"""
model.py
--------
Phase 2: LightGBM cross-sectional ranker with walk-forward validation.

Feature set covers all OHLCV-derived signals:
  - Short and long-horizon momentum
  - Mean reversion (RSI, z-score)
  - 52-week high proximity, MACD, MAX effect
  - Amihud illiquidity, idiosyncratic volatility
  - Realized volatility, volatility regime
  - Volume spike, volume-price trend
  - Ranked versions of all composite signals
"""

import hashlib
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _impute_for_ridge(X: np.ndarray) -> np.ndarray:
    """Per-column median imputation for Ridge/MLP fitting.

    LightGBM and XGBoost handle NaN natively and MUST receive raw NaN
    (0 != missing for momentum / amihud / etc.). Ridge and MLP cannot
    handle NaN, so we fill with the per-column median computed from the
    finite values in that column. Inf values are first replaced with NaN.
    Columns that are entirely NaN are filled with 0.
    """
    if X.size == 0:
        return X
    X = np.asarray(X, dtype=np.float32).copy()  # Opt 1: float32 (LightGBM bins to float32 internally)
    # Treat inf as missing
    X[~np.isfinite(X)] = np.nan
    # Per-column median (nan-ignoring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_median = np.nanmedian(X, axis=0)
    # Columns entirely NaN → median becomes nan; fill with 0
    col_median = np.where(np.isfinite(col_median), col_median, 0.0).astype(np.float32)
    # Fill NaN entries with the column median
    idx = np.where(np.isnan(X))
    if len(idx[0]):
        X[idx] = np.take(col_median, idx[1])
    return X


def _cs_winsorize(df: pd.DataFrame, pct_low: float = 0.01, pct_high: float = 0.99) -> pd.DataFrame:
    """Clip each date's cross-section at the given lower/upper percentiles.

    Per-date cross-sectional winsorization — only uses within-date
    information, so it introduces no temporal leakage.
    """
    lo = df.quantile(pct_low, axis=1)
    hi = df.quantile(pct_high, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. Run: pip install lightgbm")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.linear_model import Ridge
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def size_neutralize(
    feature_df: pd.DataFrame,
    log_mcap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Regress ``feature_df`` on ``log_mcap_df`` cross-sectionally per date
    and return the residuals.

    Both inputs must share index (dates) and columns (tickers). For each date
    (row), we run a single-variable OLS of feature ~ a + b * log_mcap using
    only tickers where both values are finite, and return the residual
    (feature - (a + b * log_mcap)). Rows with fewer than 10 valid tickers or
    zero variance in log_mcap are returned unchanged (after ffill alignment).
    """
    # Align
    lm = log_mcap_df.reindex(index=feature_df.index, columns=feature_df.columns)
    out = feature_df.copy()

    x_arr = lm.to_numpy(dtype=float)
    y_arr = feature_df.to_numpy(dtype=float)
    res = y_arr.copy()

    for i in range(y_arr.shape[0]):
        x = x_arr[i]
        y = y_arr[i]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        xv = x[mask]
        yv = y[mask]
        xm = xv.mean()
        ym = yv.mean()
        dx = xv - xm
        denom = (dx * dx).sum()
        if denom <= 0 or not np.isfinite(denom):
            continue
        beta = (dx * (yv - ym)).sum() / denom
        alpha = ym - beta * xm
        pred = alpha + beta * x  # full row (NaNs stay NaN)
        res[i] = y - pred
    out.iloc[:, :] = res
    return out


def build_feature_matrix(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    ranked_signals: Dict[str, pd.DataFrame],
    alt_features: Optional[Dict[str, pd.DataFrame]] = None,
    use_cache: bool = True,
    sector_map: Optional[Dict[str, str]] = None,
    size_neutralize: bool = True,  # noqa: shadows helper name intentionally (public API)
    shares_outstanding: Optional[pd.DataFrame] = None,
    sector_neutralize_features: bool = True,
    winsorize: bool = True,
    cs_zscore_all: bool = True,
    use_higher_moment: bool = True,
    use_breadth_wavelet: bool = True,
) -> pd.DataFrame:
    """
    Build long-format feature matrix for the ML model.

    Feature set (~60+ features):
      Momentum:        mom 5/10/21/63/126d, price_accel, residual_mom, efficiency
      Mean reversion:  z-score, RSI, Bollinger %B, MA50/200 distance
      Trend:           52w-high ratio, MACD histogram
      Tail / lottery:  max return 21d, tail risk 5th pct
      Volatility:      rvol 10/21/63d, vol regime, idiovol, vol-of-vol
      Beta:            market beta 63d
      Liquidity:       Amihud illiquidity 21d
      Volume:          vol spike, VPT, OBV z-score, vol trend
      Ranked signals:  rank of all 20 composite signals
      Extra:           ranked low-vol feature
      Alt data:        fundamentals, macro, VIX term structure,
                       short interest, earnings surprise (if provided)

    Parameters
    ----------
    size_neutralize : if True, add ``_sn`` variants of the most size-contaminated
        features (momentum, volatility, beta, Amihud) by regressing each one
        cross-sectionally on ``log_mcap_z`` per date and taking residuals.
    shares_outstanding : optional DataFrame (index=dates, columns=tickers) of
        shares outstanding. Used to compute market cap = close * shares. If
        None or sparse, falls back to log(21d rolling dollar volume) as a
        size proxy.
    use_higher_moment : if True, add higher-moment signals (realized skew,
        co-skew, semi-beta decomposition, tail dependence, signed jumps,
        downside/upside vol ratio, downside-beta spread, Kumar lottery
        composite). Market returns proxy is computed on-the-fly as the
        equal-weighted universe cross-sectional mean.
    use_breadth_wavelet : if True, add market-breadth features (%>200MA,
        new-highs-minus-lows, advance-decline ratio, breadth composite z)
        broadcast across tickers, plus per-ticker wavelet/FFT band energy
        decomposition (intra-week, weekly, monthly, quarterly, dominant
        frequency, spectral entropy).
    """
    # Local alias for the module-level helper (parameter of the same name
    # shadows it inside this function).
    _size_neutralize_fn = globals()["size_neutralize"]
    _do_size_neutralize = bool(size_neutralize)

    # ── feature panel cache ──────────────────────────────────────────────────
    _cache_key = hashlib.md5(
        f"{close.index[0]}_{close.index[-1]}_{close.shape}_{sorted(ranked_signals.keys())}"
        f"_{sorted(alt_features.keys()) if alt_features else []}"
        f"_sector={bool(sector_map)}"
        f"_sn={bool(size_neutralize)}_shr={shares_outstanding is not None}"
        f"_sni={bool(sector_neutralize_features)}"
        f"_wz={bool(winsorize)}_csz={bool(cs_zscore_all)}"
        f"_hm={bool(use_higher_moment)}_bw={bool(use_breadth_wavelet)}".encode()
    ).hexdigest()[:12]
    _cache_file = _CACHE_DIR / f"feature_panel_{_cache_key}.pkl"

    if use_cache and _cache_file.exists():
        print(f"      [cache] Loading feature panel from {_cache_file.name}")
        with open(_cache_file, "rb") as f:
            return pickle.load(f)
    # ─────────────────────────────────────────────────────────────────────────

    from features import (
        momentum,
        realized_volatility,
        volatility_regime,
        volume_spike,
        volume_price_trend,
        price_to_52w_high,
        max_return,
        amihud_illiquidity,
        idiosyncratic_volatility,
        macd_signal,
        zscore_reversion,
        rsi,
        market_beta,
        obv_signal,
        bollinger_pct_b,
        residual_momentum,
        price_acceleration,
        volume_trend,
        efficiency_ratio,
        ma_distance,
        vol_of_vol,
        tail_risk,
        cross_sectional_rank,
        # New signals
        hurst_exponent,
        short_term_reversal,
        volume_confirmed_reversal,
        sector_relative_momentum,
        sector_neutralize as _sector_neutralize_fn,
    )

    features = {}

    # Momentum — short, medium, long horizon
    for w in [5, 10, 21, 63, 126]:
        features[f"mom_{w}d"] = momentum(close, w)

    # Classic Jegadeesh-Titman 12-1 momentum: skip most-recent month (21d)
    # to avoid short-term reversal, measure return over the prior 11 months
    # (252 - 21 = 231 trading days). Well-documented anomaly (JT 1993,
    # Carhart 1997, Fama-French 2012). Kept alongside mom_63d/mom_126d.
    features["mom_12_1"] = close.shift(21).pct_change(252 - 21)

    # Momentum quality
    features["price_accel_21d"]  = price_acceleration(close, window=21)
    features["residual_mom_21d"] = residual_momentum(returns, window=21)
    features["efficiency_21d"]   = efficiency_ratio(close, window=21)

    # Mean reversion
    features["zscore_rev_20d"]   = zscore_reversion(close, window=20)
    features["rsi_14d"]          = rsi(close, window=14)
    features["bollinger_pct_b"]  = bollinger_pct_b(close)
    features["ma50_distance"]    = ma_distance(close, window=50)
    features["ma200_distance"]   = ma_distance(close, window=200)

    # Trend / price structure
    features["price_52w_high"]   = price_to_52w_high(close)
    features["macd_hist"]        = macd_signal(close)

    # Lottery / tail risk
    features["max_ret_21d"]      = max_return(close, window=21)
    features["tail_risk_5pct"]   = tail_risk(returns, window=21, pct=0.05)

    # Volatility
    for w in [10, 21, 63]:
        features[f"rvol_{w}d"]   = realized_volatility(returns, window=w)
    features["vol_regime"]       = volatility_regime(returns, short=10, long=60)
    features["idiovol_21d"]      = idiosyncratic_volatility(returns, window=21)
    features["vol_of_vol"]       = vol_of_vol(returns)

    # Beta / factor exposure
    features["mkt_beta_63d"]     = market_beta(returns, window=63)

    # Liquidity
    features["amihud_21d"]       = amihud_illiquidity(returns, volume, close, window=21)

    # Volume
    features["vol_spike_20d"]    = volume_spike(volume, window=20)
    features["vpt_20d"]          = volume_price_trend(close, volume, window=20)
    features["obv_zscore"]       = obv_signal(returns, volume)
    features["vol_trend_21d"]    = volume_trend(volume, window=21)

    # Ranked versions of all 20 composite signals
    for name, sig in ranked_signals.items():
        features[f"rank_{name}"] = sig

    # Cross-sectional rank of low volatility
    features["rank_rvol_21d"] = cross_sectional_rank(-features["rvol_21d"])

    # ── New reversal signals ─────────────────────────────────────────────────
    # 1-day and 2-day explicit reversal (Jegadeesh 1990) — orthogonal to
    # the 5-21d short_momentum feature already in ranked_signals.
    features["ret_reversal_1d"] = short_term_reversal(returns, window=1)
    features["ret_reversal_2d"] = short_term_reversal(returns, window=2)
    # Volume-confirmed reversal: stronger reversal signal when drop was on high volume
    features["vol_reversal"]    = volume_confirmed_reversal(returns, volume)

    # Sector-relative momentum — only when sector_map is available
    if sector_map is not None:
        features["sector_rel_mom_63d"] = sector_relative_momentum(returns, sector_map, window=63)

    # Hurst exponent (trending vs mean-reverting)
    features["hurst_63d"] = hurst_exponent(close, window=63)

    # ── Tier 6: Higher-moment / co-moment / tail signals ────────────────────
    # Realized skew, co-skew, semi-beta decomposition, tail dependence,
    # signed jumps, downside/upside vol ratio, downside-beta spread,
    # Kumar lottery composite. Market returns proxy = equal-weighted
    # cross-sectional mean of the universe (on-the-fly).
    if use_higher_moment:
        from features import (
            realized_skewness,
            co_skewness,
            semi_beta_decomposition,
            tail_dependence,
            signed_jump_intensity,
            downside_upside_vol_ratio,
            downside_beta_spread,
            kumar_lottery_composite,
        )
        market_returns = returns.mean(axis=1)

        try:
            features["realized_skew_21"] = -realized_skewness(returns, window=21)
            features["realized_skew_63"] = -realized_skewness(returns, window=63)
            features["co_skew_252"] = -co_skewness(returns, market_returns, window=252)
            sb_N, sb_P, sb_Mplus, sb_Mminus = semi_beta_decomposition(
                returns, market_returns, window=252
            )
            # Semi-beta sign convention (Bollerslev-Patton-Quaedvlieg 2022):
            #   beta^N  (concordant downside: both r_i<0, r_m<0) → POSITIVE premium → keep raw
            #   beta^P  (concordant upside:   both r_i>0, r_m>0) → NEGATIVE premium → negate
            #   beta^M+ (mixed: r_i>0, r_m<0)                   → ~zero/negative premium → negate
            #   beta^M- (mixed: r_i<0, r_m>0)                   → NEGATIVE premium → negate
            # Negating aligns each signal so that higher score = higher expected return,
            # consistent with other Tier-6 signals.
            features["semi_beta_N"] = sb_N
            features["semi_beta_P"] = -sb_P
            features["semi_beta_Mplus"] = -sb_Mplus
            features["semi_beta_Mminus"] = -sb_Mminus
            features["tail_dep_lower"] = tail_dependence(
                returns, market_returns, window=252, q=0.1
            )
            features["signed_jump_21"] = -signed_jump_intensity(
                returns, window=21, threshold=4.0
            )
            features["down_up_vol_ratio_63"] = downside_upside_vol_ratio(
                returns, window=63
            )
            features["downside_beta_spread_252"] = downside_beta_spread(
                returns, market_returns, window=252
            )
            features["kumar_lottery_21"] = kumar_lottery_composite(
                close, returns, window=21
            )
            print("      [tier6] added 11 higher-moment signals")
        except Exception as _e:
            warnings.warn(f"higher-moment signal computation failed: {_e}")

    # ── Tier 6: Market-breadth + wavelet/FFT band energy ────────────────────
    if use_breadth_wavelet:
        from features import (
            pct_above_200ma,
            new_highs_minus_lows,
            advance_decline_ratio,
            breadth_z,
            wavelet_band_energy,
        )
        # Breadth signals are per-date scalars — broadcast across tickers.
        try:
            bp = pct_above_200ma(close)
            nhnl = new_highs_minus_lows(close, window=252)
            adr = advance_decline_ratio(returns)
            bz = breadth_z(close, returns, window=252)

            def _broadcast_breadth(series_like):
                # signal functions return DataFrames with one column per
                # ticker already tiled, OR a Series. Normalize to a
                # (dates x tickers) DataFrame aligned to close.
                if isinstance(series_like, pd.DataFrame):
                    if series_like.shape[1] == len(close.columns):
                        return series_like.reindex(
                            index=close.index, columns=close.columns
                        )
                    # Single column → tile
                    col = series_like.iloc[:, 0]
                else:
                    col = series_like
                col = col.reindex(close.index)
                vals = col.to_numpy().reshape(-1, 1)
                return pd.DataFrame(
                    np.tile(vals, (1, len(close.columns))),
                    index=close.index,
                    columns=close.columns,
                )

            features["breadth_pct_200ma"] = _broadcast_breadth(bp)
            features["breadth_nh_nl"] = _broadcast_breadth(nhnl)
            features["breadth_adv_dec"] = _broadcast_breadth(adr)
            features["breadth_composite_z"] = _broadcast_breadth(bz)
            print("      [tier6] added 4 breadth signals")
        except Exception as _e:
            warnings.warn(f"breadth signal computation failed: {_e}")

        # Wavelet / FFT band-energy decomposition
        try:
            wbe = wavelet_band_energy(returns, window=256)
            for band_name, band_df in wbe.items():
                features[f"wavelet_band_{band_name}"] = band_df
            print(f"      [tier6] added {len(wbe)} wavelet band-energy signals")
        except Exception as _e:
            warnings.warn(f"wavelet band energy failed: {_e}")

    # ── Cross-sectional z-scores of key raw features ─────────────────────────
    # Ridge regression benefits from features that preserve distance information.
    # The ranked [0,1] features are good for LGBM but z-scores improve Ridge fit.
    def _cs_zscore(df: pd.DataFrame, clip: float = 4.0) -> pd.DataFrame:
        mu    = df.mean(axis=1)
        sigma = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sigma, axis=0).clip(-clip, clip)

    for raw_feat in ["rvol_21d", "idiovol_21d", "amihud_21d", "mkt_beta_63d"]:
        if raw_feat in features:
            features[f"z_{raw_feat}"] = _cs_zscore(features[raw_feat])
    for mom_w in [63, 126]:
        key = f"mom_{mom_w}d"
        if key in features:
            features[f"z_{key}"] = _cs_zscore(features[key])
    # z-score for classic 12-1 momentum
    if "mom_12_1" in features:
        features["z_mom_12_1"] = _cs_zscore(features["mom_12_1"])

    # ── Size feature: log market cap (cross-sectionally z-scored) ────────────
    # Market cap = Close * SharesOutstanding (ffilled within each ticker,
    # since shares update quarterly). If shares_outstanding is missing or
    # too sparse, fall back to log(21d rolling dollar volume) as a proxy.
    log_mcap_raw = None
    _used_shares = False
    if shares_outstanding is not None:
        try:
            shr = shares_outstanding.reindex(index=close.index, columns=close.columns)
            shr = shr.ffill()
            # require ≥20% coverage on average to consider usable
            if shr.notna().mean().mean() > 0.20:
                mcap = close * shr
                mcap = mcap.where(mcap > 0)
                log_mcap_raw = np.log(mcap)
                _used_shares = True
        except Exception:
            log_mcap_raw = None

    # ── Separate liquidity proxy (NOT a size proxy) ─────────────────────────
    # 21-day rolling dollar volume is a liquidity measure. It correlates with
    # size but is contaminated by turnover, so we expose it as its OWN
    # feature (log_liquidity_z) rather than labeling it log_mcap_z.
    _dollar_vol = (close * volume).rolling(21, min_periods=5).mean()
    _dollar_vol = _dollar_vol.where(_dollar_vol > 0)
    log_liquidity_z = _cs_zscore(np.log(_dollar_vol))
    features["log_liquidity_z"] = log_liquidity_z

    # Log price: weak size-ish proxy (price-level anomaly). Kept distinct
    # from market-cap so it does not contaminate `_sn` residuals.
    log_price_z = _cs_zscore(np.log(close.where(close > 0)))
    features["log_price_z"] = log_price_z

    if log_mcap_raw is None:
        # No usable shares coverage → use liquidity z as the size-neutralize
        # basis. We do NOT publish log_mcap_z in this case (feature is absent)
        # so the model can learn from log_liquidity_z / log_price_z instead.
        log_mcap_z = log_liquidity_z
        print(
            "      [size] log_mcap unavailable (shares coverage <20%); "
            "size-neutralization uses log_liquidity_z proxy, no log_mcap_z feature emitted"
        )
    else:
        log_mcap_z = _cs_zscore(log_mcap_raw)
        features["log_mcap_z"] = log_mcap_z
        print("      [size] log_mcap_z built from shares outstanding")

    # ── Size-neutralized variants of size-contaminated features ─────────────
    # Regress each feature cross-sectionally on log_mcap_z per date and keep
    # the residual. This purges the systematic small-cap tilt without
    # shrinking the training universe.
    if _do_size_neutralize:
        _sn_targets = [
            "mom_21d", "mom_63d", "mom_126d", "mom_12_1",
            "z_mom_63d", "z_mom_126d", "z_mom_12_1",
            "rvol_21d", "idiovol_21d",
            "z_rvol_21d", "z_idiovol_21d",
            "mkt_beta_63d", "z_mkt_beta_63d",
            "amihud_21d", "z_amihud_21d",
        ]
        _sn_added = []
        for feat_name in _sn_targets:
            if feat_name not in features:
                continue
            try:
                features[f"{feat_name}_sn"] = _size_neutralize_fn(
                    features[feat_name], log_mcap_z
                )
                _sn_added.append(feat_name)
            except Exception as _e:
                warnings.warn(f"size_neutralize failed for {feat_name}: {_e}")
        print(f"      [size] added {len(_sn_added)} size-neutralized _sn variants")

    # ── Sector-neutralized variants of sector-contaminated features ─────────
    # Subtract the per-date, per-sector mean from each feature value. This
    # removes systematic sector tilts so downstream ranking measures purely
    # within-industry stock-picking (JKP 2023, AQR best practice).
    if sector_neutralize_features and sector_map:
        _sni_targets = [
            "mom_63d", "mom_126d", "mom_12_1",
            "z_mom_63d", "z_mom_126d", "z_mom_12_1",
            "idiovol_21d", "z_idiovol_21d",
            "mkt_beta_63d", "z_mkt_beta_63d",
            "rvol_21d", "z_rvol_21d",
        ]
        _sni_added = []
        for feat_name in _sni_targets:
            if feat_name not in features:
                continue
            try:
                features[f"{feat_name}_sni"] = _sector_neutralize_fn(
                    features[feat_name], sector_map
                )
                _sni_added.append(feat_name)
            except Exception as _e:
                warnings.warn(f"sector_neutralize failed for {feat_name}: {_e}")
        print(f"      [sector] added {len(_sni_added)} sector-neutralized _sni variants")

    # ── Calendar / seasonality features ──────────────────────────────────────
    # January effect (small caps), Q4 window dressing, Q1 tax selling.
    # These are broadcast scalars (same for all tickers on a date) and act as
    # regime-conditioning variables for the ML model.
    _idx   = close.index
    _month = _idx.month.values.astype(float)
    _quarter = _idx.quarter.values.astype(float)

    def _broadcast(series_vals: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(series_vals.reshape(-1, 1), (1, len(close.columns))),
            index=close.index,
            columns=close.columns,
        )

    # Calendar seasonality features (month_sin, month_cos, is_q1, is_q4)
    # removed: broadcast scalars cause correlated daily reranking across
    # all tickers without adding cross-sectional predictive power.

    # Alternative data features.
    # Raw macro/VIX broadcast series (identical value across all tickers on a
    # given day) are excluded from direct inclusion — they have zero cross-sectional
    # variation. Static short interest snapshots are also excluded.
    # However, we CREATE INTERACTION FEATURES by multiplying macro scalars with
    # cross-sectional stock features. This gives each stock a different value
    # (e.g., high-beta stocks respond more to yield curve changes).
    _EXCLUDE = {
        # macro broadcast (used for interactions below, not raw)
        "yield_curve", "hy_spread", "vix_macro", "fed_funds",
        "unemployment", "breakeven_inflation", "yield_curve_chg5d",
        "vix_term_slope", "vix_vix_percentile", "vix_vix_change_5d",
        # static snapshot
        "short_ratio_signal", "short_pct_float_signal",
    }

    # Capture macro broadcast DataFrames for interaction features
    _macro_dfs = {}

    if alt_features:
        for feat_name, df in alt_features.items():
            if feat_name in _EXCLUDE:
                # Save macro broadcasts for interaction features
                try:
                    aligned = df.reindex(index=close.index, columns=close.columns)
                    if aligned.notna().mean().mean() > 0.10:  # require >10% coverage
                        _macro_dfs[feat_name] = aligned
                except Exception:
                    pass
                continue
            try:
                aligned = df.reindex(index=close.index, columns=close.columns)
                if aligned.notna().mean().mean() > 0.10:  # require >10% coverage
                    features[f"alt_{feat_name}"] = aligned
            except Exception:
                pass

    # ── Macro × stock interaction features ──────────────────────────────────
    # Multiply macro regime scalars by cross-sectional stock characteristics.
    # Each stock gets a different value because the stock feature varies.
    # These let the model learn regime-dependent factor exposures:
    #   - high-beta stocks respond more to yield curve changes
    #   - high-idiovol stocks respond more to credit spreads
    #   - momentum effectiveness varies with VIX regime
    _INTERACTIONS = {
        # (macro_key, stock_feature_key, output_name)
        ("yield_curve",     "mkt_beta_63d",  "beta_x_yield_curve"),
        ("hy_spread",       "idiovol_21d",   "idiovol_x_hy_spread"),
        ("vix_macro",       "mom_63d",       "mom_x_vix"),
        ("vix_vix_percentile", "rvol_21d",   "rvol_x_vix_pctile"),
        ("yield_curve",     "mom_126d",      "longmom_x_yield_curve"),
        ("hy_spread",       "amihud_21d",    "amihud_x_hy_spread"),
    }

    for macro_key, stock_key, out_name in _INTERACTIONS:
        if macro_key in _macro_dfs and stock_key in features:
            try:
                macro_df = _macro_dfs[macro_key]
                stock_df = features[stock_key]
                # Multiply: stock-level feature × macro scalar
                interaction = stock_df * macro_df
                # Cross-sectional z-score to normalize
                mu = interaction.mean(axis=1)
                sigma = interaction.std(axis=1).replace(0, np.nan)
                interaction_z = interaction.sub(mu, axis=0).div(sigma, axis=0).clip(-4, 4)
                features[f"ix_{out_name}"] = interaction_z
            except Exception:
                pass

    # ── Explicit feature interactions ────────────────────────────────────
    # These help Ridge (which can't discover interactions) and reduce the
    # tree depth LGBM needs. Focus on the highest-importance feature pairs.
    _FEATURE_INTERACTIONS = [
        ("mom_63d",       "rvol_21d",      "mom_x_vol"),
        ("mom_126d",      "hurst_63d",     "mom_x_hurst"),
        ("idiovol_21d",   "amihud_21d",    "idiovol_x_illiq"),
        ("mkt_beta_63d",  "rvol_21d",      "beta_x_vol"),
        ("price_52w_high","mom_63d",       "high52w_x_mom"),
    ]
    for feat_a, feat_b, out_name in _FEATURE_INTERACTIONS:
        if feat_a in features and feat_b in features:
            try:
                interaction = features[feat_a] * features[feat_b]
                mu = interaction.mean(axis=1)
                sigma = interaction.std(axis=1).replace(0, np.nan)
                features[f"ix_{out_name}"] = interaction.sub(mu, axis=0).div(sigma, axis=0).clip(-4, 4)
            except Exception:
                pass

    # ── Sector target encoding (leave-one-out) ─────────────────────────
    # Trailing 63-day sector average return, EXCLUDING the stock itself.
    # Without leave-one-out, the stock's own return contaminates its sector
    # signal — a subtle form of data leakage.
    if sector_map is not None:
        sector_groups_te: dict = {}
        for ticker in close.columns:
            s = sector_map.get(ticker, "Unknown")
            sector_groups_te.setdefault(s, []).append(ticker)

        sector_mom = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for sector, tickers in sector_groups_te.items():
            if len(tickers) < 3:
                continue
            sect_returns = returns[tickers].shift(1)
            sector_sum = sect_returns.sum(axis=1).rolling(63, min_periods=21).sum()
            n_stocks = sect_returns.notna().sum(axis=1).rolling(63, min_periods=21).sum()
            for t in tickers:
                # Leave-one-out: subtract this stock's contribution
                stock_contrib = sect_returns[t].rolling(63, min_periods=21).sum()
                loo_sum = sector_sum - stock_contrib.fillna(0)
                loo_n = n_stocks - sect_returns[t].notna().astype(float).rolling(63, min_periods=21).sum()
                sector_mom[t] = loo_sum / loo_n.replace(0, np.nan)

        features["sector_momentum_63d"] = sector_mom

    # ── Winsorize + cross-sectionally z-score continuous features ──────────
    # Apply to ALL continuous numeric features EXCEPT:
    #   - rank_* features (already in [0,1])
    #   - z_* features (already cross-sectionally z-scored)
    #   - boolean/binary signals
    # Per-date cross-sectional operations only use within-date information,
    # so they introduce no temporal leakage (fold-safe).
    def _is_continuous_feat(name: str, df: pd.DataFrame) -> bool:
        if name.startswith("rank_") or name.startswith("z_"):
            return False
        # Exclude booleans
        try:
            if df.dtypes.apply(lambda t: t == bool).all():
                return False
        except Exception:
            pass
        return True

    _continuous_feats = [n for n, d in features.items() if _is_continuous_feat(n, d)]

    if winsorize:
        for name in _continuous_feats:
            try:
                features[name] = _cs_winsorize(features[name], 0.01, 0.99)
            except Exception:
                pass
        print(f"      [winsor] winsorized {len(_continuous_feats)} continuous features at 1/99 pct")

    if cs_zscore_all:
        # Per-date cross-sectional z-score — only uses same-date information,
        # so introduces NO temporal leakage (no rolling window required).
        _csz_added = 0
        _csz_skipped_broadcast = 0
        for name in _continuous_feats:
            try:
                feat_df = features[name]
                # Skip broadcast features (e.g. market breadth) where every ticker
                # has the same value per date → cross-sectional std = 0 → the _csz
                # column would be all-NaN (then all-zero after fillna). Detect
                # automatically via per-date cross-sectional variance.
                if feat_df.std(axis=1).max() <= 0 or not np.isfinite(
                    feat_df.std(axis=1).max()
                ):
                    _csz_skipped_broadcast += 1
                    continue
                features[f"{name}_csz"] = _cs_zscore(feat_df)
                _csz_added += 1
            except Exception:
                pass
        print(
            f"      [csz] added {_csz_added} cross-sectional z-scored _csz variants "
            f"(skipped {_csz_skipped_broadcast} zero-variance broadcast features)"
        )

    panels = []
    for feat_name, df in features.items():
        s = df.stack(future_stack=True)
        s.name = feat_name
        panels.append(s)

    # Opt 2: free intermediate feature dict (~16GB peak) before concat
    del features
    import gc; gc.collect()

    panel = pd.concat(panels, axis=1)
    del panels; gc.collect()  # Opt 2: free stacked Series list too
    panel.index.names = ["date", "ticker"]

    # Opt 1: cast to float32 — LightGBM/XGBoost internally bin to float32
    # before split-finding (per LightGBM docs), so zero accuracy impact.
    # Saves ~50% RAM on the feature panel.
    panel = panel.astype(np.float32)

    if use_cache:
        with open(_cache_file, "wb") as f:
            pickle.dump(panel, f)
        print(f"      [cache] Feature panel saved to {_cache_file.name} (float32)")

    return panel


def build_labels(
    returns: pd.DataFrame,
    forward_window: int = 5,
    risk_adjust: bool = False,
    sector_map: Optional[Dict[str, str]] = None,
    sector_rank_weight: float = 0.0,
    beta_neutral: bool = False,
    market_returns: Optional[pd.Series] = None,
    forward_windows: Optional[List[int]] = None,
    investable_mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Build 6-grade relevance labels for LambdaRank.

    Within-industry ranking (when sector_map provided):
      Instead of ranking all 2000+ stocks together (which the model solves by
      picking volatile small-caps), rank within each sector. The top 5% of tech
      stocks is different from the top 5% of utilities. This forces the model
      to learn genuine stock selection, not sector/factor bets.

    6-grade scheme with a "superstar" tier (top 5%):
      Grade 0: bottom 20%    (worst)
      Grade 1: 20-40%
      Grade 2: 40-60%
      Grade 3: 60-80%
      Grade 4: 80-95%        (good)
      Grade 5: top 5%        (superstar — what we're hunting for)

    Risk-adjusted labels (opt-in, default OFF):
      If risk_adjust=True, ranks forward_return / trailing_vol instead of raw
      forward return. This was the historical default, but empirically vol-
      scaling systematically down-weights momentum stocks (where the alpha
      lives) and hurt realized Sharpe. Left available for experiments.

    Labels for the last `forward_window` days are NaN (unrealized returns).

    Parameters
    ----------
    beta_neutral : if True, subtract (rolling 60d beta * market_forward_return)
        from each stock's forward return to produce beta-neutral labels.
        Requires market_returns Series aligned with returns.index.
    market_returns : optional pd.Series of market (e.g., SPY) daily returns
        indexed by date. Required when beta_neutral=True.
    forward_windows : optional list of horizons (e.g., [1, 5, 21]). When
        provided, labels are the RANK-AVERAGE across horizons to reduce
        horizon sensitivity. Overrides forward_window for signal computation.
    """
    def _fwd_ret(win: int) -> pd.DataFrame:
        r = returns.shift(-win).rolling(win).sum()
        r.iloc[-win:] = np.nan
        return r

    # ── Fix 1 (ML path): Mask non-investable tickers before label ranking ──
    # The ML model trains on ALL tickers' features (more data helps), but
    # labels should rank WITHIN the investable set so the model learns which
    # stocks are best among tradeable names. Without this, the model learns
    # to predict micro-cap rankings that can never be traded.
    _inv_mask = None
    if investable_mask is not None:
        _inv_mask = investable_mask.reindex(
            index=returns.index, columns=returns.columns
        ).fillna(False)

    # ── Multi-horizon rank averaging (item 7) ──────────────────────────────
    if forward_windows is not None and len(forward_windows) > 0:
        rank_stack = []
        for win in forward_windows:
            fr = _fwd_ret(win)
            if beta_neutral and market_returns is not None:
                try:
                    mkt_fwd = market_returns.reindex(returns.index).shift(-win).rolling(win).sum()
                    mkt_fwd.iloc[-win:] = np.nan
                    # 60d rolling beta of each stock's daily ret vs market
                    mkt_daily = market_returns.reindex(returns.index)
                    cov = returns.rolling(60, min_periods=20).cov(mkt_daily)
                    var = mkt_daily.rolling(60, min_periods=20).var()
                    beta_df = cov.div(var.replace(0, np.nan), axis=0)
                    fr = fr.sub(beta_df.mul(mkt_fwd, axis=0), fill_value=0)
                except Exception:
                    pass
            # Mask non-investable before ranking so labels rank within tradeable set
            if _inv_mask is not None:
                fr = fr.where(_inv_mask)
            rank_stack.append(fr.rank(axis=1, pct=True))
        fwd_signal = sum(rank_stack) / len(rank_stack)
        # fwd_signal is already a rank in [0,1] — skip additional rank below
        universe_rank = fwd_signal
        if sector_map is not None and sector_rank_weight > 0.0:
            sector_rank = pd.DataFrame(np.nan, index=fwd_signal.index, columns=fwd_signal.columns)
            sector_groups: Dict[str, list] = {}
            for ticker in fwd_signal.columns:
                s = sector_map.get(ticker, "Unknown")
                sector_groups.setdefault(s, []).append(ticker)
            for sector, tickers in sector_groups.items():
                if len(tickers) < 5:
                    sector_rank[tickers] = universe_rank[tickers]
                else:
                    sector_rank[tickers] = fwd_signal[tickers].rank(axis=1, pct=True)
            pct_rank = (1.0 - sector_rank_weight) * universe_rank + sector_rank_weight * sector_rank
        else:
            pct_rank = universe_rank

        def rank_to_grade_mh(row):
            return pd.cut(
                row,
                bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 0.95, 1.001],
                labels=[0, 1, 2, 3, 4, 5],
            ).astype("Int64")
        grades = pct_rank.apply(rank_to_grade_mh, axis=1)
        stacked = grades.stack(future_stack=True)
        stacked.name = "label"
        return stacked

    fwd_return = _fwd_ret(forward_window)

    # ── Beta-neutral labels (item 6) ───────────────────────────────────────
    if beta_neutral and market_returns is not None:
        try:
            mkt_fwd = market_returns.reindex(returns.index).shift(-forward_window).rolling(forward_window).sum()
            mkt_fwd.iloc[-forward_window:] = np.nan
            mkt_daily = market_returns.reindex(returns.index)
            cov = returns.rolling(60, min_periods=20).cov(mkt_daily)
            var = mkt_daily.rolling(60, min_periods=20).var()
            beta_df = cov.div(var.replace(0, np.nan), axis=0)
            fwd_return = fwd_return.sub(beta_df.mul(mkt_fwd, axis=0), fill_value=0)
        except Exception as _e:
            warnings.warn(f"beta_neutral label adjustment failed: {_e}")

    if risk_adjust:
        trailing_vol = returns.rolling(21, min_periods=10).std() * np.sqrt(252)
        fwd_signal = fwd_return / trailing_vol.replace(0, np.nan)
    else:
        fwd_signal = fwd_return

    # Mask non-investable tickers before ranking (Fix 1, single-horizon path)
    if _inv_mask is not None:
        fwd_signal = fwd_signal.where(_inv_mask)

    # Within-industry ranking: rank each stock vs its sector peers.
    # This forces the model to learn stock selection, not sector bets.
    # Blend is parameterized via sector_rank_weight:
    #   0.0 = pure universe rank (default — captures cross-sector mega-cap alpha)
    #   0.5 = 50/50 blend (legacy behavior)
    #   1.0 = pure within-sector rank
    universe_rank = fwd_signal.rank(axis=1, pct=True)

    if sector_map is not None and sector_rank_weight > 0.0:
        sector_rank = pd.DataFrame(np.nan, index=fwd_signal.index, columns=fwd_signal.columns)
        sector_groups: Dict[str, list] = {}
        for ticker in fwd_signal.columns:
            s = sector_map.get(ticker, "Unknown")
            sector_groups.setdefault(s, []).append(ticker)

        for sector, tickers in sector_groups.items():
            if len(tickers) < 5:
                # Too few stocks — use universe rank
                sector_rank[tickers] = universe_rank[tickers]
            else:
                sector_rank[tickers] = fwd_signal[tickers].rank(axis=1, pct=True)

        pct_rank = (1.0 - sector_rank_weight) * universe_rank + sector_rank_weight * sector_rank
    else:
        pct_rank = universe_rank

    def rank_to_grade(row):
        return pd.cut(
            row,
            bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 0.95, 1.001],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype("Int64")

    grades = pct_rank.apply(rank_to_grade, axis=1)
    stacked = grades.stack(future_stack=True)
    stacked.name = "label"
    return stacked


# ---------------------------------------------------------------------------
# Monotone-constraint defaults for signed factors (item 11)
# ---------------------------------------------------------------------------

def _default_monotone_constraints() -> Dict[str, int]:
    """Known-signed factor directions (+1 = monotone increasing with label,
    -1 = monotone decreasing). Applied across prefix/exact-match feature names.

    Unknown features map to 0 (no constraint) at apply-time.
    """
    return {
        # Momentum: +1 (higher momentum -> higher expected return)
        "mom_5d": 1, "mom_10d": 1, "mom_21d": 1, "mom_63d": 1, "mom_126d": 1,
        "z_mom_63d": 1, "z_mom_126d": 1,
        "mom_21d_sn": 1, "mom_63d_sn": 1, "mom_126d_sn": 1,
        "z_mom_63d_sn": 1, "z_mom_126d_sn": 1,
        "mom_63d_sni": 1, "mom_126d_sni": 1,
        "z_mom_63d_sni": 1, "z_mom_126d_sni": 1,
        "residual_mom_21d": 1, "price_accel_21d": 1, "efficiency_21d": 1,
        "sector_rel_mom_63d": 1, "sector_momentum_63d": 1,
        # 52-week high proximity: +1
        "price_52w_high": 1,
        # Value / B-M / E-P: +1 (high book/market -> premium)
        "alt_book_to_market": 1, "alt_earnings_yield": 1, "alt_ep_ratio": 1,
        # Quality ROE / profitability: +1
        "alt_roe": 1, "alt_roa": 1, "alt_gross_profitability": 1, "alt_profitability": 1,
        # F-score: +1
        "alt_f_score": 1, "alt_piotroski": 1,
        # Amihud illiquidity: +1 (more illiquid -> premium)
        "amihud_21d": 1, "z_amihud_21d": 1, "amihud_21d_sn": 1, "z_amihud_21d_sn": 1,
        # BAB beta: -1 (low-beta anomaly)
        "mkt_beta_63d": -1, "z_mkt_beta_63d": -1,
        "mkt_beta_63d_sn": -1, "z_mkt_beta_63d_sn": -1,
        "mkt_beta_63d_sni": -1, "z_mkt_beta_63d_sni": -1,
        # MAX effect: -1 (lottery-stock penalty)
        "max_ret_21d": -1,
        # Idio-vol: -1
        "idiovol_21d": -1, "z_idiovol_21d": -1,
        "idiovol_21d_sn": -1, "z_idiovol_21d_sn": -1,
        "idiovol_21d_sni": -1, "z_idiovol_21d_sni": -1,
        # Accruals, NOA, asset growth, net share issuance: -1
        "alt_accruals": -1, "alt_noa": -1, "alt_asset_growth": -1,
        "alt_net_share_issuance": -1, "alt_share_issuance": -1,
        # CHS distress: -1 (higher distress -> lower expected return)
        "alt_chs_distress": -1, "alt_distress": -1,
        # Short interest change: -1
        "alt_short_interest_change": -1, "alt_short_interest_chg": -1,
    }


# ---------------------------------------------------------------------------
# HRP feature-clustering helpers (item 16)
# ---------------------------------------------------------------------------

def hrp_feature_clusters(feature_df: pd.DataFrame, threshold: float = 0.5) -> Dict[int, List[str]]:
    """Hierarchical-cluster features using 1-|corr| distance.

    Computes pairwise |Spearman| correlation on the given feature matrix
    (rows=observations, cols=features), builds distance = 1-|rho|, runs
    average-linkage clustering, cuts at ``threshold``, returns dict of
    cluster_id -> list of feature names.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Fix 2 (LOW-polish): drop zero-variance columns BEFORE computing the
    # correlation matrix. A constant column produces NaN correlations that
    # fillna(0) would turn into distance=1, scattering identical-info columns
    # across clusters. Dropping them up front avoids that pathology.
    try:
        variances = feature_df.var(axis=0)
        nonzero_cols = variances[variances > 1e-10].index
        if len(nonzero_cols) < len(feature_df.columns):
            dropped = len(feature_df.columns) - len(nonzero_cols)
            print(f"[HRP] Dropped {dropped} zero-variance features before clustering")
            feature_df = feature_df[nonzero_cols]
    except Exception:
        pass
    cols = list(feature_df.columns)
    if len(cols) < 2:
        return {1: cols}
    corr = feature_df.corr(method="spearman").fillna(0).abs()
    dist = (1.0 - corr.values)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    clusters: Dict[int, List[str]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(cols[i])
    return clusters


def select_cluster_representatives(
    feature_df: pd.DataFrame,
    returns_df: pd.Series,
    clusters: Dict[int, List[str]],
) -> List[str]:
    """Pick the highest-|IC| feature from each cluster.

    ``feature_df`` rows must align with ``returns_df`` (a Series of forward
    returns). Returns a flat list of one representative per cluster.
    """
    selected: List[str] = []
    common = feature_df.index.intersection(returns_df.dropna().index)
    for cid, feats in clusters.items():
        best_f, best_ic = None, -1.0
        for f in feats:
            if f not in feature_df.columns:
                continue
            try:
                ic = feature_df[f].loc[common].corr(returns_df.loc[common], method="spearman")
                if np.isfinite(ic) and abs(ic) > best_ic:
                    best_ic = abs(ic)
                    best_f = f
            except Exception:
                pass
        if best_f is None and feats:
            best_f = feats[0]
        if best_f is not None:
            selected.append(best_f)
    return selected


# ---------------------------------------------------------------------------
# Opt 6: Feature interaction constraints (Israel-Kelly-Moskowitz 2020)
# ---------------------------------------------------------------------------

def build_interaction_constraints(feature_names: List[str]) -> List[List[int]]:
    """Group features by family for LGBM interaction_constraints.

    Features within a group can interact freely; cross-group interactions
    are blocked unless a feature belongs to the 'interaction' group (ix_*),
    which is allowed to interact with all groups.

    Grouping rationale: Israel-Kelly-Moskowitz (2020) show that restricting
    tree splits to within-family interactions reduces overfitting on noisy
    financial cross-sections while preserving predictive power.
    """
    _PREFIX_MAP = {
        "momentum":    ["mom_"],
        "volatility":  ["rvol_", "idiovol_", "vol_", "tail_", "max_ret"],
        "liquidity":   ["amihud_", "rank_amihud"],
        "beta":        ["mkt_beta", "z_mkt_beta"],
        "fundamental": ["alt_book_", "alt_roe_", "alt_gross_", "alt_asset_",
                        "alt_leverage_", "alt_eps_"],
        "alt_data":    ["alt_analyst_", "alt_insider_", "alt_eodhd_"],
        "interaction": ["ix_"],
    }

    groups: Dict[str, List[int]] = {k: [] for k in _PREFIX_MAP}
    groups["misc"] = []

    for idx, name in enumerate(feature_names):
        matched = False
        for group_name, prefixes in _PREFIX_MAP.items():
            if any(name.startswith(p) for p in prefixes):
                groups[group_name].append(idx)
                matched = True
                break
        if not matched:
            groups["misc"].append(idx)

    # Build constraint list: each non-empty group is a list of feature indices.
    # The "interaction" group (ix_*) features are appended to EVERY other group
    # so they can cross-interact with all families.
    ix_indices = groups.pop("interaction", [])
    constraints = []
    for group_name, indices in groups.items():
        if indices:
            constraints.append(indices + ix_indices)

    return constraints


# ---------------------------------------------------------------------------
# Walk-forward trainer
# ---------------------------------------------------------------------------

class WalkForwardModel:
    """
    Walk-forward ensemble ranker: LightGBM + CatBoost + Ridge.

    Three-model ensemble for maximum diversity:
      - LightGBM (40%): histogram-based boosting, fast, captures non-linear patterns
      - CatBoost (25%): ordered boosting with native categorical support, different
        gradient computation provides genuine ensemble diversity
      - Ridge (35%): linear regularization anchor, prevents overfitting to noise

    Key design:
      - IC-based feature pruning before each window (keep top ~50 features)
      - Monthly retraining on expanding window
      - Temporal sample weighting (recent data matters more)
      - Ensemble confidence scoring (penalize model disagreement)
    """

    def __init__(
        self,
        min_train_days: int = 126,
        # Opt 4: quarterly retraining (Gu-Kelly-Xiu 2020 retrain quarterly;
        # monthly retraining adds compute with negligible IC lift)
        retrain_freq: int = 63,
        forward_window: int = 5,
        risk_adjust: bool = False,
        sector_rank_weight: float = 0.0,
        # LightGBM hyperparams
        # Opt 4: 300 rounds (early stopping typically triggers before 300;
        # Gu-Kelly-Xiu 2020 use 1-2 boosting rounds per tree depth)
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        # Opt 4: 20 leaves (Israel-Kelly-Moskowitz 2020: depth 2-4 optimal
        # for cross-sectional equity prediction; 20 leaves ~ depth 4)
        num_leaves: int = 20,
        # feature_fraction=0.5: forces each tree to see only half the features.
        # Breaks correlated cluster dominance (13 vol features were capturing
        # every split). At 0.7 the model still concentrated on vol; at 0.5
        # it must discover orthogonal signal in momentum/fundamental features.
        feature_fraction: float = 0.5,
        # bagging_fraction=0.7: row subsampling compounds with feature_fraction
        # to produce more diverse trees (de-correlation effect).
        bagging_fraction: float = 0.7,
        min_child_samples: int = 20,
        # LightGBM regularization (previously hardcoded in _fit_lgbm)
        lambda_l1: float = 5.0,
        lambda_l2: float = 10.0,
        min_gain_to_split: float = 0.05,
        # Opt 4: depth 5 (matches shallower num_leaves=20; Israel-Kelly-
        # Moskowitz 2020: shallow trees generalize better on noisy financial data)
        max_depth: int = 5,
        # Ridge alpha (previously hardcoded to 50.0)
        ridge_alpha: float = 50.0,
        # Ensemble blend weights (4 models)
        lgbm_weight: float = 0.30,
        xgb_weight: float = 0.25,
        ridge_weight: float = 0.25,
        mlp_weight: float = 0.20,
        # Feature pruning
        prune_features: bool = True,
        min_ic: float = 0.01,
        # max_feature_corr=0.70: tighter than 0.85 to break the vol cluster.
        # 13 vol features at 0.85 threshold kept ~8; at 0.70 keeps ~3-4.
        max_feature_corr: float = 0.70,
        # Rolling window: cap training data at N days instead of expanding.
        # None = expanding (use all history). 756 = ~3 years rolling.
        # Rolling drops stale regime data that confuses the model.
        max_train_days: Optional[int] = None,
        # Feature neutralization: regress out vol from all features before
        # training. Forces model to find orthogonal signal, not vol-sort.
        neutralize_vol: bool = False,
        # MLP: default OFF. MLP adds ~20% training time for marginal
        # contribution. Opt-in via use_mlp=True.
        use_mlp: bool = False,
        # ── New: ranking truncation level (item 1) ─────────────────────────
        lambdarank_truncation: int = 20,
        # ── Return-decile label gain (item 2) ─────────────────────────────
        use_return_decile_gain: bool = False,
        # ── Magnitude-weighted samples (item 3) ───────────────────────────
        use_magnitude_weights: bool = False,
        # ── New ensemble heads (items 4, 5) ───────────────────────────────
        huber_weight: float = 0.0,
        quantile_weight: float = 0.0,
        quantile_alpha: float = 0.75,
        huber_alpha: float = 1.35,
        # ── Meta-labeling (item 8) ────────────────────────────────────────
        use_meta_labeling: bool = False,
        # ── IC-based early stopping (item 9) ──────────────────────────────
        early_stop_on_ic: bool = True,
        # ── Min data in leaf (item 10) ────────────────────────────────────
        min_data_in_leaf: int = 200,
        # ── Monotonicity constraints (item 11) ────────────────────────────
        monotone_constraints: Optional[Dict[str, int]] = None,
        # ── Lopez de Prado sample weights (items 12, 13) ──────────────────
        use_uniqueness_weights: bool = True,
        use_per_date_weights: bool = True,
        # ── LGBM seed bagging (item 15) ───────────────────────────────────
        lgbm_num_seeds: int = 1,
        # ── Global random seed for all stochastic components ──────────────
        random_state: int = 42,
        # ── Embargo buffer between train and test folds (AFML Ch. 7.4) ────
        # Default equals typical forward_window=5 to avoid test-side
        # serial-correlation leak from purged train samples.
        embargo_days: int = 5,
        # ── Opt 3: adversarial validation gate ────────────────────────────
        # Trains a full LGBM classifier per window (~25 min total).
        # AUC=1.0 every time for temporal walk-forward (trivially separable).
        # Default OFF; keep code path for users who want distribution-shift logging.
        run_adversarial_validation: bool = False,
        # ── Opt 5: LGBM warm-start (incremental training) ────────────────
        # When ON, subsequent walk-forward windows init from the prior
        # window's booster, reducing num_boost_round to 100 (vs 300 cold).
        # Default OFF until validated via test suite.
        warm_start_lgbm: bool = False,
        # ── Opt 6: LGBM feature interaction constraints ───────────────────
        # Restricts which feature groups can interact in tree splits.
        # None = no constraints (backward compatible).
        # "auto" = auto-detect groups from feature name prefixes.
        # List[List[int]] = explicit constraint groups.
        interaction_constraints: object = None,  # None | "auto" | List[List[int]]
    ):
        self.min_train_days    = min_train_days
        self.retrain_freq      = retrain_freq
        self.forward_window    = forward_window
        self.risk_adjust       = risk_adjust
        self.sector_rank_weight = sector_rank_weight
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.num_leaves        = num_leaves
        self.feature_fraction  = feature_fraction
        self.bagging_fraction  = bagging_fraction
        self.min_child_samples = min_child_samples
        self.lambda_l1         = lambda_l1
        self.lambda_l2         = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.max_depth         = max_depth
        self.ridge_alpha       = ridge_alpha
        self.lgbm_weight       = lgbm_weight
        self.xgb_weight   = xgb_weight
        self.ridge_weight      = ridge_weight
        self.mlp_weight        = mlp_weight
        self.prune_features    = prune_features
        self.min_ic            = min_ic
        self.max_feature_corr  = max_feature_corr
        self.max_train_days    = max_train_days
        self.neutralize_vol    = neutralize_vol
        self.use_mlp           = use_mlp
        # New params (items 1-15)
        self.lambdarank_truncation = lambdarank_truncation
        self.use_return_decile_gain = use_return_decile_gain
        self.use_magnitude_weights = use_magnitude_weights
        self.huber_weight      = huber_weight
        self.quantile_weight   = quantile_weight
        self.quantile_alpha    = quantile_alpha
        self.huber_alpha       = huber_alpha
        self.use_meta_labeling = use_meta_labeling
        self.early_stop_on_ic  = early_stop_on_ic
        self.min_data_in_leaf  = min_data_in_leaf
        self.monotone_constraints = monotone_constraints if monotone_constraints is not None else _default_monotone_constraints()
        self.use_uniqueness_weights = use_uniqueness_weights
        self.use_per_date_weights = use_per_date_weights
        self.lgbm_num_seeds    = max(1, int(lgbm_num_seeds))
        self.random_state      = int(random_state)
        # AFML Ch. 7.4: embargo buffer on the test side of each fold.
        self.embargo_days      = max(0, int(embargo_days))
        self.run_adversarial_validation = run_adversarial_validation  # Opt 3
        self.warm_start_lgbm   = warm_start_lgbm                     # Opt 5
        self.interaction_constraints_cfg = interaction_constraints    # Opt 6

        # If MLP disabled, zero its weight and renormalize the other model
        # weights so the ensemble still sums to 1.0. User-configured
        # lgbm/xgb/ridge weights are preserved proportionally.
        if not self.use_mlp:
            self.mlp_weight = 0.0
            _other = self.lgbm_weight + self.xgb_weight + self.ridge_weight
            if _other > 0:
                self.lgbm_weight  = self.lgbm_weight  / _other
                self.xgb_weight   = self.xgb_weight   / _other
                self.ridge_weight = self.ridge_weight / _other

        self.models_: List[dict] = []
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Individual learner trainers
    # ------------------------------------------------------------------

    def _build_monotone_list(self, feature_names: List[str]) -> Optional[List[int]]:
        """Build per-feature monotone-constraint list from self.monotone_constraints."""
        if not self.monotone_constraints:
            return None
        mc = self.monotone_constraints
        out = []
        any_nonzero = False
        for fn in feature_names:
            v = int(mc.get(fn, 0))
            if v != 0:
                any_nonzero = True
            out.append(v)
        return out if any_nonzero else None

    def _spearman_ic_metric(self, preds, dtrain):
        """Custom LightGBM feval: Spearman rank IC (higher is better)."""
        labels = dtrain.get_label()
        try:
            p = pd.Series(preds)
            l = pd.Series(labels)
            ic = p.corr(l, method="spearman")
            if not np.isfinite(ic):
                ic = 0.0
        except Exception:
            ic = 0.0
        return "ic", float(ic), True

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray,
                  groups: np.ndarray, feature_names: List[str],
                  sample_weight: Optional[np.ndarray] = None,
                  init_model=None):  # Opt 5: warm-start from prior window's booster
        if not LGBM_AVAILABLE:
            return None

        # Carve out last 21 days as temporal holdout for early stopping.
        # groups[i] = number of tickers on day i, so we split by group count.
        val_days  = min(21, len(groups) // 5)
        val_rows  = int(groups[-val_days:].sum())
        X_tr, X_val = X[:-val_rows], X[-val_rows:]
        y_tr, y_val = y[:-val_rows], y[-val_rows:]
        g_tr, g_val = groups[:-val_days], groups[-val_days:]
        w_tr = sample_weight[:-val_rows] if sample_weight is not None else None

        # ── Label-gain schedule (item 2) ────────────────────────────────────
        if self.use_return_decile_gain:
            # 31-level decile scheme: exp(i/5) gain, requires labels 0..30
            label_gain = [float(np.exp(i / 5.0)) for i in range(31)]
        else:
            label_gain = [0, 0, 1, 5, 20, 100]

        dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feature_names,
                             weight=w_tr)
        dval   = lgb.Dataset(X_val, label=y_val, group=g_val, feature_name=feature_names,
                             reference=dtrain)
        params = {
            "objective":         "lambdarank",
            "metric":            "ndcg",
            # Item 1: truncation @20, eval at [5,10,20]
            "ndcg_eval_at":      [5, 10, self.lambdarank_truncation],
            "lambdarank_truncation_level": self.lambdarank_truncation,
            "label_gain":        label_gain,
            "num_leaves":        self.num_leaves,
            "learning_rate":     self.learning_rate,
            "feature_fraction":  self.feature_fraction,
            "bagging_fraction":  self.bagging_fraction,
            "bagging_freq":      1,
            "min_child_samples": self.min_child_samples,
            # Item 10: min_data_in_leaf — equity cross-section is noisy; larger
            # leaves prevent the model from fitting per-day noise.
            "min_data_in_leaf":  self.min_data_in_leaf,
            # Regularization: previously zero (no leaf weight penalty).
            # L1 encourages sparse leaf outputs, L2 shrinks all leaf weights.
            # Prevents overfitting to noise in cross-sectional financial data.
            # Values calibrated per Qlib benchmark + practitioner literature.
            "lambda_l1":         self.lambda_l1,
            "lambda_l2":         self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,  # prevent splits on noise
            "max_depth":         self.max_depth,           # secondary guard on leaf-wise depth
            "num_threads":       -1,
            "verbose":           -1,
            "device":            "gpu" if TORCH_AVAILABLE else "cpu",
            # Deterministic seeding across all stochastic LGBM components
            "seed":                     self.random_state,
            "bagging_seed":             self.random_state,
            "feature_fraction_seed":    self.random_state,
            "data_random_seed":         self.random_state,
        }
        # Item 11: monotonicity constraints on signed factors
        mc_list = self._build_monotone_list(feature_names)
        if mc_list is not None:
            params["monotone_constraints"] = mc_list

        # Opt 6: feature interaction constraints
        ic_cfg = self.interaction_constraints_cfg
        if ic_cfg is not None:
            if ic_cfg == "auto":
                params["interaction_constraints"] = build_interaction_constraints(feature_names)
            elif isinstance(ic_cfg, list):
                params["interaction_constraints"] = ic_cfg

        # Item 9: IC-based early stopping
        feval = self._spearman_ic_metric if self.early_stop_on_ic else None
        first_metric_only = bool(self.early_stop_on_ic)
        callbacks = [
            lgb.early_stopping(50, verbose=False, first_metric_only=first_metric_only),
            lgb.log_evaluation(-1),
        ]

        # Opt 5: warm-start reduces boost rounds (incremental learning on
        # top of prior window's trees; Ke et al. 2017 LightGBM paper)
        _num_boost = 100 if init_model is not None else self.n_estimators

        def _train_one(seed: int):
            p = dict(params)
            p["seed"] = seed
            p["bagging_seed"] = seed
            p["feature_fraction_seed"] = seed
            p["data_random_seed"] = seed
            return lgb.train(
                p, dtrain,
                num_boost_round=_num_boost,
                valid_sets=[dval],
                feval=feval,
                callbacks=callbacks,
                init_model=init_model,  # Opt 5: warm-start from prior booster
            )

        # Item 15: seed-bagged LGBM ensemble
        if self.lgbm_num_seeds > 1:
            booster_list = [
                _train_one(self.random_state + k)
                for k in range(self.lgbm_num_seeds)
            ]
            return booster_list
        return _train_one(self.random_state)

    def _fit_lgbm_huber(self, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str],
                        sample_weight: Optional[np.ndarray] = None):
        """Train a Huber-loss LGBM regression head (item 4)."""
        if not LGBM_AVAILABLE:
            return None
        try:
            split = max(1, int(len(X) * 0.85))
            w_tr = sample_weight[:split] if sample_weight is not None else None
            dtrain = lgb.Dataset(X[:split], label=y[:split].astype(float),
                                 feature_name=feature_names, weight=w_tr)
            dval = lgb.Dataset(X[split:], label=y[split:].astype(float),
                               feature_name=feature_names, reference=dtrain)
            params = {
                "objective": "huber",
                "alpha": float(self.huber_alpha),
                "metric": "huber",
                "seed": self.random_state,
                "bagging_seed": self.random_state,
                "feature_fraction_seed": self.random_state,
                "data_random_seed": self.random_state,
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": 1,
                "min_child_samples": self.min_child_samples,
                "min_data_in_leaf": self.min_data_in_leaf,
                "lambda_l1": self.lambda_l1,
                "lambda_l2": self.lambda_l2,
                "min_gain_to_split": self.min_gain_to_split,
                "max_depth": self.max_depth,
                "num_threads": -1,
                "verbose": -1,
            }
            mc_list = self._build_monotone_list(feature_names)
            if mc_list is not None:
                params["monotone_constraints"] = mc_list
            return lgb.train(
                params, dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
        except Exception:
            return None

    def _fit_lgbm_quantile(self, X: np.ndarray, y: np.ndarray,
                           feature_names: List[str],
                           alpha: float = 0.75,
                           sample_weight: Optional[np.ndarray] = None):
        """Train a quantile-regression LGBM head at the given alpha (item 5)."""
        if not LGBM_AVAILABLE:
            return None
        try:
            split = max(1, int(len(X) * 0.85))
            w_tr = sample_weight[:split] if sample_weight is not None else None
            dtrain = lgb.Dataset(X[:split], label=y[:split].astype(float),
                                 feature_name=feature_names, weight=w_tr)
            dval = lgb.Dataset(X[split:], label=y[split:].astype(float),
                               feature_name=feature_names, reference=dtrain)
            params = {
                "objective": "quantile",
                "alpha": float(alpha),
                "metric": "quantile",
                "seed": self.random_state,
                "bagging_seed": self.random_state,
                "feature_fraction_seed": self.random_state,
                "data_random_seed": self.random_state,
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": 1,
                "min_child_samples": self.min_child_samples,
                "min_data_in_leaf": self.min_data_in_leaf,
                "lambda_l1": self.lambda_l1,
                "lambda_l2": self.lambda_l2,
                "min_gain_to_split": self.min_gain_to_split,
                "max_depth": self.max_depth,
                "num_threads": -1,
                "verbose": -1,
            }
            mc_list = self._build_monotone_list(feature_names)
            if mc_list is not None:
                params["monotone_constraints"] = mc_list
            return lgb.train(
                params, dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
        except Exception:
            return None

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None):
        # Fix 3 (LOW-polish): Ridge's L2 penalty is scale-dependent, so
        # feature magnitudes that differ wildly (e.g. log_mcap ~20 vs
        # z-scored features ~O(1)) get effectively different regularization.
        # Even though Tier 4 added winsorize + cs_zscore_all upstream, we
        # apply an explicit StandardScaler inside Ridge for belt-and-braces
        # safety. The returned object is a dict {"model", "scaler"} — the
        # predict path must apply the scaler before calling .predict.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        model = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        model.fit(X_scaled, y, sample_weight=sample_weight)
        return {"model": model, "scaler": scaler}

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, feature_names: List[str]):
        """Train XGBoost ranker on GPU. No group size limit unlike CatBoost."""
        if not XGB_AVAILABLE:
            return None
        try:
            val_days = min(21, len(groups) // 5)
            val_rows = int(groups[-val_days:].sum())
            train_rows = int(groups[:-val_days].sum())

            # Groups must exactly match row count
            dtrain = xgb.DMatrix(X[:train_rows], label=y[:train_rows],
                                 feature_names=feature_names)
            dtrain.set_group(groups[:-val_days].astype(int))

            dval = xgb.DMatrix(X[train_rows:train_rows + val_rows],
                               label=y[train_rows:train_rows + val_rows],
                               feature_names=feature_names)
            dval.set_group(groups[-val_days:].astype(int))

            params = {
                "objective": "rank:ndcg",
                "eval_metric": f"ndcg@{self.lambdarank_truncation}",
                "tree_method": "hist",
                "device": "cuda" if TORCH_AVAILABLE else "cpu",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                # Item 1: align XGB pair sampling with LGBM truncation
                "lambdarank_num_pair_per_sample": int(self.lambdarank_truncation),
                "lambdarank_pair_method": "topk",
                "verbosity": 0,
                "seed": self.random_state,
            }

            model = xgb.train(
                params, dtrain,
                num_boost_round=200,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            return model
        except Exception:
            return None

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray,
                 dates: Optional[pd.DatetimeIndex] = None,
                 X_raw: Optional[np.ndarray] = None) -> Optional[object]:
        """
        Train a 3-layer MLP on GPU for cross-sectional stock ranking.

        Architecture: input → 128 → 64 → 32 → 1 with BatchNorm + Dropout.
        Learns different feature interactions than tree-based models,
        providing genuine ensemble diversity.

        Fix 1 (rank-scaled targets): `y` here arrives as raw integer grade
        labels, but MSE on categorical targets cannot learn a ranking. We
        convert `y` to per-date cross-sectional rank-pct in [0,1] before
        training so the MLP optimizes a monotone transformation of the
        ensemble's ranking objective.

        Fix 2 (train-only imputation): the caller passes `X_raw` (pre-impute
        matrix) so we can compute medians on the training slice only and
        re-impute val with those medians, eliminating the minor median-leak
        that arose when medians were computed on the full matrix.

        Returns a dict with 'model' and 'scaler' for prediction.
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            from sklearn.preprocessing import StandardScaler

            # Deterministic torch RNG for reproducible MLP training
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

            split_mlp = int(len(X) * 0.8)

            # ── Fix 2: recompute medians on training slice only ────────────
            # If `X_raw` (pre-impute) is provided, we can recompute
            # train-only medians and apply to val. Otherwise fall back to
            # the already-imputed `X` (legacy behavior).
            if X_raw is not None and len(X_raw) == len(X):
                Xr = np.asarray(X_raw, dtype=float).copy()
                Xr[~np.isfinite(Xr)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    train_medians = np.nanmedian(Xr[:split_mlp], axis=0)
                train_medians = np.where(
                    np.isfinite(train_medians), train_medians, 0.0
                )
                X_tr_imp = Xr[:split_mlp].copy()
                X_val_imp = Xr[split_mlp:].copy()
                idx_tr = np.where(np.isnan(X_tr_imp))
                if len(idx_tr[0]):
                    X_tr_imp[idx_tr] = np.take(train_medians, idx_tr[1])
                idx_val = np.where(np.isnan(X_val_imp))
                if len(idx_val[0]):
                    X_val_imp[idx_val] = np.take(train_medians, idx_val[1])
            else:
                X_tr_imp = X[:split_mlp]
                X_val_imp = X[split_mlp:]

            # Scale features — fit ONLY on train split to avoid leaking
            # val statistics into training (scaler sees mean/std of val data).
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_imp)
            X_val_scaled = scaler.transform(X_val_imp)

            # ── Fix 1: convert integer grade labels to per-date rank-pct ────
            # MSE on categorical integers doesn't learn ranking. Replace
            # targets with cross-sectional rank-pct per date (values in
            # [0,1]), which aligns the MLP head with the ranker objective.
            y_float = np.asarray(y, dtype=float)
            if dates is not None and len(dates) == len(y_float):
                try:
                    y_ser = pd.Series(
                        y_float, index=pd.DatetimeIndex(dates, name="date")
                    )
                    y_rank = y_ser.groupby(level="date").rank(pct=True).values
                    # Any all-NaN per-date groups fall back to 0.5 (neutral)
                    y_rank = np.where(np.isfinite(y_rank), y_rank, 0.5)
                    y_float = y_rank.astype(float)
                except Exception:
                    pass

            X_tr = torch.FloatTensor(X_tr_scaled).cuda()
            y_tr = torch.FloatTensor(y_float[:split_mlp]).cuda()
            X_val = torch.FloatTensor(X_val_scaled).cuda()
            y_val = torch.FloatTensor(y_float[split_mlp:]).cuda()

            n_features = X_tr.shape[1]

            # Simple but effective architecture
            model = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ).cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            # Train with early stopping
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            model.train()
            for epoch in range(100):
                # Mini-batch training
                perm = torch.randperm(len(X_tr))
                batch_size = min(4096, len(X_tr))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(X_tr), batch_size):
                    idx = perm[i:i + batch_size]
                    pred = model(X_tr[idx]).squeeze()
                    loss = loss_fn(pred, y_tr[idx])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val).squeeze()
                    val_loss = loss_fn(val_pred, y_val).item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Restore best model — keep on CPU to free GPU VRAM.
            # Move to GPU only during prediction.
            model.load_state_dict(best_state)
            model.cpu()
            model.eval()

            return {"model": model, "scaler": scaler}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Feature neutralization (regress out vol)
    # ------------------------------------------------------------------

    def _neutralize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional vol neutralization: for each date, regress each
        feature on realized vol, keep the residual. This forces the model
        to find signal orthogonal to the volatility factor.

        Reference: Lopez de Prado (2018), WorldQuant signal orthogonalization.
        """
        vol_col = None
        for candidate in ["rvol_21d", "rank_rvol_21d", "rvol_63d"]:
            if candidate in X.columns:
                vol_col = candidate
                break

        if vol_col is None:
            return X  # no vol feature found, skip

        X_out = X.copy()
        vol = X[vol_col].values
        vol_valid = np.isfinite(vol) & (vol != 0)

        # Skip the vol columns themselves — don't neutralize vol against vol
        vol_prefixes = ("rvol_", "rank_rvol", "idiovol_", "rank_idiovol",
                        "vol_of_vol", "rank_vol_of_vol", "tail_risk",
                        "rank_tail_risk", "vol_reversal", "rank_vol_reversal",
                        "z_rvol_", "z_idiovol_", "max_ret_", "rank_max_effect",
                        "vol_regime", "rank_vol_regime", "vol_spike")

        for col in X.columns:
            if any(col.startswith(p) for p in vol_prefixes):
                continue
            vals = X[col].values
            mask = vol_valid & np.isfinite(vals)
            if mask.sum() < 100:
                continue
            # Simple OLS: feature_i = alpha + beta * vol + residual
            v = vol[mask]
            f = vals[mask]
            v_mean = v.mean()
            f_mean = f.mean()
            beta = np.dot(v - v_mean, f - f_mean) / max(np.dot(v - v_mean, v - v_mean), 1e-10)
            residual = vals - beta * vol
            X_out[col] = residual

        return X_out

    # ------------------------------------------------------------------
    # Feature pruning (IC-threshold)
    # ------------------------------------------------------------------

    def _prune_features(self, X: pd.DataFrame, y: pd.Series,
                        max_samples: int = 200_000) -> List[str]:
        """
        Select features with meaningful IC, removing highly correlated duplicates.
        Keeps ~40-60 features from 90+.

        Uses random subsample for IC computation — 200K samples gives ±0.002
        IC precision which is more than enough for feature selection at IC > 0.01.
        """
        if not self.prune_features:
            return list(X.columns)

        # Subsample for speed: IC converges well at 200K rows
        if len(X) > max_samples:
            idx = np.random.default_rng(self.random_state).choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X
            y_sample = y

        # Compute rank IC for each feature against labels
        ics = {}
        for col in X_sample.columns:
            try:
                common = X_sample[col].dropna().index.intersection(y_sample.dropna().index)
                if len(common) < 100:
                    continue
                ic = X_sample[col].loc[common].corr(y_sample.loc[common], method="spearman")
                if np.isfinite(ic):
                    ics[col] = abs(ic)
            except Exception:
                pass

        if not ics:
            return list(X.columns)

        # Filter by minimum IC
        passing = {k: v for k, v in ics.items() if v >= self.min_ic}
        if len(passing) < 20:
            # Not enough — relax threshold, keep top 40
            passing = dict(sorted(ics.items(), key=lambda x: -x[1])[:40])

        # Remove correlated features via hierarchical clustering.
        # Groups highly-correlated features into clusters and keeps the
        # highest-IC representative from each cluster. This is order-
        # independent and captures transitive correlations (A~B and B~C
        # → all three in one cluster), unlike greedy pairwise pruning.
        # (Lopez de Prado, "Advances in Financial Machine Learning")
        feat_list = list(passing.keys())
        if len(feat_list) <= 15:
            return feat_list

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # Spearman correlation → distance: d = sqrt(0.5 * (1 - rho))
            sub = X_sample[feat_list].dropna(axis=0, how="all")
            if len(sub) < 100:
                return feat_list
            corr = sub.corr(method="spearman").fillna(0)
            dist = np.sqrt(0.5 * (1 - corr.values))
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, None)  # numerical safety

            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")

            # Cut threshold from max_feature_corr: corr=0.70 → dist=0.387
            dist_thresh = np.sqrt(0.5 * (1 - self.max_feature_corr))
            labels = fcluster(Z, t=dist_thresh, criterion="distance")

            # Pick highest-IC representative per cluster
            selected = []
            for cluster_id in sorted(set(labels)):
                members = [feat_list[i] for i, l in enumerate(labels) if l == cluster_id]
                best = max(members, key=lambda f: passing.get(f, 0))
                selected.append(best)

        except Exception:
            # Fallback to greedy if scipy unavailable
            sorted_feats = sorted(passing.items(), key=lambda x: -x[1])
            selected = []
            for feat, ic in sorted_feats:
                if feat not in X_sample.columns:
                    continue
                is_redundant = False
                for existing in selected:
                    try:
                        if abs(X_sample[feat].corr(X_sample[existing])) > self.max_feature_corr:
                            is_redundant = True
                            break
                    except Exception:
                        pass
                if not is_redundant:
                    selected.append(feat)

        return selected if len(selected) >= 15 else list(X.columns)

    # ------------------------------------------------------------------
    # Ensemble train / predict
    # ------------------------------------------------------------------

    def _train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                        max_train_samples: int = 500_000,
                        lgbm_init_model=None) -> Optional[dict]:  # Opt 5: warm-start
        # Only require a valid label — LGBM/XGB natively handle NaN features,
        # Ridge/MLP are median-imputed below. Dropping rows with any NaN
        # feature would discard most warmup-period samples.
        mask = y_train.notna()
        X_ = X_train[mask]
        y_ = y_train[mask].astype(int)

        if len(X_) < 50:
            return None

        # ── Rolling window: drop old data ────────────────────────────────
        # Expanding windows accumulate stale regime data. A 3-year rolling
        # window keeps training data relevant to current market structure.
        if self.max_train_days is not None:
            dates = X_.index.get_level_values("date")
            unique = dates.unique().sort_values()
            if len(unique) > self.max_train_days:
                cutoff = unique[-self.max_train_days]
                keep = dates >= cutoff
                X_ = X_[keep]
                y_ = y_[keep]

        # ── Feature neutralization: regress out vol ──────────────────────
        if self.neutralize_vol:
            X_ = self._neutralize_features(X_)

        # ── Feature pruning ──────────────────────────────────────────────
        selected_features = self._prune_features(X_, y_)
        X_ = X_[selected_features]

        # Cap training set size for speed. Keep the MOST RECENT samples
        # (temporal weighting already down-weights old data, so dropping
        # the oldest samples loses minimal information).
        if len(X_) > max_train_samples:
            X_ = X_.iloc[-max_train_samples:]
            y_ = y_.iloc[-max_train_samples:]

        # LambdaRank integrity: `groups` is built from
        # dates.value_counts().sort_index() which assumes X_ rows are in
        # date-sorted order. A (ticker, date) MultiIndex can produce
        # ticker-first ordering, which mis-aligns group boundaries with
        # actual cross-sections. Sort by date (stable) so groups match rows.
        if isinstance(X_.index, pd.MultiIndex):
            X_ = X_.sort_index(level="date", sort_remaining=True)
            y_ = y_.reindex(X_.index)

        feature_names         = list(X_.columns)
        ensemble: dict        = {"feature_subset": selected_features}

        # Temporal sample weighting
        dates = X_.index.get_level_values("date")
        unique_dates = dates.unique().sort_values()
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        max_idx = len(unique_dates) - 1
        sample_idx = np.array([date_to_idx[d] for d in dates])
        decay_lambda = np.log(2) / 630
        sample_weight = np.exp(-decay_lambda * (max_idx - sample_idx))

        # ── Per-date equal weighting (item 13) ──────────────────────────────
        if self.use_per_date_weights:
            counts = dates.value_counts().to_dict()
            n_per_row = np.array([counts.get(d, 1) for d in dates], dtype=float)
            pdw = 1.0 / np.maximum(n_per_row, 1.0)
            pdw = pdw / pdw.mean() if pdw.mean() > 0 else pdw
            sample_weight = sample_weight * pdw

        # ── Uniqueness/concurrency weights (item 12, AFML Ch. 4.3) ──────────
        if self.use_uniqueness_weights:
            try:
                uw = self._compute_uniqueness_weights(
                    pd.DatetimeIndex(dates), self.forward_window
                )
                uw = uw / uw.mean() if uw.mean() > 0 else uw
                sample_weight = sample_weight * uw
            except Exception:
                pass

        # ── Magnitude weights (item 3) ──────────────────────────────────────
        if self.use_magnitude_weights:
            # Use absolute cross-sectional deviation of label from its per-date
            # median as a proxy for |forward_return|. Normalized to mean 1 per date.
            try:
                y_series = pd.Series(y_.values, index=dates)
                med = y_series.groupby(level=0).transform("median")
                mag = (y_series - med).abs()
                mu_per_date = mag.groupby(level=0).transform("mean").replace(0, np.nan)
                mag_norm = (mag / mu_per_date).fillna(1.0).values
                sample_weight = sample_weight * mag_norm
            except Exception:
                pass

        # ── Decile labels for return-decile label gain (item 2) ─────────────
        if self.use_return_decile_gain:
            try:
                y_series = pd.Series(y_.values.astype(float), index=dates)
                # Cross-sectional rank per date, scaled to 0..30 integer deciles
                rnk = y_series.groupby(level=0).rank(pct=True).fillna(0.5)
                y_lgbm = np.clip((rnk.values * 30.0).round().astype(int), 0, 30)
            except Exception:
                y_lgbm = y_.values.astype(int)
        else:
            y_lgbm = y_.values.astype(int)

        X_vals, y_vals = X_.values, y_.values
        # Ridge / MLP / StandardScaler cannot handle NaN. Impute per-column
        # median once, reuse for all non-tree heads. LGBM/XGB receive raw
        # X_vals with NaN preserved (tree-native handling).
        X_imputed = _impute_for_ridge(X_vals)

        groups = dates.value_counts().sort_index().values

        if LGBM_AVAILABLE:
            lgbm = self._fit_lgbm(X_vals, y_lgbm, groups, feature_names,
                                  sample_weight=sample_weight,
                                  init_model=lgbm_init_model)  # Opt 5: warm-start
            if lgbm is not None:
                ensemble["lgbm"] = lgbm

        if XGB_AVAILABLE:
            xgb_model = self._fit_xgboost(X_vals, y_lgbm, groups, feature_names)
            if xgb_model is not None:
                ensemble["xgboost"] = xgb_model

        ensemble["ridge"] = self._fit_ridge(X_imputed, y_vals,
                                            sample_weight=sample_weight)

        # ── Huber head (item 4) ─────────────────────────────────────────────
        if self.huber_weight > 0 and LGBM_AVAILABLE:
            huber = self._fit_lgbm_huber(X_vals, y_vals.astype(float),
                                         feature_names, sample_weight=sample_weight)
            if huber is not None:
                ensemble["huber"] = huber

        # ── Quantile head (item 5) ──────────────────────────────────────────
        if self.quantile_weight > 0 and LGBM_AVAILABLE:
            qmod = self._fit_lgbm_quantile(X_vals, y_vals.astype(float),
                                           feature_names,
                                           alpha=self.quantile_alpha,
                                           sample_weight=sample_weight)
            if qmod is not None:
                ensemble["quantile"] = qmod

        if self.use_mlp and TORCH_AVAILABLE:
            # Pass raw (pre-impute) X and dates so MLP can (a) compute
            # train-only imputation medians (Fix 2) and (b) convert
            # integer labels to per-date rank-pct targets (Fix 1).
            mlp = self._fit_mlp(X_imputed, y_vals,
                                dates=dates, X_raw=X_vals)
            if mlp is not None:
                ensemble["mlp"] = mlp
            # Free GPU memory after MLP training
            torch.cuda.empty_cache()

        # ── Meta-labeling stage-2 classifier (item 8) ───────────────────────
        if self.use_meta_labeling and LGBM_AVAILABLE and "lgbm" in ensemble:
            try:
                # Get primary predictions on train set
                primary_preds = self._predict_single_lgbm(ensemble["lgbm"], X_vals)
                meta = self.fit_meta_labeler(X_, y_, primary_preds, feature_names)
                if meta is not None:
                    ensemble["meta"] = meta
            except Exception:
                pass

        return ensemble

    # ------------------------------------------------------------------
    # Helpers for uniqueness weights / meta-labeling / adversarial val
    # ------------------------------------------------------------------

    def _compute_uniqueness_weights(self, label_dates: pd.DatetimeIndex,
                                     forward_window: int) -> np.ndarray:
        """AFML Ch. 4.3 uniqueness: mean(1/concurrency) over each label span.

        For each observation dated t, the label spans [t, t+forward_window).
        concurrency[d] = number of observations whose span covers day d.
        uniqueness_i = mean_{d in span_i} 1/concurrency[d].
        """
        dates = pd.DatetimeIndex(label_dates)
        unique_dates = dates.unique().sort_values()
        date_to_pos = {d: i for i, d in enumerate(unique_dates)}
        n_dates = len(unique_dates)
        # concurrency per date: count of rows whose span [t, t+w) covers this date
        per_date_count = np.zeros(n_dates, dtype=float)
        date_counts = dates.value_counts()
        obs_per_date = np.array([date_counts.get(d, 0) for d in unique_dates], dtype=float)
        # Rolling forward_window sum of observations indicates concurrency
        for i in range(n_dates):
            lo = max(0, i - forward_window + 1)
            per_date_count[i] = obs_per_date[lo:i + 1].sum()
        per_date_count = np.maximum(per_date_count, 1.0)
        inv_conc = 1.0 / per_date_count
        # uniqueness_i = mean over [pos_i, pos_i+w) of inv_conc (clipped to end)
        pos = np.array([date_to_pos[d] for d in dates])
        out = np.zeros(len(dates), dtype=float)
        for j, p in enumerate(pos):
            hi = min(n_dates, p + forward_window)
            out[j] = inv_conc[p:hi].mean() if hi > p else 1.0
        return out

    def _predict_single_lgbm(self, lgbm_obj, X_vals: np.ndarray) -> np.ndarray:
        """Predict using either single booster or seed-bagged list."""
        if isinstance(lgbm_obj, list):
            return np.mean([b.predict(X_vals) for b in lgbm_obj], axis=0)
        return lgbm_obj.predict(X_vals)

    def fit_meta_labeler(self, train_X: pd.DataFrame, train_y: pd.Series,
                         primary_preds: np.ndarray,
                         feature_names: List[str]):
        """Stage-2 binary classifier: is primary top-quintile AND y > median?

        At predict time, multiply primary score by meta-labeler probability
        to size bets by confidence. (AFML Ch. 3)
        """
        if not LGBM_AVAILABLE:
            return None
        try:
            dates = train_X.index.get_level_values("date")
            p_series = pd.Series(primary_preds, index=train_X.index)
            rank_primary = p_series.groupby(dates).rank(pct=True)
            y_series = pd.Series(train_y.values, index=train_X.index)
            y_median = y_series.groupby(dates).transform("median")
            y_binary = ((rank_primary >= 0.80) & (y_series > y_median)).astype(int).values
            if y_binary.sum() < 50 or y_binary.sum() >= len(y_binary) - 50:
                return None
            split = max(1, int(len(train_X) * 0.85))
            dtrain = lgb.Dataset(train_X.values[:split], label=y_binary[:split],
                                 feature_name=feature_names)
            dval = lgb.Dataset(train_X.values[split:], label=y_binary[split:],
                               feature_name=feature_names, reference=dtrain)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "seed": self.random_state,
                "bagging_seed": self.random_state,
                "feature_fraction_seed": self.random_state,
                "data_random_seed": self.random_state,
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": 1,
                "min_child_samples": self.min_child_samples,
                "min_data_in_leaf": self.min_data_in_leaf,
                "lambda_l1": self.lambda_l1,
                "lambda_l2": self.lambda_l2,
                "verbose": -1,
                "num_threads": -1,
            }
            return lgb.train(
                params, dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
        except Exception:
            return None

    def adversarial_validation(self, train_X: pd.DataFrame,
                                test_X: pd.DataFrame) -> float:
        """Train a binary LGBM to distinguish train vs test; return AUC.

        AUC >> 0.5 signals distribution shift. Called per window for logging.
        """
        if not LGBM_AVAILABLE:
            return float("nan")
        try:
            from sklearn.metrics import roc_auc_score
            common_cols = [c for c in train_X.columns if c in test_X.columns]
            # LGBM handles NaN natively — pass inf→NaN only, no fillna
            Xtr = train_X[common_cols].replace([np.inf, -np.inf], np.nan).values
            Xte = test_X[common_cols].replace([np.inf, -np.inf], np.nan).values
            X = np.vstack([Xtr, Xte])
            y = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))])
            perm = np.random.default_rng(self.random_state).permutation(len(X))
            X = X[perm]; y = y[perm]
            split = int(0.8 * len(X))
            dtrain = lgb.Dataset(X[:split], label=y[:split], feature_name=common_cols)
            dval = lgb.Dataset(X[split:], label=y[split:], feature_name=common_cols,
                               reference=dtrain)
            params = {
                "objective": "binary", "metric": "auc",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                "verbose": -1, "num_threads": -1,
                "seed": self.random_state,
                "bagging_seed": self.random_state,
                "feature_fraction_seed": self.random_state,
                "data_random_seed": self.random_state,
            }
            booster = lgb.train(
                params, dtrain, num_boost_round=200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = booster.predict(X[split:])
            return float(roc_auc_score(y[split:], preds))
        except Exception:
            return float("nan")

    def _predict_ensemble(self, X_pred: pd.DataFrame, ensemble: dict) -> tuple:
        """
        Returns (blended_scores, confidence).

        Uses feature subset from training for prediction alignment.
        Confidence is average pairwise agreement across all models.
        """
        # Use only the features this ensemble was trained on.
        # fill_value=0 applies ONLY to columns that are entirely missing
        # from X_pred (shouldn't happen if feature matrix is stable).
        # Actual NaN values in present columns are preserved → LGBM/XGB
        # handle them natively; Ridge/MLP get a median-imputed copy below.
        feature_subset = ensemble.get("feature_subset", list(X_pred.columns))
        X_aligned = X_pred.reindex(columns=feature_subset, fill_value=0)
        X_vals = X_aligned.values
        _X_imputed_cache = {"arr": None}

        def _get_imputed() -> np.ndarray:
            if _X_imputed_cache["arr"] is None:
                _X_imputed_cache["arr"] = _impute_for_ridge(X_vals)
            return _X_imputed_cache["arr"]

        def ranked(raw: np.ndarray) -> np.ndarray:
            return pd.Series(raw).rank(pct=True).values

        model_ranks = []
        scores  = np.zeros(len(X_vals))
        total_w = 0.0

        if "lgbm" in ensemble:
            lgbm_obj = ensemble["lgbm"]
            if isinstance(lgbm_obj, list):
                # Seed-bagged ensemble: average predictions before ranking
                raw = np.mean([b.predict(X_vals) for b in lgbm_obj], axis=0)
            else:
                raw = lgbm_obj.predict(X_vals)
            lgbm_ranks = ranked(raw)
            scores  += lgbm_ranks * self.lgbm_weight
            total_w += self.lgbm_weight
            model_ranks.append(lgbm_ranks)

        if "xgboost" in ensemble:
            dmat = xgb.DMatrix(X_vals, feature_names=ensemble.get("feature_subset", None))
            # Use best_iteration from early stopping (not last iteration) to
            # avoid scoring with over-fit trees trained past the validation peak.
            best_iter = getattr(ensemble["xgboost"], "best_iteration", None)
            if best_iter is not None:
                xgb_raw = ensemble["xgboost"].predict(
                    dmat, iteration_range=(0, int(best_iter) + 1)
                )
            else:
                xgb_raw = ensemble["xgboost"].predict(dmat)
            xgb_ranks = ranked(xgb_raw)
            scores  += xgb_ranks * self.xgb_weight
            total_w += self.xgb_weight
            model_ranks.append(xgb_ranks)

        # Huber head (item 4)
        if "huber" in ensemble and self.huber_weight > 0:
            huber_ranks = ranked(ensemble["huber"].predict(X_vals))
            scores += huber_ranks * self.huber_weight
            total_w += self.huber_weight
            model_ranks.append(huber_ranks)

        # Quantile head (item 5)
        if "quantile" in ensemble and self.quantile_weight > 0:
            q_ranks = ranked(ensemble["quantile"].predict(X_vals))
            scores += q_ranks * self.quantile_weight
            total_w += self.quantile_weight
            model_ranks.append(q_ranks)

        # MLP predictions (GPU)
        if "mlp" in ensemble:
            try:
                mlp_model = ensemble["mlp"]["model"]
                mlp_scaler = ensemble["mlp"]["scaler"]
                X_scaled = mlp_scaler.transform(_get_imputed())
                with torch.no_grad():
                    mlp_model.cuda()  # move to GPU for inference
                    X_tensor = torch.FloatTensor(X_scaled).cuda()
                    mlp_preds = mlp_model(X_tensor).squeeze().cpu().numpy()
                    mlp_model.cpu()  # move back to free VRAM
                    torch.cuda.empty_cache()
                mlp_ranks = ranked(mlp_preds)
                scores += mlp_ranks * self.mlp_weight
                total_w += self.mlp_weight
                model_ranks.append(mlp_ranks)
            except Exception:
                pass

        ridge_w = self.ridge_weight
        if total_w < 0.01:
            ridge_w = 1.0  # only Ridge available
        # Fix 3: ensemble["ridge"] is a dict {"model", "scaler"} — apply the
        # fitted scaler before predict so features match Ridge's trained scale.
        _ridge_obj = ensemble["ridge"]
        if isinstance(_ridge_obj, dict):
            _ridge_X = _ridge_obj["scaler"].transform(_get_imputed())
            ridge_ranks = ranked(_ridge_obj["model"].predict(_ridge_X))
        else:
            # Back-compat for any cached/legacy bare-Ridge objects.
            ridge_ranks = ranked(_ridge_obj.predict(_get_imputed()))
        scores  += ridge_ranks * ridge_w
        total_w += ridge_w
        model_ranks.append(ridge_ranks)

        blended = scores / max(total_w, 1e-6)

        # Meta-labeling: multiply by stage-2 probability (item 8)
        if self.use_meta_labeling and "meta" in ensemble:
            try:
                meta_prob = ensemble["meta"].predict(X_vals)
                # Blend into scores: score * meta_prob preserves ordering
                blended = blended * meta_prob
            except Exception:
                pass

        # Confidence: average pairwise agreement across all models
        if len(model_ranks) >= 2:
            pairwise_diffs = []
            for i in range(len(model_ranks)):
                for j in range(i + 1, len(model_ranks)):
                    pairwise_diffs.append(np.abs(model_ranks[i] - model_ranks[j]))
            confidence = 1.0 - np.mean(pairwise_diffs, axis=0)
        else:
            confidence = np.ones(len(X_vals))

        return blended, confidence

    # ------------------------------------------------------------------
    # Walk-forward loop
    # ------------------------------------------------------------------

    def fit_predict(self, panel: pd.DataFrame, labels: pd.Series,
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Walk-forward training + prediction with disk caching.

        Cache key includes: panel shape, date range, number of features,
        model hyperparams. If any change, cache is invalidated.
        """
        # ── Seed all stochastic subsystems for reproducibility ───────────
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        if TORCH_AVAILABLE:
            try:
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
            except Exception:
                pass

        # ── Check cache ──────────────────────────────────────────────────
        # Fix 4: include ALL constructor hyperparameters attached to self.*
        # plus content hashes of the input panel and labels so that changes
        # to regularization (ridge_alpha, L1/L2, etc.), tree shape
        # (max_depth, min_child_samples, min_gain_to_split, n_estimators)
        # or upstream feature/label construction invalidate the cache.
        try:
            _panel_hash = hashlib.md5(
                pd.util.hash_pandas_object(panel, index=True).values
            ).hexdigest()[:8]
        except Exception:
            _panel_hash = "na"
        try:
            _labels_hash = hashlib.md5(
                pd.util.hash_pandas_object(labels, index=True).values
            ).hexdigest()[:8]
        except Exception:
            _labels_hash = "na"

        _cache_key = hashlib.md5(
            f"{panel.shape}_{panel.index.get_level_values('date').min()}_"
            f"{panel.index.get_level_values('date').max()}_"
            f"{len(panel.columns)}_{self.min_train_days}_{self.retrain_freq}_"
            f"{self.forward_window}_{self.lgbm_weight}_{self.xgb_weight}_"
            f"{self.ridge_weight}_{self.mlp_weight}_{self.num_leaves}_"
            f"{self.learning_rate}_{self.prune_features}_"
            f"{self.feature_fraction}_{self.bagging_fraction}_"
            f"{self.max_feature_corr}_{self.max_train_days}_"
            f"{self.neutralize_vol}_{self.use_mlp}_"
            f"{self.lambdarank_truncation}_{self.use_return_decile_gain}_"
            f"{self.use_magnitude_weights}_{self.huber_weight}_"
            f"{self.quantile_weight}_{self.quantile_alpha}_{self.huber_alpha}_"
            f"{self.use_meta_labeling}_{self.early_stop_on_ic}_"
            f"{self.min_data_in_leaf}_{sorted((self.monotone_constraints or {}).items())}_"
            f"{self.use_uniqueness_weights}_{self.use_per_date_weights}_"
            f"{self.lgbm_num_seeds}_{self.random_state}_"
            f"emb={self.embargo_days}_"
            # Previously-missing ctor hyperparams (Fix 4)
            f"ra={self.ridge_alpha}_md={self.max_depth}_"
            f"mgs={self.min_gain_to_split}_l1={self.lambda_l1}_"
            f"l2={self.lambda_l2}_mcs={self.min_child_samples}_"
            f"ne={self.n_estimators}_mic={self.min_ic}_"
            f"ra_adj={self.risk_adjust}_srw={self.sector_rank_weight}_"
            # Content hashes of inputs
            f"ph={_panel_hash}_lh={_labels_hash}_"
            # Opt 3/5/6 params in cache key
            f"adv={self.run_adversarial_validation}_ws={self.warm_start_lgbm}_"
            f"ic={self.interaction_constraints_cfg}".encode()
        ).hexdigest()[:12]
        _cache_file = _CACHE_DIR / f"ml_predictions_{_cache_key}.pkl"

        if use_cache and _cache_file.exists():
            print(f"  [model] Loading cached predictions from {_cache_file.name}")
            with open(_cache_file, "rb") as f:
                cached = pickle.load(f)
            self.feature_names_ = list(panel.columns)
            # Restore models_ for feature importance
            self.models_ = cached.get("models_", [])
            return cached["predictions"]

        self.feature_names_ = list(panel.columns)
        all_dates = panel.index.get_level_values("date").unique().sort_values()
        tickers   = panel.index.get_level_values("ticker").unique()

        predictions: Dict[pd.Timestamp, pd.Series] = {}

        train_cutoffs = [
            (i, date)
            for i, date in enumerate(all_dates)
            if i >= self.min_train_days and i % self.retrain_freq == 0
        ]

        print(f"  [model] Ensemble walk-forward: {len(train_cutoffs)} retraining windows "
              f"(LGBM {self.lgbm_weight:.0%} / Ridge {self.ridge_weight:.0%})")

        current_ensemble = None
        _prev_lgbm_booster = None  # Opt 5: warm-start state

        for window_idx, (i, cutoff_date) in enumerate(train_cutoffs):
            # Use all features up to the cutoff, but mask labels whose
            # forward return window extends past the cutoff (lookahead fix).
            # Labels for dates in [cutoff - forward_window, cutoff) use returns
            # that haven't fully materialized yet, so we NaN them out.
            safe_label_end = max(0, i - self.forward_window)
            train_dates = all_dates[:i]
            gap_dates   = set(all_dates[safe_label_end:i])

            train_idx   = panel.index.get_level_values("date").isin(train_dates)
            # NaN passthrough: LGBM/XGB handle NaN natively; Ridge/MLP are
            # median-imputed inside _train_ensemble. Filling with 0 would
            # bias tree splits for momentum/amihud/etc (0 != missing).
            X_train     = panel[train_idx].replace([np.inf, -np.inf], np.nan)
            y_train     = labels.reindex(X_train.index).copy()

            # NaN out labels in the gap zone (forward returns not yet realized)
            if gap_dates:
                gap_mask = y_train.index.get_level_values("date").isin(gap_dates)
                y_train[gap_mask] = np.nan

            next_cutoff = (
                train_cutoffs[window_idx + 1][1]
                if window_idx + 1 < len(train_cutoffs)
                else all_dates[-1]
            )
            # AFML Ch. 7.4 embargo: shift test window forward by embargo_days
            # to mirror the train-side purge (gap_dates). Without this, the
            # first few test days share serial-correlated residuals with the
            # last train labels, leaking information across the fold boundary.
            if self.embargo_days > 0:
                post = all_dates[all_dates > cutoff_date]
                emb_start = post[self.embargo_days] if len(post) > self.embargo_days else (
                    post[-1] if len(post) > 0 else cutoff_date
                )
                pred_dates = all_dates[(all_dates >= emb_start) & (all_dates <= next_cutoff)]
            else:
                pred_dates = all_dates[(all_dates > cutoff_date) & (all_dates <= next_cutoff)]

            print(
                f"  [model] Window {window_idx+1}/{len(train_cutoffs)}: "
                f"train {len(train_dates)}d -> predict {len(pred_dates)}d",
                end="\r",
            )

            # Opt 5: pass prior window's LGBM booster for warm-start
            _init = _prev_lgbm_booster if self.warm_start_lgbm else None
            new_ensemble = self._train_ensemble(X_train, y_train,
                                                lgbm_init_model=_init)
            if new_ensemble is not None:
                current_ensemble = new_ensemble
                # Opt 5: store booster for next window's warm-start
                if self.warm_start_lgbm and "lgbm" in new_ensemble:
                    _lgbm_obj = new_ensemble["lgbm"]
                    # For seed-bagged ensembles, use the first booster
                    _prev_lgbm_booster = _lgbm_obj[0] if isinstance(_lgbm_obj, list) else _lgbm_obj

                # Opt 3: adversarial validation gated behind flag (saves ~25 min)
                # AUC=1.0 every time for temporal walk-forward (trivially separable).
                adv_auc = float("nan")
                if self.run_adversarial_validation:
                    try:
                        if len(pred_dates) > 0:
                            pred_idx = panel.index.get_level_values("date").isin(pred_dates)
                            X_pred_chunk = panel[pred_idx].replace([np.inf, -np.inf], np.nan)
                            if len(X_pred_chunk) > 100:
                                adv_auc = self.adversarial_validation(X_train, X_pred_chunk)
                                if np.isfinite(adv_auc) and adv_auc > 0.75:
                                    print(f"\n  [adv-val] Window {window_idx+1} AUC={adv_auc:.3f} (distribution shift)")
                    except Exception:
                        pass
                self.models_.append({
                    "cutoff_date":  cutoff_date,
                    "n_train_days": len(train_dates),
                    "ensemble":     new_ensemble,
                    "adv_auc":      adv_auc,
                })

            if current_ensemble is None:
                continue

            for pred_date in pred_dates:
                if pred_date not in panel.index.get_level_values("date"):
                    continue
                X_pred = panel.xs(pred_date, level="date").reindex(tickers).replace([np.inf, -np.inf], np.nan)
                scores, conf = self._predict_ensemble(X_pred, current_ensemble)
                # Fix 1 (LOW-polish): center scores before confidence scaling so
                # the subsequent cross-sectional rank (pct=True) actually reflects
                # the confidence signal. Multiplying raw scores by a positive
                # factor is a no-op under rank — but subtracting 0.5 makes the
                # rescaling asymmetric: low-confidence predictions get
                # compressed toward neutral (0), while high-confidence ones
                # retain their distance from the middle rank.
                adjusted = (scores - 0.5) * (0.3 + 0.7 * conf)
                predictions[pred_date] = pd.Series(adjusted, index=tickers)

        print()
        print(f"  [model] Generated predictions for {len(predictions)} days")

        pred_df = pd.DataFrame(predictions).T
        pred_df.index.name = "date"
        result = pred_df.rank(axis=1, pct=True)

        # ── Save to cache ────────────────────────────────────────────────
        # Strip GPU models before pickling (can't serialize CUDA tensors)
        serializable_models = []
        for m in self.models_:
            safe_m = {
                "cutoff_date": m["cutoff_date"],
                "n_train_days": m["n_train_days"],
                "adv_auc": m.get("adv_auc", float("nan")),
            }
            # Only save feature importance info, not full models
            if "ensemble" in m and "lgbm" in m["ensemble"]:
                try:
                    lgbm_obj = m["ensemble"]["lgbm"]
                    boosters = lgbm_obj if isinstance(lgbm_obj, list) else [lgbm_obj]
                    imp_sum = None
                    feat_names = None
                    for b in boosters:
                        imp = np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
                        feat_names = b.feature_name()
                        imp_sum = imp if imp_sum is None else imp_sum + imp
                    if imp_sum is not None:
                        imp_sum = imp_sum / len(boosters)
                        safe_m["lgbm_importance"] = dict(zip(feat_names, imp_sum))
                except Exception:
                    pass
            serializable_models.append(safe_m)

        with open(_cache_file, "wb") as f:
            pickle.dump({
                "predictions": result,
                "models_": serializable_models,
            }, f)
        print(f"  [model] Predictions cached to {_cache_file.name}")

        return result

    # ------------------------------------------------------------------
    # Feature importance (from LightGBM component)
    # ------------------------------------------------------------------

    def feature_importance(self) -> Optional[pd.Series]:
        # Try live models first (full ensemble objects)
        lgbm_models = [
            m["ensemble"]["lgbm"]
            for m in self.models_
            if isinstance(m.get("ensemble"), dict) and "lgbm" in m["ensemble"]
        ]
        if lgbm_models:
            importances = []
            for m in lgbm_models:
                boosters = m if isinstance(m, list) else [m]
                for b in boosters:
                    importances.append(
                        pd.Series(b.feature_importance(importance_type="gain"),
                                  index=b.feature_name())
                    )
            return pd.DataFrame(importances).mean().sort_values(ascending=False)

        # Fallback: cached importance dicts (from serialized cache)
        cached_imps = [
            pd.Series(m["lgbm_importance"])
            for m in self.models_
            if "lgbm_importance" in m
        ]
        if cached_imps:
            return pd.DataFrame(cached_imps).mean().sort_values(ascending=False)

        return None

    def rolling_feature_importance(self) -> Optional[pd.DataFrame]:
        # Fix 4 (LOW-polish): previously this only worked when live ensemble
        # objects were attached — after a cache load those are stripped,
        # so it silently returned None. We now ALSO accept the cached
        # per-window importance dict stored in safe_m["lgbm_importance"]
        # (keyed by cutoff_date), which survives pickle round-trips.
        records = []
        for m in self.models_:
            # Preferred: introspect live boosters when available.
            if isinstance(m.get("ensemble"), dict) and "lgbm" in m["ensemble"]:
                lgbm = m["ensemble"]["lgbm"]
                boosters = lgbm if isinstance(lgbm, list) else [lgbm]
                imp_vals = np.mean(
                    [np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
                     for b in boosters], axis=0,
                )
                feat_names = boosters[0].feature_name()
                records.append(pd.Series(imp_vals, index=feat_names,
                                         name=m.get("cutoff_date")))
                continue
            # Fallback: cached importance dict {feature_name: gain}.
            if "lgbm_importance" in m:
                s = pd.Series(m["lgbm_importance"], name=m.get("cutoff_date"))
                records.append(s)
        return pd.DataFrame(records) if records else None


# ---------------------------------------------------------------------------
# Hyperparameter optimization (Optuna)
# ---------------------------------------------------------------------------

def optimize_hyperparameters(
    panel: pd.DataFrame,
    labels: pd.Series,
    n_trials: int = 50,
    n_eval_windows: int = 5,
    search_lookback_days: int = 504,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> dict:
    """
    Bayesian hyperparameter optimization using Optuna.

    Runs a mini walk-forward on the LAST n_eval_windows training windows
    and maximizes mean OOS rank IC.

    Fix 3: each Optuna trial previously ran fit_predict on the ENTIRE
    panel (~150 walk-forward windows x 4 heads per trial), which made
    tuning prohibitively expensive. We now restrict each trial's panel
    and labels to the most recent `search_lookback_days` trading days
    (default ~2 years). Callers can also bound the evaluation window
    via `eval_start`/`eval_end` (ISO date strings). Set
    `search_lookback_days=0` to restore the full-panel behavior.

    Returns the best hyperparameter dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [optuna] Not installed (pip install optuna). Using defaults.")
        return {}

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    tickers = panel.index.get_level_values("ticker").unique()

    # ── Fix 3: subset panel/labels to the trailing search window ────────
    # Each trial only trains on `search_lookback_days` of history, which
    # cuts trial cost roughly proportionally (e.g. 504/2500 ≈ 5x speedup
    # on a 10-year panel with retrain_freq=63).
    if search_lookback_days and len(all_dates) > search_lookback_days:
        subset_start = all_dates[-search_lookback_days]
        panel_dates = panel.index.get_level_values("date")
        panel = panel[panel_dates >= subset_start]
        label_dates = labels.index.get_level_values("date") \
            if isinstance(labels.index, pd.MultiIndex) else labels.index
        labels = labels[label_dates >= subset_start]
        all_dates = panel.index.get_level_values("date").unique().sort_values()

    # Use last portion of data for optimization to be fast
    _eval_start = max(0, len(all_dates) - 504)  # last ~2 years
    eval_dates = all_dates[_eval_start:]
    # Optional caller-supplied ISO date bounds on the eval window
    if eval_start is not None:
        eval_dates = eval_dates[eval_dates >= pd.Timestamp(eval_start)]
    if eval_end is not None:
        eval_dates = eval_dates[eval_dates <= pd.Timestamp(eval_end)]

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 10, 63)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        feat_frac = trial.suggest_float("feature_fraction", 0.4, 0.9)
        bag_frac = trial.suggest_float("bagging_fraction", 0.5, 0.9)
        n_est = trial.suggest_int("n_estimators", 100, 500, step=50)
        ridge_alpha = trial.suggest_float("ridge_alpha", 5.0, 200.0, log=True)
        lambda_l1 = trial.suggest_float("lambda_l1", 0.1, 50.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 0.1, 100.0, log=True)
        min_child = trial.suggest_int("min_child_samples", 10, 50)
        max_depth = trial.suggest_int("max_depth", 4, 10)
        # Ensemble weight ratios (normalized later)
        lgbm_w = trial.suggest_float("lgbm_weight", 0.1, 0.6)
        xgb_w = trial.suggest_float("xgb_weight", 0.0, 0.5)
        ridge_w = trial.suggest_float("ridge_weight", 0.1, 0.5)
        mlp_w = trial.suggest_float("mlp_weight", 0.0, 0.4)
        total_w = lgbm_w + xgb_w + ridge_w + mlp_w
        # Fix 5 (LOW-polish): defensively guard against an all-zero weight
        # sample. Optuna's lower bounds should prevent this, but a float
        # underflow or bad suggestion shouldn't crash the trial with a
        # ZeroDivisionError — just prune the trial.
        if total_w < 1e-6:
            return float("-inf")
        lgbm_w, xgb_w, ridge_w, mlp_w = [w / total_w for w in (lgbm_w, xgb_w, ridge_w, mlp_w)]

        wf = WalkForwardModel(
            min_train_days=252,
            retrain_freq=63,  # quarterly for speed
            num_leaves=num_leaves,
            learning_rate=lr,
            feature_fraction=feat_frac,
            bagging_fraction=bag_frac,
            n_estimators=n_est,
            min_child_samples=min_child,
            max_depth=max_depth,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            ridge_alpha=ridge_alpha,
            lgbm_weight=lgbm_w,
            xgb_weight=xgb_w,
            ridge_weight=ridge_w,
            mlp_weight=mlp_w,
            prune_features=False,  # skip pruning for speed
        )

        try:
            pred = wf.fit_predict(panel, labels)
            # Compute mean rank IC on the eval period
            fwd_ret = panel.index.get_level_values("date").isin(eval_dates)
            eval_pred = pred.loc[pred.index.isin(eval_dates)]

            if len(eval_pred) < 50:
                return 0.0

            # Simple IC: correlation between prediction rank and next-day return
            ics = []
            for date in eval_pred.index[:100]:  # sample for speed
                p = eval_pred.loc[date].dropna()
                if date in labels.index.get_level_values("date"):
                    l = labels.xs(date, level="date").reindex(p.index)
                    common = p.index.intersection(l.dropna().index)
                    if len(common) > 20:
                        ic = p[common].corr(l[common], method="spearman")
                        if np.isfinite(ic):
                            ics.append(ic)

            return np.mean(ics) if ics else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  [optuna] Best params (IC={study.best_value:.4f}): {best}")
    return best


# ---------------------------------------------------------------------------
# Signal blender
# ---------------------------------------------------------------------------

def blend_signals(
    rule_based: pd.DataFrame,
    ml_signal: pd.DataFrame,
    ml_weight: float = 0.5,
) -> pd.DataFrame:
    common_dates = rule_based.index.intersection(ml_signal.index)
    common_tickers = rule_based.columns.intersection(ml_signal.columns)

    rb = rule_based.loc[common_dates, common_tickers]
    ml = ml_signal.loc[common_dates, common_tickers]

    blended = (1 - ml_weight) * rb + ml_weight * ml
    return blended.rank(axis=1, pct=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_prices, get_close, get_returns, get_volume
    from features import build_composite_signal

    print("Loading data...")
    prices = load_prices(start="2018-01-01", end="2024-01-01")
    close = get_close(prices)
    returns = get_returns(prices)
    volume = get_volume(prices)

    print("Building signals...")
    composite, ranked_signals = build_composite_signal(close, returns, volume)

    print("Building feature matrix...")
    panel = build_feature_matrix(close, returns, volume, ranked_signals)
    labels = build_labels(returns, forward_window=3)

    print("Running walk-forward model...")
    wf = WalkForwardModel(min_train_days=252, retrain_freq=63, forward_window=3)
    ml_signal = wf.fit_predict(panel, labels)

    print(f"\nML signal shape: {ml_signal.shape}")
    print(ml_signal.tail(3).iloc[:, :5])

    imp = wf.feature_importance()
    if imp is not None:
        print("\nTop 10 features by importance:")
        print(imp.head(10))