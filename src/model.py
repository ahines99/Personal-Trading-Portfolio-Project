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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

def build_feature_matrix(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    ranked_signals: Dict[str, pd.DataFrame],
    alt_features: Optional[Dict[str, pd.DataFrame]] = None,
    use_cache: bool = True,
    sector_map: Optional[Dict[str, str]] = None,
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
    """
    # ── feature panel cache ──────────────────────────────────────────────────
    _cache_key = hashlib.md5(
        f"{close.index[0]}_{close.index[-1]}_{close.shape}_{sorted(ranked_signals.keys())}"
        f"_{sorted(alt_features.keys()) if alt_features else []}"
        f"_sector={bool(sector_map)}".encode()
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
    )

    features = {}

    # Momentum — short, medium, long horizon
    for w in [5, 10, 21, 63, 126]:
        features[f"mom_{w}d"] = momentum(close, w)

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

    panels = []
    for feat_name, df in features.items():
        s = df.stack(future_stack=True)
        s.name = feat_name
        panels.append(s)

    panel = pd.concat(panels, axis=1)
    panel.index.names = ["date", "ticker"]

    if use_cache:
        with open(_cache_file, "wb") as f:
            pickle.dump(panel, f)
        print(f"      [cache] Feature panel saved to {_cache_file.name}")

    return panel


def build_labels(
    returns: pd.DataFrame,
    forward_window: int = 21,
    risk_adjust: bool = True,
    sector_map: Optional[Dict[str, str]] = None,
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

    Risk-adjusted labels (default):
      Instead of ranking raw forward returns, we rank forward_return / trailing_vol.
      This trains the model to find CONSISTENT winners rather than volatile lottery
      tickets. A stock returning +8% at 15% vol scores higher than one returning
      +12% at 50% vol. For a concentrated 20-stock portfolio, consistency compounds
      better than variance.

    Labels for the last `forward_window` days are NaN (unrealized returns).
    """
    fwd_return = returns.shift(-forward_window).rolling(forward_window).sum()

    # Mask the last forward_window rows
    fwd_return.iloc[-forward_window:] = np.nan

    if risk_adjust:
        trailing_vol = returns.rolling(21, min_periods=10).std() * np.sqrt(252)
        fwd_signal = fwd_return / trailing_vol.replace(0, np.nan)
    else:
        fwd_signal = fwd_return

    # Within-industry ranking: rank each stock vs its sector peers.
    # This forces the model to learn stock selection, not sector bets.
    # Blend: 50% within-sector rank + 50% universe-wide rank.
    # Pure within-sector would ignore cross-sector opportunities.
    universe_rank = fwd_signal.rank(axis=1, pct=True)

    if sector_map is not None:
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

        pct_rank = 0.50 * universe_rank + 0.50 * sector_rank
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
        retrain_freq: int = 21,
        forward_window: int = 1,
        # LightGBM hyperparams
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        # feature_fraction=0.5: forces each tree to see only half the features.
        # Breaks correlated cluster dominance (13 vol features were capturing
        # every split). At 0.7 the model still concentrated on vol; at 0.5
        # it must discover orthogonal signal in momentum/fundamental features.
        feature_fraction: float = 0.5,
        # bagging_fraction=0.7: row subsampling compounds with feature_fraction
        # to produce more diverse trees (de-correlation effect).
        bagging_fraction: float = 0.7,
        min_child_samples: int = 20,
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
    ):
        self.min_train_days    = min_train_days
        self.retrain_freq      = retrain_freq
        self.forward_window    = forward_window
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.num_leaves        = num_leaves
        self.feature_fraction  = feature_fraction
        self.bagging_fraction  = bagging_fraction
        self.min_child_samples = min_child_samples
        self.lgbm_weight       = lgbm_weight
        self.xgb_weight   = xgb_weight
        self.ridge_weight      = ridge_weight
        self.mlp_weight        = mlp_weight
        self.prune_features    = prune_features
        self.min_ic            = min_ic
        self.max_feature_corr  = max_feature_corr
        self.max_train_days    = max_train_days
        self.neutralize_vol    = neutralize_vol

        self.models_: List[dict] = []
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Individual learner trainers
    # ------------------------------------------------------------------

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray,
                  groups: np.ndarray, feature_names: List[str]):
        if not LGBM_AVAILABLE:
            return None

        # Carve out last 21 days as temporal holdout for early stopping.
        # groups[i] = number of tickers on day i, so we split by group count.
        val_days  = min(21, len(groups) // 5)
        val_rows  = int(groups[-val_days:].sum())
        X_tr, X_val = X[:-val_rows], X[-val_rows:]
        y_tr, y_val = y[:-val_rows], y[-val_rows:]
        g_tr, g_val = groups[:-val_days], groups[-val_days:]

        dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feature_names)
        dval   = lgb.Dataset(X_val, label=y_val, group=g_val, feature_name=feature_names,
                             reference=dtrain)
        params = {
            "objective":         "lambdarank",
            "metric":            "ndcg",
            "ndcg_eval_at":      [5, 10],
            "label_gain":        [0, 0, 1, 5, 20, 100],
            "num_leaves":        self.num_leaves,
            "learning_rate":     self.learning_rate,
            "feature_fraction":  self.feature_fraction,
            "bagging_fraction":  self.bagging_fraction,
            "bagging_freq":      1,
            "min_child_samples": self.min_child_samples,
            # Regularization: previously zero (no leaf weight penalty).
            # L1 encourages sparse leaf outputs, L2 shrinks all leaf weights.
            # Prevents overfitting to noise in cross-sectional financial data.
            # Values calibrated per Qlib benchmark + practitioner literature.
            "lambda_l1":         5.0,
            "lambda_l2":         10.0,
            "min_gain_to_split": 0.05,  # prevent splits on noise
            "max_depth":         7,     # secondary guard on leaf-wise depth
            "num_threads":       -1,
            "verbose":           -1,
            "device":            "gpu" if TORCH_AVAILABLE else "cpu",
        }
        return lgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None):
        model = Ridge(alpha=50.0)
        model.fit(X, y, sample_weight=sample_weight)
        return model

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
                "eval_metric": "ndcg@5",
                "tree_method": "hist",
                "device": "cuda" if TORCH_AVAILABLE else "cpu",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "verbosity": 0,
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

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray) -> Optional[object]:
        """
        Train a 3-layer MLP on GPU for cross-sectional stock ranking.

        Architecture: input → 128 → 64 → 32 → 1 with BatchNorm + Dropout.
        Learns different feature interactions than tree-based models,
        providing genuine ensemble diversity.

        Returns a dict with 'model' and 'scaler' for prediction.
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            from sklearn.preprocessing import StandardScaler

            # Scale features — fit ONLY on train split to avoid leaking
            # val statistics into training (scaler sees mean/std of val data).
            split = int(len(X) * 0.8)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X[:split])
            X_val_scaled = scaler.transform(X[split:])

            X_tr = torch.FloatTensor(X_tr_scaled).cuda()
            y_tr = torch.FloatTensor(y[:split].astype(float)).cuda()
            X_val = torch.FloatTensor(X_val_scaled).cuda()
            y_val = torch.FloatTensor(y[split:].astype(float)).cuda()

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
            idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
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
                        max_train_samples: int = 500_000) -> Optional[dict]:
        mask = y_train.notna() & X_train.notna().all(axis=1)
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

        X_vals, y_vals        = X_.values, y_.values
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

        groups = dates.value_counts().sort_index().values

        if LGBM_AVAILABLE:
            lgbm = self._fit_lgbm(X_vals, y_vals, groups, feature_names)
            if lgbm is not None:
                ensemble["lgbm"] = lgbm

        if XGB_AVAILABLE:
            xgb_model = self._fit_xgboost(X_vals, y_vals, groups, feature_names)
            if xgb_model is not None:
                ensemble["xgboost"] = xgb_model

        ensemble["ridge"] = self._fit_ridge(X_vals, y_vals,
                                            sample_weight=sample_weight)

        if TORCH_AVAILABLE:
            mlp = self._fit_mlp(X_vals, y_vals)
            if mlp is not None:
                ensemble["mlp"] = mlp
            # Free GPU memory after MLP training
            torch.cuda.empty_cache()

        return ensemble

    def _predict_ensemble(self, X_pred: pd.DataFrame, ensemble: dict) -> tuple:
        """
        Returns (blended_scores, confidence).

        Uses feature subset from training for prediction alignment.
        Confidence is average pairwise agreement across all models.
        """
        # Use only the features this ensemble was trained on
        feature_subset = ensemble.get("feature_subset", list(X_pred.columns))
        X_aligned = X_pred.reindex(columns=feature_subset, fill_value=0)
        X_vals = X_aligned.values

        def ranked(raw: np.ndarray) -> np.ndarray:
            return pd.Series(raw).rank(pct=True).values

        model_ranks = []
        scores  = np.zeros(len(X_vals))
        total_w = 0.0

        if "lgbm" in ensemble:
            lgbm_ranks = ranked(ensemble["lgbm"].predict(X_vals))
            scores  += lgbm_ranks * self.lgbm_weight
            total_w += self.lgbm_weight
            model_ranks.append(lgbm_ranks)

        if "xgboost" in ensemble:
            dmat = xgb.DMatrix(X_vals, feature_names=ensemble.get("feature_subset", None))
            xgb_ranks = ranked(ensemble["xgboost"].predict(dmat))
            scores  += xgb_ranks * self.xgb_weight
            total_w += self.xgb_weight
            model_ranks.append(xgb_ranks)

        # MLP predictions (GPU)
        if "mlp" in ensemble:
            try:
                mlp_model = ensemble["mlp"]["model"]
                mlp_scaler = ensemble["mlp"]["scaler"]
                X_scaled = mlp_scaler.transform(X_vals)
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
        ridge_ranks = ranked(ensemble["ridge"].predict(X_vals))
        scores  += ridge_ranks * ridge_w
        total_w += ridge_w
        model_ranks.append(ridge_ranks)

        blended = scores / max(total_w, 1e-6)

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
        # ── Check cache ──────────────────────────────────────────────────
        _cache_key = hashlib.md5(
            f"{panel.shape}_{panel.index.get_level_values('date').min()}_"
            f"{panel.index.get_level_values('date').max()}_"
            f"{len(panel.columns)}_{self.min_train_days}_{self.retrain_freq}_"
            f"{self.forward_window}_{self.lgbm_weight}_{self.xgb_weight}_"
            f"{self.ridge_weight}_{self.mlp_weight}_{self.num_leaves}_"
            f"{self.learning_rate}_{self.prune_features}_"
            f"{self.feature_fraction}_{self.bagging_fraction}_"
            f"{self.max_feature_corr}_{self.max_train_days}_"
            f"{self.neutralize_vol}".encode()
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

        for window_idx, (i, cutoff_date) in enumerate(train_cutoffs):
            # Use all features up to the cutoff, but mask labels whose
            # forward return window extends past the cutoff (lookahead fix).
            # Labels for dates in [cutoff - forward_window, cutoff) use returns
            # that haven't fully materialized yet, so we NaN them out.
            safe_label_end = max(0, i - self.forward_window)
            train_dates = all_dates[:i]
            gap_dates   = set(all_dates[safe_label_end:i])

            train_idx   = panel.index.get_level_values("date").isin(train_dates)
            X_train     = panel[train_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
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
            pred_dates = all_dates[(all_dates > cutoff_date) & (all_dates <= next_cutoff)]

            print(
                f"  [model] Window {window_idx+1}/{len(train_cutoffs)}: "
                f"train {len(train_dates)}d -> predict {len(pred_dates)}d",
                end="\r",
            )

            new_ensemble = self._train_ensemble(X_train, y_train)
            if new_ensemble is not None:
                current_ensemble = new_ensemble
                self.models_.append({
                    "cutoff_date":  cutoff_date,
                    "n_train_days": len(train_dates),
                    "ensemble":     new_ensemble,
                })

            if current_ensemble is None:
                continue

            for pred_date in pred_dates:
                if pred_date not in panel.index.get_level_values("date"):
                    continue
                X_pred = panel.xs(pred_date, level="date").reindex(tickers).replace([np.inf, -np.inf], np.nan).fillna(0)
                scores, conf = self._predict_ensemble(X_pred, current_ensemble)
                # Penalize low-confidence predictions: multiply score by confidence.
                # When LGBM and Ridge disagree (conf ~0.3), the score gets dampened.
                # When they agree (conf ~0.9), the score is nearly unchanged.
                adjusted = scores * (0.3 + 0.7 * conf)  # floor at 30% of score
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
            }
            # Only save feature importance info, not full models
            if "ensemble" in m and "lgbm" in m["ensemble"]:
                try:
                    safe_m["lgbm_importance"] = dict(zip(
                        m["ensemble"]["lgbm"].feature_name(),
                        m["ensemble"]["lgbm"].feature_importance(importance_type="gain"),
                    ))
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
            importances = [
                pd.Series(m.feature_importance(importance_type="gain"), index=m.feature_name())
                for m in lgbm_models
            ]
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
        records = []
        for m in self.models_:
            if "lgbm" not in m["ensemble"]:
                continue
            lgbm = m["ensemble"]["lgbm"]
            records.append(pd.Series(
                lgbm.feature_importance(importance_type="gain"),
                index=lgbm.feature_name(),
                name=m["cutoff_date"],
            ))
        return pd.DataFrame(records) if records else None


# ---------------------------------------------------------------------------
# Hyperparameter optimization (Optuna)
# ---------------------------------------------------------------------------

def optimize_hyperparameters(
    panel: pd.DataFrame,
    labels: pd.Series,
    n_trials: int = 50,
    n_eval_windows: int = 5,
) -> dict:
    """
    Bayesian hyperparameter optimization using Optuna.

    Runs a mini walk-forward on the LAST n_eval_windows training windows
    and maximizes mean OOS rank IC.

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

    # Use last portion of data for optimization to be fast
    eval_start = max(0, len(all_dates) - 504)  # last ~2 years
    eval_dates = all_dates[eval_start:]

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 10, 50)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        feat_frac = trial.suggest_float("feature_fraction", 0.4, 0.9)
        n_est = trial.suggest_int("n_estimators", 100, 400, step=50)
        ridge_alpha = trial.suggest_float("ridge_alpha", 5.0, 200.0, log=True)

        wf = WalkForwardModel(
            min_train_days=252,
            retrain_freq=63,  # quarterly for speed
            num_leaves=num_leaves,
            learning_rate=lr,
            feature_fraction=feat_frac,
            n_estimators=n_est,
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