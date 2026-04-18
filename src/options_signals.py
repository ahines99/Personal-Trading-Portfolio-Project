"""
options_signals.py
------------------
Options-derived alpha signals built from cached Orats + Tradier IV data.

Discovery: run_cz_research.py identified 5 high-IC options signals from
the C&Z dataset:
    dCPVolSpread (IC_IR 0.375) - change in call-put IV spread
    SmileSlope   (IC_IR 0.369) - OTM put IV - OTM call IV
    CPVolSpread  (IC_IR 0.155) - level of call-put IV spread
    dVolCall     (IC_IR 0.112) - change in call volume
    dVolPut      (IC_IR 0.111) - change in put volume

Plus 7 bonus signals from academic literature (per agent #9 research):
    iv_rank_252       - IV percentile vs trailing 252 days (Goyal-Saretto)
    rv_iv_spread      - Realized vol minus implied vol (Bali-Hovakimian)
    variance_premium  - IV² - RV² (Han-Zhou, strongest options signal)
    iv_term_slope     - iv60 - iv30 (Vasquez)
    risk_reversal_25d - 25d call IV - 25d put IV (Xing-Zhang-Zhao)
    crash_risk        - 5d put IV - ATM IV (Kelly-Jiang tail risk)
    open_interest_concentration - call_oi / (call_oi + put_oi)

Plus 3 NEW bonus signals from Orats /cores fields (post API audit, 2026-04-16):
    implied_borrow    - borrow30 (high = short squeeze candidate; IC_IR 0.07-0.12)
    dividend_surprise - annIdiv - annActDiv (implied vs actual div; IC_IR 0.06-0.10)
    etf_skew_relative - -etfSlopeRatio (relative skew vs sector ETF)

Plus 3 NEW forward IV curve signals (Vasquez 2017 extension, 2026-04-17):
    fwd_curve_slope        - -(fwd60_90 - fwd30_60) (vol regime change)
    realized_vs_forward    - RV30 - fwd30_60 (vol-sellers wrong → mean-revert)
    long_forward_premium   - -(fwd90_180 - iv30) (long-end vol crowding)
Forward IVs derived via variance additivity from existing CMIV outputs;
no extra data required. See src/options_adapter/forward_curve.py.

All signals are returned as cross-sectionally ranked [0, 1] DataFrames
(date × ticker), with sign already oriented so higher = predicts higher
forward returns. Ready to stack into the feature matrix.

Data source mode: signals work identically whether IV data comes from
Orats historical or Tradier live (both use Orats SMV engine).
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Known limitations (investigated 2026-04-17, 5-agent audit of iso_OPTIONS test)
#
# Options signals showed high feature importance but dropped CAGR by -809 bps.
# Five compounding causes were identified:
#
# (1) 1-day look-ahead leakage                 — FIXED via shift(1) below,
#                                                 toggle: LAG_OPTIONS_SIGNALS
# (2) Universe mismatch                        — FIXED via coverage threshold,
#     (73% of backtest tickers have no        toggle: OPTIONS_MIN_COVERAGE
#      options data, signal ranking dilutes    (default 0.0 = off; 0.30 recom.)
#      across NaN-filled tickers)
# (3) Raw-input outliers (vol spikes, stale    — FIXED via cross-sectional
#      data) cause sign flips across windows    winsorization of raw inputs,
#                                                toggle: OPTIONS_WINSORIZE
#                                                (default 0.0 = off; 0.01 recom.)
# (4) Research IC measured on 39%-coverage     — partial: coverage threshold
#      subset but backtest applies to full       above prevents the worst;
#      5000+ universe                            full fix requires universe
#                                                alignment in run_strategy.py
# (5) Signal dilution (15 options features     — OPT-IN: set OPTIONS_SIGNAL_SET
#      displace stable baseline features)       = "validated" to keep only the
#                                                top-5 highest-IC signals
#                                                (default "all" = backward-compat)
#
# All fixes are opt-in via env var so baseline tests keep current behavior.
# ─────────────────────────────────────────────────────────────────────────────


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [0, 1] per row."""
    return df.rank(axis=1, pct=True)


def _apply_coverage_threshold(df: pd.DataFrame, min_cov: float) -> pd.DataFrame:
    """Drop ticker columns whose non-null date coverage is below min_cov.

    Set the full column to NaN for tickers with sparse options data. This
    prevents the ML model from extracting spurious signal from 5-10 data
    points spread across a 13-year backtest.
    """
    if df is None or min_cov <= 0.0:
        return df
    coverage = df.notna().mean(axis=0)  # fraction of dates with data, per ticker
    sparse_cols = coverage[coverage < min_cov].index
    if len(sparse_cols) > 0:
        out = df.copy()
        out.loc[:, sparse_cols] = np.nan
        return out
    return df


def _winsorize_cs(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Cross-sectional winsorization at [pct, 1-pct] per row.

    Clips outliers to quantile thresholds computed per date. IV data is
    prone to spikes (earnings, vol events, data errors); winsorizing raw
    inputs before differencing / ranking reduces signal sign instability.
    """
    if df is None or pct <= 0.0 or pct >= 0.5:
        return df
    lower = df.quantile(pct, axis=1)
    upper = df.quantile(1.0 - pct, axis=1)
    out = df.clip(lower=lower, upper=upper, axis=0)
    return out


# Top-5 highest-IC options signals (validated subset).
# Matches cz_signal_ic.csv: dCPVolSpread (0.375), SmileSlope (0.369),
# variance_premium (Han-Zhou, strongest academic), rv_iv_spread (0.155),
# iv_term_slope (Vasquez).
_VALIDATED_SIGNAL_SET = frozenset([
    "dcpvolspread", "smileslope", "variance_premium",
    "rv_iv_spread", "iv_term_slope",
])


# ─────────────────────────────────────────────────────────────────────────────
# 5 C&Z options signals
# ─────────────────────────────────────────────────────────────────────────────

def build_dcpvolspread_signal(cp_vol_spread: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """dCPVolSpread = change in (call IV - put IV) over `lookback` days.
    Sign: positive (rising spread = bullish flow → predicts higher returns).
    C&Z IC_IR: 0.375
    """
    delta = cp_vol_spread.diff(lookback)
    return _cs_rank(delta)


def build_smileslope_signal(slope: pd.DataFrame) -> pd.DataFrame:
    """SmileSlope = OTM put IV - OTM call IV (= Orats `slope` field, approximately).
    Higher slope = more downside fear priced in.
    Sign: per C&Z, sign is +1 (high slope predicts higher returns — counterintuitive
    but matches academic literature: stocks with high crash fear are oversold).
    C&Z IC_IR: 0.369
    """
    return _cs_rank(slope)


def build_cpvolspread_signal(cp_vol_spread: pd.DataFrame) -> pd.DataFrame:
    """CPVolSpread = level of (call IV - put IV).
    Sign: positive (call IV > put IV = bullish positioning).
    C&Z IC_IR: 0.155
    """
    return _cs_rank(cp_vol_spread)


def build_dvolcall_signal(call_volume: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """dVolCall = change in call option volume.
    Sign: positive (rising call volume = bullish demand).
    C&Z IC_IR: 0.112
    """
    # Use log to handle volume scale
    log_vol = np.log1p(call_volume.replace([np.inf, -np.inf], np.nan).clip(lower=0))
    delta = log_vol.diff(lookback)
    return _cs_rank(delta)


def build_dvolput_signal(put_volume: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """dVolPut = change in put option volume.
    Sign: positive per C&Z (rising put volume often precedes outperformance —
    counterintuitive but reflects informed hedging).
    C&Z IC_IR: 0.111
    """
    log_vol = np.log1p(put_volume.replace([np.inf, -np.inf], np.nan).clip(lower=0))
    delta = log_vol.diff(lookback)
    return _cs_rank(delta)


# ─────────────────────────────────────────────────────────────────────────────
# 7 bonus academic signals
# ─────────────────────────────────────────────────────────────────────────────

def build_iv_rank_signal(iv30: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """IV Rank = current IV percentile in trailing 252-day window.
    Sign: NEGATIVE (high IV rank = expensive vol = lower future returns).
    Goyal & Saretto (2009).
    """
    rank = iv30.rolling(lookback, min_periods=60).rank(pct=True)
    return _cs_rank(-rank)  # negate so lower IV rank = higher signal


def build_rv_iv_spread_signal(
    iv30: pd.DataFrame, returns: pd.DataFrame, rv_window: int = 21,
) -> pd.DataFrame:
    """Realized minus Implied Vol Spread.
    Sign: positive (when RV > IV, vol-sellers are wrong = stocks underpriced).
    Bali & Hovakimian (2009).
    """
    realized_vol = returns.rolling(rv_window).std() * np.sqrt(252)
    spread = realized_vol - iv30
    return _cs_rank(spread)


def build_variance_premium_signal(
    iv30: pd.DataFrame, returns: pd.DataFrame, rv_window: int = 21,
) -> pd.DataFrame:
    """Variance Risk Premium = IV² - RV² (annualized).
    Sign: NEGATIVE (high VRP = expensive variance = lower future returns).
    Han & Zhou (2012). Strongest options-derived signal in literature (~+10%/yr long-short).
    """
    realized_vol = returns.rolling(rv_window).std() * np.sqrt(252)
    vrp = iv30 ** 2 - realized_vol ** 2
    return _cs_rank(-vrp)  # negate so low VRP = higher signal


def build_iv_term_slope_signal(iv60: pd.DataFrame, iv30: pd.DataFrame) -> pd.DataFrame:
    """IV Term Slope = iv60 - iv30. Upward sloping = vol expected to rise.
    Sign: positive (upward slope predicts +0.3-0.5%/mo on equity longs).
    Vasquez (2017, JFQA).
    """
    slope = iv60 - iv30
    return _cs_rank(slope)


def build_iv_term_convexity_signal(
    iv30: pd.DataFrame, iv60: pd.DataFrame, iv90: pd.DataFrame,
) -> pd.DataFrame:
    """IV Term Convexity = iv30 - 2*iv60 + iv90 (second-difference / curvature).

    Positive = U-shaped (high near + far, low middle = vol expected to dip
    then rise — anomalous, mean-reverts).
    Negative = hump (mid-tenor expensive — typically pre-event positioning).

    Sign: NEGATIVE (negative convexity = hump = event positioning =
    typically reverses → predicts higher returns). Vasquez (2017) extension
    using term-structure curvature instead of just slope.
    """
    convexity = iv30 - 2.0 * iv60 + iv90
    return _cs_rank(-convexity)


def build_iv_short_term_slope_signal(
    iv7: pd.DataFrame, iv30: pd.DataFrame,
) -> pd.DataFrame:
    """Short-term IV slope = iv30 - iv7. Captures near-term vol expectation
    different from the 30-60d slope (events expected in <2wks vs <2mo).
    Sign: positive (steep short-end = stress that's about to resolve).
    """
    slope = iv30 - iv7
    return _cs_rank(slope)


def build_iv_long_term_slope_signal(
    iv90: pd.DataFrame, iv365: pd.DataFrame,
) -> pd.DataFrame:
    """Long-end IV slope = iv365 - iv90. Captures structural vol expectation
    (1yr vs 90d) — orthogonal to the standard 30-60d term slope.
    Sign: positive (long-end higher = expecting sustained higher vol).
    """
    slope = iv365 - iv90
    return _cs_rank(slope)


def build_implied_move_signal(implied_move: pd.DataFrame) -> pd.DataFrame:
    """Implied move = ATM straddle / spot at ~30 DTE. Sign: NEGATIVE (high
    implied move = expensive vol = expected to mean-revert lower; rich
    straddles bleed back when no event materializes). Beber-Brandt (2010).
    """
    return _cs_rank(-implied_move)


# ─────────────────────────────────────────────────────────────────────────────
# Bakshi-Kapadia-Madan (2003) model-free risk-neutral moment signals.
# Computed from the full OTM smile (not just ATM IV) by the BKM kernel
# integrals — see src/options_adapter/implied_moments.py.
# Estimated IC_IR 0.10-0.15 per Conrad-Dittmar-Ghysels (2013), Bali et al.
# ─────────────────────────────────────────────────────────────────────────────

def build_implied_skew_signal(implied_skew: pd.DataFrame) -> pd.DataFrame:
    """BKM risk-neutral skewness signal.

    RN-skew is typically NEGATIVE for equity names (downside fear in the
    smile). Cross-sectionally, names with the most negative skew tend to
    be over-hedged / oversold and mean-revert, while names with less
    negative (or positive) skew are bullishly positioned.

    Sign: POSITIVE — higher rank (= less negative skew) predicts higher
    forward returns. Conrad, Dittmar & Ghysels (2013, JoF):
    "Ex Ante Skewness and Expected Stock Returns".
    """
    return _cs_rank(implied_skew)


def build_implied_kurt_signal(implied_kurt: pd.DataFrame) -> pd.DataFrame:
    """BKM risk-neutral kurtosis signal.

    Higher RN-kurt = market pricing fatter tails / jump risk. In the
    cross-section, high-kurt names often coincide with crash-fear pricing
    that subsequently mean-reverts (Conrad-Dittmar-Ghysels 2013 find a
    positive kurtosis-return relationship).

    Sign: POSITIVE — higher rank (= higher kurtosis) predicts higher
    forward returns.
    """
    return _cs_rank(implied_kurt)


# ─────────────────────────────────────────────────────────────────────────────
# Forward IV curve signals (Vasquez 2017 extension)
#
# Forward IV is derived by variance additivity:
#     fwd_iv(t1->t2) = sqrt(max((iv_t2^2*t2 - iv_t1^2*t1) / (t2 - t1), 0))
# See src/options_adapter/forward_curve.py for full derivation + Orats
# verification (within 0.5 vol pts of Orats's published fwd30_20/fwd60_30/
# fwd90_60). Encodes how the market expects vol to evolve at the *forward*
# tenor, not just the spot term-structure level.
#
# OVERLAP WITH iv_term_convexity: convexity = iv30 - 2*iv60 + iv90 is the
# 2nd discrete difference of the spot IV curve. The forward-curve slope
# (fwd60_90 - fwd30_60) is a difference of forward rates and contains the
# same curvature information, weighted by variance time. They are
# correlated but not identical — forward IVs are the model-consistent
# transformation. Treat as related signals (decorrelate in the model
# stage rather than dropping one).
# ─────────────────────────────────────────────────────────────────────────────

def build_fwd_curve_slope_signal(
    fwd60_90: pd.DataFrame, fwd30_60: pd.DataFrame,
) -> pd.DataFrame:
    """Forward IV curve slope = fwd60_90 - fwd30_60.

    Predicts vol regime change: a steep upward forward curve means the
    market is pricing higher vol *for future periods* than it expects in
    the near term — typically over-extrapolation of recent vol fears.

    Sign: NEGATIVE (steep forward curve = vol overpriced for the future
    = the option-writer side wins on average → underlying tends to rally
    while implied vol mean-reverts down).
    """
    slope = fwd60_90 - fwd30_60
    return _cs_rank(-slope)


def build_realized_vs_forward_signal(
    returns: pd.DataFrame, fwd30_60: pd.DataFrame, rv_window: int = 30,
) -> pd.DataFrame:
    """Realized vol over past 30d minus forward expectation (fwd30_60).

    When realized vol > forward IV, the options market is under-pricing
    vol that already happened — vol-sellers are wrong, the underlying
    has been beaten down further than its forward vol suggests it should
    be → mean-reversion long bias.

    Sign: positive (realized > forward = vol-sellers wrong → stocks
    underpriced → predicts higher returns).
    """
    realized_vol = returns.rolling(rv_window).std() * np.sqrt(252)
    spread = realized_vol - fwd30_60
    return _cs_rank(spread)


def build_long_forward_premium_signal(
    fwd90_180: pd.DataFrame, iv30: pd.DataFrame,
) -> pd.DataFrame:
    """Long-forward premium = fwd90_180 - iv30.

    Captures the long-end vol premium: how expensive is the
    90-to-180-day forward vol relative to the front-month spot IV. A
    high premium means dealers are bidding up far-dated vol — typically
    crowded vol-buying that mean-reverts.

    Sign: NEGATIVE (high long-end premium = expensive long-term vol =
    over-hedged = mean-reverts → underlying outperforms).
    """
    premium = fwd90_180 - iv30
    return _cs_rank(-premium)


def build_risk_reversal_25d_signal(
    iv30_25d_call: pd.DataFrame, iv30_25d_put: pd.DataFrame,
) -> pd.DataFrame:
    """Risk Reversal at 25-delta = 25d call IV - 25d put IV.
    Sign: NEGATIVE (negative risk reversal = put skew = oversold = predicts higher returns).
    Xing, Zhang, Zhao (2010, JFQA): stocks with steepest put skew underperform by 10.9%/yr.
    """
    rr = iv30_25d_call - iv30_25d_put
    return _cs_rank(-rr)  # negate per academic finding


def build_crash_risk_signal(
    iv30_5d_put: Optional[pd.DataFrame], iv30_atm: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Crash Risk = 10-delta (deep OTM) put IV - ATM IV. Higher = more tail risk priced.
    Sign: positive (high crash risk premium predicts higher returns per Kelly-Jiang).
    Note: requires 10-delta or 5-delta put IV; we approximate with vol5 if available.
    """
    if iv30_5d_put is None:
        return None
    crash = iv30_5d_put - iv30_atm
    return _cs_rank(crash)


def build_oi_concentration_signal(
    call_oi: pd.DataFrame, put_oi: pd.DataFrame,
) -> pd.DataFrame:
    """OI Concentration = call_oi / (call_oi + put_oi).
    Sign: positive (call-OI-heavy = bullish positioning).
    """
    total = call_oi + put_oi
    concentration = call_oi / total.replace(0, np.nan)
    return _cs_rank(concentration)


def build_exern_vol_regime_signal(
    exErnIv30: pd.DataFrame, iv30: pd.DataFrame,
) -> pd.DataFrame:
    """Earnings-event-premium vol regime signal.

    spread = iv30 - exErnIv30
        = the portion of 30-day implied vol attributable to a known earnings
          event inside the window. Orats' `exErnIv30d` strips out the discrete
          event-premium analytically (the "steady-state" forward vol);
          subtracting from raw iv30 isolates the earnings premium directly.

    A high spread means the options market is pricing a large catalyst that
    has not yet occurred. Empirically the realized event move falls short of
    the implied move on average -> vol crushes post-event, so equities with
    the highest event premium underperform near-term (vol-sellers / dispersion
    traders harvest this systematically).

    Sign: NEGATIVE (high event premium -> expensive vol about to crush ->
    expected forward return is lower).

    Inputs may be raw Orats panels (vol-points 0-100) or the
    `compute_exern_proxy` output for forward dates after Orats cancellation;
    units must match between the two arguments. Output is the standard
    cross-sectional rank in [0, 1].
    """
    spread = iv30 - exErnIv30
    return _cs_rank(-spread)  # negate: high event premium -> low signal


# ─────────────────────────────────────────────────────────────────────────────
# Master builder: build all options signals from IV panels
# ─────────────────────────────────────────────────────────────────────────────

def build_options_signals(
    iv_panels: Dict[str, pd.DataFrame],
    returns: Optional[pd.DataFrame] = None,
    target_tickers: Optional[pd.Index] = None,
    target_date_index: Optional[pd.DatetimeIndex] = None,
    enable: Optional[Dict[str, bool]] = None,
    earnings_calendar: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build all options signals from cached IV panels.

    Parameters
    ----------
    iv_panels : dict
        Output from OratsLoader.load_iv_panel(), with keys like
        'iv30', 'iv60', 'slope', 'vol25', 'vol75', 'cVolu', 'pVolu', etc.
    returns : DataFrame (date × ticker), optional
        Daily returns. Required for VRP and RV-IV spread signals.
    target_tickers : Index, optional
        Ticker universe to align panels to. If None, use intersection.
    target_date_index : DatetimeIndex, optional
        Trading calendar to align/forward-fill to.
    enable : dict, optional
        Per-signal enable flags. None = enable all.

    Returns
    -------
    Dict of signal_name → (date × ticker) ranked [0, 1] DataFrame.
    """
    if enable is None:
        enable = {
            "dcpvolspread": True, "smileslope": True, "cpvolspread": True,
            "dvolcall": True, "dvolput": True,
            "iv_rank": True, "rv_iv_spread": True, "variance_premium": True,
            "iv_term_slope": True, "risk_reversal_25d": True,
            "crash_risk": True, "oi_concentration": True,
            "exern_vol_regime": True,
        }

    # OPTIONS_SIGNAL_SET: "all" (default, 15 signals) or "validated" (top-5 IC).
    # Validated subset drops 10 lower-IC signals that contribute more dilution
    # than alpha per iso_OPTIONS_baseline audit. 3 bonus Orats signals
    # (implied_borrow, dividend_surprise, etf_skew_relative) follow the same
    # rule via their default-True pattern below.
    _signal_set = os.environ.get("OPTIONS_SIGNAL_SET", "all").lower().strip()
    if _signal_set == "validated":
        enable = {k: (k in _VALIDATED_SIGNAL_SET and v) for k, v in enable.items()}
        print(f"      [opts] OPTIONS_SIGNAL_SET=validated → top-5 high-IC only "
              f"({sorted([k for k, v in enable.items() if v])})")
    elif _signal_set not in ("all", ""):
        warnings.warn(
            f"OPTIONS_SIGNAL_SET='{_signal_set}' unrecognized; using 'all'. "
            f"Valid values: 'all', 'validated'."
        )

    # OPTIONS_MIN_COVERAGE: drop ticker columns below this non-null fraction.
    # 0.30 means ticker needs options data on ≥30% of dates to be kept.
    _min_cov = float(os.environ.get("OPTIONS_MIN_COVERAGE", "0.0"))
    # OPTIONS_WINSORIZE: cross-sectional clip percentile on raw inputs (per row).
    # 0.01 means clip top/bottom 1% per date before ranking.
    _wins_pct = float(os.environ.get("OPTIONS_WINSORIZE", "0.0"))
    if _min_cov > 0 or _wins_pct > 0:
        print(f"      [opts] data-quality filters: min_coverage={_min_cov:.2f}, "
              f"winsorize_pct={_wins_pct:.3f}")

    signals: Dict[str, pd.DataFrame] = {}

    # Helper to align panels to target index, then optionally apply data-quality
    # filters (coverage threshold + cross-sectional winsorization).
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        out = df.copy()
        if target_tickers is not None:
            out = out.reindex(columns=target_tickers)
        if target_date_index is not None:
            out = out.reindex(index=target_date_index, method="ffill", limit=5)
        # Apply coverage threshold BEFORE winsorize so sparse tickers don't
        # pollute the per-row quantile bounds.
        out = _apply_coverage_threshold(out, _min_cov)
        out = _winsorize_cs(out, _wins_pct)
        return out

    # Field names verified against actual /datav2/cores response (2026-04-16)
    # Full term structure (added 2026-04-17 to enable convexity / short-end / long-end signals)
    iv7  = _align(iv_panels.get("iv7d"))    # short-end (1wk)
    iv14 = _align(iv_panels.get("iv14d"))   # 2wk
    iv30 = _align(iv_panels.get("iv30d"))   # constant maturity ATM IV (combined call+put)
    iv60 = _align(iv_panels.get("iv60d"))
    iv90 = _align(iv_panels.get("iv90d"))
    iv180 = _align(iv_panels.get("iv180d"))
    iv365 = _align(iv_panels.get("iv365d"))
    slope = _align(iv_panels.get("slope"))   # call-delta-based; positive = put skew
    # Delta-bucketed IVs at 30d (low number = low call delta = OTM call):
    #   dlt5Iv30d  = 5-delta call (deep OTM call upside skew)
    #   dlt25Iv30d = 25-delta call (OTM call)
    #   dlt75Iv30d = 75-delta call ≡ 25-delta put (OTM put)
    #   dlt95Iv30d = 95-delta call ≡ 5-delta put (deep OTM put / crash protection)
    vol25_call = _align(iv_panels.get("dlt25Iv30d"))   # 25-delta call IV
    vol25_put = _align(iv_panels.get("dlt75Iv30d"))    # 25-delta put IV (= 75d call)
    vol5_put = _align(iv_panels.get("dlt95Iv30d"))     # 5-delta put IV (deep OTM crash)
    # Proxy for cp_vol_spread (call_iv - put_iv at 25-delta, since /cores has no separate
    # call/put ATM IV — only combined iv30d). Positive = bullish call skew positioning.
    cp_spread_proxy = _align(iv_panels.get("cp_vol_spread_proxy"))
    cvolu = _align(iv_panels.get("cVolu"))
    pvolu = _align(iv_panels.get("pVolu"))
    coi = _align(iv_panels.get("cOi"))
    poi = _align(iv_panels.get("pOi"))
    # ── BONUS signal source fields ──
    borrow30 = _align(iv_panels.get("borrow30"))
    annIdiv = _align(iv_panels.get("annIdiv"))
    annActDiv = _align(iv_panels.get("annActDiv"))
    etfSlopeRatio = _align(iv_panels.get("etfSlopeRatio"))
    impliedMove = _align(iv_panels.get("impliedMove"))
    # Earnings-stripped IV (Orats `exErnIv30d` historically; for forward
    # dates after Orats cancellation use compute_exern_proxy() to populate
    # this key from iv30 + EODHD earnings calendar).
    exErnIv30 = _align(iv_panels.get("exErnIv30d"))
    # Forward IV curve panels (Vasquez 2017): pure-math derivatives of
    # the spot IV term structure. Available from chain_to_smv_summary
    # (Tradier-derived) and Orats historical alike.
    fwd30_60 = _align(iv_panels.get("fwd30_60"))
    fwd60_90 = _align(iv_panels.get("fwd60_90"))
    fwd90_180 = _align(iv_panels.get("fwd90_180"))
    # BKM (2003) risk-neutral moments at ~30 DTE — full-smile-derived skew
    # and kurtosis (NOT just ATM IV). Computed in chain_to_smv_summary.
    implied_skew_30d = _align(iv_panels.get("implied_skew_30d"))
    implied_kurt_30d = _align(iv_panels.get("implied_kurt_30d"))

    # ── 5 C&Z signals (using proxy for call/put spread) ──
    if enable.get("dcpvolspread") and cp_spread_proxy is not None:
        try:
            signals["opt_dcpvolspread_signal"] = build_dcpvolspread_signal(cp_spread_proxy)
            print("      [opts] dcpvolspread_signal built (proxy from delta-bucketed IVs)")
        except Exception as e:
            warnings.warn(f"dcpvolspread failed: {e}")

    if enable.get("smileslope") and slope is not None:
        try:
            signals["opt_smileslope_signal"] = build_smileslope_signal(slope)
            print("      [opts] smileslope_signal built")
        except Exception as e:
            warnings.warn(f"smileslope failed: {e}")

    if enable.get("cpvolspread") and cp_spread_proxy is not None:
        try:
            signals["opt_cpvolspread_signal"] = build_cpvolspread_signal(cp_spread_proxy)
            print("      [opts] cpvolspread_signal built (proxy from delta-bucketed IVs)")
        except Exception as e:
            warnings.warn(f"cpvolspread failed: {e}")

    if enable.get("dvolcall") and cvolu is not None:
        try:
            signals["opt_dvolcall_signal"] = build_dvolcall_signal(cvolu)
            print("      [opts] dvolcall_signal built")
        except Exception as e:
            warnings.warn(f"dvolcall failed: {e}")

    if enable.get("dvolput") and pvolu is not None:
        try:
            signals["opt_dvolput_signal"] = build_dvolput_signal(pvolu)
            print("      [opts] dvolput_signal built")
        except Exception as e:
            warnings.warn(f"dvolput failed: {e}")

    # ── 7 bonus academic signals ──
    if enable.get("iv_rank") and iv30 is not None:
        try:
            signals["opt_iv_rank_signal"] = build_iv_rank_signal(iv30)
            print("      [opts] iv_rank_signal built")
        except Exception as e:
            warnings.warn(f"iv_rank failed: {e}")

    if enable.get("rv_iv_spread") and iv30 is not None and returns is not None:
        try:
            signals["opt_rv_iv_spread_signal"] = build_rv_iv_spread_signal(iv30, returns)
            print("      [opts] rv_iv_spread_signal built")
        except Exception as e:
            warnings.warn(f"rv_iv_spread failed: {e}")

    if enable.get("variance_premium") and iv30 is not None and returns is not None:
        try:
            signals["opt_variance_premium_signal"] = build_variance_premium_signal(iv30, returns)
            print("      [opts] variance_premium_signal built")
        except Exception as e:
            warnings.warn(f"variance_premium failed: {e}")

    if enable.get("iv_term_slope") and iv60 is not None and iv30 is not None:
        try:
            signals["opt_iv_term_slope_signal"] = build_iv_term_slope_signal(iv60, iv30)
            print("      [opts] iv_term_slope_signal built")
        except Exception as e:
            warnings.warn(f"iv_term_slope failed: {e}")

    if enable.get("risk_reversal_25d") and vol25_call is not None and vol25_put is not None:
        try:
            signals["opt_risk_reversal_25d_signal"] = build_risk_reversal_25d_signal(
                vol25_call, vol25_put
            )
            print("      [opts] risk_reversal_25d_signal built")
        except Exception as e:
            warnings.warn(f"risk_reversal_25d failed: {e}")

    if enable.get("crash_risk") and vol5_put is not None and iv30 is not None:
        try:
            sig = build_crash_risk_signal(vol5_put, iv30)
            if sig is not None:
                signals["opt_crash_risk_signal"] = sig
                print("      [opts] crash_risk_signal built (5-delta put = dlt95Iv30d)")
        except Exception as e:
            warnings.warn(f"crash_risk failed: {e}")

    if enable.get("oi_concentration") and coi is not None and poi is not None:
        try:
            signals["opt_oi_concentration_signal"] = build_oi_concentration_signal(coi, poi)
            print("      [opts] oi_concentration_signal built")
        except Exception as e:
            warnings.warn(f"oi_concentration failed: {e}")

    # Earnings-event-premium vol regime: rank of -(iv30 - exErnIv30).
    # Captures the discrete event premium baked into front-month IV. Works
    # historically off Orats `exErnIv30d` (FROZEN at 2026-04-15); for forward
    # dates feed the panel via exern_iv_extractor.compute_exern_proxy().
    if enable.get("exern_vol_regime", True) and exErnIv30 is not None and iv30 is not None:
        try:
            signals["opt_exern_vol_regime_signal"] = build_exern_vol_regime_signal(
                exErnIv30, iv30
            )
            print("      [opts] exern_vol_regime_signal built (earnings event premium)")
        except Exception as e:
            warnings.warn(f"exern_vol_regime failed: {e}")

    # ── 3 NEW bonus signals from real Orats fields (post-API-doc audit) ──
    if enable.get("implied_borrow", True) and borrow30 is not None:
        try:
            # High implied borrow = hard-to-borrow = short squeeze candidate
            # Sign: positive (high borrow predicts higher returns near-term — squeeze)
            # Caveat: also captures crowded shorts (mean-revert later) — IC_IR 0.07-0.12
            signals["opt_implied_borrow_signal"] = _cs_rank(borrow30)
            print("      [opts] implied_borrow_signal built (squeeze candidate)")
        except Exception as e:
            warnings.warn(f"implied_borrow failed: {e}")

    if (enable.get("dividend_surprise", True) and annIdiv is not None
            and annActDiv is not None):
        try:
            # Implied dividend > actual dividend = options market expects upward revision
            # Sign: positive (positive surprise predicts higher returns) — IC_IR 0.06-0.10
            div_surprise = annIdiv - annActDiv
            signals["opt_dividend_surprise_signal"] = _cs_rank(div_surprise)
            print("      [opts] dividend_surprise_signal built (implied vs actual div)")
        except Exception as e:
            warnings.warn(f"dividend_surprise failed: {e}")

    if enable.get("etf_skew_relative", True) and etfSlopeRatio is not None:
        try:
            # Stock skew vs sector ETF skew — anomalous skew vs peers
            # Sign: negative (high relative skew = oversold = predicts higher returns)
            signals["opt_etf_skew_relative_signal"] = _cs_rank(-etfSlopeRatio)
            print("      [opts] etf_skew_relative_signal built")
        except Exception as e:
            warnings.warn(f"etf_skew_relative failed: {e}")

    # ── 4 NEW signals: term-structure curvature + maturity slopes + implied move ──
    # Added 2026-04-17 to leverage extended IV term structure (7d-365d).
    # Default ON; high-IC academic signals (Vasquez 2017, Beber-Brandt 2010).
    if (enable.get("iv_convexity", True) and iv30 is not None
            and iv60 is not None and iv90 is not None):
        try:
            signals["opt_iv_convexity_signal"] = build_iv_term_convexity_signal(iv30, iv60, iv90)
            print("      [opts] iv_convexity_signal built (Vasquez extension)")
        except Exception as e:
            warnings.warn(f"iv_convexity failed: {e}")

    if enable.get("iv_short_term_slope", True) and iv7 is not None and iv30 is not None:
        try:
            signals["opt_iv_short_term_slope_signal"] = build_iv_short_term_slope_signal(iv7, iv30)
            print("      [opts] iv_short_term_slope_signal built (1wk vs 1mo)")
        except Exception as e:
            warnings.warn(f"iv_short_term_slope failed: {e}")

    if enable.get("iv_long_term_slope", True) and iv90 is not None and iv365 is not None:
        try:
            signals["opt_iv_long_term_slope_signal"] = build_iv_long_term_slope_signal(iv90, iv365)
            print("      [opts] iv_long_term_slope_signal built (3mo vs 1yr)")
        except Exception as e:
            warnings.warn(f"iv_long_term_slope failed: {e}")

    if enable.get("implied_move", True) and impliedMove is not None:
        try:
            signals["opt_implied_move_signal"] = build_implied_move_signal(impliedMove)
            print("      [opts] implied_move_signal built (straddle/spot)")
        except Exception as e:
            warnings.warn(f"implied_move failed: {e}")

    # ── BKM (2003) risk-neutral moments: full-smile skew + kurt ─────────────
    # IC_IR 0.10-0.15 per Conrad-Dittmar-Ghysels (2013). Computed in
    # chain_to_smv_summary via OTM-strike trapezoidal integration.
    if enable.get("implied_skew", True) and implied_skew_30d is not None:
        try:
            signals["opt_implied_skew_signal"] = build_implied_skew_signal(implied_skew_30d)
            print("      [opts] implied_skew_signal built (BKM full-smile skew)")
        except Exception as e:
            warnings.warn(f"implied_skew failed: {e}")

    if enable.get("implied_kurt", True) and implied_kurt_30d is not None:
        try:
            signals["opt_implied_kurt_signal"] = build_implied_kurt_signal(implied_kurt_30d)
            print("      [opts] implied_kurt_signal built (BKM full-smile kurt)")
        except Exception as e:
            warnings.warn(f"implied_kurt failed: {e}")

    # ── Earnings IV-Crush (Beber & Brandt 2010) ─────────────────────────────
    # Requires the EODHD earnings calendar to identify event dates. Pre vs
    # post iv30d collapse around earnings; high crush = clean print = drift
    # outperformance over next ~10 trading days. Estimated IC_IR 0.12-0.18.
    if (enable.get("iv_crush", True) and iv30 is not None
            and earnings_calendar is not None):
        try:
            from earnings_iv_crush import build_iv_crush_signal
            signals["opt_iv_crush_signal"] = build_iv_crush_signal(
                iv30, earnings_calendar
            )
            print("      [opts] iv_crush_signal built (Beber-Brandt earnings event)")
        except Exception as e:
            warnings.warn(f"iv_crush failed: {e}")

    # ── 3 NEW signals: forward IV curve (Vasquez 2017 extension) ────────────
    # Default ON; pure-math derivatives of the spot IV term structure (no
    # extra data needed). Encode forward vol expectations (model-consistent
    # via variance time additivity) rather than raw spot-curve levels.
    # Overlap with iv_convexity is intentional — see docstrings of
    # build_fwd_curve_slope_signal / forward_curve.py.
    if (enable.get("fwd_curve_slope", True) and fwd60_90 is not None
            and fwd30_60 is not None):
        try:
            signals["opt_fwd_curve_slope_signal"] = build_fwd_curve_slope_signal(
                fwd60_90, fwd30_60
            )
            print("      [opts] fwd_curve_slope_signal built (forward IV slope)")
        except Exception as e:
            warnings.warn(f"fwd_curve_slope failed: {e}")

    if (enable.get("realized_vs_forward", True) and fwd30_60 is not None
            and returns is not None):
        try:
            signals["opt_realized_vs_forward_signal"] = build_realized_vs_forward_signal(
                returns, fwd30_60
            )
            print("      [opts] realized_vs_forward_signal built (RV30 - fwd30_60)")
        except Exception as e:
            warnings.warn(f"realized_vs_forward failed: {e}")

    if (enable.get("long_forward_premium", True) and fwd90_180 is not None
            and iv30 is not None):
        try:
            signals["opt_long_forward_premium_signal"] = build_long_forward_premium_signal(
                fwd90_180, iv30
            )
            print("      [opts] long_forward_premium_signal built (fwd90_180 - iv30)")
        except Exception as e:
            warnings.warn(f"long_forward_premium failed: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # POINT-IN-TIME LAG: shift all signals by 1 trading day.
    #
    # Orats publishes /cores data with tradeDate = end-of-day t (after the 4pm
    # options settlement). A signal value at tradeDate=t is not observable
    # until ~4:15pm ET on day t. The backtest builds the target portfolio
    # using signals as of the rebalance day's close and trades on the next
    # session — without this shift, day-t signals silently encode same-day
    # EOD options data in decisions meant to execute at day-t's open,
    # producing a 1-day look-ahead.
    #
    # Lag_opt=1 guarantees: signal at date t uses only tradeDate ≤ t-1 data,
    # which is unambiguously available before t's open. Applying uniformly
    # here (rather than per-builder) keeps the contract simple.
    # Env var LAG_OPTIONS_SIGNALS (default "1") allows explicit override for
    # future research into the optimal lag length (could test lag=2 or 3).
    # ────────────────────────────────────────────────────────────────────────
    import os as _os
    _lag = int(_os.environ.get("LAG_OPTIONS_SIGNALS", "1"))
    if _lag > 0 and signals:
        for _name in list(signals.keys()):
            signals[_name] = signals[_name].shift(_lag)
        print(f"      [opts] applied shift({_lag}) to {len(signals)} options signals "
              f"(point-in-time lag — prevents EOD-options leakage into open trades)")

    return signals
