"""
implied_dividend_proxy.py
-------------------------
"Poor man's annIdiv" — workaround for Orats's proprietary annualized
implied-dividend field (`annIdiv`) once the Orats subscription ends.

We rebuild the dividend_surprise signal using only Tradier (chains) +
EODHD (historical dividends) + a risk-free curve.

Method
------
Put-call parity (continuous div, no early exercise):
    C - P = S - K * exp(-r*T) - PV(div_during_T)
=>  PV_div = S - K * exp(-r*T) - (C - P)
=>  div_during_T  = PV_div * exp(r*T)
=>  annualized    = div_during_T * (365 / T_days)

We average over near-ATM strikes and DTE >= target_min_dte expirations
to dilute borrow-cost contamination (borrow noise scales as borrow_rate*S*T,
so longer-T amortizes the same dollar borrow over more "annualized div"
denominator units).

Realistic accuracy
------------------
Orats annIdiv is ~95-99% of consensus; this proxy is ~75-85%. Expect the
dividend_surprise signal IC to drop from ~0.06-0.10 -> ~0.03-0.06.

Decision criteria (validate during 30-day Orats overlap):
    corr(ours, Orats) >= 0.70  -> ship
    0.50 <= corr      < 0.70   -> ship with downweighted blend
    corr              < 0.50   -> drop dividend_surprise signal entirely
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Risk-free curve helper (FRED treasury yields by DTE)
# ---------------------------------------------------------------------------

def interp_rf(rf_curve: Dict[int, float], dte: int) -> float:
    """Linearly interpolate annualized risk-free rate at given DTE.

    rf_curve example: {30: 0.045, 90: 0.046, 180: 0.047, 365: 0.048}
    Returns 0 if curve is empty.
    """
    if not rf_curve:
        return 0.0
    knots = sorted(rf_curve.keys())
    if dte <= knots[0]:
        return rf_curve[knots[0]]
    if dte >= knots[-1]:
        return rf_curve[knots[-1]]
    for i in range(len(knots) - 1):
        a, b = knots[i], knots[i + 1]
        if a <= dte <= b:
            w = (dte - a) / (b - a)
            return rf_curve[a] * (1 - w) + rf_curve[b] * w
    return rf_curve[knots[-1]]


# ---------------------------------------------------------------------------
# Part A: Implied annual dividend from Tradier chain (put-call parity)
# ---------------------------------------------------------------------------

@dataclass
class _PCParityRow:
    strike: float
    dte: int
    call_mid: float
    put_mid: float
    moneyness: float  # |K/S - 1|


def _extract_pc_pairs(
    chain: pd.DataFrame,
    spot: float,
    today: pd.Timestamp,
    moneyness_tol: float = 0.05,
) -> pd.DataFrame:
    """Pivot a Tradier chain into one row per (strike, expiration) with
    matched call_mid / put_mid. Filters to near-ATM (|K/S - 1| <= tol)."""
    if chain.empty:
        return pd.DataFrame()
    df = chain.copy()
    # Mid price (fall back to last if bid/ask missing)
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    last = pd.to_numeric(df.get("last"), errors="coerce")
    mid = (bid + ask) / 2.0
    mid = mid.where(mid > 0, last)
    df["mid"] = mid
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "expiration_date" in df.columns:
        df["exp"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    elif "expiration" in df.columns:
        df["exp"] = pd.to_datetime(df["expiration"], errors="coerce")
    else:
        return pd.DataFrame()
    df = df.dropna(subset=["mid", "strike", "exp", "option_type"])
    df = df[df["mid"] > 0]
    df["dte"] = (df["exp"] - today).dt.days
    df = df[df["dte"] > 0]
    df["moneyness"] = (df["strike"] / spot - 1.0).abs()
    df = df[df["moneyness"] <= moneyness_tol]
    if df.empty:
        return pd.DataFrame()

    calls = df[df["option_type"] == "call"][["strike", "exp", "dte", "mid", "moneyness"]]
    puts = df[df["option_type"] == "put"][["strike", "exp", "dte", "mid"]]
    pairs = calls.merge(
        puts, on=["strike", "exp"], suffixes=("_call", "_put"), how="inner"
    ).rename(columns={"mid_call": "call_mid", "mid_put": "put_mid",
                      "dte_call": "dte"})
    if "dte_put" in pairs.columns:
        pairs = pairs.drop(columns=["dte_put"])
    return pairs


def compute_implied_annual_dividend(
    chain: pd.DataFrame,
    underlying_price: float,
    risk_free_rate_curve: Dict[int, float],
    target_min_dte: int = 30,    # Was 60; daily polls only fetch 20/45/90 DTE.
    target_max_dte: int = 180,   # Was 270; nothing past 90d in our daily data.
    moneyness_tol: float = 0.10, # Was 0.05; AAPL has $5 strike spacing →
                                 # ±5% near $260 = ±$13, often only 2-3 strikes.
    today: Optional[pd.Timestamp] = None,
    min_pairs: int = 2,          # Was 3; with single-DTE pool, 3 is too tight.
    winsor_q: float = 0.10,
) -> Optional[float]:
    """Estimate annualized implied dividend ($/year) from put-call parity.

    Picks near-ATM (call, put) pairs at expirations between
    target_min_dte and target_max_dte. Longer DTEs dilute borrow-cost
    contamination. Winsorizes to control outliers; returns None if too few
    valid pairs.

    Args:
        chain: Tradier chain DataFrame (multi-expiration). Must have
            columns: option_type, strike, bid, ask, last,
            expiration_date | expiration.
        underlying_price: spot S.
        risk_free_rate_curve: {dte_days: annualized_rate} (e.g. FRED).
        target_min_dte: minimum DTE to include (default 60).
        target_max_dte: maximum DTE to include (default 270).
        moneyness_tol: max |K/S - 1| (default 0.05 = +/-5%).
        today: reference date for DTE computation.
        min_pairs: minimum (strike, exp) pairs required (else None).
        winsor_q: winsorize the per-pair annualized estimate before mean.

    Returns:
        Implied annualized dividend in $/year, or None if insufficient data.
    """
    if chain is None or chain.empty or underlying_price is None:
        return None
    if not (underlying_price > 0):
        return None
    if today is None:
        today = pd.Timestamp.today().normalize()

    pairs = _extract_pc_pairs(chain, underlying_price, today, moneyness_tol)
    if pairs.empty:
        return None
    pairs = pairs[(pairs["dte"] >= target_min_dte) & (pairs["dte"] <= target_max_dte)]
    if len(pairs) < min_pairs:
        # Fall back: relax DTE floor to 30 if we got nothing in long-dated bucket
        pairs2 = _extract_pc_pairs(chain, underlying_price, today, moneyness_tol)
        pairs2 = pairs2[(pairs2["dte"] >= 30) & (pairs2["dte"] <= target_max_dte)]
        if len(pairs2) < min_pairs:
            return None
        pairs = pairs2

    # Per-pair implied annualized dividend
    estimates = np.empty(len(pairs))
    for i, row in enumerate(pairs.itertuples(index=False)):
        T_years = row.dte / 365.0
        r = interp_rf(risk_free_rate_curve, int(row.dte))
        pv_div = underlying_price - row.strike * math.exp(-r * T_years) - (
            row.call_mid - row.put_mid
        )
        div_during_T = pv_div * math.exp(r * T_years)
        ann = div_during_T * (365.0 / row.dte)
        estimates[i] = ann

    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < min_pairs:
        return None

    # Winsorize and average. Dividends should be >= 0; clip negatives at 0
    # (negative implied div is typically borrow noise, not signal).
    lo, hi = np.quantile(estimates, [winsor_q, 1 - winsor_q])
    estimates = np.clip(estimates, lo, hi)
    estimates = np.clip(estimates, 0.0, None)
    return float(np.mean(estimates))


# ---------------------------------------------------------------------------
# Part B: Actual annual dividend from EODHD (TTM)
# ---------------------------------------------------------------------------

def compute_actual_annual_dividend(
    eodhd_dividends: pd.DataFrame | pd.Series,
    as_of: Optional[pd.Timestamp] = None,
    lookback_days: int = 365,
) -> float:
    """TTM dividend (sum of per-share payments in last `lookback_days`).

    Accepts either:
      - DataFrame with 'value' column indexed by ex-date, or
      - Series of per-share payments indexed by ex-date.
    """
    if eodhd_dividends is None or len(eodhd_dividends) == 0:
        return 0.0
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    if isinstance(eodhd_dividends, pd.DataFrame):
        if "value" not in eodhd_dividends.columns:
            return 0.0
        s = pd.to_numeric(eodhd_dividends["value"], errors="coerce").dropna()
    else:
        s = pd.to_numeric(eodhd_dividends, errors="coerce").dropna()
    if s.empty:
        return 0.0

    cutoff = as_of - pd.Timedelta(days=lookback_days)
    window = s[(s.index > cutoff) & (s.index <= as_of)]
    return float(window.sum()) if not window.empty else 0.0


# ---------------------------------------------------------------------------
# Part C: dividend_surprise proxy
# ---------------------------------------------------------------------------

def compute_dividend_surprise_proxy(
    tradier_chain: pd.DataFrame,
    eodhd_dividends: pd.DataFrame | pd.Series,
    underlying_price: float,
    risk_free_rate_curve: Dict[int, float],
    as_of: Optional[pd.Timestamp] = None,
    target_min_dte: int = 60,
) -> Optional[float]:
    """implied_div_annual - actual_div_annual_TTM (in $/year, per share).

    Returns None if implied dividend cannot be estimated. Note: callers
    should cross-sectionally rank these dollar values to [0, 1] across the
    universe before stacking into the feature matrix (matches Orats path).
    """
    implied = compute_implied_annual_dividend(
        tradier_chain, underlying_price, risk_free_rate_curve,
        target_min_dte=target_min_dte, today=as_of,
    )
    if implied is None:
        return None
    actual = compute_actual_annual_dividend(eodhd_dividends, as_of=as_of)
    return implied - actual


def build_dividend_surprise_panel(
    chains_by_ticker: Dict[str, pd.DataFrame],
    dividends_by_ticker: Dict[str, pd.DataFrame],
    spot_by_ticker: Dict[str, float],
    risk_free_rate_curve: Dict[int, float],
    as_of: Optional[pd.Timestamp] = None,
    target_min_dte: int = 60,
) -> pd.Series:
    """Build a one-day cross-section of dividend_surprise raw values
    (in $/year). Caller should rank cross-sectionally before stacking.

    Returns Series indexed by ticker. NaN where insufficient data.
    """
    out: Dict[str, float] = {}
    for tkr, chain in chains_by_ticker.items():
        spot = spot_by_ticker.get(tkr)
        if spot is None or not (spot > 0):
            out[tkr] = np.nan
            continue
        divs = dividends_by_ticker.get(tkr)
        val = compute_dividend_surprise_proxy(
            chain, divs if divs is not None else pd.Series(dtype=float),
            spot, risk_free_rate_curve,
            as_of=as_of, target_min_dte=target_min_dte,
        )
        out[tkr] = val if val is not None else np.nan
    return pd.Series(out, name="dividend_surprise_proxy")


# ---------------------------------------------------------------------------
# Validation: compare proxy to Orats annIdiv during overlap window
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    n_obs: int
    pearson_corr: float
    spearman_corr: float
    rank_corr_cs: float       # avg cross-sectional rank corr (signal IC analog)
    bias_dollars: float       # mean(ours - orats)
    rmse_dollars: float
    decision: str             # "ship", "blend", "drop"
    notes: str

    def to_dict(self) -> dict:
        return self.__dict__


def validate_against_orats(
    ours_panel: pd.DataFrame,   # date x ticker, $/year
    orats_annIdiv: pd.DataFrame,  # date x ticker, $/year
    ship_threshold: float = 0.70,
    blend_threshold: float = 0.50,
) -> ValidationResult:
    """Compare our implied-dividend panel to Orats annIdiv across the overlap.

    The decision threshold is on the cross-sectional rank correlation
    averaged across overlap days, NOT pooled Pearson — because the actual
    feature consumed downstream is a cross-sectional rank.
    """
    common_idx = ours_panel.index.intersection(orats_annIdiv.index)
    common_cols = ours_panel.columns.intersection(orats_annIdiv.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return ValidationResult(
            n_obs=0, pearson_corr=np.nan, spearman_corr=np.nan,
            rank_corr_cs=np.nan, bias_dollars=np.nan, rmse_dollars=np.nan,
            decision="drop", notes="no overlap",
        )
    a = ours_panel.loc[common_idx, common_cols]
    b = orats_annIdiv.loc[common_idx, common_cols]
    mask = a.notna() & b.notna()
    if mask.sum().sum() < 50:
        return ValidationResult(
            n_obs=int(mask.sum().sum()), pearson_corr=np.nan,
            spearman_corr=np.nan, rank_corr_cs=np.nan,
            bias_dollars=np.nan, rmse_dollars=np.nan,
            decision="drop", notes="<50 paired obs",
        )

    flat_a = a.where(mask).stack()
    flat_b = b.where(mask).stack()
    pearson = float(flat_a.corr(flat_b, method="pearson"))
    spearman = float(flat_a.corr(flat_b, method="spearman"))

    # Cross-sectional rank-corr per day, then average (this is what matters
    # for the downstream signal, which is cross-sectionally ranked).
    rank_a = a.rank(axis=1, pct=True)
    rank_b = b.rank(axis=1, pct=True)
    daily_rc = []
    for d in common_idx:
        ra = rank_a.loc[d].dropna()
        rb = rank_b.loc[d].dropna()
        common = ra.index.intersection(rb.index)
        if len(common) >= 20:
            daily_rc.append(ra.loc[common].corr(rb.loc[common]))
    rank_cs = float(np.nanmean(daily_rc)) if daily_rc else np.nan

    bias = float((flat_a - flat_b).mean())
    rmse = float(np.sqrt(((flat_a - flat_b) ** 2).mean()))

    if rank_cs >= ship_threshold:
        decision = "ship"
        notes = "cross-sectional rank corr meets ship threshold"
    elif rank_cs >= blend_threshold:
        decision = "blend"
        notes = "downweight blend (e.g., 0.5x weight) until further validation"
    else:
        decision = "drop"
        notes = "rank corr too low; drop dividend_surprise signal"

    return ValidationResult(
        n_obs=int(mask.sum().sum()),
        pearson_corr=pearson,
        spearman_corr=spearman,
        rank_corr_cs=rank_cs,
        bias_dollars=bias,
        rmse_dollars=rmse,
        decision=decision,
        notes=notes,
    )


def run_overlap_validation(
    chains_by_date_ticker: Dict[Tuple[pd.Timestamp, str], pd.DataFrame],
    dividends_by_ticker: Dict[str, pd.DataFrame],
    spot_panel: pd.DataFrame,                # date x ticker
    rf_curve_by_date: Dict[pd.Timestamp, Dict[int, float]],
    orats_annIdiv_panel: pd.DataFrame,       # date x ticker
    target_min_dte: int = 60,
) -> ValidationResult:
    """End-to-end validator. Builds our proxy panel from Tradier+EODHD over
    the overlap window, then compares to Orats annIdiv panel."""
    if not chains_by_date_ticker:
        return ValidationResult(0, np.nan, np.nan, np.nan, np.nan, np.nan,
                                "drop", "no chain data provided")

    # Group chains by date
    by_date: Dict[pd.Timestamp, Dict[str, pd.DataFrame]] = {}
    for (d, t), chain in chains_by_date_ticker.items():
        by_date.setdefault(pd.Timestamp(d), {})[t] = chain

    panels = []
    for d, chains in by_date.items():
        spots = (spot_panel.loc[d].to_dict()
                 if d in spot_panel.index else {})
        rf = rf_curve_by_date.get(d, {})
        s = build_dividend_surprise_panel(
            chains, dividends_by_ticker, spots, rf,
            as_of=d, target_min_dte=target_min_dte,
        )
        s.name = d
        panels.append(s)
    ours = pd.concat(panels, axis=1).T  # date x ticker
    ours.index = pd.DatetimeIndex(ours.index)

    # We validate the IMPLIED component (since we want to know whether our
    # implied estimate tracks Orats's). The actual-div component is mechanical
    # and not the source of error; if we wanted to match Orats's surprise
    # field directly, callers can pass orats_annIdiv - orats_annActDiv as
    # the comparison target.
    return validate_against_orats(ours, orats_annIdiv_panel)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic chain: S=100, K=100, T=180d, r=5%, true div=$2/yr
    # PV(div over 180d) = 1.0 (half a year) discounted ~= 0.975
    # parity: C - P = 100 - 100*exp(-0.05*0.4932) - 0.975 ~= 100 - 97.566 - 0.975
    #               = 1.459
    S = 100.0
    K = 100.0
    T_days = 180
    T = T_days / 365.0
    r = 0.05
    true_div_per_year = 2.0
    pv_div = true_div_per_year * T * math.exp(-r * T)  # PV of cash divs in window
    cp_diff = S - K * math.exp(-r * T) - pv_div
    chain = pd.DataFrame([
        {"option_type": "call", "strike": K, "bid": cp_diff,  "ask": cp_diff,
         "last": cp_diff,  "expiration_date":
            (pd.Timestamp("2026-04-16") + pd.Timedelta(days=T_days)).strftime("%Y-%m-%d")},
        {"option_type": "put",  "strike": K, "bid": 0.0,      "ask": 0.0,
         "last": 0.001,    "expiration_date":
            (pd.Timestamp("2026-04-16") + pd.Timedelta(days=T_days)).strftime("%Y-%m-%d")},
    ])
    rf = {30: 0.05, 90: 0.05, 180: 0.05, 365: 0.05}
    est = compute_implied_annual_dividend(
        chain, S, rf, target_min_dte=30, today=pd.Timestamp("2026-04-16"),
        min_pairs=1, winsor_q=0.0,
    )
    print(f"true div/yr = {true_div_per_year:.4f}, estimated = {est:.4f}")
