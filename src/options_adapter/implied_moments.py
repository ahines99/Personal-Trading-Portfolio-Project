"""
implied_moments.py
------------------
Bakshi-Kapadia-Madan (2003) model-free risk-neutral moments.

Computes implied variance, skewness, and kurtosis from the full options
smile (NOT just ATM IV). Risk-neutral moments encode the market's
forward-looking distribution of underlying returns at expiration.

Reference
---------
Bakshi, G., Kapadia, N., & Madan, D. (2003). "Stock return characteristics,
skew laws, and the differential pricing of individual equity options."
Review of Financial Studies, 16(1), 101-143.

Formulas (BKM Eqs. 6-8, with τ = T - t)
---------------------------------------
Define V(t,τ), W(t,τ), X(t,τ) as integrals of (call/put)-weighted kernels
over OTM strikes K, with spot S:

  V(t,τ) = ∫_S^∞  [2 (1 - ln(K/S)) / K²]                C(K) dK
         + ∫_0^S  [2 (1 + ln(S/K)) / K²]                P(K) dK            (variance)

  W(t,τ) = ∫_S^∞  [(6 ln(K/S) - 3 ln²(K/S)) / K²]      C(K) dK
         + ∫_0^S  [(6 ln(S/K) + 3 ln²(S/K)) / K² · -1] P(K) dK            (cubic)

  X(t,τ) = ∫_S^∞  [(12 ln²(K/S) - 4 ln³(K/S)) / K²]    C(K) dK
         + ∫_0^S  [(12 ln²(S/K) + 4 ln³(S/K)) / K²]    P(K) dK            (quartic)

Then the risk-neutral mean of the log-return μ ≡ E^Q[ln(S_T/S)] is
  μ(t,τ) = e^(rτ) - 1 - (e^(rτ)/2) V - (e^(rτ)/6) W - (e^(rτ)/24) X       (BKM Eq. 9)

And the risk-neutral variance, skewness, kurtosis:
  σ²_RN = e^(rτ) V − μ²
  skew  = (e^(rτ) W − 3 μ e^(rτ) V + 2 μ³) / σ³_RN                       (BKM Eq. 10)
  kurt  = (e^(rτ) X − 4 μ e^(rτ) W + 6 e^(rτ) μ² V − 3 μ⁴) / σ⁴_RN       (BKM Eq. 11)

We discretize each integral using the trapezoidal rule on the observed
strike grid (OTM only).

Notes on the put cubic kernel sign
----------------------------------
The cubic moment is odd; for OTM puts (K < S), ln(S/K) > 0. The integrand
contribution to W must keep the same sign convention as the call side
(both signal "downside fear via puts" and "upside skew via calls" contribute
oppositely). Following Bakshi-Kapadia-Madan, the put kernel for W is
  -(6 ln(S/K) + 3 ln²(S/K)) / K²  ·  P(K)
i.e. negative of the symmetric form. Different authors write this with the
sign absorbed into the kernel; we keep the explicit negative.
"""

from __future__ import annotations
import math
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _midprice(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """(bid+ask)/2, NaN if either side missing or bid<=0."""
    bid = pd.to_numeric(bid, errors="coerce")
    ask = pd.to_numeric(ask, errors="coerce")
    mid = (bid + ask) / 2.0
    bad = (bid <= 0) | (ask <= 0) | bid.isna() | ask.isna()
    mid = mid.mask(bad)
    return mid


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal sum, robust to len<2 or NaN (returns 0 if can't integrate)."""
    if x is None or y is None or len(x) < 2:
        return 0.0
    msk = np.isfinite(x) & np.isfinite(y)
    if msk.sum() < 2:
        return 0.0
    # numpy 2.x renamed trapz -> trapezoid; fall back for 1.x.
    _trap = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201
    return float(_trap(y[msk], x[msk]))


def _select_target_dte_chain(
    chain: pd.DataFrame, target_dte: int, tol: int = 14,
) -> pd.DataFrame:
    """Pick rows from one expiration closest to `target_dte`.

    Returns empty DataFrame if no expiration falls within `tol` days.
    BKM is computed on a single expiration (no smoothing across maturities
    in this implementation — follows the original paper).
    """
    if "dte" not in chain.columns or chain.empty:
        return pd.DataFrame()
    dtes = pd.to_numeric(chain["dte"], errors="coerce")
    avail = dtes.dropna().unique()
    if len(avail) == 0:
        return pd.DataFrame()
    closest = avail[np.argmin(np.abs(avail - target_dte))]
    if abs(closest - target_dte) > tol:
        return pd.DataFrame()
    return chain[dtes == closest].copy()


def compute_bkm_moments(
    chain: pd.DataFrame,
    underlying_price: float,
    target_dte: int = 30,
    risk_free_rate: float = 0.045,
    min_otm_per_side: int = 5,
    max_spread_frac: float = 1.5,
) -> Dict[str, float]:
    """Compute Bakshi-Kapadia-Madan (2003) risk-neutral moments.

    Parameters
    ----------
    chain : DataFrame with columns:
        option_type ('call'/'put'), strike, bid, ask, expiration_date, dte
    underlying_price : current spot S
    target_dte : target maturity in days (default 30)
    risk_free_rate : annualized rf (decimal, e.g. 0.045)
    min_otm_per_side : minimum OTM strikes required on each of call & put side
    max_spread_frac : reject contracts where (ask-bid)/mid > this (illiquid)

    Returns
    -------
    {
        "implied_var":   risk-neutral variance σ²_RN  (annualized — see note),
        "implied_skew":  risk-neutral skewness (dimensionless),
        "implied_kurt":  risk-neutral kurtosis (dimensionless, >3 = fat tails)
    }
    Returns NaN values if strike density / data quality insufficient.

    Annualization
    -------------
    The integrals V, W, X yield τ-period (days/365) moments of log-return.
    `implied_var` is reported as σ²_RN / τ (annualized rate, comparable to
    iv30d²). Skewness and kurtosis are scale-invariant, so no normalization.
    """
    nan_out = {"implied_var": np.nan, "implied_skew": np.nan, "implied_kurt": np.nan}

    if chain is None or len(chain) == 0 or not (underlying_price and underlying_price > 0):
        return nan_out

    sub = _select_target_dte_chain(chain, target_dte=target_dte, tol=14)
    if sub.empty:
        return nan_out

    # Time to expiry (years) from the actual expiration we picked
    dte = float(pd.to_numeric(sub["dte"], errors="coerce").dropna().iloc[0])
    if dte <= 0:
        return nan_out
    tau = dte / 365.0
    S = float(underlying_price)
    r = float(risk_free_rate)
    erT = math.exp(r * tau)

    # Mid prices + liquidity filter
    sub = sub.copy()
    sub["mid"] = _midprice(sub.get("bid"), sub.get("ask"))
    sub = sub.dropna(subset=["mid", "strike", "option_type"])
    if sub.empty:
        return nan_out
    spread = (pd.to_numeric(sub["ask"], errors="coerce")
              - pd.to_numeric(sub["bid"], errors="coerce"))
    rel_spread = spread / sub["mid"].abs().replace(0, np.nan)
    sub = sub[(rel_spread.fillna(np.inf) <= max_spread_frac) & (sub["mid"] > 0)]
    if sub.empty:
        return nan_out

    # OTM only: calls with K > S, puts with K < S
    calls = sub[(sub["option_type"] == "call") & (sub["strike"] > S)].copy()
    puts = sub[(sub["option_type"] == "put") & (sub["strike"] < S)].copy()
    calls = calls.sort_values("strike").drop_duplicates("strike")
    puts = puts.sort_values("strike").drop_duplicates("strike")

    if len(calls) < min_otm_per_side or len(puts) < min_otm_per_side:
        return nan_out

    K_c = calls["strike"].astype(float).to_numpy()
    C = calls["mid"].astype(float).to_numpy()
    K_p = puts["strike"].astype(float).to_numpy()
    P = puts["mid"].astype(float).to_numpy()

    # ── Build kernels per BKM (2003) Eqs. 6-8 ─────────────────────────────
    # Call side (K > S)
    lnKS_c = np.log(K_c / S)
    w_V_c = (2.0 * (1.0 - lnKS_c)) / (K_c ** 2)
    w_W_c = (6.0 * lnKS_c - 3.0 * lnKS_c ** 2) / (K_c ** 2)
    w_X_c = (12.0 * lnKS_c ** 2 - 4.0 * lnKS_c ** 3) / (K_c ** 2)

    # Put side (K < S) — express in terms of ln(S/K) > 0
    lnSK_p = np.log(S / K_p)
    w_V_p = (2.0 * (1.0 + lnSK_p)) / (K_p ** 2)
    w_W_p = -((6.0 * lnSK_p + 3.0 * lnSK_p ** 2) / (K_p ** 2))   # NOTE the leading minus
    w_X_p = (12.0 * lnSK_p ** 2 + 4.0 * lnSK_p ** 3) / (K_p ** 2)

    # Trapezoidal integration over observed strike grid
    V_c = _trapz(w_V_c * C, K_c)
    V_p = _trapz(w_V_p * P, K_p)
    W_c = _trapz(w_W_c * C, K_c)
    W_p = _trapz(w_W_p * P, K_p)
    X_c = _trapz(w_X_c * C, K_c)
    X_p = _trapz(w_X_p * P, K_p)

    V = V_c + V_p
    W = W_c + W_p
    X = X_c + X_p

    if not np.isfinite(V) or V <= 0:
        return nan_out

    # ── BKM Eq. 9: risk-neutral mean log-return μ ─────────────────────────
    mu = erT - 1.0 - (erT / 2.0) * V - (erT / 6.0) * W - (erT / 24.0) * X

    # ── Risk-neutral variance, skew, kurt ─────────────────────────────────
    var_RN = erT * V - mu ** 2
    if not np.isfinite(var_RN) or var_RN <= 0:
        return nan_out
    sigma_RN = math.sqrt(var_RN)

    skew_num = erT * W - 3.0 * mu * erT * V + 2.0 * mu ** 3
    kurt_num = erT * X - 4.0 * mu * erT * W + 6.0 * erT * mu ** 2 * V - 3.0 * mu ** 4

    skew_RN = skew_num / (sigma_RN ** 3) if sigma_RN > 0 else np.nan
    kurt_RN = kurt_num / (sigma_RN ** 4) if sigma_RN > 0 else np.nan

    # Annualize variance so it's comparable to iv30² (a rate, not a τ-period
    # quantity). Skew/kurt are scale-invariant → no annualization.
    implied_var_ann = var_RN / tau

    return {
        "implied_var": float(implied_var_ann),
        "implied_skew": float(skew_RN) if np.isfinite(skew_RN) else np.nan,
        "implied_kurt": float(kurt_RN) if np.isfinite(kurt_RN) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _make_skewed_test_chain(spot: float = 100.0, dte: int = 30,
                            put_skew: float = 0.10) -> pd.DataFrame:
    """Build a synthetic chain with negative skew (typical equity smile).

    `put_skew` controls how much OTM puts trade above ATM IV.
    """
    from scipy.stats import norm
    today = pd.Timestamp("2026-04-17")
    exp_date = today + pd.Timedelta(days=dte)
    T = dte / 365.0
    r = 0.045
    base_iv = 0.25
    strikes = np.linspace(spot * 0.6, spot * 1.4, 41)
    rows = []
    for K in strikes:
        m = K / spot - 1.0
        # IV smile: higher for OTM puts (m<0), lower for OTM calls
        iv = base_iv + put_skew * max(-m, 0) + 0.02 * max(m, 0)
        d1 = (np.log(spot / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        c = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        p = K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        for opt_type, price in [("call", c), ("put", p)]:
            rows.append({
                "option_type": opt_type,
                "strike": K,
                "expiration_date": exp_date,
                "dte": dte,
                "bid": max(price - 0.05, 0.01),
                "ask": price + 0.05,
            })
    return pd.DataFrame(rows)


def _self_test() -> None:
    print("-" * 60)
    print("BKM moments self-test (synthetic chain with negative skew)")
    print("-" * 60)
    chain = _make_skewed_test_chain(spot=100.0, dte=30, put_skew=0.10)
    out = compute_bkm_moments(chain, underlying_price=100.0,
                              target_dte=30, risk_free_rate=0.045)
    print(out)
    iv_ann = math.sqrt(out["implied_var"]) if out["implied_var"] > 0 else float("nan")
    print(f"  sqrt(implied_var) = {iv_ann:.4f}  (expect ~0.25 ATM, slightly higher with skew)")
    print(f"  implied_skew      = {out['implied_skew']:.4f}  (expect NEGATIVE)")
    print(f"  implied_kurt      = {out['implied_kurt']:.4f}  (expect > 3)")


if __name__ == "__main__":
    _self_test()
