"""
tradier_orats_adapter.py
------------------------
Compute Orats `/cores`-equivalent fields (smile slope and delta-bucketed IVs)
from a raw Tradier options chain.

Tradier embeds Orats SMV greeks in their chain payload, but does NOT publish
the aggregated `/cores` fields (slope, dlt5Iv30d, ..., dlt95Iv30d). This
module reconstructs them per Orats' published methodology so a single panel
schema works for both historical (Orats direct) and live (Tradier) data.

Conventions:
    - All deltas are normalized to the *call-delta* convention in [0, 1].
      put_delta in [-1, 0]  ->  call_delta = 1 + put_delta
    - OTM filter: use calls for call_delta < 0.5, puts (re-expressed) for >= 0.5.
    - Smile fit:  IV = a + b * call_delta, where `slope = b / 10` to express
      "IV change per 10-delta increase in call-delta" (Orats convention).
      Sign: POSITIVE = put skew (deep-ITM-call / deep-OTM-put has higher IV).
    - Maturity projection: variance-time interpolation between bracketing
      expiries to a target DTE (default 30).

Public API:
    compute_delta_buckets(chain, target_dte=30, target_deltas=[...]) -> Dict
    compute_smile_slope(chain, target_dte=30) -> float
    compute_cores_row(chain, target_dte=30) -> Dict   # convenience wrapper
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# Column-name normalization (Tradier raw -> canonical names used here)
# ─────────────────────────────────────────────────────────────────────────────
_TRADIER_ALIASES = {
    "greek_delta": "delta",
    "greek_smv_vol": "iv",
    "greek_mid_iv": "iv_mid",
    "greek_bid_iv": "iv_bid",
    "greek_ask_iv": "iv_ask",
    "open_interest": "oi",
    "expiration_date": "expiry",
}


def _normalize(chain: pd.DataFrame) -> pd.DataFrame:
    df = chain.rename(columns=_TRADIER_ALIASES).copy()
    # Prefer SMV vol; fall back to mid IV.
    if "iv" not in df.columns and "iv_mid" in df.columns:
        df["iv"] = df["iv_mid"]
    if "expiry" in df.columns and not np.issubdtype(df["expiry"].dtype, np.datetime64):
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.lower().str[0]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def put_to_call_delta(delta: float | np.ndarray, option_type: str | np.ndarray):
    """Convert any delta to call-delta convention in [0, 1].

    Calls already have delta in [0, 1].  Puts have delta in [-1, 0]; the
    equivalent call-delta on the same strike is 1 + put_delta (Black-Scholes).
    """
    d = np.asarray(delta, dtype=float)
    t = np.asarray(option_type)
    is_put = (t == "p") | (t == "put") | (t == "P")
    return np.where(is_put, 1.0 + d, d)


def _quality_filter(df: pd.DataFrame, min_oi: int = 10,
                    max_iv_spread: float = 0.5) -> pd.DataFrame:
    """Drop low-quality rows (zero IV, wide IV spreads, tiny OI)."""
    keep = pd.Series(True, index=df.index)
    if "iv_bid" in df.columns:
        keep &= df["iv_bid"].fillna(0) > 0
    if "iv_bid" in df.columns and "iv_ask" in df.columns:
        spread = (df["iv_ask"] - df["iv_bid"]).abs()
        keep &= spread.fillna(np.inf) <= max_iv_spread
    if "oi" in df.columns:
        keep &= df["oi"].fillna(0) >= min_oi
    keep &= df["iv"].fillna(0) > 0
    keep &= df["iv"].fillna(0) < 5.0  # absurd IVs from stale quotes
    return df[keep].copy()


def _otm_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep OTM legs only — the Orats convention for smile fits."""
    cd = df["call_delta"].values
    is_call = (df["option_type"] == "c").values
    is_put = (df["option_type"] == "p").values
    # OTM call: call_delta < 0.5 ; OTM put: call_delta > 0.5
    keep = (is_call & (cd < 0.5)) | (is_put & (cd > 0.5))
    return df[keep].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-expiry smile fit & PCHIP interpolation
# ─────────────────────────────────────────────────────────────────────────────
def _fit_smile(deltas: np.ndarray, ivs: np.ndarray) -> Tuple[float, float, float]:
    """OLS fit IV = a + b * call_delta. Returns (a, b, r2)."""
    if len(deltas) < 3:
        return (np.nan, np.nan, 0.0)
    x = np.asarray(deltas, dtype=float)
    y = np.asarray(ivs, dtype=float)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return (float(a), float(b), float(r2))


def _pchip_iv(deltas: np.ndarray, ivs: np.ndarray,
              target_deltas: np.ndarray) -> np.ndarray:
    """Monotone PCHIP interpolation in delta-IV space.

    PCHIP preserves monotone segments and avoids the wild oscillations of
    cubic splines on noisy quotes. Fall back to linear if <2 unique deltas.
    """
    # Aggregate duplicate deltas (mean IV) and sort ascending.
    s = pd.Series(ivs, index=deltas).groupby(level=0).mean().sort_index()
    if len(s) < 2:
        return np.full_like(target_deltas, np.nan, dtype=float)
    if len(s) < 3:
        return np.interp(target_deltas, s.index.values, s.values,
                         left=np.nan, right=np.nan)
    pchip = PchipInterpolator(s.index.values, s.values, extrapolate=False)
    out = pchip(target_deltas)
    # PCHIP returns NaN outside hull; fall back to clamped linear extrapolation
    # using the two nearest points on the relevant side.
    mask_lo = target_deltas < s.index.values[0]
    mask_hi = target_deltas > s.index.values[-1]
    if mask_lo.any() or mask_hi.any():
        lin = np.interp(target_deltas, s.index.values, s.values)
        out = np.where(np.isnan(out), lin, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Maturity projection (variance-time interpolation)
# ─────────────────────────────────────────────────────────────────────────────
def _variance_time_interp(dte_a: float, iv_a: float,
                          dte_b: float, iv_b: float,
                          target_dte: float) -> float:
    """Interpolate IV linearly in *variance-time* (sigma^2 * T)."""
    if np.isnan(iv_a) and np.isnan(iv_b):
        return float("nan")
    if np.isnan(iv_a):
        return float(iv_b)
    if np.isnan(iv_b):
        return float(iv_a)
    if dte_b == dte_a:
        return float(iv_a)  # exact-expiry hit, no interpolation needed
    var_a = iv_a ** 2 * dte_a
    var_b = iv_b ** 2 * dte_b
    var_tgt = var_a + (var_b - var_a) * (target_dte - dte_a) / (dte_b - dte_a)
    var_tgt = max(var_tgt, 1e-10)
    return float(np.sqrt(var_tgt / max(target_dte, 1e-6)))


def _bracket_expiries(dtes: List[float], target: float) -> Tuple[float, float]:
    """Return two DTEs that bracket the target (or two nearest if extrapolating)."""
    arr = np.array(sorted(set(dtes)))
    if target <= arr[0]:
        return (arr[0], arr[1] if len(arr) > 1 else arr[0])
    if target >= arr[-1]:
        return (arr[-2] if len(arr) > 1 else arr[-1], arr[-1])
    hi = arr[arr >= target][0]
    lo = arr[arr <= target][-1]
    return (float(lo), float(hi))


# ─────────────────────────────────────────────────────────────────────────────
# Per-expiry computation
# ─────────────────────────────────────────────────────────────────────────────
def _per_expiry_metrics(df_exp: pd.DataFrame,
                        target_deltas: np.ndarray,
                        min_r2: float = 0.5) -> Dict[str, float]:
    """For one expiry, return {'slope': b, 'r2': r2, 'iv@d': ...}."""
    df_exp = _otm_only(df_exp)
    if len(df_exp) < 4:
        return {"slope": np.nan, "r2": 0.0,
                **{f"iv@{d:.2f}": np.nan for d in target_deltas}}

    deltas = df_exp["call_delta"].values
    ivs = df_exp["iv"].values
    a, b, r2 = _fit_smile(deltas, ivs)
    bucket_iv = _pchip_iv(deltas, ivs, target_deltas)

    # Quality gate: drop slope only when there IS a smile to fit but the fit is
    # poor.  A genuinely flat smile (low IV stdev) has near-zero R^2 yet is
    # informative (slope ~ 0), so don't punish it.
    iv_std = float(np.nanstd(ivs))
    smile_is_flat = iv_std < 0.01  # < 1 vol point of dispersion across deltas
    slope = b / 10.0 if (smile_is_flat or r2 >= min_r2) else np.nan
    return {
        "slope": slope,
        "r2": r2,
        **{f"iv@{d:.2f}": float(v) for d, v in zip(target_deltas, bucket_iv)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def compute_delta_buckets(
    chain: pd.DataFrame,
    target_dte: int = 30,
    target_deltas: Optional[List[float]] = None,
    *,
    asof: Optional[pd.Timestamp] = None,
    min_r2: float = 0.5,
) -> Dict[float, float]:
    """Return delta-bucketed IVs at `target_dte` projected via variance-time.

    Returns: {0.05: iv5, 0.25: iv25, 0.75: iv75, 0.95: iv95}
    """
    if target_deltas is None:
        target_deltas = [0.05, 0.25, 0.75, 0.95]
    tgt = np.array(target_deltas)

    df = _normalize(chain)
    df = _quality_filter(df)
    if df.empty:
        return {float(d): float("nan") for d in target_deltas}

    df["call_delta"] = put_to_call_delta(df["delta"].values,
                                         df["option_type"].values)

    asof = pd.Timestamp(asof) if asof is not None else df["expiry"].min().normalize()
    df["dte"] = (df["expiry"] - asof).dt.days.clip(lower=1)

    per_exp = {}
    for dte, grp in df.groupby("dte"):
        per_exp[float(dte)] = _per_expiry_metrics(grp, tgt, min_r2=min_r2)

    if not per_exp:
        return {float(d): float("nan") for d in target_deltas}

    lo_dte, hi_dte = _bracket_expiries(list(per_exp.keys()), float(target_dte))
    out = {}
    for d in target_deltas:
        key = f"iv@{d:.2f}"
        iv_lo = per_exp[lo_dte].get(key, np.nan)
        iv_hi = per_exp[hi_dte].get(key, np.nan)
        out[float(d)] = _variance_time_interp(lo_dte, iv_lo,
                                              hi_dte, iv_hi,
                                              float(target_dte))
    return out


def compute_smile_slope(
    chain: pd.DataFrame,
    target_dte: int = 30,
    *,
    asof: Optional[pd.Timestamp] = None,
    min_r2: float = 0.5,  # kept for signature compat; unused
) -> float:
    """Orats-style smile slope at target_dte = dlt75Iv30d - dlt25Iv30d.

    Positive = put skew (OTM puts richer than OTM calls).

    Methodology note: the previous OLS-fit slope (`IV ~ a + b*call_delta`
    over the full OTM hull) had near-zero rank correlation with Orats'
    published slope (validated 2026-04-17, N=30 large-caps: rank corr
    0.063, see results/validation/phase1_n30_2026-04-17.csv). The OTM
    smile is convex — the OLS line is dominated by tail noise (dlt5,
    dlt95) and per-ticker liquidity differences in the wings, which
    breaks cross-sectional ranking. Replacing with the bucket spread
    lifted rank corr to 0.609 (Orats' own internal consistency between
    its slope and dlt75-dlt25 spread is 0.785).
    """
    buckets = compute_delta_buckets(
        chain, target_dte=target_dte,
        target_deltas=[0.25, 0.75], asof=asof,
    )
    iv25 = buckets.get(0.25, float("nan"))
    iv75 = buckets.get(0.75, float("nan"))
    if np.isnan(iv25) or np.isnan(iv75):
        return float("nan")
    return float(iv75 - iv25)


def compute_cores_row(
    chain: pd.DataFrame,
    target_dte: int = 30,
    *,
    asof: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """Convenience: return one row of Orats /cores fields for a ticker."""
    buckets = compute_delta_buckets(chain, target_dte=target_dte, asof=asof)
    slope = compute_smile_slope(chain, target_dte=target_dte, asof=asof)
    return {
        "slope": slope,
        "dlt5Iv30d":  buckets[0.05],
        "dlt25Iv30d": buckets[0.25],
        "dlt75Iv30d": buckets[0.75],
        "dlt95Iv30d": buckets[0.95],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sample usage & self-tests
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_chain(slope_per_delta: float = 0.0,
                     atm_iv: float = 0.25,
                     dtes=(15, 30, 60),
                     asof: str = "2026-04-16") -> pd.DataFrame:
    """Build a synthetic Tradier-style chain with a controlled smile.

    IV(call_delta) = atm_iv + slope_per_delta * (call_delta - 0.5)
    -> slope coef = slope_per_delta, Orats slope = slope_per_delta / 10.
    """
    rng = np.random.default_rng(0)
    asof_ts = pd.Timestamp(asof)
    rows = []
    call_deltas = np.linspace(0.05, 0.95, 19)
    for dte in dtes:
        expiry = asof_ts + pd.Timedelta(days=dte)
        for cd in call_deltas:
            iv = atm_iv + slope_per_delta * (cd - 0.5)
            iv += rng.normal(0, 0.002)  # tiny noise
            # Call leg
            rows.append({
                "strike": round(100 * (1 - (cd - 0.5)), 2),
                "option_type": "c",
                "expiration_date": expiry,
                "greek_delta": float(cd),
                "greek_smv_vol": float(iv),
                "greek_bid_iv": float(iv) - 0.005,
                "greek_ask_iv": float(iv) + 0.005,
                "open_interest": 100,
            })
            # Mirrored put leg (same call_delta via 1 + put_delta)
            rows.append({
                "strike": round(100 * (1 - (cd - 0.5)), 2),
                "option_type": "p",
                "expiration_date": expiry,
                "greek_delta": float(cd - 1.0),
                "greek_smv_vol": float(iv),
                "greek_bid_iv": float(iv) - 0.005,
                "greek_ask_iv": float(iv) + 0.005,
                "open_interest": 100,
            })
    return pd.DataFrame(rows)


def _selftest() -> None:
    # Test 1: flat smile -> slope ~ 0
    flat = _synthetic_chain(slope_per_delta=0.0)
    s_flat = compute_smile_slope(flat, target_dte=30, asof="2026-04-16")
    assert abs(s_flat) < 0.005, f"flat smile expected slope~0, got {s_flat}"

    # Test 2: steep put skew -> POSITIVE slope (Orats convention)
    skew = _synthetic_chain(slope_per_delta=0.20)  # +0.20 IV per 1.0 delta
    s_skew = compute_smile_slope(skew, target_dte=30, asof="2026-04-16")
    assert s_skew > 0.015, f"put-skew expected positive slope, got {s_skew}"
    # Expected ~ 0.20 / 10 = 0.020
    assert abs(s_skew - 0.02) < 0.005, f"slope magnitude off: {s_skew}"

    # Test 3: call skew (negative slope)
    callskew = _synthetic_chain(slope_per_delta=-0.15)
    s_neg = compute_smile_slope(callskew, target_dte=30, asof="2026-04-16")
    assert s_neg < -0.01, f"call-skew expected negative slope, got {s_neg}"

    # Test 4: delta buckets ordering (put skew -> dlt95 > dlt5)
    b = compute_delta_buckets(skew, target_dte=30, asof="2026-04-16")
    assert b[0.95] > b[0.05], f"put skew should have iv95 > iv5: {b}"

    # Test 5: cores row schema
    row = compute_cores_row(skew, target_dte=30, asof="2026-04-16")
    assert set(row.keys()) == {
        "slope", "dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d"
    }
    print("All self-tests passed.")
    print(f"  flat slope     = {s_flat:+.5f}")
    print(f"  put-skew slope = {s_skew:+.5f}  (expected ~+0.020)")
    print(f"  call-skew slope= {s_neg:+.5f}  (expected ~-0.015)")
    print(f"  put-skew row   = {row}")


if __name__ == "__main__":
    _selftest()
