"""
chain_to_smv_summary.py
-----------------------
Master orchestrator: Tradier raw chain → Orats-equivalent SMV summary.

Composes the 4 adapter modules:
- cmiv_interpolator         → iv30d, iv60d, iv90d
- tradier_orats_adapter     → slope, dlt5/25/75/95 Iv30d
- implied_borrow            → borrow30
- implied_dividend_proxy    → annIdiv

Output schema matches Orats /datav2/cores response so downstream code
(options_signals.py) works identically whether data comes from Orats
historical or Tradier-derived live.

Fail-safe design: any single adapter failing returns NaN for that field;
never raises (so daily polling can't crash on a single bad ticker).
"""

from __future__ import annotations
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent src/ dir to path so we can import sibling modules
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from cmiv_interpolator import compute_constant_maturity_iv
from tradier_orats_adapter import compute_cores_row
from implied_borrow import (
    compute_implied_borrow_rate,
    compute_implied_borrow_rate_interpolated,
)
from implied_dividend_proxy import compute_implied_annual_dividend
from options_adapter.forward_curve import compute_forward_iv_scalar
from options_adapter.implied_moments import compute_bkm_moments


# Default targets — match Orats /cores response schema
DEFAULT_TARGETS = {
    # Term structure: 7/14d unlocks short-term IV slope; 180d/365d unlocks
    # long-end carry and "convexity" (Vasquez 2017): iv30 - 2*iv60 + iv90.
    # Adapter will only compute fields for DTEs covered by available chains.
    "constant_maturity_dtes": [7, 14, 30, 60, 90, 180, 365],
    "delta_buckets": [0.05, 0.25, 0.75, 0.95],  # call-delta convention
    "borrow_target_dte": 30,
    "dividend_min_dte": 30,    # Was 60; daily polls fetch 20/45/90 DTE so
    "dividend_max_dte": 180,   # the 60-270 band excluded everything but 90d.
                               # 30-180 captures both 45d and 90d.
}


def _safe_call(label: str, fn, *args, **kwargs):
    """Call a function, log failure, return None instead of raising."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"[smv] {label} failed: {type(e).__name__}: {e}")
        return None


def chain_to_smv_summary(
    chain: pd.DataFrame,
    ticker: str,
    underlying_price: Optional[float] = None,
    risk_free_rates: Optional[Dict[int, float]] = None,
    dividend_yield: float = 0.0,
    asof: Optional[pd.Timestamp] = None,
    targets: Optional[Dict] = None,
    enable: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """Convert Tradier raw options chain → Orats-equivalent SMV summary dict.

    Parameters
    ----------
    chain : DataFrame with Tradier chain columns:
        strike, option_type ('call'/'put'), expiration_date, bid, ask, last,
        volume, open_interest, greek_delta, greek_smv_vol, greek_mid_iv, etc.
    ticker : ticker symbol (for the output dict).
    underlying_price : current spot. If None, derived from chain.
    risk_free_rates : {dte: rate} interpolation curve. Default uses 0.045 flat.
    dividend_yield : known dividend yield (from EODHD). Used to subtract from
        residual yield to isolate borrow rate.
    asof : evaluation date. Default = today.
    targets : custom DTE/delta targets. Default = DEFAULT_TARGETS.
    enable : per-component enable flags. Default = all enabled.

    Returns
    -------
    Dict with keys (Orats-compatible):
        ticker, tradeDate, stockPrice,
        iv30d, iv60d, iv90d,
        slope, dlt5Iv30d, dlt25Iv30d, dlt75Iv30d, dlt95Iv30d,
        borrow30, annIdiv,
        derivation_notes (list of warnings/issues)

    Failures return NaN for that specific field; never raises.
    """
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
    targets = targets or DEFAULT_TARGETS
    enable = enable or {
        "cmiv": True, "delta_buckets": True,
        "borrow": True, "dividend": True,
    }
    notes: List[str] = []

    # Default risk-free curve (flat 4.5% if not provided — calling code SHOULD
    # supply real FRED DGS rates for accuracy, but this is a safe fallback)
    if risk_free_rates is None:
        risk_free_rates = {30: 0.045, 60: 0.045, 90: 0.045, 180: 0.045, 365: 0.045}
        notes.append("default_rf_curve_used")

    # Derive underlying price if not provided
    if underlying_price is None:
        if "underlying_price" in chain.columns and not chain["underlying_price"].isna().all():
            underlying_price = float(chain["underlying_price"].dropna().iloc[0])
        elif "stockPrice" in chain.columns and not chain["stockPrice"].isna().all():
            underlying_price = float(chain["stockPrice"].dropna().iloc[0])
        else:
            # Derive from put-call parity at ATM strike
            mid = (chain["bid"] + chain["ask"]) / 2 if "bid" in chain else chain["last"]
            atm_strike = chain["strike"].median()
            underlying_price = float(atm_strike)  # rough fallback
            notes.append("underlying_price_estimated_from_atm_strike")

    # Initialize result dict with NaN defaults
    result = {
        "ticker": ticker,
        "tradeDate": asof,
        "stockPrice": underlying_price,
        # Constant maturity IVs (full term structure)
        "iv7d": np.nan, "iv14d": np.nan,
        "iv30d": np.nan, "iv60d": np.nan, "iv90d": np.nan,
        "iv180d": np.nan, "iv365d": np.nan,
        # Smile slope + delta buckets
        "slope": np.nan,
        "dlt5Iv30d": np.nan, "dlt25Iv30d": np.nan,
        "dlt75Iv30d": np.nan, "dlt95Iv30d": np.nan,
        # Carry decomposition
        "borrow30": np.nan,
        "annIdiv": np.nan,
        "annActDiv": np.nan,  # caller fills from EODHD
        # Volume / OI aggregates (unblocks dvolcall, dvolput, oi_concentration)
        "cVolu": 0, "pVolu": 0, "cOi": 0, "pOi": 0,
        # Implied earnings move (straddle proxy: ATM call + put price / spot)
        "impliedMove": np.nan,
        # Forward IV curve (Vasquez 2017 extension): variance-additivity
        # derived forward vols. Computed from CMIV outputs below — pure math,
        # no extra data needed. Tradier-only deployments get these for free.
        "fwd30_60": np.nan,    # forward IV from day 30 to day 60
        "fwd60_90": np.nan,    # forward IV from day 60 to day 90
        "fwd90_180": np.nan,   # forward IV from day 90 to day 180
        # BKM (2003) model-free risk-neutral moments at ~30 DTE.
        # Computed from the full OTM smile via trapezoidal integration of
        # the BKM kernels (see options_adapter/implied_moments.py).
        # implied_var_30d : annualized risk-neutral variance (sqrt -> IV-like)
        # implied_skew_30d: dimensionless RN skewness (typically NEGATIVE for equities)
        # implied_kurt_30d: dimensionless RN kurtosis (typically > 3 = fat tails)
        "implied_var_30d": np.nan,
        "implied_skew_30d": np.nan,
        "implied_kurt_30d": np.nan,
        "derivation_notes": notes,
    }

    if chain is None or len(chain) == 0:
        notes.append("empty_chain")
        return result

    # ── Volume / OI aggregation across all expirations ─────────────────────
    # Tradier exposes per-contract volume + open_interest in the chain. Sum
    # across the full multi-expiration chain (not just nearest expiry) so
    # the model sees true bullish/bearish positioning. Unblocks 4 dead
    # signals: dvolcall, dvolput, dcpvolspread (uses cp_vol_spread_proxy
    # already, but cVolu/pVolu enable level-based variants), oi_concentration.
    if "option_type" in chain.columns:
        for side, key_v, key_oi in [("call", "cVolu", "cOi"), ("put", "pVolu", "pOi")]:
            sub = chain[chain["option_type"] == side]
            if not sub.empty:
                if "volume" in sub.columns:
                    result[key_v] = int(pd.to_numeric(sub["volume"], errors="coerce").fillna(0).sum())
                if "open_interest" in sub.columns:
                    result[key_oi] = int(pd.to_numeric(sub["open_interest"], errors="coerce").fillna(0).sum())

    # ── Implied move (straddle/spot at ~30 DTE) ─────────────────────────────
    # The ATM straddle premium normalized by spot is the market's expected
    # one-period (DTE) absolute return. Highly predictive around earnings
    # (Beber-Brandt). When iv30d is interpolated and not on an actual
    # expiration, use the closest available expiration's ATM straddle.
    if underlying_price and underlying_price > 0 and "strike" in chain.columns:
        try:
            # Pick expiration closest to 30 DTE
            if "dte" in chain.columns:
                target_dte = 30
                avail_dtes = chain["dte"].dropna().unique()
                if len(avail_dtes) > 0:
                    closest_dte = avail_dtes[np.argmin(np.abs(avail_dtes - target_dte))]
                    sub = chain[chain["dte"] == closest_dte]
                else:
                    sub = chain
            else:
                sub = chain
            calls = sub[sub["option_type"] == "call"] if "option_type" in sub.columns else pd.DataFrame()
            puts = sub[sub["option_type"] == "put"] if "option_type" in sub.columns else pd.DataFrame()
            if not calls.empty and not puts.empty:
                # ATM = strike closest to spot
                atm_c = calls.iloc[(calls["strike"] - underlying_price).abs().argsort()[:1]]
                atm_p = puts.iloc[(puts["strike"] - underlying_price).abs().argsort()[:1]]
                # Mid prices
                c_mid = float((atm_c["bid"].iloc[0] + atm_c["ask"].iloc[0]) / 2) \
                    if "bid" in atm_c.columns and "ask" in atm_c.columns else float("nan")
                p_mid = float((atm_p["bid"].iloc[0] + atm_p["ask"].iloc[0]) / 2) \
                    if "bid" in atm_p.columns and "ask" in atm_p.columns else float("nan")
                if not (np.isnan(c_mid) or np.isnan(p_mid)):
                    result["impliedMove"] = (c_mid + p_mid) / float(underlying_price)
        except Exception as e:
            notes.append(f"impliedMove_failed: {type(e).__name__}")

    # ── BKM (2003) model-free risk-neutral moments at ~30 DTE ──────────────
    # Uses the full OTM smile (not just ATM IV). Risk-free rate at 30d.
    # Synthetic / legacy chains may lack a `dte` column — derive it from
    # `expiration_date` when missing so this works uniformly.
    chain_for_bkm = chain
    if "dte" not in chain_for_bkm.columns and "expiration_date" in chain_for_bkm.columns:
        chain_for_bkm = chain_for_bkm.copy()
        chain_for_bkm["dte"] = (
            pd.to_datetime(chain_for_bkm["expiration_date"]) - asof
        ).dt.days
    if underlying_price and underlying_price > 0 and "dte" in chain_for_bkm.columns:
        rf30 = float(risk_free_rates.get(30, 0.045))
        bkm = _safe_call(
            "bkm_moments",
            compute_bkm_moments,
            chain=chain_for_bkm,
            underlying_price=underlying_price,
            target_dte=30,
            risk_free_rate=rf30,
        )
        if bkm is not None:
            iv_v = bkm.get("implied_var", np.nan)
            sk_v = bkm.get("implied_skew", np.nan)
            ku_v = bkm.get("implied_kurt", np.nan)
            if iv_v is not None and not (isinstance(iv_v, float) and np.isnan(iv_v)):
                result["implied_var_30d"] = float(iv_v)
            if sk_v is not None and not (isinstance(sk_v, float) and np.isnan(sk_v)):
                result["implied_skew_30d"] = float(sk_v)
            if ku_v is not None and not (isinstance(ku_v, float) and np.isnan(ku_v)):
                result["implied_kurt_30d"] = float(ku_v)

    # ── Component 1: Constant Maturity IVs ──────────────────────────────────
    if enable.get("cmiv", True):
        cmiv = _safe_call(
            "cmiv",
            compute_constant_maturity_iv,
            chain=chain,
            target_dtes=targets["constant_maturity_dtes"],
            risk_free_rates=risk_free_rates,
            asof=asof.to_pydatetime() if hasattr(asof, "to_pydatetime") else asof,
            spot_hint=underlying_price,
        )
        if cmiv:
            for dte, iv in cmiv.items():
                key = f"iv{dte}d"
                if key in result:
                    result[key] = float(iv) if not np.isnan(iv) else np.nan

        # ── Forward IV curve (variance-additivity, Vasquez 2017) ───────────
        # Pure math from the CMIV outputs — fwd_iv(t1->t2) = sqrt(max(
        # (iv_t2^2 * t2 - iv_t1^2 * t1) / (t2 - t1), 0)). NaN when input is
        # NaN or fwd variance is negative (rare; only on inverted curves).
        for fwd_key, t_short, t_long, iv_s_key, iv_l_key in [
            ("fwd30_60",  30,  60, "iv30d", "iv60d"),
            ("fwd60_90",  60,  90, "iv60d", "iv90d"),
            ("fwd90_180", 90, 180, "iv90d", "iv180d"),
        ]:
            iv_s = result.get(iv_s_key)
            iv_l = result.get(iv_l_key)
            if iv_s is not None and iv_l is not None \
                    and not (isinstance(iv_s, float) and np.isnan(iv_s)) \
                    and not (isinstance(iv_l, float) and np.isnan(iv_l)):
                result[fwd_key] = compute_forward_iv_scalar(iv_s, iv_l, t_short, t_long)

    # ── Component 2: Smile Slope + Delta Buckets ────────────────────────────
    if enable.get("delta_buckets", True):
        cores = _safe_call(
            "delta_buckets",
            compute_cores_row,
            chain=chain,
            target_dte=30,
        )
        if cores:
            for k in ("slope", "dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d"):
                if k in cores:
                    val = cores[k]
                    result[k] = float(val) if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ) else np.nan

    # ── Component 3: Implied Borrow Rate ────────────────────────────────────
    if enable.get("borrow", True):
        rf_30 = risk_free_rates.get(30, 0.045)
        borrow = _safe_call(
            "borrow_strike_match",
            compute_implied_borrow_rate,
            chain=chain,
            underlying_price=underlying_price,
            risk_free_rate=rf_30,
            dividend_yield=dividend_yield,
            target_dte=targets["borrow_target_dte"],
            asof=asof,
        )
        # Fallback to interpolated method if strike-match returned NaN
        if borrow is None or (isinstance(borrow, float) and np.isnan(borrow)):
            borrow = _safe_call(
                "borrow_interp",
                compute_implied_borrow_rate_interpolated,
                chain=chain,
                underlying_price=underlying_price,
                risk_free_rate=rf_30,
                dividend_yield=dividend_yield,
                target_dte=targets["borrow_target_dte"],
                asof=asof,
            )
            if borrow is not None and not (isinstance(borrow, float) and np.isnan(borrow)):
                notes.append("borrow_used_interpolated_fallback")
        if borrow is not None and not (isinstance(borrow, float) and np.isnan(borrow)):
            result["borrow30"] = float(borrow)

    # ── Component 4: Implied Annual Dividend ────────────────────────────────
    # Orats /cores publishes `annIdiv` as dividend YIELD (fraction), not dollars.
    # compute_implied_annual_dividend returns dollars-per-share-per-year; convert
    # by dividing by underlying_price to match Orats' convention.
    # (Phase 1 validation found a unit mismatch — this is the fix.)
    if enable.get("dividend", True):
        ann_div = _safe_call(
            "ann_dividend",
            compute_implied_annual_dividend,
            chain=chain,
            underlying_price=underlying_price,
            risk_free_rate_curve=risk_free_rates,
            target_min_dte=targets["dividend_min_dte"],
            target_max_dte=targets["dividend_max_dte"],
            today=asof,
        )
        if ann_div is not None and not (isinstance(ann_div, float) and np.isnan(ann_div)):
            if underlying_price is not None and underlying_price > 0:
                # Convert $/share → yield (fraction); Orats reports as decimal
                # (e.g. 0.0247 for a 2.47% yield).
                result["annIdiv"] = float(ann_div) / float(underlying_price)
            else:
                # No price → can't convert; fall back to raw $/share (rare path)
                result["annIdiv"] = float(ann_div)
                result.setdefault("derivation_notes", []).append(
                    "annIdiv reported in $/share (no underlying_price for yield conversion)"
                )

        # EODHD fallback: when the put-call-parity proxy returns NaN or 0
        # (typical for low-yield or sparse-strike names — AAPL was returning
        # 0 against an actual ~0.4% yield), use the caller-supplied
        # `dividend_yield` (sourced from EODHD TTM dividend / spot upstream).
        # This recovers signal coverage for the dividend_surprise model
        # at the cost of being identical to the EODHD value (the
        # annIdiv vs annActDiv "surprise" would degenerate to 0 — handle
        # downstream by skipping the surprise signal for fallback rows).
        cur = result.get("annIdiv")
        if (cur is None or (isinstance(cur, float) and (np.isnan(cur) or cur == 0.0))) \
                and dividend_yield and dividend_yield > 0:
            result["annIdiv"] = float(dividend_yield)
            result.setdefault("derivation_notes", []).append(
                "annIdiv from EODHD TTM fallback (proxy returned 0/NaN)"
            )

    return result


def chain_to_smv_summary_batch(
    chains_by_ticker: Dict[str, pd.DataFrame],
    underlyings: Optional[Dict[str, float]] = None,
    risk_free_rates: Optional[Dict[int, float]] = None,
    dividend_yields: Optional[Dict[str, float]] = None,
    asof: Optional[pd.Timestamp] = None,
    targets: Optional[Dict] = None,
) -> pd.DataFrame:
    """Process multiple tickers in batch, returning a DataFrame of SMV summaries.

    Used for daily polling (process universe of 1500+ tickers from Tradier).

    Returns DataFrame with one row per ticker, columns = SMV fields.
    """
    underlyings = underlyings or {}
    dividend_yields = dividend_yields or {}
    rows = []
    for ticker, chain in chains_by_ticker.items():
        if chain is None or len(chain) == 0:
            continue
        try:
            row = chain_to_smv_summary(
                chain=chain,
                ticker=ticker,
                underlying_price=underlyings.get(ticker),
                risk_free_rates=risk_free_rates,
                dividend_yield=dividend_yields.get(ticker, 0.0),
                asof=asof,
                targets=targets,
            )
            rows.append(row)
        except Exception as e:
            warnings.warn(f"[smv batch] {ticker} failed: {e}")
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Self-test with synthetic chain
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_chain(spot: float = 100.0, n_strikes: int = 21,
                     n_expiries: int = 4) -> pd.DataFrame:
    """Generate a synthetic Tradier-shaped chain for testing.

    Creates a flat-vol surface with small put skew so all derived metrics
    have known expected values:
      iv30d, iv60d, iv90d ≈ 0.25
      slope ≈ small positive (put skew)
      borrow30 ≈ 0 (no borrow cost)
      annIdiv ≈ 0 (no dividend)
    """
    today = pd.Timestamp("2026-04-17")
    dtes = [21, 49, 84, 168]  # ~3w, ~7w, ~12w, ~24w

    rows = []
    for i, dte in enumerate(dtes[:n_expiries]):
        exp_date = today + pd.Timedelta(days=dte)
        T = dte / 365.0
        strikes = np.linspace(spot * 0.7, spot * 1.3, n_strikes)
        for K in strikes:
            # Synthetic IV with small put skew
            moneyness = K / spot - 1.0
            iv_call = 0.25 + 0.05 * max(-moneyness, 0)  # higher IV for OTM puts
            iv_put = iv_call

            # Crude Black-Scholes pricing
            d1 = (np.log(spot/K) + 0.5 * iv_call**2 * T) / (iv_call * np.sqrt(T))
            d2 = d1 - iv_call * np.sqrt(T)
            from scipy.stats import norm
            call_price = spot * norm.cdf(d1) - K * np.exp(-0.045*T) * norm.cdf(d2)
            put_price = K * np.exp(-0.045*T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            call_delta = norm.cdf(d1)
            put_delta = norm.cdf(d1) - 1

            for opt_type, price, iv, delta in [
                ("call", call_price, iv_call, call_delta),
                ("put", put_price, iv_put, put_delta),
            ]:
                rows.append({
                    "ticker": "TEST",
                    "strike": K,
                    "option_type": opt_type,
                    "expiration_date": exp_date,
                    "bid": max(price - 0.05, 0.01),
                    "ask": price + 0.05,
                    "last": price,
                    "volume": 100,
                    "open_interest": 500,
                    "greek_delta": delta,
                    "greek_smv_vol": iv,
                    "greek_mid_iv": iv,
                    "greek_gamma": 0.01,
                    "greek_theta": -0.01,
                    "greek_vega": 0.1,
                    "underlying_price": spot,
                })
    return pd.DataFrame(rows)


def _test_orchestrator() -> None:
    """Run orchestrator on synthetic chain, verify expected outputs."""
    print("─" * 60)
    print("TEST: Synthetic chain → SMV summary")
    print("─" * 60)

    chain = _make_test_chain(spot=100.0)
    print(f"  Synthetic chain: {len(chain)} contracts, "
          f"{chain['expiration_date'].nunique()} expiries")

    rf_curve = {30: 0.045, 60: 0.045, 90: 0.045, 180: 0.045, 365: 0.045}
    summary = chain_to_smv_summary(
        chain=chain,
        ticker="TEST",
        underlying_price=100.0,
        risk_free_rates=rf_curve,
        dividend_yield=0.0,
        asof=pd.Timestamp("2026-04-17"),
    )

    print()
    print("SMV Summary output:")
    for k, v in summary.items():
        if k == "derivation_notes":
            continue
        print(f"  {k}: {v}")

    # Acceptance tests
    print()
    print("Acceptance checks:")

    # 1. iv30d should be near 0.25
    iv30 = summary.get("iv30d")
    if iv30 is not None and not np.isnan(iv30):
        diff = abs(iv30 - 0.25)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"  [{status}] iv30d={iv30:.4f} near 0.25 (diff={diff:.4f})")
    else:
        print(f"  [FAIL] iv30d is NaN")

    # 2. slope should be positive (put skew)
    slope = summary.get("slope")
    if slope is not None and not np.isnan(slope):
        status = "PASS" if slope > 0 else "FAIL"
        print(f"  [{status}] slope={slope:.4f} positive (put skew)")
    else:
        print(f"  [FAIL] slope is NaN")

    # 3. delta buckets should be ordered: dlt5 < dlt25 < dlt75 < dlt95 (put skew)
    buckets = [summary.get(k) for k in ("dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d")]
    if all(b is not None and not np.isnan(b) for b in buckets):
        ordered = buckets[0] <= buckets[1] <= buckets[2] <= buckets[3]
        status = "PASS" if ordered else "WARN"
        print(f"  [{status}] delta buckets ordered: {[f'{b:.4f}' for b in buckets]}")
    else:
        print(f"  [INFO] some delta buckets are NaN")

    # 4. borrow30 should be near 0
    borrow = summary.get("borrow30")
    if borrow is not None and not np.isnan(borrow):
        diff = abs(borrow)
        status = "PASS" if diff < 0.05 else "FAIL"
        print(f"  [{status}] borrow30={borrow:.4f} near 0 (diff={diff:.4f})")
    else:
        print(f"  [INFO] borrow30 is NaN")

    # 5. annIdiv should be near 0 (no dividend)
    div = summary.get("annIdiv")
    if div is not None and not np.isnan(div):
        diff = abs(div)
        status = "PASS" if diff < 0.5 else "FAIL"
        print(f"  [{status}] annIdiv={div:.4f} near 0 (diff={diff:.4f})")
    else:
        print(f"  [INFO] annIdiv is NaN")

    print()
    print(f"Notes: {summary.get('derivation_notes', [])}")


if __name__ == "__main__":
    _test_orchestrator()
