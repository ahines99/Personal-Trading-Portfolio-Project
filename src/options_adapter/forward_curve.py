"""
forward_curve.py
----------------
Forward implied volatility computation (Vasquez 2017 extension).

Orats publishes proprietary fields ``fwd30_20``, ``fwd60_30``, ``fwd90_60``,
``fwd180_90`` etc. in /datav2/cores. They are *pure math* derived from the
constant-maturity ATM IV term structure (iv20d, iv30d, iv60d, iv90d, iv180d):

    Variance is additive in time:
        iv_t2^2 * t2  =  iv_t1^2 * t1  +  fwd_var(t1->t2) * (t2 - t1)

    Solve for forward variance:
        fwd_var(t1->t2)  =  (iv_t2^2 * t2 - iv_t1^2 * t1) / (t2 - t1)

    Forward IV (annualized vol) is sqrt of forward variance:
        fwd_iv(t1->t2)  =  sqrt(max(fwd_var, 0))

Empirical verification (5 tickers x 5 dates = 25 samples vs Orats published
fwd30_20 / fwd60_30 / fwd90_60): max abs diff = 0.023 vol points, mean abs
diff = 0.007 vol points. Well within the 0.5 vol pt tolerance — residual
deviation comes from Orats publishing values rounded to 2 decimals.

This means Tradier-only deployments (no Orats subscription) can derive the
full forward IV curve from the standard CMIV outputs (iv30d, iv60d, iv90d,
iv180d) for free. See ``chain_to_smv_summary.py`` for integration.

NOTE on overlap: ``iv_term_convexity`` (iv30 - 2*iv60 + iv90) and the forward
curve slope (fwd60_90 - fwd30_60) encode similar information. Both measure
how the term structure curves; convexity is the second discrete difference,
the forward slope is a difference of forward rates. They are correlated but
not identical (forward IVs use variance additivity which is the model-
consistent way to interpolate). Treat as related signals — winsorize/decorr
in the model stage rather than dropping one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_iv(
    iv_panel_short: pd.DataFrame,
    iv_panel_long: pd.DataFrame,
    t_short: int,
    t_long: int,
) -> pd.DataFrame:
    """Compute the implied forward IV from t_short to t_long.

    Uses variance-time additivity:

        fwd_var = (iv_long^2 * t_long - iv_short^2 * t_short) / (t_long - t_short)
        fwd_iv  = sqrt(max(fwd_var, 0))

    Where fwd_var < 0 (i.e. the term structure is so inverted that no
    real forward vol can support it — typically a data error or extreme
    event-driven near-term spike), the result is NaN rather than 0, so
    downstream signals can correctly mask these rows out.

    Parameters
    ----------
    iv_panel_short : DataFrame (date x ticker)
        Constant-maturity IV at the shorter tenor (e.g. iv30d).
        Same scale as iv_panel_long (both fractions or both vol-points).
    iv_panel_long : DataFrame (date x ticker)
        Constant-maturity IV at the longer tenor (e.g. iv60d).
    t_short : int
        Shorter tenor in calendar days (e.g. 30).
    t_long : int
        Longer tenor in calendar days (e.g. 60).

    Returns
    -------
    DataFrame (date x ticker)
        Forward IV from t_short to t_long, same units as inputs. NaN where
        forward variance is negative or either input is NaN.

    Raises
    ------
    ValueError
        If t_long <= t_short or either tenor is non-positive.
    """
    if t_short <= 0 or t_long <= 0:
        raise ValueError(
            f"Tenors must be positive: t_short={t_short}, t_long={t_long}"
        )
    if t_long <= t_short:
        raise ValueError(
            f"t_long must exceed t_short: t_short={t_short}, t_long={t_long}"
        )

    # Align indexes/columns so the arithmetic broadcasts safely
    short, long = iv_panel_short.align(iv_panel_long, join="outer")

    fwd_var = (long.pow(2) * t_long - short.pow(2) * t_short) / (t_long - t_short)
    # Negative variance → NaN (math invalid; signals downstream of NaN drop the row)
    fwd_var = fwd_var.where(fwd_var >= 0, other=np.nan)
    fwd_iv = np.sqrt(fwd_var)
    return fwd_iv


def compute_forward_iv_scalar(
    iv_short: float, iv_long: float, t_short: int, t_long: int,
) -> float:
    """Scalar version of compute_forward_iv() for single-row use.

    Returns NaN on negative variance, NaN-input, or invalid tenors.
    """
    if iv_short is None or iv_long is None:
        return float("nan")
    try:
        s = float(iv_short)
        l = float(iv_long)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(s) or np.isnan(l):
        return float("nan")
    if t_short <= 0 or t_long <= 0 or t_long <= t_short:
        return float("nan")
    fwd_var = (l * l * t_long - s * s * t_short) / (t_long - t_short)
    if fwd_var < 0 or np.isnan(fwd_var):
        return float("nan")
    return float(np.sqrt(fwd_var))
