"""
factor_momentum_data.py
-----------------------
Self-contained loader for Ken French Fama-French 6-factor daily returns
(FF5 + Momentum), used as the input to a factor-momentum signal.

Downloads directly from the Ken French data library (same pattern as
`src/metrics.py::fama_french_regression`), caches a pickle under
`data/cache/ff6_factors_<start>_<end>_v1.pkl`, and computes a rolling
"12-1 style" momentum signal on factor returns themselves.

Graceful degradation: if the Ken French server is unreachable, loaders
return an empty DataFrame with a warning rather than raising, so callers
can skip the feature.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── cache directory (mirrors alt_data_loader.py convention) ─────────────────
_ROOT = Path(__file__).parent.parent
_CACHE = _ROOT / "data" / "cache"
_CACHE.mkdir(parents=True, exist_ok=True)

# Ken French daily factor URLs
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

_FF6_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]  # excludes RF


def _cache_path(start: str, end: str) -> Path:
    return _CACHE / f"ff6_factors_{start}_{end}_v1.pkl"


def _save_pkl(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pkl(path: Path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load FF6 cache {path}: {e}")
    return None


def load_ff6_factors(
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    use_cache: bool = True,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download Ken French 6-factor daily returns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF.

    Returns
    -------
    DataFrame with DateTimeIndex (business days) and columns:
        ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'RF']
    All values in DECIMAL form (not percent). Mkt-RF is excess market.

    Caches to data/cache/ff6_factors_<start>_<end>_v1.pkl to avoid re-download.

    On failure (Ken French server down), returns empty DataFrame with warning,
    so callers can degrade gracefully.
    """
    # Resolve cache path (respect user-supplied cache_dir if non-default)
    if cache_dir == "data/cache":
        cache_file = _cache_path(start, end)
    else:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cache_file = cdir / f"ff6_factors_{start}_{end}_v1.pkl"

    if use_cache:
        cached = _load_pkl(cache_file)
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

    try:
        # ── FF5 daily (Mkt-RF, SMB, HML, RMW, CMA, RF) ────────────────────
        ff = pd.read_csv(
            _FF5_URL,
            skiprows=3,
            index_col=0,
            compression="zip",
            on_bad_lines="skip",
        )
        valid_mask = ff.index.astype(str).str.match(r"^\d{8}$")
        ff = ff[valid_mask].apply(pd.to_numeric, errors="coerce") / 100.0
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
        ff.columns = [c.strip() for c in ff.columns]

        # ── Momentum daily (UMD / Mom) ────────────────────────────────────
        mom = pd.read_csv(
            _MOM_URL,
            skiprows=13,
            index_col=0,
            compression="zip",
            on_bad_lines="skip",
        )
        valid_mask_m = mom.index.astype(str).str.match(r"^\d{8}$")
        mom = mom[valid_mask_m].apply(pd.to_numeric, errors="coerce") / 100.0
        mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
        mom.columns = [c.strip() for c in mom.columns]
        mom_col = mom.columns[0]  # usually 'Mom'
        mom = mom.rename(columns={mom_col: "Mom"})[["Mom"]]

    except Exception as e:
        warnings.warn(
            f"Ken French FF6 download failed ({e!r}); "
            f"returning empty DataFrame. Caller should skip factor-momentum feature."
        )
        return pd.DataFrame(columns=_FF6_COLS)

    # ── Merge FF5 + Mom on common dates ───────────────────────────────────
    try:
        merged = ff.join(mom, how="inner")
    except Exception as e:
        warnings.warn(f"FF5/Mom join failed ({e!r}); returning empty DataFrame.")
        return pd.DataFrame(columns=_FF6_COLS)

    # Ensure all expected columns are present
    missing = [c for c in _FF6_COLS if c not in merged.columns]
    if missing:
        warnings.warn(
            f"FF6 download missing columns {missing}; returning empty DataFrame."
        )
        return pd.DataFrame(columns=_FF6_COLS)

    merged = merged[_FF6_COLS].copy()

    # ── Date window ───────────────────────────────────────────────────────
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        merged = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)]
    except Exception as e:
        warnings.warn(f"FF6 date filter failed ({e!r}); returning empty DataFrame.")
        return pd.DataFrame(columns=_FF6_COLS)

    merged = merged.dropna(how="all").sort_index()
    merged.index.name = "Date"

    if merged.empty:
        warnings.warn(
            f"FF6 data empty after filtering to [{start}, {end}]; "
            f"Ken French may not yet publish this range."
        )
        return pd.DataFrame(columns=_FF6_COLS)

    # ── Cache ──────────────────────────────────────────────────────────────
    try:
        _save_pkl(merged, cache_file)
    except Exception as e:
        warnings.warn(f"Failed to cache FF6 to {cache_file}: {e}")

    return merged


def compute_factor_momentum(
    ff6_df: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,  # standard 12-1 momentum: 252-21 = 231 days
) -> pd.DataFrame:
    """
    Compute rolling N-month return per factor (factor momentum signal).

    Per-date, per-factor: cumulative return from t-(lookback) to t-(skip).
    This is the "12-1 momentum" applied to factor returns themselves.

    Returns
    -------
    DataFrame with same index as ff6_df, columns same as ff6_df (excluding RF).
    Values are rolling cumulative returns (decimal).
    """
    if ff6_df is None or ff6_df.empty:
        return pd.DataFrame()

    factor_cols = [c for c in _FACTOR_COLS if c in ff6_df.columns]
    if not factor_cols:
        return pd.DataFrame(index=ff6_df.index)

    rets = ff6_df[factor_cols].astype(float).copy()

    # log-return trick: cumulative return over [t-lookback, t-skip]
    # = exp(sum_{i=t-lookback+1..t-skip} log(1+r_i)) - 1
    # Use log1p to stay numerically clean even for tiny factor returns.
    log_r = np.log1p(rets.fillna(0.0))

    # Rolling sum over the (lookback - skip) window, shifted by `skip_days`
    # so the window ends at t - skip_days (excludes the most-recent month).
    window = max(1, lookback_days - skip_days)
    rolling_sum = log_r.rolling(window=window, min_periods=window).sum()
    if skip_days > 0:
        rolling_sum = rolling_sum.shift(skip_days)

    fmom = np.expm1(rolling_sum)
    fmom.index = ff6_df.index
    fmom.columns = factor_cols
    return fmom


def get_factor_momentum_panel(
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Convenience: load FF6 + compute factor momentum + return panel ready for ML.

    Returns DataFrame with columns:
        ['Mkt-RF_mom', 'SMB_mom', 'HML_mom', 'RMW_mom', 'CMA_mom', 'Mom_mom']
    DateTimeIndex is business days.
    """
    ff6 = load_ff6_factors(start=start, end=end, use_cache=use_cache)
    if ff6.empty:
        return pd.DataFrame(
            columns=[f"{c}_mom" for c in _FACTOR_COLS]
        )

    fmom = compute_factor_momentum(ff6, lookback_days=252, skip_days=21)
    if fmom.empty:
        return pd.DataFrame(
            columns=[f"{c}_mom" for c in _FACTOR_COLS]
        )

    fmom = fmom.rename(columns={c: f"{c}_mom" for c in fmom.columns})
    fmom.index.name = "Date"
    return fmom


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_ff6_factors(start="2020-01-01", end="2024-01-01")
    print(f"FF6 shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty:
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Sample row:\n{df.head(1)}")

    fm = compute_factor_momentum(df, lookback_days=252, skip_days=21)
    print(f"\nFactor momentum shape: {fm.shape}")
    if not fm.empty:
        print(f"Last row:\n{fm.tail(1)}")

    panel = get_factor_momentum_panel(start="2020-01-01", end="2024-01-01")
    print(f"\nPanel shape: {panel.shape}")
    print(f"Panel columns: {panel.columns.tolist()}")
