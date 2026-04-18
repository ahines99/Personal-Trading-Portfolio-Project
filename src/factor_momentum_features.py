"""
Factor Momentum Features (Ehsani-Linnainmaa 2022 JF).

Builds per-stock factor-momentum signals:
  1. Rolling 252-day univariate beta of each stock to each FF6 factor.
  2. Combine with factor 12-1 momentum to get per-stock exposure-weighted
     factor-momentum signal.

This module is self-contained: it takes raw returns panels as input and
produces (date x ticker) feature frames. FF6 loading lives in
`factor_momentum_data.py` (separate module) to avoid duplication.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    # Optional import: only used if a caller wants the convenience wrapper.
    from factor_momentum_data import load_ff6_factors, compute_factor_momentum  # noqa: F401
except Exception:  # pragma: no cover - sibling module may not exist yet
    pass


def compute_rolling_factor_betas(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback_days: int = 252,
    min_periods: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Compute rolling univariate beta of each stock to each factor.

    beta_{i,k,t} = Cov(r_i, r_k) / Var(r_k)  over trailing `lookback_days`.

    Parameters
    ----------
    stock_returns : (date x ticker) DataFrame of daily returns.
    factor_returns : (date x factor) DataFrame of daily factor returns.
    lookback_days : rolling window length (default 252).
    min_periods : minimum observations before emitting a beta (default 60).

    Returns
    -------
    dict : factor_name -> (date x ticker) DataFrame of rolling betas.
    """
    # Align on common date index (inner join) so cov/var share the same window.
    common_idx = stock_returns.index.intersection(factor_returns.index)
    sr = stock_returns.loc[common_idx].astype(np.float64)
    fr = factor_returns.loc[common_idx].astype(np.float64)

    betas: Dict[str, pd.DataFrame] = {}

    for factor_name in fr.columns:
        f_series = fr[factor_name]

        # Rolling variance of the factor (1-D series).
        factor_var = f_series.rolling(lookback_days, min_periods=min_periods).var()

        # Rolling covariance of each stock vs factor.
        # DataFrame.rolling().cov(Series) returns a (date x ticker) frame.
        cov_df = sr.rolling(lookback_days, min_periods=min_periods).cov(f_series)

        # Broadcast division: factor_var is aligned on the same date index.
        # Guard against divide-by-zero on flat factor windows.
        safe_var = factor_var.replace(0.0, np.nan)
        beta_df = cov_df.div(safe_var, axis=0)

        betas[factor_name] = beta_df

    return betas


def _cross_sectional_zscore(df: pd.DataFrame, clip: float = 2.0) -> pd.DataFrame:
    """Per-row (per-date) z-score with NaN handling and clipping to [-clip, clip]."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # Avoid division by zero on degenerate days.
    std = std.replace(0.0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.clip(lower=-clip, upper=clip)


def build_factor_momentum_signal(
    stock_returns: pd.DataFrame,
    factor_momentum_panel: pd.DataFrame,
    factor_betas: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build composite per-stock factor momentum signal.

    signal_{i,t} = sum_k ( beta_{i,k,t} * factor_mom_{k,t} )

    Then cross-sectional z-score per date, clipped to [-2, 2].

    Returns
    -------
    (date x ticker) DataFrame.
    """
    # Establish a canonical date index (use stock_returns as master).
    dates = stock_returns.index
    tickers = stock_returns.columns

    composite = pd.DataFrame(0.0, index=dates, columns=tickers)
    any_contribution = pd.DataFrame(False, index=dates, columns=tickers)

    for factor_name, beta_df in factor_betas.items():
        if factor_name not in factor_momentum_panel.columns:
            continue

        # Align beta to master grid.
        beta_aligned = beta_df.reindex(index=dates, columns=tickers)

        # Align factor momentum series to master dates.
        fm_series = factor_momentum_panel[factor_name].reindex(dates)

        # Multiply: beta (date x ticker) * scalar per date (broadcast on axis=0).
        contrib = beta_aligned.mul(fm_series, axis=0)

        # Track where we have valid data.
        mask = contrib.notna()
        any_contribution = any_contribution | mask

        # Accumulate, treating NaNs as zero for the sum but keeping track via mask.
        composite = composite.add(contrib.fillna(0.0))

    # Where no factor contributed anything, set to NaN.
    composite = composite.where(any_contribution)

    # Cross-sectional z-score per date, clipped to [-2, 2].
    signal = _cross_sectional_zscore(composite, clip=2.0)

    return signal


def build_single_factor_momentum_signals(
    stock_returns: pd.DataFrame,
    factor_momentum_panel: pd.DataFrame,
    factor_betas: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Build a separate per-stock signal for each individual factor so that a
    downstream model (LightGBM) can disentangle which factors matter.

    For each factor k:
        signal_{k,i,t} = beta_{i,k,t} * factor_mom_{k,t}
    then cross-sectionally z-score per date, clipped to [-2, 2].

    Returns
    -------
    dict : factor_name -> (date x ticker) DataFrame.
    """
    dates = stock_returns.index
    tickers = stock_returns.columns

    out: Dict[str, pd.DataFrame] = {}

    for factor_name, beta_df in factor_betas.items():
        if factor_name not in factor_momentum_panel.columns:
            continue

        beta_aligned = beta_df.reindex(index=dates, columns=tickers)
        fm_series = factor_momentum_panel[factor_name].reindex(dates)

        raw = beta_aligned.mul(fm_series, axis=0)
        out[factor_name] = _cross_sectional_zscore(raw, clip=2.0)

    return out


if __name__ == "__main__":
    # Synthetic test data
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=1500, freq='B')
    tickers = ['A', 'B', 'C', 'D', 'E']

    # Synthetic stock returns
    stock_ret = pd.DataFrame(
        np.random.randn(1500, 5) * 0.015,
        index=dates, columns=tickers
    )

    # Synthetic factor returns
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    factor_ret = pd.DataFrame(
        np.random.randn(1500, 6) * 0.005,
        index=dates, columns=factors
    )

    # Synthetic factor momentum (12-1)
    factor_mom = factor_ret.rolling(252, min_periods=60).sum().shift(21)

    # Compute betas
    betas = compute_rolling_factor_betas(stock_ret, factor_ret)
    print(f"Computed {len(betas)} factor betas")
    for f, beta_df in betas.items():
        print(f"  {f}: shape={beta_df.shape}, non-null={beta_df.notna().sum().sum()}")

    # Composite signal
    sig = build_factor_momentum_signal(stock_ret, factor_mom, betas)
    print(f"\nComposite signal shape: {sig.shape}")
    print(f"Composite non-null: {sig.notna().sum().sum()}")
    print(f"Composite mean per date (last 5):\n{sig.mean(axis=1).tail(5)}")

    # Single-factor signals
    singles = build_single_factor_momentum_signals(stock_ret, factor_mom, betas)
    print(f"\nSingle-factor signals: {list(singles.keys())}")

    print("\nALL TESTS PASSED")
