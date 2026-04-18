"""
SRAF (Notre Dame) pre-computed LM sentiment features.

Uses the Loughran-McDonald 10X Summaries CSV (197 MB, 1.25M filings, 1993-2025)
which contains pre-computed word counts (N_Positive, N_Negative, N_Uncertainty,
N_Litigious, N_Words) for every 10-K and 10-Q filed on SEC EDGAR.

This replaces the raw-text-parsing approach in lm_sentiment.py with a much
faster pre-computed path: no EDGAR download needed, no text parsing, just
read the CSV and compute ratios.

Features produced (all float32, date x ticker cross-sectional z-scores):
    sraf_net_tone      : (positive - negative) / total_words   (HIGH = bullish)
    sraf_uncertainty   : uncertainty / total_words              (HIGH = bearish)
    sraf_litigious     : litigious / total_words                (HIGH = bearish)
    sraf_tone_change   : net_tone - prior_filing_net_tone       (NEG shock = bearish)

Sign conventions match lm_sentiment.py for drop-in substitution.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SRAF_CSV_DEFAULT = "data/cache/sraf/lm_10x_summaries_1993_2025.csv"
_MIN_WORDS = 100  # filings shorter than this are treated as garbage


# ---------------------------------------------------------------------------
# CIK -> ticker mapping
# ---------------------------------------------------------------------------

def _load_cik_to_ticker() -> Dict[int, str]:
    """Build int(CIK) -> ticker map from the lazy_prices_downloader CIK map."""
    try:
        from src.lazy_prices_downloader import get_cik_map
    except ImportError:
        try:
            from lazy_prices_downloader import get_cik_map
        except ImportError as e:
            warnings.warn(f"sraf_sentiment: failed to load CIK map: {e}")
            return {}
    try:
        cik_map = get_cik_map()  # ticker -> zero-padded str CIK
        return {int(v): k for k, v in cik_map.items()}
    except Exception as e:
        warnings.warn(f"sraf_sentiment: CIK map load error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Cross-sectional z-score
# ---------------------------------------------------------------------------

def _xsec_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (cross-sectional) z-score."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_sraf_summaries(
    csv_path: str = SRAF_CSV_DEFAULT,
    start: str = "2013-01-01",
    universe: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load SRAF 10X summaries, map CIK -> ticker, compute sentiment ratios.

    Returns a DataFrame with columns:
        ticker, filing_date, form_type, n_words,
        net_tone, uncertainty_ratio, litigious_ratio
    """
    path = Path(csv_path)
    if not path.exists():
        warnings.warn(f"sraf_sentiment: CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) == 0:
        return pd.DataFrame()

    # Parse filing date
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["filing_date"])

    # Filter to start date
    df = df[df["filing_date"] >= pd.Timestamp(start)]

    # Filter to 10-K and 10-Q forms only
    df = df[df["FORM_TYPE"].isin(["10-K", "10-Q", "10-K-A", "10-Q-A"])].copy()

    # Map CIK -> ticker
    cik_to_ticker = _load_cik_to_ticker()
    if not cik_to_ticker:
        return pd.DataFrame()

    df["ticker"] = df["CIK"].map(cik_to_ticker)
    df = df.dropna(subset=["ticker"])

    # Filter to universe if provided
    if universe is not None:
        uni_set = {t.upper() for t in universe}
        df = df[df["ticker"].isin(uni_set)]

    if len(df) == 0:
        return pd.DataFrame()

    # Drop filings with too few words
    df = df[df["N_Words"] >= _MIN_WORDS].copy()

    # Compute sentiment ratios
    df["net_tone"] = (df["N_Positive"] - df["N_Negative"]) / df["N_Words"]
    df["uncertainty_ratio"] = df["N_Uncertainty"] / df["N_Words"]
    df["litigious_ratio"] = df["N_Litigious"] / df["N_Words"]

    return df[["ticker", "filing_date", "FORM_TYPE", "N_Words",
               "net_tone", "uncertainty_ratio", "litigious_ratio"]].copy()


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_sraf_sentiment_features(
    csv_path: str = SRAF_CSV_DEFAULT,
    trading_dates: Optional[pd.DatetimeIndex] = None,
    universe: Optional[List[str]] = None,
    forward_fill_days: int = 180,
    publication_lag_days: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end SRAF sentiment feature build.

    Returns dict of z-scored float32 panels:
        sraf_net_tone, sraf_uncertainty, sraf_litigious, sraf_tone_change
    """
    panel = load_sraf_summaries(csv_path=csv_path, start="2013-01-01", universe=universe)
    if len(panel) == 0:
        warnings.warn("sraf_sentiment: no filings loaded")
        return {}

    # Sort and compute tone change (current - prior filing per ticker)
    panel = panel.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    panel["prior_net_tone"] = panel.groupby("ticker")["net_tone"].shift(1)
    panel["tone_change"] = panel["net_tone"] - panel["prior_net_tone"]

    # Publication lag
    lag = pd.Timedelta(days=int(publication_lag_days))
    panel["effective_date"] = panel["filing_date"] + lag

    # Build trading-date index
    if trading_dates is None:
        start = panel["effective_date"].min()
        end = panel["effective_date"].max() + pd.Timedelta(days=forward_fill_days + 5)
        trading_dates = pd.bdate_range(start=start, end=end)
    trading_dates = pd.DatetimeIndex(trading_dates)

    tickers = sorted(panel["ticker"].unique().tolist())

    feature_map = {
        "sraf_net_tone": "net_tone",
        "sraf_uncertainty": "uncertainty_ratio",
        "sraf_litigious": "litigious_ratio",
        "sraf_tone_change": "tone_change",
        # Phase D: filing length as complexity proxy. Longer filings = more
        # disclosure burden = potential risk signal. Log-transformed to compress
        # the heavy right tail (mega-cap 10-Ks can be 200K+ words).
        "sraf_filing_length": "N_Words",
    }

    results: Dict[str, pd.DataFrame] = {}

    for feat_name, col in feature_map.items():
        sub = panel.dropna(subset=[col])[["ticker", "effective_date", col]].copy()
        if len(sub) == 0:
            continue

        # Pivot to (effective_date x ticker)
        wide = sub.pivot_table(
            index="effective_date", columns="ticker", values=col, aggfunc="last"
        )

        # Reindex to trading calendar, forward-fill up to limit
        wide = wide.reindex(trading_dates)
        wide = wide.ffill(limit=forward_fill_days)

        # Ensure all universe tickers present
        for t in tickers:
            if t not in wide.columns:
                wide[t] = np.nan

        # Log-transform filing length before z-scoring (heavy right tail)
        if col == "N_Words":
            wide = np.log1p(wide)

        # Cross-sectional z-score
        z = _xsec_zscore(wide)
        results[feat_name] = z.astype(np.float32)

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SRAF Sentiment Features - Smoke Test")
    print("=" * 60)

    features = build_sraf_sentiment_features(
        universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    )

    if not features:
        print("No features returned (CSV missing?)")
    else:
        for name, df in features.items():
            nz = df.notna().sum().sum()
            print(f"  {name}: shape={df.shape}, non-NaN={nz:,}")
        # Sample values
        sample = features.get("sraf_net_tone")
        if sample is not None:
            print(f"\nsraf_net_tone sample (last 5 dates):")
            print(sample.tail().to_string())
