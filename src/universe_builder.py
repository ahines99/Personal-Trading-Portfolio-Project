"""
universe_builder.py
-------------------
Builds and maintains the investable ticker universe.

Sources (all free, no API key required):
  1. S&P 500       — Wikipedia table (large cap, ~500 names)
  2. S&P 400       — Wikipedia table (mid cap, ~400 names)
  3. S&P 600       — Wikipedia table (small cap, ~600 names)
  4. Russell 2000  — iShares IWM ETF holdings CSV (small cap, ~2000 names)

Together these approximate the Russell 3000 (~3000 large/mid/small cap names),
which is the standard institutional long-short equity universe.

Usage:
    python src/universe_builder.py            # rebuild ticker_master.csv
    python src/universe_builder.py --preview  # print counts, don't write
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests

_ROOT       = Path(__file__).parent.parent
_TICKER_CSV = _ROOT / "data" / "ticker_master.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_ticker(raw: str) -> str:
    """Normalize ticker: strip whitespace, replace dots with hyphens (BRK.B → BRK-B)."""
    t = str(raw).strip().upper()
    t = t.replace(".", "-")      # yfinance uses BRK-B not BRK.B
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t


def _is_valid(ticker: str) -> bool:
    return bool(ticker) and 1 <= len(ticker) <= 6


# ── source 1 & 2 & 3 : Wikipedia S&P index tables ───────────────────────────

_WIKIPEDIA_URLS = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def _fetch_wikipedia_sp(index_name: str) -> List[str]:
    url = _WIKIPEDIA_URLS[index_name]
    try:
        tables = pd.read_html(url, header=0)
    except Exception as e:
        print(f"  [universe] Wikipedia {index_name} failed: {e}")
        return []

    tickers = []
    for table in tables:
        cols = [c.lower().replace(" ", "_") for c in table.columns]
        table.columns = cols
        # Try common column names for ticker symbol
        for col in ("symbol", "ticker", "ticker_symbol", "stock_symbol"):
            if col in cols:
                tickers += table[col].dropna().tolist()
                break

    cleaned = [_clean_ticker(t) for t in tickers]
    valid   = [t for t in cleaned if _is_valid(t)]
    print(f"  [universe] {index_name}: {len(valid)} tickers")
    return valid


# ── source 4 : iShares IWM (Russell 2000) ETF holdings ──────────────────────

_IWM_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)


def _fetch_iwm_holdings() -> List[str]:
    """
    Download iShares IWM holdings CSV.  The file has a multi-line header;
    the actual data starts after the row containing 'Ticker'.
    """
    try:
        resp = requests.get(_IWM_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        lines = resp.text.splitlines()

        # Find the header row
        header_idx = next(
            (i for i, line in enumerate(lines) if "Ticker" in line), None
        )
        if header_idx is None:
            print("  [universe] IWM: could not find header row")
            return []

        from io import StringIO
        data_str = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(data_str))
        df.columns = [c.strip() for c in df.columns]

        ticker_col = next((c for c in df.columns if "ticker" in c.lower()), None)
        if ticker_col is None:
            print("  [universe] IWM: ticker column not found")
            return []

        tickers = df[ticker_col].dropna().tolist()
        cleaned = [_clean_ticker(t) for t in tickers]
        valid   = [t for t in cleaned if _is_valid(t) and t != "-"]
        print(f"  [universe] IWM (Russell 2000): {len(valid)} tickers")
        return valid

    except Exception as e:
        print(f"  [universe] IWM download failed: {e}")
        return []


# ── source 5 : iShares IWB (Russell 1000) as large-cap supplement ───────────

_IWB_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)


def _fetch_iwb_holdings() -> List[str]:
    try:
        resp = requests.get(_IWB_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        lines = resp.text.splitlines()
        header_idx = next(
            (i for i, line in enumerate(lines) if "Ticker" in line), None
        )
        if header_idx is None:
            return []

        from io import StringIO
        data_str = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(data_str))
        df.columns = [c.strip() for c in df.columns]
        ticker_col = next((c for c in df.columns if "ticker" in c.lower()), None)
        if ticker_col is None:
            return []

        tickers = df[ticker_col].dropna().tolist()
        cleaned = [_clean_ticker(t) for t in tickers]
        valid   = [t for t in cleaned if _is_valid(t) and t != "-"]
        print(f"  [universe] IWB (Russell 1000): {len(valid)} tickers")
        return valid

    except Exception as e:
        print(f"  [universe] IWB download failed: {e}")
        return []


# ── known-bad ticker patterns to exclude ─────────────────────────────────────
#
# BUG FIX: The previous implementation mixed regex patterns and literal tickers
# into a single set, then tried to detect regex via `startswith("r")` — but
# none of the patterns actually started with "r" (they start with "." or "^"),
# so ALL regex patterns were DEAD and only literal string equality worked.
# Preferred shares (BRK-PA), warrants (-W/-WS/-WT), rights (-R), and units
# (-U/-UN) were silently NOT being filtered.
#
# Fix: split into an explicit literal set and a compiled-regex list, with a
# helper that checks both.

# Literal tickers to exclude (exact match) — ETFs / funds that may sneak in.
_EXCLUDE_LITERALS = {
    "SPY", "QQQ", "IWM", "IWB", "DIA", "VOO", "VTI",
    "GLD", "SLV", "TLT", "HYG", "LQD",
}

# Regex patterns for non-common-equity instruments.
_EXCLUDE_REGEXES = [
    re.compile(r".*-P[A-Z]?$"),           # Preferred shares: BRK-PA, WFC-PL, etc.
    re.compile(r".*-W[ST]?$"),            # Warrants: XYZ-W, XYZ-WS, XYZ-WT
    re.compile(r".*-R$"),                 # Rights
    re.compile(r".*-U[N]?$"),             # Units: XYZ-U, XYZ-UN
    re.compile(r"^[A-Z]{1,5}\.P[A-Z]$"),  # Preferred shares dot-notation
]


def _is_excluded(ticker: str) -> bool:
    if ticker in _EXCLUDE_LITERALS:
        return True
    return any(rx.match(ticker) for rx in _EXCLUDE_REGEXES)


# ── master builder ────────────────────────────────────────────────────────────

def build_universe(preview: bool = False) -> pd.DataFrame:
    """
    Fetch all sources, deduplicate, filter, and return a sorted DataFrame
    with columns: ticker, source.
    Writes to data/ticker_master.csv unless preview=True.
    """
    print("\n[universe_builder] Fetching ticker universe...")

    all_tickers: dict[str, str] = {}  # ticker → first source seen

    # S&P Composite 1500 from Wikipedia
    for index_name in ("SP500", "SP400", "SP600"):
        for t in _fetch_wikipedia_sp(index_name):
            if t not in all_tickers and not _is_excluded(t):
                all_tickers[t] = index_name
        time.sleep(1.0)

    # Russell 1000 (supplement large cap)
    for t in _fetch_iwb_holdings():
        if t not in all_tickers and not _is_excluded(t):
            all_tickers[t] = "Russell1000"

    # Russell 2000 (small cap)
    for t in _fetch_iwm_holdings():
        if t not in all_tickers and not _is_excluded(t):
            all_tickers[t] = "Russell2000"

    df = pd.DataFrame(
        [{"ticker": t, "source": s} for t, s in all_tickers.items()]
    ).sort_values("ticker").reset_index(drop=True)

    print(f"\n[universe_builder] Total unique tickers: {len(df)}")
    print(df["source"].value_counts().to_string())

    if not preview:
        _TICKER_CSV.parent.mkdir(parents=True, exist_ok=True)
        # Write just the ticker column (data_loader expects single-column CSV)
        df[["ticker"]].to_csv(_TICKER_CSV, index=False)
        print(f"\n[universe_builder] Written to {_TICKER_CSV}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Print counts without writing to disk")
    args = parser.parse_args()
    build_universe(preview=args.preview)
