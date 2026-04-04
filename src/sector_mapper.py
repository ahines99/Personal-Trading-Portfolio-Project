"""
sector_mapper.py
----------------
Maps tickers to GICS sectors using SEC EDGAR SIC codes.

Why EDGAR instead of yfinance for sectors:
  - Covers delisted stocks (yfinance returns 404)
  - No rate limiting issues (10 req/sec vs yfinance's constant 401s)
  - Official SEC data, not scraped
  - Covers 10,433 tickers including historical filers

SIC → GICS sector mapping based on standard industry classification:
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC=

Rate limit: SEC EDGAR allows 10 requests/second.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_EDGAR_HEADERS = {
    "User-Agent": "quant-research-project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# ---------------------------------------------------------------------------
# SIC code → GICS Sector mapping
# ---------------------------------------------------------------------------
# SIC codes are 4-digit industry codes. We map the first 2 digits (division)
# to GICS sectors. This is an approximation but covers 95%+ of cases correctly.

_SIC_TO_SECTOR = {
    # Agriculture, Forestry, Fishing (01-09)
    range(100, 1000): "Consumer Defensive",

    # Mining (10-14)
    range(1000, 1500): "Energy",

    # Construction (15-17)
    range(1500, 1800): "Industrials",

    # Manufacturing (20-39) — subdivided
    range(2000, 2100): "Consumer Defensive",     # Food
    range(2100, 2200): "Consumer Defensive",     # Tobacco
    range(2200, 2400): "Consumer Cyclical",      # Textiles/Apparel
    range(2400, 2600): "Industrials",            # Lumber/Wood
    range(2600, 2700): "Basic Materials",        # Paper
    range(2700, 2800): "Communication Services", # Printing/Publishing
    range(2800, 2900): "Healthcare",             # Chemicals/Pharma
    range(2900, 3000): "Energy",                 # Petroleum Refining
    range(3000, 3100): "Basic Materials",        # Rubber/Plastics
    range(3100, 3200): "Consumer Cyclical",      # Leather
    range(3200, 3300): "Basic Materials",        # Stone/Glass
    range(3300, 3500): "Basic Materials",        # Primary Metals
    range(3500, 3570): "Industrials",            # Industrial Machinery
    range(3570, 3580): "Technology",             # Computer Hardware (AAPL, DELL)
    range(3580, 3600): "Industrials",            # Misc Industrial Machinery
    range(3600, 3700): "Technology",             # Electronic Equipment
    range(3700, 3800): "Industrials",            # Transportation Equipment
    range(3800, 3900): "Healthcare",             # Instruments
    range(3900, 4000): "Consumer Cyclical",      # Misc Manufacturing

    # Transportation (40-49)
    range(4000, 4500): "Industrials",            # Railroad/Trucking
    range(4500, 4600): "Industrials",            # Air Transportation
    range(4600, 4700): "Industrials",            # Pipelines
    range(4700, 4800): "Industrials",            # Transportation Services
    range(4800, 4900): "Communication Services", # Telecom
    range(4900, 5000): "Utilities",              # Electric/Gas/Water

    # Wholesale Trade (50-51)
    range(5000, 5200): "Consumer Cyclical",

    # Retail Trade (52-59)
    range(5200, 5400): "Consumer Cyclical",      # Building materials, general merch
    range(5400, 5500): "Consumer Defensive",     # Food stores
    range(5500, 5600): "Consumer Cyclical",      # Auto dealers
    range(5600, 5700): "Consumer Cyclical",      # Apparel stores
    range(5700, 5800): "Consumer Cyclical",      # Home furnishings
    range(5800, 5900): "Consumer Cyclical",      # Eating/drinking places
    range(5900, 6000): "Consumer Cyclical",      # Retail stores

    # Finance, Insurance, Real Estate (60-67)
    range(6000, 6100): "Financial Services",     # Depository institutions (banks)
    range(6100, 6200): "Financial Services",     # Non-depository credit
    range(6200, 6300): "Financial Services",     # Securities/commodities
    range(6300, 6400): "Financial Services",     # Insurance carriers
    range(6400, 6500): "Financial Services",     # Insurance agents
    range(6500, 6600): "Real Estate",            # Real estate
    range(6600, 6700): "Financial Services",     # Combined RE/insurance
    range(6700, 6800): "Financial Services",     # Holding/investment offices

    # Services (70-89) — SPECIFIC ranges first, then broad
    range(7370, 7380): "Technology",             # Computer services/software (MSFT, GOOG, etc)
    range(7300, 7360): "Industrials",            # Business services (non-IT)
    range(7360, 7370): "Industrials",            # Personnel/staffing
    range(7380, 7400): "Industrials",            # Misc business services
    range(7000, 7200): "Consumer Cyclical",      # Hotels/Lodging
    range(7200, 7300): "Consumer Cyclical",      # Personal services
    range(7400, 7500): "Industrials",            # Auto repair
    range(7500, 7600): "Consumer Cyclical",      # Auto rental
    range(7600, 7700): "Industrials",            # Misc repair
    range(7700, 7800): "Consumer Cyclical",      # Motion pictures
    range(7800, 7900): "Communication Services", # Motion picture distribution
    range(7900, 8000): "Consumer Cyclical",      # Amusement/recreation
    range(8000, 8100): "Healthcare",             # Health services
    range(8100, 8200): "Industrials",            # Legal services
    range(8200, 8300): "Consumer Cyclical",      # Educational services
    range(8300, 8400): "Industrials",            # Social services
    range(8400, 8500): "Industrials",            # Engineering/architectural services
    range(8500, 8600): "Industrials",            # Management consulting
    range(8600, 8700): "Industrials",            # Accounting/auditing services
    range(8700, 8800): "Technology",             # Engineering/R&D services
    range(8800, 8900): "Industrials",            # Other professional services
    range(8900, 9000): "Industrials",            # Misc services

    # Public Administration (90-99)
    range(9000, 10000): "Industrials",
}


def _sic_to_sector(sic_code: int) -> str:
    """Map a 4-digit SIC code to a GICS sector."""
    for sic_range, sector in _SIC_TO_SECTOR.items():
        if sic_code in sic_range:
            return sector
    return "Unknown"


# ---------------------------------------------------------------------------
# EDGAR SIC code fetcher
# ---------------------------------------------------------------------------

def _get_ticker_to_cik(use_cache: bool = True) -> Dict[str, str]:
    """Get ticker → CIK mapping from EDGAR."""
    cache_file = _CACHE_DIR / "edgar_cik_map.pkl"

    if use_cache and cache_file.exists():
        data = pickle.load(open(cache_file, "rb"))
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    mapping = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "")
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            mapping[ticker] = cik

    pickle.dump({"data": mapping, "ts": time.time()}, open(cache_file, "wb"))
    return mapping


def fetch_sic_codes(
    tickers: List[str],
    use_cache: bool = True,
    max_tickers: int = 5000,
) -> Dict[str, int]:
    """
    Fetch SIC codes from EDGAR submissions endpoint.

    Rate: 10 req/sec (SEC limit). 5000 tickers = ~8 minutes.
    Cached permanently (SIC codes don't change).
    """
    cache_file = _CACHE_DIR / "edgar_sic_codes.pkl"

    if use_cache and cache_file.exists():
        cached = pickle.load(open(cache_file, "rb"))
        coverage = sum(1 for t in tickers if t in cached) / max(len(tickers), 1)
        if coverage > 0.30:
            print(f"[sectors] SIC codes loaded from cache ({len(cached)} tickers, {coverage:.0%} coverage)")
            return cached

    ticker_to_cik = _get_ticker_to_cik(use_cache=use_cache)

    # Build list of (ticker, CIK) to fetch
    to_fetch = []
    for ticker in tickers[:max_tickers]:
        # Try direct match
        cik = ticker_to_cik.get(ticker)
        # Try base ticker for _OLD
        if not cik and "_" in ticker:
            base = ticker.split("_")[0]
            cik = ticker_to_cik.get(base)
        if cik:
            to_fetch.append((ticker, cik))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_sic(ticker_cik):
        ticker, cik = ticker_cik
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                sic = data.get("sic")
                if sic:
                    return ticker, int(sic)
        except Exception:
            pass
        return ticker, None

    # SEC allows 10 req/sec. Use 8 threads to stay safe.
    print(f"[sectors] Fetching SIC codes from EDGAR for {len(to_fetch)} tickers (8 threads)...")
    sic_codes: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_sic, tc): tc for tc in to_fetch}
        done = 0
        for future in as_completed(futures):
            ticker, sic = future.result()
            done += 1
            if sic is not None:
                sic_codes[ticker] = sic
            if done % 500 == 0:
                print(f"    {done}/{len(to_fetch)} fetched ({len(sic_codes)} with SIC)")

    pickle.dump(sic_codes, open(cache_file, "wb"))
    print(f"[sectors] Got SIC codes for {len(sic_codes)} tickers")
    return sic_codes


# ---------------------------------------------------------------------------
# Master sector mapper
# ---------------------------------------------------------------------------

def get_sectors_from_edgar(
    tickers: List[str],
    use_cache: bool = True,
) -> Dict[str, str]:
    """
    Map tickers to GICS sectors using EDGAR SIC codes.

    Pipeline:
      1. Get ticker → CIK mapping (1 API call)
      2. Fetch SIC code from EDGAR submissions (1 call per ticker, cached)
      3. Map SIC → GICS sector
      4. For _OLD tickers, try base ticker lookup

    Returns dict of ticker → sector name.
    """
    cache_file = _CACHE_DIR / "sector_map_edgar.pkl"

    if use_cache and cache_file.exists():
        cached = pickle.load(open(cache_file, "rb"))
        coverage = sum(1 for t in tickers if t in cached) / max(len(tickers), 1)
        if coverage > 0.70:
            unknown = sum(1 for t in tickers if cached.get(t) == "Unknown")
            print(f"[sectors] EDGAR sector map loaded from cache "
                  f"({len(cached)} tickers, {coverage:.0%} coverage, "
                  f"{unknown} Unknown)")
            return cached

    # Step 1-2: Get SIC codes
    sic_codes = fetch_sic_codes(tickers, use_cache=use_cache)

    # Step 3: Map SIC → GICS sector
    sector_map: Dict[str, str] = {}
    for ticker in tickers:
        sic = sic_codes.get(ticker)
        if sic:
            sector_map[ticker] = _sic_to_sector(sic)
        else:
            # Try base ticker for _OLD
            if "_" in ticker:
                base = ticker.split("_")[0]
                base_sic = sic_codes.get(base)
                if base_sic:
                    sector_map[ticker] = _sic_to_sector(base_sic)
                else:
                    sector_map[ticker] = "Unknown"
            else:
                sector_map[ticker] = "Unknown"

    # Stats
    known = sum(1 for v in sector_map.values() if v != "Unknown")
    total = len(sector_map)
    print(f"[sectors] Mapped {known}/{total} tickers to sectors ({known/total*100:.1f}%)")

    counts = pd.Series([v for v in sector_map.values() if v != "Unknown"]).value_counts()
    if len(counts) > 0:
        print(f"[sectors] Top sectors:\n{counts.head(10).to_string()}")

    pickle.dump(sector_map, open(cache_file, "wb"))
    return sector_map
