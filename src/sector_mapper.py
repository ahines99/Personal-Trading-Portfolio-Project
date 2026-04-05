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

# SEC REQUIRES a descriptive User-Agent with contact email (policy since 2021).
# Requests without this may get 403/429. Rate limit: 10 req/sec.
_EDGAR_HEADERS = {
    "User-Agent": "quant-research-project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Cache key version — bump to invalidate stale/partial caches.
_SIC_CACHE_VERSION = "v2"
_SECTOR_CACHE_VERSION = "v2"

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


# Hardcoded fallback mapping for top liquid US tickers (ETFs + mega-caps that
# may not appear in EDGAR or may have unusual SIC codes). Used as last resort.
_HARDCODED_SECTORS = {
    # Broad-market & sector ETFs (not in EDGAR)
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "VTI": "ETF",
    "VOO": "ETF", "IVV": "ETF", "VEA": "ETF", "VWO": "ETF", "EFA": "ETF",
    "EEM": "ETF", "AGG": "ETF", "BND": "ETF", "TLT": "ETF", "LQD": "ETF",
    "HYG": "ETF", "GLD": "ETF", "SLV": "ETF", "USO": "ETF", "UNG": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLV": "ETF", "XLI": "ETF",
    "XLY": "ETF", "XLP": "ETF", "XLU": "ETF", "XLB": "ETF", "XLRE": "ETF",
    "XLC": "ETF", "VNQ": "ETF", "XBI": "ETF", "SMH": "ETF", "SOXX": "ETF",
    "ARKK": "ETF", "VIG": "ETF", "SCHD": "ETF", "VYM": "ETF",
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
    max_tickers: int = 10000,
) -> Dict[str, int]:
    """
    Fetch SIC codes from EDGAR submissions endpoint.

    Rate: 10 req/sec (SEC limit). 5000 tickers = ~10 minutes with 4 threads.
    Cached permanently (SIC codes don't change).

    Robustness fixes (2026-04-05):
      - Per-thread requests.Session with keep-alive (avoids connection pool exhaustion)
      - Retry on network errors / 429 with backoff
      - Per-request throttle via threading.Lock to stay under 10 req/sec
      - Diagnostic counters logged at end
      - Merges with existing cache instead of overwriting
    """
    cache_file = _CACHE_DIR / f"edgar_sic_codes_{_SIC_CACHE_VERSION}.pkl"

    # Load existing cache to merge with
    existing: Dict[str, int] = {}
    if cache_file.exists():
        try:
            existing = pickle.load(open(cache_file, "rb"))
        except Exception:
            existing = {}

    if use_cache and existing:
        coverage = sum(1 for t in tickers if t in existing) / max(len(tickers), 1)
        if coverage > 0.60:
            print(f"[sectors] SIC codes loaded from cache ({len(existing)} tickers, {coverage:.0%} coverage)")
            return existing

    ticker_to_cik = _get_ticker_to_cik(use_cache=use_cache)

    # Build list of (ticker, CIK) to fetch — skip ones already cached
    to_fetch = []
    skipped_no_cik = 0
    for ticker in tickers[:max_tickers]:
        if ticker in existing:
            continue
        cik = ticker_to_cik.get(ticker)
        if not cik and "_" in ticker:
            base = ticker.split("_")[0]
            cik = ticker_to_cik.get(base)
        if not cik and "-" in ticker:
            base = ticker.split("-")[0]
            cik = ticker_to_cik.get(base)
        if cik:
            to_fetch.append((ticker, cik))
        else:
            skipped_no_cik += 1

    if not to_fetch:
        print(f"[sectors] All tickers covered by cache; {skipped_no_cik} had no CIK match")
        return existing

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Thread-local sessions for keep-alive
    _thread_local = threading.local()
    _rate_lock = threading.Lock()
    _last_request_time = [0.0]
    _min_interval = 0.11  # ~9 req/sec global max, safely below 10/sec SEC limit

    # Diagnostic counters
    stats = {
        "attempted": 0, "http_200": 0, "http_403": 0, "http_429": 0,
        "http_other": 0, "parsed_sic": 0, "no_sic_key": 0, "exc": 0,
    }
    stats_lock = threading.Lock()
    first_samples = []

    def _get_session():
        s = getattr(_thread_local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(_EDGAR_HEADERS)
            _thread_local.session = s
        return s

    def _throttle():
        with _rate_lock:
            now = time.time()
            wait = _min_interval - (now - _last_request_time[0])
            if wait > 0:
                time.sleep(wait)
            _last_request_time[0] = time.time()

    def _fetch_sic(ticker_cik):
        ticker, cik = ticker_cik
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        session = _get_session()
        for attempt in range(3):
            _throttle()
            try:
                resp = session.get(url, timeout=15)
                with stats_lock:
                    stats["attempted"] += 1
                    if resp.status_code == 200:
                        stats["http_200"] += 1
                    elif resp.status_code == 403:
                        stats["http_403"] += 1
                    elif resp.status_code == 429:
                        stats["http_429"] += 1
                    else:
                        stats["http_other"] += 1
                    if len(first_samples) < 5:
                        first_samples.append(
                            (ticker, cik, resp.status_code,
                             (resp.text[:120] if resp.status_code != 200 else "OK"))
                        )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        with stats_lock:
                            stats["exc"] += 1
                        return ticker, None
                    sic = data.get("sic")
                    if sic:
                        with stats_lock:
                            stats["parsed_sic"] += 1
                        try:
                            return ticker, int(sic)
                        except (ValueError, TypeError):
                            return ticker, None
                    else:
                        with stats_lock:
                            stats["no_sic_key"] += 1
                        return ticker, None
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                elif resp.status_code == 404:
                    return ticker, None
                else:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            except Exception:
                with stats_lock:
                    stats["exc"] += 1
                time.sleep(0.5 * (attempt + 1))
                continue
        return ticker, None

    # Use 4 threads + throttle to stay safely under SEC's 10 req/sec limit
    n_workers = 4
    print(f"[sectors] Fetching SIC codes from EDGAR for {len(to_fetch)} tickers ({n_workers} threads, ~9 req/sec)...")
    if skipped_no_cik:
        print(f"[sectors] (skipped {skipped_no_cik} tickers with no CIK mapping)")

    sic_codes: Dict[str, int] = dict(existing)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_sic, tc): tc for tc in to_fetch}
        done = 0
        for future in as_completed(futures):
            ticker, sic = future.result()
            done += 1
            if sic is not None:
                sic_codes[ticker] = sic
            if done % 500 == 0:
                print(f"    {done}/{len(to_fetch)} fetched ({len(sic_codes)} total with SIC)")
                # Persist partial progress so a crash doesn't lose work
                try:
                    pickle.dump(sic_codes, open(cache_file, "wb"))
                except Exception:
                    pass

    pickle.dump(sic_codes, open(cache_file, "wb"))

    # Diagnostic summary
    print(f"[sectors] Fetch stats: attempted={stats['attempted']}, "
          f"200={stats['http_200']}, 403={stats['http_403']}, 429={stats['http_429']}, "
          f"other={stats['http_other']}, exc={stats['exc']}, "
          f"sic_parsed={stats['parsed_sic']}, no_sic_key={stats['no_sic_key']}")
    if first_samples:
        print(f"[sectors] First 5 response samples:")
        for s in first_samples:
            print(f"    {s[0]} CIK={s[1]} status={s[2]} body={s[3]}")
    print(f"[sectors] Got SIC codes for {len(sic_codes)} tickers total")
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
    cache_file = _CACHE_DIR / f"sector_map_edgar_{_SECTOR_CACHE_VERSION}.pkl"

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
            continue
        # Try base ticker for _OLD / hyphenated share classes
        base_sic = None
        if "_" in ticker:
            base_sic = sic_codes.get(ticker.split("_")[0])
        if not base_sic and "-" in ticker:
            base_sic = sic_codes.get(ticker.split("-")[0])
        if base_sic:
            sector_map[ticker] = _sic_to_sector(base_sic)
            continue
        # Hardcoded fallback for ETFs / mega-caps
        if ticker in _HARDCODED_SECTORS:
            sector_map[ticker] = _HARDCODED_SECTORS[ticker]
            continue
        sector_map[ticker] = "Unknown"

    # Stats
    known = sum(1 for v in sector_map.values() if v != "Unknown")
    total = len(sector_map)
    print(f"[sectors] Mapped {known}/{total} tickers to sectors ({known/total*100:.1f}%)")
    print(f"[sectors] Final EDGAR coverage: {known}/{total} tickers ({known/total*100:.1f}%)")

    counts = pd.Series([v for v in sector_map.values() if v != "Unknown"]).value_counts()
    if len(counts) > 0:
        print(f"[sectors] Top sectors:\n{counts.head(10).to_string()}")

    pickle.dump(sector_map, open(cache_file, "wb"))
    return sector_map
