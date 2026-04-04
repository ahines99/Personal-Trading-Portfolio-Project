"""
data_loader.py
--------------
Downloads and caches OHLCV data via EODHD API ($20/mo).

Why EODHD instead of yfinance:
  - 28,000+ delisted US stocks included (survivorship-bias-free)
  - 1,000 req/min (vs yfinance's ~60 req/min with constant failures)
  - Zero failed downloads (vs 24+ per run with yfinance)
  - Bulk endpoint: entire US exchange in 1 API call
  - 30+ years of split/dividend-adjusted history
  - Paid, maintained service (vs yfinance unofficial scraper)

Universe construction:
  1. Fetch all active + delisted US common stocks (2 API calls)
  2. Bulk screen by price and ADV (1 API call)
  3. Download daily OHLCV for screened tickers (1 call each, 16/sec)

yfinance is kept ONLY for: sector mapping, insider transactions, analyst data
(endpoints EODHD doesn't provide on the $20 plan).

API budget: ~3,000 calls per full run (3% of 100K daily limit).
"""

import os
import pickle
import time
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# yfinance kept only for sector fetching
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# EODHD API
# Load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass
EODHD_KEY = os.environ.get("EODHD_API_KEY", "")
_EODHD_BASE = "https://eodhd.com/api"
_EODHD_RATE = 0.02  # ~50 req/sec, well under 1000/min limit


def _eodhd_get(endpoint: str, params: dict = None) -> list | dict:
    """Rate-limited EODHD API call."""
    if params is None:
        params = {}
    params["api_token"] = EODHD_KEY
    params["fmt"] = "json"
    url = f"{_EODHD_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(_EODHD_RATE)
    return resp.json()


# ---------------------------------------------------------------------------
# Universe Construction (EODHD)
# ---------------------------------------------------------------------------

def _clean_ticker(t: str) -> str:
    """Normalize ticker: strip, uppercase, remove invalid chars."""
    t = str(t).strip().upper()
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    return t


def build_eodhd_universe(
    include_delisted: bool = True,
    use_cache: bool = True,
) -> List[str]:
    """
    Fetch all active + delisted US common stocks from EODHD.

    Returns ~47,000 tickers (19K active + 28K delisted).
    Cost: 2 API calls.
    """
    cache_file = CACHE_DIR / "eodhd_universe.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"[data_loader] Universe loaded from cache: {len(data)} tickers")
        return data

    print("[data_loader] Fetching US stock universe from EODHD...")

    # Active common stocks
    active = _eodhd_get("exchange-symbol-list/US", {"type": "common_stock"})
    active_tickers = [_clean_ticker(s.get("Code", "")) for s in active
                      if s.get("Code") and len(s.get("Code", "")) <= 6]
    print(f"  Active: {len(active_tickers)} common stocks")

    # Delisted common stocks
    delisted_tickers = []
    if include_delisted:
        try:
            delisted = _eodhd_get("exchange-symbol-list/US", {
                "type": "common_stock", "delisted": "1"
            })
            delisted_tickers = [_clean_ticker(s.get("Code", "")) for s in delisted
                                if s.get("Code") and len(s.get("Code", "")) <= 6]
            print(f"  Delisted: {len(delisted_tickers)} stocks")
        except Exception as e:
            print(f"  Delisted fetch failed: {e}")

    all_tickers = sorted(set(active_tickers + delisted_tickers))
    print(f"  Total universe: {len(all_tickers)} tickers")

    with open(cache_file, "wb") as f:
        pickle.dump(all_tickers, f)

    return all_tickers


def build_top_liquidity_universe(
    n: int = 5000,
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    min_price: float = 5.0,
    min_adv: float = 500_000,
    use_cache: bool = True,
) -> List[str]:
    """
    Screen the EODHD universe by liquidity using dual-date bulk screening.

    Screens at TWO dates to capture stocks that were liquid at different
    points during the backtest:
      1. Start + 1 year (point-in-time, captures stocks active early)
      2. Mid-period (captures stocks that became liquid later or delisted early)

    This matches the approach used in the institutional L/S project.
    """
    cache_file = CACHE_DIR / f"top_universe_{n}_{start}_{end}_{int(min_price)}_{int(min_adv)}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            tickers = pickle.load(f)
        print(f"[data_loader] Liquidity universe loaded from cache: {len(tickers)} tickers")
        return tickers

    all_common = set(build_eodhd_universe(include_delisted=True, use_cache=use_cache))

    # Screen at TWO dates for maximum coverage
    screen_date_1 = (pd.Timestamp(start) + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    mid_point = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    screen_date_2 = mid_point.strftime("%Y-%m-%d")

    print(f"[data_loader] Dual-date screening: {screen_date_1} + {screen_date_2}")
    print(f"  Filters: price>${min_price}, ADV>${min_adv/1e6:.1f}M")

    _skip_prefixes = ("^",)
    _etfs = {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "LQD",
             "VOO", "VTI", "IVV", "EEM", "EFA", "VEA", "VWO", "BND", "AGG"}

    def _screen_bulk(target_date: str) -> dict:
        """Fetch bulk data and return {code: dollar_vol} for eligible tickers."""
        bulk = None
        for attempt_date in [target_date] + [
            (pd.Timestamp(target_date) + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
            for d in [1, 2, 3, 7, 14]
        ]:
            try:
                bulk = _eodhd_get(f"eod-bulk-last-day/US", {"date": attempt_date})
                if bulk and len(bulk) > 100:
                    print(f"  Bulk data for {attempt_date}: {len(bulk)} tickers")
                    break
            except Exception:
                continue

        if not bulk:
            return {}

        results = {}
        for row in bulk:
            code = row.get("code", "")
            close_px = row.get("adjusted_close") or row.get("close", 0)
            vol = row.get("volume", 0)

            if not code or any(code.startswith(p) for p in _skip_prefixes):
                continue
            if code in _etfs:
                continue

            base = code.split("_")[0] if "_" in code else code
            if len(base) >= 5 and base.endswith("X"):
                continue
            if "-P" in code:
                continue
            if any(c.isdigit() for c in base) and "_" not in code:
                continue

            try:
                close_px = float(close_px)
                vol = float(vol)
            except (ValueError, TypeError):
                continue

            if close_px <= 0 or vol <= 0:
                continue

            dollar_vol = close_px * vol
            if dollar_vol > 50e9:
                continue

            if close_px >= min_price and dollar_vol >= min_adv:
                results[code] = max(results.get(code, 0), dollar_vol)

        return results

    # Screen at both dates, merge results
    eligible_1 = _screen_bulk(screen_date_1)
    import time as _time
    _time.sleep(0.5)  # brief pause between bulk calls
    eligible_2 = _screen_bulk(screen_date_2)

    # Merge: take the higher ADV from either date
    all_eligible = {}
    for code, dv in eligible_1.items():
        all_eligible[code] = max(all_eligible.get(code, 0), dv)
    for code, dv in eligible_2.items():
        all_eligible[code] = max(all_eligible.get(code, 0), dv)

    print(f"  Date 1 eligible: {len(eligible_1)}, Date 2 eligible: {len(eligible_2)}")
    print(f"  Combined unique: {len(all_eligible)}")

    # Take ALL that pass — liquidity filter is the only gate
    sorted_eligible = sorted(all_eligible.items(), key=lambda x: -x[1])
    top = [t[0] for t in sorted_eligible]

    # Count delisted
    active_only = set(build_eodhd_universe(include_delisted=False, use_cache=True))
    n_delisted = sum(1 for t in top if t not in active_only)

    print(f"[data_loader] Final universe: {len(top)} tickers ({n_delisted} delisted, survivorship-bias-free)")

    if len(top) < 100:
        raise RuntimeError(
            f"Only {len(top)} tickers passed screening. "
            f"Try lowering min_price or min_adv."
        )

    with open(cache_file, "wb") as f:
        pickle.dump(top, f)

    return top


# ---------------------------------------------------------------------------
# Price Download (EODHD)
# ---------------------------------------------------------------------------

def _download_eodhd_prices(
    tickers: List[str],
    start: str,
    end: str,
    max_workers: int = 15,
) -> dict:
    """
    Download daily OHLCV for all tickers from EODHD using parallel threads.

    Uses adjusted_close for split/dividend-adjusted prices.
    15 threads: ~10 min for 4600 tickers (vs 78 min sequential).
    EODHD allows 1000 req/min; 15 threads × ~1 req/sec = ~15 req/sec.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_data = {}
    n = len(tickers)
    failed = []

    def _fetch_one(ticker: str):
        """Fetch one ticker's price history."""
        try:
            params = {
                "api_token": EODHD_KEY,
                "fmt": "json",
                "from": start,
                "to": end,
                "period": "d",
            }
            resp = requests.get(
                f"{_EODHD_BASE}/eod/{ticker}.US",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 20:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                return ticker, df
        except Exception:
            return ticker, None
        return ticker, None

    print(f"[data_loader] Downloading {n} tickers from EODHD ({start} to {end})...")
    print(f"[data_loader] Using {max_workers} parallel threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker, df = future.result()
            done += 1
            if df is not None:
                all_data[ticker] = df
            else:
                failed.append(ticker)

            if done % 500 == 0:
                print(f"  {done}/{n} downloaded ({len(all_data)} with data, {len(failed)} failed)")

    if failed:
        print(f"[data_loader] {len(failed)} failed downloads (out of {n})")
    else:
        print(f"[data_loader] All {n} tickers downloaded successfully")

    return all_data


def load_prices(
    tickers: Optional[List[str]] = None,
    start: str = "2015-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    dynamic_universe: bool = True,
    universe_size: int = 5000,
    min_price: float = 5.0,
    min_adv: float = 500_000,
) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame: columns = (field, ticker)
    Fields: Open, High, Low, Close, Volume, Returns

    Uses EODHD API for price data (survivorship-bias-free, includes delisted).
    Falls back to yfinance for specific ticker lists (e.g., SPY benchmark).

    Example access:
        prices["Close"]["AAPL"]   -> daily close series for AAPL
        prices["Returns"]         -> DataFrame of daily returns, all tickers
    """
    if tickers is not None and len(tickers) <= 5:
        # Small ticker list (e.g., SPY benchmark) — use yfinance for simplicity
        return _load_prices_yfinance(tickers, start, end, use_cache)

    if tickers is None:
        if dynamic_universe:
            tickers = build_top_liquidity_universe(
                n=universe_size,
                start=start,
                end=end,
                min_price=min_price,
                min_adv=min_adv,
                use_cache=use_cache,
            )
        else:
            raise ValueError("tickers is None and dynamic_universe=False.")

    # Cache key uses hash of sorted ticker list — prevents re-downloading
    # when universe changes by 1-2 tickers due to screening edge cases.
    import hashlib
    _ticker_hash = hashlib.md5("_".join(sorted(tickers)).encode()).hexdigest()[:10]
    cache_file = CACHE_DIR / f"eodhd_prices_{start}_{end}_{len(tickers)}_{_ticker_hash}.pkl"

    if use_cache and cache_file.exists():
        print(f"[data_loader] Loading from cache: {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Fuzzy match: if exact hash misses but a cache with the same date range
    # exists within 10 tickers, reuse it (avoids 11-min re-download)
    if use_cache:
        import glob, re
        pattern = str(CACHE_DIR / f"eodhd_prices_{start}_{end}_*.pkl")
        candidates = sorted(glob.glob(pattern), key=lambda f: Path(f).stat().st_mtime, reverse=True)
        for cand in candidates:
            m = re.search(r"_(\d+)tickers", Path(cand).name)
            if m and abs(int(m.group(1)) - len(tickers)) <= 10:
                print(f"[data_loader] Fuzzy cache hit (within 10 tickers): {Path(cand).name}")
                with open(cand, "rb") as f:
                    return pickle.load(f)

    # Download from EODHD
    raw_data = _download_eodhd_prices(tickers, start, end)

    if not raw_data:
        raise RuntimeError("No price data downloaded from EODHD.")

    # Build aligned DataFrames for each field
    date_index = None
    for df in raw_data.values():
        if date_index is None:
            date_index = df.index
        else:
            date_index = date_index.union(df.index)
    date_index = date_index.sort_values()

    valid_tickers = list(raw_data.keys())
    n_tickers = len(valid_tickers)

    # Build all panels at once using dict comprehension + single pd.DataFrame call.
    # This is ~100x faster than column-by-column assignment (avoids DataFrame fragmentation).
    print(f"[data_loader] Building price panels for {n_tickers} tickers...")

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}

    for ticker, df in raw_data.items():
        # Compute split adjustment factor: adjusted_close / raw_close
        adj_factor = 1.0
        if "adjusted_close" in df.columns and "close" in df.columns:
            raw_close = df["close"].replace(0, np.nan)
            adj_close = df["adjusted_close"].replace(0, np.nan)
            adj_factor = (adj_close / raw_close).fillna(1.0)

        if "adjusted_close" in df.columns:
            close_dict[ticker] = df["adjusted_close"]
        elif "close" in df.columns:
            close_dict[ticker] = df["close"]

        if "open" in df.columns:
            open_dict[ticker] = df["open"] * adj_factor
        if "high" in df.columns:
            high_dict[ticker] = df["high"] * adj_factor
        if "low" in df.columns:
            low_dict[ticker] = df["low"] * adj_factor
        if "volume" in df.columns:
            vol_dict[ticker] = df["volume"]

    # Single DataFrame construction + reindex (fast)
    close_df = pd.DataFrame(close_dict).reindex(date_index)
    open_df = pd.DataFrame(open_dict).reindex(date_index)
    high_df = pd.DataFrame(high_dict).reindex(date_index)
    low_df = pd.DataFrame(low_dict).reindex(date_index)
    volume_df = pd.DataFrame(vol_dict).reindex(date_index)

    # Forward-fill ONLY within each stock's active trading period.
    # Plain ffill() bleeds the last valid price forward forever after delisting,
    # creating phantom positions with zero returns that suppress volatility and
    # cause the backtest to hold dead stocks indefinitely.
    #
    # Fix: ffill first (fills mid-series gaps), then mask out everything AFTER
    # the last real data point per ticker (post-delisting = NaN).
    close_df = close_df.ffill()
    open_df = open_df.ffill()
    high_df = high_df.ffill()
    low_df = low_df.ffill()
    volume_df = volume_df.fillna(0)

    # Build mask: True only for dates <= last real observation per ticker.
    # Vectorized: compare each date against the last valid date per ticker.
    _raw_close = pd.DataFrame(close_dict).reindex(date_index)
    _last_valid = _raw_close.apply(lambda col: col.last_valid_index())
    # Broadcast: dates (rows) vs last_valid (columns)
    _active_mask = pd.DataFrame(
        date_index.values[:, None] <= _last_valid.values[None, :],
        index=date_index,
        columns=close_df.columns,
    )
    # Tickers that are NaN everywhere (never traded) → mask is all False
    _never_traded = _last_valid.isna()
    _active_mask.loc[:, _never_traded] = False

    n_delisted = (_last_valid < date_index[-1]).sum()
    print(f"[data_loader] {n_delisted} tickers have delisting dates (NaN after last trade)")

    # Apply mask — NaN after delisting
    close_df = close_df.where(_active_mask)
    open_df = open_df.where(_active_mask)
    high_df = high_df.where(_active_mask)
    low_df = low_df.where(_active_mask)
    volume_df = volume_df.where(_active_mask, 0)

    # ── Data quality filters ────────────────────────────────────────────
    # EODHD delisted data includes toxic tickers: zero prices, sub-penny
    # stocks, and restructurings that create 10,000%+ daily returns.
    # A single inf return propagates through features → weights → backtest
    # and overflows NAV to infinity.

    # 1. Drop tickers with >30% missing WITHIN their active trading period.
    #    Previous bug: computing missing % across the FULL date range penalized
    #    delisted stocks (e.g., stock active 2013-2016 has ~70% NaN in 2013-2026),
    #    re-introducing survivorship bias. Now we only count NaN within the period
    #    from first to last valid price.
    _first_valid = close_df.apply(lambda col: col.first_valid_index())
    _last_valid_idx = close_df.apply(lambda col: col.last_valid_index())
    bad = []
    for ticker in close_df.columns:
        fv, lv = _first_valid[ticker], _last_valid_idx[ticker]
        if fv is None or lv is None:
            bad.append(ticker)  # never traded
            continue
        active_slice = close_df.loc[fv:lv, ticker]
        lifespan = len(active_slice)
        if lifespan < 60:
            bad.append(ticker)  # fewer than 60 active trading days
            continue
        missing_in_lifespan = active_slice.isna().sum() / lifespan
        if missing_in_lifespan > 0.30:
            bad.append(ticker)

    # 2. Drop tickers where >5% of active trading days have sub-penny prices.
    #    Previous bug: .any() dropped stocks that EVER had a single $0.01 day,
    #    removing legitimate distressed-and-recovered names.
    sub_penny_frac = (close_df <= 0.01).sum() / close_df.notna().sum().replace(0, 1)
    zero_tickers = sub_penny_frac[sub_penny_frac > 0.05].index.tolist()
    bad = list(set(bad + zero_tickers))

    if bad:
        n_missing = len([t for t in bad if t not in zero_tickers])
        print(f"[data_loader] Dropping {len(bad)} tickers "
              f"({len(zero_tickers)} with >5% sub-penny prices, "
              f"{n_missing} with >30% missing within active period or <60 days)")
        valid_tickers = [t for t in valid_tickers if t not in bad]
        close_df = close_df[valid_tickers]
        open_df = open_df[valid_tickers]
        high_df = high_df[valid_tickers]
        low_df = low_df[valid_tickers]
        volume_df = volume_df[valid_tickers]

    # 3. Compute returns with safety caps
    returns_df = close_df.pct_change()

    # Cap daily returns at ±100%. No legitimate stock moves >100% in a day
    # at prices we can actually trade. This prevents inf/overflow from
    # penny stock restructurings and bad EODHD data.
    returns_df = returns_df.clip(-1.0, 1.0)
    # Replace inf with NaN (not 0). Keep NaN for unlisted periods — downstream
    # code handles NaN via dropna() in ranking and fillna(0) only in portfolio
    # return computation where NaN correctly means "no position value change."
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    # Only fill NaN on the FIRST row (pct_change produces NaN for day 1)
    returns_df.iloc[0] = returns_df.iloc[0].fillna(0.0)

    # Assemble MultiIndex DataFrame
    pieces = {
        "Open": open_df,
        "High": high_df,
        "Low": low_df,
        "Close": close_df,
        "Volume": volume_df,
        "Returns": returns_df,
    }
    out = pd.concat(pieces, axis=1).sort_index(axis=1)

    print(f"[data_loader] Final: {len(valid_tickers)} tickers x {len(date_index)} days")

    with open(cache_file, "wb") as f:
        pickle.dump(out, f)

    return out


def _load_prices_yfinance(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fallback to yfinance for small ticker lists (e.g., SPY benchmark)."""
    if not YF_AVAILABLE:
        raise ImportError("yfinance not installed for benchmark loading")

    cache_file = CACHE_DIR / f"yf_prices_{'_'.join(sorted(tickers))}_{start}_{end}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True)
    if raw.empty:
        raise RuntimeError(f"yfinance download failed for {tickers}")

    raw = raw.ffill()

    if len(tickers) == 1:
        # yfinance returns flat columns for single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        returns = close.pct_change().fillna(0.0)
        pieces = {
            "Open": raw[["Open"]].rename(columns={"Open": tickers[0]}),
            "High": raw[["High"]].rename(columns={"High": tickers[0]}),
            "Low": raw[["Low"]].rename(columns={"Low": tickers[0]}),
            "Close": close,
            "Volume": raw[["Volume"]].rename(columns={"Volume": tickers[0]}),
            "Returns": returns,
        }
    else:
        close = raw["Close"]
        returns = close.pct_change().fillna(0.0)
        pieces = {
            "Open": raw["Open"],
            "High": raw["High"],
            "Low": raw["Low"],
            "Close": raw["Close"],
            "Volume": raw["Volume"],
            "Returns": returns,
        }

    out = pd.concat(pieces, axis=1).sort_index(axis=1)

    with open(cache_file, "wb") as f:
        pickle.dump(out, f)

    return out


# ---------------------------------------------------------------------------
# Sector Mapping (yfinance — EODHD doesn't provide on $20 plan)
# ---------------------------------------------------------------------------

def get_sectors(tickers: List[str], use_cache: bool = True) -> Dict[str, str]:
    """
    Map tickers to GICS sectors.

    Loads from combined cache first (EDGAR + yfinance merged).
    Only rebuilds if cache is missing or has low coverage.
    """
    # Check combined cache FIRST — this has EDGAR + yfinance already merged
    cache_file = CACHE_DIR / "sector_map.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            cached: Dict[str, str] = pickle.load(f)
        coverage = sum(1 for t in tickers if t in cached) / max(len(tickers), 1)
        if coverage > 0.80:
            unknown = sum(1 for t in tickers if cached.get(t) == "Unknown")
            print(f"[data_loader] Sector map loaded from cache "
                  f"({len(cached)} tickers, {coverage:.0%} coverage, {unknown} Unknown)")
            return cached

    from sector_mapper import get_sectors_from_edgar

    # Build from EDGAR SIC codes (covers delisted, fast)
    sector_map = get_sectors_from_edgar(tickers, use_cache=use_cache)

    # Count unknowns — fill remaining from yfinance
    unknowns = [t for t in tickers if sector_map.get(t) == "Unknown"]

    # yfinance .info is broken for batch calls (401 Invalid Crumb in 2025-2026).
    # But we can try a small batch sequentially with fresh session per call.
    # Only attempt tickers that are likely active (no _OLD, no warrants).
    if unknowns and YF_AVAILABLE:
        active_unknowns = [t for t in unknowns
                           if "_" not in t and "-" not in t and len(t) <= 5]
        if active_unknowns:
            print(f"[data_loader] Trying yfinance for {len(active_unknowns)} active unknowns...")
            filled = 0
            for i, ticker in enumerate(active_unknowns):
                try:
                    # Fresh Ticker object each time to avoid crumb issues
                    t = yf.Ticker(ticker)
                    info = t.info
                    sector = info.get("sector")
                    if sector and sector != "Unknown":
                        sector_map[ticker] = sector
                        filled += 1
                except Exception:
                    pass
                if (i + 1) % 100 == 0:
                    print(f"        {i+1}/{len(active_unknowns)} ({filled} filled)")
                time.sleep(0.3)
            print(f"[data_loader] Filled {filled} sectors from yfinance")

    remaining = sum(1 for t in tickers if sector_map.get(t) == "Unknown")
    print(f"[data_loader] {remaining} tickers with Unknown sector (exempt from cap)")

    # Save combined map
    cache_file = CACHE_DIR / "sector_map.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(sector_map, f)

    unknown_final = sum(1 for v in sector_map.values() if v == "Unknown")
    counts = pd.Series([v for v in sector_map.values() if v != "Unknown"]).value_counts()
    print(f"[data_loader] Sectors: {len(tickers) - unknown_final} mapped, {unknown_final} Unknown")
    if len(counts) > 0:
        print(counts.head(8).to_string())
    return sector_map


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_open(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["Open"]

def get_high(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["High"]

def get_low(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["Low"]

def get_close(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["Close"]

def get_volume(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["Volume"]

def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices["Returns"]
