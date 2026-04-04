"""
fetch_finnhub.py
----------------
Standalone overnight job to fetch Finnhub data for ALL tickers.

Run this once overnight, then the main pipeline uses the cached results.

Usage:
    python fetch_finnhub.py              # fetch all tickers
    python fetch_finnhub.py --max 1000   # fetch top 1000 only
    python fetch_finnhub.py --resume     # resume from where it left off

Rate limit: 60 calls/min (Finnhub free tier)
At 4 endpoints per ticker:
    500 tickers  = ~35 min
    1000 tickers = ~70 min
    2000 tickers = ~140 min (2.3 hours)
    4600 tickers = ~307 min (5.1 hours)

Schedule: run once per month (data doesn't change faster than that)
    Windows: schtasks /create /tn "FinnhubFetch" /tr "python fetch_finnhub.py" /sc monthly /d 1 /st 02:00
    Linux:   crontab: 0 2 1 * * cd /path/to/project && python fetch_finnhub.py
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
_BASE = "https://finnhub.io/api/v1"
_CACHE_DIR = Path(__file__).parent / "data" / "cache" / "api"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_FILE = _CACHE_DIR / "finnhub_all_data.pkl"
_PROGRESS_FILE = _CACHE_DIR / "finnhub_progress.json"

# Rate limiting: 60 calls/min = 1 call per second
_MIN_INTERVAL = 1.05  # slightly over 1 sec to be safe
_last_call = 0.0


def _finnhub_get(endpoint: str, params: dict = None) -> dict:
    """Rate-limited Finnhub API call with retry."""
    global _last_call
    if params is None:
        params = {}
    params["token"] = FINNHUB_KEY

    # Enforce rate limit
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    url = f"{_BASE}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            _last_call = time.time()

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"    Timeout on {endpoint}, retrying...")
            time.sleep(5)
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                return {}

    return {}


def fetch_ticker_data(ticker: str) -> dict:
    """Fetch all 4 endpoints for one ticker."""
    data = {}

    # 1. Recommendation trends
    rec = _finnhub_get("stock/recommendation", {"symbol": ticker})
    if rec and isinstance(rec, list) and len(rec) > 0:
        data["recommendations"] = rec

    # 2. Price target
    pt = _finnhub_get("stock/price-target", {"symbol": ticker})
    if pt and isinstance(pt, dict) and pt.get("targetMean"):
        data["price_target"] = pt

    # 3. EPS surprises
    eps = _finnhub_get("stock/earnings", {"symbol": ticker, "limit": 4})
    if eps and isinstance(eps, list) and len(eps) > 0:
        data["eps_surprises"] = eps

    # 4. Company news (last 30 days)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    month_ago = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    news = _finnhub_get("company-news", {"symbol": ticker, "from": month_ago, "to": today})
    if news and isinstance(news, list):
        data["news_count"] = len(news)
        pos_words = {"beat", "surge", "rally", "upgrade", "record", "strong", "growth", "profit"}
        neg_words = {"miss", "drop", "fall", "downgrade", "loss", "weak", "decline", "cut"}
        pos = sum(1 for n in news if any(w in n.get("headline", "").lower() for w in pos_words))
        neg = sum(1 for n in news if any(w in n.get("headline", "").lower() for w in neg_words))
        data["news_sentiment"] = (pos - neg) / max(len(news), 1)

    return data


def load_universe() -> list:
    """Load the screened ticker universe."""
    cache_files = list(Path(__file__).parent.glob("data/cache/top_universe_*.pkl"))
    if cache_files:
        with open(cache_files[0], "rb") as f:
            return pickle.load(f)

    # Fallback: load from EODHD universe
    eodhd_file = Path(__file__).parent / "data" / "cache" / "eodhd_universe.pkl"
    if eodhd_file.exists():
        with open(eodhd_file, "rb") as f:
            return pickle.load(f)[:3000]

    raise RuntimeError("No universe cache found. Run the main pipeline first.")


def main():
    parser = argparse.ArgumentParser(description="Overnight Finnhub data fetch")
    parser.add_argument("--max", type=int, default=0, help="Max tickers (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress")
    args = parser.parse_args()

    # Load universe
    tickers = load_universe()
    if args.max > 0:
        tickers = tickers[:args.max]
    print(f"Finnhub overnight fetch: {len(tickers)} tickers, 4 endpoints each")
    print(f"Estimated time: ~{len(tickers) * 4 / 57:.0f} minutes")
    print()

    # Load existing data + progress
    results = {}
    completed = set()

    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing cache: {len(results)} tickers")

    if args.resume and _PROGRESS_FILE.exists():
        with open(_PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        completed = set(progress.get("completed", []))
        print(f"Resuming: {len(completed)} already done")

    # Filter out already-completed tickers
    remaining = [t for t in tickers if t not in completed]
    print(f"Remaining: {len(remaining)} tickers")
    print(f"Started at: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    print("-" * 50)

    start_time = time.time()
    new_data = 0

    for i, ticker in enumerate(remaining):
        data = fetch_ticker_data(ticker)

        if data:
            results[ticker] = data
            new_data += 1

        completed.add(ticker)

        # Save progress every 50 tickers
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            remaining_min = (len(remaining) - i - 1) / max(rate, 0.1)

            print(f"  {i+1}/{len(remaining)} | "
                  f"{new_data} with data | "
                  f"{rate:.0f} tickers/min | "
                  f"~{remaining_min:.0f} min remaining")

            # Save cache
            with open(_CACHE_FILE, "wb") as f:
                pickle.dump(results, f)
            with open(_PROGRESS_FILE, "w") as f:
                json.dump({"completed": list(completed)}, f)

    # Final save
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    # Clean up progress file
    if _PROGRESS_FILE.exists():
        _PROGRESS_FILE.unlink()

    elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print(f"DONE: {len(results)} tickers with data (out of {len(tickers)})")
    print(f"New this run: {new_data}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cache saved to: {_CACHE_FILE}")
    print()
    print("The main pipeline will automatically use this cached data.")


if __name__ == "__main__":
    main()
