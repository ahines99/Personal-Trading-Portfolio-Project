"""
api_data.py
-----------
Finnhub + EODHD API data loaders with rate limiting and caching.

EODHD ($20/mo All-World EOD plan — no fundamentals):
  - Bulk EOD prices for entire exchange (1 API call!)
  - Historical dividends (for dividend yield feature)
  - Historical splits
  - Exchange ticker list (51K+ symbols)
  NOTE: Fundamentals require All-In-One plan ($40+/mo). Not available.

Finnhub (free tier, 60 calls/min):
  - Recommendation trends (time-series, 4 months history)
  - Price targets (consensus)
  - EPS surprises (4 quarters)
  - Company news (1 year)

Rate limits:
  EODHD:  ~5 calls/sec (self-imposed pacing)
  Finnhub: 60 calls/min = 1 call/sec. We pace at 1.05 sec between calls.

All data cached to disk. For ~3000 tickers:
  EODHD dividends: ~3000 calls / 5 per sec = ~10 min (first time)
  Finnhub: 500 tickers * 4 endpoints / 57 per min = ~35 min (first time)
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_CACHE = _ROOT / "data" / "cache" / "api"
_CACHE.mkdir(parents=True, exist_ok=True)

# Load API keys from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed, rely on env vars

FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
EODHD_KEY = os.environ.get("EODHD_API_KEY", "")


def _cache_path(name: str) -> Path:
    return _CACHE / f"{name}.pkl"


def _save(obj, name: str):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)


def _load(name: str):
    p = _cache_path(name)
    if p.exists():
        return pickle.load(open(p, "rb"))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# EODHD API
# ═══════════════════════════════════════════════════════════════════════════

_EODHD_BASE = "https://eodhd.com/api"


def _eodhd_get(endpoint: str, params: dict = None) -> dict:
    """Rate-limited EODHD API call."""
    if params is None:
        params = {}
    params["api_token"] = EODHD_KEY
    params["fmt"] = "json"
    url = f"{_EODHD_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(0.02)  # ~50/sec, well under EODHD's 1000/min
    return resp.json()


# ---------------------------------------------------------------------------
# EODHD: Bulk EOD (1 call per exchange per day)
# ---------------------------------------------------------------------------

def load_eodhd_bulk_eod(
    exchange: str = "US",
    date: str = "2026-03-28",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch EOD data for ALL tickers on an exchange in 1 API call.
    Returns DataFrame with columns: code, open, high, low, close, volume, etc.
    """
    cache_name = f"eodhd_bulk_{exchange}_{date}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached

    print(f"  [EODHD] Fetching bulk EOD for {exchange} on {date}...")
    data = _eodhd_get(f"eod-bulk-last-day/{exchange}", {"date": date})
    df = pd.DataFrame(data)
    _save(df, cache_name)
    print(f"  [EODHD] Got {len(df)} tickers")
    return df


# ---------------------------------------------------------------------------
# EODHD: Quarterly Fundamentals
# ---------------------------------------------------------------------------

def load_eodhd_fundamentals(
    tickers: List[str],
    use_cache: bool = True,
    max_tickers: int = 3000,
) -> Dict[str, dict]:
    """
    Fetch quarterly fundamentals (income stmt, balance sheet, cash flow)
    for each ticker. Returns dict of ticker -> raw JSON response.

    Rate: ~5 calls/sec, ~3000 tickers = ~10 min first time.
    Cached indefinitely (fundamentals change quarterly).
    """
    cache_name = "eodhd_fundamentals"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"  [EODHD] Fundamentals loaded from cache ({len(cached)} tickers)")
            return cached

    subset = tickers[:max_tickers]
    print(f"  [EODHD] Fetching fundamentals for {len(subset)} tickers...")

    results = {}
    for i, ticker in enumerate(subset):
        try:
            # EODHD uses TICKER.EXCHANGE format
            symbol = f"{ticker}.US"
            data = _eodhd_get(f"fundamentals/{symbol}", {
                "filter": "Financials,General,Highlights,Valuation"
            })
            if data and isinstance(data, dict):
                results[ticker] = data
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(subset)} tickers fetched...")

    _save(results, cache_name)
    print(f"  [EODHD] Fundamentals fetched for {len(results)} tickers")
    return results


def parse_eodhd_quarterly_financials(
    fundamentals: Dict[str, dict],
    tickers: List[str],
) -> pd.DataFrame:
    """
    Parse quarterly financial statements into a structured DataFrame.

    Extracts key fields from each quarterly report:
      - Revenue, net income, gross profit, operating income
      - Total assets, total liabilities, total equity
      - Operating cash flow, capital expenditures, free cash flow
      - EPS, book value per share
      - Filing date for point-in-time alignment

    Returns MultiIndex DataFrame: columns = (ticker, metric), index = filing_date
    """
    records = []

    for ticker in tickers:
        if ticker not in fundamentals:
            continue
        data = fundamentals[ticker]

        financials = data.get("Financials", {})
        if not financials:
            continue

        # Quarterly income statement
        income_q = financials.get("Income_Statement", {}).get("quarterly", {})
        balance_q = financials.get("Balance_Sheet", {}).get("quarterly", {})
        cashflow_q = financials.get("Cash_Flow", {}).get("quarterly", {})

        # Also grab highlights for current metrics
        highlights = data.get("Highlights", {})
        valuation = data.get("Valuation", {})

        # Parse quarterly data
        for date_key, income in income_q.items():
            try:
                filing_date = income.get("filing_date") or income.get("date") or date_key

                balance = balance_q.get(date_key, {})
                cashflow = cashflow_q.get(date_key, {})

                record = {
                    "ticker": ticker,
                    "date": pd.Timestamp(filing_date),
                    "report_date": pd.Timestamp(date_key),
                    # Income statement
                    "revenue": _safe_float(income.get("totalRevenue")),
                    "gross_profit": _safe_float(income.get("grossProfit")),
                    "operating_income": _safe_float(income.get("operatingIncome")),
                    "net_income": _safe_float(income.get("netIncome")),
                    "ebitda": _safe_float(income.get("ebitda")),
                    "eps": _safe_float(income.get("epsDiluted") or income.get("eps")),
                    # Balance sheet
                    "total_assets": _safe_float(balance.get("totalAssets")),
                    "total_liabilities": _safe_float(balance.get("totalLiab")),
                    "total_equity": _safe_float(balance.get("totalStockholderEquity")),
                    "total_debt": _safe_float(balance.get("longTermDebt")),
                    "cash": _safe_float(balance.get("cash")),
                    "shares_outstanding": _safe_float(balance.get("commonStockSharesOutstanding")),
                    # Cash flow
                    "operating_cf": _safe_float(cashflow.get("totalCashFromOperatingActivities")),
                    "capex": _safe_float(cashflow.get("capitalExpenditures")),
                    "dividends_paid": _safe_float(cashflow.get("dividendsPaid")),
                }
                # Derived
                rev = record["revenue"]
                gp = record["gross_profit"]
                ni = record["net_income"]
                ta = record["total_assets"]
                te = record["total_equity"]
                ocf = record["operating_cf"]
                capex = record["capex"]

                record["gross_margin"] = gp / rev if rev and rev > 0 else None
                record["operating_margin"] = record["operating_income"] / rev if rev and rev > 0 else None
                record["net_margin"] = ni / rev if rev and rev > 0 else None
                record["roe"] = ni / te if te and te > 0 else None
                record["roa"] = ni / ta if ta and ta > 0 else None
                record["debt_to_equity"] = record["total_debt"] / te if te and te > 0 else None
                record["fcf"] = (ocf or 0) - abs(capex or 0) if ocf is not None else None
                record["accruals"] = ((ni or 0) - (ocf or 0)) / ta if ta and ta > 0 else None

                records.append(record)
            except Exception:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])
    return df


def _safe_float(val):
    """Safely convert to float, handling None/str/etc."""
    if val is None or val == "None" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# EODHD: Dividends and Splits
# ---------------------------------------------------------------------------

def load_eodhd_dividends(
    tickers: List[str],
    start: str = "2013-01-01",
    use_cache: bool = True,
    max_workers: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical dividends for tickers. Parallel: ~3 min for 4600 tickers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_name = f"eodhd_dividends_{start}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"  [EODHD] Dividends loaded from cache ({len(cached)} tickers)")
            return cached

    def _fetch_div(ticker):
        try:
            params = {"api_token": EODHD_KEY, "fmt": "json", "from": start}
            resp = requests.get(f"{_EODHD_BASE}/div/{ticker}.US", params=params, timeout=15)
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    return ticker, df.set_index("date").sort_index()
        except Exception:
            pass
        return ticker, None

    print(f"  [EODHD] Fetching dividends for {len(tickers)} tickers ({max_workers} threads)...")
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_div, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker, df = future.result()
            done += 1
            if df is not None:
                results[ticker] = df
            if done % 500 == 0:
                print(f"    {done}/{len(tickers)} ({len(results)} with data)")

    _save(results, cache_name)
    print(f"  [EODHD] Dividends for {len(results)} tickers")
    return results


# ---------------------------------------------------------------------------
# EODHD: Sentiment (daily pre-computed sentiment scores)
# ---------------------------------------------------------------------------

def load_eodhd_sentiment(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    use_cache: bool = True,
    max_tickers: int = 3000,
    max_workers: int = 15,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily sentiment scores from EODHD. Parallel: ~3 min for 3000 tickers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_name = f"eodhd_sentiment_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"  [EODHD] Sentiment loaded from cache ({len(cached)} tickers)")
            return cached

    subset = tickers[:max_tickers]

    def _fetch_sent(ticker):
        try:
            params = {"api_token": EODHD_KEY, "fmt": "json",
                      "s": f"{ticker}.US", "from": start, "to": end}
            resp = requests.get(f"{_EODHD_BASE}/sentiments", params=params, timeout=15)
            data = resp.json()
            key = f"{ticker}.US"
            if isinstance(data, dict) and key in data:
                records = data[key]
                if records and len(records) > 10:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                    return ticker, df.set_index("date").sort_index()
        except Exception:
            pass
        return ticker, None

    print(f"  [EODHD] Fetching sentiment for {len(subset)} tickers ({max_workers} threads)...")
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_sent, t): t for t in subset}
        done = 0
        for future in as_completed(futures):
            ticker, df = future.result()
            done += 1
            if df is not None:
                results[ticker] = df
            if done % 500 == 0:
                print(f"    {done}/{len(subset)} ({len(results)} with data)")

    _save(results, cache_name)
    print(f"  [EODHD] Sentiment for {len(results)} tickers")
    return results


# ---------------------------------------------------------------------------
# EODHD: Live quotes (for live trading mode)
# ---------------------------------------------------------------------------

def load_eodhd_live_quote(ticker: str) -> dict:
    """Get real-time delayed quote for a single ticker."""
    return _eodhd_get(f"real-time/{ticker}.US")


# ═══════════════════════════════════════════════════════════════════════════
# FINNHUB API
# ═══════════════════════════════════════════════════════════════════════════

_FINNHUB_BASE = "https://finnhub.io/api/v1"
_finnhub_last_call = 0.0


def _finnhub_get(endpoint: str, params: dict = None) -> dict:
    """Rate-limited Finnhub API call. Enforces 60 calls/min (1/sec)."""
    global _finnhub_last_call
    if params is None:
        params = {}
    params["token"] = FINNHUB_KEY

    # Enforce rate limit: 1 call per 1.05 seconds
    elapsed = time.time() - _finnhub_last_call
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)

    url = f"{_FINNHUB_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    _finnhub_last_call = time.time()

    if resp.status_code == 429:
        # Rate limited — wait and retry once
        print("    [Finnhub] Rate limited, waiting 60s...")
        time.sleep(60)
        resp = requests.get(url, params=params, timeout=15)

    resp.raise_for_status()
    return resp.json()


def load_finnhub_data(
    tickers: List[str],
    max_tickers: int = 500,
    use_cache: bool = True,
) -> Dict[str, dict]:
    """
    Fetch all Finnhub free-tier data for up to max_tickers stocks.

    For each ticker, fetches 4 endpoints:
      1. Recommendation trends (analyst consensus time-series)
      2. Price target (consensus target price)
      3. EPS surprises (last 4 quarters)
      4. Company news (last 30 days)

    At 60 calls/min, 500 tickers * 4 endpoints = 2000 calls = ~35 min.
    Cached after first fetch.
    """
    cache_name = "finnhub_all_data"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"  [Finnhub] Data loaded from cache ({len(cached)} tickers)")
            return cached

    subset = tickers[:max_tickers]
    print(f"  [Finnhub] Fetching data for {len(subset)} tickers (4 endpoints each)...")
    print(f"  [Finnhub] Estimated time: ~{len(subset) * 4 / 57:.0f} min")

    results = {}
    for i, ticker in enumerate(subset):
        ticker_data = {}

        # 1. Recommendation trends
        try:
            rec = _finnhub_get("stock/recommendation", {"symbol": ticker})
            if rec and isinstance(rec, list):
                ticker_data["recommendations"] = rec
        except Exception:
            pass

        # 2. Price target
        try:
            pt = _finnhub_get("stock/price-target", {"symbol": ticker})
            if pt and isinstance(pt, dict):
                ticker_data["price_target"] = pt
        except Exception:
            pass

        # 3. EPS surprises
        try:
            eps = _finnhub_get("stock/earnings", {"symbol": ticker, "limit": 4})
            if eps and isinstance(eps, list):
                ticker_data["eps_surprises"] = eps
        except Exception:
            pass

        # 4. Company news (last 30 days)
        try:
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            month_ago = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            news = _finnhub_get("company-news", {
                "symbol": ticker, "from": month_ago, "to": today
            })
            if news and isinstance(news, list):
                ticker_data["news_count"] = len(news)
                # Simple sentiment: count positive vs negative headlines
                pos_words = {"beat", "surge", "rally", "upgrade", "record", "strong", "growth", "profit"}
                neg_words = {"miss", "drop", "fall", "downgrade", "loss", "weak", "decline", "cut"}
                pos = sum(1 for n in news if any(w in n.get("headline", "").lower() for w in pos_words))
                neg = sum(1 for n in news if any(w in n.get("headline", "").lower() for w in neg_words))
                ticker_data["news_sentiment"] = (pos - neg) / max(len(news), 1)
        except Exception:
            pass

        if ticker_data:
            results[ticker] = ticker_data

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(subset)} tickers ({len(results)} with data)")

    _save(results, cache_name)
    print(f"  [Finnhub] Done — {len(results)} tickers with data")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS (convert raw API data to model features)
# ═══════════════════════════════════════════════════════════════════════════

def build_eodhd_features(
    quarterly_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional features from EODHD quarterly fundamentals.

    Features (all cross-sectionally ranked [0,1]):
      quarterly_revenue_growth  — QoQ revenue growth rate
      quarterly_roe             — return on equity (quarterly)
      quarterly_gross_margin    — gross profit / revenue
      quarterly_fcf_yield       — free cash flow / market cap proxy
      quarterly_accruals        — (net income - operating CF) / total assets
      quarterly_debt_change     — QoQ change in debt/equity ratio
      quarterly_eps_growth      — QoQ EPS growth
    """
    signals = {}

    if quarterly_df is None or quarterly_df.empty:
        return signals

    def _cs_rank(df):
        return df.rank(axis=1, pct=True)

    # Build per-ticker time series, then broadcast to daily
    metrics = [
        "revenue", "gross_margin", "roe", "roa", "net_margin",
        "operating_margin", "fcf", "accruals", "debt_to_equity", "eps",
    ]

    for metric in metrics:
        if metric not in quarterly_df.columns:
            continue

        panel = pd.DataFrame(np.nan, index=date_index, columns=tickers)

        for ticker in tickers:
            tk_data = quarterly_df[quarterly_df["ticker"] == ticker].sort_values("date")
            if tk_data.empty or metric not in tk_data.columns:
                continue

            # Point-in-time: use filing date, forward-fill
            for _, row in tk_data.iterrows():
                d = row["date"]
                if pd.notna(d) and d in date_index:
                    panel.loc[d:, ticker] = row[metric]
                elif pd.notna(d):
                    valid = date_index[date_index >= d]
                    if len(valid) > 0:
                        panel.loc[valid[0]:, ticker] = row[metric]

        # Forward fill within ticker columns
        panel = panel.ffill()

        if panel.notna().any().any():
            # For growth metrics, compute QoQ change
            if metric in ("revenue", "eps"):
                growth = panel.pct_change(63)  # ~1 quarter of trading days
                growth = growth.clip(-2, 10)  # clip extreme values
                signals[f"q_{metric}_growth"] = _cs_rank(growth)
            elif metric in ("accruals",):
                # Lower accruals = better (earnings backed by cash flow)
                signals[f"q_{metric}"] = _cs_rank(-panel)
            elif metric in ("debt_to_equity",):
                # Lower leverage = better
                signals[f"q_{metric}"] = _cs_rank(-panel)
            else:
                signals[f"q_{metric}"] = _cs_rank(panel)

    return signals


def build_eodhd_sentiment_features(
    sentiment_data: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Build features from EODHD daily sentiment scores.

    Features:
      eodhd_sentiment       — daily sentiment score (0-1), cross-sectionally ranked
      eodhd_sentiment_chg   — 5-day change in sentiment (momentum in sentiment)
      eodhd_news_volume     — article count (attention proxy), cross-sectionally ranked
    """
    signals = {}

    if not sentiment_data:
        return signals

    def _cs_rank(df):
        return df.rank(axis=1, pct=True)

    # Build panels
    sent_panel = pd.DataFrame(np.nan, index=date_index, columns=tickers)
    count_panel = pd.DataFrame(np.nan, index=date_index, columns=tickers)

    for ticker, df in sentiment_data.items():
        if ticker not in tickers:
            continue
        if "normalized" in df.columns:
            # shift(1): sentiment aggregated at end-of-day is not available
            # until the next morning. Without shift, we'd trade on today's
            # sentiment using today's close — lookahead bias.
            aligned = df["normalized"].reindex(date_index, method="ffill").shift(1)
            sent_panel[ticker] = aligned
        if "count" in df.columns:
            aligned = df["count"].reindex(date_index, method="ffill").shift(1)
            count_panel[ticker] = aligned

    if sent_panel.notna().any().any():
        signals["eodhd_sentiment"] = _cs_rank(sent_panel)
        # Sentiment momentum: 5-day change
        sent_chg = sent_panel.diff(5)
        if sent_chg.notna().any().any():
            signals["eodhd_sentiment_chg"] = _cs_rank(sent_chg)

    if count_panel.notna().any().any():
        signals["eodhd_news_volume"] = _cs_rank(count_panel)

    return signals


def build_finnhub_features(
    finnhub_data: Dict[str, dict],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional features from Finnhub data.

    Features (all cross-sectionally ranked [0,1]):
      fh_recommendation_score  — weighted analyst consensus (higher = more bullish)
      fh_target_upside         — (target price / current) - 1
      fh_eps_surprise          — most recent EPS surprise %
      fh_news_sentiment        — positive - negative news ratio
    """
    signals = {}

    if not finnhub_data:
        return signals

    def _cs_rank(df):
        return df.rank(axis=1, pct=True)

    # 1. Recommendation score
    rec_scores = {}
    for ticker, data in finnhub_data.items():
        recs = data.get("recommendations", [])
        if recs and len(recs) > 0:
            latest = recs[0]  # most recent period
            # Weighted score: strongBuy=5, buy=4, hold=3, sell=2, strongSell=1
            total = (latest.get("strongBuy", 0) + latest.get("buy", 0) +
                     latest.get("hold", 0) + latest.get("sell", 0) +
                     latest.get("strongSell", 0))
            if total > 0:
                score = (latest.get("strongBuy", 0) * 5 +
                         latest.get("buy", 0) * 4 +
                         latest.get("hold", 0) * 3 +
                         latest.get("sell", 0) * 2 +
                         latest.get("strongSell", 0) * 1) / total
                rec_scores[ticker] = score

    if rec_scores:
        s = pd.Series(rec_scores).reindex(tickers)
        df = pd.DataFrame(
            np.tile(s.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        df.loc[:, s.isna()] = np.nan
        signals["fh_recommendation_score"] = _cs_rank(df)

    # 2. Target upside
    target_upside = {}
    for ticker, data in finnhub_data.items():
        pt = data.get("price_target", {})
        target = pt.get("targetMean") or pt.get("targetMedian")
        last_price = pt.get("lastUpdatedPrice")
        if target and last_price and last_price > 0:
            target_upside[ticker] = target / last_price - 1

    if target_upside:
        s = pd.Series(target_upside).reindex(tickers)
        df = pd.DataFrame(
            np.tile(s.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        df.loc[:, s.isna()] = np.nan
        signals["fh_target_upside"] = _cs_rank(df)

    # 3. EPS surprise
    eps_surprise = {}
    for ticker, data in finnhub_data.items():
        eps = data.get("eps_surprises", [])
        if eps and len(eps) > 0:
            latest = eps[0]
            surprise_pct = latest.get("surprisePercent")
            if surprise_pct is not None:
                eps_surprise[ticker] = surprise_pct

    if eps_surprise:
        s = pd.Series(eps_surprise).reindex(tickers)
        df = pd.DataFrame(
            np.tile(s.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        df.loc[:, s.isna()] = np.nan
        signals["fh_eps_surprise"] = _cs_rank(df)

    # 4. News sentiment
    sentiment = {}
    for ticker, data in finnhub_data.items():
        sent = data.get("news_sentiment")
        if sent is not None:
            sentiment[ticker] = sent

    if sentiment:
        s = pd.Series(sentiment).reindex(tickers)
        df = pd.DataFrame(
            np.tile(s.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        df.loc[:, s.isna()] = np.nan
        signals["fh_news_sentiment"] = _cs_rank(df)

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# MASTER LOADER (call this from run_strategy.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_all_api_data(
    tickers: List[str],
    date_index: pd.DatetimeIndex,
    finnhub_max: int = 500,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load and build features from both EODHD and Finnhub.

    Returns a dict of feature_name -> DataFrame(date x ticker) that can be
    merged directly into the alt_features pipeline.

    Total first-time fetch: ~10 min EODHD + ~35 min Finnhub = ~45 min.
    Subsequent runs: instant from cache.
    """
    all_features = {}
    tickers_index = pd.Index(tickers)

    # ── EODHD: Quarterly Fundamentals (requires All-In-One plan) ─────────
    # NOTE: The $20 All-World plan does NOT include fundamentals.
    # Uncomment below if you upgrade to All-In-One ($40+/mo).
    # print("\n[API] Loading EODHD quarterly fundamentals...")
    # raw_fundamentals = load_eodhd_fundamentals(tickers, use_cache=use_cache)
    # if raw_fundamentals:
    #     quarterly_df = parse_eodhd_quarterly_financials(raw_fundamentals, tickers)
    #     if not quarterly_df.empty:
    #         eodhd_features = build_eodhd_features(quarterly_df, tickers_index, date_index)
    #         all_features.update(eodhd_features)

    # ── EODHD: Dividends (available on All-World plan) ───────────────────
    print("[API] Loading EODHD dividends...")
    dividends = load_eodhd_dividends(tickers, use_cache=use_cache)
    if dividends:
        # Build dividend yield feature
        div_panel = pd.DataFrame(0.0, index=date_index, columns=tickers_index)
        for ticker, div_df in dividends.items():
            if ticker in div_panel.columns and "value" in div_df.columns:
                annual_div = div_df["value"].resample("YE").sum()
                annual_div = annual_div.reindex(date_index, method="ffill")
                div_panel[ticker] = annual_div
        if div_panel.sum().sum() > 0:
            all_features["dividend_yield"] = div_panel.rank(axis=1, pct=True)
            print(f"  [EODHD] Built dividend yield feature")

    # ── EODHD: Sentiment (daily scores + article counts) ──────────────────
    print("[API] Loading EODHD sentiment data...")
    sentiment_data = load_eodhd_sentiment(
        tickers, start=date_index[0].strftime("%Y-%m-%d"),
        end=date_index[-1].strftime("%Y-%m-%d"), use_cache=use_cache,
    )
    if sentiment_data:
        sent_features = build_eodhd_sentiment_features(sentiment_data, tickers_index, date_index)
        all_features.update(sent_features)
        print(f"  [EODHD] Built {len(sent_features)} sentiment features")

    # ─────────────────────────────────────────────────────────────────────
    # Finnhub: auto-gated on cached-ticker coverage.
    #
    # Finnhub features are auto-disabled below FINNHUB_MIN_COVERAGE cached
    # tickers. Partial data creates sparse features that distort the model.
    # Run `python fetch_finnhub.py --max 3000` overnight to populate the
    # cache; then Finnhub features auto-activate on the next run.
    #
    # This is an opt-in-via-cache-completion path. Do NOT remove — leave the
    # code path live so that once the cache is filled the features light up.
    # ─────────────────────────────────────────────────────────────────────
    FINNHUB_MIN_COVERAGE = 1000  # min cached tickers before Finnhub activates
    cached_fh = _load("finnhub_all_data")
    if cached_fh and len(cached_fh) >= FINNHUB_MIN_COVERAGE:
        fh_features = build_finnhub_features(cached_fh, tickers_index, date_index)
        all_features.update(fh_features)
        print(f"  [Finnhub] Built {len(fh_features)} features "
              f"(coverage: {len(cached_fh)} tickers)")
    else:
        n_cached = len(cached_fh) if cached_fh else 0
        print(f"[API] Finnhub: disabled ({n_cached}/{FINNHUB_MIN_COVERAGE} "
              f"cached tickers). Run `python fetch_finnhub.py --max 3000` "
              f"overnight to populate; features auto-activate once cache "
              f"reaches threshold.")

    print(f"\n[API] Total new features from paid APIs: {len(all_features)}")
    return all_features
