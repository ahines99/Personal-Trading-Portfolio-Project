"""
alt_data_loader.py
------------------
Loaders for five free alternative data sources:

  1. SEC EDGAR  — fundamental factors via XBRL Company Facts API
  2. FRED        — macro regime series (yield curve, HY spread, VIX, fed funds, etc.)
  3. CBOE VIX   — VIX term structure (VIX9D / VIX / VIX3M / VIX6M)
  4. FINRA SI    — short interest via yfinance shortRatio (point-in-time limitation noted)
  5. yfinance    — earnings calendar + EPS surprise

All loaders write pickle caches under data/cache/ and respect a
`use_cache=True` parameter for fast reloads.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ── cache directory ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_CACHE = _ROOT / "data" / "cache"
_CACHE.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return _CACHE / f"{name}.pkl"


def _save(obj, name: str):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)


def _load(name: str):
    p = _cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SEC EDGAR  — XBRL Company Facts
# ─────────────────────────────────────────────────────────────────────────────

_EDGAR_HEADERS = {
    "User-Agent": "quant-research-project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

_EDGAR_CONCEPTS = {
    # concept : (taxonomy, label)
    # ── Core (existing) ──────────────────────────────────────────────────
    "Assets":                                ("us-gaap", "Total Assets"),
    "Liabilities":                           ("us-gaap", "Total Liabilities"),
    "StockholdersEquity":                    ("us-gaap", "Shareholders Equity"),
    "NetIncomeLoss":                         ("us-gaap", "Net Income"),
    "GrossProfit":                           ("us-gaap", "Gross Profit"),
    "Revenues":                              ("us-gaap", "Revenue"),
    "EarningsPerShareBasic":                 ("us-gaap", "EPS Basic"),
    # ── New: for robust anomalies ────────────────────────────────────────
    # Accruals (Sloan 1996): NetIncome - OCF → high accruals = overearning
    "NetCashProvidedByOperatingActivities":  ("us-gaap", "Operating Cash Flow"),
    # Net Operating Assets (Hirshleifer 2004): (Assets-Cash)-(Liab-Debt)
    "CashAndCashEquivalentsAtCarryingValue": ("us-gaap", "Cash"),
    "LongTermDebtNoncurrent":                ("us-gaap", "Long Term Debt"),
    # Net Share Issuance (Pontiff/Woodgate 2008): change in shares out
    "CommonStockSharesOutstanding":          ("us-gaap", "Shares Outstanding"),
    # Operating Profitability (Fama-French RMW): (Rev-COGS-SGA-IntExp)/Equity
    "SellingGeneralAndAdministrativeExpense": ("us-gaap", "SGA"),
    "InterestExpense":                       ("us-gaap", "Interest Expense"),
    # Piotroski F-Score / Current Ratio
    "AssetsCurrent":                         ("us-gaap", "Current Assets"),
    "LiabilitiesCurrent":                    ("us-gaap", "Current Liabilities"),
}


def _get_cik_map(tickers: List[str]) -> Dict[str, str]:
    """Download SEC ticker→CIK mapping (cached 7 days)."""
    cache_name = "edgar_cik_map"
    cached = _load(cache_name)
    if cached is not None and (time.time() - cached["ts"]) < 7 * 86400:
        return cached["data"]

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        mapping = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in raw.values()
        }
        _save({"ts": time.time(), "data": mapping}, cache_name)
        return mapping
    except Exception as e:
        warnings.warn(f"EDGAR CIK map download failed: {e}")
        return cached["data"] if cached else {}


def _fetch_company_facts(cik: str) -> Optional[dict]:
    """Fetch raw XBRL company facts for one CIK."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _extract_annual_series(facts: dict, concept: str, taxonomy: str = "us-gaap") -> pd.Series:
    """
    Extract annual (10-K) filings for a concept into a date-indexed Series.
    Uses 'filed' date (point-in-time) not period end date to avoid look-ahead.
    """
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        # prefer USD, fall back to shares for EPS concepts
        unit_key = "USD" if "USD" in units else list(units.keys())[0]
        rows = units[unit_key]
        records = []
        for r in rows:
            if r.get("form") in ("10-K", "10-K/A") and "filed" in r and "val" in r:
                records.append({"filed": pd.to_datetime(r["filed"]), "val": r["val"]})
        if not records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(records).sort_values("filed").drop_duplicates("filed", keep="last")
        return df.set_index("filed")["val"]
    except (KeyError, IndexError):
        return pd.Series(dtype=float)


def load_edgar_fundamentals(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    max_tickers: int = 10_000,
    sleep_sec: float = 0.12,   # ~8 req/sec, well within SEC's 10 req/s limit
) -> pd.DataFrame:
    """
    Returns a DataFrame with MultiIndex columns (ticker, factor) and DatetimeIndex.
    Factors: book_to_market, roe, gross_margin, asset_growth, leverage, eps_trend.
    Values are filled forward to daily frequency and lagged 1 day (point-in-time).

    Parameters
    ----------
    max_tickers : safety ceiling on tickers to fetch (default = all)
    """
    cache_name = f"edgar_fundamentals_{start}_{end}_{max_tickers}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [EDGAR] loaded from cache ({cached.shape})")
            return cached

    print(f"      [EDGAR] fetching fundamentals for up to {max_tickers} tickers...")
    cik_map   = _get_cik_map(tickers)
    date_idx  = pd.bdate_range(start, end)
    results   = {}

    subset = [t for t in tickers if t in cik_map][:max_tickers]
    for i, ticker in enumerate(subset):
        cik   = cik_map[ticker]
        facts = _fetch_company_facts(cik)
        if facts is None:
            continue

        assets  = _extract_annual_series(facts, "Assets")
        liab    = _extract_annual_series(facts, "Liabilities")
        equity  = _extract_annual_series(facts, "StockholdersEquity")
        net_inc = _extract_annual_series(facts, "NetIncomeLoss")
        gross   = _extract_annual_series(facts, "GrossProfit")
        rev     = _extract_annual_series(facts, "Revenues")
        eps     = _extract_annual_series(facts, "EarningsPerShareBasic")

        ticker_df = pd.DataFrame(index=date_idx)

        def _daily(series: pd.Series, shift: int = 1) -> pd.Series:
            """Align annual series to daily, ffill, shift for point-in-time."""
            if series.empty:
                return pd.Series(np.nan, index=date_idx)
            s = series.reindex(date_idx, method="ffill")
            return s.shift(shift)

        # Book-to-Market proxy: equity / assets (higher = cheaper)
        eq_d  = _daily(equity)
        ast_d = _daily(assets)
        ticker_df["book_to_market"] = (eq_d / ast_d.replace(0, np.nan)).clip(-5, 5)

        # ROE: net_income / equity
        ni_d = _daily(net_inc)
        ticker_df["roe"] = (ni_d / eq_d.replace(0, np.nan)).clip(-2, 2)

        # Gross Margin: gross_profit / revenue
        gp_d  = _daily(gross)
        rev_d = _daily(rev)
        ticker_df["gross_margin"] = (gp_d / rev_d.replace(0, np.nan)).clip(-1, 1)

        # Asset Growth: YoY change in assets (lower = better, contrarian)
        ticker_df["asset_growth"] = ast_d.pct_change(252, fill_method=None).clip(-1, 2)

        # Leverage: liabilities / assets
        lb_d = _daily(liab)
        ticker_df["leverage"] = (lb_d / ast_d.replace(0, np.nan)).clip(0, 5)

        # EPS Trend: 1-year change in EPS
        eps_d = _daily(eps)
        ticker_df["eps_trend"] = eps_d.diff(252)

        # Gross Profitability: (Revenue - COGS) / Assets = Gross Profit / Assets
        # Novy-Marx (2013) "The Other Side of Value" — one of the most robust
        # anomalies, survives Hou/Xue/Zhang (2020) replication.
        ticker_df["gross_profitability"] = (gp_d / ast_d.replace(0, np.nan)).clip(-1, 2)

        # ── New robust anomalies ─────────────────────────────────────────
        # Extract new EDGAR concepts (graceful if missing)
        ocf     = _extract_annual_series(facts, "NetCashProvidedByOperatingActivities")
        cash    = _extract_annual_series(facts, "CashAndCashEquivalentsAtCarryingValue")
        lt_debt = _extract_annual_series(facts, "LongTermDebtNoncurrent")
        shares  = _extract_annual_series(facts, "CommonStockSharesOutstanding")
        sga     = _extract_annual_series(facts, "SellingGeneralAndAdministrativeExpense")
        intexp  = _extract_annual_series(facts, "InterestExpense")
        ca      = _extract_annual_series(facts, "AssetsCurrent")
        cl      = _extract_annual_series(facts, "LiabilitiesCurrent")

        ocf_d   = _daily(ocf)
        cash_d  = _daily(cash)
        debt_d  = _daily(lt_debt)
        shr_d   = _daily(shares)
        sga_d   = _daily(sga)
        int_d   = _daily(intexp)
        ca_d    = _daily(ca)
        cl_d    = _daily(cl)

        # Accruals (Sloan 1996): (NetIncome - OCF) / Assets
        # High accruals = earnings exceed cash flow = negative signal
        ticker_df["accruals"] = ((ni_d - ocf_d) / ast_d.replace(0, np.nan)).clip(-2, 2)

        # Net Operating Assets (Hirshleifer 2004): (Assets-Cash)-(Liab-Debt) / Assets
        # High NOA = "balance sheet bloat" = negative signal
        op_assets = ast_d - cash_d.fillna(0)
        op_liab = lb_d - debt_d.fillna(0)
        ticker_df["net_operating_assets"] = ((op_assets - op_liab) / ast_d.replace(0, np.nan)).clip(-2, 5)

        # Net Share Issuance (Pontiff/Woodgate 2008): log change in shares
        # Firms issuing shares underperform; repurchasers outperform
        if not shr_d.isna().all():
            ticker_df["net_share_issuance"] = np.log(
                shr_d / shr_d.shift(252).replace(0, np.nan)
            ).clip(-1, 1)

        # Operating Profitability (FF5 RMW): (Rev-COGS-SGA-IntExp) / Equity
        op_profit = gp_d - sga_d.fillna(0) - int_d.fillna(0)
        ticker_df["operating_profitability"] = (op_profit / eq_d.replace(0, np.nan)).clip(-2, 5)

        # Current Ratio change (Piotroski F-Score component)
        cr = ca_d / cl_d.replace(0, np.nan)
        ticker_df["current_ratio_chg"] = cr.diff(252).clip(-2, 2)

        results[ticker] = ticker_df
        time.sleep(sleep_sec)

        if (i + 1) % 50 == 0:
            print(f"        {i+1}/{len(subset)} tickers fetched")

    if not results:
        print("      [EDGAR] no data retrieved")
        empty = pd.DataFrame(index=date_idx)
        return empty

    out = pd.concat(results, axis=1)
    out.index = pd.DatetimeIndex(out.index)
    _save(out, cache_name)
    print(f"      [EDGAR] done — {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FRED  — macro regime series
# ─────────────────────────────────────────────────────────────────────────────

_FRED_SERIES = {
    "T10Y2Y":       "Yield Curve (10Y-2Y)",
    "BAMLH0A0HYM2": "HY OAS Spread",
    "VIXCLS":       "VIX Close",
    "DFF":          "Fed Funds Rate",
    "UNRATE":       "Unemployment Rate",
    "T10YIE":       "10Y Breakeven Inflation",
    "NAPM":         "ISM Manufacturing PMI",
    "ICSA":         "Initial Jobless Claims",
    "PERMIT":       "Building Permits",
}


def load_fred_macro(
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame (DatetimeIndex × macro series) at daily frequency.
    Values are forward-filled to business days and shifted 1 day (point-in-time).
    """
    cache_name = f"fred_macro_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [FRED] loaded from cache ({cached.shape})")
            return cached

    print("      [FRED] downloading macro series...")
    date_idx = pd.bdate_range(start, end)
    frames   = {}

    for series_id, label in _FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index)
            s  = df.iloc[:, 0].replace(".", np.nan).astype(float)
            s  = s.reindex(date_idx, method="ffill").shift(1)  # point-in-time
            frames[series_id] = s
            print(f"        {series_id}: {s.notna().sum()} valid obs")
        except Exception as e:
            warnings.warn(f"FRED {series_id} failed: {e}")

    # ── yfinance fallback for critical series that FRED missed ────────────
    critical_missing = {"T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DFF"} - set(frames.keys())
    if critical_missing:
        print(f"      [FRED] {len(critical_missing)} critical series missing, trying yfinance fallback...")
        try:
            import yfinance as yf

            def _yf_series(ticker, start, end):
                """Download single ticker from yfinance, return as 1D Series."""
                df = yf.download(ticker, start=start, end=end, progress=False)
                if df.empty:
                    return pd.Series(dtype=float)
                col = df["Close"]
                # yfinance may return DataFrame with multi-level columns
                if hasattr(col, 'columns'):
                    col = col.iloc[:, 0]
                return col

            # VIX
            if "VIXCLS" not in frames:
                vix_s = _yf_series("^VIX", start, end)
                if len(vix_s) > 0:
                    s = vix_s.reindex(date_idx, method="ffill").shift(1)
                    frames["VIXCLS"] = s
                    print(f"        VIXCLS (from ^VIX): {s.notna().sum()} valid obs")

            # Yield curve proxy: 10Y - 3M
            if "T10Y2Y" not in frames:
                tnx_s = _yf_series("^TNX", start, end)
                irx_s = _yf_series("^IRX", start, end)
                if len(tnx_s) > 0 and len(irx_s) > 0:
                    yc = (tnx_s - irx_s.reindex(tnx_s.index, method="ffill")).reindex(date_idx, method="ffill").shift(1)
                    frames["T10Y2Y"] = yc
                    print(f"        T10Y2Y (from ^TNX-^IRX): {yc.notna().sum()} valid obs")

            # HY spread proxy: HYG vs LQD return differential
            if "BAMLH0A0HYM2" not in frames:
                hyg_s = _yf_series("HYG", start, end)
                lqd_s = _yf_series("LQD", start, end)
                if len(hyg_s) > 0 and len(lqd_s) > 0:
                    spread = (lqd_s.pct_change() - hyg_s.reindex(lqd_s.index, method="ffill").pct_change()).rolling(21).mean() * 10000
                    spread = spread.reindex(date_idx, method="ffill").shift(1)
                    frames["BAMLH0A0HYM2"] = spread
                    print(f"        BAMLH0A0HYM2 (from HYG-LQD): {spread.notna().sum()} valid obs")

            # Fed funds proxy: ^IRX (3-month T-bill rate)
            if "DFF" not in frames:
                irx_s = _yf_series("^IRX", start, end)
                if len(irx_s) > 0:
                    s = irx_s.reindex(date_idx, method="ffill").shift(1)
                    frames["DFF"] = s
                    print(f"        DFF (from ^IRX): {s.notna().sum()} valid obs")

        except Exception as e:
            print(f"      [FRED] yfinance fallback failed: {e}")

    if not frames:
        return pd.DataFrame(index=date_idx)

    out = pd.DataFrame(frames, index=date_idx)
    _save(out, cache_name)
    print(f"      [FRED] done — {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CBOE VIX Term Structure
# ─────────────────────────────────────────────────────────────────────────────

_VIX_URLS = {
    "VIX":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
}


def load_vix_term_structure(
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      VIX9D, VIX, VIX3M, VIX6M,
      term_slope      (VIX / VIX3M — >1 = backwardation / fear)
      vix_percentile  (rolling 252d rank of VIX)
      vix_change_5d   (5d % change in VIX)
    All shifted 1 day for point-in-time correctness.
    """
    cache_name = f"vix_term_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [VIX] loaded from cache ({cached.shape})")
            return cached

    print("      [VIX] downloading term structure...")
    date_idx = pd.bdate_range(start, end)
    raw      = {}

    for name, url in _VIX_URLS.items():
        try:
            df = pd.read_csv(url, skiprows=0)
            # CBOE format: DATE, OPEN, HIGH, LOW, CLOSE
            df.columns = [c.strip().upper() for c in df.columns]
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df.set_index("DATE").sort_index()
            raw[name] = df["CLOSE"].astype(float)
            print(f"        {name}: {len(raw[name])} rows")
        except Exception as e:
            warnings.warn(f"VIX {name} download failed: {e}")

    if "VIX" not in raw:
        return pd.DataFrame(index=date_idx)

    out = pd.DataFrame(raw, index=pd.DatetimeIndex(list(raw["VIX"].index)))
    out = out.reindex(date_idx, method="ffill")

    # Derived features
    if "VIX3M" in out.columns:
        out["term_slope"] = out["VIX"] / out["VIX3M"].replace(0, np.nan)
    else:
        out["term_slope"] = np.nan

    out["vix_percentile"] = (
        out["VIX"]
        .rolling(252, min_periods=63)
        .rank(pct=True)
    )
    out["vix_change_5d"] = out["VIX"].pct_change(5, fill_method=None)

    out = out.shift(1)  # point-in-time
    _save(out, cache_name)
    print(f"      [VIX] done — {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FINRA Short Interest  (via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_short_interest(
    tickers: List[str],
    use_cache: bool = True,
    cache_days: int = 7,
) -> pd.DataFrame:
    """
    Returns a DataFrame (tickers as columns) with shortRatio and shortPercentOfFloat.
    NOTE: yfinance only provides *current* short interest snapshot (no historical series).
    We store the snapshot with its retrieval timestamp and use it as a static signal.
    For real backtesting, the FINRA bi-monthly archive (finra.org/investors/learn-to-invest/
    advanced-investing/short-selling/regsho/short-sale-volume-data) would be needed.
    """
    cache_name = "short_interest_snapshot"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            age_days = (time.time() - cached["ts"]) / 86400
            if age_days < cache_days:
                print(f"      [SI] loaded snapshot from cache (age {age_days:.1f}d)")
                return cached["data"]

    print(f"      [SI] fetching short interest for {len(tickers)} tickers...")
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not installed — skipping short interest")
        return pd.DataFrame()

    records = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            records[ticker] = {
                "short_ratio":   info.get("shortRatio", np.nan),
                "short_pct_float": info.get("shortPercentOfFloat", np.nan),
            }
        except Exception:
            records[ticker] = {"short_ratio": np.nan, "short_pct_float": np.nan}

        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(tickers)}")
        time.sleep(0.20)

    df = pd.DataFrame(records).T
    _save({"ts": time.time(), "data": df}, cache_name)
    print(f"      [SI] done — {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  yfinance Earnings Calendar + EPS Surprise
# ─────────────────────────────────────────────────────────────────────────────

def load_earnings_calendar(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict mapping ticker → DataFrame with columns:
      earnings_date, eps_estimate, eps_actual, eps_surprise, eps_surprise_pct
    Uses yfinance .get_earnings_dates() (returns ~8 quarters of history).
    """
    cache_name = f"earnings_calendar_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [EARN] loaded from cache ({len(cached)} tickers)")
            return cached

    print(f"      [EARN] fetching earnings for {len(tickers)} tickers...")
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not installed — skipping earnings calendar")
        return {}

    results = {}
    for i, ticker in enumerate(tickers):
        try:
            t    = yf.Ticker(ticker)
            earn = t.get_earnings_dates(limit=40)  # ~10 years
            if earn is None or earn.empty:
                continue
            earn = earn.reset_index()
            earn.columns = [c.strip() for c in earn.columns]

            # Normalize column names across yfinance versions
            col_map = {}
            for c in earn.columns:
                cl = c.lower().replace(" ", "_")
                if "earnings" in cl and "date" in cl:
                    col_map[c] = "earnings_date"
                elif "eps_estimate" in cl or "estimate" in cl:
                    col_map[c] = "eps_estimate"
                elif "reported" in cl or "actual" in cl:
                    col_map[c] = "eps_actual"
                elif "surprise" in cl and "%" not in c:
                    col_map[c] = "eps_surprise"
            earn = earn.rename(columns=col_map)

            if "earnings_date" not in earn.columns:
                earn = earn.rename(columns={earn.columns[0]: "earnings_date"})

            earn["earnings_date"] = pd.to_datetime(
                earn["earnings_date"], utc=True
            ).dt.tz_localize(None)

            # Compute surprise % if we have estimate and actual
            if "eps_estimate" in earn.columns and "eps_actual" in earn.columns:
                est = pd.to_numeric(earn["eps_estimate"], errors="coerce")
                act = pd.to_numeric(earn["eps_actual"],   errors="coerce")
                earn["eps_surprise_pct"] = (
                    (act - est) / est.abs().replace(0, np.nan)
                ).clip(-2, 2)
            else:
                earn["eps_surprise_pct"] = np.nan

            results[ticker] = earn.set_index("earnings_date").sort_index()

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(tickers)}")
        time.sleep(0.20)

    _save(results, cache_name)
    print(f"      [EARN] done — {len(results)} tickers with data")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Insider Transactions  (via SEC EDGAR Submissions API — metadata only)
# ─────────────────────────────────────────────────────────────────────────────

def load_insider_transactions(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    max_tickers: int = 10_000,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict mapping ticker → DataFrame with columns:
      date, form (always '4')

    Uses SEC EDGAR Submissions API to get Form 4 filing dates directly
    from the submissions metadata — one API call per ticker, no XML parsing.
    Each Form 4 filing represents an insider transaction event.

    The feature builder counts filings over rolling windows to create
    insider activity signals. This is a proxy for buy/sell activity:
    more Form 4 filings = more insider trading = potential signal.
    """
    cache_name = f"insider_subs_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [INSIDER] loaded from cache ({len(cached)} tickers)")
            return cached

    print(f"      [INSIDER] fetching Form 4 metadata from SEC EDGAR...")
    cik_map = _get_cik_map(tickers)
    dt_start = pd.Timestamp(start)
    dt_end = pd.Timestamp(end)

    results = {}
    subset = [t for t in tickers if t in cik_map][:max_tickers]

    for i, ticker in enumerate(subset):
        cik = cik_map[ticker]
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
            if resp.status_code != 200:
                time.sleep(0.12)
                continue
            subs = resp.json()
        except Exception:
            time.sleep(0.12)
            continue

        # Extract Form 4 filing dates from recent filings
        recent = subs.get("filings", {}).get("recent", {})
        if not recent:
            time.sleep(0.12)
            continue

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])

        # Collect all Form 4 filing dates within range
        form4_dates = []
        for j, form in enumerate(forms):
            if form != "4" or j >= len(dates):
                continue
            filing_date = pd.Timestamp(dates[j])
            if dt_start <= filing_date <= dt_end:
                form4_dates.append(filing_date)

        if form4_dates:
            df = pd.DataFrame({"date": form4_dates, "form": "4"})
            df = df.set_index("date").sort_index()
            results[ticker] = df

        if (i + 1) % 200 == 0:
            print(f"        {i+1}/{len(subset)} tickers ({len(results)} with data)")
        time.sleep(0.12)  # respect SEC rate limit

    _save(results, cache_name)
    print(f"      [INSIDER] done — {len(results)} tickers with data")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Analyst Upgrades / Downgrades  (via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_earnings_estimates(
    tickers: List[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load analyst earnings estimates from Yahoo Finance.
    Returns DataFrame with tickers as index and columns like
    'targetMeanPrice', 'currentPrice', 'targetUpside',
    'numberOfAnalysts', 'recommendationMean'.
    Cached to avoid repeated API calls.
    """
    cache_name = "earnings_estimates"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [ESTIMATES] loaded from cache ({cached.shape})")
            return cached

    print(f"      [ESTIMATES] fetching estimates for {len(tickers)} tickers...")
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not installed — skipping earnings estimates")
        return pd.DataFrame()

    estimates = {}
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"        [estimates] {i}/{len(tickers)}...")
        try:
            t = yf.Ticker(ticker)
            # Get earnings estimate data
            info = t.info
            target_price = info.get('targetMeanPrice', None)
            current_price = info.get('currentPrice', None) or info.get('regularMarketPrice', None)

            estimates[ticker] = {
                'targetMeanPrice': target_price,
                'currentPrice': current_price,
                'targetUpside': (target_price / current_price - 1) if (target_price and current_price and current_price > 0) else None,
                'numberOfAnalysts': info.get('numberOfAnalystOpinions', None),
                'recommendationMean': info.get('recommendationMean', None),  # 1=strong buy, 5=sell
            }
        except Exception:
            pass

        if i % 50 == 49:
            time.sleep(1)

    result = pd.DataFrame(estimates).T

    _save(result, cache_name)
    print(f"      [ESTIMATES] Loaded estimates for {len(result)} tickers")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Analyst Upgrades / Downgrades  (via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_analyst_actions(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    max_tickers: int = 10_000,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict mapping ticker → DataFrame with columns:
      date, firm, to_grade, from_grade, action (e.g. 'up', 'down', 'main', 'init')
    Uses yfinance .upgrades_downgrades for historical analyst actions.
    """
    cache_name = f"analyst_actions_{start}_{end}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [ANALYST] loaded from cache ({len(cached)} tickers)")
            return cached

    print(f"      [ANALYST] fetching analyst actions for up to {max_tickers} tickers...")
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not installed — skipping analyst actions")
        return {}

    results = {}
    subset = tickers[:max_tickers]
    for i, ticker in enumerate(subset):
        try:
            t = yf.Ticker(ticker)
            ud = t.upgrades_downgrades
            if ud is None or ud.empty:
                continue

            ud = ud.copy()
            # Index is typically GradeDate (datetime)
            if not isinstance(ud.index, pd.DatetimeIndex):
                ud.index = pd.to_datetime(ud.index, utc=True)
            ud.index = ud.index.tz_localize(None) if ud.index.tz else ud.index

            col_lower = {c: c.lower().replace(" ", "_") for c in ud.columns}
            ud = ud.rename(columns=col_lower)

            dt_start = pd.Timestamp(start)
            dt_end = pd.Timestamp(end)
            ud = ud.loc[(ud.index >= dt_start) & (ud.index <= dt_end)]
            if not ud.empty:
                results[ticker] = ud

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(subset)}")
        time.sleep(0.20)

    _save(results, cache_name)
    print(f"      [ANALYST] done — {len(results)} tickers with data")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Institutional Holdings Snapshot (via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_institutional_holders(
    tickers: List[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load institutional holder counts and % held from yfinance.

    Returns DataFrame with tickers as index and columns:
      'institutionCount', 'pctHeldByInstitutions', 'pctHeldByInsiders'

    This is a current snapshot, not time-series. Broadcast to all dates
    in the feature builder (same as short interest).
    """
    cache_name = "institutional_holders"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [13F] loaded from cache ({len(cached)} tickers)")
            return cached

    import yfinance as yf

    print(f"      [13F] fetching institutional data for {len(tickers)} tickers...")
    records = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            records[ticker] = {
                "institutionCount": info.get("institutionCount", None),
                "pctHeldByInstitutions": info.get("heldPercentInstitutions", None),
                "pctHeldByInsiders": info.get("heldPercentInsiders", None),
            }
        except Exception:
            pass

        if i % 100 == 99:
            print(f"        {i+1}/{len(tickers)}")
            time.sleep(1)

    result = pd.DataFrame(records).T
    _save(result, cache_name)
    print(f"      [13F] Loaded for {len(result)} tickers")
    return result
