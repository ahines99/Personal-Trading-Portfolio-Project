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

# ── cache version ────────────────────────────────────────────────────────────
# _ALT_DATA_CACHE_VERSION bumped 2026-04-05 (Tier 8): forces rebuild of
# fred_macro, vix_term, analyst_actions, insider_subs, earnings_calendar,
# short_interest_snapshot, earnings_estimates, institutional_holders,
# edgar_cik_map, dxy, breakeven, vvix, oil_wti, copper_gold, cross_asset_panel,
# ig_oas, sector_oas, ebp, treasury_yield_curve caches after Tier 1-6 feature
# set changes. EDGAR fundamentals already have _v2raw suffix and are untouched.
# On next run, these caches will be rebuilt from upstream APIs (one-time cost).
_ALT_DATA_CACHE_VERSION = "v2"


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
    """
    Download SEC ticker→CIK mapping (cached 7 days).

    Combines two SEC sources to improve coverage for delisted tickers:
      1. company_tickers.json          — currently-listed firms (primary)
      2. company_tickers_exchange.json — broader, sometimes includes former
                                         tickers / additional exchange listings
    Both sources are still survivorship-biased (SEC does not publish a full
    historical former-ticker archive here); truly delisted tickers may still
    fail lookup. Per-CIK former tickers live under
    https://data.sec.gov/submissions/CIK{cik}.json which would require
    per-ticker crawling — not done here to keep rate-limit footprint small.
    """
    cache_name = f"edgar_cik_map_{_ALT_DATA_CACHE_VERSION}"
    cached = _load(cache_name)
    if cached is not None and (time.time() - cached["ts"]) < 7 * 86400:
        return cached["data"]

    mapping: Dict[str, str] = {}

    # Primary: currently-listed company tickers
    url_primary = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url_primary, headers=_EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        for v in raw.values():
            t = str(v.get("ticker", "")).upper()
            cik = v.get("cik_str")
            if t and cik is not None:
                mapping[t] = str(cik).zfill(10)
    except Exception as e:
        warnings.warn(f"EDGAR primary CIK map download failed: {e}")

    # Supplementary: exchange-listed tickers (broader coverage)
    url_exchange = "https://www.sec.gov/files/company_tickers_exchange.json"
    try:
        resp = requests.get(url_exchange, headers=_EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        # format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
        fields = raw.get("fields", [])
        data = raw.get("data", [])
        if fields and data:
            try:
                i_cik = fields.index("cik")
                i_tic = fields.index("ticker")
                for row in data:
                    t = str(row[i_tic] or "").upper()
                    cik = row[i_cik]
                    if t and cik is not None and t not in mapping:
                        mapping[t] = str(cik).zfill(10)
            except ValueError:
                pass
    except Exception as e:
        warnings.warn(f"EDGAR exchange CIK map download failed: {e}")

    if mapping:
        _save({"ts": time.time(), "data": mapping}, cache_name)
        return mapping

    # Total failure — fall back to previously cached map if available
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


def _extract_annual_series(
    facts: dict,
    concept: str,
    taxonomy: str = "us-gaap",
    include_quarterly: bool = True,
) -> pd.Series:
    """
    Extract fundamental filings for a concept into a date-indexed Series.
    Uses 'filed' date (point-in-time) not period end date to avoid look-ahead.

    By default now includes 10-Q (quarterly) in addition to 10-K (annual),
    giving ~4x the data resolution and much fresher signals. Set
    include_quarterly=False to restore annual-only behavior.

    CAVEAT — 10-Q data is PARTIAL-YEAR year-to-date, not standalone-quarter:
      - Q1 10-Q  = 3 months YTD
      - Q2 10-Q  = 6 months YTD
      - Q3 10-Q  = 9 months YTD
      - Q4       = 10-K (12 months FY)
    Flow concepts (Revenues, NetIncomeLoss, GrossProfit, OCF, InterestExpense,
    SGA) are YTD-accumulated within a fiscal year, so raw diff(252) in
    downstream features compares mixed periods and should be replaced by
    same-quarter YoY diffs (Q3 2024 vs Q3 2023) for flow items.
    Stock concepts (Assets, Liabilities, Equity, Shares, Cash, LTDebt,
    AssetsCurrent, LiabilitiesCurrent, EPS) are point-in-time balances and
    not affected by the YTD accumulation issue.
    TODO: downstream flow-factor features should switch to same-quarter YoY.
    """
    allowed_forms = ("10-K", "10-K/A")
    if include_quarterly:
        allowed_forms = allowed_forms + ("10-Q", "10-Q/A")
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        # prefer USD, fall back to shares for EPS concepts
        unit_key = "USD" if "USD" in units else list(units.keys())[0]
        rows = units[unit_key]
        records = []
        for r in rows:
            if r.get("form") in allowed_forms and "filed" in r and "val" in r:
                records.append({"filed": pd.to_datetime(r["filed"]), "val": r["val"]})
        if not records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(records).sort_values("filed").drop_duplicates("filed", keep="last")
        return df.set_index("filed")["val"]
    except (KeyError, IndexError):
        return pd.Series(dtype=float)


def _extract_annual_series_multi(
    facts: dict,
    concepts: List[str],
    taxonomy: str = "us-gaap",
    include_quarterly: bool = True,
) -> pd.Series:
    """Try multiple XBRL concepts in order; return first non-empty series.

    Useful when companies report under different tag names
    (e.g. DeferredRevenueCurrent vs ContractWithCustomerLiabilityCurrent under ASC 606).
    """
    for c in concepts:
        s = _extract_annual_series(facts, c, taxonomy, include_quarterly)
        if not s.empty:
            return s
    return pd.Series(dtype=float)


def load_edgar_fundamentals_extra(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    max_tickers: int = 10_000,
    sleep_sec: float = 0.12,
) -> pd.DataFrame:
    """Incremental EDGAR loader for Phase F C&Z signal raw fields.

    Loads ONLY the 8 new XBRL concepts needed for the C&Z accounting signals
    (XFIN, PayoutYield, NetPayoutYield, OperProfRD, Tax, cfp, DelDRC).
    Cached separately from main EDGAR fundamentals so adding these doesn't
    invalidate the 1.9GB main cache.

    Returns DataFrame with MultiIndex columns (ticker, raw_*) and DatetimeIndex.
    """
    cache_name = f"edgar_fundamentals_extra_{start}_{end}_{max_tickers}_v1"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [EDGAR-EXTRA] loaded from cache ({cached.shape})")
            return cached

    print(f"      [EDGAR-EXTRA] fetching extra XBRL fields for up to {max_tickers} tickers...")
    cik_map = _get_cik_map(tickers)
    date_idx = pd.bdate_range(start, end)
    results = {}

    subset = [t for t in tickers if t in cik_map][:max_tickers]
    n = len(subset)
    print(f"      [EDGAR-EXTRA] processing {n} tickers (~{n*sleep_sec/60:.0f} min)")

    for i, ticker in enumerate(subset):
        cik = cik_map[ticker]
        facts = _fetch_company_facts(cik)
        if facts is None:
            continue

        # Cash flow statement items
        dividends_paid = _extract_annual_series_multi(facts, [
            "PaymentsOfDividends",
            "PaymentsOfDividendsCommonStock",
        ])
        buybacks = _extract_annual_series_multi(facts, [
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfEquity",
        ])
        stock_iss = _extract_annual_series_multi(facts, [
            "ProceedsFromIssuanceOfCommonStock",
            "ProceedsFromIssuanceOfEquityCommon",
        ])
        debt_iss = _extract_annual_series_multi(facts, [
            "ProceedsFromIssuanceOfLongTermDebt",
            "ProceedsFromIssuanceOfDebt",
        ])
        debt_repay = _extract_annual_series_multi(facts, [
            "RepaymentsOfLongTermDebt",
            "RepaymentsOfDebt",
        ])

        # Balance sheet
        deferred_rev = _extract_annual_series_multi(facts, [
            "ContractWithCustomerLiabilityCurrent",  # ASC 606 (post-2018)
            "DeferredRevenueCurrent",                 # legacy
            "DeferredRevenue",
        ])

        # Income statement
        rd_expense = _extract_annual_series(facts, "ResearchAndDevelopmentExpense")
        tax_expense = _extract_annual_series_multi(facts, [
            "IncomeTaxExpenseBenefit",
            "IncomeTaxesPaid",
            "IncomeTaxesPaidNet",
        ])

        ticker_df = pd.DataFrame(index=date_idx)

        def _daily(series: pd.Series, shift: int = 1) -> pd.Series:
            if series.empty:
                return pd.Series(np.nan, index=date_idx)
            return series.reindex(date_idx, method="ffill").shift(shift)

        ticker_df["raw_dividends_paid"]    = _daily(dividends_paid)
        ticker_df["raw_buybacks"]          = _daily(buybacks)
        ticker_df["raw_stock_issuance"]    = _daily(stock_iss)
        ticker_df["raw_lt_debt_issuance"]  = _daily(debt_iss)
        ticker_df["raw_lt_debt_repayment"] = _daily(debt_repay)
        ticker_df["raw_deferred_revenue"]  = _daily(deferred_rev)
        ticker_df["raw_rd"]                = _daily(rd_expense)
        ticker_df["raw_taxes"]             = _daily(tax_expense)

        results[ticker] = ticker_df
        time.sleep(sleep_sec)

        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{n} tickers fetched")

    if not results:
        print("      [EDGAR-EXTRA] no data retrieved")
        return pd.DataFrame(index=date_idx)

    out = pd.concat(results, axis=1)
    out.index = pd.DatetimeIndex(out.index)
    _save(out, cache_name)
    print(f"      [EDGAR-EXTRA] done — {out.shape}")
    return out


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
    cache_name = f"edgar_fundamentals_{start}_{end}_{max_tickers}_v2raw"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            print(f"      [EDGAR] loaded from cache ({cached.shape})")
            return cached

    print(f"      [EDGAR] fetching fundamentals for up to {max_tickers} tickers...")
    cik_map   = _get_cik_map(tickers)
    date_idx  = pd.bdate_range(start, end)
    results   = {}

    missing = [t for t in tickers if t not in cik_map]
    if missing:
        print(
            f"      [EDGAR] Warning: {len(missing)} tickers have no CIK mapping "
            f"(likely delisted / non-SEC registrants — no fundamentals available)"
        )
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

        # ── Raw fields (exposed for downstream Tier 3B signals) ──────────
        # Expose raw daily-aligned underlying values so downstream feature
        # builders can compose new ratios (value composites, Piotroski
        # F-Score, cash-based operating profitability, etc.).
        ticker_df["raw_assets"]             = ast_d
        ticker_df["raw_liabilities"]        = lb_d
        ticker_df["raw_equity"]             = eq_d
        ticker_df["raw_net_income"]         = ni_d
        ticker_df["raw_gross_profit"]       = gp_d
        ticker_df["raw_revenue"]            = rev_d
        ticker_df["raw_eps"]                = eps_d
        ticker_df["raw_operating_cf"]       = ocf_d
        ticker_df["raw_cash"]               = cash_d
        ticker_df["raw_lt_debt"]            = debt_d
        ticker_df["raw_shares_out"]         = shr_d
        ticker_df["raw_sga"]                = sga_d
        ticker_df["raw_interest_expense"]   = int_d
        ticker_df["raw_current_assets"]     = ca_d
        ticker_df["raw_current_liabilities"] = cl_d

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
    cache_name = f"fred_macro_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"vix_term_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"short_interest_snapshot_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"earnings_calendar_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
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

            # Add before_after_market field (yfinance doesn't provide; default unknown)
            # so downstream consumers expecting this field don't crash
            if "before_after_market" not in earn.columns:
                earn["before_after_market"] = "unknown"

            df_t = earn.set_index("earnings_date").sort_index()
            # FIX (2026-04-16): dedup duplicate dates to prevent downstream reindex errors
            # in build_earnings_signals (streak_ts.reindex). yfinance occasionally returns
            # duplicate earnings dates (same announcement listed twice with different metadata).
            if df_t.index.duplicated().any():
                df_t["_nan_count"] = df_t.isna().sum(axis=1)
                df_t = df_t.sort_values(["_nan_count"], ascending=True)
                df_t = df_t[~df_t.index.duplicated(keep="first")]
                df_t = df_t.drop(columns=["_nan_count"]).sort_index()
            results[ticker] = df_t

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(tickers)}")
        time.sleep(0.20)

    _save(results, cache_name)
    print(f"      [EARN] done — {len(results)} tickers with data")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Earnings calendar via EODHD bulk endpoint (replaces yfinance scraper)
# ─────────────────────────────────────────────────────────────────────────────

def load_earnings_calendar_eodhd(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2026-04-12",
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Bulk earnings calendar via EODHD /api/calendar/earnings.

    Returns same contract as load_earnings_calendar():
        Dict[ticker -> DataFrame] with index=earnings_date and columns:
        eps_estimate, eps_actual, eps_surprise_pct

    Advantages over yfinance:
    - 25+ years of history (vs yfinance's ~8 quarters)
    - Single bulk API call (1 API call cost) vs ticker-by-ticker scraping
    - Stable endpoint (vs yfinance schema breakage)
    """
    import os
    api_key = os.environ.get("EODHD_API_KEY", "")
    if not api_key:
        warnings.warn("[EARN-EODHD] no EODHD_API_KEY, falling back to yfinance")
        return load_earnings_calendar(tickers, start=start, end=end, use_cache=use_cache)

    cache_name = f"earnings_calendar_eodhd_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            # MIGRATION (2026-04-17): older caches were built with a parser that
            # silently always wrote "unknown" into `before_after_market` (camelCase
            # vs snake_case mismatch against EODHD response). Detect those caches
            # and force a re-fetch so the Phase D EAR-timing fix actually activates.
            needs_refetch = False
            if not isinstance(cached, dict) or len(cached) == 0:
                needs_refetch = False  # nothing to validate
            else:
                missing_col = 0
                all_unknown = 0
                sampled = 0
                for _t, _df in cached.items():
                    if not isinstance(_df, pd.DataFrame) or _df.empty:
                        continue
                    sampled += 1
                    if "before_after_market" not in _df.columns:
                        missing_col += 1
                        continue
                    vals = _df["before_after_market"].astype(str).str.lower().str.strip()
                    non_unknown = vals[~vals.isin(["unknown", "nan", "none", ""])]
                    if len(non_unknown) == 0:
                        all_unknown += 1
                    if sampled >= 50:  # sample is enough to decide
                        break
                if sampled > 0 and (missing_col + all_unknown) >= max(1, int(0.9 * sampled)):
                    needs_refetch = True
                    warnings.warn(
                        "[EARN-EODHD] cache appears to predate the "
                        "before_after_market fix — every sampled ticker has the field "
                        "missing or always 'unknown'. Forcing one-time re-fetch from "
                        "EODHD so the Phase D EAR-timing window activates. "
                        "(Cache file: " + str(_cache_path(cache_name)) + ")"
                    )
            if not needs_refetch:
                print(f"      [EARN-EODHD] loaded from cache ({len(cached)} tickers)")
                return cached

    print(f"      [EARN-EODHD] fetching bulk earnings {start} -> {end}...")

    url = "https://eodhd.com/api/calendar/earnings"
    all_records = []

    # Fetch in 6-month chunks to stay within response limits
    # FIX (2026-04-16): advance chunk_start by 1 day to prevent boundary overlap
    # (EODHD treats both `from` and `to` as inclusive, double-counting events on boundaries)
    chunk_start = pd.Timestamp(start)
    chunk_end = pd.Timestamp(end)
    while chunk_start < chunk_end:
        chunk_stop = min(chunk_start + pd.DateOffset(months=6), chunk_end)
        params = {
            "api_token": api_key,
            "from": str(chunk_start.date()),
            "to": str(chunk_stop.date()),
            "fmt": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                earnings = data.get("earnings", []) if isinstance(data, dict) else data
                all_records.extend(earnings)
            else:
                warnings.warn(f"[EARN-EODHD] HTTP {resp.status_code} for {chunk_start.date()}")
        except Exception as e:
            warnings.warn(f"[EARN-EODHD] request failed: {e}")
        # +1 day prevents boundary-day double-count (was: chunk_start = chunk_stop)
        chunk_start = chunk_stop + pd.Timedelta(days=1)
        time.sleep(0.1)

    if not all_records:
        warnings.warn("[EARN-EODHD] no records returned, falling back to yfinance")
        return load_earnings_calendar(tickers, start=start, end=end, use_cache=use_cache)

    print(f"      [EARN-EODHD] {len(all_records)} raw records fetched")

    # Parse into per-ticker DataFrames matching yfinance contract
    ticker_set = {t.upper() for t in tickers}
    results: Dict[str, pd.DataFrame] = {}

    for rec in all_records:
        code = rec.get("code", "")
        # EODHD returns "AAPL.US" format — strip exchange suffix
        ticker = code.split(".")[0].upper() if "." in code else code.upper()
        if ticker not in ticker_set:
            continue

        report_date = rec.get("report_date")
        if not report_date:
            continue

        eps_est = rec.get("estimate")
        eps_act = rec.get("actual")
        eps_surprise_pct = rec.get("percent")

        # Compute surprise_pct if not provided but estimate + actual exist
        if eps_surprise_pct is None and eps_est is not None and eps_act is not None:
            try:
                est_f = float(eps_est)
                act_f = float(eps_act)
                if abs(est_f) > 1e-9:
                    eps_surprise_pct = ((act_f - est_f) / abs(est_f)) * 100
            except (TypeError, ValueError):
                pass

        row = {
            "earnings_date": pd.to_datetime(report_date, errors="coerce"),
            "eps_estimate": _safe_float(eps_est),
            "eps_actual": _safe_float(eps_act),
            "eps_surprise_pct": np.clip(_safe_float(eps_surprise_pct) / 100, -2, 2)
            if eps_surprise_pct is not None else np.nan,
            # FIX (2026-04-17): EODHD API response uses snake_case
            # `before_after_market` exclusively (verified via test_eodhd_earnings_api.py).
            # Earlier code defensively also tried a camelCase variant which never
            # exists in the response — removed to prevent confusion. The Phase D
            # EAR-timing fix in alt_features.py keys off this field; if it is missing
            # or always "unknown" the fix silently degrades to the default window.
            "before_after_market": str(
                rec.get("before_after_market", "unknown")
            ).lower().strip(),
        }

        if ticker not in results:
            results[ticker] = []
        results[ticker].append(row)

    # Convert lists to DataFrames + DEDUPLICATE
    # FIX (2026-04-16): EODHD calendar returns spurious fiscal-quarter-end markers
    # (rows with eps_estimate=NaN) alongside real earnings events. These doubled
    # AAPL's count from expected 52 to actual 81. Solution:
    #   1. Drop rows with no eps_estimate (the spurious markers)
    #   2. Then dedup by date keeping row with most complete data
    final: Dict[str, pd.DataFrame] = {}
    for ticker, rows in results.items():
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["earnings_date"])
        # Drop spurious quarter-end markers (no estimate AND no actual = not a real event)
        valid_mask = df["eps_estimate"].notna() | df["eps_actual"].notna()
        df = df[valid_mask]
        df = df.set_index("earnings_date").sort_index()
        # When duplicates remain on same date, keep row with fewest NaNs (most complete)
        if df.index.duplicated().any():
            df["_nan_count"] = df.isna().sum(axis=1)
            df = df.sort_values(["_nan_count"], ascending=True)  # complete rows first
            df = df[~df.index.duplicated(keep="first")]
            df = df.drop(columns=["_nan_count"]).sort_index()
        final[ticker] = df

    _save(final, cache_name)
    print(f"      [EARN-EODHD] done — {len(final)} tickers with data")
    return final


def _safe_float(val) -> float:
    """Safely convert to float, returning NaN on failure."""
    if val is None or val == "":
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


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
    cache_name = f"insider_subs_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"earnings_estimates_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"analyst_actions_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
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
    cache_name = f"institutional_holders_{_ALT_DATA_CACHE_VERSION}"
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


# ─────────────────────────────────────────────────────────────────────────────
# 10+. Additional macro / cross-asset loaders (Tier 6+)
# ─────────────────────────────────────────────────────────────────────────────

def _fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Low-level helper: fetch one FRED series as date-indexed float Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    s = df.iloc[:, 0].replace(".", np.nan).astype(float)
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


def _yf_close(ticker: str, start: str, end: str) -> pd.Series:
    """Low-level helper: fetch yfinance Close as 1-D Series."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.Series(dtype=float)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        col = df["Close"]
        if hasattr(col, "columns"):
            col = col.iloc[:, 0]
        col.index = pd.DatetimeIndex(col.index).tz_localize(None) if col.index.tz else pd.DatetimeIndex(col.index)
        return col.astype(float)
    except Exception:
        return pd.Series(dtype=float)


def load_dxy(start: str = "2013-01-01", end: str = "2026-01-01",
             use_cache: bool = True) -> pd.DataFrame:
    """Broad trade-weighted dollar (FRED DTWEXBGS). Returns DataFrame col='dxy'."""
    cache_name = f"dxy_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    try:
        s = _fred_series("DTWEXBGS", start, end)
        s = s.reindex(date_idx, method="ffill").shift(1)
    except Exception as e:
        warnings.warn(f"DXY load failed: {e}")
        s = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame({"dxy": s}, index=date_idx)
    _save(out, cache_name)
    return out


def load_breakeven_inflation(start: str = "2013-01-01", end: str = "2026-01-01",
                             use_cache: bool = True) -> pd.DataFrame:
    """10-yr breakeven inflation (FRED T10YIE). Returns DataFrame col='breakeven'."""
    cache_name = f"breakeven_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    try:
        s = _fred_series("T10YIE", start, end)
        s = s.reindex(date_idx, method="ffill").shift(1)
    except Exception as e:
        warnings.warn(f"Breakeven load failed: {e}")
        s = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame({"breakeven": s}, index=date_idx)
    _save(out, cache_name)
    return out


def load_vvix(start: str = "2013-01-01", end: str = "2026-01-01",
              use_cache: bool = True) -> pd.DataFrame:
    """VVIX (vol of VIX). Try yfinance ^VVIX, fallback to CBOE CSV.
    Returns DataFrame col='vvix'."""
    cache_name = f"vvix_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    s = _yf_close("^VVIX", start, end)
    if s is None or s.empty:
        try:
            url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv"
            df = pd.read_csv(url)
            df.columns = [c.strip().upper() for c in df.columns]
            date_col = "DATE" if "DATE" in df.columns else df.columns[0]
            close_col = "VVIX" if "VVIX" in df.columns else ("CLOSE" if "CLOSE" in df.columns else df.columns[-1])
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[close_col].astype(float).sort_index()
        except Exception as e:
            warnings.warn(f"VVIX CBOE fallback failed: {e}")
            s = pd.Series(dtype=float)
    if s is None or s.empty:
        s = pd.Series(np.nan, index=date_idx)
    else:
        s = s.reindex(date_idx, method="ffill").shift(1)
    out = pd.DataFrame({"vvix": s}, index=date_idx)
    _save(out, cache_name)
    return out


def load_oil_wti(start: str = "2013-01-01", end: str = "2026-01-01",
                 use_cache: bool = True) -> pd.DataFrame:
    """WTI crude (FRED DCOILWTICO). Returns DataFrame col='oil_wti'."""
    cache_name = f"oil_wti_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    try:
        s = _fred_series("DCOILWTICO", start, end)
        s = s.reindex(date_idx, method="ffill").shift(1)
    except Exception as e:
        warnings.warn(f"Oil WTI load failed: {e}")
        s = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame({"oil_wti": s}, index=date_idx)
    _save(out, cache_name)
    return out


def load_copper_gold(start: str = "2013-01-01", end: str = "2026-01-01",
                     use_cache: bool = True) -> pd.DataFrame:
    """Copper (HG=F) and Gold (GC=F) via yfinance. Returns DataFrame
    with cols: copper, gold, cu_au_ratio."""
    cache_name = f"copper_gold_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    cu = _yf_close("HG=F", start, end)
    au = _yf_close("GC=F", start, end)
    cu_d = cu.reindex(date_idx, method="ffill").shift(1) if not cu.empty else pd.Series(np.nan, index=date_idx)
    au_d = au.reindex(date_idx, method="ffill").shift(1) if not au.empty else pd.Series(np.nan, index=date_idx)
    ratio = cu_d / au_d.replace(0, np.nan)
    out = pd.DataFrame({"copper": cu_d, "gold": au_d, "cu_au_ratio": ratio}, index=date_idx)
    _save(out, cache_name)
    return out


def load_cross_asset_etf_panel(start: str = "2013-01-01", end: str = "2026-01-01",
                               use_cache: bool = True) -> pd.DataFrame:
    """Six-ETF cross-asset panel (SPY, TLT, DBC, UUP, GLD, HYG). Returns a
    DataFrame with a 'close_<ticker>' column for each and 'mom12_1_<ticker>'
    (12-1 momentum: 252d return minus 21d return)."""
    cache_name = f"cross_asset_panel_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    etfs = ["SPY", "TLT", "DBC", "UUP", "GLD", "HYG"]
    cols = {}
    for etf in etfs:
        s = _yf_close(etf, start, end)
        if s is None or s.empty:
            cols[f"close_{etf}"] = pd.Series(np.nan, index=date_idx)
            cols[f"mom12_1_{etf}"] = pd.Series(np.nan, index=date_idx)
            continue
        s = s.reindex(date_idx, method="ffill").shift(1)
        cols[f"close_{etf}"] = s
        # 12-1 momentum: (P_{t-21}/P_{t-252}) - 1
        mom = (s.shift(21) / s.shift(252) - 1.0)
        cols[f"mom12_1_{etf}"] = mom
    out = pd.DataFrame(cols, index=date_idx)
    _save(out, cache_name)
    return out


def load_ig_oas(start: str = "2013-01-01", end: str = "2026-01-01",
                use_cache: bool = True) -> pd.DataFrame:
    """Investment grade OAS (FRED BAMLC0A0CM). Returns DataFrame col='ig_oas'."""
    cache_name = f"ig_oas_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    try:
        s = _fred_series("BAMLC0A0CM", start, end)
        s = s.reindex(date_idx, method="ffill").shift(1)
    except Exception as e:
        warnings.warn(f"IG OAS load failed: {e}")
        s = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame({"ig_oas": s}, index=date_idx)
    _save(out, cache_name)
    return out


def load_sector_oas(start: str = "2013-01-01", end: str = "2026-01-01",
                    use_cache: bool = True) -> pd.DataFrame:
    """Rating-bucket OAS series from FRED. Returns DataFrame with columns
    BBB, BB, B, CCC (from BAMLC0A4CBBB, BAMLH0A1HYBB, BAMLH0A2HYB, BAMLH0A3HYC)."""
    cache_name = f"sector_oas_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    mapping = {
        "BBB": "BAMLC0A4CBBB",
        "BB":  "BAMLH0A1HYBB",
        "B":   "BAMLH0A2HYB",
        "CCC": "BAMLH0A3HYC",
    }
    frames = {}
    for name, sid in mapping.items():
        try:
            s = _fred_series(sid, start, end)
            frames[name] = s.reindex(date_idx, method="ffill").shift(1)
        except Exception as e:
            warnings.warn(f"sector OAS {sid} load failed: {e}")
            frames[name] = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame(frames, index=date_idx)
    _save(out, cache_name)
    return out


def load_excess_bond_premium(start: str = "2013-01-01", end: str = "2026-01-01",
                             use_cache: bool = True,
                             cache_days: int = 30) -> pd.DataFrame:
    """Federal Reserve Excess Bond Premium (Gilchrist-Zakrajsek 2012).
    Downloads ebp_csv.csv and returns DataFrame with columns
    'ebp' and 'gz_spread' at daily bd frequency (monthly values ffilled).
    Cache auto-refreshes every ``cache_days`` days."""
    cache_name = f"ebp_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            ts = cached.get("ts") if isinstance(cached, dict) else None
            if ts is not None and (time.time() - ts) < cache_days * 86400:
                return cached["data"]
    date_idx = pd.bdate_range(start, end)
    try:
        url = "https://www.federalreserve.gov/data/ebp_csv.csv"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = None
        for c in df.columns:
            if "date" in c:
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        # find ebp and gz_spread columns
        ebp_col = next((c for c in df.columns if "ebp" in c), None)
        gz_col = next((c for c in df.columns if "gz" in c), None)
        out_cols = {}
        if ebp_col:
            out_cols["ebp"] = pd.to_numeric(df[ebp_col], errors="coerce")
        if gz_col:
            out_cols["gz_spread"] = pd.to_numeric(df[gz_col], errors="coerce")
        monthly = pd.DataFrame(out_cols)
        # reindex to daily bd, ffill monthly values, shift 1 for point-in-time
        daily = monthly.reindex(date_idx, method="ffill").shift(1)
    except Exception as e:
        warnings.warn(f"EBP load failed: {e}")
        daily = pd.DataFrame(
            {"ebp": np.nan, "gz_spread": np.nan}, index=date_idx
        )
    _save({"ts": time.time(), "data": daily}, cache_name)
    return daily


def load_treasury_yield_curve(start: str = "2013-01-01", end: str = "2026-01-01",
                              use_cache: bool = True) -> pd.DataFrame:
    """Full Treasury yield curve from FRED (DGS1MO, DGS3MO, DGS6MO, DGS1,
    DGS2, DGS5, DGS10, DGS30). Returns DataFrame with these columns at bd freq."""
    cache_name = f"treasury_yield_curve_{start}_{end}_{_ALT_DATA_CACHE_VERSION}"
    if use_cache:
        cached = _load(cache_name)
        if cached is not None:
            return cached
    date_idx = pd.bdate_range(start, end)
    series_ids = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    frames = {}
    for sid in series_ids:
        try:
            s = _fred_series(sid, start, end)
            frames[sid] = s.reindex(date_idx, method="ffill").shift(1)
        except Exception as e:
            warnings.warn(f"Treasury {sid} load failed: {e}")
            frames[sid] = pd.Series(np.nan, index=date_idx)
    out = pd.DataFrame(frames, index=date_idx)
    _save(out, cache_name)
    return out
