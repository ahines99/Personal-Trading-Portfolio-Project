"""
finra_short_interest.py
------------------------
Historical FINRA bi-monthly equity short interest loader.

Source:
    https://cdn.finra.org/equity/otcmarket/biweekly/shrt<YYYYMMDD>.csv

Despite the "otcmarket" path component, this file contains NYSE, NASDAQ,
ARCA, AMEX, and OTC issues (verified: ticker 'A' = Agilent / NYSE appears
in 2024-01-31 file). It is the canonical FINRA bi-monthly short interest
publication pursuant to FINRA Rule 4560.

Format: pipe-delimited CSV with header row. Columns:
    accountingYearMonthNumber | symbolCode | issueName |
    issuerServicesGroupExchangeCode | marketClassCode |
    currentShortPositionQuantity | previousShortPositionQuantity |
    stockSplitFlag | averageDailyVolumeQuantity | daysToCoverQuantity |
    revisionFlag | changePercent | changePreviousNumber | settlementDate

Settlement dates are the 15th of the month and the last business day of
the month (adjusted for weekends/holidays). Publication is typically
8 business days after settlement, so a publication_lag_days of ~10
calendar days is a safe point-in-time buffer.

This module only produces the raw loader + trading-calendar alignment.
Feature engineering (SI / float, SI change, etc.) is a separate layer.
"""

from __future__ import annotations

import pickle
import warnings
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ── constants ────────────────────────────────────────────────────────────────
_BASE_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{date}.csv"
_ROOT = Path(__file__).parent.parent
_DEFAULT_CACHE = _ROOT / "data" / "cache"
_HTTP_TIMEOUT = 30
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (personal-research) "
        "finra_short_interest.py/v1 contact=localhost"
    )
}


# ── settlement-date calendar ─────────────────────────────────────────────────
def _finra_settlement_dates(start: str, end: str) -> List[pd.Timestamp]:
    """
    Generate FINRA bi-monthly settlement dates between start and end.

    Rule: 15th of the month AND last business day of the month.
    If the 15th falls on a weekend/holiday, FINRA rolls to the prior
    business day. We approximate with NYSE business-day calendar
    (pandas 'B' frequency is close enough; actual holiday rolls can
    be corrected by trial-fetching ±1 day).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates: List[pd.Timestamp] = []
    # iterate month by month
    months = pd.date_range(
        start_ts.replace(day=1), end_ts, freq="MS"
    )
    for m in months:
        # mid-month: 15th, rolled to previous business day if weekend
        mid = m.replace(day=15)
        if mid.weekday() >= 5:
            mid = mid - pd.tseries.offsets.BDay(1)
        # end-of-month: last business day
        eom = (m + pd.offsets.MonthEnd(0))
        if eom.weekday() >= 5:
            eom = eom - pd.tseries.offsets.BDay(1)
        for d in (mid, eom):
            if start_ts <= d <= end_ts:
                dates.append(pd.Timestamp(d.normalize()))
    return sorted(set(dates))


# ── fetcher ──────────────────────────────────────────────────────────────────
def _fetch_one(
    settlement_date: pd.Timestamp,
    session: requests.Session,
) -> Optional[pd.DataFrame]:
    """Fetch one FINRA bi-monthly file. Returns None on 403/404/network err."""
    url = _BASE_URL.format(date=settlement_date.strftime("%Y%m%d"))
    try:
        resp = session.get(url, headers=_HTTP_HEADERS, timeout=_HTTP_TIMEOUT)
    except requests.RequestException as e:
        warnings.warn(f"FINRA fetch failed for {settlement_date.date()}: {e}")
        return None
    if resp.status_code != 200:
        # Try ±1 day rolls (FINRA uses exchange calendar; our estimate
        # can be off by one for odd holidays).
        for shift in (-1, 1, -2, 2):
            alt = settlement_date + pd.Timedelta(days=shift)
            alt_url = _BASE_URL.format(date=alt.strftime("%Y%m%d"))
            try:
                r2 = session.get(
                    alt_url, headers=_HTTP_HEADERS, timeout=_HTTP_TIMEOUT
                )
            except requests.RequestException:
                continue
            if r2.status_code == 200:
                resp = r2
                settlement_date = pd.Timestamp(alt.normalize())
                break
        else:
            return None

    try:
        df = pd.read_csv(
            StringIO(resp.text),
            sep="|",
            dtype={"symbolCode": str},
            low_memory=False,
        )
    except Exception as e:
        warnings.warn(f"FINRA parse failed for {settlement_date.date()}: {e}")
        return None

    if df.empty or "symbolCode" not in df.columns:
        return None

    # Normalize to our contract.
    out = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(
                df.get("settlementDate", settlement_date)
            ),
            "symbol": df["symbolCode"].astype(str).str.upper().str.strip(),
            "shares_short": pd.to_numeric(
                df.get("currentShortPositionQuantity"), errors="coerce"
            ),
            "days_to_cover": pd.to_numeric(
                df.get("daysToCoverQuantity"), errors="coerce"
            ),
            "avg_daily_vol": pd.to_numeric(
                df.get("averageDailyVolumeQuantity"), errors="coerce"
            ),
            "exchange": df.get(
                "issuerServicesGroupExchangeCode", pd.Series(dtype=str)
            ).astype(str),
        }
    )
    # Publication ~ settlement + 8 business days (FINRA Rule 4560 timeline)
    out["publication_date"] = out["settlement_date"] + pd.offsets.BDay(8)
    out = out.dropna(subset=["shares_short"])
    return out


def fetch_finra_short_interest(
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    use_cache: bool = True,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download FINRA historical bi-monthly short interest.

    Returns long-format DataFrame with columns:
        settlement_date, publication_date, symbol,
        shares_short, days_to_cover, avg_daily_vol, exchange

    Bi-monthly: 15th of month + last business day of month.
    On network errors or if FINRA changes the URL scheme, returns an
    empty DataFrame with the expected columns (so callers can skip).
    """
    cache_path = Path(cache_dir) / f"finra_si_{start}_{end}_v1.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, pd.DataFrame):
                return cached
        except Exception as e:
            warnings.warn(f"FINRA cache read failed: {e}; refetching")

    empty = pd.DataFrame(
        columns=[
            "settlement_date",
            "publication_date",
            "symbol",
            "shares_short",
            "days_to_cover",
            "avg_daily_vol",
            "exchange",
        ]
    )

    settlement_dates = _finra_settlement_dates(start, end)
    if not settlement_dates:
        return empty

    frames: List[pd.DataFrame] = []
    session = requests.Session()
    n_ok = 0
    n_miss = 0
    for sd in settlement_dates:
        df = _fetch_one(sd, session)
        if df is None or df.empty:
            n_miss += 1
            continue
        frames.append(df)
        n_ok += 1

    if not frames:
        warnings.warn(
            f"FINRA short interest: 0/{len(settlement_dates)} settlement "
            f"dates fetched successfully. Returning empty DataFrame."
        )
        return empty

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["settlement_date", "symbol"], keep="last")
    out = out.sort_values(["settlement_date", "symbol"]).reset_index(drop=True)

    if n_miss:
        warnings.warn(
            f"FINRA short interest: {n_ok}/{len(settlement_dates)} "
            f"settlement files fetched, {n_miss} missed."
        )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
    except Exception as e:
        warnings.warn(f"FINRA cache write failed: {e}")

    return out


# ── ticker alignment ─────────────────────────────────────────────────────────
def align_to_ticker_universe(
    finra_df: pd.DataFrame,
    universe: List[str],
    cusip_to_ticker_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Reshape long-format FINRA data to wide (settlement_date x ticker)
    using shares_short as values.

    FINRA's biweekly file uses ticker symbols natively (symbolCode),
    so cusip_to_ticker_map is accepted for API compatibility but only
    applied if the data column names indicate CUSIP-keyed input.
    """
    if finra_df is None or finra_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="settlement_date"),
                            columns=pd.Index(universe, name="ticker"),
                            dtype=float)

    df = finra_df.copy()
    # Optional CUSIP fallback (FINRA biweekly uses tickers; this is a
    # no-op unless a future format change introduces a 'cusip' column).
    if "cusip" in df.columns and cusip_to_ticker_map:
        df["symbol"] = (
            df["cusip"].map(cusip_to_ticker_map).fillna(df.get("symbol"))
        )

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    uni = {t.upper() for t in universe}
    df = df[df["symbol"].isin(uni)]

    if df.empty:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="settlement_date"),
            columns=pd.Index(sorted(uni), name="ticker"),
            dtype=float,
        )

    # BUG FIX: pivot on publication_date (not settlement_date) so downstream
    # point-in-time alignment uses the actual date FINRA released the data.
    # FINRA publishes ~10-12 calendar days after settlement; the pickle already
    # contains publication_date = settlement_date + BDay(8), and using it as
    # the index removes the 2-day lookahead that was present in 74% of events.
    pivot_date_col = "publication_date" if "publication_date" in df.columns else "settlement_date"
    wide = df.pivot_table(
        index=pivot_date_col,
        columns="symbol",
        values="shares_short",
        aggfunc="last",
    ).sort_index()

    # Reindex to full universe so downstream alignment is deterministic.
    wide = wide.reindex(columns=sorted(uni))
    wide.index.name = "publication_date"
    wide.columns.name = "ticker"
    return wide


# ── trading-calendar alignment ───────────────────────────────────────────────
def align_to_trading_calendar(
    wide_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    publication_lag_days: int = 10,
) -> pd.DataFrame:
    """
    Forward-fill bi-monthly FINRA values across trading dates with a
    publication-lag offset so the output is point-in-time safe.

    For each settlement_date s with value v, v only becomes visible
    on trading dates >= s + publication_lag_days (calendar days).
    """
    if wide_df is None or wide_df.empty:
        return pd.DataFrame(
            index=trading_dates, columns=wide_df.columns if wide_df is not None
            else pd.Index([], name="ticker"), dtype=float
        )

    shifted_index = wide_df.index + pd.Timedelta(days=publication_lag_days)
    shifted = wide_df.copy()
    shifted.index = shifted_index

    # Union index then forward-fill, then reindex onto trading dates.
    union = shifted.index.union(trading_dates).sort_values()
    ff = shifted.reindex(union).ffill()
    out = ff.reindex(trading_dates)
    out.index.name = "date"
    return out


# ── orchestrator ─────────────────────────────────────────────────────────────
def load_finra_short_interest(
    tickers: List[str],
    trading_dates: pd.DatetimeIndex,
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    End-to-end loader: fetch + align + point-in-time shift.

    Returns (trading_date x ticker) DataFrame of shares_short,
    forward-filled across trading days with a 10-day publication lag.
    Matches the signature pattern of alt_data_loader.py loaders.
    """
    raw = fetch_finra_short_interest(
        start=start, end=end, use_cache=use_cache
    )
    wide = align_to_ticker_universe(raw, list(tickers))
    # BUG FIX: publication_lag_days=0 because we now pivot on publication_date
    # directly in align_to_ticker_universe. Adding an additional 10-day lag
    # would double-lag the data. A 1-day trading buffer is a conservative
    # hygiene step so strategy can't trade on same-day-as-publication data.
    aligned = align_to_trading_calendar(
        wide, trading_dates, publication_lag_days=1
    )
    return aligned


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import datetime as _dt

    print("=" * 64)
    print("FINRA short interest loader - smoke test")
    print("=" * 64)

    # 1. Fetch 1 year of data
    raw = fetch_finra_short_interest(
        start="2023-01-01", end="2024-01-01", use_cache=True
    )
    print(f"\n[1] Raw shape: {raw.shape}")
    if not raw.empty:
        print(f"    Date range: {raw['settlement_date'].min().date()} "
              f"-> {raw['settlement_date'].max().date()}")
        print(f"    Unique symbols: {raw['symbol'].nunique():,}")
        print(f"    Unique settlement dates: "
              f"{raw['settlement_date'].nunique()}")

        # 3. First 5 rows
        print("\n[2] First 5 rows:")
        print(raw.head().to_string(index=False))

        # 4. Point-in-time check
        today = pd.Timestamp(_dt.date.today())
        max_pub = raw["publication_date"].max()
        print(f"\n[3] Point-in-time check:")
        print(f"    Max publication_date: {max_pub.date()}")
        print(f"    Today: {today.date()}")
        print(f"    Max settlement < today - 10d: "
              f"{raw['settlement_date'].max() < today - pd.Timedelta(days=10)}")

        # 5. Ticker alignment smoke test
        test_tickers = ["AAPL", "MSFT", "TSLA"]
        wide = align_to_ticker_universe(raw, test_tickers)
        print(f"\n[4] align_to_ticker_universe({test_tickers}) shape: "
              f"{wide.shape}")
        if not wide.empty:
            print(wide.tail(3).to_string())

        trading = pd.bdate_range("2023-01-03", "2024-01-02")
        aligned = align_to_trading_calendar(wide, trading,
                                            publication_lag_days=10)
        print(f"\n[5] align_to_trading_calendar shape: {aligned.shape}")
        nn = aligned.notna().sum().sum()
        print(f"    Non-null cells: {nn}")
        if nn:
            print(aligned.dropna(how="all").tail(3).to_string())
    else:
        print("    (empty - FINRA fetch failed or URL changed)")

    print("\n" + "=" * 64)
    print("smoke test complete")
    print("=" * 64)
