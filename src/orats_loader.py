"""
orats_loader.py
---------------
Orats Historical Options Data loader.

Two delivery modes (Orats supports both):
  1. API mode — REST calls to docs.orats.io endpoints (Live Data API tier)
  2. FTP/CSV mode — bulk file download (Historical Data product, $399 one-time)

This module supports both. After data is loaded, it's cached as parquet
for fast reload during backtests.

After cancellation: per Orats §6.3, the data remains usable forever for
internal/personal use.

Usage:
    # API mode (during 30-day Orats subscription):
    loader = OratsLoader(api_token=os.environ["ORATS_TOKEN"])
    loader.bulk_download_historical(
        tickers=our_universe,
        start_date="2013-01-01",
        end_date="2025-12-31",
    )

    # FTP/CSV mode (after one-time bulk purchase):
    loader = OratsLoader(ftp_dir="/path/to/orats_csv_dump/")
    loader.import_csv_dump()

    # After load, get the unified IV panel:
    panel = loader.load_iv_panel()
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


ORATS_API_BASE = "https://api.orats.io/datav2"

# Cache structure:
#   data/cache/options/orats_raw/{ticker}_{year}.parquet  — raw downloads
#   data/cache/options/iv_panel.parquet                    — combined cross-ticker IV panel
DEFAULT_CACHE = Path("data/cache/options")


class OratsLoader:
    """Orats historical options data loader (API or FTP/CSV)."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        ftp_dir: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            api_token: Orats API token (for Live Data API tier $199/mo)
            ftp_dir: Path to bulk CSV dump (for one-time Historical Data $399)
            cache_dir: Where to store parquet cache (default: data/cache/options/)
        """
        self.api_token = api_token or os.environ.get("ORATS_TOKEN")
        self.ftp_dir = Path(ftp_dir) if ftp_dir else None
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "orats_raw").mkdir(exist_ok=True)

        if not self.api_token and not self.ftp_dir:
            raise ValueError(
                "Provide either api_token (env ORATS_TOKEN) or ftp_dir "
                "(path to bulk CSV files)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # API mode: download daily summaries from Orats REST endpoints
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_ticker_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch Orats Core SMV summary history for one ticker.

        Endpoint: /datav2/hist/cores
        Returns daily records (verified field names from real /cores API response):
            tradeDate, ticker, pxAtmIv, iv10d/iv20d/iv30d/iv60d/iv90d/iv6m/iv1yr,
            atmIvM1-M4, dlt5Iv30d/dlt25Iv30d/dlt75Iv30d/dlt95Iv30d, slope, deriv,
            cVolu, pVolu, cOi, pOi, borrow30, annIdiv, annActDiv, etfSlopeRatio, etc.
        Does NOT include separate cVol/pVol (call vs put ATM IV) — only combined iv30d.
        """
        if not self.api_token:
            raise RuntimeError("API mode requires api_token")

        cache_path = self.cache_dir / "orats_raw" / f"{ticker}_{start_date}_{end_date}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        url = f"{ORATS_API_BASE}/hist/cores"
        params = {
            "token": self.api_token,
            "ticker": ticker,
            "tradeDate": f"{start_date},{end_date}",  # Orats date range syntax
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data or not data["data"]:
                return pd.DataFrame()
            df = pd.DataFrame(data["data"])
            df["tradeDate"] = pd.to_datetime(df["tradeDate"])
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as e:
            print(f"  [orats] {ticker} failed: {e}")
            return pd.DataFrame()

    def fetch_batch_history(
        self,
        tickers: List[str],
        endpoint: str = "cores",
        start_date: str = "2013-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Fetch one batch of up to 10 tickers in a single API call (Orats max).

        Multi-ticker support verified for: cores, summaries, ivrank, dailies, hvs.
        Comma-delimited tickers per Orats API docs.

        Returns combined DataFrame with all tickers' history.
        """
        if not self.api_token:
            raise RuntimeError("API mode requires api_token")
        if len(tickers) > 10:
            raise ValueError(f"Max 10 tickers per Orats batch call, got {len(tickers)}")

        # Cache by hash of ticker batch + endpoint + dates (atomic write)
        batch_key = "_".join(sorted(tickers))[:80]  # truncate for filesystem
        cache_path = (self.cache_dir / "orats_raw" /
                      f"{endpoint}_{batch_key}_{start_date}_{end_date}.parquet")
        tmp_path = cache_path.with_suffix(".parquet.tmp")
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        url = f"{ORATS_API_BASE}/hist/{endpoint}"
        # NOTE: /hist/* endpoints return FULL history when called with just ticker.
        # Adding tradeDate range parameter causes 404. We filter dates client-side.
        params = {
            "token": self.api_token,
            "ticker": ",".join(tickers),
        }
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data or not data["data"]:
                return pd.DataFrame()
            df = pd.DataFrame(data["data"])
            if "tradeDate" in df.columns:
                df["tradeDate"] = pd.to_datetime(df["tradeDate"])
                # Filter to requested date range client-side
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date)
                df = df[(df["tradeDate"] >= start_dt) & (df["tradeDate"] <= end_dt)]
            elif "earnDate" in df.columns:  # earnings endpoint
                df["earnDate"] = pd.to_datetime(df["earnDate"], errors="coerce")
            # Coerce mixed-type columns to string to allow parquet write
            for col in df.columns:
                if df[col].dtype == "object":
                    # Try numeric first, fall back to string
                    try:
                        df[col] = pd.to_numeric(df[col], errors="raise")
                    except (ValueError, TypeError):
                        df[col] = df[col].astype(str)
            # Atomic write
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(cache_path)
            return df
        except Exception as e:
            print(f"  [orats] batch {tickers[:3]}... failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return pd.DataFrame()

    def bulk_download_historical(
        self,
        tickers: List[str],
        start_date: str = "2013-01-01",
        end_date: str = "2025-12-31",
        sleep_sec: float = 0.6,  # ~100 calls/min, well under rate limit
        max_workers: int = 1,
        endpoint: str = "cores",
        batch_size: int = 10,  # Orats API max
        request_budget: int = 18000,  # safety: leave 2K buffer of 20K monthly
    ) -> Dict[str, pd.DataFrame]:
        """Download history for all tickers using multi-ticker batching.

        With batch_size=10, 1500 tickers = 150 calls per endpoint (vs 1500 single).
        Tracks request budget to avoid exceeding 20K monthly limit.
        """
        if not self.api_token:
            raise RuntimeError("API mode requires api_token")

        results = {}
        n = len(tickers)
        n_batches = (n + batch_size - 1) // batch_size
        t0 = time.time()
        api_calls_made = 0

        for b_idx in range(n_batches):
            if api_calls_made >= request_budget:
                print(f"  [orats] STOPPING: budget exhausted ({api_calls_made}/{request_budget})")
                break
            batch = tickers[b_idx * batch_size : (b_idx + 1) * batch_size]
            df = self.fetch_batch_history(batch, endpoint, start_date, end_date)
            api_calls_made += 1
            if not df.empty and "ticker" in df.columns:
                for ticker, group in df.groupby("ticker"):
                    results[ticker] = group.copy()
            time.sleep(sleep_sec)
            if (b_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (b_idx + 1) * (n_batches - b_idx - 1)
                print(f"  [orats] batch {b_idx+1}/{n_batches} done "
                      f"({api_calls_made} calls, {elapsed:.0f}s, ~{eta:.0f}s left)")

        # Backwards-compatible: also iterate per-ticker if endpoint doesn't support batching
        # (e.g., divs, earnings, splits)
        if endpoint in ("divs", "earnings", "splits"):
            print(f"  [orats] {endpoint} doesn't support batching — per-ticker mode")
            for i, ticker in enumerate(tickers):
                if api_calls_made >= request_budget:
                    print(f"  [orats] STOPPING: budget exhausted")
                    break
                df = self.fetch_ticker_history(ticker, start_date, end_date)
                api_calls_made += 1
                if not df.empty:
                    results[ticker] = df
                time.sleep(sleep_sec)
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (n - i - 1)
                    print(f"  [orats] {i+1}/{n} downloaded ({elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining)")
        print(f"  [orats] Done: {len(results)}/{n} tickers, {time.time()-t0:.0f}s")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # FTP/CSV mode: import a bulk CSV dump
    # ─────────────────────────────────────────────────────────────────────────

    def import_csv_dump(self, pattern: str = "*.csv") -> int:
        """Import bulk Orats CSV dump (one file per day or per year).
        Saves to per-ticker parquet cache. Returns count of records imported.
        """
        if not self.ftp_dir or not self.ftp_dir.exists():
            raise RuntimeError(f"ftp_dir does not exist: {self.ftp_dir}")

        csv_files = sorted(self.ftp_dir.glob(pattern))
        print(f"  [orats] Importing {len(csv_files)} CSV files from {self.ftp_dir}")

        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                if "tradeDate" in df.columns:
                    df["tradeDate"] = pd.to_datetime(df["tradeDate"])
                all_data.append(df)
            except Exception as e:
                print(f"  [orats] Failed to read {csv_file}: {e}")

        if not all_data:
            return 0

        combined = pd.concat(all_data, ignore_index=True)
        # Save per-ticker for fast random access
        for ticker, ticker_df in combined.groupby("ticker"):
            cache_path = self.cache_dir / "orats_raw" / f"{ticker}_bulk.parquet"
            ticker_df.to_parquet(cache_path, index=False)

        print(f"  [orats] Imported {len(combined):,} records, "
              f"{combined['ticker'].nunique():,} tickers")
        return len(combined)

    # ─────────────────────────────────────────────────────────────────────────
    # Unified IV panel: combine all cached data into (date × ticker) panels
    # ─────────────────────────────────────────────────────────────────────────

    def load_iv_panel(self) -> Dict[str, pd.DataFrame]:
        """Load all cached Orats data and pivot to (date × ticker) panels.

        Returns dict with one DataFrame per IV field:
            {
                'iv30': DataFrame(date × ticker),
                'iv60': DataFrame(date × ticker),
                'slope': DataFrame(date × ticker),
                'cp_vol_spread': DataFrame(date × ticker),
                'call_volume': DataFrame(date × ticker),
                ...
            }
        """
        cache_files = list((self.cache_dir / "orats_raw").glob("*.parquet"))
        if not cache_files:
            print("  [orats] No cached data found")
            return {}

        all_data = []
        for f in cache_files:
            try:
                all_data.append(pd.read_parquet(f))
            except Exception:
                continue
        if not all_data:
            return {}

        combined = pd.concat(all_data, ignore_index=True)
        if "tradeDate" not in combined.columns or "ticker" not in combined.columns:
            print("  [orats] Missing required columns (tradeDate, ticker)")
            return {}

        combined["tradeDate"] = pd.to_datetime(combined["tradeDate"])

        # Build panels for the C&Z signal-relevant fields.
        # Field names verified against actual /datav2/cores response (2026-04-16).
        signal_fields = [
            # Constant maturity ATM IVs (single combined value, not call/put split)
            "iv10d", "iv20d", "iv30d", "iv60d", "iv90d", "iv6m", "iv1yr",
            # ATM IVs at standard monthly expiries
            "atmIvM1", "atmIvM2", "atmIvM3", "atmIvM4",
            # Smile slope variants (positive = put skew, call-delta-based)
            "slope", "slopeInf", "slopeFcst",
            # Smile curvature
            "deriv", "derivFcst",
            # Delta-bucketed IVs at 30d (low number = low call delta = OTM call)
            #   dlt5Iv30d  = 5-delta call (deep OTM call upside skew)
            #   dlt25Iv30d = 25-delta call (OTM call)
            #   dlt75Iv30d = 75-delta call ≡ 25-delta put (OTM put)
            #   dlt95Iv30d = 95-delta call ≡ 5-delta put (deep OTM put / crash)
            "dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d",
            # Earnings-stripped versions (cleaner for cross-period comparison)
            "exErnIv30d", "exErnDlt25Iv30d", "exErnDlt75Iv30d", "exErnDlt95Iv30d",
            # Call/put volume + OI
            "cVolu", "pVolu", "cOi", "pOi",
            # Forecast vols (Orats proprietary)
            "orFcst20d", "orIvFcst20d",
            # Realized vols (multiple windows + earnings-stripped)
            "orHv10d", "orHv20d", "orHv60d", "orHv90d", "orHv252d",
            "orHvXern20d", "orHvXern60d",
            # Forward IVs (term structure decomposition)
            "fwd30_20", "fwd60_30", "fwd90_30",
            # ── BONUS signal fields (from agent research) ──
            "borrow30", "borrow2yr",       # implied borrow rate (squeeze signal)
            "annIdiv", "annActDiv",        # implied vs actual dividend (surprise)
            "etfSlopeRatio",               # skew vs sector ETF (relative skew)
            "volOfVol", "volOfIvol",       # second-order vol metrics
            "ivPctile1m", "ivPctile1y",    # IV rank pre-computed
            "ivPctileSpy", "ivPctileEtf",  # relative IV vs benchmarks
            "correlSpy1m", "correlSpy1y",  # SPY correlation (beta drift)
            "beta1m", "beta1y",            # market beta
            "impliedMove", "impliedEarningsMove",  # earnings move signals
            "absAvgErnMv", "daysToNextErn",        # earnings calendar
            "tkOver",                              # takeover indicator
            "iv200Ma",                             # IV 200-day MA
            "stockPrice", "pxAtmIv",               # spot price reference
        ]
        panels = {}
        missing_fields = []
        for field in signal_fields:
            if field in combined.columns:
                panel = combined.pivot_table(
                    index="tradeDate", columns="ticker", values=field, aggfunc="first"
                )
                panels[field] = panel
            else:
                missing_fields.append(field)
        if missing_fields:
            print(f"  [orats] Missing fields ({len(missing_fields)}): {missing_fields[:10]}"
                  f"{'...' if len(missing_fields) > 10 else ''}")

        # Derived: cp_vol_spread proxy using delta-bucketed IVs
        # Orats /cores does NOT publish separate call/put ATM IVs (single iv30d field).
        # Best proxy: dlt75Iv30d (~25-delta put) - dlt25Iv30d (~25-delta call)
        # Captures same skew information as call_atm_iv - put_atm_iv would.
        if "dlt75Iv30d" in combined.columns and "dlt25Iv30d" in combined.columns:
            # Note: dlt75 - dlt25 is positive when put skew > call skew (bearish positioning)
            # For dCPVolSpread sign convention (positive = bullish), we'd negate this
            combined["cp_vol_spread_proxy"] = (
                combined["dlt25Iv30d"] - combined["dlt75Iv30d"]
            )  # call_25d - put_25d, positive = call skew (bullish)
            panels["cp_vol_spread_proxy"] = combined.pivot_table(
                index="tradeDate", columns="ticker",
                values="cp_vol_spread_proxy", aggfunc="first"
            )

        print(f"  [orats] Loaded {len(panels)} IV panels, "
              f"{combined['ticker'].nunique()} unique tickers")
        return panels
