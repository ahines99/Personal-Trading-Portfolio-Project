"""
tradier_client.py
-----------------
Tradier Brokerage API client for ongoing daily options data.

Tradier embeds Orats-computed Greeks/IV in their chains endpoint
(verified via Tradier docs: "Greeks and volatility data have been included
courtesy of the ORATS APIs"). This means train/serve consistency:
- Historical training data: Orats direct ($399 one-time)
- Live serving data: Tradier API (free with brokerage account)
Same SMV engine, no train/serve skew.

Usage:
    client = TradierClient(token=os.environ["TRADIER_TOKEN"])
    chain = client.fetch_chain("AAPL", target_dte=30)
    panel = client.fetch_universe_panel(tickers, target_dte=30)
"""

from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import requests


TRADIER_BASE = "https://api.tradier.com/v1"
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"


def _to_tradier_symbol(ticker: str) -> str:
    """Translate our internal symbology to Tradier's.

    Our backtest universe uses hyphen for share-class suffixes (BRK-B, BF-B,
    HEI-A, LEN-B). Tradier uses slash (BRK/B). Coverage probe (Apr 2026)
    showed `BRK-B` returned zero expirations while `BRK/B` returned 16.
    """
    if not ticker:
        return ticker
    # Only convert hyphens that look like share-class suffixes (e.g. XYZ-A,
    # XYZ-B). Leave tickers with longer hyphenated forms alone (none in US
    # listed options as of 2026, but future-proof).
    if "-" in ticker:
        parts = ticker.split("-")
        if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
            return f"{parts[0]}/{parts[1]}"
    return ticker


class TradierClient:
    """Tradier API client for options chain data with Orats-embedded Greeks."""

    def __init__(self, token: Optional[str] = None, sandbox: bool = False,
                 max_retries: int = 3, timeout: float = 15.0):
        self.token = token or os.environ.get("TRADIER_TOKEN")
        if not self.token:
            raise ValueError("Tradier token required (env TRADIER_TOKEN or pass token=)")
        self.base = TRADIER_SANDBOX if sandbox else TRADIER_BASE
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        self.max_retries = max_retries
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base}{path}"
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, headers=self.headers,
                                    timeout=self.timeout)
                if resp.status_code == 429:  # rate limit
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        return {}

    def get_expirations(self, ticker: str) -> List[str]:
        """Returns list of expiration dates (YYYY-MM-DD)."""
        data = self._get("/markets/options/expirations",
                         {"symbol": _to_tradier_symbol(ticker),
                          "includeAllRoots": "true"})
        if data is None or "expirations" not in data or data["expirations"] is None:
            return []
        exps = data["expirations"].get("date", [])
        return exps if isinstance(exps, list) else [exps]

    def get_chain(self, ticker: str, expiration: str, greeks: bool = True) -> pd.DataFrame:
        """Returns options chain DataFrame with optional Greeks (Orats-supplied)."""
        data = self._get("/markets/options/chains", {
            "symbol": _to_tradier_symbol(ticker), "expiration": expiration,
            "greeks": "true" if greeks else "false",
        })
        if data is None or "options" not in data or data["options"] is None:
            return pd.DataFrame()
        opts = data["options"].get("option", [])
        if not isinstance(opts, list):
            opts = [opts]
        df = pd.DataFrame(opts)
        if "greeks" in df.columns and len(df) > 0:
            # Flatten greeks dict into separate columns
            greeks_df = pd.json_normalize(df["greeks"].fillna({}))
            greeks_df.columns = [f"greek_{c}" for c in greeks_df.columns]
            df = pd.concat([df.drop(columns=["greeks"]), greeks_df], axis=1)
        return df

    def find_target_expiration(self, ticker: str, target_dte: int = 30) -> Optional[str]:
        """Find expiration closest to target DTE (skipping <5 day to avoid noise)."""
        exps = self.get_expirations(ticker)
        if not exps:
            return None
        today = pd.Timestamp.today().normalize()
        exp_dates = pd.to_datetime(exps)
        dtes = (exp_dates - today).days
        valid_mask = dtes >= 5
        if not valid_mask.any():
            return None
        valid_dtes = dtes[valid_mask]
        valid_exps = [e for e, ok in zip(exps, valid_mask) if ok]
        diffs = abs(valid_dtes - target_dte)
        best_idx = diffs.argmin()
        return valid_exps[best_idx]

    def fetch_ticker_summary(self, ticker: str, target_dte: int = 30) -> Optional[dict]:
        """Fetch a single ticker's options summary at target DTE.

        Returns dict with computed signal-level fields:
          - iv30_call_atm, iv30_put_atm, iv30_atm
          - iv30_25d_call, iv30_25d_put, slope
          - call_volume, put_volume, call_oi, put_oi
        """
        exp = self.find_target_expiration(ticker, target_dte)
        if exp is None:
            return None
        chain = self.get_chain(ticker, exp, greeks=True)
        if chain.empty:
            return None

        # Get spot price (use ATM strike midpoint as proxy)
        # Tradier returns 'underlying' field with each contract sometimes
        if "underlying" in chain.columns:
            underlying = chain["underlying"].iloc[0]
            # Fetch quote for underlying separately if needed; here use ATM strike approx
        spot = chain["strike"].median()  # rough — better would be a quotes call

        return _summarize_chain(chain, spot, exp_date=exp)

    def fetch_universe_panel(
        self, tickers: List[str], target_dte: int = 30,
        max_workers: int = 8, on_progress: Optional[callable] = None,
    ) -> pd.DataFrame:
        """Fetch options summaries for a universe of tickers. Returns DataFrame
        with one row per ticker, columns are signal-level fields."""
        results = {}
        n = len(tickers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.fetch_ticker_summary, t, target_dte): t
                       for t in tickers}
            done = 0
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = None
                done += 1
                if on_progress is not None and done % 50 == 0:
                    on_progress(done, n)

        df = pd.DataFrame(
            {t: r for t, r in results.items() if r is not None}
        ).T
        df.index.name = "ticker"
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Orats-equivalent SMV panel (used by daily poll → IV panel pipeline)
    #
    # The legacy fetch_universe_panel produces a flat "Tradier-legacy" summary
    # (iv30_atm, iv30_25d_call, slope, …). This is NOT what options_signals.py
    # reads — that module expects Orats /cores schema (iv30d, iv60d, slope,
    # dlt5/25/75/95Iv30d, borrow30, annIdiv). The methods below run each
    # ticker's multi-expiration chain through chain_to_smv_summary to produce
    # the full Orats-equivalent row, unlocking all 11 signal-consumable fields.
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_multi_exp_chain(
        self, ticker: str, target_dtes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Fetch chains spanning multiple expirations for CMIV interpolation.

        Returns a single DataFrame with all strikes × all selected expirations
        stacked. Adds `expiration_date` and `dte` columns.
        """
        if target_dtes is None:
            # 5 expirations: covers full term structure 7-365d for richer
            # IV-curve signals (vol convexity, term-slope, short-end skew).
            # Cost: ~1.5x daily-poll API budget vs 3 expirations.
            target_dtes = [10, 30, 60, 120, 270]
        exps = self.get_expirations(ticker)
        if not exps:
            return pd.DataFrame()
        import numpy as np
        today = pd.Timestamp.today().normalize()
        exp_dates = pd.to_datetime(exps)
        dtes = np.asarray((exp_dates - today).days, dtype=int)
        # For each target DTE, pick the closest listed expiration (≥5 days out)
        chosen: List[str] = []
        for tgt in target_dtes:
            mask = dtes >= 5
            if not mask.any():
                continue
            valid_dtes = dtes[mask]
            valid_exps = [e for e, ok in zip(exps, mask) if ok]
            idx = int(np.argmin(np.abs(valid_dtes - tgt)))
            pick = valid_exps[idx]
            if pick not in chosen:
                chosen.append(pick)
        if not chosen:
            return pd.DataFrame()
        # Fetch each chosen expiration serially (per-ticker parallelism is
        # handled at the universe level)
        pieces = []
        for exp in chosen:
            piece = self.get_chain(ticker, exp, greeks=True)
            if piece.empty:
                continue
            piece = piece.copy()
            piece["expiration_date"] = pd.to_datetime(exp)
            piece["dte"] = (piece["expiration_date"] - today).dt.days
            pieces.append(piece)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    def fetch_ticker_smv_summary(
        self, ticker: str, target_dtes: Optional[List[int]] = None,
        dividend_yield: float = 0.0,
        risk_free_rates: Optional[Dict[int, float]] = None,
    ) -> Optional[dict]:
        """Fetch a ticker's multi-expiration chain and run it through
        chain_to_smv_summary to produce an Orats /cores-equivalent row.

        Returns a dict with keys: ticker, tradeDate, stockPrice, iv30d, iv60d,
        iv90d, slope, dlt5/25/75/95Iv30d, borrow30, annIdiv, derivation_notes.

        dividend_yield (decimal, e.g. 0.005 = 0.5% TTM div yield) is wired
        into the borrow-rate parity solver — without it, the borrow extractor
        absorbs the dividend signal as negative borrow and clips to floor.
        Pass from EODHD: TTM dividend / spot.
        """
        # Local import to avoid circulars at module load
        from options_adapter.chain_to_smv_summary import chain_to_smv_summary

        chain = self.fetch_multi_exp_chain(ticker, target_dtes=target_dtes)
        if chain.empty:
            return None

        # Underlying price: use 'underlying' column if present, else ATM-ish
        # strike midpoint as a fallback.
        spot = None
        if "underlying" in chain.columns:
            try:
                spot = float(chain["underlying"].dropna().iloc[0])
            except (IndexError, ValueError, TypeError):
                spot = None
        if spot is None or spot <= 0:
            try:
                spot = float(chain["strike"].median())
            except Exception:
                spot = None

        try:
            summary = chain_to_smv_summary(
                chain=chain, ticker=ticker, underlying_price=spot,
                dividend_yield=dividend_yield,
                risk_free_rates=risk_free_rates,
                asof=pd.Timestamp.today().normalize(),
            )
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
        return summary

    def fetch_universe_smv_panel(
        self, tickers: List[str], target_dtes: Optional[List[int]] = None,
        max_workers: int = 8, on_progress: Optional[callable] = None,
        dividend_yields: Optional[Dict[str, float]] = None,
        risk_free_rates: Optional[Dict[int, float]] = None,
    ) -> pd.DataFrame:
        """Parallel fetch of full Orats-equivalent SMV summaries across a
        universe. Returns DataFrame with one row per ticker (index) and
        Orats /cores schema columns.
        """
        results: Dict[str, Optional[dict]] = {}
        n = len(tickers)
        dy_map = dividend_yields or {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.fetch_ticker_smv_summary, t, target_dtes,
                          dy_map.get(t, 0.0), risk_free_rates): t
                for t in tickers
            }
            done = 0
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = None
                done += 1
                if on_progress is not None and done % 50 == 0:
                    on_progress(done, n)

        rows = {t: r for t, r in results.items()
                if r is not None and isinstance(r, dict) and "error" not in r}
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).T
        df.index.name = "ticker"
        return df


def _summarize_chain(chain: pd.DataFrame, spot: float, exp_date: str) -> dict:
    """Reduce a full options chain to signal-level summary fields."""
    if chain.empty:
        return {}
    chain = chain.copy()
    # Tradier columns: strike, option_type ('call'/'put'), bid, ask, last,
    # volume, open_interest, greek_delta, greek_gamma, greek_theta, greek_vega,
    # greek_smv_vol (Orats SMV IV), greek_mid_iv, greek_bid_iv, greek_ask_iv
    iv_col = "greek_smv_vol" if "greek_smv_vol" in chain.columns else "greek_mid_iv"
    delta_col = "greek_delta"

    if iv_col not in chain.columns:
        return {"exp_date": exp_date}  # No Greeks returned (sandbox or error)

    calls = chain[chain["option_type"] == "call"].copy()
    puts = chain[chain["option_type"] == "put"].copy()

    # ATM = closest to spot
    out = {"exp_date": exp_date}
    if not calls.empty:
        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        out["iv30_call_atm"] = atm_call[iv_col].iloc[0] if iv_col in atm_call else None
        out["call_volume"] = int(calls["volume"].fillna(0).sum())
        out["call_oi"] = int(calls["open_interest"].fillna(0).sum())

    if not puts.empty:
        atm_put = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        out["iv30_put_atm"] = atm_put[iv_col].iloc[0] if iv_col in atm_put else None
        out["put_volume"] = int(puts["volume"].fillna(0).sum())
        out["put_oi"] = int(puts["open_interest"].fillna(0).sum())

    # 25-delta strikes
    if delta_col in chain.columns and not calls.empty:
        c25 = calls.iloc[(calls[delta_col].abs() - 0.25).abs().argsort()[:1]]
        out["iv30_25d_call"] = c25[iv_col].iloc[0] if iv_col in c25 else None
    if delta_col in chain.columns and not puts.empty:
        p25 = puts.iloc[(puts[delta_col].abs() - 0.25).abs().argsort()[:1]]
        out["iv30_25d_put"] = p25[iv_col].iloc[0] if iv_col in p25 else None

    # ATM IV (mean of call and put for stability)
    if "iv30_call_atm" in out and "iv30_put_atm" in out:
        if out.get("iv30_call_atm") is not None and out.get("iv30_put_atm") is not None:
            out["iv30_atm"] = (out["iv30_call_atm"] + out["iv30_put_atm"]) / 2

    # Slope (proxy for SmileSlope: put_25d_iv - call_25d_iv) — high = bearish skew
    if out.get("iv30_25d_call") is not None and out.get("iv30_25d_put") is not None:
        out["slope"] = out["iv30_25d_put"] - out["iv30_25d_call"]

    # CPVolSpread: call_iv - put_iv (positive = bullish flow)
    if out.get("iv30_call_atm") is not None and out.get("iv30_put_atm") is not None:
        out["cp_vol_spread"] = out["iv30_call_atm"] - out["iv30_put_atm"]

    return out
