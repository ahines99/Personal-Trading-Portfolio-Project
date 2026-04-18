"""
exern_iv_extractor.py
---------------------
Extract Orats `exErnIv*d` (earnings-stripped constant-maturity IV) from cached
/cores parquets, and provide a forward-computable proxy for after the Orats
subscription ends.

Why exErn IV matters
~~~~~~~~~~~~~~~~~~~~
Orats' raw `iv30d` mixes (i) pure forward vol expectation with (ii) a discrete
event-premium for any earnings announcement falling within the 30-day window.
That event premium can dominate the panel for ~2-3 weeks per quarter on every
single name. `exErnIv30d` strips out the event-premium, leaving a clean
"steady-state" 30d vol estimate that's directly comparable across (a) the
same name pre- vs post-earnings and (b) across names in different reporting
windows.

This is *exactly* what we want for vol-regime / variance-premium signals.

Two builders:
  1. extract_exern_panel(cache_dir, output_path) — historical, FROZEN.
     Reads ALL cached cores_*.parquet files, pulls exErnIv30d/60d/90d, pivots
     to date x ticker, saves a pickle. Run once on the 2026-04-15 snapshot;
     the panel does not refresh after the Orats cancellation.

  2. compute_exern_proxy(iv30, earnings_calendar, ...) — forward, LIVE.
     For dates after the freeze, approximates exErnIv30 by subtracting an
     estimated event-premium term that decays away from any earnings date:
        event_premium ≈ ALPHA * implied_move / sqrt(days_to_event + 1)
        exErn_proxy   = iv30 - event_premium
     ALPHA is tuned against the frozen panel (default 0.3).

After the Orats subscription lapses we splice: use the frozen panel where
available; use the proxy from the freeze date onward (Tradier-derived iv30
+ EODHD earnings calendar = both inputs available with no Orats dependency).
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default fields to extract from each cores parquet. exErnIv6m and exErnIv1yr
# are also available but are not currently consumed by any signal builder.
DEFAULT_EXERN_FIELDS: Tuple[str, ...] = ("exErnIv30d", "exErnIv60d", "exErnIv90d")

# IV values are reported by Orats in vol-points (e.g. 28.07 = 28.07% annualized).
# Anything outside this band is almost certainly a data error / divide-by-zero
# (we observed e+36 outliers on TSLA without clipping).
_IV_LOW_CLIP = 0.0
_IV_HIGH_CLIP = 500.0  # 500 vol-points (~5x annualized vol) is the realistic ceiling

# Default cache locations (relative to repo root)
_DEFAULT_CACHE_DIR = "data/cache/options/orats_raw"
_DEFAULT_OUTPUT = "data/cache/options/iv_panels_exern.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Historical extractor (FROZEN at cache date, 2026-04-15)
# ─────────────────────────────────────────────────────────────────────────────

def _ticker_from_filename(fname: str) -> Optional[str]:
    """Pull TICKER out of `cores_TICKER_<from>_<to>.parquet`."""
    base = os.path.basename(fname)
    if not base.startswith("cores_"):
        return None
    parts = base[len("cores_"):].split("_")
    if not parts:
        return None
    return parts[0]


def _clean_iv_series(s: pd.Series) -> pd.Series:
    """Coerce to float, NaN-out garbage, clip to realistic IV range."""
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # Very negative or very large values are data errors (e+36 spikes seen)
    out = out.where((out >= _IV_LOW_CLIP) & (out <= _IV_HIGH_CLIP))
    return out


def extract_exern_panel(
    cache_dir: Union[str, Path] = _DEFAULT_CACHE_DIR,
    output_path: Union[str, Path] = _DEFAULT_OUTPUT,
    fields: Iterable[str] = DEFAULT_EXERN_FIELDS,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Extract exErnIv* panels from ALL cached Orats cores parquets.

    This is a one-time historical extraction. The output is FROZEN at the
    cache date (typically 2026-04-15, when Orats was last polled). It does
    NOT refresh after Orats cancellation — for live forward dates, splice
    in `compute_exern_proxy` output.

    Parameters
    ----------
    cache_dir : path to dir containing cores_*.parquet files.
    output_path : where to save the pickled dict {field: DataFrame}.
    fields : exErn fields to extract (default exErnIv30d/60d/90d).
    verbose : print progress every 100 files.

    Returns
    -------
    Dict[field_name, DataFrame[date, ticker]]
        e.g. {"exErnIv30d": <date x ticker>, "exErnIv60d": ..., ...}
        Also written to `output_path` as a pickle of the same dict.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Orats cache directory not found: {cache_dir}")

    files = sorted(cache_dir.glob("cores_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cores_*.parquet files in {cache_dir}")

    if verbose:
        print(f"[exern] scanning {len(files)} cores parquets in {cache_dir}")

    fields = list(fields)
    cols_needed = ["tradeDate"] + fields

    # Per-field, per-ticker time series — collected then concat'd at the end
    # (faster than incremental DataFrame growth).
    series_buckets: Dict[str, Dict[str, pd.Series]] = {f: {} for f in fields}

    t0 = time.time()
    n_skipped = 0
    n_ok = 0

    for i, fp in enumerate(files):
        ticker = _ticker_from_filename(fp.name)
        if ticker is None:
            n_skipped += 1
            continue

        try:
            df = pd.read_parquet(fp, columns=cols_needed)
        except Exception as e:
            # Some parquets may be missing exErn columns (older schema)
            try:
                df_full = pd.read_parquet(fp)
                missing = [c for c in cols_needed if c not in df_full.columns]
                if missing:
                    n_skipped += 1
                    if verbose and i < 10:
                        print(f"[exern] skip {fp.name}: missing {missing}")
                    continue
                df = df_full[cols_needed]
            except Exception as e2:
                n_skipped += 1
                if verbose and i < 10:
                    print(f"[exern] skip {fp.name}: read error {type(e2).__name__}")
                continue

        # tradeDate as datetime index
        try:
            df = df.copy()
            df["tradeDate"] = pd.to_datetime(df["tradeDate"], errors="coerce")
            df = df.dropna(subset=["tradeDate"])
            # Some tickers have duplicate trade dates (caching artifact); keep last
            df = df.drop_duplicates(subset=["tradeDate"], keep="last")
            df = df.set_index("tradeDate").sort_index()
        except Exception:
            n_skipped += 1
            continue

        for f in fields:
            if f not in df.columns:
                continue
            s = _clean_iv_series(df[f])
            if s.notna().any():
                series_buckets[f][ticker] = s

        n_ok += 1
        if verbose and (i + 1) % 250 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            print(f"[exern]   {i + 1}/{len(files)} parquets read "
                  f"({rate:.0f}/s, {elapsed:.0f}s elapsed, {n_skipped} skipped)")

    if verbose:
        print(f"[exern] read complete: {n_ok} ok, {n_skipped} skipped, "
              f"{time.time() - t0:.1f}s")
        print(f"[exern] pivoting per-ticker series -> date × ticker panels")

    panels: Dict[str, pd.DataFrame] = {}
    for f in fields:
        bucket = series_buckets[f]
        if not bucket:
            if verbose:
                print(f"[exern]   {f}: no data found")
            continue
        # concat with keys=tickers gives a 2-level column MultiIndex; unstack
        # by aligning to a union date index.
        panel = pd.DataFrame(bucket)  # Series dict -> DataFrame, columns=tickers
        panel.index.name = "date"
        panel = panel.sort_index()
        panels[f] = panel
        if verbose:
            print(f"[exern]   {f}: shape={panel.shape}, "
                  f"dates={panel.index.min().date()}->{panel.index.max().date()}, "
                  f"tickers={panel.shape[1]}, "
                  f"density={panel.notna().mean().mean():.1%}")

    # Write pickle (frozen panel)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panels, output_path)
    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"[exern] FROZEN panel saved: {output_path} ({size_mb:.1f} MB)")
        print(f"[exern] NOTE: this panel does not refresh after Orats cancellation.")
        print(f"[exern]       Use compute_exern_proxy() for forward dates.")

    return panels


# ─────────────────────────────────────────────────────────────────────────────
# Forward-computable proxy (post-Orats)
# ─────────────────────────────────────────────────────────────────────────────

def _earnings_dates_panel(
    earnings_calendar: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """For each (date, ticker), return signed days-to-next-earnings.

    Positive = days until upcoming earnings.
    Negative = days since most-recent past earnings (we use abs() for proximity).
    NaN = no earnings on either side within the calendar.

    The proximity factor only cares about |days_to_event|, so the sign is
    informational; we return signed values so callers can also implement
    "post-event drift" features if desired.
    """
    out = pd.DataFrame(np.nan, index=date_index, columns=tickers, dtype="float32")

    # Convert cal entries -> sorted DatetimeIndex per ticker once
    cal: Dict[str, pd.DatetimeIndex] = {}
    for t in tickers:
        if t not in earnings_calendar:
            continue
        df = earnings_calendar[t]
        if df is None or len(df) == 0:
            continue
        # earnings_date may be the index OR a column
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif "earnings_date" in df.columns:
            dates = pd.to_datetime(df["earnings_date"], errors="coerce")
        elif "report_date" in df.columns:
            dates = pd.to_datetime(df["report_date"], errors="coerce")
        else:
            continue
        dates = pd.DatetimeIndex(dates).dropna().normalize().unique().sort_values()
        if len(dates):
            cal[t] = dates

    if not cal:
        return out

    # For each ticker, compute signed days-to-nearest-earnings using searchsorted.
    # Vectorized: numpy diff over the date_index works fine for thousands of dates.
    date_arr = date_index.values.astype("datetime64[D]")
    for t, ev in cal.items():
        if t not in out.columns:
            continue
        ev_arr = np.array(ev.values, dtype="datetime64[D]")
        # For each date d, find the nearest ev:
        # idx = searchsorted insertion point (right of any equal entries)
        idx = np.searchsorted(ev_arr, date_arr)
        # Candidate before and after
        n = len(ev_arr)
        prev_ev = np.where(idx > 0, ev_arr[np.clip(idx - 1, 0, n - 1)],
                           np.datetime64("NaT", "D"))
        next_ev = np.where(idx < n, ev_arr[np.clip(idx, 0, n - 1)],
                           np.datetime64("NaT", "D"))
        # Days deltas (signed: positive = upcoming, negative = past)
        days_next = (next_ev - date_arr).astype("timedelta64[D]").astype(float)
        days_prev = (prev_ev - date_arr).astype("timedelta64[D]").astype(float)
        # Pick whichever is smaller in magnitude (treat NaN as +inf)
        abs_n = np.where(np.isnan(days_next), np.inf, np.abs(days_next))
        abs_p = np.where(np.isnan(days_prev), np.inf, np.abs(days_prev))
        chosen = np.where(abs_n <= abs_p, days_next, days_prev)
        chosen = np.where(np.isfinite(chosen), chosen, np.nan)
        out[t] = chosen.astype("float32")

    return out


def compute_exern_proxy(
    iv30_panel: pd.DataFrame,
    earnings_calendar: Dict[str, pd.DataFrame],
    implied_move_panel: Optional[pd.DataFrame] = None,
    max_days_to_earnings: int = 30,
    alpha: float = 3.0,
) -> pd.DataFrame:
    """Approximate exErnIv30 from iv30 + earnings calendar (no Orats needed).

    Logic
    -----
    For any date within `max_days_to_earnings` of an earnings announcement
    (past or future) we estimate the event-premium embedded in iv30 and
    subtract it:

        event_premium = alpha * implied_move_proxy / sqrt(|days_to_event| + 1)
        exern_proxy   = iv30  -  event_premium

    `implied_move_proxy` is the ATM straddle / spot fraction (Tradier gives
    us this directly via `impliedMove`). When it isn't available, we fall
    back to using `iv30` itself scaled to a 1-month horizon as a crude proxy:

        implied_move ≈ iv30 / 100 * sqrt(30/365)

    Sign convention: the proxy returns the same convention as Orats —
    `exern_proxy <= iv30` whenever earnings are nearby; identical to iv30
    otherwise. ALPHA=0.3 was calibrated to match the median (iv30 - exErnIv30)
    spread we observed in the 2026-04-15 frozen panel (~2 vol points
    near earnings, ~0 outside the 30-day window).

    Parameters
    ----------
    iv30_panel : DataFrame[date, ticker] — 30-day constant-maturity IV.
        Whatever units iv30 is in (Orats uses vol-points 0-100, our Tradier
        adapter uses fractions 0-1). The proxy preserves units.
    earnings_calendar : dict {ticker: DataFrame} from EODHD cache.
    implied_move_panel : optional DataFrame[date, ticker] of ATM-straddle/spot
        for the same dates/tickers; if None we derive from iv30.
    max_days_to_earnings : window where we apply the event-premium subtract.
        Outside this window, exern_proxy = iv30 exactly.
    alpha : event-premium coefficient. Default 0.3. Tune higher if you find
        the proxy systematically over-estimates exErn (i.e. proxy < true).

    Returns
    -------
    DataFrame[date, ticker] — same shape as iv30_panel.
    """
    if iv30_panel is None or iv30_panel.empty:
        return iv30_panel

    iv30 = iv30_panel.astype("float32", copy=False)
    date_idx = pd.DatetimeIndex(iv30.index)
    tickers = iv30.columns

    # Build proximity panel
    days_to_earn = _earnings_dates_panel(earnings_calendar, tickers, date_idx)

    # Out-of-window mask -> no event premium
    in_window = days_to_earn.abs() <= float(max_days_to_earnings)

    # Implied move proxy (units-aware: fraction)
    if implied_move_panel is not None:
        im = implied_move_panel.reindex(index=date_idx, columns=tickers).astype(
            "float32", copy=False
        )
    else:
        # Heuristic: 30-day move ≈ iv * sqrt(30/365). iv30 may be 0-1 (fraction)
        # or 0-100 (vol-points); detect from median magnitude.
        med = float(np.nanmedian(iv30.values)) if iv30.size else np.nan
        scale = 100.0 if (np.isfinite(med) and med > 5.0) else 1.0
        im = (iv30 / scale) * float(np.sqrt(30.0 / 365.0))

    # Event premium (only inside window)
    days_abs = days_to_earn.abs()
    decay = 1.0 / np.sqrt(days_abs + 1.0)  # NaN where no earnings nearby
    event_premium = (alpha * im * decay).where(in_window, 0.0)
    # When iv30 is in vol-points (0-100), implied_move (~0.05) is in fractions;
    # scale event_premium back to vol-points to subtract on equal footing.
    med = float(np.nanmedian(iv30.values)) if iv30.size else np.nan
    if np.isfinite(med) and med > 5.0:
        event_premium = event_premium * 100.0  # back to vol-points

    proxy = iv30 - event_premium.astype("float32")
    # Floor at 0 — IV cannot be negative
    proxy = proxy.clip(lower=0.0)
    return proxy


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point: run the historical extraction once
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract exErnIv panels from Orats cache")
    p.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    p.add_argument("--output", default=_DEFAULT_OUTPUT)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    panels = extract_exern_panel(
        cache_dir=args.cache_dir,
        output_path=args.output,
        verbose=not args.quiet,
    )
    print(f"\nDone. {len(panels)} panels extracted: {list(panels.keys())}")
