"""
merge_tradier_daily_to_iv_panel.py
-----------------------------------
Bridge between daily Tradier polling and the IV-panel format that
`src/options_signals.py` consumes.

Data flow:

    data/cache/options/tradier_daily/<YYYY-MM-DD>.parquet
          (one row per ticker, columns = per-ticker summary fields
           written by run_options_setup.py::cmd_daily_poll)
                              |
                              v
                     [this script]
                              |
                              v
    data/cache/options/iv_panels_tradier.pkl
          (dict of field -> date x ticker DataFrame, matching the
           schema produced by src/orats_loader.py::load_iv_panel)
                              |
                              v
    loaded by src/options_signals.py via run_strategy.py
          (consumes keys: iv30d, iv60d, iv90d, slope, dlt5Iv30d,
           dlt25Iv30d, dlt75Iv30d, dlt95Iv30d, borrow30, annIdiv,
           cp_vol_spread_proxy)

Schema handling
---------------
The daily snapshot schema depends on which pipeline wrote it:

  (a) Tradier-legacy (current cmd_daily_poll, via tradier_client._summarize_chain):
      iv30_call_atm, iv30_put_atm, iv30_atm, iv30_25d_call, iv30_25d_put,
      slope, cp_vol_spread, call_volume, put_volume, call_oi, put_oi, exp_date

  (b) Orats-SMV (when chain_to_smv_summary is applied upstream):
      iv30d, iv60d, iv90d, slope, dlt5Iv30d, dlt25Iv30d, dlt75Iv30d,
      dlt95Iv30d, borrow30, annIdiv, annActDiv, stockPrice, tradeDate

This script auto-detects which schema is in use and maps Tradier-legacy
fields onto the Orats-equivalent keys expected downstream. Missing fields
become all-NaN panels so options_signals.py can skip those signals cleanly.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Keys consumed by src/options_signals.py::build_options_signals
EXPECTED_KEYS: Tuple[str, ...] = (
    "iv30d",
    "iv60d",
    "iv90d",
    "slope",
    "dlt5Iv30d",
    "dlt25Iv30d",
    "dlt75Iv30d",
    "dlt95Iv30d",
    "borrow30",
    "annIdiv",
    "cp_vol_spread_proxy",
)

# Map Tradier-legacy summary column -> Orats-equivalent panel key.
# When the daily file already uses Orats-SMV naming, we pass through instead.
TRADIER_TO_ORATS = {
    "iv30_atm": "iv30d",
    "slope": "slope",
    "iv30_25d_call": "dlt25Iv30d",
    "iv30_25d_put": "dlt75Iv30d",
    # cp_vol_spread comes straight from _summarize_chain as call_atm - put_atm;
    # options_signals uses cp_vol_spread_proxy = dlt25Iv30d - dlt75Iv30d.
    # We expose BOTH: the raw spread if present and the proxy computed below.
    "cp_vol_spread": "cp_vol_spread_proxy",
}

# Orats-SMV keys that pass through unchanged when present in the daily file.
ORATS_PASSTHROUGH = {
    "iv30d", "iv60d", "iv90d", "slope",
    "dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d",
    "borrow30", "annIdiv",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-dir", default="data/cache/options/tradier_daily",
                   help="Directory of <date>.parquet daily snapshots.")
    p.add_argument("--output", default="data/cache/options/iv_panels_tradier.pkl",
                   help="Pickle path for the combined panel dict.")
    p.add_argument("--start-date", default=None,
                   help="Inclusive lower bound on file date (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None,
                   help="Inclusive upper bound on file date (YYYY-MM-DD).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would happen; do not write output.")
    p.add_argument("--yes", action="store_true",
                   help="Allow overwriting existing --output when no date range "
                        "is given (full reprocess).")
    return p.parse_args()


def _enumerate_daily_files(input_dir: Path,
                           start: Optional[str],
                           end: Optional[str]) -> List[Tuple[pd.Timestamp, Path]]:
    """Return sorted [(date, path)] for every <date>.parquet in input_dir."""
    if not input_dir.exists():
        return []
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    out: List[Tuple[pd.Timestamp, Path]] = []
    for p in sorted(input_dir.glob("*.parquet")):
        try:
            d = pd.Timestamp(p.stem)
        except (ValueError, TypeError):
            print(f"  [skip] {p.name}: filename is not a date")
            continue
        if start_ts is not None and d < start_ts:
            continue
        if end_ts is not None and d > end_ts:
            continue
        out.append((d, p))
    return out


def _load_snapshot(path: Path) -> pd.DataFrame:
    """Load one daily parquet; coerce ticker column into the index."""
    df = pd.read_parquet(path)
    if df.index.name != "ticker":
        # cmd_daily_poll writes with index.name="ticker"; guard for drift.
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        else:
            df.index.name = "ticker"
    return df


def _extract_fields(snapshot: pd.DataFrame,
                    date: pd.Timestamp) -> Dict[str, pd.Series]:
    """Pull each expected field out of one snapshot as a ticker-indexed Series.

    Returns dict {field -> Series(index=ticker)} ready to be stacked into the
    date x ticker pivot later. Missing fields are omitted (caller builds NaN
    rows when a field never appears).
    """
    out: Dict[str, pd.Series] = {}

    # First: any Orats-SMV columns that pass through unchanged.
    for col in ORATS_PASSTHROUGH:
        if col in snapshot.columns:
            out[col] = snapshot[col]

    # Second: Tradier-legacy columns mapped onto Orats-equivalent keys.
    # Do NOT overwrite an Orats passthrough if both happen to be present.
    for src_col, dst_key in TRADIER_TO_ORATS.items():
        if src_col in snapshot.columns and dst_key not in out:
            out[dst_key] = snapshot[src_col]

    # Third: derive cp_vol_spread_proxy from delta buckets when possible.
    # Prefer this over the raw Tradier cp_vol_spread because it matches the
    # definition options_signals.py expects (dlt25Iv30d - dlt75Iv30d).
    if "dlt25Iv30d" in out and "dlt75Iv30d" in out:
        out["cp_vol_spread_proxy"] = out["dlt25Iv30d"] - out["dlt75Iv30d"]

    return out


def _collect(daily_files: List[Tuple[pd.Timestamp, Path]]) -> Dict[str, pd.DataFrame]:
    """Loop over daily files, pivot each field into date x ticker."""
    # field -> list of (date, Series) to stack later
    per_field: Dict[str, List[Tuple[pd.Timestamp, pd.Series]]] = {}
    for date, path in daily_files:
        try:
            snap = _load_snapshot(path)
        except Exception as e:  # pragma: no cover - corrupt parquet
            print(f"  [error] {path.name}: {type(e).__name__}: {e}")
            continue

        n_rows = len(snap)
        fields = _extract_fields(snap, date)
        missing_here = [k for k in EXPECTED_KEYS if k not in fields]
        per_ticker_missing = 0
        if fields:
            # Count tickers that have NaN across all expected fields present.
            present_mat = pd.concat(
                {k: fields[k] for k in fields}, axis=1
            )
            per_ticker_missing = int(present_mat.isna().all(axis=1).sum())

        print(f"  [ok]   {path.name}: {n_rows} rows, "
              f"{len(fields)}/{len(EXPECTED_KEYS)} fields, "
              f"{len(missing_here)} missing, "
              f"{per_ticker_missing} tickers all-NaN")

        for key, series in fields.items():
            per_field.setdefault(key, []).append((date, series))

    # Pivot each field into a date x ticker DataFrame.
    panels: Dict[str, pd.DataFrame] = {}
    for key, entries in per_field.items():
        entries.sort(key=lambda x: x[0])
        frame = pd.DataFrame(
            {date: series for date, series in entries}
        ).T
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        frame.columns.name = "ticker"
        panels[key] = frame.sort_index()
    return panels


def _merge_with_existing(new_panels: Dict[str, pd.DataFrame],
                         output_path: Path) -> Dict[str, pd.DataFrame]:
    """Combine new panels with an existing pickle via outer-index union."""
    if not output_path.exists():
        return new_panels
    try:
        with open(output_path, "rb") as f:
            prior = pickle.load(f)
    except Exception as e:
        print(f"  [warn] could not read existing {output_path}: {e}; "
              f"treating as empty.")
        return new_panels
    if not isinstance(prior, dict):
        print(f"  [warn] existing {output_path} is not a dict; overwriting.")
        return new_panels

    merged: Dict[str, pd.DataFrame] = {}
    all_keys = set(prior) | set(new_panels)
    for key in all_keys:
        old_df = prior.get(key)
        new_df = new_panels.get(key)
        if old_df is None:
            merged[key] = new_df
        elif new_df is None:
            merged[key] = old_df
        else:
            # Union both axes; newer dates overwrite older overlaps.
            combined = old_df.combine_first(new_df)
            # combine_first prefers old; we want new to win on overlap.
            combined.update(new_df)
            merged[key] = combined.sort_index()
    return merged


def _ensure_expected_schema(panels: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """Add empty DataFrames for any missing expected keys. Return (present, missing)."""
    present: List[str] = []
    missing: List[str] = []
    for key in EXPECTED_KEYS:
        if key in panels and panels[key] is not None and not panels[key].empty:
            present.append(key)
        else:
            panels[key] = pd.DataFrame()  # empty placeholder
            missing.append(key)
    return present, missing


def _print_schema_report(panels: Dict[str, pd.DataFrame]) -> None:
    present, missing = _ensure_expected_schema(panels)
    print()
    print("-" * 60)
    print("Schema validation vs src/options_signals.py")
    print("-" * 60)
    print(f"Present ({len(present)}/{len(EXPECTED_KEYS)}):")
    for key in present:
        df = panels[key]
        print(f"    {key:<22} shape={df.shape}")
    if missing:
        print(f"Missing ({len(missing)}/{len(EXPECTED_KEYS)}):")
        for key in missing:
            print(f"    {key}")


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if (output_path.exists() and args.start_date is None
            and not args.yes and not args.dry_run):
        print(f"ERROR: {output_path} exists and --start-date was not supplied.")
        print("       Re-run with --start-date to append a range, or --yes to "
              "overwrite the full panel.")
        return 2

    daily_files = _enumerate_daily_files(
        input_dir, args.start_date, args.end_date
    )
    print(f"Scanning {input_dir} ...")
    print(f"  found {len(daily_files)} daily file(s) in range "
          f"{args.start_date or 'MIN'} .. {args.end_date or 'MAX'}")
    if not daily_files:
        print("Nothing to process.")
        return 0

    print()
    print("Per-file ingest:")
    new_panels = _collect(daily_files)

    if args.dry_run:
        print()
        print("[dry-run] Would merge and write to:", output_path)
        _print_schema_report(new_panels)
        return 0

    merged = _merge_with_existing(new_panels, output_path)
    _print_schema_report(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)
    print()
    print(f"Wrote {output_path} with {len(merged)} panel keys.")
    return 0


# ---------------------------------------------------------------------------
# Self-test (no __main__ wrapper; call test_merge_tradier() explicitly)
# ---------------------------------------------------------------------------

def test_merge_tradier(tmp_root: Optional[Path] = None) -> None:
    """Synthetic 2-ticker x 3-day fake input; verify output schema and pivots."""
    import shutil
    import tempfile

    root = Path(tmp_root or tempfile.mkdtemp(prefix="merge_tradier_test_"))
    input_dir = root / "tradier_daily"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path = root / "iv_panels_tradier.pkl"

    dates = ["2026-04-14", "2026-04-15", "2026-04-16"]
    tickers = ["AAPL", "MSFT"]
    for i, d in enumerate(dates):
        df = pd.DataFrame(
            {
                # Tradier-legacy schema as produced by _summarize_chain
                "iv30_call_atm": [0.25 + 0.01 * i, 0.22 + 0.01 * i],
                "iv30_put_atm":  [0.27 + 0.01 * i, 0.24 + 0.01 * i],
                "iv30_atm":      [0.26 + 0.01 * i, 0.23 + 0.01 * i],
                "iv30_25d_call": [0.24 + 0.01 * i, 0.21 + 0.01 * i],
                "iv30_25d_put":  [0.28 + 0.01 * i, 0.25 + 0.01 * i],
                "slope":         [0.04, 0.04],
                "cp_vol_spread": [-0.02, -0.02],
                "call_volume":   [100, 80],
                "put_volume":    [90, 70],
                "call_oi":       [1000, 800],
                "put_oi":        [900, 700],
                "exp_date":      ["2026-05-16", "2026-05-16"],
            },
            index=pd.Index(tickers, name="ticker"),
        )
        df.to_parquet(input_dir / f"{d}.parquet")

    # Run via argparse-style namespace
    ns = argparse.Namespace(
        input_dir=str(input_dir),
        output=str(output_path),
        start_date=None,
        end_date=None,
        dry_run=False,
        yes=True,
    )
    rc = run(ns)
    assert rc == 0, f"run() returned {rc}"

    with open(output_path, "rb") as f:
        panels = pickle.load(f)

    # Every expected key must exist in the output dict.
    for key in EXPECTED_KEYS:
        assert key in panels, f"missing expected key: {key}"

    # Keys that should have non-empty 3x2 pivots given our synthetic input.
    non_empty = {"iv30d", "slope", "dlt25Iv30d", "dlt75Iv30d", "cp_vol_spread_proxy"}
    for key in non_empty:
        df = panels[key]
        assert not df.empty, f"{key} unexpectedly empty"
        assert df.shape == (3, 2), f"{key} shape={df.shape}, expected (3, 2)"
        assert set(df.columns) == set(tickers), f"{key} cols={list(df.columns)}"

    # Keys we cannot derive from Tradier-legacy should be empty placeholders.
    for key in ("iv60d", "iv90d", "dlt5Iv30d", "dlt95Iv30d", "borrow30", "annIdiv"):
        assert panels[key].empty, f"{key} should be empty placeholder"

    # cp_vol_spread_proxy should equal dlt25Iv30d - dlt75Iv30d elementwise.
    proxy = panels["cp_vol_spread_proxy"]
    diff = panels["dlt25Iv30d"] - panels["dlt75Iv30d"]
    assert ((proxy - diff).abs().fillna(0).values < 1e-9).all(), \
        "cp_vol_spread_proxy mismatch vs dlt25 - dlt75"

    print()
    print("test_merge_tradier: PASS")
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
