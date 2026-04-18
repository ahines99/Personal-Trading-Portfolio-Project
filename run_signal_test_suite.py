"""
run_signal_test_suite.py
========================

Orchestrates clean, properly-isolated A/B tests of additive signals (C&Z subsets
+ Options subsets) against a frozen baseline.

Each test runs `run_strategy.py` in a fresh subprocess with a unique
`--results-dir`, captures its stdout to `logs/suite_<name>.log`, parses the
resulting `tearsheet.csv`, and writes a comparison row to a CSV. Optional
between-test cache clearing enforces strict isolation.

Manifest entries record the feature-panel hash and ml-predictions hash of every
run, proving each test produced (and consumed) a unique cache key. That is the
single best automated check that signal flags are actually flowing through the
pipeline.

Safety rules:
  * Cache files are NEVER deleted unless `--clear-cache-between-tests` is set.
  * No subprocess is launched if `--dry-run` is set.
  * An existing results CSV is NEVER overwritten silently — a timestamp suffix
    is appended.

Usage:
    python run_signal_test_suite.py --dry-run
    python run_signal_test_suite.py
    python run_signal_test_suite.py --only T2_cz_coskewness_only
    python run_signal_test_suite.py --start-from T5_cz_all_signals
    python run_signal_test_suite.py --clear-cache-between-tests
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_STRATEGY = PROJECT_ROOT / "run_strategy.py"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "signal_test_suite.csv"
DEFAULT_MANIFEST_JSON = RESULTS_DIR / "signal_test_suite_manifest.json"

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------
# Frozen baseline flags applied to every test. Anything in TESTS["args"] is
# ADDED to (or overrides) these. By keeping the baseline in one place we make
# the deltas in the comparison table unambiguous.
BASELINE_ARGS = [
    "--use-finra-short-interest",
    "--use-tax-aware",
    "--no-use-eodhd-earnings",  # T0 explicitly disables EODHD earnings
]

# NOTE: This script depends on run_strategy.py exposing per-signal CLI flags
# `--cz-only=<csv>` (added by another agent). The script does NOT validate that
# they exist; if a flag is unknown, run_strategy.py will fail and the suite
# will record a non-zero exit code for that test.
TESTS = [
    {
        "name": "T0_baseline_locked",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
        ],
    },
    {
        "name": "T1_baseline_with_eodhd",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--use-eodhd-earnings",
        ],
    },
    {
        "name": "T2_cz_coskewness_only",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-cz-signals",
            "--cz-only=coskewness",
        ],
    },
    {
        "name": "T3_cz_xfin_only",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-cz-signals",
            "--cz-only=xfin",
        ],
    },
    {
        "name": "T4_cz_net_payout_only",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-cz-signals",
            "--cz-only=net_payout_yield",
        ],
    },
    {
        "name": "T5_cz_all_signals",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-cz-signals",
        ],
    },
    {
        "name": "T6_options_only",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-options-signals",
        ],
    },
    {
        "name": "T7_cz_plus_options",
        "args": [
            "--use-finra-short-interest",
            "--use-tax-aware",
            "--no-use-eodhd-earnings",
            "--use-cz-signals",
            "--use-options-signals",
        ],
    },
]

BASELINE_NAME = "T0_baseline_locked"

# Metrics extracted from tearsheet.csv (Metric -> column header in our CSV)
METRIC_MAP = {
    "CAGR": "CAGR",
    "Total Return": "TotalReturn",
    "Annualized Volatility": "Vol",
    "Sharpe Ratio": "Sharpe",
    "Max Drawdown": "MaxDD",
    "Beta to Market": "Beta",
    "After-Tax CAGR (est.)": "AfterTaxCAGR",
    "Total Transaction Costs": "Costs",
}

# Log-line regexes — depend on the print() statements in src/model.py:
#   "      [cache] Loading feature panel from feature_panel_<hash>.pkl"
#   "      [cache] Feature panel saved to feature_panel_<hash>.pkl (float32)"
#   "  [model] Loading cached predictions from ml_predictions_<hash>.pkl"
#   "  [model] Predictions cached to ml_predictions_<hash>.pkl"
RE_FEATURE_HASH = re.compile(r"feature_panel_([0-9a-f]+)\.pkl")
RE_ML_HASH = re.compile(r"ml_predictions_([0-9a-f]+)\.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_tearsheet(tearsheet_path: Path) -> dict[str, str]:
    """Read tearsheet.csv (Metric,Value rows) and return the metrics we care
    about, keyed by our short column names. Missing metrics map to ''."""
    out = {col: "" for col in METRIC_MAP.values()}
    if not tearsheet_path.exists():
        return out
    with tearsheet_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header row "Metric,Value"
        for row in reader:
            if len(row) < 2:
                continue
            metric, value = row[0].strip(), row[1].strip()
            if metric in METRIC_MAP:
                out[METRIC_MAP[metric]] = value
    return out


def extract_hashes(log_path: Path) -> tuple[str, str]:
    """Parse the log file once, return (feature_panel_hash, ml_predictions_hash).
    Either may be '' if not present (e.g. crash before that stage)."""
    if not log_path.exists():
        return ("", "")
    feature_hash = ""
    ml_hash = ""
    # Read line-by-line to keep memory bounded on very large logs.
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not feature_hash:
                m = RE_FEATURE_HASH.search(line)
                if m:
                    feature_hash = m.group(1)
            if not ml_hash:
                m = RE_ML_HASH.search(line)
                if m:
                    ml_hash = m.group(1)
            if feature_hash and ml_hash:
                break
    return (feature_hash, ml_hash)


def clear_caches() -> list[str]:
    """Delete feature_panel_*.pkl and ml_predictions_*.pkl. Returns the list of
    deleted file names. Caller MUST gate this on --clear-cache-between-tests."""
    deleted: list[str] = []
    if not CACHE_DIR.exists():
        return deleted
    for pattern in ("feature_panel_*.pkl", "ml_predictions_*.pkl"):
        for p in CACHE_DIR.glob(pattern):
            try:
                p.unlink()
                deleted.append(p.name)
            except OSError as exc:
                print(f"  [cache] WARN: could not delete {p.name}: {exc}")
    return deleted


def parse_pct(s: str) -> float | None:
    """'20.83%' -> 20.83. Returns None if unparseable."""
    if not s:
        return None
    s = s.strip().rstrip("%").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def safe_results_csv(path: Path) -> Path:
    """If `path` exists, append a timestamp suffix so we never silently
    overwrite a prior comparison run."""
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


# ---------------------------------------------------------------------------
# Per-test runner
# ---------------------------------------------------------------------------
def run_one_test(test: dict, *, dry_run: bool) -> dict:
    """Execute a single test. Returns the manifest entry."""
    name: str = test["name"]
    extra_args: list[str] = list(test["args"])
    results_subdir = RESULTS_DIR / f"_suite_{name}"
    log_path = LOGS_DIR / f"suite_{name}.log"

    cmd = [
        sys.executable,
        "-u",
        str(RUN_STRATEGY),
        "--skip-robustness",
        "--results-dir",
        str(results_subdir),
        *extra_args,
    ]

    entry: dict = {
        "name": name,
        "cmd": cmd,
        "results_dir": str(results_subdir),
        "log_path": str(log_path),
        "start_time": "",
        "end_time": "",
        "duration_sec": None,
        "exit_code": None,
        "feature_panel_hash": "",
        "ml_predictions_hash": "",
        "metrics": {col: "" for col in METRIC_MAP.values()},
        "dry_run": dry_run,
    }

    print(f"\n{'=' * 72}")
    print(f"[{name}]")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  results_dir: {results_subdir}")
    print(f"  log: {log_path}")

    if dry_run:
        print("  [dry-run] not executing")
        return entry

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results_subdir.mkdir(parents=True, exist_ok=True)

    entry["start_time"] = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        # We tee nothing to stdout — the suite's own progress prints are
        # enough; full child output stays in the per-test log file.
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    entry["end_time"] = datetime.now().isoformat(timespec="seconds")
    entry["duration_sec"] = round(time.time() - t0, 1)
    entry["exit_code"] = proc.returncode

    # Always attempt to parse hashes + tearsheet, even on non-zero exit, so we
    # can see how far the run got.
    fh, mh = extract_hashes(log_path)
    entry["feature_panel_hash"] = fh
    entry["ml_predictions_hash"] = mh
    entry["metrics"] = parse_tearsheet(results_subdir / "tearsheet.csv")

    status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
    print(
        f"  -> {status}  duration={entry['duration_sec']}s  "
        f"feat={fh or '?'}  ml={mh or '?'}  "
        f"CAGR={entry['metrics'].get('CAGR') or '?'}"
    )
    return entry


# ---------------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------------
def write_comparison_csv(results_csv: Path, entries: list[dict]) -> None:
    """One row per test. Includes raw metrics + delta-vs-baseline columns."""
    baseline = next((e for e in entries if e["name"] == BASELINE_NAME), None)
    base_cagr = parse_pct(baseline["metrics"].get("CAGR")) if baseline else None
    base_sharpe = (
        float(baseline["metrics"]["Sharpe"])
        if baseline and baseline["metrics"].get("Sharpe")
        else None
    )

    columns = [
        "name",
        "exit_code",
        "duration_sec",
        *METRIC_MAP.values(),
        "delta_CAGR_vs_baseline",
        "delta_Sharpe_vs_baseline",
        "feature_panel_hash",
        "ml_predictions_hash",
        "results_dir",
        "log_path",
    ]

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for e in entries:
            row = {
                "name": e["name"],
                "exit_code": e["exit_code"],
                "duration_sec": e["duration_sec"],
                "feature_panel_hash": e["feature_panel_hash"],
                "ml_predictions_hash": e["ml_predictions_hash"],
                "results_dir": e["results_dir"],
                "log_path": e["log_path"],
            }
            for col in METRIC_MAP.values():
                row[col] = e["metrics"].get(col, "")

            cagr = parse_pct(e["metrics"].get("CAGR"))
            row["delta_CAGR_vs_baseline"] = (
                f"{cagr - base_cagr:+.2f}"
                if (cagr is not None and base_cagr is not None)
                else ""
            )
            try:
                sharpe = float(e["metrics"].get("Sharpe", "") or "nan")
            except ValueError:
                sharpe = None
            row["delta_Sharpe_vs_baseline"] = (
                f"{sharpe - base_sharpe:+.3f}"
                if (sharpe is not None and base_sharpe is not None)
                else ""
            )
            w.writerow(row)
    print(f"\n[suite] Comparison CSV written: {results_csv}")


def print_comparison_table(entries: list[dict]) -> None:
    """Pretty-print sorted-by-CAGR table with delta vs baseline."""
    baseline = next((e for e in entries if e["name"] == BASELINE_NAME), None)
    base_cagr = parse_pct(baseline["metrics"].get("CAGR")) if baseline else None

    rows = []
    for e in entries:
        cagr = parse_pct(e["metrics"].get("CAGR"))
        rows.append(
            {
                "name": e["name"],
                "exit": e["exit_code"] if e["exit_code"] is not None else "-",
                "cagr": cagr,
                "cagr_str": e["metrics"].get("CAGR") or "-",
                "sharpe": e["metrics"].get("Sharpe") or "-",
                "maxdd": e["metrics"].get("MaxDD") or "-",
                "beta": e["metrics"].get("Beta") or "-",
                "feat": (e["feature_panel_hash"] or "-")[:12],
                "ml": (e["ml_predictions_hash"] or "-")[:12],
            }
        )
    # Tests with no CAGR sort to the bottom.
    rows.sort(key=lambda r: (r["cagr"] is None, -(r["cagr"] or 0)))

    print("\n" + "=" * 96)
    print("Signal Test Suite — sorted by CAGR (delta vs " + BASELINE_NAME + ")")
    print("=" * 96)
    header = (
        f"{'name':<28} {'exit':>4}  {'CAGR':>8} {'dCAGR':>7}  "
        f"{'Sharpe':>6}  {'MaxDD':>8}  {'Beta':>6}  {'feat':<12} {'ml':<12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["cagr"] is not None and base_cagr is not None:
            dcagr = f"{r['cagr'] - base_cagr:+.2f}"
        else:
            dcagr = "-"
        marker = "  <-- baseline" if r["name"] == BASELINE_NAME else ""
        print(
            f"{r['name']:<28} {str(r['exit']):>4}  {r['cagr_str']:>8} "
            f"{dcagr:>7}  {r['sharpe']:>6}  {r['maxdd']:>8}  {r['beta']:>6}  "
            f"{r['feat']:<12} {r['ml']:<12}{marker}"
        )

    # Cache-isolation sanity check: each completed test should have a unique
    # feature_panel_hash AND ml_predictions_hash. Duplicates suggest a flag
    # didn't actually change the cache key.
    feat_hashes = [e["feature_panel_hash"] for e in entries if e["feature_panel_hash"]]
    ml_hashes = [e["ml_predictions_hash"] for e in entries if e["ml_predictions_hash"]]
    if len(feat_hashes) != len(set(feat_hashes)):
        print(
            "\n[suite] WARNING: duplicate feature_panel hashes detected — "
            "two tests produced the same feature cache. Either the tested "
            "flag does not affect features, or the cache key is missing the "
            "flag."
        )
    if len(ml_hashes) != len(set(ml_hashes)):
        print(
            "[suite] WARNING: duplicate ml_predictions hashes detected — "
            "two tests produced the same model output cache. Investigate."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--start-from",
        default=None,
        metavar="TEST_NAME",
        help="Skip all tests defined before this one (resume after crash).",
    )
    p.add_argument(
        "--only",
        default=None,
        metavar="TEST_NAME",
        help="Run only this single test.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    p.add_argument(
        "--clear-cache-between-tests",
        action="store_true",
        help="Delete feature_panel_*.pkl and ml_predictions_*.pkl between tests.",
    )
    p.add_argument(
        "--results-csv",
        default=str(DEFAULT_RESULTS_CSV),
        help=f"Comparison CSV path (default: {DEFAULT_RESULTS_CSV}).",
    )
    p.add_argument(
        "--manifest-json",
        default=str(DEFAULT_MANIFEST_JSON),
        help=f"Per-test manifest JSON path (default: {DEFAULT_MANIFEST_JSON}).",
    )
    args = p.parse_args()

    # Validate test names referenced via CLI.
    test_names = [t["name"] for t in TESTS]
    if args.only and args.only not in test_names:
        print(f"ERROR: --only={args.only!r} not in TESTS: {test_names}")
        return 2
    if args.start_from and args.start_from not in test_names:
        print(f"ERROR: --start-from={args.start_from!r} not in TESTS: {test_names}")
        return 2

    # Build the actual run list.
    if args.only:
        run_list = [t for t in TESTS if t["name"] == args.only]
    elif args.start_from:
        idx = test_names.index(args.start_from)
        run_list = TESTS[idx:]
    else:
        run_list = list(TESTS)

    # Resolve output paths (timestamp-suffix on collision).
    results_csv = safe_results_csv(Path(args.results_csv))
    manifest_json = safe_results_csv(Path(args.manifest_json))
    if str(results_csv) != args.results_csv:
        print(
            f"[suite] NOTE: {args.results_csv} exists; writing to {results_csv} instead."
        )
    if str(manifest_json) != args.manifest_json:
        print(
            f"[suite] NOTE: {args.manifest_json} exists; writing to {manifest_json} instead."
        )

    print(f"[suite] {len(run_list)} test(s) to run "
          f"(dry_run={args.dry_run}, clear_cache_between_tests="
          f"{args.clear_cache_between_tests})")
    for t in run_list:
        print(f"  - {t['name']}")

    entries: list[dict] = []
    for i, test in enumerate(run_list):
        # Cache hygiene BEFORE the test runs (so the very first test also gets
        # a clean slate when the user asked for it).
        if args.clear_cache_between_tests and not args.dry_run:
            deleted = clear_caches()
            print(
                f"[cache] cleared {len(deleted)} file(s) before "
                f"{test['name']}: {deleted if deleted else '(none)'}"
            )
        elif args.clear_cache_between_tests and args.dry_run:
            print(f"[cache] [dry-run] would clear caches before {test['name']}")

        entry = run_one_test(test, dry_run=args.dry_run)
        entries.append(entry)

        # Persist the manifest after every test so a crash mid-suite still
        # leaves us with partial data.
        if not args.dry_run:
            try:
                manifest_json.parent.mkdir(parents=True, exist_ok=True)
                with manifest_json.open("w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2)
            except OSError as exc:
                print(f"[suite] WARN: could not write manifest: {exc}")

    if args.dry_run:
        print("\n[suite] dry-run complete; no tests executed, no files written.")
        return 0

    # Final comparison artifacts.
    write_comparison_csv(results_csv, entries)
    print_comparison_table(entries)

    # Exit non-zero if ANY test failed, so this script is CI-friendly.
    failed = [e for e in entries if e["exit_code"] not in (0, None)]
    if failed:
        print(
            f"\n[suite] {len(failed)} test(s) failed: "
            f"{[e['name'] for e in failed]}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
