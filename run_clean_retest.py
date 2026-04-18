"""Clean Retest: Re-evaluate all additives against the recovered yfinance baseline.

This script runs after the yfinance baseline confirms recovery to ~25.67% CAGR.
It tests each major additive INDEPENDENTLY against the clean baseline to determine
which actually help (vs which we previously rejected as bad due to confounded data).

Test sequence:
  T0_baseline       Clean yfinance baseline (no extras) — should match ~25.67%
  T1_eodhd_only     yfinance baseline + switch to EODHD earnings (with dedup fixes)
  T2_phaseD_only    EODHD earnings + 6 Phase D features
  T3_cz_only        yfinance baseline + 10 C&Z signals
  T4_phaseD_cz      EODHD + Phase D + C&Z (full stack — original Phase D test)
  T5_cz_yfinance    yfinance baseline + C&Z signals (tests C&Z without EODHD confound)

After all 5 tests complete, build comparison table:
  - Which additive genuinely improved CAGR vs clean baseline?
  - Are there positive interactions (Phase D + C&Z > sum of parts)?
  - Should we keep, drop, or modify each addition?

Each test takes ~45 min, total ~4 hours sequential.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESULTS_FILE = Path("results/clean_retest.csv")
ARCHIVE = Path("results/archive")
ARCHIVE.mkdir(parents=True, exist_ok=True)

# Each config: (name, additional_args)
# Note: --use-eodhd-earnings defaults to True per current code, so we explicitly
# negate where needed via --no-use-eodhd-earnings
CONFIGS = [
    # T0: Clean yfinance baseline (control)
    ("T0_baseline_yfinance", [
        "--no-use-eodhd-earnings",
    ]),

    # T1: + EODHD earnings only (with dedup fixes applied)
    # Tests whether EODHD source itself is broken or just had duplicates
    ("T1_eodhd_only", [
        "--use-eodhd-earnings",
    ]),

    # T2: yfinance baseline + C&Z signals (no Phase D / no EODHD)
    # Tests C&Z independently of EODHD confound
    ("T2_cz_yfinance", [
        "--no-use-eodhd-earnings",
        "--use-cz-signals",
    ]),

    # T3: EODHD + C&Z (matches our previous C&Z test setup)
    # Should reproduce ~21.16% if dedup fix didn't help, OR show recovery
    ("T3_eodhd_cz", [
        "--use-eodhd-earnings",
        "--use-cz-signals",
    ]),

    # T4: yfinance + C&Z + tax-aware (tests tax-aware independently)
    ("T4_yfinance_cz_tax", [
        "--no-use-eodhd-earnings",
        "--use-cz-signals",
        "--use-tax-aware",
    ]),
]


def parse_tearsheet(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                out[row[0]] = row[1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-from", default=None,
                   help="Skip configs before this one (resume after crash)")
    p.add_argument("--only", default=None,
                   help="Run only this config (debug)")
    args = p.parse_args()

    print("=" * 70)
    print("CLEAN RETEST — Re-evaluate all additives against recovered baseline")
    print("=" * 70)
    print(f"Configs to run: {len(CONFIGS)}")
    print()

    # Track existing results to allow resume
    existing = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                existing.add(row["name"])

    first = not RESULTS_FILE.exists()
    started = False
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow([
                "name", "CAGR", "Sharpe", "MaxDD", "Beta", "Vol",
                "AfterTax_HIFO", "Costs", "vs_baseline_bps", "flags",
            ])

        baseline_cagr = None

        for name, flags in CONFIGS:
            # Resume support
            if args.start_from and not started:
                if name == args.start_from:
                    started = True
                else:
                    continue
            if args.only and name != args.only:
                continue
            if name in existing and not args.only:
                print(f"[skip] {name} (already done)")
                continue

            rdir = Path(f"results/_retest_{name}")
            rdir.mkdir(parents=True, exist_ok=True)
            tearsheet_path = rdir / "tearsheet.csv"
            if tearsheet_path.exists():
                tearsheet_path.unlink()

            cmd = [sys.executable, "-u", "run_strategy.py",
                   "--skip-robustness",
                   "--results-dir", str(rdir)] + flags

            print(f"\n{'='*60}")
            print(f"[run] {name}")
            print(f"  flags: {' '.join(flags)}")
            print(f"{'='*60}", flush=True)

            t0 = time.time()
            log_path = Path(f"logs/{name}.log")
            log_path.parent.mkdir(exist_ok=True)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            with open(log_path, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s (exit {r.returncode})")

            if r.returncode != 0:
                print(f"  FAILED — see {log_path}")
                w.writerow([name, "FAIL", "", "", "", "", "", "", "", "exit=" + str(r.returncode)])
                f.flush()
                continue

            t = parse_tearsheet(tearsheet_path)
            cagr_str = t.get("CAGR", "").strip("%")
            sharpe = t.get("Sharpe Ratio", "")
            maxdd = t.get("Max Drawdown", "").strip("%")
            beta = t.get("Beta to Market", "")
            vol = t.get("Annualized Volatility", "").strip("%")
            after_tax = t.get("After-Tax CAGR (HIFO)", "").strip("%")
            costs = t.get("Total Transaction Costs", "").strip('"').replace("$", "").replace(",", "")

            # Compute vs-baseline delta
            vs_baseline = ""
            try:
                cagr_f = float(cagr_str)
                if name == "T0_baseline_yfinance":
                    baseline_cagr = cagr_f
                if baseline_cagr is not None:
                    vs_baseline = f"{(cagr_f - baseline_cagr) * 100:+.0f}"
            except (ValueError, TypeError):
                pass

            w.writerow([name, cagr_str, sharpe, maxdd, beta, vol,
                        after_tax, costs, vs_baseline, ""])
            f.flush()

            print(f"  Result: CAGR {cagr_str}% Sharpe {sharpe} MaxDD {maxdd}% "
                  f"AfterTax {after_tax}% (vs baseline: {vs_baseline} bps)")

            # Archive tearsheet
            shutil.copy(tearsheet_path, ARCHIVE / f"retest_{name}_tearsheet.csv")

    # Print summary
    print()
    print("=" * 70)
    print("RETEST SUMMARY")
    print("=" * 70)
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f)
            print(f"{'Name':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'vs_base':>10}")
            print("-" * 75)
            for row in reader:
                print(f"{row['name']:<30} {row['CAGR']:>7}% {row['Sharpe']:>8} "
                      f"{row['MaxDD']:>7}% {row.get('vs_baseline_bps',''):>9} bps")
    print()
    print(f"Results: {RESULTS_FILE}")
    print()
    print("INTERPRETATION GUIDE:")
    print("  T0 should match ~25.67% original baseline (confirms recovery)")
    print("  T1 (EODHD only): if > T0, EODHD source is fine after dedup fix")
    print("                   if < T0, EODHD itself is the regression source")
    print("  T2 (yfinance + C&Z): isolates C&Z impact, no EODHD confound")
    print("  T3 (EODHD + C&Z): reproduces our previous C&Z test setup")
    print("  T4 (yfinance + C&Z + tax-aware): full clean stack")


if __name__ == "__main__":
    main()
