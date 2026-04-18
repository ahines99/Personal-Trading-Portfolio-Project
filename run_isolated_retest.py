"""Isolated Feature Retest — one feature at a time vs clean baseline.

Once the yfinance baseline confirms recovery to ~25.67%, run this script to
re-evaluate each addition INDIVIDUALLY against the clean baseline.

Tests organized in 3 groups:
  C&Z tests      — test C&Z signal groups (price-only, accounting, all)
  Phase D tests  — test Phase D additions (EODHD, EAR fix, etc.)
  Options tests  — test options signals (after Orats download done)

Each test uses the SAME baseline configuration with ONE additional change.
This isolates the marginal impact of each feature.

Usage:
    # Run all tests
    python run_isolated_retest.py

    # Run only C&Z group
    python run_isolated_retest.py --group cz

    # Run specific test
    python run_isolated_retest.py --only CZ_price_only

    # Resume from specific point
    python run_isolated_retest.py --start-from PHASE_D_eodhd
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESULTS_FILE = Path("results/isolated_retest.csv")
ARCHIVE = Path("results/archive")
ARCHIVE.mkdir(parents=True, exist_ok=True)

# Baseline config (yfinance earnings, all defaults)
# This is what should produce the 25.67% if our recovery hypothesis is correct
BASELINE_FLAGS = ["--no-use-eodhd-earnings"]

# C&Z signal name groups (used to deny specific signals via env var)
CZ_PRICE_SIGNALS = [
    "coskewness_signal", "coskewness_signal_csz", "coskewness_signal_sn",
    "coskewness_signal_sni", "coskewness_signal_sn_csz",
    "coskew_acx_signal", "coskew_acx_signal_csz", "coskew_acx_signal_sn",
    "coskew_acx_signal_sni", "coskew_acx_signal_sn_csz",
    "mom_season_signal", "mom_season_signal_csz", "mom_season_signal_sn",
    "mom_season_signal_sni", "mom_season_signal_sn_csz",
]
CZ_ACCOUNTING_SIGNALS = [
    "payout_yield_signal", "payout_yield_signal_csz", "payout_yield_signal_sn",
    "net_payout_yield_signal", "net_payout_yield_signal_csz",
    "xfin_signal", "xfin_signal_csz", "xfin_signal_sn",
    "cfp_signal", "cfp_signal_csz", "cfp_signal_sn",
    "operprof_rd_signal", "operprof_rd_signal_csz",
    "tax_signal", "tax_signal_csz",
    "deldrc_signal", "deldrc_signal_csz",
]

# Each test: (name, group, flags, env_extra_deny, description)
TESTS = [
    # ─── BASELINE (control) ───────────────────────────────────────────────
    ("BASELINE_yfinance", "baseline", BASELINE_FLAGS, [],
     "Clean yfinance baseline — should match ~25.67%"),

    # ─── C&Z TESTS ────────────────────────────────────────────────────────
    ("CZ_all_signals", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     [],
     "All 10 C&Z signals (price + accounting)"),

    ("CZ_price_only", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     CZ_ACCOUNTING_SIGNALS,
     "Only the 3 price-based C&Z signals (coskewness, coskew_acx, mom_season)"),

    ("CZ_accounting_only", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     CZ_PRICE_SIGNALS,
     "Only the 7 accounting C&Z signals (payout, xfin, cfp, etc.)"),

    # Top individual signals by C&Z research IC_IR
    ("CZ_coskewness_only", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     [s for s in CZ_PRICE_SIGNALS + CZ_ACCOUNTING_SIGNALS
      if not s.startswith("coskewness")],
     "Only coskewness signal (highest C&Z IC_IR 0.20)"),

    ("CZ_xfin_only", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     [s for s in CZ_PRICE_SIGNALS + CZ_ACCOUNTING_SIGNALS
      if not s.startswith("xfin")],
     "Only xfin signal (IC_IR 0.18)"),

    ("CZ_net_payout_only", "cz",
     BASELINE_FLAGS + ["--use-cz-signals"],
     [s for s in CZ_PRICE_SIGNALS + CZ_ACCOUNTING_SIGNALS
      if not s.startswith("net_payout_yield")],
     "Only net_payout_yield signal (IC_IR 0.18)"),

    # ─── PHASE D TESTS ────────────────────────────────────────────────────
    ("PHASE_D_eodhd_only", "phase_d",
     ["--use-eodhd-earnings"],  # switch from yfinance to EODHD
     [],
     "Just switch earnings source from yfinance to EODHD (with dedup fix)"),

    ("PHASE_D_eodhd_sraf", "phase_d",
     ["--use-eodhd-earnings", "--use-sraf-sentiment"],
     [],
     "EODHD + SRAF sentiment (filing_length signal)"),

    ("PHASE_D_no_sraf", "phase_d",
     BASELINE_FLAGS + ["--use-sraf-sentiment"],
     [],
     "Just SRAF sentiment alone (no EODHD)"),

    ("PHASE_D_full", "phase_d",
     ["--use-eodhd-earnings", "--use-sraf-sentiment"],
     [],
     "Full Phase D: EODHD + SRAF (the original Phase D config)"),

    # ─── OPTIONS TESTS (only run after Orats download done) ──────────────
    ("OPTIONS_baseline", "options",
     BASELINE_FLAGS + ["--use-options-signals"],
     [],
     "Baseline + 13 options signals (requires Orats data)"),

    ("OPTIONS_with_cz", "options",
     BASELINE_FLAGS + ["--use-cz-signals", "--use-options-signals"],
     [],
     "Baseline + C&Z + options (full alpha stack)"),

    ("OPTIONS_no_div_surprise", "options",
     BASELINE_FLAGS + ["--use-options-signals"],
     ["opt_dividend_surprise_signal", "opt_dividend_surprise_signal_csz",
      "opt_dividend_surprise_signal_sn", "opt_dividend_surprise_signal_sni"],
     "Options signals excluding dividend_surprise (proprietary Orats gap)"),

    ("OPTIONS_top5_cz", "options",
     BASELINE_FLAGS + ["--use-options-signals"],
     # Keep only top-5 options signals (the original C&Z options 5)
     # Deny everything else
     ["opt_iv_rank_signal", "opt_iv_rank_signal_csz", "opt_iv_rank_signal_sn",
      "opt_rv_iv_spread_signal", "opt_rv_iv_spread_signal_csz",
      "opt_variance_premium_signal", "opt_variance_premium_signal_csz",
      "opt_iv_term_slope_signal", "opt_iv_term_slope_signal_csz",
      "opt_risk_reversal_25d_signal", "opt_risk_reversal_25d_signal_csz",
      "opt_crash_risk_signal", "opt_crash_risk_signal_csz",
      "opt_oi_concentration_signal", "opt_oi_concentration_signal_csz",
      "opt_implied_borrow_signal", "opt_implied_borrow_signal_csz",
      "opt_dividend_surprise_signal", "opt_dividend_surprise_signal_csz",
      "opt_etf_skew_relative_signal", "opt_etf_skew_relative_signal_csz"],
     "Only top-5 C&Z options (dCPVolSpread, SmileSlope, CPVolSpread, dVolCall, dVolPut)"),

    # ─── COMBINED TESTS ──────────────────────────────────────────────────
    ("COMBINED_yfinance_cz", "combined",
     BASELINE_FLAGS + ["--use-cz-signals", "--use-tax-aware"],
     [],
     "yfinance + C&Z + tax-aware (clean recommended config)"),

    ("COMBINED_eodhd_cz_phaseD", "combined",
     ["--use-eodhd-earnings", "--use-sraf-sentiment", "--use-cz-signals",
      "--use-tax-aware"],
     [],
     "Full kitchen sink: EODHD + Phase D + C&Z + tax-aware"),
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
    p.add_argument("--group", default=None,
                   choices=["baseline", "cz", "phase_d", "options", "combined"],
                   help="Run only tests in this group")
    p.add_argument("--only", default=None,
                   help="Run only this specific test")
    p.add_argument("--start-from", default=None,
                   help="Skip tests before this one (resume after crash)")
    p.add_argument("--skip-options", action="store_true",
                   help="Skip options tests (default if Orats data not downloaded)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview tests without executing")
    p.add_argument("--baseline-from", default=None,
                   help="Path to existing tearsheet to use as baseline reference "
                        "(e.g. results/_yfinance_test/tearsheet.csv)")
    args = p.parse_args()

    # Check if Orats data exists for options tests
    iv_panels_path = Path("data/cache/options/iv_panels_orats.pkl")
    if not iv_panels_path.exists() and not args.skip_options:
        print(f"[notice] Orats IV panels not found ({iv_panels_path})")
        print(f"[notice] Options tests will be skipped automatically")
        args.skip_options = True

    # Filter tests
    tests_to_run = TESTS
    if args.group:
        tests_to_run = [t for t in tests_to_run if t[1] == args.group]
    if args.only:
        tests_to_run = [t for t in tests_to_run if t[0] == args.only]
    if args.skip_options:
        tests_to_run = [t for t in tests_to_run if t[1] != "options"]

    print("=" * 70)
    print("ISOLATED FEATURE RETEST — one variable at a time vs baseline")
    print("=" * 70)
    print(f"Tests to run: {len(tests_to_run)}")
    for t in tests_to_run:
        print(f"  - {t[0]} ({t[1]}): {t[4]}")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting without executing tests.")
        print(f"Estimated total time: ~{len(tests_to_run) * 45} min sequential")
        return

    # Resume support
    existing = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                existing.add(row["name"])

    first = not RESULTS_FILE.exists()
    started = bool(not args.start_from)
    baseline_cagr = None

    # If --baseline-from is provided, load it as the reference baseline
    if args.baseline_from:
        baseline_path = Path(args.baseline_from)
        if baseline_path.exists():
            baseline_t = parse_tearsheet(baseline_path)
            baseline_cagr_str = baseline_t.get("CAGR", "").strip("%")
            try:
                baseline_cagr = float(baseline_cagr_str)
                print(f"[baseline] Using {baseline_path}: CAGR={baseline_cagr}%")
            except (ValueError, TypeError):
                print(f"[baseline] Could not parse CAGR from {baseline_path}")
        else:
            print(f"[baseline] WARNING: {baseline_path} not found")

    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow([
                "name", "group", "CAGR", "Sharpe", "MaxDD", "Beta", "Vol",
                "AfterTax_HIFO", "Costs", "vs_baseline_bps", "description",
            ])

        for name, group, flags, env_deny, desc in tests_to_run:
            if not started:
                if name == args.start_from:
                    started = True
                else:
                    continue
            if name in existing and not args.only:
                print(f"[skip] {name} (already done)")
                continue

            rdir = Path(f"results/_iso_{name}")
            rdir.mkdir(parents=True, exist_ok=True)
            tearsheet_path = rdir / "tearsheet.csv"
            if tearsheet_path.exists():
                tearsheet_path.unlink()

            cmd = [sys.executable, "-u", "run_strategy.py",
                   "--skip-robustness",
                   "--results-dir", str(rdir)] + flags

            print(f"\n{'='*60}")
            print(f"[run] {name} ({group})")
            print(f"  {desc}")
            print(f"  flags: {' '.join(flags)}")
            if env_deny:
                print(f"  env deny: {len(env_deny)} features")
            print(f"{'='*60}", flush=True)

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            if env_deny:
                env["EXTRA_DENY_FEATURES"] = ",".join(env_deny)

            t0 = time.time()
            log_path = Path(f"logs/iso_{name}.log")
            log_path.parent.mkdir(exist_ok=True)
            with open(log_path, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s (exit {r.returncode})")

            if r.returncode != 0:
                print(f"  FAILED — see {log_path}")
                w.writerow([name, group, "FAIL", "", "", "", "", "", "",
                           "", desc[:60]])
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

            vs_baseline = ""
            try:
                cagr_f = float(cagr_str)
                if name == "BASELINE_yfinance":
                    baseline_cagr = cagr_f
                if baseline_cagr is not None:
                    vs_baseline = f"{(cagr_f - baseline_cagr) * 100:+.0f}"
            except (ValueError, TypeError):
                pass

            w.writerow([name, group, cagr_str, sharpe, maxdd, beta, vol,
                        after_tax, costs, vs_baseline, desc[:60]])
            f.flush()

            print(f"  Result: CAGR {cagr_str}% Sharpe {sharpe} MaxDD {maxdd}% "
                  f"(vs baseline: {vs_baseline} bps)")

            shutil.copy(tearsheet_path, ARCHIVE / f"iso_{name}_tearsheet.csv")

    # Summary
    print()
    print("=" * 70)
    print("ISOLATED RETEST SUMMARY")
    print("=" * 70)
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group results
        for grp in ["baseline", "phase_d", "cz", "options", "combined"]:
            grp_rows = [r for r in rows if r["group"] == grp]
            if not grp_rows:
                continue
            print(f"\n{grp.upper()} GROUP:")
            print(f"  {'Name':<30} {'CAGR':>8} {'Sharpe':>8} {'vs_base':>10}")
            print(f"  {'-'*60}")
            for r in grp_rows:
                vs = r.get("vs_baseline_bps", "")
                color = ""
                try:
                    if int(vs) > 30:
                        color = "✓ KEEP"
                    elif int(vs) < -30:
                        color = "✗ DROP"
                except (ValueError, TypeError):
                    pass
                print(f"  {r['name']:<30} {r['CAGR']:>7}% {r['Sharpe']:>8} "
                      f"{vs:>9} bps  {color}")

    print()
    print("DECISION RULES (per existing protocol):")
    print("  Keep feature: ΔCAGR ≥ +30 bps AND ΔSharpe ≥ -0.02")
    print("  Reject feature: ΔCAGR < +30 bps OR ΔSharpe < -0.02")
    print(f"\nResults: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
