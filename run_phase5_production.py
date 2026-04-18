"""
run_phase5_production.py
========================

Phase 5 production orchestrator for the personal ML US-equity portfolio.

Locked baseline config is Phase 1.8d. Phase 5 differs from prior phases only
in that `--skip-robustness` is REMOVED, enabling bootstrap CIs, Fama-French
factor regression, stress tests, and the rest of the robustness pack that
already lives inside ``run_strategy.py``.

Usage
-----
Full production run (2-3 hours)::

    python run_phase5_production.py

Smoke test (re-post-processes existing results/*.csv without re-running)::

    python run_phase5_production.py --dry-run

This script NEVER edits any existing file in the repo. It only creates:
    results/archive/phase1_8d_baseline_<ts>/...
    results/phase5_run.log
    results/phase5_summary.md
    results/production_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Locked Phase 1.8d config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"
LOG_PATH = RESULTS_DIR / "phase5_run.log"
SUMMARY_PATH = RESULTS_DIR / "phase5_summary.md"
CONFIG_PATH = RESULTS_DIR / "production_config.json"
RUN_STRATEGY = PROJECT_ROOT / "run_strategy.py"

LOCKED_FLAGS: dict = {
    "start": "2013-01-01",
    "forward_window": 21,
    "signal_smooth_halflife": 10.0,
    "quality_percentile": 0.0,
    "max_stock_vol": 1.0,
    "min_adv_for_selection": 0,
    "min_holding_overlap": 0.50,
    "n_positions": 50,
    "ml_blend": 0.30,
    "vol_target": 0.30,
    "max_leverage": 1.8,
    "vol_ceiling": 0.40,
    "quality_tilt": 0.35,
    "cash_in_bear": 0.30,
    "no_cs_zscore_all": True,
    "sample_weight_halflife_years": 4.5,
}


def build_reproduction_command() -> list[str]:
    """Return the exact argv list to reproduce Phase 5."""
    cmd: list[str] = [
        sys.executable,
        str(RUN_STRATEGY),
        "--start", LOCKED_FLAGS["start"],
        "--forward-window", str(LOCKED_FLAGS["forward_window"]),
        "--signal-smooth-halflife", str(LOCKED_FLAGS["signal_smooth_halflife"]),
        "--quality-percentile", str(LOCKED_FLAGS["quality_percentile"]),
        "--max-stock-vol", str(LOCKED_FLAGS["max_stock_vol"]),
        "--min-adv-for-selection", str(LOCKED_FLAGS["min_adv_for_selection"]),
        "--min-holding-overlap", str(LOCKED_FLAGS["min_holding_overlap"]),
        "--n-positions", str(LOCKED_FLAGS["n_positions"]),
        "--ml-blend", str(LOCKED_FLAGS["ml_blend"]),
        "--vol-target", str(LOCKED_FLAGS["vol_target"]),
        "--max-leverage", str(LOCKED_FLAGS["max_leverage"]),
        "--vol-ceiling", str(LOCKED_FLAGS["vol_ceiling"]),
        "--quality-tilt", str(LOCKED_FLAGS["quality_tilt"]),
        "--cash-in-bear", str(LOCKED_FLAGS["cash_in_bear"]),
        "--no-cs-zscore-all",
        "--sample-weight-halflife-years", str(LOCKED_FLAGS["sample_weight_halflife_years"]),
    ]
    return cmd


def pretty_command(argv: list[str]) -> str:
    """Human-readable one-liner for the reproduction command."""
    parts = []
    for token in argv:
        if " " in token or "\\" in token:
            parts.append(f'"{token}"')
        else:
            parts.append(token)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(dry_run: bool) -> None:
    print("[preflight] Checking environment...")
    if not RUN_STRATEGY.exists():
        raise SystemExit(f"[preflight] FATAL: {RUN_STRATEGY} not found")

    cache_dir = PROJECT_ROOT / "data" / "cache"
    if not cache_dir.exists():
        raise SystemExit(f"[preflight] FATAL: {cache_dir} not found")

    required_cache_hints = [
        "eodhd_universe",
        "top_universe_3000",
        "feature_panel_",
        "ml_predictions_",
    ]
    cache_files = [p.name for p in cache_dir.iterdir() if p.is_file()]
    missing = [hint for hint in required_cache_hints
               if not any(hint in name for name in cache_files)]
    if missing:
        msg = f"[preflight] WARN: missing cache hints {missing} in {cache_dir}"
        if dry_run:
            print(msg)
        else:
            raise SystemExit(msg.replace("WARN", "FATAL"))

    if LOG_PATH.exists():
        print(f"[preflight] WARN: existing {LOG_PATH} will be overwritten")

    print("[preflight] OK")


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def archive_existing_results(timestamp: str) -> Path:
    target = ARCHIVE_DIR / f"phase1_8d_baseline_{timestamp}"
    target.mkdir(parents=True, exist_ok=True)
    if not RESULTS_DIR.exists():
        print(f"[archive] {RESULTS_DIR} does not exist; nothing to archive")
        return target
    print(f"[archive] Copying results/ -> {target}")
    for item in RESULTS_DIR.iterdir():
        if item.name == "archive":
            continue
        dest = target / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[archive] WARN could not copy {item.name}: {exc}")
    print(f"[archive] Done ({sum(1 for _ in target.rglob('*'))} entries)")
    return target


# ---------------------------------------------------------------------------
# Backtest invocation
# ---------------------------------------------------------------------------

def run_backtest(argv: list[str]) -> int:
    print(f"[backtest] Launching: {pretty_command(argv)}")
    print(f"[backtest] Log -> {LOG_PATH}")
    print("[backtest] This takes 2-3 hours with --skip-robustness removed.")
    start = time.time()
    with LOG_PATH.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# Phase 5 production run {datetime.now().isoformat()}\n")
        log_fh.write(f"# CMD: {pretty_command(argv)}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_heartbeat = time.time()
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            # Forward every line, but also print a heartbeat every 60s
            sys.stdout.write(line)
            now = time.time()
            if now - last_heartbeat > 60:
                elapsed = now - start
                print(f"[backtest] ... still running, elapsed {elapsed/60:.1f} min")
                last_heartbeat = now
        rc = proc.wait()
    dur = time.time() - start
    print(f"[backtest] Exit code {rc} after {dur/60:.1f} min")
    return rc


# ---------------------------------------------------------------------------
# Result parsing helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[list[str]]:
    import csv
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def _percent_to_float(text: str) -> float | None:
    if text is None:
        return None
    text = text.strip().replace("+", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def _try_float(text: str) -> float | None:
    if text is None:
        return None
    text = text.strip().replace("+", "")
    if text.endswith("%"):
        return _percent_to_float(text + "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_tearsheet(path: Path) -> dict:
    rows = _read_csv_rows(path)
    out: dict = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        key, val = row[0].strip(), row[1].strip()
        out[key] = val
    return out


def parse_fama_french(path: Path) -> dict:
    rows = _read_csv_rows(path)
    out: dict = {}
    if not rows:
        return out
    # Header: ,Coefficient,Std Error,t-stat,p-value,Ann. Alpha,R²
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        name = row[0].strip()
        entry = {
            "coef": _try_float(row[1]) if len(row) > 1 else None,
            "se": _try_float(row[2]) if len(row) > 2 else None,
            "t": _try_float(row[3]) if len(row) > 3 else None,
            "p": _try_float(row[4]) if len(row) > 4 else None,
            "ann_alpha": _try_float(row[5]) if len(row) > 5 else None,
            "r2": _try_float(row[6]) if len(row) > 6 else None,
        }
        out[name] = entry
    return out


def parse_oos(path: Path) -> dict:
    rows = _read_csv_rows(path)
    out: dict = {}
    if not rows:
        return out
    header = rows[0]
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        label = row[0].strip()
        entry = {}
        for i, col in enumerate(header[1:], start=1):
            if i < len(row):
                entry[col.strip()] = row[i].strip()
        out[label] = entry
    return out


def parse_bootstrap(path: Path) -> dict:
    rows = _read_csv_rows(path)
    out: dict = {}
    if not rows:
        return out
    for row in rows[1:]:
        if not row:
            continue
        out[row[0].strip()] = {
            "observed": _try_float(row[1]) if len(row) > 1 else None,
            "p5": _try_float(row[2]) if len(row) > 2 else None,
            "p50": _try_float(row[3]) if len(row) > 3 else None,
            "p95": _try_float(row[4]) if len(row) > 4 else None,
            "p_gt_0": row[5].strip() if len(row) > 5 else None,
        }
    return out


def parse_regime(path: Path) -> list[dict]:
    rows = _read_csv_rows(path)
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    out = []
    for row in rows[1:]:
        if not row:
            continue
        out.append({h: (row[i] if i < len(row) else "") for i, h in enumerate(header)})
    return out


def parse_stress(path: Path) -> list[dict]:
    return parse_regime(path)  # same structure


# ---------------------------------------------------------------------------
# Deflated Sharpe with multi-trial correction
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Beasley-Springer/Moro approximation
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return -float("inf")
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


EULER_MASCHERONI = 0.5772156649015329


def expected_max_sr(n_trials: int, var_sr_across_trials: float) -> float:
    """Bailey & Lopez de Prado (2014) expected max SR under H0.

    E[max SR] ~= sqrt(Var(SR)) * ((1 - gamma) * Z^{-1}(1 - 1/N) + gamma * Z^{-1}(1 - 1/(N*e)))
    """
    if n_trials <= 1:
        return 0.0
    sigma = math.sqrt(max(var_sr_across_trials, 0.0))
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe(sr_hat: float, T: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    var_sr: float | None = None) -> float:
    """Probability the true Sharpe exceeds the null-adjusted threshold.

    Uses PSR formulation from Bailey-Lopez de Prado (2014).
    ``var_sr`` is the cross-trial variance of Sharpes. If unknown, a
    reasonable default of 0.5 is used (the baseline DSR already reported
    at n_trials=1 implicitly assumes variance ~ 1 / T).
    """
    if var_sr is None:
        # fall back to 1/T asymptotic variance of Sharpe under iid normal
        var_sr = max(1.0 / max(T, 1), 1e-9)
    sr0 = expected_max_sr(n_trials, var_sr)
    denom = math.sqrt(max((1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2) / max(T - 1, 1), 1e-12))
    z = (sr_hat - sr0) / denom
    return _norm_cdf(z)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _yn(flag: bool) -> str:
    return "Y" if flag else "N"


def post_process() -> dict:
    print("[post] Parsing result CSVs...")
    tear = parse_tearsheet(RESULTS_DIR / "tearsheet.csv")
    ff = parse_fama_french(RESULTS_DIR / "fama_french.csv")
    stress = parse_stress(RESULTS_DIR / "stress_test.csv")
    regime = parse_regime(RESULTS_DIR / "regime_performance.csv")
    oos = parse_oos(RESULTS_DIR / "oos_tearsheet.csv")
    boot = parse_bootstrap(RESULTS_DIR / "bootstrap_ci.csv")

    cagr = _percent_to_float(tear.get("CAGR", ""))
    spy_cagr = _percent_to_float(tear.get("SPY CAGR", ""))
    sharpe = _try_float(tear.get("Sharpe Ratio", ""))
    max_dd = _percent_to_float(tear.get("Max Drawdown", ""))
    dsr_1 = _try_float(tear.get("Deflated Sharpe (DSR)", ""))
    psr = _try_float(tear.get("Probabilistic Sharpe (PSR)", ""))

    # Approx T from "Total Return" history is unavailable in tearsheet; use
    # the OOS tearsheet which has Days.
    T = None
    for key, entry in oos.items():
        try:
            d = int(float(entry.get("Days", "0")))
            T = (T or 0) + d
        except (ValueError, TypeError):
            pass
    if not T:
        # fallback: 2013-01-01 to ~2026-03-01, 252 * 13
        T = 252 * 13

    # Multi-trial DSR correction (n=15).
    dsr_15 = None
    if sharpe is not None:
        # Daily Sharpe basis for PSR formula
        sr_daily = sharpe / math.sqrt(252.0)
        # variance of Sharpe across trials — use conservative default
        var_sr = 1.0 / max(T, 1)
        dsr_15 = deflated_sharpe(sr_daily, T=T, n_trials=15,
                                 skew=0.0, kurt=3.0, var_sr=var_sr)

    # Alpha p-value from FF6
    alpha_p = None
    if "Alpha" in ff:
        alpha_p = ff["Alpha"].get("p")
    rmw_coef = None
    if "RMW" in ff:
        rmw_coef = ff["RMW"].get("coef")

    # OOS vs IS Sharpe
    is_sharpe = None
    oos_sharpe = None
    for key, entry in oos.items():
        s = _try_float(entry.get("Sharpe", ""))
        k = key.lower().replace("_", "-").replace(" ", "-")
        if k in ("in-sample", "insample"):
            is_sharpe = s
        elif k in ("out-of-sample", "outofsample", "oos"):
            oos_sharpe = s
    if oos and (is_sharpe is None or oos_sharpe is None):
        print(f"[post] WARN: could not parse IS/OOS sharpe from keys={list(oos.keys())} "
              f"(is={is_sharpe}, oos={oos_sharpe})")

    checks: list[tuple[str, bool | None, str]] = []
    checks.append(("CAGR > SPY",
                   (cagr is not None and spy_cagr is not None and cagr > spy_cagr),
                   f"{(cagr or 0)*100:.2f}% vs SPY {(spy_cagr or 0)*100:.2f}%"))
    checks.append(("Sharpe > 0.60",
                   (sharpe is not None and sharpe > 0.60),
                   f"{sharpe:.3f}" if sharpe is not None else "n/a"))
    checks.append(("DSR (n=1) > 0.95",
                   (dsr_1 is not None and dsr_1 > 0.95),
                   f"{dsr_1:.3f}" if dsr_1 is not None else "n/a"))
    checks.append(("DSR (n=15 corrected) > 0.80",
                   (dsr_15 is not None and dsr_15 > 0.80),
                   f"{dsr_15:.3f}" if dsr_15 is not None else "n/a"))
    checks.append(("Alpha p-value < 0.10",
                   (alpha_p is not None and alpha_p < 0.10),
                   f"p={alpha_p:.4f}" if alpha_p is not None else "n/a"))
    checks.append(("MaxDD > -40%",
                   (max_dd is not None and max_dd > -0.40),
                   f"{(max_dd or 0)*100:.2f}%"))
    checks.append(("OOS Sharpe > Full-sample Sharpe",
                   (oos_sharpe is not None and sharpe is not None and oos_sharpe > sharpe),
                   f"IS={is_sharpe} OOS={oos_sharpe} Full={sharpe}"))
    checks.append(("FF6 RMW loading > -0.20",
                   (rmw_coef is not None and rmw_coef > -0.20),
                   f"{rmw_coef:.3f}" if rmw_coef is not None else "n/a"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    score = passed / total if total else 0.0

    metrics = {
        "cagr": cagr,
        "spy_cagr": spy_cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "dsr_n1": dsr_1,
        "dsr_n15": dsr_15,
        "psr": psr,
        "alpha_p_value": alpha_p,
        "rmw_loading": rmw_coef,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "T_days": T,
    }

    return {
        "tear": tear,
        "ff": ff,
        "stress": stress,
        "regime": regime,
        "oos": oos,
        "bootstrap": boot,
        "metrics": metrics,
        "checks": checks,
        "score": score,
        "passed": passed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(analysis: dict, timestamp: str, dry_run: bool) -> None:
    tear = analysis["tear"]
    metrics = analysis["metrics"]
    checks = analysis["checks"]
    passed = analysis["passed"]
    total = analysis["total"]
    score = analysis["score"]

    lines: list[str] = []
    lines.append("# Phase 5 Production Summary")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Mode: {'DRY RUN (re-processed existing results)' if dry_run else 'FULL PRODUCTION RUN'}")
    lines.append(f"- Config phase: 1.8d (locked)")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    headline_keys = [
        "CAGR", "SPY CAGR", "Alpha vs SPY", "Annualized Volatility",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown",
        "Beta to Market", "Probabilistic Sharpe (PSR)",
        "Deflated Sharpe (DSR)", "DSR n_trials",
    ]
    for k in headline_keys:
        if k in tear:
            lines.append(f"- **{k}**: {tear[k]}")
    lines.append("")
    lines.append("## Multi-Trial Deflated Sharpe (Bailey & Lopez de Prado 2014)")
    lines.append("")
    lines.append(f"- Trials assumed across all phases: **15**")
    lines.append(f"- DSR @ n_trials=1 (reported): **{metrics['dsr_n1']}**")
    lines.append(f"- DSR @ n_trials=15 (corrected): **{metrics['dsr_n15']:.4f}**"
                 if metrics['dsr_n15'] is not None else "- DSR @ n_trials=15: n/a")
    lines.append(f"- Sample length T = {metrics['T_days']} days")
    lines.append("")
    lines.append("## Fama-French 6-Factor Regression")
    lines.append("")
    ff = analysis["ff"]
    if ff:
        lines.append("| Factor | Coef | t-stat | p-value |")
        lines.append("|---|---:|---:|---:|")
        for name, entry in ff.items():
            coef = entry.get("coef")
            t = entry.get("t")
            p = entry.get("p")
            lines.append(f"| {name} | {coef:.4f} | {t:.2f} | {p:.4f} |"
                         if coef is not None else f"| {name} | n/a | n/a | n/a |")
    lines.append("")
    lines.append("## OOS vs In-Sample")
    lines.append("")
    oos = analysis["oos"]
    for k, entry in oos.items():
        lines.append(f"- **{k}**: Sharpe={entry.get('Sharpe','?')}, "
                     f"Ann Ret={entry.get('Ann. Return','?')}, "
                     f"MaxDD={entry.get('Max DD','?')}, Beta={entry.get('Beta','?')}")
    lines.append("")
    lines.append("## Stress Tests")
    lines.append("")
    for row in analysis["stress"]:
        lines.append(f"- **{row.get('period','?')}** "
                     f"[{row.get('start','?')} -> {row.get('end','?')}]: "
                     f"ret={row.get('total_return','?')}, "
                     f"dd={row.get('max_drawdown','?')}, "
                     f"vs bench={row.get('vs_benchmark','?')}")
    lines.append("")
    lines.append("## Regime Performance")
    lines.append("")
    for row in analysis["regime"]:
        lines.append(f"- **{row.get('regime','?')}** "
                     f"({row.get('pct_time','?')} of time): "
                     f"Sharpe={row.get('sharpe','?')}, "
                     f"ann_ret={row.get('ann_return','?')}, "
                     f"active={row.get('active_return','?')}")
    lines.append("")
    lines.append("## Bootstrap 90% Confidence Intervals")
    lines.append("")
    def _fmt(v):
        return "n/a" if v is None else v
    for name, entry in analysis["bootstrap"].items():
        lines.append(f"- **{name}**: observed={_fmt(entry['observed'])}, "
                     f"p5={_fmt(entry['p5'])}, p95={_fmt(entry['p95'])}, "
                     f"P(>0)={_fmt(entry['p_gt_0'])}")
    lines.append("")
    lines.append("## Production Readiness Checklist")
    lines.append("")
    lines.append("| # | Check | Pass | Detail |")
    lines.append("|---|---|:---:|---|")
    for i, (name, ok, detail) in enumerate(checks, start=1):
        lines.append(f"| {i} | {name} | {_yn(bool(ok))} | {detail} |")
    lines.append("")
    lines.append(f"**Score: {passed}/{total} ({score*100:.0f}%)**")
    lines.append("")
    verdict = "GO — production ready" if score >= 0.75 else \
              ("CAUTION — marginal, review failures" if score >= 0.5 else
               "NO-GO — too many failing gates")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(pretty_command(build_reproduction_command()))
    lines.append("```")
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[post] Wrote {SUMMARY_PATH}")


def write_config(analysis: dict, timestamp: str) -> None:
    metrics = analysis["metrics"]

    def _jsonable(v):
        if v is None:
            return None
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        return v

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False,
        ).stdout.strip() or None
    except Exception:
        sha = None
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False,
        ).stdout
        git_clean = (porcelain.strip() == "")
    except Exception:
        git_clean = None

    payload = {
        "phase": "1.8d",
        "locked_date": timestamp,
        "git_commit_sha": sha,
        "git_clean": git_clean,
        "metrics": {k: _jsonable(v) for k, v in metrics.items()},
        "flags": LOCKED_FLAGS,
        "reproduction_command": pretty_command(build_reproduction_command()),
        "readiness": {
            "passed": analysis["passed"],
            "total": analysis["total"],
            "score": analysis["score"],
            "checks": [
                {"name": n, "pass": bool(ok), "detail": d}
                for (n, ok, d) in analysis["checks"]
            ],
        },
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[post] Wrote {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 production orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip backtest, re-post-process existing results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 72)
    print(f"Phase 5 production orchestrator  [{timestamp}]")
    print("=" * 72)

    preflight(dry_run=args.dry_run)

    if not args.dry_run:
        try:
            archive_existing_results(timestamp)
        except Exception as exc:  # noqa: BLE001
            print(f"[archive] FATAL: {exc}")
            return 2

        argv = build_reproduction_command()
        rc = run_backtest(argv)
        if rc != 0:
            print(f"[main] Backtest failed (exit {rc}); skipping post-process.")
            print(f"[main] Archive preserved at results/archive/phase1_8d_baseline_{timestamp}/")
            return rc
    else:
        print("[main] DRY RUN — skipping archive + backtest, post-processing existing results/.")

    analysis = post_process()
    write_summary(analysis, timestamp, dry_run=args.dry_run)
    write_config(analysis, timestamp)

    print("")
    print("=" * 72)
    print(f"Done. Passed {analysis['passed']}/{analysis['total']} gates.")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Config : {CONFIG_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
