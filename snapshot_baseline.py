"""snapshot_baseline.py

Capture and lock the current "true baseline" state of the trading-portfolio
project for reproducibility. Solves the recurring "where is the real baseline?"
confusion by writing an immutable, timestamped JSON manifest that records:

    * git HEAD + dirty flag
    * SHA256 of every file under src/ and of run_strategy.py
    * Feature-panel and ML-prediction cache hashes (parsed from the run log)
    * Tearsheet metrics (CAGR, Sharpe, MaxDD, Beta, Vol, AfterTax, Costs)
    * Feature count parsed from the log
    * Relevant environment variables
    * The exact CLI command (if recorded in the log)

Usage
-----
    python snapshot_baseline.py --label v2_recovery
    python snapshot_baseline.py --label v2_recovery --results-dir results/_my_run
    python snapshot_baseline.py --validate v2_recovery

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUN_STRATEGY = PROJECT_ROOT / "run_strategy.py"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "_baseline_recovery_v2"
BASELINES_DIR = PROJECT_ROOT / "baselines"
LOGS_DIR = PROJECT_ROOT / "logs"

RELEVANT_ENV_VARS = (
    "ENABLE_PHASE_D_SIGNALS",
    "EXTRA_DENY_FEATURES",
    "DENY_LIST_HASH_SUFFIX",
)

TEARSHEET_KEYS_OF_INTEREST = {
    "CAGR": "CAGR",
    "Sharpe Ratio": "Sharpe",
    "Max Drawdown": "MaxDD",
    "Beta to Market": "Beta",
    "Annualized Volatility": "Vol",
    "After-Tax CAGR (est.)": "AfterTax",
    "Total Transaction Costs": "Costs",
}


# --------------------------------------------------------------------------- #
# Hashing                                                                     #
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_src_tree(src_dir: Path) -> Dict[str, str]:
    """SHA256 every regular file under src/, keyed by POSIX-style relative path."""
    manifest: Dict[str, str] = {}
    if not src_dir.is_dir():
        return manifest
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src_dir).as_posix()
        manifest[rel] = sha256_file(p)
    return manifest


# --------------------------------------------------------------------------- #
# Git                                                                         #
# --------------------------------------------------------------------------- #
def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state() -> Dict[str, object]:
    head = _git("rev-parse", "HEAD")
    msg = _git("log", "-1", "--pretty=%s")
    porcelain = _git("status", "--porcelain")
    return {
        "head": head,
        "message": msg,
        "dirty": bool(porcelain),
        "dirty_files": porcelain.splitlines() if porcelain else [],
    }


# --------------------------------------------------------------------------- #
# Tearsheet                                                                   #
# --------------------------------------------------------------------------- #
def parse_tearsheet(results_dir: Path) -> Dict[str, str]:
    ts_path = results_dir / "tearsheet.csv"
    if not ts_path.is_file():
        # Fall back to the canonical results/tearsheet.csv if the per-run dir is empty.
        ts_path = PROJECT_ROOT / "results" / "tearsheet.csv"
    out: Dict[str, str] = {"_source": str(ts_path) if ts_path.is_file() else "missing"}
    if not ts_path.is_file():
        return out
    with ts_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            metric, value = row[0].strip(), row[1].strip()
            if metric in TEARSHEET_KEYS_OF_INTEREST:
                out[TEARSHEET_KEYS_OF_INTEREST[metric]] = value
    return out


# --------------------------------------------------------------------------- #
# Log parsing                                                                 #
# --------------------------------------------------------------------------- #
_FEATURE_PANEL_RE = re.compile(r"feature_panel_([0-9a-fA-F]{6,})\.pkl")
_ML_CACHE_RE = re.compile(r"(?:ml_predictions|ml_preds|predictions)_([0-9a-fA-F]{6,})\.pkl")
_SAMPLES_RE = re.compile(r"(\d[\d,]*)\s*samples?\s*x\s*(\d+)\s*features", re.IGNORECASE)
_PYTHON_INVOCATION_RE = re.compile(r"^(?:\$\s*)?(python[\w\.\-]*\s+[^\n]+)$", re.MULTILINE)


def find_log_for(label: str, results_dir: Path) -> Optional[Path]:
    """Best-effort: pick a log whose name matches the results-dir basename or label."""
    if not LOGS_DIR.is_dir():
        return None
    candidates: List[Path] = []
    needles = {results_dir.name.lstrip("_"), label}
    for log in LOGS_DIR.glob("*.log"):
        stem = log.stem.lstrip("_")
        if any(n and n in stem for n in needles):
            candidates.append(log)
    if not candidates:
        return None
    # Prefer the most recently modified candidate.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_log(log_path: Optional[Path]) -> Dict[str, object]:
    info: Dict[str, object] = {
        "log_path": str(log_path) if log_path else None,
        "feature_panel_hash": None,
        "ml_predictions_hash": None,
        "feature_count": None,
        "sample_count": None,
        "cli_command": "unknown",
    }
    if log_path is None or not log_path.is_file():
        return info
    text = log_path.read_text(encoding="utf-8", errors="replace")

    if (m := _FEATURE_PANEL_RE.search(text)) is not None:
        info["feature_panel_hash"] = m.group(1)
    if (m := _ML_CACHE_RE.search(text)) is not None:
        info["ml_predictions_hash"] = m.group(1)
    if (m := _SAMPLES_RE.search(text)) is not None:
        info["sample_count"] = int(m.group(1).replace(",", ""))
        info["feature_count"] = int(m.group(2))
    if (m := _PYTHON_INVOCATION_RE.search(text)) is not None:
        info["cli_command"] = m.group(1).strip()
    return info


# --------------------------------------------------------------------------- #
# Snapshot writer                                                             #
# --------------------------------------------------------------------------- #
def build_manifest(label: str, results_dir: Path) -> Tuple[Dict[str, object], str]:
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = find_log_for(label, results_dir)
    manifest: Dict[str, object] = {
        "label": label,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "git": git_state(),
        "src_hashes": hash_src_tree(SRC_DIR),
        "run_strategy_hash": sha256_file(RUN_STRATEGY) if RUN_STRATEGY.is_file() else None,
        "tearsheet": parse_tearsheet(results_dir),
        "log": parse_log(log_path),
        "env": {k: os.environ.get(k) for k in RELEVANT_ENV_VARS},
    }
    return manifest, timestamp


def write_snapshot(manifest: Dict[str, object], label: str, timestamp: str) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINES_DIR / f"{label}_{timestamp}.json"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {out_path}")
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def latest_snapshot_for(label: str) -> Optional[Path]:
    if not BASELINES_DIR.is_dir():
        return None
    matches = sorted(BASELINES_DIR.glob(f"{label}_*.json"))
    return matches[-1] if matches else None


def validate(label: str) -> int:
    snap_path = latest_snapshot_for(label)
    if snap_path is None:
        print(f"[error] no snapshot found for label '{label}' in {BASELINES_DIR}", file=sys.stderr)
        return 2
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    print(f"[validate] comparing against {snap_path.name}")

    saved_src = snap.get("src_hashes", {}) or {}
    current_src = hash_src_tree(SRC_DIR)
    saved_keys, current_keys = set(saved_src), set(current_src)

    added = sorted(current_keys - saved_keys)
    removed = sorted(saved_keys - current_keys)
    changed = sorted(k for k in saved_keys & current_keys if saved_src[k] != current_src[k])

    saved_rs = snap.get("run_strategy_hash")
    current_rs = sha256_file(RUN_STRATEGY) if RUN_STRATEGY.is_file() else None
    rs_changed = saved_rs != current_rs

    saved_git = (snap.get("git") or {}).get("head")
    current_git = git_state().get("head")

    if not (added or removed or changed or rs_changed):
        print("[validate] OK — src/ tree and run_strategy.py match snapshot.")
        if saved_git and current_git and saved_git != current_git:
            print(f"[validate] note: git HEAD differs ({saved_git[:10]} -> {current_git[:10]})")
        return 0

    print(f"[validate] DRIFT detected vs snapshot taken {snap.get('timestamp')}")
    if rs_changed:
        print(f"  run_strategy.py changed: {saved_rs} -> {current_rs}")
    for k in changed:
        print(f"  modified: src/{k}  ({saved_src[k][:10]} -> {current_src[k][:10]})")
    for k in added:
        print(f"  added:    src/{k}")
    for k in removed:
        print(f"  removed:  src/{k}")
    if saved_git != current_git:
        print(f"  git HEAD: {saved_git} -> {current_git}")
    return 1


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture or validate a reproducibility baseline snapshot."
    )
    parser.add_argument("--label", help="Human-readable label for the snapshot.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory holding the run output (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--validate",
        metavar="LABEL",
        help="Validate current src/ tree against the latest snapshot for LABEL.",
    )
    args = parser.parse_args(argv)

    if args.validate:
        return validate(args.validate)

    if not args.label:
        parser.error("--label is required when not using --validate")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (PROJECT_ROOT / results_dir).resolve()

    manifest, timestamp = build_manifest(args.label, results_dir)
    out_path = write_snapshot(manifest, args.label, timestamp)
    print(f"[snapshot] wrote {out_path}")
    ts = manifest["tearsheet"]
    if ts and any(k in ts for k in TEARSHEET_KEYS_OF_INTEREST.values()):
        bits = [f"{k}={ts[k]}" for k in TEARSHEET_KEYS_OF_INTEREST.values() if k in ts]
        print(f"[snapshot] tearsheet: {', '.join(bits)}")
    log_info = manifest["log"]
    if log_info.get("feature_count"):
        print(f"[snapshot] features={log_info['feature_count']} samples={log_info.get('sample_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
