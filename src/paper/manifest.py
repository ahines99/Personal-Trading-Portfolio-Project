from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline_resolver import resolve_baseline_path
from .verify import compute_config_hash


def build_manifest(
    config: Any,
    *,
    config_path: str | Path,
    results_dir: str | Path,
    baseline_path: str | Path | None = None,
    signal_path: str | Path | None = None,
    paper_run_id: str | None = None,
    strategy_version: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = _resolve_repo_root(repo_root)
    resolved_baseline_path = Path(
        baseline_path or resolve_baseline_path(config, repo_root=repo_root_path)
    ).resolve()
    resolved_signal_path = Path(
        signal_path or resolved_baseline_path / "final_signal.parquet"
    ).resolve()
    resolved_config_path = Path(config_path).resolve()
    git_state = collect_git_state(repo_root_path)

    return {
        "run_timestamp": _utc_now_iso(),
        "git_sha": git_state["git_sha"],
        "git_dirty": git_state["git_dirty"],
        "git_dirty_files": git_state["git_dirty_files"],
        "config_hash": compute_config_hash(config),
        "config_path": str(resolved_config_path),
        "strategy_version": strategy_version or resolved_baseline_path.name,
        "baseline_path": str(resolved_baseline_path),
        "input_signal_path": str(resolved_signal_path),
        "input_signal_hash": compute_file_hash(resolved_signal_path),
        "python_version": sys.version.split()[0],
        "pip_freeze_hash": compute_pip_freeze_hash(),
        "stage": _read_attr(config, "stage"),
        "broker": _read_attr(config, "broker"),
        "capital_mode": _read_attr(config, "capital_mode"),
        "mode": _derive_legacy_mode(config),
        "paper_run_id": paper_run_id or str(uuid.uuid4()),
        "results_dir": str(Path(results_dir).resolve()),
    }


def write_manifest(results_dir: str | Path, manifest: dict[str, Any]) -> Path:
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir_path / "manifest.json"

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=results_dir_path,
        prefix="manifest.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)

    os.replace(temp_path, manifest_path)
    return manifest_path


def build_and_write_manifest(
    config: Any,
    *,
    config_path: str | Path,
    results_dir: str | Path,
    baseline_path: str | Path | None = None,
    signal_path: str | Path | None = None,
    paper_run_id: str | None = None,
    strategy_version: str | None = None,
    repo_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest = build_manifest(
        config,
        config_path=config_path,
        results_dir=results_dir,
        baseline_path=baseline_path,
        signal_path=signal_path,
        paper_run_id=paper_run_id,
        strategy_version=strategy_version,
        repo_root=repo_root,
    )
    manifest_path = write_manifest(results_dir, manifest)
    return manifest_path, manifest


def collect_git_state(repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root_path = _resolve_repo_root(repo_root)
    git_sha = _run_git(repo_root_path, "rev-parse", "HEAD")
    status_output = _run_git(repo_root_path, "status", "--porcelain")
    dirty_entries = [
        _parse_porcelain_line(line)
        for line in (status_output or "").splitlines()
        if line.strip()
    ]
    truncated_entries = dirty_entries[:100]
    if len(dirty_entries) > 100:
        truncated_entries.append(f"+{len(dirty_entries) - 100} more")
    return {
        "git_sha": git_sha or "unknown",
        "git_dirty": bool(dirty_entries),
        "git_dirty_files": truncated_entries,
    }


def compute_file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_pip_freeze_hash() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None

    payload = result.stdout.replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _parse_porcelain_line(line: str) -> str:
    if len(line) < 4:
        return line.strip()
    return line[3:].strip()


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_attr(config: Any, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _derive_legacy_mode(config: Any) -> str | None:
    broker = _read_attr(config, "broker")
    if not broker:
        return None
    if broker == "mock":
        return "mock"
    if "sandbox" in str(broker):
        return "sandbox"
    if "live" in str(broker):
        return "live"
    return str(broker)
