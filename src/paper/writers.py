from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional
from uuid import uuid4

import pandas as pd


def _repo_root(explicit_root: Optional[Path] = None) -> Path:
    return Path(explicit_root).resolve() if explicit_root else Path(__file__).resolve().parents[2]


@contextmanager
def _file_lock(path: Path, *, timeout_seconds: float = 10.0, stale_seconds: float = 60.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
            break
        except FileExistsError:
            try:
                age_seconds = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_seconds >= stale_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring lock for {path}")
            time.sleep(0.05)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def atomic_update_text(
    path: Path,
    updater: Callable[[str], str],
    *,
    encoding: str = "utf-8",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        current = path.read_text(encoding=encoding) if path.exists() else ""
        atomic_write_text(path, updater(current), encoding=encoding)


def _atomic_write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.decode("utf-8", errors="replace").strip()


def capture_git_state(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = _repo_root(repo_root)
    porcelain = _run_git(root, "status", "--porcelain")
    dirty_files = porcelain.splitlines() if porcelain else []
    return {
        "head": _run_git(root, "rev-parse", "HEAD"),
        "message": _run_git(root, "log", "-1", "--pretty=%s"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def _coerce_target_weights(target_weights: Any) -> pd.DataFrame:
    if isinstance(target_weights, pd.Series):
        frame = target_weights.dropna().rename("weight").reset_index()
        frame.columns = ["ticker", "weight"]
        return frame

    if isinstance(target_weights, pd.DataFrame):
        frame = target_weights.copy()
        lowered = {str(col).lower(): col for col in frame.columns}
        if "ticker" in lowered:
            ticker_col = lowered["ticker"]
            weight_col = lowered.get("weight", lowered.get("target_weight"))
            if weight_col is None:
                raise ValueError("target_weights is missing a weight column")
            out = frame.loc[:, [ticker_col, weight_col]].copy()
            out.columns = ["ticker", "weight"]
            return out
        if frame.empty:
            return pd.DataFrame(columns=["ticker", "weight"])
        latest = frame.iloc[-1].dropna().rename("weight").reset_index()
        latest.columns = ["ticker", "weight"]
        return latest

    if isinstance(target_weights, Mapping):
        return pd.DataFrame(
            [{"ticker": str(ticker), "weight": weight} for ticker, weight in target_weights.items()]
        )

    raise TypeError(f"Unsupported target_weights type: {type(target_weights)!r}")


def _coerce_intended_trades(intended_trades: Any) -> pd.DataFrame:
    if intended_trades is None:
        return pd.DataFrame()
    if isinstance(intended_trades, pd.DataFrame):
        return intended_trades.copy()
    if isinstance(intended_trades, Mapping):
        return pd.DataFrame([intended_trades])
    if isinstance(intended_trades, Iterable) and not isinstance(intended_trades, (str, bytes)):
        return pd.DataFrame(list(intended_trades))
    raise TypeError(f"Unsupported intended_trades type: {type(intended_trades)!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def write_daily_bundle(
    target_weights: Any,
    intended_trades: Any,
    as_of_date: Any,
    git_state: Optional[Dict[str, Any]],
    repo_root: Optional[Path] = None,
) -> Path:
    """Persist a dated paper-shadow bundle with manifest and checksum file."""
    root = _repo_root(repo_root)
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    bundle_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_dir = results_root / f"_paper_shadow_{bundle_stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    target_df = _coerce_target_weights(target_weights)
    target_df["ticker"] = target_df["ticker"].astype(str)
    target_df["weight"] = pd.to_numeric(target_df["weight"], errors="coerce").fillna(0.0)
    target_df = target_df.sort_values("ticker").reset_index(drop=True)

    intended_df = _coerce_intended_trades(intended_trades)

    target_path = bundle_dir / "target_weights.csv"
    intended_path = bundle_dir / "intended_trades.csv"
    target_df.to_csv(target_path, index=False)
    intended_df.to_csv(intended_path, index=False)

    git_payload = git_state or capture_git_state(root)
    dirty_files = list(git_payload.get("dirty_files") or [])
    target_hash = _sha256_file(target_path)
    intended_hash = _sha256_file(intended_path)

    buy_notional = 0.0
    sell_notional = 0.0
    if "delta_notional" in intended_df.columns:
        delta = pd.to_numeric(intended_df["delta_notional"], errors="coerce").fillna(0.0)
        buy_notional = float(delta[delta > 0].sum())
        sell_notional = float((-delta[delta < 0]).sum())

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of_date).date()),
        "bundle_dir": str(bundle_dir),
        "git": {
            "head": git_payload.get("head"),
            "message": git_payload.get("message"),
            "dirty": bool(git_payload.get("dirty")),
            "dirty_files": dirty_files[:50],
        },
        "hashes": {
            "target_weights": target_hash,
            "intended_trades": intended_hash,
        },
        "stats": {
            "n_target_positions": int((target_df["weight"].abs() > 0).sum()),
            "n_intended_trades": int(len(intended_df)),
            "total_buy_notional": buy_notional,
            "total_sell_notional": sell_notional,
        },
    }

    _atomic_write_text(
        bundle_dir / "manifest.json",
        json.dumps(manifest, indent=2, default=_json_default, sort_keys=True),
    )
    checksums = [
        f"{target_hash}  target_weights.csv",
        f"{intended_hash}  intended_trades.csv",
    ]
    _atomic_write_text(bundle_dir / "checksums.sha256", "\n".join(checksums) + "\n")
    return bundle_dir
