from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_signal_path(
    results_dir: Optional[Any] = None,
    config: Any = None,
) -> Path:
    if results_dir:
        candidate = Path(results_dir)
        if not candidate.is_absolute():
            candidate = (_repo_root() / candidate).resolve()
        if candidate.is_file():
            if candidate.name != "final_signal.parquet":
                raise FileNotFoundError(f"Expected final_signal.parquet, got: {candidate}")
            return candidate
        signal_path = candidate / "final_signal.parquet"
        if signal_path.is_file():
            return signal_path
        raise FileNotFoundError(f"Could not find final_signal.parquet under {candidate}")

    baseline_path = _config_value(config, "baseline_path")
    if baseline_path:
        baseline_root = Path(baseline_path)
        if not baseline_root.is_absolute():
            baseline_root = (_repo_root() / baseline_root).resolve()
        signal_path = baseline_root / "final_signal.parquet"
        if signal_path.is_file():
            return signal_path
        raise FileNotFoundError(f"Configured baseline_path is missing final_signal.parquet: {baseline_root}")

    results_root = _repo_root() / "results"
    candidates = sorted(
        results_root.glob("*/final_signal.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not locate any final_signal.parquet under results/")
    return candidates[0]


def _extract_signal_date(signal_path: Path) -> date:
    panel = pd.read_parquet(signal_path)
    if isinstance(panel.index, pd.MultiIndex):
        level = pd.to_datetime(panel.index.get_level_values(0), errors="coerce")
        signal_ts = level.max()
    else:
        signal_ts = pd.to_datetime(panel.index, errors="coerce").max()

    if pd.isna(signal_ts):
        for candidate in ("date", "as_of_date", "timestamp"):
            if candidate in panel.columns:
                signal_ts = pd.to_datetime(panel[candidate], errors="coerce").max()
                if not pd.isna(signal_ts):
                    break

    if pd.isna(signal_ts):
        raise ValueError(f"Could not extract a signal date from {signal_path}")
    return pd.Timestamp(signal_ts).date()


def check_signal_staleness(
    as_of_date: Optional[Any] = None,
    results_dir: Optional[Any] = None,
    config: Any = None,
) -> Dict[str, Any]:
    """Verify the latest final_signal snapshot is fresh enough for paper shadow."""
    signal_path = _resolve_signal_path(results_dir=results_dir, config=config)
    signal_date = _extract_signal_date(signal_path)
    reference_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else pd.Timestamp.today().date()
    max_allowed_days = int(_config_value(config, "signal_max_staleness_days", 7))
    age_days = int((reference_date - signal_date).days)

    payload = {
        "signal_path": str(signal_path),
        "signal_date": signal_date.isoformat(),
        "age_days": age_days,
        "max_allowed_days": max_allowed_days,
        "status": "OK" if age_days <= max_allowed_days else "STALE",
    }
    if age_days > max_allowed_days:
        raise ValueError(
            "final_signal.parquet is stale: "
            f"signal_date={payload['signal_date']}, age_days={age_days}, max_allowed_days={max_allowed_days}"
        )
    return payload
