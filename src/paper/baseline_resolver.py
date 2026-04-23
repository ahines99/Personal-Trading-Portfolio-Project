from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_CURRENT_BASELINE_RE = re.compile(
    r"Current adopted clean canonical baseline:\s*`?([^`\r\n]+)`?",
    re.IGNORECASE,
)


def resolve_baseline_path(
    config: Any,
    *,
    current_baseline_doc: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> str:
    """Resolve the active baseline directory and verify final_signal.parquet exists."""
    repo_root_path = _resolve_repo_root(repo_root)
    explicit_path = _read_attr(config, "baseline_path")

    if explicit_path:
        candidate = _coerce_path(explicit_path, repo_root_path)
        return _validate_baseline_dir(
            candidate,
            source="config.baseline_path",
        )

    baseline_doc_path = Path(current_baseline_doc or repo_root_path / "CURRENT_BASELINE.md")
    if not baseline_doc_path.exists():
        raise ValueError(
            f"Unable to resolve baseline path: missing {baseline_doc_path}."
        )

    match = _CURRENT_BASELINE_RE.search(
        baseline_doc_path.read_text(encoding="utf-8")
    )
    if not match:
        raise ValueError(
            "Unable to resolve baseline path from CURRENT_BASELINE.md: "
            "missing 'Current adopted clean canonical baseline:' line."
        )

    candidate = _coerce_path(match.group(1).strip(), repo_root_path)
    return _validate_baseline_dir(
        candidate,
        source=str(baseline_doc_path),
    )


def resolve_signal_path(
    config: Any,
    *,
    current_baseline_doc: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    baseline_dir = Path(
        resolve_baseline_path(
            config,
            current_baseline_doc=current_baseline_doc,
            repo_root=repo_root,
        )
    )
    return baseline_dir / "final_signal.parquet"


def _validate_baseline_dir(path: Path, *, source: str) -> str:
    if not path.exists():
        raise ValueError(
            f"Resolved baseline path from {source} does not exist: {path}"
        )
    if not path.is_dir():
        raise ValueError(
            f"Resolved baseline path from {source} is not a directory: {path}"
        )

    signal_path = path / "final_signal.parquet"
    if not signal_path.exists():
        raise ValueError(
            f"Resolved baseline path from {source} is missing final_signal.parquet: "
            f"{signal_path}"
        )

    return str(path)


def _coerce_path(value: str | os.PathLike[str], repo_root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded
    return repo_root / expanded


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def _read_attr(config: Any, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)
