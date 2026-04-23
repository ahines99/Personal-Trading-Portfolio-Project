"""Paper-trading schema models and append-only ledger helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

SCHEMA_VERSION = "paper_trading_schema_v1"
JSONL_ENCODING = "utf-8"
_LOCK_BYTES = 1

ModelT = TypeVar("ModelT", bound=BaseModel)


def utc_now_iso() -> str:
    """Return a canonical UTC timestamp with a trailing Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def canonicalize_utc_timestamp(value: Any) -> str:
    """Normalize datetimes and ISO strings into canonical UTC Z form."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_ticker(value: str) -> str:
    """Normalize ticker-like identifiers to uppercase ASCII text."""
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Ticker cannot be empty.")
    return normalized


def _lock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, _LOCK_BYTES)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl_record(path: str | Path, payload: dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file under a process-safe lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode(JSONL_ENCODING)

    with target.open("a+b") as handle:
        _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into plain dictionaries."""
    target = Path(path)
    if not target.exists():
        return []

    records: list[dict[str, Any]] = []
    with target.open("r", encoding=JSONL_ENCODING) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def read_jsonl_models(path: str | Path, model_cls: type[ModelT]) -> list[ModelT]:
    """Read and validate a JSONL file as Pydantic models."""
    return [model_cls.model_validate(record) for record in read_jsonl_records(path)]


from .broker_positions import BrokerPosition, BrokerPositionsSnapshot
from .fill_blotter import FillBlotter, FillBlotterRow
from .order_blotter import OrderBlotter, OrderBlotterRow, OrderSide, OrderStatus
from .portfolio_state import PortfolioStateLedger, PortfolioStateRow
from .reconciliation import ReconciliationReport, SlippageSummary, WeightComparison
from .target_book import TargetBookSnapshot, TargetPosition

__all__ = [
    "BrokerPosition",
    "BrokerPositionsSnapshot",
    "FillBlotter",
    "FillBlotterRow",
    "OrderBlotter",
    "OrderBlotterRow",
    "OrderSide",
    "OrderStatus",
    "PortfolioStateLedger",
    "PortfolioStateRow",
    "ReconciliationReport",
    "SCHEMA_VERSION",
    "SlippageSummary",
    "TargetBookSnapshot",
    "TargetPosition",
    "WeightComparison",
    "append_jsonl_record",
    "canonicalize_utc_timestamp",
    "normalize_ticker",
    "read_jsonl_models",
    "read_jsonl_records",
    "utc_now_iso",
]
