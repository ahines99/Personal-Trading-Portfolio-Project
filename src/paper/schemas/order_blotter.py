"""Append-only order blotter for paper trading."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import (
    SCHEMA_VERSION,
    append_jsonl_record,
    canonicalize_utc_timestamp,
    normalize_ticker,
    read_jsonl_models,
    utc_now_iso,
)

DEFAULT_ORDER_BLOTTER_PATH = Path("paper_trading/blotter/orders.jsonl")


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PREVIEW_PENDING = "PREVIEW_PENDING"
    PREVIEW_APPROVED = "PREVIEW_APPROVED"
    PREVIEW_REJECTED = "PREVIEW_REJECTED"
    APPROVAL_PENDING = "APPROVAL_PENDING"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


_TERMINAL_STATUSES = {
    OrderStatus.PREVIEW_REJECTED,
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
}

_ALLOWED_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PREVIEW_PENDING: {
        OrderStatus.PREVIEW_APPROVED,
        OrderStatus.PREVIEW_REJECTED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PREVIEW_APPROVED: {
        OrderStatus.APPROVAL_PENDING,
        OrderStatus.APPROVED,
        OrderStatus.SUBMITTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.PREVIEW_REJECTED: set(),
    OrderStatus.APPROVAL_PENDING: {
        OrderStatus.APPROVED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.APPROVED: {
        OrderStatus.SUBMITTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


class OrderBlotterRow(BaseModel):
    """One immutable order-log row in the append-only blotter."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.order_blotter.v1"},
    )

    order_id: UUID
    rebalance_id: UUID
    timestamp: str = Field(default_factory=utc_now_iso)
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0.0)
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None
    duration: str = "day"
    status: OrderStatus = OrderStatus.PREVIEW_PENDING
    broker_order_id: str | None = None
    preview_result: dict[str, Any] | None = None
    approval_record: dict[str, Any] | None = None
    submission_timestamp: str | None = None
    terminal_timestamp: str | None = None
    parent_intent_hash: str
    supersedes: UUID | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator(
        "timestamp",
        "submission_timestamp",
        "terminal_timestamp",
        "created_at",
        mode="before",
    )
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> Any:
        if value is None:
            return None
        return canonicalize_utc_timestamp(value)


class OrderBlotter:
    """Append-only JSONL ledger for order lifecycle tracking."""

    def __init__(self, path: str | Path = DEFAULT_ORDER_BLOTTER_PATH) -> None:
        self.path = Path(path)

    def append_order(self, row: OrderBlotterRow | dict[str, Any]) -> OrderBlotterRow:
        record = row if isinstance(row, OrderBlotterRow) else OrderBlotterRow.model_validate(row)
        append_jsonl_record(self.path, record.model_dump(mode="json", round_trip=True))
        return record

    def get_order(self, order_id: UUID | str) -> OrderBlotterRow | None:
        order_uuid = UUID(str(order_id))
        matches = [row for row in self._rows() if row.order_id == order_uuid]
        return matches[-1] if matches else None

    def get_order_history(self, order_id: UUID | str) -> list[OrderBlotterRow]:
        order_uuid = UUID(str(order_id))
        return [row for row in self._rows() if row.order_id == order_uuid]

    def get_orders_by_status(
        self, status: OrderStatus | str, latest_only: bool = True
    ) -> list[OrderBlotterRow]:
        target_status = OrderStatus(status)
        rows = self._rows()
        if not latest_only:
            return [row for row in rows if row.status == target_status]

        latest: dict[UUID, OrderBlotterRow] = {}
        for row in rows:
            latest[row.order_id] = row
        return [row for row in latest.values() if row.status == target_status]

    def update_status(
        self,
        order_id: UUID | str,
        new_status: OrderStatus | str,
        updates: dict[str, Any] | None = None,
    ) -> OrderBlotterRow:
        current = self.get_order(order_id)
        if current is None:
            raise KeyError(f"Unknown order_id: {order_id}")

        target_status = OrderStatus(new_status)
        if target_status not in _ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(
                f"Illegal order status transition: {current.status} -> {target_status}"
            )

        payload = current.model_dump(mode="python", round_trip=True)
        payload["status"] = target_status
        payload["timestamp"] = utc_now_iso()
        if updates:
            payload.update(updates)

        if target_status == OrderStatus.SUBMITTED and not payload.get("submission_timestamp"):
            payload["submission_timestamp"] = utc_now_iso()
        if target_status in _TERMINAL_STATUSES and not payload.get("terminal_timestamp"):
            payload["terminal_timestamp"] = utc_now_iso()

        return self.append_order(payload)

    def _rows(self) -> list[OrderBlotterRow]:
        return read_jsonl_models(self.path, OrderBlotterRow)
