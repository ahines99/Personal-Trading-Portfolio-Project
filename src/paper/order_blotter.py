"""Execution-friendly wrapper around the append-only order blotter schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .schemas import (
    OrderBlotter as SchemaOrderBlotter,
    OrderBlotterRow,
    OrderSide,
    OrderStatus,
    read_jsonl_models,
    utc_now_iso,
)
from .schemas.order_blotter import DEFAULT_ORDER_BLOTTER_PATH

OrderRecord = OrderBlotterRow


class OrderBlotter:
    """Thin runtime wrapper over the schema-backed append-only order ledger."""

    def __init__(self, path: str | Path = DEFAULT_ORDER_BLOTTER_PATH) -> None:
        self.path = Path(path)
        self._ledger = SchemaOrderBlotter(self.path)

    def create_order(
        self,
        *,
        rebalance_id: UUID | str,
        symbol: str,
        side: OrderSide | str,
        qty: float,
        order_type: str,
        parent_intent_hash: str,
        duration: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        status: OrderStatus | str = OrderStatus.PREVIEW_PENDING,
        order_id: UUID | str | None = None,
        preview_result: dict[str, Any] | None = None,
        approval_record: dict[str, Any] | None = None,
        broker_order_id: str | None = None,
    ) -> OrderRecord:
        """Create and append a new order intent row."""
        payload = {
            "order_id": UUID(str(order_id)) if order_id is not None else uuid4(),
            "rebalance_id": UUID(str(rebalance_id)),
            "symbol": symbol,
            "side": OrderSide(side),
            "qty": float(qty),
            "order_type": str(order_type),
            "duration": str(duration),
            "limit_price": limit_price,
            "stop_price": stop_price,
            "status": OrderStatus(status),
            "preview_result": preview_result,
            "approval_record": approval_record,
            "broker_order_id": broker_order_id,
            "parent_intent_hash": str(parent_intent_hash),
            "timestamp": utc_now_iso(),
        }
        return self.append_order(payload)

    def append_order(self, row: OrderRecord | dict[str, Any]) -> OrderRecord:
        """Append one immutable row using the schema ledger's locked JSONL write."""
        return self._ledger.append_order(row)

    def get_order(
        self,
        order_id: UUID | str,
        *,
        required: bool = False,
    ) -> OrderRecord | None:
        """Return the latest row for an order id."""
        record = self._ledger.get_order(order_id)
        if record is None and required:
            raise KeyError(f"Unknown order_id: {order_id}")
        return record

    def get_order_history(self, order_id: UUID | str) -> list[OrderRecord]:
        """Return the full append history for one order id."""
        return self._ledger.get_order_history(order_id)

    def get_orders_by_status(
        self,
        status: OrderStatus | str,
        *,
        latest_only: bool = True,
    ) -> list[OrderRecord]:
        """Return orders filtered by status."""
        return self._ledger.get_orders_by_status(status, latest_only=latest_only)

    def list_orders(
        self,
        *,
        latest_only: bool = True,
        rebalance_id: UUID | str | None = None,
        symbol: str | None = None,
        status: OrderStatus | str | None = None,
    ) -> list[OrderRecord]:
        """Return latest or full rows with optional execution-facing filters."""
        rows = self._rows()
        if latest_only:
            latest: dict[UUID, OrderRecord] = {}
            for row in rows:
                latest[row.order_id] = row
            rows = list(latest.values())

        if rebalance_id is not None:
            rebalance_uuid = UUID(str(rebalance_id))
            rows = [row for row in rows if row.rebalance_id == rebalance_uuid]
        if symbol is not None:
            needle = str(symbol).strip().upper()
            rows = [row for row in rows if row.symbol == needle]
        if status is not None:
            target_status = OrderStatus(status)
            rows = [row for row in rows if row.status == target_status]
        return rows

    def update_order(self, order_id: UUID | str, **updates: Any) -> OrderRecord:
        """Append a non-status metadata revision for an existing order."""
        current = self.get_order(order_id, required=True)
        assert current is not None

        if "status" in updates:
            target_status = OrderStatus(updates["status"])
            if target_status != current.status:
                raise ValueError("Use update_status() for status transitions.")
            updates = dict(updates)
            updates.pop("status", None)

        payload = current.model_dump(mode="python", round_trip=True)
        payload.update(updates)
        payload["timestamp"] = utc_now_iso()
        return self.append_order(payload)

    def update_status(
        self,
        order_id: UUID | str,
        new_status: OrderStatus | str,
        *,
        updates: dict[str, Any] | None = None,
    ) -> OrderRecord:
        """Append a validated status transition row."""
        return self._ledger.update_status(order_id, new_status, updates=updates)

    def get_status_history(self, order_id: UUID | str) -> list[dict[str, Any]]:
        """Return a compact status/timestamp history for one order."""
        history: list[dict[str, Any]] = []
        for row in self.get_order_history(order_id):
            history.append(
                {
                    "order_id": str(row.order_id),
                    "status": row.status.value,
                    "timestamp": row.timestamp,
                    "broker_order_id": row.broker_order_id,
                    "submission_timestamp": row.submission_timestamp,
                    "terminal_timestamp": row.terminal_timestamp,
                }
            )
        return history

    def attach_preview(
        self,
        order_id: UUID | str,
        preview_result: dict[str, Any],
        *,
        approved: bool | None = None,
    ) -> OrderRecord:
        """Store preview payload and optionally advance preview status."""
        status = None
        if approved is True:
            status = OrderStatus.PREVIEW_APPROVED
        elif approved is False:
            status = OrderStatus.PREVIEW_REJECTED

        if status is None:
            return self.update_order(order_id, preview_result=preview_result)
        return self.update_status(
            order_id,
            status,
            updates={"preview_result": preview_result},
        )

    def attach_approval(
        self,
        order_id: UUID | str,
        approval_record: dict[str, Any],
        *,
        approved: bool | None = None,
    ) -> OrderRecord:
        """Store approval payload and optionally advance approval status."""
        status = None
        if approved is True:
            status = OrderStatus.APPROVED
        elif approved is False:
            status = OrderStatus.REJECTED

        if status is None:
            return self.update_order(order_id, approval_record=approval_record)
        return self.update_status(
            order_id,
            status,
            updates={"approval_record": approval_record},
        )

    def mark_submitted(
        self,
        order_id: UUID | str,
        *,
        broker_order_id: str,
    ) -> OrderRecord:
        """Mark an approved order as submitted with its broker id."""
        return self.update_status(
            order_id,
            OrderStatus.SUBMITTED,
            updates={"broker_order_id": str(broker_order_id)},
        )

    def _rows(self) -> list[OrderRecord]:
        return read_jsonl_models(self.path, OrderRecord)


__all__ = [
    "DEFAULT_ORDER_BLOTTER_PATH",
    "OrderBlotter",
    "OrderRecord",
    "OrderSide",
    "OrderStatus",
]
