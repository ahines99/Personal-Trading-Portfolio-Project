"""Prior-day reconciliation gate for Stage 3 and Stage 4 runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .fill_blotter import FillBlotter
from .order_blotter import OrderBlotter, OrderStatus

_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
}


@dataclass(frozen=True)
class UnreconciledOrdersException(RuntimeError):
    """Raised when the prior trading day has unresolved orders or fills."""

    unreconciled_order_ids: list[str]
    message: str

    def __str__(self) -> str:
        return self.message


def verify_prior_day_reconciliation(
    *,
    as_of_date: date | str | None = None,
    order_blotter: OrderBlotter | None = None,
    fill_blotter: FillBlotter | None = None,
    repo_root: str | Path | None = None,
    quantity_tolerance_ratio: float = 0.005,
) -> bool:
    """Block a new rebalance when the previous trading day is unresolved."""
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    gate_date = _coerce_date(as_of_date)
    orders_ledger = order_blotter or OrderBlotter(
        root / "paper_trading" / "blotter" / "orders.jsonl"
    )
    fills_ledger = fill_blotter or FillBlotter(
        root / "paper_trading" / "blotter" / "fills.jsonl"
    )

    latest_orders = orders_ledger.list_orders(latest_only=True)
    prior_day = _previous_order_day(latest_orders, gate_date)
    if prior_day is None:
        return True

    prior_orders = [
        order
        for order in latest_orders
        if _coerce_date(order.submission_timestamp or order.timestamp) == prior_day
    ]
    unresolved_ids: list[str] = []
    discrepancies: list[str] = []

    for order in prior_orders:
        if order.status not in _TERMINAL_STATUSES:
            unresolved_ids.append(str(order.order_id))
            continue
        if order.status == OrderStatus.FILLED:
            fills = fills_ledger.get_fills_by_order(order.order_id)
            filled_qty = sum(fill.qty for fill in fills)
            tolerance = max(order.qty * quantity_tolerance_ratio, 1e-6)
            if abs(filled_qty - order.qty) > tolerance:
                unresolved_ids.append(str(order.order_id))
                discrepancies.append(
                    f"{order.order_id}: fills {filled_qty:.6f} vs order {order.qty:.6f}"
                )

    report_path = root / "paper_trading" / "reconciliation" / f"{prior_day.isoformat()}.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(payload.get("reconciliation_ok", False)):
            discrepancies.extend(str(item) for item in payload.get("discrepancies", []))

    if unresolved_ids or discrepancies:
        details = []
        if unresolved_ids:
            details.append("unreconciled order_ids=" + ", ".join(sorted(set(unresolved_ids))))
        if discrepancies:
            details.append("discrepancies=" + " | ".join(discrepancies))
        raise UnreconciledOrdersException(
            unreconciled_order_ids=sorted(set(unresolved_ids)),
            message="Prior-day reconciliation failed: " + "; ".join(details),
        )

    return True


def _previous_order_day(latest_orders: list[Any], gate_date: date) -> date | None:
    candidates = {
        _coerce_date(order.submission_timestamp or order.timestamp)
        for order in latest_orders
        if _coerce_date(order.submission_timestamp or order.timestamp) < gate_date
    }
    if not candidates:
        return None
    return max(candidates)


def _coerce_date(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


__all__ = ["UnreconciledOrdersException", "verify_prior_day_reconciliation"]
