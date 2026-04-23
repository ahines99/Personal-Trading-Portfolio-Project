from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


def is_canary_mode_active(trading_day_count: int, *, duration_trading_days: int = 10) -> bool:
    return int(trading_day_count) < int(duration_trading_days)


def load_canary_state(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {
            "trading_day_count": 0,
            "completed_dates": [],
            "last_date": None,
            "active": True,
        }
    return json.loads(target.read_text(encoding="utf-8"))


def save_canary_state(path: str | Path, state: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return target


def record_canary_day(
    path: str | Path,
    *,
    as_of_date: date | str,
    duration_trading_days: int = 10,
) -> dict[str, Any]:
    state = load_canary_state(path)
    day_str = _coerce_date(as_of_date).isoformat()
    completed_dates = list(state.get("completed_dates") or [])
    if day_str not in completed_dates:
        completed_dates.append(day_str)
    trading_day_count = len(completed_dates)
    state.update(
        {
            "trading_day_count": trading_day_count,
            "completed_dates": completed_dates,
            "last_date": day_str,
            "active": is_canary_mode_active(
                trading_day_count,
                duration_trading_days=duration_trading_days,
            ),
        }
    )
    save_canary_state(path, state)
    return state


def apply_canary_caps(
    orders: Iterable[dict[str, Any]],
    *,
    max_single_order_notional_usd: float = 500.0,
    max_orders_per_day: int = 5,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for order in orders:
        order_copy = dict(order)
        notional = abs(_order_notional(order_copy))
        if notional > max_single_order_notional_usd:
            continue
        if len(accepted) >= max_orders_per_day:
            continue
        order_copy["canary_status"] = "accepted"
        accepted.append(order_copy)
    return accepted


def evaluate_canary_orders(
    orders: Iterable[dict[str, Any]],
    *,
    max_single_order_notional_usd: float = 500.0,
    max_orders_per_day: int = 5,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    capped: list[dict[str, Any]] = []

    for order in orders:
        order_copy = dict(order)
        notional = abs(_order_notional(order_copy))
        if notional > max_single_order_notional_usd:
            order_copy["canary_status"] = "canary_capped"
            capped.append(order_copy)
            continue
        if len(accepted) >= max_orders_per_day:
            order_copy["canary_status"] = "queued_next_day"
            deferred.append(order_copy)
            continue
        order_copy["canary_status"] = "accepted"
        accepted.append(order_copy)

    metrics = {
        "orders_submitted": len(accepted),
        "orders_capped": len(capped),
        "orders_deferred": len(deferred),
        "total_notional_submitted": sum(abs(_order_notional(item)) for item in accepted),
        "max_single_notional_submitted": max(
            [abs(_order_notional(item)) for item in accepted] or [0.0]
        ),
        "status": "ok" if not capped and not deferred else "constrained",
    }
    return {
        "accepted_orders": accepted,
        "capped_orders": capped,
        "deferred_orders": deferred,
        "metrics": metrics,
    }


def _order_notional(order: dict[str, Any]) -> float:
    for key in ("delta_notional", "target_notional", "notional", "order_value"):
        value = order.get(key)
        if value is not None:
            return float(value)
    qty = float(order.get("qty") or order.get("quantity") or 0.0)
    price = float(order.get("limit_price") or order.get("price") or order.get("reference_price") or 0.0)
    return qty * price


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


__all__ = [
    "apply_canary_caps",
    "evaluate_canary_orders",
    "is_canary_mode_active",
    "load_canary_state",
    "record_canary_day",
    "save_canary_state",
]
