"""Mock broker implementation for Stage 2 tests and dry integration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time
from typing import Sequence

from .interface import (
    TERMINAL_ORDER_STATUSES,
    BrokerBalances,
    BrokerClient,
    BrokerPositionRecord,
    BrokerProfile,
    normalize_equity_order_request,
)


class MockBrokerClient(BrokerClient):
    """Simple fixture-backed broker client with no external dependencies."""

    def __init__(
        self,
        profile: BrokerProfile,
        balances: BrokerBalances,
        positions: Sequence[BrokerPositionRecord],
        fill_after_polls: int = 3,
        starting_order_id: int = 100000,
        **_: object,
    ) -> None:
        self._profile = deepcopy(dict(profile))
        self._balances = deepcopy(dict(balances))
        self._positions = [deepcopy(dict(position)) for position in positions]
        self._fill_after_polls = max(1, int(fill_after_polls))
        self._next_preview_id = 1
        self._next_order_id = max(1, int(starting_order_id))
        self._orders: dict[str, dict[str, object]] = {}

    def ping(self) -> bool:
        """Return a healthy connection state for tests and local dry runs."""
        return True

    def get_profile(self) -> BrokerProfile:
        """Return the configured broker profile without mutating internal state."""
        return deepcopy(self._profile)

    def get_balances(self) -> BrokerBalances:
        """Return the configured broker balances without mutating internal state."""
        return deepcopy(self._balances)

    def get_positions(self) -> list[BrokerPositionRecord]:
        """Return the configured broker positions without mutating internal state."""
        return deepcopy(self._positions)

    def preview_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ) -> dict[str, object]:
        request = normalize_equity_order_request(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )
        preview_id = f"mock-preview-{self._next_preview_id:06d}"
        tag = f"pt-mock-{self._next_preview_id:06d}"
        self._next_preview_id += 1

        reference_price = self._reference_price(request["symbol"])
        execution_price = (
            float(request["limit_price"])
            if request["limit_price"] is not None
            else reference_price
        )
        estimated_cost = round(float(request["qty"]) * execution_price, 4)
        side_is_buy = request["side"] in {"buy", "buy_to_cover"}
        buying_power = float(self._balances.get("buying_power", 0.0))
        result = (not side_is_buy) or estimated_cost <= buying_power
        warnings: list[str] = []
        if side_is_buy and not result:
            warnings.append(
                f"Estimated notional {estimated_cost:.2f} exceeds buying power {buying_power:.2f}"
            )

        return {
            "preview_id": preview_id,
            "status": "ok" if result else "rejected",
            "result": result,
            "symbol": request["symbol"],
            "quantity": request["qty"],
            "side": request["side"],
            "type": request["order_type"],
            "duration": request["duration"],
            "limit_price": request["limit_price"],
            "estimated_cost": estimated_cost,
            "estimated_fees": 0.0,
            "order_cost": estimated_cost,
            "margin_change": round(estimated_cost * 0.5, 4) if side_is_buy else 0.0,
            "warnings": warnings,
            "tag": tag,
            "request_timestamp": _utc_now_iso(),
            "raw": {
                "order": {
                    "status": "ok" if result else "rejected",
                    "result": result,
                    "symbol": request["symbol"],
                    "quantity": request["qty"],
                    "side": request["side"],
                    "type": request["order_type"],
                    "duration": request["duration"],
                    "price": request["limit_price"],
                    "order_cost": estimated_cost,
                    "margin_change": round(estimated_cost * 0.5, 4)
                    if side_is_buy
                    else 0.0,
                    "fees": 0.0,
                    "tag": tag,
                }
            },
        }

    def place_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
        preview_result: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request = normalize_equity_order_request(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )
        preview = _validate_preview_result(preview_result, request)

        broker_order_id = str(self._next_order_id)
        self._next_order_id += 1
        execution_price = (
            float(request["limit_price"])
            if request["limit_price"] is not None
            else self._reference_price(request["symbol"])
        )
        order = {
            "broker_order_id": broker_order_id,
            "status": "pending",
            "symbol": request["symbol"],
            "quantity": request["qty"],
            "filled_quantity": 0.0,
            "remaining_quantity": request["qty"],
            "side": request["side"],
            "type": request["order_type"],
            "duration": request["duration"],
            "limit_price": request["limit_price"],
            "avg_fill_price": None,
            "last_fill_price": None,
            "last_fill_quantity": 0.0,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "tag": preview["tag"],
            "reason": None,
            "execution_price": execution_price,
            "polls_remaining": self._fill_after_polls,
            "preview_id": preview["preview_id"],
        }
        self._orders[broker_order_id] = order

        placed = self.get_order_status(broker_order_id)
        placed["accepted"] = True
        placed["preview_id"] = preview["preview_id"]
        placed["placement_status"] = "ok"
        return placed

    def cancel_order(self, broker_order_id: str) -> dict[str, object]:
        order = self._require_order(broker_order_id)
        if order["status"] not in TERMINAL_ORDER_STATUSES:
            order["status"] = "canceled"
            order["remaining_quantity"] = max(
                float(order["quantity"]) - float(order["filled_quantity"]), 0.0
            )
            order["updated_at"] = _utc_now_iso()
            order["reason"] = "Canceled via MockBrokerClient"
        result = self.get_order_status(broker_order_id)
        result["canceled"] = result["status"] == "canceled"
        return result

    def replace_order(
        self,
        broker_order_id: str,
        *,
        qty: float | None = None,
        limit_price: float | None = None,
    ) -> dict[str, object]:
        order = self._require_order(broker_order_id)
        if order["status"] in TERMINAL_ORDER_STATUSES:
            raise ValueError(
                f"Cannot replace terminal mock order {broker_order_id}: {order['status']}"
            )
        if qty is None and limit_price is None:
            raise ValueError("replace_order requires qty and/or limit_price")

        if qty is not None:
            normalized_qty = float(qty)
            if normalized_qty <= 0:
                raise ValueError("qty must be positive")
            order["quantity"] = normalized_qty
            order["remaining_quantity"] = max(
                normalized_qty - float(order["filled_quantity"]), 0.0
            )

        if limit_price is not None:
            normalized_price = float(limit_price)
            if normalized_price <= 0:
                raise ValueError("limit_price must be positive")
            order["limit_price"] = normalized_price
            order["type"] = "limit"
            order["execution_price"] = normalized_price

        order["status"] = "pending"
        order["polls_remaining"] = self._fill_after_polls
        order["updated_at"] = _utc_now_iso()
        replaced = self.get_order_status(broker_order_id)
        replaced["replaced"] = True
        return replaced

    def get_order_status(self, broker_order_id: str) -> dict[str, object]:
        order = self._require_order(broker_order_id)
        status = str(order["status"])
        return {
            "broker_order_id": str(order["broker_order_id"]),
            "status": status,
            "is_terminal": status in TERMINAL_ORDER_STATUSES,
            "symbol": str(order["symbol"]),
            "quantity": float(order["quantity"]),
            "filled_quantity": float(order["filled_quantity"]),
            "remaining_quantity": float(order["remaining_quantity"]),
            "side": str(order["side"]),
            "type": str(order["type"]),
            "duration": str(order["duration"]),
            "limit_price": order["limit_price"],
            "avg_fill_price": order["avg_fill_price"],
            "last_fill_price": order["last_fill_price"],
            "last_fill_quantity": float(order["last_fill_quantity"]),
            "created_at": str(order["created_at"]),
            "updated_at": str(order["updated_at"]),
            "reason": order["reason"],
            "tag": str(order["tag"]),
            "raw": deepcopy(order),
        }

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        interval = max(0.0, float(poll_interval_seconds))

        while True:
            current = self.get_order_status(broker_order_id)
            if current["is_terminal"]:
                return current
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Mock order {broker_order_id} did not reach a terminal state within "
                    f"{timeout_seconds:.1f}s"
                )
            self._advance_order(broker_order_id)
            if interval:
                time.sleep(interval)

    def _advance_order(self, broker_order_id: str) -> None:
        order = self._require_order(broker_order_id)
        if order["status"] in TERMINAL_ORDER_STATUSES:
            return

        polls_remaining = int(order["polls_remaining"])
        qty = float(order["quantity"])
        execution_price = float(order["execution_price"])

        if polls_remaining <= 1:
            remaining_before_fill = max(qty - float(order["filled_quantity"]), 0.0)
            order["status"] = "filled"
            order["filled_quantity"] = qty
            order["remaining_quantity"] = 0.0
            order["avg_fill_price"] = execution_price
            order["last_fill_price"] = execution_price
            order["last_fill_quantity"] = remaining_before_fill or qty
        elif polls_remaining == self._fill_after_polls:
            order["status"] = "open"
        else:
            partial_fill = min(qty, round(qty * 0.5, 8))
            order["status"] = "partially_filled"
            order["filled_quantity"] = partial_fill
            order["remaining_quantity"] = max(qty - partial_fill, 0.0)
            order["avg_fill_price"] = execution_price
            order["last_fill_price"] = execution_price
            order["last_fill_quantity"] = partial_fill

        order["polls_remaining"] = max(polls_remaining - 1, 0)
        order["updated_at"] = _utc_now_iso()

    def _reference_price(self, symbol: str) -> float:
        normalized_symbol = str(symbol).strip().upper()
        for position in self._positions:
            if str(position.get("ticker", "")).strip().upper() != normalized_symbol:
                continue
            quantity = float(position.get("quantity", 0.0))
            market_value = float(position.get("market_value", 0.0))
            if quantity > 0 and market_value > 0:
                return round(market_value / quantity, 6)
        return 100.0

    def _require_order(self, broker_order_id: str) -> dict[str, object]:
        normalized_order_id = str(broker_order_id).strip()
        if normalized_order_id not in self._orders:
            raise KeyError(f"Unknown mock broker order: {broker_order_id!r}")
        return self._orders[normalized_order_id]


def _validate_preview_result(
    preview_result: dict[str, object] | None,
    request: dict[str, object],
) -> dict[str, object]:
    if preview_result is None:
        raise ValueError("preview_result is required for place_equity_order")
    preview = dict(preview_result)
    if not preview.get("result", False):
        raise ValueError("preview_result must indicate a successful preview")

    for key, preview_key in (
        ("symbol", "symbol"),
        ("qty", "quantity"),
        ("side", "side"),
        ("order_type", "type"),
        ("limit_price", "limit_price"),
    ):
        if preview.get(preview_key) != request[key]:
            raise ValueError(
                f"preview_result does not match order request for {key}: "
                f"{preview.get(preview_key)!r} != {request[key]!r}"
            )
    if not str(preview.get("preview_id", "")).strip():
        raise ValueError("preview_result must include a preview_id")
    if not str(preview.get("tag", "")).strip():
        raise ValueError("preview_result must include a tag")
    return preview


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
