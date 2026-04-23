"""Read-only broker interface for Stage 2 paper trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict

TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "rejected", "expired", "error"}
)
NON_TERMINAL_ORDER_STATUSES = frozenset(
    {"pending", "open", "partially_filled", "pending_cancel"}
)
ALLOWED_EQUITY_ORDER_SIDES = frozenset({"buy", "sell", "sell_short", "buy_to_cover"})
ALLOWED_EQUITY_ORDER_TYPES = frozenset({"market", "limit"})


class BrokerProfile(TypedDict):
    """Normalized broker-account profile returned by `get_profile()`."""

    account_id: str
    account_type: str
    status: str
    user_name: NotRequired[str | None]
    email: NotRequired[str | None]


class BrokerBalances(TypedDict):
    """Normalized broker-balance payload returned by `get_balances()`."""

    account_id: str
    account_type: str
    cash: float
    buying_power: float
    equity: float
    market_value: NotRequired[float]
    total_equity: NotRequired[float]
    maintenance_requirement: NotRequired[float]
    options_buying_power: NotRequired[float]


class BrokerPositionRecord(TypedDict):
    """One normalized broker position returned by `get_positions()`."""

    ticker: str
    quantity: float
    cost_basis: float
    market_value: float
    current_weight: float
    unrealized_pnl: float
    entry_date: NotRequired[str | None]


def normalize_equity_order_request(
    symbol: str,
    qty: float,
    side: str,
    order_type: str,
    limit_price: float | None = None,
) -> dict[str, Any]:
    """Normalize and validate the Stage 3 equity-order request surface."""
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    try:
        normalized_qty = float(qty)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"qty must be numeric, got {qty!r}") from exc
    if normalized_qty <= 0:
        raise ValueError("qty must be positive")

    normalized_side = str(side).strip().lower()
    if normalized_side not in ALLOWED_EQUITY_ORDER_SIDES:
        raise ValueError(
            f"Unsupported equity order side {side!r}; expected one of "
            f"{sorted(ALLOWED_EQUITY_ORDER_SIDES)!r}"
        )

    normalized_type = str(order_type).strip().lower()
    if normalized_type not in ALLOWED_EQUITY_ORDER_TYPES:
        raise ValueError(
            f"Unsupported equity order type {order_type!r}; expected one of "
            f"{sorted(ALLOWED_EQUITY_ORDER_TYPES)!r}"
        )

    normalized_price: float | None = None
    if normalized_type == "limit":
        if limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        try:
            normalized_price = float(limit_price)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"limit_price must be numeric, got {limit_price!r}"
            ) from exc
        if normalized_price <= 0:
            raise ValueError("limit_price must be positive")
    elif limit_price is not None:
        raise ValueError("limit_price is only valid for limit orders")

    return {
        "symbol": normalized_symbol,
        "qty": normalized_qty,
        "side": normalized_side,
        "order_type": normalized_type,
        "limit_price": normalized_price,
        "duration": "day",
    }


class BrokerClient(ABC):
    """Abstract broker surface used by paper-trading execution stages."""

    @abstractmethod
    def ping(self) -> bool:
        """Return `True` when the broker connection and account look healthy."""

    @abstractmethod
    def get_profile(self) -> BrokerProfile:
        """Return a normalized broker-account profile."""

    @abstractmethod
    def get_balances(self) -> BrokerBalances:
        """Return normalized cash, equity, and buying-power balances."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPositionRecord]:
        """Return normalized current positions for the configured account."""

    def preview_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Preview an equity order before submission."""
        # Stage 3
        raise NotImplementedError("Stage 3")

    def place_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
        preview_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit an equity order after preview and approval."""
        # Stage 3
        raise NotImplementedError("Stage 3")

    def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        """Cancel an outstanding broker order."""
        # Stage 3
        raise NotImplementedError("Stage 3")

    def replace_order(
        self,
        broker_order_id: str,
        *,
        qty: float | None = None,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Replace an existing broker order."""
        # Stage 3
        raise NotImplementedError("Stage 3")

    def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        """Fetch the latest broker-reported status for an order."""
        # Stage 3
        raise NotImplementedError("Stage 3")

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Poll an order until it reaches a terminal state."""
        # Stage 3
        raise NotImplementedError("Stage 3")
