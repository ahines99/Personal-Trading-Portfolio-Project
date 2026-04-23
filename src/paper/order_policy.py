"""Stage 3 order policy for turning rebalance proposals into broker-ready orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from typing import Any, Iterable, Sequence

from .diff_engine import ProposedTrade, ProposedTrades

_ORDER_TYPE_MARKET = "MARKET"
_ORDER_TYPE_LIMIT = "LIMIT"
_SIDE_BUY = "BUY"
_SIDE_SELL = "SELL"
_NON_EQUITY_TYPES = {"etf", "fund", "index", "future", "futures", "option", "crypto"}


class CapExceededException(ValueError):
    """Raised when the proposed order batch violates a hard order-count cap."""


@dataclass(frozen=True)
class OrderSpec:
    """One executable order derived from a proposed trade."""

    order_spec_id: str
    proposal_id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    limit_price: float | None
    est_price: float
    est_notional: float
    target_weight: float
    current_weight: float
    split_index: int = 1
    split_count: int = 1
    security_type: str = "equity"
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview_payload(self) -> dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "order_type": self.order_type.lower(),
        }
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        return payload

    def as_record(self) -> dict[str, Any]:
        payload = {
            "order_spec_id": self.order_spec_id,
            "proposal_id": self.proposal_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "est_price": self.est_price,
            "est_notional": self.est_notional,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "split_index": self.split_index,
            "split_count": self.split_count,
            "security_type": self.security_type,
            "warnings": list(self.warnings),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OrderPolicyResult:
    """Resolved order batch plus non-blocking policy warnings."""

    orders: tuple[OrderSpec, ...]
    warnings: tuple[str, ...] = ()
    skipped: tuple[dict[str, Any], ...] = ()

    def as_records(self) -> list[dict[str, Any]]:
        return [order.as_record() for order in self.orders]

    @property
    def total_orders(self) -> int:
        return len(self.orders)

    @property
    def total_notional(self) -> float:
        return sum(order.est_notional for order in self.orders)


class OrderPolicy:
    """Transform rebalance proposals into broker-preview-ready order specs."""

    def __init__(
        self,
        max_single_order_notional: float = 50_000.0,
        max_order_count: int = 20,
        qty_rounding: str = "lot",
        order_type: str = _ORDER_TYPE_MARKET,
        *,
        allowed_order_types: Iterable[str] | None = None,
        limit_buffer_bps: float = 200.0,
    ) -> None:
        if max_single_order_notional <= 0:
            raise ValueError("max_single_order_notional must be positive")
        if max_order_count <= 0:
            raise ValueError("max_order_count must be positive")
        normalized_rounding = str(qty_rounding).strip().lower()
        if normalized_rounding not in {"lot", "share", "dollar"}:
            raise ValueError("qty_rounding must be one of: lot, share, dollar")
        normalized_order_type = str(order_type).strip().upper()
        if normalized_order_type not in {_ORDER_TYPE_MARKET, _ORDER_TYPE_LIMIT}:
            raise ValueError("order_type must be MARKET or LIMIT")

        normalized_allowed = {
            str(item).strip().upper()
            for item in (allowed_order_types or [normalized_order_type])
            if str(item).strip()
        }
        if not normalized_allowed:
            raise ValueError("allowed_order_types cannot be empty")
        if normalized_order_type not in normalized_allowed:
            raise ValueError(
                f"Configured order_type {normalized_order_type} is not in allowed_order_types"
            )
        if limit_buffer_bps < 0:
            raise ValueError("limit_buffer_bps cannot be negative")

        self.max_single_order_notional = float(max_single_order_notional)
        self.max_order_count = int(max_order_count)
        self.qty_rounding = normalized_rounding
        self.order_type = normalized_order_type
        self.allowed_order_types = normalized_allowed
        self.limit_buffer_bps = float(limit_buffer_bps)

    def build_order_specs(
        self,
        proposed_trades: ProposedTrades | Sequence[ProposedTrade],
    ) -> OrderPolicyResult:
        proposals = (
            tuple(proposed_trades.proposals)
            if isinstance(proposed_trades, ProposedTrades)
            else tuple(proposed_trades)
        )

        warnings: list[str] = []
        skipped: list[dict[str, Any]] = []
        built_orders: list[OrderSpec] = []

        for proposal in proposals:
            if proposal.security_type.lower() in _NON_EQUITY_TYPES:
                warning = (
                    f"Skipping {proposal.symbol}: security_type={proposal.security_type} "
                    "is not an equity order."
                )
                warnings.append(warning)
                skipped.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "symbol": proposal.symbol,
                        "reason": "non_equity",
                        "security_type": proposal.security_type,
                    }
                )
                continue

            split_quantities = self._split_quantities(proposal)
            if not split_quantities:
                warnings.append(
                    f"Skipping {proposal.symbol}: rounded quantity is below minimum executable size."
                )
                skipped.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "symbol": proposal.symbol,
                        "reason": "rounded_to_zero",
                    }
                )
                continue

            split_count = len(split_quantities)
            for split_index, split_qty in enumerate(split_quantities, start=1):
                est_notional = split_qty * proposal.est_price
                if est_notional > self.max_single_order_notional + 1e-6:
                    raise ValueError(
                        f"{proposal.symbol} split order exceeds max_single_order_notional: "
                        f"{est_notional:.2f} > {self.max_single_order_notional:.2f}"
                    )
                order_spec = OrderSpec(
                    order_spec_id=_build_order_spec_id(
                        proposal_id=proposal.proposal_id,
                        split_index=split_index,
                        split_qty=split_qty,
                    ),
                    proposal_id=proposal.proposal_id,
                    symbol=proposal.symbol,
                    qty=split_qty,
                    side=proposal.side,
                    order_type=self.order_type,
                    limit_price=self._limit_price(
                        side=proposal.side,
                        est_price=proposal.est_price,
                    ),
                    est_price=proposal.est_price,
                    est_notional=est_notional,
                    target_weight=proposal.target_weight,
                    current_weight=proposal.current_weight,
                    split_index=split_index,
                    split_count=split_count,
                    security_type=proposal.security_type,
                    warnings=proposal.warnings,
                    metadata={"adv_participation_pct": proposal.adv_participation_pct, **proposal.metadata},
                )
                built_orders.append(order_spec)

        built_orders.sort(
            key=lambda order: (
                0 if order.side == _SIDE_SELL else 1,
                -order.est_notional,
                order.symbol,
                order.split_index,
            )
        )

        if len(built_orders) > self.max_order_count:
            raise CapExceededException(
                f"Order count cap exceeded: {len(built_orders)} > {self.max_order_count}"
            )

        return OrderPolicyResult(
            orders=tuple(built_orders),
            warnings=tuple(dict.fromkeys(warnings)),
            skipped=tuple(skipped),
        )

    def _split_quantities(self, proposal: ProposedTrade) -> list[float]:
        rounded_total = self._round_quantity(
            raw_qty=proposal.qty,
            est_price=proposal.est_price,
            est_notional=proposal.est_notional,
        )
        if rounded_total < 1.0:
            return []

        max_chunk_qty = self._round_quantity(
            raw_qty=self.max_single_order_notional / proposal.est_price,
            est_price=proposal.est_price,
            est_notional=self.max_single_order_notional,
        )
        if max_chunk_qty < 1.0:
            raise ValueError(
                f"{proposal.symbol} price {proposal.est_price:.2f} is too high for "
                f"max_single_order_notional={self.max_single_order_notional:.2f} under "
                f"{self.qty_rounding} rounding."
            )

        remaining = rounded_total
        chunks: list[float] = []
        while remaining > 0:
            if remaining <= max_chunk_qty:
                candidate = self._round_quantity(
                    raw_qty=remaining,
                    est_price=proposal.est_price,
                    est_notional=remaining * proposal.est_price,
                )
            else:
                candidate = max_chunk_qty
            if candidate < 1.0:
                break
            chunks.append(candidate)
            remaining = max(remaining - candidate, 0.0)
            if remaining < 1.0:
                break

        if not chunks:
            return []

        remainder = rounded_total - sum(chunks)
        if remainder >= 1.0:
            last_chunk = chunks[-1] + remainder
            if (last_chunk * proposal.est_price) <= self.max_single_order_notional + 1e-6:
                chunks[-1] = last_chunk
            else:
                candidate = self._round_quantity(
                    raw_qty=remainder,
                    est_price=proposal.est_price,
                    est_notional=remainder * proposal.est_price,
                )
                if candidate >= 1.0:
                    chunks.append(candidate)

        return [chunk for chunk in chunks if chunk >= 1.0]

    def _round_quantity(
        self,
        *,
        raw_qty: float,
        est_price: float,
        est_notional: float,
    ) -> float:
        if raw_qty <= 0 or est_price <= 0:
            return 0.0

        if self.qty_rounding == "lot":
            return math.floor(raw_qty / 100.0) * 100.0
        if self.qty_rounding == "share":
            return float(math.floor(raw_qty))

        notional = max(math.floor(est_notional), 0)
        if notional <= 0:
            return 0.0
        quantity = notional / est_price
        return math.floor(quantity * 10000.0) / 10000.0

    def _limit_price(self, *, side: str, est_price: float) -> float | None:
        if self.order_type == _ORDER_TYPE_MARKET:
            return None
        if self.order_type not in self.allowed_order_types:
            raise ValueError(f"Order type {self.order_type} is not allowed")
        buffer_multiplier = self.limit_buffer_bps / 10000.0
        if side == _SIDE_BUY:
            limit_price = est_price * (1.0 - buffer_multiplier)
        else:
            limit_price = est_price * (1.0 + buffer_multiplier)
        return round(limit_price, 4)


def _build_order_spec_id(*, proposal_id: str, split_index: int, split_qty: float) -> str:
    digest = sha256()
    digest.update(f"{proposal_id}|{split_index}|{split_qty:.8f}".encode("utf-8"))
    return digest.hexdigest()[:24]
