"""Thin Stage 3 wrapper around the schema-level fill blotter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from .schemas import canonicalize_utc_timestamp, normalize_ticker, read_jsonl_models, utc_now_iso
from .schemas.fill_blotter import (
    DEFAULT_FILL_BLOTTER_PATH,
    FillBlotter as SchemaFillBlotter,
    FillBlotterRow,
)
from .schemas.order_blotter import OrderSide
from .schemas.reconciliation import SlippageSummary


def _coerce_uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _coerce_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    return canonicalize_utc_timestamp(value)


def _coerce_side(value: OrderSide | str | None) -> OrderSide | None:
    if value is None:
        return None
    return value if isinstance(value, OrderSide) else OrderSide(str(value))


@dataclass(frozen=True)
class FillRecord:
    """Validated fill record wrapper for reporting and reconciliation helpers."""

    fill_id: UUID
    order_id: UUID
    parent_intent_hash: str
    symbol: str
    side: OrderSide | str
    qty: float
    price: float
    commission: float = 0.0
    timestamp: str = field(default_factory=utc_now_iso)
    broker_fill_id: str | None = None
    expected_price: float | None = None
    slippage_bps: float | None = None
    expected_vs_actual_notional: float | None = None
    supersedes: UUID | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        row = FillBlotterRow.model_validate(asdict(self))
        normalized = row.model_dump(mode="python", round_trip=True)
        for key, value in normalized.items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_row(cls, row: FillBlotterRow | dict[str, Any]) -> "FillRecord":
        validated = row if isinstance(row, FillBlotterRow) else FillBlotterRow.model_validate(row)
        return cls(**validated.model_dump(mode="python", round_trip=True))

    def to_row(self) -> FillBlotterRow:
        return FillBlotterRow.model_validate(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return self.to_row().model_dump(mode="json", round_trip=True)

    @property
    def gross_notional(self) -> float:
        return self.qty * self.price

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == OrderSide.BUY else -self.qty

    @property
    def net_cash_flow(self) -> float:
        if self.side == OrderSide.BUY:
            return -(self.gross_notional + self.commission)
        return self.gross_notional - self.commission

    @property
    def side_adjusted_slippage_notional(self) -> float | None:
        if self.expected_price is None:
            return None
        raw = (self.price - self.expected_price) * self.qty
        return raw if self.side == OrderSide.BUY else -raw

    @property
    def side_adjusted_slippage_bps(self) -> float | None:
        if self.expected_price is None:
            return None
        expected_notional = self.expected_price * self.qty
        if expected_notional <= 0.0:
            return None
        slippage_notional = self.side_adjusted_slippage_notional
        if slippage_notional is None:
            return None
        return (slippage_notional / expected_notional) * 10000.0


class FillBlotter:
    """Thin wrapper around the schema ledger with reporting helpers."""

    def __init__(self, path: str | Path = DEFAULT_FILL_BLOTTER_PATH) -> None:
        self.path = Path(path)
        self._ledger = SchemaFillBlotter(self.path)

    def append_fill(
        self,
        row: FillRecord | FillBlotterRow | dict[str, Any],
        submission_timestamp: str | None = None,
    ) -> FillRecord:
        record = self._coerce_record(row)
        if submission_timestamp is not None:
            submitted_at = canonicalize_utc_timestamp(submission_timestamp)
            if record.timestamp < submitted_at:
                raise ValueError(
                    "Fill timestamp cannot be earlier than order submission time: "
                    f"{record.timestamp} < {submitted_at}"
                )
        stored = self._ledger.append_fill(record.to_row())
        return FillRecord.from_row(stored)

    def append_fills(
        self,
        rows: Iterable[FillRecord | FillBlotterRow | dict[str, Any]],
        submission_timestamp: str | None = None,
    ) -> list[FillRecord]:
        return [self.append_fill(row, submission_timestamp=submission_timestamp) for row in rows]

    def get_fill(self, fill_id: UUID | str) -> FillRecord | None:
        fill_uuid = _coerce_uuid(fill_id)
        for row in self._rows():
            if row.fill_id == fill_uuid:
                return FillRecord.from_row(row)
        return None

    def get_fills_by_order(self, order_id: UUID | str) -> list[FillRecord]:
        return [FillRecord.from_row(row) for row in self._ledger.get_fills_by_order(order_id)]

    def get_fills_by_symbol(
        self,
        symbol: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> list[FillRecord]:
        return self.query_fills(
            symbol=symbol,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    def query_fills(
        self,
        order_id: UUID | str | None = None,
        symbol: str | None = None,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
        side: OrderSide | str | None = None,
        broker_fill_id: str | None = None,
    ) -> list[FillRecord]:
        order_uuid = _coerce_uuid(order_id) if order_id is not None else None
        normalized_symbol = normalize_ticker(symbol) if symbol else None
        normalized_start = _coerce_timestamp(start_timestamp)
        normalized_end = _coerce_timestamp(end_timestamp)
        normalized_side = _coerce_side(side)

        records: list[FillRecord] = []
        for row in self._rows():
            if order_uuid is not None and row.order_id != order_uuid:
                continue
            if normalized_symbol is not None and row.symbol != normalized_symbol:
                continue
            if normalized_side is not None and row.side != normalized_side:
                continue
            if broker_fill_id is not None and row.broker_fill_id != broker_fill_id:
                continue
            if normalized_start is not None and row.timestamp < normalized_start:
                continue
            if normalized_end is not None and row.timestamp > normalized_end:
                continue
            records.append(FillRecord.from_row(row))

        records.sort(key=lambda record: (record.timestamp, str(record.fill_id)))
        return records

    def aggregate_fills_for_order(self, order_id: UUID | str) -> dict[str, Any]:
        return self._aggregate(self.get_fills_by_order(order_id), grouping="order", key=str(order_id))

    def aggregate_fills_for_symbol(
        self,
        symbol: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> dict[str, Any]:
        normalized_symbol = normalize_ticker(symbol)
        return self._aggregate(
            self.get_fills_by_symbol(
                normalized_symbol,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            ),
            grouping="symbol",
            key=normalized_symbol,
        )

    def build_slippage_summary(
        self,
        order_id: UUID | str | None = None,
        symbol: str | None = None,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> SlippageSummary:
        records = self.query_fills(
            order_id=order_id,
            symbol=symbol,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        values = [
            record.side_adjusted_slippage_bps
            for record in records
            if record.side_adjusted_slippage_bps is not None
        ]
        return SlippageSummary.from_bps(values)

    def _aggregate(
        self,
        records: list[FillRecord],
        *,
        grouping: str,
        key: str,
    ) -> dict[str, Any]:
        if not records:
            return {
                "grouping": grouping,
                "key": key,
                "fill_count": 0,
                "symbols": [],
                "order_ids": [],
                "total_qty": 0.0,
                "vwap": 0.0,
                "total_notional": 0.0,
                "total_commission": 0.0,
                "net_cash_flow": 0.0,
                "expected_notional": 0.0,
                "side_adjusted_slippage_notional": 0.0,
                "slippage_bps": 0.0,
                "first_fill_timestamp": None,
                "last_fill_timestamp": None,
                "slippage_summary": SlippageSummary.from_bps([]).model_dump(mode="json"),
            }

        total_qty = sum(record.qty for record in records)
        total_notional = sum(record.gross_notional for record in records)
        total_commission = sum(record.commission for record in records)
        net_cash_flow = sum(record.net_cash_flow for record in records)
        expected_notional = sum(
            record.expected_price * record.qty
            for record in records
            if record.expected_price is not None
        )
        slippage_notional = sum(
            record.side_adjusted_slippage_notional or 0.0 for record in records
        )
        slippage_bps = 0.0
        if expected_notional > 0.0:
            slippage_bps = (slippage_notional / expected_notional) * 10000.0

        return {
            "grouping": grouping,
            "key": key,
            "fill_count": len(records),
            "symbols": sorted({record.symbol for record in records}),
            "order_ids": sorted({str(record.order_id) for record in records}),
            "total_qty": total_qty,
            "vwap": total_notional / total_qty,
            "total_notional": total_notional,
            "total_commission": total_commission,
            "net_cash_flow": net_cash_flow,
            "expected_notional": expected_notional,
            "side_adjusted_slippage_notional": slippage_notional,
            "slippage_bps": slippage_bps,
            "first_fill_timestamp": min(record.timestamp for record in records),
            "last_fill_timestamp": max(record.timestamp for record in records),
            "slippage_summary": SlippageSummary.from_bps(
                [
                    record.side_adjusted_slippage_bps
                    for record in records
                    if record.side_adjusted_slippage_bps is not None
                ]
            ).model_dump(mode="json"),
        }

    def _coerce_record(
        self, row: FillRecord | FillBlotterRow | dict[str, Any]
    ) -> FillRecord:
        if isinstance(row, FillRecord):
            return row
        if isinstance(row, FillBlotterRow):
            return FillRecord.from_row(row)
        return FillRecord.from_row(row)

    def _rows(self) -> list[FillBlotterRow]:
        return read_jsonl_models(self.path, FillBlotterRow)


__all__ = ["FillBlotter", "FillRecord"]
