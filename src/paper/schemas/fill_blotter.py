"""Append-only fill blotter for paper trading."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import (
    SCHEMA_VERSION,
    append_jsonl_record,
    canonicalize_utc_timestamp,
    normalize_ticker,
    read_jsonl_models,
    utc_now_iso,
)
from .order_blotter import OrderSide

DEFAULT_FILL_BLOTTER_PATH = Path("paper_trading/blotter/fills.jsonl")


class FillBlotterRow(BaseModel):
    """One immutable fill row in the append-only fill blotter."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.fill_blotter.v1"},
    )

    fill_id: UUID
    order_id: UUID
    broker_fill_id: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    expected_price: float | None = Field(default=None, gt=0.0)
    slippage_bps: float | None = None
    expected_vs_actual_notional: float | None = None
    parent_intent_hash: str
    supersedes: UUID | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("timestamp", "created_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        return canonicalize_utc_timestamp(value)

    @model_validator(mode="after")
    def _derive_slippage_fields(self) -> "FillBlotterRow":
        if self.expected_price is None:
            return self

        raw_bps = ((self.price / self.expected_price) - 1.0) * 10000.0
        raw_notional = (self.price - self.expected_price) * self.qty

        if self.slippage_bps is None:
            object.__setattr__(self, "slippage_bps", raw_bps)
        if self.expected_vs_actual_notional is None:
            object.__setattr__(self, "expected_vs_actual_notional", raw_notional)
        return self


class FillBlotter:
    """Append-only JSONL ledger for fill tracking and aggregation."""

    def __init__(self, path: str | Path = DEFAULT_FILL_BLOTTER_PATH) -> None:
        self.path = Path(path)

    def append_fill(self, row: FillBlotterRow | dict[str, Any]) -> FillBlotterRow:
        record = row if isinstance(row, FillBlotterRow) else FillBlotterRow.model_validate(row)
        append_jsonl_record(self.path, record.model_dump(mode="json", round_trip=True))
        return record

    def get_fills_by_order(self, order_id: UUID | str) -> list[FillBlotterRow]:
        order_uuid = UUID(str(order_id))
        return [row for row in self._rows() if row.order_id == order_uuid]

    def aggregate_fills_for_order(self, order_id: UUID | str) -> dict[str, float]:
        fills = self.get_fills_by_order(order_id)
        if not fills:
            return {
                "total_qty": 0.0,
                "vwap": 0.0,
                "total_commission": 0.0,
                "expected_vs_actual_notional": 0.0,
                "slippage_bps": 0.0,
            }

        total_qty = sum(fill.qty for fill in fills)
        total_notional = sum(fill.qty * fill.price for fill in fills)
        total_commission = sum(fill.commission for fill in fills)
        expected_vs_actual_notional = sum(
            fill.expected_vs_actual_notional or 0.0 for fill in fills
        )
        expected_notional = sum(
            fill.qty * fill.expected_price
            for fill in fills
            if fill.expected_price is not None
        )
        slippage_bps = 0.0
        if expected_notional:
            slippage_bps = (expected_vs_actual_notional / expected_notional) * 10000.0

        return {
            "total_qty": total_qty,
            "vwap": total_notional / total_qty,
            "total_commission": total_commission,
            "expected_vs_actual_notional": expected_vs_actual_notional,
            "slippage_bps": slippage_bps,
        }

    def _rows(self) -> list[FillBlotterRow]:
        return read_jsonl_models(self.path, FillBlotterRow)
