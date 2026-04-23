"""Broker-position snapshot schemas for paper trading."""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import SCHEMA_VERSION, canonicalize_utc_timestamp, normalize_ticker, utc_now_iso

_NAV_TOLERANCE = 1e-4


class BrokerPosition(BaseModel):
    """One broker-reported position at a point in time."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.broker_position.v1"},
    )

    ticker: str
    quantity: float
    cost_basis: float
    market_value: float
    current_weight: float
    entry_date: date | None = None
    unrealized_pnl: float

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return normalize_ticker(value)


class BrokerPositionsSnapshot(BaseModel):
    """Immutable broker-account snapshot used for paper reconciliation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.broker_positions.v1"},
    )

    as_of_date: date
    broker_timestamp: str
    cash_balance: float
    equity_value: float
    nav: float
    total_return_pct: float
    gross_exposure: float
    net_exposure: float
    positions: dict[str, BrokerPosition]
    snapshot_id: UUID
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("broker_timestamp", "created_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        return canonicalize_utc_timestamp(value)

    @field_validator("positions")
    @classmethod
    def _normalize_position_keys(
        cls, positions: dict[str, BrokerPosition]
    ) -> dict[str, BrokerPosition]:
        normalized: dict[str, BrokerPosition] = {}
        for ticker, position in positions.items():
            key = normalize_ticker(ticker)
            if key != position.ticker:
                raise ValueError(
                    f"Position key {key!r} does not match payload ticker "
                    f"{position.ticker!r}."
                )
            normalized[key] = position
        return normalized

    @model_validator(mode="after")
    def _validate_nav_identity(self) -> "BrokerPositionsSnapshot":
        market_value_total = sum(position.market_value for position in self.positions.values())
        nav_identity = self.cash_balance + market_value_total
        if abs(nav_identity - self.nav) > _NAV_TOLERANCE:
            raise ValueError(
                "cash_balance + sum(market_value) must reconcile to nav: "
                f"got {nav_identity:.6f} vs {self.nav:.6f}."
            )
        return self

    def position_records(self) -> list[dict[str, Any]]:
        """Return deterministic per-position records suitable for parquet export."""
        records: list[dict[str, Any]] = []
        for ticker in sorted(self.positions):
            position = self.positions[ticker]
            records.append(
                {
                    "as_of_date": self.as_of_date.isoformat(),
                    "snapshot_id": str(self.snapshot_id),
                    "broker_timestamp": self.broker_timestamp,
                    "ticker": ticker,
                    "quantity": position.quantity,
                    "cost_basis": position.cost_basis,
                    "market_value": position.market_value,
                    "current_weight": position.current_weight,
                    "entry_date": (
                        position.entry_date.isoformat() if position.entry_date else None
                    ),
                    "unrealized_pnl": position.unrealized_pnl,
                    "cash_balance": self.cash_balance,
                    "equity_value": self.equity_value,
                    "nav": self.nav,
                    "gross_exposure": self.gross_exposure,
                    "net_exposure": self.net_exposure,
                    "total_return_pct": self.total_return_pct,
                    "created_at": self.created_at,
                }
            )
        return records

    def header_payload(self) -> dict[str, Any]:
        """Return a compact summary payload for JSON sidecars."""
        payload = self.model_dump(mode="json", exclude={"positions"}, round_trip=True)
        payload["schema_version"] = f"{SCHEMA_VERSION}.broker_positions.v1"
        payload["position_count"] = len(self.positions)
        payload["market_value_total"] = round(
            sum(position.market_value for position in self.positions.values()), 10
        )
        return payload
