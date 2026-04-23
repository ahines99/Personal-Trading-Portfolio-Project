"""Target-book schemas for paper-trading snapshots."""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import SCHEMA_VERSION, canonicalize_utc_timestamp, normalize_ticker, utc_now_iso

_WEIGHT_TOLERANCE = 1e-4


class TargetPosition(BaseModel):
    """One target position in the intended rebalance book."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.target_position.v1"},
    )

    ticker: str
    target_weight: float = Field(ge=0.0, le=1.0)
    signal_score: float
    signal_rank: int = Field(ge=1)
    sector: str | None = None
    rationale: str | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return normalize_ticker(value)


class TargetBookSnapshot(BaseModel):
    """Immutable target-book snapshot used by the paper-trading controller."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.target_book.v1"},
    )

    as_of_date: date
    signal_hash: str
    signal_timestamp: str
    strategy_version: str
    n_positions: int = Field(ge=0)
    target_positions: dict[str, TargetPosition]
    target_cash_pct: float = Field(ge=0.0, le=1.0)
    rebalance_id: UUID
    model_ensemble: tuple[str, ...] = ()
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("signal_timestamp", "created_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        return canonicalize_utc_timestamp(value)

    @field_validator("target_positions")
    @classmethod
    def _normalize_position_keys(
        cls, positions: dict[str, TargetPosition]
    ) -> dict[str, TargetPosition]:
        normalized: dict[str, TargetPosition] = {}
        for ticker, position in positions.items():
            key = normalize_ticker(ticker)
            if key != position.ticker:
                raise ValueError(
                    f"Target-position key {key!r} does not match payload ticker "
                    f"{position.ticker!r}."
                )
            normalized[key] = position
        return normalized

    @model_validator(mode="after")
    def _validate_weights(self) -> "TargetBookSnapshot":
        if self.n_positions != len(self.target_positions):
            raise ValueError(
                f"n_positions={self.n_positions} does not match "
                f"{len(self.target_positions)} target positions."
            )

        weight_sum = self.target_cash_pct + sum(
            position.target_weight for position in self.target_positions.values()
        )
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                "Target-book weights must sum to 1.0 including cash: "
                f"got {weight_sum:.6f}."
            )
        return self

    def position_records(self) -> list[dict[str, Any]]:
        """Return deterministic per-position records suitable for parquet export."""
        records: list[dict[str, Any]] = []
        for ticker in sorted(self.target_positions):
            position = self.target_positions[ticker]
            records.append(
                {
                    "as_of_date": self.as_of_date.isoformat(),
                    "rebalance_id": str(self.rebalance_id),
                    "signal_hash": self.signal_hash,
                    "signal_timestamp": self.signal_timestamp,
                    "strategy_version": self.strategy_version,
                    "model_ensemble": list(self.model_ensemble),
                    "ticker": ticker,
                    "target_weight": position.target_weight,
                    "signal_score": position.signal_score,
                    "signal_rank": position.signal_rank,
                    "sector": position.sector,
                    "rationale": position.rationale,
                    "target_cash_pct": self.target_cash_pct,
                    "created_at": self.created_at,
                }
            )
        return records

    def header_payload(self) -> dict[str, Any]:
        """Return the deterministic header payload for the JSON sidecar."""
        payload = self.model_dump(mode="json", exclude={"target_positions"}, round_trip=True)
        payload["schema_version"] = f"{SCHEMA_VERSION}.target_book.v1"
        payload["target_weight_sum"] = round(
            sum(position.target_weight for position in self.target_positions.values()), 10
        )
        return payload
