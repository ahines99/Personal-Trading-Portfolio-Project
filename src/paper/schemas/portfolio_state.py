"""Append-only portfolio-state ledger for paper trading."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import (
    SCHEMA_VERSION,
    append_jsonl_record,
    canonicalize_utc_timestamp,
    read_jsonl_models,
    utc_now_iso,
)

DEFAULT_PORTFOLIO_STATE_PATH = Path("paper_trading/portfolio_state/daily_state.jsonl")
_NAV_TOLERANCE = 1e-4


class PortfolioStateRow(BaseModel):
    """One immutable daily portfolio-state observation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.portfolio_state.v1"},
    )

    date: date
    nav: float
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    num_positions: int = Field(ge=0)
    daily_return: float
    daily_pnl: float
    daily_pnl_realized: float
    daily_pnl_unrealized: float
    cum_pnl: float
    ytd_return: float
    inception_return: float
    ytd_sharpe: float | None = None
    ytd_sortino: float | None = None
    ytd_max_dd: float | None = None
    realized_vol_30d: float | None = None
    drift_l1_today: float | None = None
    snapshot_id: UUID
    supersedes: UUID | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        return canonicalize_utc_timestamp(value)

    @model_validator(mode="after")
    def _validate_nav_identity(self) -> "PortfolioStateRow":
        if abs((self.cash + self.equity) - self.nav) > _NAV_TOLERANCE:
            raise ValueError(
                "cash + equity must reconcile to nav: "
                f"{self.cash + self.equity:.6f} vs {self.nav:.6f}."
            )
        return self


class PortfolioStateLedger:
    """Append-only JSONL ledger for daily portfolio-state tracking."""

    def __init__(self, path: str | Path = DEFAULT_PORTFOLIO_STATE_PATH) -> None:
        self.path = Path(path)

    def append_state(
        self, row: PortfolioStateRow | dict[str, Any]
    ) -> PortfolioStateRow:
        record = row if isinstance(row, PortfolioStateRow) else PortfolioStateRow.model_validate(row)
        append_jsonl_record(self.path, record.model_dump(mode="json", round_trip=True))
        return record

    def get_latest(self) -> PortfolioStateRow | None:
        rows = self._rows()
        return rows[-1] if rows else None

    def get_state_for_date(self, value: date | str) -> PortfolioStateRow | None:
        target_date = date.fromisoformat(value) if isinstance(value, str) else value
        matches = [row for row in self._rows() if row.date == target_date]
        return matches[-1] if matches else None

    def history(self) -> list[PortfolioStateRow]:
        return self._rows()

    def append_correction(
        self, snapshot_id: UUID | str, updates: dict[str, Any]
    ) -> PortfolioStateRow:
        target_snapshot = UUID(str(snapshot_id))
        current = next(
            (row for row in reversed(self._rows()) if row.snapshot_id == target_snapshot),
            None,
        )
        if current is None:
            raise KeyError(f"Unknown snapshot_id: {snapshot_id}")

        payload = current.model_dump(mode="python", round_trip=True)
        payload.update(updates)
        payload["supersedes"] = current.snapshot_id
        payload["created_at"] = utc_now_iso()
        return self.append_state(payload)

    def _rows(self) -> list[PortfolioStateRow]:
        return read_jsonl_models(self.path, PortfolioStateRow)
