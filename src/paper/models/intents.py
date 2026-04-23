"""Intent and preview models for Stage 3 paper-trading execution."""

from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ..schemas import SCHEMA_VERSION, normalize_ticker
from .approval import ApprovalRecord, ApprovalStatus

_MARKET_ORDER_TYPES = {"market", "moo", "market_on_open"}
_LIMIT_ORDER_TYPES = {"limit", "loo", "limit_on_open"}


class OrderSide(str, Enum):
    """Supported equity-order sides for manual paper trading."""

    BUY = "BUY"
    SELL = "SELL"


class IntentBundleStatus(str, Enum):
    """Allowed lifecycle states for a pending intent bundle."""

    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


def _normalize_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported datetime value: {type(value)!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _normalize_text_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        candidate_values = [values]
    else:
        candidate_values = list(values)

    normalized: list[str] = []
    for item in candidate_values:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _normalize_weights(payload: Any) -> dict[str, float]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("Weight payloads must be dictionaries keyed by ticker.")

    normalized: dict[str, float] = {}
    for key, value in payload.items():
        ticker = normalize_ticker(str(key))
        normalized[ticker] = float(value)
    return normalized


class OrderSpec(BaseModel):
    """One normalized equity order proposal prior to broker preview."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.order_spec.v1"},
    )

    symbol: str
    qty: float = Field(gt=0.0)
    side: OrderSide
    order_type: str
    limit_price: float | None = Field(default=None, gt=0.0)

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: Any) -> str:
        return normalize_ticker(str(value))

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("order_type must be non-empty.")
        return text

    @model_validator(mode="after")
    def _validate_limit_usage(self) -> "OrderSpec":
        if self.order_type in _MARKET_ORDER_TYPES and self.limit_price is not None:
            raise ValueError(
                f"limit_price must be omitted for order_type={self.order_type!r}."
            )
        if self.order_type in _LIMIT_ORDER_TYPES and self.limit_price is None:
            raise ValueError(
                f"limit_price is required for order_type={self.order_type!r}."
            )
        return self

    def cache_key_payload(self) -> dict[str, Any]:
        """Return a deterministic payload for preview-cache keys."""
        return self.model_dump(mode="json", round_trip=True)


class PreviewOrderResult(BaseModel):
    """Normalized preview result for a single proposed broker order."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.preview_order_result.v1"},
    )

    order: OrderSpec
    preview_id: str | None = None
    broker_status: str | None = None
    estimated_notional: float = 0.0
    estimated_commission: float = 0.0
    cash_required: float = 0.0
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False

    @field_validator("preview_id", "broker_status", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("warnings", "errors", mode="before")
    @classmethod
    def _normalize_messages(cls, value: Any) -> tuple[str, ...]:
        return _normalize_text_tuple(value)

    @property
    def ok(self) -> bool:
        return not self.errors


class PreviewBatchTotals(BaseModel):
    """Aggregate totals across a preview batch."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.preview_batch_totals.v1"},
    )

    total_notional: float = 0.0
    commission: float = 0.0
    cash_needed: float = 0.0
    order_count: int = Field(ge=0)
    cache_hits: int = Field(ge=0, default=0)
    warning_count: int = Field(ge=0, default=0)
    error_count: int = Field(ge=0, default=0)


class PreviewBatch(BaseModel):
    """Batch preview output consumed by approval and submission layers."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.preview_batch.v1"},
    )

    generated_at: datetime
    results: tuple[PreviewOrderResult, ...]
    totals: PreviewBatchTotals
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @field_validator("generated_at", mode="before")
    @classmethod
    def _validate_generated_at(cls, value: Any) -> datetime:
        return _normalize_datetime(value)

    @field_validator("warnings", "errors", mode="before")
    @classmethod
    def _normalize_messages(cls, value: Any) -> tuple[str, ...]:
        return _normalize_text_tuple(value)

    @model_validator(mode="after")
    def _validate_counts(self) -> "PreviewBatch":
        if self.totals.order_count != len(self.results):
            raise ValueError(
                "Preview batch order_count must match the number of preview results."
            )
        return self

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return _serialize_datetime(value)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors) or any(result.errors for result in self.results)


class IntentAggregate(BaseModel):
    """Aggregate statistics stored alongside a pending intent bundle."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.intent_aggregate.v1"},
    )

    total_notional: float = 0.0
    commission: float = 0.0
    cash_needed: float = 0.0
    turnover_pct: float = Field(ge=0.0, default=0.0)

    @classmethod
    def from_preview_batch(
        cls,
        preview_batch: PreviewBatch,
        *,
        turnover_pct: float = 0.0,
    ) -> "IntentAggregate":
        return cls(
            total_notional=preview_batch.totals.total_notional,
            commission=preview_batch.totals.commission,
            cash_needed=preview_batch.totals.cash_needed,
            turnover_pct=turnover_pct,
        )


class IntentBundle(BaseModel):
    """Pending intent bundle written by Phase A and consumed by Phase B."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.intent_bundle.v1"},
    )

    as_of_date: date | None = None
    rebalance_id: str | None = None
    generated_at: datetime
    approval_deadline: datetime
    status: IntentBundleStatus
    signal_hash: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    target_weights: dict[str, float]
    current_holdings: dict[str, float]
    proposed_orders: tuple[OrderSpec, ...]
    aggregate: IntentAggregate
    approval_record: ApprovalRecord | None = None
    preview_batch: PreviewBatch | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at", "approval_deadline", mode="before")
    @classmethod
    def _validate_datetimes(cls, value: Any) -> datetime:
        return _normalize_datetime(value)

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, value: Any) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        return date.fromisoformat(str(value)[:10])

    @field_validator("signal_hash", "config_hash", mode="before")
    @classmethod
    def _strip_required_hashes(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Hash fields must be non-empty.")
        return text

    @field_validator("target_weights", "current_holdings", mode="before")
    @classmethod
    def _validate_weight_maps(cls, value: Any) -> dict[str, float]:
        return _normalize_weights(value)

    @model_validator(mode="after")
    def _validate_approval_state(self) -> "IntentBundle":
        if self.approval_deadline <= self.generated_at:
            raise ValueError("approval_deadline must be later than generated_at.")

        if self.status == IntentBundleStatus.AWAITING_APPROVAL:
            if self.approval_record is not None:
                raise ValueError(
                    "approval_record must be omitted while status is AWAITING_APPROVAL."
                )
            return self

        if self.approval_record is None:
            raise ValueError(
                "approval_record is required when intent status is APPROVED or REJECTED."
            )

        if self.status == IntentBundleStatus.APPROVED:
            expected = ApprovalStatus.APPROVED
        else:
            expected = ApprovalStatus.REJECTED

        if self.approval_record.status != expected:
            raise ValueError(
                "Intent bundle status and approval_record.status must match."
            )
        return self

    @field_serializer("generated_at", "approval_deadline")
    def _serialize_datetimes(self, value: datetime) -> str:
        return _serialize_datetime(value)

    def with_approval(self, approval_record: ApprovalRecord) -> "IntentBundle":
        """Return an updated immutable bundle carrying the provided approval."""
        next_status = (
            IntentBundleStatus.APPROVED
            if approval_record.status == ApprovalStatus.APPROVED
            else IntentBundleStatus.REJECTED
        )
        return self.model_copy(
            update={
                "status": next_status,
                "approval_record": approval_record,
            }
        )
