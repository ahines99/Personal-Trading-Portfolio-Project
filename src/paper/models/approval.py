"""Approval models for Stage 3 manual and Stage 4 automatic approvals."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..schemas import SCHEMA_VERSION


class ApprovalStatus(str, Enum):
    """Allowed approval decisions for pending intent bundles."""

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


class ApprovalRecord(BaseModel):
    """Immutable audit record for a manual or automatic approval decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.approval_record.v1"},
    )

    approved_at: datetime = Field(
        description="UTC decision timestamp captured at approval time."
    )
    operator: str = Field(min_length=1, description="Username or email of approver.")
    status: ApprovalStatus
    comment: str | None = Field(
        default=None,
        description="Optional rationale or rejection reason.",
    )
    hostname: str = Field(
        min_length=1,
        description="Host that captured the approval for auditability.",
    )
    rebalance_id: str | None = Field(
        default=None,
        description="Optional rebalance identifier carried through Phase A and B.",
    )

    @field_validator("approved_at", mode="before")
    @classmethod
    def _validate_approved_at(cls, value: Any) -> datetime:
        return _normalize_datetime(value)

    @field_validator("operator", "hostname", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Approval text fields must be non-empty.")
        return text

    @field_validator("comment", "rebalance_id", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_serializer("approved_at")
    def _serialize_approved_at(self, value: datetime) -> str:
        return _serialize_datetime(value)
