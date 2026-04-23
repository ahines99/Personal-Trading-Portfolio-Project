from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.models import (  # noqa: E402
    ApprovalRecord,
    ApprovalStatus,
    IntentAggregate,
    IntentBundle,
    IntentBundleStatus,
    OrderSide,
    OrderSpec,
)
from paper.verify import (  # noqa: E402
    validate_approval_freshness,
    verify_approval_freshness,
)


def test_validate_approval_freshness_accepts_approved_bundle() -> None:
    generated_at = datetime(2026, 1, 7, 21, 45, tzinfo=timezone.utc)
    approval_deadline = generated_at + timedelta(hours=12)
    approved_at = generated_at + timedelta(hours=8)
    bundle = _base_bundle(
        generated_at=generated_at,
        approval_deadline=approval_deadline,
    ).with_approval(
        ApprovalRecord(
            approved_at=approved_at,
            operator="alice",
            status=ApprovalStatus.APPROVED,
            hostname="paper-host",
        )
    )

    resolved = validate_approval_freshness(bundle)

    assert resolved == approved_at
    assert verify_approval_freshness(bundle) is True


def test_validate_approval_freshness_accepts_legacy_approval_payload() -> None:
    generated_at = datetime(2026, 1, 7, 21, 45, tzinfo=timezone.utc)
    approval_deadline = generated_at + timedelta(hours=12)
    approved_at = generated_at + timedelta(hours=8)
    bundle = _base_bundle(
        generated_at=generated_at,
        approval_deadline=approval_deadline,
    ).model_dump(mode="json", round_trip=True)
    bundle["status"] = "APPROVED"
    legacy_approval = {
        "approved": True,
        "approved_at": approved_at.isoformat().replace("+00:00", "Z"),
    }

    resolved = validate_approval_freshness(bundle, legacy_approval)

    assert resolved == approved_at


def test_validate_approval_freshness_requires_approved_at() -> None:
    bundle = _base_bundle().with_approval(
        ApprovalRecord(
            approved_at=datetime(2026, 1, 8, 5, 0, tzinfo=timezone.utc),
            operator="alice",
            status=ApprovalStatus.APPROVED,
            hostname="paper-host",
        )
    )
    payload = bundle.model_dump(mode="json", round_trip=True)
    payload["approval_record"].pop("approved_at")

    with pytest.raises(ValueError, match="missing required approved_at"):
        validate_approval_freshness(payload)

    assert verify_approval_freshness(payload) is False


def test_validate_approval_freshness_rejects_status_mismatch() -> None:
    bundle = _base_bundle().with_approval(
        ApprovalRecord(
            approved_at=datetime(2026, 1, 8, 5, 0, tzinfo=timezone.utc),
            operator="alice",
            status=ApprovalStatus.APPROVED,
            hostname="paper-host",
        )
    )
    payload = bundle.model_dump(mode="json", round_trip=True)
    payload["approval_record"]["status"] = "REJECTED"

    with pytest.raises(ValueError, match="Approval record status mismatch"):
        validate_approval_freshness(payload)


def test_validate_approval_freshness_rejects_approval_after_deadline() -> None:
    generated_at = datetime(2026, 1, 7, 21, 45, tzinfo=timezone.utc)
    approval_deadline = generated_at + timedelta(hours=12)
    bundle = _base_bundle(
        generated_at=generated_at,
        approval_deadline=approval_deadline,
    ).with_approval(
        ApprovalRecord(
            approved_at=approval_deadline + timedelta(seconds=1),
            operator="alice",
            status=ApprovalStatus.APPROVED,
            hostname="paper-host",
        )
    )

    with pytest.raises(ValueError, match="exceeds bundle approval deadline"):
        validate_approval_freshness(bundle)


def test_validate_approval_freshness_rejects_approval_before_generation() -> None:
    generated_at = datetime(2026, 1, 7, 21, 45, tzinfo=timezone.utc)
    bundle = _base_bundle(generated_at=generated_at).with_approval(
        ApprovalRecord(
            approved_at=generated_at - timedelta(seconds=1),
            operator="alice",
            status=ApprovalStatus.REJECTED,
            hostname="paper-host",
            comment="Market conditions changed.",
        )
    )

    with pytest.raises(ValueError, match="predates bundle generation"):
        validate_approval_freshness(bundle)


def _base_bundle(
    *,
    generated_at: datetime | None = None,
    approval_deadline: datetime | None = None,
) -> IntentBundle:
    generated_at = generated_at or datetime(2026, 1, 7, 21, 45, tzinfo=timezone.utc)
    approval_deadline = approval_deadline or generated_at + timedelta(hours=12)
    return IntentBundle(
        generated_at=generated_at,
        approval_deadline=approval_deadline,
        status=IntentBundleStatus.AWAITING_APPROVAL,
        signal_hash="signal-hash",
        config_hash="config-hash",
        target_weights={"AAA": 0.6, "BBB": 0.4},
        current_holdings={"AAA": 0.5, "BBB": 0.5},
        proposed_orders=(
            OrderSpec(symbol="AAA", qty=10.0, side=OrderSide.BUY, order_type="moo"),
            OrderSpec(symbol="BBB", qty=8.0, side=OrderSide.SELL, order_type="moo"),
        ),
        aggregate=IntentAggregate(
            total_notional=1500.0,
            commission=0.0,
            cash_needed=900.0,
            turnover_pct=0.12,
        ),
    )
