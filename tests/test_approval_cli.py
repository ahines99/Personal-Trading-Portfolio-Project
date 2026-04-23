from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.approval_cli import main  # noqa: E402
from paper.models import (  # noqa: E402
    IntentAggregate,
    IntentBundle,
    IntentBundleStatus,
    OrderSide,
    OrderSpec,
)


def test_approval_cli_approves_pending_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_path = _write_bundle(tmp_path / "intents.json")

    exit_code = main(
        [
            str(bundle_path),
            "--approve",
            "--operator",
            "alice",
            "--comment",
            "Reviewed and ready.",
            "--hostname",
            "paper-host",
            "--yes",
        ]
    )

    assert exit_code == 0
    updated = _read_bundle(bundle_path)
    assert updated["status"] == "APPROVED"
    assert updated["approval_record"]["status"] == "APPROVED"
    assert updated["approval_record"]["operator"] == "alice"
    assert updated["approval_record"]["comment"] == "Reviewed and ready."
    assert updated["approval_record"]["hostname"] == "paper-host"
    assert "APPROVED bundle written" in capsys.readouterr().out


def test_approval_cli_rejects_pending_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "intents.json")

    exit_code = main(
        [
            str(bundle_path),
            "--reject",
            "--operator",
            "alice",
            "--comment",
            "Broker state drifted overnight.",
            "--hostname",
            "paper-host",
            "--yes",
        ]
    )

    assert exit_code == 0
    updated = _read_bundle(bundle_path)
    assert updated["status"] == "REJECTED"
    assert updated["approval_record"]["status"] == "REJECTED"
    assert updated["approval_record"]["comment"] == "Broker state drifted overnight."


def test_approval_cli_errors_when_bundle_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.json"

    exit_code = main(
        [
            str(missing_path),
            "--approve",
            "--operator",
            "alice",
            "--hostname",
            "paper-host",
            "--yes",
        ]
    )

    assert exit_code == 1
    assert "Bundle file not found" in capsys.readouterr().err


def test_approval_cli_errors_when_deadline_expired(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(timezone.utc)
    bundle_path = _write_bundle(
        tmp_path / "expired.json",
        generated_at=now - timedelta(hours=2),
        approval_deadline=now - timedelta(minutes=1),
    )

    exit_code = main(
        [
            str(bundle_path),
            "--approve",
            "--operator",
            "alice",
            "--hostname",
            "paper-host",
            "--yes",
        ],
        now=now,
    )

    assert exit_code == 1
    assert _read_bundle(bundle_path)["status"] == "AWAITING_APPROVAL"
    assert "Approval deadline expired" in capsys.readouterr().err


def test_approval_cli_errors_on_invalid_status_transition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_path = _write_bundle(
        tmp_path / "approved.json",
        status=IntentBundleStatus.APPROVED,
        approval_record={
            "approved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "operator": "prior-operator",
            "status": "APPROVED",
            "comment": "Already approved.",
            "hostname": "paper-host",
        },
    )

    exit_code = main(
        [
            str(bundle_path),
            "--approve",
            "--operator",
            "alice",
            "--hostname",
            "paper-host",
            "--yes",
        ]
    )

    assert exit_code == 1
    assert _read_bundle(bundle_path)["approval_record"]["operator"] == "prior-operator"
    assert "only AWAITING_APPROVAL can be updated" in capsys.readouterr().err


def _write_bundle(
    path: Path,
    *,
    generated_at: datetime | None = None,
    approval_deadline: datetime | None = None,
    status: IntentBundleStatus = IntentBundleStatus.AWAITING_APPROVAL,
    approval_record: dict[str, object] | None = None,
) -> Path:
    generated_at = generated_at or datetime.now(timezone.utc) - timedelta(minutes=5)
    approval_deadline = approval_deadline or generated_at + timedelta(hours=4)

    bundle = IntentBundle(
        generated_at=generated_at,
        approval_deadline=approval_deadline,
        status=status,
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
        approval_record=approval_record,
    )
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path


def _read_bundle(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
