from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_ROOT))

import run_paper_phase_a  # noqa: E402
from paper.approval_cli import main as approval_main  # noqa: E402
from paper.phase_b_executor import PhaseBExecutor  # noqa: E402
from paper_e2e_helpers import (  # noqa: E402
    E2EMockBroker,
    base_config,
    build_fake_repo,
    latest_rows_by_id,
    read_json,
    read_jsonl,
)


def test_stage3_phase_a_approval_cli_phase_b_e2e(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = build_fake_repo(tmp_path)
    config = base_config(repo_root, baseline_dir)

    phase_a_result = run_paper_phase_a.run_phase_a(
        config=config,
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents_path = bundle_dir / "intents.json"
    pending_intents = read_json(intents_path)
    assert pending_intents["status"] == "AWAITING_APPROVAL"
    assert pending_intents["proposed_orders"]
    generated_at = datetime.fromisoformat(pending_intents["generated_at"].replace("Z", "+00:00"))
    approval_deadline = datetime.fromisoformat(pending_intents["approval_deadline"].replace("Z", "+00:00"))
    approval_time = generated_at + (approval_deadline - generated_at) / 2

    exit_code = approval_main(
        [
            str(intents_path),
            "--approve",
            "--operator",
            "worker-4",
            "--comment",
            "E2E approval.",
            "--hostname",
            "paper-e2e",
            "--yes",
        ],
        now=approval_time.astimezone(timezone.utc),
    )
    assert exit_code == 0
    approved_intents = read_json(intents_path)
    assert approved_intents["status"] == "APPROVED"
    assert approved_intents["approval_record"]["operator"] == "worker-4"

    broker = E2EMockBroker()
    result = PhaseBExecutor(config=config, repo_root=repo_root, broker_client=broker).run(bundle_dir=bundle_dir)

    assert result["submitted_order_count"] == phase_a_result["order_count"]
    assert result["executed_trade_count"] == phase_a_result["order_count"]
    assert result["unfilled_order_count"] == 0
    assert Path(result["reconciliation_report"]).exists()
    assert Path(result["reconciliation_report_json"]).exists()
    assert (bundle_dir / "phase_b_summary.json").exists()
    assert read_json(bundle_dir / "phase_b_summary.json")["executed_trade_count"] == phase_a_result["order_count"]
    assert read_json(bundle_dir / "reconciliation_report.json")["executed_trades"]
    assert broker.place_calls

    orders = latest_rows_by_id(read_jsonl(repo_root / "paper_trading" / "blotter" / "orders.jsonl"))
    assert {row["status"] for row in orders.values()} == {"FILLED"}
    assert read_jsonl(repo_root / "paper_trading" / "blotter" / "fills.jsonl")


def test_stage4_auto_approval_happy_path_e2e(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = build_fake_repo(tmp_path)
    config = base_config(repo_root, baseline_dir, stage=4, stage4_mode="auto")

    phase_a_result = run_paper_phase_a.run_phase_a(
        config=config,
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    auto_intents = read_json(bundle_dir / "intents.json")
    assert auto_intents["status"] == "APPROVED"
    assert auto_intents["approval_record"]["operator"] == "stage4-auto"

    approval_log = repo_root / "paper_trading" / "reports" / as_of_date / "approvals.jsonl"
    approval_records = read_jsonl(approval_log)
    assert len(approval_records) == 1
    assert approval_records[0]["status"] == "APPROVED"

    broker = E2EMockBroker()
    result = PhaseBExecutor(config=config, repo_root=repo_root, broker_client=broker).run(bundle_dir=bundle_dir)

    assert result["executed_trade_count"] == phase_a_result["order_count"]
    assert result["reconciliation_ok"] is True
    assert Path(result["reconciliation_report"]).exists()
    assert read_json(bundle_dir / "phase_b_summary.json")["submitted_order_count"] == phase_a_result["order_count"]


@pytest.mark.parametrize("filename", ["approval.template.json", "manifest.json", "checksums.sha256"])
def test_phase_a_e2e_bundle_contains_operator_artifacts(tmp_path: Path, filename: str) -> None:
    repo_root, baseline_dir, as_of_date = build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=base_config(repo_root, baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )

    assert (Path(phase_a_result["bundle_dir"]) / filename).exists()
