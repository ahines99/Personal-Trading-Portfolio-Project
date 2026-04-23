from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import run_paper_phase_a  # noqa: E402
from paper.config import PaperTradingConfig  # noqa: E402
from paper.phase_b_executor import PhaseBExecutor  # noqa: E402
from paper.post_run_reconciliation import generate_reconciliation_report  # noqa: E402
from tests.paper_e2e_helpers import E2EMockBroker, base_config, build_fake_repo, read_json  # noqa: E402


def test_phase_b_executor_records_tracking_vs_backtest_metrics(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = build_fake_repo(tmp_path)
    config = base_config(repo_root, baseline_dir)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=config,
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    (bundle_dir / "approval.json").write_text(
        json.dumps(
            {
                "status": "APPROVED",
                "approved": True,
                "approved_at": _valid_approval_timestamp(intents),
                "approver": "tester",
                "config_hash": intents["config_hash"],
                "rebalance_id": intents["rebalance_id"],
            }
        ),
        encoding="utf-8",
    )

    executor = PhaseBExecutor(
        config=config,
        repo_root=repo_root,
        broker_client=E2EMockBroker(),
    )
    result = executor.run(bundle_dir=bundle_dir)

    tracking = result["tracking_vs_backtest"]
    assert tracking["available"] is True
    assert tracking["return_date"] == as_of_date
    assert tracking["return_date_source"] == "bundle_as_of_date"
    assert tracking["backtest_predicted_return"] == pytest.approx(-0.002, abs=1e-12)
    assert "tracking_vs_backtest" in read_json(bundle_dir / "reconciliation_report.json")

    summary_rows = _read_jsonl(repo_root / "paper_trading" / "reports" / "reconciliation_summary.jsonl")
    latest = summary_rows[-1]
    assert latest["return_date_source"] == "bundle_as_of_date"
    assert latest["backtest_predicted_return"] == pytest.approx(-0.002, abs=1e-12)
    assert latest["tracking_error_observations"] == 1
    assert latest["monthly_win_rate_observations"] == 1


def test_generate_reconciliation_report_rolls_tracking_error_and_monthly_win_rate(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "paper_trading" / "reports" / "reconciliation_summary.jsonl"
    day_one_context = {
        "repo_root": tmp_path,
        "summary_path": summary_path,
        "target_weights": {"AAA": 0.60, "BBB": 0.40},
        "actual_weights": {"AAA": 0.60, "BBB": 0.40},
        "intended_trades": [{"ticker": "AAA"}],
        "fills": [],
        "tracking_error_window_days": 2,
        "tracking_error_annualization_days": 252,
        "tracking_error_alert_threshold": 0.10,
        "monthly_win_rate_min_threshold": 0.55,
        "tracking_vs_backtest": {
            "available": True,
            "status": "OK",
            "return_date": "2026-04-21",
            "return_date_source": "submission_day",
            "paper_realized_return": 0.02,
            "backtest_predicted_return": 0.01,
            "return_excess_vs_backtest": 0.01,
            "paper_realized_pnl": 2000.0,
            "backtest_predicted_pnl": 1000.0,
            "pnl_reconciled_vs_backtest": 1000.0,
            "tracking_capital_base": 100000.0,
        },
    }
    day_two_context = {
        **day_one_context,
        "tracking_vs_backtest": {
            "available": True,
            "status": "OK",
            "return_date": "2026-04-22",
            "return_date_source": "submission_day",
            "paper_realized_return": -0.01,
            "backtest_predicted_return": 0.01,
            "return_excess_vs_backtest": -0.02,
            "paper_realized_pnl": -1000.0,
            "backtest_predicted_pnl": 1000.0,
            "pnl_reconciled_vs_backtest": -2000.0,
            "tracking_capital_base": 100000.0,
        },
    }

    report_path = generate_reconciliation_report("2026-04-21", day_one_context)
    report_path = generate_reconciliation_report("2026-04-22", day_two_context)

    latest = _read_jsonl(summary_path)[-1]
    expected_tracking_error = 0.015 * math.sqrt(252.0)
    assert latest["tracking_error_observations"] == 2
    assert latest["tracking_error_rolling_ann"] == pytest.approx(expected_tracking_error, rel=1e-9)
    assert latest["tracking_error_rolling_ann_pct"] == pytest.approx(expected_tracking_error * 100.0, rel=1e-9)
    assert latest["tracking_error_gate_status"] == "ALERT"
    assert latest["monthly_positive_excess_days"] == 1
    assert latest["monthly_win_rate_observations"] == 2
    assert latest["monthly_win_rate_vs_backtest"] == pytest.approx(0.5, abs=1e-12)
    assert latest["monthly_win_rate_gate_status"] == "ALERT"

    report_text = Path(report_path).read_text(encoding="utf-8")
    assert "## Tracking vs Backtest" in report_text
    assert "Rolling tracking error (2d ann.): 23.812%" in report_text


def test_paper_trading_config_accepts_tracking_error_fields() -> None:
    config = PaperTradingConfig.model_validate(
        {
            "stage": 3,
            "broker": "mock",
            "capital_mode": "paper",
            "account_id": "PAPER-12345",
            "tracking_error_window_days": 10,
            "tracking_error_annualization_days": 252,
            "tracking_error_alert_pct": 0.75,
            "tracking_monthly_win_rate_min_pct": 55.0,
        }
    )

    assert config.tracking_error_window_days == 10
    assert config.tracking_error_alert_pct == pytest.approx(0.75, abs=1e-12)
    assert config.tracking_monthly_win_rate_min_pct == pytest.approx(55.0, abs=1e-12)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _valid_approval_timestamp(intents: dict[str, object]) -> str:
    generated_at = datetime.fromisoformat(str(intents["generated_at"]).replace("Z", "+00:00"))
    approval_deadline = datetime.fromisoformat(str(intents["approval_deadline"]).replace("Z", "+00:00"))
    midpoint = generated_at + (approval_deadline - generated_at) / 2
    return midpoint.isoformat().replace("+00:00", "Z")
