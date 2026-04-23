from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.order_blotter import OrderBlotter  # noqa: E402
from paper.post_run_reconciliation import (  # noqa: E402
    build_reconciliation_drift_snapshot,
    generate_reconciliation_report,
)
from paper.reconciliation_engine import ReconciliationEngine  # noqa: E402


class DriftBroker:
    def get_order_status(self, broker_order_id: str) -> dict[str, object]:
        return {"broker_order_id": broker_order_id, "status": "filled", "fills": []}

    def get_balances(self) -> dict[str, object]:
        return {
            "cash": 4000.0,
            "equity": 99000.0,
            "total_equity": 99000.0,
        }

    def get_positions(self) -> list[dict[str, object]]:
        return [
            {
                "ticker": "AAA",
                "quantity": 610.0,
                "cost_basis": 60000.0,
                "market_value": 61000.0,
                "unrealized_pnl": 1000.0,
            },
            {
                "ticker": "BBB",
                "quantity": 340.0,
                "cost_basis": 35000.0,
                "market_value": 34000.0,
                "unrealized_pnl": -1000.0,
            },
        ]


def test_reconciliation_engine_computes_cash_nav_and_weight_drift(tmp_path: Path) -> None:
    engine = ReconciliationEngine(
        DriftBroker(),
        order_blotter=OrderBlotter(tmp_path / "orders.jsonl"),
        repo_root=tmp_path,
    )

    result = engine.reconcile_and_update_positions(
        as_of_date="2026-04-22",
        target_weights={"AAA": 0.60, "BBB": 0.35},
        expected_nav=100000.0,
    )

    payload = json.loads(Path(result.reconciliation_path).read_text(encoding="utf-8"))
    report = payload["report"]
    assert report["target_weights"] == {"AAA": 0.6, "BBB": 0.35}
    assert report["actual_weights"]["AAA"] == 61000.0 / 99000.0
    assert report["cash_drift"] == pytest.approx(-1000.0)
    assert report["nav_drift"] == pytest.approx(-1000.0)
    assert report["weight_drift_l1"] > 0.01


def test_reconciliation_report_surfaces_drift_thresholds(tmp_path: Path) -> None:
    context = {
        "repo_root": tmp_path,
        "target_weights": {"AAA": 0.60, "BBB": 0.35},
        "broker_balances": {"cash": 4000.0, "equity": 99000.0},
        "broker_positions": DriftBroker().get_positions(),
        "expected_nav": 100000.0,
        "cash_drift_abs_threshold": 100.0,
        "nav_drift_abs_threshold": 100.0,
        "weight_drift_alert_threshold": 0.01,
    }

    snapshot = build_reconciliation_drift_snapshot(context)
    assert snapshot["cash_drift"] == pytest.approx(-1000.0)
    assert snapshot["nav_drift"] == pytest.approx(-1000.0)
    assert snapshot["thresholds"]["cash_drift_abs"] == 100.0
    assert "largest_weight_drifts" in snapshot

    report_path = generate_reconciliation_report("2026-04-22", context)
    text = Path(report_path).read_text(encoding="utf-8")
    assert "Status: ALERT_REQUIRED" in text
    assert "Cash drift: -1000.00" in text
    assert "NAV drift threshold: 100.00" in text


def test_reconciliation_summary_replaces_same_day_row_atomically(tmp_path: Path) -> None:
    summary_path = tmp_path / "paper_trading" / "reports" / "reconciliation_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                json.dumps({"as_of_date": "2026-04-21", "status": "CLEAN", "weight_drift_l1": 0.0}, sort_keys=True),
                json.dumps(
                    {"as_of_date": "2026-04-22", "status": "CLEAN", "weight_drift_l1": 0.0},
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generate_reconciliation_report(
        "2026-04-22",
        {
            "repo_root": tmp_path,
            "summary_path": summary_path,
            "target_weights": {"AAA": 0.60, "BBB": 0.35},
            "broker_balances": {"cash": 4000.0, "equity": 99000.0},
            "broker_positions": DriftBroker().get_positions(),
            "expected_nav": 100000.0,
            "cash_drift_abs_threshold": 100.0,
            "nav_drift_abs_threshold": 100.0,
            "weight_drift_alert_threshold": 0.01,
        },
    )

    rows = [
        json.loads(line)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["as_of_date"] for row in rows] == ["2026-04-21", "2026-04-22"]
    assert rows[0]["status"] == "CLEAN"
    assert rows[1]["status"] == "ALERT_REQUIRED"
    assert rows[1]["cash_drift"] == pytest.approx(-1000.0)
