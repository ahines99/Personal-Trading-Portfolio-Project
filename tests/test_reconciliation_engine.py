from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.order_blotter import OrderBlotter, OrderStatus  # noqa: E402
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


class FillStatusBroker:
    def __init__(
        self,
        statuses: dict[str, dict[str, object]],
        *,
        balances: dict[str, object] | None = None,
        positions: list[dict[str, object]] | None = None,
    ) -> None:
        self._statuses = statuses
        self._balances = balances or {
            "cash": 10000.0,
            "equity": 10000.0,
            "total_equity": 10000.0,
        }
        self._positions = positions or []

    def get_order_status(self, broker_order_id: str) -> dict[str, object]:
        return dict(self._statuses[broker_order_id])

    def get_balances(self) -> dict[str, object]:
        return dict(self._balances)

    def get_positions(self) -> list[dict[str, object]]:
        return [dict(position) for position in self._positions]


def _append_terminal_order(
    blotter: OrderBlotter,
    *,
    broker_order_id: str,
    qty: float = 10.0,
    symbol: str = "AAA",
) -> tuple[UUID, str, str]:
    order = blotter.create_order(
        rebalance_id=uuid4(),
        symbol=symbol,
        side="BUY",
        qty=qty,
        order_type="market",
        parent_intent_hash=f"intent-{symbol.lower()}",
        status=OrderStatus.APPROVED,
    )
    submitted = blotter.update_status(
        order.order_id,
        OrderStatus.SUBMITTED,
        updates={"broker_order_id": broker_order_id},
    )
    filled = blotter.update_status(order.order_id, OrderStatus.FILLED)
    assert filled.submission_timestamp is not None
    return order.order_id, submitted.submission_timestamp[:10], submitted.submission_timestamp


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


def test_reconciliation_engine_flags_malformed_fill_payloads(tmp_path: Path) -> None:
    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    order_id, submission_date, _ = _append_terminal_order(
        blotter,
        broker_order_id="broker-bad-fill",
    )
    engine = ReconciliationEngine(
        FillStatusBroker(
            {
                "broker-bad-fill": {
                    "broker_order_id": "broker-bad-fill",
                    "status": "filled",
                    "fills": [
                        {"qty": "abc", "price": 101.0, "timestamp": "2026-04-22T20:00:00Z"},
                        {"qty": 4.0, "price": 0.0, "timestamp": "2026-04-22T20:01:00Z"},
                    ],
                }
            }
        ),
        order_blotter=blotter,
        repo_root=tmp_path,
    )

    result = engine.reconcile_and_update_positions(as_of_date=submission_date)

    assert result.reconciliation_ok is False
    assert result.fills_written == 0
    assert any("invalid qty='abc'" in item for item in result.discrepancies)
    assert any("non-positive price=0.0" in item for item in result.discrepancies)
    assert any("has no broker fill details" in item for item in result.discrepancies)

    payload = json.loads(Path(result.reconciliation_path).read_text(encoding="utf-8"))
    anomalies = payload["report"]["anomalies"]
    assert any("invalid qty='abc'" in item for item in anomalies)
    assert any("non-positive price=0.0" in item for item in anomalies)
    fills_path = tmp_path / "paper_trading" / "blotter" / "fills.jsonl"
    assert not fills_path.exists()
    assert engine.fill_blotter.get_fills_by_order(order_id) == []


def test_reconciliation_engine_preserves_valid_fills_while_flagging_malformed_ones(
    tmp_path: Path,
) -> None:
    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    order_id, submission_date, submission_timestamp = _append_terminal_order(
        blotter,
        broker_order_id="broker-mixed-fill",
    )
    engine = ReconciliationEngine(
        FillStatusBroker(
            {
                "broker-mixed-fill": {
                    "broker_order_id": "broker-mixed-fill",
                    "status": "filled",
                    "fills": [
                        {
                            "broker_fill_id": "fill-good-1",
                            "qty": 6.0,
                            "price": 101.0,
                            "commission": 0.5,
                            "timestamp": submission_timestamp,
                        },
                        {
                            "broker_fill_id": "fill-bad-2",
                            "qty": 4.0,
                            "price": "bad-price",
                            "timestamp": submission_timestamp,
                        },
                    ],
                }
            }
        ),
        order_blotter=blotter,
        repo_root=tmp_path,
    )

    result = engine.reconcile_and_update_positions(as_of_date=submission_date)

    assert result.reconciliation_ok is False
    assert result.fills_written == 1
    assert any("invalid price='bad-price'" in item for item in result.discrepancies)
    assert any("filled_qty mismatch" in item for item in result.discrepancies)

    stored_fills = engine.fill_blotter.get_fills_by_order(order_id)
    assert len(stored_fills) == 1
    assert stored_fills[0].broker_fill_id == "fill-good-1"
    assert stored_fills[0].qty == pytest.approx(6.0)
    assert stored_fills[0].price == pytest.approx(101.0)
