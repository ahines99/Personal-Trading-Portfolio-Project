from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.order_blotter import OrderBlotter, OrderStatus  # noqa: E402
from paper.reconciliation_engine import ReconciliationEngine  # noqa: E402
from paper.submission_engine import SubmissionEngine  # noqa: E402


class FakeBroker:
    def __init__(self) -> None:
        self.placed_orders: list[tuple[str, str, float]] = []
        self.cancel_calls: list[str] = []
        self._poll_payloads: dict[str, dict[str, object]] = {}
        self._poll_errors: dict[str, Exception] = {}
        self._cancel_payloads: dict[str, dict[str, object]] = {}
        self._poll_hooks: dict[str, Callable[[], None]] = {}

    def place_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
        preview_result: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del order_type, limit_price, preview_result
        broker_order_id = f"broker-{symbol}-{len(self.placed_orders) + 1}"
        self.placed_orders.append((symbol, side, qty))
        return {"broker_order_id": broker_order_id}

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, object]:
        del timeout_seconds, poll_interval_seconds
        hook = self._poll_hooks.get(broker_order_id)
        if hook is not None:
            hook()
        if broker_order_id in self._poll_errors:
            raise self._poll_errors[broker_order_id]
        return dict(self._poll_payloads[broker_order_id])

    def cancel_order(self, broker_order_id: str) -> dict[str, object]:
        self.cancel_calls.append(broker_order_id)
        payload = self._cancel_payloads.get(
            broker_order_id,
            {"broker_order_id": broker_order_id, "status": "cancelled"},
        )
        return dict(payload)

    def get_order_status(self, broker_order_id: str) -> dict[str, object]:
        return dict(self._poll_payloads[broker_order_id])

    def get_balances(self) -> dict[str, object]:
        return {
            "cash": 6125.0,
            "equity": 25875.0,
            "total_equity": 25875.0,
        }

    def get_positions(self) -> list[dict[str, object]]:
        return [
            {
                "ticker": "AAA",
                "quantity": 80.0,
                "cost_basis": 8000.0,
                "market_value": 8400.0,
                "current_weight": 0.325,
                "unrealized_pnl": 400.0,
            },
            {
                "ticker": "BBB",
                "quantity": 120.0,
                "cost_basis": 12000.0,
                "market_value": 17475.0,
                "current_weight": 0.675,
                "unrealized_pnl": 5475.0,
            },
        ]


def test_submission_engine_sells_first_and_records_terminal_states(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker._poll_payloads = {
        "broker-AAA-1": {
            "status": "filled",
            "qty_filled": 10.0,
            "fills": [
                {
                    "fill_id": "fill-aaa-1",
                    "qty": 10.0,
                    "price": 101.0,
                    "commission": 1.0,
                    "timestamp": "2026-04-22T16:00:00Z",
                }
            ],
        },
        "broker-BBB-2": {
            "status": "canceled",
            "qty_filled": 4.0,
            "fills": [
                {
                    "fill_id": "fill-bbb-1",
                    "qty": 4.0,
                    "price": 52.5,
                    "commission": 0.5,
                    "timestamp": "2026-04-22T16:00:01Z",
                }
            ],
        },
        "broker-CCC-3": {"status": "rejected", "qty_filled": 0.0, "fills": []},
    }

    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    preview_batch = {
        "intent-sell": _preview("AAA", 10, "SELL", expected_price=100.0),
        "intent-buy-bbb": _preview("BBB", 8, "BUY", expected_price=52.0),
        "intent-buy-ccc": _preview("CCC", 5, "BUY", expected_price=10.0),
    }
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}
    order_specs = [
        {
            "symbol": "BBB",
            "qty": 8,
            "side": "BUY",
            "order_type": "market",
            "parent_intent_hash": "intent-buy-bbb",
        },
        {
            "symbol": "AAA",
            "qty": 10,
            "side": "SELL",
            "order_type": "market",
            "parent_intent_hash": "intent-sell",
        },
        {
            "symbol": "CCC",
            "qty": 5,
            "side": "BUY",
            "order_type": "market",
            "parent_intent_hash": "intent-buy-ccc",
        },
    ]

    result = engine.submit_and_poll(order_specs, preview_batch, approval_record)

    assert broker.placed_orders == [
        ("AAA", "SELL", 10.0),
        ("BBB", "BUY", 8.0),
        ("CCC", "BUY", 5.0),
    ]
    assert result.submitted == 3
    assert result.filled == 1
    assert result.partial == 1
    assert result.rejected == 1
    assert result.halted == 0

    latest = blotter.list_orders(latest_only=True)
    statuses = {row.symbol: row.status for row in latest}
    assert statuses["AAA"] == OrderStatus.FILLED
    assert statuses["BBB"] == OrderStatus.PARTIALLY_FILLED
    assert statuses["CCC"] == OrderStatus.REJECTED


def test_submission_engine_requires_approved_record(tmp_path: Path) -> None:
    engine = SubmissionEngine(
        FakeBroker(),
        order_blotter=OrderBlotter(tmp_path / "orders.jsonl"),
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    with pytest.raises(PermissionError, match="APPROVED"):
        engine.submit_and_poll([], None, {"status": "REJECTED"})


def test_submission_engine_halts_when_kill_switch_appears_mid_loop(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker._poll_payloads = {
        "broker-AAA-1": {"status": "filled", "qty_filled": 10.0, "fills": []},
    }
    kill_switch_path = tmp_path / "paper_trading" / "state" / "KILL_SWITCH"

    def _activate_kill_switch() -> None:
        kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        kill_switch_path.write_text("engaged\n", encoding="utf-8")

    broker._poll_hooks["broker-AAA-1"] = _activate_kill_switch

    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=kill_switch_path,
    )
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}

    with pytest.raises(RuntimeError, match="Kill switch engaged during submission"):
        engine.submit_and_poll(
            [
                {
                    "symbol": "BBB",
                    "qty": 8,
                    "side": "BUY",
                    "order_type": "market",
                    "parent_intent_hash": "intent-buy-bbb",
                },
                {
                    "symbol": "AAA",
                    "qty": 10,
                    "side": "SELL",
                    "order_type": "market",
                    "parent_intent_hash": "intent-sell",
                },
            ],
            {
                "intent-sell": _preview("AAA", 10, "SELL", expected_price=100.0),
                "intent-buy-bbb": _preview("BBB", 8, "BUY", expected_price=52.0),
            },
            approval_record,
        )

    latest = blotter.list_orders(latest_only=True)
    assert broker.placed_orders == [("AAA", "SELL", 10.0)]
    assert len(latest) == 1
    assert latest[0].status == OrderStatus.FILLED
    assert latest[0].preview_result["kill_switch_halt"]["stage"] == "after_poll"


def test_submission_engine_cancels_once_and_halts_on_poll_failure(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker._poll_errors["broker-AAA-1"] = TimeoutError("poll timeout")
    broker._cancel_payloads["broker-AAA-1"] = {
        "broker_order_id": "broker-AAA-1",
        "status": "cancelled",
    }

    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}

    with pytest.raises(RuntimeError, match="cancel attempted and batch halted"):
        engine.submit_and_poll(
            [
                {
                    "symbol": "AAA",
                    "qty": 10,
                    "side": "SELL",
                    "order_type": "market",
                    "parent_intent_hash": "intent-aaa",
                }
            ],
            {"intent-aaa": _preview("AAA", 10, "SELL", expected_price=100.0)},
            approval_record,
        )

    latest = blotter.list_orders(latest_only=True)
    assert broker.cancel_calls == ["broker-AAA-1"]
    assert len(latest) == 1
    assert latest[0].status == OrderStatus.CANCELLED
    assert latest[0].broker_order_id == "broker-AAA-1"
    assert latest[0].preview_result["poll_error"] == "poll timeout"
    assert latest[0].preview_result["cancellation_attempt"]["response"]["status"] == "cancelled"


def test_reconciliation_engine_writes_fills_and_positions(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker._poll_payloads = {"broker-AAA-1": {"status": "filled", "qty_filled": 10.0, "fills": []}}
    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}
    engine.submit_and_poll(
        [
            {
                "symbol": "AAA",
                "qty": 10,
                "side": "SELL",
                "order_type": "market",
                "parent_intent_hash": "intent-aaa",
            }
        ],
        {"intent-aaa": _preview("AAA", 10, "SELL", expected_price=100.0)},
        approval_record,
    )

    submission_timestamp = blotter.list_orders(latest_only=True)[0].submission_timestamp
    broker._poll_payloads["broker-AAA-1"] = {
        "status": "filled",
        "qty_filled": 10.0,
        "fills": [
            {
                "fill_id": "fill-aaa-1",
                "qty": 10.0,
                "price": 101.0,
                "commission": 1.0,
                "timestamp": submission_timestamp,
            }
        ],
    }

    recon = ReconciliationEngine(
        broker,
        order_blotter=blotter,
        repo_root=tmp_path,
    )
    submission_date = blotter.list_orders(latest_only=True)[0].submission_timestamp[:10]
    result = recon.reconcile_and_update_positions(as_of_date=submission_date)

    assert result.reconciliation_ok is True
    assert result.fills_written == 1
    assert Path(result.positions_path).exists()
    assert Path(result.reconciliation_path).exists()
    fills_path = tmp_path / "paper_trading" / "blotter" / "fills.jsonl"
    assert fills_path.exists()
    payload = Path(result.positions_path).read_text(encoding="utf-8")
    assert "\"AAA\"" in payload


def test_submission_engine_blocks_batch_when_preview_missing_for_any_order(tmp_path: Path) -> None:
    broker = FakeBroker()
    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}

    with pytest.raises(ValueError, match="missing preview coverage"):
        engine.submit_and_poll(
            [
                {
                    "symbol": "AAA",
                    "qty": 10,
                    "side": "SELL",
                    "order_type": "market",
                    "parent_intent_hash": "intent-aaa",
                },
                {
                    "symbol": "BBB",
                    "qty": 8,
                    "side": "BUY",
                    "order_type": "market",
                    "parent_intent_hash": "intent-bbb",
                },
            ],
            {
                "intent-aaa": _preview("AAA", 10, "SELL", expected_price=100.0),
            },
            approval_record,
        )

    assert broker.placed_orders == []
    assert blotter.list_orders(latest_only=True) == []


def test_submission_engine_blocks_batch_when_preview_mismatches_order_request(tmp_path: Path) -> None:
    broker = FakeBroker()
    blotter = OrderBlotter(tmp_path / "orders.jsonl")
    engine = SubmissionEngine(
        broker,
        order_blotter=blotter,
        kill_switch_path=tmp_path / "KILL_SWITCH",
    )
    approval_record = {"status": "APPROVED", "rebalance_id": str(uuid4())}

    with pytest.raises(ValueError, match="preview does not match order request"):
        engine.submit_and_poll(
            [
                {
                    "symbol": "AAA",
                    "qty": 10,
                    "side": "SELL",
                    "order_type": "market",
                    "parent_intent_hash": "intent-aaa",
                }
            ],
            {
                "intent-aaa": _preview("AAA", 9, "SELL", expected_price=100.0),
            },
            approval_record,
        )

    assert broker.placed_orders == []
    assert blotter.list_orders(latest_only=True) == []


def _preview(
    symbol: str,
    qty: float,
    side: str,
    *,
    order_type: str = "market",
    limit_price: float | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": symbol,
        "quantity": qty,
        "side": side,
        "type": order_type,
        "limit_price": limit_price,
    }
    payload.update(extra)
    return payload
