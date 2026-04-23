from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.daily_gate import (  # noqa: E402
    UnreconciledOrdersException,
    verify_prior_day_reconciliation,
)
from paper.fill_blotter import FillBlotter, FillRecord  # noqa: E402
from paper.order_blotter import OrderBlotter, OrderStatus, OrderSide  # noqa: E402


def test_prior_day_gate_allows_clean_friday_to_monday(tmp_path: Path) -> None:
    order_blotter = OrderBlotter(tmp_path / "orders.jsonl")
    fill_blotter = FillBlotter(tmp_path / "fills.jsonl")
    rebalance_id = uuid4()
    order = order_blotter.create_order(
        rebalance_id=rebalance_id,
        symbol="AAA",
        side="SELL",
        qty=10,
        order_type="market",
        status=OrderStatus.APPROVED,
        parent_intent_hash="intent-aaa",
    )
    order_blotter.update_status(
        order.order_id,
        OrderStatus.SUBMITTED,
        updates={
            "broker_order_id": "broker-aaa",
            "submission_timestamp": "2026-04-17T13:30:00Z",
        },
    )
    filled = order_blotter.update_status(
        order.order_id,
        OrderStatus.FILLED,
        updates={"broker_order_id": "broker-aaa"},
    )
    fill_blotter.append_fill(
        FillRecord(
            fill_id=uuid4(),
            order_id=filled.order_id,
            parent_intent_hash=filled.parent_intent_hash,
            symbol=filled.symbol,
            side=filled.side,
            qty=10.0,
            price=101.0,
            commission=1.0,
            timestamp="2026-04-17T13:31:00Z",
        ),
        submission_timestamp="2026-04-17T13:30:00Z",
    )

    recon_dir = tmp_path / "paper_trading" / "reconciliation"
    recon_dir.mkdir(parents=True, exist_ok=True)
    (recon_dir / "2026-04-17.json").write_text(
        json.dumps({"reconciliation_ok": True, "discrepancies": []}),
        encoding="utf-8",
    )

    assert (
        verify_prior_day_reconciliation(
            as_of_date="2026-04-20",
            order_blotter=order_blotter,
            fill_blotter=fill_blotter,
            repo_root=tmp_path,
        )
        is True
    )


def test_prior_day_gate_blocks_non_terminal_orders(tmp_path: Path) -> None:
    order_blotter = OrderBlotter(tmp_path / "orders.jsonl")
    order = order_blotter.create_order(
        rebalance_id=uuid4(),
        symbol="BBB",
        side="BUY",
        qty=5,
        order_type="market",
        status=OrderStatus.APPROVED,
        parent_intent_hash="intent-bbb",
    )
    order_blotter.update_status(
        order.order_id,
        OrderStatus.SUBMITTED,
        updates={"submission_timestamp": "2026-04-21T13:30:00Z"},
    )

    with pytest.raises(UnreconciledOrdersException) as excinfo:
        verify_prior_day_reconciliation(
            as_of_date="2026-04-22",
            order_blotter=order_blotter,
            fill_blotter=FillBlotter(tmp_path / "fills.jsonl"),
            repo_root=tmp_path,
        )

    assert str(order.order_id) in str(excinfo.value)


def test_prior_day_gate_blocks_fill_qty_mismatch(tmp_path: Path) -> None:
    order_blotter = OrderBlotter(tmp_path / "orders.jsonl")
    fill_blotter = FillBlotter(tmp_path / "fills.jsonl")
    order = order_blotter.create_order(
        rebalance_id=uuid4(),
        symbol="CCC",
        side=OrderSide.BUY,
        qty=20,
        order_type="market",
        status=OrderStatus.APPROVED,
        parent_intent_hash="intent-ccc",
    )
    order_blotter.update_status(
        order.order_id,
        OrderStatus.SUBMITTED,
        updates={
            "broker_order_id": "broker-ccc",
            "submission_timestamp": "2026-04-21T13:30:00Z",
        },
    )
    filled = order_blotter.update_status(
        order.order_id,
        OrderStatus.FILLED,
        updates={"broker_order_id": "broker-ccc"},
    )
    fill_blotter.append_fill(
        FillRecord(
            fill_id=uuid4(),
            order_id=filled.order_id,
            parent_intent_hash=filled.parent_intent_hash,
            symbol=filled.symbol,
            side=filled.side,
            qty=19.0,
            price=10.0,
            commission=0.5,
            timestamp="2026-04-21T13:31:00Z",
        ),
        submission_timestamp="2026-04-21T13:30:00Z",
    )

    with pytest.raises(UnreconciledOrdersException, match="fills 19.000000 vs order 20.000000"):
        verify_prior_day_reconciliation(
            as_of_date="2026-04-22",
            order_blotter=order_blotter,
            fill_blotter=fill_blotter,
            repo_root=tmp_path,
        )
