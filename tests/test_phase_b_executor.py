from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import run_paper_phase_a  # noqa: E402
from paper.phase_b_executor import PhaseBExecutor  # noqa: E402
from paper.brokerage.interface import BrokerClient  # noqa: E402
from paper.order_blotter import OrderBlotter, OrderStatus  # noqa: E402


class ExecutionMockBroker(BrokerClient):
    def __init__(self, *, drift_weights: dict[str, float] | None = None) -> None:
        self._profile = {
            "account_id": "PAPER-12345",
            "account_type": "paper",
            "status": "active",
        }
        self._balances = {
            "account_id": "PAPER-12345",
            "account_type": "paper",
            "cash": 10000.0,
            "buying_power": 30000.0,
            "equity": 30000.0,
        }
        self._positions = [
            {
                "ticker": "AAA",
                "quantity": 550.0,
                "cost_basis": 100.0,
                "market_value": 55000.0,
                "current_weight": 0.55,
                "unrealized_pnl": 0.0,
            },
            {
                "ticker": "BBB",
                "quantity": 450.0,
                "cost_basis": 100.0,
                "market_value": 45000.0,
                "current_weight": 0.45,
                "unrealized_pnl": 0.0,
            },
        ]
        self._drift_weights = drift_weights
        self.preview_calls: list[dict[str, Any]] = []
        self.place_calls: list[dict[str, Any]] = []
        self._orders: dict[str, dict[str, Any]] = {}

    def ping(self) -> bool:
        return True

    def get_profile(self) -> dict[str, Any]:
        return dict(self._profile)

    def get_balances(self) -> dict[str, Any]:
        return dict(self._balances)

    def get_positions(self) -> list[dict[str, Any]]:
        if self._drift_weights is not None:
            return [
                {
                    "ticker": ticker,
                    "quantity": 100.0,
                    "cost_basis": 100.0,
                    "market_value": 10000.0,
                    "current_weight": weight,
                    "unrealized_pnl": 0.0,
                }
                for ticker, weight in self._drift_weights.items()
            ]
        total_notional = sum(abs(position["market_value"]) for position in self._positions)
        rows = []
        for position in self._positions:
            row = dict(position)
            row["current_weight"] = row["market_value"] / total_notional if total_notional else 0.0
            rows.append(row)
        return rows

    def preview_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "expected_price": 100.0,
            "reference_price": 100.0,
        }
        self.preview_calls.append(payload)
        return payload

    def place_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
        preview_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        broker_order_id = f"broker-{len(self._orders) + 1}"
        self._orders[broker_order_id] = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "preview_result": dict(preview_result or {}),
        }
        self.place_calls.append(dict(self._orders[broker_order_id]))
        return {"broker_order_id": broker_order_id}

    def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        return {"broker_order_id": broker_order_id, "status": "canceled"}

    def replace_order(
        self,
        broker_order_id: str,
        *,
        qty: float | None = None,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        return {
            "broker_order_id": broker_order_id,
            "qty": qty,
            "limit_price": limit_price,
        }

    def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        return {"broker_order_id": broker_order_id, "status": "filled"}

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        order = self._orders[broker_order_id]
        symbol = order["symbol"]
        qty = float(order["qty"])
        side = order["side"]
        price = 101.0 if symbol == "AAA" else 99.5
        self._apply_fill(symbol=symbol, qty=qty, side=side, price=price)
        fill_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "status": "filled",
            "timestamp": fill_timestamp,
            "filled_qty": qty,
            "avg_fill_price": price,
            "commission": 1.0,
            "fills": [
                {
                    "fill_id": str(uuid4()),
                    "broker_fill_id": f"fill-{broker_order_id}",
                    "qty": qty,
                    "price": price,
                    "commission": 1.0,
                    "timestamp": fill_timestamp,
                    "expected_price": 100.0,
                }
            ],
        }

    def _apply_fill(self, *, symbol: str, qty: float, side: str, price: float) -> None:
        sign = 1.0 if side == "buy" else -1.0
        matched = False
        for position in self._positions:
            if position["ticker"] == symbol:
                position["quantity"] += sign * qty
                position["market_value"] = position["quantity"] * price
                matched = True
                break
        if not matched:
            self._positions.append(
                {
                    "ticker": symbol,
                    "quantity": sign * qty,
                    "cost_basis": price,
                    "market_value": sign * qty * price,
                    "current_weight": 0.0,
                    "unrealized_pnl": 0.0,
                }
            )


class BrokenBalanceBroker(ExecutionMockBroker):
    def get_balances(self) -> dict[str, Any]:
        raise RuntimeError("balance failure")


def test_phase_b_executor_submits_and_reconciles(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=_base_config(baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approved_at": _valid_approval_timestamp(intents),
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")

    broker = ExecutionMockBroker()
    executor = PhaseBExecutor(config=_base_config(baseline_dir), repo_root=repo_root, broker_client=broker)
    result = executor.run(bundle_dir=bundle_dir)

    assert result["executed_trade_count"] == 2
    assert result["unfilled_order_count"] == 0
    reconciliation = json.loads((bundle_dir / "reconciliation_report.json").read_text(encoding="utf-8"))
    assert reconciliation["executed_trades"]
    assert (repo_root / "paper_trading" / "blotter" / "fills.jsonl").exists()
    orders = [
        json.loads(line)
        for line in (repo_root / "paper_trading" / "blotter" / "orders.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["status"] == "FILLED" for row in orders)


def test_phase_b_executor_default_mock_backend_replays_bundle_state(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config={
            **_base_config(baseline_dir),
            "submission_poll_seconds": 0,
            "submission_poll_interval_seconds": 0,
            "mock_fill_after_polls": 1,
        },
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approved_at": _valid_approval_timestamp(intents),
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")
    unrelated_blotter = OrderBlotter(repo_root / "paper_trading" / "blotter" / "orders.jsonl")
    unrelated_blotter.create_order(
        rebalance_id=uuid4(),
        symbol="ZZZ",
        side="SELL",
        qty=1.0,
        order_type="market",
        status=OrderStatus.APPROVED,
        parent_intent_hash="unrelated-intent",
    )

    executor = PhaseBExecutor(
        config={
            **_base_config(baseline_dir),
            "submission_poll_seconds": 0,
            "submission_poll_interval_seconds": 0,
            "mock_fill_after_polls": 1,
        },
        repo_root=repo_root,
    )
    result = executor.run(bundle_dir=bundle_dir)

    assert result["status"] == "SUCCESS"
    assert result["executed_trade_count"] == 2
    assert result["reconciliation_ok"] is True
    reconciliation = json.loads((bundle_dir / "reconciliation_report.json").read_text(encoding="utf-8"))
    actual_weights = reconciliation["actual_weights"]
    assert actual_weights["AAA"] == pytest.approx(0.6, rel=1e-4)
    assert actual_weights["BBB"] == pytest.approx(0.4, rel=1e-4)


def test_phase_b_executor_halts_on_overnight_drift(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=_base_config(baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approved_at": _valid_approval_timestamp(intents),
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")

    broker = ExecutionMockBroker(drift_weights={"AAA": 0.10, "BBB": 0.90})
    executor = PhaseBExecutor(config=_base_config(baseline_dir), repo_root=repo_root, broker_client=broker)
    with pytest.raises(RuntimeError, match="Overnight drift check failed"):
        executor.run(bundle_dir=bundle_dir)


def test_phase_b_executor_rejects_legacy_approval_without_approved_at(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=_base_config(baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")

    broker = ExecutionMockBroker()
    executor = PhaseBExecutor(config=_base_config(baseline_dir), repo_root=repo_root, broker_client=broker)
    with pytest.raises(ValueError, match="approved_at"):
        executor.run(bundle_dir=bundle_dir)

    summary = json.loads((bundle_dir / "phase_b_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["failure_stage"] == "resolve_approval"


def test_phase_b_executor_rejects_approval_after_deadline(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=_base_config(baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents_path = bundle_dir / "intents.json"
    intents = json.loads(intents_path.read_text(encoding="utf-8"))
    generated_at = datetime.fromisoformat(intents["generated_at"].replace("Z", "+00:00"))
    approval_deadline = generated_at + timedelta(minutes=5)
    intents["approval_deadline"] = approval_deadline.isoformat().replace("+00:00", "Z")
    intents_path.write_text(json.dumps(intents), encoding="utf-8")
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approved_at": (approval_deadline + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")

    broker = ExecutionMockBroker()
    executor = PhaseBExecutor(config=_base_config(baseline_dir), repo_root=repo_root, broker_client=broker)
    with pytest.raises(ValueError, match="approval deadline"):
        executor.run(bundle_dir=bundle_dir)

    summary = json.loads((bundle_dir / "phase_b_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["failure_stage"] == "resolve_approval"


def test_phase_b_executor_writes_failure_artifacts_when_reconciliation_raises(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    phase_a_result = run_paper_phase_a.run_phase_a(
        config=_base_config(baseline_dir),
        repo_root=repo_root,
        as_of_date=as_of_date,
    )
    bundle_dir = Path(phase_a_result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    approval_record = {
        "status": "APPROVED",
        "approved": True,
        "approved_at": _valid_approval_timestamp(intents),
        "approver": "tester",
        "config_hash": intents["config_hash"],
        "rebalance_id": intents["rebalance_id"],
    }
    (bundle_dir / "approval.json").write_text(json.dumps(approval_record), encoding="utf-8")

    broker = BrokenBalanceBroker()
    executor = PhaseBExecutor(config=_base_config(baseline_dir), repo_root=repo_root, broker_client=broker)
    with pytest.raises(RuntimeError, match="balance failure"):
        executor.run(bundle_dir=bundle_dir)

    summary = json.loads((bundle_dir / "phase_b_summary.json").read_text(encoding="utf-8"))
    report_payload = json.loads((bundle_dir / "reconciliation_report.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["failure_stage"] == "reconcile"
    assert Path(summary["reconciliation_report"]).exists()
    assert report_payload["status"] == "FAILED"
    assert report_payload["error_type"] == "RuntimeError"
    assert "balance failure" in report_payload["error"]


def _base_config(baseline_dir: Path) -> dict[str, Any]:
    return {
        "stage": 3,
        "broker": "mock",
        "capital_mode": "paper",
        "account_id": "PAPER-12345",
        "baseline_path": str(baseline_dir),
        "signal_max_staleness_days": 365,
        "results_dir": "results",
        "portfolio_notional_usd": 100000.0,
        "min_trade_notional_usd": 50.0,
        "max_single_order_notional_usd": 75000.0,
        "allowed_order_types": ["market"],
        "approval_deadline_time_et": "08:00",
        "overnight_drift_halt_pct": 5.0,
        "submission_poll_seconds": 5,
        "submission_poll_interval_seconds": 1,
    }


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo_root = tmp_path / "repo"
    results_dir = repo_root / "results" / "_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "CURRENT_BASELINE.md").write_text(
        "# Current Baseline Status\n- Current adopted clean canonical baseline: `results/_baseline`\n",
        encoding="utf-8",
    )

    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    signal = pd.DataFrame(
        {
            "AAA": [0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
            "BBB": [0.80, 0.81, 0.82, 0.83, 0.84, 0.85],
            "CCC": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
        },
        index=dates,
    )
    signal.to_parquet(results_dir / "final_signal.parquet")

    returns = pd.DataFrame(
        {
            "AAA": [0.0, 0.01, -0.01, 0.02, 0.00, 0.01],
            "BBB": [0.0, 0.00, 0.01, -0.01, 0.01, 0.00],
            "CCC": [0.0, -0.01, 0.00, 0.01, -0.01, 0.00],
        },
        index=dates,
    )
    returns.to_parquet(results_dir / "returns_panel.parquet")

    weights_history = pd.DataFrame(
        {
            "AAA": [0.55, 0.55, 0.60, 0.60, 0.60, 0.60],
            "BBB": [0.45, 0.45, 0.40, 0.40, 0.40, 0.40],
            "CCC": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    weights_history.to_parquet(results_dir / "weights_history.parquet")

    run_config = {
        "args": {
            "n_positions": 2,
            "weighting": "signal",
            "concentration": 1.0,
            "max_weight": 0.60,
            "max_sector_pct": 1.0,
            "cash_in_bear": 0.0,
            "use_vol_buckets": False,
            "max_selection_pool": 100,
            "spy_core": 0.0,
            "spy_ticker": "SPY",
            "force_mega_caps": False,
            "signal_smooth_halflife": 0.0,
            "apply_rank_normal": False,
            "min_holding_overlap": 0.0,
            "mid_month_refresh": False,
            "min_adv_for_selection": 0.0,
            "max_stock_vol": 1.0,
            "quality_percentile": 0.0,
            "quality_tilt": 0.0,
            "vol_target": 0.0,
            "no_vol_target": True,
            "max_leverage": 1.0,
            "min_leverage": 1.0,
            "vol_floor": 0.08,
            "vol_ceiling": 0.25,
            "credit_overlay": False,
            "use_sector_map": False,
            "bsc_scaling": False,
        }
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    return repo_root, results_dir, "2026-01-06"


def _valid_approval_timestamp(intents: dict[str, Any]) -> str:
    generated_at = datetime.fromisoformat(intents["generated_at"].replace("Z", "+00:00"))
    approval_deadline = datetime.fromisoformat(intents["approval_deadline"].replace("Z", "+00:00"))
    midpoint = generated_at + (approval_deadline - generated_at) / 2
    return midpoint.isoformat().replace("+00:00", "Z")
