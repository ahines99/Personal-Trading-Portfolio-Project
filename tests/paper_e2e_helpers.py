from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from paper.brokerage.interface import BrokerClient


class E2EMockBroker(BrokerClient):
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
            "equity": 100000.0,
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

        total_value = sum(abs(float(position["market_value"])) for position in self._positions)
        rows = []
        for position in self._positions:
            row = dict(position)
            row["current_weight"] = float(row["market_value"]) / total_value if total_value else 0.0
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
            "side": str(side).lower(),
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "ok",
            "result": True,
            "expected_price": 100.0,
            "reference_price": 100.0,
            "estimated_notional": abs(float(qty) * 100.0),
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
        order = {
            "symbol": symbol,
            "qty": qty,
            "side": str(side).lower(),
            "order_type": order_type,
            "limit_price": limit_price,
            "preview_result": dict(preview_result or {}),
        }
        self._orders[broker_order_id] = order
        self.place_calls.append(dict(order))
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
        return {"broker_order_id": broker_order_id, "qty": qty, "limit_price": limit_price}

    def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        order = self._orders[broker_order_id]
        qty = float(order["qty"])
        price = _fill_price(order["symbol"])
        fill_timestamp = _utc_now_iso()
        return {
            "broker_order_id": broker_order_id,
            "status": "filled",
            "filled_qty": qty,
            "avg_fill_price": price,
            "commission": 1.0,
            "fills": [
                {
                    "fill_id": f"fill-{broker_order_id}",
                    "broker_fill_id": f"fill-{broker_order_id}",
                    "qty": qty,
                    "price": price,
                    "commission": 1.0,
                    "timestamp": fill_timestamp,
                    "expected_price": 100.0,
                }
            ],
        }

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        del timeout_seconds, poll_interval_seconds
        order = self._orders[broker_order_id]
        qty = float(order["qty"])
        price = _fill_price(order["symbol"])
        self._apply_fill(symbol=order["symbol"], qty=qty, side=order["side"], price=price)
        fill_timestamp = _utc_now_iso()
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
        sign = 1.0 if side.lower() in {"buy", "buy_to_cover"} else -1.0
        matched = False
        for position in self._positions:
            if position["ticker"] == symbol:
                position["quantity"] = float(position["quantity"]) + sign * qty
                position["market_value"] = float(position["quantity"]) * price
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


def base_config(
    repo_root: Path,
    baseline_dir: Path,
    *,
    stage: int = 3,
    stage4_mode: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "stage": stage,
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
        "submission_poll_seconds": 0,
        "submission_poll_interval_seconds": 0,
        "kill_switch_path": str(repo_root / "paper_trading" / "state" / "KILL_SWITCH"),
        "alert_log_path": str(repo_root / "paper_trading" / "monitoring_alerts.jsonl"),
    }
    if stage4_mode is not None:
        config["stage4_mode"] = stage4_mode
    return config


def build_fake_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo_root = tmp_path / "repo"
    baseline_dir = repo_root / "results" / "_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
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
    signal.to_parquet(baseline_dir / "final_signal.parquet")

    returns = pd.DataFrame(
        {
            "AAA": [0.0, 0.01, -0.01, 0.02, 0.00, 0.01],
            "BBB": [0.0, 0.00, 0.01, -0.01, 0.01, 0.00],
            "CCC": [0.0, -0.01, 0.00, 0.01, -0.01, 0.00],
        },
        index=dates,
    )
    returns.to_parquet(baseline_dir / "returns_panel.parquet")

    weights_history = pd.DataFrame(
        {
            "AAA": [0.55, 0.55, 0.60, 0.60, 0.60, 0.60],
            "BBB": [0.45, 0.45, 0.40, 0.40, 0.40, 0.40],
            "CCC": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    weights_history.to_parquet(baseline_dir / "weights_history.parquet")

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
    (baseline_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    return repo_root, baseline_dir, "2026-01-06"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["order_id"]): row for row in rows}


def _fill_price(symbol: str) -> float:
    return 101.0 if symbol == "AAA" else 99.5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
