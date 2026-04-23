from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.alerting import build_daily_digest, send_alert  # noqa: E402
from paper.auto_approval_gate import auto_approve  # noqa: E402
from paper.calendar import get_rebalance_dates, is_rebalance_day  # noqa: E402
from paper.canary_mode import (  # noqa: E402
    evaluate_canary_orders,
    is_canary_mode_active,
    record_canary_day,
)
from paper.drawdown_circuit_breaker import check_drawdown  # noqa: E402
from paper.post_run_reconciliation import generate_reconciliation_report  # noqa: E402
from paper.preflight_gate import (  # noqa: E402
    activate_kill_switch,
    clear_kill_switch,
    preflight_passes,
)
from paper.verify import compute_config_hash  # noqa: E402


class _Broker:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def ping(self) -> bool:
        return self.healthy


def test_calendar_matches_strategy_day_offset_and_override() -> None:
    index = pd.bdate_range("2026-01-01", "2026-03-31")
    dates = get_rebalance_dates(2026, signal_index=index)
    assert pd.Timestamp("2026-01-05") in dates
    assert is_rebalance_day("2026-02-04", signal_index=index) is True
    assert is_rebalance_day("2026-02-03", signal_index=index) is False

    explicit = {
        "rebalance_calendar_source": "explicit",
        "rebalance_calendar": [{"date": "2026-02-17", "config_hash": "abc"}],
    }
    assert get_rebalance_dates(2026, signal_index=index, config=explicit) == {
        pd.Timestamp("2026-02-17")
    }


def test_preflight_and_kill_switch(tmp_path: Path) -> None:
    repo_root = tmp_path
    signal_path = repo_root / "results" / "_baseline" / "final_signal.parquet"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text("signal", encoding="utf-8")

    config = {
        "stage": 4,
        "broker": "tradier_sandbox",
        "capital_mode": "paper",
        "account_id": "ACC-1",
        "baseline_path": str(signal_path.parent),
        "kill_switch_path": str(repo_root / "paper_trading" / "state" / "KILL_SWITCH"),
        "alert_log_path": str(repo_root / "paper_trading" / "monitoring_alerts.jsonl"),
    }
    context = {
        "config": config,
        "config_path": repo_root / "config.yaml",
        "results_dir": repo_root / "paper_trading" / "reports",
        "manifest": {"config_hash": compute_config_hash(config)},
        "signal_path": signal_path,
        "broker_client": _Broker(True),
        "prior_reconciliation": {"status": "CLEAN", "unresolved_diff": 0.0},
        "reconciliation_tolerance": 0.01,
    }
    ok, reasons = preflight_passes(context)
    assert ok is True
    assert reasons == []

    activate_kill_switch(config["kill_switch_path"], reason="manual test", operator="tester")
    blocked, reasons = preflight_passes(context)
    assert blocked is False
    assert any("kill switch" in reason for reason in reasons)
    alert_records = [
        json.loads(line)
        for line in Path(config["alert_log_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(alert_records) == 1
    assert alert_records[0]["alert_type"] == "preflight_failure"
    assert alert_records[0]["severity"] == "error"
    assert alert_records[0]["details"]["passed"] is False
    assert "kill_switch" in alert_records[0]["details"]["failing_checks"]
    clear_kill_switch(config["kill_switch_path"], reason="clear", operator="tester")


def test_auto_approve_is_idempotent(tmp_path: Path) -> None:
    approval_path = tmp_path / "paper_trading" / "reports" / "2026-04-22" / "approvals.jsonl"
    context = {
        "config": {"stage": 4},
        "config_hash": "cfg",
        "signal_hash": "sig",
        "is_rebalance_day": True,
        "approval_log_path": approval_path,
        "as_of_date": "2026-04-22",
    }
    previews = [{"status": "OK", "symbol": "AAA", "qty": 10}]
    approved, _ = auto_approve(context, previews, (True, []))
    assert approved is True
    approved, _ = auto_approve(context, previews, (True, []))
    assert approved is True
    lines = approval_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_drawdown_breaker_creates_kill_switch(tmp_path: Path) -> None:
    alert_log_path = tmp_path / "paper_trading" / "monitoring_alerts.jsonl"
    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    kill_switch_path = tmp_path / "paper_trading" / "state" / "KILL_SWITCH"
    passed, reason = check_drawdown(
        [
            {"trade_date": "2026-04-22", "realized_pnl": -7000.0},
            {"trade_date": "2026-01-05", "realized_pnl": -25000.0},
        ],
        initial_capital=100000.0,
        trading_day="2026-04-22",
        kill_switch_path=kill_switch_path,
        log_path=tmp_path / "paper_trading" / "circuit_breaker_log.jsonl",
        alert_log_path=alert_log_path,
        current_time=now,
    )
    assert passed is False
    assert "YTD DD" in reason
    assert kill_switch_path.exists()
    alert_records = [
        json.loads(line)
        for line in alert_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["alert_type"] for record in alert_records] == [
        "circuit_breaker_triggered",
        "kill_switch_activated",
    ]
    assert all(record["severity"] == "critical" for record in alert_records)
    assert alert_records[0]["details"]["decision"] == "ytd_drawdown_breach"
    assert alert_records[1]["details"]["kill_switch_path"] == str(kill_switch_path)


def test_canary_caps_and_tracking(tmp_path: Path) -> None:
    path = tmp_path / "paper_trading" / "canary_state.json"
    state = record_canary_day(path, as_of_date="2026-04-22", duration_trading_days=2)
    assert state["trading_day_count"] == 1
    assert is_canary_mode_active(state["trading_day_count"], duration_trading_days=2) is True
    state = record_canary_day(path, as_of_date="2026-04-23", duration_trading_days=2)
    assert state["active"] is False

    report = evaluate_canary_orders(
        [
            {"symbol": "AAA", "delta_notional": 200.0},
            {"symbol": "BBB", "delta_notional": 600.0},
            {"symbol": "CCC", "delta_notional": 100.0},
        ],
        max_single_order_notional_usd=500.0,
        max_orders_per_day=1,
    )
    assert len(report["accepted_orders"]) == 1
    assert len(report["capped_orders"]) == 1
    assert len(report["deferred_orders"]) == 1


def test_reconciliation_report_and_alert_digest(tmp_path: Path) -> None:
    report_path = generate_reconciliation_report(
        "2026-04-22",
        {
            "repo_root": tmp_path,
            "target_weights": {"AAA": 0.6, "BBB": 0.4},
            "actual_weights": {"AAA": 0.57, "BBB": 0.43},
            "intended_trades": [{"ticker": "AAA"}],
            "fills": [
                {
                    "symbol": "AAA",
                    "side": "buy",
                    "qty": 10,
                    "price": 101.0,
                    "expected_price": 100.0,
                    "timestamp": "2026-04-22T16:00:00Z",
                    "submission_timestamp": "2026-04-22T15:50:00Z",
                }
            ],
        },
    )
    assert Path(report_path).exists()
    text = Path(report_path).read_text(encoding="utf-8")
    assert "Status: ALERT_REQUIRED" in text

    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    send_alert(
        "preflight_failure",
        "composite gate failed",
        "critical",
        config={
            "alert_log_path": str(tmp_path / "paper_trading" / "monitoring_alerts.jsonl"),
            "desktop_alerts_enabled": False,
            "email_alerts_enabled": False,
            "alert_mute_types": [],
        },
        current_time=now,
    )
    digest = build_daily_digest(
        "2026-04-22",
        alert_log_path=tmp_path / "paper_trading" / "monitoring_alerts.jsonl",
    )
    assert digest["count"] == 1
    assert digest["by_severity"]["critical"] == 1
