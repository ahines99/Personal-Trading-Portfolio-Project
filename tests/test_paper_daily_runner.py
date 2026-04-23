from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_paper_daily  # noqa: E402


def test_auto_stops_after_approved_phase_a_by_default(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, "APPROVED")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        return _completed({"bundle_dir": str(bundle_dir), "order_count": 1})

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_OK
    assert summary["status"] == run_paper_daily.STATUS_APPROVED_READY
    assert summary["approval_status"] == "APPROVED"
    assert summary["phase_b"] is None
    assert "next-open Phase B" in summary["phase_b_skipped_reason"]
    assert len(calls) == 1


def test_auto_can_chain_phase_b_when_explicitly_requested(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, "APPROVED")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        if command[1].endswith("run_paper_phase_a.py"):
            return _completed({"bundle_dir": str(bundle_dir), "order_count": 1})
        return _completed({"bundle_dir": str(bundle_dir), "executed_trade_count": 1})

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto", chain_approved_phase_b=True),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_OK
    assert summary["status"] == run_paper_daily.STATUS_SUCCESS
    assert summary["approval_status"] == "APPROVED"
    assert len(calls) == 2
    assert calls[1][-2:] == ["--bundle-dir", str(bundle_dir)]


def test_auto_exits_awaiting_approval_without_running_phase_b(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, "AWAITING_APPROVAL")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        return _completed({"bundle_dir": str(bundle_dir), "order_count": 1})

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_AWAITING_APPROVAL
    assert summary["status"] == run_paper_daily.STATUS_AWAITING_APPROVAL
    assert summary["approval_status"] == "AWAITING_APPROVAL"
    assert len(calls) == 1


def test_auto_stops_before_phase_a_when_kill_switch_is_active(tmp_path: Path) -> None:
    kill_switch_path = tmp_path / "paper_trading" / "state" / "KILL_SWITCH"
    kill_switch_path.parent.mkdir(parents=True)
    kill_switch_path.write_text("halt", encoding="utf-8")

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        raise AssertionError(f"runner should not be called: {command}")

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_ERROR
    assert summary["status"] == run_paper_daily.STATUS_KILL_SWITCH_ACTIVE
    assert summary["phase_a"]["blocked_by_kill_switch"] is True
    assert summary["phase_a"]["kill_switch_path"] == str(kill_switch_path)


def test_phase_b_uses_configured_kill_switch_path_for_top_level_guard(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "paper.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("kill_switch_path: custom/halts/STOP\n", encoding="utf-8")
    kill_switch_path = tmp_path / "custom" / "halts" / "STOP"
    kill_switch_path.parent.mkdir(parents=True)
    kill_switch_path.write_text("halt", encoding="utf-8")

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        raise AssertionError(f"runner should not be called: {command}")

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="phase-b", config=str(config_path), bundle_dir="results/_paper_phase_a_1"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_ERROR
    assert summary["status"] == run_paper_daily.STATUS_KILL_SWITCH_ACTIVE
    assert summary["kill_switch_path"] == str(kill_switch_path)
    assert summary["phase_b"]["blocked_by_kill_switch"] is True
    assert summary["phase_b"]["kill_switch_path"] == str(kill_switch_path)


def test_phase_a_passes_config_date_and_output_dir(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        return _completed({"bundle_dir": str(tmp_path / "out" / "_paper_phase_a_1")})

    summary, exit_code = run_paper_daily.run_daily(
        _args(
            mode="phase-a",
            config="config/paper.yaml",
            as_of_date="2026-04-22",
            output_dir=str(tmp_path / "out"),
        ),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_OK
    assert summary["status"] == run_paper_daily.STATUS_SUCCESS
    json.dumps(summary)
    command = calls[0]
    assert command[1].endswith("run_paper_phase_a.py")
    assert "--config" in command
    assert "--as-of-date" in command
    assert "--results-dir" in command


def test_phase_b_passes_config_and_bundle_dir(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        return _completed({"bundle_dir": str(tmp_path / "bundle"), "executed_trade_count": 0})

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="phase-b", config="config/paper.yaml", bundle_dir="results/_paper_phase_a_1"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_OK
    assert summary["status"] == run_paper_daily.STATUS_SUCCESS
    command = calls[0]
    assert command[1].endswith("run_paper_phase_b.py")
    assert "--config" in command
    assert command[-2:] == ["--bundle-dir", "results/_paper_phase_a_1"]


def test_subprocess_failure_returns_failed_summary(tmp_path: Path) -> None:
    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=3, stdout="partial output", stderr="boom")

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="phase-a"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_ERROR
    assert summary["status"] == run_paper_daily.STATUS_FAILED
    assert summary["phase_a"]["returncode"] == 3
    assert summary["phase_a"]["stderr_tail"] == "boom"


def test_auto_fails_when_phase_a_does_not_report_or_write_bundle(tmp_path: Path) -> None:
    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        return _completed({"order_count": 0})

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto", output_dir=str(tmp_path / "empty-results")),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert exit_code == run_paper_daily.EXIT_ERROR
    assert summary["status"] == run_paper_daily.STATUS_FAILED
    assert "no _paper_phase_a_* bundle" in summary["error"]


def test_chained_phase_b_failure_writes_bundle_artifact_without_changing_approval(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path, "APPROVED")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        if command[1].endswith("run_paper_phase_a.py"):
            return _completed({"bundle_dir": str(bundle_dir), "order_count": 1})
        return SimpleNamespace(returncode=2, stdout="phase b partial", stderr="phase b boom")

    summary, exit_code = run_paper_daily.run_daily(
        _args(mode="auto", chain_approved_phase_b=True),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    artifact_path = bundle_dir / run_paper_daily.PHASE_B_FAILURE_ARTIFACT
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))

    assert exit_code == run_paper_daily.EXIT_ERROR
    assert summary["status"] == run_paper_daily.STATUS_FAILED
    assert len(calls) == 2
    assert artifact_path.exists()
    assert summary["phase_b_failure_artifact"] == str(artifact_path)
    assert artifact["status"] == "PHASE_B_FAILED_AFTER_PHASE_A"
    assert "Leave approval state unchanged" in artifact["next_action"]
    assert artifact["phase_b"]["returncode"] == 2
    assert intents["status"] == "APPROVED"


def _args(**overrides: Any) -> Namespace:
    values = {
        "mode": "auto",
        "config": None,
        "as_of_date": None,
        "bundle_dir": None,
        "output_dir": None,
        "python": sys.executable,
        "chain_approved_phase_b": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _completed(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload, indent=2), stderr="")


def _write_bundle(tmp_path: Path, status: str) -> Path:
    bundle_dir = tmp_path / "results" / "_paper_phase_a_20260422_000000"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "intents.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    return bundle_dir
