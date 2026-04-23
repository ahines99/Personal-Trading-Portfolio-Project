from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_AWAITING_APPROVAL = 10
EXIT_NOT_APPROVED = 11

STATUS_SUCCESS = "SUCCESS"
STATUS_AWAITING_APPROVAL = "AWAITING_APPROVAL"
STATUS_APPROVED_READY = "APPROVED_READY_FOR_PHASE_B"
STATUS_REJECTED = "REJECTED"
STATUS_FAILED = "FAILED"
STATUS_KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"

PHASE_B_FAILURE_ARTIFACT = "paper_daily_runner_phase_b_failure.json"


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    parsed: dict[str, Any] | None

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": self.command,
            "returncode": self.returncode,
        }
        if self.parsed is not None:
            payload["result"] = self.parsed
        if self.returncode != 0:
            payload["stdout_tail"] = _tail(self.stdout)
            payload["stderr_tail"] = _tail(self.stderr)
        return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe daily paper-trading runner for Phase A, Phase B, or Stage 4 auto orchestration."
    )
    parser.add_argument(
        "--mode",
        choices=("phase-a", "phase-b", "auto"),
        default="auto",
        help="phase-a generates intents, phase-b executes an approved bundle, auto runs A then B only when approved.",
    )
    parser.add_argument("--config", default=None, help="Optional YAML/JSON paper config path")
    parser.add_argument("--as-of-date", default=None, help="Phase A as-of date override (YYYY-MM-DD)")
    parser.add_argument("--bundle-dir", default=None, help="Phase B bundle directory")
    parser.add_argument(
        "--output-dir",
        "--results-dir",
        dest="output_dir",
        default=None,
        help="Phase A output root; passed to run_paper_phase_a.py as --results-dir",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke phase entrypoints.",
    )
    parser.add_argument(
        "--chain-approved-phase-b",
        action="store_true",
        help=(
            "Run Phase B immediately when auto-generated Phase A is already APPROVED. "
            "Leave unset for the normal T+1 contract."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent

    try:
        summary, exit_code = run_daily(args, repo_root=repo_root)
    except Exception as exc:
        summary = {
            "mode": getattr(args, "mode", None),
            "status": STATUS_FAILED,
            "exit_code": EXIT_ERROR,
            "error": str(exc),
        }
        exit_code = EXIT_ERROR

    summary["exit_code"] = exit_code
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


def run_daily(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    runner: Any = subprocess.run,
) -> tuple[dict[str, Any], int]:
    config = _load_runner_config(getattr(args, "config", None), repo_root=repo_root)
    kill_switch_path = _resolve_kill_switch_path(repo_root, config)
    summary: dict[str, Any] = {
        "mode": args.mode,
        "status": STATUS_SUCCESS,
        "phase_a": None,
        "phase_b": None,
        "kill_switch_path": str(kill_switch_path),
    }

    if args.mode in {"phase-a", "auto"}:
        kill_switch_error = _kill_switch_error(kill_switch_path)
        if kill_switch_error is not None:
            summary["status"] = STATUS_KILL_SWITCH_ACTIVE
            summary["error"] = kill_switch_error
            summary["phase_a"] = _blocked_step_summary(
                name="phase_a",
                command=_phase_a_command(args, repo_root=repo_root),
                error=kill_switch_error,
                kill_switch_path=kill_switch_path,
            )
            return summary, EXIT_ERROR

    if args.mode == "phase-a":
        phase_a = _run_phase_a(args, repo_root=repo_root, runner=runner)
        summary["phase_a"] = phase_a.summary()
        if phase_a.returncode != 0:
            summary["status"] = STATUS_FAILED
            return summary, EXIT_ERROR
        bundle_dir = _bundle_dir_from_phase_a(phase_a, args, repo_root)
        summary["bundle_dir"] = str(bundle_dir) if bundle_dir is not None else None
        return summary, EXIT_OK

    if args.mode == "phase-b":
        kill_switch_error = _kill_switch_error(kill_switch_path)
        if kill_switch_error is not None:
            summary["status"] = STATUS_KILL_SWITCH_ACTIVE
            summary["error"] = kill_switch_error
            summary["phase_b"] = _blocked_step_summary(
                name="phase_b",
                command=_phase_b_command(args, repo_root=repo_root, bundle_dir=args.bundle_dir),
                error=kill_switch_error,
                kill_switch_path=kill_switch_path,
            )
            return summary, EXIT_ERROR
        phase_b = _run_phase_b(args, repo_root=repo_root, runner=runner, bundle_dir=args.bundle_dir)
        summary["phase_b"] = phase_b.summary()
        if phase_b.returncode != 0:
            summary["status"] = STATUS_FAILED
            return summary, EXIT_ERROR
        if phase_b.parsed and phase_b.parsed.get("bundle_dir"):
            summary["bundle_dir"] = phase_b.parsed["bundle_dir"]
        return summary, EXIT_OK

    phase_a = _run_phase_a(args, repo_root=repo_root, runner=runner)
    summary["phase_a"] = phase_a.summary()
    if phase_a.returncode != 0:
        summary["status"] = STATUS_FAILED
        return summary, EXIT_ERROR

    bundle_dir = _bundle_dir_from_phase_a(phase_a, args, repo_root)
    if bundle_dir is None:
        summary["status"] = STATUS_FAILED
        summary["error"] = "Phase A completed but no _paper_phase_a_* bundle was found."
        return summary, EXIT_ERROR

    summary["bundle_dir"] = str(bundle_dir)
    approval_status = _intent_status(bundle_dir) if bundle_dir is not None else None
    summary["approval_status"] = approval_status

    if approval_status is None:
        summary["status"] = STATUS_FAILED
        summary["error"] = f"Phase A bundle is missing a readable status: {bundle_dir / 'intents.json'}"
        return summary, EXIT_ERROR

    if approval_status != "APPROVED":
        summary["status"] = _status_for_unapproved_bundle(approval_status)
        return summary, _exit_for_unapproved_bundle(approval_status)

    if not bool(getattr(args, "chain_approved_phase_b", False)):
        summary["status"] = STATUS_APPROVED_READY
        summary["phase_b_skipped_reason"] = (
            "Bundle is approved. Run --mode phase-b against this bundle at the "
            "next-open Phase B window."
        )
        return summary, EXIT_OK

    kill_switch_error = _kill_switch_error(kill_switch_path)
    if kill_switch_error is not None:
        summary["status"] = STATUS_KILL_SWITCH_ACTIVE
        summary["error"] = kill_switch_error
        summary["phase_b"] = _blocked_step_summary(
            name="phase_b",
            command=_phase_b_command(args, repo_root=repo_root, bundle_dir=str(bundle_dir)),
            error=kill_switch_error,
            kill_switch_path=kill_switch_path,
        )
        artifact_path = _write_phase_b_failure_artifact(
            bundle_dir,
            kill_switch_path=kill_switch_path,
            failure_reason=kill_switch_error,
            phase_b=summary["phase_b"],
        )
        summary["phase_b_failure_artifact"] = str(artifact_path)
        return summary, EXIT_ERROR

    phase_b = _run_phase_b(args, repo_root=repo_root, runner=runner, bundle_dir=str(bundle_dir))
    summary["phase_b"] = phase_b.summary()
    if phase_b.returncode != 0:
        summary["status"] = STATUS_FAILED
        artifact_path = _write_phase_b_failure_artifact(
            bundle_dir,
            kill_switch_path=kill_switch_path,
            failure_reason="Chained Phase B failed after Phase A completed successfully.",
            phase_b=summary["phase_b"],
        )
        summary["phase_b_failure_artifact"] = str(artifact_path)
        summary["error"] = (
            "Chained Phase B failed. Approval state is unchanged; investigate the bundle artifact "
            "and rerun --mode phase-b against this bundle in the next Phase B window."
        )
        return summary, EXIT_ERROR
    return summary, EXIT_OK


def _run_phase_a(args: argparse.Namespace, *, repo_root: Path, runner: Any) -> StepResult:
    command = _phase_a_command(args, repo_root=repo_root)
    return _run_step("phase_a", command, repo_root=repo_root, runner=runner)


def _run_phase_b(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    runner: Any,
    bundle_dir: str | None,
) -> StepResult:
    command = _phase_b_command(args, repo_root=repo_root, bundle_dir=bundle_dir)
    return _run_step("phase_b", command, repo_root=repo_root, runner=runner)


def _phase_a_command(args: argparse.Namespace, *, repo_root: Path) -> list[str]:
    command = [args.python, str(repo_root / "run_paper_phase_a.py")]
    if args.config:
        command.extend(["--config", args.config])
    if args.as_of_date:
        command.extend(["--as-of-date", args.as_of_date])
    if args.output_dir:
        command.extend(["--results-dir", args.output_dir])
    return command


def _phase_b_command(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    bundle_dir: str | None,
) -> list[str]:
    command = [args.python, str(repo_root / "run_paper_phase_b.py")]
    if args.config:
        command.extend(["--config", args.config])
    if bundle_dir:
        command.extend(["--bundle-dir", bundle_dir])
    return command


def _run_step(name: str, command: list[str], *, repo_root: Path, runner: Any) -> StepResult:
    completed = runner(
        command,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return StepResult(
        name=name,
        command=command,
        returncode=int(completed.returncode),
        stdout=stdout,
        stderr=stderr,
        parsed=_extract_json_object(stdout),
    )


def _bundle_dir_from_phase_a(
    phase_a: StepResult,
    args: argparse.Namespace,
    repo_root: Path,
) -> Path | None:
    parsed_bundle = (phase_a.parsed or {}).get("bundle_dir")
    if parsed_bundle:
        candidate = Path(str(parsed_bundle))
        return candidate if candidate.is_absolute() else repo_root / candidate

    output_root = Path(str(args.output_dir or "results"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    candidates = sorted(output_root.glob("_paper_phase_a_*"))
    return candidates[-1] if candidates else None


def _intent_status(bundle_dir: Path | None) -> str | None:
    if bundle_dir is None:
        return None
    intents_path = bundle_dir / "intents.json"
    if not intents_path.exists():
        return None
    payload = json.loads(intents_path.read_text(encoding="utf-8"))
    return str(payload.get("status") or "").strip().upper() or None


def _status_for_unapproved_bundle(approval_status: str | None) -> str:
    if approval_status in {"", "AWAITING_APPROVAL"}:
        return STATUS_AWAITING_APPROVAL
    if approval_status == "REJECTED":
        return STATUS_REJECTED
    return f"NOT_APPROVED:{approval_status}"


def _exit_for_unapproved_bundle(approval_status: str | None) -> int:
    if approval_status in {"", "AWAITING_APPROVAL"}:
        return EXIT_AWAITING_APPROVAL
    return EXIT_NOT_APPROVED


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _tail(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _load_runner_config(path: str | None, *, repo_root: Path) -> dict[str, Any]:
    if not path:
        return {}
    try:
        from src.paper.loader import load_config as external_load_config

        loaded = external_load_config(path)
        if hasattr(loaded, "model_dump"):
            return dict(loaded.model_dump())
        if hasattr(loaded, "dict"):
            return dict(loaded.dict())
    except Exception:
        pass
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _coerce_scalar(value.strip())
    return config


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _resolve_kill_switch_path(repo_root: Path, config: dict[str, Any]) -> Path:
    candidate = Path(str(config.get("kill_switch_path") or "paper_trading/state/KILL_SWITCH"))
    return candidate if candidate.is_absolute() else repo_root / candidate


def _kill_switch_error(kill_switch_path: Path) -> str | None:
    if kill_switch_path.exists():
        return f"Kill switch engaged: {kill_switch_path}"
    return None


def _blocked_step_summary(
    *,
    name: str,
    command: list[str],
    error: str,
    kill_switch_path: Path,
) -> dict[str, Any]:
    return {
        "name": name,
        "command": command,
        "returncode": None,
        "blocked_by_kill_switch": True,
        "kill_switch_path": str(kill_switch_path),
        "error": error,
    }


def _write_phase_b_failure_artifact(
    bundle_dir: Path,
    *,
    kill_switch_path: Path,
    failure_reason: str,
    phase_b: dict[str, Any] | None,
) -> Path:
    artifact_path = bundle_dir / PHASE_B_FAILURE_ARTIFACT
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PHASE_B_FAILED_AFTER_PHASE_A",
        "failure_reason": failure_reason,
        "kill_switch_path": str(kill_switch_path),
        "next_action": (
            "Leave approval state unchanged. Investigate the failure, clear the kill switch if needed, "
            "and rerun `run_paper_daily.py --mode phase-b --bundle-dir <bundle_dir>` in the next Phase B window."
        ),
        "phase_b": phase_b,
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


if __name__ == "__main__":
    raise SystemExit(main())
