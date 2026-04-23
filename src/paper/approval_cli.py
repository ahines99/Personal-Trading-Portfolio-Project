from __future__ import annotations

import argparse
import getpass
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, TextIO

from pydantic import ValidationError

try:
    from .models import (
        ApprovalRecord,
        ApprovalStatus,
        IntentBundle,
        IntentBundleStatus,
    )
except ImportError:  # pragma: no cover - script fallback
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from paper.models import (  # type: ignore[no-redef]
            ApprovalRecord,
            ApprovalStatus,
            IntentBundle,
            IntentBundleStatus,
        )
    else:  # pragma: no cover
        raise


DEFAULT_BUNDLE_PATH = Path("paper_trading/current/intents.json")


class ApprovalCliError(RuntimeError):
    """Raised for recoverable approval CLI failures."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Approve or reject a pending Stage 3 paper-trading intent bundle."
    )
    parser.add_argument(
        "bundle_path",
        nargs="?",
        default=str(DEFAULT_BUNDLE_PATH),
        help="Path to intents.json written by Phase A.",
    )
    decision_group = parser.add_mutually_exclusive_group()
    decision_group.add_argument(
        "--approve",
        action="store_true",
        help="Approve the pending bundle.",
    )
    decision_group.add_argument(
        "--reject",
        action="store_true",
        help="Reject the pending bundle.",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Approver identity. Defaults to the current OS user.",
    )
    parser.add_argument(
        "--comment",
        default=None,
        help="Optional rationale. Rejections require a non-empty comment.",
    )
    parser.add_argument(
        "--hostname",
        default=None,
        help="Override the captured hostname for auditability.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final confirmation prompt.",
    )
    return parser


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def load_bundle(path: Path) -> IntentBundle:
    if not path.is_file():
        raise ApprovalCliError(f"Bundle file not found: {path}")
    try:
        return IntentBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ApprovalCliError(f"Failed to load intent bundle: {exc}") from exc


def save_bundle(path: Path, bundle: IntentBundle) -> None:
    try:
        _atomic_write_text(path, bundle.model_dump_json(indent=2))
    except OSError as exc:
        raise ApprovalCliError(f"Failed to persist updated bundle: {exc}") from exc


def _prompt(prompt_text: str, input_fn: Callable[[str], str]) -> str:
    return input_fn(prompt_text).strip()


def _resolve_decision(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> ApprovalStatus:
    if args.approve:
        return ApprovalStatus.APPROVED
    if args.reject:
        return ApprovalStatus.REJECTED

    while True:
        decision = _prompt("Decision [a=approve / r=reject]: ", input_fn).lower()
        if decision in {"a", "approve"}:
            return ApprovalStatus.APPROVED
        if decision in {"r", "reject"}:
            return ApprovalStatus.REJECTED


def _resolve_operator(
    operator: str | None,
    *,
    input_fn: Callable[[str], str],
) -> str:
    if operator:
        text = operator.strip()
        if text:
            return text

    default_user = getpass.getuser().strip()
    if default_user:
        return default_user

    prompted = _prompt("Operator: ", input_fn)
    if prompted:
        return prompted
    raise ApprovalCliError("Operator is required.")


def _resolve_comment(
    decision: ApprovalStatus,
    comment: str | None,
    *,
    input_fn: Callable[[str], str],
) -> str | None:
    text = (comment or "").strip() or None
    if decision == ApprovalStatus.REJECTED and not text:
        text = _prompt("Rejection reason: ", input_fn) or None
    if decision == ApprovalStatus.REJECTED and not text:
        raise ApprovalCliError("Rejections require a non-empty comment.")
    return text


def _confirm(
    bundle: IntentBundle,
    decision: ApprovalStatus,
    operator: str,
    comment: str | None,
    *,
    input_fn: Callable[[str], str],
    stdout: TextIO,
) -> bool:
    summary = [
        f"Bundle: pending {bundle.status.value}",
        f"Signal hash: {bundle.signal_hash}",
        f"Orders: {len(bundle.proposed_orders)}",
        f"Decision: {decision.value}",
        f"Operator: {operator}",
    ]
    if comment:
        summary.append(f"Comment: {comment}")
    stdout.write("\n".join(summary) + "\n")
    response = _prompt("Confirm update? [y/N]: ", input_fn).lower()
    return response in {"y", "yes"}


def validate_pending_bundle(bundle: IntentBundle, *, now: datetime) -> None:
    if bundle.status != IntentBundleStatus.AWAITING_APPROVAL:
        raise ApprovalCliError(
            f"Bundle status is {bundle.status.value}; only AWAITING_APPROVAL can be updated."
        )
    if now >= bundle.approval_deadline:
        raise ApprovalCliError(
            f"Approval deadline expired at {bundle.approval_deadline.isoformat()}."
        )


def apply_decision(
    bundle: IntentBundle,
    *,
    decision: ApprovalStatus,
    operator: str,
    hostname: str,
    comment: str | None,
    now: datetime,
) -> IntentBundle:
    validate_pending_bundle(bundle, now=now)
    approval_record = ApprovalRecord(
        approved_at=now,
        operator=operator,
        status=decision,
        comment=comment,
        hostname=hostname,
        rebalance_id=bundle.rebalance_id,
    )
    return bundle.with_approval(approval_record)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: datetime | None = None,
    input_fn: Callable[[str], str] = input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        decision = _resolve_decision(args, input_fn=input_fn)
        operator = _resolve_operator(args.operator, input_fn=input_fn)
        comment = _resolve_comment(decision, args.comment, input_fn=input_fn)
        hostname = (args.hostname or socket.gethostname()).strip()
        if not hostname:
            raise ApprovalCliError("Hostname is required.")

        bundle_path = Path(args.bundle_path).resolve()
        bundle = load_bundle(bundle_path)
        decision_time = now or datetime.now(timezone.utc)
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=timezone.utc)
        else:
            decision_time = decision_time.astimezone(timezone.utc)

        if not args.yes and not _confirm(
            bundle,
            decision,
            operator,
            comment,
            input_fn=input_fn,
            stdout=stdout,
        ):
            raise ApprovalCliError("Approval update cancelled by operator.")

        updated_bundle = apply_decision(
            bundle,
            decision=decision,
            operator=operator,
            hostname=hostname,
            comment=comment,
            now=decision_time,
        )
        save_bundle(bundle_path, updated_bundle)
        stdout.write(
            f"{updated_bundle.status.value} bundle written to {bundle_path}\n"
        )
        return 0
    except ApprovalCliError as exc:
        stderr.write(f"error: {exc}\n")
        return 1
    except KeyboardInterrupt:
        stderr.write("error: interrupted by operator\n")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
