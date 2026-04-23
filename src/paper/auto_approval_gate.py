from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .verify import compute_config_hash


def auto_approve(
    context: dict[str, Any],
    previews: Iterable[dict[str, Any]],
    preflight_result: tuple[bool, list[str]] | dict[str, Any],
) -> tuple[bool, str]:
    """Return an automated approval decision and persist the approval record."""
    approved, preflight_reasons = _normalize_preflight_result(preflight_result)
    preview_list = [dict(item) for item in previews]

    preview_statuses = [_preview_is_ok(item) for item in preview_list]
    all_previews_ok = all(preview_statuses) and bool(preview_list)
    is_rebalance_day = bool(context.get("is_rebalance_day") or context.get("override_rebalance_gate"))

    reasons: list[str] = []
    status = "APPROVED"
    if not approved:
        reasons.append("preflight failed")
        reasons.extend(preflight_reasons)
        status = "REJECTED"
    if not all_previews_ok:
        reasons.append("preview check failed")
        status = "REJECTED"
    if not is_rebalance_day:
        reasons.append("not a rebalance day")
        status = "REJECTED"

    if status == "APPROVED":
        reason = "Auto-approved: preflight ok, previews ok, rebalance day"
    else:
        reason = "Auto-rejected: " + "; ".join(reasons)

    record = build_approval_record(
        context,
        preview_list,
        status=status,
        reason=reason,
    )
    append_approval_record(record, approval_log_path(context))
    return status == "APPROVED", reason


def build_approval_record(
    context: dict[str, Any],
    previews: list[dict[str, Any]],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    config = context.get("config")
    config_hash = context.get("config_hash") or compute_config_hash(config)
    signal_hash = str(context.get("signal_hash") or "unknown")
    preview_hashes = [_preview_hash(item) for item in previews]
    payload = {
        "timestamp": _utc_now_iso(),
        "status": status,
        "reason": reason,
        "approval_type": "auto",
        "signal_hash": signal_hash,
        "config_hash": config_hash,
        "preview_hashes": preview_hashes,
        "preview_count": len(previews),
        "rebalance_date": str(context.get("as_of_date") or ""),
    }
    payload["record_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def append_approval_record(record: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_hashes = {
        item.get("record_hash")
        for item in _load_jsonl(target)
        if isinstance(item, dict)
    }
    if record.get("record_hash") in existing_hashes:
        return target

    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")
    return target


def approval_log_path(context: dict[str, Any]) -> Path:
    explicit = context.get("approval_log_path")
    if explicit:
        return Path(explicit)

    repo_root = Path(context.get("repo_root") or Path(__file__).resolve().parents[2])
    as_of_date = str(context.get("as_of_date") or datetime.now(timezone.utc).date().isoformat())
    return repo_root / "paper_trading" / "reports" / as_of_date / "approvals.jsonl"


def _normalize_preflight_result(
    result: tuple[bool, list[str]] | dict[str, Any]
) -> tuple[bool, list[str]]:
    if isinstance(result, dict):
        return bool(result.get("passed")), list(result.get("reasons") or [])
    passed, reasons = result
    return bool(passed), list(reasons)


def _preview_is_ok(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or payload.get("broker_status") or "").strip().lower()
    if status in {"ok", "approved", "pass"}:
        return True
    errors = payload.get("errors") or []
    if status in {"warn", "warning", "fail", "rejected"}:
        return False
    if payload.get("result") is not None:
        return bool(payload.get("result"))
    return not bool(errors)


def _preview_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["append_approval_record", "approval_log_path", "auto_approve", "build_approval_record"]
