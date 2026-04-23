from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .alerting import send_alert
from .baseline_resolver import resolve_signal_path
from .manifest import build_manifest
from .verify import validate_manifest_config_hash

CRITICAL_ALERT_LEVELS = {"critical", "fatal"}


def preflight_passes(
    context: dict[str, Any],
    *,
    check_now: datetime | None = None,
) -> tuple[bool, list[str]]:
    """Run the Stage 4 composite preflight and return (passed, reasons)."""
    result = run_preflight_checks(context, check_now=check_now)
    return result["passed"], list(result["reasons"])


def run_preflight_checks(
    context: dict[str, Any],
    *,
    check_now: datetime | None = None,
) -> dict[str, Any]:
    now = _coerce_now(check_now)
    config = context.get("config")
    repo_root = Path(context.get("repo_root") or Path(__file__).resolve().parents[2])
    intents_only = bool(context.get("intents_only"))
    allow_intents_bypass = bool(context.get("allow_intents_only_with_kill_switch"))
    results_dir = Path(context.get("results_dir") or repo_root / "paper_trading" / "reports")
    kill_switch_path = Path(
        context.get("kill_switch_path")
        or _read_attr(config, "kill_switch_path")
        or repo_root / "paper_trading" / "state" / "KILL_SWITCH"
    )
    alert_log_path = Path(
        context.get("alert_log_path")
        or _read_attr(config, "alert_log_path")
        or repo_root / "paper_trading" / "monitoring_alerts.jsonl"
    )

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    kill_switch_active = is_kill_switch_active(kill_switch_path)
    if kill_switch_active:
        if intents_only and allow_intents_bypass:
            checks.append(
                {
                    "name": "kill_switch",
                    "passed": True,
                    "reason": "kill switch active but bypassed for intents-only mode",
                }
            )
        else:
            checks.append(
                {
                    "name": "kill_switch",
                    "passed": False,
                    "reason": f"kill switch active at {kill_switch_path}",
                }
            )
            reasons.append(f"kill switch active at {kill_switch_path}")
    else:
        checks.append({"name": "kill_switch", "passed": True, "reason": ""})

    freshness_ok, freshness_reason = _check_signal_freshness(context, now=now)
    checks.append({"name": "signal_freshness", "passed": freshness_ok, "reason": freshness_reason})
    if not freshness_ok:
        reasons.append(freshness_reason)

    manifest_ok, manifest_reason = _check_manifest_hash(context, repo_root=repo_root)
    checks.append({"name": "manifest_hash", "passed": manifest_ok, "reason": manifest_reason})
    if not manifest_ok:
        reasons.append(manifest_reason)

    broker_ok, broker_reason = _check_broker_health(context)
    checks.append({"name": "broker_ping", "passed": broker_ok, "reason": broker_reason})
    if not broker_ok:
        reasons.append(broker_reason)

    recon_ok, recon_reason = _check_prior_reconciliation(context)
    checks.append({"name": "prior_reconciliation", "passed": recon_ok, "reason": recon_reason})
    if not recon_ok:
        reasons.append(recon_reason)

    alerts_ok, alerts_reason = _check_recent_alerts(alert_log_path, now=now)
    checks.append({"name": "recent_alerts", "passed": alerts_ok, "reason": alerts_reason})
    if not alerts_ok:
        reasons.append(alerts_reason)

    passed = not reasons
    log_payload = {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "passed": passed,
        "reasons": reasons,
        "intents_only": intents_only,
        "checks": checks,
    }
    write_preflight_log(results_dir, log_payload, check_date=now.date())
    if not passed:
        _emit_preflight_failure_alert(
            context,
            log_payload=log_payload,
            repo_root=repo_root,
            alert_log_path=alert_log_path,
            current_time=now,
        )
    return log_payload


def is_kill_switch_active(path: str | Path) -> bool:
    return Path(path).exists()


def activate_kill_switch(
    path: str | Path,
    *,
    reason: str,
    operator: str = "manual",
    log_path: str | Path | None = None,
) -> Path:
    kill_path = Path(path)
    kill_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_now_iso()
    kill_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "reason": reason,
                "operator": operator,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _append_kill_switch_log(
        log_path or kill_path.parent / "kill_switch_log.jsonl",
        action="activate",
        reason=reason,
        operator=operator,
    )
    return kill_path


def clear_kill_switch(
    path: str | Path,
    *,
    reason: str,
    operator: str = "manual",
    log_path: str | Path | None = None,
) -> None:
    kill_path = Path(path)
    if kill_path.exists():
        kill_path.unlink()
    _append_kill_switch_log(
        log_path or kill_path.parent / "kill_switch_log.jsonl",
        action="clear",
        reason=reason,
        operator=operator,
    )


def write_preflight_log(
    results_dir: str | Path,
    payload: dict[str, Any],
    *,
    check_date: datetime.date,
) -> Path:
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"preflight_log_{check_date.isoformat()}.json"
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target_path


def _check_signal_freshness(
    context: dict[str, Any],
    *,
    now: datetime,
) -> tuple[bool, str]:
    config = context.get("config")
    signal_path = context.get("signal_path")
    if signal_path is None:
        signal_path = resolve_signal_path(
            config,
            repo_root=context.get("repo_root"),
        )
    target = Path(signal_path)
    if not target.exists():
        return False, f"signal file missing: {target}"

    max_age_hours = float(context.get("signal_max_age_hours") or 24.0)
    age = now - datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc)
    if age > timedelta(hours=max_age_hours):
        return False, f"signal file stale: {target} age={age}"
    return True, ""


def _check_manifest_hash(
    context: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[bool, str]:
    config = context.get("config")
    manifest = context.get("manifest")
    if manifest is None:
        manifest_path = context.get("manifest_path")
        if manifest_path:
            path = Path(manifest_path)
            if not path.exists():
                return False, f"manifest missing: {path}"
            manifest = json.loads(path.read_text(encoding="utf-8"))
        elif config is not None and context.get("config_path") and context.get("results_dir"):
            manifest = build_manifest(
                config,
                config_path=context["config_path"],
                results_dir=context["results_dir"],
                repo_root=repo_root,
            )
        else:
            return False, "manifest not supplied"

    try:
        validate_manifest_config_hash(config, manifest)
    except Exception as exc:
        return False, f"manifest config hash mismatch: {exc}"
    return True, ""


def _check_broker_health(context: dict[str, Any]) -> tuple[bool, str]:
    if "broker_ping" in context:
        return (bool(context["broker_ping"]), "" if context["broker_ping"] else "broker ping failed")

    broker_client = context.get("broker_client")
    if broker_client is None:
        return False, "broker client missing"

    try:
        broker_ok = bool(broker_client.ping())
    except Exception as exc:
        return False, f"broker ping failed: {exc}"
    if not broker_ok:
        return False, "broker ping failed"
    return True, ""


def _check_prior_reconciliation(context: dict[str, Any]) -> tuple[bool, str]:
    tolerance = float(context.get("reconciliation_tolerance") or 0.0)
    record = context.get("prior_reconciliation")
    if record is None:
        summary_path = context.get("reconciliation_summary_path")
        if summary_path:
            record = _load_latest_jsonl_record(summary_path)

    if record is None:
        return True, ""

    status = str(record.get("status") or "").strip().upper()
    unresolved = float(
        record.get("unresolved_diff", record.get("weight_drift_l1", 0.0)) or 0.0
    )
    if status == "ALERT_REQUIRED":
        return False, "prior-day reconciliation status is ALERT_REQUIRED"
    if unresolved > tolerance:
        return False, f"prior-day reconciliation unresolved_diff {unresolved:.6f} > {tolerance:.6f}"
    return True, ""


def _check_recent_alerts(
    alert_log_path: str | Path,
    *,
    now: datetime,
) -> tuple[bool, str]:
    path = Path(alert_log_path)
    if not path.exists():
        return True, ""

    cutoff = now - timedelta(hours=6)
    for record in _load_jsonl(path):
        severity = str(record.get("severity") or "").strip().lower()
        if severity not in CRITICAL_ALERT_LEVELS:
            continue
        timestamp = record.get("timestamp")
        if timestamp is None:
            continue
        event_time = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if event_time >= cutoff:
            alert_type = str(record.get("alert_type") or record.get("type") or "unknown")
            return False, f"recent critical alert: {alert_type}"
    return True, ""


def _emit_preflight_failure_alert(
    context: dict[str, Any],
    *,
    log_payload: dict[str, Any],
    repo_root: Path,
    alert_log_path: Path,
    current_time: datetime,
) -> None:
    reasons = [str(reason) for reason in log_payload.get("reasons", []) if str(reason).strip()]
    failing_checks = [
        str(check.get("name"))
        for check in log_payload.get("checks", [])
        if not bool(check.get("passed"))
    ]
    if reasons:
        message = f"Composite preflight gate failed: {reasons[0]}"
    else:
        message = "Composite preflight gate failed"
    send_alert(
        "preflight_failure",
        message,
        "error",
        details={
            "source": "preflight_gate",
            "passed": False,
            "reasons": reasons,
            "failing_checks": failing_checks,
            "intents_only": bool(log_payload.get("intents_only")),
            "checks": log_payload.get("checks", []),
        },
        config=context.get("config"),
        repo_root=repo_root,
        alert_log_path=alert_log_path,
        current_time=current_time,
    )


def _append_kill_switch_log(
    path: str | Path,
    *,
    action: str,
    reason: str,
    operator: str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": _utc_now_iso(),
        "action": action,
        "reason": reason,
        "operator": operator,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


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


def _load_latest_jsonl_record(path: str | Path) -> dict[str, Any] | None:
    records = _load_jsonl(path)
    return records[-1] if records else None


def _coerce_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _read_attr(config: Any, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "activate_kill_switch",
    "clear_kill_switch",
    "is_kill_switch_active",
    "preflight_passes",
    "run_preflight_checks",
    "write_preflight_log",
]
