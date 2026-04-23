from __future__ import annotations

import json
import os
import smtplib
import subprocess
from datetime import date, datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def send_alert(
    alert_type: str,
    message: str,
    severity: str,
    *,
    details: dict[str, Any] | None = None,
    config: Any | None = None,
    repo_root: str | Path | None = None,
    alert_log_path: str | Path | None = None,
    recipients: list[str] | None = None,
    smtp_transport: Any | None = None,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    timestamp = _coerce_now(current_time)
    normalized_type = str(alert_type).strip()
    normalized_severity = str(severity).strip().lower()
    mute_types = set(_list_value(config, "alert_mute_types"))
    payload = {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "alert_type": normalized_type,
        "message": str(message).strip(),
        "severity": normalized_severity,
        "muted": normalized_type in mute_types,
        "channels_attempted": [],
    }
    if details:
        payload["details"] = _json_safe(details)

    if payload["muted"]:
        _append_alert_log(_alert_log_path(config, repo_root=repo_root, alert_log_path=alert_log_path), payload)
        return payload

    target_recipients = recipients or list(_list_value(config, "alert_recipients"))
    if bool(_read_attr(config, "desktop_alerts_enabled")):
        payload["channels_attempted"].append("desktop")
        payload["desktop_sent"] = _send_desktop_alert(
            normalized_type,
            payload["message"],
            normalized_severity,
        )

    if bool(_read_attr(config, "email_alerts_enabled")) and target_recipients:
        payload["channels_attempted"].append("email")
        payload["email_sent"] = _send_email_alert(
            normalized_type,
            payload["message"],
            normalized_severity,
            recipients=target_recipients,
            details=payload.get("details"),
            smtp_profile=_read_attr(config, "alert_smtp_profile"),
            smtp_transport=smtp_transport,
        )

    _append_alert_log(_alert_log_path(config, repo_root=repo_root, alert_log_path=alert_log_path), payload)
    return payload


def build_daily_digest(
    digest_date: date | str,
    *,
    alert_log_path: str | Path,
) -> dict[str, Any]:
    target_date = _coerce_date(digest_date)
    alerts = [
        record
        for record in _load_jsonl(alert_log_path)
        if _coerce_date(record["timestamp"]) == target_date
    ]
    by_severity: dict[str, int] = {}
    for record in alerts:
        severity = str(record.get("severity") or "unknown")
        by_severity[severity] = by_severity.get(severity, 0) + 1

    return {
        "date": target_date.isoformat(),
        "count": len(alerts),
        "by_severity": by_severity,
        "alerts": alerts,
    }


def _send_email_alert(
    alert_type: str,
    message: str,
    severity: str,
    *,
    recipients: list[str],
    details: Any | None,
    smtp_profile: str | None,
    smtp_transport: Any | None,
) -> bool:
    server = os.environ.get("PAPER_TRADING_SMTP_SERVER")
    username = os.environ.get("PAPER_TRADING_SMTP_USERNAME")
    password = os.environ.get("PAPER_TRADING_SMTP_PASSWORD")
    port = int(os.environ.get("PAPER_TRADING_SMTP_PORT", "587"))
    sender = os.environ.get("PAPER_TRADING_SMTP_SENDER", username or "")
    if not server or not sender:
        return False

    msg = EmailMessage()
    msg["Subject"] = f"[paper-trading] {severity.upper()} {alert_type}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    if smtp_profile:
        msg["X-SMTP-Profile"] = smtp_profile
    msg.set_content(_email_body(message, details))

    if smtp_transport is not None:
        smtp_transport(server, port, username, password, msg)
        return True

    with smtplib.SMTP(server, port, timeout=5) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)
    return True


def _send_desktop_alert(alert_type: str, message: str, severity: str) -> bool:
    if os.name != "nt":
        return False
    title = f"paper-trading {severity.upper()} {alert_type}"
    script = (
        "$wshell = New-Object -ComObject WScript.Shell; "
        f"$wshell.Popup('{_ps_escape(message)}', 5, '{_ps_escape(title)}', 0x0)"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return False
    return True


def _append_alert_log(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _alert_log_path(
    config: Any | None,
    *,
    repo_root: str | Path | None,
    alert_log_path: str | Path | None = None,
) -> Path:
    if alert_log_path:
        return Path(alert_log_path)
    explicit = _read_attr(config, "alert_log_path")
    if explicit:
        return Path(explicit)
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    return root / "paper_trading" / "monitoring_alerts.jsonl"


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


def _read_attr(config: Any | None, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _list_value(config: Any | None, field_name: str) -> list[str]:
    value = _read_attr(config, field_name)
    if not value:
        return []
    return [str(item) for item in value]


def _ps_escape(value: str) -> str:
    return str(value).replace("'", "''")


def _email_body(message: str, details: Any | None) -> str:
    if details is None:
        return message
    return f"{message}\n\nDetails:\n{json.dumps(details, indent=2, sort_keys=True)}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _coerce_now(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _coerce_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


__all__ = ["build_daily_digest", "send_alert"]
