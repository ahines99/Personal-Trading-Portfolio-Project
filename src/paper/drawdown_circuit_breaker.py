from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .alerting import send_alert
from .preflight_gate import activate_kill_switch


def check_drawdown(
    blotter: Any,
    *,
    initial_capital: float,
    ytd_limit: float = -0.30,
    daily_limit: float = -0.05,
    trading_day: date | str | None = None,
    kill_switch_path: str | Path | None = None,
    log_path: str | Path | None = None,
    operator: str = "PT-S4-006",
    canary_active: bool = False,
    config: Any | None = None,
    repo_root: str | Path | None = None,
    alert_log_path: str | Path | None = None,
    current_time: datetime | None = None,
) -> tuple[bool, str]:
    """Return (passed, reason) and optionally activate the kill switch on breach."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    effective_ytd_limit = adjusted_limit(ytd_limit, canary_active=canary_active)
    effective_daily_limit = adjusted_limit(daily_limit, canary_active=canary_active)
    daily_pnl, ytd_pnl = compute_realized_pnl(blotter, trading_day=trading_day)

    ytd_ratio = ytd_pnl / initial_capital
    daily_ratio = daily_pnl / initial_capital

    passed = True
    reason = ""
    decision = "pass"
    if ytd_ratio <= effective_ytd_limit:
        passed = False
        decision = "ytd_drawdown_breach"
        reason = f"YTD DD {ytd_ratio * 100:.1f}%"
    elif daily_ratio <= effective_daily_limit:
        passed = False
        decision = "daily_loss_breach"
        reason = f"Daily loss {daily_ratio * 100:.1f}%"

    event = {
        "timestamp": _coerce_now(current_time).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "passed": passed,
        "decision": decision,
        "reason": reason,
        "daily_pnl": daily_pnl,
        "ytd_pnl": ytd_pnl,
        "daily_ratio": daily_ratio,
        "ytd_ratio": ytd_ratio,
        "daily_limit": effective_daily_limit,
        "ytd_limit": effective_ytd_limit,
        "canary_active": canary_active,
        "initial_capital": initial_capital,
        "operator": operator,
        "kill_switch_activated": bool(not passed and kill_switch_path is not None),
    }
    _append_log(log_path, event)

    if not passed:
        _emit_circuit_breaker_alert(
            event,
            config=config,
            repo_root=repo_root,
            alert_log_path=alert_log_path,
            current_time=current_time,
        )

    if not passed and kill_switch_path is not None:
        activate_kill_switch(
            kill_switch_path,
            reason=reason,
            operator=operator,
        )
        _emit_kill_switch_alert(
            event,
            config=config,
            repo_root=repo_root,
            alert_log_path=alert_log_path,
            current_time=current_time,
            kill_switch_path=kill_switch_path,
        )
    return passed, reason


def compute_realized_pnl(
    blotter: Any,
    *,
    trading_day: date | str | None = None,
) -> tuple[float, float]:
    rows = list(_coerce_records(blotter))
    target_day = _coerce_date(trading_day) if trading_day is not None else None

    daily_pnl = 0.0
    ytd_pnl = 0.0
    for row in rows:
        pnl = _extract_realized_pnl(row)
        if pnl is None:
            continue
        row_date = _extract_row_date(row)
        ytd_pnl += pnl
        if target_day is not None and row_date == target_day:
            daily_pnl += pnl

    if target_day is None:
        daily_pnl = ytd_pnl
    return daily_pnl, ytd_pnl


def adjusted_limit(limit_pct: float, *, canary_active: bool) -> float:
    if not canary_active:
        return limit_pct
    return limit_pct * 1.5


def _coerce_records(blotter: Any) -> Iterable[Any]:
    if blotter is None:
        return []
    if isinstance(blotter, list):
        return blotter
    if hasattr(blotter, "list_orders"):
        return blotter.list_orders(latest_only=True)
    if hasattr(blotter, "query_fills"):
        return blotter.query_fills()
    if hasattr(blotter, "__iter__"):
        return blotter
    raise TypeError(f"Unsupported blotter type: {type(blotter)!r}")


def _extract_realized_pnl(row: Any) -> float | None:
    if isinstance(row, dict):
        for key in ("realized_pnl", "pnl", "net_pnl"):
            if key in row and row[key] is not None:
                return float(row[key])
        return None

    for key in ("realized_pnl", "pnl", "net_pnl"):
        value = getattr(row, key, None)
        if value is not None:
            return float(value)
    return None


def _extract_row_date(row: Any) -> date | None:
    if isinstance(row, dict):
        for key in ("trade_date", "date", "timestamp"):
            value = row.get(key)
            if value:
                return _coerce_date(value)
        return None

    for key in ("trade_date", "date", "timestamp"):
        value = getattr(row, key, None)
        if value:
            return _coerce_date(value)
    return None


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _append_log(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _emit_circuit_breaker_alert(
    event: dict[str, Any],
    *,
    config: Any | None,
    repo_root: str | Path | None,
    alert_log_path: str | Path | None,
    current_time: datetime | None,
) -> None:
    details = {
        "source": "drawdown_circuit_breaker",
        **event,
    }
    send_alert(
        "circuit_breaker_triggered",
        str(event.get("reason") or "Drawdown circuit breaker triggered"),
        "critical",
        details=details,
        config=config,
        repo_root=repo_root,
        alert_log_path=alert_log_path,
        current_time=current_time,
    )


def _emit_kill_switch_alert(
    event: dict[str, Any],
    *,
    config: Any | None,
    repo_root: str | Path | None,
    alert_log_path: str | Path | None,
    current_time: datetime | None,
    kill_switch_path: str | Path,
) -> None:
    send_alert(
        "kill_switch_activated",
        f"Kill switch activated by drawdown circuit breaker: {event.get('reason') or 'threshold breach'}",
        "critical",
        details={
            "source": "drawdown_circuit_breaker",
            **event,
            "kill_switch_path": str(kill_switch_path),
        },
        config=config,
        repo_root=repo_root,
        alert_log_path=alert_log_path,
        current_time=current_time,
    )


def _coerce_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["adjusted_limit", "check_drawdown", "compute_realized_pnl"]
