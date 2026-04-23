from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .schemas.reconciliation import ReconciliationReport, SlippageSummary
from .writers import atomic_update_text, atomic_write_text


def generate_reconciliation_report(
    as_of_date: date | str,
    context: dict[str, Any],
) -> str:
    report_date = _coerce_date(as_of_date)
    report_dir = Path(
        context.get("report_dir")
        or Path(context.get("repo_root") or Path(__file__).resolve().parents[2])
        / "paper_trading"
        / "reports"
        / report_date.isoformat()
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    target_weights = _normalize_weights(context.get("target_weights"))
    balances = dict(context.get("balances") or context.get("broker_balances") or {})
    broker_positions = context.get("broker_positions")
    expected_nav = _optional_float(
        context.get("expected_nav", context.get("portfolio_notional_usd", context.get("notional")))
    )
    expected_cash = _optional_float(context.get("expected_cash"))
    actual_weights = _normalize_weights(
        context.get("actual_weights")
        or _weights_from_positions(
            broker_positions,
            nav=_actual_nav(balances, broker_positions),
        )
    )
    intended_trades = list(context.get("intended_trades") or [])
    executed_trades = list(context.get("executed_trades") or context.get("fills") or [])
    unfilled_orders = list(context.get("unfilled_orders") or [])
    cancellations = list(context.get("cancellations") or [])
    fills = list(context.get("fills") or [])
    expected_prices = dict(context.get("expected_prices") or {})

    slippage_values = _slippage_bps(fills, expected_prices=expected_prices)
    weight_comparisons = ReconciliationReport.build_weight_comparisons(
        target_weights,
        actual_weights,
    )
    weight_drift_l1 = sum(abs(item.drift) for item in weight_comparisons)
    thresholds = _drift_thresholds(context)
    cash_drift = _cash_drift(
        context=context,
        target_weights=target_weights,
        balances=balances,
        broker_positions=broker_positions,
        expected_cash=expected_cash,
        expected_nav=expected_nav,
    )
    nav_drift = _nav_drift(
        context=context,
        balances=balances,
        broker_positions=broker_positions,
        expected_nav=expected_nav,
    )
    anomalies = _detect_anomalies(
        unfilled_orders=unfilled_orders,
        cancellations=cancellations,
        fills=fills,
    )
    anomalies.extend(_normalize_anomalies(context.get("anomalies")))
    anomalies.extend(
        _drift_anomalies(
            weight_drift_l1=weight_drift_l1,
            cash_drift=cash_drift,
            nav_drift=nav_drift,
            thresholds=thresholds,
        )
    )
    status = _reconciliation_status(
        weight_drift_l1=weight_drift_l1,
        cash_drift=cash_drift,
        nav_drift=nav_drift,
        anomalies=anomalies,
        thresholds=thresholds,
    )
    summary_path = Path(
        context.get("summary_path") or report_dir.parent / "reconciliation_summary.jsonl"
    )
    existing_summary_rows = _load_summary_rows(summary_path)
    tracking_summary = _build_tracking_summary(
        as_of_date=report_date,
        tracking_payload=context.get("tracking_vs_backtest"),
        existing_rows=existing_summary_rows,
        settings=_tracking_settings(context),
    )

    report = ReconciliationReport(
        as_of_date=report_date,
        intended_trades=intended_trades,
        executed_trades=executed_trades,
        unfilled_orders=unfilled_orders,
        cancellations=cancellations,
        slippage_summary=SlippageSummary.from_bps(slippage_values),
        target_weights=target_weights,
        actual_weights=actual_weights,
        weight_comparisons=weight_comparisons,
        weight_drift_l1=weight_drift_l1,
        cash_drift=cash_drift,
        nav_drift=nav_drift,
        pnl_reconciled_vs_backtest=tracking_summary.get("pnl_reconciled_vs_backtest"),
        anomalies=anomalies,
    )

    report_path = report_dir / "reconciliation.md"
    atomic_write_text(
        report_path,
        _render_markdown(
            report,
            status=status,
            thresholds=thresholds,
            tracking_summary=tracking_summary,
        ),
    )
    summary_row = _build_summary_row(
        report,
        status=status,
        thresholds=thresholds,
        tracking_summary=tracking_summary,
    )
    _write_summary_row(summary_row, summary_path=summary_path)
    return str(report_path)


def build_reconciliation_drift_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic drift metrics for JSON execution summaries."""
    target_weights = _normalize_weights(context.get("target_weights"))
    balances = dict(context.get("balances") or context.get("broker_balances") or {})
    broker_positions = context.get("broker_positions")
    expected_nav = _optional_float(
        context.get("expected_nav", context.get("portfolio_notional_usd", context.get("notional")))
    )
    expected_cash = _optional_float(context.get("expected_cash"))
    actual_weights = _normalize_weights(
        context.get("actual_weights")
        or _weights_from_positions(
            broker_positions,
            nav=_actual_nav(balances, broker_positions),
        )
    )
    weight_comparisons = ReconciliationReport.build_weight_comparisons(
        target_weights,
        actual_weights,
    )
    weight_drift_l1 = sum(abs(item.drift) for item in weight_comparisons)
    thresholds = _drift_thresholds(context)
    cash_drift = _cash_drift(
        context=context,
        target_weights=target_weights,
        balances=balances,
        broker_positions=broker_positions,
        expected_cash=expected_cash,
        expected_nav=expected_nav,
    )
    nav_drift = _nav_drift(
        context=context,
        balances=balances,
        broker_positions=broker_positions,
        expected_nav=expected_nav,
    )
    anomalies = _drift_anomalies(
        weight_drift_l1=weight_drift_l1,
        cash_drift=cash_drift,
        nav_drift=nav_drift,
        thresholds=thresholds,
    )
    status = _reconciliation_status(
        weight_drift_l1=weight_drift_l1,
        cash_drift=cash_drift,
        nav_drift=nav_drift,
        anomalies=anomalies,
        thresholds=thresholds,
    )
    largest_drifts = sorted(
        [
            {
                "ticker": item.ticker,
                "target": item.target,
                "actual": item.actual,
                "drift": item.drift,
                "drift_bps": item.drift_bps,
            }
            for item in weight_comparisons
        ],
        key=lambda row: abs(float(row["drift"])),
        reverse=True,
    )
    return {
        "status": status,
        "target_weights": target_weights,
        "actual_weights": actual_weights,
        "weight_drift_l1": weight_drift_l1,
        "weight_drift_l1_pct": weight_drift_l1 * 100.0,
        "cash_drift": cash_drift,
        "cash_drift_abs": abs(cash_drift),
        "nav_drift": nav_drift,
        "nav_drift_abs": abs(nav_drift),
        "thresholds": thresholds,
        "largest_weight_drifts": largest_drifts[:10],
        "anomalies": anomalies,
    }


def _render_markdown(
    report: ReconciliationReport,
    *,
    status: str,
    thresholds: dict[str, float],
    tracking_summary: dict[str, Any],
) -> str:
    lines = [
        f"# Reconciliation Report - {report.as_of_date.isoformat()}",
        "",
        f"Status: {status}",
        "",
        "## Summary",
        "",
        f"- Intended trades: {len(report.intended_trades)}",
        f"- Executed trades: {len(report.executed_trades)}",
        f"- Unfilled orders: {len(report.unfilled_orders)}",
        f"- Cancellations: {len(report.cancellations)}",
        f"- Weight drift L1: {report.weight_drift_l1:.4f}",
        f"- Weight drift alert threshold: {thresholds['weight_drift_alert']:.4f}",
        f"- Cash drift: {report.cash_drift:+.2f}",
        f"- Cash drift threshold: {thresholds['cash_drift_abs']:.2f}",
        f"- NAV drift: {report.nav_drift:+.2f}",
        f"- NAV drift threshold: {thresholds['nav_drift_abs']:.2f}",
        f"- Slippage mean bps: {report.slippage_summary.mean_bps:.2f}",
        "",
        "## Tracking vs Backtest",
        "",
        f"- Tracking status: {tracking_summary['tracking_status']}",
        f"- Return date: {tracking_summary['return_date'] or 'N/A'}",
        f"- Return date source: {tracking_summary['return_date_source'] or 'N/A'}",
        f"- Backtest-predicted return: {tracking_summary['backtest_predicted_return']:+.6f}",
        f"- Paper realized return: {tracking_summary['paper_realized_return']:+.6f}",
        f"- Excess return vs backtest: {tracking_summary['return_excess_vs_backtest']:+.6f}",
        f"- Rolling tracking error ({tracking_summary['tracking_error_window_days']}d ann.): "
        f"{tracking_summary['tracking_error_rolling_ann_pct']:.3f}%",
        f"- Tracking error threshold: {tracking_summary['tracking_error_alert_threshold_pct']:.3f}%",
        f"- Monthly win rate vs backtest: {tracking_summary['monthly_win_rate_vs_backtest_pct']:.2f}%",
        f"- Monthly win-rate threshold: {tracking_summary['monthly_win_rate_min_threshold_pct']:.2f}%",
        "",
        "## Intended vs Executed",
        "",
        _ascii_table(
            ["Ticker", "Target", "Actual", "Diff"],
            [
                [
                    item.ticker,
                    f"{item.target:.4f}",
                    f"{item.actual:.4f}",
                    f"{item.drift:+.4f}",
                ]
                for item in report.weight_comparisons
            ],
        ),
        "",
        "## Anomalies",
        "",
    ]
    if report.anomalies:
        for anomaly in report.anomalies:
            lines.append(f"- {anomaly}")
    else:
        lines.append("- None")
    if tracking_summary.get("tracking_reason"):
        lines.append(f"- tracking_vs_backtest: {tracking_summary['tracking_reason']}")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            _next_steps(report, status=status),
            "",
        ]
    )
    return "\n".join(lines)


def _build_summary_row(
    report: ReconciliationReport,
    *,
    status: str,
    thresholds: dict[str, float],
    tracking_summary: dict[str, Any],
) -> dict[str, Any]:
    summary_row = report.summary_row()
    summary_row["status"] = status
    summary_row.update(
        {
            "weight_drift_minor_threshold": thresholds["weight_drift_minor"],
            "weight_drift_alert_threshold": thresholds["weight_drift_alert"],
            "cash_drift_abs_threshold": thresholds["cash_drift_abs"],
            "nav_drift_abs_threshold": thresholds["nav_drift_abs"],
        }
    )
    summary_row.update(tracking_summary)
    return summary_row


def _write_summary_row(summary_row: dict[str, Any], *, summary_path: str | Path) -> None:
    path = Path(summary_path)

    def merge_summary(existing_text: str) -> str:
        existing = [json.loads(line) for line in existing_text.splitlines() if line.strip()]
        filtered = [
            row for row in existing if str(row.get("as_of_date")) != str(summary_row.get("as_of_date"))
        ]
        filtered.append(summary_row)
        filtered.sort(key=lambda row: str(row.get("as_of_date") or ""))
        return "".join(json.dumps(row, sort_keys=True) + "\n" for row in filtered)

    atomic_update_text(path, merge_summary)


def _load_summary_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tracking_settings(context: dict[str, Any]) -> dict[str, float | int]:
    return {
        "tracking_error_window_days": int(context.get("tracking_error_window_days", 21)),
        "tracking_error_annualization_days": int(context.get("tracking_error_annualization_days", 252)),
        "tracking_error_alert_threshold": _float_setting(context, "tracking_error_alert_threshold", 0.0075),
        "monthly_win_rate_min_threshold": _float_setting(context, "monthly_win_rate_min_threshold", 0.55),
    }


def _build_tracking_summary(
    *,
    as_of_date: date,
    tracking_payload: Any,
    existing_rows: list[dict[str, Any]],
    settings: dict[str, float | int],
) -> dict[str, Any]:
    window_days = int(settings["tracking_error_window_days"])
    annualization_days = int(settings["tracking_error_annualization_days"])
    tracking_threshold = float(settings["tracking_error_alert_threshold"])
    monthly_win_rate_threshold = float(settings["monthly_win_rate_min_threshold"])
    summary: dict[str, Any] = {
        "tracking_status": "UNAVAILABLE",
        "tracking_reason": None,
        "return_date": None,
        "return_date_source": None,
        "backtest_predicted_return": 0.0,
        "paper_realized_return": 0.0,
        "return_excess_vs_backtest": 0.0,
        "backtest_predicted_pnl": 0.0,
        "paper_realized_pnl": 0.0,
        "pnl_reconciled_vs_backtest": None,
        "tracking_capital_base": None,
        "tracking_error_window_days": window_days,
        "tracking_error_annualization_days": annualization_days,
        "tracking_error_observations": 0,
        "tracking_error_rolling_ann": 0.0,
        "tracking_error_rolling_ann_pct": 0.0,
        "tracking_error_alert_threshold": tracking_threshold,
        "tracking_error_alert_threshold_pct": tracking_threshold * 100.0,
        "tracking_error_gate_status": "UNAVAILABLE",
        "monthly_win_rate_observations": 0,
        "monthly_positive_excess_days": 0,
        "monthly_win_rate_vs_backtest": 0.0,
        "monthly_win_rate_vs_backtest_pct": 0.0,
        "monthly_win_rate_min_threshold": monthly_win_rate_threshold,
        "monthly_win_rate_min_threshold_pct": monthly_win_rate_threshold * 100.0,
        "monthly_win_rate_gate_status": "UNAVAILABLE",
    }
    payload = dict(tracking_payload or {})
    if not payload.get("available"):
        summary["tracking_reason"] = str(payload.get("reason") or "tracking_vs_backtest unavailable")
        return summary

    summary.update(
        {
            "tracking_status": str(payload.get("status") or "OK"),
            "tracking_reason": payload.get("reason"),
            "return_date": payload.get("return_date"),
            "return_date_source": payload.get("return_date_source"),
            "backtest_predicted_return": float(payload.get("backtest_predicted_return") or 0.0),
            "paper_realized_return": float(payload.get("paper_realized_return") or 0.0),
            "return_excess_vs_backtest": float(payload.get("return_excess_vs_backtest") or 0.0),
            "backtest_predicted_pnl": _optional_float(payload.get("backtest_predicted_pnl")),
            "paper_realized_pnl": _optional_float(payload.get("paper_realized_pnl")),
            "pnl_reconciled_vs_backtest": _optional_float(payload.get("pnl_reconciled_vs_backtest")),
            "tracking_capital_base": _optional_float(payload.get("tracking_capital_base")),
        }
    )

    current_date = as_of_date.isoformat()
    dated_rows = _rows_with_excess_returns(existing_rows)
    dated_rows = [
        row for row in dated_rows if str(row.get("as_of_date")) != current_date
    ]
    dated_rows.append(
        {
            "as_of_date": current_date,
            "return_excess_vs_backtest": summary["return_excess_vs_backtest"],
        }
    )
    dated_rows.sort(key=lambda row: str(row["as_of_date"]))

    rolling_rows = dated_rows[-window_days:]
    rolling_excess = [float(row["return_excess_vs_backtest"]) for row in rolling_rows]
    summary["tracking_error_observations"] = len(rolling_excess)
    tracking_error_ann = _annualized_tracking_error(
        rolling_excess,
        annualization_days=annualization_days,
    )
    summary["tracking_error_rolling_ann"] = tracking_error_ann
    summary["tracking_error_rolling_ann_pct"] = tracking_error_ann * 100.0
    summary["tracking_error_gate_status"] = (
        "ALERT" if tracking_error_ann > tracking_threshold else "OK"
    )

    month_prefix = current_date[:7]
    month_rows = [row for row in dated_rows if str(row["as_of_date"]).startswith(month_prefix)]
    monthly_excess = [float(row["return_excess_vs_backtest"]) for row in month_rows]
    positive_days = sum(1 for value in monthly_excess if value > 0.0)
    win_rate = (positive_days / len(monthly_excess)) if monthly_excess else 0.0
    summary["monthly_win_rate_observations"] = len(monthly_excess)
    summary["monthly_positive_excess_days"] = positive_days
    summary["monthly_win_rate_vs_backtest"] = win_rate
    summary["monthly_win_rate_vs_backtest_pct"] = win_rate * 100.0
    summary["monthly_win_rate_gate_status"] = (
        "ALERT" if monthly_excess and win_rate < monthly_win_rate_threshold else "OK"
    )
    return summary


def _rows_with_excess_returns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        as_of_date = str(row.get("as_of_date") or "").strip()
        excess = _optional_float(row.get("return_excess_vs_backtest"))
        if not as_of_date or excess is None:
            continue
        filtered.append(
            {
                "as_of_date": as_of_date,
                "return_excess_vs_backtest": excess,
            }
        )
    return filtered


def _annualized_tracking_error(
    values: list[float],
    *,
    annualization_days: int,
) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return (variance**0.5) * (float(annualization_days) ** 0.5)


def _normalize_weights(payload: Any) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return {str(key).strip().upper(): float(value) for key, value in payload.items()}
    if hasattr(payload, "to_dict"):
        return {str(key).strip().upper(): float(value) for key, value in payload.to_dict().items()}
    raise TypeError(f"Unsupported weights payload: {type(payload)!r}")


def _weights_from_positions(positions: Any, *, nav: float | None = None) -> dict[str, float]:
    if not positions:
        return {}
    if isinstance(positions, dict):
        iterable = positions.values()
    else:
        iterable = positions
    weights: dict[str, float] = {}
    for item in iterable:
        if isinstance(item, dict):
            ticker = str(item.get("ticker") or "").strip().upper()
            market_value = _optional_float(item.get("market_value"))
            weight = market_value / nav if market_value is not None and nav and abs(nav) > 1e-9 else item.get("current_weight")
        else:
            ticker = str(getattr(item, "ticker", "")).strip().upper()
            market_value = _optional_float(getattr(item, "market_value", None))
            weight = (
                market_value / nav
                if market_value is not None and nav and abs(nav) > 1e-9
                else getattr(item, "current_weight", None)
            )
        if ticker and weight is not None:
            weights[ticker] = float(weight)
    return weights


def _actual_cash(balances: dict[str, Any]) -> float | None:
    return _optional_float(balances.get("cash", balances.get("total_cash")))


def _actual_nav(balances: dict[str, Any], positions: Any) -> float | None:
    nav = _optional_float(balances.get("nav", balances.get("equity", balances.get("total_equity"))))
    if nav is not None:
        return nav
    cash = _actual_cash(balances) or 0.0
    market_value_total = _market_value_total(positions)
    if market_value_total is None:
        return None
    return cash + market_value_total


def _market_value_total(positions: Any) -> float | None:
    if not positions:
        return 0.0
    iterable = positions.values() if isinstance(positions, dict) else positions
    total = 0.0
    found = False
    for item in iterable:
        value = (
            _optional_float(item.get("market_value"))
            if isinstance(item, dict)
            else _optional_float(getattr(item, "market_value", None))
        )
        if value is not None:
            found = True
            total += value
    return total if found else None


def _cash_drift(
    *,
    context: dict[str, Any],
    target_weights: dict[str, float],
    balances: dict[str, Any],
    broker_positions: Any,
    expected_cash: float | None,
    expected_nav: float | None,
) -> float:
    legacy = _optional_float(context.get("cash_drift"))
    actual_cash = _actual_cash(balances)
    if actual_cash is None:
        return legacy or 0.0
    cash_reference = expected_cash
    if cash_reference is None and target_weights:
        nav_reference = expected_nav if expected_nav is not None else _actual_nav(balances, broker_positions)
        if nav_reference is not None:
            cash_reference = (1.0 - sum(target_weights.values())) * nav_reference
    if cash_reference is None:
        return legacy or 0.0
    return actual_cash - cash_reference


def _nav_drift(
    *,
    context: dict[str, Any],
    balances: dict[str, Any],
    broker_positions: Any,
    expected_nav: float | None,
) -> float:
    legacy = _optional_float(context.get("nav_drift"))
    actual_nav = _actual_nav(balances, broker_positions)
    if actual_nav is None:
        return legacy or 0.0
    if expected_nav is not None:
        return actual_nav - expected_nav
    cash = _actual_cash(balances)
    market_value_total = _market_value_total(broker_positions)
    if cash is not None and market_value_total is not None:
        return actual_nav - (cash + market_value_total)
    return legacy or 0.0


def _drift_thresholds(context: dict[str, Any]) -> dict[str, float]:
    return {
        "weight_drift_minor": _float_setting(context, "weight_drift_minor_threshold", 0.02),
        "weight_drift_alert": _float_setting(context, "weight_drift_alert_threshold", 0.05),
        "cash_drift_abs": _float_setting(context, "cash_drift_abs_threshold", 1.0),
        "nav_drift_abs": _float_setting(context, "nav_drift_abs_threshold", 1.0),
    }


def _drift_anomalies(
    *,
    weight_drift_l1: float,
    cash_drift: float,
    nav_drift: float,
    thresholds: dict[str, float],
) -> list[str]:
    anomalies: list[str] = []
    if weight_drift_l1 > thresholds["weight_drift_alert"]:
        anomalies.append(
            "weight drift "
            f"{weight_drift_l1:.6f} exceeds {thresholds['weight_drift_alert']:.6f}"
        )
    if abs(cash_drift) > thresholds["cash_drift_abs"]:
        anomalies.append(
            "cash drift "
            f"{cash_drift:+.2f} exceeds +/-{thresholds['cash_drift_abs']:.2f}"
        )
    if abs(nav_drift) > thresholds["nav_drift_abs"]:
        anomalies.append(
            "NAV drift "
            f"{nav_drift:+.2f} exceeds +/-{thresholds['nav_drift_abs']:.2f}"
        )
    return anomalies


def _slippage_bps(
    fills: Iterable[dict[str, Any]],
    *,
    expected_prices: dict[str, float],
) -> list[float]:
    values: list[float] = []
    for fill in fills:
        expected = fill.get("expected_price")
        symbol = str(fill.get("symbol") or fill.get("ticker") or "").strip().upper()
        if expected is None and symbol in expected_prices:
            expected = expected_prices[symbol]
        if expected in (None, 0):
            continue
        qty = float(fill.get("qty") or fill.get("quantity") or 0.0)
        price = float(fill.get("price") or fill.get("avg_fill_price") or 0.0)
        if qty <= 0 or price <= 0:
            continue
        side = str(fill.get("side") or "buy").strip().lower()
        signed = (price - float(expected)) * qty
        if side == "sell":
            signed *= -1.0
        values.append((signed / (float(expected) * qty)) * 10000.0)
    return values


def _detect_anomalies(
    *,
    unfilled_orders: list[dict[str, Any]],
    cancellations: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> list[str]:
    anomalies: list[str] = []
    if unfilled_orders:
        anomalies.append(f"{len(unfilled_orders)} unfilled orders")
    if cancellations:
        anomalies.append(f"{len(cancellations)} cancellations")
    for fill in fills:
        submitted = fill.get("submission_timestamp")
        timestamp = fill.get("timestamp")
        if submitted and timestamp:
            start = datetime.fromisoformat(str(submitted).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if (end - start).total_seconds() > 1800:
                anomalies.append(f"slow fill {fill.get('symbol') or fill.get('ticker')}")
    return anomalies


def _normalize_anomalies(payload: Any) -> list[str]:
    if not payload:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []
    values: list[str] = []
    for item in payload:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _reconciliation_status(
    *,
    weight_drift_l1: float,
    cash_drift: float,
    nav_drift: float,
    anomalies: list[str],
    thresholds: dict[str, float],
) -> str:
    if (
        anomalies
        or weight_drift_l1 > thresholds["weight_drift_alert"]
        or abs(cash_drift) > thresholds["cash_drift_abs"]
        or abs(nav_drift) > thresholds["nav_drift_abs"]
    ):
        return "ALERT_REQUIRED"
    if weight_drift_l1 > thresholds["weight_drift_minor"]:
        return "MINOR_DRIFT"
    return "CLEAN"


def _next_steps(report: ReconciliationReport, *, status: str) -> str:
    if status == "CLEAN":
        return "No action required. Continue scheduled automation."
    if status == "MINOR_DRIFT":
        return "Review next rebalance window and monitor drift before resubmitting."
    return "Investigate anomalies and consider manual rebalance before next Phase A run."


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    table = [render_row(headers), separator]
    for row in rows:
        table.append(render_row([str(cell) for cell in row]))
    return "\n".join(table)


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_setting(context: dict[str, Any], key: str, default: float) -> float:
    value = _optional_float(context.get(key))
    return default if value is None else value


__all__ = ["build_reconciliation_drift_snapshot", "generate_reconciliation_report"]
