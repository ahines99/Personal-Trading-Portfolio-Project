"""Stage 3 fill reconciliation and end-of-day position snapshotting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from .brokerage.interface import BrokerClient
from .fill_blotter import FillBlotter, FillRecord
from .order_blotter import OrderBlotter, OrderStatus
from .schemas import ReconciliationReport, SlippageSummary

_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.PARTIALLY_FILLED,
}


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of an end-of-day reconciliation pass."""

    reconciliation_ok: bool
    discrepancies: list[str] = field(default_factory=list)
    fills_written: int = 0
    positions_path: str | None = None
    reconciliation_path: str | None = None


class ReconciliationEngine:
    """Fetch broker fills, persist them, and refresh the paper position snapshot."""

    def __init__(
        self,
        broker_client: BrokerClient,
        *,
        order_blotter: OrderBlotter | None = None,
        fill_blotter: FillBlotter | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.broker_client = broker_client
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        self.order_blotter = order_blotter or OrderBlotter(
            self.repo_root / "paper_trading" / "blotter" / "orders.jsonl"
        )
        self.fill_blotter = fill_blotter or FillBlotter(
            self.repo_root / "paper_trading" / "blotter" / "fills.jsonl"
        )

    def reconcile_and_update_positions(
        self,
        *,
        end_of_day: bool = True,
        as_of_date: date | str | None = None,
        target_weights: dict[str, float] | None = None,
        expected_cash: float | None = None,
        expected_nav: float | None = None,
    ) -> ReconciliationResult:
        del end_of_day
        reconcile_date = self._coerce_date(as_of_date)
        discrepancies: list[str] = []
        fills_written = 0

        orders = self._orders_for_date(reconcile_date)
        executed_trades: list[dict[str, Any]] = []
        unfilled_orders: list[dict[str, Any]] = []
        cancellations: list[dict[str, Any]] = []

        for order in orders:
            if order.status not in _TERMINAL_STATUSES or not order.broker_order_id:
                discrepancies.append(
                    f"Order {order.order_id} is not terminal or lacks broker_order_id."
                )
                continue

            broker_status = self.broker_client.get_order_status(order.broker_order_id)
            fills = self._extract_fills(order, broker_status)

            if order.status == OrderStatus.FILLED and not fills:
                discrepancies.append(
                    f"Filled order {order.order_id} has no broker fill details."
                )

            for fill in fills:
                existing = self.fill_blotter.get_fill(fill.fill_id)
                if existing is None:
                    self.fill_blotter.append_fill(
                        fill,
                        submission_timestamp=order.submission_timestamp,
                    )
                    fills_written += 1
                executed_trades.append(fill.to_dict())

            filled_qty = sum(fill.qty for fill in fills)
            if order.status == OrderStatus.CANCELLED:
                cancellations.append(
                    self._order_summary(order, broker_status, filled_qty=filled_qty)
                )
            elif filled_qty <= 0.0:
                unfilled_orders.append(
                    self._order_summary(order, broker_status, filled_qty=filled_qty)
                )

            if order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                tolerance = max(order.qty * 0.005, 1e-6)
                if abs(filled_qty - order.qty) > tolerance:
                    discrepancies.append(
                        f"Order {order.order_id} filled_qty mismatch: "
                        f"{filled_qty:.6f} vs requested {order.qty:.6f}"
                    )

        balances = self.broker_client.get_balances()
        positions = self.broker_client.get_positions()
        positions_payload = self._build_positions_payload(
            reconcile_date,
            balances=balances,
            positions=positions,
        )

        positions_path = self.repo_root / "paper_trading" / "positions.json"
        self._atomic_write_json(positions_path, positions_payload)

        report = self._build_report(
            reconcile_date,
            orders=orders,
            executed_trades=executed_trades,
            unfilled_orders=unfilled_orders,
            cancellations=cancellations,
            balances=balances,
            positions=positions,
            discrepancies=discrepancies,
            target_weights=target_weights,
            expected_cash=expected_cash,
            expected_nav=expected_nav,
        )
        reconciliation_path = (
            self.repo_root
            / "paper_trading"
            / "reconciliation"
            / f"{reconcile_date.isoformat()}.json"
        )
        self._atomic_write_json(
            reconciliation_path,
            {
                "reconciliation_ok": not discrepancies,
                "discrepancies": list(discrepancies),
                "report": report.model_dump(mode="json", round_trip=True),
                "positions_path": str(positions_path),
            },
        )

        return ReconciliationResult(
            reconciliation_ok=not discrepancies,
            discrepancies=discrepancies,
            fills_written=fills_written,
            positions_path=str(positions_path),
            reconciliation_path=str(reconciliation_path),
        )

    def _orders_for_date(self, reconcile_date: date) -> list[Any]:
        latest = self.order_blotter.list_orders(latest_only=True)
        orders = [
            order
            for order in latest
            if self._coerce_date(order.submission_timestamp or order.timestamp) == reconcile_date
        ]
        orders.sort(key=lambda row: row.submission_timestamp or row.timestamp)
        return orders

    def _extract_fills(self, order: Any, broker_status: dict[str, Any]) -> list[FillRecord]:
        terminal_poll = None
        payloads = broker_status.get("fills") or broker_status.get("fill") or []
        if not payloads and isinstance(order.preview_result, dict):
            terminal_poll = order.preview_result.get("terminal_poll")
            if isinstance(terminal_poll, dict):
                payloads = terminal_poll.get("fills") or terminal_poll.get("fill") or []
        if isinstance(payloads, dict):
            payloads = [payloads]
        if not isinstance(payloads, list):
            payloads = []

        expected_price = None
        if isinstance(order.preview_result, dict):
            expected_price = self._coerce_float(
                order.preview_result.get("expected_price", order.preview_result.get("reference_price"))
            )
            if expected_price <= 0.0:
                expected_price = None

        records: list[FillRecord] = []
        for index, payload in enumerate(payloads):
            qty = self._coerce_float(
                payload.get("qty", payload.get("quantity", payload.get("filled_qty")))
            )
            price = self._coerce_float(
                payload.get("price", payload.get("avg_fill_price", broker_status.get("avg_fill_price")))
            )
            if qty <= 0.0 or price <= 0.0:
                continue

            broker_fill_id = payload.get("broker_fill_id") or payload.get("fill_id") or payload.get("id")
            timestamp = (
                payload.get("timestamp")
                or payload.get("filled_at")
                or broker_status.get("filled_at")
                or (terminal_poll.get("timestamp") if isinstance(terminal_poll, dict) else None)
            )
            fill_id_seed = f"{order.order_id}:{broker_fill_id or index}:{timestamp or ''}:{qty}:{price}"
            records.append(
                FillRecord(
                    fill_id=uuid5(NAMESPACE_URL, fill_id_seed),
                    order_id=UUID(str(order.order_id)),
                    parent_intent_hash=order.parent_intent_hash,
                    symbol=order.symbol,
                    side=order.side,
                    qty=qty,
                    price=price,
                    commission=self._coerce_float(
                        payload.get("commission", broker_status.get("commission"))
                    ),
                    timestamp=timestamp or broker_status.get("timestamp") or order.timestamp,
                    broker_fill_id=str(broker_fill_id) if broker_fill_id else None,
                    expected_price=expected_price,
                )
            )
        return records

    def _build_positions_payload(
        self,
        reconcile_date: date,
        *,
        balances: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cash = self._coerce_float(
            balances.get("cash", balances.get("total_cash"))
        )
        payload_positions: dict[str, Any] = {}
        for position in positions:
            ticker = str(position["ticker"]).strip().upper()
            quantity = self._coerce_float(position.get("quantity"))
            gross_cost = self._coerce_float(position.get("cost_basis"))
            vwap = gross_cost / quantity if abs(quantity) > 1e-9 else 0.0
            payload_positions[ticker] = {
                "qty": quantity,
                "vwap": vwap,
                "gross_cost": gross_cost,
                "market_value": self._coerce_float(position.get("market_value")),
                "current_weight": self._coerce_float(position.get("current_weight")),
                "unrealized_pnl": self._coerce_float(position.get("unrealized_pnl")),
            }
        return {
            "snapshot_time": reconcile_date.isoformat(),
            "cash": cash,
            "positions": dict(sorted(payload_positions.items())),
        }

    def _build_report(
        self,
        reconcile_date: date,
        *,
        orders: list[Any],
        executed_trades: list[dict[str, Any]],
        unfilled_orders: list[dict[str, Any]],
        cancellations: list[dict[str, Any]],
        balances: dict[str, Any],
        positions: list[dict[str, Any]],
        discrepancies: list[str],
        target_weights: dict[str, float] | None = None,
        expected_cash: float | None = None,
        expected_nav: float | None = None,
    ) -> ReconciliationReport:
        balance_snapshot = self._balance_snapshot(balances, positions)
        nav = balance_snapshot["actual_nav"]
        actual_weights = self._actual_weights_from_positions(positions, nav=nav)
        normalized_targets = self._normalize_weight_map(target_weights)
        if not normalized_targets:
            normalized_targets = dict(actual_weights)
        cash_drift, nav_drift = self._cash_nav_drift(
            target_weights=normalized_targets if target_weights else {},
            actual_cash=balance_snapshot["actual_cash"],
            actual_nav=nav,
            expected_cash=expected_cash,
            expected_nav=expected_nav,
        )
        weight_comparisons = ReconciliationReport.build_weight_comparisons(
            normalized_targets,
            actual_weights,
        )
        slippage_values = []
        for trade in executed_trades:
            value = trade.get("slippage_bps")
            if value is not None:
                slippage_values.append(float(value))

        return ReconciliationReport(
            as_of_date=reconcile_date,
            intended_trades=[self._order_summary(order, {}, filled_qty=0.0) for order in orders],
            executed_trades=executed_trades,
            unfilled_orders=unfilled_orders,
            cancellations=cancellations,
            slippage_summary=SlippageSummary.from_bps(slippage_values),
            target_weights=normalized_targets,
            actual_weights=actual_weights,
            weight_comparisons=weight_comparisons,
            weight_drift_l1=sum(abs(item.drift) for item in weight_comparisons),
            cash_drift=cash_drift,
            nav_drift=nav_drift,
            pnl_reconciled_vs_backtest=None,
            anomalies=list(discrepancies),
        )

    def _order_summary(
        self,
        order: Any,
        broker_status: dict[str, Any],
        *,
        filled_qty: float,
    ) -> dict[str, Any]:
        return {
            "order_id": str(order.order_id),
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "qty": order.qty,
            "filled_qty": filled_qty,
            "status": order.status.value,
            "broker_status": broker_status.get("status"),
        }

    def _coerce_date(self, value: date | str | None) -> date:
        if value is None:
            return date.today()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    def _coerce_float(self, value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _coerce_optional_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_weight_map(self, payload: dict[str, float] | None) -> dict[str, float]:
        if not payload:
            return {}
        return {
            str(ticker).strip().upper(): float(weight)
            for ticker, weight in payload.items()
            if abs(float(weight)) > 1e-12
        }

    def _balance_snapshot(
        self,
        balances: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> dict[str, float]:
        actual_cash = self._coerce_float(balances.get("cash", balances.get("total_cash")))
        actual_nav = self._coerce_float(
            balances.get("nav", balances.get("equity", balances.get("total_equity")))
        )
        market_value_total = sum(
            self._coerce_float(position.get("market_value")) for position in positions
        )
        if abs(actual_nav) <= 1e-9:
            actual_nav = actual_cash + market_value_total
        return {
            "actual_cash": actual_cash,
            "actual_nav": actual_nav,
            "market_value_total": market_value_total,
        }

    def _actual_weights_from_positions(
        self,
        positions: list[dict[str, Any]],
        *,
        nav: float,
    ) -> dict[str, float]:
        actual_weights: dict[str, float] = {}
        for position in positions:
            ticker = str(position["ticker"]).strip().upper()
            market_value = self._coerce_optional_float(position.get("market_value"))
            if market_value is not None and abs(nav) > 1e-9:
                actual_weights[ticker] = market_value / nav
                continue
            actual_weights[ticker] = self._coerce_float(position.get("current_weight"))
        return {
            ticker: weight
            for ticker, weight in actual_weights.items()
            if abs(weight) > 1e-12
        }

    def _cash_nav_drift(
        self,
        *,
        target_weights: dict[str, float],
        actual_cash: float,
        actual_nav: float,
        expected_cash: float | None,
        expected_nav: float | None,
    ) -> tuple[float, float]:
        nav_reference = expected_nav if expected_nav is not None else actual_nav
        cash_reference = expected_cash
        if cash_reference is None and target_weights and abs(nav_reference) > 1e-9:
            target_cash_weight = 1.0 - sum(target_weights.values())
            cash_reference = target_cash_weight * nav_reference
        cash_drift = actual_cash - cash_reference if cash_reference is not None else 0.0
        nav_drift = actual_nav - expected_nav if expected_nav is not None else 0.0
        return cash_drift, nav_drift

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)


__all__ = ["ReconciliationEngine", "ReconciliationResult"]
