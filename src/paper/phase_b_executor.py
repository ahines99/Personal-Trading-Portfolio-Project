from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .daily_gate import verify_prior_day_reconciliation
from .brokerage.mock import MockBrokerClient
from .loaders import resolve_baseline_path
from .models import ApprovalRecord, ApprovalStatus, IntentBundle, IntentBundleStatus
from .order_blotter import OrderBlotter
from .post_run_reconciliation import (
    build_reconciliation_drift_snapshot,
    generate_reconciliation_report,
)
from .reconciliation_engine import ReconciliationEngine
from .submission_engine import SubmissionEngine
from .verify import (
    compute_config_hash,
    validate_approval_freshness,
    validate_rebalance_approval,
)
from .writers import atomic_write_text


@dataclass(frozen=True)
class PhaseBBundle:
    bundle_dir: Path
    bundle: IntentBundle
    raw_payload: dict[str, Any]


class PhaseBExecutor:
    def __init__(
        self,
        config: Any | None = None,
        *,
        repo_root: str | Path | None = None,
        broker_client: Any | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        self.logger = logging.getLogger("paper.stage3.phase_b")
        self._broker_client = broker_client

    def run(
        self,
        *,
        bundle_dir: str | Path | None = None,
        approval_path: str | Path | None = None,
    ) -> dict[str, Any]:
        bundle: PhaseBBundle | None = None
        effective_bundle: IntentBundle | None = None
        broker: Any | None = None
        order_specs: list[dict[str, Any]] = []
        fills: list[dict[str, Any]] = []
        submission_day = datetime.now(timezone.utc).date()
        failure_stage = "kill_switch_preflight"

        try:
            self._check_kill_switch()
            failure_stage = "load_bundle"
            bundle = self._load_bundle(bundle_dir)
            current_config_hash = compute_config_hash(self.config)
            if bundle.bundle.config_hash != current_config_hash:
                raise ValueError(
                    "Phase A bundle config hash mismatch: "
                    f"current={current_config_hash} bundle={bundle.bundle.config_hash}"
                )

            bundle_date = bundle.bundle.as_of_date or bundle.bundle.generated_at.date()
            verify_prior_day_reconciliation(as_of_date=bundle_date, repo_root=self.repo_root)

            failure_stage = "resolve_approval"
            approval_record = self._resolve_approval_record(bundle, approval_path)
            effective_bundle = bundle.bundle.with_approval(approval_record)
            validate_rebalance_approval(current_config_hash, {"config_hash": bundle.bundle.config_hash})
            validate_approval_freshness(effective_bundle, approval_record)
            if approval_record.status != ApprovalStatus.APPROVED:
                raise ValueError("Phase B requires an approved intent bundle.")

            failure_stage = "broker_preflight"
            broker = self._get_broker_client(effective_bundle)
            if not broker.ping():
                raise RuntimeError("Broker connectivity failed during Phase B preflight.")

            drift_summary = self._check_overnight_drift(
                broker_positions=broker.get_positions(),
                current_holdings=effective_bundle.current_holdings,
            )
            self.logger.info(
                "phase_b.drift weight_drift_l1_pct=%.4f threshold_pct=%.4f",
                drift_summary["weight_drift_l1_pct"],
                drift_summary["threshold_pct"],
            )

            submission_engine = SubmissionEngine(
                broker,
                order_blotter=OrderBlotter(self.repo_root / "paper_trading" / "blotter" / "orders.jsonl"),
                timeout_seconds=float(self.config.get("submission_poll_seconds") or 60),
                poll_interval_seconds=float(self.config.get("submission_poll_interval_seconds") or 5),
                kill_switch_path=self._resolved_kill_switch_path(),
            )
            preview_lookup = self._preview_lookup(effective_bundle)
            order_specs = self._submission_specs(effective_bundle)
            approval_payload = approval_record.model_dump(mode="json", round_trip=True)
            approval_payload["status"] = approval_record.status.value
            approval_payload["config_hash"] = bundle.bundle.config_hash

            failure_stage = "submit_and_poll"
            submission_result = submission_engine.submit_and_poll(
                order_specs,
                preview_lookup,
                approval_payload,
            )

            failure_stage = "reconcile"
            reconciliation_engine = ReconciliationEngine(broker, repo_root=self.repo_root)
            reconciliation_result = reconciliation_engine.reconcile_and_update_positions(
                as_of_date=submission_day,
                target_weights=effective_bundle.target_weights,
                expected_nav=_expected_nav(self.config),
                rebalance_id=effective_bundle.rebalance_id,
                order_ids=[item["order_id"] for item in order_specs if item.get("order_id")],
            )
            fills = _fills_for_order_ids(
                self.repo_root / "paper_trading" / "blotter" / "fills.jsonl",
                {item["order_id"] for item in order_specs if item.get("order_id")},
            )
            report_context = self._build_report_context(
                bundle=effective_bundle,
                broker=broker,
                submission_day=submission_day,
                order_specs=order_specs,
                fills=fills,
            )
            drift_snapshot = build_reconciliation_drift_snapshot(report_context)
            actual_weights = drift_snapshot["actual_weights"]

            report_path = generate_reconciliation_report(
                submission_day,
                report_context,
            )
            bundle_report_path = bundle.bundle_dir / "reconciliation_report.json"
            _atomic_write_json(
                bundle_report_path,
                {
                    "status": "SUCCESS",
                    "as_of_date": submission_day.isoformat(),
                    "target_weights": effective_bundle.target_weights,
                    "actual_weights": actual_weights,
                    "executed_trades": fills,
                    "unfilled_orders": [],
                    "cancellations": [],
                    "reconciliation_ok": reconciliation_result.reconciliation_ok,
                    "drift": drift_snapshot,
                    "cash_drift": drift_snapshot["cash_drift"],
                    "nav_drift": drift_snapshot["nav_drift"],
                    "weight_drift_l1": drift_snapshot["weight_drift_l1"],
                    "tracking_vs_backtest": report_context.get("tracking_vs_backtest"),
                },
            )

            summary = {
                "status": "SUCCESS",
                "bundle_dir": str(bundle.bundle_dir),
                "executed_trade_count": submission_result.filled + submission_result.partial,
                "submitted_order_count": submission_result.submitted,
                "rejected_order_count": submission_result.rejected,
                "halted_order_count": submission_result.halted,
                "unfilled_order_count": submission_result.cancelled + submission_result.rejected,
                "weight_drift_l1": drift_summary["weight_drift_l1"],
                "post_weight_drift_l1": drift_snapshot["weight_drift_l1"],
                "post_weight_drift_l1_pct": drift_snapshot["weight_drift_l1_pct"],
                "cash_drift": drift_snapshot["cash_drift"],
                "nav_drift": drift_snapshot["nav_drift"],
                "drift_thresholds": drift_snapshot["thresholds"],
                "reconciliation_ok": reconciliation_result.reconciliation_ok,
                "reconciliation_report": report_path,
                "reconciliation_report_json": str(bundle_report_path),
                "reconciliation_json": reconciliation_result.reconciliation_path,
                "tracking_vs_backtest": report_context.get("tracking_vs_backtest"),
            }
            _atomic_write_json(bundle.bundle_dir / "phase_b_summary.json", summary)
            return summary
        except Exception as exc:
            if bundle is not None:
                self._write_failure_artifacts(
                    bundle=bundle,
                    effective_bundle=effective_bundle or bundle.bundle,
                    broker=broker,
                    submission_day=submission_day,
                    order_specs=order_specs,
                    fills=fills,
                    failure_stage=failure_stage,
                    error=exc,
                )
            raise

    def _load_bundle(self, bundle_dir: str | Path | None) -> PhaseBBundle:
        resolved_dir = _resolve_bundle_dir(self.repo_root, self.config, bundle_dir)
        intents_path = resolved_dir / "intents.json"
        if not intents_path.exists():
            raise FileNotFoundError(f"Missing Phase A intents artifact: {intents_path}")
        raw_payload = json.loads(intents_path.read_text(encoding="utf-8"))
        bundle = IntentBundle.model_validate(raw_payload)
        return PhaseBBundle(bundle_dir=resolved_dir, bundle=bundle, raw_payload=raw_payload)

    def _resolve_approval_record(
        self,
        bundle: PhaseBBundle,
        approval_path: str | Path | None,
    ) -> ApprovalRecord:
        if (
            bundle.bundle.status == IntentBundleStatus.APPROVED
            and bundle.bundle.approval_record is not None
        ):
            return bundle.bundle.approval_record

        path = _resolve_approval_path(bundle.bundle_dir, approval_path)
        if not path.exists():
            raise FileNotFoundError(
                "Bundle is not approved in intents.json and no legacy approval.json was found."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        approved_config_hash = payload.get("config_hash")
        if approved_config_hash not in (None, "") and str(approved_config_hash) != bundle.bundle.config_hash:
            raise ValueError(
                "Legacy approval config hash mismatch: "
                f"bundle={bundle.bundle.config_hash} approval={approved_config_hash}"
            )
        status = ApprovalStatus.APPROVED if bool(payload.get("approved")) else ApprovalStatus.REJECTED
        approved_at = payload.get("approved_at")
        if approved_at in (None, ""):
            raise ValueError("Legacy approval payload is missing required approved_at.")
        operator = payload.get("operator") or payload.get("approver") or "legacy-approval"
        comment = payload.get("comment")
        hostname = payload.get("hostname") or "legacy-approval"
        rebalance_id = payload.get("rebalance_id") or bundle.bundle.rebalance_id
        return ApprovalRecord(
            approved_at=approved_at,
            operator=operator,
            status=status,
            comment=comment,
            hostname=hostname,
            rebalance_id=rebalance_id,
        )

    def _get_broker_client(self, bundle: IntentBundle | None = None) -> Any:
        if self._broker_client is not None:
            return self._broker_client
        broker_name = str(self.config.get("broker") or "mock").strip().lower()
        if broker_name == "mock":
            return self._build_mock_broker_client(bundle)

        from .brokerage.factory import create_broker_client
        client = create_broker_client(self.config)
        if client is None:
            raise RuntimeError("Unable to create configured broker client.")
        return client

    def _build_mock_broker_client(self, bundle: IntentBundle | None) -> MockBrokerClient:
        notional = _expected_nav(self.config) or 100000.0
        holdings = dict(bundle.current_holdings) if bundle is not None else {}
        reference_prices = {}
        if bundle is not None and isinstance(bundle.metadata, dict):
            reference_prices = {
                str(ticker).strip().upper(): float(price)
                for ticker, price in dict(bundle.metadata.get("reference_prices") or {}).items()
                if price is not None and float(price) > 0
            }

        positions = []
        invested_value = 0.0
        for ticker, weight in holdings.items():
            normalized_ticker = str(ticker).strip().upper()
            normalized_weight = float(weight)
            if abs(normalized_weight) <= 1e-12:
                continue
            price = float(reference_prices.get(normalized_ticker, 100.0))
            market_value = normalized_weight * notional
            quantity = market_value / price if price > 0 else 0.0
            invested_value += market_value
            positions.append(
                {
                    "ticker": normalized_ticker,
                    "quantity": quantity,
                    "cost_basis": price,
                    "market_value": market_value,
                    "current_weight": normalized_weight,
                    "unrealized_pnl": 0.0,
                }
            )

        cash = notional - invested_value
        return MockBrokerClient(
            profile={
                "account_id": str(self.config.get("account_id") or "PAPER-12345"),
                "account_type": "paper",
                "status": "active",
                "user_name": "Phase B Mock",
                "email": None,
            },
            balances={
                "account_id": str(self.config.get("account_id") or "PAPER-12345"),
                "account_type": "paper",
                "cash": cash,
                "buying_power": max(notional, cash),
                "equity": notional,
                "market_value": invested_value,
                "total_equity": notional,
                "maintenance_requirement": 0.0,
            },
            positions=positions,
            fill_after_polls=max(1, int(self.config.get("mock_fill_after_polls", 3) or 3)),
        )

    def _check_kill_switch(self) -> None:
        kill_switch_path = self._resolved_kill_switch_path()
        if kill_switch_path.exists():
            raise RuntimeError(f"Kill switch engaged: {kill_switch_path}")

    def _resolved_kill_switch_path(self) -> Path:
        kill_switch_path = Path(str(self.config.get("kill_switch_path") or "paper_trading/state/KILL_SWITCH"))
        if not kill_switch_path.is_absolute():
            kill_switch_path = self.repo_root / kill_switch_path
        return kill_switch_path

    def _check_overnight_drift(
        self,
        *,
        broker_positions: list[dict[str, Any]],
        current_holdings: dict[str, float],
    ) -> dict[str, Any]:
        actual_weights = _positions_to_weight_map(broker_positions)
        drift_l1 = _weight_drift_l1(current_holdings, actual_weights)
        threshold_pct = float(self.config.get("overnight_drift_halt_pct") or 0.0)
        drift_l1_pct = drift_l1 * 100.0
        if threshold_pct > 0 and drift_l1_pct > threshold_pct:
            raise RuntimeError(
                "Overnight drift check failed: "
                f"{drift_l1_pct:.4f}% > {threshold_pct:.4f}%"
            )
        return {
            "weight_drift_l1": drift_l1,
            "weight_drift_l1_pct": drift_l1_pct,
            "threshold_pct": threshold_pct,
        }

    def _preview_lookup(self, bundle: IntentBundle) -> dict[str, dict[str, Any]]:
        if bundle.preview_batch is None:
            return {}
        lookup: dict[str, dict[str, Any]] = {}
        for result in bundle.preview_batch.results:
            lookup[_intent_hash(result.order)] = result.model_dump(mode="json", round_trip=True)
        return lookup

    def _submission_specs(self, bundle: IntentBundle) -> list[dict[str, Any]]:
        return [
            {
                "order_id": _find_seeded_order_id(
                    self.repo_root / "paper_trading" / "blotter" / "orders.jsonl",
                    bundle.rebalance_id,
                    _intent_hash(order),
                ),
                "rebalance_id": bundle.rebalance_id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side.value,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "parent_intent_hash": _intent_hash(order),
            }
            for order in bundle.proposed_orders
        ]

    def _build_report_context(
        self,
        *,
        bundle: IntentBundle,
        broker: Any | None,
        submission_day: date,
        order_specs: list[dict[str, Any]],
        fills: list[dict[str, Any]],
        anomalies: list[str] | None = None,
    ) -> dict[str, Any]:
        broker_positions = _safe_broker_call(broker, "get_positions")
        broker_balances = _safe_broker_call(broker, "get_balances")
        actual_weights = _coerce_actual_weights(
            broker_positions=broker_positions,
            broker_balances=broker_balances,
            fallback=bundle.current_holdings,
        )
        tracking_metrics = _compute_tracking_vs_backtest(
            repo_root=self.repo_root,
            config=self.config,
            bundle=bundle,
            submission_day=submission_day,
            target_weights=bundle.target_weights,
            actual_weights=actual_weights,
            tracking_notional=_expected_nav(self.config),
        )
        context: dict[str, Any] = {
            "repo_root": self.repo_root,
            "target_weights": bundle.target_weights,
            "actual_weights": actual_weights,
            "expected_nav": _expected_nav(self.config),
            "intended_trades": [dict(item) for item in order_specs],
            "fills": fills,
            "anomalies": list(anomalies or []),
            "tracking_vs_backtest": tracking_metrics,
            "pnl_reconciled_vs_backtest": tracking_metrics.get("pnl_reconciled_vs_backtest"),
            **_tracking_threshold_context(self.config),
            **_reconciliation_threshold_context(self.config),
        }
        if broker_positions is not None:
            context["broker_positions"] = broker_positions
        if broker_balances is not None:
            context["broker_balances"] = broker_balances
        return context

    def _write_failure_artifacts(
        self,
        *,
        bundle: PhaseBBundle,
        effective_bundle: IntentBundle,
        broker: Any | None,
        submission_day: date,
        order_specs: list[dict[str, Any]],
        fills: list[dict[str, Any]],
        failure_stage: str,
        error: Exception,
    ) -> None:
        anomaly = f"{failure_stage}: {error.__class__.__name__}: {error}"
        report_context = self._build_report_context(
            bundle=effective_bundle,
            broker=broker,
            submission_day=submission_day,
            order_specs=order_specs,
            fills=fills,
            anomalies=[anomaly],
        )
        drift_snapshot: dict[str, Any]
        try:
            drift_snapshot = build_reconciliation_drift_snapshot(report_context)
        except Exception as snapshot_exc:
            self.logger.exception("phase_b.failure_snapshot_failed: %s", snapshot_exc)
            drift_snapshot = {
                "status": "FAILED",
                "target_weights": effective_bundle.target_weights,
                "actual_weights": report_context.get("actual_weights", effective_bundle.current_holdings),
                "weight_drift_l1": 0.0,
                "weight_drift_l1_pct": 0.0,
                "cash_drift": 0.0,
                "nav_drift": 0.0,
                "thresholds": _reconciliation_threshold_context(self.config),
                "anomalies": [anomaly, f"snapshot_failed: {snapshot_exc}"],
            }

        report_path: str | None = None
        try:
            report_path = generate_reconciliation_report(submission_day, report_context)
        except Exception as report_exc:
            self.logger.exception("phase_b.failure_report_failed: %s", report_exc)

        bundle_report_path = bundle.bundle_dir / "reconciliation_report.json"
        _atomic_write_json(
            bundle_report_path,
            {
                "status": "FAILED",
                "failure_stage": failure_stage,
                "error_type": error.__class__.__name__,
                "error": str(error),
                "as_of_date": submission_day.isoformat(),
                "target_weights": effective_bundle.target_weights,
                "actual_weights": drift_snapshot.get("actual_weights", effective_bundle.current_holdings),
                "executed_trades": fills,
                "unfilled_orders": [],
                "cancellations": [],
                "reconciliation_ok": False,
                "drift": drift_snapshot,
                "cash_drift": drift_snapshot.get("cash_drift", 0.0),
                "nav_drift": drift_snapshot.get("nav_drift", 0.0),
                "weight_drift_l1": drift_snapshot.get("weight_drift_l1", 0.0),
                "anomalies": list(drift_snapshot.get("anomalies", [])),
                "tracking_vs_backtest": report_context.get("tracking_vs_backtest"),
            },
        )
        _atomic_write_json(
            bundle.bundle_dir / "phase_b_summary.json",
            {
                "status": "FAILED",
                "bundle_dir": str(bundle.bundle_dir),
                "failure_stage": failure_stage,
                "error_type": error.__class__.__name__,
                "error": str(error),
                "reconciliation_ok": False,
                "reconciliation_report": report_path,
                "reconciliation_report_json": str(bundle_report_path),
                "post_weight_drift_l1": drift_snapshot.get("weight_drift_l1", 0.0),
                "post_weight_drift_l1_pct": drift_snapshot.get("weight_drift_l1_pct", 0.0),
                "cash_drift": drift_snapshot.get("cash_drift", 0.0),
                "nav_drift": drift_snapshot.get("nav_drift", 0.0),
                "drift_thresholds": drift_snapshot.get("thresholds", _reconciliation_threshold_context(self.config)),
                "tracking_vs_backtest": report_context.get("tracking_vs_backtest"),
            },
        )


def _resolve_bundle_dir(repo_root: Path, config: dict[str, Any], bundle_dir: str | Path | None) -> Path:
    if bundle_dir:
        candidate = Path(bundle_dir)
        return candidate if candidate.is_absolute() else repo_root / candidate
    results_dir = Path(str(config.get("results_dir") or "results"))
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    candidates = sorted(results_dir.glob("_paper_phase_a_*"))
    if not candidates:
        raise FileNotFoundError("No Phase A bundle found under results/.")
    return candidates[-1]


def _resolve_approval_path(bundle_dir: Path, approval_path: str | Path | None) -> Path:
    if approval_path:
        candidate = Path(approval_path)
        return candidate if candidate.is_absolute() else bundle_dir / candidate
    return bundle_dir / "approval.json"


def _coerce_config(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    return {}


def _positions_to_weight_map(positions: list[dict[str, Any]]) -> dict[str, float]:
    rows = {
        str(item["ticker"]).strip().upper(): float(item.get("current_weight") or 0.0)
        for item in positions
    }
    return {ticker: weight for ticker, weight in rows.items() if abs(weight) > 1e-12}


def _coerce_actual_weights(
    *,
    broker_positions: Any,
    broker_balances: Any,
    fallback: dict[str, float],
) -> dict[str, float]:
    if not broker_positions:
        return {
            str(ticker).strip().upper(): float(weight)
            for ticker, weight in (fallback or {}).items()
            if abs(float(weight)) > 1e-12
        }

    nav = _broker_nav(broker_balances, broker_positions)
    iterable = broker_positions.values() if isinstance(broker_positions, dict) else broker_positions
    rows: dict[str, float] = {}
    for item in iterable:
        if isinstance(item, dict):
            ticker = str(item.get("ticker") or "").strip().upper()
            market_value = _optional_float(item.get("market_value"))
            current_weight = _optional_float(item.get("current_weight"))
        else:
            ticker = str(getattr(item, "ticker", "")).strip().upper()
            market_value = _optional_float(getattr(item, "market_value", None))
            current_weight = _optional_float(getattr(item, "current_weight", None))
        if not ticker:
            continue
        if current_weight is not None:
            rows[ticker] = current_weight
        elif market_value is not None and nav is not None and abs(nav) > 1e-9:
            rows[ticker] = market_value / nav
    return {ticker: weight for ticker, weight in rows.items() if abs(weight) > 1e-12}


def _weight_drift_l1(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(abs(float(left.get(ticker, 0.0)) - float(right.get(ticker, 0.0))) for ticker in set(left) | set(right))


def _expected_nav(config: dict[str, Any]) -> float | None:
    value = config.get("portfolio_notional_usd", config.get("notional"))
    if value in (None, ""):
        return None
    return float(value)


def _compute_tracking_vs_backtest(
    *,
    repo_root: Path,
    config: dict[str, Any],
    bundle: IntentBundle,
    submission_day: date,
    target_weights: dict[str, float],
    actual_weights: dict[str, float],
    tracking_notional: float | None,
) -> dict[str, Any]:
    try:
        baseline_dir = resolve_baseline_path(
            config=config,
            baseline_path=config.get("baseline_path"),
            repo_root=repo_root,
        )
    except Exception as exc:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": f"baseline_resolve_failed: {exc}",
        }

    returns_path = baseline_dir / "returns_panel.parquet"
    if not returns_path.exists():
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": f"missing returns_panel artifact: {returns_path}",
            "baseline_path": str(baseline_dir),
        }

    returns_panel = pd.read_parquet(returns_path)
    if returns_panel.empty:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": f"returns_panel is empty: {returns_path}",
            "baseline_path": str(baseline_dir),
            "returns_path": str(returns_path),
        }

    returns_panel.index = pd.to_datetime(returns_panel.index).normalize()
    returns_panel = returns_panel.sort_index()
    return_date, return_date_source = _select_tracking_return_date(
        returns_panel.index,
        submission_day=submission_day,
        bundle=bundle,
    )
    if return_date is None:
        bundle_date = bundle.as_of_date.isoformat() if bundle.as_of_date is not None else None
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "reason": (
                "no return row found for submission_day="
                f"{submission_day.isoformat()} or bundle_as_of_date={bundle_date}"
            ),
            "baseline_path": str(baseline_dir),
            "returns_path": str(returns_path),
        }

    returns_row = returns_panel.loc[return_date]
    if isinstance(returns_row, pd.DataFrame):
        returns_row = returns_row.iloc[-1]
    returns_row = pd.to_numeric(returns_row, errors="coerce").fillna(0.0)
    aligned_index = pd.Index(sorted(set(returns_row.index) | set(target_weights) | set(actual_weights)))
    aligned_returns = returns_row.reindex(aligned_index).fillna(0.0).astype("float64")
    target_series = pd.Series(target_weights, dtype="float64").reindex(aligned_index).fillna(0.0)
    actual_series = pd.Series(actual_weights, dtype="float64").reindex(aligned_index).fillna(0.0)

    backtest_predicted_return = float((target_series * aligned_returns).sum())
    paper_realized_return = float((actual_series * aligned_returns).sum())
    return_excess = paper_realized_return - backtest_predicted_return
    capital_base = float(tracking_notional) if tracking_notional not in (None, "") else 1.0

    return {
        "available": True,
        "status": "OK",
        "baseline_path": str(baseline_dir),
        "returns_path": str(returns_path),
        "return_date": pd.Timestamp(return_date).date().isoformat(),
        "return_date_source": return_date_source,
        "paper_realized_return": paper_realized_return,
        "backtest_predicted_return": backtest_predicted_return,
        "return_excess_vs_backtest": return_excess,
        "paper_realized_pnl": paper_realized_return * capital_base,
        "backtest_predicted_pnl": backtest_predicted_return * capital_base,
        "pnl_reconciled_vs_backtest": return_excess * capital_base,
        "tracking_capital_base": capital_base,
        "tracked_symbol_count": int((aligned_returns.abs() > 0).sum()),
    }


def _select_tracking_return_date(
    index: pd.Index,
    *,
    submission_day: date,
    bundle: IntentBundle,
) -> tuple[pd.Timestamp | None, str | None]:
    normalized_index = pd.DatetimeIndex(pd.to_datetime(index)).normalize().sort_values()
    submission_ts = pd.Timestamp(submission_day).normalize()
    if submission_ts in normalized_index:
        return submission_ts, "submission_day"

    bundle_ts = pd.Timestamp(bundle.as_of_date).normalize() if bundle.as_of_date is not None else None
    if bundle_ts is not None and bundle_ts in normalized_index:
        return bundle_ts, "bundle_as_of_date"

    prior_submission = normalized_index[normalized_index <= submission_ts]
    if len(prior_submission) > 0:
        return pd.Timestamp(prior_submission[-1]).normalize(), "latest_on_or_before_submission_day"

    if bundle_ts is not None:
        prior_bundle = normalized_index[normalized_index <= bundle_ts]
        if len(prior_bundle) > 0:
            return pd.Timestamp(prior_bundle[-1]).normalize(), "latest_on_or_before_bundle_as_of_date"
    return None, None


def _broker_nav(balances: Any, positions: Any) -> float | None:
    if isinstance(balances, dict):
        for key in ("nav", "equity", "total_equity"):
            value = _optional_float(balances.get(key))
            if value is not None and abs(value) > 1e-9:
                return value
        cash = _optional_float(balances.get("cash", balances.get("total_cash")))
    else:
        cash = None

    total_market_value = 0.0
    found_market_value = False
    iterable = positions.values() if isinstance(positions, dict) else (positions or [])
    for item in iterable:
        if isinstance(item, dict):
            market_value = _optional_float(item.get("market_value"))
        else:
            market_value = _optional_float(getattr(item, "market_value", None))
        if market_value is None:
            continue
        total_market_value += market_value
        found_market_value = True
    if cash is not None and found_market_value:
        return cash + total_market_value
    return None


def _reconciliation_threshold_context(config: dict[str, Any]) -> dict[str, float]:
    weight_alert_pct = float(
        config.get(
            "reconciliation_weight_drift_alert_pct",
            config.get("overnight_drift_halt_pct", 5.0),
        )
    )
    weight_minor_pct = float(config.get("reconciliation_weight_drift_minor_pct", 2.0))
    return {
        "weight_drift_alert_threshold": weight_alert_pct / 100.0,
        "weight_drift_minor_threshold": weight_minor_pct / 100.0,
        "cash_drift_abs_threshold": float(config.get("reconciliation_cash_drift_usd", 1.0)),
        "nav_drift_abs_threshold": float(config.get("reconciliation_nav_drift_usd", 1.0)),
    }


def _tracking_threshold_context(config: dict[str, Any]) -> dict[str, float | int]:
    return {
        "tracking_error_window_days": int(config.get("tracking_error_window_days", 21)),
        "tracking_error_annualization_days": int(config.get("tracking_error_annualization_days", 252)),
        "tracking_error_alert_threshold": float(config.get("tracking_error_alert_pct", 0.75)) / 100.0,
        "monthly_win_rate_min_threshold": float(config.get("tracking_monthly_win_rate_min_pct", 55.0)) / 100.0,
    }


def _intent_hash(order: Any) -> str:
    payload = order.cache_key_payload() if hasattr(order, "cache_key_payload") else dict(order)
    return compute_config_hash(payload)


def _find_seeded_order_id(path: Path, rebalance_id: str | None, parent_intent_hash: str) -> str | None:
    if not path.exists():
        return None
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[row["order_id"]] = row
    for row in latest.values():
        if rebalance_id and str(row.get("rebalance_id") or "") != str(rebalance_id):
            continue
        if str(row.get("parent_intent_hash") or "") == parent_intent_hash:
            return str(row["order_id"])
    return None


def _fills_for_order_ids(path: Path, order_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists() or not order_ids:
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("order_id")) in order_ids:
            rows.append(row)
    return rows


def _safe_broker_call(broker: Any | None, method_name: str) -> Any | None:
    if broker is None:
        return None
    method = getattr(broker, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True),
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["PhaseBExecutor"]
