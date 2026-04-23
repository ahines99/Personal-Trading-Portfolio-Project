from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from zoneinfo import ZoneInfo

from src.paper.brokerage.factory import create_broker_client
from src.paper.brokerage.mock import MockBrokerClient
from src.paper.auto_approval_gate import auto_approve
from src.paper.diff_engine import DiffEngine
from src.paper.loaders import load_target_book
from src.paper.models import (
    ApprovalRecord,
    ApprovalStatus,
    IntentAggregate,
    IntentBundle,
    IntentBundleStatus,
    OrderSpec as IntentOrderSpec,
    PreviewBatch,
    PreviewBatchTotals,
)
from src.paper.order_blotter import OrderBlotter, OrderStatus
from src.paper.order_policy import OrderPolicy
from src.paper.canary_mode import evaluate_canary_orders
from src.paper.preflight_gate import preflight_passes
from src.paper.preview_engine import PreviewEngine
from src.paper.verify import compute_config_hash

EASTERN_TZ = ZoneInfo("America/New_York")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 Phase A paper-trading intent generator")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON paper config path")
    parser.add_argument("--as-of-date", default=None, help="Override as-of date (YYYY-MM-DD)")
    parser.add_argument("--results-dir", default=None, help="Override bundle output root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    config = _load_config(args.config)
    if args.results_dir:
        config["results_dir"] = args.results_dir
    log_date = pd.Timestamp(args.as_of_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")).strftime("%Y-%m-%d")
    _configure_logging(repo_root, log_date)
    logger = logging.getLogger("paper.stage3.phase_a.cli")

    try:
        result = run_phase_a(config=config, repo_root=repo_root, as_of_date=args.as_of_date)
    except Exception:
        logger.exception("paper phase A failed")
        return 1

    print(json.dumps(result, indent=2))
    logger.info(
        "paper phase A complete bundle_dir=%s order_count=%s skipped_count=%s",
        result["bundle_dir"],
        result["order_count"],
        result["skipped_count"],
    )
    return 0


def run_phase_a(
    *,
    config: dict[str, Any],
    repo_root: Path,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    logger = logging.getLogger("paper.stage3.phase_a")
    config_hash = compute_config_hash(config)
    as_of_ts = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp.utcnow().normalize()
    generated_at = datetime.now(timezone.utc)

    target_weights, prior_weights, metadata = load_target_book(
        as_of_date=as_of_ts,
        baseline_path=config.get("baseline_path"),
        config=config,
        repo_root=repo_root,
    )
    rebalance_id = str(uuid4())
    notional = float(config.get("portfolio_notional_usd", config.get("notional", 100000.0)))
    min_trade_notional = float(config.get("min_trade_notional_usd", config.get("min_trade_notional", 50.0)))
    reference_prices = _reference_prices_for_phase_a(metadata, target_weights, prior_weights, notional=notional)

    broker_client, broker_state = _build_broker_context(
        config=config,
        repo_root=repo_root,
        current_holdings=prior_weights,
        reference_prices=reference_prices,
        notional=notional,
    )
    current_holdings = _current_holdings_from_broker_state(broker_state) or _series_weight_map(prior_weights)

    diff_rows: list[dict[str, Any]] = []
    skipped_orders: list[dict[str, Any]] = []
    policy_orders: list[Any] = []
    preflight_result: tuple[bool, list[str]] = (True, [])

    if bool(metadata.get("is_rebalance_day")):
        target_payload = _build_target_payload(target_weights, reference_prices)
        previous_payload = _build_previous_payload(prior_weights, reference_prices)
        diff_engine = DiffEngine(min_trade_notional=min_trade_notional)
        proposed_trades = diff_engine.compute_trades(
            target_payload,
            broker_state,
            previous_positions=previous_payload,
            total_portfolio_value=notional,
        )
        diff_rows = proposed_trades.as_records()

        policy = OrderPolicy(
            max_single_order_notional=float(
                config.get("max_single_order_notional_usd", config.get("max_single_order_notional", 50000.0))
            ),
            max_order_count=int(config.get("max_order_count", 10)),
            qty_rounding="share",
            order_type=_policy_order_type(config),
            allowed_order_types={_policy_order_type(config)},
        )
        policy_result = policy.build_order_specs(proposed_trades)
        policy_orders = list(policy_result.orders)
        skipped_orders.extend(list(policy_result.skipped))
        skipped_orders.extend({"reason": "policy_warning", "warning": warning} for warning in policy_result.warnings)

        if bool(config.get("canary_mode")) and policy_orders:
            canary_eval = evaluate_canary_orders(
                [order.as_record() for order in policy_orders],
                max_single_order_notional_usd=float(
                    config.get("canary_max_single_order_notional_usd", 500.0)
                ),
                max_orders_per_day=int(config.get("canary_max_order_count", 5)),
            )
            accepted_ids = {
                item["order_spec_id"]
                for item in canary_eval["accepted_orders"]
                if item.get("order_spec_id")
            }
            policy_orders = [
                order for order in policy_orders if order.order_spec_id in accepted_ids
            ]
            skipped_orders.extend(canary_eval["capped_orders"])
            skipped_orders.extend(canary_eval["deferred_orders"])
    else:
        skipped_orders.append(
            {
                "reason": "non_rebalance_day",
                "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
            }
        )

    preview_specs = [IntentOrderSpec.model_validate(order.preview_payload()) for order in policy_orders]
    preview_engine = PreviewEngine(broker_client)
    preview_batch = preview_engine.preview_batch(preview_specs, abort_on_error=False) if preview_specs else None
    if int(config.get("stage", 1)) >= 4:
        preflight_result = preflight_passes(
            {
                "config": config,
                "repo_root": repo_root,
                "results_dir": repo_root / "paper_trading" / "reports",
                "manifest": {"config_hash": config_hash},
                "signal_path": Path(str(config.get("baseline_path") or metadata.get("baseline_path"))) / "final_signal.parquet",
                "broker_client": broker_client,
                "is_rebalance_day": bool(metadata.get("is_rebalance_day")),
                "kill_switch_path": config.get("kill_switch_path"),
                "alert_log_path": config.get("alert_log_path"),
            }
        )
        if config.get("stage4_mode") == "auto" and not preflight_result[0]:
            raise RuntimeError("Stage 4 auto preflight failed: " + "; ".join(preflight_result[1]))

    proposed_orders: list[IntentOrderSpec] = []
    valid_preview_results = []
    order_blotter = OrderBlotter(repo_root / "paper_trading" / "blotter" / "orders.jsonl")
    preview_lookup: dict[str, Any] = {}

    if preview_batch is not None:
        order_by_hash = {
            _intent_hash(IntentOrderSpec.model_validate(order.preview_payload())): order
            for order in policy_orders
        }
        for preview_result in preview_batch.results:
            intent_hash = _intent_hash(preview_result.order)
            order = order_by_hash[intent_hash]
            row = order_blotter.create_order(
                rebalance_id=rebalance_id,
                symbol=preview_result.order.symbol,
                side=preview_result.order.side.value,
                qty=preview_result.order.qty,
                order_type=preview_result.order.order_type,
                limit_price=preview_result.order.limit_price,
                parent_intent_hash=intent_hash,
                status=OrderStatus.PREVIEW_PENDING,
            )
            preview_payload = preview_result.model_dump(mode="json", round_trip=True)
            preview_payload["order_id"] = str(row.order_id)
            preview_payload["metadata"] = order.as_record()

            if preview_result.errors:
                order_blotter.attach_preview(row.order_id, preview_payload, approved=False)
                skipped_orders.append(
                    {
                        "symbol": preview_result.order.symbol,
                        "reason": "preview_error",
                        "errors": list(preview_result.errors),
                    }
                )
                continue

            order_blotter.attach_preview(row.order_id, preview_payload, approved=True)
            order_blotter.update_status(row.order_id, OrderStatus.APPROVAL_PENDING)
            preview_lookup[intent_hash] = preview_result
            proposed_orders.append(preview_result.order)
            valid_preview_results.append(preview_result)

    filtered_preview_batch = _filtered_preview_batch(preview_batch, valid_preview_results)
    turnover_pct = (
        sum(abs(item.get("est_notional", item.get("delta_notional", 0.0))) for item in diff_rows) / notional * 100.0
        if diff_rows and notional > 0
        else 0.0
    )
    aggregate = (
        IntentAggregate.from_preview_batch(filtered_preview_batch, turnover_pct=turnover_pct)
        if filtered_preview_batch is not None
        else IntentAggregate(turnover_pct=turnover_pct)
    )

    bundle = IntentBundle(
        as_of_date=as_of_ts.date(),
        rebalance_id=rebalance_id,
        generated_at=generated_at,
        approval_deadline=_approval_deadline(generated_at, config),
        status=IntentBundleStatus.AWAITING_APPROVAL,
        signal_hash=_signal_hash(target_weights, metadata),
        config_hash=config_hash,
        target_weights=_series_weight_map(target_weights),
        current_holdings=current_holdings,
        proposed_orders=tuple(proposed_orders),
        aggregate=aggregate,
        preview_batch=filtered_preview_batch,
        metadata={
            "baseline_path": str(config.get("baseline_path") or metadata.get("baseline_path") or ""),
            "is_rebalance_day": bool(metadata.get("is_rebalance_day")),
            "weight_source": metadata.get("weight_source"),
            "reference_prices": reference_prices,
            "target_positions_count": int((target_weights.abs() > 1e-12).sum()),
            "skipped_orders": skipped_orders,
        },
    )
    if (
        int(config.get("stage", 1)) >= 4
        and str(config.get("stage4_mode") or "manual").lower() == "auto"
        and filtered_preview_batch is not None
    ):
        approved, reason = auto_approve(
            {
                "config": config,
                "config_hash": config_hash,
                "signal_hash": bundle.signal_hash,
                "is_rebalance_day": bool(metadata.get("is_rebalance_day")),
                "repo_root": repo_root,
                "as_of_date": as_of_ts.date().isoformat(),
            },
            [result.model_dump(mode="json", round_trip=True) for result in filtered_preview_batch.results],
            preflight_result,
        )
        bundle = bundle.with_approval(
            ApprovalRecord(
                approved_at=generated_at,
                operator="stage4-auto",
                status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
                comment=reason,
                hostname="stage4-auto",
                rebalance_id=rebalance_id,
            )
        )

    bundle_dir = _write_phase_a_bundle(
        repo_root=repo_root,
        config=config,
        bundle=bundle,
        target_weights=target_weights,
        prior_weights=pd.Series(current_holdings, dtype=float),
        intended_trades=pd.DataFrame(diff_rows),
        skipped_orders=skipped_orders,
    )
    logger.info(
        "phase_a.bundle written bundle_dir=%s order_count=%s skipped_count=%s rebalance_day=%s",
        bundle_dir,
        len(proposed_orders),
        len(skipped_orders),
        bool(metadata.get("is_rebalance_day")),
    )
    return {
        "bundle_dir": str(bundle_dir),
        "rebalance_id": rebalance_id,
        "order_count": len(proposed_orders),
        "skipped_count": len(skipped_orders),
    }


def _build_broker_context(
    *,
    config: dict[str, Any],
    repo_root: Path,
    current_holdings: pd.Series,
    reference_prices: dict[str, float],
    notional: float,
) -> tuple[Any, dict[str, Any]]:
    broker_name = str(config.get("broker") or "mock").strip().lower()
    if broker_name == "mock":
        client, state = _build_mock_broker_context(
            config=config,
            current_holdings=current_holdings,
            reference_prices=reference_prices,
            notional=notional,
        )
        return client, state

    broker_client = create_broker_client(config)
    if broker_client is None:
        raise RuntimeError("Unable to create configured broker client.")
    if hasattr(broker_client, "get_broker_snapshot"):
        snapshot = broker_client.get_broker_snapshot(as_of_date=pd.Timestamp.utcnow().date())
        return broker_client, snapshot

    balances = broker_client.get_balances()
    positions = broker_client.get_positions()
    return broker_client, {
        "positions": {
            str(position["ticker"]).strip().upper(): {
                "quantity": float(position.get("quantity") or 0.0),
                "market_value": float(position.get("market_value") or 0.0),
                "current_weight": float(position.get("current_weight") or 0.0),
                "cost_basis": float(position.get("cost_basis") or 0.0),
            }
            for position in positions
        },
        "cash": float(balances.get("cash", balances.get("total_cash", 0.0)) or 0.0),
        "equity": float(balances.get("equity", balances.get("total_equity", notional)) or notional),
    }


def _build_mock_broker_context(
    *,
    config: dict[str, Any],
    current_holdings: pd.Series,
    reference_prices: dict[str, float],
    notional: float,
) -> tuple[MockBrokerClient, dict[str, Any]]:
    positions = []
    invested_value = 0.0
    for ticker, weight in current_holdings[current_holdings.abs() > 1e-12].items():
        price = float(reference_prices.get(ticker, 100.0))
        market_value = float(weight) * notional
        quantity = market_value / price if price > 0 else 0.0
        invested_value += market_value
        positions.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "cost_basis": market_value,
                "market_value": market_value,
                "current_weight": float(weight),
                "unrealized_pnl": 0.0,
            }
        )
    cash = max(notional - invested_value, 0.0)
    balances = {
        "account_id": str(config.get("account_id") or "PAPER-12345"),
        "account_type": "paper",
        "cash": cash,
        "buying_power": max(notional, cash),
        "equity": notional,
        "market_value": invested_value,
        "total_equity": notional,
        "maintenance_requirement": 0.0,
    }
    client = MockBrokerClient(
        profile={
            "account_id": balances["account_id"],
            "account_type": "paper",
            "status": "active",
            "user_name": "Phase A Mock",
            "email": None,
        },
        balances=balances,
        positions=positions,
    )
    state = {
        "positions": {
            item["ticker"]: {
                "quantity": item["quantity"],
                "market_value": item["market_value"],
                "current_weight": item["current_weight"],
                "cost_basis": item["cost_basis"],
                "price": reference_prices.get(item["ticker"]),
            }
            for item in positions
        },
        "cash": cash,
        "equity": notional,
    }
    return client, state


def _build_target_payload(
    target_weights: pd.Series,
    reference_prices: dict[str, float],
) -> dict[str, Any]:
    positions = {}
    for ticker, weight in target_weights[target_weights.abs() > 1e-12].items():
        payload: dict[str, Any] = {"target_weight": float(weight)}
        price = reference_prices.get(ticker)
        if price is not None and price > 0:
            payload["est_price"] = float(price)
        positions[str(ticker).strip().upper()] = payload
    return {"positions": positions}


def _build_previous_payload(
    prior_weights: pd.Series,
    reference_prices: dict[str, float],
) -> dict[str, Any]:
    positions = {}
    for ticker, weight in prior_weights[prior_weights.abs() > 1e-12].items():
        payload: dict[str, Any] = {"weight": float(weight)}
        price = reference_prices.get(ticker)
        if price is not None and price > 0:
            payload["price"] = float(price)
        positions[str(ticker).strip().upper()] = payload
    return positions


def _reference_prices_for_phase_a(
    metadata: dict[str, Any],
    target_weights: pd.Series,
    prior_weights: pd.Series,
    *,
    notional: float,
) -> dict[str, float]:
    if metadata.get("reference_prices"):
        return {
            str(ticker).strip().upper(): float(price)
            for ticker, price in dict(metadata["reference_prices"]).items()
            if price is not None and float(price) > 0
        }

    prices: dict[str, float] = {}
    for series in (target_weights, prior_weights):
        for ticker, weight in series[series.abs() > 1e-12].items():
            market_value = abs(float(weight) * notional)
            if market_value > 0:
                prices[str(ticker).strip().upper()] = 100.0
    return prices


def _current_holdings_from_broker_state(broker_state: Any) -> dict[str, float]:
    if hasattr(broker_state, "positions"):
        return {
            str(symbol).strip().upper(): float(position.current_weight)
            for symbol, position in broker_state.positions.items()
            if abs(float(position.current_weight)) > 1e-12
        }
    if isinstance(broker_state, dict):
        positions = broker_state.get("positions") or {}
        return {
            str(symbol).strip().upper(): float(payload.get("current_weight", payload.get("weight", 0.0)))
            for symbol, payload in positions.items()
            if abs(float(payload.get("current_weight", payload.get("weight", 0.0)))) > 1e-12
        }
    return {}


def _series_weight_map(series: pd.Series) -> dict[str, float]:
    return {
        str(ticker).strip().upper(): float(weight)
        for ticker, weight in series[series.abs() > 1e-12].items()
    }


def _policy_order_type(config: dict[str, Any]) -> str:
    allowed = list(config.get("allowed_order_types") or ["market"])
    first = str(allowed[0]).strip().lower()
    if first in {"market", "moo", "market_on_open"}:
        return "MARKET"
    if first in {"limit", "loo", "limit_on_open"}:
        return "LIMIT"
    raise ValueError(f"Unsupported allowed_order_types entry for Phase A policy: {first!r}")


def _filtered_preview_batch(
    preview_batch: PreviewBatch | None,
    results: list[Any],
) -> PreviewBatch | None:
    if preview_batch is None:
        return None
    if not results:
        return PreviewBatch(
            generated_at=preview_batch.generated_at,
            results=(),
            totals=PreviewBatchTotals(order_count=0),
            warnings=preview_batch.warnings,
            errors=preview_batch.errors,
        )
    return PreviewBatch(
        generated_at=preview_batch.generated_at,
        results=tuple(results),
        totals=PreviewBatchTotals(
            total_notional=sum(item.estimated_notional for item in results),
            commission=sum(item.estimated_commission for item in results),
            cash_needed=sum(max(item.cash_required, 0.0) for item in results),
            order_count=len(results),
            cache_hits=sum(1 for item in results if item.cache_hit),
            warning_count=sum(len(item.warnings) for item in results),
            error_count=sum(len(item.errors) for item in results),
        ),
        warnings=tuple(dict.fromkeys(msg for item in results for msg in item.warnings)),
        errors=(),
    )


def _approval_deadline(generated_at: datetime, config: dict[str, Any]) -> datetime:
    hour_str, minute_str = str(config.get("approval_deadline_time_et") or "08:00").split(":", maxsplit=1)
    local_now = generated_at.astimezone(EASTERN_TZ)
    deadline_date = local_now.date() + timedelta(days=1)
    deadline_local = datetime.combine(
        deadline_date,
        time(hour=int(hour_str), minute=int(minute_str)),
        tzinfo=EASTERN_TZ,
    )
    return deadline_local.astimezone(timezone.utc)


def _intent_hash(order_spec: IntentOrderSpec) -> str:
    payload = json.dumps(
        order_spec.cache_key_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _signal_hash(target_weights: pd.Series, metadata: dict[str, Any]) -> str:
    payload = {
        "as_of_date": metadata.get("as_of_date"),
        "weights": _series_weight_map(target_weights),
        "weight_source": metadata.get("weight_source"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_phase_a_bundle(
    *,
    repo_root: Path,
    config: dict[str, Any],
    bundle: IntentBundle,
    target_weights: pd.Series,
    prior_weights: pd.Series,
    intended_trades: pd.DataFrame,
    skipped_orders: list[dict[str, Any]],
) -> Path:
    results_root = Path(str(config.get("results_dir") or "results"))
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    bundle_dir = results_root / f"_paper_phase_a_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    target_frame = (
        pd.DataFrame({"ticker": target_weights.index, "weight": target_weights.values})
        .query("weight != 0")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    prior_frame = (
        pd.DataFrame({"ticker": prior_weights.index, "weight": prior_weights.values})
        .query("weight != 0")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    target_path = bundle_dir / "target_weights.csv"
    prior_path = bundle_dir / "prior_weights.csv"
    trades_path = bundle_dir / "intended_trades.csv"
    target_frame.to_csv(target_path, index=False)
    prior_frame.to_csv(prior_path, index=False)
    intended_trades.to_csv(trades_path, index=False)

    manifest_payload = {
        "bundle_type": "paper_phase_a",
        "generated_at": bundle.generated_at.isoformat().replace("+00:00", "Z"),
        "config_hash": bundle.config_hash,
        "rebalance_id": bundle.rebalance_id,
        "stats": {
            "target_positions": int(len(target_frame)),
            "prior_positions": int(len(prior_frame)),
            "intended_trade_count": int(len(intended_trades)),
            "order_count": int(len(bundle.proposed_orders)),
            "skipped_count": int(len(skipped_orders)),
        },
        "hashes": {
            "target_weights": _sha256_file(target_path),
            "prior_weights": _sha256_file(prior_path),
            "intended_trades": _sha256_file(trades_path),
        },
    }

    (bundle_dir / "intents.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    (bundle_dir / "approval.template.json").write_text(
        json.dumps(
            {
                "note": "Use `python -m src.paper.approval_cli <bundle_dir>/intents.json --approve --yes --operator <name>` or --reject.",
                "rebalance_id": bundle.rebalance_id,
                "config_hash": bundle.config_hash,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    (bundle_dir / "checksums.sha256").write_text(
        "\n".join(
            [
                f"{manifest_payload['hashes']['target_weights']}  target_weights.csv",
                f"{manifest_payload['hashes']['prior_weights']}  prior_weights.csv",
                f"{manifest_payload['hashes']['intended_trades']}  intended_trades.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _configure_logging(repo_root: Path, log_date: str) -> None:
    log_dir = repo_root / "logs" / "paper_phase_a"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_date}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _load_config(path: str | None) -> dict[str, Any]:
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
        config_path = Path(__file__).resolve().parent / config_path
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


if __name__ == "__main__":
    raise SystemExit(main())
