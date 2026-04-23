"""Stage 3 sequential submission and polling engine."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID, uuid4

from .brokerage.interface import BrokerClient, normalize_equity_order_request
from .order_blotter import OrderBlotter, OrderRecord, OrderSide, OrderStatus
from .preflight_gate import is_kill_switch_active

_SELL_ORDER = {"SELL", "SELL_SHORT"}
_APPROVED_STATUS = "APPROVED"
_TERMINAL_BROKER_STATUSES = {"filled", "canceled", "cancelled", "rejected", "expired", "error"}
_DEFAULT_KILL_SWITCH_PATH = Path("paper_trading/state/KILL_SWITCH")
_PREVIEW_FAILURE_STATUSES = {"error", "errors", "failed", "failure", "rejected", "invalid"}


@dataclass(frozen=True)
class SubmissionResult:
    """Aggregate result for one sequential submission batch."""

    submitted: int = 0
    filled: int = 0
    partial: int = 0
    rejected: int = 0
    halted: int = 0
    cancelled: int = 0
    order_ids: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


class SubmissionEngine:
    """Submit approved orders sequentially and persist every lifecycle step."""

    def __init__(
        self,
        broker_client: BrokerClient,
        *,
        order_blotter: OrderBlotter | None = None,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        kill_switch_path: str | Path | None = None,
    ) -> None:
        self.broker_client = broker_client
        self.order_blotter = order_blotter or OrderBlotter()
        self.timeout_seconds = float(timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.kill_switch_path = self._resolve_kill_switch_path(kill_switch_path)

    def submit_and_poll(
        self,
        order_specs: Iterable[dict[str, Any]],
        preview_batch: Any,
        approval_record: dict[str, Any],
    ) -> SubmissionResult:
        """Submit orders one by one, sell side first, and poll to terminal or halt."""
        status = str(approval_record.get("status", "")).strip().upper()
        if status != _APPROVED_STATUS:
            raise PermissionError(
                "Submission blocked: approval_record.status must equal APPROVED."
            )

        result = SubmissionResult()
        ordered_specs = sorted(
            [self._normalize_spec(spec) for spec in order_specs],
            key=self._sort_key,
        )
        preview_results = [
            self._require_valid_preview(
                self._resolve_preview(preview_batch, spec, index),
                spec=spec,
            )
            for index, spec in enumerate(ordered_specs)
        ]

        for spec, preview_result in zip(ordered_specs, preview_results):
            self._ensure_kill_switch_clear(stage="before_prepare")
            order_row = self._create_blotter_order(
                spec=spec,
                approval_record=approval_record,
                preview_result=preview_result,
            )
            current_preview_result = preview_result

            result.order_ids.append(str(order_row.order_id))
            self._ensure_kill_switch_clear(
                stage="before_submit",
                order_row=order_row,
                preview_result=current_preview_result,
            )

            try:
                self.order_blotter.update_status(
                    order_row.order_id,
                    OrderStatus.SUBMITTED,
                )
                submission_payload = self.broker_client.place_equity_order(
                    spec["symbol"],
                    spec["qty"],
                    spec["side"].value,
                    spec["order_type"],
                    limit_price=spec.get("limit_price"),
                    preview_result=current_preview_result,
                )
                broker_order_id = self._extract_broker_order_id(submission_payload)
                current_preview_result = self._merge_payload(
                    current_preview_result,
                    {"submission_response": submission_payload},
                )
                self.order_blotter.update_order(
                    order_row.order_id,
                    broker_order_id=broker_order_id,
                    preview_result=current_preview_result,
                )
                result = self._replace_result(
                    result,
                    submitted=result.submitted + 1,
                )
            except Exception as exc:
                self.order_blotter.update_status(
                    order_row.order_id,
                    OrderStatus.REJECTED,
                    updates={
                        "preview_result": self._merge_payload(
                            current_preview_result,
                            {"submission_error": str(exc)},
                        )
                    },
                )
                result = self._replace_result(
                    result,
                    rejected=result.rejected + 1,
                    failures=result.failures
                    + [
                        {
                            "order_id": str(order_row.order_id),
                            "symbol": spec["symbol"],
                            "stage": "submit",
                            "error": str(exc),
                        }
                    ],
                )
                continue

            self._ensure_kill_switch_clear(
                stage="before_poll",
                order_row=order_row,
                broker_order_id=broker_order_id,
                preview_result=current_preview_result,
                cancel_live_order=True,
            )
            try:
                poll_payload = self.broker_client.poll_until_terminal(
                    broker_order_id,
                    timeout_seconds=self.timeout_seconds,
                    poll_interval_seconds=self.poll_interval_seconds,
                )
            except Exception as exc:
                self._cancel_and_raise(
                    order_row=order_row,
                    broker_order_id=broker_order_id,
                    preview_result=current_preview_result,
                    halt_payload={"poll_error": str(exc)},
                    message=(
                        f"Polling failed after submission for order {order_row.order_id}; "
                        "cancel attempted and batch halted."
                    ),
                    cause=exc,
                )

            current_preview_result = self._merge_payload(
                current_preview_result,
                {
                    "broker_order_id": broker_order_id,
                    "terminal_poll": poll_payload,
                },
            )
            transition_status, bucket = self._classify_terminal_state(
                spec=spec,
                payload=poll_payload,
            )
            updates = {
                "preview_result": current_preview_result,
                "broker_order_id": broker_order_id,
            }

            if transition_status is None:
                self.order_blotter.update_order(order_row.order_id, **updates)
                result = self._replace_result(
                    result,
                    halted=result.halted + 1,
                    failures=result.failures
                    + [
                        {
                            "order_id": str(order_row.order_id),
                            "symbol": spec["symbol"],
                            "stage": "terminal",
                            "error": f"Non-terminal broker state: {poll_payload}",
                        }
                    ],
                )
                continue

            self.order_blotter.update_status(
                order_row.order_id,
                transition_status,
                updates=updates,
            )
            result = self._increment_bucket(result, bucket)
            self._ensure_kill_switch_clear(
                stage="after_poll",
                order_row=order_row,
                broker_order_id=broker_order_id,
                preview_result=current_preview_result,
            )

        return result

    def _create_blotter_order(
        self,
        *,
        spec: dict[str, Any],
        approval_record: dict[str, Any],
        preview_result: dict[str, Any] | None,
    ) -> OrderRecord:
        existing_order_id = spec.get("order_id")
        if existing_order_id:
            current = self.order_blotter.get_order(existing_order_id, required=True)
            assert current is not None
            updates = {
                "preview_result": preview_result,
                "approval_record": dict(approval_record),
            }
            if current.status == OrderStatus.APPROVAL_PENDING:
                return self.order_blotter.update_status(
                    current.order_id,
                    OrderStatus.APPROVED,
                    updates=updates,
                )
            return self.order_blotter.update_order(current.order_id, **updates)

        rebalance_id = spec.get("rebalance_id") or approval_record.get("rebalance_id") or uuid4()
        return self.order_blotter.create_order(
            rebalance_id=rebalance_id,
            symbol=spec["symbol"],
            side=spec["side"],
            qty=spec["qty"],
            order_type=spec["order_type"],
            duration=str(spec.get("duration", "day")),
            limit_price=spec.get("limit_price"),
            status=OrderStatus.APPROVED,
            parent_intent_hash=spec["parent_intent_hash"],
            preview_result=preview_result,
            approval_record=dict(approval_record),
        )

    def _normalize_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_equity_order_request(
            spec["symbol"],
            spec["qty"],
            spec["side"],
            spec["order_type"],
            limit_price=spec.get("limit_price"),
        )
        side = normalized["side"].upper()
        return {
            "symbol": normalized["symbol"],
            "qty": normalized["qty"],
            "side": OrderSide(side if side in OrderSide._value2member_map_ else side.upper()),
            "order_type": normalized["order_type"],
            "limit_price": normalized.get("limit_price"),
            "duration": normalized.get("duration", "day"),
            "parent_intent_hash": str(
                spec.get("parent_intent_hash")
                or spec.get("intent_hash")
                or f"{normalized['symbol']}:{normalized['side']}:{normalized['qty']}"
            ),
            "rebalance_id": spec.get("rebalance_id"),
            "order_id": spec.get("order_id"),
        }

    def _sort_key(self, spec: dict[str, Any]) -> tuple[int, str]:
        priority = 0 if spec["side"].value in _SELL_ORDER else 1
        return (priority, spec["symbol"])

    def _resolve_preview(
        self,
        preview_batch: Any,
        spec: dict[str, Any],
        index: int,
    ) -> dict[str, Any] | None:
        if preview_batch is None:
            return None
        if isinstance(preview_batch, list):
            if index < len(preview_batch):
                preview = preview_batch[index]
                return dict(preview) if isinstance(preview, dict) else None
            return None
        if isinstance(preview_batch, dict):
            for key in (
                spec["parent_intent_hash"],
                spec["symbol"],
                f"{spec['symbol']}:{spec['side'].value}",
                str(index),
            ):
                if key in preview_batch and isinstance(preview_batch[key], dict):
                    return dict(preview_batch[key])
        return None

    def _require_valid_preview(
        self,
        preview_result: dict[str, Any] | None,
        *,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(preview_result, dict):
            raise ValueError(
                "Submission blocked: missing preview coverage for "
                f"{spec['symbol']} {spec['side'].value} {spec['qty']}."
            )

        preview_errors = self._collect_preview_errors(preview_result)
        if preview_errors:
            raise ValueError(
                "Submission blocked: preview contains errors for "
                f"{spec['symbol']}: {'; '.join(preview_errors)}"
            )

        if self._preview_indicates_failure(preview_result):
            raise ValueError(
                "Submission blocked: preview does not indicate success for "
                f"{spec['symbol']}."
            )

        preview_request = self._extract_preview_request(preview_result)
        if preview_request is None:
            raise ValueError(
                "Submission blocked: preview coverage is incomplete for "
                f"{spec['symbol']} {spec['side'].value} {spec['qty']}."
            )

        expected_request = normalize_equity_order_request(
            spec["symbol"],
            spec["qty"],
            spec["side"].value,
            spec["order_type"],
            limit_price=spec.get("limit_price"),
        )
        for key in ("symbol", "qty", "side", "order_type", "limit_price"):
            if preview_request[key] != expected_request[key]:
                raise ValueError(
                    "Submission blocked: preview does not match order request for "
                    f"{spec['symbol']} {key}: "
                    f"{preview_request[key]!r} != {expected_request[key]!r}"
                )
        return preview_result

    def _extract_preview_request(
        self,
        preview_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        symbol = self._preview_field(preview_result, "symbol")
        qty = self._preview_field(preview_result, "quantity", "qty")
        side = self._preview_field(preview_result, "side")
        order_type = self._preview_field(preview_result, "type", "order_type")
        limit_price = self._preview_field(preview_result, "limit_price")
        if symbol in (None, "") or qty in (None, "") or side in (None, "") or order_type in (None, ""):
            return None
        return normalize_equity_order_request(
            symbol,
            qty,
            side,
            order_type,
            limit_price=limit_price,
        )

    def _preview_field(
        self,
        preview_result: dict[str, Any],
        *keys: str,
    ) -> Any:
        for payload in self._preview_payload_sources(preview_result):
            for key in keys:
                if key in payload:
                    value = payload.get(key)
                    if value is not None:
                        return value
        return None

    def _preview_payload_sources(
        self,
        preview_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sources = [preview_result]
        order_payload = preview_result.get("order")
        if isinstance(order_payload, dict):
            sources.append(order_payload)
        raw_payload = preview_result.get("raw_payload")
        if isinstance(raw_payload, dict):
            sources.append(raw_payload)
        return sources

    def _collect_preview_errors(
        self,
        preview_result: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        for payload in self._preview_payload_sources(preview_result):
            for key in ("errors", "error", "fault"):
                errors.extend(self._flatten_preview_messages(payload.get(key)))
        deduped: list[str] = []
        seen: set[str] = set()
        for error in errors:
            if error in seen:
                continue
            seen.add(error)
            deduped.append(error)
        return deduped

    def _flatten_preview_messages(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, dict):
            messages: list[str] = []
            for item in value.values():
                messages.extend(self._flatten_preview_messages(item))
            return messages
        if isinstance(value, (list, tuple, set)):
            messages: list[str] = []
            for item in value:
                messages.extend(self._flatten_preview_messages(item))
            return messages
        text = str(value).strip()
        return [text] if text else []

    def _preview_indicates_failure(
        self,
        preview_result: dict[str, Any],
    ) -> bool:
        result_value = self._preview_field(preview_result, "result")
        if result_value is not None and not self._coerce_preview_success(result_value):
            return True

        for payload in self._preview_payload_sources(preview_result):
            for key in ("broker_status", "status", "outcome"):
                status = payload.get(key)
                if status is None:
                    continue
                if str(status).strip().lower() in _PREVIEW_FAILURE_STATUSES:
                    return True
        return False

    def _coerce_preview_success(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0
        text = str(value).strip().lower()
        if not text:
            return False
        return text not in _PREVIEW_FAILURE_STATUSES and text not in {"false", "0", "no", "off"}

    def _extract_broker_order_id(self, payload: dict[str, Any]) -> str:
        for key in ("broker_order_id", "order_id", "id"):
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
        raise ValueError(f"Broker submission payload missing order id: {payload}")

    def _classify_terminal_state(
        self,
        *,
        spec: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[OrderStatus | None, str | None]:
        raw_status = str(payload.get("status", "")).strip().lower()
        qty_filled = self._coerce_float(
            payload.get("qty_filled", payload.get("filled_qty", payload.get("filled_quantity")))
        )
        qty_requested = self._coerce_float(
            payload.get("qty_requested", payload.get("qty", spec["qty"]))
        )

        if raw_status not in _TERMINAL_BROKER_STATUSES:
            if qty_filled > 0.0:
                return OrderStatus.PARTIALLY_FILLED, "partial"
            return None, None

        if qty_requested > 0.0 and 0.0 < qty_filled < qty_requested:
            return OrderStatus.PARTIALLY_FILLED, "partial"

        if raw_status == "filled":
            return OrderStatus.FILLED, "filled"
        if raw_status in {"canceled", "cancelled", "expired"}:
            return OrderStatus.CANCELLED, "cancelled"
        return OrderStatus.REJECTED, "rejected"

    def _increment_bucket(self, result: SubmissionResult, bucket: str | None) -> SubmissionResult:
        if bucket == "filled":
            return self._replace_result(result, filled=result.filled + 1)
        if bucket == "partial":
            return self._replace_result(result, partial=result.partial + 1)
        if bucket == "cancelled":
            return self._replace_result(
                result,
                cancelled=result.cancelled + 1,
                rejected=result.rejected + 1,
            )
        if bucket == "rejected":
            return self._replace_result(result, rejected=result.rejected + 1)
        return result

    def _replace_result(self, result: SubmissionResult, **updates: Any) -> SubmissionResult:
        payload = {
            "submitted": result.submitted,
            "filled": result.filled,
            "partial": result.partial,
            "rejected": result.rejected,
            "halted": result.halted,
            "cancelled": result.cancelled,
            "order_ids": list(result.order_ids),
            "failures": list(result.failures),
        }
        payload.update(updates)
        return SubmissionResult(**payload)

    def _resolve_kill_switch_path(self, kill_switch_path: str | Path | None) -> Path:
        if kill_switch_path is not None:
            return Path(kill_switch_path)

        blotter_path = self.order_blotter.path
        blotter_dir = blotter_path.parent
        if blotter_dir.name == "blotter" and blotter_dir.parent.name == "paper_trading":
            return blotter_dir.parent / "state" / "KILL_SWITCH"
        return _DEFAULT_KILL_SWITCH_PATH

    def _ensure_kill_switch_clear(
        self,
        *,
        stage: str,
        order_row: OrderRecord | None = None,
        broker_order_id: str | None = None,
        preview_result: dict[str, Any] | None = None,
        cancel_live_order: bool = False,
    ) -> None:
        if not is_kill_switch_active(self.kill_switch_path):
            return

        halt_payload = {
            "kill_switch_halt": {
                "stage": stage,
                "path": str(self.kill_switch_path),
            }
        }
        if order_row is not None:
            if cancel_live_order and broker_order_id is not None:
                self._cancel_and_raise(
                    order_row=order_row,
                    broker_order_id=broker_order_id,
                    preview_result=preview_result,
                    halt_payload=halt_payload,
                    message=f"Kill switch engaged during submission: {self.kill_switch_path}",
                )
            self.order_blotter.update_order(
                order_row.order_id,
                broker_order_id=broker_order_id,
                preview_result=self._merge_payload(preview_result, halt_payload),
            )
        raise RuntimeError(f"Kill switch engaged during submission: {self.kill_switch_path}")

    def _cancel_and_raise(
        self,
        *,
        order_row: OrderRecord,
        broker_order_id: str,
        preview_result: dict[str, Any] | None,
        halt_payload: dict[str, Any],
        message: str,
        cause: Exception | None = None,
    ) -> None:
        cancellation_attempt: dict[str, Any] = {
            "attempted": True,
            "broker_order_id": broker_order_id,
            "timestamp": self._utc_now_iso(),
        }
        transition_status: OrderStatus | None = None

        try:
            cancel_payload = self.broker_client.cancel_order(broker_order_id)
            cancellation_attempt["response"] = cancel_payload
            if self._is_cancelled_payload(cancel_payload):
                transition_status = OrderStatus.CANCELLED
        except Exception as cancel_exc:
            cancellation_attempt["error"] = str(cancel_exc)

        updates = {
            "broker_order_id": broker_order_id,
            "preview_result": self._merge_payload(
                preview_result,
                {
                    "broker_order_id": broker_order_id,
                    **halt_payload,
                    "cancellation_attempt": cancellation_attempt,
                },
            ),
        }
        if transition_status is OrderStatus.CANCELLED:
            self.order_blotter.update_status(
                order_row.order_id,
                transition_status,
                updates=updates,
            )
        else:
            self.order_blotter.update_order(order_row.order_id, **updates)

        if cause is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from cause

    def _is_cancelled_payload(self, payload: dict[str, Any]) -> bool:
        status = str(payload.get("status", "")).strip().lower()
        return status in {"canceled", "cancelled", "expired"}

    def _merge_payload(
        self,
        preview_result: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if isinstance(preview_result, dict):
            merged.update(preview_result)
        merged.update(extra)
        return merged

    def _coerce_float(self, value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["SubmissionEngine", "SubmissionResult"]
