"""Order-preview batching with short-lived broker preview caching."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .brokerage.interface import BrokerClient
from .models import OrderSpec, PreviewBatch, PreviewBatchTotals, PreviewOrderResult


class PreviewEngineError(RuntimeError):
    """Base exception for preview-engine failures."""


class PreviewBatchError(PreviewEngineError):
    """Raised when a preview batch contains one or more blocking errors."""

    def __init__(self, preview_batch: PreviewBatch) -> None:
        self.preview_batch = preview_batch
        message = (
            "Preview batch contains blocking errors for "
            f"{preview_batch.totals.error_count} order(s)."
        )
        super().__init__(message)


@dataclass(slots=True)
class _CachedPreviewEntry:
    preview: PreviewOrderResult
    cached_at: float


class PreviewEngine:
    """Batch preview order specs through the broker with a short TTL cache."""

    def __init__(
        self,
        broker_client: BrokerClient,
        *,
        ttl_seconds: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self.broker_client = broker_client
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock or time.monotonic
        self._cache: dict[str, _CachedPreviewEntry] = {}

    def preview_batch(
        self,
        order_specs: Iterable[OrderSpec | dict[str, Any]],
        *,
        abort_on_error: bool = True,
    ) -> PreviewBatch:
        """Preview a batch of orders and optionally abort on any blocking error."""
        now = self._clock()
        self._purge_expired(now)

        normalized_specs = [
            spec if isinstance(spec, OrderSpec) else OrderSpec.model_validate(spec)
            for spec in order_specs
        ]

        results: list[PreviewOrderResult] = []
        warnings: list[str] = []
        errors: list[str] = []
        total_notional = 0.0
        total_commission = 0.0
        total_cash_needed = 0.0
        cache_hits = 0

        for order_spec in normalized_specs:
            preview = self._get_or_create_preview(order_spec, now)
            results.append(preview)
            total_notional += abs(preview.estimated_notional)
            total_commission += preview.estimated_commission
            total_cash_needed += max(preview.cash_required, 0.0)
            warnings.extend(preview.warnings)
            errors.extend(preview.errors)
            if preview.cache_hit:
                cache_hits += 1

        batch = PreviewBatch(
            generated_at=datetime.now(timezone.utc),
            results=tuple(results),
            totals=PreviewBatchTotals(
                total_notional=total_notional,
                commission=total_commission,
                cash_needed=total_cash_needed,
                order_count=len(results),
                cache_hits=cache_hits,
                warning_count=sum(len(result.warnings) for result in results),
                error_count=sum(len(result.errors) for result in results),
            ),
            warnings=tuple(_unique_preserve_order(warnings)),
            errors=tuple(_unique_preserve_order(errors)),
        )

        if abort_on_error and batch.has_errors:
            raise PreviewBatchError(batch)
        return batch

    def invalidate(self, order_spec: OrderSpec | dict[str, Any] | None = None) -> None:
        """Clear either one cached preview or the entire cache."""
        if order_spec is None:
            self._cache.clear()
            return

        spec = order_spec if isinstance(order_spec, OrderSpec) else OrderSpec.model_validate(order_spec)
        self._cache.pop(self._cache_key(spec), None)

    def _purge_expired(self, now: float) -> None:
        expired_keys = [
            cache_key
            for cache_key, entry in self._cache.items()
            if (now - entry.cached_at) >= self.ttl_seconds
        ]
        for cache_key in expired_keys:
            self._cache.pop(cache_key, None)

    def _get_or_create_preview(
        self,
        order_spec: OrderSpec,
        now: float,
    ) -> PreviewOrderResult:
        cache_key = self._cache_key(order_spec)
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached.cached_at) < self.ttl_seconds:
            return cached.preview.model_copy(update={"cache_hit": True})

        raw_payload = self.broker_client.preview_equity_order(
            symbol=order_spec.symbol,
            qty=order_spec.qty,
            side=order_spec.side.value,
            order_type=order_spec.order_type,
            limit_price=order_spec.limit_price,
        )
        preview = _normalize_preview_payload(order_spec, raw_payload)
        self._cache[cache_key] = _CachedPreviewEntry(preview=preview, cached_at=now)
        return preview

    def _cache_key(self, order_spec: OrderSpec) -> str:
        payload = json.dumps(
            order_spec.cache_key_payload(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _normalize_preview_payload(
    order_spec: OrderSpec,
    raw_payload: dict[str, Any] | None,
) -> PreviewOrderResult:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    warnings = _extract_messages(payload, primary_keys=("warnings", "warning"))
    errors = _extract_messages(
        payload,
        primary_keys=("errors", "error", "fault", "message"),
        treat_status_as_error=True,
    )

    estimated_notional = _extract_float(
        payload,
        "estimated_notional",
        "notional",
        "order_cost",
        "estimated_cost",
        "amount",
    )
    estimated_commission = _extract_float(
        payload,
        "estimated_commission",
        "commission",
        "commission_cost",
    )
    cash_required = _extract_float(
        payload,
        "cash_required",
        "required_cash",
        "cash_needed",
        "required_margin",
        default=None,
    )

    if estimated_notional is None:
        if order_spec.limit_price is not None:
            estimated_notional = order_spec.qty * order_spec.limit_price
        else:
            estimated_notional = 0.0

    if cash_required is None:
        if order_spec.side.value == "BUY":
            cash_required = estimated_notional + estimated_commission
        else:
            cash_required = estimated_commission

    preview_id = _extract_text(payload, "preview_id", "id", "previewId")
    broker_status = _extract_text(payload, "status", "result", "outcome")

    return PreviewOrderResult(
        order=order_spec,
        preview_id=preview_id,
        broker_status=broker_status,
        estimated_notional=float(estimated_notional or 0.0),
        estimated_commission=float(estimated_commission or 0.0),
        cash_required=float(cash_required or 0.0),
        warnings=tuple(warnings),
        errors=tuple(errors),
        raw_payload=_json_safe_dict(payload),
        cache_hit=False,
    )


def _extract_messages(
    payload: dict[str, Any],
    *,
    primary_keys: tuple[str, ...],
    treat_status_as_error: bool = False,
) -> list[str]:
    extracted: list[str] = []
    for key in primary_keys:
        if key not in payload:
            continue
        extracted.extend(_flatten_messages(payload.get(key)))

    if treat_status_as_error:
        status = _extract_text(payload, "status", "result", "outcome")
        if status and status.strip().lower() in {"error", "rejected", "failed", "invalid"}:
            detail = _extract_text(payload, "message", "reason", "detail")
            extracted.append(detail or f"Broker preview returned status={status}.")

    return _unique_preserve_order(extracted)


def _flatten_messages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        messages: list[str] = []
        for item in value.values():
            messages.extend(_flatten_messages(item))
        return messages
    if isinstance(value, (list, tuple, set)):
        messages = []
        for item in value:
            messages.extend(_flatten_messages(item))
        return messages
    text = str(value).strip()
    return [text] if text else []


def _extract_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_float(
    payload: dict[str, Any],
    *keys: str,
    default: float | None = 0.0,
) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
    return default


def _json_safe_dict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
        )
    except TypeError:
        return {"raw": str(payload)}


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
