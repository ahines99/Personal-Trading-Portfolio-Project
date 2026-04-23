from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import requests

from .interface import (
    NON_TERMINAL_ORDER_STATUSES,
    TERMINAL_ORDER_STATUSES,
    BrokerClient,
    normalize_equity_order_request,
)
from ..schemas import BrokerPosition, BrokerPositionsSnapshot, normalize_ticker

TRADIER_BASE_URLS = {
    "tradier_live": "https://api.tradier.com/v1",
    "tradier_sandbox": "https://sandbox.tradier.com/v1",
}


class TradierBrokerError(RuntimeError):
    """Raised for broker connectivity or response-shape errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        response_payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.response_payload = response_payload


class TradierAuthenticationError(TradierBrokerError):
    """Raised for 401/403 responses."""


class TradierRateLimitError(TradierBrokerError):
    """Raised for Tradier 429 responses."""


class TradierTransientError(TradierBrokerError):
    """Raised for retryable 5xx/transport failures."""


class TradierTransportError(TradierTransientError):
    """Raised for retryable connection-level request failures."""


class TradierNotFoundError(TradierBrokerError):
    """Raised for 404 responses."""


class TradierValidationError(TradierBrokerError):
    """Raised for non-retryable request validation failures."""


class TradierResponseShapeError(TradierBrokerError):
    """Raised when Tradier returns an unexpected payload shape."""


class TradierDuplicateOrderError(TradierBrokerError):
    """Raised when multiple live orders share the same client tag."""


class TradierBrokerClient(BrokerClient):
    def __init__(
        self,
        *,
        account_id: str,
        token: str,
        broker: str = "tradier_sandbox",
        timeout: float = 15.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        normalized_broker = str(broker).strip().lower()
        if normalized_broker not in TRADIER_BASE_URLS:
            raise ValueError(f"Unsupported Tradier broker backend: {broker!r}")
        if not str(account_id).strip():
            raise ValueError("Tradier account_id must be non-empty")
        if not str(token).strip():
            raise ValueError("Tradier API token must be non-empty")

        self.account_id = str(account_id).strip()
        self.broker = normalized_broker
        self.mode = "sandbox" if normalized_broker.endswith("sandbox") else "live"
        self.base_url = TRADIER_BASE_URLS[normalized_broker]
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {str(token).strip()}",
                "Accept": "application/json",
            }
        )

    def ping(self) -> bool:
        try:
            self._get(f"/accounts/{self.account_id}")
            return True
        except TradierBrokerError:
            return False

    def get_profile(self) -> dict[str, Any]:
        payload = self._get("/user/profile")
        profile = _as_dict(payload.get("profile"))
        account_payload = profile.get("account")
        if account_payload is None:
            accounts_wrapper = _as_dict(profile.get("accounts"))
            account_payload = accounts_wrapper.get("account") or profile.get("accounts")
        accounts = [
            _normalize_account_entry(item)
            for item in _ensure_list(account_payload)
            if _as_dict(item)
        ]
        selected_account = next(
            (
                account
                for account in accounts
                if account.get("account_number") == self.account_id
            ),
            accounts[0] if accounts else {},
        )
        return {
            "account_id": self.account_id,
            "account_type": _coerce_text(selected_account.get("classification")) or "unknown",
            "status": _coerce_text(selected_account.get("status")) or "unknown",
            "user_name": _coerce_text(profile.get("name")),
            "email": _coerce_text(profile.get("email")),
            "accounts": accounts,
            "user_id": _coerce_text(profile.get("id")),
        }

    def get_balances(self) -> dict[str, Any]:
        payload = self._get(f"/accounts/{self.account_id}/balances")
        balances = _as_dict(payload.get("balances"))
        margin = _as_dict(balances.get("margin"))
        cash = _as_dict(balances.get("cash"))
        pdt = _as_dict(balances.get("pdt"))
        buying_power = {
            "option_buying_power": _coerce_float(
                margin.get("option_buying_power"), default=None
            ),
            "stock_buying_power": _coerce_float(
                margin.get("stock_buying_power"), default=None
            ),
            "cash_available": _coerce_float(cash.get("cash_available"), default=None),
            "unsettled_funds": _coerce_float(cash.get("unsettled_funds"), default=None),
            "pdt_option_buying_power": _coerce_float(
                pdt.get("option_buying_power"), default=None
            ),
            "pdt_stock_buying_power": _coerce_float(
                pdt.get("stock_buying_power"), default=None
            ),
        }
        return {
            "account_id": _coerce_text(
                balances.get("account_number"), default=self.account_id
            ),
            "account_type": _coerce_text(balances.get("account_type")),
            "cash": _coerce_float(balances.get("total_cash")),
            "buying_power": _coerce_float(
                margin.get("stock_buying_power"),
                default=_coerce_float(cash.get("cash_available")),
            ),
            "equity": _coerce_float(
                balances.get("total_equity"),
                default=_coerce_float(balances.get("market_value")),
            ),
            "total_equity": _coerce_float(balances.get("total_equity")),
            "total_cash": _coerce_float(balances.get("total_cash")),
            "market_value": _coerce_float(balances.get("market_value")),
            "long_market_value": _coerce_float(balances.get("long_market_value")),
            "short_market_value": _coerce_float(balances.get("short_market_value")),
            "stock_long_value": _coerce_float(balances.get("stock_long_value")),
            "stock_short_value": _coerce_float(balances.get("stock_short_value")),
            "option_long_value": _coerce_float(balances.get("option_long_value")),
            "option_short_value": _coerce_float(balances.get("option_short_value")),
            "open_pl": _coerce_float(balances.get("open_pl")),
            "close_pl": _coerce_float(balances.get("close_pl")),
            "maintenance_requirement": _coerce_float(
                balances.get("current_requirement"), default=None
            ),
            "pending_orders_count": _coerce_int(balances.get("pending_orders_count")),
            "buying_power": buying_power,
            "options_buying_power": buying_power.get("option_buying_power"),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        payload = self._get(f"/accounts/{self.account_id}/positions")
        positions_wrapper = _as_dict(payload.get("positions"))
        position_payload = positions_wrapper.get("position")
        balances = self.get_balances()
        equity = _coerce_float(
            balances.get("equity"),
            default=_coerce_float(balances.get("total_equity")),
        )
        return [
            _normalize_position_entry(item, equity=equity)
            for item in _ensure_list(position_payload)
            if _as_dict(item)
        ]

    def preview_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        request = normalize_equity_order_request(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )
        tag = f"pt-{uuid4().hex}"
        payload = self._post(
            f"/accounts/{self.account_id}/orders",
            data=self._build_order_form(request, preview=True, tag=tag),
            retryable=True,
        )
        order = _extract_order_node(payload)
        return _normalize_preview_response(
            order_payload=order,
            request=request,
            tag=tag,
        )

    def place_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: float | None = None,
        preview_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = normalize_equity_order_request(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )
        preview = _validate_preview_result(preview_result, request)
        preview_tag = str(preview["tag"])
        existing_order = self._find_order_by_tag(preview_tag)
        if existing_order is not None:
            return self._build_submission_result(
                order_payload=existing_order,
                request=request,
                preview=preview,
                raw_payload={"order": deepcopy_safe(existing_order)},
                placement_status="recovered_existing",
                recovered_by_tag=True,
            )

        try:
            payload = self._post(
                f"/accounts/{self.account_id}/orders",
                data=self._build_order_form(
                    request,
                    preview=False,
                    tag=preview_tag,
                ),
                retryable=False,
            )
        except (TradierRateLimitError, TradierTransientError) as exc:
            recovered = self._recover_order_by_tag(
                preview_tag,
                request=request,
                preview=preview,
                error=exc,
            )
            if recovered is not None:
                return recovered
            raise

        order = _extract_order_node(payload)
        return self._build_submission_result(
            order_payload=order,
            request=request,
            preview=preview,
            raw_payload=payload,
            placement_status=_coerce_text(order.get("status"), default="unknown"),
            recovered_by_tag=False,
        )

    def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        normalized_order_id = _require_order_id(broker_order_id)
        payload = self._delete(
            f"/accounts/{self.account_id}/orders/{normalized_order_id}",
            retryable=False,
        )
        latest_status: dict[str, Any] | None = None
        try:
            latest_status = self.get_order_status(normalized_order_id)
            if latest_status["status"] in {"pending_cancel", "open", "pending"}:
                latest_status = self.poll_until_terminal(
                    normalized_order_id,
                    timeout_seconds=10.0,
                    poll_interval_seconds=1.0,
                )
        except (TradierBrokerError, TimeoutError):
            latest_status = None

        result = {
            "broker_order_id": normalized_order_id,
            "canceled": bool(
                latest_status and latest_status["status"] == "canceled"
            ),
            "raw": deepcopy_safe(payload),
        }
        if latest_status is not None:
            result.update(latest_status)
            result["latest_status"] = latest_status
        else:
            result["status"] = "pending_cancel"
            result["is_terminal"] = False
        return result

    def replace_order(
        self,
        broker_order_id: str,
        *,
        qty: float | None = None,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        normalized_order_id = _require_order_id(broker_order_id)
        if qty is None and limit_price is None:
            raise ValueError("replace_order requires qty and/or limit_price")

        current_status = self.get_order_status(normalized_order_id)
        if current_status["status"] in TERMINAL_ORDER_STATUSES:
            raise ValueError(
                f"Cannot replace terminal Tradier order {normalized_order_id}: "
                f"{current_status['status']}"
            )

        current_qty = float(current_status["quantity"])
        if qty is not None and abs(float(qty) - current_qty) > 1e-9:
            raise ValueError(
                "Tradier change-order API does not support changing quantity; "
                "cancel and submit a new order instead"
            )

        current_type = str(current_status["type"]).lower()
        if current_type not in {"market", "limit"}:
            raise ValueError(
                f"Tradier replace_order only supports market/limit orders on this "
                f"surface, got {current_type!r}"
            )

        order_type = "limit" if limit_price is not None else current_type
        request = normalize_equity_order_request(
            symbol=str(current_status["symbol"]),
            qty=current_qty,
            side=str(current_status["side"]),
            order_type=order_type,
            limit_price=limit_price
            if limit_price is not None
            else current_status.get("limit_price"),
        )
        payload = self._put(
            f"/accounts/{self.account_id}/orders/{normalized_order_id}",
            data=self._build_replace_form(
                request=request,
                tag=_coerce_text(current_status.get("tag")),
            ),
            retryable=False,
        )

        latest_status = self.get_order_status(normalized_order_id)
        result = {
            "broker_order_id": normalized_order_id,
            "replaced": True,
            "raw": deepcopy_safe(payload),
        }
        result.update(latest_status)
        result["latest_status"] = latest_status
        return result

    def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        normalized_order_id = _require_order_id(broker_order_id)
        payload = self._get(
            f"/accounts/{self.account_id}/orders/{normalized_order_id}",
            params={"includeTags": "true"},
        )
        order = _extract_order_node(payload)
        return _normalize_order_status(order)

    def poll_until_terminal(
        self,
        broker_order_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        normalized_order_id = _require_order_id(broker_order_id)
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        interval = max(0.1, float(poll_interval_seconds))

        while True:
            status = self.get_order_status(normalized_order_id)
            if status["is_terminal"]:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Tradier order {normalized_order_id} did not reach a terminal state "
                    f"within {timeout_seconds:.1f}s; last status={status['status']!r}"
                )
            time.sleep(interval)

    def get_broker_snapshot(
        self,
        *,
        as_of_date: date | None = None,
        balances: dict[str, Any] | None = None,
        positions: list[dict[str, Any]] | None = None,
    ) -> BrokerPositionsSnapshot:
        balances = balances or self.get_balances()
        positions = positions or self.get_positions()
        snapshot_date = as_of_date or datetime.now(timezone.utc).date()

        cash_balance = _coerce_float(
            balances.get("cash"),
            default=_coerce_float(balances.get("total_cash")),
        )
        open_pl = _coerce_float(balances.get("open_pl"))
        raw_total_equity = _coerce_float(
            balances.get("equity"),
            default=_coerce_float(balances.get("total_equity"), default=cash_balance),
        )

        position_models: dict[str, BrokerPosition] = {}
        position_market_total = 0.0
        gross_notional_total = 0.0
        net_notional_total = 0.0

        for item in positions:
            ticker = normalize_ticker(item["ticker"])
            market_value = _coerce_float(item.get("market_value"), default=0.0)
            cost_basis = _coerce_float(item.get("cost_basis"), default=0.0)
            unrealized_pnl = _coerce_float(
                item.get("unrealized_pnl"), default=market_value - cost_basis
            )
            position_market_total += market_value
            gross_notional_total += abs(market_value)
            net_notional_total += market_value
            position_models[ticker] = BrokerPosition(
                ticker=ticker,
                quantity=_coerce_float(item.get("quantity")),
                cost_basis=cost_basis,
                market_value=market_value,
                current_weight=_coerce_float(item.get("current_weight"), default=0.0),
                entry_date=_parse_entry_date(
                    item.get("entry_date") or item.get("date_acquired")
                ),
                unrealized_pnl=unrealized_pnl,
            )

        nav = cash_balance + position_market_total
        if nav <= 0 and raw_total_equity > 0:
            nav = raw_total_equity

        weighted_models: dict[str, BrokerPosition] = {}
        for ticker, position in position_models.items():
            current_weight = position.market_value / nav if nav else 0.0
            weighted_models[ticker] = position.model_copy(
                update={"current_weight": current_weight}
            )

        gross_exposure = gross_notional_total / nav if nav else 0.0
        net_exposure = net_notional_total / nav if nav else 0.0
        starting_capital = nav - open_pl
        total_return_pct = (
            (open_pl / starting_capital) * 100.0
            if abs(starting_capital) > 1e-9
            else 0.0
        )

        return BrokerPositionsSnapshot(
            as_of_date=snapshot_date,
            broker_timestamp=_utc_now_iso(),
            cash_balance=cash_balance,
            equity_value=position_market_total,
            nav=nav,
            total_return_pct=total_return_pct,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            positions=weighted_models,
            snapshot_id=uuid4(),
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_json(
            "GET",
            path,
            params=params,
            retryable=True,
        )

    def _post(
        self,
        path: str,
        *,
        data: dict[str, Any],
        retryable: bool,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            path,
            data=data,
            retryable=retryable,
        )

    def _put(
        self,
        path: str,
        *,
        data: dict[str, Any],
        retryable: bool,
    ) -> dict[str, Any]:
        return self._request_json(
            "PUT",
            path,
            data=data,
            retryable=retryable,
        )

    def _delete(
        self,
        path: str,
        *,
        retryable: bool,
    ) -> dict[str, Any]:
        return self._request_json(
            "DELETE",
            path,
            retryable=retryable,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        retryable: bool,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        method_name = str(method).upper()
        request_fn = getattr(self.session, method_name.lower())
        headers = None
        if method_name in {"POST", "PUT"}:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
        attempt_budget = max(1, self.max_retries + 1)

        for attempt in range(attempt_budget):
            try:
                request_kwargs: dict[str, Any] = {
                    "timeout": self.timeout,
                }
                if params is not None:
                    request_kwargs["params"] = params
                if data is not None:
                    request_kwargs["data"] = data
                if headers is not None:
                    request_kwargs["headers"] = headers
                response = request_fn(url, **request_kwargs)
            except requests.RequestException as exc:
                error = _classify_transport_error(
                    exc,
                    path=path,
                    broker=self.broker,
                    method_name=method_name,
                )
                if retryable and error.retryable and attempt < attempt_budget - 1:
                    time.sleep(_retry_delay_seconds(attempt=attempt, response=None))
                    continue
                raise error from exc

            if response.status_code >= 400:
                error = _classify_http_error(
                    response,
                    path=path,
                    broker=self.broker,
                    method_name=method_name,
                )
                if retryable and error.retryable and attempt < attempt_budget - 1:
                    time.sleep(_retry_delay_seconds(attempt=attempt, response=response))
                    continue
                raise error

            try:
                payload = response.json()
            except ValueError as exc:
                raise TradierResponseShapeError(
                    f"Tradier request returned invalid JSON for {path} [{self.broker}] "
                    f"via {method_name}",
                    error_code="invalid_json",
                    response_payload=response.text,
                ) from exc
            if not isinstance(payload, dict):
                raise TradierResponseShapeError(
                    f"Unexpected Tradier payload type for {path}: {type(payload)!r}"
                )
            return _normalize_tradier_payload(payload)
        raise TradierTransientError(
            f"Tradier request exhausted retries for {path} [{self.broker}] via {method_name}",
            error_code="retries_exhausted",
            retryable=False,
        )

    def _build_order_form(
        self,
        request: dict[str, Any],
        *,
        preview: bool,
        tag: str,
    ) -> dict[str, Any]:
        form: dict[str, Any] = {
            "class": "equity",
            "symbol": request["symbol"],
            "side": request["side"],
            "quantity": _format_quantity(request["qty"]),
            "type": request["order_type"],
            "duration": request["duration"],
            "tag": tag,
        }
        if preview:
            form["preview"] = "true"
        if request["limit_price"] is not None:
            form["price"] = _format_price(request["limit_price"])
        return form

    def _build_replace_form(
        self,
        *,
        request: dict[str, Any],
        tag: str | None,
    ) -> dict[str, Any]:
        form: dict[str, Any] = {
            "type": request["order_type"],
            "duration": request["duration"],
        }
        if request["limit_price"] is not None:
            form["price"] = _format_price(request["limit_price"])
        if tag:
            form["tag"] = tag
        return form

    def _list_orders(self) -> list[dict[str, Any]]:
        payload = self._get(
            f"/accounts/{self.account_id}/orders",
            params={"includeTags": "true"},
        )
        return _extract_order_nodes(payload)

    def _find_order_by_tag(self, tag: str) -> dict[str, Any] | None:
        normalized_tag = str(tag).strip()
        if not normalized_tag:
            return None

        matches = [
            order
            for order in self._list_orders()
            if _coerce_text(order.get("tag")) == normalized_tag
        ]
        if not matches:
            return None
        unique_ids = {
            _coerce_text(match.get("id"))
            for match in matches
            if _coerce_text(match.get("id"))
        }
        if len(unique_ids) > 1:
            raise TradierDuplicateOrderError(
                f"Multiple Tradier orders share client tag {normalized_tag!r}; "
                f"refusing duplicate-unsafe submission recovery.",
                error_code="duplicate_tag_orders",
                response_payload=deepcopy_safe(matches),
            )
        return matches[0]

    def _recover_order_by_tag(
        self,
        tag: str,
        *,
        request: dict[str, Any],
        preview: dict[str, Any],
        error: TradierBrokerError,
    ) -> dict[str, Any] | None:
        try:
            existing_order = self._find_order_by_tag(tag)
        except TradierBrokerError:
            raise error
        if existing_order is None:
            return None
        return self._build_submission_result(
            order_payload=existing_order,
            request=request,
            preview=preview,
            raw_payload={
                "order": deepcopy_safe(existing_order),
                "recovered_after_error": str(error),
            },
            placement_status="recovered_existing",
            recovered_by_tag=True,
        )

    def _build_submission_result(
        self,
        *,
        order_payload: dict[str, Any],
        request: dict[str, Any],
        preview: dict[str, Any],
        raw_payload: dict[str, Any],
        placement_status: str,
        recovered_by_tag: bool,
    ) -> dict[str, Any]:
        broker_order_id = _coerce_text(order_payload.get("id"))
        latest_status: dict[str, Any] | None = None
        if broker_order_id:
            try:
                latest_status = self.get_order_status(broker_order_id)
            except TradierBrokerError:
                latest_status = None
        normalized_order = latest_status or _normalize_order_status(order_payload)
        accepted = normalized_order["status"] not in {"rejected", "error"}

        result = {
            "broker_order_id": broker_order_id,
            "accepted": accepted,
            "placement_status": placement_status,
            "partner_id": _coerce_text(order_payload.get("partner_id")),
            "preview_id": preview["preview_id"],
            "tag": preview["tag"],
            "symbol": request["symbol"],
            "quantity": request["qty"],
            "side": request["side"],
            "type": request["order_type"],
            "duration": request["duration"],
            "limit_price": request["limit_price"],
            "recovered_by_tag": recovered_by_tag,
            "raw": deepcopy_safe(raw_payload),
        }
        result.update(
            {
                "status": normalized_order["status"],
                "is_terminal": normalized_order["is_terminal"],
                "latest_status": normalized_order,
            }
        )
        return result


def _normalize_tradier_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_tradier_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_tradier_payload(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "null":
            return None
        return stripped
    return value


def _extract_order_node(payload: dict[str, Any]) -> dict[str, Any]:
    orders = _extract_order_nodes(payload)
    if len(orders) == 1:
        return orders[0]
    if not orders:
        raise TradierResponseShapeError(
            f"Tradier payload did not include an order node: {payload!r}",
            error_code="missing_order",
            response_payload=deepcopy_safe(payload),
        )
    raise TradierResponseShapeError(
        f"Tradier payload unexpectedly included multiple orders: {payload!r}",
        error_code="multiple_orders",
        response_payload=deepcopy_safe(payload),
    )


def _extract_order_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    if "order" in payload:
        candidates.extend(_ensure_list(payload.get("order")))
    orders = payload.get("orders")
    if isinstance(orders, dict):
        candidates.extend(_ensure_list(orders.get("order")))
    elif orders is not None:
        candidates.extend(_ensure_list(orders))
    return [node for node in (_as_dict(item) for item in candidates) if node]


def _normalize_preview_response(
    *,
    order_payload: dict[str, Any],
    request: dict[str, Any],
    tag: str,
) -> dict[str, Any]:
    result = _coerce_bool(order_payload.get("result"), default=False)
    warnings: list[str] = []
    errors = order_payload.get("errors")
    if errors:
        warnings.append(str(errors))
    return {
        "preview_id": f"tradier-preview-{tag}",
        "status": _coerce_text(order_payload.get("status"), default="unknown"),
        "result": result,
        "symbol": request["symbol"],
        "quantity": request["qty"],
        "side": request["side"],
        "type": request["order_type"],
        "duration": request["duration"],
        "limit_price": request["limit_price"],
        "estimated_cost": _coerce_float(
            order_payload.get("cost"),
            default=_coerce_float(order_payload.get("order_cost"), default=0.0),
        ),
        "estimated_fees": _coerce_float(order_payload.get("fees"), default=0.0),
        "order_cost": _coerce_float(order_payload.get("order_cost"), default=0.0),
        "margin_change": _coerce_float(
            order_payload.get("margin_change"),
            default=0.0,
        ),
        "warnings": warnings,
        "tag": tag,
        "request_timestamp": _coerce_text(
            order_payload.get("request_date"),
            default=_utc_now_iso(),
        ),
        "raw": {"order": deepcopy_safe(order_payload)},
    }


def _normalize_order_status(order_payload: dict[str, Any]) -> dict[str, Any]:
    status = _coerce_text(order_payload.get("status"), default="unknown")
    normalized_status = str(status).strip().lower()
    quantity = _coerce_float(order_payload.get("quantity"), default=0.0)
    filled_quantity = _coerce_float(order_payload.get("exec_quantity"), default=0.0)
    remaining_quantity = _coerce_float(
        order_payload.get("remaining_quantity"),
        default=max(quantity - filled_quantity, 0.0),
    )
    avg_fill_price = _coerce_float(order_payload.get("avg_fill_price"), default=None)
    last_fill_price = _coerce_float(order_payload.get("last_fill_price"), default=None)
    reason = _coerce_text(order_payload.get("reason_description"))
    errors = order_payload.get("errors")
    if errors and not reason:
        reason = str(errors)

    return {
        "broker_order_id": _coerce_text(order_payload.get("id")),
        "status": normalized_status,
        "is_terminal": normalized_status in TERMINAL_ORDER_STATUSES,
        "is_non_terminal": normalized_status in NON_TERMINAL_ORDER_STATUSES,
        "symbol": normalize_ticker(
            _coerce_text(order_payload.get("symbol"), default="UNKNOWN")
        ),
        "quantity": quantity,
        "filled_quantity": filled_quantity,
        "remaining_quantity": remaining_quantity,
        "side": _coerce_text(order_payload.get("side"), default="unknown"),
        "type": _coerce_text(order_payload.get("type"), default="unknown"),
        "duration": _coerce_text(order_payload.get("duration"), default="unknown"),
        "limit_price": _coerce_float(order_payload.get("price"), default=None),
        "avg_fill_price": avg_fill_price,
        "last_fill_price": last_fill_price,
        "last_fill_quantity": _coerce_float(
            order_payload.get("last_fill_quantity"), default=0.0
        ),
        "created_at": _coerce_text(order_payload.get("create_date")),
        "updated_at": _coerce_text(
            order_payload.get("transaction_date"),
            default=_coerce_text(order_payload.get("create_date")),
        ),
        "reason": reason,
        "tag": _coerce_text(order_payload.get("tag")),
        "raw": deepcopy_safe(order_payload),
    }


def _validate_preview_result(
    preview_result: dict[str, Any] | None,
    request: dict[str, Any],
) -> dict[str, Any]:
    if preview_result is None:
        raise ValueError("preview_result is required for place_equity_order")
    preview = dict(preview_result)
    if not _coerce_bool(preview.get("result"), default=False):
        raise ValueError("preview_result must indicate a successful preview")
    if not str(preview.get("preview_id", "")).strip():
        raise ValueError("preview_result must include a non-empty preview_id")
    if not str(preview.get("tag", "")).strip():
        raise ValueError("preview_result must include a non-empty tag")
    for key, preview_key in (
        ("symbol", "symbol"),
        ("qty", "quantity"),
        ("side", "side"),
        ("order_type", "type"),
        ("limit_price", "limit_price"),
    ):
        if preview.get(preview_key) != request[key]:
            raise ValueError(
                f"preview_result does not match order request for {key}: "
                f"{preview.get(preview_key)!r} != {request[key]!r}"
            )
    return preview


def _require_order_id(broker_order_id: str) -> str:
    normalized = str(broker_order_id).strip()
    if not normalized:
        raise ValueError("broker_order_id must be non-empty")
    return normalized


def _format_quantity(value: Any) -> str:
    quantity = _coerce_float(value, default=0.0)
    if abs(quantity - round(quantity)) < 1e-9:
        return str(int(round(quantity)))
    return f"{quantity:.8f}".rstrip("0").rstrip(".")


def _format_price(value: Any) -> str:
    price = _coerce_float(value, default=0.0)
    return f"{price:.8f}".rstrip("0").rstrip(".")


def deepcopy_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: deepcopy_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deepcopy_safe(item) for item in value]
    return value


def _normalize_account_entry(payload: Any) -> dict[str, Any]:
    account = _as_dict(payload)
    return {
        "account_number": _coerce_text(
            account.get("account_number") or account.get("number") or account.get("account")
        ),
        "classification": _coerce_text(
            account.get("classification") or account.get("type")
        ),
        "status": _coerce_text(account.get("status")),
        "nickname": _coerce_text(account.get("nickname")),
    }


def _normalize_position_entry(payload: Any, *, equity: float = 0.0) -> dict[str, Any]:
    position = _as_dict(payload)
    market_value = _coerce_float(
        position.get("market_value"),
        default=_coerce_float(position.get("cost_basis"), default=0.0),
    )
    cost_basis = _coerce_float(position.get("cost_basis"))
    current_weight = market_value / equity if abs(equity) > 1e-9 else 0.0
    return {
        "ticker": normalize_ticker(
            _coerce_text(position.get("symbol"), default="UNKNOWN")
        ),
        "quantity": _coerce_float(position.get("quantity")),
        "cost_basis": cost_basis,
        "market_value": market_value,
        "current_weight": current_weight,
        "unrealized_pnl": market_value - cost_basis,
        "entry_date": _coerce_text(position.get("date_acquired")),
        "id": _coerce_text(position.get("id")),
    }


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _coerce_text(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "null":
        return default
    return text


def _parse_entry_date(value: Any) -> date | None:
    text = _coerce_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _error_detail(response: requests.Response | None) -> str:
    if response is None:
        return "no response"
    try:
        payload = response.json()
    except ValueError:
        payload = response.text.strip()
    return f"{response.status_code} {payload}"


def _safe_json_payload(response: requests.Response) -> Any | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return _normalize_tradier_payload(payload)


def _classify_http_error(
    response: requests.Response,
    *,
    path: str,
    broker: str,
    method_name: str,
) -> TradierBrokerError:
    payload = _safe_json_payload(response)
    detail = _error_detail(response)
    message = (
        f"Tradier request failed for {path} [{broker}] via {method_name}: {detail}"
    )
    status_code = int(response.status_code)

    if status_code == 429:
        return TradierRateLimitError(
            message,
            status_code=status_code,
            error_code="rate_limit",
            retryable=True,
            response_payload=payload,
        )
    if status_code in {401, 403}:
        return TradierAuthenticationError(
            message,
            status_code=status_code,
            error_code="authentication",
            retryable=False,
            response_payload=payload,
        )
    if status_code == 404:
        return TradierNotFoundError(
            message,
            status_code=status_code,
            error_code="not_found",
            retryable=False,
            response_payload=payload,
        )
    if status_code in {400, 409, 422}:
        return TradierValidationError(
            message,
            status_code=status_code,
            error_code="validation",
            retryable=False,
            response_payload=payload,
        )
    if status_code >= 500:
        return TradierTransientError(
            message,
            status_code=status_code,
            error_code="server_error",
            retryable=True,
            response_payload=payload,
        )
    return TradierBrokerError(
        message,
        status_code=status_code,
        error_code="http_error",
        retryable=False,
        response_payload=payload,
    )


def _classify_transport_error(
    exc: requests.RequestException,
    *,
    path: str,
    broker: str,
    method_name: str,
) -> TradierBrokerError:
    retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError))
    error_cls = TradierTransportError if retryable else TradierBrokerError
    return error_cls(
        f"Tradier transport error for {path} [{broker}] via {method_name}: {exc}",
        error_code="transport",
        retryable=retryable,
    )


def _retry_delay_seconds(
    *,
    attempt: int,
    response: requests.Response | None,
) -> float:
    retry_after = _parse_retry_after_seconds(response)
    if retry_after is not None:
        return retry_after
    return min(2**attempt, 8)


def _parse_retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, seconds)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
