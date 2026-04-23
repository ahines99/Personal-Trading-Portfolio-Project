from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.brokerage.tradier import (  # noqa: E402
    TradierAuthenticationError,
    TradierBrokerClient,
    TradierDuplicateOrderError,
    TradierTransientError,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.text = str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RouteSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._routes: dict[tuple[str, str], list[Any]] = {}

    def queue(self, method: str, suffix: str, *responses: Any) -> None:
        self._routes[(method.upper(), suffix)] = list(responses)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("POST", url, kwargs)

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("PUT", url, kwargs)

    def delete(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("DELETE", url, kwargs)

    def _dispatch(self, method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        for (route_method, suffix), queue in self._routes.items():
            if route_method == method and url.endswith(suffix):
                if not queue:
                    raise AssertionError(f"No queued responses left for {method} {suffix}")
                response = queue.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"Unexpected call: {method} {url}")


def test_preview_retries_rate_limit_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RouteSession()
    session.queue(
        "POST",
        "/accounts/PAPER-123/orders",
        FakeResponse(
            {"errors": {"error": "slow down"}},
            status_code=429,
            headers={"Retry-After": "3"},
        ),
        FakeResponse(
            {
                "order": {
                    "status": "ok",
                    "result": "true",
                    "cost": "401.25",
                    "fees": "0.00",
                    "order_cost": "401.25",
                    "margin_change": "401.25",
                }
            }
        ),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        max_retries=2,
        session=session,  # type: ignore[arg-type]
    )
    sleeps: list[float] = []
    monkeypatch.setattr("paper.brokerage.tradier.time.sleep", sleeps.append)

    preview = client.preview_equity_order("SPY", 1.0, "buy", "market")

    assert preview["result"] is True
    assert preview["status"] == "ok"
    assert sleeps == [3.0]
    post_calls = [call for call in session.calls if call[0] == "POST"]
    assert len(post_calls) == 2


def test_get_profile_raises_authentication_error_on_401() -> None:
    session = RouteSession()
    session.queue(
        "GET",
        "/user/profile",
        FakeResponse({"errors": {"error": "auth"}}, status_code=401),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        session=session,  # type: ignore[arg-type]
    )

    with pytest.raises(TradierAuthenticationError, match="401"):
        client.get_profile()


def test_place_equity_order_reuses_existing_tagged_order_without_reposting() -> None:
    session = RouteSession()
    session.queue(
        "GET",
        "/accounts/PAPER-123/orders",
        FakeResponse(
            {
                "orders": {
                    "order": {
                        "id": "existing-1",
                        "tag": "pt-dup",
                        "symbol": "SPY",
                        "quantity": "1",
                        "side": "buy",
                        "type": "market",
                        "duration": "day",
                        "status": "open",
                    }
                }
            }
        ),
    )
    session.queue(
        "GET",
        "/accounts/PAPER-123/orders/existing-1",
        FakeResponse(
            {
                "order": {
                    "id": "existing-1",
                    "tag": "pt-dup",
                    "symbol": "SPY",
                    "quantity": "1",
                    "side": "buy",
                    "type": "market",
                    "duration": "day",
                    "status": "open",
                }
            }
        ),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        session=session,  # type: ignore[arg-type]
    )

    placed = client.place_equity_order(
        "SPY",
        1.0,
        "buy",
        "market",
        preview_result=_preview_result(tag="pt-dup"),
    )

    assert placed["broker_order_id"] == "existing-1"
    assert placed["recovered_by_tag"] is True
    assert placed["placement_status"] == "recovered_existing"
    assert not [call for call in session.calls if call[0] == "POST"]


def test_place_equity_order_recovers_after_timeout_by_querying_tag() -> None:
    session = RouteSession()
    session.queue(
        "GET",
        "/accounts/PAPER-123/orders",
        FakeResponse({"orders": "null"}),
        FakeResponse(
            {
                "orders": {
                    "order": [
                        {
                            "id": "existing-2",
                            "tag": "pt-recover",
                            "symbol": "SPY",
                            "quantity": "1",
                            "side": "buy",
                            "type": "market",
                            "duration": "day",
                            "status": "open",
                        }
                    ]
                }
            }
        ),
    )
    session.queue(
        "POST",
        "/accounts/PAPER-123/orders",
        requests.Timeout("timed out"),
    )
    session.queue(
        "GET",
        "/accounts/PAPER-123/orders/existing-2",
        FakeResponse(
            {
                "order": {
                    "id": "existing-2",
                    "tag": "pt-recover",
                    "symbol": "SPY",
                    "quantity": "1",
                    "side": "buy",
                    "type": "market",
                    "duration": "day",
                    "status": "filled",
                    "exec_quantity": "1",
                }
            }
        ),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        session=session,  # type: ignore[arg-type]
    )

    placed = client.place_equity_order(
        "SPY",
        1.0,
        "buy",
        "market",
        preview_result=_preview_result(tag="pt-recover"),
    )

    assert placed["broker_order_id"] == "existing-2"
    assert placed["status"] == "filled"
    assert placed["recovered_by_tag"] is True
    assert len([call for call in session.calls if call[0] == "POST"]) == 1


def test_place_equity_order_raises_duplicate_tag_error_for_multiple_matches() -> None:
    session = RouteSession()
    session.queue(
        "GET",
        "/accounts/PAPER-123/orders",
        FakeResponse(
            {
                "orders": {
                    "order": [
                        {
                            "id": "existing-1",
                            "tag": "pt-dup",
                            "symbol": "SPY",
                            "quantity": "1",
                            "side": "buy",
                            "type": "market",
                            "duration": "day",
                            "status": "open",
                        },
                        {
                            "id": "existing-2",
                            "tag": "pt-dup",
                            "symbol": "SPY",
                            "quantity": "1",
                            "side": "buy",
                            "type": "market",
                            "duration": "day",
                            "status": "open",
                        },
                    ]
                }
            }
        ),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        session=session,  # type: ignore[arg-type]
    )

    with pytest.raises(TradierDuplicateOrderError, match="pt-dup"):
        client.place_equity_order(
            "SPY",
            1.0,
            "buy",
            "market",
            preview_result=_preview_result(tag="pt-dup"),
        )


def test_retryable_5xx_exhaustion_raises_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RouteSession()
    session.queue(
        "GET",
        "/accounts/PAPER-123/balances",
        FakeResponse({"errors": {"error": "oops"}}, status_code=503),
        FakeResponse({"errors": {"error": "oops"}}, status_code=503),
        FakeResponse({"errors": {"error": "oops"}}, status_code=503),
    )
    client = TradierBrokerClient(
        account_id="PAPER-123",
        token="sandbox-token",
        broker="tradier_sandbox",
        max_retries=2,
        session=session,  # type: ignore[arg-type]
    )
    sleeps: list[float] = []
    monkeypatch.setattr("paper.brokerage.tradier.time.sleep", sleeps.append)

    with pytest.raises(TradierTransientError, match="503"):
        client.get_balances()

    assert sleeps == [1, 2]


def _preview_result(*, tag: str) -> dict[str, Any]:
    return {
        "preview_id": f"tradier-preview-{tag}",
        "result": True,
        "symbol": "SPY",
        "quantity": 1.0,
        "side": "buy",
        "type": "market",
        "limit_price": None,
        "tag": tag,
    }
