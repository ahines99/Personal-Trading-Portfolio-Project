from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from paper.tools.tradier_sandbox_smoke import (  # noqa: E402
    PreviewOrderRequest,
    require_explicit_write_flag,
    run_tradier_sandbox_smoke,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


class FakeTradierSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/accounts/PAPER-123"):
            return FakeResponse({"account": {"account_number": "PAPER-123"}})
        if url.endswith("/user/profile"):
            return FakeResponse(
                {
                    "profile": {
                        "id": "user-1",
                        "name": "Sandbox User",
                        "account": {
                            "account_number": "PAPER-123",
                            "classification": "individual",
                            "status": "active",
                        },
                    }
                }
            )
        if url.endswith("/accounts/PAPER-123/balances"):
            return FakeResponse(
                {
                    "balances": {
                        "account_number": "PAPER-123",
                        "account_type": "margin",
                        "total_cash": "1000.00",
                        "total_equity": "1000.00",
                        "market_value": "0.00",
                        "margin": {
                            "stock_buying_power": "2000.00",
                            "option_buying_power": "1000.00",
                        },
                        "cash": {"cash_available": "1000.00"},
                    }
                }
            )
        if url.endswith("/accounts/PAPER-123/positions"):
            return FakeResponse({"positions": "null"})
        return FakeResponse({"errors": {"error": "not found"}}, status_code=404)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/accounts/PAPER-123/orders"):
            return FakeResponse(
                {
                    "order": {
                        "status": "ok",
                        "result": "true",
                        "cost": "401.25",
                        "fees": "0.00",
                        "order_cost": "401.25",
                        "margin_change": "401.25",
                        "request_date": "2026-04-23T10:00:00Z",
                    }
                }
            )
        return FakeResponse({"errors": {"error": "not found"}}, status_code=404)


def test_default_smoke_is_read_only(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    session = FakeTradierSession()

    result = run_tradier_sandbox_smoke(
        config_path=config_path,
        token_loader=lambda config: "sandbox-token",
        session=session,  # type: ignore[arg-type]
    )

    assert result.ok is True
    assert result.preview is None
    assert [check.name for check in result.checks] == [
        "config",
        "sandbox_guard",
        "credentials",
        "account_ping",
        "profile",
        "balances",
        "positions",
        "order_preview",
    ]
    assert {method for method, _, _ in session.calls} == {"GET"}


def test_preview_order_uses_preview_flag_and_never_places(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    session = FakeTradierSession()

    result = run_tradier_sandbox_smoke(
        config_path=config_path,
        preview_order=True,
        preview_request=PreviewOrderRequest(symbol="SPY", quantity=1.0),
        token_loader=lambda config: "sandbox-token",
        session=session,  # type: ignore[arg-type]
    )

    assert result.ok is True
    assert result.preview is not None
    assert result.preview["would_place_order"] is False

    post_calls = [call for call in session.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    method, url, kwargs = post_calls[0]
    assert method == "POST"
    assert url.endswith("/accounts/PAPER-123/orders")
    assert kwargs["data"]["preview"] == "true"
    assert kwargs["data"]["quantity"] == "1"


def test_missing_config_fails_gracefully(tmp_path: Path) -> None:
    result = run_tradier_sandbox_smoke(config_path=tmp_path / "missing.yaml")

    assert result.ok is False
    assert result.checks[0].name == "config"
    assert result.checks[0].ok is False
    assert "paper trading config not found" in result.checks[0].detail


def test_non_sandbox_config_is_rejected_before_token_load(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, broker="mock")

    result = run_tradier_sandbox_smoke(
        config_path=config_path,
        token_loader=lambda config: pytest.fail("token loader should not run"),
    )

    assert result.ok is False
    assert result.checks[-1].name == "sandbox_guard"
    assert "expected broker=tradier_sandbox" in result.checks[-1].detail


def test_missing_token_fails_gracefully(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = run_tradier_sandbox_smoke(
        config_path=config_path,
        token_loader=lambda config: (_ for _ in ()).throw(RuntimeError("no token")),
    )

    assert result.ok is False
    assert result.checks[-1].name == "credentials"
    assert result.checks[-1].ok is False
    assert "no token" in result.checks[-1].detail


def test_future_write_guard_requires_explicit_flag() -> None:
    with pytest.raises(PermissionError, match="--allow-write"):
        require_explicit_write_flag(allow_write=False)

    require_explicit_write_flag(allow_write=True)


def _write_config(
    tmp_path: Path,
    *,
    broker: str = "tradier_sandbox",
    capital_mode: str = "paper",
) -> Path:
    config_path = tmp_path / "paper_trading.yaml"
    config_path.write_text(
        "\n".join(
            [
                "stage: 2",
                f"broker: {broker}",
                f"capital_mode: {capital_mode}",
                "account_id: PAPER-123",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
