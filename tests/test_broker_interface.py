from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fixtures.broker_fixtures import (  # noqa: E402
    broker_balances,
    broker_positions,
    broker_profile,
    clone_balances,
    clone_positions,
    clone_profile,
    mock_broker_client,
)
from paper.brokerage import BrokerClient, MockBrokerClient  # noqa: E402
from paper.brokerage.factory import create_broker_client  # noqa: E402


class _MinimalBrokerClient(BrokerClient):
    def __init__(self) -> None:
        self._profile = clone_profile()
        self._balances = clone_balances()
        self._positions = clone_positions()

    def ping(self) -> bool:
        return True

    def get_profile(self) -> dict[str, Any]:
        return self._profile

    def get_balances(self) -> dict[str, Any]:
        return self._balances

    def get_positions(self) -> list[dict[str, Any]]:
        return self._positions


class BrokerContractTests:
    def make_client(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> BrokerClient:
        raise NotImplementedError

    def test_ping(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> None:
        client = self.make_client(broker_profile, broker_balances, broker_positions)
        assert client.ping() is True

    def test_get_profile_schema(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> None:
        client = self.make_client(broker_profile, broker_balances, broker_positions)
        profile = client.get_profile()

        assert isinstance(profile, dict)
        assert {"account_id", "account_type", "status"} <= set(profile)
        assert profile["account_id"] == "PAPER-12345"

    def test_get_balances_schema(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> None:
        client = self.make_client(broker_profile, broker_balances, broker_positions)
        balances = client.get_balances()

        assert isinstance(balances, dict)
        assert {"account_id", "account_type", "cash", "buying_power", "equity"} <= set(
            balances
        )
        assert balances["cash"] == pytest.approx(12500.0)

    def test_get_positions_schema(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> None:
        client = self.make_client(broker_profile, broker_balances, broker_positions)
        positions = client.get_positions()

        assert isinstance(positions, list)
        assert len(positions) == 2
        assert {"ticker", "quantity", "cost_basis", "market_value", "current_weight"} <= set(
            positions[0]
        )
        assert positions[0]["ticker"] == "AAA"

    def test_methods_handle_errors(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> None:
        client = self.make_client(broker_profile, broker_balances, broker_positions)

        if isinstance(client, MockBrokerClient):
            preview = client.preview_equity_order("AAA", 10.0, "BUY", "market")
            assert preview["symbol"] == "AAA"
            placed = client.place_equity_order(
                "AAA",
                10.0,
                "BUY",
                "market",
                preview_result=preview,
            )
            assert placed["broker_order_id"]
            assert client.cancel_order(placed["broker_order_id"])["broker_order_id"] == placed["broker_order_id"]
            return

        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.preview_equity_order("AAA", 10.0, "BUY", "market")
        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.place_equity_order("AAA", 10.0, "BUY", "market")
        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.cancel_order("broker-1")
        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.replace_order("broker-1", qty=5.0)
        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.get_order_status("broker-1")
        with pytest.raises(NotImplementedError, match="Stage 3"):
            client.poll_until_terminal("broker-1")


class TestMockBrokerClient(BrokerContractTests):
    def make_client(
        self,
        broker_profile: dict[str, object],
        broker_balances: dict[str, object],
        broker_positions: list[dict[str, object]],
    ) -> BrokerClient:
        return MockBrokerClient(
            profile=broker_profile,
            balances=broker_balances,
            positions=broker_positions,
        )


def test_broker_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        BrokerClient()


def test_read_only_subclass_can_call_methods() -> None:
    client = _MinimalBrokerClient()

    assert client.ping() is True
    assert client.get_profile()["account_id"] == "PAPER-12345"
    assert client.get_balances()["equity"] == pytest.approx(32000.0)
    assert client.get_positions()[1]["ticker"] == "BBB"


def test_mock_broker_client_returns_defensive_copies(
    mock_broker_client: MockBrokerClient,
) -> None:
    profile = mock_broker_client.get_profile()
    balances = mock_broker_client.get_balances()
    positions = mock_broker_client.get_positions()

    profile["status"] = "mutated"
    balances["cash"] = 1.0
    positions[0]["ticker"] = "ZZZ"

    assert mock_broker_client.get_profile()["status"] == "active"
    assert mock_broker_client.get_balances()["cash"] == pytest.approx(12500.0)
    assert mock_broker_client.get_positions()[0]["ticker"] == "AAA"


def test_factory_returns_mock_client_for_mock_backend() -> None:
    client = create_broker_client(
        {
            "stage": 1,
            "broker": "mock",
            "capital_mode": "paper",
            "account_id": "PAPER-12345",
        }
    )
    assert isinstance(client, MockBrokerClient)
    assert client.ping() is True


def test_mock_broker_client_updates_positions_after_fill(
    mock_broker_client: MockBrokerClient,
) -> None:
    preview = mock_broker_client.preview_equity_order("AAA", 10.0, "SELL", "market")
    placed = mock_broker_client.place_equity_order(
        "AAA",
        10.0,
        "SELL",
        "market",
        preview_result=preview,
    )

    terminal = mock_broker_client.poll_until_terminal(
        placed["broker_order_id"],
        timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert terminal["status"] == "filled"
    aaa = next(
        row for row in mock_broker_client.get_positions() if row["ticker"] == "AAA"
    )
    assert aaa["quantity"] == pytest.approx(110.0)
