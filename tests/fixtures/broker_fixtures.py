from __future__ import annotations

from copy import deepcopy

import pytest

from paper.brokerage.mock import MockBrokerClient


_PROFILE = {
    "account_id": "PAPER-12345",
    "account_type": "margin",
    "status": "active",
    "user_name": "Paper Trader",
    "email": "paper@example.com",
}

_BALANCES = {
    "account_id": "PAPER-12345",
    "account_type": "margin",
    "cash": 12500.0,
    "buying_power": 25000.0,
    "equity": 32000.0,
    "market_value": 19500.0,
    "total_equity": 32000.0,
    "maintenance_requirement": 6000.0,
}

_POSITIONS = [
    {
        "ticker": "AAA",
        "quantity": 120.0,
        "cost_basis": 48.25,
        "market_value": 6000.0,
        "current_weight": 0.1875,
        "unrealized_pnl": 210.0,
        "entry_date": "2026-04-01",
    },
    {
        "ticker": "BBB",
        "quantity": 80.0,
        "cost_basis": 82.10,
        "market_value": 7500.0,
        "current_weight": 0.234375,
        "unrealized_pnl": -95.0,
        "entry_date": "2026-03-18",
    },
]


def clone_profile() -> dict[str, object]:
    return deepcopy(_PROFILE)


def clone_balances() -> dict[str, object]:
    return deepcopy(_BALANCES)


def clone_positions() -> list[dict[str, object]]:
    return deepcopy(_POSITIONS)


@pytest.fixture
def broker_profile() -> dict[str, object]:
    return clone_profile()


@pytest.fixture
def broker_balances() -> dict[str, object]:
    return clone_balances()


@pytest.fixture
def broker_positions() -> list[dict[str, object]]:
    return clone_positions()


@pytest.fixture
def mock_broker_client(
    broker_profile: dict[str, object],
    broker_balances: dict[str, object],
    broker_positions: list[dict[str, object]],
) -> MockBrokerClient:
    return MockBrokerClient(
        profile=broker_profile,
        balances=broker_balances,
        positions=broker_positions,
    )
