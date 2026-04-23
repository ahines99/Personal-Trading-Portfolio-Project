from __future__ import annotations

from typing import Any

from .keyring_manager import load_token_for_config
from .mock import MockBrokerClient
from .tradier import TradierBrokerClient


def broker_enabled(config: Any) -> bool:
    broker = str(_read_attr(config, "broker") or "mock").strip().lower()
    return broker in {"mock", "tradier_sandbox", "tradier_live"}


def create_broker_client(
    config: Any,
    *,
    token: str | None = None,
    timeout: float = 15.0,
    max_retries: int = 3,
) -> Any:
    if not broker_enabled(config):
        return None

    broker = str(_read_attr(config, "broker") or "").strip().lower()
    account_id = str(_require_attr(config, "account_id")).strip()

    if broker == "mock":
        return MockBrokerClient(
            profile={
                "account_id": account_id,
                "account_type": "paper",
                "status": "active",
                "user_name": "Mock Broker",
                "email": None,
            },
            balances={
                "account_id": account_id,
                "account_type": "paper",
                "cash": 0.0,
                "buying_power": 0.0,
                "equity": 0.0,
                "market_value": 0.0,
                "total_equity": 0.0,
                "maintenance_requirement": 0.0,
            },
            positions=[],
        )

    if broker in {"tradier_sandbox", "tradier_live"}:
        client_token = token or load_token_for_config(config)
        return TradierBrokerClient(
            account_id=account_id,
            token=client_token,
            broker=broker,
            timeout=timeout,
            max_retries=max_retries,
        )

    raise ValueError(f"Unsupported broker backend: {broker!r}")


def describe_broker_backend(config: Any) -> dict[str, str | int]:
    broker = str(_read_attr(config, "broker") or "mock").strip().lower()
    stage = int(_read_attr(config, "stage") or 1)
    capital_mode = str(_read_attr(config, "capital_mode") or "paper").strip().lower()
    account_id = str(_read_attr(config, "account_id") or "").strip()
    return {
        "broker": broker,
        "stage": stage,
        "capital_mode": capital_mode,
        "account_id": account_id,
    }


def _require_attr(config: Any, field_name: str) -> Any:
    value = _read_attr(config, field_name)
    if value in (None, ""):
        raise ValueError(f"Paper trading config missing required field: {field_name}")
    return value


def _read_attr(config: Any, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)
