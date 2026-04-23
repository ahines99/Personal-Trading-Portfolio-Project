from __future__ import annotations

from typing import Any

from ..secrets import load_tradier_token, store_tradier_token


def broker_mode_name(broker: str) -> str:
    normalized = str(broker).strip().lower()
    if normalized == "tradier_sandbox":
        return "sandbox"
    if normalized == "tradier_live":
        return "live"
    raise ValueError(f"Unsupported Tradier broker backend: {broker!r}")


def load_token_for_config(config: Any) -> str:
    broker = _require_attr(config, "broker")
    account_id = _require_attr(config, "account_id")
    return load_tradier_token(broker_mode_name(str(broker)), str(account_id))


def store_token_for_broker(broker: str, account_id: str, token: str) -> None:
    store_tradier_token(broker_mode_name(broker), account_id, token)


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
