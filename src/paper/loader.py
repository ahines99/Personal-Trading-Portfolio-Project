from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import AliasChoices

from .config import PaperTradingConfig

DEFAULT_CONFIG_PATH = Path("config") / "paper_trading.yaml"
ENV_PREFIX = "PAPER_TRADING_"
REQUIRED_CONFIG_KEYS = ("account_id", "stage", "broker", "capital_mode")


def _parse_env_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _env_field_name(field_name: str) -> str:
    return f"{ENV_PREFIX}{field_name.upper()}"


def _field_env_names(field_name: str) -> list[str]:
    names = [_env_field_name(field_name)]
    field_info = PaperTradingConfig.model_fields[field_name]
    validation_alias = field_info.validation_alias

    if isinstance(validation_alias, AliasChoices):
        for alias in validation_alias.choices:
            if isinstance(alias, str):
                names.append(_env_field_name(alias))
    elif isinstance(validation_alias, str):
        names.append(_env_field_name(validation_alias))

    deduped_names: list[str] = []
    for name in names:
        if name not in deduped_names:
            deduped_names.append(name)
    return deduped_names


def _collect_env_overrides(env_overrides: Mapping[str, str] | None) -> dict[str, Any]:
    source = os.environ if env_overrides is None else env_overrides
    overrides: dict[str, Any] = {}

    for field_name in PaperTradingConfig.model_fields:
        for env_name in _field_env_names(field_name):
            if env_name in source:
                overrides[field_name] = _parse_env_value(source[env_name])
                break

    if "confirm_live_trading" not in overrides:
        if "PAPER_TRADING_CONFIRM_LIVE_TRADING" in source:
            overrides["confirm_live_trading"] = _parse_env_value(
                source["PAPER_TRADING_CONFIRM_LIVE_TRADING"]
            )
        elif "CONFIRM_LIVE_TRADING" in source:
            overrides["confirm_live_trading"] = _parse_env_value(
                source["CONFIRM_LIVE_TRADING"]
            )

    return overrides


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"paper trading config not found: {path}")

    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed paper trading YAML in {path}: {exc}") from exc

    if raw_data is None:
        return {}
    if not isinstance(raw_data, dict):
        raise ValueError(
            f"paper trading config at {path} must be a YAML mapping, got {type(raw_data).__name__}"
        )
    return dict(raw_data)


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    env_overrides: Mapping[str, str] | None = None,
) -> PaperTradingConfig:
    config_path = Path(path)
    yaml_config = _read_yaml_config(config_path)
    merged_config = dict(yaml_config)
    merged_config.update(_collect_env_overrides(env_overrides))

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in merged_config]
    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(
            f"paper trading config missing required keys after env merge: {missing_str}"
        )

    return PaperTradingConfig.model_validate(merged_config)
