from __future__ import annotations

from datetime import date as date_type
from typing import Any, Iterable

import pandas as pd

import portfolio as portfolio_module


def get_rebalance_dates(
    year: int,
    signal_index: pd.DatetimeIndex | Iterable[Any] | None = None,
    config: Any | None = None,
) -> set[pd.Timestamp]:
    """Return the paper-trading rebalance dates for one calendar year."""
    if _read_attr(config, "rebalance_calendar_source") == "explicit":
        return _explicit_rebalance_dates(year, config)

    trading_index = _coerce_index(signal_index, year)
    return {
        pd.Timestamp(item).normalize()
        for item in portfolio_module._get_optimal_rebalance_dates(
            trading_index,
            day_offset=2,
        )
    }


def is_rebalance_day(
    value: Any,
    signal_index: pd.DatetimeIndex | Iterable[Any] | None = None,
    config: Any | None = None,
) -> bool:
    trading_day = pd.Timestamp(value).normalize()
    return trading_day in get_rebalance_dates(
        trading_day.year,
        signal_index=signal_index,
        config=config,
    )


def _explicit_rebalance_dates(year: int, config: Any | None) -> set[pd.Timestamp]:
    entries = _read_attr(config, "rebalance_calendar") or []
    dates: set[pd.Timestamp] = set()
    for entry in entries:
        raw_date = entry.get("date") if isinstance(entry, dict) else getattr(entry, "date", None)
        if raw_date is None:
            continue
        timestamp = pd.Timestamp(raw_date).normalize()
        if timestamp.year == year:
            dates.add(timestamp)
    return dates


def _coerce_index(
    signal_index: pd.DatetimeIndex | Iterable[Any] | None,
    year: int,
) -> pd.DatetimeIndex:
    if signal_index is None:
        return pd.bdate_range(f"{year}-01-01", f"{year}-12-31")

    if isinstance(signal_index, pd.DatetimeIndex):
        index = signal_index
    else:
        index = pd.DatetimeIndex(signal_index)

    normalized = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    filtered = normalized[normalized.year == year]
    if filtered.empty:
        raise ValueError(f"signal_index has no trading dates for {year}")
    return filtered


def _read_attr(config: Any, field_name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


__all__ = ["get_rebalance_dates", "is_rebalance_day"]
