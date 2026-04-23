from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class RebalanceCalendarEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    config_hash: str

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            return date.fromisoformat(str(value)).isoformat()
        except ValueError as exc:
            raise ValueError("rebalance calendar date must be an ISO string") from exc

    @field_validator("config_hash")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rebalance calendar config_hash cannot be empty")
        return value


class PaperTradingConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, frozen=False, extra="forbid")

    stage: Literal[1, 2, 3, 4]
    broker: Literal["mock", "tradier_sandbox", "tradier_live"] = "mock"
    capital_mode: Literal["paper", "live"] = "paper"
    account_id: str
    results_dir: str = "results"
    baseline_path: str | None = None
    signal_max_staleness_days: int = 7
    rebalance_calendar_source: Literal["strategy", "explicit"] = "strategy"
    rebalance_calendar: list[RebalanceCalendarEntry] = Field(default_factory=list)
    portfolio_notional_usd: float = Field(
        default=100000.0,
        validation_alias=AliasChoices("portfolio_notional_usd", "notional_usd", "notional"),
    )
    min_trade_notional_usd: float = Field(
        default=50.0,
        validation_alias=AliasChoices(
            "min_trade_notional_usd", "min_trade_notional"
        ),
    )
    max_order_count: int = 10
    max_single_order_notional_usd: float = 50000.0
    allowed_order_types: list[str] = Field(default_factory=lambda: ["market"])
    dry_run: bool = True
    intents_only: bool = False
    stage4_mode: Literal["manual", "auto"] = "manual"
    kill_switch_path: str = "paper_trading/state/KILL_SWITCH"
    alert_channel: Literal["desktop", "email", "slack"] | None = None
    desktop_alerts_enabled: bool = True
    email_alerts_enabled: bool = False
    alert_recipients: list[str] = Field(default_factory=list)
    alert_log_path: str = "paper_trading/monitoring_alerts.jsonl"
    alert_smtp_profile: str | None = None
    alert_mute_types: list[str] = Field(default_factory=list)
    canary_mode: bool = False
    canary_days: int = 10
    canary_max_single_order_notional_usd: float = 500.0
    canary_max_order_count: int = 5
    canary_tracking_path: str = "paper_trading/canary_state.json"
    overnight_drift_halt_pct: float = 5.0
    turnover_cap_pct: float = 25.0
    estimated_cost_cap_pct: float = 0.15
    max_margin_usage_pct: float = 80.0
    submission_poll_seconds: int = 60
    submission_poll_interval_seconds: int = 5
    approval_deadline_time_et: str = "08:00"
    hard_halt_drawdown_pct: float | None = None
    hard_halt_daily_loss_pct: float | None = None
    confirm_live_trading: bool = Field(default=False, exclude=True, repr=False)

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("account_id cannot be empty")
        return value

    @field_validator(
        "results_dir",
        "kill_switch_path",
        "alert_log_path",
        "canary_tracking_path",
    )
    @classmethod
    def validate_required_path_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("path-like config values cannot be empty")
        return value

    @field_validator("baseline_path")
    @classmethod
    def validate_baseline_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None

        baseline_dir = Path(value)
        if not baseline_dir.exists():
            raise ValueError(f"baseline_path does not exist: {baseline_dir}")
        if not baseline_dir.is_dir():
            raise ValueError(f"baseline_path must be a directory: {baseline_dir}")
        signal_path = baseline_dir / "final_signal.parquet"
        if not signal_path.exists():
            raise ValueError(
                f"baseline_path must contain final_signal.parquet: {baseline_dir}"
            )
        return str(baseline_dir)

    @field_validator("signal_max_staleness_days")
    @classmethod
    def validate_signal_max_staleness_days(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("signal_max_staleness_days must be positive")
        return value

    @field_validator(
        "portfolio_notional_usd",
        "max_single_order_notional_usd",
        "canary_max_single_order_notional_usd",
    )
    @classmethod
    def validate_positive_notional(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("notional config values must be positive")
        return value

    @field_validator("min_trade_notional_usd")
    @classmethod
    def validate_min_trade_notional_usd(cls, value: float) -> float:
        if value < 0:
            raise ValueError("min_trade_notional_usd cannot be negative")
        return value

    @field_validator("max_order_count")
    @classmethod
    def validate_max_order_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_order_count must be positive")
        return value

    @field_validator(
        "canary_days",
        "canary_max_order_count",
        "submission_poll_seconds",
        "submission_poll_interval_seconds",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("integer config values must be positive")
        return value

    @field_validator("allowed_order_types")
    @classmethod
    def validate_allowed_order_types(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            order_type = str(item).strip().lower()
            if not order_type:
                raise ValueError("allowed_order_types cannot contain empty values")
            normalized.append(order_type)
        if not normalized:
            raise ValueError("allowed_order_types cannot be empty")
        return normalized

    @field_validator("alert_recipients", "alert_mute_types")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            normalized_item = str(item).strip()
            if not normalized_item:
                raise ValueError("list config values cannot contain empty strings")
            normalized.append(normalized_item)
        return normalized

    @field_validator("approval_deadline_time_et")
    @classmethod
    def validate_approval_deadline_time_et(cls, value: str) -> str:
        value = value.strip()
        if len(value) != 5 or value[2] != ":":
            raise ValueError("approval_deadline_time_et must be HH:MM")
        hour_str, minute_str = value.split(":", maxsplit=1)
        if not hour_str.isdigit() or not minute_str.isdigit():
            raise ValueError("approval_deadline_time_et must be HH:MM")
        hour = int(hour_str)
        minute = int(minute_str)
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError("approval_deadline_time_et must be a valid 24h time")
        return f"{hour:02d}:{minute:02d}"

    @field_validator(
        "overnight_drift_halt_pct",
        "turnover_cap_pct",
        "estimated_cost_cap_pct",
        "max_margin_usage_pct",
        "hard_halt_drawdown_pct",
        "hard_halt_daily_loss_pct",
    )
    @classmethod
    def validate_optional_pct(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("hard halt percentages must be positive when set")
        return value

    @model_validator(mode="after")
    def validate_rebalance_settings(self) -> "PaperTradingConfig":
        if self.rebalance_calendar_source == "explicit" and not self.rebalance_calendar:
            raise ValueError(
                "rebalance_calendar is required when rebalance_calendar_source='explicit'"
            )
        if self.rebalance_calendar_source == "strategy" and self.rebalance_calendar:
            raise ValueError(
                "rebalance_calendar must be empty when rebalance_calendar_source='strategy'"
            )
        return self

    @model_validator(mode="after")
    def validate_stage3_stage4_settings(self) -> "PaperTradingConfig":
        if self.min_trade_notional_usd > self.max_single_order_notional_usd:
            raise ValueError(
                "min_trade_notional_usd cannot exceed max_single_order_notional_usd"
            )

        if (
            self.canary_max_single_order_notional_usd
            > self.max_single_order_notional_usd
        ):
            raise ValueError(
                "canary_max_single_order_notional_usd cannot exceed "
                "max_single_order_notional_usd"
            )

        if self.canary_max_order_count > self.max_order_count:
            raise ValueError("canary_max_order_count cannot exceed max_order_count")

        if self.submission_poll_interval_seconds > self.submission_poll_seconds:
            raise ValueError(
                "submission_poll_interval_seconds cannot exceed submission_poll_seconds"
            )

        if self.stage4_mode == "auto" and self.stage != 4:
            raise PermissionError("stage4_mode='auto' requires stage=4")

        if self.canary_mode and self.stage != 4:
            raise PermissionError("canary_mode=true requires stage=4")

        return self

    @model_validator(mode="after")
    def validate_execution_permissions(self) -> "PaperTradingConfig":
        if self.stage == 1 and self.broker != "mock":
            raise PermissionError("stage=1 requires broker='mock'")

        if self.broker == "tradier_live" and self.stage < 4:
            raise PermissionError("broker='tradier_live' requires stage=4")

        if self.broker == "tradier_live" and self.capital_mode != "live":
            raise PermissionError(
                "broker='tradier_live' requires capital_mode='live'"
            )

        if self.capital_mode == "live":
            if self.stage != 4 or self.broker != "tradier_live":
                raise PermissionError(
                    "capital_mode='live' requires stage=4 and broker='tradier_live'"
                )

            env_confirmed = _is_truthy(os.environ.get("CONFIRM_LIVE_TRADING"))
            if not self.confirm_live_trading and not env_confirmed:
                raise PermissionError(
                    "live trading requires CONFIRM_LIVE_TRADING=true or explicit confirmation"
                )

        return self
