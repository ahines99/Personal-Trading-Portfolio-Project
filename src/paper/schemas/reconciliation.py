"""Daily reconciliation schemas for paper trading."""

from __future__ import annotations

from datetime import date
from statistics import mean, median, pstdev
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import SCHEMA_VERSION

_DRIFT_TOLERANCE = 1e-4


class SlippageSummary(BaseModel):
    """Robust summary statistics over fill slippage observations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.slippage_summary.v1"},
    )

    count: int = Field(ge=0)
    mean_bps: float
    median_bps: float
    min_bps: float
    max_bps: float
    std_bps: float

    @classmethod
    def from_bps(cls, values: list[float]) -> "SlippageSummary":
        if not values:
            return cls(
                count=0,
                mean_bps=0.0,
                median_bps=0.0,
                min_bps=0.0,
                max_bps=0.0,
                std_bps=0.0,
            )
        return cls(
            count=len(values),
            mean_bps=mean(values),
            median_bps=median(values),
            min_bps=min(values),
            max_bps=max(values),
            std_bps=pstdev(values) if len(values) > 1 else 0.0,
        )


class WeightComparison(BaseModel):
    """Per-symbol target-vs-actual weight comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.weight_comparison.v1"},
    )

    ticker: str
    target: float
    actual: float
    drift: float
    drift_bps: float


class ReconciliationReport(BaseModel):
    """End-of-day reconciliation artifact for one paper-trading date."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"schema_version": f"{SCHEMA_VERSION}.reconciliation.v1"},
    )

    as_of_date: date
    intended_trades: list[dict[str, Any]]
    executed_trades: list[dict[str, Any]]
    unfilled_orders: list[dict[str, Any]]
    cancellations: list[dict[str, Any]]
    slippage_summary: SlippageSummary
    target_weights: dict[str, float]
    actual_weights: dict[str, float]
    weight_comparisons: list[WeightComparison]
    weight_drift_l1: float = Field(ge=0.0)
    cash_drift: float
    nav_drift: float
    pnl_reconciled_vs_backtest: float | None = None
    anomalies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_weight_drift(self) -> "ReconciliationReport":
        implied_l1 = sum(abs(comparison.drift) for comparison in self.weight_comparisons)
        if abs(implied_l1 - self.weight_drift_l1) > _DRIFT_TOLERANCE:
            raise ValueError(
                "weight_drift_l1 does not match the sum of absolute comparison drifts: "
                f"{self.weight_drift_l1:.6f} vs {implied_l1:.6f}."
            )
        return self

    @classmethod
    def build_weight_comparisons(
        cls, target_weights: dict[str, float], actual_weights: dict[str, float]
    ) -> list[WeightComparison]:
        comparisons: list[WeightComparison] = []
        for ticker in sorted(set(target_weights) | set(actual_weights)):
            target = target_weights.get(ticker, 0.0)
            actual = actual_weights.get(ticker, 0.0)
            drift = actual - target
            comparisons.append(
                WeightComparison(
                    ticker=ticker,
                    target=target,
                    actual=actual,
                    drift=drift,
                    drift_bps=drift * 10000.0,
                )
            )
        return comparisons

    def summary_row(self) -> dict[str, Any]:
        """Return a flattened summary row suitable for a parquet time series."""
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "intended_trade_count": len(self.intended_trades),
            "executed_trade_count": len(self.executed_trades),
            "unfilled_order_count": len(self.unfilled_orders),
            "cancellation_count": len(self.cancellations),
            "weight_drift_l1": self.weight_drift_l1,
            "cash_drift": self.cash_drift,
            "nav_drift": self.nav_drift,
            "slippage_count": self.slippage_summary.count,
            "slippage_mean_bps": self.slippage_summary.mean_bps,
            "slippage_median_bps": self.slippage_summary.median_bps,
            "slippage_std_bps": self.slippage_summary.std_bps,
            "pnl_reconciled_vs_backtest": self.pnl_reconciled_vs_backtest,
            "anomaly_count": len(self.anomalies),
        }
