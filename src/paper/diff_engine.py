"""Stage 3 target-vs-actual diff engine for paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from typing import Any, Iterable, Mapping

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is already a repo dependency
    pd = None

from .schemas.broker_positions import BrokerPositionsSnapshot
from .schemas.target_book import TargetBookSnapshot

_SIDE_BUY = "BUY"
_SIDE_SELL = "SELL"
_EQUITY_TYPES = {"equity", "stock", "common_stock", "adr", "common"}
_PRICE_KEYS = (
    "est_price",
    "price",
    "reference_price",
    "last_price",
    "market_price",
    "close",
)
_ADV_DOLLAR_KEYS = ("adv", "avg_daily_dollar_volume", "average_daily_dollar_volume")
_ADV_SHARE_KEYS = ("adv_shares", "avg_daily_share_volume", "average_daily_share_volume")
_SECURITY_TYPE_KEYS = ("security_type", "asset_type", "instrument_type", "type")


def _normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    return symbol


def _to_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _coerce_mapping_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python", round_trip=True))
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return {}


@dataclass(frozen=True)
class ProposedTrade:
    """One explicit target-vs-actual rebalance proposal."""

    proposal_id: str
    symbol: str
    side: str
    qty: float
    est_price: float
    est_notional: float
    target_weight: float
    current_weight: float
    target_qty: float
    current_qty: float
    delta_weight: float
    delta_notional: float
    available_cash: float
    security_type: str = "equity"
    adv_participation_pct: float | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        payload = {
            "proposal_id": self.proposal_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "est_price": self.est_price,
            "est_notional": self.est_notional,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "target_qty": self.target_qty,
            "current_qty": self.current_qty,
            "delta_weight": self.delta_weight,
            "delta_notional": self.delta_notional,
            "available_cash": self.available_cash,
            "security_type": self.security_type,
            "adv_participation_pct": self.adv_participation_pct,
            "warnings": list(self.warnings),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ProposedTrades:
    """Batch of proposed trades plus execution-side summary state."""

    proposals: tuple[ProposedTrade, ...]
    available_cash: float
    gross_buy_notional: float
    gross_sell_notional: float
    projected_cash_after_sells: float
    net_cash_required: float
    cash_reconciled: bool
    warnings: tuple[str, ...] = ()

    def as_records(self) -> list[dict[str, Any]]:
        return [proposal.as_record() for proposal in self.proposals]

    @property
    def count(self) -> int:
        return len(self.proposals)


class CashConstraintViolation(ValueError):
    """Raised when the proposed buy book exceeds available cash plus sells."""


class DiffEngine:
    """Compare target weights against broker state and emit explicit trade proposals."""

    def __init__(
        self,
        *,
        min_trade_notional: float = 10.0,
        max_adv_participation: float = 0.10,
        cash_tolerance_usd: float = 1.0,
    ) -> None:
        if min_trade_notional < 0:
            raise ValueError("min_trade_notional cannot be negative")
        if max_adv_participation <= 0:
            raise ValueError("max_adv_participation must be positive")
        if cash_tolerance_usd < 0:
            raise ValueError("cash_tolerance_usd cannot be negative")
        self.min_trade_notional = float(min_trade_notional)
        self.max_adv_participation = float(max_adv_participation)
        self.cash_tolerance_usd = float(cash_tolerance_usd)

    def compute_trades(
        self,
        target_weights: Any,
        broker_positions: Any,
        previous_positions: Any = None,
        *,
        total_portfolio_value: float | None = None,
    ) -> ProposedTrades:
        target_state = _normalize_target_state(target_weights)
        broker_state = _normalize_broker_state(broker_positions)
        previous_state = _normalize_previous_state(previous_positions)

        portfolio_value = (
            float(total_portfolio_value)
            if total_portfolio_value is not None
            else broker_state["portfolio_value"]
        )
        if portfolio_value <= 0:
            raise ValueError("total_portfolio_value must be positive")

        available_cash = broker_state["available_cash"]
        proposal_warnings: list[str] = []
        proposals: list[ProposedTrade] = []

        universe = sorted(
            set(target_state["positions"])
            | set(broker_state["positions"])
            | set(previous_state)
        )

        for symbol in universe:
            target_payload = target_state["positions"].get(symbol, {})
            broker_payload = broker_state["positions"].get(symbol, {})
            previous_payload = previous_state.get(symbol, {})

            target_weight = _to_float(target_payload.get("target_weight"), default=0.0) or 0.0
            current_qty = _to_float(broker_payload.get("quantity"), default=0.0) or 0.0
            current_notional = _to_float(broker_payload.get("market_value"))
            if current_notional is None:
                current_weight = _to_float(
                    broker_payload.get("current_weight"),
                    default=0.0,
                ) or 0.0
                current_notional = current_weight * portfolio_value
            current_weight = (
                _to_float(broker_payload.get("current_weight"))
                if broker_payload.get("current_weight") is not None
                else current_notional / portfolio_value
            )
            current_weight = current_weight or 0.0

            est_price = _resolve_est_price(target_payload, broker_payload, previous_payload)
            target_notional = target_weight * portfolio_value
            delta_notional = target_notional - current_notional
            if abs(delta_notional) < self.min_trade_notional:
                continue
            if est_price is None or est_price <= 0:
                raise ValueError(
                    f"Missing usable est_price for {symbol}; cannot compute trade quantity."
                )

            target_qty = target_notional / est_price
            delta_qty = target_qty - current_qty
            if abs(delta_qty * est_price) < self.min_trade_notional:
                continue
            side = _SIDE_BUY if delta_qty > 0 else _SIDE_SELL
            delta_weight = target_weight - current_weight
            security_type = _resolve_security_type(
                target_payload,
                broker_payload,
                previous_payload,
            )
            adv_participation_pct, adv_warning = _compute_adv_participation(
                delta_qty=delta_qty,
                est_price=est_price,
                target_payload=target_payload,
                broker_payload=broker_payload,
                previous_payload=previous_payload,
                max_adv_participation=self.max_adv_participation,
            )

            warnings: list[str] = []
            if adv_warning:
                warnings.append(adv_warning)
                proposal_warnings.append(f"{symbol}: {adv_warning}")
            if security_type not in _EQUITY_TYPES:
                warning = f"non-equity security_type={security_type}"
                warnings.append(warning)
                proposal_warnings.append(f"{symbol}: {warning}")

            proposal_id = _build_proposal_id(
                symbol=symbol,
                side=side,
                target_weight=target_weight,
                current_weight=current_weight,
                qty=abs(delta_qty),
                est_price=est_price,
            )
            proposals.append(
                ProposedTrade(
                    proposal_id=proposal_id,
                    symbol=symbol,
                    side=side,
                    qty=abs(delta_qty),
                    est_price=est_price,
                    est_notional=abs(delta_qty) * est_price,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    target_qty=target_qty,
                    current_qty=current_qty,
                    delta_weight=delta_weight,
                    delta_notional=delta_notional,
                    available_cash=available_cash,
                    security_type=security_type,
                    adv_participation_pct=adv_participation_pct,
                    warnings=tuple(warnings),
                    metadata=_compact_metadata(
                        target_payload=target_payload,
                        broker_payload=broker_payload,
                        previous_payload=previous_payload,
                    ),
                )
            )

        proposals.sort(
            key=lambda proposal: (-abs(proposal.qty), proposal.symbol, proposal.side)
        )

        gross_buy_notional = sum(
            proposal.est_notional for proposal in proposals if proposal.side == _SIDE_BUY
        )
        gross_sell_notional = sum(
            proposal.est_notional for proposal in proposals if proposal.side == _SIDE_SELL
        )
        projected_cash_after_sells = available_cash + gross_sell_notional
        cash_reconciled = gross_buy_notional <= (
            projected_cash_after_sells + self.cash_tolerance_usd
        )
        if not cash_reconciled:
            deficit = gross_buy_notional - projected_cash_after_sells
            raise CashConstraintViolation(
                "Proposed buys exceed available cash plus sells: "
                f"need {gross_buy_notional:.2f}, have {projected_cash_after_sells:.2f}, "
                f"deficit {deficit:.2f}."
            )

        return ProposedTrades(
            proposals=tuple(proposals),
            available_cash=available_cash,
            gross_buy_notional=gross_buy_notional,
            gross_sell_notional=gross_sell_notional,
            projected_cash_after_sells=projected_cash_after_sells,
            net_cash_required=max(gross_buy_notional - gross_sell_notional, 0.0),
            cash_reconciled=cash_reconciled,
            warnings=tuple(dict.fromkeys(proposal_warnings)),
        )


def _normalize_target_state(target_weights: Any) -> dict[str, Any]:
    if isinstance(target_weights, TargetBookSnapshot):
        positions = {
            symbol: {
                "target_weight": position.target_weight,
                "signal_score": position.signal_score,
                "signal_rank": position.signal_rank,
                "sector": position.sector,
                "rationale": position.rationale,
            }
            for symbol, position in target_weights.target_positions.items()
        }
        return {"positions": positions, "cash_weight": target_weights.target_cash_pct}

    if isinstance(target_weights, Mapping):
        payload = dict(target_weights)
        position_payload = payload.get("positions")
        if position_payload is not None:
            positions = _normalize_symbol_payload_map(
                position_payload,
                weight_keys=("target_weight", "weight"),
                default_weight_key="target_weight",
            )
            cash_weight = _to_float(payload.get("cash_weight"), default=0.0) or 0.0
            return {"positions": positions, "cash_weight": cash_weight}

    positions = _normalize_symbol_payload_map(
        target_weights,
        weight_keys=("target_weight", "weight"),
        default_weight_key="target_weight",
    )
    return {"positions": positions, "cash_weight": 0.0}


def _normalize_broker_state(broker_positions: Any) -> dict[str, Any]:
    if isinstance(broker_positions, BrokerPositionsSnapshot):
        positions = {
            symbol: {
                "quantity": position.quantity,
                "market_value": position.market_value,
                "current_weight": position.current_weight,
                "cost_basis": position.cost_basis,
                "unrealized_pnl": position.unrealized_pnl,
                "entry_date": (
                    position.entry_date.isoformat() if position.entry_date else None
                ),
            }
            for symbol, position in broker_positions.positions.items()
        }
        return {
            "positions": positions,
            "available_cash": broker_positions.cash_balance,
            "portfolio_value": broker_positions.nav,
        }

    if isinstance(broker_positions, Mapping):
        payload = dict(broker_positions)
        position_payload = payload.get("positions")
        if position_payload is not None:
            positions = _normalize_symbol_payload_map(
                position_payload,
                weight_keys=("current_weight", "weight"),
                default_weight_key="current_weight",
            )
            available_cash = (
                _to_float(payload.get("cash_balance"))
                or _to_float(payload.get("cash"))
                or 0.0
            )
            portfolio_value = (
                _to_float(payload.get("nav"))
                or _to_float(payload.get("equity_value"))
                or _to_float(payload.get("equity"))
            )
            if portfolio_value is None:
                portfolio_value = available_cash + sum(
                    _to_float(item.get("market_value"), default=0.0) or 0.0
                    for item in positions.values()
                )
            return {
                "positions": positions,
                "available_cash": available_cash,
                "portfolio_value": portfolio_value,
            }

    if isinstance(broker_positions, Iterable) and not isinstance(
        broker_positions, (str, bytes, Mapping)
    ):
        positions = _normalize_symbol_payload_map(
            broker_positions,
            weight_keys=("current_weight", "weight"),
            default_weight_key="current_weight",
        )
        portfolio_value = sum(
            _to_float(item.get("market_value"), default=0.0) or 0.0
            for item in positions.values()
        )
        return {
            "positions": positions,
            "available_cash": 0.0,
            "portfolio_value": portfolio_value,
        }

    raise TypeError(f"Unsupported broker_positions input: {type(broker_positions)!r}")


def _normalize_previous_state(previous_positions: Any) -> dict[str, dict[str, Any]]:
    if previous_positions is None:
        return {}
    return _normalize_symbol_payload_map(
        previous_positions,
        weight_keys=("weight", "target_weight", "current_weight"),
        default_weight_key="weight",
    )


def _normalize_symbol_payload_map(
    value: Any,
    *,
    weight_keys: tuple[str, ...],
    default_weight_key: str,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}

    if pd is not None and isinstance(value, pd.Series):
        return {
            _normalize_symbol(symbol): {default_weight_key: _to_float(weight, default=0.0) or 0.0}
            for symbol, weight in value.dropna().items()
        }

    if pd is not None and isinstance(value, pd.DataFrame):
        lowered = {str(column).lower(): column for column in value.columns}
        if "ticker" in lowered or "symbol" in lowered:
            symbol_column = lowered.get("ticker") or lowered.get("symbol")
            records: dict[str, dict[str, Any]] = {}
            for record in value.to_dict(orient="records"):
                symbol = _normalize_symbol(record[symbol_column])
                payload = {
                    key: record.get(key)
                    for key in record
                    if key != symbol_column and record.get(key) is not None
                }
                payload[default_weight_key] = _extract_weight(
                    payload,
                    weight_keys=weight_keys,
                    default_weight_key=default_weight_key,
                )
                records[symbol] = payload
            return records

        if value.empty:
            return {}
        latest = value.iloc[-1].dropna().to_dict()
        return {
            _normalize_symbol(symbol): {default_weight_key: _to_float(weight, default=0.0) or 0.0}
            for symbol, weight in latest.items()
        }

    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = []
        for record in value:
            payload = _coerce_mapping_payload(record)
            symbol = payload.get("ticker", payload.get("symbol"))
            if symbol is None:
                raise ValueError("iterable record is missing ticker/symbol")
            items.append((symbol, payload))
    else:
        raise TypeError(f"Unsupported position payload: {type(value)!r}")

    normalized: dict[str, dict[str, Any]] = {}
    for symbol, raw_payload in items:
        normalized_symbol = _normalize_symbol(symbol)
        if isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)
        else:
            payload = {default_weight_key: raw_payload}
        if not isinstance(payload, Mapping):
            payload = {default_weight_key: payload}
        payload = dict(payload)
        payload[default_weight_key] = _extract_weight(
            payload,
            weight_keys=weight_keys,
            default_weight_key=default_weight_key,
        )
        normalized[normalized_symbol] = payload
    return normalized


def _extract_weight(
    payload: Mapping[str, Any],
    *,
    weight_keys: tuple[str, ...],
    default_weight_key: str,
) -> float:
    for key in weight_keys:
        if key in payload:
            return _to_float(payload.get(key), default=0.0) or 0.0
    return _to_float(payload.get(default_weight_key), default=0.0) or 0.0


def _resolve_est_price(
    target_payload: Mapping[str, Any],
    broker_payload: Mapping[str, Any],
    previous_payload: Mapping[str, Any],
) -> float | None:
    for payload in (broker_payload, target_payload, previous_payload):
        quantity = _to_float(payload.get("quantity"))
        market_value = _to_float(payload.get("market_value"))
        if quantity and market_value and quantity > 0:
            return market_value / quantity
        for key in _PRICE_KEYS:
            price = _to_float(payload.get(key))
            if price and price > 0:
                return price
    return None


def _resolve_security_type(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in _SECURITY_TYPE_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
    return "equity"


def _compute_adv_participation(
    *,
    delta_qty: float,
    est_price: float,
    target_payload: Mapping[str, Any],
    broker_payload: Mapping[str, Any],
    previous_payload: Mapping[str, Any],
    max_adv_participation: float,
) -> tuple[float | None, str | None]:
    for payload in (target_payload, broker_payload, previous_payload):
        for key in _ADV_DOLLAR_KEYS:
            adv_dollar = _to_float(payload.get(key))
            if adv_dollar and adv_dollar > 0:
                participation = abs(delta_qty) * est_price / adv_dollar
                if participation > max_adv_participation:
                    return (
                        participation,
                        f"ADV participation {participation:.2%} exceeds {max_adv_participation:.2%}",
                    )
                return participation, None
        for key in _ADV_SHARE_KEYS:
            adv_shares = _to_float(payload.get(key))
            if adv_shares and adv_shares > 0:
                participation = abs(delta_qty) / adv_shares
                if participation > max_adv_participation:
                    return (
                        participation,
                        f"ADV participation {participation:.2%} exceeds {max_adv_participation:.2%}",
                    )
                return participation, None
    return None, None


def _compact_metadata(
    *,
    target_payload: Mapping[str, Any],
    broker_payload: Mapping[str, Any],
    previous_payload: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("signal_score", "signal_rank", "sector", "rationale", "entry_date"):
        for payload in (target_payload, broker_payload, previous_payload):
            if key in payload and payload[key] is not None:
                metadata[key] = payload[key]
                break
    return metadata


def _build_proposal_id(
    *,
    symbol: str,
    side: str,
    target_weight: float,
    current_weight: float,
    qty: float,
    est_price: float,
) -> str:
    digest = sha256()
    digest.update(
        "|".join(
            [
                symbol,
                side,
                f"{target_weight:.10f}",
                f"{current_weight:.10f}",
                f"{qty:.8f}",
                f"{est_price:.8f}",
            ]
        ).encode("utf-8")
    )
    return digest.hexdigest()[:24]
