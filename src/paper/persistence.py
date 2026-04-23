from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _repo_root(explicit_root: Optional[Path] = None) -> Path:
    return Path(explicit_root).resolve() if explicit_root else Path(__file__).resolve().parents[2]


def _paper_root(repo_root: Optional[Path] = None) -> Path:
    return _repo_root(repo_root) / "paper_trading"


def _history_path(repo_root: Optional[Path] = None) -> Path:
    return _paper_root(repo_root) / "history" / "blotter.jsonl"


def _current_state_path(repo_root: Optional[Path] = None) -> Path:
    return _paper_root(repo_root) / "current" / "state.json"


def _ensure_layout(repo_root: Optional[Path] = None) -> None:
    _history_path(repo_root).parent.mkdir(parents=True, exist_ok=True)
    _current_state_path(repo_root).parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _coerce_trades_frame(intended_trades: Any) -> pd.DataFrame:
    if intended_trades is None:
        return pd.DataFrame()
    if isinstance(intended_trades, pd.DataFrame):
        return intended_trades.copy()
    if isinstance(intended_trades, dict):
        return pd.DataFrame([intended_trades])
    if isinstance(intended_trades, Iterable) and not isinstance(intended_trades, (str, bytes)):
        return pd.DataFrame(list(intended_trades))
    raise TypeError(f"Unsupported intended_trades type: {type(intended_trades)!r}")


def load_current_state(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    path = _current_state_path(repo_root)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_blotter_history(limit: Optional[int] = None, repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = _history_path(repo_root)
    if not path.is_file():
        return []

    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if limit == 0:
        return []
    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def append_blotter_entry(
    as_of_date: Any,
    intended_trades: Any,
    bundle_dir: Any,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a shadow-trade entry and refresh the current state snapshot."""
    _ensure_layout(repo_root)
    trades = _coerce_trades_frame(intended_trades)
    if not trades.empty:
        trades = trades.copy()
        for column in (
            "prior_weight",
            "target_weight",
            "prior_notional",
            "target_notional",
            "delta_notional",
            "shares_to_trade",
        ):
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")

    total_buy_notional = 0.0
    total_sell_notional = 0.0
    if "delta_notional" in trades.columns:
        delta = trades["delta_notional"].fillna(0.0)
        total_buy_notional = float(delta[delta > 0].sum())
        total_sell_notional = float((-delta[delta < 0]).sum())

    event_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle_value = str(Path(bundle_dir))
    entry = {
        "date": str(pd.Timestamp(as_of_date).date()),
        "timestamp": event_timestamp,
        "bundle_dir": bundle_value,
        "trades": trades.where(pd.notna(trades), None).to_dict(orient="records"),
        "total_buy_notional": total_buy_notional,
        "total_sell_notional": total_sell_notional,
    }

    blotter_path = _history_path(repo_root)
    with blotter_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")

    current_state = load_current_state(repo_root)
    positions = dict(current_state.get("positions") or {})
    for trade in entry["trades"]:
        ticker = str(trade.get("ticker", "")).strip()
        if not ticker:
            continue
        target_weight = float(trade.get("target_weight") or 0.0)
        target_notional = float(trade.get("target_notional") or 0.0)
        if abs(target_weight) < 1e-12 and abs(target_notional) < 1e-6:
            positions.pop(ticker, None)
            continue
        position_payload = {
            "weight": target_weight,
            "notional": target_notional,
            "updated_at": event_timestamp,
        }
        shares_value = trade.get("shares_to_trade")
        if shares_value is not None:
            try:
                position_payload["shares_last_trade"] = float(shares_value)
            except (TypeError, ValueError):
                pass
        positions[ticker] = position_payload

    gross_notional = float(sum(abs(float(item.get("notional", 0.0))) for item in positions.values()))
    snapshot = {
        "as_of_date": entry["date"],
        "updated_at": event_timestamp,
        "bundle_dir": bundle_value,
        "positions": dict(sorted(positions.items())),
        "n_positions": int(len(positions)),
        "gross_notional": gross_notional,
    }
    _atomic_write_json(_current_state_path(repo_root), snapshot)
    return entry
