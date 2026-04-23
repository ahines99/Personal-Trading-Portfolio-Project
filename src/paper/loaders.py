from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src import portfolio as portfolio_module
    from src.features import momentum as compute_momentum
    from src.features import realized_volatility
except ModuleNotFoundError:  # pragma: no cover - supports paper-style imports
    import portfolio as portfolio_module
    from features import momentum as compute_momentum
    from features import realized_volatility


def load_target_book(
    as_of_date: str | pd.Timestamp | None = None,
    baseline_path: str | Path | None = None,
    config: Any | None = None,
    repo_root: str | Path | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    repo_root_path = _resolve_repo_root(repo_root)
    config_dict = _coerce_config(config)
    baseline_dir = resolve_baseline_path(
        config=config_dict,
        baseline_path=baseline_path,
        repo_root=repo_root_path,
    )

    signal = _load_signal_frame(baseline_dir)
    if signal.empty:
        raise ValueError(f"No rows found in {baseline_dir / 'final_signal.parquet'}")

    as_of_ts = _resolve_as_of_date(as_of_date, signal.index.max())
    staleness = _check_signal_staleness(signal.index.max(), as_of_ts, config_dict)
    rebalance_dates = portfolio_module._get_optimal_rebalance_dates(signal.index, day_offset=2)
    is_rebalance_day = as_of_ts in rebalance_dates

    prior_state = _load_portfolio_state(repo_root_path, signal.columns, as_of_date=as_of_ts)
    historical_weights = _load_weights_history(baseline_dir, signal.columns)
    prior_weights = _resolve_prior_weights(
        prior_state=prior_state,
        historical_weights=historical_weights,
        as_of_date=as_of_ts,
        universe=signal.columns,
    )

    metadata: dict[str, Any] = {
        "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "signal_date": signal.index.max().strftime("%Y-%m-%d"),
        "baseline_path": str(baseline_dir),
        "is_rebalance_day": bool(is_rebalance_day),
        "signal_age_days": int(staleness["age_days"]),
        "signal_max_staleness_days": int(staleness["max_allowed_days"]),
        "builder_called": False,
        "weight_source": "state_hold" if not prior_weights.empty else "empty",
        "reference_prices_available": False,
        "missing_inputs": [],
    }

    if not is_rebalance_day:
        target_weights = _resolve_hold_weights(
            prior_state=prior_state,
            historical_weights=historical_weights,
            as_of_date=as_of_ts,
            universe=signal.columns,
        )
        metadata["weight_source"] = (
            "state_hold"
            if prior_state.get("weights")
            else "weights_history_hold"
            if historical_weights is not None and not historical_weights.empty
            else "empty"
        )
        metadata["target_positions_count"] = int((target_weights > 1e-6).sum())
        metadata["build_params_hash"] = _hash_payload({"hold_only": True, "as_of_date": metadata["as_of_date"]})
        return target_weights, prior_weights, metadata

    run_config = _load_run_config(baseline_dir)
    args = run_config.get("args", {})
    returns_panel = _load_returns_panel(baseline_dir, signal.columns)
    build_context = _build_strategy_context(
        signal=signal,
        returns_panel=returns_panel,
        args=args,
    )
    missing_inputs = build_context.pop("missing_inputs")

    builder_weights = _build_target_weights(signal, build_context, args)
    metadata["builder_called"] = True
    metadata["missing_inputs"] = missing_inputs

    exact_weights = _weights_row_for_date(historical_weights, as_of_ts, signal.columns)
    if exact_weights is not None:
        target_weights = exact_weights
        metadata["weight_source"] = "weights_history_exact"
    else:
        target_weights = _safe_series(builder_weights.loc[as_of_ts], signal.columns)
        metadata["weight_source"] = "rebuilt_from_artifacts"

    reference_prices = build_context.get("reference_prices")
    if isinstance(reference_prices, pd.Series):
        metadata["reference_prices_available"] = bool(reference_prices.notna().any())
        metadata["reference_prices"] = {
            ticker: float(price)
            for ticker, price in reference_prices.dropna().items()
        }

    build_params = {
        "args": _subset_build_args(args),
        "missing_inputs": missing_inputs,
        "weight_source": metadata["weight_source"],
    }
    metadata["build_params_hash"] = _hash_payload(build_params)
    metadata["target_positions_count"] = int((target_weights > 1e-6).sum())
    metadata["artifact_paths"] = {
        "final_signal": str(baseline_dir / "final_signal.parquet"),
        "run_config": str(baseline_dir / "run_config.json"),
        "returns_panel": str(baseline_dir / "returns_panel.parquet"),
        "weights_history": str(baseline_dir / "weights_history.parquet"),
    }
    return target_weights, prior_weights, metadata


def inspect_rebalance_day(
    as_of_date: str | pd.Timestamp | None = None,
    baseline_path: str | Path | None = None,
    config: Any | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = _resolve_repo_root(repo_root)
    config_dict = _coerce_config(config)
    baseline_dir = resolve_baseline_path(
        config=config_dict,
        baseline_path=baseline_path,
        repo_root=repo_root_path,
    )
    signal = _load_signal_frame(baseline_dir)
    if signal.empty:
        raise ValueError(f"No rows found in {baseline_dir / 'final_signal.parquet'}")
    as_of_ts = _resolve_as_of_date(as_of_date, pd.Timestamp.utcnow().normalize())
    rebalance_dates = portfolio_module._get_optimal_rebalance_dates(signal.index, day_offset=2)
    return {
        "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "signal_date": signal.index.max().strftime("%Y-%m-%d"),
        "is_rebalance_day": bool(as_of_ts in rebalance_dates),
        "baseline_path": str(baseline_dir),
    }


def resolve_baseline_path(
    config: Any | None = None,
    baseline_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    repo_root_path = _resolve_repo_root(repo_root)
    if baseline_path:
        candidate = Path(baseline_path)
        return candidate if candidate.is_absolute() else repo_root_path / candidate

    config_dict = _coerce_config(config)
    configured_path = config_dict.get("baseline_path")
    if configured_path:
        candidate = Path(configured_path)
        return candidate if candidate.is_absolute() else repo_root_path / candidate

    try:
        from .baseline_resolver import resolve_baseline_path as external_resolver

        resolved = external_resolver(config=config, repo_root=repo_root_path)
        if resolved:
            return Path(resolved)
    except Exception:
        pass

    current_baseline = repo_root_path / "CURRENT_BASELINE.md"
    if not current_baseline.exists():
        raise FileNotFoundError(
            "Could not resolve baseline path: CURRENT_BASELINE.md is missing and no baseline_path was provided."
        )

    pattern = re.compile(r"Current adopted clean canonical baseline:\s*`([^`]+)`")
    for line in current_baseline.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            return repo_root_path / match.group(1)

    raise ValueError("Could not parse current baseline path from CURRENT_BASELINE.md")


def check_signal_staleness(
    as_of_date: str | pd.Timestamp | None = None,
    baseline_path: str | Path | None = None,
    config: Any | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = _resolve_repo_root(repo_root)
    config_dict = _coerce_config(config)
    baseline_dir = resolve_baseline_path(
        config=config_dict,
        baseline_path=baseline_path,
        repo_root=repo_root_path,
    )
    signal = _load_signal_frame(baseline_dir)
    if signal.empty:
        raise ValueError(f"No rows found in {baseline_dir / 'final_signal.parquet'}")
    as_of_ts = _resolve_as_of_date(as_of_date, pd.Timestamp.utcnow().normalize())
    return _check_signal_staleness(signal.index.max(), as_of_ts, config_dict)


def _build_target_weights(
    signal: pd.DataFrame,
    context: dict[str, Any],
    args: dict[str, Any],
) -> pd.DataFrame:
    weights, _ = portfolio_module.build_monthly_portfolio(
        signal=signal,
        n_positions=int(args.get("n_positions", 30)),
        weighting=args.get("weighting", "signal"),
        concentration=float(args.get("concentration", 1.0)),
        max_weight=float(args.get("max_weight", 0.10)),
        min_weight=0.02,
        sector_map=context.get("sector_map"),
        max_sector_pct=float(args.get("max_sector_pct", 0.35)),
        momentum_filter=context.get("momentum_filter"),
        realized_vol=context.get("realized_vol"),
        returns=context.get("returns"),
        regime=None,
        adv=context.get("adv"),
        cash_in_bear=float(args.get("cash_in_bear", 0.15)),
        quality_filter=context.get("quality_filter"),
        earnings_dates=context.get("earnings_dates"),
        spy_trend_filter=None,
        use_vol_buckets=bool(args.get("use_vol_buckets", False)),
        max_selection_pool=int(args.get("max_selection_pool", 1500)),
        spy_core_weight=float(args.get("spy_core", 0.0)),
        spy_ticker=args.get("spy_ticker", "SPY"),
        force_mega_caps=bool(args.get("force_mega_caps", False)),
        signal_smooth_halflife=float(args.get("signal_smooth_halflife", 10.0)),
        apply_rank_normal=bool(args.get("apply_rank_normal", True)),
        min_holding_overlap=float(args.get("min_holding_overlap", 0.7)),
        mid_month_refresh=bool(args.get("mid_month_refresh", False)),
        min_adv_for_selection=float(args.get("min_adv_for_selection", 5_000_000)),
        max_stock_vol=float(args.get("max_stock_vol", 0.6)),
        quality_percentile=float(args.get("quality_percentile", 0.7)),
        quality_tilt=float(args.get("quality_tilt", 0.35)),
        rvol=context.get("rvol_63d"),
    )

    if bool(args.get("bsc_scaling", False)):
        weights = portfolio_module.apply_bsc_scaling(
            weights,
            daily_returns=context["returns"],
            target_vol=float(args.get("bsc_target_vol", 0.20)),
            max_leverage=float(args.get("max_leverage", 2.0)),
            min_leverage=float(args.get("bsc_min_leverage", 0.3)),
            lookback=int(args.get("bsc_lookback", 126)),
        )
    else:
        vol_target_active = (not bool(args.get("no_vol_target", False))) and float(args.get("vol_target", 0.0)) > 0
        if vol_target_active:
            weights = portfolio_module.apply_vol_targeting(
                weights,
                realized_vol=context["realized_vol"],
                target_vol=float(args.get("vol_target", 0.40)),
                max_leverage=float(args.get("max_leverage", 2.0)),
                min_leverage=float(args.get("min_leverage", 0.8)),
                vol_floor=float(args.get("vol_floor", 0.08)),
                vol_ceiling=float(args.get("vol_ceiling", 0.25)),
            )

    if bool(args.get("credit_overlay", False)):
        context.setdefault("missing_inputs", []).append("credit_overlay_series")

    return weights


def _build_strategy_context(
    signal: pd.DataFrame,
    returns_panel: pd.DataFrame,
    args: dict[str, Any],
) -> dict[str, Any]:
    returns_panel = returns_panel.reindex(columns=signal.columns).astype("float32")
    pseudo_close = (1.0 + returns_panel.fillna(0.0)).cumprod()

    context: dict[str, Any] = {
        "returns": returns_panel,
        "realized_vol": realized_volatility(returns_panel, window=21),
        "momentum_filter": compute_momentum(pseudo_close, 63),
        "rvol_63d": returns_panel.rolling(63, min_periods=21).std() * np.sqrt(252),
        "reference_prices": pseudo_close.iloc[-1].replace({0.0: np.nan}),
        "sector_map": None,
        "adv": None,
        "quality_filter": None,
        "earnings_dates": None,
        "missing_inputs": [],
    }

    if bool(args.get("use_sector_map", False)):
        try:
            from data_loader import get_sectors

            sectors = get_sectors(signal.columns.tolist())
            if sectors:
                context["sector_map"] = sectors
            else:
                context["missing_inputs"].append("sector_map")
        except Exception:
            context["missing_inputs"].append("sector_map")

    context["missing_inputs"].extend(["adv", "quality_filter", "earnings_dates"])
    return context


def _load_signal_frame(baseline_dir: Path) -> pd.DataFrame:
    signal_path = baseline_dir / "final_signal.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing signal artifact: {signal_path}")
    signal = pd.read_parquet(signal_path)
    signal.index = pd.to_datetime(signal.index).normalize()
    return signal.sort_index()


def _load_returns_panel(baseline_dir: Path, universe: pd.Index) -> pd.DataFrame:
    returns_path = baseline_dir / "returns_panel.parquet"
    if returns_path.exists():
        returns_panel = pd.read_parquet(returns_path)
        returns_panel.index = pd.to_datetime(returns_panel.index).normalize()
        return returns_panel.sort_index().reindex(columns=universe)

    signal = pd.read_parquet(baseline_dir / "final_signal.parquet")
    aligned = signal.reindex(columns=universe).copy()
    aligned.loc[:, :] = 0.0
    return aligned


def _load_run_config(baseline_dir: Path) -> dict[str, Any]:
    config_path = baseline_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_weights_history(baseline_dir: Path, universe: pd.Index) -> pd.DataFrame | None:
    path = baseline_dir / "weights_history.parquet"
    if not path.exists():
        return None
    weights = pd.read_parquet(path)
    weights.index = pd.to_datetime(weights.index).normalize()
    return weights.sort_index().reindex(columns=universe).fillna(0.0)


def _resolve_prior_weights(
    prior_state: dict[str, Any],
    historical_weights: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    universe: pd.Index,
) -> pd.Series:
    if prior_state.get("weights"):
        return _safe_series(pd.Series(prior_state["weights"], dtype="float64"), universe)

    if historical_weights is None or historical_weights.empty:
        return pd.Series(0.0, index=universe, dtype="float64")

    earlier = historical_weights.loc[historical_weights.index < as_of_date]
    if earlier.empty:
        return pd.Series(0.0, index=universe, dtype="float64")
    return _safe_series(earlier.iloc[-1], universe)


def _resolve_hold_weights(
    prior_state: dict[str, Any],
    historical_weights: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    universe: pd.Index,
) -> pd.Series:
    if prior_state.get("weights"):
        return _safe_series(pd.Series(prior_state["weights"], dtype="float64"), universe)

    exact = _weights_row_for_date(historical_weights, as_of_date, universe)
    if exact is not None:
        return exact

    if historical_weights is not None and not historical_weights.empty:
        earlier = historical_weights.loc[historical_weights.index <= as_of_date]
        if not earlier.empty:
            return _safe_series(earlier.iloc[-1], universe)

    return pd.Series(0.0, index=universe, dtype="float64")


def _weights_row_for_date(
    historical_weights: pd.DataFrame | None,
    as_of_date: pd.Timestamp,
    universe: pd.Index,
) -> pd.Series | None:
    if historical_weights is None or historical_weights.empty or as_of_date not in historical_weights.index:
        return None
    return _safe_series(historical_weights.loc[as_of_date], universe)


def _load_portfolio_state(
    repo_root: Path,
    universe: pd.Index,
    *,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    candidates = [
        repo_root / "paper_trading" / "state" / "portfolio_state.json",
        repo_root / "paper_trading" / "current" / "state.json",
    ]
    for state_path in candidates:
        if not state_path.exists():
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        snapshot_as_of = _payload_as_of_date(payload)
        if (
            as_of_date is not None
            and snapshot_as_of is not None
            and snapshot_as_of > as_of_date.normalize()
        ):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
            payload["weights"] = {
                ticker: float(weight)
                for ticker, weight in payload["weights"].items()
                if ticker in universe and abs(float(weight)) > 0
            }
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("positions"), dict):
            payload["weights"] = {
                ticker: float(position.get("weight", 0.0))
                for ticker, position in payload["positions"].items()
                if ticker in universe and abs(float(position.get("weight", 0.0))) > 0
            }
            return payload
    return {}


def _payload_as_of_date(payload: dict[str, Any]) -> pd.Timestamp | None:
    if not isinstance(payload, dict):
        return None
    raw_value = payload.get("as_of_date")
    if raw_value in (None, ""):
        return None
    try:
        return pd.Timestamp(raw_value).normalize()
    except (TypeError, ValueError):
        return None


def _check_signal_staleness(
    signal_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_allowed_days = int(config.get("signal_max_staleness_days", 7))
    age_days = int((as_of_date.normalize() - signal_date.normalize()).days)
    status = "OK" if age_days <= max_allowed_days else "STALE"
    if status == "STALE":
        raise ValueError(
            f"Signal is stale: latest={signal_date.strftime('%Y-%m-%d')} "
            f"as_of={as_of_date.strftime('%Y-%m-%d')} age={age_days}d "
            f"max_allowed={max_allowed_days}d"
        )
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "age_days": age_days,
        "max_allowed_days": max_allowed_days,
        "status": status,
    }


def _subset_build_args(args: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_positions",
        "weighting",
        "concentration",
        "max_weight",
        "max_sector_pct",
        "cash_in_bear",
        "use_vol_buckets",
        "max_selection_pool",
        "spy_core",
        "spy_ticker",
        "force_mega_caps",
        "signal_smooth_halflife",
        "apply_rank_normal",
        "min_holding_overlap",
        "mid_month_refresh",
        "min_adv_for_selection",
        "max_stock_vol",
        "quality_percentile",
        "quality_tilt",
        "bsc_scaling",
        "bsc_target_vol",
        "bsc_min_leverage",
        "bsc_lookback",
        "vol_target",
        "no_vol_target",
        "max_leverage",
        "min_leverage",
        "vol_floor",
        "vol_ceiling",
        "credit_overlay",
    ]
    return {key: args.get(key) for key in keys if key in args}


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _coerce_config(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return {}


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_as_of_date(as_of_date: str | pd.Timestamp | None, default: pd.Timestamp) -> pd.Timestamp:
    if as_of_date is None:
        return pd.Timestamp(default).normalize()
    return pd.Timestamp(as_of_date).normalize()


def _safe_series(series: pd.Series, universe: pd.Index) -> pd.Series:
    aligned = series.reindex(universe).fillna(0.0).astype("float64")
    return aligned
