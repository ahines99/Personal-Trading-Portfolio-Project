from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .loaders import check_signal_staleness, load_target_book, resolve_baseline_path


class PaperTradingController:
    def __init__(self, config: Any | None = None, repo_root: str | Path | None = None):
        self.config = _coerce_config(config)
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        self.logger = logging.getLogger("paper.stage1")

    def run_daily(self, as_of_date: str | pd.Timestamp | None = None) -> dict[str, Any]:
        as_of_ts = _resolve_as_of_date(as_of_date)
        self.logger.info("stage1.start as_of_date=%s", as_of_ts.strftime("%Y-%m-%d"))

        preflight = self._check_signal_staleness(as_of_ts)
        self.logger.info(
            "stage1.preflight status=%s signal_date=%s age_days=%s",
            preflight["status"],
            preflight["signal_date"],
            preflight["age_days"],
        )

        broker_preflight = self._check_broker_connectivity()
        if broker_preflight is not None:
            preflight["broker"] = broker_preflight
            self.logger.info(
                "stage2.preflight status=%s broker=%s account_id=%s positions=%s",
                broker_preflight["status"],
                broker_preflight["broker"],
                broker_preflight["account_id"],
                broker_preflight["position_count"],
            )

        target_weights, prior_weights, metadata = self._load_target_book(as_of_ts)
        self.logger.info(
            "stage1.target loaded rebalance=%s source=%s positions=%s",
            metadata.get("is_rebalance_day"),
            metadata.get("weight_source"),
            metadata.get("target_positions_count"),
        )

        intended_trades = self._compute_intended_trades(target_weights, prior_weights, metadata)
        self.logger.info("stage1.intents generated n_intents=%s", len(intended_trades))

        bundle_dir, manifest = self._write_daily_bundle(
            target_weights=target_weights,
            intended_trades=intended_trades,
            as_of_date=as_of_ts,
            git_state=_git_state(self.repo_root),
            metadata=metadata,
            preflight=preflight,
            prior_weights=prior_weights,
        )
        self.logger.info("stage1.bundle written bundle_dir=%s", bundle_dir)

        if not self._is_dry_run():
            self._append_blotter_entry(
                as_of_date=as_of_ts,
                target_weights=target_weights,
                prior_weights=prior_weights,
                intended_trades=intended_trades,
                bundle_dir=bundle_dir,
                manifest=manifest,
                metadata=metadata,
            )
            self.logger.info("stage1.persistence updated")
        else:
            self.logger.info("stage1.persistence skipped dry_run=true")

        return {
            "bundle_dir": str(bundle_dir),
            "n_intents": int(len(intended_trades)),
            "bundle_manifest": manifest,
        }

    def _check_signal_staleness(self, as_of_date: pd.Timestamp) -> dict[str, Any]:
        try:
            from .preflight import check_signal_staleness as external_preflight

            return external_preflight(
                as_of_date=as_of_date,
                results_dir=self.config.get("baseline_path"),
                config=self.config,
            )
        except Exception:
            return check_signal_staleness(
                as_of_date=as_of_date,
                baseline_path=self.config.get("baseline_path"),
                config=self.config,
                repo_root=self.repo_root,
            )

    def _load_target_book(self, as_of_date: pd.Timestamp) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
        return load_target_book(
            as_of_date=as_of_date,
            baseline_path=self.config.get("baseline_path"),
            config=self.config,
            repo_root=self.repo_root,
        )

    def _check_broker_connectivity(self) -> dict[str, Any] | None:
        broker = str(self.config.get("broker") or "mock").strip().lower()
        stage = int(self.config.get("stage") or 1)
        if broker == "mock" or stage < 2:
            return None

        from .brokerage.factory import create_broker_client

        client = create_broker_client(self.config)
        if client is None:
            return None
        if not client.ping():
            raise RuntimeError(
                f"Broker connectivity preflight failed for broker={broker} "
                f"account_id={self.config.get('account_id')}"
            )
        return {
            "status": "ok",
            "broker": broker,
            "account_id": str(self.config.get("account_id") or ""),
        }

    def _compute_intended_trades(
        self,
        target_weights: pd.Series,
        prior_weights: pd.Series,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        try:
            from .generators import compute_intended_trades as external_generator

            return external_generator(
                target_weights=target_weights,
                prior_holdings=prior_weights,
                notional=float(self.config.get("notional", 100_000)),
            )
        except Exception:
            return _fallback_compute_intended_trades(
                target_weights=target_weights,
                prior_weights=prior_weights,
                metadata=metadata,
                notional=float(self.config.get("notional", 100_000)),
                min_trade_notional=float(self.config.get("min_trade_notional", 100.0)),
                max_single_order_notional=self.config.get("max_single_order_notional"),
            )

    def _write_daily_bundle(
        self,
        target_weights: pd.Series,
        intended_trades: pd.DataFrame,
        as_of_date: pd.Timestamp,
        git_state: dict[str, Any],
        metadata: dict[str, Any],
        preflight: dict[str, Any],
        prior_weights: pd.Series,
    ) -> tuple[Path, dict[str, Any]]:
        try:
            from .writers import write_daily_bundle as external_writer

            bundle_dir = external_writer(
                target_weights=target_weights,
                intended_trades=intended_trades,
                as_of_date=as_of_date,
                git_state=git_state,
                repo_root=self.repo_root,
            )
            manifest_path = Path(bundle_dir) / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = {"bundle_dir": str(bundle_dir)}
            manifest["preflight"] = preflight
            manifest["metadata"] = metadata
            manifest["dry_run"] = bool(self._is_dry_run())
            _atomic_write_json(manifest_path, manifest)
            return Path(bundle_dir), manifest
        except Exception:
            return _fallback_write_daily_bundle(
                repo_root=self.repo_root,
                results_root=self.config.get("results_dir"),
                as_of_date=as_of_date,
                target_weights=target_weights,
                prior_weights=prior_weights,
                intended_trades=intended_trades,
                git_state=git_state,
                metadata=metadata,
                preflight=preflight,
                dry_run=self._is_dry_run(),
            )

    def _append_blotter_entry(
        self,
        as_of_date: pd.Timestamp,
        target_weights: pd.Series,
        prior_weights: pd.Series,
        intended_trades: pd.DataFrame,
        bundle_dir: Path,
        manifest: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            from .persistence import append_blotter_entry as external_append

            external_append(
                as_of_date=as_of_date,
                intended_trades=intended_trades,
                bundle_dir=bundle_dir,
                repo_root=self.repo_root,
            )
            external_persisted = True
        except Exception:
            external_persisted = False

        history_dir = self.repo_root / "paper_trading" / "history"
        state_dir = self.repo_root / "paper_trading" / "state"
        history_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        blotter_path = history_dir / "blotter.jsonl"
        if not external_persisted:
            entry = {
                "date": as_of_date.strftime("%Y-%m-%d"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "manifest_hash": manifest.get("hashes", {}),
                "metadata": {
                    "is_rebalance_day": bool(metadata.get("is_rebalance_day")),
                    "weight_source": metadata.get("weight_source"),
                    "build_params_hash": metadata.get("build_params_hash"),
                },
                "trades": intended_trades.to_dict(orient="records"),
                "total_buy_notional": float(intended_trades.loc[intended_trades["delta_notional"] > 0, "delta_notional"].sum()),
                "total_sell_notional": float((-intended_trades.loc[intended_trades["delta_notional"] < 0, "delta_notional"]).sum()),
            }
            with blotter_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

        state_payload = {
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_path": str(resolve_baseline_path(self.config, self.config.get("baseline_path"), self.repo_root)),
            "weights": {
                ticker: float(weight)
                for ticker, weight in target_weights[target_weights.abs() > 1e-10].sort_values(ascending=False).items()
            },
            "prior_weights": {
                ticker: float(weight)
                for ticker, weight in prior_weights[prior_weights.abs() > 1e-10].sort_values(ascending=False).items()
            },
            "bundle_dir": str(bundle_dir),
            "last_manifest": manifest,
            "last_metadata": metadata,
        }
        _atomic_write_json(state_dir / "portfolio_state.json", state_payload)

    def _is_dry_run(self) -> bool:
        return bool(self.config.get("dry_run", False))


def _fallback_compute_intended_trades(
    target_weights: pd.Series,
    prior_weights: pd.Series,
    metadata: dict[str, Any],
    notional: float,
    min_trade_notional: float,
    max_single_order_notional: float | None,
) -> pd.DataFrame:
    aligned_target = target_weights.fillna(0.0)
    aligned_prior = prior_weights.reindex(aligned_target.index).fillna(0.0)
    reference_prices = metadata.get("reference_prices", {})

    rows: list[dict[str, Any]] = []
    for ticker in aligned_target.index:
        prior_weight = float(aligned_prior.get(ticker, 0.0))
        target_weight = float(aligned_target.get(ticker, 0.0))
        delta_weight = target_weight - prior_weight
        delta_notional = delta_weight * notional
        if abs(delta_notional) < min_trade_notional:
            continue
        if max_single_order_notional is not None and abs(delta_notional) > float(max_single_order_notional):
            raise ValueError(
                f"Trade for {ticker} exceeds max_single_order_notional: "
                f"{abs(delta_notional):.2f} > {float(max_single_order_notional):.2f}"
            )
        reference_price = reference_prices.get(ticker)
        shares_to_trade = None
        if reference_price and reference_price > 0:
            shares_to_trade = delta_notional / float(reference_price)
        rows.append(
            {
                "ticker": ticker,
                "action": "BUY" if delta_notional > 0 else "SELL",
                "prior_weight": prior_weight,
                "target_weight": target_weight,
                "prior_notional": prior_weight * notional,
                "target_notional": target_weight * notional,
                "delta_notional": delta_notional,
                "shares_to_trade": shares_to_trade,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "prior_weight",
                "target_weight",
                "prior_notional",
                "target_notional",
                "delta_notional",
                "shares_to_trade",
            ]
        )

    frame["abs_delta_notional"] = frame["delta_notional"].abs()
    frame["sort_bucket"] = frame["action"].map({"BUY": 0, "SELL": 1})
    frame = frame.sort_values(["sort_bucket", "abs_delta_notional"], ascending=[True, False]).drop(
        columns=["abs_delta_notional", "sort_bucket"]
    )
    return frame.reset_index(drop=True)


def _fallback_write_daily_bundle(
    repo_root: Path,
    results_root: str | Path | None,
    as_of_date: pd.Timestamp,
    target_weights: pd.Series,
    prior_weights: pd.Series,
    intended_trades: pd.DataFrame,
    git_state: dict[str, Any],
    metadata: dict[str, Any],
    preflight: dict[str, Any],
    dry_run: bool,
) -> tuple[Path, dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    base_dir = Path(results_root) if results_root else repo_root / "results"
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = base_dir / f"_paper_shadow_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    target_frame = (
        pd.DataFrame({"ticker": target_weights.index, "target_weight": target_weights.values})
        .query("target_weight != 0")
        .sort_values("target_weight", ascending=False)
        .reset_index(drop=True)
    )
    prior_frame = (
        pd.DataFrame({"ticker": prior_weights.index, "prior_weight": prior_weights.values})
        .query("prior_weight != 0")
        .sort_values("prior_weight", ascending=False)
        .reset_index(drop=True)
    )

    target_path = bundle_dir / "target_weights.csv"
    prior_path = bundle_dir / "prior_weights.csv"
    trades_path = bundle_dir / "intended_trades.csv"
    manifest_path = bundle_dir / "manifest.json"

    target_frame.to_csv(target_path, index=False)
    prior_frame.to_csv(prior_path, index=False)
    intended_trades.to_csv(trades_path, index=False)

    manifest = {
        "timestamp": now_utc.isoformat(),
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "dry_run": bool(dry_run),
        "git": git_state,
        "preflight": preflight,
        "metadata": metadata,
        "stats": {
            "n_target_positions": int(len(target_frame)),
            "n_prior_positions": int(len(prior_frame)),
            "n_intended_trades": int(len(intended_trades)),
            "total_buy_notional": float(intended_trades.loc[intended_trades["delta_notional"] > 0, "delta_notional"].sum())
            if not intended_trades.empty
            else 0.0,
            "total_sell_notional": float((-intended_trades.loc[intended_trades["delta_notional"] < 0, "delta_notional"]).sum())
            if not intended_trades.empty
            else 0.0,
        },
        "hashes": {
            "target_weights": _sha256_file(target_path),
            "prior_weights": _sha256_file(prior_path),
            "intended_trades": _sha256_file(trades_path),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    return bundle_dir, manifest


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    head = _run_git("rev-parse", "HEAD")
    message = _run_git("log", "-1", "--pretty=%s")
    dirty_files = _run_git("status", "--porcelain").splitlines()
    return {
        "head": head,
        "message": message,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:50],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


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


def _resolve_as_of_date(as_of_date: str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of_date is None:
        return pd.Timestamp.utcnow().normalize()
    return pd.Timestamp(as_of_date).normalize()
