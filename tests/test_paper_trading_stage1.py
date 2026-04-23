from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper.controller import PaperTradingController
from paper import loaders


def test_load_target_book_rebalance_calls_builder_and_prefers_exact_history(tmp_path, monkeypatch):
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    called = {}

    def fake_builder(**kwargs):
        called["kwargs"] = kwargs
        signal = kwargs["signal"]
        weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        weights.loc[:, "AAA"] = 0.7
        weights.loc[:, "BBB"] = 0.3
        return weights, [as_of_date]

    monkeypatch.setattr(loaders.portfolio_module, "build_monthly_portfolio", fake_builder)

    target, prior, metadata = loaders.load_target_book(
        as_of_date=as_of_date,
        config={"baseline_path": str(baseline_dir), "signal_max_staleness_days": 365},
        repo_root=repo_root,
    )

    assert "kwargs" in called
    assert metadata["builder_called"] is True
    assert metadata["is_rebalance_day"] is True
    assert metadata["weight_source"] == "weights_history_exact"
    assert pytest.approx(target["AAA"], rel=1e-9) == 0.60
    assert pytest.approx(target["BBB"], rel=1e-9) == 0.40
    assert pytest.approx(prior["AAA"], rel=1e-9) == 0.55


def test_load_target_book_non_rebalance_holds_state(tmp_path):
    repo_root, baseline_dir, _ = _build_fake_repo(tmp_path)
    state_dir = repo_root / "paper_trading" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "portfolio_state.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-01-05",
                "weights": {"AAA": 0.50, "BBB": 0.50},
            }
        ),
        encoding="utf-8",
    )

    target, prior, metadata = loaders.load_target_book(
        as_of_date="2026-01-05",
        config={"baseline_path": str(baseline_dir), "signal_max_staleness_days": 365},
        repo_root=repo_root,
    )

    assert metadata["is_rebalance_day"] is False
    assert metadata["weight_source"] == "state_hold"
    assert pytest.approx(target["AAA"], rel=1e-9) == 0.50
    assert pytest.approx(prior["BBB"], rel=1e-9) == 0.50


def test_controller_run_daily_dry_run_writes_bundle_without_state(tmp_path):
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    controller = PaperTradingController(
        config={
            "baseline_path": str(baseline_dir),
            "signal_max_staleness_days": 365,
            "dry_run": True,
            "results_dir": "results",
            "notional": 100_000,
        },
        repo_root=repo_root,
    )

    result = controller.run_daily(as_of_date=as_of_date)

    bundle_dir = Path(result["bundle_dir"])
    assert bundle_dir.exists()
    assert (bundle_dir / "target_weights.csv").exists()
    assert (bundle_dir / "intended_trades.csv").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert result["n_intents"] >= 1
    assert not (repo_root / "paper_trading" / "state" / "portfolio_state.json").exists()


def test_controller_run_daily_persists_state_when_not_dry_run(tmp_path):
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    controller = PaperTradingController(
        config={
            "baseline_path": str(baseline_dir),
            "signal_max_staleness_days": 365,
            "dry_run": False,
            "results_dir": "results",
            "notional": 100_000,
        },
        repo_root=repo_root,
    )

    controller.run_daily(as_of_date=as_of_date)

    state_path = repo_root / "paper_trading" / "state" / "portfolio_state.json"
    blotter_path = repo_root / "paper_trading" / "history" / "blotter.jsonl"
    assert state_path.exists()
    assert blotter_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["as_of_date"] == "2026-01-06"
    assert "weights" in payload


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo_root = tmp_path / "repo"
    results_dir = repo_root / "results" / "_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "paper").mkdir(parents=True, exist_ok=True)
    (repo_root / "CURRENT_BASELINE.md").write_text(
        "# Current Baseline Status\n- Current adopted clean canonical baseline: `results/_baseline`\n",
        encoding="utf-8",
    )

    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    signal = pd.DataFrame(
        {
            "AAA": [0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
            "BBB": [0.80, 0.81, 0.82, 0.83, 0.84, 0.85],
            "CCC": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
        },
        index=dates,
    )
    signal.to_parquet(results_dir / "final_signal.parquet")

    returns = pd.DataFrame(
        {
            "AAA": [0.0, 0.01, -0.01, 0.02, 0.00, 0.01],
            "BBB": [0.0, 0.00, 0.01, -0.01, 0.01, 0.00],
            "CCC": [0.0, -0.01, 0.00, 0.01, -0.01, 0.00],
        },
        index=dates,
    )
    returns.to_parquet(results_dir / "returns_panel.parquet")

    weights_history = pd.DataFrame(
        {
            "AAA": [0.55, 0.55, 0.60, 0.60, 0.60, 0.60],
            "BBB": [0.45, 0.45, 0.40, 0.40, 0.40, 0.40],
            "CCC": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    weights_history.to_parquet(results_dir / "weights_history.parquet")

    run_config = {
        "args": {
            "n_positions": 2,
            "weighting": "signal",
            "concentration": 1.0,
            "max_weight": 0.60,
            "max_sector_pct": 1.0,
            "cash_in_bear": 0.0,
            "use_vol_buckets": False,
            "max_selection_pool": 100,
            "spy_core": 0.0,
            "spy_ticker": "SPY",
            "force_mega_caps": False,
            "signal_smooth_halflife": 0.0,
            "apply_rank_normal": False,
            "min_holding_overlap": 0.0,
            "mid_month_refresh": False,
            "min_adv_for_selection": 0.0,
            "max_stock_vol": 1.0,
            "quality_percentile": 0.0,
            "quality_tilt": 0.0,
            "vol_target": 0.0,
            "no_vol_target": True,
            "max_leverage": 1.0,
            "min_leverage": 1.0,
            "vol_floor": 0.08,
            "vol_ceiling": 0.25,
            "credit_overlay": False,
            "use_sector_map": False,
            "bsc_scaling": False,
        }
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    return repo_root, results_dir, "2026-01-06"
