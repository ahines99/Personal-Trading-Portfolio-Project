from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import run_paper_phase_a  # noqa: E402


def test_run_phase_a_writes_pending_intents_and_order_blotter(tmp_path: Path) -> None:
    repo_root, baseline_dir, as_of_date = _build_fake_repo(tmp_path)
    result = run_paper_phase_a.run_phase_a(
        config={
            "stage": 3,
            "broker": "mock",
            "capital_mode": "paper",
            "account_id": "PAPER-12345",
            "baseline_path": str(baseline_dir),
            "signal_max_staleness_days": 365,
            "results_dir": "results",
            "portfolio_notional_usd": 100000.0,
            "min_trade_notional_usd": 50.0,
            "allowed_order_types": ["market"],
            "approval_deadline_time_et": "08:00",
        },
        repo_root=repo_root,
        as_of_date=as_of_date,
    )

    bundle_dir = Path(result["bundle_dir"])
    intents = json.loads((bundle_dir / "intents.json").read_text(encoding="utf-8"))
    assert intents["status"] == "AWAITING_APPROVAL"
    assert intents["rebalance_id"]
    assert intents["proposed_orders"]
    assert (bundle_dir / "approval.template.json").exists()

    orders_path = repo_root / "paper_trading" / "blotter" / "orders.jsonl"
    assert orders_path.exists()
    orders = [json.loads(line) for line in orders_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest_by_order = {row["order_id"]: row for row in orders}
    assert len(latest_by_order) == len(intents["proposed_orders"])
    assert {row["status"] for row in latest_by_order.values()} == {"APPROVAL_PENDING"}


def _build_fake_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo_root = tmp_path / "repo"
    results_dir = repo_root / "results" / "_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
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
