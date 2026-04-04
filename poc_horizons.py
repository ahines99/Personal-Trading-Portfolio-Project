"""
poc_horizons.py
---------------
Quick POC: test forward_window = 1, 5, 10, 21 with reduced windows.

Uses quarterly retraining (retrain_freq=63) instead of monthly (21) = 3x faster.
Skips MLP to save GPU time (~15s/window saved).
Loads cached feature panel (no rebuild).

Estimated: ~50 min total for all 4 horizons on RTX 4070 Super.

Usage:
    python poc_horizons.py
"""

import hashlib
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_prices, get_close, get_returns, get_volume, get_sectors
from features import build_composite_signal, realized_volatility, momentum
from model import build_labels, WalkForwardModel
from portfolio import build_monthly_portfolio
from backtest import run_backtest, TransactionCostModel
from metrics import sharpe_ratio, annualized_return, max_drawdown, sortino_ratio, annualized_volatility
from regime import detect_combined_regime, build_market_series

CACHE_DIR = Path("data/cache")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS = [1, 5, 10, 21]
ML_BLENDS = [0.70]          # Just test the default blend for now
RETRAIN_FREQ = 63            # Quarterly = 51 windows vs 151
MLP_WEIGHT = 0.0             # Skip MLP to save time (20% weight -> 0)
# Redistribute MLP weight equally to other 3 models
LGBM_WEIGHT = 0.375          # 0.30 + 0.05
XGB_WEIGHT = 0.3125          # 0.25 + 0.0625
RIDGE_WEIGHT = 0.3125        # 0.25 + 0.0625

START = "2013-01-01"
END = "2026-03-01"


def load_data():
    """Load data + feature panel from cache."""
    print("=" * 60)
    print("  LOADING CACHED DATA")
    print("=" * 60)

    # Load prices
    prices = load_prices(start=START, end=END)
    close = get_close(prices)
    returns = get_returns(prices)
    volume = get_volume(prices)
    sector_map = get_sectors(close.columns.tolist())

    # Signals for portfolio construction
    composite, ranked_signals = build_composite_signal(
        close, returns, volume, sector_map=sector_map, use_ic_weights=True,
    )
    rvol = realized_volatility(returns, window=21)
    mom_63d = momentum(close, 63)
    adv_30d = (close * volume).rolling(30, min_periods=10).mean()

    # Regime
    mkt_price, mkt_return = build_market_series(prices)
    regime = detect_combined_regime(mkt_return, mkt_price)

    # SPY benchmark
    try:
        spy_raw = load_prices(["SPY"], start=START, end=END, use_cache=True)
        spy_ret = get_returns(spy_raw)
        if isinstance(spy_ret, pd.DataFrame):
            spy_series = spy_ret.iloc[:, 0] if "SPY" not in spy_ret.columns else spy_ret["SPY"]
            if isinstance(spy_series, pd.DataFrame):
                spy_series = spy_series.iloc[:, 0]
        else:
            spy_series = spy_ret
    except Exception:
        spy_series = pd.Series(0.0, index=returns.index)

    # Load cached feature panel
    import glob
    panel_files = sorted(glob.glob(str(CACHE_DIR / "feature_panel_*.pkl")),
                         key=lambda f: Path(f).stat().st_mtime, reverse=True)
    if not panel_files:
        raise FileNotFoundError("No cached feature panel found! Run run_strategy.py first.")

    print(f"  Loading feature panel: {Path(panel_files[0]).name}")
    t0 = time.time()
    with open(panel_files[0], "rb") as f:
        panel = pickle.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s — shape {panel.shape}")

    return {
        "prices": prices, "close": close, "returns": returns, "volume": volume,
        "sector_map": sector_map, "rvol": rvol, "mom_63d": mom_63d,
        "adv_30d": adv_30d, "regime": regime, "spy_series": spy_series,
        "panel": panel,
    }


def run_horizon_test(data, forward_window, ml_blend=0.70):
    """Train ML at given horizon, blend with momentum, run backtest."""
    mom_blend = 1.0 - ml_blend
    returns = data["returns"]
    close = data["close"]
    panel = data["panel"]
    sector_map = data["sector_map"]

    # Build labels for this horizon
    print(f"\n  Building labels (fwd_window={forward_window})...")
    labels = build_labels(returns, forward_window=forward_window, sector_map=sector_map)

    # Walk-forward ML (quarterly retraining, no MLP)
    wf = WalkForwardModel(
        min_train_days=252,
        retrain_freq=RETRAIN_FREQ,
        forward_window=forward_window,
        lgbm_weight=LGBM_WEIGHT,
        xgb_weight=XGB_WEIGHT,
        ridge_weight=RIDGE_WEIGHT,
        mlp_weight=MLP_WEIGHT,
    )

    t0 = time.time()
    ml_signal = wf.fit_predict(panel, labels)
    train_time = time.time() - t0
    ml_smooth = ml_signal.rolling(5, min_periods=1).mean()

    # Momentum blend
    mom_12_1 = close.shift(21).pct_change(252)
    mom_ranked = mom_12_1.rank(axis=1, pct=True)
    mom_aligned = mom_ranked.reindex(index=ml_smooth.index, columns=ml_smooth.columns)
    final_signal = ml_blend * ml_smooth + mom_blend * mom_aligned.fillna(0.5)
    final_signal = final_signal.rank(axis=1, pct=True)

    # Compute IC at this horizon
    # Compare signal with actual forward returns
    fwd_ret = returns.shift(-forward_window).rolling(forward_window).sum()
    common_dates = final_signal.index.intersection(fwd_ret.index)
    daily_ics = []
    for d in common_dates[-500:]:  # Last 500 days for speed
        sig = final_signal.loc[d].dropna()
        ret = fwd_ret.loc[d].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) > 50:
            ic = sig[common].corr(ret[common], method="spearman")
            if np.isfinite(ic):
                daily_ics.append(ic)
    mean_ic = np.mean(daily_ics) if daily_ics else np.nan
    ic_std = np.std(daily_ics) if daily_ics else np.nan

    # Portfolio construction (same params for all tests)
    quality_signal = None
    try:
        from alt_features import build_alt_features
    except Exception:
        pass

    weights, rebalance_dates = build_monthly_portfolio(
        signal=final_signal,
        n_positions=20,
        weighting="signal",
        concentration=1.0,
        max_weight=0.10,
        min_weight=0.02,
        sector_map=sector_map,
        max_sector_pct=0.35,
        momentum_filter=data["mom_63d"],
        realized_vol=data["rvol"],
        returns=returns,
        regime=data["regime"],
        adv=data["adv_30d"],
        cash_in_bear=0.15,
    )

    # Backtest
    cost_model = TransactionCostModel(spread_bps=3.0, commission_bps=0.0, slippage_bps=2.0)
    result = run_backtest(
        weights, data["prices"],
        initial_capital=100_000,
        cost_model=cost_model,
        rebalance_dates=set(rebalance_dates),
        adv=data["adv_30d"],
        stop_loss_pct=0.15,
    )

    r = result.daily_returns
    eq = result.equity_curve
    spy = data["spy_series"].reindex(r.index).fillna(0)

    return {
        "forward_window": forward_window,
        "ml_blend": ml_blend,
        "sharpe": sharpe_ratio(r),
        "ann_return": annualized_return(r),
        "ann_vol": annualized_volatility(r),
        "sortino": sortino_ratio(r),
        "max_dd": max_drawdown(eq),
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "final_nav": eq.iloc[-1],
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": mean_ic / ic_std if ic_std and ic_std > 0 else np.nan,
        "train_time_min": train_time / 60,
        "n_windows": len(wf.models_),
    }


def main():
    data = load_data()

    results = []
    total_start = time.time()

    print("\n" + "=" * 60)
    print(f"  POC HORIZON TEST: {len(HORIZONS)} horizons x {len(ML_BLENDS)} blends")
    print(f"  Retrain freq: {RETRAIN_FREQ}d (quarterly)")
    print(f"  MLP: OFF (LGBM {LGBM_WEIGHT:.1%} / XGB {XGB_WEIGHT:.1%} / Ridge {RIDGE_WEIGHT:.1%})")
    print("=" * 60)

    for fw in HORIZONS:
        for blend in ML_BLENDS:
            print(f"\n{'─'*60}")
            print(f"  TESTING: forward_window={fw}d, ml_blend={blend:.0%}")
            print(f"{'─'*60}")

            t0 = time.time()
            metrics = run_horizon_test(data, forward_window=fw, ml_blend=blend)
            elapsed = time.time() - t0

            results.append(metrics)

            print(f"\n  RESULT: fwd={fw}d | "
                  f"Sharpe={metrics['sharpe']:.3f} | "
                  f"CAGR={metrics['ann_return']*100:.1f}% | "
                  f"MaxDD={metrics['max_dd']*100:.1f}% | "
                  f"IC={metrics['mean_ic']:.4f} | "
                  f"ICIR={metrics['icir']:.3f} | "
                  f"NAV=${metrics['final_nav']:,.0f} | "
                  f"Time={elapsed/60:.1f}min")

    # ── Summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    total_time = (time.time() - total_start) / 60
    print("\n" + "=" * 60)
    print(f"  POC RESULTS (total time: {total_time:.0f} min)")
    print("=" * 60)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    summary_cols = ["forward_window", "sharpe", "ann_return", "ann_vol", "sortino",
                    "max_dd", "total_return", "final_nav", "mean_ic", "icir", "train_time_min"]
    print(df[summary_cols].to_string(index=False))

    # Save
    out_file = RESULTS_DIR / "poc_horizon_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\n  Results saved to {out_file}")

    # ── Key insight ──────────────────────────────────────────────────────
    best = df.iloc[0]
    worst = df.iloc[-1]
    print(f"\n  BEST:  fwd={int(best['forward_window'])}d → Sharpe={best['sharpe']:.3f}, CAGR={best['ann_return']*100:.1f}%")
    print(f"  WORST: fwd={int(worst['forward_window'])}d → Sharpe={worst['sharpe']:.3f}, CAGR={worst['ann_return']*100:.1f}%")

    if best["forward_window"] != 21:
        print(f"\n  ** CONFIRMED: Changing forward_window from 21d to {int(best['forward_window'])}d improves Sharpe by "
              f"{best['sharpe'] - df[df['forward_window']==21]['sharpe'].values[0]:.3f} **")


if __name__ == "__main__":
    main()
