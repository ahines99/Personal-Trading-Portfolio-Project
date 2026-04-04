"""
test_horizons.py
----------------
Academically-grounded test suite for forward_window, signal design,
and portfolio construction optimization.

THEORY & MOTIVATION
-------------------

Problem diagnosis:
  - 21d IC is NEGATIVE (-0.0046). The ML model is anti-predictive at our
    holding period. More ML = worse returns.
  - Feature importance dominated by volatility features (top 10 are all vol).
    The model is essentially learning the low-vol anomaly (Ang et al., 2006),
    which works short-term but mean-reverts — hence the IC decay.
  - 12-1 month momentum alone (0/100 blend) gives Sharpe 0.433, better than
    any ML-weighted blend.

Academic grounding for tests:

  1. HORIZON ALIGNMENT — IC is positive at 1-5d but negative at 21d.
     Train ML at the horizon where it actually predicts.
     (Baz et al., 2015 "Dissecting Investment Strategies in the Cross Section")

  2. VOL-SCALED MOMENTUM — Barroso & Santa-Clara (2015):
     mom / trailing_vol reduces drawdowns by 50%+ during momentum crashes.
     We risk-adjust ML labels but NOT the momentum component.

  3. MULTI-HORIZON ENSEMBLE — DeMiguel et al. (2020):
     Blend predictions from multiple horizons with IC-proportional weights.
     Diversifies across horizon-specific patterns.

  4. REBALANCE FREQUENCY — If signal works at 5d, monthly rebalance
     captures ~25% of the signal. Weekly rebalance matches the horizon.
     Cost difference: ~1.4% annual (acceptable if signal improves >2%).

  5. LABEL DESIGN — Current: fwd_return / trailing_vol (risk-adjusted).
     With vol-dominated features, this double-doses on vol sorting.
     Test raw returns to see if the model learns different patterns.
     (Lopez de Prado, 2018 "Advances in Financial Machine Learning")

  6. MOMENTUM VARIANT — Novy-Marx (2012): 7-month momentum is the
     strongest single component. Test 126d vs 252d windows.

  7. FEATURE ABLATION — If the model is just vol-sorting, dropping vol
     features should reveal whether remaining features have independent IC.
     (Gu, Kelly, Xiu, 2020 "Empirical Asset Pricing via Machine Learning")

PHASES
------
  POC (~45 min):
    Phase 1 — Horizon scan: fw=1,3,5,10,21 at quarterly retrain
    Phase 2 — Vol-scaled momentum (no retrain, just modify blend)

  OVERNIGHT (~6-8 hours):
    Phase 3 — Blend sweep on best horizon
    Phase 4 — Multi-horizon ensemble
    Phase 5 — Weekly rebalance at best horizon
    Phase 6 — Raw labels (no risk-adjust) at best horizon
    Phase 7 — Feature ablation (drop vol features, retrain)
    Phase 8 — Final cross of all winners

Usage:
    python test_horizons.py                 # POC only (~45 min)
    python test_horizons.py --overnight     # full suite (~6-8 hrs)
    python test_horizons.py --phase 1       # specific phase
"""

import argparse
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
from model import build_labels, build_feature_matrix, WalkForwardModel
from portfolio import build_monthly_portfolio
from backtest import run_backtest, TransactionCostModel
from metrics import (sharpe_ratio, annualized_return, max_drawdown,
                     sortino_ratio, annualized_volatility, calmar_ratio)
from regime import detect_combined_regime, build_market_series

CACHE_DIR = Path("data/cache")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "test_horizons_results.csv"

START = "2013-01-01"
END   = "2026-03-01"


# ─────────────────────────────────────────────────────────────────────────────
# Model config for POC (quarterly, no MLP) vs overnight (monthly, full)
# ─────────────────────────────────────────────────────────────────────────────

POC_CONFIG = dict(
    retrain_freq=63,      # quarterly = 51 windows
    lgbm_weight=0.375,    # redistribute MLP share
    xgb_weight=0.3125,
    ridge_weight=0.3125,
    mlp_weight=0.0,
    # New defaults from audit:
    feature_fraction=0.5,   # was 0.7, breaks vol cluster dominance
    bagging_fraction=0.7,   # was 0.8, more diverse trees
    max_feature_corr=0.70,  # was 0.85, tighter dedup
)

# VOL-NEUTRALIZED variant
POC_NEUTRAL = dict(
    **POC_CONFIG,
    neutralize_vol=True,    # regress out vol from all features
)

# ROLLING 3-YEAR variant
POC_ROLLING = dict(
    **POC_CONFIG,
    max_train_days=756,     # ~3 years rolling (drops stale data)
)

FULL_CONFIG = dict(
    retrain_freq=21,      # monthly = 151 windows
    lgbm_weight=0.30,
    xgb_weight=0.25,
    ridge_weight=0.25,
    mlp_weight=0.20,
    feature_fraction=0.5,
    bagging_fraction=0.7,
    max_feature_corr=0.70,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    prices = load_prices(start=START, end=END)
    close = get_close(prices)
    returns = get_returns(prices)
    volume = get_volume(prices)
    sector_map = get_sectors(close.columns.tolist())

    composite, ranked_signals = build_composite_signal(
        close, returns, volume, sector_map=sector_map, use_ic_weights=True)
    rvol = realized_volatility(returns, window=21)
    mom_63d = momentum(close, 63)
    adv_30d = (close * volume).rolling(30, min_periods=10).mean()

    mkt_price, mkt_return = build_market_series(prices)
    regime = detect_combined_regime(mkt_return, mkt_price)

    # SPY
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

    # Feature panel from cache
    import glob
    panels = sorted(glob.glob(str(CACHE_DIR / "feature_panel_*.pkl")),
                    key=lambda f: Path(f).stat().st_mtime, reverse=True)
    if not panels:
        raise FileNotFoundError("No cached feature panel. Run run_strategy.py first.")
    print(f"  Loading feature panel: {Path(panels[0]).name}")
    t0 = time.time()
    with open(panels[0], "rb") as f:
        panel = pickle.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s — {panel.shape}")

    return {
        "prices": prices, "close": close, "returns": returns, "volume": volume,
        "sector_map": sector_map, "rvol": rvol, "mom_63d": mom_63d,
        "adv_30d": adv_30d, "regime": regime, "spy_series": spy_series,
        "panel": panel, "ranked_signals": ranked_signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal generators
# ─────────────────────────────────────────────────────────────────────────────

def train_ml_signal(data, forward_window, config, risk_adjust=True,
                    drop_vol_features=False):
    """Train ML model, return raw ranked ML signal (no momentum blend)."""
    panel = data["panel"]

    # Feature ablation: drop volatility features
    if drop_vol_features:
        vol_prefixes = ["rvol_", "rank_rvol", "idiovol_", "rank_idiovol",
                        "vol_of_vol", "rank_vol_of_vol", "vol_regime",
                        "rank_vol_regime", "tail_risk", "rank_tail_risk",
                        "vol_reversal", "rank_vol_reversal", "vol_spike",
                        "z_rvol_", "z_idiovol_", "max_ret_", "rank_max_effect"]
        drop_cols = [c for c in panel.columns
                     if any(c.startswith(p) for p in vol_prefixes)]
        if drop_cols:
            panel = panel.drop(columns=drop_cols, errors="ignore")
            print(f"  [ablation] Dropped {len(drop_cols)} vol features, {len(panel.columns)} remaining")

    labels = build_labels(data["returns"], forward_window=forward_window,
                          risk_adjust=risk_adjust, sector_map=data["sector_map"])

    wf = WalkForwardModel(
        min_train_days=252,
        forward_window=forward_window,
        **config,
    )
    ml_signal = wf.fit_predict(panel, labels)
    return ml_signal.rolling(5, min_periods=1).mean(), wf


def build_momentum_signal(close, window=252, skip=21, vol_scale=False,
                          returns=None):
    """
    Build momentum signal.

    Standard: Jegadeesh-Titman 12-1 month (skip 1 month, look back 12).
    Vol-scaled: Barroso & Santa-Clara (2015) — mom / trailing_vol.
    """
    raw_mom = close.shift(skip).pct_change(window)

    if vol_scale and returns is not None:
        # Trailing 63-day realized vol, annualized
        trailing_vol = returns.rolling(63, min_periods=21).std() * np.sqrt(252)
        raw_mom = raw_mom / trailing_vol.replace(0, np.nan)

    return raw_mom.rank(axis=1, pct=True)


def blend_signals(ml_signal, mom_signal, ml_blend, orthogonalize=False):
    """Blend ML and momentum signals.

    If orthogonalize=True, residualize momentum against ML first
    (Ehsani & Linnainmaa 2019). The ML model already captures momentum
    through its features — residualizing extracts the "pure momentum not
    already in ML" component, avoiding double-counting.
    """
    mom_blend = 1.0 - ml_blend
    if ml_blend < 0.001:
        aligned = mom_signal.reindex(index=ml_signal.index, columns=ml_signal.columns)
        return aligned.fillna(0.5).rank(axis=1, pct=True)

    mom_aligned = mom_signal.reindex(index=ml_signal.index, columns=ml_signal.columns).fillna(0.5)

    if orthogonalize and ml_blend > 0.001:
        # Cross-sectional residualization per date: remove the part of
        # momentum already captured by the ML signal
        mom_resid = mom_aligned.copy()
        for date in ml_signal.index:
            ml_row = ml_signal.loc[date].values
            mom_row = mom_aligned.loc[date].values
            mask = np.isfinite(ml_row) & np.isfinite(mom_row)
            if mask.sum() < 50:
                continue
            x, y = ml_row[mask], mom_row[mask]
            beta = np.dot(x - x.mean(), y - y.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-10)
            mom_resid.loc[date] = mom_row - beta * ml_row
        mom_aligned = mom_resid.rank(axis=1, pct=True)

    raw = ml_blend * ml_signal + mom_blend * mom_aligned
    return raw.rank(axis=1, pct=True)


def build_multi_horizon_signal(ml_signals, ics, mom_signal, ml_blend=0.70):
    """
    IC-weighted blend of ML predictions from multiple horizons.
    DeMiguel et al. (2020): diversify across horizon-specific patterns.

    ml_signals: dict {fw: ml_signal_df}
    ics: dict {fw: mean_ic}
    """
    # Only include horizons with positive IC
    valid = {fw: ic for fw, ic in ics.items() if ic > 0}
    if not valid:
        # Fall back to equal weight of all
        valid = {fw: 1.0 for fw in ml_signals}

    total_ic = sum(valid.values())
    weights = {fw: ic / total_ic for fw, ic in valid.items()}

    # Find common index
    ref = list(ml_signals.values())[0]
    blended = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for fw, w in weights.items():
        if fw in ml_signals:
            aligned = ml_signals[fw].reindex(index=ref.index, columns=ref.columns).fillna(0.5)
            blended += w * aligned

    blended_ranked = blended.rank(axis=1, pct=True)
    return blend_signals(blended_ranked, mom_signal, ml_blend)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio + backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_test(name, data, signal, n_positions=20, max_weight=0.10,
             cash_in_bear=0.15, use_momentum_filter=True,
             weekly_rebalance=False):
    """Run portfolio + backtest, return metrics dict."""
    t0 = time.time()
    mom_filter = data["mom_63d"] if use_momentum_filter else None

    try:
        # Build weights
        weights, rebalance_dates = build_monthly_portfolio(
            signal=signal,
            n_positions=n_positions,
            weighting="signal",
            concentration=1.0,
            max_weight=max_weight,
            min_weight=0.02,
            sector_map=data["sector_map"],
            max_sector_pct=0.35,
            momentum_filter=mom_filter,
            realized_vol=data["rvol"],
            returns=data["returns"],
            regime=data["regime"],
            adv=data["adv_30d"],
            cash_in_bear=cash_in_bear,
        )

        # Weekly rebalance override: add every 5th trading day
        if weekly_rebalance:
            all_dates = signal.index
            weekly_dates = set(all_dates[::5])
            rebalance_dates = rebalance_dates | weekly_dates

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

        # Compute IC for the final blended signal
        fwd_ret_21d = data["returns"].shift(-21).rolling(21).sum()
        common = signal.index.intersection(fwd_ret_21d.index)
        daily_ics = []
        for d in common[-500:]:
            sig_d = signal.loc[d].dropna()
            ret_d = fwd_ret_21d.loc[d].dropna()
            shared = sig_d.index.intersection(ret_d.index)
            if len(shared) > 50:
                ic = sig_d[shared].corr(ret_d[shared], method="spearman")
                if np.isfinite(ic):
                    daily_ics.append(ic)
        signal_ic = np.mean(daily_ics) if daily_ics else np.nan

        # OOS (2024+)
        oos_mask = r.index >= "2024-01-01"
        if oos_mask.sum() > 60:
            oos_r = r[oos_mask]
            oos_sharpe = oos_r.mean() / oos_r.std() * np.sqrt(252) if oos_r.std() > 0 else 0
            oos_cagr = (1 + oos_r).prod() ** (252 / len(oos_r)) - 1
        else:
            oos_sharpe, oos_cagr = np.nan, np.nan

        # Costs and turnover
        total_costs = result.transaction_costs.sum() if hasattr(result, 'transaction_costs') else 0
        ann_turnover = result.turnover.sum() / (len(r) / 252) if hasattr(result, 'turnover') and len(r) > 0 else 0

        metrics = {
            "test": name,
            "sharpe": sharpe_ratio(r),
            "ann_return": annualized_return(r),
            "ann_vol": annualized_volatility(r),
            "sortino": sortino_ratio(r),
            "calmar": calmar_ratio(r, eq),
            "max_dd": max_drawdown(eq),
            "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
            "final_nav": eq.iloc[-1],
            "signal_ic_21d": signal_ic,
            "oos_sharpe": oos_sharpe,
            "oos_cagr": oos_cagr,
            "total_costs": total_costs,
            "ann_turnover": ann_turnover,
            "elapsed_min": (time.time() - t0) / 60,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        metrics = {"test": name, "sharpe": np.nan, "error": str(e)[:200],
                   "elapsed_min": (time.time() - t0) / 60}

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Save / resume / print
# ─────────────────────────────────────────────────────────────────────────────

def load_existing():
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        return set(df["test"].tolist()), df
    return set(), pd.DataFrame()

def save_result(metrics, df):
    new = pd.DataFrame([metrics])
    combined = pd.concat([df, new], ignore_index=True) if not df.empty else new
    combined.to_csv(RESULTS_FILE, index=False)
    return combined

def pr(m):
    """Print result line."""
    s = m.get("sharpe", np.nan)
    r = m.get("ann_return", np.nan)
    dd = m.get("max_dd", np.nan)
    nav = m.get("final_nav", np.nan)
    oos = m.get("oos_sharpe", np.nan)
    ic = m.get("signal_ic_21d", np.nan)
    t = m.get("elapsed_min", 0)
    err = m.get("error", "")
    if err:
        print(f"    ERROR: {err}")
    else:
        print(f"    Sharpe={s:.3f}  CAGR={r*100:.1f}%  DD={dd*100:.1f}%  "
              f"NAV=${nav:,.0f}  IC={ic:.4f}  OOS={oos:.3f}  ({t:.1f}m)")

def skip_or_run(name, done):
    if name in done:
        print(f"\n  [{name}] CACHED")
        return True
    print(f"\n  [{name}]", end=" ", flush=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# POC PHASES (quick, ~45 min total)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_horizon_scan(data, done, df):
    """
    PHASE 1 — HORIZON ALIGNMENT (Baz et al., 2015)
    ------------------------------------------------
    Train ML at fw=1,3,5,10,21 with same blend (70/30).
    This isolates the horizon effect: does matching the label horizon
    to the IC peak (1-5d) make ML additive instead of destructive?

    Quarterly retrain (51 windows) for speed.
    ~12 min per horizon, ~60 min total.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: HORIZON SCAN")
    print("  Theory: Baz et al. (2015) — train at horizon where IC is positive")
    print("  Current: fw=21d has IC=-0.0046, fw=1d has IC=+0.0167")
    print("=" * 60)

    # Standard 12-1 month momentum for blending
    mom_signal = build_momentum_signal(data["close"])
    ml_signals = {}

    for fw in [1, 3, 5, 10, 21]:
        name = f"P1_fw{fw}d"
        if skip_or_run(name, done):
            continue

        print(f"Training ML at {fw}d horizon...", flush=True)
        ml_sig, _ = train_ml_signal(data, fw, POC_CONFIG)
        ml_signals[fw] = ml_sig
        signal = blend_signals(ml_sig, mom_signal, ml_blend=0.70)

        m = run_test(name, data, signal)
        pr(m)
        df = save_result(m, df)
        done.add(name)

    return done, df, ml_signals


def phase2_vol_scaled_momentum(data, done, df):
    """
    PHASE 2 — VOL-SCALED MOMENTUM (Barroso & Santa-Clara, 2015)
    -------------------------------------------------------------
    Standard momentum crashes (-80% in 2009). Vol-scaling the momentum
    component by trailing vol halves drawdowns with minimal return loss.

    We risk-adjust ML labels but NOT the momentum blend component.
    This tests 3 momentum variants, all with 0% ML (pure momentum).
    No retraining needed — instant.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: MOMENTUM VARIANTS")
    print("  Theory: Barroso & Santa-Clara (2015) — vol-scaled momentum")
    print("          Novy-Marx (2012) — 7-month momentum is strongest")
    print("=" * 60)

    close = data["close"]
    returns = data["returns"]
    rvol = data["rvol"]

    # Get ML signal from cache for blending (use fw=5 or fw=21 if available)
    import glob
    ml_sig = None
    for fw_try in [5, 21, 10, 1]:
        name_try = f"P1_fw{fw_try}d"
        # Check if prediction is cached by WalkForwardModel
        wf_test = WalkForwardModel(min_train_days=252, forward_window=fw_try, **POC_CONFIG)
        cache_key = hashlib.md5(
            f"{data['panel'].shape}_{data['panel'].index.get_level_values('date').min()}_"
            f"{data['panel'].index.get_level_values('date').max()}_"
            f"{len(data['panel'].columns)}_{wf_test.min_train_days}_{wf_test.retrain_freq}_"
            f"{wf_test.forward_window}_{wf_test.lgbm_weight}_{wf_test.xgb_weight}_"
            f"{wf_test.ridge_weight}_{wf_test.mlp_weight}_{wf_test.num_leaves}_"
            f"{wf_test.learning_rate}_{wf_test.prune_features}".encode()
        ).hexdigest()[:12]
        cache_file = CACHE_DIR / f"ml_predictions_{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            ml_sig = cached["predictions"].rolling(5, min_periods=1).mean()
            print(f"  Using cached ML signal from fw={fw_try}d")
            break

    tests = [
        # (name, mom_window, mom_skip, vol_scale, ml_blend)
        ("P2_mom_12_1_raw",        252, 21, False, 0.0),    # Standard JT momentum
        ("P2_mom_12_1_volscaled",  252, 21, True,  0.0),    # Barroso vol-scaled
        ("P2_mom_7_1_raw",         147, 21, False, 0.0),    # Novy-Marx 7-month
        ("P2_mom_7_1_volscaled",   147, 21, True,  0.0),    # 7-month vol-scaled
        ("P2_mom_6_1_raw",         126, 21, False, 0.0),    # 6-month
    ]

    # Also test blending vol-scaled momentum with ML (if available)
    if ml_sig is not None:
        tests.extend([
            ("P2_ml70_mom_volscaled",  252, 21, True,  0.70),
            ("P2_ml50_mom_volscaled",  252, 21, True,  0.50),
            ("P2_ml30_mom_volscaled",  252, 21, True,  0.30),
        ])

    for name, window, skip, vol_scale, ml_blend in tests:
        if skip_or_run(name, done):
            continue

        print(f"mom_window={window}d, vol_scale={vol_scale}, ml={ml_blend:.0%}...", flush=True)
        mom = build_momentum_signal(close, window=window, skip=skip,
                                    vol_scale=vol_scale, returns=returns)

        if ml_blend > 0.001 and ml_sig is not None:
            signal = blend_signals(ml_sig, mom, ml_blend)
        else:
            # Pure momentum — use ML signal index for alignment
            ref_idx = data["panel"].index.get_level_values("date").unique()
            signal = mom.reindex(index=ref_idx).rank(axis=1, pct=True)

        m = run_test(name, data, signal, use_momentum_filter=False)
        pr(m)
        df = save_result(m, df)
        done.add(name)

    return done, df


# ─────────────────────────────────────────────────────────────────────────────
# OVERNIGHT PHASES (comprehensive, ~6-8 hours)
# ─────────────────────────────────────────────────────────────────────────────

def phase3_blend_sweep(data, done, df):
    """
    PHASE 3 — BLEND SWEEP ON BEST HORIZON
    Reuses cached ML predictions. ~2 min per test.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: BLEND SWEEP ON BEST HORIZON")
    print("=" * 60)

    p1 = df[df["test"].str.startswith("P1_")].dropna(subset=["sharpe"])
    if p1.empty:
        print("  Need Phase 1 results first.")
        return done, df

    best = p1.loc[p1["sharpe"].idxmax()]
    best_fw = int(best["test"].replace("P1_fw", "").replace("d", ""))
    print(f"  Best horizon: {best_fw}d (Sharpe={best['sharpe']:.3f})")

    ml_sig, _ = train_ml_signal(data, best_fw, POC_CONFIG)

    # Test both standard and vol-scaled momentum
    for mom_type, vol_scale in [("std", False), ("vs", True)]:
        mom = build_momentum_signal(data["close"], vol_scale=vol_scale,
                                    returns=data["returns"])
        for ml_pct in [90, 70, 50, 30, 10, 0]:
            name = f"P3_fw{best_fw}_ml{ml_pct}_{mom_type}"
            if skip_or_run(name, done):
                continue
            print(f"ML={ml_pct}%, mom={mom_type}...", flush=True)
            signal = blend_signals(ml_sig, mom, ml_pct / 100)
            m = run_test(name, data, signal, use_momentum_filter=False)
            pr(m)
            df = save_result(m, df)
            done.add(name)

    return done, df


def phase4_multi_horizon(data, done, df):
    """
    PHASE 4 — MULTI-HORIZON ENSEMBLE (DeMiguel et al., 2020)
    Blend ML predictions from horizons with positive IC.
    Weight by IC magnitude.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: MULTI-HORIZON ENSEMBLE")
    print("  Theory: DeMiguel et al. (2020) — IC-weighted horizon blend")
    print("=" * 60)

    # Known IC values from our factor_decay analysis
    known_ics = {1: 0.0167, 3: 0.012, 5: 0.0107, 10: 0.0029, 21: -0.0046}

    # Train all horizons we need (will use cache)
    ml_signals = {}
    for fw in [1, 3, 5, 10]:
        print(f"  Loading/training fw={fw}d...")
        ml_sig, _ = train_ml_signal(data, fw, POC_CONFIG)
        ml_signals[fw] = ml_sig

    mom_std = build_momentum_signal(data["close"])
    mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

    # Multi-horizon ML blended with different momentum types
    for mom_type, mom_sig in [("std", mom_std), ("vs", mom_vs)]:
        for ml_pct in [70, 50, 30]:
            name = f"P4_multi_ml{ml_pct}_{mom_type}"
            if skip_or_run(name, done):
                continue
            print(f"IC-weighted multi-horizon, ML={ml_pct}%, mom={mom_type}...", flush=True)
            signal = build_multi_horizon_signal(
                ml_signals, known_ics, mom_sig, ml_blend=ml_pct/100)
            m = run_test(name, data, signal, use_momentum_filter=False)
            pr(m)
            df = save_result(m, df)
            done.add(name)

    return done, df


def phase5_weekly_rebalance(data, done, df):
    """
    PHASE 5 — WEEKLY REBALANCE
    If signal works at 5d, monthly rebalance wastes 75% of it.
    Test weekly rebal with best signals. Higher turnover but captures signal.
    """
    print("\n" + "=" * 60)
    print("  PHASE 5: WEEKLY REBALANCE")
    print("=" * 60)

    # Use best signal from earlier phases
    p_all = df[df["sharpe"].notna()].sort_values("sharpe", ascending=False)
    if p_all.empty:
        print("  Need earlier phase results first.")
        return done, df

    # Test weekly rebal with a few promising signals
    for fw in [5, 1]:
        ml_sig, _ = train_ml_signal(data, fw, POC_CONFIG)
        mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

        for ml_pct in [70, 50, 0]:
            name = f"P5_weekly_fw{fw}_ml{ml_pct}"
            if skip_or_run(name, done):
                continue
            print(f"Weekly rebal, fw={fw}d, ML={ml_pct}%...", flush=True)
            signal = blend_signals(ml_sig, mom_vs, ml_pct / 100)
            m = run_test(name, data, signal, weekly_rebalance=True,
                         use_momentum_filter=False, cash_in_bear=0.0)
            pr(m)
            df = save_result(m, df)
            done.add(name)

    return done, df


def phase6_raw_labels(data, done, df):
    """
    PHASE 6 — RAW LABELS (no risk-adjustment)
    Current: fwd_return / trailing_vol (double-doses vol with vol features).
    Test raw returns to see if model learns different patterns.
    """
    print("\n" + "=" * 60)
    print("  PHASE 6: RAW LABELS (no risk-adjustment)")
    print("  Theory: de Prado (2018) — label design affects what model learns")
    print("=" * 60)

    mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

    for fw in [5, 10]:
        name = f"P6_rawlabel_fw{fw}"
        if skip_or_run(name, done):
            continue
        print(f"Raw labels, fw={fw}d...", flush=True)
        ml_sig, _ = train_ml_signal(data, fw, POC_CONFIG, risk_adjust=False)
        signal = blend_signals(ml_sig, mom_vs, 0.70)
        m = run_test(name, data, signal, use_momentum_filter=False)
        pr(m)
        df = save_result(m, df)
        done.add(name)

    return done, df


def phase7_feature_ablation(data, done, df):
    """
    PHASE 7 — FEATURE ABLATION
    Drop all volatility features. If remaining features have better
    long-horizon IC, the model was just vol-sorting.
    (Gu, Kelly, Xiu, 2020)
    """
    print("\n" + "=" * 60)
    print("  PHASE 7: FEATURE ABLATION (drop vol features)")
    print("  Theory: Gu, Kelly, Xiu (2020) — feature set diversity matters")
    print("=" * 60)

    mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

    for fw in [5, 10, 21]:
        name = f"P7_novol_fw{fw}"
        if skip_or_run(name, done):
            continue
        print(f"No vol features, fw={fw}d...", flush=True)
        ml_sig, _ = train_ml_signal(data, fw, POC_CONFIG, drop_vol_features=True)
        signal = blend_signals(ml_sig, mom_vs, 0.70)
        m = run_test(name, data, signal, use_momentum_filter=False)
        pr(m)
        df = save_result(m, df)
        done.add(name)

    return done, df


def phase8_model_improvements(data, done, df):
    """
    PHASE 8 — MODEL IMPROVEMENTS FROM AUDIT
    Test vol-neutralization, rolling window, and combined at best horizon.
    These are the top 3 code-level changes identified by the research.
    """
    print("\n" + "=" * 60)
    print("  PHASE 8: MODEL IMPROVEMENTS")
    print("  8a: Vol-neutralization (de Prado / WorldQuant)")
    print("  8b: Rolling 3yr window (drops stale regime data)")
    print("  8c: Combined (neutral + rolling)")
    print("=" * 60)

    mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

    for fw in [5, 10]:
        # 8a: Vol-neutralized
        name = f"P8a_neutral_fw{fw}"
        if not skip_or_run(name, done):
            print(f"Vol-neutral, fw={fw}d...", flush=True)
            ml_sig, _ = train_ml_signal(data, fw, POC_NEUTRAL)
            signal = blend_signals(ml_sig, mom_vs, 0.70)
            m = run_test(name, data, signal, use_momentum_filter=False)
            pr(m)
            df = save_result(m, df)
            done.add(name)

        # 8b: Rolling 3yr
        name = f"P8b_rolling_fw{fw}"
        if not skip_or_run(name, done):
            print(f"Rolling 3yr, fw={fw}d...", flush=True)
            ml_sig, _ = train_ml_signal(data, fw, POC_ROLLING)
            signal = blend_signals(ml_sig, mom_vs, 0.70)
            m = run_test(name, data, signal, use_momentum_filter=False)
            pr(m)
            df = save_result(m, df)
            done.add(name)

        # 8c: Combined (neutral + rolling)
        combined_cfg = dict(**POC_CONFIG, neutralize_vol=True, max_train_days=756)
        name = f"P8c_neutral_rolling_fw{fw}"
        if not skip_or_run(name, done):
            print(f"Neutral+rolling, fw={fw}d...", flush=True)
            ml_sig, _ = train_ml_signal(data, fw, combined_cfg)
            signal = blend_signals(ml_sig, mom_vs, 0.70)
            m = run_test(name, data, signal, use_momentum_filter=False)
            pr(m)
            df = save_result(m, df)
            done.add(name)

    return done, df


def phase9_final_combos(data, done, df):
    """
    PHASE 9 — FINAL CROSS OF ALL WINNERS
    Combine best horizon + best momentum + best portfolio params +
    best model improvements.
    """
    print("\n" + "=" * 60)
    print("  PHASE 9: FINAL COMBINATIONS")
    print("=" * 60)

    all_done = df[df["sharpe"].notna()].sort_values("sharpe", ascending=False)
    if all_done.empty:
        return done, df

    print("  Top 5 so far:")
    for _, r in all_done.head(5).iterrows():
        print(f"    {r['test']:40s}  Sharpe={r['sharpe']:.3f}  CAGR={r['ann_return']*100:.1f}%")

    # Portfolio param variants from B+C findings
    portfolio_configs = [
        ("base",    dict()),
        ("15pos",   dict(n_positions=15)),
        ("opt",     dict(n_positions=15, max_weight=0.15, cash_in_bear=0.0)),
    ]

    mom_vs = build_momentum_signal(data["close"], vol_scale=True, returns=data["returns"])

    # Test the best model configs against portfolio variants
    model_configs = [
        ("default", POC_CONFIG),
        ("neutral", POC_NEUTRAL),
        ("rolling", POC_ROLLING),
        ("nr",      dict(**POC_CONFIG, neutralize_vol=True, max_train_days=756)),
    ]

    for fw in [5]:
        for model_name, model_cfg in model_configs:
            ml_sig, _ = train_ml_signal(data, fw, model_cfg)
            for ml_pct in [70, 50, 30]:
                # Standard blend + orthogonalized blend
                for orth, orth_label in [(False, ""), (True, "_orth")]:
                    signal = blend_signals(ml_sig, mom_vs, ml_pct / 100,
                                           orthogonalize=orth)
                    for port_name, port_kw in portfolio_configs:
                        name = f"P9_fw{fw}_{model_name}_ml{ml_pct}{orth_label}_{port_name}"
                        if skip_or_run(name, done):
                            continue
                        print(f"fw{fw} {model_name} ml{ml_pct}{orth_label} {port_name}...", flush=True)
                        m = run_test(name, data, signal, use_momentum_filter=False, **port_kw)
                        pr(m)
                        df = save_result(m, df)
                        done.add(name)

    return done, df


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    if not RESULTS_FILE.exists():
        return
    df = pd.read_csv(RESULTS_FILE)
    df = df.drop_duplicates(subset="test", keep="last")
    df = df.dropna(subset=["sharpe"])
    df = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 60)
    print("  FULL RESULTS (sorted by Sharpe)")
    print("=" * 60)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    cols = ["test", "sharpe", "ann_return", "ann_vol", "max_dd", "final_nav",
            "signal_ic_21d", "oos_sharpe", "oos_cagr", "total_costs", "elapsed_min"]
    avail = [c for c in cols if c in df.columns]
    print("\n" + df[avail].to_string(index=False))

    best = df.iloc[0]
    spy_cagr = 0.133
    alpha = best["ann_return"] - spy_cagr

    print(f"\n{'='*60}")
    print(f"  BEST CONFIG: {best['test']}")
    print(f"{'='*60}")
    print(f"  Sharpe:      {best['sharpe']:.3f}")
    print(f"  CAGR:        {best['ann_return']*100:.1f}%")
    print(f"  Max DD:      {best['max_dd']*100:.1f}%")
    print(f"  Final NAV:   ${best['final_nav']:,.0f}")
    if pd.notna(best.get("oos_sharpe")):
        print(f"  OOS Sharpe:  {best['oos_sharpe']:.3f}")
    if pd.notna(best.get("oos_cagr")):
        print(f"  OOS CAGR:    {best['oos_cagr']*100:.1f}%")
    print(f"\n  vs SPY:      {alpha*100:+.1f}% ({'BEATS SPY' if alpha > 0 else 'BELOW SPY'})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overnight", action="store_true",
                        help="Run full overnight suite (phases 1-8)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-8). 0 = POC only.")
    args = parser.parse_args()

    data = load_data()
    done, df = load_existing()
    total_t0 = time.time()

    if args.phase > 0:
        # Run specific phase
        phases = {
            1: phase1_horizon_scan,
            2: phase2_vol_scaled_momentum,
            3: phase3_blend_sweep,
            4: phase4_multi_horizon,
            5: phase5_weekly_rebalance,
            6: phase6_raw_labels,
            7: phase7_feature_ablation,
            8: phase8_model_improvements,
            9: phase9_final_combos,
        }
        fn = phases.get(args.phase)
        if fn:
            result = fn(data, done, df)
            if len(result) == 3:
                done, df, _ = result
            else:
                done, df = result
    elif args.overnight:
        # Full overnight: all 9 phases
        done, df, ml_signals = phase1_horizon_scan(data, done, df)
        done, df = phase2_vol_scaled_momentum(data, done, df)
        done, df = phase3_blend_sweep(data, done, df)
        done, df = phase4_multi_horizon(data, done, df)
        done, df = phase5_weekly_rebalance(data, done, df)
        done, df = phase6_raw_labels(data, done, df)
        done, df = phase7_feature_ablation(data, done, df)
        done, df = phase8_model_improvements(data, done, df)
        done, df = phase9_final_combos(data, done, df)
    else:
        # POC: phases 1-2 only
        done, df, ml_signals = phase1_horizon_scan(data, done, df)
        done, df = phase2_vol_scaled_momentum(data, done, df)

    total_min = (time.time() - total_t0) / 60
    print(f"\n  Total time: {total_min:.0f} min")
    print_summary()


if __name__ == "__main__":
    main()
