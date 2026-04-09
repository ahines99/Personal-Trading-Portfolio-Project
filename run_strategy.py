"""
run_strategy.py
---------------
Personal Trading Portfolio — ML-ranked concentrated long-only strategy.

Goal: Beat SPY by 5-10%+ annualized with 15-20 stocks, monthly rebalance.
Designed to be actually traded with personal capital.

Pipeline:
  Phase 1: Load data (3000+ stock universe, alt data)
  Phase 2: Feature engineering + walk-forward ML training
  Phase 3: Long-only portfolio construction (top 20, monthly)
  Phase 4: Backtest with realistic personal-scale costs
  Phase 5: Analysis + interactive dashboard

Usage:
  python run_strategy.py                          # full run
  python run_strategy.py --skip-ml                # rule-based only
  python run_strategy.py --skip-alt-data          # skip SEC/FRED/insider data
  python run_strategy.py --n-positions 15         # fewer positions
  python run_strategy.py --capital 50000          # personal capital size
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader     import load_prices, get_close, get_returns, get_volume, get_sectors
from features        import (build_composite_signal, realized_volatility,
                              factor_decay_analysis, momentum)
from portfolio       import (build_monthly_portfolio, compute_portfolio_stats,
                              get_current_holdings, sector_allocation)
from backtest        import run_backtest, TransactionCostModel
from metrics         import (compute_full_tearsheet, monte_carlo_sharpe,
                              monthly_returns_table, annual_returns, wealth_growth,
                              after_tax_estimate)
from model           import build_feature_matrix, build_labels, WalkForwardModel
from regime          import (detect_combined_regime, build_market_series,
                              performance_by_regime, stress_test)
from robustness      import bootstrap_tearsheet
from dashboard       import build_dashboard
from alt_data_loader import (load_edgar_fundamentals, load_fred_macro,
                              load_vix_term_structure, load_short_interest,
                              load_earnings_calendar, load_insider_transactions,
                              load_analyst_actions, load_earnings_estimates,
                              load_institutional_holders,
                              load_dxy, load_breakeven_inflation, load_vvix,
                              load_oil_wti, load_copper_gold,
                              load_cross_asset_etf_panel, load_ig_oas,
                              load_sector_oas, load_excess_bond_premium,
                              load_treasury_yield_curve)
from alt_features    import build_alt_features
from api_data        import load_all_api_data


def parse_args():
    # Speed-optimized defaults (Tier 9):
    # retrain_freq=63 (quarterly, per Gu-Kelly-Xiu 2020)
    # num_leaves=20 (shallow, per Israel-Kelly-Moskowitz 2020)
    # n_estimators=300 (early stopping catches most windows before 300)
    # max_depth=5 (consistent with num_leaves=20)
    # adversarial_validation=False (AUC=1.0 on temporal splits is uninformative)
    p = argparse.ArgumentParser(description="Personal Trading Portfolio")
    p.add_argument("--start",           default="2013-01-01")
    p.add_argument("--end",             default="2026-03-01")
    p.add_argument("--capital",         default=100_000, type=float,
                   help="Starting capital (default $100K)")
    p.add_argument("--n-positions",     default=20, type=int,
                   help="Number of stocks to hold (default 20)")
    p.add_argument("--weighting",       default="signal",
                   choices=["equal", "signal", "inverse_vol"],
                   help="Position weighting scheme")
    p.add_argument("--concentration",   default=1.0, type=float,
                   help="Signal weighting concentration (0=equal, 0.5=sqrt, 1=linear)")
    p.add_argument("--max-weight",      default=0.10, type=float,
                   help="Max weight per position (default 10%%)")
    p.add_argument("--max-sector-pct",  default=0.35, type=float,
                   help="Max allocation to any single sector (default 35%%)")
    p.add_argument("--skip-ml",         action="store_true")
    p.add_argument("--skip-alt-data",   action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--n-bootstrap",     default=500, type=int)
    p.add_argument("--forward-window",  default=5, type=int,
                   help="Forward return window for ML labels (default 5 = weekly; "
                        "aligns with where alpha decays to — 21d IC is negative, 5d IC_IR=0.062)")
    p.add_argument("--retrain-freq", default=63, type=int,
                   help="ML retraining frequency in trading days (default 63 = quarterly, "
                        "per Gu-Kelly-Xiu 2020; applies to BOTH Optuna search and final fit)")
    p.add_argument("--min-train-days", default=252, type=int,
                   help="Minimum training window for walk-forward (default 252 = 1 year)")
    p.add_argument("--risk-adjust-labels", action="store_true",
                   help="Vol-scale forward returns before ranking labels (default OFF — "
                        "vol-scaling down-weights momentum stocks where alpha lives)")
    p.add_argument("--sector-rank-weight", default=0.0, type=float,
                   help="Weight on within-sector rank in label construction "
                        "(0.0 = pure universe rank, default; 0.5 = legacy 50/50 blend)")
    p.add_argument("--beta-neutral-labels", action="store_true",
                   help="Residualize forward-return labels vs SPY beta (Tier 1 fix for "
                        "beta drag at the LABEL layer; OFF by default). When ON, passes "
                        "market_returns to build_labels and WalkForwardModel.")
    p.add_argument("--stop-loss",       default=0.15, type=float,
                   help="Stop-loss threshold (default 15%%)")
    p.add_argument("--cash-in-bear",    default=0.15, type=float,
                   help="Fraction of portfolio in cash during bear regimes (default 15%%)")
    p.add_argument("--optimize-ml",     action="store_true",
                   help="Run Optuna Bayesian hyperparameter search before final fit")
    p.add_argument("--optuna-trials",   default=40, type=int,
                   help="Number of Optuna trials when --optimize-ml is set (default 40)")
    p.add_argument("--vol-target",      default=0.16, type=float,
                   help="Annualized vol target for portfolio leverage scaling "
                        "(default 0.16 = 16%%, ENABLED by default per Moreira-Muir 2017 "
                        "and DeMiguel et al. JF 2024; pass 0 or --no-vol-target to disable)")
    p.add_argument("--no-vol-target",   action="store_true",
                   help="Disable vol targeting overlay (overrides --vol-target)")
    p.add_argument("--max-leverage",    default=1.3, type=float,
                   help="Max leverage cap for vol targeting (default 1.3 = up to 30%% margin; "
                        "Moreira-Muir requires leverage > 1 to capture the Sharpe lift during "
                        "low-vol regimes. IBKR Reg-T allows 2:1, so 1.3 is conservative)")
    p.add_argument("--use-vol-buckets", action="store_true",
                   help="Use vol-bucket-neutral selection (V4 default, now OFF — suppresses beta)")
    p.add_argument("--max-selection-pool", default=1500, type=int,
                   help="Cap selection candidates to top-N most liquid by ADV (0 = disabled)")
    p.add_argument("--spy-core",        default=0.0, type=float,
                   help="SPY core weight for core+satellite mode (e.g. 0.4 = 40%% SPY, 60%% picks)")
    p.add_argument("--spy-ticker",      default="SPY",
                   help="SPY ticker symbol in the price panel (default 'SPY')")
    p.add_argument("--force-mega-caps", action="store_true",
                   help="Force-include top-10 mega-caps at min 3%% when model ranks them in top 40%%")
    # ── Optuna pruner / warm-start / search controls ─────────────────────
    p.add_argument("--prune-threshold-sharpe", default=0.0, type=float,
                   help="ThresholdPruner: prune trials with intermediate Sharpe below this "
                        "(default 0.0)")
    p.add_argument("--prune-threshold-ic", default=0.2, type=float,
                   help="ThresholdPruner: prune trials with intermediate IC-IR below this "
                        "(default 0.2)")
    p.add_argument("--optuna-warm-start", action=argparse.BooleanOptionalAction, default=True,
                   help="Warm-start Optuna from prior best trials in "
                        "data/cache/optuna_best_params.pkl (default ON)")

    # ── Tier 9: Speed / LGBM tree-structure flags ─────────────────────────
    p.add_argument("--num-leaves", default=20, type=int,
                   help="LightGBM num_leaves (default 20, shallow per Israel-Kelly-Moskowitz 2020)")
    p.add_argument("--n-estimators", default=300, type=int,
                   help="LightGBM n_estimators (default 300; early stopping catches most windows)")
    p.add_argument("--max-depth", default=5, type=int,
                   help="LightGBM max_depth (default 5, consistent with num_leaves=20)")
    p.add_argument("--warm-start-lgbm", action="store_true",
                   help="Enable LGBM warm-starting from prior walk-forward window "
                        "(3-5x faster per window)")
    p.add_argument("--adversarial-validation", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Run adversarial validation diagnostic (default OFF — AUC=1.0 on "
                        "temporal walk-forward splits is uninformative; saves ~25 min)")
    p.add_argument("--interaction-constraints", default="none",
                   choices=["none", "auto"],
                   help="Feature interaction constraints for LGBM: 'auto' groups by feature "
                        "family (Israel-Kelly-Moskowitz 2020)")

    # ── Tier 1-7: Label / ML objective flags ─────────────────────────────
    p.add_argument("--forward-windows", default=None,
                   type=lambda s: [int(x) for x in s.split(",")],
                   help="Comma-separated forward-horizon list for multi-horizon "
                        "ensemble labels (e.g. '1,5,21'). If omitted, uses the "
                        "single --forward-window horizon.")
    p.add_argument("--lambdarank-truncation", default=20, type=int,
                   help="LambdaRank truncation level (default 20)")
    p.add_argument("--use-return-decile-gain", action="store_true",
                   help="Use return-decile gain mapping in ranking objective")
    p.add_argument("--use-magnitude-weights", action="store_true",
                   help="Weight samples by absolute forward-return magnitude")
    p.add_argument("--huber-weight", default=0.0, type=float,
                   help="Blend weight on Huber regression head (default 0.0)")
    p.add_argument("--quantile-weight", default=0.0, type=float,
                   help="Blend weight on quantile regression head (default 0.0)")
    p.add_argument("--quantile-alpha", default=0.75, type=float,
                   help="Quantile level for quantile regression head (default 0.75)")
    p.add_argument("--use-meta-labeling", action="store_true",
                   help="Enable meta-labeling second-stage classifier")

    # ── Sample-weight + regularization flags ─────────────────────────────
    p.add_argument("--min-data-in-leaf", default=200, type=int,
                   help="LightGBM min_data_in_leaf (default 200)")
    p.add_argument("--use-uniqueness-weights", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply label uniqueness sample-weights (default ON)")
    p.add_argument("--use-per-date-weights", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply per-date sample-weight normalization (default ON)")
    p.add_argument("--early-stop-on-ic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Early-stop boosters on rank-IC rather than loss (default ON)")
    p.add_argument("--lgbm-num-seeds", default=1, type=int,
                   help="Number of LightGBM seeds to average (default 1)")
    p.add_argument("--use-mlp", action="store_true",
                   help="Include MLP head in ensemble (default OFF)")
    p.add_argument("--random-state", default=42, type=int,
                   help="Random seed for ML models (default 42)")

    # ── Feature panel flags ──────────────────────────────────────────────
    p.add_argument("--size-neutralize", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Size-neutralize features (default ON)")
    p.add_argument("--sector-neutralize-features", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Sector-neutralize features (default ON)")
    p.add_argument("--winsorize", action=argparse.BooleanOptionalAction, default=True,
                   help="Winsorize feature panel (default ON)")
    p.add_argument("--cs-zscore-all", action=argparse.BooleanOptionalAction, default=True,
                   help="Cross-sectional z-score all features (default ON)")
    p.add_argument("--use-higher-moment", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include higher-moment (skew/kurt) features (default ON)")
    p.add_argument("--use-breadth-wavelet", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include breadth + wavelet features (default ON)")

    # ── Alt-feature flags ────────────────────────────────────────────────
    p.add_argument("--use-tier3-academic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable Tier-3 academic value/quality + EAR gate (default ON)")
    p.add_argument("--sector-neutralize-fundamentals", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Sector-neutralize fundamentals (default ON)")

    # ── Portfolio signal shaping flags ───────────────────────────────────
    p.add_argument("--signal-smooth-halflife", default=5.0, type=float,
                   help="Exp half-life (days) for signal smoothing in portfolio "
                        "construction (default 5.0)")
    p.add_argument("--apply-rank-normal", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply rank-normal transform to signals pre-selection "
                        "(default ON)")

    # ── ML / momentum blend ──────────────────────────────────────────────
    p.add_argument("--ml-blend", default=0.70, type=float,
                   help="ML weight in final signal blend; momentum weight = "
                        "1 - ml_blend (default 0.70 = 70%% ML / 30%% momentum)")

    return p.parse_args()


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Optuna wrapper with HyperbandPruner + Threshold pruning + warm-start
# ---------------------------------------------------------------------------
def optimize_hyperparameters_pruned(
    panel, labels,
    n_trials: int = 40,
    n_eval_folds: int = 5,
    prune_threshold_sharpe: float = 0.0,
    prune_threshold_ic: float = 0.2,
    warm_start: bool = True,
    warm_start_path: Path = Path("data/cache/optuna_best_params.pkl"),
    min_train_days: int = 252,
    retrain_freq: int = 21,
) -> dict:
    """
    Bayesian hyperparameter optimization with HyperbandPruner, per-fold
    ThresholdPruner, and warm-start from prior best trials.

    Uses a mini walk-forward over `n_eval_folds` folds; reports intermediate
    per-fold rank-IC to Optuna so it can prune unpromising trials early.
    """
    try:
        import optuna
        import pickle
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [optuna] Not installed (pip install optuna). Using defaults.")
        return {}

    from model import WalkForwardModel

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    eval_start = max(0, len(all_dates) - 504)
    eval_dates = all_dates[eval_start:]
    # Split eval dates into folds for walk-forward IC reporting
    fold_size = max(20, len(eval_dates) // max(1, n_eval_folds))
    folds = [eval_dates[i * fold_size:(i + 1) * fold_size]
             for i in range(n_eval_folds)]
    folds = [f for f in folds if len(f) > 0]

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 10, 63)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        feat_frac = trial.suggest_float("feature_fraction", 0.4, 0.9)
        bag_frac = trial.suggest_float("bagging_fraction", 0.5, 0.9)
        n_est = trial.suggest_int("n_estimators", 100, 500, step=50)
        ridge_alpha = trial.suggest_float("ridge_alpha", 5.0, 200.0, log=True)
        lambda_l1 = trial.suggest_float("lambda_l1", 0.1, 50.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 0.1, 100.0, log=True)
        min_child = trial.suggest_int("min_child_samples", 10, 50)
        max_depth = trial.suggest_int("max_depth", 4, 10)
        lgbm_w = trial.suggest_float("lgbm_weight", 0.1, 0.6)
        xgb_w = trial.suggest_float("xgb_weight", 0.0, 0.5)
        ridge_w = trial.suggest_float("ridge_weight", 0.1, 0.5)
        mlp_w = trial.suggest_float("mlp_weight", 0.0, 0.4)
        total_w = lgbm_w + xgb_w + ridge_w + mlp_w
        lgbm_w, xgb_w, ridge_w, mlp_w = [
            w / total_w for w in (lgbm_w, xgb_w, ridge_w, mlp_w)
        ]

        wf = WalkForwardModel(
            min_train_days=min_train_days, retrain_freq=retrain_freq,
            num_leaves=num_leaves, learning_rate=lr,
            feature_fraction=feat_frac, bagging_fraction=bag_frac,
            n_estimators=n_est, min_child_samples=min_child,
            max_depth=max_depth, lambda_l1=lambda_l1, lambda_l2=lambda_l2,
            ridge_alpha=ridge_alpha,
            lgbm_weight=lgbm_w, xgb_weight=xgb_w,
            ridge_weight=ridge_w, mlp_weight=mlp_w,
            prune_features=False,
        )
        try:
            pred = wf.fit_predict(panel, labels)
        except Exception:
            return 0.0

        fold_ics = []
        for fold_idx, fold_dates in enumerate(folds):
            ics = []
            for date in fold_dates[:50]:
                if date not in pred.index:
                    continue
                p = pred.loc[date].dropna()
                if date in labels.index.get_level_values("date"):
                    l = labels.xs(date, level="date").reindex(p.index)
                    common = p.index.intersection(l.dropna().index)
                    if len(common) > 20:
                        ic = p[common].corr(l[common], method="spearman")
                        if np.isfinite(ic):
                            ics.append(ic)
            if not ics:
                continue
            fold_ic = float(np.mean(ics))
            fold_ics.append(fold_ic)

            # Running mean IC, IC-IR, and proxy "Sharpe" (IC * sqrt(252))
            running_mean = float(np.mean(fold_ics))
            running_std = float(np.std(fold_ics)) if len(fold_ics) > 1 else 1.0
            ic_ir = running_mean / (running_std + 1e-8) \
                if len(fold_ics) > 1 else running_mean
            proxy_sharpe = running_mean * np.sqrt(252.0)

            # Manual ThresholdPruner
            if (proxy_sharpe < prune_threshold_sharpe
                    or ic_ir < prune_threshold_ic):
                raise optuna.TrialPruned()

            # HyperbandPruner decision
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_ics)) if fold_ics else 0.0

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3,
        ),
    )

    # ── Warm-start: load prior top-5 configs ────────────────────────────
    if warm_start and warm_start_path.exists():
        try:
            with open(warm_start_path, "rb") as f:
                prior = pickle.load(f)
            for params in (prior or [])[:5]:
                try:
                    study.enqueue_trial(params)
                except Exception:
                    pass
            print(f"  [optuna] Warm-started with {len(prior)} prior configs")
        except Exception as e:
            print(f"  [optuna] Warm-start load failed: {e}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Save top-5 best params for future warm-starts ───────────────────
    try:
        warm_start_path.parent.mkdir(parents=True, exist_ok=True)
        completed = [t for t in study.trials
                     if t.state.name == "COMPLETE" and t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=True)
        top5 = [t.params for t in completed[:5]]
        with open(warm_start_path, "wb") as f:
            pickle.dump(top5, f)
    except Exception as e:
        print(f"  [optuna] Failed to save warm-start cache: {e}")

    best = study.best_params if study.best_trial is not None else {}
    if best:
        print(f"  [optuna] Best params (IC={study.best_value:.4f}): {best}")
    return best


def run(args):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    section("PERSONAL TRADING PORTFOLIO")
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Positions:  {args.n_positions}")
    print(f"  Weighting:  {args.weighting} (concentration={args.concentration})")
    print(f"  Sector cap: {args.max_sector_pct*100:.0f}%")

    # ------------------------------------------------------------------
    # PHASE 1 — Data
    # ------------------------------------------------------------------
    section("Phase 1 — Data Loading")

    print("[1/6] Loading price data (large universe)...")
    # $1M ADV: maximize universe for small-cap alpha. Risk-adjusted labels +
    # vol-bucket selection prevent loading up on volatile junk. Liquidity-scaled
    # transaction costs penalize micro-caps realistically.
    prices = load_prices(
        start=args.start,
        end=args.end,
        dynamic_universe=True,
        universe_size=3000,
        min_price=5.0,
        min_adv=500_000,
    )
    close   = get_close(prices)
    returns = get_returns(prices)
    volume  = get_volume(prices)
    print(f"      {len(close.columns)} tickers x {len(close)} trading days")

    print("[1b/6] Loading sector map...")
    sector_map = get_sectors(close.columns.tolist())

    # ------------------------------------------------------------------
    # Alternative data
    # ------------------------------------------------------------------
    alt_features_dict = None
    if not args.skip_alt_data:
        print("[1c/6] Loading alternative data sources...")
        tickers_list = close.columns.tolist()

        edgar_df = load_edgar_fundamentals(tickers_list, start=args.start, end=args.end)
        fred_df  = load_fred_macro(start=args.start, end=args.end)
        vix_df   = load_vix_term_structure(start=args.start, end=args.end)
        si_df    = load_short_interest(tickers_list)

        earnings_dict = load_earnings_calendar(tickers_list, start=args.start, end=args.end)
        insider_dict  = load_insider_transactions(tickers_list, start=args.start, end=args.end)
        analyst_dict  = load_analyst_actions(tickers_list, start=args.start, end=args.end)
        estimates_df  = load_earnings_estimates(tickers_list)
        inst_df       = load_institutional_holders(tickers_list)

        # Tier 6: macro / cross-asset / distress auxiliary data
        dxy_df_           = load_dxy(start=args.start, end=args.end)
        breakeven_df_     = load_breakeven_inflation(start=args.start, end=args.end)
        vvix_df_          = load_vvix(start=args.start, end=args.end)
        oil_df_           = load_oil_wti(start=args.start, end=args.end)
        copper_gold_df_   = load_copper_gold(start=args.start, end=args.end)
        cross_asset_df_   = load_cross_asset_etf_panel(start=args.start, end=args.end)
        ig_oas_df_        = load_ig_oas(start=args.start, end=args.end)
        sector_oas_df_    = load_sector_oas(start=args.start, end=args.end)
        ebp_df_           = load_excess_bond_premium(start=args.start, end=args.end)
        treasury_df_      = load_treasury_yield_curve(start=args.start, end=args.end)

        alt_features_dict = build_alt_features(
            tickers    = close.columns,
            date_index = close.index,
            edgar_df   = edgar_df if not edgar_df.empty else None,
            fred_df    = fred_df  if not fred_df.empty  else None,
            vix_df     = vix_df   if not vix_df.empty   else None,
            si_df      = si_df,
            earnings_dict  = earnings_dict,
            insider_dict   = insider_dict,
            analyst_dict   = analyst_dict,
            estimates_df    = estimates_df if not estimates_df.empty else None,
            institutional_df = inst_df if not inst_df.empty else None,
            sector_map      = sector_map,
            close           = close,
            # Point-in-time gates: keep OFF until historical time-series data
            # is integrated (FINRA short-interest archive; SEC 13F-HR filings).
            # Current loaders return a single snapshot that would be broadcast
            # across the full backtest, which is a lookahead bias.
            use_short_interest         = False,
            use_institutional_holdings = False,
            # Tier 3 academic value/quality + EAR gate (fundamentals)
            use_tier3_academic             = args.use_tier3_academic,
            sector_neutralize_fundamentals = args.sector_neutralize_fundamentals,
            # Tier 6 macro / distress auxiliary panels
            dxy_df            = dxy_df_,
            breakeven_df      = breakeven_df_,
            vvix_df           = vvix_df_,
            oil_df            = oil_df_,
            copper_gold_df    = copper_gold_df_,
            cross_asset_panel = cross_asset_df_,
            ig_oas_df         = ig_oas_df_,
            sector_oas_df     = sector_oas_df_,
            ebp_df            = ebp_df_,
            treasury_yields_df = treasury_df_,
        )
        n_alt = len(alt_features_dict)
        print(f"      {n_alt} alt feature panels loaded")
    else:
        print("  [alt data skipped]")
        alt_features_dict = {}

    # ------------------------------------------------------------------
    # Paid API data (EODHD + Finnhub)
    # ------------------------------------------------------------------
    if not args.skip_alt_data:
        print("[1d/6] Loading EODHD + Finnhub API data...")
        api_features = load_all_api_data(
            tickers=close.columns.tolist(),
            date_index=close.index,
            finnhub_max=500,  # top 500 tickers for Finnhub (rate limit)
        )
        if api_features:
            if alt_features_dict is None:
                alt_features_dict = {}
            alt_features_dict.update(api_features)
            print(f"      Total features (free + paid): {len(alt_features_dict)}")

    # Free raw alt-data DataFrames no longer needed (signals already built).
    # Keeps alt_features_dict (the signal panels) but drops the heavy source data.
    if not args.skip_alt_data:
        del edgar_df, fred_df, vix_df, si_df
        del earnings_dict, insider_dict, analyst_dict, estimates_df, inst_df
        del dxy_df_, breakeven_df_, vvix_df_, oil_df_, copper_gold_df_
        del cross_asset_df_, ig_oas_df_, sector_oas_df_, ebp_df_, treasury_df_
        gc.collect()

    # ------------------------------------------------------------------
    # PHASE 2 — Feature Engineering + ML
    # ------------------------------------------------------------------
    section("Phase 2 — Feature Engineering + ML Model")

    # ── Fix 1: Compute investable mask (top 1500 by ADV per date) ──────
    # Signal rankings should be computed WITHIN the investable universe so
    # that IC predicts top-1500 returns, not all-stock returns. Without this,
    # the signal's alpha concentrates in illiquid micro-caps that the ADV
    # filter replaces at portfolio construction time (97% pick replacement).
    adv_for_mask = (close * volume).rolling(21).mean()
    n_investable = min(args.max_selection_pool, adv_for_mask.shape[1])
    investable_mask = adv_for_mask.rank(axis=1, pct=True) >= (1 - n_investable / adv_for_mask.shape[1])
    print(f"  [investable mask] top {n_investable} by ADV per date")

    print("[2/6] Computing alpha signals...")
    composite, ranked_signals = build_composite_signal(
        close, returns, volume,
        sector_map=sector_map,
        use_ic_weights=True,
        investable_mask=investable_mask,
    )
    rvol = realized_volatility(returns, window=21)

    decay = factor_decay_analysis(composite, returns, horizons=[1, 5, 10, 21])
    decay.to_csv(results_dir / "factor_decay.csv")
    print(decay.to_string())

    # Momentum filter: 63-day (3-month) momentum for pre-screening
    mom_63d = momentum(close, 63)

    # Compute ADV for liquidity-aware costs and position sizing
    adv_30d = (close * volume).rolling(30, min_periods=10).mean()

    # Regime detection — computed early so portfolio can use it
    print("[2b/6] Computing market regime...")
    mkt_price, mkt_return = build_market_series(prices)
    regime = detect_combined_regime(mkt_return, mkt_price)

    # ------------------------------------------------------------------
    # ML signal (or fall back to rule-based composite)
    # ------------------------------------------------------------------
    final_signal = composite
    feature_imp  = None

    if not args.skip_ml:
        print("[3/6] Building feature matrix...")
        panel  = build_feature_matrix(
            close, returns, volume, ranked_signals,
            alt_features=alt_features_dict,
            sector_map=sector_map,
            use_higher_moment=args.use_higher_moment,
            use_breadth_wavelet=args.use_breadth_wavelet,
            size_neutralize=args.size_neutralize,
            sector_neutralize_features=args.sector_neutralize_features,
            winsorize=args.winsorize,
            cs_zscore_all=args.cs_zscore_all,
        )
        # Equal-weighted market-return proxy for beta-neutral labels
        mkt_return = returns.mean(axis=1) if args.beta_neutral_labels else None
        _label_kwargs = dict(
            risk_adjust=args.risk_adjust_labels,
            sector_map=sector_map,
            sector_rank_weight=args.sector_rank_weight,
            beta_neutral=args.beta_neutral_labels,
            market_returns=mkt_return,
            investable_mask=investable_mask,  # Fix 1: rank labels within investable set
        )
        if args.forward_windows is not None:
            _label_kwargs["forward_windows"] = args.forward_windows
        else:
            _label_kwargs["forward_window"] = args.forward_window
        labels = build_labels(returns, **_label_kwargs)
        print(f"       {panel.shape[0]:,} samples x {panel.shape[1]} features")

        # Optional: Bayesian hyperparameter search via Optuna
        ml_kwargs = dict(
            min_train_days=args.min_train_days,
            retrain_freq=args.retrain_freq,
            forward_window=args.forward_window,
            risk_adjust=args.risk_adjust_labels,
            sector_rank_weight=args.sector_rank_weight,
            num_leaves=args.num_leaves,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            warm_start_lgbm=args.warm_start_lgbm,
            run_adversarial_validation=args.adversarial_validation,
            interaction_constraints=args.interaction_constraints if args.interaction_constraints != "none" else None,
            learning_rate=0.05,
            lgbm_weight=0.60,
            ridge_weight=0.40,
            lambdarank_truncation=args.lambdarank_truncation,
            use_return_decile_gain=args.use_return_decile_gain,
            use_magnitude_weights=args.use_magnitude_weights,
            huber_weight=args.huber_weight,
            quantile_weight=args.quantile_weight,
            quantile_alpha=args.quantile_alpha,
            use_meta_labeling=args.use_meta_labeling,
            min_data_in_leaf=args.min_data_in_leaf,
            use_uniqueness_weights=args.use_uniqueness_weights,
            use_per_date_weights=args.use_per_date_weights,
            early_stop_on_ic=args.early_stop_on_ic,
            lgbm_num_seeds=args.lgbm_num_seeds,
            use_mlp=args.use_mlp,
            random_state=args.random_state,
        )
        if args.forward_windows is not None:
            ml_kwargs["forward_windows"] = args.forward_windows
        if args.optimize_ml:
            print(f"[3a/6] Optuna hyperparameter search ({args.optuna_trials} trials)...")
            best = optimize_hyperparameters_pruned(
                panel, labels,
                n_trials=args.optuna_trials,
                prune_threshold_sharpe=args.prune_threshold_sharpe,
                prune_threshold_ic=args.prune_threshold_ic,
                warm_start=args.optuna_warm_start,
                warm_start_path=Path(__file__).parent / "data" / "cache" / "optuna_best_params.pkl",
                min_train_days=args.min_train_days,
                retrain_freq=args.retrain_freq,
            )
            if best:
                # Override tunable params (keep min_train_days/retrain_freq/forward_window)
                for k in ("num_leaves", "learning_rate", "feature_fraction",
                          "bagging_fraction", "n_estimators", "ridge_alpha",
                          "lambda_l1", "lambda_l2", "min_child_samples", "max_depth",
                          "lgbm_weight", "xgb_weight", "ridge_weight", "mlp_weight"):
                    if k in best:
                        ml_kwargs[k] = best[k]
                # Normalize ensemble weights if all four were suggested
                w = [ml_kwargs.get(k, 0) for k in ("lgbm_weight", "xgb_weight", "ridge_weight", "mlp_weight")]
                if sum(w) > 0:
                    s = sum(w)
                    ml_kwargs["lgbm_weight"]  = w[0] / s
                    ml_kwargs["xgb_weight"]   = w[1] / s
                    ml_kwargs["ridge_weight"] = w[2] / s
                    ml_kwargs["mlp_weight"]   = w[3] / s
                print(f"       Applied: {best}")

        print("[3b/6] Walk-forward LightGBM training...")
        # 252-day warmup (1 year). 504 was too conservative — lost a full year
        # of predictions. 252 gives enough data for 90+ features while maximizing
        # the OOS evaluation period.
        wf = WalkForwardModel(**ml_kwargs)
        ml_signal = wf.fit_predict(panel, labels)

        # Light smoothing (5-day)
        ml_signal_smooth = ml_signal.rolling(5, min_periods=1).mean()

        # ── Momentum blend ───────────────────────────────────────────────
        # 70% ML + 30% classic 12-1 month momentum.
        # V4 used 60/40 which gave too much weight to momentum (a known factor)
        # and not enough to the ML's stock-specific signal. Shifting to 70/30
        # trusts the ML more while keeping momentum as an anchor.
        print("[3c/6] Blending ML signal with 12-1 month momentum...")
        mom_12_1 = close.shift(21).pct_change(252)
        mom_12_1_ranked = mom_12_1.rank(axis=1, pct=True)

        mom_aligned = mom_12_1_ranked.reindex(
            index=ml_signal_smooth.index, columns=ml_signal_smooth.columns
        )
        final_signal = (args.ml_blend * ml_signal_smooth
                        + (1.0 - args.ml_blend) * mom_aligned.fillna(0.5))
        final_signal = final_signal.rank(axis=1, pct=True)

        feature_imp = wf.feature_importance()
        if feature_imp is not None:
            feature_imp.to_csv(results_dir / "feature_importance.csv")
            print("\n  Top 10 features:")
            print(feature_imp.head(10).to_string())
    else:
        print("  [ML skipped — using rule-based composite signal]")

    # ------------------------------------------------------------------
    # PHASE 3 — Portfolio Construction
    # ------------------------------------------------------------------
    section("Phase 3 — Long-Only Portfolio Construction")

    # Quality filter: use alt ROE signal if available, else use
    # profitability proxy (low idiosyncratic vol = more stable/profitable).
    quality_signal = None
    if alt_features_dict and "roe_signal" in alt_features_dict:
        quality_signal = alt_features_dict["roe_signal"]
        print("  Using ROE quality filter")
    elif alt_features_dict and "gross_margin_signal" in alt_features_dict:
        quality_signal = alt_features_dict["gross_margin_signal"]
        print("  Using gross margin quality filter")

    # SPY trend filter: signal is positive when SPY > 200d MA, negative when below
    # SPY trend overlay DISABLED — V4 showed it ate too much exposure.
    # With beta 0.53, the portfolio was a de-levered index fund.
    # Regime-based cash (15%) is sufficient downside protection.
    spy_trend = None

    # Earnings dates for exclusion zone
    earnings_days_df = None
    if alt_features_dict and "days_to_earnings_signal" in alt_features_dict:
        # The raw days-to-earnings panel (before cross-sectional ranking)
        # We need raw days, not ranks. Use the signal as a proxy:
        # higher rank = closer to earnings. Invert: rank 1.0 = 0 days away.
        earnings_days_df = alt_features_dict.get("days_to_earnings_signal")

    # SPY core+satellite mode requires SPY to be a column in the signal panel.
    # Training universe may not include SPY by default — warn if so.
    if args.spy_core > 0 and args.spy_ticker not in final_signal.columns:
        print(f"  [WARNING] --spy-core {args.spy_core} requested but '{args.spy_ticker}' "
              f"is not in the price/signal panel. SPY core allocation will be SKIPPED. "
              f"Add SPY to the universe (e.g. via load_prices) to enable this feature.")

    print("[4/6] Building monthly portfolio...")
    weights, rebalance_dates = build_monthly_portfolio(
        signal          = final_signal,
        n_positions     = args.n_positions,
        weighting       = args.weighting,
        concentration   = args.concentration,
        max_weight      = args.max_weight,
        min_weight      = 0.02,
        sector_map      = sector_map,
        max_sector_pct  = args.max_sector_pct,
        momentum_filter = mom_63d,
        realized_vol    = rvol,
        returns         = returns,
        regime          = regime,
        adv             = adv_30d,
        cash_in_bear    = args.cash_in_bear,
        quality_filter  = quality_signal,
        earnings_dates  = earnings_days_df,
        spy_trend_filter= spy_trend,
        use_vol_buckets = args.use_vol_buckets,
        max_selection_pool = args.max_selection_pool,
        spy_core_weight = args.spy_core,
        spy_ticker      = args.spy_ticker,
        force_mega_caps = args.force_mega_caps,
        signal_smooth_halflife = args.signal_smooth_halflife,
        apply_rank_normal = args.apply_rank_normal,
    )

    # ── Volatility targeting overlay (ENABLED by default) ─────────────
    # Moreira & Muir (2017) "Volatility-Managed Portfolios" + DeMiguel et al.
    # (JF 2024) — scaling exposure inversely to ex-ante portfolio vol lifts
    # net Sharpe ~0.2-0.4 (up to +13% on multifactor portfolios). This REQUIRES
    # leverage > 1 during low-vol regimes to capture the full lift; a
    # de-lever-only overlay (max_leverage=1.0) leaves most of the Sharpe
    # gain on the table.
    #
    # SPY-core interaction: vol targeting is applied to the *entire* weight
    # vector here, including the SPY core slice. This is a known simplification
    # — ideally we would only scale the satellite (non-SPY) book so the SPY
    # anchor stays at its specified target weight. For now, the effective
    # leverage on SPY is identical to satellite leverage; if spy_core is large,
    # revisit to apply vol-targeting only to the risky satellite portion.
    vol_target_active = (not args.no_vol_target) and args.vol_target > 0
    if vol_target_active:
        from portfolio import apply_vol_targeting
        print(f"  [vol-target] Scaling exposure to {args.vol_target*100:.0f}% annualized vol "
              f"(max_leverage={args.max_leverage:.2f})")
        weights = apply_vol_targeting(
            weights,
            realized_vol=rvol,
            target_vol=args.vol_target,
            max_leverage=args.max_leverage,
            min_leverage=0.8,
        )
    else:
        print("  [vol-target] disabled")

    port_stats = compute_portfolio_stats(weights, rebalance_dates)
    print(f"      Positions:  {port_stats['avg_positions']:.0f} avg")
    print(f"      Max weight: {port_stats['avg_max_weight']*100:.1f}% avg")
    print(f"      Top-5 wt:   {port_stats['avg_top5_weight']*100:.1f}% avg")
    print(f"      Rebalances: {port_stats['n_rebalances']}")
    print(f"      Ann. turn:  {port_stats['annualized_turnover']*100:.0f}%")

    # Current holdings (latest rebalance)
    holdings = get_current_holdings(weights, final_signal, sector_map)
    if not holdings.empty:
        print("\n  Latest holdings:")
        print(holdings.to_string(index=False))
        holdings.to_csv(results_dir / "current_holdings.csv", index=False)

    # ------------------------------------------------------------------
    # PHASE 4 — Backtest
    # ------------------------------------------------------------------
    section("Phase 4 — Backtest")

    print("[5/6] Running backtest...")
    # Personal-scale costs: $0 commission (IBKR/Schwab), ~3bps base spread
    # (scaled by liquidity per stock), minimal slippage at personal sizes.
    cost_model = TransactionCostModel(
        spread_bps=3.0,
        commission_bps=0.0,
        slippage_bps=2.0,
    )
    result = run_backtest(
        weights,
        prices,
        initial_capital=args.capital,
        cost_model=cost_model,
        rebalance_dates=set(rebalance_dates),
        adv=adv_30d,
        stop_loss_pct=args.stop_loss,
        monthly_loss_limit=0.0,  # disabled — too defensive in V4
    )

    # SPY benchmark
    try:
        spy_raw    = load_prices(["SPY"], start=args.start, end=args.end, use_cache=False)
        spy_ret    = get_returns(spy_raw)
        # Force to 1D Series (yfinance may return multi-level columns)
        if isinstance(spy_ret, pd.DataFrame):
            if "SPY" in spy_ret.columns:
                spy_series = spy_ret["SPY"]
            else:
                spy_series = spy_ret.iloc[:, 0]
            # Flatten if still multi-level
            if isinstance(spy_series, pd.DataFrame):
                spy_series = spy_series.iloc[:, 0]
        else:
            spy_series = spy_ret
        spy_series = spy_series.reindex(result.daily_returns.index).fillna(0)
        spy_equity = (1 + spy_series).cumprod() * args.capital
    except Exception:
        spy_series = pd.Series(0.0, index=result.daily_returns.index)
        spy_equity = pd.Series(args.capital, index=result.daily_returns.index)

    dsr_n_trials = args.optuna_trials if args.optimize_ml else 1
    tearsheet = compute_full_tearsheet(result, benchmark_returns=spy_series,
                                       n_trials=dsr_n_trials)
    print("\n  Performance Tearsheet:")
    print(tearsheet.to_string())
    tearsheet.to_csv(results_dir / "tearsheet.csv")

    # Annual returns
    annual = annual_returns(result.daily_returns, benchmark_returns=spy_series)
    print("\n  Annual Returns:")
    print(annual.to_string())
    annual.to_csv(results_dir / "annual_returns.csv")

    # Monthly returns heatmap
    monthly = monthly_returns_table(result.daily_returns)
    print("\n  Monthly Returns (%):")
    print(monthly.to_string())
    monthly.to_csv(results_dir / "monthly_returns.csv")

    # Save daily weights history for time-travel dashboard
    if not result.weights_history.empty:
        result.weights_history.to_parquet(results_dir / "weights_history.parquet")
        print(f"  Weights history saved: {result.weights_history.shape}")

    # Wealth growth comparison
    wealth = wealth_growth(result.daily_returns, spy_series, initial=args.capital)
    wealth.to_csv(results_dir / "wealth_growth.csv")
    print(f"\n  Final portfolio:  ${wealth['Strategy'].iloc[-1]:>12,.0f}")
    print(f"  SPY buy & hold:   ${wealth['SPY Buy & Hold'].iloc[-1]:>12,.0f}")
    print(f"  Excess wealth:    ${wealth['Excess'].iloc[-1]:>+12,.0f}")

    # After-tax estimate
    pretax_cagr = tearsheet.loc["CAGR", "Value"] if "CAGR" in tearsheet.index else 0.0
    if isinstance(pretax_cagr, str):
        pretax_cagr = float(pretax_cagr.replace("%", "")) / 100
    avg_monthly_turnover = result.turnover.mean() / 12 if hasattr(result, 'turnover') and len(result.turnover) > 0 else 0.50
    tax_info = after_tax_estimate(pretax_cagr, monthly_turnover_oneway=min(avg_monthly_turnover, 1.0))
    print("\n  After-Tax Estimate:")
    for k, v in tax_info.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:>8.2%}" if abs(v) < 10 else f"    {k:30s}: ${v:>12,.0f}")
        else:
            print(f"    {k:30s}: {v}")

    # ------------------------------------------------------------------
    # PHASE 5 — Robustness + Dashboard
    # ------------------------------------------------------------------
    section("Phase 5 — Analysis & Dashboard")

    # Regime analysis (regime already computed in Phase 2)
    print("[5b/6] Regime analysis...")
    perf_regime = performance_by_regime(
        result.daily_returns, regime, benchmark_returns=spy_series)
    print(perf_regime.to_string())
    perf_regime.to_csv(results_dir / "regime_performance.csv")

    print("\n  Stress test:")
    stress = stress_test(result.daily_returns, spy_series)
    print(stress.to_string())
    stress.to_csv(results_dir / "stress_test.csv")

    # OOS split
    from metrics import oos_split_tearsheet, fama_french_regression, strategy_correlation_analysis
    oos_df = None
    ff_df = None
    factor_corr_df = None

    print("\n  In-sample vs out-of-sample split...")
    try:
        oos_df = oos_split_tearsheet(result, benchmark_returns=spy_series,
                                     oos_start="2024-01-01")
        print(oos_df.to_string())
        oos_df.to_csv(results_dir / "oos_tearsheet.csv")
    except Exception as e:
        print(f"  [OOS split failed: {e}]")

    print("\n  Fama-French 5-factor + Momentum regression...")
    try:
        ff_df = fama_french_regression(result.daily_returns)
        if "error" not in ff_df.columns:
            print(ff_df.to_string())
            ff_df.to_csv(results_dir / "fama_french.csv")
        else:
            print(f"  [FF regression failed: {ff_df['error'].iloc[0]}]")
            ff_df = None
    except Exception as e:
        print(f"  [FF regression failed: {e}]")

    print("\n  Strategy correlation to known factors...")
    try:
        factor_corr_df = strategy_correlation_analysis(
            result.daily_returns, benchmark_returns=spy_series)
        if "error" not in factor_corr_df.columns:
            print(factor_corr_df.to_string())
            factor_corr_df.to_csv(results_dir / "factor_correlation.csv")
        else:
            factor_corr_df = None
    except Exception as e:
        print(f"  [Factor correlation failed: {e}]")

    # Bootstrap (optional)
    bootstrap_df = None
    if not args.skip_robustness:
        print("\n  Bootstrap confidence intervals...")
        bootstrap_df = bootstrap_tearsheet(result.daily_returns, n_simulations=args.n_bootstrap)
        print(bootstrap_df.to_string())
        bootstrap_df.to_csv(results_dir / "bootstrap_ci.csv")

    # Sector allocation
    sector_alloc = sector_allocation(weights, sector_map)
    sector_alloc.to_csv(results_dir / "sector_allocation.csv")

    # Dashboard
    print("\n[6/6] Building dashboard...")
    build_dashboard(
        result             = result,
        benchmark_equity   = spy_equity,
        benchmark_returns  = spy_series,
        perf_by_regime     = perf_regime,
        feature_importance = feature_imp,
        bootstrap_df       = bootstrap_df,
        oos_df             = oos_df,
        factor_corr_df     = factor_corr_df,
        monthly_returns    = monthly,
        annual_df          = annual,
        holdings_df        = holdings,
        sector_alloc_df    = sector_alloc,
        initial_capital    = args.capital,
        output_path        = str(results_dir / "dashboard.html"),
    )

    section("ALL OUTPUTS SAVED TO results/")
    print("""
  tearsheet.csv            Strategy performance metrics
  annual_returns.csv       Year-by-year returns vs SPY
  monthly_returns.csv      Calendar monthly returns (%)
  wealth_growth.csv        Cumulative wealth: strategy vs SPY
  current_holdings.csv     Latest portfolio holdings
  sector_allocation.csv    Sector weights over time
  feature_importance.csv   ML feature importances
  regime_performance.csv   Performance by market regime
  stress_test.csv          Historical stress events
  oos_tearsheet.csv        In-sample vs out-of-sample
  fama_french.csv          Factor regression results
  factor_correlation.csv   Correlation to known factors
  bootstrap_ci.csv         Bootstrap confidence intervals
  dashboard.html           <- Open in browser
""")

    return result, tearsheet, annual, monthly


if __name__ == "__main__":
    run(parse_args())
