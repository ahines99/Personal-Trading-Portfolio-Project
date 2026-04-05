"""
test_suite.py
-------------
Systematic parameter sweep for the Personal Trading Portfolio.

Tests hyperparameters, portfolio construction, feature ablation,
holding periods, and signal blends. Results save incrementally.

Categories (prefix → purpose):
    A, B, C, D, E, F, G, H, V, X — original hyperparam / construction sweeps
    P   — Portfolio beta-attack (SPY core, mega-cap forcing, vol-tgt x leverage,
          signal post-proc, factor neutralization, concentration, stacked)
    L   — Label engineering (clean horizons, beta-neutral, multi-horizon,
          lambdarank truncation, huber/quantile heads, stacks)
    R   — Regularization (min_data_in_leaf sweep)
    Z   — Training hygiene stacks (uniqueness/per-date weights, seed bagging)
    FA, FB, FG, FH — Feature ablation (size-neutralize, sector-neutralize,
          distress signals, macro cross-section)

Usage:
    python test_suite.py              # full suite
    python test_suite.py --quick      # portfolio-only (uses cached ML)
    python test_suite.py --category B # specific category

Results: results/test_suite_results.csv
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_prices, get_close, get_returns, get_volume, get_sectors
from features import build_composite_signal, realized_volatility, momentum
from model import build_feature_matrix, build_labels, WalkForwardModel
from portfolio import build_monthly_portfolio, compute_portfolio_stats, apply_vol_targeting
from backtest import run_backtest, TransactionCostModel
from metrics import (compute_full_tearsheet, sharpe_ratio, annualized_return,
                     max_drawdown, sortino_ratio, annualized_volatility,
                     beta_to_market, win_rate, calmar_ratio, oos_split_tearsheet)
from regime import detect_combined_regime, build_market_series

RESULTS_FILE = Path("results/test_suite_results.csv")


# ---------------------------------------------------------------------------
# Data loading (shared across all tests)
# ---------------------------------------------------------------------------

def load_shared_data(args):
    """Load all data once, return as dict."""
    print("Loading shared data...")
    prices = load_prices(start=args.start, end=args.end)
    close = get_close(prices)
    returns = get_returns(prices)
    volume = get_volume(prices)
    sector_map = get_sectors(close.columns.tolist())

    composite, ranked_signals = build_composite_signal(
        close, returns, volume, sector_map=sector_map, use_ic_weights=True,
    )
    rvol = realized_volatility(returns, window=21)
    mom_63d = momentum(close, 63)
    adv_30d = (close * volume).rolling(30, min_periods=10).mean()

    mkt_price, mkt_return = build_market_series(prices)
    regime = detect_combined_regime(mkt_return, mkt_price)

    # SPY benchmark
    try:
        spy_raw = load_prices(["SPY"], start=args.start, end=args.end, use_cache=True)
        spy_ret = get_returns(spy_raw)
        if isinstance(spy_ret, pd.DataFrame):
            spy_series = spy_ret.iloc[:, 0] if "SPY" not in spy_ret.columns else spy_ret["SPY"]
            if isinstance(spy_series, pd.DataFrame):
                spy_series = spy_series.iloc[:, 0]
        else:
            spy_series = spy_ret
    except Exception:
        spy_series = pd.Series(0.0, index=returns.index)

    # Alt features
    alt_features_dict = None
    try:
        from alt_data_loader import (load_edgar_fundamentals, load_fred_macro,
                                      load_vix_term_structure, load_short_interest,
                                      load_earnings_calendar, load_insider_transactions,
                                      load_analyst_actions)
        from alt_features import build_alt_features

        tickers_list = close.columns.tolist()
        edgar_df = load_edgar_fundamentals(tickers_list, start=args.start, end=args.end)
        fred_df = load_fred_macro(start=args.start, end=args.end)
        vix_df = load_vix_term_structure(start=args.start, end=args.end)
        si_df = load_short_interest(tickers_list)
        earnings_dict = load_earnings_calendar(tickers_list, start=args.start, end=args.end)
        insider_dict = load_insider_transactions(tickers_list, start=args.start, end=args.end)
        analyst_dict = load_analyst_actions(tickers_list, start=args.start, end=args.end)

        alt_features_dict = build_alt_features(
            tickers=close.columns, date_index=close.index,
            edgar_df=edgar_df if not edgar_df.empty else None,
            fred_df=fred_df if not fred_df.empty else None,
            vix_df=vix_df if not vix_df.empty else None,
            si_df=si_df,
            earnings_dict=earnings_dict,
            insider_dict=insider_dict,
            analyst_dict=analyst_dict,
            sector_map=sector_map,
            close=close,
            # Point-in-time gates: keep OFF until historical time-series data
            # is integrated (FINRA short-interest archive; SEC 13F-HR filings).
            use_short_interest=False,
            use_institutional_holdings=False,
            # Tier 3 academic value/quality + EAR gate (fundamentals)
            use_tier3_academic=True,
            sector_neutralize_fundamentals=True,
        )
    except Exception as e:
        print(f"  Alt data loading failed: {e}")

    # Feature panel + labels
    panel = build_feature_matrix(
        close, returns, volume, ranked_signals,
        alt_features=alt_features_dict, sector_map=sector_map,
        use_higher_moment=True, use_breadth_wavelet=True,
    )

    return {
        "prices": prices, "close": close, "returns": returns, "volume": volume,
        "sector_map": sector_map, "composite": composite, "ranked_signals": ranked_signals,
        "rvol": rvol, "mom_63d": mom_63d, "adv_30d": adv_30d,
        "regime": regime, "spy_series": spy_series, "panel": panel,
        "alt_features_dict": alt_features_dict,
        "mkt_return": mkt_return,
    }


# ---------------------------------------------------------------------------
# Feature-ablation panel rebuilder (Fix 6)
# ---------------------------------------------------------------------------

def build_panel_for_config(data, panel_flags, alt_flags, use_cache=True):
    """Rebuild feature_panel and alt_features with custom flags for
    feature-ablation tests.

    panel_flags: dict of build_feature_matrix kwargs (size_neutralize,
                 sector_neutralize_features, winsorize, cs_zscore_all,
                 use_higher_moment, use_breadth_wavelet)
    alt_flags: dict of build_alt_features kwargs (use_chs, use_dtd,
               use_altman_z, use_macro_cross_section, use_tier3_academic,
               sector_neutralize_fundamentals, use_sni_variants)

    Returns (feature_panel, alt_features_dict)
    """
    close = data["close"]
    returns = data["returns"]
    volume = data["volume"]
    sector_map = data["sector_map"]
    ranked_signals = data["ranked_signals"]

    # Rebuild alt features if any alt_flags supplied
    alt_dict = data.get("alt_features_dict")
    if alt_flags:
        try:
            from alt_data_loader import (load_edgar_fundamentals, load_fred_macro,
                                          load_vix_term_structure, load_short_interest,
                                          load_earnings_calendar, load_insider_transactions,
                                          load_analyst_actions)
            from alt_features import build_alt_features

            tickers_list = close.columns.tolist()
            start = str(close.index.min().date())
            end = str(close.index.max().date())
            edgar_df = load_edgar_fundamentals(tickers_list, start=start, end=end)
            fred_df = load_fred_macro(start=start, end=end)
            vix_df = load_vix_term_structure(start=start, end=end)
            si_df = load_short_interest(tickers_list)
            earnings_dict = load_earnings_calendar(tickers_list, start=start, end=end)
            insider_dict = load_insider_transactions(tickers_list, start=start, end=end)
            analyst_dict = load_analyst_actions(tickers_list, start=start, end=end)

            alt_kwargs = dict(
                tickers=close.columns, date_index=close.index,
                edgar_df=edgar_df if not edgar_df.empty else None,
                fred_df=fred_df if not fred_df.empty else None,
                vix_df=vix_df if not vix_df.empty else None,
                si_df=si_df,
                earnings_dict=earnings_dict,
                insider_dict=insider_dict,
                analyst_dict=analyst_dict,
                sector_map=sector_map,
                close=close,
                use_short_interest=False,
                use_institutional_holdings=False,
                use_tier3_academic=True,
                sector_neutralize_fundamentals=True,
            )
            alt_kwargs.update(alt_flags)
            alt_dict = build_alt_features(**alt_kwargs)
        except Exception as e:
            print(f"  [panel rebuild] alt rebuild failed: {e}")

    # Rebuild feature panel with merged flags
    panel_kwargs = dict(
        size_neutralize=True,
        sector_neutralize_features=True,
        winsorize=True,
        cs_zscore_all=True,
        use_higher_moment=True,
        use_breadth_wavelet=True,
    )
    panel_kwargs.update(panel_flags or {})
    panel = build_feature_matrix(
        close, returns, volume, ranked_signals,
        alt_features=alt_dict, sector_map=sector_map,
        use_cache=use_cache,
        **panel_kwargs,
    )
    return panel, alt_dict


# ---------------------------------------------------------------------------
# ML signal generation (cached)
# ---------------------------------------------------------------------------

def generate_ml_signal(data, forward_window=5, min_train_days=126,
                       ml_blend=0.70, mom_blend=0.30,
                       risk_adjust=False, sector_rank_weight=0.0,
                       beta_neutral=False, forward_windows=None,
                       panel=None, **model_kwargs):
    """Train ML model and blend with momentum. Uses prediction cache.

    Label kwargs (risk_adjust, sector_rank_weight, beta_neutral, forward_windows)
    are forwarded to build_labels. When beta_neutral=True, pulls market_returns
    from data['mkt_return']. Optional `panel` kwarg overrides data['panel'] for
    feature-ablation tests.
    """
    market_returns = data.get("mkt_return") if beta_neutral else None
    labels = build_labels(
        data["returns"],
        forward_window=forward_window,
        sector_map=data["sector_map"],
        risk_adjust=risk_adjust,
        sector_rank_weight=sector_rank_weight,
        beta_neutral=beta_neutral,
        market_returns=market_returns,
        forward_windows=forward_windows,
    )

    wf = WalkForwardModel(
        min_train_days=min_train_days,
        retrain_freq=21,
        forward_window=forward_window,
        risk_adjust=risk_adjust,
        sector_rank_weight=sector_rank_weight,
        **model_kwargs,
    )
    panel_to_use = panel if panel is not None else data["panel"]
    ml_signal = wf.fit_predict(panel_to_use, labels)
    ml_smooth = ml_signal.rolling(5, min_periods=1).mean()

    # Momentum blend
    mom_12_1 = data["close"].shift(21).pct_change(252)
    mom_ranked = mom_12_1.rank(axis=1, pct=True)
    mom_aligned = mom_ranked.reindex(index=ml_smooth.index, columns=ml_smooth.columns)

    final = ml_blend * ml_smooth + mom_blend * mom_aligned.fillna(0.5)
    return final.rank(axis=1, pct=True), wf


# ---------------------------------------------------------------------------
# Single test runner
# ---------------------------------------------------------------------------

def run_single_test(test_name, data, signal, portfolio_kwargs, spy_series,
                    cost_model=None, capital=100_000, vol_target=0.0):
    """Run portfolio construction + backtest + metrics for one config."""
    t0 = time.time()

    if cost_model is None:
        cost_model = TransactionCostModel(spread_bps=3.0, commission_bps=0.0, slippage_bps=2.0)

    # Fix 3: read max_leverage from portfolio_kwargs (Tier 4 default 1.3)
    max_leverage = portfolio_kwargs.pop("max_leverage", 1.3)

    try:
        weights, rebalance_dates = build_monthly_portfolio(signal=signal, **portfolio_kwargs)

        # Optional volatility targeting overlay
        if vol_target is not None and vol_target > 0:
            weights = apply_vol_targeting(
                weights, realized_vol=data["rvol"],
                target_vol=vol_target, max_leverage=max_leverage, min_leverage=0.3,
            )

        result = run_backtest(
            weights, data["prices"],
            initial_capital=capital,
            cost_model=cost_model,
            rebalance_dates=set(rebalance_dates),
            adv=data["adv_30d"],
            stop_loss_pct=0.15,
        )

        r = result.daily_returns
        eq = result.equity_curve
        spy = spy_series.reindex(r.index).fillna(0)

        # Compute metrics
        metrics = {
            "test": test_name,
            "sharpe": sharpe_ratio(r),
            "ann_return": annualized_return(r),
            "ann_vol": annualized_volatility(r),
            "sortino": sortino_ratio(r),
            "calmar": calmar_ratio(r, eq),
            "max_dd": max_drawdown(eq),
            "beta": beta_to_market(r, spy),
            "win_rate": win_rate(r),
            "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
            "final_nav": eq.iloc[-1],
        }

        # OOS metrics
        try:
            oos = oos_split_tearsheet(result, benchmark_returns=spy, oos_start="2024-01-01")
            if "Out-of-Sample" in oos.index:
                oos_row = oos.loc["Out-of-Sample"]
                metrics["oos_sharpe"] = float(str(oos_row.get("Sharpe", "0")).replace("%", ""))
                metrics["oos_return"] = str(oos_row.get("Ann. Return", "0%"))
        except Exception:
            metrics["oos_sharpe"] = np.nan

        metrics["elapsed_sec"] = time.time() - t0

    except Exception as e:
        metrics = {
            "test": test_name,
            "sharpe": np.nan,
            "ann_return": np.nan,
            "error": str(e)[:200],
            "elapsed_sec": time.time() - t0,
        }

    return metrics


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def get_baseline_portfolio_kwargs(data):
    """Default portfolio parameters."""
    quality_signal = data["alt_features_dict"].get("roe_signal") if data["alt_features_dict"] else None
    return {
        "n_positions": 20,
        "weighting": "signal",
        "concentration": 1.0,
        "max_weight": 0.10,
        "min_weight": 0.02,
        "sector_map": data["sector_map"],
        "max_sector_pct": 0.35,
        "momentum_filter": data["mom_63d"],
        "realized_vol": data["rvol"],
        "returns": data["returns"],
        "regime": data["regime"],
        "adv": data["adv_30d"],
        "cash_in_bear": 0.15,
        "quality_filter": quality_signal,
    }


def build_tests(data):
    """Return list of (test_name, signal_kwargs, portfolio_kwargs_override) tuples.

    signal_kwargs may include a reserved 'vol_target' key (float) which is
    stripped before passing to the model and applied to portfolio weights.
    """
    tests = []
    base_pk = get_baseline_portfolio_kwargs(data)

    # ── A: HOLDING PERIOD / FORWARD WINDOW ────────────────────────────
    # This is the highest-impact parameter based on IC decay analysis
    tests.append(("A1_fwd5d", {"forward_window": 5}, {}))
    tests.append(("A2_fwd10d", {"forward_window": 10}, {}))
    tests.append(("A3_fwd21d_baseline", {"forward_window": 21}, {}))
    tests.append(("A4_fwd1d", {"forward_window": 1}, {}))

    # ── B: ML/MOMENTUM BLEND ─────────────────────────────────────────
    tests.append(("B1_ml90_mom10", {"ml_blend": 0.90, "mom_blend": 0.10}, {}))
    tests.append(("B2_ml70_mom30_baseline", {"ml_blend": 0.70, "mom_blend": 0.30}, {}))
    tests.append(("B3_ml50_mom50", {"ml_blend": 0.50, "mom_blend": 0.50}, {}))
    tests.append(("B4_ml30_mom70", {"ml_blend": 0.30, "mom_blend": 0.70}, {}))
    tests.append(("B5_mom_only", {"ml_blend": 0.0, "mom_blend": 1.0}, {}))

    # ── C: PORTFOLIO CONSTRUCTION ─────────────────────────────────────
    tests.append(("C1_pos15", {}, {"n_positions": 15}))
    tests.append(("C2_pos20_baseline", {}, {"n_positions": 20}))
    tests.append(("C3_pos25", {}, {"n_positions": 25}))
    tests.append(("C4_pos30", {}, {"n_positions": 30}))
    tests.append(("C5_conc0.5", {}, {"concentration": 0.5}))
    tests.append(("C6_conc1.0_baseline", {}, {"concentration": 1.0}))
    tests.append(("C7_conc1.5", {}, {"concentration": 1.5}))
    tests.append(("C8_equal_weight", {}, {"weighting": "equal"}))
    tests.append(("C9_maxwt5pct", {}, {"max_weight": 0.05}))
    tests.append(("C10_maxwt15pct", {}, {"max_weight": 0.15}))
    tests.append(("C11_bear_cash0", {}, {"cash_in_bear": 0.0}))
    tests.append(("C12_bear_cash15_baseline", {}, {"cash_in_bear": 0.15}))
    tests.append(("C13_bear_cash30", {}, {"cash_in_bear": 0.30}))
    tests.append(("C14_no_quality_filter", {}, {"quality_filter": None}))
    tests.append(("C15_no_momentum_filter", {}, {"momentum_filter": None}))

    # ── D: MODEL HYPERPARAMETERS ──────────────────────────────────────
    tests.append(("D1_leaves15", {"num_leaves": 15}, {}))
    tests.append(("D2_leaves31_baseline", {"num_leaves": 31}, {}))
    tests.append(("D3_leaves50", {"num_leaves": 50}, {}))
    tests.append(("D4_lr0.03", {"learning_rate": 0.03}, {}))
    tests.append(("D5_lr0.05_baseline", {"learning_rate": 0.05}, {}))
    tests.append(("D6_lr0.10", {"learning_rate": 0.10}, {}))
    tests.append(("D7_lgbm50_ridge50", {"lgbm_weight": 0.50, "xgb_weight": 0.0,
                                         "ridge_weight": 0.50, "mlp_weight": 0.0}, {}))
    tests.append(("D8_lgbm30_xgb25_ridge25_mlp20_baseline", {}, {}))

    # ── E: LIGHTGBM REGULARIZATION (NEW — previously hardcoded) ────────
    # L1/L2 and min_gain_to_split were hardcoded at 5.0/10.0/0.05. Test whether
    # current defaults are over- or under-regularized for this feature count.
    tests.append(("E1_l1_0.5_l2_1",   {"lambda_l1": 0.5,  "lambda_l2": 1.0},   {}))
    tests.append(("E2_l1_2_l2_5",     {"lambda_l1": 2.0,  "lambda_l2": 5.0},   {}))
    tests.append(("E3_l1_5_l2_10_baseline", {"lambda_l1": 5.0, "lambda_l2": 10.0}, {}))
    tests.append(("E4_l1_10_l2_20",   {"lambda_l1": 10.0, "lambda_l2": 20.0},  {}))
    tests.append(("E5_l1_25_l2_50",   {"lambda_l1": 25.0, "lambda_l2": 50.0},  {}))
    tests.append(("E6_mindata50",     {"min_child_samples": 50},               {}))
    tests.append(("E7_mindata100",    {"min_child_samples": 100},              {}))
    tests.append(("E8_maxdepth4",     {"max_depth": 4},                        {}))
    tests.append(("E9_maxdepth10",    {"max_depth": 10},                       {}))
    tests.append(("E10_featfrac0.3",  {"feature_fraction": 0.3},               {}))
    tests.append(("E11_featfrac0.7",  {"feature_fraction": 0.7},               {}))
    tests.append(("E12_featfrac0.9",  {"feature_fraction": 0.9},               {}))

    # ── F: RIDGE ALPHA (previously hardcoded at 50.0, Optuna suggestion ignored) ─
    tests.append(("F1_ridge5",    {"ridge_alpha": 5.0},   {}))
    tests.append(("F2_ridge20",   {"ridge_alpha": 20.0},  {}))
    tests.append(("F3_ridge50_baseline", {"ridge_alpha": 50.0}, {}))
    tests.append(("F4_ridge100",  {"ridge_alpha": 100.0}, {}))
    tests.append(("F5_ridge200",  {"ridge_alpha": 200.0}, {}))

    # ── G: ENSEMBLE MIX (never walk-forward validated) ─────────────────
    tests.append(("G1_lgbm_only",   {"lgbm_weight": 1.0, "xgb_weight": 0.0,
                                      "ridge_weight": 0.0, "mlp_weight": 0.0}, {}))
    tests.append(("G2_ridge_only",  {"lgbm_weight": 0.0, "xgb_weight": 0.0,
                                      "ridge_weight": 1.0, "mlp_weight": 0.0}, {}))
    tests.append(("G3_gbm_heavy",   {"lgbm_weight": 0.50, "xgb_weight": 0.30,
                                      "ridge_weight": 0.20, "mlp_weight": 0.0}, {}))
    # Fix 5: G4 and G6 have mlp_weight>0 → must set use_mlp=True or MLP is
    # silently zeroed and the "linear-heavy" / "no-xgb" configs mis-measure.
    tests.append(("G4_linear_heavy",{"lgbm_weight": 0.25, "xgb_weight": 0.15,
                                      "ridge_weight": 0.50, "mlp_weight": 0.10,
                                      "use_mlp": True}, {}))
    tests.append(("G5_no_mlp",      {"lgbm_weight": 0.40, "xgb_weight": 0.30,
                                      "ridge_weight": 0.30, "mlp_weight": 0.0}, {}))
    tests.append(("G6_no_xgb",      {"lgbm_weight": 0.40, "xgb_weight": 0.0,
                                      "ridge_weight": 0.35, "mlp_weight": 0.25,
                                      "use_mlp": True}, {}))

    # ── H: TRAINING WINDOW (min_train_days was never tested) ───────────
    tests.append(("H1_train126",  {"min_train_days": 126},  {}))
    tests.append(("H2_train252_baseline", {"min_train_days": 252}, {}))
    tests.append(("H3_train504",  {"min_train_days": 504},  {}))
    tests.append(("H4_train756",  {"min_train_days": 756},  {}))

    # ── V: VOLATILITY TARGETING (Moreira & Muir 2017) ──────────────────
    # Scale gross exposure inversely to ex-ante portfolio vol.
    # Expected +0.2 to +0.4 Sharpe lift per academic literature.
    tests.append(("V1_voltgt10",  {"vol_target": 0.10}, {}))
    tests.append(("V2_voltgt12",  {"vol_target": 0.12}, {}))
    tests.append(("V3_voltgt15",  {"vol_target": 0.15}, {}))
    tests.append(("V4_voltgt18",  {"vol_target": 0.18}, {}))
    tests.append(("V5_voltgt20",  {"vol_target": 0.20}, {}))

    # ── X: COMBINED "BEST GUESS" CONFIGS ───────────────────────────────
    # Stack known winners: no momentum filter, momentum-heavy blend, vol target.
    tests.append(("X1_no_mom_filter_voltgt15", {"vol_target": 0.15}, {"momentum_filter": None}))
    tests.append(("X2_mom_heavy_voltgt15", {"ml_blend": 0.30, "mom_blend": 0.70, "vol_target": 0.15}, {}))
    tests.append(("X3_pure_mom_voltgt12", {"ml_blend": 0.0, "mom_blend": 1.0, "vol_target": 0.12}, {}))
    tests.append(("X4_ml50_mom50_voltgt15_15pos",
                  {"ml_blend": 0.50, "mom_blend": 0.50, "vol_target": 0.15}, {"n_positions": 15}))
    tests.append(("X5_no_mom_filter_mom_heavy",
                  {"ml_blend": 0.30, "mom_blend": 0.70}, {"momentum_filter": None}))
    tests.append(("X6_ensemble_gbm_heavy_voltgt15",
                  {"lgbm_weight": 0.50, "xgb_weight": 0.30, "ridge_weight": 0.20,
                   "mlp_weight": 0.0, "vol_target": 0.15}, {}))

    # ──────────────────────────────────────────────────────────────────────
    # P-series: Portfolio beta-attack (quick-mode compatible, reuses cached ML)
    # ──────────────────────────────────────────────────────────────────────
    # P1 — SPY Core Blend (direct beta attack)
    tests.append(("P1a_spy0_baseline", {}, {"spy_core_weight": 0.0}))
    tests.append(("P1b_spy20",          {}, {"spy_core_weight": 0.20}))
    tests.append(("P1c_spy30",          {}, {"spy_core_weight": 0.30}))
    tests.append(("P1d_spy40",          {}, {"spy_core_weight": 0.40}))
    tests.append(("P1e_spy50",          {}, {"spy_core_weight": 0.50}))
    tests.append(("P1f_spy60",          {}, {"spy_core_weight": 0.60}))

    # P2 — Mega-Cap Forcing (beta fix without index)
    tests.append(("P2a_mega_only",      {}, {"force_mega_caps": True, "spy_core_weight": 0.0}))
    tests.append(("P2b_mega_spy20",     {}, {"force_mega_caps": True, "spy_core_weight": 0.20}))
    tests.append(("P2c_mega_spy40",     {}, {"force_mega_caps": True, "spy_core_weight": 0.40}))
    tests.append(("P2d_no_mega_spy40",  {}, {"force_mega_caps": False, "spy_core_weight": 0.40}))
    tests.append(("P2e_control",        {}, {"force_mega_caps": False, "spy_core_weight": 0.0}))

    # P4 — Vol Targeting × Leverage
    tests.append(("P4d_vt18_lev15",     {"vol_target": 0.18}, {"max_leverage": 1.5}))
    tests.append(("P4f_no_vt",          {"vol_target": 0.0},  {"max_leverage": 1.0}))

    # P5 — Signal Post-Processing
    tests.append(("P5b_rank_only_baseline", {}, {"signal_smooth_halflife": 0.0, "apply_rank_normal": True}))
    tests.append(("P5d_smooth5",            {}, {"signal_smooth_halflife": 5.0, "apply_rank_normal": True}))

    # P6 — Factor Neutralization
    tests.append(("P6a_none_baseline",  {}, {"neutralize_factors": None}))
    tests.append(("P6b_size",           {}, {"neutralize_factors": ["size"]}))

    # P7 — Stacked Beta-Fix ("best guess" X-style for Tier 1)
    tests.append(("P7a_spy40_mega_vt16", {"vol_target": 0.16},
                  {"spy_core_weight": 0.40, "force_mega_caps": True, "max_leverage": 1.3}))
    tests.append(("P7b_spy30_mega_neutsize", {},
                  {"spy_core_weight": 0.30, "force_mega_caps": True, "neutralize_factors": ["size"]}))
    tests.append(("P7d_full_stack", {"vol_target": 0.16},
                  {"spy_core_weight": 0.30, "force_mega_caps": True, "use_vol_buckets": True,
                   "max_leverage": 1.3, "signal_smooth_halflife": 5.0,
                   "neutralize_factors": ["size"]}))
    tests.append(("P7e_aggressive", {"vol_target": 0.18},
                  {"spy_core_weight": 0.20, "force_mega_caps": True,
                   "max_leverage": 1.5, "signal_smooth_halflife": 3.0}))

    # P8 — Concentration × SPY Core
    tests.append(("P8b_15pos_spy40", {},
                  {"n_positions": 15, "spy_core_weight": 0.40, "max_weight": 0.12}))
    tests.append(("P8e_20pos_spy40", {},
                  {"n_positions": 20, "spy_core_weight": 0.40, "max_weight": 0.10}))

    # ──────────────────────────────────────────────────────────────────────
    # L-series: Label engineering (requires ML retrain)
    # ──────────────────────────────────────────────────────────────────────
    tests.append(("L1_fwd5_clean_default",
                  {"forward_window": 5, "risk_adjust": False, "sector_rank_weight": 0.0}, {}))
    tests.append(("L1_fwd5_riskadj_legacy",
                  {"forward_window": 5, "risk_adjust": True, "sector_rank_weight": 0.5}, {}))
    tests.append(("L2_bn_fwd5",
                  {"forward_window": 5, "beta_neutral": True}, {}))
    tests.append(("L3_mh_1_5_21",
                  {"forward_windows": [1, 5, 21]}, {}))
    tests.append(("L4_trunc20_default",
                  {"lambdarank_truncation": 20}, {}))
    tests.append(("L5_huber_0.2",
                  {"huber_weight": 0.2}, {}))
    tests.append(("L5_quantile_0.2_a75",
                  {"quantile_weight": 0.2, "quantile_alpha": 0.75}, {}))
    tests.append(("L8_stack_core",
                  {"forward_window": 5, "beta_neutral": True,
                   "lambdarank_truncation": 20}, {}))
    tests.append(("L8_stack_full",
                  {"forward_windows": [1, 5, 21], "beta_neutral": True,
                   "lambdarank_truncation": 20, "huber_weight": 0.2,
                   "quantile_weight": 0.2, "quantile_alpha": 0.75,
                   "use_magnitude_weights": True, "use_meta_labeling": True}, {}))

    # ──────────────────────────────────────────────────────────────────────
    # R-series: Regularization (min_data_in_leaf)
    # ──────────────────────────────────────────────────────────────────────
    tests.append(("R1_4_mdil200_baseline", {"min_data_in_leaf": 200}, {}))
    tests.append(("R1_5_mdil300",          {"min_data_in_leaf": 300}, {}))

    # ──────────────────────────────────────────────────────────────────────
    # Z-series: Training hygiene stacks
    # ──────────────────────────────────────────────────────────────────────
    tests.append(("Z1_hygiene_full",
                  {"min_data_in_leaf": 300, "early_stop_on_ic": True,
                   "use_uniqueness_weights": True, "use_per_date_weights": True,
                   "lgbm_num_seeds": 5}, {}))
    tests.append(("Z5_hygiene_no_seeds",
                  {"min_data_in_leaf": 300, "early_stop_on_ic": True,
                   "use_uniqueness_weights": True, "use_per_date_weights": True,
                   "lgbm_num_seeds": 1}, {}))

    # ──────────────────────────────────────────────────────────────────────
    # FA/FB/FG/FH: Feature ablation (panel rebuild + ML retrain)
    # panel_flags/alt_flags are stashed inside signal_kwargs under reserved
    # keys that main() pops before calling generate_ml_signal.
    # ──────────────────────────────────────────────────────────────────────
    tests.append(("FA1_baseline",
                  {"__panel_flags__": {"size_neutralize": True,
                                        "sector_neutralize_features": True,
                                        "winsorize": True, "cs_zscore_all": True,
                                        "use_higher_moment": True,
                                        "use_breadth_wavelet": True},
                   "__alt_flags__": {}}, {}))
    tests.append(("FA2_sn_off",
                  {"__panel_flags__": {"size_neutralize": False},
                   "__alt_flags__": {}}, {}))
    tests.append(("FB2_sni_off",
                  {"__panel_flags__": {"sector_neutralize_features": False},
                   "__alt_flags__": {}}, {}))
    tests.append(("FG2_all_distress_off",
                  {"__panel_flags__": {},
                   "__alt_flags__": {"use_chs": False, "use_dtd": False,
                                      "use_altman_z": False}}, {}))
    tests.append(("FH2_macro_off",
                  {"__panel_flags__": {},
                   "__alt_flags__": {"use_macro_cross_section": False}}, {}))

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--quick", action="store_true",
                        help="Portfolio tests only (skip ML retraining)")
    parser.add_argument("--category", default=None,
                        help="Run specific category: A, B, C, D, E, F, G, H, V, X, "
                             "P, L, R, Z, FA, FB, FG, FH "
                             "(comma-separate for multiple, e.g. 'V,X' or 'P1,P7')")
    args = parser.parse_args()

    RESULTS_FILE.parent.mkdir(exist_ok=True)

    # Load all data
    data = load_shared_data(args)

    # Build test list
    all_tests = build_tests(data)

    # Filter by category (accepts comma-separated list)
    if args.category:
        cats = tuple(c.strip() for c in args.category.split(",") if c.strip())
        all_tests = [(n, s, p) for n, s, p in all_tests if n.startswith(cats)]
    elif args.quick:
        # Quick mode: portfolio + vol-target tests that reuse cached ML
        all_tests = [(n, s, p) for n, s, p in all_tests
                     if n.startswith(("B", "C", "V", "X", "P"))]

    print(f"\n{'='*60}")
    print(f"  TEST SUITE: {len(all_tests)} tests")
    print(f"{'='*60}\n")

    # Load existing results (crash-safe resume)
    existing = set()
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        existing = set(df["test"].tolist())
        print(f"Resuming: {len(existing)} tests already complete")

    results = []
    base_pk = get_baseline_portfolio_kwargs(data)
    spy_series = data["spy_series"]

    for i, (test_name, signal_kwargs, pk_override) in enumerate(all_tests):
        if test_name in existing:
            print(f"[{i+1}/{len(all_tests)}] {test_name}: CACHED (skipping)")
            continue

        print(f"[{i+1}/{len(all_tests)}] {test_name}...", end=" ", flush=True)

        # Generate ML signal (may use cache if params match baseline)
        sig_params = {
            # Tier 2: horizon fix — clean 5d forward window, shorter warmup
            "forward_window": 5,
            "min_train_days": 126,
            "ml_blend": 0.70,
            "mom_blend": 0.30,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "min_child_samples": 20,
            "lambda_l1": 5.0,
            "lambda_l2": 10.0,
            "max_depth": 7,
            "ridge_alpha": 50.0,
            "lgbm_weight": 0.30,
            "xgb_weight": 0.25,
            "ridge_weight": 0.25,
            "mlp_weight": 0.20,
            # Tier 6 label/model defaults — matched to model.py defaults so
            # signal_kwargs overrides compose cleanly via .update()
            "risk_adjust": False,
            "sector_rank_weight": 0.0,
            "beta_neutral": False,
            "forward_windows": None,
            "lambdarank_truncation": 20,
            "use_return_decile_gain": False,
            "use_magnitude_weights": False,
            "huber_weight": 0.0,
            "quantile_weight": 0.0,
            "quantile_alpha": 0.75,
            "use_meta_labeling": False,
            "min_data_in_leaf": 200,
            "early_stop_on_ic": True,
            "use_uniqueness_weights": True,
            "use_per_date_weights": True,
            "lgbm_num_seeds": 1,
            "use_mlp": False,
            "monotone_constraints": None,  # None → default_dict inside WalkForwardModel
        }
        sig_params.update(signal_kwargs)

        # Extract non-model keys
        ml_blend = sig_params.pop("ml_blend")
        mom_blend = sig_params.pop("mom_blend")
        fwd_window = sig_params.pop("forward_window")
        min_train = sig_params.pop("min_train_days")
        vol_target = sig_params.pop("vol_target", 0.0)
        # Fix 6: extract feature-ablation flags and rebuild panel if provided
        panel_flags = sig_params.pop("__panel_flags__", None)
        alt_flags = sig_params.pop("__alt_flags__", None)

        custom_panel = None
        if panel_flags is not None or alt_flags is not None:
            custom_panel, _ = build_panel_for_config(
                data,
                panel_flags=panel_flags or {},
                alt_flags=alt_flags or {},
            )

        signal, wf = generate_ml_signal(
            data, forward_window=fwd_window, min_train_days=min_train,
            ml_blend=ml_blend, mom_blend=mom_blend,
            panel=custom_panel, **sig_params,
        )

        # Build portfolio kwargs
        pk = dict(base_pk)
        pk.update(pk_override)

        metrics = run_single_test(test_name, data, signal, pk, spy_series,
                                  vol_target=vol_target)
        results.append(metrics)

        # Print summary
        s = metrics.get("sharpe", np.nan)
        r = metrics.get("ann_return", np.nan)
        dd = metrics.get("max_dd", np.nan)
        t = metrics.get("elapsed_sec", 0)
        print(f"Sharpe={s:.3f}  Return={r*100:.1f}%  MaxDD={dd*100:.1f}%  ({t:.0f}s)")

        # Save incrementally
        all_results = pd.DataFrame(results)
        if RESULTS_FILE.exists():
            prev = pd.read_csv(RESULTS_FILE)
            all_results = pd.concat([prev, all_results], ignore_index=True)
        all_results.to_csv(RESULTS_FILE, index=False)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}\n")

    df = pd.read_csv(RESULTS_FILE)
    df = df.sort_values("sharpe", ascending=False)

    print("TOP 10 by Sharpe:")
    print(df[["test", "sharpe", "ann_return", "max_dd", "total_return"]].head(10).to_string(index=False))
    print()
    print("BOTTOM 5 by Sharpe:")
    print(df[["test", "sharpe", "ann_return", "max_dd", "total_return"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
