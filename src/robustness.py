"""
robustness.py
-------------
Phase 3: Rigorous robustness testing.

These tests answer the hardest interview question: "How do you know
your backtest isn't just overfitted luck?"

Three approaches:
  1. Bootstrap (resample returns with replacement)
     -> Confidence interval on Sharpe. If p5 > 0, signal is likely real.

  2. Permutation test (shuffle signal labels)
     -> If random signals score as well as yours, your signal has no edge.

  3. Capacity analysis (vary AUM)
     -> At what AUM does transaction cost drag kill the strategy?
     -> Required for any serious quant interview discussion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(
    returns:      pd.Series,
    metric_fn,
    n_simulations: int = 2000,
    seed:         int = 42,
    block_size:   int = 20,
) -> Dict:
    """
    Block bootstrap for time-series data.

    Standard (IID) bootstrap assumes returns are independent.
    Block bootstrap preserves short-range autocorrelation by
    resampling contiguous blocks — more appropriate for financial data.

    Parameters
    ----------
    returns       : daily return series
    metric_fn     : function that takes a pd.Series and returns a float
    n_simulations : number of bootstrap replications
    block_size    : number of consecutive days per block (~1 month)

    Returns
    -------
    dict with observed, p5, p25, p50, p75, p95, std, prob_positive
    """
    rng         = np.random.default_rng(seed)
    observed    = metric_fn(returns)
    n           = len(returns)
    n_blocks    = int(np.ceil(n / block_size))
    r_arr       = returns.values
    bootstrapped = []

    for _ in range(n_simulations):
        # Sample block start indices
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r_arr[s:s + block_size] for s in block_starts])[:n]
        bootstrapped.append(metric_fn(pd.Series(sample)))

    bootstrapped = np.array(bootstrapped)

    return {
        "observed":      observed,
        "p5":            np.percentile(bootstrapped, 5),
        "p25":           np.percentile(bootstrapped, 25),
        "p50":           np.percentile(bootstrapped, 50),
        "p75":           np.percentile(bootstrapped, 75),
        "p95":           np.percentile(bootstrapped, 95),
        "std":           bootstrapped.std(),
        "prob_positive": (bootstrapped > 0).mean(),
        "n_simulations": n_simulations,
    }


def bootstrap_tearsheet(
    returns:       pd.Series,
    n_simulations: int = 1000,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for multiple metrics simultaneously.
    Returns a formatted DataFrame suitable for the README / report.
    """
    from metrics import sharpe_ratio, sortino_ratio, annualized_return

    metric_fns = {
        "Sharpe Ratio":       lambda r: sharpe_ratio(r),
        "Sortino Ratio":      lambda r: sortino_ratio(r),
        "Annualized Return":  lambda r: annualized_return(r),
    }

    rows = []
    for name, fn in metric_fns.items():
        res = bootstrap_metric(returns, fn, n_simulations=n_simulations)
        rows.append({
            "metric":    name,
            "observed":  f"{res['observed']:.3f}",
            "p5":        f"{res['p5']:.3f}",
            "p50":       f"{res['p50']:.3f}",
            "p95":       f"{res['p95']:.3f}",
            "p(>0)":     f"{res['prob_positive']*100:.1f}%",
        })

    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------------------
# 2. Permutation / label-shuffle test
# ---------------------------------------------------------------------------

def permutation_test(
    signal:    pd.DataFrame,
    returns:   pd.DataFrame,
    n_permutations: int = 200,
    forward_window: int = 5,
    seed:      int = 42,
) -> Dict:
    """
    Permutation test: shuffle the cross-sectional signal labels and
    compute IC on shuffled data.

    If your true IC is not significantly above the shuffled distribution,
    your signal has no edge beyond noise.

    Returns
    -------
    dict with: true_IC, shuffled_mean, shuffled_std, p_value, z_score
    """
    from features import factor_decay_analysis

    rng = np.random.default_rng(seed)

    # True IC
    true_decay = factor_decay_analysis(signal, returns, horizons=[forward_window])
    true_ic    = true_decay.loc[f"{forward_window}d_IC", "mean_IC"]

    # Shuffled ICs
    shuffled_ics = []
    tickers      = signal.columns.tolist()

    for i in range(n_permutations):
        print(f"  [permutation] {i+1}/{n_permutations}", end="\r")
        # Shuffle ticker labels cross-sectionally on each day
        shuffled = signal.copy()
        for date in shuffled.index:
            row = shuffled.loc[date].values.copy()
            rng.shuffle(row)
            shuffled.loc[date] = row

        decay     = factor_decay_analysis(shuffled, returns, horizons=[forward_window])
        shuf_ic   = decay.loc[f"{forward_window}d_IC", "mean_IC"]
        shuffled_ics.append(shuf_ic)

    print()
    shuffled_ics = np.array(shuffled_ics)
    shuf_mean    = shuffled_ics.mean()
    shuf_std     = shuffled_ics.std()
    z_score      = (true_ic - shuf_mean) / shuf_std if shuf_std > 0 else 0
    p_value      = (shuffled_ics >= true_ic).mean()

    return {
        "true_IC":       true_ic,
        "shuffled_mean": shuf_mean,
        "shuffled_std":  shuf_std,
        "z_score":       z_score,
        "p_value":       p_value,
        "significant":   p_value < 0.05,
        "interpretation": (
            "Signal is statistically significant (p < 0.05)" if p_value < 0.05
            else "Signal is NOT significant — may be noise"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Capacity analysis
# ---------------------------------------------------------------------------

def capacity_analysis(
    weights:      pd.DataFrame,
    prices:       pd.DataFrame,
    aum_levels:   List[float] = None,
    adv_fraction: float = 0.10,  # max % of ADV we're willing to trade
) -> pd.DataFrame:
    """
    How does strategy performance degrade as AUM increases?

    The key insight: slippage scales with trade_size / ADV.
    A $1M trade in a stock with $100M ADV = 1% of ADV (low impact).
    A $100M trade in the same stock = 100% of ADV (massive impact, illiquid).

    We model this using a square-root market impact model:
        impact_bps = base_impact_bps * sqrt(trade_size / ADV)

    This is the industry-standard Almgren-Chriss impact approximation.

    Parameters
    ----------
    aum_levels    : list of AUM values to test (dollars)
    adv_fraction  : warn if trade > this fraction of ADV

    Returns
    -------
    DataFrame: rows = AUM levels, columns = net Sharpe, net return, total cost
    """
    from backtest    import run_backtest, TransactionCostModel
    from metrics     import sharpe_ratio, annualized_return

    if aum_levels is None:
        aum_levels = [1e6, 10e6, 50e6, 100e6, 250e6, 500e6, 1e9]

    volume  = prices["Volume"]
    close   = prices["Close"]
    avg_adv = (volume * close).rolling(20).mean()  # rolling ADV in dollars

    records = []

    for aum in aum_levels:
        # Compute position-level dollar sizes
        position_dollars = weights.abs() * aum

        # Average trade size across all positions and days
        trade_size     = (weights.diff().abs() * aum)
        avg_trade      = trade_size.mean().mean()

        # Compare against ADV
        common_dates   = trade_size.index.intersection(avg_adv.index)
        common_tickers = trade_size.columns.intersection(avg_adv.columns)

        if len(common_dates) == 0 or len(common_tickers) == 0:
            pct_adv = np.nan
        else:
            ts_aligned  = trade_size.loc[common_dates, common_tickers]
            adv_aligned = avg_adv.loc[common_dates, common_tickers]
            adv_ratio   = (ts_aligned / adv_aligned.replace(0, np.nan)).mean().mean()
            pct_adv     = adv_ratio * 100

        # Slippage scales with sqrt(trade / ADV) — square root impact model
        # base_slippage_bps = 5 bps at $10M AUM
        base_aum        = 10e6
        base_slip       = 5.0
        scaled_slippage = base_slip * np.sqrt(aum / base_aum)
        scaled_slippage = min(scaled_slippage, 200.0)  # cap at 200 bps

        cost_model = TransactionCostModel(
            spread_bps=3.0,
            commission_bps=1.0,
            slippage_bps=scaled_slippage,
        )

        result    = run_backtest(
            weights, prices,
            initial_capital=aum,
            cost_model=cost_model,
            verbose=False,
        )
        sharpe    = sharpe_ratio(result.daily_returns)
        ann_ret   = annualized_return(result.daily_returns)
        total_tc  = result.transaction_costs.sum()
        tc_drag   = total_tc / aum * 100  # as % of initial AUM

        records.append({
            "AUM":              f"${aum/1e6:.0f}M",
            "scaled_slip_bps":  f"{scaled_slippage:.1f}",
            "pct_of_ADV":       f"{pct_adv:.2f}%" if not np.isnan(pct_adv) else "N/A",
            "net_sharpe":       f"{sharpe:.2f}",
            "net_ann_return":   f"{ann_ret*100:.1f}%",
            "total_cost_drag":  f"{tc_drag:.2f}%",
            "viable":           "✓" if sharpe > 0.5 else "✗",
        })

    df = pd.DataFrame(records).set_index("AUM")
    return df


@dataclass
class RobustnessReport:
    """Full robustness analysis output, ready for README/presentation."""
    bootstrap:    pd.DataFrame = None
    permutation:  Dict         = None
    capacity:     pd.DataFrame = None

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  ROBUSTNESS REPORT")
        print("=" * 60)

        if self.bootstrap is not None:
            print("\n--- Bootstrap Confidence Intervals ---")
            print(self.bootstrap.to_string())

        if self.permutation is not None:
            print("\n--- Permutation Test ---")
            for k, v in self.permutation.items():
                if k != "interpretation":
                    print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
            print(f"\n  -> {self.permutation['interpretation']}")

        if self.capacity is not None:
            print("\n--- Capacity Analysis ---")
            print(self.capacity.to_string())
        print()


def run_full_robustness(
    strategy_returns: pd.Series,
    signal:           pd.DataFrame,
    returns:          pd.DataFrame,
    weights:          pd.DataFrame,
    prices:           pd.DataFrame,
    n_bootstrap:      int = 500,
    n_permutations:   int = 100,
) -> RobustnessReport:
    """Run all robustness checks and return a consolidated report."""
    report = RobustnessReport()

    print("[robustness] Running bootstrap confidence intervals...")
    report.bootstrap = bootstrap_tearsheet(strategy_returns, n_simulations=n_bootstrap)

    print("[robustness] Running capacity analysis...")
    report.capacity  = capacity_analysis(weights, prices)

    # Permutation test is slow — only run on a sample if requested
    if n_permutations > 0:
        print(f"[robustness] Running permutation test ({n_permutations} shuffles)...")
        report.permutation = permutation_test(
            signal, returns, n_permutations=n_permutations
        )

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_prices, get_close, get_returns, get_volume
    from features    import build_composite_signal, realized_volatility
    from portfolio   import compute_target_weights
    from backtest    import run_backtest

    prices   = load_prices(start="2018-01-01", end="2024-01-01")
    close    = get_close(prices)
    returns  = get_returns(prices)
    volume   = get_volume(prices)

    composite, _ = build_composite_signal(close, returns, volume)
    rvol         = realized_volatility(returns, window=21)
    weights      = compute_target_weights(composite, rvol)
    result       = run_backtest(weights, prices, verbose=False)

    print("Bootstrap tearsheet:")
    bt = bootstrap_tearsheet(result.daily_returns, n_simulations=500)
    print(bt)

    print("\nCapacity analysis:")
    cap = capacity_analysis(weights, prices)
    print(cap)
