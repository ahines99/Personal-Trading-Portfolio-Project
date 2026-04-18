"""
regime.py
---------
Phase 3: Market regime detection and performance attribution.

WHY REGIME ANALYSIS MATTERS IN INTERVIEWS:
  Every interviewer at a top quant fund will ask: "How does your strategy
  perform in different market environments?" If you can't answer this,
  you haven't thought hard enough about your strategy.

  Regime analysis answers:
  - Does your momentum signal break down in bear markets? (It usually does)
  - Does mean-reversion work better in high-vol environments? (Often yes)
  - Is your Sharpe consistent, or driven by one lucky period?

REGIMES IMPLEMENTED:
  1. Volatility regime: rolling VIX-proxy (realized market vol)
     Low vol = calm bull / High vol = fear / Stress = crisis
  2. Trend regime: market in uptrend vs downtrend (200-day MA)
  3. Combined: 4-state (bull calm, bull volatile, bear calm, bear volatile)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_volatility_regime(
    market_returns: pd.Series,
    short_window:   int = 21,
    low_vol_pct:    float = 0.33,
    high_vol_pct:   float = 0.67,
) -> pd.Series:
    """
    Classify each day as low / medium / high volatility regime.

    Uses rolling realized volatility relative to its own history
    (expanding percentile rank) — avoids forward-looking thresholds.

    Returns
    -------
    regime : pd.Series with values "low_vol", "med_vol", "high_vol"
    """
    rvol        = market_returns.rolling(short_window).std() * np.sqrt(252)
    # Rolling 252-day percentile rank (no lookahead, avoids small-sample
    # bias from expanding rank where early days rank among 20 observations).
    # PERF fix (LOW): previously used .apply(lambda x: pd.Series(x).rank(...)
    # .iloc[-1], raw=False) which reconstructs a Series on every window and
    # runs ~20x slower than pandas' native vectorized rolling.rank. The
    # .rank(pct=True) call returns the percentile rank of the LAST element
    # within each window — which is exactly the semantic we want.
    rvol_rank   = rvol.rolling(252, min_periods=63).rank(pct=True)

    regime = pd.Series("med_vol", index=market_returns.index)
    regime[rvol_rank <= low_vol_pct]  = "low_vol"
    regime[rvol_rank >= high_vol_pct] = "high_vol"

    return regime


def detect_trend_regime(
    market_prices: pd.Series,
    ma_window:     int = 200,
) -> pd.Series:
    """
    Classify each day as bull or bear based on 200-day moving average.

    price > 200d MA → bull
    price < 200d MA → bear

    The 200d MA is the most widely-used trend signal in institutional trading.
    """
    ma     = market_prices.rolling(ma_window, min_periods=ma_window // 2).mean()
    # FIX (MED bug): previously returned "bear" by default, which mislabeled the
    # MA warmup window (first ~100 days where MA is NaN) as bear markets even
    # during raging bull runs. Now we emit "unknown" until the MA is populated
    # so downstream consumers can exclude those dates from regime analysis.
    regime = pd.Series(index=market_prices.index, dtype=object)
    regime[ma.isna()] = "unknown"
    regime[(ma.notna()) & (market_prices > ma)]  = "bull"
    regime[(ma.notna()) & (market_prices <= ma)] = "bear"
    return regime


def detect_combined_regime(
    market_returns: pd.Series,
    market_prices:  pd.Series,
) -> pd.Series:
    """
    4-state combined regime:
      bull_calm    : uptrend + low vol  → momentum works best
      bull_volatile: uptrend + high vol → mixed
      bear_calm    : downtrend + low vol → mean-reversion opportunity
      bear_volatile: downtrend + high vol → most dangerous
    """
    vol_regime   = detect_volatility_regime(market_returns)
    trend_regime = detect_trend_regime(market_prices)

    # Simplify vol to binary
    vol_binary   = vol_regime.map({"low_vol": "calm", "med_vol": "calm", "high_vol": "volatile"})

    combined = trend_regime + "_" + vol_binary
    # If trend_regime is "unknown" (MA warmup) the concatenation above yields
    # "unknown_calm" / "unknown_volatile"; collapse both to a single "unknown"
    # label so downstream regime attribution treats the warmup window as one
    # excludable bucket rather than two noisy states.
    combined[trend_regime == "unknown"] = "unknown"
    return combined


def build_market_series(
    prices:          pd.DataFrame,
    market_ticker:   str = "SPY",
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract market price and return series.
    Falls back to equal-weight index if SPY not in universe.
    """
    close = prices["Close"]

    if market_ticker in close.columns:
        mkt_price  = close[market_ticker]
        mkt_return = mkt_price.pct_change()
    else:
        # FIX (MED bug): previously the equal-weight fallback used ALL columns,
        # meaning stray ETFs, warrants, or bad-data tickers with blown-up price
        # series could pollute the market proxy. Warn the user and defensively
        # drop tickers whose cumulative return over the sample is implausible
        # (e.g. delisted-to-zero survivors or 100x meme-stock outliers).
        import warnings
        warnings.warn(
            f"{market_ticker} not in price panel. Falling back to equal-weighted "
            f"universe proxy (filtered for sane cumulative returns)."
        )

        first_valid = close.bfill().iloc[0]
        last_valid  = close.ffill().iloc[-1]
        cum_ret     = last_valid / first_valid - 1
        valid_mask  = (cum_ret > -0.99) & (cum_ret < 50.0) & first_valid.notna() & last_valid.notna()
        valid_close = close.loc[:, valid_mask]

        mkt_return = valid_close.pct_change().mean(axis=1)
        mkt_price  = (1 + mkt_return.fillna(0)).cumprod()

    return mkt_price, mkt_return


# ---------------------------------------------------------------------------
# Performance attribution by regime
# ---------------------------------------------------------------------------

def performance_by_regime(
    strategy_returns: pd.Series,
    regime:           pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate:   float = 0.02,
) -> pd.DataFrame:
    """
    Compute key performance metrics for each regime state.

    This is the table you show in an interview to demonstrate
    you understand when your strategy works and when it doesn't.

    Returns
    -------
    DataFrame: rows = regime states, columns = metrics
    """
    from metrics import sharpe_ratio, annualized_return, max_drawdown, win_rate

    common = strategy_returns.index.intersection(regime.index)
    strat  = strategy_returns.loc[common]
    reg    = regime.loc[common]

    # Phase 1.8 fix: coerce mixed-type regime values to strings so sorted()
    # doesn't crash when float NaNs coexist with string labels.
    reg    = reg.astype(str).replace("nan", None).dropna()
    strat  = strat.loc[reg.index]

    records = []
    for regime_state in sorted(reg.unique()):
        mask  = reg == regime_state
        r_sub = strat[mask]

        if len(r_sub) < 20:
            continue

        ann_ret = annualized_return(r_sub)
        ann_vol = r_sub.std() * np.sqrt(252)
        sharpe  = sharpe_ratio(r_sub, risk_free_rate)
        mdd     = max_drawdown((1 + r_sub).cumprod())
        wr      = win_rate(r_sub)
        n_days  = len(r_sub)
        pct_time = n_days / len(strat) * 100

        row = {
            "regime":          regime_state,
            "n_days":          n_days,
            "pct_time":        f"{pct_time:.1f}%",
            "ann_return":      f"{ann_ret*100:.1f}%",
            "ann_volatility":  f"{ann_vol*100:.1f}%",
            "sharpe":          f"{sharpe:.2f}",
            "max_drawdown":    f"{mdd*100:.1f}%",
            "win_rate":        f"{wr*100:.1f}%",
        }

        # Active return vs benchmark if provided
        if benchmark_returns is not None:
            bench_sub   = benchmark_returns.reindex(r_sub.index).fillna(0)
            active      = r_sub - bench_sub
            active_ann  = annualized_return(active)
            row["active_return"] = f"{active_ann*100:.1f}%"

        records.append(row)

    df = pd.DataFrame(records).set_index("regime")
    return df


def regime_transition_analysis(regime: pd.Series) -> pd.DataFrame:
    """
    Analyze how often and how long each regime persists.
    Useful for understanding strategy timing risk.
    """
    records = []
    current_regime  = None
    current_start   = None
    current_count   = 0
    prev_date       = None

    # BOUNDARY fix (LOW): previously the "end" of closed episodes was set to
    # `date` (the first day of the NEXT regime), while the final episode's
    # "end" was the last day of data (still in the current regime). That made
    # durations inconsistent — closed episodes were half-open [start, end)
    # while the final one was fully inclusive [start, end]. Now ALL episodes
    # use inclusive [start, end] within the same regime state; the trailing
    # episode is flagged `ongoing=True` so callers know it may extend beyond
    # the sample.
    for date, state in regime.items():
        if state != current_regime:
            if current_regime is not None:
                records.append({
                    "regime":    current_regime,
                    "start":     current_start,
                    "end":       prev_date,    # last day IN this regime
                    "duration":  current_count,
                    "ongoing":   False,
                })
            current_regime = state
            current_start  = date
            current_count  = 1
        else:
            current_count += 1
        prev_date = date

    if current_regime is not None:
        records.append({
            "regime":   current_regime,
            "start":    current_start,
            "end":      regime.index[-1],
            "duration": current_count,
            "ongoing":  True,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    summary = df.groupby("regime")["duration"].agg(
        avg_duration="mean",
        max_duration="max",
        n_episodes="count",
    ).round(1)

    return summary


# ---------------------------------------------------------------------------
# Stress test: performance in specific historical episodes
# ---------------------------------------------------------------------------

HISTORICAL_STRESS_PERIODS = {
    "COVID crash (2020)":        ("2020-02-19", "2020-03-23"),
    "Post-COVID rally (2020)":   ("2020-03-23", "2020-08-31"),
    "Rate hike cycle (2022)":    ("2022-01-01", "2022-12-31"),
    "GFC peak-to-trough":        ("2007-10-09", "2009-03-09"),
    "Dot-com bust":              ("2000-03-10", "2002-10-09"),
    "2018 Q4 selloff":           ("2018-09-20", "2018-12-24"),
    "COVID recovery rally":      ("2020-04-01", "2021-01-31"),
    "2022 bear market":          ("2022-01-03", "2022-10-12"),
    "SVB/regional bank crisis":  ("2023-03-08", "2023-03-24"),
    "2025 tariff shock":         ("2025-04-01", "2025-04-10"),
}


def stress_test(
    strategy_returns:  pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Performance during named historical stress periods.

    Having this table in your README demonstrates awareness of
    tail risk — something quant funds care deeply about.
    """
    from metrics import sharpe_ratio, max_drawdown

    records = []
    for name, (start, end) in HISTORICAL_STRESS_PERIODS.items():
        mask = (strategy_returns.index >= start) & (strategy_returns.index <= end)
        r    = strategy_returns[mask]

        if len(r) < 5:
            records.append({"period": name, "status": "not in sample"})
            continue

        total_ret  = (1 + r).prod() - 1
        ann_vol    = r.std() * np.sqrt(252)
        mdd        = max_drawdown((1 + r).cumprod())

        row = {
            "period":       name,
            "start":        start,
            "end":          end,
            "n_days":       len(r),
            "total_return": f"{total_ret*100:.1f}%",
            "ann_vol":      f"{ann_vol*100:.1f}%",
            "max_drawdown": f"{mdd*100:.1f}%",
        }

        if benchmark_returns is not None:
            b      = benchmark_returns.reindex(r.index).fillna(0)
            b_ret  = (1 + b).prod() - 1
            active = total_ret - b_ret
            row["vs_benchmark"] = f"{active*100:+.1f}%"

        records.append(row)

    return pd.DataFrame(records).set_index("period")


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

    mkt_price, mkt_return = build_market_series(prices)
    regime = detect_combined_regime(mkt_return, mkt_price)

    print("=== Performance by Regime ===")
    print(performance_by_regime(result.daily_returns, regime))

    print("\n=== Regime Duration Analysis ===")
    print(regime_transition_analysis(regime))

    print("\n=== Stress Test ===")
    print(stress_test(result.daily_returns, mkt_return))
