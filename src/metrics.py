"""
metrics.py
----------
Hedge fund-grade performance analytics.

Every metric here is something you will be asked about in interviews.
Comments explain not just HOW to compute each metric, but WHY it matters
and what a "good" value looks like at firms like Citadel or DE Shaw.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of total risk.

    Why it matters: the universal benchmark for risk-adjusted returns.
    A Sharpe of 1.0 is "acceptable", 1.5+ is "good", 2.0+ is "great"
    for an equity L/S strategy. Jane Street pods typically target 2+.

    Note: risk_free_rate default is ~2% (average over 2013-2024 period).
    """
    daily_rf  = risk_free_rate / TRADING_DAYS
    excess    = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Like Sharpe, but penalizes only downside volatility using semi-deviation.

    Semi-deviation = sqrt(E[min(r - target, 0)^2]) — penalizes all returns
    below the target, not just the std of negative returns. This is the
    correct Sortino denominator per Sortino & van der Meer (1991).
    """
    daily_rf       = risk_free_rate / TRADING_DAYS
    excess         = returns - daily_rf
    # Semi-deviation: sqrt of mean squared downside deviation
    downside       = np.minimum(returns - daily_rf, 0.0)
    downside_vol   = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_vol == 0:
        return 0.0
    return (excess.mean() * TRADING_DAYS) / downside_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in NAV.
    Return value is negative (e.g. -0.15 = -15% drawdown).

    Why it matters: max drawdown determines strategy survivability.
    A -30% drawdown typically triggers redemptions or risk manager intervention.
    Most L/S equity books target max DD < -15%.
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Full drawdown time series (useful for plotting)."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Annualized return divided by max drawdown magnitude.
    Calmar > 1 is considered good. Measures return per unit of worst loss.
    """
    ann_return = annualized_return(returns)
    mdd        = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_return / mdd


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    IR = mean(active return) / std(active return)

    Active return = strategy return - benchmark return (daily).

    This is arguably MORE important than Sharpe for L/S strategies because:
    - Sharpe measures absolute risk-adjusted return
    - IR measures skill at OUTPERFORMING a benchmark consistently
    - IR > 0.5 is acceptable, > 1.0 is very good

    For a market-neutral strategy, benchmark_returns should be zeros.
    For a long-biased strategy, use SPY daily returns.
    """
    active = strategy_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return (active.mean() / active.std()) * np.sqrt(TRADING_DAYS)


def annualized_return(returns: pd.Series) -> float:
    """Compound annualized growth rate (CAGR)."""
    total    = (1 + returns).prod()
    n_years  = len(returns) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with positive P&L."""
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """
    Sum of gains / sum of losses (absolute).
    > 1 means strategy makes more on up days than it loses on down days.
    """
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf
    return gains / losses


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """
    95th percentile return / abs(5th percentile return).
    > 1 = right-skewed (good). < 1 = left-skewed (bad — fat left tail).
    """
    p95 = np.percentile(returns, 100 - percentile)
    p05 = abs(np.percentile(returns, percentile))
    if p05 == 0:
        return 0.0
    return p95 / p05


def avg_drawdown_duration(equity_curve: pd.Series) -> float:
    """
    Average number of trading days spent in drawdown.
    Shorter is better — means the strategy recovers quickly.
    """
    dd    = drawdown_series(equity_curve)
    in_dd = dd < 0
    durations = []
    current   = 0
    for x in in_dd:
        if x:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    return np.mean(durations) if durations else 0.0


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """
    Rolling Sharpe ratio over a window (default: 6 months).
    Useful for detecting regime changes and strategy degradation over time.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)


def beta_to_market(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Market beta: sensitivity of strategy returns to market moves.
    Target for market-neutral L/S: beta near 0.
    """
    aligned = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    cov    = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    mkt_var = np.var(aligned.iloc[:, 1], ddof=1)
    if mkt_var == 0:
        return 0.0
    return cov[0, 1] / mkt_var


def probabilistic_sharpe_ratio(
    sharpe_ratio: float,
    n_obs: int,
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: P(true_SR > benchmark_SR) with skew/kurtosis adjustment.

    Reference: Bailey & Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier".

    Returns probability in [0, 1] that the true (non-annualized) Sharpe exceeds
    the benchmark, given the observed sample Sharpe and the return distribution's
    higher moments.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3 or n_obs < 3:
        return float("nan")

    # Non-annualized Sharpe (per-period). Caller may pass annualized; convert.
    # We compute PSR on the per-period Sharpe; expected input is annualized,
    # so de-annualize by sqrt(TRADING_DAYS).
    sr = sharpe_ratio / np.sqrt(TRADING_DAYS)
    sr_bench = benchmark_sharpe / np.sqrt(TRADING_DAYS)

    skew = float(stats.skew(arr, bias=False)) if len(arr) > 2 else 0.0
    kurt = float(stats.kurtosis(arr, bias=False, fisher=True)) if len(arr) > 3 else 0.0

    # Bailey-Lopez de Prado PSR formula
    denom = np.sqrt(max(1e-12, 1 - skew * sr + ((kurt) / 4.0) * sr ** 2))
    z = (sr - sr_bench) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_trials: int,
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> dict:
    """
    Deflated Sharpe Ratio: correct observed Sharpe for multiple-testing bias and
    non-normality of returns.

    Reference: Bailey & Lopez de Prado 2014, Journal of Portfolio Management.

    Args:
        sharpe_ratio: observed (annualized) Sharpe
        n_trials: number of strategies/hyperparameter configs tested
        returns: return series used to compute Sharpe
        benchmark_sharpe: threshold to test against (annualized)

    Returns:
        dict with keys: 'sharpe', 'deflated_sharpe', 'expected_max_sharpe',
        'probability'
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_obs = len(arr)

    if n_obs < 3 or n_trials < 1:
        return {
            "sharpe": sharpe_ratio,
            "deflated_sharpe": float("nan"),
            "expected_max_sharpe": float("nan"),
            "probability": float("nan"),
        }

    # Expected maximum Sharpe under the null (per-period units)
    # E[max SR] ≈ sqrt(2 ln N) - (euler + ln(ln N)) / sqrt(2 ln N)
    # multiplied by the std of the sample-SR estimator (assumed ≈ 1 in
    # standardized units; Bailey-Lopez de Prado use this approximation).
    euler = 0.5772156649015329
    N = max(2, int(n_trials))
    sqrt_2lnN = np.sqrt(2.0 * np.log(N))
    expected_max_sr_per_period = sqrt_2lnN - (euler + np.log(np.log(N))) / sqrt_2lnN

    # Annualize the expected max for reporting
    expected_max_sr_annual = expected_max_sr_per_period * np.sqrt(TRADING_DAYS) \
        if n_obs > 1 else expected_max_sr_per_period
    # Actually, the expected max under H0 is in units of per-period SR with
    # unit variance. We report it re-annualized as a reference benchmark.
    # A more precise formulation uses std(SR_hat); we use the standard
    # simplification that the test statistic is already standardized.

    # DSR = PSR against benchmark = expected_max_sharpe (annualized)
    prob = probabilistic_sharpe_ratio(
        sharpe_ratio=sharpe_ratio,
        n_obs=n_obs,
        returns=arr,
        benchmark_sharpe=expected_max_sr_annual if benchmark_sharpe == 0.0
                         else max(benchmark_sharpe, expected_max_sr_annual),
    )

    return {
        "sharpe": float(sharpe_ratio),
        "deflated_sharpe": float(prob),  # probability that SR > expected max null
        "expected_max_sharpe": float(expected_max_sr_annual),
        "probability": float(prob),
    }


def compute_full_tearsheet(
    result,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    n_trials: int = 1,
) -> pd.DataFrame:
    """
    Generate a performance tearsheet for a long-only personal portfolio.

    Emphasizes metrics that matter for personal investing:
    total return, alpha vs SPY, CAGR, and practical risk measures.
    """
    r  = result.daily_returns
    eq = result.equity_curve

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(r.index).fillna(0)
    else:
        bench = pd.Series(0.0, index=r.index)

    ann_ret = annualized_return(r)
    bench_ann_ret = annualized_return(bench)
    # Ensure scalar (yfinance may return Series with multi-level columns)
    if hasattr(bench_ann_ret, 'iloc'):
        bench_ann_ret = float(bench_ann_ret.iloc[0]) if len(bench_ann_ret) > 0 else 0.0
    if hasattr(ann_ret, 'iloc'):
        ann_ret = float(ann_ret.iloc[0]) if len(ann_ret) > 0 else 0.0
    alpha = ann_ret - bench_ann_ret

    metrics = {
        # Return metrics
        "CAGR":                      f"{ann_ret*100:.2f}%",
        "Total Return":              f"{(eq.iloc[-1]/eq.iloc[0]-1)*100:.2f}%",
        "SPY CAGR":                  f"{bench_ann_ret*100:.2f}%",
        "Alpha vs SPY":              f"{alpha*100:+.2f}%",
        "Annualized Volatility":     f"{annualized_volatility(r)*100:.2f}%",

        # Risk-adjusted metrics
        "Sharpe Ratio":              f"{sharpe_ratio(r, risk_free_rate):.3f}",
        "Sortino Ratio":             f"{sortino_ratio(r, risk_free_rate):.3f}",
        "Calmar Ratio":              f"{calmar_ratio(r, eq):.3f}",
        "Information Ratio":         f"{information_ratio(r, bench):.3f}",

        # Drawdown
        "Max Drawdown":              f"{max_drawdown(eq)*100:.2f}%",
        "Avg Drawdown Duration":     f"{avg_drawdown_duration(eq):.0f} days",

        # Win/loss
        "Daily Win Rate":            f"{win_rate(r)*100:.1f}%",
        "Monthly Win Rate":          f"{_monthly_win_rate(r)*100:.1f}%",
        "Profit Factor":             f"{profit_factor(r):.2f}",
        "Tail Ratio":                f"{tail_ratio(r):.2f}",

        # Costs
        "Total Transaction Costs":   f"${result.transaction_costs.sum():,.0f}",

        # Market exposure
        "Beta to Market":            f"{beta_to_market(r, bench):.3f}",
    }

    # Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014) — corrects for
    # multiple-testing bias and non-normality. Passed n_trials from caller
    # (typically args.optuna_trials if hyperparameter search was run).
    try:
        sr_val = sharpe_ratio(r, risk_free_rate)
        dsr_info = deflated_sharpe_ratio(
            sharpe_ratio=sr_val,
            n_trials=max(1, int(n_trials)),
            returns=r.values,
            benchmark_sharpe=0.0,
        )
        psr_val = probabilistic_sharpe_ratio(
            sharpe_ratio=sr_val,
            n_obs=len(r),
            returns=r.values,
            benchmark_sharpe=0.0,
        )
        metrics["Probabilistic Sharpe (PSR)"] = f"{psr_val:.3f}"
        metrics["Deflated Sharpe (DSR)"] = f"{dsr_info['deflated_sharpe']:.3f}"
        metrics["Expected Max SR (null)"] = f"{dsr_info['expected_max_sharpe']:.3f}"
        metrics["DSR n_trials"] = f"{int(n_trials)}"
    except Exception:
        pass

    tearsheet = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    tearsheet.index.name = "Metric"
    return tearsheet


def _monthly_win_rate(daily_returns: pd.Series) -> float:
    """Fraction of months with positive returns."""
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return (monthly > 0).mean() if len(monthly) > 0 else 0.0


def monte_carlo_sharpe(
    returns: pd.Series,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval on Sharpe ratio.
    This answers: "Is this Sharpe ratio statistically significant,
    or could it have arisen by chance?"

    Returns p5, p50, p95 of the Sharpe distribution.
    Presented in interviews as: "My backtest Sharpe is 1.4 and the
    bootstrap 5th percentile is 0.9, suggesting robustness."
    """
    rng     = np.random.default_rng(seed)
    sharpes = []
    n       = len(returns)
    vals    = returns.values

    # Block bootstrap (Politis & Romano 1994) preserves autocorrelation.
    # Block length ~ sqrt(n) is standard for daily return data.
    block_size = max(1, int(np.sqrt(n)))

    for _ in range(n_simulations):
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        sample = np.concatenate([vals[s:s + block_size] for s in starts])[:n]
        s      = sharpe_ratio(pd.Series(sample))
        sharpes.append(s)

    sharpes = sorted(sharpes)
    return {
        "observed_sharpe": sharpe_ratio(returns),
        "p5":  np.percentile(sharpes, 5),
        "p50": np.percentile(sharpes, 50),
        "p95": np.percentile(sharpes, 95),
        "prob_sharpe_positive": (np.array(sharpes) > 0).mean(),
    }


# ---------------------------------------------------------------------------
# Out-of-Sample Split Tearsheet
# ---------------------------------------------------------------------------

def oos_split_tearsheet(
    result,
    benchmark_returns: Optional[pd.Series] = None,
    oos_start: str = "2024-01-01",
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Split backtest results into in-sample (IS) and out-of-sample (OOS)
    periods, computing a tearsheet for each.

    This is the most compelling evidence against overfitting: if OOS
    performance is comparable to IS, the signal is likely real.
    """
    r = result.daily_returns
    eq = result.equity_curve
    bench = benchmark_returns.reindex(r.index).fillna(0) if benchmark_returns is not None \
            else pd.Series(0.0, index=r.index)

    oos_date = pd.Timestamp(oos_start)

    # Rebase OOS equity to start at 1.0 so max drawdown is computed from
    # the OOS start, not from an IS peak (which would overstate OOS drawdown).
    eq_is = eq[eq.index < oos_date]
    eq_oos = eq[eq.index >= oos_date]
    if len(eq_oos) > 0:
        eq_oos = eq_oos / eq_oos.iloc[0]

    splits = {
        "In-Sample": (r[r.index < oos_date], eq_is, bench[bench.index < oos_date]),
        "Out-of-Sample": (r[r.index >= oos_date], eq_oos, bench[bench.index >= oos_date]),
    }

    rows = {}
    for label, (ret, equity, b) in splits.items():
        if len(ret) < 20:
            continue
        ann_ret = annualized_return(ret)
        ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
        daily_rf = risk_free_rate / TRADING_DAYS
        excess = ret - daily_rf
        sr = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else 0
        dd = max_drawdown(equity) if len(equity) > 1 else 0

        rows[label] = {
            "Days": len(ret),
            "Ann. Return": f"{ann_ret*100:.2f}%",
            "Ann. Vol": f"{ann_vol*100:.2f}%",
            "Sharpe": f"{sr:.3f}",
            "Sortino": f"{sortino_ratio(ret, risk_free_rate):.3f}",
            "Max DD": f"{dd*100:.2f}%",
            "Win Rate": f"{win_rate(ret)*100:.1f}%",
            "Beta": f"{beta_to_market(ret, b):.3f}",
        }

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Fama-French 5-Factor Regression
# ---------------------------------------------------------------------------

def fama_french_regression(
    strategy_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Regress strategy returns against the Fama-French 5 factors
    (Mkt-RF, SMB, HML, RMW, CMA) + Momentum (UMD).

    Downloads factor data from Ken French's data library.
    Returns: alpha (annualized), factor loadings, t-stats, R².

    This answers: "Is your alpha explained by known risk premia?"
    A significant positive alpha after controlling for FF5+Mom is
    strong evidence of genuine stock-selection skill.
    """
    try:
        ff_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
        mom_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                   "ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")

        ff = pd.read_csv(ff_url, skiprows=3, index_col=0,
                         compression="zip", on_bad_lines="skip")
        # Find where data ends (blank row or text)
        valid_mask = ff.index.astype(str).str.match(r"^\d{8}$")
        ff = ff[valid_mask].apply(pd.to_numeric, errors="coerce") / 100
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")

        mom = pd.read_csv(mom_url, skiprows=13, index_col=0,
                          compression="zip", on_bad_lines="skip")
        valid_mask_m = mom.index.astype(str).str.match(r"^\d{8}$")
        mom = mom[valid_mask_m].apply(pd.to_numeric, errors="coerce") / 100
        mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
        mom.columns = [c.strip() for c in mom.columns]
        mom_col = mom.columns[0]  # Usually 'Mom   ' or similar

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

    # Align dates
    common = strategy_returns.index.intersection(ff.index).intersection(mom.index)
    if len(common) < 60:
        return pd.DataFrame({"error": ["Insufficient overlapping dates"]})

    ff_aligned = ff.loc[common]
    ff_aligned.columns = [c.strip() for c in ff_aligned.columns]

    # Use EXCESS returns (strategy - rf) on left side of regression.
    # Previous bug: used raw returns, overstating alpha by ~rf/year.
    rf_daily = ff_aligned["RF"].values if "RF" in ff_aligned.columns else np.zeros(len(common))
    y = strategy_returns.loc[common].values - rf_daily

    # Build factor matrix: Mkt-RF, SMB, HML, RMW, CMA, Mom
    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X_factors = ff_aligned[factor_names].copy()
    X_factors["Mom"] = mom.loc[common, mom_col].values

    # Drop any rows with NaN
    mask = X_factors.notna().all(axis=1) & np.isfinite(y)
    X_factors = X_factors[mask]
    y = y[mask]

    # OLS regression with intercept
    X = np.column_stack([np.ones(len(y)), X_factors.values])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return pd.DataFrame({"error": ["Regression failed"]})

    y_hat = X @ beta
    residuals = y - y_hat
    n, k = X.shape
    r_squared = 1 - (residuals ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # Newey-West HAC standard errors (robust to heteroskedasticity &
    # autocorrelation). Lag length = floor(n^(1/3)) per Andrews (1991).
    max_lag = int(np.floor(n ** (1 / 3)))
    XtX_inv = np.linalg.inv(X.T @ X)

    # Meat of the sandwich: S = sum of Gamma_j weighted by Bartlett kernel
    e = residuals.reshape(-1, 1)
    Xe = X * e  # n x k, each row = x_i * e_i
    S = Xe.T @ Xe  # Gamma_0
    for j in range(1, max_lag + 1):
        w = 1 - j / (max_lag + 1)  # Bartlett kernel
        Gamma_j = Xe[j:].T @ Xe[:-j]
        S += w * (Gamma_j + Gamma_j.T)

    # Sandwich: V = (X'X)^{-1} S (X'X)^{-1}
    V_nw = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V_nw))
    t_stats = beta / se

    labels = ["Alpha"] + list(X_factors.columns)
    results = pd.DataFrame({
        "Coefficient": beta,
        "Std Error": se,
        "t-stat": t_stats,
        "p-value": [2 * (1 - stats.t.cdf(abs(t), n - k)) for t in t_stats],
    }, index=labels)

    # Annualize alpha
    results.loc["Alpha", "Ann. Alpha"] = beta[0] * 252 * 100  # in percent
    results.loc["Alpha", "R²"] = r_squared

    return results


# ---------------------------------------------------------------------------
# Rolling Beta
# ---------------------------------------------------------------------------

def rolling_beta(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Rolling market beta (6-month window)."""
    aligned = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    aligned.columns = ["strat", "mkt"]
    cov = aligned["strat"].rolling(window, min_periods=window // 2).cov(aligned["mkt"])
    var = aligned["mkt"].rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0, np.nan)).dropna()


# ---------------------------------------------------------------------------
# Known Strategy Correlation
# ---------------------------------------------------------------------------

def strategy_correlation_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute correlation of strategy returns against known factor strategies.
    Downloads Fama-French factors and momentum.

    Low correlation to all factors = genuinely novel alpha source.
    """
    try:
        ff_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
        mom_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                   "ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")

        ff = pd.read_csv(ff_url, skiprows=3, index_col=0,
                         compression="zip", on_bad_lines="skip")
        valid_mask = ff.index.astype(str).str.match(r"^\d{8}$")
        ff = ff[valid_mask].apply(pd.to_numeric, errors="coerce") / 100
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")

        mom = pd.read_csv(mom_url, skiprows=13, index_col=0,
                          compression="zip", on_bad_lines="skip")
        valid_mask_m = mom.index.astype(str).str.match(r"^\d{8}$")
        mom = mom[valid_mask_m].apply(pd.to_numeric, errors="coerce") / 100
        mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
        mom.columns = [c.strip() for c in mom.columns]
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

    common = strategy_returns.index.intersection(ff.index).intersection(mom.index)
    if len(common) < 60:
        return pd.DataFrame({"error": ["Insufficient data"]})

    ff_aligned = ff.loc[common]
    ff_aligned.columns = [c.strip() for c in ff_aligned.columns]

    factors = pd.DataFrame({
        "Strategy": strategy_returns.loc[common],
        "Market (Mkt-RF)": ff_aligned["Mkt-RF"],
        "Size (SMB)": ff_aligned["SMB"],
        "Value (HML)": ff_aligned["HML"],
        "Profitability (RMW)": ff_aligned["RMW"],
        "Investment (CMA)": ff_aligned["CMA"],
        "Momentum (UMD)": mom.loc[common].iloc[:, 0],
    })

    if benchmark_returns is not None:
        factors["SPY"] = benchmark_returns.reindex(common).fillna(0)

    corr = factors.corr()["Strategy"].drop("Strategy")
    result = pd.DataFrame({
        "Correlation": corr,
        "Abs Correlation": corr.abs(),
    }).sort_values("Abs Correlation", ascending=False)

    return result


# ---------------------------------------------------------------------------
# Monthly Returns Table (Calendar Heatmap)
# ---------------------------------------------------------------------------

def monthly_returns_table(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Build a year × month table of returns (percentages).

    Perfect for a calendar heatmap in the dashboard. Each cell is the
    total return for that month, plus a YTD column.
    """
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    table = pd.DataFrame(index=sorted(monthly.index.year.unique()))
    table.index.name = "Year"

    for month_num in range(1, 13):
        month_name = pd.Timestamp(2000, month_num, 1).strftime("%b")
        vals = monthly[monthly.index.month == month_num]
        table[month_name] = vals.groupby(vals.index.year).first().reindex(table.index)

    # YTD column
    yearly = daily_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    table["YTD"] = yearly.groupby(yearly.index.year).first().reindex(table.index)

    return (table * 100).round(2)


# ---------------------------------------------------------------------------
# Wealth Growth Comparison
# ---------------------------------------------------------------------------

def wealth_growth(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial: float = 100_000,
) -> pd.DataFrame:
    """
    Side-by-side cumulative wealth: strategy vs benchmark.
    Starting from a given initial investment.
    """
    strat_eq = (1 + strategy_returns).cumprod() * initial
    bench_eq = (1 + benchmark_returns.reindex(strategy_returns.index).fillna(0)).cumprod() * initial

    return pd.DataFrame({
        "Strategy": strat_eq,
        "SPY Buy & Hold": bench_eq,
        "Excess": strat_eq - bench_eq,
    })


# ---------------------------------------------------------------------------
# Annual Returns Summary
# ---------------------------------------------------------------------------

def annual_returns(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Year-by-year returns for strategy and benchmark."""
    yearly_s = strategy_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    result = pd.DataFrame({"Strategy": yearly_s * 100})
    result.index = result.index.year
    result.index.name = "Year"

    if benchmark_returns is not None:
        yearly_b = benchmark_returns.reindex(strategy_returns.index).fillna(0) \
                   .resample("YE").apply(lambda x: (1 + x).prod() - 1)
        result["SPY"] = yearly_b.values * 100
        result["Alpha"] = result["Strategy"] - result["SPY"]

    return result.round(2)


# ---------------------------------------------------------------------------
# After-Tax Return Estimate
# ---------------------------------------------------------------------------

def after_tax_returns(
    daily_returns: pd.Series,
    st_tax_rate: float = 0.37,
    lt_tax_rate: float = 0.20,
    avg_holding_months: float = 1.0,
) -> dict:
    """
    Estimate after-tax performance for a personal portfolio.

    Since monthly rebalancing produces mostly short-term gains (held < 12 months),
    the majority of gains are taxed at the higher short-term rate. Some positions
    held across multiple rebalances may qualify for long-term treatment.

    Parameters
    ----------
    st_tax_rate         : short-term capital gains rate (default 37% top bracket)
    lt_tax_rate         : long-term capital gains rate (default 20%)
    avg_holding_months  : average holding period in months.
                          1 = all short-term, 12+ = all long-term.

    Returns
    -------
    dict with pre-tax and after-tax metrics
    """
    # Estimate fraction of gains that are long-term
    lt_fraction = min(1.0, max(0.0, (avg_holding_months - 1) / 11.0))
    effective_tax_rate = lt_fraction * lt_tax_rate + (1 - lt_fraction) * st_tax_rate

    # Separate gains and losses
    gains = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns <= 0]

    # Tax only applies to gains; losses reduce taxable income
    after_tax_daily = daily_returns.copy()
    after_tax_daily[daily_returns > 0] = gains * (1 - effective_tax_rate)
    # Losses are kept as-is (they offset gains)

    pre_tax_cagr = annualized_return(daily_returns)
    after_tax_cagr = annualized_return(after_tax_daily)
    tax_drag = pre_tax_cagr - after_tax_cagr

    return {
        "Pre-Tax CAGR": f"{pre_tax_cagr*100:.2f}%",
        "Effective Tax Rate": f"{effective_tax_rate*100:.1f}%",
        "After-Tax CAGR": f"{after_tax_cagr*100:.2f}%",
        "Tax Drag": f"{tax_drag*100:.2f}%",
        "LT Fraction": f"{lt_fraction*100:.0f}%",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_prices, get_close, get_returns, get_volume
    from features    import build_composite_signal, realized_volatility
    from portfolio   import compute_target_weights
    from backtest    import run_backtest

    prices   = load_prices(start="2020-01-01", end="2024-01-01")
    close    = get_close(prices)
    returns  = get_returns(prices)
    volume   = get_volume(prices)

    composite, _ = build_composite_signal(close, returns, volume)
    rvol         = realized_volatility(returns, window=21)
    weights      = compute_target_weights(composite, rvol)
    result       = run_backtest(weights, prices)

    tearsheet = compute_full_tearsheet(result)
    print("\n=== Performance Tearsheet ===")
    print(tearsheet.to_string())

    print("\n=== Monte Carlo Sharpe ===")
    mc = monte_carlo_sharpe(result.daily_returns)
    for k, v in mc.items():
        print(f"  {k}: {v:.3f}")
