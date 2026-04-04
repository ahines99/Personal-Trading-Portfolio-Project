"""
features.py
-----------
Computes alpha signals (features) from raw price/volume data.

THE GOLDEN RULE enforced throughout this file:
    Feature at row T may ONLY use data from rows 0..T.
    Never use df.rolling().apply() with a function that peeks forward.
    Never normalize using full-sample statistics.

Signal families:
    1. Short-horizon momentum (reduced weight — exhibits reversal in recent markets)
    2. Long-horizon momentum (3m/6m — tends toward continuation)
    3. Mean reversion (RSI, z-score — primary signal given negative short IC)
    4. Volatility regime
    5. Volume-price trend
    6. Cross-sectional rank
"""

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Price Momentum
# ---------------------------------------------------------------------------

def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Return over the past `window` trading days, skipping the most recent day.
    The skip (shift(1)) avoids the well-known short-term reversal at lag-1.

    Signal at T = close[T-1] / close[T-1-window] - 1
    """
    return close.shift(1).pct_change(window)


def short_horizon_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    Shorter-horizon momentum (5d, 10d, 21d).

    NOTE: Short-horizon momentum shows negative IC in recent US equity data,
    indicating reversal rather than continuation at these timeframes.
    Weight is reduced in the composite; use long_horizon_momentum as the
    primary momentum signal.
    """
    m5 = momentum(close, 5)
    m10 = momentum(close, 10)
    m21 = momentum(close, 21)

    composite = (m5 + m10 + m21) / 3
    return composite.rename(columns={c: c for c in composite.columns})


def long_horizon_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    Longer-horizon momentum (63d / 126d).

    3–6 month momentum historically exhibits return continuation rather
    than the reversal observed at 5–21 day horizons.
    Skips the most recent 21 days to avoid contamination from short-term
    reversal (standard Jegadeesh-Titman skip).

    Signal at T = close[T-21] / close[T-21-window] - 1
    """
    m63 = close.shift(21).pct_change(63)
    m126 = close.shift(21).pct_change(126)
    composite = (m63 + m126) / 2
    return composite.rename(columns={c: c for c in composite.columns})


# ---------------------------------------------------------------------------
# 2. Mean Reversion
# ---------------------------------------------------------------------------

def zscore_reversion(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Z-score of price relative to its rolling mean.
    Negative z-score = oversold = mean-reversion BUY signal.
    We negate so that higher score = more attractive (oversold).

    z = -(close - rolling_mean) / rolling_std
    """
    roll_mean = close.rolling(window, min_periods=window // 2).mean()
    roll_std = close.rolling(window, min_periods=window // 2).std()
    z = (close - roll_mean) / roll_std.replace(0, np.nan)
    return -z


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index. Values < 30 = oversold, > 70 = overbought.
    We return (50 - RSI) so that oversold stocks get a higher score,
    consistent with a mean-reversion signal.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_ = 100 - (100 / (1 + rs))
    return 50 - rsi_


def composite_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    """
    Combines z-score reversion and RSI into a single mean-reversion signal.

    Both signals agree: high score = oversold = likely to revert upward.
    Average after cross-sectional ranking to equalize scale.
    """
    zrev = cross_sectional_rank(zscore_reversion(close, window=20))
    rsi_ = cross_sectional_rank(rsi(close, window=14))
    return (zrev + rsi_) / 2


# ---------------------------------------------------------------------------
# 3. Volatility Signals
# ---------------------------------------------------------------------------

def realized_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Annualized realized volatility over a rolling window.
    Used both as a signal and for position sizing.
    """
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def volatility_regime(returns: pd.DataFrame, short: int = 10, long: int = 60) -> pd.DataFrame:
    """
    Ratio of short-term vol to long-term vol.
    > 1 = vol expanding
    < 1 = vol contracting

    We negate so calm regime → higher score.
    """
    short_vol = realized_volatility(returns, short)
    long_vol = realized_volatility(returns, long)
    ratio = short_vol / long_vol.replace(0, np.nan)
    return -ratio


# ---------------------------------------------------------------------------
# 4. 52-Week High Proximity  (George & Hwang 2004)
# ---------------------------------------------------------------------------

def price_to_52w_high(close: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of yesterday's close to the 52-week rolling high.
    Range: (0, 1]. Higher = closer to the annual high = stronger trend anchor.

    George & Hwang (2004) show this predicts returns better than raw momentum
    because it anchors investor reference points.

    Signal at T = close[T-1] / max(close[T-252 : T-1])
    """
    high_252 = close.shift(1).rolling(252, min_periods=126).max()
    return close.shift(1) / high_252.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 5. MAX Effect — Lottery Stock Avoidance  (Bali, Cakici & Whitelaw 2011)
# ---------------------------------------------------------------------------

def max_return(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Negated maximum daily return over the past `window` days.

    Stocks with very large recent single-day gains (lottery-like payoff) tend
    to be overpriced by retail investors and subsequently underperform.
    We negate so that HIGH score = LOW recent max return = avoids lottery stocks.

    Signal at T = -max(daily_return[T-window : T-1])
    """
    daily_ret = close.pct_change().shift(1)
    return -daily_ret.rolling(window, min_periods=window // 2).max()


# ---------------------------------------------------------------------------
# 6. Amihud Illiquidity  (Amihud 2002)
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Mean ratio of |daily return| to dollar volume over a rolling window.

    Less-liquid stocks require a larger price move per dollar traded — they
    earn an illiquidity premium. Higher Amihud ratio = less liquid = positive
    expected-return signal.

    Amihud illiquidity = mean(|r_t| / (P_t * V_t))
    """
    dollar_vol = (close * volume).replace(0, np.nan)
    daily_illiq = returns.abs() / dollar_vol
    return daily_illiq.rolling(window, min_periods=window // 2).mean()


# ---------------------------------------------------------------------------
# 7. Idiosyncratic Volatility  (Ang, Hodrick, Xing & Zhang 2006)
# ---------------------------------------------------------------------------

def idiosyncratic_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Volatility of the market-residual return (CAPM alpha residual).

    Uses an equal-weighted cross-sectional average as the market proxy (avoids
    needing external benchmark data). Residual = stock return - market return.

    The low-idiovol anomaly: stocks with LOW idiosyncratic volatility tend to
    OUTPERFORM — opposite of what naive risk-return intuition suggests.

    We negate so that HIGH score = LOW idiovol = positive expected return.
    """
    mkt = returns.mean(axis=1)               # equal-weighted market proxy
    residuals = returns.sub(mkt, axis=0)     # idiosyncratic component
    ivol = residuals.rolling(window, min_periods=window // 2).std()
    return -ivol                              # negate: low ivol → buy signal


# ---------------------------------------------------------------------------
# 8. MACD Signal  (trend-following complement to mean reversion)
# ---------------------------------------------------------------------------

def macd_signal(
    close: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """
    MACD histogram: (MACD line) - (signal line).

    MACD line   = EMA(fast) - EMA(slow)
    Signal line = EMA(signal_span) of MACD line
    Histogram   = MACD line - signal line

    Positive histogram → bullish momentum (fast EMA accelerating above slow).
    Negative histogram → bearish momentum.

    Complements mean reversion: on longer timeframes the two signals diverge
    and the ML model can learn when trend vs. reversion is dominant.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig_line = macd.ewm(span=signal_span, adjust=False).mean()
    return macd - sig_line


# ---------------------------------------------------------------------------
# 9. Volume Anomaly
# ---------------------------------------------------------------------------

def volume_spike(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume today vs rolling average volume.
    spike = volume / rolling_avg_volume
    """
    avg_vol = volume.rolling(window, min_periods=window // 2).mean()
    return volume / avg_vol.replace(0, np.nan)


def volume_price_trend(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume-weighted price trend: accumulates (price change * volume) over window.
    Positive = buying pressure. Negative = selling pressure.
    Normalized to rolling z-score.
    """
    daily_return = close.pct_change()
    vpt_raw = (daily_return * volume).rolling(window, min_periods=window // 2).sum()

    roll_mean = vpt_raw.rolling(window * 3, min_periods=window).mean()
    roll_std = vpt_raw.rolling(window * 3, min_periods=window).std()
    return (vpt_raw - roll_mean) / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 10. Market Beta  (Frazzini & Pedersen 2014 — Betting Against Beta)
# ---------------------------------------------------------------------------

def market_beta(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Rolling OLS beta of each stock against the equal-weighted market.

    Frazzini & Pedersen (2014) show that low-beta assets outperform on a
    risk-adjusted basis (BAB factor). We negate so low-beta = high score.

    beta = Cov(r_i, r_m) / Var(r_m)
    Cov estimated as E[r_i * r_m] - E[r_i] * E[r_m]
    """
    mkt = returns.mean(axis=1)
    r_mean = returns.rolling(window, min_periods=window // 2).mean()
    m_mean = mkt.rolling(window, min_periods=window // 2).mean()
    cov_im = (
        returns.mul(mkt, axis=0)
        .rolling(window, min_periods=window // 2).mean()
        .sub(r_mean.mul(m_mean, axis=0))
    )
    var_m = mkt.rolling(window, min_periods=window // 2).var()
    beta = cov_im.div(var_m.replace(0, np.nan), axis=0)
    return -beta   # negate: low beta → buy signal


# ---------------------------------------------------------------------------
# 11. On-Balance Volume  (Granville 1963)
# ---------------------------------------------------------------------------

def obv_signal(returns: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Cumulative OBV normalized to a rolling z-score.

    OBV accumulates signed volume (positive on up days, negative on down days).
    Rising OBV with rising price = confirmed trend; divergence = warning.
    """
    direction = np.sign(returns)
    obv_cum = (direction * volume).cumsum()
    roll_mean = obv_cum.rolling(window * 3, min_periods=window).mean()
    roll_std  = obv_cum.rolling(window * 3, min_periods=window).std()
    return (obv_cum - roll_mean) / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 12. Bollinger Band %B  (Bollinger 1992)
# ---------------------------------------------------------------------------

def bollinger_pct_b(close: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Position of price within its Bollinger Bands.

    %B = (price - lower_band) / (upper_band - lower_band)
    0 = at lower band (oversold), 1 = at upper band (overbought).
    We negate so that low %B (oversold) → high score (mean-reversion buy).
    """
    ma    = close.rolling(window, min_periods=window // 2).mean()
    std   = close.rolling(window, min_periods=window // 2).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return -pct_b


# ---------------------------------------------------------------------------
# 13. Residual Momentum  (idiosyncratic price trend)
# ---------------------------------------------------------------------------

def residual_momentum(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Market-neutral momentum: cumulative idiosyncratic return over `window` days.

    Strips out the common market factor so the signal captures purely
    stock-specific trends, not sector or market beta.

    residual_return = stock_return - equal_weighted_market_return
    residual_mom    = sum(residual_return[T-window-1 : T-1])
    """
    mkt = returns.mean(axis=1)
    residuals = returns.sub(mkt, axis=0)
    return residuals.shift(1).rolling(window, min_periods=window // 2).sum()


# ---------------------------------------------------------------------------
# 14. Price Acceleration  (momentum of momentum)
# ---------------------------------------------------------------------------

def price_acceleration(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Change in medium-term momentum over the past `window` days.

    acceleration = mom(T) - mom(T - window)

    Accelerating momentum (positive) indicates a strengthening trend;
    decelerating (negative) may signal an imminent reversal.
    """
    mom_now  = momentum(close, window)
    mom_prev = momentum(close, window).shift(window)
    return mom_now - mom_prev


# ---------------------------------------------------------------------------
# 15. Volume Trend  (sustained volume growth)
# ---------------------------------------------------------------------------

def volume_trend(volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Rolling z-score of the 21-day percentage change in average daily volume.

    Rising volume = increasing market participation in a move.
    Normalised cross-sectionally to remove market-wide volume shifts.
    """
    vol_ma     = volume.rolling(window, min_periods=window // 2).mean()
    vol_growth = vol_ma.pct_change(window)
    roll_mean  = vol_growth.rolling(window * 3, min_periods=window).mean()
    roll_std   = vol_growth.rolling(window * 3, min_periods=window).std()
    return (vol_growth - roll_mean) / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 16. Kaufman Efficiency Ratio  (trend quality)
# ---------------------------------------------------------------------------

def efficiency_ratio(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Ratio of net directional price move to total path length over `window` days.

    efficiency = |close[T] - close[T-window]| / sum(|daily_changes|)

    Range [0, 1]. High ER = price moved efficiently in one direction (strong trend).
    Low ER = noisy, choppy price action.
    """
    net_move  = close.diff(window).abs()
    total_path = close.diff().abs().rolling(window, min_periods=window // 2).sum()
    return net_move / total_path.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 17. Moving Average Distance  (trend deviation)
# ---------------------------------------------------------------------------

def ma_distance(close: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Signed distance of price from its `window`-day moving average, normalised.

    (close - MA) / MA

    Negated for mean-reversion: stocks far above MA → oversold signal.
    Can also be used as a trend filter (positive = above MA = in uptrend).
    """
    ma = close.rolling(window, min_periods=window // 2).mean()
    return -(close - ma) / ma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 18. Volatility of Volatility  (regime stability)
# ---------------------------------------------------------------------------

def vol_of_vol(returns: pd.DataFrame, short: int = 5, long: int = 21) -> pd.DataFrame:
    """
    Rolling standard deviation of short-term realized volatility.

    Stocks with LOW vol-of-vol have stable, predictable volatility — they are
    easier to hedge and tend to outperform erratic peers.
    Negated so stable (low VoV) → high score.
    """
    short_vol = returns.rolling(short, min_periods=short // 2).std()
    vov = short_vol.rolling(long, min_periods=long // 2).std()
    return -vov


# ---------------------------------------------------------------------------
# 19. Tail Risk  (left-tail exposure)
# ---------------------------------------------------------------------------

def tail_risk(returns: pd.DataFrame, window: int = 21, pct: float = 0.05) -> pd.DataFrame:
    """
    Rolling 5th-percentile daily return (left tail).

    Stocks with severe downside tail risk (very negative 5th percentile)
    tend to underperform due to crash aversion and leverage constraints.
    Negated so that low tail risk (less negative) → high score.
    """
    quantile = returns.shift(1).rolling(window, min_periods=window // 2).quantile(pct)
    return -quantile


# ---------------------------------------------------------------------------
# 20. Short-Term Reversal  (Jegadeesh 1990)
# ---------------------------------------------------------------------------

def short_term_reversal(returns: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Negated short-term return over `window` trading days.

    Stocks that decline in the past 1-2 days tend to mean-revert upward
    over the next 1-5 days. This is one of the most robust cross-sectional
    signals in U.S. equities, strongest at the 1-day horizon.

    We negate so that recent losers (negative return) get a positive score.

    Signal at T = -sum(returns[T-window : T-1])
    """
    return -returns.shift(1).rolling(window, min_periods=1).sum()


# ---------------------------------------------------------------------------
# 21. Volume-Confirmed Reversal
# ---------------------------------------------------------------------------

def volume_confirmed_reversal(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    avg_window: int = 20,
) -> pd.DataFrame:
    """
    1-day reversal amplified by the volume spike ratio.

    A stock that drops on HIGH relative volume tends to reverse more strongly
    because the selling is exhausted. A drop on thin volume may continue.

    Signal = -return_yesterday × clip(vol_yesterday / avg_vol, 0.5, 3.0)

    avg_window : lookback for the rolling average volume baseline.
    """
    avg_vol   = volume.rolling(avg_window, min_periods=avg_window // 2).mean()
    vol_spike = (volume.shift(1) / avg_vol.shift(1).replace(0, np.nan)).clip(0.5, 3.0)
    return -returns.shift(1) * vol_spike


# ---------------------------------------------------------------------------
# 22. Sector-Relative Momentum
# ---------------------------------------------------------------------------

def sector_relative_momentum(
    returns: pd.DataFrame,
    sector_map: dict,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling cumulative return of each stock minus its GICS sector peer average.

    Captures 'winner within sector' alpha — orthogonal to both market-level
    momentum (which residual_momentum already measures) and sector tilts.
    Stocks that outperform their sector peer group over 63 days tend to
    continue outperforming those peers.

    Uses shift(1) to avoid lookahead bias.
    Sectors with fewer than 3 stocks fall back to absolute rolling return.
    """
    shifted = returns.shift(1)

    sector_groups: dict = {}
    for ticker in returns.columns:
        sector = sector_map.get(ticker, "Unknown")
        sector_groups.setdefault(sector, []).append(ticker)

    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for sector, tickers in sector_groups.items():
        sub        = shifted[tickers]
        stock_cum  = sub.rolling(window, min_periods=window // 2).sum()
        if len(tickers) < 3:
            result[tickers] = stock_cum
        else:
            sector_avg_cum = sub.mean(axis=1).rolling(window, min_periods=window // 2).sum()
            result[tickers] = stock_cum.sub(sector_avg_cum, axis=0)

    return result


# ---------------------------------------------------------------------------
# Hurst Exponent (trending vs mean-reverting)
# ---------------------------------------------------------------------------

def hurst_exponent(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Simplified Hurst exponent via rescaled range analysis.

    H > 0.5 → trending (momentum works)
    H < 0.5 → mean-reverting (reversal works)
    H = 0.5 → random walk

    Uses a fast approximation: variance ratio of returns at lag 1 vs lag N.
    """
    log_ret = np.log(close / close.shift(1))

    var_1 = log_ret.rolling(window, min_periods=window // 2).var()
    # Variance of aggregated returns at lag N (N=window//4)
    lag_n = max(2, window // 4)
    agg_ret = log_ret.rolling(lag_n).sum()
    var_n = agg_ret.rolling(window, min_periods=window // 2).var()

    # Variance ratio: if trending, var_n > lag_n * var_1
    expected_var_n = var_1 * lag_n
    ratio = var_n / expected_var_n.replace(0, np.nan)

    # Map to approximate Hurst: ratio > 1 → H > 0.5 (trending)
    hurst_approx = 0.5 * np.log2(ratio.clip(0.01, 10))  # maps ratio=1 to H~0.5
    return hurst_approx.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Cross-Asset Sector Signals
# ---------------------------------------------------------------------------

def cross_asset_sector_signal(
    close: pd.DataFrame,
    sector_map: dict,
    commodity_data: Optional[dict] = None,
    window: int = 21,
) -> pd.DataFrame:
    """
    Sector-level macro tailwind signal.

    Maps each stock to its sector's relevant commodity/macro momentum.
    When a sector's macro driver is trending up, stocks in that sector
    get a positive signal.

    Uses yfinance commodity proxies: CL=F (oil), GC=F (gold), HG=F (copper).
    Falls back to sector-level stock momentum if commodity data unavailable.
    """
    result = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # Compute sector momentum as fallback
    sector_groups: dict = {}
    for ticker in close.columns:
        sector = sector_map.get(ticker, "Unknown")
        sector_groups.setdefault(sector, []).append(ticker)

    for sector, tickers in sector_groups.items():
        if len(tickers) < 3:
            continue
        # Sector average momentum
        sector_mom = close[tickers].pct_change(window).mean(axis=1)
        for t in tickers:
            result[t] = sector_mom

    return result


# ---------------------------------------------------------------------------
# Net Share Issuance (buyback signal)
# ---------------------------------------------------------------------------

def net_share_issuance_signal(
    close: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Proxy for net share issuance using split-adjusted price behavior.

    When a company buys back shares, the remaining shares represent more
    of the company, creating a subtle upward drift in per-share metrics.

    True net issuance requires shares outstanding data (from EDGAR).
    This proxy uses the ratio of actual returns to price returns as
    an approximation. Negated so buybacks (negative issuance) get high scores.

    Note: This is a weak proxy. For production, use quarterly shares
    outstanding from SEC filings.
    """
    # Without shares outstanding data, return zeros (placeholder)
    # Will be populated when EDGAR shares data is integrated
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)


# ---------------------------------------------------------------------------
# 23. IC-Weighted Composite Builder
# ---------------------------------------------------------------------------

def build_ic_weighted_composite(
    ranked_signals: dict,
    returns: pd.DataFrame,
    ic_window: int = 63,
) -> pd.DataFrame:
    """
    Build a composite signal weighted by each sub-signal's rolling IC.

    IC(signal_i, T) = Spearman rank correlation of signal_i[T] with return[T+1]
    rolling_IC_i    = rolling mean of IC over the past ic_window days
    weight_i        = |rolling_IC_i| / Σ|rolling_IC_j|

    Rolling IC is shifted by 1 day before use to eliminate any lookahead.
    Falls back to equal-weighting when insufficient IC history exists.

    WHY: Fixed composite weights assume each signal contributes equally at
    all times. In reality, some factors work better in certain regimes.
    IC-weighting dynamically upweights what is currently predictive and
    down-weights what is not — without ever looking at future data.
    """
    fwd_ret = returns.shift(-1)     # 1-day forward return (IC target)

    # Compute rolling IC for each signal (vectorised via corrwith)
    rolling_ics: dict = {}
    for name, sig in ranked_signals.items():
        sig_ranked = sig.rank(axis=1, pct=True)
        fwd_ranked = fwd_ret.rank(axis=1, pct=True)
        daily_ic   = sig_ranked.corrwith(fwd_ranked, axis=1)
        rolling_ics[name] = (
            daily_ic.rolling(ic_window, min_periods=ic_window // 2).mean()
            .shift(1)           # no lookahead: use yesterday's IC to weight today
        )

    # |IC| weights, normalised each day; equal-weight fallback
    ic_df      = pd.DataFrame(rolling_ics).abs()
    row_sum    = ic_df.sum(axis=1).replace(0, np.nan)
    n_signals  = len(ranked_signals)
    ic_norm    = ic_df.div(row_sum, axis=0).fillna(1.0 / n_signals)

    first = next(iter(ranked_signals.values()))
    composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, sig in ranked_signals.items():
        composite += sig.multiply(ic_norm[name], axis=0)

    return composite


# ---------------------------------------------------------------------------
# 5. Cross-Sectional Ranking
# ---------------------------------------------------------------------------

def cross_sectional_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Rank stocks cross-sectionally on each day.
    Output is uniform in [0, 1] where 1 = most attractive.
    """
    return signal.rank(axis=1, pct=True, na_option="keep")


def demean_cross_sectional(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract cross-sectional mean on each day.
    Useful for constructing dollar-neutral signals.
    """
    return signal.sub(signal.mean(axis=1), axis=0)


# ---------------------------------------------------------------------------
# 6. Composite Signal Builder
# ---------------------------------------------------------------------------

def build_composite_signal(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    weights: dict = None,
    sector_map: dict = None,
    use_ic_weights: bool = False,
):
    """
    Combines individual signals into a single composite alpha score.

    Signal roster (24 signals):
      Core anomalies:  idiovol, amihud, long_momentum, market_beta, residual_momentum
      Reversal:        reversal_1d, reversal_2d, vol_reversal  ← NEW
      Sector alpha:    sector_rel_mom (if sector_map provided)  ← NEW
      Trend:           macd, price_52w_high, efficiency_ratio, price_acceleration
      Mean reversion:  mean_reversion, bollinger_pct_b, ma_distance
      Risk / tail:     max_effect, tail_risk, vol_of_vol
      Volume:          obv, volume_trend, vpt
      Regime:          vol_regime, short_momentum

    Parameters
    ----------
    close           : (date x ticker) adjusted close prices
    returns         : (date x ticker) daily returns
    volume          : (date x ticker) daily volume
    weights         : dict of {signal_name: weight}, must sum to 1.0
    sector_map      : dict of ticker → GICS sector (enables sector_rel_mom)
    use_ic_weights  : if True, replace fixed weights with rolling-IC weights

    Returns
    -------
    composite_ranked : (date x ticker) DataFrame, cross-sectionally ranked [0, 1]
    ranked_signals   : dict of individual ranked signal DataFrames
    """
    has_sector = sector_map is not None

    if weights is None:
        # Base weights for signals present regardless of sector_map.
        # New signals (reversal family) added; existing weights scaled down ~12%
        # to make room while keeping the sum = 1.0.
        weights = {
            # --- core anomalies ---
            "idiovol":            0.08,
            "amihud":             0.06,
            "long_momentum":      0.08,
            "market_beta":        0.06,
            "residual_momentum":  0.05,
            # --- reversal (NEW) ---
            "reversal_1d":        0.05,  # 1-day reversal — strongest short-term signal
            "reversal_2d":        0.03,  # 2-day reversal complement
            "vol_reversal":       0.03,  # volume-confirmed reversal
            # --- sector relative momentum (NEW, conditional) ---
            "sector_rel_mom":     0.04 if has_sector else 0.0,
            # --- trend ---
            "macd":               0.05,
            "price_52w_high":     0.06,
            "efficiency_ratio":   0.03,
            "price_acceleration": 0.04,
            # --- mean reversion ---
            "mean_reversion":     0.06,
            "bollinger_pct_b":    0.04,
            "ma_distance":        0.03,
            # --- risk / tail ---
            "max_effect":         0.04,
            "tail_risk":          0.04,
            "vol_of_vol":         0.03,
            # --- volume ---
            "obv":                0.03,
            "volume_trend":       0.02,
            "vpt":                0.01,
            # --- regime ---
            "vol_regime":         0.02,
            "short_momentum":     0.01,
        }
        # Renormalise to exactly 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    raw_signals = {
        # --- core anomalies ---
        "idiovol":            idiosyncratic_volatility(returns, window=21),
        "amihud":             amihud_illiquidity(returns, volume, close, window=21),
        "long_momentum":      long_horizon_momentum(close),
        "market_beta":        market_beta(returns, window=63),
        "residual_momentum":  residual_momentum(returns, window=21),
        # --- reversal signals (NEW) ---
        "reversal_1d":        short_term_reversal(returns, window=1),
        "reversal_2d":        short_term_reversal(returns, window=2),
        "vol_reversal":       volume_confirmed_reversal(returns, volume),
        # --- trend ---
        "macd":               macd_signal(close),
        "price_52w_high":     price_to_52w_high(close),
        "efficiency_ratio":   efficiency_ratio(close, window=21),
        "price_acceleration": price_acceleration(close, window=21),
        # --- mean reversion ---
        "mean_reversion":     composite_mean_reversion(close),
        "bollinger_pct_b":    bollinger_pct_b(close),
        "ma_distance":        ma_distance(close, window=50),
        # --- risk / tail ---
        "max_effect":         max_return(close, window=21),
        "tail_risk":          tail_risk(returns, window=21),
        "vol_of_vol":         vol_of_vol(returns),
        # --- volume ---
        "obv":                obv_signal(returns, volume),
        "volume_trend":       volume_trend(volume, window=21),
        "vpt":                volume_price_trend(close, volume, window=20),
        # --- regime ---
        "vol_regime":         volatility_regime(returns, short=10, long=60),
        "short_momentum":     short_horizon_momentum(close),
    }

    # Sector-relative momentum — only if sector_map is provided
    if has_sector:
        raw_signals["sector_rel_mom"] = sector_relative_momentum(returns, sector_map, window=63)

    ranked_signals = {
        name: cross_sectional_rank(sig)
        for name, sig in raw_signals.items()
        if name in weights  # only rank signals that have a weight
    }

    if use_ic_weights:
        composite = build_ic_weighted_composite(ranked_signals, returns)
    else:
        composite = sum(
            ranked_signals[name] * w
            for name, w in weights.items()
            if name in ranked_signals
        )

    composite_ranked = cross_sectional_rank(composite)

    return composite_ranked, ranked_signals


# ---------------------------------------------------------------------------
# Factor decay analysis
# ---------------------------------------------------------------------------

def factor_decay_analysis(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: list = [1, 5, 10, 20, 60],
) -> pd.DataFrame:
    """
    Measures how predictive the signal is at different forward horizons.

    Returns a DataFrame of IC (Information Coefficient = rank correlation)
    at each horizon.
    """
    results = {}

    for h in horizons:
        fwd_return = returns.shift(-h).rolling(h).sum()

        ics = []
        for date in signal.index:
            s = signal.loc[date].dropna()
            r = fwd_return.loc[date].dropna() if date in fwd_return.index else pd.Series(dtype=float)

            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue

            ic = s[common].corr(r[common], method="spearman")
            ics.append(ic)

        ic_series = pd.Series(ics, dtype=float)

        results[f"{h}d_IC"] = {
            "mean_IC": ic_series.mean(),
            "IC_IR": ic_series.mean() / ic_series.std() if ic_series.std() and ic_series.std() > 0 else 0,
            "pct_positive": (ic_series > 0).mean(),
        }

    return pd.DataFrame(results).T


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_prices, get_close, get_returns, get_volume

    prices = load_prices(start="2020-01-01", end="2024-01-01")
    close = get_close(prices)
    returns = get_returns(prices)
    volume = get_volume(prices)

    composite, components = build_composite_signal(close, returns, volume)

    print("Composite signal (last 5 rows, first 5 tickers):")
    print(composite.iloc[-5:, :5])

    print("\nFactor decay analysis (composite):")
    decay = factor_decay_analysis(composite, returns, horizons=[1, 3, 5, 10, 20])
    print(decay)
