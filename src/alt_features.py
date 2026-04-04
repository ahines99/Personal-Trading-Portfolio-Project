"""
alt_features.py
---------------
Alpha signal functions derived from the five alternative data sources.

Sources → signals:

  SEC EDGAR fundamentals
    book_to_market_signal   — high book/market (cheap stocks)
    roe_signal              — high return-on-equity
    gross_margin_signal     — high gross margin
    asset_growth_signal     — negated asset growth (low growth = good)
    leverage_signal         — negated leverage (low debt = good)
    earnings_trend_signal   — rising EPS trend

  FRED macro
    macro_regime_features   — broadcast scalar macro series to all tickers
                              (yield_curve, hy_spread, vix_macro, fed_funds,
                               unemployment, breakeven_inflation)

  CBOE VIX term structure
    vix_regime_features     — term_slope, vix_percentile, vix_change_5d
                              broadcast to all tickers

  FINRA short interest
    short_interest_signal   — short ratio and short % of float cross-sectional rank

  Earnings
    earnings_surprise_signal — post-earnings drift (eps_surprise_pct carried forward)
    days_to_earnings_signal  — distance to next earnings (near = higher implied vol)

All cross-sectional signals are ranked [0, 1] so they can be stacked directly
into the feature matrix.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [0, 1] each row."""
    return df.rank(axis=1, pct=True)


def _broadcast(series: pd.Series, columns: pd.Index) -> pd.DataFrame:
    """Broadcast a date-indexed scalar series to a ticker-wide DataFrame."""
    return pd.DataFrame(
        np.tile(series.values.reshape(-1, 1), (1, len(columns))),
        index=series.index,
        columns=columns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SEC EDGAR Fundamental Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_fundamental_signals(
    edgar_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    edgar_df : MultiIndex-column DataFrame (ticker, factor) from load_edgar_fundamentals()
    tickers  : ticker columns to align to
    date_index : business-day DatetimeIndex to align to

    Returns
    -------
    dict of signal_name → cross-sectionally ranked DataFrame (dates × tickers)
    """
    signals = {}

    if edgar_df is None or edgar_df.empty:
        return signals

    factors = {
        # ── Core (existing) ──────────────────────────────────────────────
        "book_to_market":         True,   # higher = stronger (cheap)
        "roe":                    True,   # higher = stronger
        "gross_margin":           True,   # higher = stronger
        "asset_growth":           False,  # lower = stronger (negate before ranking)
        "leverage":               False,  # lower = stronger
        "eps_trend":              True,   # higher = stronger
        # ── Robust anomalies (new, all survive HXZ replication) ──────────
        # Novy-Marx (2013): GrossProfit / Assets
        "gross_profitability":    True,
        # Sloan (1996): (NetIncome - OCF) / Assets — lower = stronger
        "accruals":               False,
        # Hirshleifer (2004): Net Operating Assets / Assets — lower = stronger
        "net_operating_assets":   False,
        # Pontiff/Woodgate (2008): log change in shares — lower = stronger
        "net_share_issuance":     False,
        # Fama-French RMW: (Rev-COGS-SGA-IntExp) / Equity
        "operating_profitability": True,
        # Piotroski F-Score component: current ratio improvement
        "current_ratio_chg":      True,
    }

    # 10-K filings are not publicly available until ~45 days after period-end
    # (60 days for large accelerated filers, 90 for smaller). Shifting the
    # data index forward by 45 days before forward-filling prevents lookahead.
    _FILING_LAG = pd.Timedelta(days=45)

    for factor, higher_is_better in factors.items():
        frames = {}
        for ticker in tickers:
            if (ticker, factor) in edgar_df.columns:
                raw = edgar_df[(ticker, factor)].dropna()
                # Shift index forward by filing lag before forward-filling
                lagged = raw.copy()
                lagged.index = lagged.index + _FILING_LAG
                s = lagged.reindex(date_index, method="ffill")
                frames[ticker] = s

        if not frames:
            continue

        df = pd.DataFrame(frames, index=date_index).reindex(columns=tickers)
        if not higher_is_better:
            df = -df
        signals[f"{factor}_signal"] = _cs_rank(df)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FRED Macro Features
# ─────────────────────────────────────────────────────────────────────────────

_FRED_RENAME = {
    "T10Y2Y":       "yield_curve",
    "BAMLH0A0HYM2": "hy_spread",
    "VIXCLS":       "vix_macro",
    "DFF":          "fed_funds",
    "UNRATE":       "unemployment",
    "T10YIE":       "breakeven_inflation",
    "NAPM":         "ism_pmi",
    "ICSA":         "initial_claims",
    "PERMIT":       "building_permits",
}


def build_macro_features(
    fred_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of macro_feature_name → DataFrame (dates × tickers).
    Each value is the same scalar broadcast to every ticker — no cross-sectional
    ranking since these are market-wide conditioning variables.
    """
    features = {}

    if fred_df is None or fred_df.empty:
        return features

    fred_aligned = fred_df.reindex(date_index, method="ffill")

    for raw_col, nice_name in _FRED_RENAME.items():
        if raw_col not in fred_aligned.columns:
            continue
        s = fred_aligned[raw_col]
        # z-score relative to 252-day rolling window for stationarity
        mu  = s.rolling(252, min_periods=63).mean()
        std = s.rolling(252, min_periods=63).std().replace(0, np.nan)
        s_z = ((s - mu) / std).clip(-4, 4)
        features[nice_name] = _broadcast(s_z, tickers)

    # Derived: yield curve change (momentum in macro)
    if "T10Y2Y" in fred_aligned.columns:
        yc       = fred_aligned["T10Y2Y"]
        yc_chg5d = yc.diff(5)
        features["yield_curve_chg5d"] = _broadcast(yc_chg5d, tickers)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CBOE VIX Term Structure Features
# ─────────────────────────────────────────────────────────────────────────────

def build_vix_features(
    vix_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of vix_feature_name → DataFrame (dates × tickers).
    Columns: term_slope, vix_percentile, vix_change_5d
    """
    features = {}

    if vix_df is None or vix_df.empty:
        return features

    vix_aligned = vix_df.reindex(date_index, method="ffill")

    for col in ("term_slope", "vix_percentile", "vix_change_5d"):
        if col in vix_aligned.columns:
            features[f"vix_{col}"] = _broadcast(vix_aligned[col], tickers)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Short Interest Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_short_interest_signals(
    si_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Since yfinance only provides a current snapshot, we broadcast the static
    cross-sectional ranks to all dates. This won't capture time variation but
    still provides a useful structural signal (persistently high SI stocks
    tend to underperform or are crowded shorts — use carefully).

    Returns dict with:
      short_ratio_signal       — cross-sectional rank of short ratio (high = bearish)
      short_pct_float_signal   — cross-sectional rank of short % of float
    """
    signals = {}

    if si_df is None or si_df.empty:
        return signals

    for col, signal_name in [
        ("short_ratio",     "short_ratio_signal"),
        ("short_pct_float", "short_pct_float_signal"),
    ]:
        if col not in si_df.columns:
            continue
        s = si_df[col].reindex(tickers).astype(float)
        # Cross-sectional rank (static snapshot, broadcast to all dates)
        rank_val = s.rank(pct=True)
        df = pd.DataFrame(
            np.tile(rank_val.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index,
            columns=tickers,
        )
        # Only fill where we have the ticker
        valid_tickers = rank_val.dropna().index
        df.loc[:, ~df.columns.isin(valid_tickers)] = np.nan
        signals[signal_name] = df

    # NOTE: Short interest delta requires time-series data (bi-monthly FINRA).
    # Current implementation uses a static snapshot. Future improvement: load
    # historical short interest and compute short_interest_change_30d.

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Earnings Calendar Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_earnings_signals(
    earnings_dict: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    surprise_carry_days: int = 60,
    days_to_earnings_window: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    earnings_dict  : ticker → DataFrame with DatetimeIndex (earnings_date) and
                     eps_surprise_pct column (from load_earnings_calendar)
    surprise_carry_days  : forward-fill EPS surprise for this many days post-earnings
    days_to_earnings_window : only flag stocks within this many days of next earnings

    Returns
    -------
    dict with:
      earnings_surprise_signal  — cross-sectionally ranked post-earnings drift
      days_to_earnings_signal   — negated days to next earnings (nearer = higher rank)
                                  clipped to [0, window]; NaN if beyond window
    """
    signals = {}

    surprise_panel = pd.DataFrame(np.nan, index=date_index, columns=tickers)
    days_panel     = pd.DataFrame(np.nan, index=date_index, columns=tickers)

    for ticker in tickers:
        if ticker not in earnings_dict:
            continue
        earn = earnings_dict[ticker]
        if earn is None or earn.empty:
            continue
        if "eps_surprise_pct" not in earn.columns:
            continue

        earn_dates = earn.index.normalize()

        # EPS surprise: set value on earnings date, then forward-fill up to carry days
        surprise_ts = pd.Series(np.nan, index=date_index)
        for edate, row in earn.iterrows():
            edate_norm = pd.Timestamp(edate).normalize()
            # Find the closest business day >= earnings_date
            valid_dates = date_index[date_index >= edate_norm]
            if len(valid_dates) == 0:
                continue
            post_dates = valid_dates[:surprise_carry_days]
            surprise_ts.loc[post_dates] = row.get("eps_surprise_pct", np.nan)
        surprise_panel[ticker] = surprise_ts

        # Days to next earnings
        # For each date in date_index, find the next earnings date
        earn_dates_sorted = np.sort(earn_dates.unique())
        date_arr = date_index.values.astype("datetime64[D]")
        earn_arr = earn_dates_sorted.astype("datetime64[D]") if len(earn_dates_sorted) > 0 else np.array([], dtype="datetime64[D]")

        if len(earn_arr) == 0:
            continue

        days_to_next = np.full(len(date_arr), np.nan)
        for i, d in enumerate(date_arr):
            future = earn_arr[earn_arr > d]
            if len(future) > 0:
                days_to_next[i] = (future[0] - d).astype(int)

        # Only keep if within window; NaN otherwise
        within = days_to_next <= days_to_earnings_window
        days_ts = np.where(within, days_to_next, np.nan)
        days_panel[ticker] = days_ts

    # Cross-sectional rank surprise (higher surprise = better)
    if surprise_panel.notna().any().any():
        signals["earnings_surprise_signal"] = _cs_rank(surprise_panel)

    # Negate days so *closer* = higher rank (more signal)
    if days_panel.notna().any().any():
        signals["days_to_earnings_signal"] = _cs_rank(-days_panel)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Insider Transaction Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_insider_signals(
    insider_dict: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    lookback: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional signals from insider filing activity.

    Features:
      insider_activity_signal — rolling count of Form 4 filings (higher = more insider activity)

    Each Form 4 filing represents an insider transaction event. Higher filing
    frequency indicates more active insider trading. Academic evidence shows
    that spikes in insider activity (especially buying) precede positive returns
    (Lakonishok & Lee 2001).

    Uses filing date metadata from SEC EDGAR Submissions API — no XML parsing needed.
    """
    signals = {}

    activity_panel = pd.DataFrame(0.0, index=date_index, columns=tickers)
    net_buy_panel  = pd.DataFrame(0.0, index=date_index, columns=tickers)

    # Columns that may indicate transaction direction
    _DIR_COLS = {"type", "transactiontype", "acquisitionordisposition",
                 "transaction_type", "acquisition_or_disposition"}
    _BUY_VALS  = {"a", "p", "buy", "purchase"}
    _SELL_VALS = {"d", "s", "sale", "sell"}

    for ticker in tickers:
        if ticker not in insider_dict:
            continue
        txns = insider_dict[ticker]
        if txns is None or txns.empty:
            continue

        # Detect which column (if any) carries buy/sell direction
        dir_col = None
        for c in txns.columns:
            if c.lower().replace(" ", "_") in _DIR_COLS:
                dir_col = c
                break

        for idx_date, row in txns.iterrows():
            d = pd.Timestamp(idx_date).normalize()
            if d not in date_index:
                valid = date_index[date_index >= d]
                if len(valid) == 0:
                    continue
                d = valid[0]
            activity_panel.loc[d, ticker] += 1

            # Determine buy vs sell direction
            if dir_col is not None:
                val = str(row.get(dir_col, "")).strip().lower()
                if val in _BUY_VALS:
                    net_buy_panel.loc[d, ticker] += 1
                elif val in _SELL_VALS:
                    net_buy_panel.loc[d, ticker] -= 1

    # Rolling sum over lookback window
    activity_rolling = activity_panel.rolling(lookback, min_periods=1).sum()
    net_buy_rolling  = net_buy_panel.rolling(lookback, min_periods=1).sum()

    # Cross-sectional rank
    if activity_rolling.abs().sum().sum() > 0:
        signals["insider_activity_signal"] = _cs_rank(activity_rolling)
    if net_buy_rolling.abs().sum().sum() > 0:
        signals["insider_net_buy_signal"] = _cs_rank(net_buy_rolling)

    # Cluster buying detection: 3+ insiders buying within 30 days
    # This is one of the strongest insider signals — hard to coordinate
    # buying among multiple officers unless they know something
    cluster_panel = pd.DataFrame(0.0, index=date_index, columns=tickers)
    for ticker in tickers:
        if ticker not in insider_dict:
            continue
        txns = insider_dict[ticker]
        if txns is None or txns.empty:
            continue

        # Count distinct buy events per 30-day rolling window
        buy_dates = []
        for idx_date, row in txns.iterrows():
            # Detect if this is a buy transaction
            is_buy = False
            for col in txns.columns:
                val = str(row.get(col, '')).upper()
                if val in ('A', 'P', 'BUY', 'PURCHASE'):
                    is_buy = True
                    break
            if is_buy:
                d = pd.Timestamp(idx_date).normalize()
                buy_dates.append(d)

        if len(buy_dates) < 3:
            continue

        buy_series = pd.Series(1, index=pd.DatetimeIndex(buy_dates))
        buy_daily = buy_series.resample('D').sum().reindex(date_index, fill_value=0)
        cluster_count = buy_daily.rolling(30, min_periods=1).sum()
        cluster_panel[ticker] = (cluster_count >= 3).astype(float)

    if cluster_panel.sum().sum() > 0:
        signals["insider_cluster_buy_signal"] = _cs_rank(cluster_panel)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Analyst Upgrade / Downgrade Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_analyst_signals(
    analyst_dict: Dict[str, pd.DataFrame],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    lookback: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional signals from analyst upgrades/downgrades.

    Features:
      analyst_revision_signal — net upgrades minus downgrades (higher = bullish)
      analyst_coverage_signal — total analyst actions (higher coverage = more info)

    Analyst revisions are among the most documented alpha signals in institutional
    quant (Womack 1996, Barber et al. 2001). Upgrades precede positive returns;
    downgrades precede negative returns.
    """
    signals = {}

    revision_panel = pd.DataFrame(0.0, index=date_index, columns=tickers)
    coverage_panel = pd.DataFrame(0.0, index=date_index, columns=tickers)

    for ticker in tickers:
        if ticker not in analyst_dict:
            continue
        actions = analyst_dict[ticker]
        if actions is None or actions.empty:
            continue

        # Find action column
        action_col = None
        for c in actions.columns:
            if "action" in c.lower():
                action_col = c
                break

        for idx_date, row in actions.iterrows():
            d = pd.Timestamp(idx_date).normalize()
            if d not in date_index:
                valid = date_index[date_index >= d]
                if len(valid) == 0:
                    continue
                d = valid[0]

            coverage_panel.loc[d, ticker] += 1

            if action_col:
                action = str(row.get(action_col, "")).lower()
                if action in ("up", "upgrade"):
                    revision_panel.loc[d, ticker] += 1
                elif action in ("down", "downgrade"):
                    revision_panel.loc[d, ticker] -= 1
                elif action in ("main", "maintain", "reiterated"):
                    pass  # neutral
                elif action in ("init", "initiated"):
                    revision_panel.loc[d, ticker] += 0.5  # initiations are mildly bullish

    # Rolling sums
    revision_rolling = revision_panel.rolling(lookback, min_periods=1).sum()
    coverage_rolling = coverage_panel.rolling(lookback, min_periods=1).sum()

    if revision_rolling.abs().sum().sum() > 0:
        signals["analyst_revision_signal"] = _cs_rank(revision_rolling)
    if coverage_rolling.abs().sum().sum() > 0:
        signals["analyst_coverage_signal"] = _cs_rank(coverage_rolling)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Analyst Estimate Signals (target price, rating)
# ─────────────────────────────────────────────────────────────────────────────

def build_estimate_signals(
    estimates_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional signals from analyst estimates (snapshot data).

    Features:
      target_upside_signal    — ranked upside to analyst target price
      analyst_rating_signal   — ranked recommendation (lower=more bullish)
    """
    signals = {}

    if estimates_df is None or estimates_df.empty:
        return signals

    if 'targetUpside' in estimates_df.columns:
        upside = estimates_df['targetUpside'].reindex(tickers).astype(float)
        rank_val = upside.rank(pct=True)
        df = pd.DataFrame(
            np.tile(rank_val.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        valid = rank_val.dropna().index
        df.loc[:, ~df.columns.isin(valid)] = np.nan
        signals["target_upside_signal"] = df

    if 'recommendationMean' in estimates_df.columns:
        rating = -estimates_df['recommendationMean'].reindex(tickers).astype(float)
        rank_val = rating.rank(pct=True)
        df = pd.DataFrame(
            np.tile(rank_val.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        valid = rank_val.dropna().index
        df.loc[:, ~df.columns.isin(valid)] = np.nan
        signals["analyst_rating_signal"] = df

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Institutional Holdings Signals
# ─────────────────────────────────────────────────────────────────────────────

def build_institutional_signals(
    inst_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Build cross-sectional signals from institutional holder data (snapshot).

    Features:
      institutional_ownership_signal — ranked % held by institutions (higher = more owned)
      insider_ownership_signal       — ranked % held by insiders (higher = more skin in game)
    """
    signals = {}
    if inst_df is None or inst_df.empty:
        return signals

    if 'pctHeldByInstitutions' in inst_df.columns:
        pct_inst = inst_df['pctHeldByInstitutions'].reindex(tickers).astype(float)
        rank_val = pct_inst.rank(pct=True)
        df = pd.DataFrame(
            np.tile(rank_val.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        valid = rank_val.dropna().index
        df.loc[:, ~df.columns.isin(valid)] = np.nan
        signals["institutional_ownership_signal"] = df

    if 'pctHeldByInsiders' in inst_df.columns:
        pct_ins = inst_df['pctHeldByInsiders'].reindex(tickers).astype(float)
        rank_val = pct_ins.rank(pct=True)
        df = pd.DataFrame(
            np.tile(rank_val.values.reshape(1, -1), (len(date_index), 1)),
            index=date_index, columns=tickers,
        )
        valid = rank_val.dropna().index
        df.loc[:, ~df.columns.isin(valid)] = np.nan
        signals["insider_ownership_signal"] = df

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_alt_features(
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    edgar_df: Optional[pd.DataFrame] = None,
    fred_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    si_df: Optional[pd.DataFrame] = None,
    earnings_dict: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper that calls all five signal builders and returns a
    combined dict of signal_name → DataFrame (dates × tickers).
    """
    all_signals: Dict[str, pd.DataFrame] = {}

    def _merge(d: dict):
        for k, v in d.items():
            if v is not None and not v.empty:
                all_signals[k] = v.reindex(index=date_index, columns=tickers, method="ffill")

    _merge(build_fundamental_signals(edgar_df, tickers, date_index))
    _merge(build_macro_features(fred_df, tickers, date_index))
    _merge(build_vix_features(vix_df, tickers, date_index))
    _merge(build_short_interest_signals(si_df, tickers, date_index))

    if earnings_dict is not None:
        _merge(build_earnings_signals(earnings_dict, tickers, date_index))

    if kwargs.get("insider_dict"):
        _merge(build_insider_signals(kwargs["insider_dict"], tickers, date_index))

    if kwargs.get("analyst_dict"):
        _merge(build_analyst_signals(kwargs["analyst_dict"], tickers, date_index))

    if kwargs.get("estimates_df") is not None:
        _merge(build_estimate_signals(kwargs["estimates_df"], tickers, date_index))

    if kwargs.get("institutional_df") is not None:
        _merge(build_institutional_signals(kwargs["institutional_df"], tickers, date_index))

    return all_signals
