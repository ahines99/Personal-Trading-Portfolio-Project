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
    sector_map: Optional[Dict[str, str]] = None,
    sector_neutralize_signals: bool = True,
    close: Optional[pd.DataFrame] = None,
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

    # ── Tier 3B: helper to build daily panel of a raw fundamentals field ────
    _FILING_LAG_RAW = _FILING_LAG

    def _raw_panel(field: str) -> Optional[pd.DataFrame]:
        """Build (dates × tickers) panel of a raw_* edgar field, filing-lagged."""
        frames = {}
        for ticker in tickers:
            if (ticker, field) in edgar_df.columns:
                raw = edgar_df[(ticker, field)].dropna()
                if raw.empty:
                    continue
                lagged = raw.copy()
                lagged.index = lagged.index + _FILING_LAG_RAW
                frames[ticker] = lagged.reindex(date_index, method="ffill")
        if not frames:
            return None
        return pd.DataFrame(frames, index=date_index).reindex(columns=tickers)

    assets_p   = _raw_panel("raw_assets")
    liab_p     = _raw_panel("raw_liabilities")
    equity_p   = _raw_panel("raw_equity")
    netinc_p   = _raw_panel("raw_net_income")
    gprofit_p  = _raw_panel("raw_gross_profit")
    revenue_p  = _raw_panel("raw_revenue")
    ocf_p      = _raw_panel("raw_operating_cf")
    cash_p     = _raw_panel("raw_cash")
    ltdebt_p   = _raw_panel("raw_lt_debt")
    shares_p   = _raw_panel("raw_shares_out")
    sga_p      = _raw_panel("raw_sga")
    intexp_p   = _raw_panel("raw_interest_expense")
    ca_p       = _raw_panel("raw_current_assets")
    cl_p       = _raw_panel("raw_current_liabilities")

    # Market cap (Close × Shares Outstanding) — required for value yields
    market_cap = None
    if close is not None and shares_p is not None:
        close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
        market_cap = close_aligned * shares_p

    # ── Tier 3B Feature 1: Multi-factor Value Composite ────────────────────
    # Asness-Frazzini "Devil in HML's Details" (2013); Loughran-Wellman (2011)
    new_signals: Dict[str, pd.DataFrame] = {}

    if market_cap is not None:
        mc_safe = market_cap.replace(0, np.nan)

        # earnings_yield = Net Income / Market Cap
        if netinc_p is not None:
            ey = (netinc_p / mc_safe).clip(-5, 5)
            new_signals["earnings_yield_signal"] = _cs_rank(ey)

        # sales_yield = Revenue / Market Cap
        if revenue_p is not None:
            sy = (revenue_p / mc_safe).clip(-50, 50)
            new_signals["sales_yield_signal"] = _cs_rank(sy)

        # fcf_yield ≈ Operating CF / Market Cap (Capex unavailable)
        if ocf_p is not None:
            fy = (ocf_p / mc_safe).clip(-5, 5)
            new_signals["fcf_yield_signal"] = _cs_rank(fy)

        # ebit_to_ev = (NetIncome + InterestExpense) / (MarketCap + Debt - Cash)
        if netinc_p is not None and intexp_p is not None:
            ebit_num = netinc_p + intexp_p.fillna(0)
            debt_term = ltdebt_p.fillna(0) if ltdebt_p is not None else 0
            cash_term = cash_p.fillna(0) if cash_p is not None else 0
            ev = (market_cap + debt_term - cash_term).replace(0, np.nan)
            ebit_ev = (ebit_num / ev).clip(-5, 5)
            new_signals["ebit_ev_signal"] = _cs_rank(ebit_ev)

        # Composite: mean of ranked yields across available ones per row
        yield_ranks = [
            new_signals[k] for k in
            ("earnings_yield_signal", "sales_yield_signal",
             "fcf_yield_signal", "ebit_ev_signal")
            if k in new_signals
        ]
        if yield_ranks:
            composite = sum(yield_ranks) / len(yield_ranks)
            # composite is already in [0,1] range (mean of ranks), rerank
            # cross-sectionally so it's a clean [0,1] signal
            new_signals["value_composite_signal"] = _cs_rank(composite)

    # ── Tier 3B Feature 2: Cash-Based Operating Profitability ──────────────
    # Ball, Gerakos, Linnainmaa, Nikolaev (2016)
    # = (Revenue - COGS - SGA - ΔWC) / Total Assets
    # where COGS = Revenue - GrossProfit, WC = CurrentAssets - CurrentLiabilities
    if (revenue_p is not None and gprofit_p is not None and assets_p is not None):
        cogs_p = revenue_p - gprofit_p
        sga_term = sga_p.fillna(0) if sga_p is not None else 0
        if ca_p is not None and cl_p is not None:
            wc = ca_p - cl_p
            delta_wc = wc.diff(252)
        else:
            delta_wc = 0
        cbop_num = revenue_p - cogs_p - sga_term - (delta_wc if not isinstance(delta_wc, int) else 0)
        cbop = (cbop_num / assets_p.replace(0, np.nan)).clip(-5, 5)
        new_signals["cash_based_op_prof_signal"] = _cs_rank(cbop)

    # ── Tier 3B Feature 3: Piotroski F-Score (9-signal composite) ──────────
    # Piotroski (2000)
    if assets_p is not None and netinc_p is not None:
        assets_safe = assets_p.replace(0, np.nan)
        roa = netinc_p / assets_safe

        f1 = (roa > 0).astype(float)                                     # 1
        f2 = ((ocf_p > 0).astype(float)
              if ocf_p is not None else pd.DataFrame(0.0, index=date_index, columns=tickers))  # 2
        f3 = (roa.diff(252) > 0).astype(float)                           # 3
        if ocf_p is not None:
            f4 = (ocf_p > netinc_p).astype(float)                        # 4
        else:
            f4 = pd.DataFrame(0.0, index=date_index, columns=tickers)
        # 5: Long-term Debt / Assets decreasing YoY (use total Debt proxy if needed)
        debt_for_lev = ltdebt_p if ltdebt_p is not None else liab_p
        if debt_for_lev is not None:
            lev_ratio = debt_for_lev / assets_safe
            f5 = (lev_ratio.diff(252) < 0).astype(float)
        else:
            f5 = pd.DataFrame(0.0, index=date_index, columns=tickers)
        # 6: Current Ratio increasing YoY
        if ca_p is not None and cl_p is not None:
            curr_ratio = ca_p / cl_p.replace(0, np.nan)
            f6 = (curr_ratio.diff(252) > 0).astype(float)
        else:
            f6 = pd.DataFrame(0.0, index=date_index, columns=tickers)
        # 7: Shares NOT increasing YoY
        if shares_p is not None:
            f7 = (shares_p.diff(252) <= 0).astype(float)
        else:
            f7 = pd.DataFrame(0.0, index=date_index, columns=tickers)
        # 8: Gross Margin increasing YoY
        if gprofit_p is not None and revenue_p is not None:
            gm = gprofit_p / revenue_p.replace(0, np.nan)
            f8 = (gm.diff(252) > 0).astype(float)
        else:
            f8 = pd.DataFrame(0.0, index=date_index, columns=tickers)
        # 9: Asset Turnover increasing YoY
        if revenue_p is not None:
            at_ratio = revenue_p / assets_safe
            f9 = (at_ratio.diff(252) > 0).astype(float)
        else:
            f9 = pd.DataFrame(0.0, index=date_index, columns=tickers)

        f_score = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        # mask out rows where we have no fundamentals at all (roa all NaN)
        f_score = f_score.where(roa.notna())
        new_signals["piotroski_f_score_signal"] = _cs_rank(f_score)

    # merge the Tier 3B signals into the main dict
    signals.update(new_signals)

    # ── Sector-neutralized variants ─────────────────────────────────────────
    # Subtract per-date sector mean from each fundamental signal so the signal
    # captures within-industry differences rather than sector-level tilts
    # (JKP 2023, AQR best practice for quality / value / momentum).
    if sector_neutralize_signals and sector_map:
        try:
            from features import sector_neutralize as _sector_neutralize_fn
        except ImportError:
            from src.features import sector_neutralize as _sector_neutralize_fn
        _sni_factors = [
            "book_to_market", "roe", "gross_margin", "asset_growth",
            "leverage", "gross_profitability", "accruals",
            "net_operating_assets", "net_share_issuance",
            "operating_profitability", "current_ratio_chg",
            # Tier 3B additions
            "earnings_yield", "sales_yield", "fcf_yield", "ebit_ev",
            "value_composite", "cash_based_op_prof", "piotroski_f_score",
        ]
        for factor in _sni_factors:
            sig_name = f"{factor}_signal"
            if sig_name not in signals:
                continue
            try:
                signals[f"{sig_name}_sni"] = _sector_neutralize_fn(
                    signals[sig_name], sector_map
                )
            except Exception:
                pass

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
    close: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
    sector_neutralize_signals: bool = True,
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

    # ── Tier 3B Feature 4: 3-day Earnings Announcement Return (EAR) ─────────
    # Brandt-Kishore-Santa-Clara-Venkatachalam (2008) — orthogonal to PEAD.
    # For each earnings date t, compute (close[t+1] - close[t-1]) / close[t-1]
    # and carry forward for `surprise_carry_days` business days.
    if close is not None:
        close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
        ear_panel = pd.DataFrame(np.nan, index=date_index, columns=tickers)

        for ticker in tickers:
            if ticker not in earnings_dict:
                continue
            earn = earnings_dict[ticker]
            if earn is None or earn.empty:
                continue
            if ticker not in close_aligned.columns:
                continue
            px = close_aligned[ticker]
            if px.dropna().empty:
                continue

            ear_ts = pd.Series(np.nan, index=date_index)
            for edate in earn.index:
                edate_norm = pd.Timestamp(edate).normalize()
                # Anchor t = first business day >= announcement
                valid = date_index[date_index >= edate_norm]
                if len(valid) == 0:
                    continue
                t_idx = date_index.get_loc(valid[0])
                if t_idx < 1 or t_idx + 1 >= len(date_index):
                    continue
                p_pre = px.iloc[t_idx - 1]
                p_post = px.iloc[t_idx + 1]
                if pd.isna(p_pre) or pd.isna(p_post) or p_pre == 0:
                    continue
                ear_val = (p_post - p_pre) / p_pre
                # Carry forward for surprise_carry_days
                post_dates = date_index[t_idx:t_idx + surprise_carry_days]
                ear_ts.loc[post_dates] = ear_val
            ear_panel[ticker] = ear_ts

        if ear_panel.notna().any().any():
            signals["earnings_ann_return_signal"] = _cs_rank(ear_panel)

    # ── Sector-neutralized variants (Tier 3A / 3B) ─────────────────────────
    if sector_neutralize_signals and sector_map:
        try:
            from features import sector_neutralize as _sector_neutralize_fn
        except ImportError:
            from src.features import sector_neutralize as _sector_neutralize_fn
        for base in ("earnings_ann_return_signal",):
            if base not in signals:
                continue
            try:
                signals[f"{base}_sni"] = _sector_neutralize_fn(
                    signals[base], sector_map
                )
            except Exception:
                pass

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
    sector_map: Optional[Dict[str, str]] = None,
    close: Optional[pd.DataFrame] = None,
    use_short_interest: bool = False,
    use_institutional_holdings: bool = False,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper that calls all five signal builders and returns a
    combined dict of signal_name → DataFrame (dates × tickers).

    Parameters
    ----------
    use_short_interest : bool, default False
        Gate for short-interest signals (short_ratio_signal, short_pct_float_signal).
        DISABLED until historical FINRA bi-monthly short-interest archive is
        integrated. Current snapshot-broadcast breaks point-in-time correctness
        and creates lookahead bias.
    use_institutional_holdings : bool, default False
        Gate for institutional / insider ownership signals
        (institutional_ownership_signal, insider_ownership_signal).
        DISABLED until time-series 13F data is integrated (e.g., via SEC EDGAR
        13F-HR filings). Current snapshot-broadcast breaks point-in-time
        correctness.
    """
    all_signals: Dict[str, pd.DataFrame] = {}

    def _merge(d: dict):
        for k, v in d.items():
            if v is not None and not v.empty:
                all_signals[k] = v.reindex(index=date_index, columns=tickers, method="ffill")

    _merge(build_fundamental_signals(
        edgar_df, tickers, date_index, sector_map=sector_map, close=close,
    ))
    _merge(build_macro_features(fred_df, tickers, date_index))
    _merge(build_vix_features(vix_df, tickers, date_index))

    # DISABLED until historical FINRA bi-monthly short-interest archive is
    # integrated. Current snapshot-broadcast breaks point-in-time correctness
    # and creates lookahead bias. Opt-in via use_short_interest=True.
    if use_short_interest:
        _merge(build_short_interest_signals(si_df, tickers, date_index))

    if earnings_dict is not None:
        _merge(build_earnings_signals(
            earnings_dict, tickers, date_index,
            close=close, sector_map=sector_map,
        ))

    if kwargs.get("insider_dict"):
        _merge(build_insider_signals(kwargs["insider_dict"], tickers, date_index))

    if kwargs.get("analyst_dict"):
        _merge(build_analyst_signals(kwargs["analyst_dict"], tickers, date_index))

    if kwargs.get("estimates_df") is not None:
        _merge(build_estimate_signals(kwargs["estimates_df"], tickers, date_index))

    # DISABLED until time-series 13F data is integrated (e.g., via SEC EDGAR
    # 13F-HR filings). Current snapshot-broadcast breaks point-in-time
    # correctness. Opt-in via use_institutional_holdings=True.
    if use_institutional_holdings and kwargs.get("institutional_df") is not None:
        _merge(build_institutional_signals(kwargs["institutional_df"], tickers, date_index))

    return all_signals
