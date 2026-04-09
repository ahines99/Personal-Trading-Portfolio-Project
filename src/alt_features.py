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
    use_tier3_academic: bool = True,
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

    if use_tier3_academic and market_cap is not None:
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
    if use_tier3_academic and (revenue_p is not None and gprofit_p is not None and assets_p is not None):
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
    if use_tier3_academic and assets_p is not None and netinc_p is not None:
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
    use_tier3_academic: bool = True,
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
    if use_tier3_academic and close is not None:
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
# Tier 6: Distress / Default Signals
# ─────────────────────────────────────────────────────────────────────────────

def _get_sector_neutralize():
    try:
        from features import sector_neutralize as _fn
    except ImportError:
        from src.features import sector_neutralize as _fn
    return _fn


def _raw_panel_from_edgar(
    edgar_df: pd.DataFrame,
    field: str,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    filing_lag_days: int = 45,
) -> Optional[pd.DataFrame]:
    """Build a daily (dates × tickers) panel of a raw_* EDGAR field."""
    if edgar_df is None or edgar_df.empty:
        return None
    lag = pd.Timedelta(days=filing_lag_days)
    frames = {}
    for ticker in tickers:
        if (ticker, field) in edgar_df.columns:
            raw = edgar_df[(ticker, field)].dropna()
            if raw.empty:
                continue
            lagged = raw.copy()
            lagged.index = lagged.index + lag
            frames[ticker] = lagged.reindex(date_index, method="ffill")
    if not frames:
        return None
    return pd.DataFrame(frames, index=date_index).reindex(columns=tickers)


def build_chs_distress_signal(
    edgar_df: pd.DataFrame,
    close: pd.DataFrame,
    market_returns: pd.Series,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Campbell-Hilscher-Szilagyi (2008) 8-variable distress score.
    Output is cross-sectionally ranked on descending distress so that
    high distress maps to HIGH rank (short candidates). The downstream
    interpretation is that distressed stocks underperform (NEGATIVE premium),
    and the ranked signal is returned ready for a short tilt.
    """
    signals: Dict[str, pd.DataFrame] = {}
    if close is None or edgar_df is None or edgar_df.empty:
        return signals

    equity_p  = _raw_panel_from_edgar(edgar_df, "raw_equity",       tickers, date_index)
    liab_p    = _raw_panel_from_edgar(edgar_df, "raw_liabilities",  tickers, date_index)
    netinc_p  = _raw_panel_from_edgar(edgar_df, "raw_net_income",   tickers, date_index)
    cash_p    = _raw_panel_from_edgar(edgar_df, "raw_cash",         tickers, date_index)
    shares_p  = _raw_panel_from_edgar(edgar_df, "raw_shares_out",   tickers, date_index)
    if liab_p is None or netinc_p is None or shares_p is None:
        return signals

    close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
    me = (close_aligned * shares_p).clip(lower=1e-6)          # market equity
    tl = liab_p.fillna(0)
    mta = (me + tl).replace(0, np.nan)                        # market total assets

    nimta = netinc_p / mta
    # 4-qtr geom-weighted avg ≈ 252-day rolling mean of NIMTA
    nimtaavg = nimta.rolling(252, min_periods=63).mean()

    tlmta = tl / mta
    cashmta = (cash_p.fillna(0) / mta) if cash_p is not None else pd.DataFrame(0.0, index=date_index, columns=tickers)

    # Excess returns vs market (SPY)
    rets = close_aligned.pct_change()
    mkt = market_returns.reindex(date_index).fillna(0.0) if market_returns is not None else pd.Series(0.0, index=date_index)
    exret = rets.subtract(mkt, axis=0)
    # geometrically-decayed 12m average: approximate with EWMA half-life ~63d
    exretavg = exret.ewm(halflife=63, min_periods=63).mean()

    # 3-month daily vol
    sigma = rets.rolling(63, min_periods=21).std() * np.sqrt(252)

    # Relative size: log(ME / SPX total market cap) — proxy SPX ME with sum
    total_me = me.sum(axis=1).replace(0, np.nan)
    rsize = np.log(me.div(total_me, axis=0).replace(0, np.nan))

    # Market-to-book
    book = equity_p.replace(0, np.nan) if equity_p is not None else tl.replace(0, np.nan)
    mb = (me / book).clip(-50, 50)

    # Log price, capped at 15
    price_log = np.log(close_aligned.clip(upper=15).replace(0, np.nan))

    chs = (
        -9.164
        - 20.264 * nimtaavg
        + 1.416  * tlmta
        - 7.129  * exretavg
        + 1.411  * sigma
        - 0.045  * rsize
        - 2.132  * cashmta
        + 0.075  * mb
        - 0.058  * price_log
    )

    # High CHS → more distress → short candidate → rank descending
    signals["chs_distress_signal"] = _cs_rank(chs)
    return signals


def build_naive_dtd_signal(
    edgar_df: pd.DataFrame,
    close: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Bharath-Shumway (2008) naive Distance-to-Default."""
    signals: Dict[str, pd.DataFrame] = {}
    if close is None or edgar_df is None or edgar_df.empty:
        return signals
    liab_p   = _raw_panel_from_edgar(edgar_df, "raw_liabilities",  tickers, date_index)
    shares_p = _raw_panel_from_edgar(edgar_df, "raw_shares_out",   tickers, date_index)
    debt_p   = _raw_panel_from_edgar(edgar_df, "raw_lt_debt",      tickers, date_index)
    if liab_p is None or shares_p is None:
        return signals

    close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
    E = (close_aligned * shares_p).clip(lower=1e-6)
    # Face debt: short-term debt + 0.5 * long-term debt — approximate with total liabilities
    F = liab_p.clip(lower=1e-6) if debt_p is None else (liab_p.fillna(0) - 0.5 * debt_p.fillna(0)).clip(lower=1e-6)
    V = E + F

    rets = close_aligned.pct_change()
    sigma_E = rets.rolling(252, min_periods=63).std() * np.sqrt(252)
    # asset vol
    sigma_V = (E / V) * sigma_E + (F / V) * (0.05 + 0.25 * sigma_E)
    # trailing 1y log return
    mu = np.log(close_aligned / close_aligned.shift(252))
    T = 1.0
    num = np.log(V / F) + (mu - 0.5 * sigma_V**2) * T
    den = (sigma_V * np.sqrt(T)).replace(0, np.nan)
    dtd = num / den

    # High DtD = safe, low DtD = distressed.
    signals["naive_dtd_signal"] = _cs_rank(dtd)
    return signals


def build_altman_z_signal(
    edgar_df: pd.DataFrame,
    close: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Altman Z-score: low Z = distress."""
    signals: Dict[str, pd.DataFrame] = {}
    if close is None or edgar_df is None or edgar_df.empty:
        return signals
    assets_p  = _raw_panel_from_edgar(edgar_df, "raw_assets",       tickers, date_index)
    liab_p    = _raw_panel_from_edgar(edgar_df, "raw_liabilities",  tickers, date_index)
    equity_p  = _raw_panel_from_edgar(edgar_df, "raw_equity",       tickers, date_index)
    netinc_p  = _raw_panel_from_edgar(edgar_df, "raw_net_income",   tickers, date_index)
    intexp_p  = _raw_panel_from_edgar(edgar_df, "raw_interest_expense", tickers, date_index)
    rev_p     = _raw_panel_from_edgar(edgar_df, "raw_revenue",      tickers, date_index)
    shares_p  = _raw_panel_from_edgar(edgar_df, "raw_shares_out",   tickers, date_index)
    ca_p      = _raw_panel_from_edgar(edgar_df, "raw_current_assets", tickers, date_index)
    cl_p      = _raw_panel_from_edgar(edgar_df, "raw_current_liabilities", tickers, date_index)
    if assets_p is None or liab_p is None or equity_p is None or shares_p is None:
        return signals

    ta = assets_p.replace(0, np.nan)
    wc = (ca_p - cl_p) if (ca_p is not None and cl_p is not None) else (assets_p - liab_p)
    # Retained earnings proxy = equity (paid-in-capital not available)
    re = equity_p
    ebit = (netinc_p.fillna(0) + (intexp_p.fillna(0) if intexp_p is not None else 0))
    close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
    me = close_aligned * shares_p
    sales = rev_p if rev_p is not None else pd.DataFrame(0.0, index=date_index, columns=tickers)

    z = (
        1.2 * (wc / ta)
        + 1.4 * (re / ta)
        + 3.3 * (ebit / ta)
        + 0.6 * (me / liab_p.replace(0, np.nan))
        + 1.0 * (sales / ta)
    ).clip(-20, 20)

    signals["altman_z_signal"] = _cs_rank(z)
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Tier 6: Macro cross-sectional beta signals
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_beta(y: pd.DataFrame, x: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    """Compute rolling beta of each column of y on scalar x."""
    x_s = x.reindex(y.index)
    x_mean = x_s.rolling(window, min_periods=min_periods).mean()
    x_var = x_s.rolling(window, min_periods=min_periods).var().replace(0, np.nan)
    out = {}
    for col in y.columns:
        ys = y[col]
        y_mean = ys.rolling(window, min_periods=min_periods).mean()
        cov = (ys * x_s).rolling(window, min_periods=min_periods).mean() - (y_mean * x_mean)
        out[col] = cov / x_var
    return pd.DataFrame(out, index=y.index)


def build_dxy_beta_signal(
    returns: pd.DataFrame,
    dxy_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """36-month rolling beta to ΔDXY. Emit -β (long when DXY falls), plus
    interaction with 3M DXY momentum regime."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or dxy_df is None or dxy_df.empty or "dxy" not in dxy_df.columns:
        return signals
    rets = returns.reindex(index=date_index, columns=tickers)
    dxy = dxy_df["dxy"].reindex(date_index).ffill()
    ddxy = dxy.pct_change()
    window = 756  # ~36 months of bdays
    beta = _rolling_beta(rets, ddxy, window=window, min_periods=252)
    neg_beta = -beta
    signals["dxy_beta_signal"] = _cs_rank(neg_beta)

    # 3M dxy momentum regime sign
    dxy_mom = dxy.pct_change(63)
    regime = np.sign(dxy_mom).fillna(0.0)
    interaction = neg_beta.mul(regime, axis=0)
    signals["dxy_beta_x_dxy_mom_regime"] = _cs_rank(interaction)
    return signals


def build_yield_curve_pca_betas(
    returns: pd.DataFrame,
    yields_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """PCA on weekly yield changes → PC1 (level), PC2 (slope), PC3 (curvature).
    For each stock compute 60-month rolling beta to PC2 changes."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or yields_df is None or yields_df.empty:
        return signals
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return signals

    yld = yields_df.reindex(date_index).ffill()
    # weekly changes
    weekly = yld.resample("W-FRI").last().diff().dropna(how="all")
    weekly = weekly.dropna(axis=1, how="all").dropna(how="any")
    if weekly.empty or weekly.shape[1] < 3 or weekly.shape[0] < 30:
        return signals
    try:
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(weekly.values)
    except Exception:
        return signals
    pc2_weekly = pd.Series(pcs[:, 1], index=weekly.index)
    # expand back to daily
    pc2_daily = pc2_weekly.reindex(date_index, method="ffill")

    rets = returns.reindex(index=date_index, columns=tickers)
    window = 1260  # ~60 months bd
    beta = _rolling_beta(rets, pc2_daily.diff(), window=window, min_periods=252)
    signals["pc2_slope_beta_signal"] = _cs_rank(beta)
    return signals


def build_copper_gold_regime_features(
    returns: pd.DataFrame,
    cu_au_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Copper/gold ratio z-score, 3m rate-of-change, and per-stock 60m beta."""
    signals: Dict[str, pd.DataFrame] = {}
    if cu_au_df is None or cu_au_df.empty or "cu_au_ratio" not in cu_au_df.columns:
        return signals
    ratio = cu_au_df["cu_au_ratio"].reindex(date_index).ffill()
    mu = ratio.rolling(252, min_periods=63).mean()
    sd = ratio.rolling(252, min_periods=63).std().replace(0, np.nan)
    z = ((ratio - mu) / sd).clip(-4, 4)
    roc_3m = ratio.pct_change(63)

    signals["cu_au_zscore"] = _broadcast(z, tickers)
    signals["cu_au_roc_3m"] = _broadcast(roc_3m, tickers)

    if returns is not None:
        rets = returns.reindex(index=date_index, columns=tickers)
        beta = _rolling_beta(rets, ratio.pct_change(), window=1260, min_periods=252)
        signals["cyclicality_cu_au_signal"] = _cs_rank(beta)
    return signals


def build_credit_beta_signal(
    returns: pd.DataFrame,
    hy_oas: pd.Series,
    ig_oas: pd.Series,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """36m rolling beta of each stock to Δ(HY-IG) spread."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or hy_oas is None or ig_oas is None:
        return signals
    hy = hy_oas.reindex(date_index).ffill()
    ig = ig_oas.reindex(date_index).ffill()
    spread = (hy - ig).diff()
    rets = returns.reindex(index=date_index, columns=tickers)
    beta = _rolling_beta(rets, spread, window=756, min_periods=252)
    signals["credit_beta_signal"] = _cs_rank(beta)
    return signals


def build_cross_asset_momentum_features(
    cross_asset_panel: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Broadcast 12-1 momentum of SPY/TLT/DBC/UUP/GLD/HYG as global features."""
    signals: Dict[str, pd.DataFrame] = {}
    if cross_asset_panel is None or cross_asset_panel.empty:
        return signals
    panel = cross_asset_panel.reindex(date_index).ffill()
    for etf in ("SPY", "TLT", "DBC", "UUP", "GLD", "HYG"):
        col = f"mom12_1_{etf}"
        if col in panel.columns:
            signals[f"mom_{etf.lower()}_12_1"] = _broadcast(panel[col], tickers)
    return signals


def build_breakeven_inflation_beta_signal(
    returns: pd.DataFrame,
    breakeven_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """60m rolling inflation beta, interacted with sign(Δbreakeven)."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or breakeven_df is None or breakeven_df.empty:
        return signals
    col = "breakeven" if "breakeven" in breakeven_df.columns else breakeven_df.columns[0]
    bei = breakeven_df[col].reindex(date_index).ffill()
    dbei = bei.diff()
    rets = returns.reindex(index=date_index, columns=tickers)
    beta = _rolling_beta(rets, dbei, window=1260, min_periods=252)
    regime = np.sign(dbei).fillna(0.0)
    interaction = beta.mul(regime, axis=0)
    signals["inflation_sensitivity_signal"] = _cs_rank(interaction)
    return signals


def build_oil_beta_signal(
    returns: pd.DataFrame,
    oil_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """60m rolling beta to oil monthly returns + interaction with 3m oil momentum."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or oil_df is None or oil_df.empty:
        return signals
    col = "oil_wti" if "oil_wti" in oil_df.columns else oil_df.columns[0]
    oil = oil_df[col].reindex(date_index).ffill()
    oil_mret = oil.pct_change(21)
    rets = returns.reindex(index=date_index, columns=tickers)
    # use daily returns beta on oil daily returns (simpler, stable)
    oil_dret = oil.pct_change()
    beta = _rolling_beta(rets, oil_dret, window=1260, min_periods=252)
    signals["oil_beta_signal"] = _cs_rank(beta)

    regime = np.sign(oil_mret).fillna(0.0)
    interaction = beta.mul(regime, axis=0)
    signals["oil_beta_x_oil_momentum"] = _cs_rank(interaction)
    return signals


def build_vvix_features(
    vvix_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """VVIX/VIX ratio and 5-day change, broadcast as globals."""
    signals: Dict[str, pd.DataFrame] = {}
    if vvix_df is None or vvix_df.empty:
        return signals
    vvix = vvix_df["vvix"].reindex(date_index).ffill() if "vvix" in vvix_df.columns else None
    # pull VIX from vix_df (CBOE term structure)
    vix = None
    if vix_df is not None and not vix_df.empty and "VIX" in vix_df.columns:
        vix = vix_df["VIX"].reindex(date_index).ffill()
    if vvix is None or vix is None:
        return signals
    ratio = (vvix / vix.replace(0, np.nan))
    chg_5d = ratio.diff(5)
    signals["vvix_vix_ratio"] = _broadcast(ratio, tickers)
    signals["vvix_vix_ratio_chg5d"] = _broadcast(chg_5d, tickers)
    return signals


def build_sector_oas_momentum_signal(
    sector_oas_df: pd.DataFrame,
    sector_map: Dict[str, str],
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Map tickers to sectors, then to rating-bucket OAS changes.
    Rationale: map sectors to a representative credit-rating bucket based on
    sector-average issuer quality. High-quality / defensive sectors
    (Financials, Utilities, Consumer Defensive/Staples, Healthcare,
    Technology, Communication Services) sit in IG territory → BBB bucket.
    Cyclical / commodity-sensitive sectors (Industrials, Consumer
    Cyclical/Discretionary, Basic Materials/Materials, Energy, Real Estate)
    have more crossover exposure → BB bucket. We deliberately do NOT use
    the CCC (distress) bucket — no broad GICS sector averages CCC. Callers
    who disagree with this assignment should override via a parameter or
    pre-remap their sector_map before calling. If no map, broadcast mean
    OAS change. Bucket series names (BBB/BB/B/CCC) are preserved in the
    input sector_oas_df; we simply don't reference B/CCC here."""
    signals: Dict[str, pd.DataFrame] = {}
    if sector_oas_df is None or sector_oas_df.empty:
        return signals
    panel = sector_oas_df.reindex(date_index).ffill()
    # Negative of 21d OAS change: tightening (falling OAS) → positive signal
    d_oas = -panel.diff(21)

    # Corrected mapping (2026-04-05): previous version inverted quality,
    # mapping Tech/Comms/Real Estate to CCC (distress) and Staples/Health
    # Care to B, which contaminated the signal. High-quality sectors → BBB,
    # cyclicals → BB. Both GICS and Yahoo/EODHD sector names are included.
    _GICS_TO_BUCKET = {
        # IG (BBB) — high-quality / defensive
        "Financials": "BBB", "Financial Services": "BBB",
        "Utilities": "BBB",
        "Consumer Staples": "BBB", "Consumer Defensive": "BBB",
        "Health Care": "BBB", "Healthcare": "BBB",
        "Information Technology": "BBB", "Technology": "BBB",
        "Communication Services": "BBB",
        # Crossover (BB) — cyclical / commodity-sensitive
        "Industrials": "BB",
        "Consumer Discretionary": "BB", "Consumer Cyclical": "BB",
        "Materials": "BB", "Basic Materials": "BB",
        "Energy": "BB",
        "Real Estate": "BB",
    }

    if not sector_map:
        # broadcast mean signal
        avg = d_oas.mean(axis=1)
        signals["sector_oas_momentum_signal"] = _broadcast(avg, tickers)
        return signals

    out = pd.DataFrame(np.nan, index=date_index, columns=tickers)
    for ticker in tickers:
        sec = sector_map.get(ticker)
        if sec is None:
            continue
        bucket = _GICS_TO_BUCKET.get(sec, "BBB")
        if bucket in d_oas.columns:
            out[ticker] = d_oas[bucket]
    signals["sector_oas_momentum_signal"] = _cs_rank(out)
    return signals


def build_ebp_beta_signal(
    returns: pd.DataFrame,
    ebp_df: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """36m rolling beta to Excess Bond Premium changes."""
    signals: Dict[str, pd.DataFrame] = {}
    if returns is None or ebp_df is None or ebp_df.empty or "ebp" not in ebp_df.columns:
        return signals
    ebp = ebp_df["ebp"].reindex(date_index).ffill()
    debp = ebp.diff()
    rets = returns.reindex(index=date_index, columns=tickers)
    beta = _rolling_beta(rets, debp, window=756, min_periods=252)
    signals["ebp_beta_signal"] = _cs_rank(beta)
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
    # ── Tier 6: macro / distress signal gates ────────────────────────────
    use_chs: bool = True,
    use_dtd: bool = True,
    use_altman_z: bool = True,
    use_macro_cross_section: bool = True,
    use_sni_variants: bool = True,
    # ── Tier 3 academic fundamentals / earnings gate ─────────────────────
    use_tier3_academic: bool = True,
    # ── Passthrough alias controlling sector-neutralization of fundamentals
    # Forwarded to build_fundamental_signals' sector_neutralize_signals param.
    # ONLY gates fundamental signal _sni variants (NOT Tier 6 distress/macro
    # _sni variants, which remain under `use_sni_variants`).
    sector_neutralize_fundamentals: bool = True,
    dxy_df: Optional[pd.DataFrame] = None,
    breakeven_df: Optional[pd.DataFrame] = None,
    vvix_df: Optional[pd.DataFrame] = None,
    oil_df: Optional[pd.DataFrame] = None,
    copper_gold_df: Optional[pd.DataFrame] = None,
    cross_asset_panel: Optional[pd.DataFrame] = None,
    ig_oas_df: Optional[pd.DataFrame] = None,
    sector_oas_df: Optional[pd.DataFrame] = None,
    ebp_df: Optional[pd.DataFrame] = None,
    treasury_yields_df: Optional[pd.DataFrame] = None,
    market_returns: Optional[pd.Series] = None,
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
    use_tier3_academic : bool, default True
        Gate for Tier 3 academic value/quality signals built from raw EDGAR
        fundamentals + market cap, plus the earnings-announcement-return (EAR)
        signal. When False, the following are skipped (and their _sni variants
        are never produced):
            value_composite_signal, earnings_yield_signal, sales_yield_signal,
            fcf_yield_signal, ebit_ev_signal, cash_based_op_prof_signal,
            piotroski_f_score_signal, earnings_ann_return_signal
        Core Tier 1-2 fundamental signals (book_to_market, roe, gross_margin,
        gross_profitability, accruals, NOA, issuance, op_prof, current_ratio_chg)
        are NOT affected by this flag.
    sector_neutralize_fundamentals : bool, default True
        PASSTHROUGH alias from the top-level build_alt_features API to
        build_fundamental_signals' `sector_neutralize_signals` parameter.
        When True (default), sector-neutralize fundamental signals and emit
        `_sni` variants. When False, skip sector-neutralization of
        fundamentals entirely — no `_sni` variants for fundamentals.
        Interaction: this ONLY controls fundamental signal `_sni` variants.
        The existing `use_sni_variants` flag is separate and gates Tier 6
        distress / macro `_sni` variants only.
    """
    all_signals: Dict[str, pd.DataFrame] = {}

    def _merge(d: dict):
        for k, v in d.items():
            if v is not None and not v.empty:
                all_signals[k] = v.reindex(index=date_index, columns=tickers, method="ffill").astype(np.float32)

    _merge(build_fundamental_signals(
        edgar_df, tickers, date_index, sector_map=sector_map, close=close,
        sector_neutralize_signals=sector_neutralize_fundamentals,
        use_tier3_academic=use_tier3_academic,
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
            use_tier3_academic=use_tier3_academic,
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

    # ────────────────────────────────────────────────────────────────────
    # Tier 6: distress & macro cross-sectional signals
    # ────────────────────────────────────────────────────────────────────
    # Build per-stock returns panel from close (needed by macro-beta signals)
    returns_panel = None
    if close is not None and not close.empty:
        close_a = close.reindex(index=date_index, columns=tickers, method="ffill")
        returns_panel = close_a.pct_change()

    mkt_rets = market_returns
    if mkt_rets is None and returns_panel is not None:
        mkt_rets = returns_panel.mean(axis=1)

    if use_chs and edgar_df is not None and close is not None:
        _merge(build_chs_distress_signal(edgar_df, close, mkt_rets, tickers, date_index))
    if use_dtd and edgar_df is not None and close is not None:
        _merge(build_naive_dtd_signal(edgar_df, close, tickers, date_index))
    if use_altman_z and edgar_df is not None and close is not None:
        _merge(build_altman_z_signal(edgar_df, close, tickers, date_index))

    if use_macro_cross_section:
        if dxy_df is not None and returns_panel is not None:
            _merge(build_dxy_beta_signal(returns_panel, dxy_df, tickers, date_index))
        if treasury_yields_df is not None and returns_panel is not None:
            _merge(build_yield_curve_pca_betas(
                returns_panel, treasury_yields_df, tickers, date_index))
        if copper_gold_df is not None:
            _merge(build_copper_gold_regime_features(
                returns_panel, copper_gold_df, tickers, date_index))
        if ig_oas_df is not None and fred_df is not None and returns_panel is not None:
            hy_series = None
            if "BAMLH0A0HYM2" in fred_df.columns:
                hy_series = fred_df["BAMLH0A0HYM2"]
            ig_series = ig_oas_df["ig_oas"] if "ig_oas" in ig_oas_df.columns else None
            if hy_series is not None and ig_series is not None:
                _merge(build_credit_beta_signal(
                    returns_panel, hy_series, ig_series, tickers, date_index))
        if cross_asset_panel is not None:
            _merge(build_cross_asset_momentum_features(
                cross_asset_panel, tickers, date_index))
        if breakeven_df is not None and returns_panel is not None:
            _merge(build_breakeven_inflation_beta_signal(
                returns_panel, breakeven_df, tickers, date_index))
        if oil_df is not None and returns_panel is not None:
            _merge(build_oil_beta_signal(returns_panel, oil_df, tickers, date_index))
        if vvix_df is not None:
            _merge(build_vvix_features(vvix_df, vix_df, tickers, date_index))
        if sector_oas_df is not None:
            _merge(build_sector_oas_momentum_signal(
                sector_oas_df, sector_map, tickers, date_index))
        if ebp_df is not None and returns_panel is not None:
            _merge(build_ebp_beta_signal(returns_panel, ebp_df, tickers, date_index))

    # ────────────────────────────────────────────────────────────────────
    # Sector-neutralized variants for new cross-sectional signals
    # ────────────────────────────────────────────────────────────────────
    if use_sni_variants and sector_map:
        try:
            _sn = _get_sector_neutralize()
            _sni_candidates = [
                "chs_distress_signal",
                "naive_dtd_signal",
                "altman_z_signal",
                "dxy_beta_signal",
                "pc2_slope_beta_signal",
                "cyclicality_cu_au_signal",
                "credit_beta_signal",
                "inflation_sensitivity_signal",
                "oil_beta_signal",
                "ebp_beta_signal",
                "sector_oas_momentum_signal",
            ]
            for base in _sni_candidates:
                if base in all_signals:
                    try:
                        all_signals[f"{base}_sni"] = _sn(all_signals[base], sector_map)
                    except Exception:
                        pass
        except Exception:
            pass

    return all_signals
