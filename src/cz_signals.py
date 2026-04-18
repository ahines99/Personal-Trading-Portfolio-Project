"""
cz_signals.py
-------------
Chen-Zimmermann anomaly signals built from our live data sources.

Discovery: run_cz_research.py identified 10 high-IC novel signals from the
C&Z Open Source Asset Pricing dataset that we don't currently have.

This module re-implements those signals from scratch using our own data
(EDGAR, prices) so they update in real-time for live trading.

PHASE 1 — Price-only signals (built first, no EDGAR changes needed):
    coskewness_signal       Harvey & Siddique (2000), C&Z IC_IR 0.20
    coskew_acx_signal       Ang, Chen & Xing (2006), C&Z IC_IR 0.12
    mom_season_signal       Heston & Sadka (2008), C&Z IC_IR 0.15

PHASE 2 — Accounting signals (require new EDGAR fields, see roadmap):
    payout_yield_signal     Boudoukh et al. (2007), C&Z IC_IR 0.15
    net_payout_yield_signal Boudoukh et al. (2007), C&Z IC_IR 0.18
    xfin_signal             Bradshaw, Richardson, Sloan (2006), C&Z IC_IR 0.18
    cfp_signal              Desai et al. (2004), C&Z IC_IR 0.11
    operprof_rd_signal      Ball et al. (2016), C&Z IC_IR 0.13
    tax_signal              Lev & Nissim (2004), C&Z IC_IR 0.12
    deldrc_signal           Prakash & Sinha (2013), C&Z IC_IR 0.14

All signals are returned as cross-sectionally ranked [0, 1] DataFrames
(date × ticker), with sign already oriented so higher = predicts higher
forward returns. Stack directly into the feature matrix.
"""

from __future__ import annotations
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [0, 1] per row."""
    return df.rank(axis=1, pct=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coskewness (Harvey & Siddique 2000) — IC_IR 0.20 in our universe
# ─────────────────────────────────────────────────────────────────────────────

def build_coskewness_signal(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback_months: int = 60,
    min_months: int = 24,
) -> pd.DataFrame:
    """Coskewness signal using past 60 months of monthly data.

    Per C&Z definition (Harvey & Siddique 2000):
        coskew_i = E[r_i * r_m^2] / (sigma_i * sigma_m^2)
    where r_i and r_m are de-meaned returns.

    Sign: NEGATIVE coskewness predicts HIGHER returns (reversed for ranking).

    Parameters
    ----------
    returns : DataFrame (daily date × ticker)
    market_returns : Series (daily date), e.g. SPY returns
    lookback_months : 60 (per C&Z spec)
    min_months : 24 minimum for valid signal

    Returns
    -------
    (date × ticker) DataFrame, cross-sectionally ranked [0, 1]
    """
    # Convert to monthly
    monthly = (1.0 + returns.replace([np.inf, -np.inf], np.nan)).resample("ME").prod() - 1.0
    monthly_mkt = (1.0 + market_returns.replace([np.inf, -np.inf], np.nan)).resample("ME").prod() - 1.0

    # De-mean rolling
    mkt_demean = monthly_mkt - monthly_mkt.rolling(lookback_months, min_periods=min_months).mean()
    mkt_var = monthly_mkt.rolling(lookback_months, min_periods=min_months).var()
    mkt_std = np.sqrt(mkt_var)

    # For each ticker, compute coskewness
    # E[r_i_demean * r_m_demean^2] / (std_i * var_m)
    coskew_panel = pd.DataFrame(index=monthly.index, columns=monthly.columns, dtype=float)
    mkt_sq_demean = mkt_demean ** 2

    for ticker in monthly.columns:
        r_i = monthly[ticker]
        r_i_demean = r_i - r_i.rolling(lookback_months, min_periods=min_months).mean()
        std_i = r_i.rolling(lookback_months, min_periods=min_months).std()

        # E[r_i_demean * r_m_sq_demean]
        prod = r_i_demean * mkt_sq_demean
        e_prod = prod.rolling(lookback_months, min_periods=min_months).mean()

        denom = std_i * mkt_var
        coskew = e_prod / denom.replace(0, np.nan)
        coskew_panel[ticker] = coskew.clip(-3, 3)

    # Negate (negative coskew predicts higher returns)
    coskew_panel = -coskew_panel

    # Forward-fill monthly signal to daily trading days
    daily = coskew_panel.reindex(returns.index, method="ffill", limit=25)

    # Cross-sectional rank
    return _cs_rank(daily)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CoskewACX (Ang, Chen, Xing 2006) — IC_IR 0.12, daily-data variant
# ─────────────────────────────────────────────────────────────────────────────

def build_coskew_acx_signal(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback_days: int = 252,
    min_days: int = 126,
) -> pd.DataFrame:
    """Coskewness using past year of DAILY returns (Ang/Chen/Xing variant).

    Same formula as coskewness but with daily data and 252-day window.
    Sign: NEGATIVE coskew predicts HIGHER returns (reversed for ranking).

    Returns (date × ticker) cross-sectionally ranked [0, 1].
    """
    rets = returns.replace([np.inf, -np.inf], np.nan)
    mkt = market_returns.replace([np.inf, -np.inf], np.nan)

    # Align market series to returns index
    mkt = mkt.reindex(rets.index)

    mkt_demean = mkt - mkt.rolling(lookback_days, min_periods=min_days).mean()
    mkt_sq_demean = mkt_demean ** 2
    mkt_var = mkt.rolling(lookback_days, min_periods=min_days).var()

    coskew_panel = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=np.float32)

    for ticker in rets.columns:
        r_i = rets[ticker]
        r_i_demean = r_i - r_i.rolling(lookback_days, min_periods=min_days).mean()
        std_i = r_i.rolling(lookback_days, min_periods=min_days).std()
        prod = r_i_demean * mkt_sq_demean
        e_prod = prod.rolling(lookback_days, min_periods=min_days).mean()
        denom = std_i * mkt_var
        coskew = e_prod / denom.replace(0, np.nan)
        coskew_panel[ticker] = coskew.clip(-3, 3).astype(np.float32)

    # Negate
    coskew_panel = -coskew_panel

    return _cs_rank(coskew_panel)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MomSeason06YrPlus (Heston & Sadka 2008) — IC_IR 0.15
# ─────────────────────────────────────────────────────────────────────────────

def build_mom_season_signal(
    returns: pd.DataFrame,
    lookback_years_min: int = 6,
    lookback_years_max: int = 10,
) -> pd.DataFrame:
    """Same-month seasonal momentum (Heston & Sadka 2008).

    For each (date t, ticker), signal = average return in the same calendar
    month over the preceding `lookback_years_min` to `lookback_years_max` years.

    E.g. for April 2026, average April returns from April 2016-April 2020 (years 6-10).

    Sign: positive (higher historical same-month return predicts higher returns).

    Returns (date × ticker) cross-sectionally ranked [0, 1].
    """
    rets = returns.replace([np.inf, -np.inf], np.nan)

    # Convert to monthly returns
    monthly = (1.0 + rets).resample("ME").prod() - 1.0

    # For each month-end date, find returns from same calendar month
    # in years t-6 through t-10 (or whatever range)
    signal_monthly = pd.DataFrame(
        index=monthly.index, columns=monthly.columns, dtype=np.float32
    )

    for date in monthly.index:
        # Look back lookback_years_min..lookback_years_max years ago, same month
        target_dates = []
        for years_back in range(lookback_years_min, lookback_years_max + 1):
            target = date - pd.DateOffset(years=years_back)
            # Snap to month-end
            target_me = target + pd.offsets.MonthEnd(0) if target.day != target.days_in_month \
                        else target
            if target_me in monthly.index:
                target_dates.append(target_me)

        if len(target_dates) >= 2:
            # Average returns across those months
            avg_returns = monthly.loc[target_dates].mean(axis=0)
            signal_monthly.loc[date] = avg_returns.values.astype(np.float32)

    # Forward-fill to daily
    daily = signal_monthly.reindex(rets.index, method="ffill", limit=25)

    return _cs_rank(daily)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Accounting-based C&Z signals (need EDGAR extra fields)
# ─────────────────────────────────────────────────────────────────────────────

_FILING_LAG = pd.Timedelta(days=45)  # 10-K filing lag (matches alt_features.py)


def _raw_panel_from_edgar(
    edgar_df: pd.DataFrame,
    field: str,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """Extract a (date × ticker) panel from MultiIndex edgar_df with filing lag."""
    if edgar_df is None or edgar_df.empty:
        return None
    frames = {}
    for ticker in tickers:
        if (ticker, field) in edgar_df.columns:
            raw = edgar_df[(ticker, field)].dropna()
            if raw.empty:
                continue
            lagged = raw.copy()
            lagged.index = lagged.index + _FILING_LAG
            frames[ticker] = lagged.reindex(date_index, method="ffill")
    if not frames:
        return None
    return pd.DataFrame(frames, index=date_index).reindex(columns=tickers)


def build_payout_yield_signal(
    edgar_extra: pd.DataFrame,
    market_cap: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """PayoutYield = (dividends + buybacks) / market_cap_lagged_6mo.
    Boudoukh et al. (2007). Sign: +1.
    """
    div_p = _raw_panel_from_edgar(edgar_extra, "raw_dividends_paid", tickers, date_index)
    bb_p = _raw_panel_from_edgar(edgar_extra, "raw_buybacks", tickers, date_index)
    if div_p is None and bb_p is None:
        return None
    div_p = div_p.fillna(0) if div_p is not None else 0
    bb_p = bb_p.fillna(0) if bb_p is not None else 0
    payout = div_p + bb_p
    mc_lag = market_cap.shift(126).replace(0, np.nan)  # ~6 months lag
    yield_ratio = (payout / mc_lag).clip(-1, 5)
    return _cs_rank(yield_ratio)


def build_net_payout_yield_signal(
    edgar_extra: pd.DataFrame,
    market_cap: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """NetPayoutYield = (dividends + buybacks - issuance) / market_cap_lagged_6mo.
    Boudoukh et al. (2007). Sign: +1.
    """
    div_p = _raw_panel_from_edgar(edgar_extra, "raw_dividends_paid", tickers, date_index)
    bb_p = _raw_panel_from_edgar(edgar_extra, "raw_buybacks", tickers, date_index)
    iss_p = _raw_panel_from_edgar(edgar_extra, "raw_stock_issuance", tickers, date_index)
    if div_p is None and bb_p is None and iss_p is None:
        return None
    div_p = div_p.fillna(0) if div_p is not None else 0
    bb_p = bb_p.fillna(0) if bb_p is not None else 0
    iss_p = iss_p.fillna(0) if iss_p is not None else 0
    net_payout = div_p + bb_p - iss_p
    mc_lag = market_cap.shift(126).replace(0, np.nan)
    yield_ratio = (net_payout / mc_lag).clip(-1, 5)
    return _cs_rank(yield_ratio)


def build_xfin_signal(
    edgar_extra: pd.DataFrame,
    assets: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """XFIN = (sstk - dvc - prstkc + dltis - dltr) / total_assets.
    Bradshaw, Richardson, Sloan (2006). Sign: -1 (high external financing predicts LOW returns).
    """
    iss_p = _raw_panel_from_edgar(edgar_extra, "raw_stock_issuance", tickers, date_index)
    div_p = _raw_panel_from_edgar(edgar_extra, "raw_dividends_paid", tickers, date_index)
    bb_p = _raw_panel_from_edgar(edgar_extra, "raw_buybacks", tickers, date_index)
    debt_iss = _raw_panel_from_edgar(edgar_extra, "raw_lt_debt_issuance", tickers, date_index)
    debt_repay = _raw_panel_from_edgar(edgar_extra, "raw_lt_debt_repayment", tickers, date_index)
    if all(p is None for p in [iss_p, div_p, bb_p, debt_iss, debt_repay]):
        return None
    iss_p = iss_p.fillna(0) if iss_p is not None else 0
    div_p = div_p.fillna(0) if div_p is not None else 0
    bb_p = bb_p.fillna(0) if bb_p is not None else 0
    debt_iss = debt_iss.fillna(0) if debt_iss is not None else 0
    debt_repay = debt_repay.fillna(0) if debt_repay is not None else 0
    xfin_num = iss_p - div_p - bb_p + debt_iss - debt_repay
    assets_safe = assets.replace(0, np.nan) if assets is not None else None
    if assets_safe is None:
        return None
    xfin = (xfin_num / assets_safe).clip(-2, 2)
    # Negate for ranking (high XFIN predicts LOW returns per C&Z sign=-1)
    return _cs_rank(-xfin)


def build_cfp_signal(
    operating_cf: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """cfp = Operating Cash Flow / Market Cap.
    Desai, Rajgopal, Venkatachalam (2004). Sign: +1.
    """
    if operating_cf is None or market_cap is None:
        return None
    mc_safe = market_cap.replace(0, np.nan)
    cfp = (operating_cf / mc_safe).clip(-5, 5)
    return _cs_rank(cfp)


def build_operprof_rd_signal(
    revenue: pd.DataFrame,
    gross_profit: pd.DataFrame,
    sga: pd.DataFrame,
    edgar_extra: pd.DataFrame,
    assets: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """OperProfRD = (Revenue - COGS - (SGA - R&D)) / Assets.
    Ball et al. (2016): R&D-adjusted operating profitability.
    Sign: +1.
    """
    if revenue is None or gross_profit is None or assets is None:
        return None
    rd_p = _raw_panel_from_edgar(edgar_extra, "raw_rd", tickers, date_index)
    rd_p = rd_p.fillna(0) if rd_p is not None else 0
    sga_term = sga.fillna(0) if sga is not None else 0
    cogs = revenue - gross_profit
    op_profit = revenue - cogs - (sga_term - rd_p)
    op_rd_ratio = (op_profit / assets.replace(0, np.nan)).clip(-2, 5)
    return _cs_rank(op_rd_ratio)


def build_tax_signal(
    edgar_extra: pd.DataFrame,
    net_income: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    tax_rate: float = 0.21,  # Post-TCJA federal corporate rate
) -> Optional[pd.DataFrame]:
    """Tax = taxes_paid / (statutory_tax_rate * net_income).
    Lev & Nissim (2004). Sign: +1.
    Captures information in tax expense beyond what's in book income.
    """
    tax_p = _raw_panel_from_edgar(edgar_extra, "raw_taxes", tickers, date_index)
    if tax_p is None or net_income is None:
        return None
    expected_tax = tax_rate * net_income.replace(0, np.nan)
    tax_ratio = (tax_p / expected_tax).clip(-2, 5)
    return _cs_rank(tax_ratio)


def build_deldrc_signal(
    edgar_extra: pd.DataFrame,
    assets: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    """DelDRC = annual change in deferred revenue / avg(assets_t, assets_t-1).
    Prakash & Sinha (2013). Sign: +1.
    Increased deferred revenue = future revenue locked in = bullish.
    """
    drc_p = _raw_panel_from_edgar(edgar_extra, "raw_deferred_revenue", tickers, date_index)
    if drc_p is None or assets is None:
        return None
    drc_chg = drc_p.diff(252)  # YoY change
    avg_assets = (assets + assets.shift(252)) / 2
    avg_assets_safe = avg_assets.replace(0, np.nan)
    deldrc = (drc_chg / avg_assets_safe).clip(-1, 1)
    return _cs_rank(deldrc)


def build_cz_accounting_signals(
    edgar_extra: pd.DataFrame,
    edgar_main: pd.DataFrame,
    tickers: pd.Index,
    date_index: pd.DatetimeIndex,
    close: Optional[pd.DataFrame] = None,
    enable: Optional[Dict[str, bool]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build C&Z accounting signals from EDGAR extra fields.

    Requires:
        edgar_extra : DataFrame from load_edgar_fundamentals_extra()
        edgar_main : DataFrame from load_edgar_fundamentals() (for raw_assets, etc.)
        close : (date × ticker) close prices (for market cap)
    """
    if enable is None:
        enable = {
            "payout_yield": True, "net_payout_yield": True, "xfin": True,
            "cfp": True, "operprof_rd": True, "tax": True, "deldrc": True,
        }

    signals: Dict[str, pd.DataFrame] = {}

    # Build raw panels from MAIN edgar (assets, equity, net_income, etc.)
    assets_p = _raw_panel_from_edgar(edgar_main, "raw_assets", tickers, date_index)
    netinc_p = _raw_panel_from_edgar(edgar_main, "raw_net_income", tickers, date_index)
    revenue_p = _raw_panel_from_edgar(edgar_main, "raw_revenue", tickers, date_index)
    gprofit_p = _raw_panel_from_edgar(edgar_main, "raw_gross_profit", tickers, date_index)
    sga_p = _raw_panel_from_edgar(edgar_main, "raw_sga", tickers, date_index)
    ocf_p = _raw_panel_from_edgar(edgar_main, "raw_operating_cf", tickers, date_index)
    shares_p = _raw_panel_from_edgar(edgar_main, "raw_shares_out", tickers, date_index)

    # Market cap = Close × Shares Outstanding
    market_cap = None
    if close is not None and shares_p is not None:
        close_aligned = close.reindex(index=date_index, columns=tickers, method="ffill")
        market_cap = close_aligned * shares_p

    if enable.get("payout_yield", False) and market_cap is not None:
        sig = build_payout_yield_signal(edgar_extra, market_cap, tickers, date_index)
        if sig is not None:
            signals["payout_yield_signal"] = sig
            print(f"      [CZ] payout_yield_signal built")

    if enable.get("net_payout_yield", False) and market_cap is not None:
        sig = build_net_payout_yield_signal(edgar_extra, market_cap, tickers, date_index)
        if sig is not None:
            signals["net_payout_yield_signal"] = sig
            print(f"      [CZ] net_payout_yield_signal built")

    if enable.get("xfin", False) and assets_p is not None:
        sig = build_xfin_signal(edgar_extra, assets_p, tickers, date_index)
        if sig is not None:
            signals["xfin_signal"] = sig
            print(f"      [CZ] xfin_signal built")

    if enable.get("cfp", False) and ocf_p is not None and market_cap is not None:
        sig = build_cfp_signal(ocf_p, market_cap)
        if sig is not None:
            signals["cfp_signal"] = sig
            print(f"      [CZ] cfp_signal built")

    if enable.get("operprof_rd", False) and assets_p is not None:
        sig = build_operprof_rd_signal(revenue_p, gprofit_p, sga_p, edgar_extra,
                                        assets_p, tickers, date_index)
        if sig is not None:
            signals["operprof_rd_signal"] = sig
            print(f"      [CZ] operprof_rd_signal built")

    if enable.get("tax", False) and netinc_p is not None:
        sig = build_tax_signal(edgar_extra, netinc_p, tickers, date_index)
        if sig is not None:
            signals["tax_signal"] = sig
            print(f"      [CZ] tax_signal built")

    if enable.get("deldrc", False) and assets_p is not None:
        sig = build_deldrc_signal(edgar_extra, assets_p, tickers, date_index)
        if sig is not None:
            signals["deldrc_signal"] = sig
            print(f"      [CZ] deldrc_signal built")

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Master builder: all C&Z signals
# ─────────────────────────────────────────────────────────────────────────────

def build_cz_price_signals(
    returns: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    enable: Optional[Dict[str, bool]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build all C&Z price-only signals.

    Parameters
    ----------
    returns : (date × ticker) daily returns
    market_returns : Series of market index daily returns (e.g. SPY).
                     Required for coskewness signals. If None, those are skipped.
    enable : dict like {"coskewness": True, "coskew_acx": False, ...}
             None = enable all.

    Returns
    -------
    Dict of signal_name → (date × ticker) ranked [0, 1] DataFrame.
    """
    if enable is None:
        enable = {"coskewness": True, "coskew_acx": True, "mom_season": True}

    signals: Dict[str, pd.DataFrame] = {}

    if enable.get("coskewness", False) and market_returns is not None:
        try:
            signals["coskewness_signal"] = build_coskewness_signal(returns, market_returns)
            print(f"      [CZ] coskewness_signal built ({signals['coskewness_signal'].shape})")
        except Exception as e:
            warnings.warn(f"coskewness_signal failed: {e}")

    if enable.get("coskew_acx", False) and market_returns is not None:
        try:
            signals["coskew_acx_signal"] = build_coskew_acx_signal(returns, market_returns)
            print(f"      [CZ] coskew_acx_signal built ({signals['coskew_acx_signal'].shape})")
        except Exception as e:
            warnings.warn(f"coskew_acx_signal failed: {e}")

    if enable.get("mom_season", False):
        try:
            signals["mom_season_signal"] = build_mom_season_signal(returns)
            print(f"      [CZ] mom_season_signal built ({signals['mom_season_signal'].shape})")
        except Exception as e:
            warnings.warn(f"mom_season_signal failed: {e}")

    return signals
