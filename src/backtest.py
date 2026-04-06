"""
backtest.py
-----------
Event-driven backtesting engine.

Execution model (T+1):
  - Signal computed at close of day T using data up to and including T
  - Order submitted at close of day T
  - Order FILLS at open of day T+1
  - P&L accrues at close of T+1 relative to fill price

Transaction cost model:
  - Spread:      half the bid-ask spread (paid on both entry and exit)
  - Slippage:    market impact as % of ADV (scales with trade size)
  - Commission:  flat per-share or bps of notional

All costs reduce net_pnl and are tracked separately for analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Transaction Cost Assumptions (2024-2026 retail, $100K-$500K account):
#
# Commission: $0 at Schwab/Fidelity/IBKR Lite (zero-commission era since 2019)
# Spread: 1-3 bps half-spread for large/mid-cap US equities after price improvement
#   (SEC Rule 606 data shows retail gets 1-2 bps price improvement from wholesalers)
# Slippage: <1 bps at $5K-$25K trade sizes (square-root model: sigma*sqrt(Q/V) ~ 0.2 bps
#   for $5K trade vs $10M ADV; Frazzini-Israel-Moskowitz 2018 confirms)
#
# Total: ~3 bps/side (6 bps round-trip) for large/mid-cap
# Conservative stress test: 5-10 bps/side for small-cap or illiquid names
#
# References:
#   Schwarz et al. (2025) "The Actual Retail Price of Equity Trades" JF
#   Frazzini, Israel & Moskowitz (2018) "Trading Costs" SSRN
#   Novy-Marx & Velikov (2016) "A Taxonomy of Anomalies and Their Trading Costs"


@dataclass
class TransactionCostModel:
    """
    Realistic all-in transaction cost model for US equities.
    Defaults calibrated to 2024-2026 retail execution quality.
    """
    spread_bps:     float = 2.0   # was 3.0 -- half-spread for large/mid-cap after price improvement
    commission_bps: float = 0.0   # was 1.0 -- $0 commission era (Schwab/Fidelity/IBKR Lite)
    slippage_bps:   float = 1.0   # was 5.0 -- negligible impact at $5K-$25K retail trades
    # Total: 3 bps per side (was 9). Based on Schwarz et al. (2025 JF)
    # and Frazzini-Israel-Moskowitz (2018) empirical retail cost estimates.

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.commission_bps + self.slippage_bps

    def compute_cost(self, trade_notional: float) -> float:
        """Cost in dollars for a trade of given notional value (flat model)."""
        return abs(trade_notional) * self.total_bps / 10_000

    def compute_cost_with_liquidity(self, trade_notional: float, adv: float) -> float:
        """Cost in dollars, with spread scaled by stock liquidity.

        Micro-cap stocks ($1M ADV) get ~20bps spread.
        Large-cap stocks ($100M+ ADV) get ~1bps spread.
        Formula: spread_bps = base_spread_bps * 5 / sqrt(ADV_in_millions)
        Clipped to [1, 50] bps.
        """
        if adv > 0:
            adv_millions = adv / 1e6
            scaled_spread = self.spread_bps * 5.0 / max(adv_millions ** 0.5, 0.1)
            scaled_spread = min(max(scaled_spread, 1.0), 50.0)
        else:
            scaled_spread = self.spread_bps
        total = scaled_spread + self.commission_bps + self.slippage_bps
        return abs(trade_notional) * total / 10_000

    def compute_tiered_cost(self, trade_notional: float, adv: float) -> float:
        """Compute transaction cost with ADV-based tiering.

        Tiers based on empirical bid-ask spreads by liquidity (Novy-Marx & Velikov 2016,
        Frazzini-Israel-Moskowitz 2018, Schwarz et al. 2025):

            ADV > $20M:   2 bps/side  (mega/large-cap, tight spreads)
            ADV $5-20M:   5 bps/side  (mid-cap)
            ADV $1-5M:   10 bps/side  (small-cap, wider spreads)
            ADV < $1M:   25 bps/side  (micro-cap, should generally be excluded)

        Returns dollar cost for one side of the trade.
        """
        if adv <= 0 or pd.isna(adv):
            bps = 10.0  # conservative fallback
        elif adv >= 20_000_000:
            bps = 2.0
        elif adv >= 5_000_000:
            bps = 5.0
        elif adv >= 1_000_000:
            bps = 10.0
        else:
            bps = 25.0

        return abs(trade_notional) * bps / 10_000


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    equity_curve:     pd.Series       = field(default_factory=pd.Series)
    daily_returns:    pd.Series       = field(default_factory=pd.Series)
    positions:        pd.DataFrame    = field(default_factory=pd.DataFrame)
    weights_history:  pd.DataFrame    = field(default_factory=pd.DataFrame)
    gross_pnl:        pd.Series       = field(default_factory=pd.Series)
    transaction_costs: pd.Series      = field(default_factory=pd.Series)
    net_pnl:          pd.Series       = field(default_factory=pd.Series)
    turnover:         pd.Series       = field(default_factory=pd.Series)
    metadata:         dict            = field(default_factory=dict)


def run_backtest(
    weights:         pd.DataFrame,
    prices:          pd.DataFrame,
    initial_capital: float = 10_000_000,
    cost_model:      Optional[TransactionCostModel] = None,
    rebalance_dates: Optional[set] = None,
    adv:             Optional[pd.DataFrame] = None,
    use_tiered_costs: bool = True,
    stop_loss_pct:   float = 0.0,
    drawdown_halt_pct: float = 0.0,  # disabled by default — whipsaws in practice
    monthly_loss_limit: float = 0.0,
    risk_free_series: Optional[pd.Series] = None,
    verbose:         bool = True,
) -> BacktestResult:
    """
    Core backtest loop with optional buy-and-hold between rebalance dates.

    Parameters
    ----------
    weights         : (date x ticker) target portfolio weights. Must be
                      forward-filled to daily frequency internally — sparse
                      rebalance-only frames are reindexed at the top of this
                      function so weights.loc[date] is always defined.
    prices          : full price DataFrame from data_loader.py
    initial_capital : starting NAV in dollars
    cost_model      : TransactionCostModel instance (uses defaults if None)
    rebalance_dates : if provided, only rebalance on these dates.
                      Between rebalances, positions are held (drift naturally).
                      If None, rebalances every day (original behavior).
    adv             : (date x ticker) rolling average daily dollar volume.
                      If provided, uses liquidity-scaled transaction costs.
    use_tiered_costs : if True (default), use ADV-tiered cost model per trade.
                      Computes ADV from prices if adv DataFrame not provided.
                      If False, uses flat cost model (backward compatible).
    stop_loss_pct   : if > 0, sell positions that drop this much from their
                      entry price. E.g. 0.15 = sell if down 15% from purchase.
                      Only active between rebalance dates.
    monthly_loss_limit : if > 0, liquidate to cash for the rest of the
                      calendar month after losing this fraction of
                      month-start NAV. DISABLED by default (0.0) — V4 showed
                      an 8% limit killed recovery trades. Pass 0.08 (or any
                      positive value) explicitly to re-enable.
    risk_free_series : optional daily series of annualized T-bill yields
                      (e.g. FRED DGS3MO expressed as a decimal, 0.04 = 4%).
                      If provided, uninvested cash accrues rf_annual/252
                      each trading day. Missing dates are forward-filled
                      from the last valid observation. If None, cash earns
                      0% (preserves original behavior). Over a 13-year
                      backtest a 4% T-bill on a 15% cash sleeve is ~60bps
                      of annual drag, so supplying this is recommended.
    verbose         : print progress

    Returns
    -------
    BacktestResult with full history
    """
    if cost_model is None:
        cost_model = TransactionCostModel()

    close  = prices["Close"]
    open_  = prices["Open"]

    # Compute ADV from prices if tiered costs are requested but no ADV provided
    if use_tiered_costs and adv is None:
        adv = (prices["Close"] * prices["Volume"]).rolling(21).mean()

    # Align weights to dates available in price data.
    # IMPORTANT: weights must be daily-broadcast (forward-filled across all
    # trading days), NOT sparse on rebalance rows only. The turnover loop
    # below does weights.loc[prev_date], which fails if prev_date isn't in
    # the index. Reindexing with ffill makes weights.loc[date] always valid
    # and represents the standing target between rebalances.
    common_dates = close.index[
        (close.index >= weights.index.min()) & (close.index <= weights.index.max())
    ]
    weights = weights.reindex(common_dates, method="ffill").fillna(0.0)

    # Prepare risk-free rate series (daily annualized yields, forward-filled).
    # rf_daily = rf_annual / 252 applied each day to the cash balance.
    if risk_free_series is not None:
        rf_series = risk_free_series.reindex(common_dates, method="ffill").fillna(0.0)
    else:
        rf_series = pd.Series(0.0, index=common_dates)

    # -----------------------------------------------------------------------
    # State variables
    # -----------------------------------------------------------------------
    _last_known_price = {}  # {ticker: last valid close} for delisting exit
    nav          = initial_capital    # current portfolio value
    cash         = initial_capital    # uninvested cash
    positions    = {}                 # {ticker: shares_held}

    equity_curve      = []
    daily_returns_lst = []
    gross_pnl_lst     = []
    tc_lst            = []
    net_pnl_lst       = []
    turnover_lst      = []
    dates_lst         = []
    weights_hist      = []
    positions_hist    = []

    prev_weights = pd.Series(0.0, index=weights.columns)

    # Stop-loss tracking: entry prices for each position
    entry_prices = {}  # {ticker: fill_price}
    peak_nav = initial_capital  # for portfolio-level drawdown halt
    in_drawdown_halt = False    # flag: halved exposure due to drawdown

    # Monthly loss budget: if portfolio drops > monthly_loss_limit within
    # a calendar month, go to cash for the rest of that month.
    month_start_nav = initial_capital
    current_month = None
    in_monthly_halt = False

    if verbose:
        print(f"[backtest] Starting NAV: ${initial_capital:,.0f}")
        print(f"[backtest] Universe: {len(weights.columns)} tickers")
        print(f"[backtest] Period: {common_dates[0].date()} → {common_dates[-1].date()}")
        print(f"[backtest] Cost model: {cost_model.total_bps:.1f} bps total")
        if stop_loss_pct > 0:
            print(f"[backtest] Stop-loss: {stop_loss_pct*100:.0f}%")
        print("-" * 60)

    # -----------------------------------------------------------------------
    # Main loop: iterate over each trading day
    # -----------------------------------------------------------------------
    # Track whether we need to rebalance. If rebalance_dates is None,
    # rebalance every day (original behavior). If provided, only trade
    # on those dates — between rebalances, positions drift naturally.
    pending_rebalance = False

    for i, date in enumerate(common_dates):

        # Skip first day — no prior weights, no fills to process
        if i == 0:
            equity_curve.append(nav)
            daily_returns_lst.append(0.0)
            gross_pnl_lst.append(0.0)
            tc_lst.append(0.0)
            net_pnl_lst.append(0.0)
            turnover_lst.append(0.0)
            dates_lst.append(date)
            weights_hist.append(prev_weights.to_dict())
            positions_hist.append({})
            prev_weights = weights.loc[date].fillna(0.0)
            pending_rebalance = True
            continue

        prev_date = common_dates[i - 1]

        # ---- Accrue T-bill interest on uninvested cash --------------------
        # Cash balance grows at rf_annual / 252 per trading day. If no
        # risk_free_series was supplied, rf_series is 0 everywhere and this
        # is a no-op (preserves legacy behavior).
        rf_annual = rf_series.loc[date]
        if cash > 0 and rf_annual > 0:
            cash *= (1.0 + rf_annual / 252.0)

        # ---- Monthly loss budget check ------------------------------------
        date_month = (date.year, date.month)
        if current_month != date_month:
            # New month — reset
            current_month = date_month
            month_start_nav = nav
            in_monthly_halt = False

        if monthly_loss_limit > 0 and not in_monthly_halt:
            month_return = (nav / month_start_nav) - 1 if month_start_nav > 0 else 0
            if month_return <= -monthly_loss_limit:
                # Sell everything and sit in cash for rest of month
                in_monthly_halt = True
                today_open_ml = open_.loc[date] if date in open_.index else close.loc[date]
                for ticker in list(positions.keys()):
                    if positions[ticker] != 0:
                        sell_price = today_open_ml.get(ticker, 0)
                        if pd.notna(sell_price) and sell_price > 0:
                            trade_val = -positions[ticker] * sell_price
                            if use_tiered_costs and adv is not None and ticker in adv.columns and date in adv.index:
                                ticker_adv = adv.loc[date, ticker]
                                tc = cost_model.compute_tiered_cost(
                                    trade_val, ticker_adv if pd.notna(ticker_adv) else 0
                                )
                            else:
                                tc = cost_model.compute_cost(trade_val)
                            cash -= (trade_val + tc)
                        positions[ticker] = 0.0
                entry_prices.clear()

        # -------------------------------------------------------------------
        # Step 1: Execute orders at today's open (T+1 fill)
        # -------------------------------------------------------------------
        # Decide whether to trade today:
        #   - If rebalance_dates is None: trade every day (original behavior)
        #   - If rebalance_dates is set: only trade when pending from a rebalance date
        should_trade = pending_rebalance if rebalance_dates is not None else True
        pending_rebalance = False

        # Override: don't trade if in monthly halt
        if in_monthly_halt:
            should_trade = False

        gross_cost = 0.0
        today_open = open_.loc[date] if date in open_.index else close.loc[date]

        # ---- Force-close delisted positions + stop-loss check ---------------
        for ticker, shares in list(positions.items()):
            if shares <= 0:
                continue
            current_price = today_open.get(ticker, np.nan) if ticker in today_open.index else np.nan

            # DELISTED: price is NaN → exit at last known price (not $0).
            # ~70% of delistings are acquisitions where shareholders get paid.
            # Exiting at last price ≈ 0% delisting return (close to reality).
            # Exiting at $0 would be -100% for every delisting (too harsh).
            if pd.isna(current_price) or current_price <= 0:
                exit_price = _last_known_price.get(ticker, 0)
                if exit_price > 0 and shares > 0:
                    # Credit cash at last known price (as if sold before delisting)
                    trade_value = -shares * exit_price
                    if use_tiered_costs and adv is not None and ticker in adv.columns and date in adv.index:
                        ticker_adv = adv.loc[date, ticker]
                        tc = cost_model.compute_tiered_cost(
                            trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                        )
                    else:
                        tc = cost_model.compute_cost(trade_value)
                    cash -= (trade_value + tc)
                positions[ticker] = 0.0
                if ticker in entry_prices:
                    del entry_prices[ticker]
                _last_known_price.pop(ticker, None)
                continue

            # STOP-LOSS: price dropped >stop_loss_pct from entry
            if stop_loss_pct > 0 and ticker in entry_prices:
                entry_px = entry_prices[ticker]
                if entry_px > 0 and (current_price / entry_px - 1) <= -stop_loss_pct:
                    trade_value = -shares * current_price
                    if use_tiered_costs and adv is not None and ticker in adv.columns and date in adv.index:
                        ticker_adv = adv.loc[date, ticker]
                        tc = cost_model.compute_tiered_cost(
                            trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                        )
                    elif adv is not None and ticker in adv.columns and date in adv.index:
                        ticker_adv = adv.loc[date, ticker]
                        tc = cost_model.compute_cost_with_liquidity(
                            trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                        )
                    else:
                        tc = cost_model.compute_cost(trade_value)
                    gross_cost += tc
                    cash -= (trade_value + tc)
                    positions[ticker] = 0.0
                    del entry_prices[ticker]

        # ---- Scheduled rebalance trades -----------------------------------
        if should_trade:
            target_weights = prev_weights
            target_dollars = target_weights * nav

            for ticker in weights.columns:
                target_shares = 0.0
                if ticker in target_dollars.index and today_open.get(ticker, 0) > 0:
                    price = today_open[ticker]
                    if pd.notna(price) and price > 0:
                        target_shares = target_dollars.get(ticker, 0.0) / price

                current_shares = positions.get(ticker, 0.0)
                trade_shares   = target_shares - current_shares

                if abs(trade_shares) < 1e-6:
                    continue

                trade_value = trade_shares * today_open.get(ticker, 0)
                if use_tiered_costs and adv is not None and ticker in adv.columns and date in adv.index:
                    ticker_adv = adv.loc[date, ticker]
                    tc = cost_model.compute_tiered_cost(
                        trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                    )
                elif adv is not None and ticker in adv.columns and date in adv.index:
                    ticker_adv = adv.loc[date, ticker]
                    tc = cost_model.compute_cost_with_liquidity(
                        trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                    )
                else:
                    tc = cost_model.compute_cost(trade_value)
                gross_cost += tc
                cash -= (trade_value + tc)
                positions[ticker] = target_shares

                # Track entry price for stop-loss
                if target_shares > 0:
                    fill_price = today_open.get(ticker, 0)
                    if pd.notna(fill_price) and fill_price > 0:
                        entry_prices[ticker] = fill_price
                elif target_shares == 0 and ticker in entry_prices:
                    del entry_prices[ticker]

        # -------------------------------------------------------------------
        # Step 2: Mark-to-market at today's close
        # -------------------------------------------------------------------
        portfolio_value = cash
        position_values = {}

        for ticker, shares in list(positions.items()):
            price = close.loc[date, ticker] if ticker in close.columns else np.nan
            if pd.notna(price) and price > 0:
                value = shares * price
                portfolio_value += value
                position_values[ticker] = value
                # Track last known price for delisting exit
                _last_known_price[ticker] = price
            elif shares > 0:
                # Already handled by force-close above — shouldn't reach here
                # But as safety net: use last known price
                lkp = _last_known_price.get(ticker, 0)
                if lkp > 0:
                    portfolio_value += shares * lkp
                    position_values[ticker] = shares * lkp
                positions[ticker] = 0.0
                if ticker in entry_prices:
                    del entry_prices[ticker]

        prev_nav  = nav
        nav       = portfolio_value
        gross_pnl = nav - prev_nav + gross_cost
        net_pnl   = nav - prev_nav
        daily_ret = (nav / prev_nav) - 1 if prev_nav > 0 else 0.0

        # Portfolio-level drawdown circuit breaker
        peak_nav = max(peak_nav, nav)
        current_dd = (nav / peak_nav) - 1 if peak_nav > 0 else 0.0

        if drawdown_halt_pct > 0 and current_dd <= -drawdown_halt_pct and not in_drawdown_halt:
            # Sell half of all positions
            in_drawdown_halt = True
            for ticker in list(positions.keys()):
                if positions[ticker] > 0:
                    sell_shares = positions[ticker] * 0.5
                    sell_price = close.loc[date, ticker] if ticker in close.columns else 0
                    if pd.notna(sell_price) and sell_price > 0:
                        trade_value = -sell_shares * sell_price
                        if use_tiered_costs and adv is not None and ticker in adv.columns and date in adv.index:
                            ticker_adv = adv.loc[date, ticker]
                            tc = cost_model.compute_tiered_cost(
                                trade_value, ticker_adv if pd.notna(ticker_adv) else 0
                            )
                        else:
                            tc = cost_model.compute_cost(trade_value)
                        gross_cost += tc
                        cash -= (trade_value + tc)
                        positions[ticker] -= sell_shares
            # Update nav after forced sells
            portfolio_value = cash
            for ticker, shares in positions.items():
                price = close.loc[date, ticker] if ticker in close.columns else 0
                if pd.notna(price) and price > 0:
                    portfolio_value += shares * price
            nav = portfolio_value
        elif current_dd > -drawdown_halt_pct * 0.5:
            # Reset halt when drawdown recovers to half the threshold
            in_drawdown_halt = False

        # -------------------------------------------------------------------
        # Step 3: Compute turnover
        # -------------------------------------------------------------------
        current_w = pd.Series(position_values, dtype=float)
        if nav > 0:
            current_w = current_w / nav
        current_w = current_w.reindex(weights.columns, fill_value=0.0)

        # Turnover = change in TARGET weights day-over-day (NOT drift-vs-target).
        # Between rebalances the target is constant, so turnover is zero on
        # non-trading days. At a rebalance, it jumps by |new_target - old_target|/2.
        # Summing this across the year gives the true annual fraction of the
        # portfolio that was traded, without inflating from intraday drift.
        # weights is daily-broadcast (ffilled at top of function), so
        # weights.loc[prev_date] and weights.loc[date] are both guaranteed
        # to exist. Between rebalances they are identical, so turnover is
        # zero on non-trading days — exactly what we want.
        today_target = weights.loc[date].fillna(0.0)
        prev_target  = weights.loc[prev_date].fillna(0.0)
        turnover = (today_target - prev_target).abs().sum() / 2.0

        # -------------------------------------------------------------------
        # Step 4: Record state
        # -------------------------------------------------------------------
        equity_curve.append(nav)
        daily_returns_lst.append(daily_ret)
        gross_pnl_lst.append(gross_pnl)
        tc_lst.append(gross_cost)
        net_pnl_lst.append(net_pnl)
        turnover_lst.append(turnover)
        dates_lst.append(date)
        weights_hist.append(current_w.to_dict())
        positions_hist.append(position_values.copy())

        # Check if today is a rebalance date — if so, update target weights
        # and flag that we need to trade tomorrow (T+1 execution)
        if rebalance_dates is not None:
            if date in rebalance_dates:
                prev_weights = weights.loc[date].fillna(0.0)
                pending_rebalance = True
            # Otherwise prev_weights stays stale — no trading tomorrow
        else:
            prev_weights = weights.loc[date].fillna(0.0)

        if verbose and i % 252 == 0:
            ytd_ret = (nav / initial_capital - 1) * 100
            print(f"  {date.date()}  NAV: ${nav:>12,.0f}  Cumulative: {ytd_ret:+.1f}%")

    # -----------------------------------------------------------------------
    # Package results
    # -----------------------------------------------------------------------
    idx = pd.DatetimeIndex(dates_lst)

    result = BacktestResult(
        equity_curve     = pd.Series(equity_curve,       index=idx, name="NAV"),
        daily_returns    = pd.Series(daily_returns_lst,  index=idx, name="Returns"),
        gross_pnl        = pd.Series(gross_pnl_lst,      index=idx, name="Gross PnL"),
        transaction_costs= pd.Series(tc_lst,             index=idx, name="Transaction Costs"),
        net_pnl          = pd.Series(net_pnl_lst,        index=idx, name="Net PnL"),
        turnover         = pd.Series(turnover_lst,       index=idx, name="Turnover"),
        positions        = pd.DataFrame(positions_hist,  index=idx),
        weights_history  = pd.DataFrame(weights_hist,    index=idx),
        metadata={
            "initial_capital": initial_capital,
            "cost_model_bps":  cost_model.total_bps,
            "n_tickers":       len(weights.columns),
            "start_date":      str(common_dates[0].date()),
            "end_date":        str(common_dates[-1].date()),
        },
    )

    final_ret = (result.equity_curve.iloc[-1] / initial_capital - 1) * 100
    total_tc  = result.transaction_costs.sum()
    if verbose:
        print("-" * 60)
        print(f"[backtest] Final NAV:           ${result.equity_curve.iloc[-1]:>12,.0f}")
        print(f"[backtest] Total return:         {final_ret:+.2f}%")
        print(f"[backtest] Total transaction costs: ${total_tc:>10,.0f}")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_prices, get_close, get_returns, get_volume
    from features    import build_composite_signal, realized_volatility
    from portfolio   import compute_target_weights

    prices  = load_prices(start="2020-01-01", end="2024-01-01")
    close   = get_close(prices)
    returns = get_returns(prices)
    volume  = get_volume(prices)

    composite, _ = build_composite_signal(close, returns, volume)
    rvol         = realized_volatility(returns, window=21)
    weights      = compute_target_weights(composite, rvol)

    result = run_backtest(weights, prices, initial_capital=10_000_000)
    print("\nEquity curve tail:")
    print(result.equity_curve.tail())
