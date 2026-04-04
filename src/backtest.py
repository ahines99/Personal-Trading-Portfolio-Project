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


@dataclass
class TransactionCostModel:
    """
    Realistic all-in transaction cost model for US equities.
    Defaults are conservative estimates for liquid large-cap stocks.
    """
    spread_bps:     float = 3.0   # half spread, paid on each side
    commission_bps: float = 1.0   # broker commission
    slippage_bps:   float = 5.0   # market impact / slippage

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.commission_bps + self.slippage_bps

    def compute_cost(self, trade_notional: float) -> float:
        """Cost in dollars for a trade of given notional value."""
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
    stop_loss_pct:   float = 0.0,
    drawdown_halt_pct: float = 0.0,  # disabled by default — whipsaws in practice
    monthly_loss_limit: float = 0.08,
    verbose:         bool = True,
) -> BacktestResult:
    """
    Core backtest loop with optional buy-and-hold between rebalance dates.

    Parameters
    ----------
    weights         : (date x ticker) target portfolio weights.
    prices          : full price DataFrame from data_loader.py
    initial_capital : starting NAV in dollars
    cost_model      : TransactionCostModel instance (uses defaults if None)
    rebalance_dates : if provided, only rebalance on these dates.
                      Between rebalances, positions are held (drift naturally).
                      If None, rebalances every day (original behavior).
    adv             : (date x ticker) rolling average daily dollar volume.
                      If provided, uses liquidity-scaled transaction costs.
    stop_loss_pct   : if > 0, sell positions that drop this much from their
                      entry price. E.g. 0.15 = sell if down 15% from purchase.
                      Only active between rebalance dates.
    verbose         : print progress

    Returns
    -------
    BacktestResult with full history
    """
    if cost_model is None:
        cost_model = TransactionCostModel()

    close  = prices["Close"]
    open_  = prices["Open"]

    # Align weights to dates available in price data
    common_dates = weights.index.intersection(close.index)
    weights = weights.loc[common_dates]

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
                    if adv is not None and ticker in adv.columns and date in adv.index:
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
                if adv is not None and ticker in adv.columns and date in adv.index:
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

        today_target = weights.loc[date].fillna(0.0)
        turnover = (today_target - current_w).abs().sum() / 2.0

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
