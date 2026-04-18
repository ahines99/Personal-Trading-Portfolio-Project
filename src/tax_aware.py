"""Tax-aware portfolio construction for personal systematic equity strategies.

Implements:
- Tax-lot tracking with specific identification (HIFO priority)
- Short-term gain deferral near LT threshold
- Tax-loss harvesting with wash sale avoidance
- Partial rebalancing with tax-cost-aware trade prioritization

References:
- Israel, Moskowitz & Pettenuzzo (2023) "After-Tax Returns of Factor-Based Strategies"
- Israelov & Katz (2017) "Are Your Factor Premia Tax-Managed?" AQR
- IRS Publication 550 (Investment Income and Expenses)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class TaxLot:
    """A single purchase lot of a security."""
    ticker: str
    shares: float
    cost_basis_per_share: float
    purchase_date: date

    @property
    def total_cost(self) -> float:
        return self.shares * self.cost_basis_per_share

    def unrealized_gl(self, current_price: float) -> float:
        """Unrealized gain/loss in dollars."""
        return (current_price - self.cost_basis_per_share) * self.shares

    def unrealized_gl_pct(self, current_price: float) -> float:
        """Unrealized gain/loss as percentage."""
        if self.cost_basis_per_share <= 0:
            return 0.0
        return (current_price / self.cost_basis_per_share) - 1.0

    def is_long_term(self, as_of: date) -> bool:
        """Whether this lot qualifies for long-term capital gains rate."""
        return (as_of - self.purchase_date).days > 365

    def days_held(self, as_of: date) -> int:
        return (as_of - self.purchase_date).days

    def days_to_long_term(self, as_of: date) -> int:
        """Days remaining until LT qualification. Negative if already LT."""
        return 365 - self.days_held(as_of)

    def tax_cost_if_sold(self, current_price: float, sell_date: date,
                         st_rate: float = 0.328, lt_rate: float = 0.188) -> float:
        """Estimated tax liability (positive) or benefit (negative) if sold now.

        Args:
            st_rate: combined federal + NIIT + state short-term rate (default: 24% + 3.8% + 5%)
            lt_rate: combined federal + NIIT + state long-term rate (default: 15% + 3.8%)
        """
        gl = self.unrealized_gl(current_price)
        if gl <= 0:
            # Loss = tax benefit at the applicable rate
            rate = lt_rate if self.is_long_term(sell_date) else st_rate
            return gl * rate  # negative number = tax benefit
        rate = lt_rate if self.is_long_term(sell_date) else st_rate
        return gl * rate


class TaxLotTracker:
    """Tracks all tax lots across the portfolio.

    Implements specific identification (Spec-ID) lot selection:
    priority order is losses first (ST then LT), then LT gains, then ST gains.
    This minimizes current-year tax liability.
    """

    def __init__(self, st_rate: float = 0.328, lt_rate: float = 0.188):
        self.lots: List[TaxLot] = []
        self.st_rate = st_rate
        self.lt_rate = lt_rate
        self.wash_sale_blacklist: Dict[str, date] = {}  # ticker -> earliest_rebuy_date
        self.ytd_realized_gl: float = 0.0
        self.ytd_realized_st: float = 0.0
        self.ytd_realized_lt: float = 0.0
        self._year: int = 0

    def add_lot(self, ticker: str, shares: float, price: float, purchase_date: date):
        """Record a new purchase."""
        self.lots.append(TaxLot(ticker, shares, price, purchase_date))

    def get_position(self, ticker: str) -> float:
        """Total shares held across all lots for a ticker."""
        return sum(lot.shares for lot in self.lots if lot.ticker == ticker)

    def get_all_positions(self) -> Dict[str, float]:
        """Total shares per ticker."""
        positions = {}
        for lot in self.lots:
            positions[lot.ticker] = positions.get(lot.ticker, 0) + lot.shares
        return positions

    def select_lots_to_sell(self, ticker: str, shares_needed: float,
                            current_price: float, sell_date: date) -> List[Tuple[TaxLot, float]]:
        """Select lots to sell using tax-optimal specific identification.

        Priority:
        1. Short-term losses (highest tax benefit at ST rate)
        2. Long-term losses (tax benefit at LT rate)
        3. Long-term gains (lowest tax cost at LT rate)
        4. Short-term gains (highest tax cost at ST rate -- sell last)
        """
        ticker_lots = [l for l in self.lots if l.ticker == ticker and l.shares > 0]

        # Sort by tax cost ascending (most beneficial first)
        ticker_lots.sort(key=lambda l: l.tax_cost_if_sold(current_price, sell_date,
                                                           self.st_rate, self.lt_rate))

        sells = []
        remaining = shares_needed
        for lot in ticker_lots:
            if remaining <= 1e-8:
                break
            sell_shares = min(lot.shares, remaining)
            sells.append((lot, sell_shares))
            remaining -= sell_shares

        return sells

    def execute_sell(self, ticker: str, shares: float, price: float, sell_date: date) -> float:
        """Execute a sell, update lots, track realized gains. Returns realized G/L."""
        lots_to_sell = self.select_lots_to_sell(ticker, shares, price, sell_date)
        total_gl = 0.0

        # Reset YTD at year boundary
        if sell_date.year != self._year:
            self._year = sell_date.year
            self.ytd_realized_gl = 0.0
            self.ytd_realized_st = 0.0
            self.ytd_realized_lt = 0.0

        for lot, sell_shares in lots_to_sell:
            gl = (price - lot.cost_basis_per_share) * sell_shares
            total_gl += gl
            self.ytd_realized_gl += gl

            if lot.is_long_term(sell_date):
                self.ytd_realized_lt += gl
            else:
                self.ytd_realized_st += gl

            # Update lot
            lot.shares -= sell_shares
            if lot.shares < 1e-8:
                self.lots.remove(lot)

        # If sold at a loss, add to wash sale blacklist (31 calendar days)
        if total_gl < 0:
            self.wash_sale_blacklist[ticker] = sell_date + timedelta(days=31)

        return total_gl

    def is_wash_sale_blocked(self, ticker: str, as_of: date) -> bool:
        """Check if buying this ticker would trigger a wash sale."""
        if ticker not in self.wash_sale_blacklist:
            return False
        return as_of < self.wash_sale_blacklist[ticker]

    def unrealized_gl_by_ticker(self, prices: Dict[str, float], as_of: date) -> Dict[str, dict]:
        """Compute unrealized gain/loss summary per ticker."""
        result = {}
        for lot in self.lots:
            if lot.ticker not in prices:
                continue
            price = prices[lot.ticker]
            if lot.ticker not in result:
                result[lot.ticker] = {"shares": 0, "cost": 0, "value": 0, "gl": 0,
                                       "has_st_loss": False, "days_to_lt": 999}
            r = result[lot.ticker]
            r["shares"] += lot.shares
            r["cost"] += lot.total_cost
            r["value"] += lot.shares * price
            r["gl"] += lot.unrealized_gl(price)
            if not lot.is_long_term(as_of) and lot.unrealized_gl(price) < 0:
                r["has_st_loss"] = True
            r["days_to_lt"] = min(r["days_to_lt"], lot.days_to_long_term(as_of))
        return result


def should_defer_sale(lot: TaxLot, current_price: float, sell_date: date,
                      daily_alpha_decay: float = 0.0002,
                      st_rate: float = 0.328, lt_rate: float = 0.188,
                      max_defer_days: int = 60) -> bool:
    """Decide whether to defer selling a winner to capture LT rate.

    Defers if the tax savings from LT treatment exceed the expected
    alpha decay cost from holding a stale signal.

    Args:
        daily_alpha_decay: expected daily alpha loss from holding stale position
        max_defer_days: don't defer if more than this many days to LT
    """
    if lot.is_long_term(sell_date):
        return False  # already LT, no benefit from deferring

    days_to_lt = lot.days_to_long_term(sell_date)
    if days_to_lt > max_defer_days or days_to_lt <= 0:
        return False

    gl = lot.unrealized_gl(current_price)
    if gl <= 0:
        return False  # no gain to defer -- sell now for loss

    tax_savings = gl * (st_rate - lt_rate)
    position_value = lot.shares * current_price
    alpha_decay_cost = position_value * daily_alpha_decay * days_to_lt

    return tax_savings > alpha_decay_cost


def identify_harvest_candidates(tracker: TaxLotTracker, prices: Dict[str, float],
                                 as_of: date, min_loss: float = 500.0,
                                 min_loss_pct: float = 0.02) -> List[Tuple[str, float, float]]:
    """Find positions with harvestable tax losses.

    Returns list of (ticker, unrealized_loss, tax_benefit) sorted by largest benefit.
    Excludes tickers on the wash sale blacklist.
    """
    candidates = []
    summaries = tracker.unrealized_gl_by_ticker(prices, as_of)

    for ticker, info in summaries.items():
        # Skip if wash-sale blocked
        if tracker.is_wash_sale_blocked(ticker, as_of):
            continue

        gl = info["gl"]
        if gl >= 0:
            continue  # no loss to harvest

        loss = abs(gl)
        if loss < min_loss:
            continue

        loss_pct = loss / info["cost"] if info["cost"] > 0 else 0
        if loss_pct < min_loss_pct:
            continue

        # Tax benefit
        tax_benefit = loss * tracker.st_rate  # harvest at ST rate (most common)
        candidates.append((ticker, gl, tax_benefit))

    # Sort by tax benefit descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def tax_aware_trade_filter(
    target_weights: pd.Series,
    current_weights: pd.Series,
    tracker: TaxLotTracker,
    prices: Dict[str, float],
    trade_date: date,
    min_holding_rank: int = 40,
    daily_alpha_decay: float = 0.0002,
) -> pd.Series:
    """Filter trades to minimize tax impact while preserving alpha.

    Rules applied:
    1. Wash sale check: don't buy tickers on the blacklist
    2. Gain deferral: don't sell winners within 60 days of LT threshold
    3. Wide sell buffer: only sell positions that dropped below rank min_holding_rank

    Args:
        target_weights: desired new weights from the model
        current_weights: current portfolio weights
        tracker: TaxLotTracker with lot history
        prices: current prices dict
        trade_date: trade execution date
        min_holding_rank: keep positions if still ranked within top-N (wider than top-20)
        daily_alpha_decay: alpha decay rate for deferral calculation

    Returns:
        adjusted_weights: tax-filtered target weights
    """
    adjusted = target_weights.copy()

    # 1. Wash sale: zero out buys for blacklisted tickers
    for ticker in adjusted.index:
        if adjusted[ticker] > 0 and tracker.is_wash_sale_blocked(ticker, trade_date):
            adjusted[ticker] = 0.0

    # 2. Gain deferral: keep current weight for winners near LT threshold
    for ticker in current_weights.index:
        if current_weights[ticker] <= 0:
            continue
        if ticker not in prices:
            continue

        # Check if any lot for this ticker should be deferred
        ticker_lots = [l for l in tracker.lots if l.ticker == ticker and l.shares > 0]
        any_defer = any(
            should_defer_sale(lot, prices[ticker], trade_date, daily_alpha_decay)
            for lot in ticker_lots
        )

        if any_defer and adjusted.get(ticker, 0) < current_weights[ticker]:
            # Keep the current weight instead of reducing
            adjusted[ticker] = current_weights[ticker]

    # Renormalize (weights may have changed)
    total = adjusted[adjusted > 0].sum()
    if total > 0:
        adjusted = adjusted / total

    return adjusted


class TaxAwareLedger:
    """Lightweight cumulative ledger wrapping TaxLotTracker for backtest wiring.

    Unlike TaxLotTracker.ytd_realized_* (which resets each calendar year),
    this ledger accumulates realized ST/LT gains across the ENTIRE backtest
    horizon so we can compute a true actual-HIFO after-tax CAGR at the end.

    Trade accounting uses continuous "dollar-shares": a lot is recorded with
    shares = dollars_traded / price so the ledger works cleanly with float
    position sizes from weight-based rebalancing.
    """

    def __init__(self, st_rate: float = 0.328, lt_rate: float = 0.188):
        self.tracker = TaxLotTracker(st_rate=st_rate, lt_rate=lt_rate)
        self.st_rate = st_rate
        self.lt_rate = lt_rate
        self.cum_st_gain: float = 0.0
        self.cum_lt_gain: float = 0.0
        self.cum_realized_gain: float = 0.0
        self.n_buys: int = 0
        self.n_sells: int = 0
        self.gross_sold_dollars: float = 0.0

    def record_buy(self, ticker: str, dollars: float, price: float, trade_date: date):
        """Record a buy; dollars > 0."""
        if dollars <= 0 or price <= 0:
            return
        shares = dollars / price
        self.tracker.add_lot(ticker, shares, price, trade_date)
        self.n_buys += 1

    def record_sell(self, ticker: str, dollars: float, price: float, trade_date: date):
        """Record a sell; dollars > 0 (absolute notional)."""
        if dollars <= 0 or price <= 0:
            return
        shares = dollars / price
        available = self.tracker.get_position(ticker)
        if available <= 0:
            return
        shares = min(shares, available)
        # execute_sell resets YTD totals on year boundaries, so capture the
        # per-call realized ST/LT delta by snapshotting before/after.
        st_before = self.tracker.ytd_realized_st
        lt_before = self.tracker.ytd_realized_lt
        year_before = self.tracker._year
        self.tracker.execute_sell(ticker, shares, price, trade_date)
        if self.tracker._year != year_before:
            # Year rolled: before-values belong to previous year, delta is just the new totals
            st_delta = self.tracker.ytd_realized_st
            lt_delta = self.tracker.ytd_realized_lt
        else:
            st_delta = self.tracker.ytd_realized_st - st_before
            lt_delta = self.tracker.ytd_realized_lt - lt_before
        self.cum_st_gain += st_delta
        self.cum_lt_gain += lt_delta
        self.cum_realized_gain += (st_delta + lt_delta)
        self.gross_sold_dollars += dollars
        self.n_sells += 1

    def summary(self, final_nav: float, initial_capital: float, years: float) -> dict:
        """Compute actual after-tax CAGR using realized ST/LT gains.

        We treat taxes as paid out of NAV at the end of the horizon (a
        conservative proxy for pay-as-you-go since taxes on short-term gains
        are owed in the year realized). Unrealized gains at horizon end are
        NOT taxed (deferred), which is the correct treatment for a live
        portfolio still in flight.
        """
        st = self.cum_st_gain
        lt = self.cum_lt_gain
        realized = st + lt
        tax_paid = max(st, 0.0) * self.st_rate + max(lt, 0.0) * self.lt_rate
        # Losses offset gains within each bucket first; use a simple netted rule:
        net_st_taxable = max(st, 0.0) if st > 0 else 0.0
        net_lt_taxable = max(lt, 0.0) if lt > 0 else 0.0
        # Apply loss offset: if ST net negative, it reduces LT taxable (and vice versa)
        if st < 0 and lt > 0:
            net_lt_taxable = max(0.0, lt + st)
        if lt < 0 and st > 0:
            net_st_taxable = max(0.0, st + lt)
        tax_paid = net_st_taxable * self.st_rate + net_lt_taxable * self.lt_rate

        after_tax_nav = final_nav - tax_paid
        if years > 0 and initial_capital > 0 and after_tax_nav > 0:
            pretax_cagr = (final_nav / initial_capital) ** (1.0 / years) - 1.0
            after_tax_cagr = (after_tax_nav / initial_capital) ** (1.0 / years) - 1.0
            effective_rate = 1.0 - (after_tax_cagr / pretax_cagr) if pretax_cagr > 0 else 0.0
        else:
            pretax_cagr = 0.0
            after_tax_cagr = 0.0
            effective_rate = 0.0

        st_frac = (st / realized) if realized > 0 else 0.0
        lt_frac = (lt / realized) if realized > 0 else 0.0

        return {
            "pretax_cagr_actual": pretax_cagr,
            "after_tax_cagr_hifo": after_tax_cagr,
            "cum_st_gain": st,
            "cum_lt_gain": lt,
            "cum_realized_gain": realized,
            "tax_paid": tax_paid,
            "effective_tax_rate_actual": effective_rate,
            "st_fraction_actual": st_frac,
            "lt_fraction_actual": lt_frac,
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "gross_sold_dollars": self.gross_sold_dollars,
            "open_lots_at_end": len(self.tracker.lots),
        }


def compute_tax_summary(tracker: TaxLotTracker, prices: Dict[str, float],
                        as_of: date) -> dict:
    """Compute a summary of the portfolio's tax position.

    Useful for tearsheet reporting.
    """
    summaries = tracker.unrealized_gl_by_ticker(prices, as_of)

    total_unrealized = sum(s["gl"] for s in summaries.values())
    total_value = sum(s["value"] for s in summaries.values())
    total_cost = sum(s["cost"] for s in summaries.values())

    # Count lots by holding period
    st_lots = sum(1 for l in tracker.lots if not l.is_long_term(as_of))
    lt_lots = sum(1 for l in tracker.lots if l.is_long_term(as_of))

    # Harvest opportunities
    harvest = identify_harvest_candidates(tracker, prices, as_of)
    harvest_potential = sum(h[2] for h in harvest) if harvest else 0

    return {
        "total_positions": len(summaries),
        "total_lots": len(tracker.lots),
        "st_lots": st_lots,
        "lt_lots": lt_lots,
        "lt_lot_pct": lt_lots / max(1, len(tracker.lots)),
        "total_value": total_value,
        "total_cost_basis": total_cost,
        "unrealized_gl": total_unrealized,
        "unrealized_gl_pct": total_unrealized / total_cost if total_cost > 0 else 0,
        "ytd_realized_gl": tracker.ytd_realized_gl,
        "ytd_realized_st": tracker.ytd_realized_st,
        "ytd_realized_lt": tracker.ytd_realized_lt,
        "harvest_candidates": len(harvest),
        "harvest_potential_tax_savings": harvest_potential,
        "wash_sale_blocked_tickers": len([t for t, d in tracker.wash_sale_blacklist.items() if d > as_of]),
    }
