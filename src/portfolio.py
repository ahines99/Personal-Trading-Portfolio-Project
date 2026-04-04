"""
portfolio.py
------------
Long-only concentrated portfolio construction for personal trading.

V4 design — balanced alpha capture:
  - Long-only with market beta (it's free return over decades).
  - Concentrated (20 positions) but VOL-BUCKET NEUTRAL: 4 stocks from each
    volatility quintile so the portfolio isn't just a leveraged small-cap bet.
  - Signal-weighted with moderate concentration (0.75 = between equal and linear).
  - Monthly rebalance with momentum pre-filter and IPO buffer.
  - Sector caps prevent single-sector blowups.
  - Correlation filter prevents holding 5 correlated semiconductor stocks.
  - Regime-aware: modestly reduce exposure in bear markets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core: build the monthly portfolio
# ---------------------------------------------------------------------------

def build_monthly_portfolio(
    signal:           pd.DataFrame,
    n_positions:      int   = 20,
    weighting:        str   = "signal",
    concentration:    float = 1.0,
    max_weight:       float = 0.10,
    min_weight:       float = 0.02,
    sector_map:       Optional[Dict[str, str]] = None,
    max_sector_pct:   float = 0.35,
    momentum_filter:  Optional[pd.DataFrame] = None,
    realized_vol:     Optional[pd.DataFrame] = None,
    returns:          Optional[pd.DataFrame] = None,
    regime:           Optional[pd.Series] = None,
    adv:              Optional[pd.DataFrame] = None,
    regime_n_map:     Optional[Dict[str, int]] = None,
    cash_in_bear:     float = 0.30,
    quality_filter:   Optional[pd.DataFrame] = None,
    earnings_dates:   Optional[pd.DataFrame] = None,
    spy_trend_filter: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """
    Build a long-only concentrated portfolio with monthly rebalancing.

    V4 features:
      - Vol-bucket neutral selection
      - Regime-aware sizing (fewer positions + cash in bear)
      - Adaptive position count by signal quality
      - Earnings exclusion zone (don't hold into imminent earnings if low-conviction)
      - SPY trend overlay (reduce exposure when SPY below 200d MA)
      - Optimal rebalance day (3rd-5th trading day, avoids turn-of-month noise)
      - IPO buffer, momentum filter, quality filter, correlation filter
    """
    if regime_n_map is None:
        regime_n_map = {
            "bull_calm":     n_positions,
            "bull_volatile": n_positions,
            "bear_calm":     max(12, n_positions - 5),
            "bear_volatile": max(10, n_positions - 8),
        }

    all_dates = signal.index
    # Optimal rebalance day: 3rd trading day of each month instead of 1st.
    # Avoids turn-of-month institutional flow noise.
    rebalance_dates = _get_optimal_rebalance_dates(all_dates, day_offset=2)

    first_valid_date = None
    if returns is not None:
        first_valid_date = returns.apply(pd.Series.first_valid_index)

    weights = pd.DataFrame(0.0, index=all_dates, columns=signal.columns)
    current_weights = pd.Series(0.0, index=signal.columns)

    for date in all_dates:
        if date in rebalance_dates:
            # ── Regime-aware position count ───────────────────────────────
            effective_n = n_positions
            cash_frac = 0.0
            if regime is not None:
                prev_dates = regime.index[regime.index < date]
                if len(prev_dates) > 0:
                    current_regime = regime.loc[prev_dates[-1]]
                    effective_n = regime_n_map.get(current_regime, n_positions)
                    if "bear" in str(current_regime):
                        cash_frac = cash_in_bear

            # ── Adaptive sizing by signal quality ─────────────────────────
            # When model has high conviction (large spread between top and
            # median scores), use more positions. When low conviction, fewer.
            scores_today = signal.loc[date].dropna()
            if len(scores_today) > 100:
                top_mean = scores_today.nlargest(50).mean()
                all_mean = scores_today.mean()
                dispersion = top_mean - all_mean
                if dispersion > 0.20:
                    effective_n = min(effective_n + 3, 25)
                elif dispersion < 0.08:
                    effective_n = max(effective_n - 3, 15)

            # ── SPY trend overlay ─────────────────────────────────────────
            # When SPY is below 200d MA, reduce exposure by 30%.
            if spy_trend_filter is not None and date in spy_trend_filter.index:
                if spy_trend_filter.loc[date] < 0:
                    cash_frac = min(cash_frac + 0.30, 0.70)

            new_weights = _select_and_weight(
                signal=signal,
                date=date,
                n_positions=effective_n,
                weighting=weighting,
                concentration=concentration,
                max_weight=max_weight,
                min_weight=min_weight,
                sector_map=sector_map,
                max_sector_pct=max_sector_pct,
                momentum_filter=momentum_filter,
                realized_vol=realized_vol,
                returns=returns,
                adv=adv,
                quality_filter=quality_filter,
                first_valid_date=first_valid_date,
                earnings_dates=earnings_dates,
            )

            if cash_frac > 0:
                new_weights = new_weights * (1 - cash_frac)

            current_weights = new_weights

        # ── Mid-month signal refresh ─────────────────────────────────────
        # On ~15th of each month, check if any held stock's signal decayed
        # badly. Swap at most 3 positions with much higher-ranked replacements.
        elif date.day >= 14 and date.day <= 16 and current_weights.sum() > 0.1:
            held = current_weights[current_weights > 0.005].index.tolist()
            if len(held) > 0 and date in signal.index:
                day_scores = signal.loc[date].dropna()
                if len(day_scores) > 50:
                    held_ranks = day_scores.reindex(held).rank(pct=True, ascending=False)
                    # Find held stocks that fell below top 5% (rank > 0.05)
                    bad_held = held_ranks[held_ranks > 0.05].sort_values(ascending=False)
                    # Find non-held stocks in top 0.5%
                    non_held = day_scores.drop(held, errors="ignore")
                    top_replacements = non_held.nlargest(5)
                    top_rep_rank = day_scores.rank(pct=True)
                    top_rep_rank = top_rep_rank.reindex(top_replacements.index)
                    great_replacements = top_rep_rank[top_rep_rank >= 0.995].index.tolist()

                    swaps = 0
                    for bad_ticker in bad_held.index:
                        if swaps >= 3 or len(great_replacements) == 0:
                            break
                        replacement = great_replacements.pop(0)
                        old_w = current_weights[bad_ticker]
                        current_weights[bad_ticker] = 0.0
                        current_weights[replacement] = old_w
                        swaps += 1

        weights.loc[date] = current_weights

    return weights, rebalance_dates


def _get_monthly_rebalance_dates(dates: pd.DatetimeIndex) -> set:
    """First trading day of each month in the date index."""
    monthly = dates.to_series().groupby([dates.year, dates.month]).first()
    return set(monthly.values)


def _get_optimal_rebalance_dates(
    dates: pd.DatetimeIndex,
    day_offset: int = 2,
) -> set:
    """
    Nth trading day of each month (default: 3rd, with day_offset=2).

    Why not the 1st? Turn-of-month effect creates artificial buying pressure
    from institutional flows in the first 1-2 days. By the 3rd day, prices
    have normalized and we get cleaner fills.
    """
    grouped = dates.to_series().groupby([dates.year, dates.month])
    rebalance = set()
    for _, group in grouped:
        trading_days = group.values
        idx = min(day_offset, len(trading_days) - 1)
        rebalance.add(trading_days[idx])
    return rebalance


# ---------------------------------------------------------------------------
# Stock selection pipeline
# ---------------------------------------------------------------------------

def _select_and_weight(
    signal:          pd.DataFrame,
    date:            pd.Timestamp,
    n_positions:     int,
    weighting:       str,
    concentration:   float,
    max_weight:      float,
    min_weight:      float,
    sector_map:      Optional[Dict[str, str]],
    max_sector_pct:  float,
    momentum_filter: Optional[pd.DataFrame],
    realized_vol:    Optional[pd.DataFrame],
    returns:         Optional[pd.DataFrame] = None,
    adv:             Optional[pd.DataFrame] = None,
    quality_filter:  Optional[pd.DataFrame] = None,
    first_valid_date: Optional[pd.Series] = None,
    earnings_dates:  Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Full selection pipeline for a single rebalance date:
      1. Pre-filters: momentum, quality (soft), IPO buffer, earnings exclusion
      2. Vol-bucket candidate generation (balanced across vol quintiles)
      3. Sector caps
      4. Correlation filter
      5. Weighting with liquidity awareness
    """
    scores = signal.loc[date].dropna()

    if len(scores) < n_positions * 2:
        return pd.Series(0.0, index=signal.columns)

    # ── Pre-filter 1: Momentum ───────────────────────────────────────────
    if momentum_filter is not None and date in momentum_filter.index:
        mom = momentum_filter.loc[date].reindex(scores.index)
        # Use percentile rank > 0.40 (top 60%) instead of raw return > 0.
        # Raw return > 0 lets through stocks with +0.001% momentum (no filter).
        mom_rank = mom.rank(pct=True)
        positive_mom = mom_rank[mom_rank > 0.40].index
        if len(positive_mom) >= n_positions * 3:
            scores = scores.loc[positive_mom]

    # ── Pre-filter 2: Quality (soft) ─────────────────────────────────────
    # Require above 15th percentile (very soft — only excludes the worst junk).
    # This still allows growth stocks without earnings but removes the bottom trash.
    if quality_filter is not None and date in quality_filter.index:
        qual = quality_filter.loc[date].reindex(scores.index)
        passing = qual[qual > 0.15].index
        if len(passing) >= n_positions * 3:
            scores = scores.loc[passing]

    # ── Pre-filter 3: IPO buffer ─────────────────────────────────────────
    # Skip stocks with less than ~180 trading days of history. Recent IPOs
    # are volatile, have no historical features, and often crash after lockup.
    if first_valid_date is not None:
        ipo_cutoff = date - pd.Timedelta(days=270)  # ~180 trading days
        mature = first_valid_date.reindex(scores.index)
        mature_mask = mature.notna() & (mature <= ipo_cutoff)
        mature_tickers = mature_mask[mature_mask].index
        if len(mature_tickers) >= n_positions * 3:
            scores = scores.loc[mature_tickers]

    # ── Pre-filter 4: Earnings exclusion zone ───────────────────────────
    # The earnings signal is cross-sectionally ranked [0,1] where HIGH rank
    # means CLOSE to earnings. Exclude stocks in the top 10% (nearest to
    # earnings) unless the model is very confident (top 1% ML rank).
    if earnings_dates is not None and date in earnings_dates.index:
        earn_rank = earnings_dates.loc[date].reindex(scores.index)
        imminent = earn_rank >= 0.90  # top 10% = closest to earnings
        top_1pct = scores.rank(pct=True) >= 0.99
        exclude = imminent.fillna(False) & ~top_1pct
        safe_tickers = scores.index[~exclude]
        if len(safe_tickers) >= n_positions * 3:
            scores = scores.loc[safe_tickers]

    # ── Step 1: Vol-bucket candidate generation ──────────────────────────
    # Instead of just picking top N (which loads up on high-vol junk),
    # pick from each volatility quintile equally. This ensures the portfolio
    # spans the vol spectrum: low-vol compounders AND high-vol alpha plays.
    if realized_vol is not None and date in realized_vol.index:
        vol_today = realized_vol.loc[date].reindex(scores.index)
        # 3 vol terciles (low/mid/high) — less restrictive than 5 quintiles.
        # V4 used 5 buckets which diluted alpha too much (beta 0.53).
        raw_candidates = _vol_bucket_candidates(
            scores, vol_today, n_per_bucket=max(6, n_positions // 3 * 2),
            n_buckets=3,
        )
        # Use the vol-bucket pool for downstream selection
        scores_pool = scores.loc[scores.index.isin(raw_candidates)]
        if len(scores_pool) < n_positions:
            scores_pool = scores  # fallback to full universe
    else:
        scores_pool = scores

    # ── Step 2: Sector caps ──────────────────────────────────────────────
    n_candidates = min(n_positions * 2, len(scores_pool))
    candidates = _select_with_sector_caps(
        scores=scores_pool,
        n_positions=n_candidates,
        sector_map=sector_map,
        max_sector_pct=max_sector_pct,
    )

    # ── Step 3: Correlation filter ───────────────────────────────────────
    # Prune stocks with >0.80 pairwise correlation to existing picks.
    if returns is not None and len(candidates) > n_positions:
        selected = _correlation_filter(
            candidates=candidates,
            returns=returns,
            date=date,
            n_positions=n_positions,
            max_corr=0.80,
        )
    else:
        selected = candidates[:n_positions]

    if len(selected) == 0:
        return pd.Series(0.0, index=signal.columns)

    # ── Step 4: Compute weights ──────────────────────────────────────────
    adv_series = None
    if adv is not None and date in adv.index:
        adv_series = adv.loc[date].reindex(selected)

    w = _compute_weights(
        selected_scores=scores.loc[selected],
        weighting=weighting,
        concentration=concentration,
        max_weight=max_weight,
        min_weight=min_weight,
        realized_vol=realized_vol.loc[date, selected] if realized_vol is not None and date in realized_vol.index else None,
        adv=adv_series,
    )

    full_weights = pd.Series(0.0, index=signal.columns)
    full_weights[w.index] = w.values
    return full_weights


# ---------------------------------------------------------------------------
# Vol-bucket neutral candidate selection
# ---------------------------------------------------------------------------

def _vol_bucket_candidates(
    scores:       pd.Series,
    realized_vol: pd.Series,
    n_per_bucket: int = 8,
    n_buckets:    int = 5,
) -> List[str]:
    """
    Select top candidates from each volatility quintile.

    This is the key mechanism that prevents the portfolio from becoming
    a pure high-vol bet. Each quintile contributes equal candidates,
    so the final portfolio has a balanced vol profile.
    """
    common = scores.index.intersection(realized_vol.dropna().index)
    if len(common) < 20:
        return scores.nlargest(n_per_bucket * n_buckets).index.tolist()

    vol = realized_vol.loc[common]
    sc = scores.loc[common]
    vol_rank = vol.rank(pct=True)

    selected = []
    for i in range(n_buckets):
        q_lo = i / n_buckets
        q_hi = (i + 1) / n_buckets
        if i == n_buckets - 1:
            mask = (vol_rank >= q_lo) & (vol_rank <= 1.0)
        else:
            mask = (vol_rank >= q_lo) & (vol_rank < q_hi)
        bucket = sc[mask]
        if len(bucket) > 0:
            top = bucket.nlargest(min(n_per_bucket, len(bucket)))
            selected.extend(top.index.tolist())

    return selected


# ---------------------------------------------------------------------------
# Sector cap selection
# ---------------------------------------------------------------------------

def _select_with_sector_caps(
    scores:         pd.Series,
    n_positions:    int,
    sector_map:     Optional[Dict[str, str]],
    max_sector_pct: float,
) -> List[str]:
    """
    Greedy sector-capped selection from ranked candidates.

    "Unknown" sector is EXEMPT from the cap. These are mostly delisted stocks
    and small caps without yfinance sector data. Capping them as one group
    would artificially limit selection from 58% of our universe, defeating
    the purpose of including delisted stocks. The cap only applies to known
    sectors (Technology, Healthcare, etc.) to prevent concentrated industry bets.
    """
    ranked = scores.sort_values(ascending=False)

    if sector_map is None:
        return ranked.index[:n_positions].tolist()

    max_per_sector = max(1, int(n_positions * max_sector_pct))
    sector_counts: Dict[str, int] = {}
    selected: List[str] = []

    for ticker in ranked.index:
        if len(selected) >= n_positions:
            break
        sector = sector_map.get(ticker, "Unknown")

        # Unknown sector exempt from cap — don't limit delisted/small-cap selection
        if sector != "Unknown":
            current_count = sector_counts.get(sector, 0)
            if current_count >= max_per_sector:
                continue
            sector_counts[sector] = current_count + 1

        selected.append(ticker)

    return selected


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------

def _correlation_filter(
    candidates:  List[str],
    returns:     pd.DataFrame,
    date:        pd.Timestamp,
    n_positions: int,
    max_corr:    float = 0.80,
    lookback:    int   = 63,
) -> List[str]:
    """Prune candidates that are too correlated with already-selected stocks."""
    date_loc = returns.index.get_loc(date) if date in returns.index else None
    if date_loc is None or date_loc < lookback:
        return candidates[:n_positions]

    window_returns = returns.iloc[date_loc - lookback:date_loc][candidates]
    try:
        corr_matrix = window_returns.corr()
    except Exception:
        return candidates[:n_positions]

    selected: List[str] = []
    for ticker in candidates:
        if len(selected) >= n_positions:
            break
        if ticker not in corr_matrix.columns:
            selected.append(ticker)
            continue
        if selected:
            max_existing_corr = corr_matrix.loc[ticker, selected].abs().max()
            if max_existing_corr > max_corr:
                continue
        selected.append(ticker)

    return selected


# ---------------------------------------------------------------------------
# Position weighting
# ---------------------------------------------------------------------------

def _compute_weights(
    selected_scores: pd.Series,
    weighting:       str,
    concentration:   float,
    max_weight:      float,
    min_weight:      float,
    realized_vol:    Optional[pd.Series],
    adv:             Optional[pd.Series] = None,
) -> pd.Series:
    """Compute normalized weights with liquidity-aware caps."""
    n = len(selected_scores)

    if weighting == "equal":
        w = pd.Series(1.0 / n, index=selected_scores.index)
    elif weighting == "signal":
        ranks = selected_scores.rank(ascending=True, method="first")
        w = ranks ** concentration
        w = w / w.sum()
    elif weighting == "inverse_vol":
        if realized_vol is not None:
            vol = realized_vol.reindex(selected_scores.index).replace(0, np.nan)
            w = (1.0 / vol).fillna(1.0)
        else:
            w = pd.Series(1.0, index=selected_scores.index)
        w = w / w.sum()
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")

    # Liquidity-aware cap for illiquid stocks
    if adv is not None:
        for ticker in w.index:
            ticker_adv = adv.get(ticker, np.nan)
            if pd.notna(ticker_adv) and ticker_adv > 0:
                adv_millions = ticker_adv / 1e6
                liq_cap = min(max_weight, 0.01 * adv_millions ** 0.5)
                liq_cap = max(liq_cap, min_weight)
                w[ticker] = min(w[ticker], liq_cap)

    for _ in range(5):
        w = w.clip(min_weight, max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        if w.max() <= max_weight + 1e-6 and w.min() >= min_weight - 1e-6:
            break

    return w


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def compute_portfolio_stats(
    weights:          pd.DataFrame,
    rebalance_dates:  Optional[List[pd.Timestamp]] = None,
) -> dict:
    gross_exposure = weights.sum(axis=1)
    n_positions = (weights > 0.001).sum(axis=1)

    daily_turnover = []
    for i in range(1, len(weights)):
        diff = (weights.iloc[i] - weights.iloc[i - 1]).abs().sum() / 2.0
        daily_turnover.append(diff)
    daily_turnover = pd.Series(daily_turnover)

    n_rebalances = len(rebalance_dates) if rebalance_dates else 0
    max_weight_ts = weights.max(axis=1)
    top5_weight_ts = weights.apply(lambda row: row.nlargest(5).sum(), axis=1)

    return {
        "avg_positions":       n_positions.mean(),
        "avg_gross_exposure":  gross_exposure.mean(),
        "avg_max_weight":      max_weight_ts.mean(),
        "avg_top5_weight":     top5_weight_ts.mean(),
        "n_rebalances":        n_rebalances,
        "avg_daily_turnover":  daily_turnover.mean(),
        "annualized_turnover": daily_turnover.mean() * 252,
    }


# ---------------------------------------------------------------------------
# Holdings analysis
# ---------------------------------------------------------------------------

def get_current_holdings(
    weights: pd.DataFrame,
    signal:  pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    latest = weights.iloc[-1]
    held = latest[latest > 0.001].sort_values(ascending=False)

    records = []
    for ticker, weight in held.items():
        record = {
            "ticker": ticker,
            "weight": weight,
            "signal_score": signal.iloc[-1].get(ticker, np.nan),
        }
        if sector_map:
            record["sector"] = sector_map.get(ticker, "Unknown")
        records.append(record)

    return pd.DataFrame(records)


def sector_allocation(
    weights:    pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    sectors = {}
    for ticker in weights.columns:
        s = sector_map.get(ticker, "Unknown")
        sectors.setdefault(s, []).append(ticker)

    alloc = pd.DataFrame(index=weights.index)
    for sector, tickers in sectors.items():
        alloc[sector] = weights[tickers].sum(axis=1)

    return alloc
