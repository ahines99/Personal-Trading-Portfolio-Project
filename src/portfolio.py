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
# Signal post-processing utilities
# ---------------------------------------------------------------------------

def ema_smooth_signals(signal_df: pd.DataFrame, halflife: float = 5.0) -> pd.DataFrame:
    """Apply per-stock EWMA with given halflife (in trading days) to reduce signal
    jitter and turnover.

    Reference: Garleanu-Pedersen 2013 "Dynamic Trading with Predictable Returns".
    """
    # Per-ticker (column) EWMA along the date axis
    return signal_df.ewm(halflife=halflife, min_periods=1).mean()


def cs_rank_normal(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank-normalization: rank -> uniform -> inverse-normal CDF.

    Stabilizes signal distribution across dates and regimes.
    """
    from scipy.stats import norm
    ranks = signal_df.rank(axis=1, pct=True)
    # Clip to avoid infinite values from ppf at 0/1
    ranks = ranks.clip(lower=1e-4, upper=1 - 1e-4)
    return pd.DataFrame(
        norm.ppf(ranks.values),
        index=signal_df.index,
        columns=signal_df.columns,
    )


def factor_neutralize_signal(
    signal_df: pd.DataFrame,
    factor_panel: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Residualize the signal cross-sectionally per date vs supplied factor exposures.

    factor_panel: dict of factor_name -> DataFrame(index=dates, columns=tickers).

    Reference: Asness-Porter-Stevens (2000) 'Predicting Stock Returns Using
    Industry-Relative Firm Characteristics'.
    """
    if not factor_panel:
        return signal_df.copy()

    out = signal_df.copy() * np.nan
    factor_names = list(factor_panel.keys())

    for date in signal_df.index:
        y = signal_df.loc[date]
        # Assemble factor exposures for this date
        cols = []
        for fname in factor_names:
            fdf = factor_panel[fname]
            if date in fdf.index:
                cols.append(fdf.loc[date].reindex(signal_df.columns))
            else:
                cols.append(pd.Series(np.nan, index=signal_df.columns))
        X_df = pd.concat(cols, axis=1)
        X_df.columns = factor_names

        # Keep only tickers with no NaNs across y and all factors
        valid_mask = y.notna() & X_df.notna().all(axis=1)
        if valid_mask.sum() < max(10, len(factor_names) + 2):
            out.loc[date] = y
            continue

        y_v = y[valid_mask].values.astype(float)
        X_v = X_df[valid_mask].values.astype(float)
        # Add intercept
        X_full = np.column_stack([np.ones(len(y_v)), X_v])
        try:
            beta, *_ = np.linalg.lstsq(X_full, y_v, rcond=None)
            resid = y_v - X_full @ beta
            row = pd.Series(np.nan, index=signal_df.columns)
            row.loc[y[valid_mask].index] = resid
            out.loc[date] = row
        except np.linalg.LinAlgError:
            out.loc[date] = y

    return out


# Hardcoded top-10 SPY mega caps (yfinance ticker format: BRK-B not BRK.B).
# Used when force_mega_caps=True to prevent breadth collapse from pushing
# model-liked mega-caps out via vol-bucket/sector-cap constraints.
MEGA_CAPS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "LLY", "JPM",
]


# ---------------------------------------------------------------------------
# Volatility targeting overlay
# ---------------------------------------------------------------------------

def apply_vol_targeting(
    weights: pd.DataFrame,
    realized_vol: pd.DataFrame,
    target_vol: float = 0.16,
    avg_correlation: float = 0.45,
    max_leverage: float = 1.3,
    min_leverage: float = 0.8,
    vol_floor: float = 0.08,
    vol_ceiling: float = 0.25,
) -> pd.DataFrame:
    """
    Scale portfolio gross exposure inversely to ex-ante portfolio vol.

    Moreira & Muir (2017) "Volatility-Managed Portfolios" and DeMiguel et al.
    (JF 2024) — scaling by 1/vol has historically lifted net Sharpe ~0.2-0.4
    (and up to +13% on multifactor portfolios) for US equity strategies.

    Leverage note: values > 1.0 imply MARGIN use. Retail IBKR Reg-T typically
    allows up to 2:1 overnight leverage on marginable equities, so the 1.3
    default is deliberately conservative (30% margin ceiling) while still
    capturing the Moreira-Muir Sharpe lift, which requires the ability to
    *amplify* exposure during low-vol regimes (not just de-lever in high-vol).

    Ex-ante portfolio vol is estimated as
        sigma_p ≈ sqrt( sum(w_i^2 * s_i^2) + rho * (sum(w_i*s_i))^2 - rho * sum(w_i^2*s_i^2) )
    using a constant average pairwise correlation assumption.

    Fix notes (2026-04-05, Tier-8 beta rescue):
      - min_leverage raised from 0.5 to 0.8. The prior 0.5 floor let outlier
        high-vol months slash net exposure to 50%, collapsing net beta to SPY
        toward 0.25. User wants vol targeting to RANGE in [0.8, 1.3], not
        [0.5, 1.3], so de-leveraging doesn't strip market exposure.
      - Ex-ante vol estimate is clipped to [vol_floor, vol_ceiling] before
        computing leverage. Outlier months (churn, ETF contamination,
        small-cap spikes) otherwise drive leverage to the min floor and
        destroy beta. 8-25% brackets the realistic range of a 20-stock
        diversified long-only equity book.
      - Diagnostic logging added: median / min / max leverage and the date
        of the heaviest de-leveraging event.

    Parameters
    ----------
    weights         : (date x ticker) target weights (may sum to <= 1.0)
    realized_vol    : (date x ticker) annualized realized volatility
    target_vol      : desired annualized portfolio vol (e.g. 0.16 = 16%)
    avg_correlation : assumed average pairwise correlation between holdings
    max_leverage    : cap scaling factor (1.3 = up to 30% margin; 1.0 = no margin)
    min_leverage    : floor scaling factor (default 0.8; prevents beta collapse)
    vol_floor       : clamp ex-ante vol estimate (annualized) from below
    vol_ceiling     : clamp ex-ante vol estimate (annualized) from above
    """
    aligned_vol = realized_vol.reindex(index=weights.index, columns=weights.columns)
    w = weights.fillna(0.0)
    s = aligned_vol.fillna(aligned_vol.median(axis=1).median())

    # Per-row ex-ante vol using constant-correlation assumption
    diag = (w.pow(2) * s.pow(2)).sum(axis=1)
    cross = (w * s).sum(axis=1).pow(2)
    port_var = diag + avg_correlation * (cross - diag)
    port_vol = port_var.clip(lower=1e-8).pow(0.5)

    # Clip the vol estimate to a sane range before computing leverage.
    # This prevents outlier churn months from distorting leverage (see docstring).
    port_vol_clipped = port_vol.clip(lower=vol_floor, upper=vol_ceiling)

    leverage = (target_vol / port_vol_clipped).clip(lower=min_leverage, upper=max_leverage)

    # Diagnostic logging — only log for rows where we actually have weights.
    active_mask = w.abs().sum(axis=1) > 1e-6
    if active_mask.any():
        lev_active = leverage[active_mask]
        vol_active = port_vol[active_mask]
        try:
            min_lev_date = lev_active.idxmin()
            print(
                f"  [vol-target] leverage: median={lev_active.median():.3f} "
                f"min={lev_active.min():.3f} max={lev_active.max():.3f} | "
                f"ex-ante vol: median={vol_active.median():.3f} "
                f"min={vol_active.min():.3f} max={vol_active.max():.3f} | "
                f"heaviest de-lever on {min_lev_date.strftime('%Y-%m-%d')}"
            )
        except Exception:
            pass

    return weights.mul(leverage, axis=0)


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
    use_vol_buckets:  bool  = False,
    max_selection_pool: int = 1500,
    spy_core_weight:  float = 0.0,
    spy_ticker:       str   = "SPY",
    force_mega_caps:  bool  = False,
    signal_smooth_halflife: float = 5.0,
    apply_rank_normal: bool = True,
    neutralize_factors: Optional[List[str]] = None,
    factor_panel:     Optional[Dict[str, pd.DataFrame]] = None,
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

    # ── Signal post-processing transforms ────────────────────────────────
    # Applied in order: EMA smoothing -> factor neutralization -> rank-normal.
    # The factor_panel (dict of factor_name -> DataFrame[date, ticker]) must
    # be supplied by the caller; e.g. load size/value/momentum exposures from
    # alt_features_dict and pass them in via `factor_panel=...`.
    if signal_smooth_halflife and signal_smooth_halflife > 0:
        signal = ema_smooth_signals(signal, halflife=signal_smooth_halflife)

    if neutralize_factors and factor_panel:
        sub_panel = {k: factor_panel[k] for k in neutralize_factors if k in factor_panel}
        if sub_panel:
            signal = factor_neutralize_signal(signal, sub_panel)

    if apply_rank_normal:
        signal = cs_rank_normal(signal)

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
                use_vol_buckets=use_vol_buckets,
                max_selection_pool=max_selection_pool,
                force_mega_caps=force_mega_caps,
            )

            if cash_frac > 0:
                new_weights = new_weights * (1 - cash_frac)

            # ── SPY core + satellite overlay ─────────────────────────────
            # When spy_core_weight > 0, scale all stock picks by (1-core)
            # and allocate `core` to SPY directly. Instant beta ~1.0 anchor.
            if spy_core_weight > 0 and spy_ticker in new_weights.index:
                satellite_scale = 1.0 - spy_core_weight
                new_weights = new_weights * satellite_scale
                # Add SPY core weight (additive; SPY likely wasn't picked)
                new_weights[spy_ticker] = new_weights.get(spy_ticker, 0.0) + spy_core_weight

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
    use_vol_buckets: bool = False,
    max_selection_pool: int = 1500,
    force_mega_caps: bool = False,
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

    # Pipeline funnel counters — emitted at the end if final count < n_positions
    stage_counts = {"initial": len(scores)}

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
    stage_counts["post_momentum"] = len(scores)

    # ── Pre-filter 2: Quality (soft) ─────────────────────────────────────
    # Require above 15th percentile (very soft — only excludes the worst junk).
    # This still allows growth stocks without earnings but removes the bottom trash.
    if quality_filter is not None and date in quality_filter.index:
        qual = quality_filter.loc[date].reindex(scores.index)
        passing = qual[qual > 0.15].index
        if len(passing) >= n_positions * 3:
            scores = scores.loc[passing]
    stage_counts["post_quality"] = len(scores)

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
    stage_counts["post_ipo"] = len(scores)

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
    stage_counts["post_earnings"] = len(scores)

    # ── Pre-filter 5: Liquidity pool cap ─────────────────────────────────
    # Training universe stays broad (~3000) but SELECTION only picks from
    # top-N most liquid names by 21-day ADV as of this rebalance date.
    # Prevents picking illiquid micro-caps while keeping ML training data rich.
    if adv is not None and date in adv.index and max_selection_pool > 0:
        adv_today = adv.loc[date].reindex(scores.index).dropna()
        if len(adv_today) > max_selection_pool:
            liquid_tickers = adv_today.nlargest(max_selection_pool).index
            pooled = scores.loc[scores.index.isin(liquid_tickers)]
            if len(pooled) >= n_positions * 2:
                scores = pooled
    stage_counts["post_liquidity"] = len(scores)

    # ── Step 1: Candidate pool (optionally vol-bucket-neutral) ───────────
    # Vol buckets force balanced picks across volatility terciles, which
    # structurally suppresses beta (V4 hit beta=0.53 from this). Default OFF.
    # When OFF, we select top candidates directly by signal score, still
    # subject to sector caps + correlation filter downstream.
    if use_vol_buckets and realized_vol is not None and date in realized_vol.index:
        vol_today = realized_vol.loc[date].reindex(scores.index)
        raw_candidates = _vol_bucket_candidates(
            scores, vol_today, n_per_bucket=max(6, n_positions // 3 * 2),
            n_buckets=3,
        )
        scores_pool = scores.loc[scores.index.isin(raw_candidates)]
        if len(scores_pool) < n_positions:
            scores_pool = scores  # fallback to full universe
    else:
        scores_pool = scores
    stage_counts["post_vol_bucket"] = len(scores_pool)

    # ── Step 2: Sector caps ──────────────────────────────────────────────
    n_candidates = min(n_positions * 2, len(scores_pool))
    candidates = _select_with_sector_caps(
        scores=scores_pool,
        n_positions=n_candidates,
        sector_map=sector_map,
        max_sector_pct=max_sector_pct,
    )
    stage_counts["post_sector_cap"] = len(candidates)

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
    stage_counts["post_correlation"] = len(selected)

    # Diagnostic: if selection fell short of the target, log the funnel.
    if len(selected) < n_positions:
        funnel = " -> ".join(f"{k}={v}" for k, v in stage_counts.items())
        print(
            f"  [select] {date.strftime('%Y-%m-%d')} short "
            f"(got {len(selected)}/{n_positions}): {funnel}"
        )

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

    # ── Step 5: Force mega-cap inclusion ─────────────────────────────────
    # Breadth collapse 2015-2024: if the model likes a mega-cap but
    # vol/sector caps pushed it out, force-include at minimum 3% weight.
    # Only includes mega-caps whose raw signal rank is in top 40% of universe.
    if force_mega_caps:
        raw_scores = signal.loc[date].dropna()
        if len(raw_scores) > 0:
            mega_rank = raw_scores.rank(pct=True)
            min_mega_w = 0.03
            to_force: List[str] = []
            for mc in MEGA_CAPS:
                if mc in w.index:
                    continue  # already picked
                if mc not in mega_rank.index:
                    continue  # not in universe this date
                if mega_rank.loc[mc] >= 0.60:  # top 40%
                    to_force.append(mc)
            if to_force:
                forced_total = min_mega_w * len(to_force)
                # Scale existing weights down to make room
                scale = max(0.0, 1.0 - forced_total)
                w = w * scale
                for mc in to_force:
                    w[mc] = min_mega_w
                # Re-normalize just in case
                total = w.sum()
                if total > 0:
                    w = w / total

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
    scores:                 pd.Series,
    n_positions:            int,
    sector_map:             Optional[Dict[str, str]],
    max_sector_pct:         float,
    unknown_sector_exempt:  bool = False,
) -> List[str]:
    """
    Greedy sector-capped selection from ranked candidates.

    By default, "Unknown" sector is treated as a pseudo-sector subject to the
    SAME cap as other sectors. With roughly half the universe classified as
    "Unknown" (mostly delisted / small-cap illiquid names), exempting it would
    defeat the cap's purpose and concentrate risk in fragile names.

    Set unknown_sector_exempt=True to restore the old behavior (Unknown
    bypasses the cap entirely).
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

        # Unknown is capped by default; only exempt if explicitly requested.
        if unknown_sector_exempt and sector == "Unknown":
            selected.append(ticker)
            continue

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

    # Turnover must measure ACTUAL trades (position reshuffles at rebalances),
    # NOT day-over-day weight drift caused by vol-targeting leverage scaling.
    # Previously this diffed the raw weights DataFrame: since apply_vol_targeting
    # rescales gross exposure on every single day (leverage shifts with daily
    # realized-vol), the day-over-day diff was nonzero on every row and the
    # `* 252` annualization turned that drift into phantom 1500%+ turnover.
    # Fix: normalize each row to gross=1 BEFORE diffing (removes leverage
    # churn so only actual position changes count), then annualize by the
    # true elapsed years instead of mean*252.
    n_rebalances = len(rebalance_dates) if rebalance_dates else 0
    gross_for_norm = weights.sum(axis=1).replace(0, np.nan)
    normed = weights.div(gross_for_norm, axis=0).fillna(0.0)

    daily_turnover = []
    for i in range(1, len(normed)):
        diff = (normed.iloc[i] - normed.iloc[i - 1]).abs().sum() / 2.0
        daily_turnover.append(diff)
    daily_turnover = pd.Series(daily_turnover)

    n_years = len(weights) / 252.0 if len(weights) > 0 else 1.0
    annualized_turnover = daily_turnover.sum() / n_years if n_years > 0 else 0.0

    max_weight_ts = weights.max(axis=1)
    top5_weight_ts = weights.apply(lambda row: row.nlargest(5).sum(), axis=1)

    return {
        "avg_positions":       n_positions.mean(),
        "avg_gross_exposure":  gross_exposure.mean(),
        "avg_max_weight":      max_weight_ts.mean(),
        "avg_top5_weight":     top5_weight_ts.mean(),
        "n_rebalances":        n_rebalances,
        "avg_daily_turnover":  daily_turnover.mean(),
        "annualized_turnover": annualized_turnover,
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
