# Paper Trading Contract

**Last updated:** 2026-04-22
**Version:** 1.0

## Purpose

This document defines the pre-committed rules, gates, and operational thresholds that govern paper trading via the Tradier sandbox environment and the transition to live capital deployment. It establishes a read-only contract: all numerical gates, rollback triggers, stage exit criteria, and performance benchmarks are decided *before* paper trading begins and cannot be revised during the validation period. This contract serves as the rulebook for go-live decisions, operational pause points, and forced rollback triggers. It protects both the baseline strategy performance (27.86% CAGR, 0.838 Sharpe, 0.672 DSR, -45% max drawdown) and the operational stability required for live trading.

## Scope

**In scope:**
- Paper trading via Tradier sandbox API (read-only and execution modes)
- Rebalancing, order placement, and position tracking on paper
- Signal generation and transmission to paper broker
- Slippage measurement and operational metrics during paper
- Transition from paper to live capital deployment in four progressive stages (Stages 1–4)
- Monitoring of drift, drawdown, Sharpe, win rates, and signal reliability

**Out of scope:**
- Options strategies (equity long-only only)
- Short selling (long positions only)
- Intraday trading (EOD rebalance only)
- Ad hoc changes to the promoted strategy's configured overlays or leverage policy during paper. Paper trading mirrors the adopted baseline's configured overlays and leverage, subject to broker/account limits and the hard risk gates below.
- Changes to the feature set or signal model during paper (frozen at paper start)
- Modifications to this contract between version bumps

## Pre-Committed Go-Live Gates

All of the following gates must be met for **6 consecutive calendar months** of automated paper trading (measured from month 4 onward — see Provisional Gate State below) before deploying real capital. A single failure resets the counter.

All windows use 21- or 63-trading-day rolling metrics that are fully defined by the end of month 3. Months 1-3 operate under looser "provisional" thresholds and do not contribute to the 6-month pass counter.
Earliest live eligibility is therefore the end of Stage 4 month 9: months 1-3 provisional, then 6 consecutive full-threshold months.

| Metric | Threshold (Month 4+) | Measurement Window | Backtest Reference | Rationale |
|--------|----------------------|--------------------|--------------------|-----------|
| Sharpe (rolling 63d) | ≥ 0.50 | Rolling 63 trading days | 0.838 | 60% of backtest Sharpe. 63 days ≈ 3 months; Sharpe stabilizes after ~60 samples; fully populated by month 3. |
| Tracking error vs backtest | ≤ 0.75% annualized | Rolling 21 trading days | 0.80% | Daily realized paper return vs daily backtest-predicted return on **same signal and same prices**. NOT vs SPY. 21-day rolling detects execution drift early. |
| Max drawdown | ≥ −25% | Inception-to-date OR calendar year (max of both) | −45.06% | Inception-to-date always defined. Limits tail risk while allowing Monte Carlo variability. |
| Realized volatility | 28–40% annualized | Rolling 21 trading days | 35.1% | Matches tracking-error window. Band ±20% from backtest detects regime shift. |
| Average slippage | ≤ 3.0 bps/side | Each calendar month | 0 (backtest) | Real brokers incur slippage; 3 bps/side is empirically realistic for liquid equities. Measured as (execution price − midpoint at submission) / midpoint. |
| Max L1 drift sustained | ≤ 12% | Daily | 0 (backtest) | Gross position weight divergence. Daily check catches rebalance failures same-day. |
| Monthly win rate | ≥ 55% | Each calendar month | 56% | Days with positive excess return vs backtest-predicted. Confirms signal separates winners. |
| FF Mkt-RF beta | [1.35, 1.65] | Rolling 63 trading days | 1.54 | Band ±10% from backtest. Detects systematic factor shift. Fully populated month 3+. |

### Months 1–3: Provisional Gate State

During the first three months of paper trading, rolling 63-day windows are not fully populated. Operate under provisional thresholds:

| Metric | Provisional Threshold (Months 1–3) | Basis |
|--------|-------------------------------------|-------|
| Sharpe (since-inception) | ≥ 0.35 | 42% of backtest; limited data + early volatility |
| Tracking error vs backtest | ≤ 0.75% (same as full) | Same threshold; monitor closely during ramp |
| Max drawdown | ≥ −25% (same as full) | Inception-to-date always defined |
| Realized volatility | 25–45% annualized | Wider band for noise |
| Average slippage | ≤ 3.0 bps/side (same as full) | Always measurable |
| Max L1 drift | ≤ 12% (same as full) | Always measurable |
| Monthly win rate | ≥ 55% (same as full) | Always measurable after month 1 |
| FF Mkt-RF beta (since-inception) | [1.20, 1.80] | Wider band for noise |

Months 1–3 gate results labeled **PROVISIONAL** and do not contribute to the 6-consecutive-month counter. Full gates (and the counter) activate on day 1 of month 4.

**Gate enforcement:**
- Gates measured on paper trading account data, not backtest or holdout.
- Months 1–3: provisional gates logged but do not start the 6-month clock.
- Month 4+: all 8 full gates must PASS; any single violation resets the counter to 0.
- No deviation from these thresholds is permitted mid-validation. A version bump is required to modify any gate.

## Rollback / Pause Triggers

The following conditions automatically halt paper trading or trigger forced investigation before any further trading continues:

| Trigger | Condition | Action | Investigation Required |
|---------|-----------|--------|------------------------|
| Sharpe collapse | Sharpe < 0.0 for 1 calendar month | Immediate pause | Yes — re-run backtest sensitivity, check signal generation |
| Tracking error spike | Annualized tracking error > 5.0% | Halt + no rebalance | Yes — audit slippage, order routing, position reconciliation |
| Signal failure | Signal generation fails for 2 consecutive trading days | Halt immediately | Yes — check data pipeline, feature computation, cache state |
| Drift runaway | Gross position weight drift > 15% for 7+ consecutive days | Forced rebalance + halt | Yes — diagnose root cause (failed rebalance, new fill delays, market moves) |
| Drawdown alert | YTD drawdown exceeds -30% | High alert + investigation | Yes — re-run backtest out-of-sample test, check for regime shift |
| Single-day loss | Single trading day loss > -5% notional | Auto kill-switch + hard halt | Yes — forensic review of that day's execution, identify root cause |

**Hard halt rules:**
- Any trigger marked "Immediate pause" or "Auto kill-switch" creates a kill-switch file at `paper_trading/state/KILL_SWITCH` and halts all further trading until manually cleared.
- An investigation summary must be written and reviewed before the kill-switch is cleared.

## Normal Drawdown vs Real Failure

Paper trading will experience volatility and occasional drawdown. The contract distinguishes normal strategy variance from actual strategy breakdown:

**Normal (CONTINUE):**
- Paper account hits -20% YTD drawdown, *AND*
- Sharpe remains > 0.40, *AND*
- Monthly win rate remains > 50%, *AND*
- Position drift < 10%

In this scenario: strategy is within normal operating envelope. Continue paper trading and monitoring.

**Failure (STOP and investigate):**
- Paper account hits -20% YTD drawdown, *AND*
- Sharpe < 0.20, *AND*
- Position drift > 20%, *AND*
- Monthly win rate < 40%

In this scenario: multiple independent signals suggest strategy breakdown. Halt immediately, root-cause analysis required, backtest re-validation required before resuming.

## Stage Exit Criteria

Paper trading progression follows four stages plus live deployment. Exit from one stage to the next requires the following conditions:

**Stage 1 → Stage 2: Shadow Trading (Orders Queued, Not Executed)**
- Shadow run executes for 10 consecutive weekdays without interruption.
- Idempotency verified: running same day's signals twice produces identical order queue.
- Artifacts (order queue, signal log, position snapshot) pass validation checks.
- All integration points (data fetch, signal compute, order format) are confirmed stable.

**Stage 2 → Stage 3: Broker Read-Only Operations**
- Broker ping + account snapshot operations remain reliable for 5 consecutive days.
- Account read (holdings, cash, margin requirements) succeeds ≥ 95% of attempts.
- No auth or network timeout errors on broker API calls.
- Latency for snapshot queries stays < 2 seconds (p95).

**Stage 3 → Stage 4: Automated Paper Execution (Tradier Sandbox, Two-Phase T+1)**
- 2 full monthly rebalance cycles executed under Phase A (intent generation at T close, 16:45 ET) + manual approval + Phase B (submission at T+1 open, 09:15 ET).
- Zero operational errors across Phase A and Phase B (intent generation failures, approval gate bypasses, submission failures, position mismatches, reconciliation gaps).
- Realized slippage remains ≤ 3 bps/side for 90%+ of fills (measured on Phase B market-on-open or limit-at-open execution).
- All Phase B submissions occur in the 09:15–09:30 ET pre-open/early-open window; no same-evening submissions.
- No kill-switch triggers; all Phase A and Phase B processes run to completion.

**Stage 4 → Live: Months 1-3 Provisional, Months 4-9 Full Gate Validation**
- Months 1-3 (Provisional): all provisional gates PASS or WATCH; no operational halts except kill-switch triggers.
- Months 4-9 (Full gates): all 8 go-live gates met for 6 consecutive calendar months at full thresholds. A single failure resets the counter to 0.
- Final month clean: zero pause triggers, drift ≤ 12%, no daily loss > −3%, Sharpe > 0.50.
- Kill-switch never activated during months 4-9.
- Operational team has reviewed all monthly reconciliation reports and confirmed no hidden slippage or hidden losses.

## Change Management

This document is **read-only between version bumps**. Changes to gates, triggers, or criteria are not permitted during a paper trading run.

**Change process:**
1. Proposed change requires written justification (1 paragraph minimum: what is changing, why, what is the impact).
2. Change must be submitted as a pull request with the version bump embedded in the document header.
3. The version is incremented: `v1.0` → `v1.1` (minor gate adjustment), or `v1.0` → `v2.0` (major operational change).
4. A brief note is appended to the version history table below.
5. The change takes effect on the next paper trading campaign, not retroactively.

**Version history:**

| Version | Date | Summary | Change Owner |
|---------|------|---------|--------------|
| 1.0 | 2026-04-22 | Initial contract: 8 go-live gates, stage progression, rollback triggers | System |

## Initial Live Deployment Rules

Once paper trading clears all gates and exits Stage 4, live capital deployment follows:

1. **Start with 25–40% of target AUM** in the live broker account (e.g., if target is $100k, deploy $25–40k initially).
2. **Keep remaining capital in SPY and/or cash** in the same account for 2–3 months while monitoring live strategy performance.
3. **Require live strategy performance to match paper within tolerance:**
   - Sharpe within 0.15 of paper (e.g., if paper is 0.55, live must be ≥ 0.40).
   - Monthly returns within 1.5% of paper (absolute deviation).
   - Max drawdown within -5% of paper YTD.
4. **Only after 2–3 months of in-tolerance live trading**, progressively ramp capital to full target size (in 25% tranches per month).
5. **Maintain kill-switch capability** throughout ramp and beyond. Live kill-switch is checked every 5 minutes during market hours.

## Kill Switch Protocol

**File location:** `paper_trading/state/KILL_SWITCH`

**Purpose:** Stops all trading (paper or live) immediately and prevents rebalance until manually cleared.

**Who can create:**
- Operator (human on keyboard)
- Automated pause trigger (Sharpe < 0, signal failure, drift > 15%, drawdown > -30%, single-day loss > -5%)

**What it halts:**
- Order submission to broker (paper or live)
- Position rebalance logic
- New trades are queued but not executed

**How to clear:**
1. Create investigation summary file: `paper_trading/kill_switch_investigation_<YYYY-MM-DD_HHmmss>.md`
2. Root-cause analysis: what happened, why it triggered, what was fixed
3. Recommendation: resume or keep halted
4. Delete `paper_trading/state/KILL_SWITCH` (operator only)
5. Restart trading on next scheduled rebalance window

**Monitoring:**
- Kill-switch file is checked every 60 seconds during trading hours.
- If kill-switch exists, all downstream processes exit cleanly (no partial orders, no orphaned state).

## Related Documents

- [PAPER_TRADING_PLAN.md](PAPER_TRADING_PLAN.md) — Four-stage architecture, ticket index, and execution guide.
- [PAPER_TRADING_OPS.md](PAPER_TRADING_OPS.md) — Daily/weekly/monthly operational runbook.
- [PAPER_TRADING_TICKETS.md](PAPER_TRADING_TICKETS.md) — Full ticket specifications (68 tickets).
- [CURRENT_BASELINE.md](CURRENT_BASELINE.md) — Current adopted baseline (the strategy being paper-traded).
- [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) — T+1 execution contract that paper trading honors.

## Governance

- This document is **read-only** between version bumps. No edits are made to gates, thresholds, or triggers mid-validation.
- Any proposed modification requires a version bump (recorded in the version history table above).
- Paper trading team lead must approve all version bumps before deployment.
- This contract overrides any ad-hoc operational decisions. If a conflict arises, the contract takes precedence.
- All deviations from the contract (emergency pause, manual override) must be logged in `paper_trading/deviation_log.txt` with timestamp and justification.
