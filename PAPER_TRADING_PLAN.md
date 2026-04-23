# Paper Trading Plan

**Last updated:** 2026-04-22
**Version:** 1.0

Master reference and ticket index for the four-stage paper trading rollout. Defines the vision, architecture, ticket organization, dependencies, timeline, and explicit decision gates for progressively moving from intent-generation-only to controlled algorithmic order submission.

## Purpose

**Audience:** portfolio operator (primary decision-maker), developers (implementation and code review), stakeholders (status and risk visibility).

**Relationship to other documents:**
- [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md) — immutable pre-commits: acceptance criteria, non-goals, risk bounds, architectural constraints. Do not change without explicit user approval.
- [PAPER_TRADING_OPS.md](PAPER_TRADING_OPS.md) — mutable runbook: procedural checklists, deployment scripts, monitoring dashboards, incident response. Update freely as operations evolve.
- [PAPER_TRADING_TICKETS.md](PAPER_TRADING_TICKETS.md) — full ticket specifications for all 68 tickets, organized by stage and cross-cutting series.
- [CURRENT_BASELINE.md](CURRENT_BASELINE.md) — reference for current adopted baseline (27.86% / 0.838 / 0.672 DSR) to compare paper trading performance against.
- [DEVELOPMENT_PROCESS.md](DEVELOPMENT_PROCESS.md) — research workflow, run classes, promotion gates for the underlying strategy research.

## Four-Stage Architecture

### Overview

Paper trading progresses through four stages, each with explicit entry prerequisites and exit criteria. Stages are ordered by risk and automation level, with human oversight built into Stages 1–3 and gated automation introduced in Stage 4.

**Stage flow:** intent generation only → read-only broker surface → manual approval per order → controlled automation with preflight gates.

### Stage 1: Shadow Controller

**Goal:** Intent generation only. Prove signal logic, rebalance math, and artifact production work without any broker interface or order submission.

**What happens:**
- New `run_paper_shadow.py` entrypoint (NOT `run_strategy.py`) wakes at 16:45 ET on the 3rd trading day of each month (the real strategy's rebalance calendar per `portfolio._get_optimal_rebalance_dates(day_offset=2)`)
- Loads the adopted baseline's `final_signal.parquet` (path resolved from `PaperTradingConfig.baseline_path` or `CURRENT_BASELINE.md`)
- Calls `portfolio.build_monthly_portfolio()` - the same function the real strategy uses - with the same inputs, then reapplies the promoted strategy's configured post-build overlays (for example vol targeting or credit overlays) before intents are generated
- Computes target weights + intended trades (target-vs-prior diff)
- On non-rebalance days: holds prior-day weights (no rebalance, no intents)
- Writes dated bundle `results/_paper_shadow_<YYYYMMDD_HHMMSS>/`
- Persists state to `paper_trading/history/blotter.jsonl` (append-only)
- No broker calls. No order submission.

**Non-goals:** any broker API contact; reconciliation against live state; coupling to `run_strategy.py` or the options Tradier poll path.

**Entry prerequisites:** none — Stage 1 is the starting point.

**Exit criteria:** 10 consecutive weekdays of clean shadow runs; idempotency verified; manifest hashes validated; operator explicit go-ahead.

**Typical duration:** 2–3 weeks.

**Risk level:** Very Low. Zero market exposure.

### Stage 2: Broker Client Abstraction

**Goal:** Read-only broker integration. Prove we can ping accounts, read balances, and snapshot positions without placing orders.

**What happens:**
- `BrokerClient` abstract interface introduced in `src/paper/brokerage/`
- Two implementations: `TradierBrokerClient` (sandbox + live) and `MockBrokerClient` (tests)
- Broker backend is a first-class config switch with fail-closed behavior
- Read-only operations: `ping()`, `get_profile()`, `get_balances()`, `get_positions()`
- Tokens stored in Windows Credential Manager via `keyring`; `.env.example` updated

**Non-goals:** any order placement; order preview; position mutations; coupling to run_strategy.py.

**Entry prerequisites:** Stage 1 exit criteria met; Tradier sandbox account + token configured.

**Exit criteria:** broker ping + snapshot stable for 5 consecutive days; ≥95% success on reads; p95 latency <2s.

**Typical duration:** 1–2 weeks.

**Risk level:** Very Low. Read-only.

### Stage 3: Manual-Approval Paper Orders

**Goal:** First real orders to Tradier sandbox. Every single order previewed and human-approved before submission. Fills reconciled into immutable blotter.

**What happens:**
- Order methods added to `BrokerClient` interface: preview, place, cancel, replace, status polling
- Diff engine produces proposed trades; order policy layer turns them into order specs
- **Every order previewed via `preview=true`** before submission
- **Phase A writes pending intents; operator approves them via `python -m src.paper.approve`; Phase B submits only approved bundles**
- Sequential submission with status polling until terminal
- Fills written to append-only `fills.jsonl`; prior-day reconciliation gate before next run

**Non-goals:** automated submission; options; shorts; pre/post-market; advanced conditional orders.

**Entry prerequisites:** Stage 2 exit criteria met; `run_paper_phase_a.py` and `run_paper_phase_b.py` tested end-to-end against mock broker with config `stage=3, broker=mock, capital_mode=paper`.

**Exit criteria:** 2 full monthly rebalance cycles with zero operational errors; slippage ≤ 3 bps/side on ≥90% of fills; no bypass of approval gates.

**Typical duration:** 6-8 weeks (must span at least 2 calendar-month rebalance cycles).

**Risk level:** Low. Paper only (sandbox, no real capital). Human review catches operational errors.

### Stage 4: Controlled Automation

**Goal:** Auto-submit on scheduled rebalance days, gated by multi-layer preflight and pre-committed caps. Kill switch mandatory. `--intents-only` escape valve always available.

**What happens:**
- Rebalance calendar: 3rd trading day of each month (matches real strategy; reuses `portfolio._get_optimal_rebalance_dates(day_offset=2)`)
- Two-phase T+1 execution honoring [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md):
  - **Phase A** (T 16:45 ET): compute intents, write pending-order bundle, NO submission
  - **Phase B** (T+1 09:15 ET): load pending intents, submit as market-on-open or limit-at-open, poll until fills terminal
- Two scheduled Windows tasks: `PaperTradingPhaseA` and `PaperTradingPhaseB`
- Composite preflight gate (both phases): fresh data, config hash match, broker ping, prior-day reconciliation clean, no open critical alerts
- Kill switch file at `paper_trading/state/KILL_SWITCH` halts all execution immediately
- Auto-approval **only** when all previews green AND all preflight checks pass (Phase A writes approval record; Phase B verifies it)
- Canary mode: first 10 trading days capped at $500/order and 5 orders/day
- Drawdown circuit breaker: YTD DD < -30% or daily loss < -5% → auto-create kill switch

**Non-goals:** automation without preflight; options; shorts; intra-day trading; override of any failed gate.

**Entry prerequisites:** Stage 3 exit criteria met; pre-committed gate thresholds documented; kill switch tested.

**Exit criteria (canary → full):** 10 canary days clean; 6 consecutive monthly gate checks all passing per CONTRACT.

**Typical duration:** canary 2 weeks; then Stage 4 months 1-3 provisional and months 4-9 counted. Earliest go-live eligibility is end of Stage 4 month 9.

**Risk level:** Medium. Automation reduces manual work but introduces operational dependencies.

## Execution Guide

Roadmap translator: 68 tickets sound like a lot. This section explains what's actually hard, which 10 tickets carry disproportionate risk, what subset gets you to Stage 1, and realistic calendar estimates for solo build + operational validation.

### What's Actually Hard (10 Load-Bearing Tickets)

These 10 carry disproportionate design and implementation risk. Bugs here either block entire stages or corrupt the blotter. The rest is plumbing: necessary but well-defined.

| ID | Title | Why it's hard |
|---|---|---|
| PT-S1-004 | Target book loader | Must call `build_monthly_portfolio()` identically to the real strategy; signature drift causes months of hidden tracking errors. |
| PT-S2-002 | TradierBrokerClient | First external API integration; authentication, rate limiting, sandbox/live routing must fail closed; handle `"null"` string quirks. |
| PT-S3-001 | Order methods on BrokerClient | Preview, place, cancel, replace, poll-until-terminal surface; timing bugs corrupt the fill blotter. |
| PT-S3-004 | Diff engine | Target vs actual → proposed trades; off-by-one or float rounding creates phantom orders. |
| PT-S3-008 | Sequential submission with polling | Orders must block until terminal; race conditions lose fills. |
| PT-S3-009 | Fill reconciliation | Order IDs → fills in append-only blotter; data integrity linchpin. |
| PT-S3-011 | Phase A + Phase B entrypoints | Two-phase T+1 choreography; coupling errors skip entire rebalances. |
| PT-S3-014 | Approval tool | Must persist approval decisions immutably; bugs bypass the human gate in Stage 4 auto mode. |
| PT-S4-002 | Composite preflight gate | Gatekeeper for all automation; false negatives disable Stage 4, false positives leak bad orders. |
| PT-S4-006 | Drawdown circuit breaker | Auto-halt on YTD < −30% or daily < −5%; logic bugs either kill healthy strategies or fail to trigger. |

Bugs in the other 58 tickets are recoverable (restart, rerun bundle, skip a day). Bugs here cost data integrity or stage gates.

### Complexity by Category

All 68 tickets grouped by implementation weight, not by series.

| Category | Count | Complexity | Rationale |
|---|---|---|---|
| Config / schema / plumbing | 15 | Low | Pydantic models, YAML loaders, JSONL schemas. Well-specified. |
| Broker integration | 11 | Medium | Tradier API client, keyring secrets, rate limits, sandbox/live routing. |
| Execution layer | 10 | High | Submission, polling, reconciliation, diff engine, state machine. |
| Safety & gates | 8 | Medium | Preflight, approval, circuit breaker, kill switch. Must fail-closed. |
| Alerting & reporting | 6 | Low-Medium | Email, desktop notifications, reconciliation reports. Glue code. |
| Dashboard | 7 | Low | Views, timeseries, status tiles. Read-only visualization. |
| Tests (embedded) | 7 | Low-Medium | Integration, smoke, contract tests inside PT-S*-011/012/013. |
| Docs & governance | 4 | Trivial | Exit criteria docs, `.env.example`, runbook checklists. |

Only ~10 tickets are actually hard. The rest is scaffolding with well-defined specs.

### MVP Path — 24 Tickets to Intent Generation (~60 hours)

Minimum viable subset to get Stage 1 shadow running (scheduled intent bundles, no broker contact):

- **All 15 PT-X cross-cutting** — config, schemas, secrets, manifests, blotters, persistence, baseline resolver. Foundation; do not cut.
- **9 of 12 PT-S1 tickets** — all except PT-S1-010 (Task Scheduler launcher; run manually), PT-S1-011 (integration tests; defer), PT-S1-012 (exit docs; defer).
- **Nothing else.** Stage 1 alone is shadow mode with zero broker contact.

That's 24 tickets to get intent generation live.

### Full Build Path by Stage

| Phase | Tickets | Effort | What it unlocks |
|---|---|---|---|
| Foundation (PT-X all) | 15 | ~60h | Config, schemas, blotters, baseline resolver. Prereq for everything. |
| Stage 1 Shadow | 12 | ~44h | Intent generation, no broker. Monthly nightly bundles. |
| Stage 2 Read-only | 10 | ~29h | Broker abstraction, `ping()` + snapshot reads. |
| Stage 3 Manual paper | 14 | ~90h | First real paper orders with human approval. The 10 hard tickets land here. |
| Stage 4 Auto | 10 | ~47h | Automation, canary, kill switch, circuit breaker. |
| Dashboard | 7 | ~30h | Operator visibility. Parallel with Stage 3+. |
| **Build total** | **68** | **~300h** | Full four-stage system, operator-ready. |

### Calendar Estimates

Time from "start coding" to "system ready for Stage 4 go-live decision":

| Pace | Build phase | Operational validation | Total |
|---|---|---|---|
| Full-time solo (8 hr/day, 5 day/week) | 7-8 weeks | ~10-11 months | ~12-13 months |
| Part-time solo (4 hr/day) | 14-16 weeks | ~10-11 months | ~13-15 months |
| Evenings/weekends (10 hr/week) | ~30 weeks | ~10-11 months | ~17-18 months |

**Critical caveat:** Build is still roughly 8-10 weeks solo full-time, but operational validation is materially longer than 6 months. Stage 3 must span 2 monthly rebalance cycles, and Stage 4 requires 3 provisional months plus 6 consecutive full-threshold months. The operational clock starts from Stage 3 - not from build kickoff.

### What's Safe to Defer or Skip

| Ticket(s) | Consequence of skipping |
|---|---|
| PT-X-007 (bootstrap wizard) | Run setup manually via `keyring` Python one-liner. |
| PT-S1-010 (Task Scheduler launcher) | Run shadow manually until tired of it. |
| PT-S1-011, PT-S3-012 (integration tests) | Defer — not ideal, commit to backfill before Stage 4 canary. |
| PT-S4-007 (alerting) | Check logs manually the first month. |
| PT-S4-009 (canary auto-run) | Skip if disciplined about starting small. |
| All PT-D-* (dashboard) | Read CSVs and JSONL blotters in a text editor. |

Skipping those drops scope to **~55 tickets, ~220 hours**. Still a complete system; just coarser operator experience and no automation for the first month.

### Recommended Phased Execution

Four phases within the build cycle:

| Phase | Scope | Effort | Duration | Outcome |
|---|---|---|---|---|
| Phase 1 | PT-X foundation + Stage 1 shadow | ~104h | 2 weeks | Intent bundles drop nightly; rebalance math validated |
| Phase 2 | Stage 2 broker read-only | ~29h | 1 week | `BrokerClient.ping()` works; Tradier integration sanity |
| Phase 3 | Stage 3 manual paper | ~90h | 6-8 weeks | Real paper orders with human approval; zero op errors across 2 rebalance cycles |
| Phase 4 | Stage 4 automation + dashboard | ~77h | 1-2 weeks build + 9+ months validation | Scheduled Phase A/B; canary; kill switch; monthly gate program |

Calendar: code can still be built in roughly 7-8 weeks solo full-time, but earliest live eligibility is not until Stage 4 month 9 after the Stage 3 two-cycle manual-paper window. Dashboard can run in parallel from Phase 3 onward.

### Why 68 Tickets Is the Right Scope

The system has four fundamental pieces: intent generation (what to trade), broker integration (read positions, submit orders), execution layer (sequential, stateful order + fill handling), and automation (gates, approval, kill switch). 68 tickets is what it takes to specify those four pieces rigorously enough to ship. Before the audit, the target-book construction didn't match the real strategy — a month-3 surprise waiting to happen. The 68 tickets are not busywork; they're the specification gap closure.

## Ticket Organization

### ID Convention

- **PT-X-nnn:** Cross-cutting (config, schemas, secrets, persistence)
- **PT-S1-nnn:** Stage 1 (Shadow Controller)
- **PT-S2-nnn:** Stage 2 (Broker Client Abstraction)
- **PT-S3-nnn:** Stage 3 (Manual-Approval Paper Orders)
- **PT-S4-nnn:** Stage 4 (Controlled Automation)
- **PT-D-nnn:** Dashboard & Monitoring

### Priority Labels

- **P0:** Blocker — stage cannot exit without it.
- **P1:** High — should be done before stage exits.
- **P2:** Nice-to-have — can defer.

### Status Labels

- TODO / IN PROGRESS / IN REVIEW / DONE

## Ticket Index (Summary)

Full specifications for every ticket below live in [PAPER_TRADING_TICKETS.md](PAPER_TRADING_TICKETS.md). Summary below shows ID, one-line scope, priority, effort, and primary dependencies.

### Cross-Cutting (X series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-X-001 | PaperTradingConfig Pydantic model | P0 | 5pt | — |
| PT-X-002 | Config file format + loader (YAML) | P0 | 5pt | PT-X-001 |
| PT-X-003 | Manifest writer with git SHA + dirty state | P0 | 8pt | PT-X-002 |
| PT-X-004 | Secrets loader via keyring | P1 | 6pt | PT-X-002 |
| PT-X-005 | `.env.example` updates (Tradier vars) | P1 | 2pt | — |
| PT-X-006 | Config hash verification for rebalance approval | P1 | 4pt | PT-X-001, PT-X-003 |
| PT-X-007 | Bootstrap wizard (`python -m src.paper.setup`) | P1 | 6pt | PT-X-001..005 |
| PT-X-008 | TargetBookSnapshot schema (parquet + json) | P0 | 3pt | — |
| PT-X-009 | BrokerPositionsSnapshot schema | P0 | 2pt | — |
| PT-X-010 | OrderBlotter append-only JSONL | P0 | 4pt | PT-X-008 |
| PT-X-011 | FillBlotter append-only JSONL | P0 | 3pt | PT-X-010 |
| PT-X-012 | ReconciliationReport daily artifact | P1 | 4pt | PT-X-008..011 |
| PT-X-013 | PortfolioState rolling ledger | P1 | 3pt | PT-X-009, PT-X-012 |
| PT-X-014 | Directory structure + retention policy | P2 | 2pt | All X |
| PT-X-015 | Baseline path resolver (reads CURRENT_BASELINE.md) | P0 | 2pt | PT-X-001 |

### Stage 1: Shadow Controller (S1 series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-S1-001 | Create `src/paper/` package skeleton | P0 | 2h | — |
| PT-S1-002 | Paper trading config schema | P0 | 4h | PT-S1-001 |
| PT-S1-003 | PaperTradingController orchestrator | P0 | 5h | PT-S1-002 |
| PT-S1-004 | Target book loader | P0 | 4h | PT-S1-001 |
| PT-S1-005 | Intent generator (target-vs-prior diff) | P0 | 4h | PT-S1-001 |
| PT-S1-006 | Daily paper bundle writer | P0 | 5h | PT-S1-001 |
| PT-S1-007 | State persistence layer (JSONL blotter) | P1 | 4h | PT-S1-001 |
| PT-S1-008 | Signal staleness preflight | P0 | 2h | PT-S1-001 |
| PT-S1-009 | Shadow entrypoint script (`run_paper_shadow.py`) | P0 | 3h | PT-S1-003 |
| PT-S1-010 | Windows Task Scheduler launcher | P1 | 3h | PT-S1-009 |
| PT-S1-011 | Stage 1 integration tests | P1 | 6h | PT-S1-003, 006, 007 |
| PT-S1-012 | Stage 1 exit criteria documentation | P0 | 2h | PT-S1-011 |

**Stage 1 total: ~44 hours.**

### Stage 2: Broker Client Abstraction (S2 series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-S2-001 | Define `BrokerClient` abstract interface | P0 | 3h | — |
| PT-S2-002 | Implement `TradierBrokerClient` | P0 | 6h | PT-S2-001, PT-S2-005, PT-X-004 |
| PT-S2-003 | Implement `MockBrokerClient` | P0 | 2h | PT-S2-001 |
| PT-S2-004 | Broker factory + config resolution | P0 | 2h | PT-S2-001..003 |
| PT-S2-005 | Broker backend switch (fail-closed) | P0 | 2h | — |
| PT-S2-006 | Secrets management via keyring | P1 | 3h | PT-X-004 |
| PT-S2-007 | Rate limiter + retry middleware (tenacity) | P1 | 4h | PT-S2-002 |
| PT-S2-008 | Connection health preflight (`ping()`) | P1 | 2h | PT-S2-001, 002 |
| PT-S2-009 | Read-only broker snapshot CLI tool | P2 | 1.5h | PT-S2-002, 004 |
| PT-S2-010 | Stage 2 integration + contract tests | P1 | 3h | PT-S2-001..004 |

**Stage 2 total: ~28.5 hours.**

### Stage 3: Manual-Approval Paper Orders (S3 series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-S3-001 | Extend BrokerClient with order methods | P0 | 8h | PT-S2-* |
| PT-S3-002 | Order blotter schema + persistence | P0 | 6h | PT-S3-001 |
| PT-S3-003 | Fill blotter schema + persistence | P0 | 5h | PT-S3-002 |
| PT-S3-004 | Target-vs-actual diff engine | P1 | 6h | PT-S1-*, PT-S2-* |
| PT-S3-005 | Order policy layer (equity only, caps) | P1 | 6h | PT-S3-004 |
| PT-S3-006 | Preview all orders before submission | P0 | 4h | PT-S3-001, 005 |
| PT-S3-007 | Approval CLI prompt (Y/N gate) | P0 | 5h | PT-S3-006 |
| PT-S3-008 | Sequential order submission + polling | P0 | 7h | PT-S3-001, 002, 007 |
| PT-S3-009 | Fill reconciliation into blotter | P1 | 6h | PT-S3-003, 008 |
| PT-S3-010 | Prior-day reconciliation gate | P0 | 4h | PT-S3-002, 009 |
| PT-S3-011 | Paper trading Phase A + Phase B entrypoints | P0 | 12h | PT-S3-001..010, 014 |
| PT-S3-012 | Stage 3 integration + smoke tests | P1 | 10h | PT-S3-001..011 |
| PT-S3-013 | Phase B execution job (market-on-open submission) | P0 | 7h | PT-S3-008, 010 |
| PT-S3-014 | Approval tool + record schema (`python -m src.paper.approve`) | P0 | 8h | PT-S3-007 |

**Stage 3 total: ~90 hours.** Critical path: PT-S3-001 -> 002 -> 009 -> 010 -> 011 (~37h).

### Stage 4: Controlled Automation (S4 series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-S4-001 | Rebalance calendar | P0 | 4h | PT-S3-* |
| PT-S4-002 | Composite preflight gate function | P0 | 6h | PT-S4-001, PT-S3-* |
| PT-S4-003 | Kill switch file mechanism | P0 | 2h | — |
| PT-S4-004 | Auto-approval gate (preflight + previews) | P0 | 5h | PT-S4-001..003 |
| PT-S4-005 | `--intents-only` mode | P0 | 4h | PT-S3-* |
| PT-S4-006 | Max daily loss / drawdown auto-halt | P1 | 5h | PT-S4-003, PT-S3-* |
| PT-S4-007 | Alerting integration (email/desktop) | P1 | 6h | PT-S4-002, 006 |
| PT-S4-008 | Post-run reconciliation report | P1 | 5h | PT-S4-004, 005 |
| PT-S4-009 | Canary auto-run (first 10 trading days) | P1 | 4h | PT-S4-001, 004, 008 |
| PT-S4-010 | Stage 4 runbook + exit criteria | P0 | 6h | All S4 |

**Stage 4 total: ~47 hours.**

### Dashboard (D series)

| ID | Title | Pri | Effort | Depends |
|---|---|---|---|---|
| PT-D-001 | Paper trading dashboard entrypoint | P0 | 3–5d | — |
| PT-D-002 | Current holdings view (broker vs target) | P0 | 2–3d | PT-D-001 |
| PT-D-003 | Pending orders view | P0 | 2d | PT-D-001 |
| PT-D-004 | Fills view + slippage analysis | P1 | 2–3d | PT-D-001 |
| PT-D-005 | Reconciliation status (red/yellow/green) | P0 | 2d | PT-D-001 |
| PT-D-006 | P&L and metrics timeseries | P1 | 3d | PT-D-001 |
| PT-D-007 | Alerts & incidents log view | P1 | 2d | PT-D-001 |

**Dashboard total: 16–20 days.** Suggested order: PT-D-001 first, then PT-D-005 (most valuable for ops confidence), then PT-D-002/003/004 in parallel, finally PT-D-006/007.

## Dependency Graph

```
PT-X-001..014 (Cross-cutting foundation)
      │
      ▼
PT-S1-001..012 (Shadow Controller)
      │
      ▼  [Stage 1 exit gate]
      │
PT-S2-001..010 (Broker Client Abstraction, read-only)
      │
      ▼  [Stage 2 exit gate]
      │
PT-S3-001..012 (Manual-Approval Paper Orders)
      │
      ├─ PT-D-001..007 (dashboards can run in parallel from here)
      │
      ▼  [Stage 3 exit gate: 2 monthly cycles clean]
      │
PT-S4-001..010 (Controlled Automation)
      │
      ▼  [Stage 4 canary gate: 10 clean days]
      │
Full Stage 4 operation → 6-month gate check → Live deployment
```

**Critical path:** PT-X-001..003 → PT-S1-001..009 → PT-S2-001..004 → PT-S3-001, 002, 006, 007, 008, 011 → PT-S4-001, 002, 003, 004, 005, 009 → operational validation.

## Timeline

### Build Phase (4 weeks)

| Week | Focus | Tickets |
|---|---|---|
| Week 1 | Cross-cutting foundation + Stage 1 | PT-X-001..014; PT-S1-001..012 |
| Week 2 | Stage 2 broker abstraction | PT-S2-001..010 |
| Week 3 | Stage 3 manual-approval workflow | PT-S3-001..012 |
| Week 4 | Stage 4 automation (canary config) + Dashboard P0s | PT-S4-001..010; PT-D-001, 002, 005 |

### Validation Timeline (earliest live eligibility no sooner than Stage 4 month 9)

| Weeks | Focus | Gate |
|---|---|---|
| 1-2 | Stage 1 build + shadow ops | 10 clean weekday shadow runs |
| 3 | Stage 2 read-only | 5 clean broker snapshot days |
| 4-11 | Stage 3 manual paper | 2 full monthly rebalance cycles |
| 12-13 | Stage 4 canary | 10 canary days clean |
| 14-17 | Stage 4 Month 1 provisional | provisional gate check |
| 18-21 | Stage 4 Month 2 provisional | provisional gate check |
| 22-25 | Stage 4 Month 3 provisional | provisional window complete; full counter activates next month |
| 26-29 | Stage 4 Month 4 | full gate month 1/6 |
| 30-33 | Stage 4 Month 5 | full gate month 2/6 |
| 34-37 | Stage 4 Month 6 | full gate month 3/6 |
| 38-41 | Stage 4 Month 7 | full gate month 4/6 |
| 42-45 | Stage 4 Month 8 | full gate month 5/6 |
| 46-49 | Stage 4 Month 9 | full gate month 6/6; earliest go-live decision |

**Go-live (if all gates pass):** week 50+ on the optimistic path. Start at 25-40% of target AUM, ramp over 2-3 months.

## Decision Points

Each point requires explicit operator go/no-go; no implicit progression.

### End of Week 1: Stage 1 Exit Gate

**Go criteria:** 10 weekday shadow runs clean, idempotency verified, manifest hashes valid, exit criteria per PT-S1-012 all met.

**No-go options:** fix bugs, extend shadow by 1 week, retry.

### End of Week 2: Stage 2 Exit Gate

**Go criteria:** broker ping + snapshot stable 5 consecutive days, ≥95% read success, no auth errors, p95 latency <2s.

**No-go options:** fix API integration issues; delay Stage 3 by 1 week.

### End of Week 11 (approx): Stage 3 Exit Gate

**Go criteria:** 2 full monthly rebalance cycles with zero operational errors; slippage ≤ 3 bps/side on ≥90% of fills; no approval gate bypasses.

**No-go options:** fix workflow issues; extend manual mode by 2 weeks.

### End of Week 13 (approx): Stage 4 Canary Gate

**Go criteria:** 10 canary days with caps; all preflight checks pass; order downsizing reasonable; no unhandled escalations.

**No-go options:** extend canary, tighten gates, or rollback to Stage 3.

### End of Month 1, 2, 3: Interim Gate Checks

Per [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md) gate table. Months 1-3 are provisional only and do not start the 6-month counter.

### End of Stage 4 Month 9: Final Go-Live Gate

**Go criteria:** all 8 CONTRACT gates passing for 6 consecutive full-threshold months (Stage 4 months 4-9); final counted month clean; kill switch never activated.

**No-go options:** extend paper; root-cause any gate failures; no live deployment under threshold.

## Non-Goals (Explicit Rejections)

The following are out of scope for this rollout:

1. **Wiring order submission into `run_strategy.py`.** Paper trading has its own entrypoints (`run_paper_shadow.py`, `run_paper_phase_a.py`, `run_paper_phase_b.py`). Research and execution are decoupled.
2. **Coupling to the options Tradier poll path.** Paper trading is equities-only. Options cron stays separate.
3. **Full automation before manual loop proves stable.** Stages 1–3 are sequential gates. No skipping to Stage 4.
4. **Options or short positions.** Strategy is long-only.
5. **Intraday trading or high-frequency rebalancing.** Monthly cadence only.
6. **Bypassing preflight gates in Stage 4.** All gates mandatory. Operator override via kill switch only.
7. **Operating during market stress or data gaps.** If market closed, data stale, or broker API down, rebalances skipped.
8. **Sandbox-only without documented migration path.** Live deployment plan tracked in CONTRACT and OPS.

## Governance

### Change Management

This document is **append-only between major version bumps**. Minor updates (ticket status changes, timeline adjustments) get appended as dated notes. Major structural changes (new stage, altered gates) require explicit version bump (v1.0 → v2.0) and sign-off.

### Ownership

- **Portfolio Operator:** gate decisions, go/no-go calls, live capital allocation.
- **Lead Developer:** ticket implementation, code review, technical risk assessment.
- **QA/Reviewer:** acceptance testing against tickets and contract gates.

### Document Relationships

- **PAPER_TRADING_CONTRACT.md** (immutable) overrides this document on conflict. Gates live there, not here.
- **PAPER_TRADING_OPS.md** (mutable runbook) is the authoritative source for daily operational procedures.
- **PAPER_TRADING_TICKETS.md** holds full ticket specifications; this document holds the summary index.
- **CURRENT_BASELINE.md** provides the performance reference for all comparisons.
- **DEVELOPMENT_PROCESS.md** defines research workflow; paper trading operates within its constraints without modifying it.

## Version History

| Version | Date | Summary |
|---|---|---|
| 1.0 | 2026-04-22 | Initial four-stage architecture, 68 tickets indexed, timeline and gates established. Post-audit additions: PT-X-015 baseline resolver, PT-S3-013 Phase B executor, PT-S3-014 approval tool. |

## Related Documents

- [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md) — Pre-committed gates, rollback triggers, stage exit criteria.
- [PAPER_TRADING_OPS.md](PAPER_TRADING_OPS.md) — Daily/weekly/monthly operational runbook.
- [PAPER_TRADING_TICKETS.md](PAPER_TRADING_TICKETS.md) — Full ticket specifications for all 68 tickets.
- [CURRENT_BASELINE.md](CURRENT_BASELINE.md) — Current adopted baseline.
- [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) — Feature timing and price-basis contract.
- [DEVELOPMENT_PROCESS.md](DEVELOPMENT_PROCESS.md) — Research workflow and promotion gates.
