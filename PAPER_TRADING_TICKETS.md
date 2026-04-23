# Paper Trading Tickets

**Last updated:** 2026-04-22
**Version:** 1.0

Full specifications for all 68 paper trading tickets. Organized by series and dependency order within each stage. Summary index and four-stage architecture in [PAPER_TRADING_PLAN.md](PAPER_TRADING_PLAN.md). Gates and rollback rules in [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md). Daily/weekly/monthly operations in [PAPER_TRADING_OPS.md](PAPER_TRADING_OPS.md).

## Ticket Format

Every ticket uses:
- **ID** (e.g., PT-S1-001)
- **Stage** (1–4, X for cross-cutting, D for dashboard)
- **Priority** (P0 blocker / P1 high / P2 nice-to-have)
- **Effort** (hours or story points)
- **Depends on** (ticket IDs or "—")
- **Scope** (files, classes, function signatures)
- **Acceptance criteria** (testable checklist)
- **Files touched** (new + modified)
- **Notes / risks**

---

# Cross-Cutting Tickets (PT-X Series)

Foundation tickets used across every stage: config schema, secrets, manifest, state models, blotter schemas, retention policy.

## PT-X-001: PaperTradingConfig Pydantic Model (Three-Field Mode)
**Stage:** X | **Priority:** P0 | **Effort:** 5pt | **Depends on:** —

**Scope:** `src/paper/config.py` — Pydantic v2 `BaseModel` with strict validation. Replaces the overloaded single-field `mode` with three orthogonal fields:

- `stage: Literal[1, 2, 3, 4]` (required; execution stage)
  - 1 = Shadow (intent generation only, no broker contact)
  - 2 = Broker read-only (snapshot + connectivity)
  - 3 = Manual approval (submit orders with human Y/N gate)
  - 4 = Controlled auto (submit with auto-approval + preflight gates)
- `broker: Literal["mock", "tradier_sandbox", "tradier_live"]` (default `"mock"`; broker backend)
- `capital_mode: Literal["paper", "live"]` (default `"paper"`; safety level)
- `account_id: str` (required)
- `baseline_path: Optional[str] = None` (null → resolve from CURRENT_BASELINE.md; else explicit path)
- `rebalance_calendar_source: Literal["strategy", "explicit"] = "strategy"` ("strategy" → use `portfolio._get_optimal_rebalance_dates(day_offset=2)` = 3rd trading day; "explicit" → list below)
- `rebalance_calendar: list[dict]` (only used if source="explicit"; `[{"date": "YYYY-MM-DD", "config_hash": "..."}, ...]`)
- `max_order_count: int = 10`
- `max_single_order_notional_usd: float = 50000`
- `allowed_order_types: list[str] = ["market"]`
- `dry_run: bool = True`
- `kill_switch_path: str = "paper_trading/state/KILL_SWITCH"`
- `alert_channel: Literal["email","slack",None] | None`
- `canary_mode: bool = False`
- `hard_halt_drawdown_pct: float | None`
- `hard_halt_daily_loss_pct: float | None`

**Validation rules (Pydantic validators):**
- `stage == 1` → `broker` MUST be `"mock"` (intent-generation-only)
- `capital_mode == "live"` → `stage == 4` AND `broker == "tradier_live"` AND env var `CONFIRM_LIVE_TRADING=true` OR CLI flag `--confirm-live-trading`
- `broker == "tradier_live"` AND `stage < 4` → `PermissionError`
- `broker == "tradier_live"` AND `capital_mode != "live"` → `PermissionError`
- `baseline_path` if provided: must exist and contain `final_signal.parquet`

**Valid combinations:**

| stage | broker | capital_mode | Use case |
|---|---|---|---|
| 1 | mock | paper | Shadow smoke test |
| 2 | tradier_sandbox | paper | Broker read-only sanity |
| 3 | tradier_sandbox | paper | Real paper with manual approval |
| 4 | tradier_sandbox | paper | Auto paper (canary) |
| 4 | tradier_live | live | Go-live (requires explicit confirmation) |

**Acceptance criteria:**
- Pydantic v2 `ConfigDict(validate_assignment=True, frozen=False)`
- All three fields present; no single overloaded `mode` field
- Validators enforce 5 valid combinations and reject 7+ forbidden ones
- `baseline_path` validator: file exists and readable (if non-null)
- Rebalance calendar entries validated as ISO dates (if source="explicit")
- Unit tests cover 5 valid combos + 5 invalid combos + baseline_path resolution

**Files touched:** new `src/paper/config.py`; new `tests/test_paper_config.py`.

**Notes:** Separate from research `args` namespace. Explicit three-field design eliminates the ambiguity between the previous `mode` (stage+safety) and PT-S2-005 `mode` (broker). See [PAPER_TRADING_PLAN.md](PAPER_TRADING_PLAN.md) Non-Goals section for forbidden combinations.

---

## PT-X-002: Config File Format + Loader (YAML)
**Stage:** X | **Priority:** P0 | **Effort:** 5pt | **Depends on:** PT-X-001

**Scope:** YAML config at `config/paper_trading.yaml` (gitignored); example at `config/paper_trading.example.yaml` (committed). Loader: `load_config(path, env_overrides) -> PaperTradingConfig`.

**Behavior:**
- YAML read → env var overrides merged (YAML first, then `PAPER_TRADING_*` env vars) → Pydantic validation
- `.gitignore` excludes `config/paper_trading.yaml` but keeps `.example.yaml`

**Acceptance criteria:**
- Example YAML documents every field
- Env overrides use `PAPER_TRADING_*` prefix and win over YAML
- Malformed YAML produces clear error
- Missing `account_id`, `stage`, `broker`, or `capital_mode` fails loudly (all three-field schema fields required)
- Integration test: load example, override `stage` via env, assert override applied
- Integration test: override invalid combination (e.g., `stage=1 + broker=tradier_sandbox`) → `PermissionError`

**Files touched:** new `src/paper/loader.py`; new `config/paper_trading.example.yaml`; modified `.gitignore`.

**Notes:** Example YAML includes full comment block explaining every field.

---

## PT-X-003: Manifest Writer with Git State
**Stage:** X | **Priority:** P0 | **Effort:** 8pt | **Depends on:** PT-X-002

**Scope:** `src/paper/manifest.py`. Per-paper-run manifest written to `<results_dir>/manifest.json` at start. Fields:
- `run_timestamp` (ISO 8601 UTC)
- `git_sha`, `git_dirty` (bool), `git_dirty_files` (list, truncated at 100 with `+N more`)
- `config_hash` (SHA256 of PaperTradingConfig as sorted JSON)
- `config_path` (absolute path)
- `strategy_version` (baseline label)
- `input_signal_hash` (SHA256 of signal file)
- `python_version`, `pip_freeze_hash`
- `stage`, `broker`, `capital_mode`, `paper_run_id`

**Acceptance criteria:**
- Git SHA + dirty flag via `git rev-parse HEAD` and `git status --porcelain`
- Config hash deterministic (stable JSON serialization)
- Manifest written atomically (temp file + rename)
- Unit test: mock git, verify schema

**Files touched:** new `src/paper/manifest.py`; new tests.

**Notes:** Differs from `snapshot_baseline.py` — manifests are per-run; snapshots are per-promotion.

---

## PT-X-004: Secrets Loader via Keyring
**Stage:** X | **Priority:** P1 | **Effort:** 6pt | **Depends on:** PT-X-002

**Scope:** `src/paper/secrets.py`. Functions:
- `load_tradier_token(mode: str, account_id: str) -> str` — reads from Windows Credential Manager via `keyring` (service `"tradier"`, account `f"{mode}_{account_id}"`). Falls back to env var `TRADIER_API_TOKEN` with loud stderr warning.
- `store_tradier_token(mode, account_id, token)` — writes to keyring.
- CLI: `python -m src.paper.secrets --store sandbox ABC123`

**Acceptance criteria:**
- Keyring query succeeds on Windows Credential Manager
- Fallback warning emitted to stderr, not logged to file
- Token never appears in logs or error messages
- Unit test: mock keyring, verify fallback path

**Files touched:** new `src/paper/secrets.py`; modified `requirements.txt` (add `keyring`); new tests.

**Notes:** Only Tradier token stored here. Account IDs are plain text (not secrets).

---

## PT-X-005: .env.example Updates
**Stage:** X | **Priority:** P1 | **Effort:** 2pt | **Depends on:** —

**Scope:** Append to `.env.example`:
```
# Paper Trading (Tradier Broker)
TRADIER_ACCOUNT_ID_SANDBOX=your_sandbox_account_id
TRADIER_ACCOUNT_ID_LIVE=your_live_account_id
TRADIER_MODE=sandbox
PAPER_TRADING_CONFIG=config/paper_trading.yaml
# NOTE: TRADIER_API_TOKEN should NOT be in .env.
# Use: python -m src.paper.secrets --store to store in Windows Credential Manager.
```

**Acceptance criteria:**
- All fields documented with inline comments
- TRADIER_API_TOKEN explicitly NOT in `.env.example`
- Explanation of keyring preference

**Files touched:** modified `.env.example`.

---

## PT-X-006: Config Hash Verification
**Stage:** X | **Priority:** P1 | **Effort:** 4pt | **Depends on:** PT-X-001, PT-X-003

**Scope:** `src/paper/verify.py`.
- `compute_config_hash(config: PaperTradingConfig) -> str` (SHA256 hex of JSON with sorted keys, no whitespace)
- `verify_rebalance_approval(config_hash, rebalance_entry) -> bool`
- Every paper run: compute current config hash; compare against manifest's stored hash; abort if mismatched. Manual approval never bypasses a config-hash mismatch.

**Acceptance criteria:**
- Hash deterministic (same config → same hash always)
- Mutating one field changes hash
- Rebalance calendar entries require matching hash
- Mismatch triggers clear error with both hashes

**Files touched:** new `src/paper/verify.py`; new tests.

**Notes:** Approval workflow: admin approves rebalance on date D with config hash H; manifest records H; execution checks current hash == H.

---

## PT-X-007: Bootstrap Wizard (`python -m src.paper.setup`)
**Stage:** X | **Priority:** P1 | **Effort:** 6pt | **Depends on:** PT-X-001..005

**Scope:** `src/paper/setup.py`. Interactive CLI:
1. Mode: `[1] sandbox` or `[2] live`
2. Account ID entry (non-empty validation)
3. Token storage to keyring (echo off; confirms success)
4. Broker ping (fetch account summary; print balance)
5. Generate `config/paper_trading.yaml` from `.example.yaml` template + user inputs (rebalance cadence, max order count, alert channel)
6. Summary: config path, next steps

**Acceptance criteria:**
- Wizard non-interactive on success path (default values accepted)
- Invalid inputs prompt retry
- Broker ping confirms connectivity before writing config
- Idempotent re-run asks to overwrite
- Exit 0 on success

**Files touched:** new `src/paper/setup.py`; new `tests/test_setup_wizard.py`.

**Notes:** Scripted: `python -m src.paper.setup --mode sandbox --account-id ABC123 --token XYZ`.

---

## PT-X-008: TargetBookSnapshot Schema
**Stage:** X | **Priority:** P0 | **Effort:** 3pt | **Depends on:** —

**Scope:** `src/paper/schemas/target_book.py`. Pydantic model:
```python
class TargetPosition: ticker, target_weight, signal_score, signal_rank, sector, rationale
class TargetBookSnapshot: as_of_date, signal_hash, signal_timestamp, strategy_version,
    n_positions, target_positions (dict[ticker→TargetPosition]), target_cash_pct,
    rebalance_id (UUID), model_ensemble, created_at
```

**Persistence:**
- Primary: `paper_trading/target_books/{YYYY-MM-DD}.parquet`
- Sidecar: `paper_trading/target_books/{YYYY-MM-DD}.json` (header only)

**Acceptance criteria:**
- Captures all signal components (score, rank, sector, rationale)
- Serialization deterministic (JSON Schema version pinned)
- Example round-trips through pandas without loss
- `target_cash_pct + sum(weights) ≈ 1.0`

**Files touched:** new `src/paper/schemas/target_book.py`; new tests.

---

## PT-X-009: BrokerPositionsSnapshot Schema
**Stage:** X | **Priority:** P0 | **Effort:** 2pt | **Depends on:** —

**Scope:** `src/paper/schemas/broker_positions.py`. Pydantic model:
```python
class BrokerPosition: ticker, quantity (signed), cost_basis, market_value, current_weight,
    entry_date, unrealized_pnl
class BrokerPositionsSnapshot: as_of_date, broker_timestamp, cash_balance, equity_value,
    nav, total_return_pct, gross_exposure, net_exposure, positions (dict), snapshot_id, created_at
```

**Persistence:** `paper_trading/broker_snapshots/{YYYY-MM-DD}.parquet` + `.json` summary.

**Acceptance criteria:**
- Captures all Tradier account fields (cash, positions, unrealized P&L)
- Handles edge cases: zero positions, all-cash, margin account
- Reconciliation `sum(market_value) + cash == nav` is testable

**Files touched:** new schema file + tests.

---

## PT-X-010: OrderBlotter Append-Only JSONL
**Stage:** X | **Priority:** P0 | **Effort:** 4pt | **Depends on:** PT-X-008

**Scope:** `src/paper/schemas/order_blotter.py`. JSONL at `paper_trading/blotter/orders.jsonl`. Row schema:
```
order_id (UUID), rebalance_id, timestamp, symbol, side (BUY/SELL), qty, order_type,
limit_price, stop_price, duration, status (PREVIEW_PENDING..REJECTED), broker_order_id,
preview_result, approval_record, submission_timestamp, terminal_timestamp,
parent_intent_hash, supersedes (null|order_id), created_at
```

**Class `OrderBlotter`:** `append_order(row)`, `get_order(id)`, `get_orders_by_status(s)`, `update_status(id, new_status, updates)` (appends new row; never modifies).

**Acceptance criteria:**
- Append-only semantics (no mutations)
- Status transitions PREVIEW_PENDING → PREVIEW_APPROVED → SUBMITTED → FILLED tracked via append
- `update_status` creates new row; original preserved
- Concurrent append doesn't corrupt file (file-lock or `O_APPEND`)

**Files touched:** new `src/paper/schemas/order_blotter.py`; new `paper_trading/blotter/` dir (with `.gitkeep`); new tests.

**Notes:** Use `datetime.utcnow().isoformat() + "Z"` for all timestamps.

---

## PT-X-011: FillBlotter Append-Only JSONL
**Stage:** X | **Priority:** P0 | **Effort:** 3pt | **Depends on:** PT-X-010

**Scope:** `src/paper/schemas/fill_blotter.py`. JSONL at `paper_trading/blotter/fills.jsonl`. Row:
```
fill_id, order_id, broker_fill_id, timestamp, symbol, side, qty, price, commission,
expected_price, slippage_bps, expected_vs_actual_notional, parent_intent_hash, supersedes, created_at
```

**Class `FillBlotter`:** `append_fill(row)`, `get_fills_by_order(order_id)`, `aggregate_fills_for_order(order_id) -> VWAP + total commission`.

**Acceptance criteria:**
- Each fill immutable
- Partial fills aggregate to correct VWAP
- Slippage negative for buys below limit, positive for sells above
- Commission always positive; deducted from net proceeds

**Files touched:** new schema + tests.

**Notes:** For MARKET orders `expected_price` is reference (last close or VWAP at submit).

---

## PT-X-012: ReconciliationReport Daily Artifact
**Stage:** X | **Priority:** P1 | **Effort:** 4pt | **Depends on:** PT-X-008..011

**Scope:** `src/paper/schemas/reconciliation.py`. Per-day JSON at `paper_trading/reconciliation/{YYYY-MM-DD}.json`:
- `intended_trades`, `executed_trades`, `unfilled_orders`, `cancellations`
- `slippage_summary` (count, mean_bps, median_bps, min/max/std)
- `target_weights`, `actual_weights`, `weight_comparisons` (list of `{ticker, target, actual, drift, drift_bps}`)
- `weight_drift_l1`, `cash_drift`, `nav_drift`
- `pnl_reconciled_vs_backtest` (optional)
- `anomalies` (list[str])

**Acceptance criteria:**
- Sums of positions + cash reconcile to NAV
- Median alongside mean for robustness to outliers
- Anomalies actionable (clear, timestamped)
- Aggregated into `paper_trading/reconciliation_summary.parquet` time-series

**Files touched:** new schema + tests.

---

## PT-X-013: PortfolioState Rolling Ledger
**Stage:** X | **Priority:** P1 | **Effort:** 3pt | **Depends on:** PT-X-009, PT-X-012

**Scope:** `src/paper/schemas/portfolio_state.py`. Append-only JSONL at `paper_trading/portfolio_state/daily_state.jsonl`. Row:
```
date, nav, cash, equity, gross_exposure, net_exposure, num_positions,
daily_return, daily_pnl, daily_pnl_realized, daily_pnl_unrealized,
cum_pnl, ytd_return, inception_return,
ytd_sharpe, ytd_sortino, ytd_max_dd, realized_vol_30d, drift_l1_today,
snapshot_id, created_at
```

**Acceptance criteria:**
- Daily state queryable by Plotly dashboard
- Rolling metrics (Sharpe/Sortino/vol) computed on 252-day windows
- Append-only; use `supersedes` for corrections
- `summary.parquet` rebuilt weekly from JSONL

**Files touched:** new schema + tests.

---

## PT-X-015: Baseline Path Resolver
**Stage:** X | **Priority:** P0 | **Effort:** 2pt | **Depends on:** PT-X-001

**Scope:** `src/paper/baseline_resolver.py::resolve_baseline_path(config: PaperTradingConfig) -> str`. Resolution order:
1. If `config.baseline_path` non-empty: return it verbatim (explicit config wins)
2. Else: read `CURRENT_BASELINE.md`; parse the "Current adopted clean canonical baseline:" line; extract `results/_...` path
3. Verify resolved path exists and contains `final_signal.parquet`
4. Raise `ValueError` if both missing or resolved path invalid

**Acceptance criteria:**
- Explicit `config.baseline_path` returned when set
- `CURRENT_BASELINE.md` parsed via regex or structured lookup
- File existence verified; clear error on missing path or missing `final_signal.parquet`
- Unit tests: (a) explicit path, (b) null path + valid CURRENT_BASELINE.md, (c) null path + malformed CURRENT_BASELINE.md, (d) resolved path missing

**Files touched:** new `src/paper/baseline_resolver.py`; new `tests/test_baseline_resolver.py`.

**Notes:** Single source of truth for the signal source. When a new baseline is promoted, operators update CURRENT_BASELINE.md once; paper trading auto-picks up the new pointer on next run. An explicit `config.baseline_path` is available for pinning a specific run during investigation.

---

## PT-X-014: Directory Structure + Retention Policy
**Stage:** X | **Priority:** P2 | **Effort:** 2pt | **Depends on:** All X

**Scope:** `paper_trading/README.md` documenting directory tree + `src/paper/retention_policy.py` for cleanup logic.

**Directory layout:**
```
paper_trading/
├── README.md
├── target_books/{YYYY-MM-DD}.parquet + .json
├── broker_snapshots/{YYYY-MM-DD}.parquet + .json
├── blotter/
│   ├── orders.jsonl (append-only)
│   └── fills.jsonl (append-only)
├── reconciliation/{YYYY-MM-DD}.json + summary.parquet
├── portfolio_state/daily_state.jsonl + summary.parquet
├── monthly/{YYYY-MM}/report.md + tearsheet.csv + factor_regression.csv + gate_check.md
├── weekly/{YYYY}-W{WW}/report.md
├── daily/{YYYY-MM-DD}_blotter.csv + _reconciliation.json + _report.md
├── state/
│   ├── portfolio_state.json (overwritable current snapshot)
│   ├── KILL_SWITCH (presence = halt)
│   ├── SKIP_UNTIL (optional)
│   └── MANUAL_OVERRIDE.json (optional)
├── dashboard/latest.html + {YYYY-MM-DD}/index.html
└── audit/checksums.sha256 + policy.json
```

**Retention:**
- Blotters (JSONL), snapshots (parquet), reconciliations, portfolio state: **forever** (audit trail)
- Materialized views (rebuilt from JSONL): 2 years
- Dashboard cache: overwrite daily
- Immutable files: chmod 444 on Unix; Windows equivalent

**Acceptance criteria:**
- Layout prevents accidental overwrites (immutable files read-only)
- Materialized views rebuild correctly from JSONL source
- Checksums detect bitrot; daily cron validates all immutable files

**Files touched:** new `paper_trading/README.md`; new `src/paper/retention_policy.py`.

---

# Stage 1 Tickets (PT-S1 Series): Shadow Controller

Intent generation only. No broker calls. Daily bundle + append-only blotter.

## PT-S1-001: Create `src/paper/` Package Skeleton
**Stage:** 1 | **Priority:** P0 | **Effort:** 2h | **Depends on:** —

**Scope:** New Python package:
- `src/paper/__init__.py`
- `src/paper/config.py`
- `src/paper/controller.py`
- `src/paper/loaders.py`
- `src/paper/generators.py`
- `src/paper/writers.py`
- `src/paper/persistence.py`
- `src/paper/preflight.py`

**Acceptance criteria:**
- All 8 files exist, importable
- Each has module docstring
- `import src.paper` succeeds from repo root

**Files touched:** new files above.

---

## PT-S1-002: Paper Trading Config Schema
**Stage:** 1 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S1-001

**Scope:** Implement `src/paper/config.py` (satisfies PT-X-001 as stage-local version):
- Fields per PT-X-001
- Loader `load_config()` returning singleton `PaperTradingConfig`
- Validation: raise `ValueError` on missing/invalid fields

**Acceptance criteria:**
- Schema validates all fields
- Three variants load cleanly (shadow, dry_run, kill_switch_enabled)
- Missing `account_id` fails
- `load_config()` is singleton

**Files touched:** modified `src/paper/config.py`; new `paper_trading/config.toml` example.

---

## PT-S1-003: PaperTradingController Orchestrator
**Stage:** 1 | **Priority:** P0 | **Effort:** 5h | **Depends on:** PT-S1-002

**Scope:** `src/paper/controller.py::PaperTradingController`:
- `__init__(config, repo_root)`
- `.run_daily(as_of_date=None) -> dict`:
  1. `preflight.check_signal_staleness(as_of_date)`
  2. `loaders.load_target_book(as_of_date)` → `(target_weights_df, prior_holdings_df)`
  3. `generators.compute_intended_trades(target, prior)` → `intended_trades_df`
  4. `writers.write_daily_bundle(target, intended, date, git_state)` → `bundle_dir`
  5. `persistence.append_blotter_entry(...)` if not dry_run
  6. Return `{"bundle_dir": str, "n_intents": int, "bundle_manifest": dict}`
- Structured INFO logging per stage

**Acceptance criteria:**
- Instantiates with valid config
- `run_daily()` returns dict with expected keys
- Preflight failures raise
- Dry-run skips persistence but returns same dict
- All sub-stages mockable

**Files touched:** modified `src/paper/controller.py`.

---

## PT-S1-004: Target Book Loader (Build Monthly Portfolio)
**Stage:** 1 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S1-001, PT-X-015

**Scope:** `src/paper/loaders.py::load_target_book(as_of_date=None, baseline_path=None)`. Returns `(target_weights_series, prior_holdings, metadata)`.

**Must reuse the real strategy's portfolio builder — NOT a custom "latest-row-normalize" shortcut.** Paper trading that uses a different target-book construction validates a different portfolio.

Logic:
1. Resolve baseline path via `baseline_resolver.resolve_baseline_path(config)` (PT-X-015)
2. Load signal DataFrame from `{baseline_path}/final_signal.parquet`
3. Check if `as_of_date` is a rebalance day via `portfolio._get_optimal_rebalance_dates(signal.index, day_offset=2)` (3rd trading day of each month — matches real strategy per `src/portfolio.py:601`)
4. If YES (rebalance day):
   - Call `portfolio.build_monthly_portfolio(signal, n_positions=..., weighting=..., concentration=..., sector_map=..., momentum_filter=..., quality_filter=..., realized_vol=..., adv=..., rvol=..., ...)` with the same kwargs the promoted strategy run used
   - Extract weights for `as_of_date` from the returned weight panel
5. If NO (non-rebalance day):
   - Load prior-day weights from `paper_trading/state/portfolio_state.json`
   - Return unchanged
6. Load prior holdings from `paper_trading/state/portfolio_state.json`
7. Metadata: `{signal_date, baseline_path, is_rebalance_day, target_positions_count, build_params_hash}`
8. Raise `ValueError` if signal is stale (> `config.signal_max_staleness_days`)

**Inputs that must be loaded:** sector_map, momentum_filter, quality_filter, realized_vol, adv, rvol, shares_outstanding — read from the baseline run's `data/cache/` or the bundle directory alongside `final_signal.parquet`. If any missing, raise with clear error.

**Acceptance criteria:**
- `build_monthly_portfolio()` called on rebalance days with identical kwargs to the promoted run (verified by reading the run's `run_config.json`)
- Weights sum to ~1.0 on rebalance days; held constant on non-rebalance days
- Metadata includes `is_rebalance_day` flag and `build_params_hash` (reproducibility)
- Unit tests mock `build_monthly_portfolio` and verify target outputs match strategy outputs on the same inputs
- Stale signal raises clear error with age and threshold

**Files touched:** modified `src/paper/loaders.py`; new `tests/test_target_book_loader.py`.

**Notes:** DO reuse `portfolio.build_monthly_portfolio`. The previous ticket text (auto-detect latest row, normalize, explicitly NOT reuse the builder) was an architectural bug — that approach validates a different portfolio than the actual strategy. Real strategy: `src/portfolio.py:463, 601`.

---

## PT-S1-005: Intent Generator
**Stage:** 1 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S1-001

**Scope:** `src/paper/generators.py::compute_intended_trades(target_weights, prior_holdings, notional=100_000)`. Returns DF `[ticker, action, prior_weight, target_weight, prior_notional, target_notional, delta_notional, shares_to_trade]`.
- Merge target × prior on ticker (left outer; prior missing → 0)
- `delta_notional = (target_weight - prior_weight) × notional`
- Filter `|delta_notional| < $100` (noise threshold)
- Action: BUY (delta > 0) / SELL (delta < 0) / HOLD (≈ 0); drop HOLD rows
- Validate `|delta_notional| ≤ max_single_order_notional`; raise if violated
- Sort: BUY ascending, then SELL descending |notional|

**Acceptance criteria:**
- Output columns as specified
- Buy sum ≈ sell sum within 1%
- HOLD rows filtered
- Over-size raises
- Tickers in target but not prior → new positions; in prior but not target → liquidations

**Files touched:** modified `src/paper/generators.py`.

---

## PT-S1-006: Daily Paper Bundle Writer
**Stage:** 1 | **Priority:** P0 | **Effort:** 5h | **Depends on:** PT-S1-001

**Scope:** `src/paper/writers.py::write_daily_bundle(target_weights, intended_trades, as_of_date, git_state, repo_root=None) -> Path`.
- Creates `results/_paper_shadow_<YYYYMMDD_HHMMSS>/`
- Writes `target_weights.csv`, `intended_trades.csv`
- Writes `manifest.json` with:
  - `timestamp` (ISO 8601 UTC)
  - `as_of_date`
  - `git.head`, `git.message`, `git.dirty`, `git.dirty_files` (truncated at 50)
  - `hashes.target_weights` (SHA256 of CSV), `hashes.intended_trades`
  - `stats.n_target_positions`, `stats.n_intended_trades`, `stats.total_buy_notional`, `stats.total_sell_notional`

**Acceptance criteria:**
- Bundle dir created with timestamp
- CSVs written and readable
- Manifest valid JSON with all required fields
- Git state captured via `subprocess.run("git ...")`
- Hashes match re-read files

**Files touched:** modified `src/paper/writers.py`.

**Notes:** Reuse `snapshot_baseline.py` git_state pattern.

---

## PT-S1-007: State Persistence Layer (JSONL)
**Stage:** 1 | **Priority:** P1 | **Effort:** 4h | **Depends on:** PT-S1-001

**Scope:** `src/paper/persistence.py`. Append-only JSONL at `paper_trading/history/blotter.jsonl`; current snapshot at `paper_trading/current/state.json`.
- `append_blotter_entry(as_of_date, intended_trades, bundle_dir)`:
  - Append one JSON object per line: `{date, timestamp, bundle_dir, trades: [...], total_buy_notional, total_sell_notional}`
  - Update `current/state.json` with cumulative position snapshot
- `load_current_state() -> dict` (empty if missing)
- `load_blotter_history(limit=None) -> list[dict]`

**Acceptance criteria:**
- Appends don't truncate/lose data
- Current state updated per run
- History returns entries in order
- Load returns `{}` if missing
- After 5 runs: 5 entries in blotter, current state reflects latest

**Files touched:** modified `src/paper/persistence.py`; new `paper_trading/history/`, `paper_trading/current/`.

**Notes:** Atomic writes (temp file + rename) to prevent corruption.

---

## PT-S1-008: Signal Staleness Preflight
**Stage:** 1 | **Priority:** P0 | **Effort:** 2h | **Depends on:** PT-S1-001

**Scope:** `src/paper/preflight.py::check_signal_staleness(as_of_date=None, results_dir=None, config=None)`:
- Locate latest `final_signal.parquet`
- Parse signal date from parquet metadata (or first/last index row)
- Compare to `as_of_date` (or today)
- Raise `ValueError` if age > `config.signal_max_staleness_days` (default 7)
- Return `{signal_date, age_days, max_allowed_days, status: "OK"|"STALE"}`

**Acceptance criteria:**
- Correct age detection
- Raises when stale
- Returns OK dict when fresh
- Missing signal file raises with clear message

**Files touched:** modified `src/paper/preflight.py`.

---

## PT-S1-009: Shadow Entrypoint Script
**Stage:** 1 | **Priority:** P0 | **Effort:** 3h | **Depends on:** PT-S1-003

**Scope:** `run_paper_shadow.py` at repo root. CLI: `python run_paper_shadow.py [--config <path>] [--as-of-date YYYY-MM-DD] [--dry-run] [--results-dir <path>]`.
- Parse args → load config → instantiate PaperTradingController → `run_daily()`
- Logs to `logs/paper_shadow/<YYYY-MM-DD>.log`
- Stdout: bundle path + stats summary
- Exit 0 on success, 1 on any failure

**Acceptance criteria:**
- Runs cleanly: `python run_paper_shadow.py --dry-run`
- Structured logs to dated log file
- Stdout shows bundle directory
- Exit codes correct
- `--help` shows usage

**Files touched:** new `run_paper_shadow.py`.

**Notes:** Does NOT import from `run_strategy.py`.

---

## PT-S1-010: Windows Task Scheduler Launcher
**Stage:** 1 | **Priority:** P1 | **Effort:** 3h | **Depends on:** PT-S1-009

**Scope:** `automation/run_paper_shadow.ps1`. Parameters: `RepoRoot`, `RunTag`, `DryRun`, `RebalanceCalendar`.
- Skip if non-rebalance day or weekend/holiday
- Otherwise invoke `python run_paper_shadow.py`
- Logs: `logs/paper_shadow_<RunTag>_meta.log`, `.log`, `.stderr.log`
- Task Scheduler registration: `schtasks /Create /SC DAILY /TN "PaperShadowDaily" /TR "...ps1" /ST 16:45 /F`

**Acceptance criteria:**
- Runs: `powershell -ExecutionPolicy Bypass -File automation/run_paper_shadow.ps1 -DryRun`
- Skips weekends/holidays
- Produces three log files
- Registration command in script comments works

**Files touched:** new `automation/run_paper_shadow.ps1`; new `logs/paper_shadow/`.

---

## PT-S1-011: Stage 1 Integration Tests
**Stage:** 1 | **Priority:** P1 | **Effort:** 6h | **Depends on:** PT-S1-003, 006, 007

**Scope:** `tests/test_paper_trading_stage1.py`. Tests:
- `test_controller_instantiation()`
- `test_run_daily_end_to_end()` (dry_run=True; bundle created, state NOT persisted)
- `test_idempotency()` (two runs same date → identical manifest hashes of CSVs; only ONE blotter entry)
- `test_manifest_validity()`
- `test_signal_staleness_preflight()`
- `test_invalid_config()`
- `test_persistence_append()` (dry_run=False; blotter appends)
- `test_bundle_hashes()`
- `simulate_5_day_rebalances()` (5 consecutive trading days; 5 bundles, 5 blotter entries; current state = latest)

**Acceptance criteria:**
- All tests pass with `pytest tests/test_paper_trading_stage1.py -v`
- Coverage > 80% for `src/paper/`
- Fixtures clean up temp files
- 5-day simulation completes without errors

**Files touched:** new test file + fixtures `tests/fixtures/sample_signal_*.parquet`, `sample_holdings_*.csv`.

---

## PT-S1-012: Stage 1 Exit Criteria Documentation
**Stage:** 1 | **Priority:** P0 | **Effort:** 2h | **Depends on:** PT-S1-011

**Scope:** Stage 1 section in [PAPER_TRADING_PLAN.md](PAPER_TRADING_PLAN.md) enumerating:
- Goal, non-goals, duration target (2 weeks)
- Exit criteria (all 12 tickets merged; 100% test pass; 10 consecutive clean trading days; no manual intervention; 100% success rate; no errors in logs; performance <10s/run; 2-week disk usage <10MB)
- Metrics tracked: success rate, bundle creation time, blotter entries, idempotency verification, manifest validity
- Blockers for Stage 2
- Rollback plan (disable task, archive results, inspect blotter)

**Acceptance criteria:**
- Document complete, references all 12 tickets
- Exit criteria measurable (not vague)
- Rollback commands copy-paste ready
- Committed alongside code

**Files touched:** modified `PAPER_TRADING_PLAN.md`.

---

# Stage 2 Tickets (PT-S2 Series): Broker Client Abstraction

Read-only broker API. No order placement.

## PT-S2-001: Define `BrokerClient` Abstract Interface
**Stage:** 2 | **Priority:** P0 | **Effort:** 3h | **Depends on:** —

**Scope:** `src/paper/brokerage/interface.py`. ABC with:
- `ping() -> bool`
- `get_profile() -> dict`
- `get_balances() -> dict`
- `get_positions() -> list[dict]`
- Stubbed abstract order methods (`NotImplementedError` with `# Stage 3` comment): `preview_equity_order`, `place_equity_order`, `cancel_order`, `replace_order`, `get_order_status`, `poll_until_terminal`

**Acceptance criteria:**
- Pure ABC, no implementation
- Typed return schemas with docstrings
- Stage-3 stubs raise NotImplementedError
- Unit test: mock subclass can call read-only methods

**Files touched:** new `src/paper/brokerage/__init__.py`, `src/paper/brokerage/interface.py`.

---

## PT-S2-002: Implement `TradierBrokerClient`
**Stage:** 2 | **Priority:** P0 | **Effort:** 6h | **Depends on:** PT-S2-001, PT-S2-005, PT-X-004

**Scope:** `src/paper/brokerage/tradier.py`. Class `TradierBrokerClient(BrokerClient)`:
- `__init__(account_id, token, sandbox: bool, ...)`
- Base URL: `https://sandbox.tradier.com/v1/` or `https://api.tradier.com/v1/`
- Headers: `Authorization: Bearer <token>`, `Accept: application/json`
- Handle Tradier quirks: literal string `"null"` for empty lists; single-item vs list responses; `application/x-www-form-urlencoded` on POST

**Endpoints (read-only for Stage 2):**
- `GET /user/profile`
- `GET /accounts/{id}/balances` → map to `{account_type, cash, buying_power, equity, ...}`
- `GET /accounts/{id}/positions` → normalize `"null"` to `[]`; wrap single positions in list

**Acceptance criteria:**
- `get_profile()` returns dict with required keys
- `get_balances()` maps Tradier fields correctly
- `get_positions()` handles all three cases (empty, single, multi)
- Unit test with MockBrokerClient asserts interface compliance
- Integration test against Tradier sandbox (skipped if creds not set)

**Files touched:** new `src/paper/brokerage/tradier.py`.

---

## PT-S2-003: Implement `MockBrokerClient`
**Stage:** 2 | **Priority:** P0 | **Effort:** 2h | **Depends on:** PT-S2-001

**Scope:** `src/paper/brokerage/mock.py`. `MockBrokerClient(profile, balances, positions, **kwargs)`:
- All read-only methods return constructor-supplied values verbatim
- `ping()` always True
- No state mutation

**Acceptance criteria:**
- Accepts fixture dicts/lists
- All read-only methods return provided values
- Pytest fixtures usable without external creds
- Interface-compliance test passes

**Files touched:** new `src/paper/brokerage/mock.py`.

---

## PT-S2-004: Broker Factory + Config Resolution
**Stage:** 2 | **Priority:** P0 | **Effort:** 2h | **Depends on:** PT-S2-001..003

**Scope:** `src/paper/brokerage/factory.py::create_broker_client(config) -> BrokerClient`:
- `config.broker == "mock"` -> MockBrokerClient(fixtures)
- `config.broker == "tradier_sandbox"` -> TradierBrokerClient(..., sandbox=True)
- `config.broker == "tradier_live"` -> TradierBrokerClient(..., sandbox=False)
- Token from keyring (PT-X-004) + account_id from config
- Raise `ValueError` if broker backend unknown
- Graceful failure if token missing

**Acceptance criteria:**
- Factory selects correct implementation
- MockBrokerClient requires no external secrets
- TradierBrokerClient fails gracefully on missing token
- Unit tests cover each broker backend

**Files touched:** new `src/paper/brokerage/factory.py`.

---

## PT-S2-005: Broker Backend Switch (Fail-Closed)
**Stage:** 2 | **Priority:** P0 | **Effort:** 2h | **Depends on:** PT-X-001

**Scope:** In `PaperTradingConfig`, the `broker` field (`Literal["mock", "tradier_sandbox", "tradier_live"]`) selects the backend. Related to `stage` and `capital_mode` fields (PT-X-001); all three are orthogonal.

**Specific requirements:**
- Default `broker == "mock"` for safety
- If `broker == "tradier_live"`:
  - `capital_mode` MUST be `"live"` — else `PermissionError`
  - `stage` MUST be `4` — else `PermissionError`
  - Require env var `CONFIRM_LIVE_TRADING=true` OR CLI flag `--confirm-live-trading` — else `PermissionError`
  - Log WARNING on entry: "LIVE BROKER MODE ENABLED — REAL MONEY AT RISK"
- Fail-closed: no silent fallback; explicit validation errors with actionable messages
- Factory `BrokerFactory.create(config)` returns `MockBrokerClient`, `TradierBrokerClient(sandbox=True)`, or `TradierBrokerClient(sandbox=False)`

**Acceptance criteria:**
- Config validation rejects `tradier_live` + `stage < 4`
- Config validation rejects `tradier_live` + `capital_mode != "live"`
- Log warns on live broker mode
- Unit tests cover all 3 broker options + validation errors + missing confirmation env var

**Files touched:** modified `src/paper/config.py`; modified `src/paper/brokerage/factory.py`.

**Notes:** This ticket is renamed from "Sandbox vs Live Config Switch" and restructured. The field is `broker` (backend), not `mode` (which previously overloaded stage + broker + capital). See PT-X-001 for the full three-field design.

---

## PT-S2-006: Secrets Management via Keyring
**Stage:** 2 | **Priority:** P1 | **Effort:** 3h | **Depends on:** PT-X-004

**Scope:** Integrates PT-X-004. `src/paper/brokerage/keyring_manager.py::get_tradier_token(backend)`:
- Service: `"tradier"`; account: `"sandbox"` or `"live"`
- Call `keyring.get_password(service, account)`
- Fallback to env var `TRADIER_TOKEN_<MODE>` with loud warning
- Raise `ValueError` if not found

**Acceptance criteria:**
- Keyring query succeeds on Windows Credential Manager
- Fallback warning emitted
- Token never logged
- Unit test: mock keyring for fallback path
- `docs/SECRETS_SETUP.md` documents pattern

**Files touched:** new `src/paper/brokerage/keyring_manager.py`; new `docs/SECRETS_SETUP.md`; modified `.env.example`.

---

## PT-S2-007: Rate Limiter + Retry Middleware
**Stage:** 2 | **Priority:** P1 | **Effort:** 4h | **Depends on:** PT-S2-002

**Scope:** `src/paper/brokerage/rate_limiter.py`:
- Token-bucket: 60 req/min (trading tier)
- Raise `RateLimitExceeded` if exhausted (client waits)
- tenacity decorator: exponential backoff 1, 2, 4, 8, 16s; max 5 retries
- Retry on: 429, 5xx, `requests.Timeout`
- Do NOT retry: 4xx (except 429), 401, 403
- **Critical:** do NOT retry POST /orders on 5xx (not idempotent); use tag-based reconciliation

**Acceptance criteria:**
- Rate limiter enforces 60/min
- Exponential backoff verified
- 429 triggers wait-then-retry
- 401 fails immediately
- Unit tests mock `requests` for retry/rate-limit
- `requirements.txt` adds `tenacity`

**Files touched:** new `src/paper/brokerage/rate_limiter.py`; modified `src/paper/brokerage/tradier.py`; modified `requirements.txt`.

---

## PT-S2-008: Connection Health Preflight (`ping()`)
**Stage:** 2 | **Priority:** P1 | **Effort:** 2h | **Depends on:** PT-S2-001, 002

**Scope:** `TradierBrokerClient.ping()`:
- Call `/accounts/{id}` (lightweight, idempotent)
- Return `True` if OK; `False` if auth fails (401)
- No exception on expected failures

Controller integration: call `broker.ping()` before any run; raise `BrokerUnavailableError` if False.

**Acceptance criteria:**
- Succeeds with valid account
- Fails gracefully if token invalid (returns False, no raise)
- Controller refuses run on False
- Unit + integration tests

**Files touched:** modified `src/paper/brokerage/interface.py`, `tradier.py`, `controller.py`.

---

## PT-S2-009: Read-Only Broker Snapshot CLI Tool
**Stage:** 2 | **Priority:** P2 | **Effort:** 1.5h | **Depends on:** PT-S2-002, 004

**Scope:** `src/paper/tools/broker_snapshot.py`. CLI: `python -m src.paper.tools.broker_snapshot --mode sandbox --account-id ACC123`. Prints profile, balances, positions in human-readable format.

**Acceptance criteria:**
- Prints formatted output
- Works with all broker implementations
- Graceful handling of missing positions
- Exit 0 on success

**Files touched:** new `src/paper/tools/__init__.py`, `broker_snapshot.py`.

---

## PT-S2-010: Stage 2 Integration + Contract Tests
**Stage:** 2 | **Priority:** P1 | **Effort:** 3h | **Depends on:** PT-S2-001..004

**Scope:** `tests/test_broker_interface.py`. Contract ABC `BrokerContractTest` with tests any BrokerClient must pass: `test_ping`, `test_get_profile_schema`, `test_get_balances_schema`, `test_get_positions_schema`, `test_methods_handle_errors`.

Concrete: `MockBrokerClientTest`, `TradierBrokerClientTest` (skip if Tradier env vars missing). Factory test ensures correct implementation per mode.

**Acceptance criteria:**
- Contract tests pass for MockBrokerClient always
- Pass for TradierBrokerClient when env vars set (skip otherwise)
- Factory test covers each mode
- CI runs mock-only path

**Files touched:** new `tests/test_broker_interface.py`, `tests/fixtures/broker_fixtures.py`.

---

# Stage 3 Tickets (PT-S3 Series): Manual-Approval Paper Orders

First real orders to sandbox. Every order previewed and Y/N approved before submission.

## PT-S3-001: Extend BrokerClient with Order Methods
**Stage:** 3 | **Priority:** P0 | **Effort:** 8h | **Depends on:** PT-S2-*

**Scope:** Implement stubbed abstract methods in interface + Tradier + Mock:
- `preview_equity_order(symbol, qty, side, order_type, limit_price=None) -> PreviewResult`
- `place_equity_order(symbol, qty, side, order_type, limit_price=None, preview_result=required) -> OrderConfirmation` (requires `preview_result` arg — no silent bypass)
- `cancel_order(broker_order_id) -> CancelResult`
- `replace_order(broker_order_id, new_qty=None, new_price=None) -> ReplaceResult`
- `get_order_status(broker_order_id) -> OrderStatus`
- `poll_until_terminal(broker_order_id, timeout=60) -> FinalOrderStatus`

Terminal states: FILLED, CANCELED, REJECTED, EXPIRED, ERROR. Non-terminal: PENDING, PARTIALLY_FILLED.

**Acceptance criteria:**
- `preview` returns cost estimate + warnings
- `place` requires non-None preview_result (no silent override)
- `poll_until_terminal` supports all states
- MockBroker fills in 1–10 polls (configurable); TradierBroker polls real sandbox
- Idempotent where applicable (polling same order returns same status)
- Use UUID tag on every order for idempotency recovery (PT-S2-007)

**Files touched:** modified `src/paper/brokerage/interface.py`, `tradier.py`, `mock.py` (~700 lines total).

---

## PT-S3-002: Order Blotter Schema + Persistence
**Stage:** 3 | **Priority:** P0 | **Effort:** 6h | **Depends on:** PT-S3-001

**Scope:** Integrates PT-X-010. `src/paper/order_blotter.py`:
- `OrderBlotter` class with file lock for concurrent append
- `OrderRecord` dataclass per PT-X-010 schema
- `append_order`, `get_order`, `get_orders_by_status`, `update_status` (append new line)

**Acceptance criteria:**
- Each `append_order` writes exactly one newline-delimited JSON record; no truncation
- Human-readable via `tail -f`
- Status transitions tracked via appends
- `update_status` appends new row; original untouched
- Concurrent append via multiple processes: no corruption (file-lock)

**Files touched:** new `src/paper/order_blotter.py`.

---

## PT-S3-003: Fill Blotter Schema + Persistence
**Stage:** 3 | **Priority:** P0 | **Effort:** 5h | **Depends on:** PT-S3-002

**Scope:** Integrates PT-X-011. `src/paper/fill_blotter.py`:
- `FillBlotter` class + `FillRecord` dataclass
- `append_fill`, `get_fills_by_order`, `aggregate_fills_for_order` (VWAP, total commission)

**Acceptance criteria:**
- One immutable record per fill
- Partial fills aggregate correctly (VWAP = `sum(qty × price) / sum(qty)`)
- Slippage sign correct (bid/ask convention)
- Commission positive and deducted from proceeds
- Fill timestamp ≥ order submission time

**Files touched:** new `src/paper/fill_blotter.py`.

---

## PT-S3-004: Target-vs-Actual Diff Engine
**Stage:** 3 | **Priority:** P1 | **Effort:** 6h | **Depends on:** PT-S1-*, PT-S2-*

**Scope:** `src/paper/diff_engine.py::DiffEngine`:
- `compute_trades(target_weights, broker_positions, previous_positions=None) -> ProposedTrades`
- Load broker cash + stock positions
- `target_qty = target_weight × total_portfolio_value`
- `delta_qty = target_qty - current_qty`
- Filter deltas < $10 notional
- Rank by `|delta_qty|` descending
- Return `ProposedTrade(symbol, side, qty, est_price, est_notional, target_weight, current_weight)`

**Acceptance criteria:**
- SELL for over-target positions; BUY for under-target
- Cash reconciliation: `sum(costs) ≤ available_cash`
- No trade > 10% of daily volume (flags warning)
- Zero-weight existing → explicit SELL
- New target not held → explicit BUY

**Files touched:** new `src/paper/diff_engine.py`; export `build_monthly_portfolio` return type from `src/portfolio.py`.

---

## PT-S3-005: Order Policy Layer
**Stage:** 3 | **Priority:** P1 | **Effort:** 6h | **Depends on:** PT-S3-004

**Scope:** `src/paper/order_policy.py::OrderPolicy`:
- `__init__(max_single_order_notional=50000, max_order_count=20, qty_rounding="lot", order_type="MARKET")`
- `build_order_specs(proposed_trades) -> OrderSpecs`:
  - Equity only (skip ETFs/indices/futures; log warnings)
  - MARKET default; optional conservative LIMIT (-2% buys, +2% sells)
  - Rounding: lot (100-share) / share / dollar
  - Single-order cap: split if needed
  - Order count cap: raise `CapExceededException` if over
  - Return list of `OrderSpec(symbol, qty, side, order_type, limit_price)`

**Acceptance criteria:**
- MARKET orders no limit_price; LIMIT have it
- Quantity ≥ 1 (drop if rounded to 0)
- Notional ≤ max_single_order_notional
- Total ≤ max_order_count (else raise)
- Non-equity logged and filtered

**Files touched:** new `src/paper/order_policy.py`.

---

## PT-S3-006: Preview All Orders Before Submission
**Stage:** 3 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S3-001, 005

**Scope:** `src/paper/preview_engine.py::PreviewEngine`:
- `preview_batch(order_specs, abort_on_error=True) -> PreviewBatch`:
  - For each order call `broker_client.preview_equity_order()`
  - Collect all `PreviewResult`s
  - Abort if ANY has errors
  - Compute batch totals: notional, commission, cash required
  - Cache previews 30s (avoid re-previewing same order)

**Acceptance criteria:**
- No submission without successful preview
- Any preview error → retain all orders for review (no partial submission)
- Preview cache hit within 30s
- Warnings collected without blocking

**Files touched:** new `src/paper/preview_engine.py`.

---

## PT-S3-007: Approval CLI Prompt
**Stage:** 3 | **Priority:** P0 | **Effort:** 5h | **Depends on:** PT-S3-006

**Scope:** `src/paper/approval_cli.py::prompt_for_approval(preview_batch, target_weights, current_weights) -> ApprovalRecord`:
- Display summary table (drift, action, shares, est. cost)
- Preview results (total notional, commission, cash required, warnings)
- Capture operator identity (default `$USER`)
- Capture rebalance note (optional)
- Require explicit Y (case-insensitive)
- Log approval record (immutable)
- Rejection → `ApprovalRecord(approved=False, reason="user_declined")`

**Acceptance criteria:**
- Prompt blocks submission until Y typed
- Operator identity captured
- Summary table ≤80 chars wide
- Approval record immutable after log
- Rejection is graceful

**Files touched:** new `src/paper/approval_cli.py`; new `logs/approvals.jsonl`.

---

## PT-S3-008: Sequential Order Submission + Status Polling
**Stage:** 3 | **Priority:** P0 | **Effort:** 7h | **Depends on:** PT-S3-001, 007, 002

**Scope:** `src/paper/submission_engine.py::SubmissionEngine`:
- `submit_and_poll(order_specs, preview_batch, approval_record) -> SubmissionResult`:
  - Prioritize SELL first, then BUY
  - Per order:
    1. Create OrderRecord (status=SUBMITTED); append to blotter
    2. `place_equity_order(spec, preview_result=...)`
    3. Update OrderRecord: broker_order_id, status=PENDING
    4. `poll_until_terminal(order_id, timeout=60)`
    5. Log state transitions
    6. Final status → update OrderRecord
  - Partial fills recorded
  - Continue on failures (partial execution allowed)
  - Return `SubmissionResult(submitted, filled, partial, rejected, halted)`

**Acceptance criteria:**
- Sequential submission (not parallel)
- Refuses if `approval_record.status != "APPROVED"`
- Polling terminal within 60s; timeout → HALTED
- Partial fills recorded with `qty_filled < qty_requested`
- Blotter updated after each state change

**Files touched:** new `src/paper/submission_engine.py`.

---

## PT-S3-009: Fill Reconciliation into Blotter
**Stage:** 3 | **Priority:** P1 | **Effort:** 6h | **Depends on:** PT-S3-008, 003

**Scope:** `src/paper/reconciliation_engine.py::ReconciliationEngine`:
- `reconcile_and_update_positions(end_of_day=True) -> ReconciliationResult`:
  - After all orders terminal, fetch final order list from broker; cross-check broker_order_id
  - For each FILLED order, fetch fill details; write to `fills.jsonl`
  - Aggregate: VWAP, commission, gross/net proceeds
  - Update `paper_trading/positions.json`: `{snapshot_time, cash, positions: {ticker: {qty, vwap, gross_cost}}}`
  - Return `ReconciliationResult(reconciliation_ok, discrepancies)`

**Acceptance criteria:**
- All FILLED orders have FillRecords
- VWAP = `sum(qty × price) / sum(qty)`
- Commission deducted from cash
- Positions updated only after fills written
- Discrepancies logged (e.g., qty off by 1 due to broker rounding)

**Files touched:** new `src/paper/reconciliation_engine.py`; `paper_trading/positions.json` (regenerated EOD).

**Notes:** Broker fills may lag 30s–2min; don't reconcile too early.

---

## PT-S3-010: Prior-Day Reconciliation Gate
**Stage:** 3 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S3-002, 009

**Scope:** `src/paper/daily_gate.py::verify_prior_day_reconciliation() -> bool`:
- Load previous day's blotter entries
- Check all terminal (FILLED, CANCELED, REJECTED)
- If any PENDING/HALTED → raise `UnreconciledOrdersException` with list
- Load previous day's fills; verify qty reconciles with orders
- Return True if clean; else raise

**Acceptance criteria:**
- Rebalance cannot start if prior-day unreconciled
- Exception lists unreconciled order IDs
- Operator manually cancels/accepts halted orders; then retry gate
- Once clean, rebalance proceeds

**Files touched:** new `src/paper/daily_gate.py`.

**Notes:** Weekend/holiday gap handling: Friday→Monday is OK.

---

## PT-S3-011: Paper Trading Phase A + Phase B Entrypoints
**Stage:** 3 | **Priority:** P0 | **Effort:** 12h | **Depends on:** PT-S3-001..010, PT-S3-013, PT-S3-014

**Scope:** TWO new entrypoints (not one), decoupling intent generation from execution so Stage 3 works under headless Task Scheduler. These replace the previous single-entry design.

**`run_paper_phase_a.py` — Daily 16:45 ET via Task Scheduler:**
Scheduled for Stages 2+ (Stage 1 uses `run_paper_shadow.py`).
1. Preflight (data freshness, broker ping, config hash, prior-day reconciliation, kill switch)
2. Resolve baseline path; load signal
3. If rebalance day: `portfolio.build_monthly_portfolio()` → target_weights
4. `diff_engine.compute_trades()`
5. `order_policy.build_order_specs()`
6. `preview_engine.preview_batch()` — abort on any preview error
7. Write intent bundle to `paper_trading/pending/<YYYY-MM-DD>/intents.json` with `status="AWAITING_APPROVAL"`
8. Stage 3: emit desktop notification. Stage 4: call `auto_approval_gate.auto_approve()`; if green, write machine-generated approval record to `intents.json` with `status="APPROVED"`
9. Log to `logs/paper_trading/phase_a_<YYYY-MM-DD>.log`
10. Exit. NO order submission.

**`run_paper_phase_b.py` — Daily 09:15 ET via Task Scheduler:**
Scheduled for Stages 2+ (Stage 2 is no-op; Stages 3–4 submit if approved).
1. Preflight (same as Phase A)
2. Scan `paper_trading/pending/<YESTERDAY>/` for intents.json
3. For each pending bundle:
   - Verify `approval_record.status == "APPROVED"`; skip on REJECTED or AWAITING_APPROVAL
   - Recompute broker holdings; if drift > 5% from Phase A snapshot, HALT + alert
   - `submission_engine.submit_and_poll()` — market-on-open or limit-at-open
   - Log fills to `fills.jsonl`
4. `reconciliation_engine.reconcile_and_update_positions()`
5. Update `paper_trading/state/portfolio_state.json`
6. Summary written to `paper_trading/reports/<YYYY-MM-DD>/phase_b_summary.md`

**Stage 3 day-in-the-life:**
- **Day N 16:45 ET:** Phase A auto-runs → `intents.json` with status="AWAITING_APPROVAL"
- **Day N 16:50 ET - Day N+1 08:00 ET:** operator runs `python -m src.paper.approve` at own computer; approval record written back
- **Day N+1 09:15 ET:** Phase B auto-runs → reads approval, submits if APPROVED

**Acceptance criteria:**
- Phase A runs headless; generates bundle without submitting
- Phase B verifies approval_record before any submission
- Missing / rejected approval → Phase B skips gracefully (no orders, log entry)
- Overnight state drift > 5% → Phase B halts with operator alert
- `--auto-approve` flag explicit in Phase A for Stage 4 (not hidden, not default)
- Integration tests cover: (a) Phase A → approval → Phase B happy path, (b) Phase A → rejection → Phase B skips, (c) Phase A → no approval → Phase B skips, (d) drift halt

**Files touched:** new `run_paper_phase_a.py`; new `run_paper_phase_b.py`; no single-entry execution wrapper.

**Notes:** This design honors the T+1 contract from [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) (signals at close T, fills at open T+1) AND works under non-interactive Task Scheduler (approval is a separate interactive command, not part of the scheduled job).

---

## PT-S3-012: Stage 3 Integration + Smoke Tests
**Stage:** 3 | **Priority:** P1 | **Effort:** 10h | **Depends on:** PT-S3-001..011

**Scope:** `tests/test_paper_stage3.py`. Categories:
- Broker interface tests (Mock deterministic; Tradier preview rejects insufficient BP)
- Blotter tests (append-only, VWAP aggregation, transitions logged)
- Diff engine tests
- Policy layer tests (rounding, caps)
- Preview engine tests
- Approval CLI tests (Y/N captured; rejection prevents submission)
- Submission engine tests
- Reconciliation tests
- Daily gate tests (unreconciled blocks new rebalance)
- Integration: full flow with mock broker; shadow mode co-run
- Blotter append-only across multiple runs
- Approval gate cannot be bypassed (hidden auto-approve flag)

**Acceptance criteria:**
- All tests pass
- Integration runs full flow
- Shadow vs paper-mode produces identical target weights
- Append-only semantics preserved
- Approval gate safe (auto-approve requires explicit use)

**Files touched:** new `tests/test_paper_stage3.py` (~600 lines); new test config.

**Notes:** Integration tests use mock broker only (no real Tradier).

---

## PT-S3-013: Phase B Execution Job (Market-on-Open Submission)
**Stage:** 3 | **Priority:** P0 | **Effort:** 7h | **Depends on:** PT-S3-008, PT-S3-010, PT-S3-014

**Scope:** `src/paper/phase_b_executor.py::PhaseBExecutor`:
- `load_pending_intents(date: str) -> PendingOrderBundle` — reads `paper_trading/pending/<date>/intents.json`
- `verify_approval(bundle) -> bool` — checks `approval_record.status == "APPROVED"`; returns False (skip) otherwise
- `check_overnight_drift(bundle, broker_positions) -> bool` — if broker state differs > 5% from Phase A snapshot, return False (halt)
- For each order (sell first, then buy):
  - Submit as MOO (market-on-open) or limit-at-open (yesterday's close ±0.5% buffer)
  - Poll `broker.get_order_status()` every 5s, up to 60s
  - Record fill to `paper_trading/daily/<YYYY-MM-DD>_fills.jsonl`
- Reconcile fills vs intents; tolerance ±1% VWAP
- Update `paper_trading/state/portfolio_state.json`
- Return `PhaseBResult(approved, submitted, filled, rejected, halted, discrepancies)`

**Acceptance criteria:**
- Loads yesterday's intents correctly
- Submits only if `approval_record.status == "APPROVED"`
- Drift detection halts execution cleanly when broker state moved overnight
- Submission window: 09:15–09:30 ET pre-open; MOO honored by broker
- Fills logged with timestamp, qty, price, commission, slippage_bps
- Discrepancies flagged in alert log
- Portfolio state updated after all orders terminal
- Unit tests mock broker for: (a) happy path all-filled, (b) partial fills, (c) timeouts, (d) overnight drift halt, (e) no approval

**Files touched:** new `src/paper/phase_b_executor.py`; new `tests/test_phase_b_executor.py`.

**Notes:** This is the sole order-submission path in Stages 2–4. Phase A never submits; Phase B never generates intents. Strict phase separation enforces T+1 contract.

---

## PT-S3-014: Approval Tool & Record Schema
**Stage:** 3 | **Priority:** P0 | **Effort:** 8h | **Depends on:** PT-S3-007

**Scope:** Interactive CLI tool `python -m src.paper.approve` + `ApprovalRecord` schema. Replaces the previous in-scheduled-job Y/N prompt (infeasible under headless Task Scheduler).

**`ApprovalRecord` schema (`src/paper/models/approval.py`):**
```python
class ApprovalRecord(BaseModel):
    approved_at: datetime          # UTC ISO
    operator: str                  # username or email
    status: Literal["APPROVED", "REJECTED"]
    comment: str | None            # optional free-form rationale
    hostname: str                  # for audit; where approval ran
```

**`IntentBundle` schema (`src/paper/models/intents.py`):**
```python
class IntentBundle(BaseModel):
    generated_at: datetime
    approval_deadline: datetime    # next morning 08:00 ET
    status: Literal["AWAITING_APPROVAL", "APPROVED", "REJECTED"]
    signal_hash: str
    config_hash: str
    target_weights: dict[str, float]
    current_holdings: dict[str, float]
    proposed_orders: list[OrderSpec]
    aggregate: dict                # total_notional, commission, cash_needed, turnover_pct
    approval_record: ApprovalRecord | None
```

**CLI `python -m src.paper.approve` flow:**
1. Scan `paper_trading/pending/*/intents.json` for bundles with `status="AWAITING_APPROVAL"`
2. For each pending bundle:
   a. Display summary table (generated_at, target vs current holdings, proposed trades, aggregate stats)
   b. Display warnings: high turnover (> 0.25), high concentration (any position > 0.15), high est-cost (> 0.30%)
   c. Prompt: "APPROVE ORDER BUNDLE for `<date>`? (Y/N/DETAILS/SKIP)"
      - Y → write `ApprovalRecord(status="APPROVED")` to bundle
      - N → write `ApprovalRecord(status="REJECTED")` to bundle
      - DETAILS → show full per-order breakdown; re-prompt
      - SKIP → leave as-is; exit (operator will return later)
3. If multiple pending bundles, prompt per bundle in chronological order
4. Append to `logs/approvals.jsonl` for audit

**Acceptance criteria:**
- CLI runs on operator's machine; no server required
- Clear summary display in ≤ 80-char wide table
- Warnings surface without blocking approval
- Approval record persisted in `intents.json` and in append-only `logs/approvals.jsonl`
- SKIP leaves bundle AWAITING_APPROVAL; next run re-prompts
- Operator identity captured via `getpass.getuser()` + hostname
- Unit tests cover: approve, reject, skip, details, multiple pending bundles

**Files touched:** new `src/paper/approval_cli.py`; new `src/paper/models/approval.py`, `src/paper/models/intents.py`; new `tests/test_approval_cli.py`.

**Notes:** This is the operator's manual-approval interface for Stage 3. Stage 4 auto-approval bypasses this (Phase A writes approval record directly). Either way, Phase B only cares that a valid APPROVED record exists before submission.

### Safety Invariants (Stage 3)

Must hold unconditionally:
1. **Cannot submit without preview** — `submission_engine` checks `preview_batch is not None`; else `PreviewMissingException`
2. **Cannot submit without approval** — checks `approval_record.status == "APPROVED"`; else exit early
3. **Cannot start if prior-day unreconciled** — preflight gate raises `UnreconciledOrdersException`
4. **Kill switch halts pending orders** — `kill_switch.py` module cancels pending, sets HALTED, logs reason

### Manual Override (Stage 3)

- **Skip today:** `run_paper_phase_a.py --skip-today --reason "travel"` logs skip
- **Cancel order:** `run_paper_phase_b.py --cancel-order=ORD-YYYYMMDD-001`
- **Replace order:** `run_paper_phase_b.py --replace-order=ORD-... --new-qty=100 --new-price=150.00`
- **Record fill manually:** `run_paper_phase_b.py --record-fill='{...}'`

---

# Stage 4 Tickets (PT-S4 Series): Controlled Automation

Auto-submit on rebalance days, multi-layer preflight, kill switch, intents-only escape valve.

## PT-S4-001: Rebalance Calendar (3rd Trading Day)
**Stage:** 4 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S3-*

**Scope:** Reuse `portfolio._get_optimal_rebalance_dates(all_dates, day_offset=2)` — the same calendar the real strategy uses (`src/portfolio.py:601`). For each month, this selects the 3rd trading day (not last-business-day).

Helpers in `src/paper/calendar.py`:
- `get_rebalance_dates(year, signal_index=None) -> set[pd.Timestamp]` — thin wrapper over `_get_optimal_rebalance_dates`
- `is_rebalance_day(date, signal_index=None) -> bool`
- Optional override via `config.rebalance_calendar` if `config.rebalance_calendar_source == "explicit"` (emergency manual override only)

Paper trading calls these to determine which days trigger `portfolio.build_monthly_portfolio()` in PT-S1-004.

**Acceptance criteria:**
- 3rd-trading-day dates match `portfolio._get_optimal_rebalance_dates(day_offset=2)` exactly (regression test: generate 2013–2026 rebalance dates from both paths, diff must be empty)
- Feb (28/29 days), holiday weeks, leap year: all handled correctly via the shared underlying code
- Dry-run with `--intents-only` verifies paper Phase A fires only on rebalance days
- Explicit override mode tested separately

**Files touched:** new `src/paper/calendar.py` (thin wrapper); NO changes to `src/portfolio.py` (reuses existing function).

**Notes:** DO NOT create a parallel rebalance calendar (e.g., last-business-day). Paper trading must match the real strategy's calendar exactly. Previous ticket text specifying `pd.bdate_range()` last-business-day was an architectural bug.

---

## PT-S4-002: Composite Preflight Gate
**Stage:** 4 | **Priority:** P0 | **Effort:** 6h | **Depends on:** PT-S4-001, PT-S3-*

**Scope:** `src/preflight_gate.py::preflight_passes(context) -> tuple[bool, list[str]]`. Checks:
1. `final_signal.parquet` mtime < 24h
2. Config hash matches manifest hash
3. Broker ping succeeds
4. Prior-day reconciliation clean (no `unresolved_diff > tolerance`)
5. No CRITICAL/FATAL alerts in `monitoring_alerts.jsonl` from last 6h

Returns `(True, [])` on pass; `(False, [reasons])` on fail. Each check independently testable via mock context.

**Acceptance criteria:**
- Failing one check returns appropriate reason
- Broker connectivity timeout: 5s; returns False + reason (not raise)
- Config hash via `hashlib.sha256(json.dumps(config, sort_keys=True))`
- `--intents-only` runs preflight but does NOT halt execution
- All five checks logged to console + `preflight_log_<date>.json`

**Files touched:** new `src/preflight_gate.py`, `paper_trading/monitoring_alerts.jsonl`; modified `run_strategy.py` / controller.

---

## PT-S4-003: Kill Switch File Mechanism
**Stage:** 4 | **Priority:** P0 | **Effort:** 2h | **Depends on:** —

**Scope:** File at `paper_trading/state/KILL_SWITCH` (presence = halt) + audit log at `paper_trading/state/kill_switch_log.jsonl`.

Modified: preflight gate checks file presence at start; controller entrypoint checks before auto-submission.

**Acceptance criteria:**
- File present → Stage 4 halts within 100ms
- Manual creation logged (timestamp + reason)
- Removal logged
- `--intents-only` still works when kill switch active
- Clearing verified by absence + confirmation log

**Files touched:** new `paper_trading/state/KILL_SWITCH` (empty on first create), `paper_trading/state/kill_switch_log.jsonl`; modified `src/preflight_gate.py`.

**Notes:** Log entries: `{"timestamp", "action": "activate|clear", "reason", "operator": "manual|PT-S4-006"}`.

---

## PT-S4-004: Auto-Approval Gate
**Stage:** 4 | **Priority:** P0 | **Effort:** 5h | **Depends on:** PT-S4-001..003

**Scope:** `src/auto_approval_gate.py::auto_approve(context, previews, preflight_result) -> tuple[bool, str]`:
- Auto-approve IFF:
  1. preflight_result[0] == True
  2. All three previews status == "OK"
  3. Rebalance day (or override flag set)
- Machine-generated approval record to `approvals_<date>.jsonl`

Modified: controller replaces human Y/N prompt with `auto_approve()` in Stage 4 mode; new config key `stage4_mode: "auto"|"manual"`.

**Acceptance criteria:**
- Auto-approval granted only when all checks green
- Record contains: timestamp, reason_string, signal_hash, config_hash, three preview hashes
- Any "WARN" or "FAIL" → denial + reason logged
- Reason deterministic: "Auto-approved: preflight ✓, previews ✓, rebalance day"
- Denial → controller goes to `--intents-only` mode
- Idempotent logging (no duplicates on retry)

**Files touched:** new `src/auto_approval_gate.py`; new `paper_trading/reports/<date>/approvals.jsonl`; modified controller.

---

## PT-S4-005: `--intents-only` Mode
**Stage:** 4 | **Priority:** P0 | **Effort:** 4h | **Depends on:** PT-S3-*

**Scope:** CLI flag `--intents-only` on controller. When set:
1. Run all signal generation + rebalance + preview steps
2. Generate full intent bundle: `intents_<date>.json` (target weights, proposed trades, risk metrics)
3. Generate all previews
4. DO NOT submit orders
5. DO NOT update live portfolio state
6. DO NOT trigger "order filled" alerts

Order submission branch guarded: `if not args.intents_only: submit_orders()`.

**Acceptance criteria:**
- Intent bundle generated but no orders submitted
- `intents_<date>.json` contains target_weights, current_weights, proposed_trades, risk_metrics
- Dry-run matches what Stage 3 manual would accept
- Preflight runs but does NOT halt
- Kill switch still halts execution
- Combinable with other flags for testing

**Files touched:** modified `run_strategy.py` / controller; new `paper_trading/reports/<date>/intents_<date>.json`.

---

## PT-S4-006: Max Daily Loss / Drawdown Auto-Halt
**Stage:** 4 | **Priority:** P1 | **Effort:** 5h | **Depends on:** PT-S4-003, PT-S3-*

**Scope:** `src/drawdown_circuit_breaker.py`:
- `check_drawdown(blotter, ytd_limit=-0.30, daily_limit=-0.05) -> tuple[bool, str]`
- Computes YTD realized PnL from closed positions
- Computes daily PnL from today's fills
- If breached → auto-create `paper_trading/state/KILL_SWITCH` with reason
- Log to `paper_trading/circuit_breaker_log.jsonl`

Config keys: `hard_halt_drawdown_pct` (-0.30), `hard_halt_daily_loss_pct` (-0.05).
Modified: preflight includes circuit-breaker check; order submission loop checks after each fill.

**Acceptance criteria:**
- Uses realized PnL (not unrealized)
- YTD DD = `sum(realized_gain_loss) / initial_capital`
- Daily loss = sum of all fills today
- Breach creates kill switch with reason "YTD DD -31.2%" or "Daily loss -5.8%"
- Operator must manually clear
- Synthetic blotter test verifies trigger at threshold

**Files touched:** new `src/drawdown_circuit_breaker.py`; new log file; modified preflight gate + submission loop; modified config schema.

**Notes:** Thresholds tunable. Test with 10 synthetic losing trades.

---

## PT-S4-007: Alerting Integration
**Stage:** 4 | **Priority:** P1 | **Effort:** 6h | **Depends on:** PT-S4-002, 006

**Scope:** `src/alerting.py::send_alert(alert_type, message, severity)`. Types: preflight_failure, order_rejected, slippage_breach, drift_alert, kill_switch_activated, circuit_breaker_triggered. Channels: email (SMTP), desktop (pywin32 or win-notification), log file.

Config section `alerting`:
```yaml
alerting:
  email_enabled: true
  email_recipient: "user@example.com"
  email_smtp_server: "smtp.gmail.com"
  email_port: 587
  desktop_notifications_enabled: true
  alert_thresholds:
    slippage_pct: 0.02
    drift_pct: 0.05
```

**Acceptance criteria:**
- Alerts sent within 5s
- Email contains timestamp, type, message, severity
- Desktop notifications for CRITICAL
- Multiple recipients support
- Dry-run SMTP verifies email body
- Daily digest option (batch end-of-day)
- Mute list per alert type
- SMTP creds in `.env` or keyring, NOT config

**Files touched:** new `src/alerting.py`; new `paper_trading/config/alerting.json` template; modified preflight gate + circuit breaker + submission + kill switch; modified run config.

---

## PT-S4-008: Post-Run Reconciliation Report
**Stage:** 4 | **Priority:** P1 | **Effort:** 5h | **Depends on:** PT-S4-004, 005, PT-S3-*

**Scope:** `src/post_run_reconciliation.py::generate_reconciliation_report(date, context) -> str`. Outputs `paper_trading/reports/<date>/reconciliation.md` with sections:
1. Summary (# intended, # fills, % diff)
2. Intended vs executed (table: ticker, target_wt, actual_wt, diff_pct)
3. Slippage analysis (per-trade expected vs actual, aggregate $)
4. Drift report (post-execution weights vs targets)
5. Anomalies (rejected, partial fills, trades > 30min to fill, data gaps)
6. Status: CLEAN / MINOR_DRIFT / ALERT_REQUIRED
7. Next steps (if drift > 5%, recommend manual rebalance)

**Acceptance criteria:**
- Generated after every Stage 4 auto-run
- Intended weights from approval record JSON; actual from blotter
- Slippage = `(avg_fill - expected) × shares / order_value`
- Drift flags positions > 2% off target
- Status = ALERT_REQUIRED if drift > 5% or anomalies present
- Markdown with ASCII tables
- Metadata appended to `reconciliation_summary.jsonl`

**Files touched:** new `src/post_run_reconciliation.py`; new report + summary files; modified preflight gate (checks prior recon status).

**Notes:** Expected fill price = previous close. Report idempotent (overwrite same-date rerun).

---

## PT-S4-009: Canary Auto-Run
**Stage:** 4 | **Priority:** P1 | **Effort:** 4h | **Depends on:** PT-S4-001, 004, 008

**Scope:** `src/canary_mode.py`:
- `is_canary_mode_active(trading_day_count) -> bool`
- `apply_canary_caps(orders) -> filtered_orders`
- Tracks trading day count in `paper_trading/canary_state.json` (incremented daily)

Config:
```yaml
canary_mode:
  enabled: true
  duration_trading_days: 10
  max_single_order_notional_usd: 500
  max_orders_per_day: 5
  tracking_file: "paper_trading/canary_state.json"
```

Modified: submission loop applies canary caps; circuit breaker relaxes thresholds by 50% during canary.

**Acceptance criteria:**
- Active for first 10 trading days
- Orders > $500 notional filtered (logged as "canary_capped")
- Days > 5 orders: remaining queued for next day
- Daily metrics logged (orders submitted, total notional, max single, status)
- After 10 days: auto-disable; caps widen
- Operator can extend via config
- Unit test: 10-day mock, verify caps applied then released

**Files touched:** new `src/canary_mode.py`, `paper_trading/canary_state.json`; modified submission loop + circuit breaker; modified config.

---

## PT-S4-010: Stage 4 Runbook + Exit Criteria
**Stage:** 4 | **Priority:** P0 | **Effort:** 6h | **Depends on:** All S4

**Scope:** Final `PAPER_TRADING_OPS.md` section + `paper_trading/DEPLOYMENT_CHECKLIST.md`.

Sections in OPS runbook:
1. Stage 4 architecture overview (controller → preflight → approval → submission → reconciliation)
2. Deployment checklist (20 points: kill switch file, config validation, alerting tested, canary enabled, etc.)
3. Daily operations (logs, alert patterns, response playbooks)
4. Monitoring & alerting
5. Incident response (kill switch, rollback to Stage 3)
6. Rollback to Stage 3 (flip `stage4_mode: "manual"`)
7. Exit criteria to live capital:
   - 20+ trading days zero preflight failures
   - Reconciliation CLEAN for 10 consecutive rebalances
   - Circuit breaker never triggered
   - Canary metrics stable (avg slippage < 1%, no rejections)
   - Live backtest Sharpe > 0.60
   - 5 stress-test scenarios completed
8. Post-go-live monitoring

**Acceptance criteria:**
- Covers all Stage 4 subsystems
- Rollback procedure 2–3 steps
- Exit criteria specific, measurable
- Checklist executable (✓ mark each item)
- Incident response as decision trees
- Stress-test scenarios documented
- Runbook readable by non-programmer
- Versioned (v1.0, date, operator)

**Files touched:** modified `PAPER_TRADING_OPS.md`; new `paper_trading/DEPLOYMENT_CHECKLIST.md`.

---

# Dashboard Tickets (PT-D Series)

Paper trading operator visibility. Static HTML, reuses `dashboard.py` patterns.

## PT-D-001: Paper Trading Dashboard Entrypoint
**Stage:** D | **Priority:** P0 | **Effort:** 3–5d | **Depends on:** —

**Scope:** `paper_dashboard.py` at repo root. Function `build_paper_dashboard(as_of_date, data_dir, output_path, **kwargs)`.

**Inputs:** `paper_trading/` directory with `portfolio_state.jsonl`, `broker_positions.json`, `pending_orders.jsonl`, `fill_log.jsonl`, `reconciliation_log.jsonl`, `alerts.jsonl`.

**Output:** static HTML at `paper_trading/dashboard/<YYYY-MM-DD>/index.html` + rolling `latest.html`.

**Parameters:** `as_of_date` (defaults today), `highlight_alerts`, `compare_to_backtest` (optional backtest path).

**Technical:** Reuse COLORS, LAYOUT_DEFAULTS, `_apply_defaults` from `src/dashboard.py`. Six main chart objects (PT-D-002..007). Plotly.js CDN. No callbacks; pre-computed data embedded.

**Acceptance criteria:**
- Renders in browser without server
- Loads `portfolio_state.jsonl` and generates holdings/NAV/returns
- Handles missing data gracefully (show "No pending orders")
- Auto-generated after every paper run
- CLI flag `--gen-paper-dashboard` regenerates on demand

**Files touched:** new `paper_dashboard.py`.

---

## PT-D-002: Current Holdings View (Broker vs Target)
**Stage:** D | **Priority:** P0 | **Effort:** 2–3d | **Depends on:** PT-D-001

**Scope:** Side-by-side table: broker actual vs target. Columns sorted by `|drift|` desc: Ticker, Actual Wt %, Target Wt %, Drift % (red if > 2%), Actual Shares, Target Shares, Diff Shares, Notional to Trade.

Summary stats: Total Drift ($), Max Single Drift, Rebalance Notional.

**Acceptance criteria:**
- Correct calculations
- Sorted by |drift| desc
- Handles missing tickers (in signal but not broker → target_shares with diff_shares > 0)
- Zero/negative portfolio values handled

**Files touched:** new function in `paper_dashboard.py`.

---

## PT-D-003: Pending Orders View
**Stage:** D | **Priority:** P0 | **Effort:** 2d | **Depends on:** PT-D-001

**Scope:** Table of open/partial orders. Columns: Order ID, Symbol, Side, Qty, Submitted At, Latest Status, Qty Filled, Qty Remaining, Time in Status, Price.

Alerts: green (filled/recent), yellow (pending > 30min), red (canceled/error). Red highlight on "Time in Status" > 1 hour.

Summary: Orders Submitted (24h), Avg Time to Fill, % Partial Fills. "All orders resolved" if none pending.

**Acceptance criteria:**
- Sorted newest submitted first
- Status colors correct
- Handles missing `pending_orders.jsonl`

**Files touched:** new function in `paper_dashboard.py`.

---

## PT-D-004: Fills View + Slippage Analysis
**Stage:** D | **Priority:** P1 | **Effort:** 2–3d | **Depends on:** PT-D-001

**Scope:** Table of last 50–100 fills (configurable): Fill ID, Symbol, Side, Qty, Exec Price, Expected Price, Slippage Bps, Fill Time, Notional.

Line chart of rolling 20-fill mean slippage. Summary stats: mean bps, p95 bps, total notional traded, implied cost %.

Alerts: amber if mean > 3 bps; red if p95 > 10 bps.

**Acceptance criteria:**
- Correct slippage calculations
- Rolling stats update with new fills
- Fallback for missing expected_price (use VWAP or day open)

**Files touched:** new functions in `paper_dashboard.py`.

---

## PT-D-005: Reconciliation Status (Red/Yellow/Green)
**Stage:** D | **Priority:** P0 | **Effort:** 2d | **Depends on:** PT-D-001

**Scope:** Five tiles for health checks:
1. **Prior-Day Reconciliation** — green (reconciled within 1h), yellow (> 2h late), red (unreconciled/stale > 24h)
2. **Drift Threshold** — green (< 1%), yellow (1–3%), red (> 3%)
3. **Preflight Gate** — green (all pass), yellow (some fail but trading allowed), red (critical failed)
4. **Kill Switch Status** — green (off), red (on)
5. **Broker Connectivity** — green (last ping < 5min), yellow (5–30min), red (> 30min)

Layout: five tiles in row; warning banner at top if any RED. Hover details.

**Acceptance criteria:**
- All five tiles update from respective sources
- Colors match operational intent
- Missing data → assume RED

**Files touched:** new function in `paper_dashboard.py`.

---

## PT-D-006: P&L and Metrics Timeseries
**Stage:** D | **Priority:** P1 | **Effort:** 3d | **Depends on:** PT-D-001

**Scope:** Four line charts:
1. NAV Over Time — paper NAV, optional backtest overlay, go-live marker
2. YTD Cumulative Return — paper vs SPY
3. Rolling Sharpe (252-day) — with y=1 threshold
4. Max Drawdown (YTD + All-Time)

Summary KPI tiles: Current NAV, YTD Return %, YTD Sharpe, Days Since Go-Live, Max DD (All-Time), Max DD (YTD).

**Acceptance criteria:**
- NAV matches `portfolio_state.jsonl`
- Sharpe calc matches backtest dashboard
- Optional backtest overlay works
- Charts responsive

**Files touched:** new functions in `paper_dashboard.py`.

---

## PT-D-007: Alerts & Incidents Log View
**Stage:** D | **Priority:** P1 | **Effort:** 2d | **Depends on:** PT-D-001

**Scope:** Table of latest 30 alerts from `alerts.jsonl`. Columns: Timestamp, Severity, Type, Message, Context.

Row color by severity: green (INFO), yellow (WARN), orange (ERROR), red (CRITICAL). Latest CRITICAL/ERROR pinned on top.

HALT handling: if unresolved HALT, red banner above table: "TRADING HALTED — <reason> — <timestamp>". If resolved, green: "Halt resolved at <timestamp>; trading resumed."

Optional dropdown filters (severity, type). Summary: Critical (24h), Warnings (7d). "All systems normal" if none.

**Acceptance criteria:**
- Latest 30 events in order
- HALT surfaces prominently
- Row colors match severity
- Handles missing `alerts.jsonl`

**Files touched:** new function in `paper_dashboard.py`.

---

# Ticket Summary

**Total: 68 tickets** (post-audit additions: PT-X-015, PT-S3-013, PT-S3-014)
- Cross-cutting (PT-X): 15 tickets (~59 effort points)
- Stage 1 (PT-S1): 12 tickets (~44 hours)
- Stage 2 (PT-S2): 10 tickets (~28.5 hours)
- Stage 3 (PT-S3): 14 tickets (~90 hours, with Phase A/B split)
- Stage 4 (PT-S4): 10 tickets (~47 hours)
- Dashboard (PT-D): 7 tickets (~16–20 days)

**Build timeline (coding only):** ~4 weeks on the optimistic path (Week 1: X + S1; Week 2: S2; Week 3: S3 code; Week 4: S4 canary wiring + D P0s).
**Operational timeline:** earliest live eligibility is materially longer: Stage 3 must span 2 monthly rebalance cycles, and Stage 4 requires 3 provisional months plus 6 full-threshold months. On the optimistic path that puts the earliest go-live decision around week 50+ from kickoff.

**Go-live prerequisites:** all 8 gates in [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md) PASS for 6 consecutive full-threshold months after the Stage 4 months 1-3 provisional window.

## Non-Goals (Do Not Build)

Explicitly out of scope for v1:
- Options or short positions (equity long-only)
- Intraday trading (monthly rebalance only)
- Automation before Stages 1–3 prove stable
- Bypassing any preflight gate
- Coupling to `run_strategy.py` main loop or options Tradier poll
- Operating during market stress / data gaps (skip, don't degrade)

## References

- [PAPER_TRADING_PLAN.md](PAPER_TRADING_PLAN.md) — Master plan, stages, dependency graph, timeline
- [PAPER_TRADING_CONTRACT.md](PAPER_TRADING_CONTRACT.md) — Pre-committed gates, rollback triggers, stage exits
- [PAPER_TRADING_OPS.md](PAPER_TRADING_OPS.md) — Daily/weekly/monthly runbook, failure modes, troubleshooting
- [CURRENT_BASELINE.md](CURRENT_BASELINE.md) — Adopted baseline performance
- [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) — Signal timing contract
- [DEVELOPMENT_PROCESS.md](DEVELOPMENT_PROCESS.md) — Research run classes and promotion gates
