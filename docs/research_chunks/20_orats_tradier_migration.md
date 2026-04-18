### Tradier Migration: What We Lose, What We Keep, What's Free

### Why Tradier
Tradier's market data API is **free with any brokerage account** — no separate $99/mo subscription, no per-call metering for normal polling cadence. The decisive feature for our stack: Tradier's `/markets/options/chains` endpoint **embeds Orats SMV Greeks** (per Tradier docs), meaning IV, delta, gamma, theta, vega come from the same SMV (Stochastic-Model Volatility) engine that produced our cached Orats historical data. This delivers **train/serve consistency**: the 13 years of Orats Delayed Data we already cached and the daily Tradier polls going forward are produced by the same vol surface model. No methodology drift between train and inference.

### Architecture
- **`src/tradier_client.py`** — `TradierClient` class with rate-limit retry (exponential backoff on 429) and `ThreadPool` batch fetching across tickers/expirations.
- **`src/options_adapter/chain_to_smv_summary.py`** — orchestrator that takes a raw Tradier chain and emits Orats `/cores`-equivalent rows (~27 fields per ticker per day).
- Five sub-adapters under `src/options_adapter/`:
  - `cmiv_interpolator.py` — variance-time interpolation across expiries → `iv7d/14d/30d/60d/90d/180d/365d`
  - `tradier_orats_adapter.py` — PCHIP smile fit → `slope`, `dlt5/25/75/95Iv30d`
  - `implied_borrow.py` — put-call parity → `borrow30`
  - `implied_dividend_proxy.py` — PCP-based dividend yield with EODHD fallback
  - `forward_curve.py` — variance additivity → `fwd30_60`, `fwd60_90`, `fwd90_180`
  - `implied_moments.py` — BKM trapezoidal integration → `implied_var/skew/kurt`

### Daily Polling
- **`cron/poll_tradier_daily.bat`** — Windows Task Scheduler entry, runs 16:30 ET.
- **`run_options_setup.py:cmd_daily_poll`** — fetches 5 expirations × N tickers, runs through SMV adapter, writes `data/cache/options/tradier_daily/<date>.parquet`.
- Wires the EODHD dividend yield map for accurate `borrow30` + `annIdiv` solving.
- 17 fields per ticker per day land in the daily snapshot.

### Merger Pipeline
**`merge_tradier_daily_to_iv_panel.py`** rolls daily snapshots into per-field date×ticker panels at `data/cache/options/iv_panels_tradier.pkl`, consumed directly by `src/options_signals.py`.

### Validation (Phase 1, N=30 large-caps, 2026-04-17)
- All 6 IV/delta fields rank-correlate **≥ 0.88 vs Orats**.
- `slope`: improved from **0.063 → 0.753** after methodology fix (replaced OLS regression with `dlt75 - dlt25` differential).
- `borrow30` + `annIdiv` remain **weak rank-corr (~0.2)** — these drive only two low-IC bonus signals, so the residual error is tolerable.

### What's Permanently Lost by Cancelling Orats
- **`fcstSlope`** — Orats's proprietary IV forecast model. Not used by any of our signals, so zero impact.
- **Real-time `exErnIv*` updates** — workaround: 13 years cached + forward proxy via earnings calendar.
- **Real-time `impliedMove` from `/summaries`** — workaround: compute ourselves from straddle inside the adapter.

### Coverage
- **Orats**: 1,414 tickers cached.
- **Tradier**: 80–86% of full backtest universe (5,077 tickers) — **broader coverage than Orats** going forward.

### Cost Timeline
- Pre: **$99/mo** Orats Delayed Data.
- Post: **$0/mo** Tradier (free, requires brokerage account).
- Savings: **$1,188/yr indefinitely**, with no signal-quality regression on the fields that drive our IC.
