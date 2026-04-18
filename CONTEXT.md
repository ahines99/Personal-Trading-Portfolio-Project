# Personal Trading Portfolio — Master Context

> Paste this file into any new chat to give Claude full project context.
> Compiled 2026-04-17 from 24 parallel repo-exploration passes. ~35K LOC, ~27 GB cache, 13 yrs data.
> **Audited 2026-04-17** via 20 parallel verification passes; corrections applied inline (see §0 changelog).

---

## 0. Audit corrections (2026-04-17)

**2026-04-17 evening update — honest baseline recovered to 22.58% CAGR / 0.715 Sharpe via `EMPTY_DENY_LIST=1`.** The 101-entry `_DENY_LIST` in [model.py:800](src/model.py#L800) was the primary regression cause. Clearing it recovers +377 bps CAGR / +0.093 Sharpe in a single env-var flip. Full tearsheet: [results/_repro_fixed2_v3/tearsheet.csv](results/_repro_fixed2_v3/tearsheet.csv). Remaining gap to Fixed2 (25.67%) is −309 bps, likely from feature-order non-determinism / XGBoost version drift / missing panel features — under investigation. See [REPRODUCTION.md](REPRODUCTION.md) for the reproducible command. Two prior theories (EODHD sentiment lookahead, fundamentals crash) are DISPROVEN via code inspection.

20 verification agents cross-checked every claim. Overall: most claims correct; 11 corrections applied. Notable:

| Area | Claim → Actual |
|---|---|
| §5 module lines | `options_signals.py` 512 → **837**; `chain_to_smv_summary.py` 297 → **564**; +4 files added (earnings_iv_crush, forward_curve, implied_moments, exern_iv_extractor) |
| §8 ensemble weights | 40/25/35 → **LGBM 0.30 / XGB 0.25 / Ridge 0.25 / MLP 0.20** |
| §9 quality_percentile | 70th → **50th** (portfolio.py:399 default) |
| §16 _DENY_LIST | 93 → **101 entries** (Phase E prune partially reverted at model.py:852) |
| §22 bottom-decile range | 365–480 → **366–480** (exact gains listed) |
| §24 "unused" claims | multiple WRONG: `load_insider_transactions`, `load_earnings_calendar_eodhd`, `load_fred_macro`, `UNRATE`, `VIX3M`, `before_after_market`, EODHD `count` — all ACTUALLY USED. Truly unused: NAPM/ICSA/PERMIT/VIX9D/VIX6M/institutionCount/targetMeanPrice/currentPrice/numberOfAnalysts |
| §30 forward_window | 7 only → CLI default 7 (prod path); WalkForwardModel constructor default is 5 |
| §34 cron | "NOT registered" → **IS registered** (Windows TaskScheduler `TradierDailyPoll` Ready, 16:30 ET) |
| §41 _sni variants | 13 → **12** (amihud is NOT sector-neutralized) |
| §63 OPTIONS_* defaults | OPTIONS_MIN_COVERAGE/WINSORIZE claimed 0.30/0.01 → code defaults are **0.0** (set explicitly). FINNHUB_MIN_COVERAGE is doc-only, not wired. |
| §64 oil_beta_signal | "negated" → **NOT negated** in alt_features.py:1403. Either real bug or doc error. |

Verified clean: §21 horizon CSV (exact match), §23 C&Z IC top-20 (exact match), §25/§26/§27 sweeps/regime/annual CSVs (exact match), §31 costs $40,002 (exact match), §32 FF6 all coefficients (exact match), §42-§48 features.py line anchors (all correct), §50-§57 alt_features.py line anchors (all correct), §58-§60 specialty modules (28/28 correct), §14 tearsheet columns (33 actual vs 29 claimed — superset), all recent commit hashes (5/5 correct), all cache filename patterns (8/8 correct).

---

## 1. Identity in one breath

Personal ML-driven **long-only US equity** portfolio. Monthly rebalance, 20–50 positions selected from a 3,000-ticker liquidity-screened universe (2013‑01‑01 → 2026-present). LightGBM + XGBoost + Ridge ensemble on 60+ engineered features. Runs on user's own capital (~$15–50K AUM; do NOT propose hedge-fund infrastructure). Primary data source: EODHD $20–60/mo, with free EDGAR/FRED/CBOE + optional Tradier (options). Target: beat SPY after tax and costs.

---

## 2. Headline numbers (as of 2026-04-17)

| Run | CAGR | Sharpe | Alpha vs SPY | Max DD | Beta | Notes |
|---|---|---|---|---|---|---|
| **Honest baseline** (`EMPTY_DENY_LIST=1`) | **22.58%** | **0.715** | +8.60% | -54.76% | 1.491 | PSR 0.995 / DSR 0.947. Reproducible. See REPRODUCTION.md |
| **Master default** (deny-list active) | 18.81% | 0.622 | +4.82% | -59.23% | ~1.5 | lower bound; 101-entry deny-list over-prunes |
| **Fixed2 claim** (Apr 15, uncommitted) | 25.67% | 0.785 | — | -52.88% | — | partial target; -309 bps gap remaining under investigation |
| **Phase5 / Phase 1.8d** (production_config.json) | 14.24% | 0.655 | — | -36.57% | 0.904 | N=50, tilt=0.35, fwd=21 — risk-reduced variant |
| **SPY benchmark** | 13.99% | — | — | — | 1.0 | 2013–2026 |

**Deflated Sharpe** 0.947 (PSR 0.995) on honest baseline run. **Capacity**: viable to ~$100M AUM. After-tax CAGR 16.18% (Roth IRA recommended to avoid 32.8% drag).

**Resolved 2026-04-17 evening:** `EMPTY_DENY_LIST=1` env-var gate in [model.py:859](src/model.py#L859) recovers 22.58% CAGR. Two prior regression theories (EODHD sentiment lookahead, fundamentals crash) are DISPROVEN. Remaining 309 bps to Fixed2's 25.67% likely from feature-column-order non-determinism ([model.py:359](src/model.py#L359), [alt_features.py:173](src/alt_features.py#L173) — dict iterations without `sorted()`), XGBoost 3.2.0 vs pinned `<3.0.0` drift, MLP unseeded `torch.randperm` ([model.py:1937](src/model.py#L1937)), and/or Fixed2's uncommitted feature panel (had cluster: mom_dbc/gld/spy 12-1, alt_initial_claims, SRAF, alt_dxy_beta, z_idiovol_21d, vol_regime, breadth; lacked cluster: 18 alt_q_*, beat_streak, surprise_consistency, hmlint, days_to_earnings).

---

## 3. Tech stack

- **ML**: lightgbm ≥4.0 (LambdaRank NDCG), xgboost ≥2.0, catboost ≥1.2, torch 2.6 (CUDA 12.4), optuna
- **Data**: pandas 2.x, numpy, yfinance, scikit-learn, statsmodels (Newey-West)
- **Dashboards**: Plotly (static HTML) + Dash (interactive app)
- **Platform**: Windows 11 + Git Bash; paths use Unix-style forward slashes inside Python (`pathlib` bug on Windows backslashes fixed in `lazy_prices_features.py:236`)
- **Env**: `EODHD_API_KEY`, `FINNHUB_API_KEY` in `.env`

---

## 4. Repository layout

```
Personal-Trading-Portfolio-Project/
├── src/                        # core library (~23K LOC, 35 modules)
├── data/cache/                 # ~27 GB cached data (pickles + parquet)
│   ├── api/                    # EODHD/Finnhub pickles (v2 versioned)
│   ├── edgar_text/             # 94,768 10-K/10-Q JSON files
│   ├── chen_zimmermann/        # 1.3 GB signed_predictors parquet
│   ├── sraf/                   # Loughran-McDonald pre-parsed summaries
│   ├── options/                # Tradier daily parquet + Orats IV panels
│   └── permno_link/
├── results/                    # every run's CSVs + dashboard.html
│   ├── archive/                # 100+ tearsheet CSVs from prior experiments
│   ├── _iso_*/, _phase_*/      # isolated/phase experiment outputs
│   └── validation/             # Tradier-vs-Orats validation HTML/CSV
├── logs/                       # 104+ per-experiment .log files
├── baselines/                  # immutable snapshots via snapshot_baseline.py
├── notebooks/                  # chain_to_smv_summary.py helper
├── cron/                       # register_cron.bat + poll_tradier_daily.bat
├── docs/test_suite_design.md
├── *.py                        # ~40 run_*/test_* orchestration scripts
├── run_strategy.py             # MAIN ENTRY (1528 lines, ~80 CLI flags)
├── README.md, REPRODUCTION.md
└── requirements.txt, .env.example
```

---

## 5. `src/` module catalog

| Module | Lines | Purpose |
|---|---|---|
| `data_loader.py` | 1121 | EODHD universe + price loader, delisting masks, quality filters |
| `universe_builder.py` | 252 | Wikipedia + iShares CSV → `data/ticker_master.csv` |
| `sector_mapper.py` | 451 | EDGAR SIC → GICS sector (10 sectors + Unknown) |
| `api_data.py` | 1034 | EODHD fundamentals/dividends/sentiment + Finnhub |
| `alt_data_loader.py` | 1610 | EDGAR XBRL facts, FRED (9 macro), VIX term, earnings (yf+EODHD), insider, analyst, institutional, cross-asset (DXY, Cu/Au, oil, ETFs, OAS, EBP, Treasury curve) |
| `alt_features.py` | 2056 | ~80 alt-data signals (value/quality/distress/macro-betas/earnings/FINRA short) |
| `features.py` | 1530 | Core 24-signal composite: momentum, MR, vol, Amihud, skew/coskew, wavelets |
| `model.py` | 3497 | WalkForwardModel, feature matrix build, labels, LGBM/XGB/Ridge/MLP ensemble, meta-labeling, era-weighting, Optuna |
| `portfolio.py` | 1153 | `build_monthly_portfolio()`, vol targeting, BSC scaling, credit overlay, sector caps |
| `tax_aware.py` | 469 | HIFO tax-lot tracker, wash-sale, ST/LT classification |
| `backtest.py` | 621 | T+1 event-driven loop, tiered cost model (1/3/8/20 bps), delisting, stops |
| `metrics.py` | 920 | Tearsheet, Sharpe/Sortino/PSR/DSR, FF6 regression, after-tax modeling |
| `dashboard.py` | 654 | Static HTML+Plotly → `results/dashboard.html` |
| `regime.py` | 373 | Vol+trend combined regime (4 states), stress test periods |
| `robustness.py` | 365 | Block bootstrap, permutation, capacity analysis |
| `cz_signals.py` | 522 | Chen-Zimmermann: coskew, mom_season, payout_yield, xfin, cfp, operprof_rd, tax, deldrc |
| `factor_momentum_data.py` | 269 | Ken French FF6 daily factors + 12-1 momentum |
| `factor_momentum_features.py` | 210 | Per-stock rolling factor betas × factor momentum (Ehsani-Linnainmaa 2022) |
| `cmiv_interpolator.py` | 417 | Constant-maturity 30/60/90d ATM IV from options chain |
| `cross_asset_leadlag.py` | 240 | Hong-Stein lead-lag: oil/Cu/Au/DXY/HY → 15 features |
| `finra_short_interest.py` | 415 | FINRA bi-monthly short interest → wide panel (publication_date pivot, not settlement) |
| `implied_borrow.py` | 369 | Put-call parity → borrow30 (clip ±0.50→tightened ±[-0.05, 0.50]) |
| `implied_dividend_proxy.py` | 459 | PCP → annual implied div (~75-85% of Orats accuracy) |
| `insider_eodhd_loader.py` | 286 | EODHD insider transactions (form 4 detail) |
| `opportunistic_insider_signal.py` | 373 | Cohen-Malloy-Pomorski routine/opportunistic classifier |
| `lazy_prices_downloader.py` | 516 | SEC EDGAR 10-K/10-Q downloader, 9 req/sec, Item 1A+7 extraction |
| `lazy_prices_features.py` | 581 | TF-IDF cosine text similarity (lower = more change = bearish) |
| `lm_sentiment.py` | 615 | Loughran-McDonald dict from pysentiment2/SRAF, scored on Item 7 |
| `sraf_sentiment.py` | 241 | SRAF 10X pre-parsed counts (197 MB, 1.25M filings) |
| `options_signals.py` | 837 | 15 options signals (5 C&Z high-IC + 7 academic + 3 Orats bonus), LAG=1 day |
| `orats_loader.py` | 403 | Orats `/datav2/cores` API/FTP loader, parquet cache |
| `tradier_client.py` | 382 | Tradier REST client, `greek_smv_vol` preferred over `greek_mid_iv` |
| `tradier_orats_adapter.py` | 391 | Tradier chain → Orats `/cores` schema (slope, delta buckets via PCHIP) |
| `earnings_iv_crush.py` | 191 | IV crush around earnings (post-audit addition) |
| `options_adapter/chain_to_smv_summary.py` | 564 | Master orchestrator: cmiv + delta_buckets + borrow + dividend |
| `options_adapter/forward_curve.py` | 123 | Forward-curve construction for options pricing |
| `options_adapter/implied_moments.py` | 291 | Higher-order implied moments from options chain |
| `options_adapter/exern_iv_extractor.py` | 407 | IV extraction for extended hours / external sources |
| `pandas_compat.py` | 23 | StringDtype patch for old pickle loads |

---

## 6. Pipeline (run_strategy.py order)

Line anchors in `run_strategy.py`:

1. **CLI parse** (~70-420): ~80 flags covering universe, features, model, portfolio, overlays, output
2. **Prices** (619): `load_prices(universe_size=3000, dynamic_universe=True, min_adv=500K)` — 3000 live-screened tickers, SPY force-injected at 643 (fixes regime detector fallback)
3. **Alt data** (684): EDGAR fundamentals + extra, FRED (9 series), VIX term, EODHD earnings, insider/analyst/est/inst, DXY/VVIX/oil/Cu/Au/ETFs/OAS/EBP/Treasury
4. **Build alt features** (730): `build_alt_features()` → ~80 signal panels
5. **C&Z signals** (799): price + accounting anomalies if `--use-cz-signals`
6. **Options signals** (865): lazy-import; reads `data/cache/options/iv_panels_orats.pkl`
7. **Credit overlay stash** (890): extract NFCI+HY-OAS BEFORE `fred_df` deletion (906)
8. **Core signals** (916): `build_composite_signal()` with investable mask (top-1500 by ADV)
9. **Labels** (995): 5-grade LambdaRank or rank-targets; horizon = `--forward-window` (default 7d)
10. **Optuna** (1040) + **Ensemble** (1072): `WalkForwardModel.fit_predict()` → ranked predictions
11. **Blend** (1084): 70% ML + 30% 12-1 momentum, rank-normalized
12. **Portfolio** (1142): `build_monthly_portfolio()` — quality gate, earnings blackout, sector caps, mega-cap force, SPY core
13. **Vol targeting** (1208) / **BSC scaling** / **credit overlay** (1229)
14. **Backtest** (1269): `run_backtest()` T+1 fills, tiered costs, delisting, stops
15. **Tearsheet + analysis** (1323): 15+ CSVs → `results/*.csv`
16. **Dashboard** (1488): `results/dashboard.html`

---

## 7. Feature catalog

### Core (from features.py::build_composite_signal)
Momentum (m5/m10/m21/m63/m126 skip-21), mean-reversion (z-score, RSI, Bollinger), volatility (realized, regime, vol-of-vol), Amihud illiquidity, idiosyncratic vol, price/52w-high, MACD, efficiency ratio, OBV, VPT, obv signal, sector-relative momentum, short-term reversal, volume-confirmed reversal, realized skew, coskewness (Harvey-Siddique, 252d window), semi-beta (4 components: N/P/M+/M-), tail dependence, signed jump, downside/upside vol ratio, downside beta spread, Kumar lottery composite, breadth (pct>200MA, new_highs-lows, AD ratio, breadth_z), wavelet band energy (FFT 256d, chunked 500 tickers to avoid 30GB peak).

**Sign convention**: EVERY signal ranked so **higher = higher expected return** (mean-reversion negated, vol negated, MAX negated, etc.)

### Alt data (from alt_features.py)
- **Fundamentals (EDGAR)**: book_to_market, roe, gross_margin, asset_growth (-), leverage (-), eps_trend, gross_profitability, accruals (-), NOA (-), net_share_issuance (-), op_profitability, current_ratio_chg
- **Tier3 academic**: earnings_yield, sales_yield, fcf_yield, ebit_ev, cbop, piotroski_f_score, EAR, beat_streak, surprise_consistency
- **Macro broadcasts** (z-scored, clipped ±4): yield_curve, hy_spread, vix_macro, fed_funds, unemployment, breakeven_inflation, ism_pmi, initial_claims, building_permits
- **VIX term**: term_slope, vix_percentile, vix_change_5d
- **Macro betas** (cross-sectional): dxy_beta, pc2_slope_beta, cyclicality_cu_au, credit_beta, inflation_sensitivity, oil_beta, ebp_beta, sector_oas_momentum
- **Earnings**: earnings_surprise, days_to_earnings, EAR_timing, beat_streak, surprise_consistency (all Phase-D gated by `ENABLE_PHASE_D_SIGNALS=1`)
- **Insider**: activity, net_buy, cluster_buy; opportunistic (CMP classification)
- **Analyst**: revisions, coverage; target_upside, rating
- **Distress**: chs_distress (Campbell-Hilscher-Szilagyi), naive_dtd (Bharath-Shumway), altman_z
- **Text**: lazy_prices (item7, item1a, avg), lm_sentiment, sraf_sentiment
- **FINRA short interest**: delta_si_21d, delta_si_63d, si_level (pivoted on publication_date to fix 2-day lookahead)
- Most with `_sni` sector-neutralized variants

### C&Z (cz_signals.py)
Price: coskewness, coskew_acx, mom_season. Accounting: payout_yield, net_payout_yield, xfin, cfp, operprof_rd, tax, deldrc.

### Factor momentum (Ehsani-Linnainmaa)
Per-stock 252d betas to FF6, × factor 12-1 momentum → composite or 6 granular signals.

### Options (options_signals.py, 15 total, LAG=1)
C&Z high-IC: dCPVolSpread (0.375), SmileSlope (0.369), cpVolSpread, dVolCall, dVolPut. Academic: iv_rank, rv_iv_spread, variance_premium, iv_term_slope, risk_reversal_25d, crash_risk, oi_concentration. Orats bonus: borrow30, dividend_surprise, etf_skew_relative.

### Cross-asset lead-lag
oil/copper/gold/DXY/HY-OAS × {5, 10, 20} day horizons = 15 features.

### Deny-list (model.py:800)
~93 features auto-excluded (zero LGBM importance in prior runs: many `_sn`/`_sni` variants, wavelet components, low-IC alt signals). Overridable via `EXTRA_DENY_FEATURES=f1,f2` or `EMPTY_DENY_LIST=1`.

---

## 8. Model architecture (model.py)

**WalkForwardModel** (class ~1298+): expanding/rolling window, `max_train_days=756`, `embargo_days=5` (AFML), retrain every `retrain_freq=63` (quarterly per GKX 2020).

**Ensemble** (heads + weights, verified 2026-04-17 audit from model.py:1351-1354): LGBM 0.30 + XGBoost 0.25 + Ridge 0.25 + MLP 0.20 (4 heads sum to 1.0). Optional Huber + Quantile + CatBoost heads available. MLP is 3-layer PyTorch 128→64→32→1 on GPU with per-date rank targets. (Earlier docs claimed 40/25/35 — that was aspirational, not code.)

**Meta-labeling**: stage-2 LightGBM binary classifier (top-quintile AND y>median) as reject gate (line 2751).

**Era-weighting** (Phase E): rolling per-model Spearman IC buffer → dynamic 50/50 mix of fixed weights + era weights.

**Calibration**: confidence = 1 - mean pairwise rank diff across heads; scores adjusted `(scores-0.5) × (0.3 + 0.7×conf)`.

**Caching**: cache-key MD5 includes panel shape, date range, ALL constructor hyperparams, content hash, `EXTRA_DENY_FEATURES`, `DENY_LIST_HASH_SUFFIX`. Pickle at `data/cache/feature_panel_{hash}.pkl` + `ml_predictions_{hash}.pkl`.

**Memory opts** (recent commits): `_f32()` float32 construction, `ranked_signals` float32, `size_neutralize` OLS in float32, inf replacement done once post-build, MultiIndex C-order ravel.

---

## 9. Portfolio construction (portfolio.py)

`build_monthly_portfolio()` (line 366): 3rd trading day of month (day_offset=2), long-only, signal-weighted with `ranks**concentration` (default 1.0), cross-sectional rank→normal CDF transform.

**Constraints**: n_positions=20 default (SWEEP-1 winner at 17.81% CAGR; prod is 50), max_weight=10%, max_sector=35%, min_position=2%, min_ADV=$5M, max_stock_vol=60%, quality_percentile=**50th** (portfolio.py:399 default — SWEEP-2U tested 70th as a winner but default remained 50th).

**Overlays**:
- **Quality tilt**: soft multiplicative `1 + 0.35×quality_z` (Asness 2018 QMJ)
- **Vol targeting**: 0.16–0.40 target, leverage [0.8, 1.3], vol clip [0.08, 0.25] (Moreira-Muir 2017)
- **BSC scaling**: Barroso-Santa Clara strategy-own-vol, [0.5, 1.5] leverage
- **Credit overlay**: NFCI + HY-OAS risk-off switches (default thresholds 0.0 / 5.0%)
- **Regime-aware N**: bear markets compress to 8-10 positions
- **Mega-cap force**: top-10 SPY names at 3% if ranked in top 40%
- **SPY core + satellite**: `--spy-core 0.0–1.0` carves baseline SPY allocation

**Tax-aware** (tax_aware.py, optional): HIFO lot selection (ST-losses → LT-losses → LT-gains → ST-gains), 31-day wash-sale blacklist, defer-winner rule for positions <60d from LT (ST 32.8%, LT 18.8% defaults).

---

## 10. Backtest & metrics

**Backtest** (backtest.py): T+1 fills at open, mark-to-close. **Tiered costs** (commit 2026-04-13 retail-calibrated):
- ADV ≥ $20M: 1 bps/side
- $5-20M: 3 bps
- $1-5M: 8 bps
- <$1M: 20 bps

Delisting exits at last known price (~70% acquisition-value proxy, not $0). Stop-loss 15% active between rebalances. Drawdown halt sells 50% of positions.

**Metrics** (metrics.py): CAGR, Sharpe, Sortino, Calmar, IR, Max DD, PSR (skew+kurtosis corrected), DSR (Bailey-Lopez de Prado multi-testing), FF6 regression with Newey-West HAC (lag=⌊n^(1/3)⌋), rolling beta, stress periods (COVID, GFC, SVB, 2025 tariff shock), `after_tax_estimate()` with ST/LT decay model + TLH benefit + account recommendation.

---

## 11. Caching conventions

**Universal**: `data/cache/*.pkl` (pickle) or `*.parquet`. Version key: `_ALT_DATA_CACHE_VERSION = "v2"` (bumped 2026-04-05).

**Filename patterns**:
- `eodhd_prices_{start}_{end}_{n_tickers}_{md5}.pkl` (fuzzy-match on ticker count ±10)
- `top_universe_{n}_{start}_{end}_{min_price}_{min_adv}_v3_us_only.pkl`
- `edgar_fundamentals_{start}_{end}_{max_tickers}_v2raw.pkl`
- `fred_macro_{start}_{end}_v2.pkl`, `ff6_factors_{start}_{end}_v1.pkl`
- `feature_panel_{hash}.pkl`, `ml_predictions_{hash}.pkl`
- Options: `iv_panels_orats.pkl`, `tradier_daily/{YYYY-MM-DD}.parquet`

**Invalidation**: `--no-cache` sets random `DENY_LIST_HASH_SUFFIX`; `--clear-cache` deletes `feature_panel_*.pkl` + `ml_predictions_*.pkl`. Cron jobs update options panels; alt loaders auto-rebuild on version bump.

**Cleanup utility**: `clean_stale_caches.py --list / --archive --keep H1,H2 / --restore HASH`.

---

## 12. Environment variables

Set in `.env` or shell:

| Var | Purpose |
|---|---|
| `EODHD_API_KEY` | required for prices/fundamentals/sentiment |
| `FINNHUB_API_KEY` | optional, secondary sentiment/estimates |
| `ENABLE_PHASE_D_SIGNALS` | default "1"; gates beat_streak/surprise_consistency/EAR_timing |
| `DISABLE_DIVIDEND_YIELD` | gate to skip redundant div feature when CZ payout_yield active |
| `DISABLE_CASH_BASED_OP_PROF` | gate cash-based op profitability (Ball et al. 2016) |
| `DISABLE_CO_SKEW_252` | bypass 252d coskewness (memory pressure) |
| `EXTRA_DENY_FEATURES` | comma-separated features added to `_DENY_LIST` |
| `EMPTY_DENY_LIST` | "1" = clear 93-entry default deny-list (ablation) |
| `DENY_LIST_HASH_SUFFIX` | force cache-key mismatch (isolation tests) |
| `LAG_OPTIONS_SIGNALS` | default "1"; options signal shift days |
| `PYTHONIOENCODING=utf-8` | set by subprocess orchestrators for Windows |

---

## 13. Orchestration scripts (root)

| Script | Purpose |
|---|---|
| **`run_strategy.py`** | MAIN pipeline entry, 1528 lines, ~80 CLI flags |
| `test_suite.py` | ~125 regression tests across A-H/P/L/R/Z/FA-FH/QA-QE families |
| `test_horizons.py` | Academic horizon study (Baz 2015, B-SC 2015, Lopez de Prado) |
| `poc_horizons.py` | Fast fw=1/5/10/21 IC comparison (~50 min) |
| `run_phase5_production.py` | Locked Phase 1.8d production run → phase5_summary.md + production_config.json |
| `run_signal_test_suite.py` | 8 additive-signal A/B tests w/ cache-hash tracking |
| `run_clean_retest.py` | 5-config recovery validation after yfinance baseline |
| `run_isolated_retest.py` | 19 feature-by-feature isolation tests (baseline/cz/phase_d/options groups) |
| `run_cz_research.py` | Chen-Zimmermann 209-signal IC discovery |
| `run_phase_b.py`, `run_phase_b_fixed.py` | SRAF/FINRA/lazy/LM feature flip tests |
| `run_phase_d_bisect.py` | Leave-one-out attribution for Phase D regression |
| `run_stack_test.py` | SRAF+FINRA combined ± tax-aware |
| `run_sweep*.py` | Hyperparameter grids (vol, leverage×fwd, quality, N positions) |
| `run_edgar_download.py`, `run_edgar_parallel.py` | 10-K/10-Q bulk download (6-worker, 9 req/sec) |
| `fetch_finnhub.py` | Overnight Finnhub fetch (`--max N --resume`) |
| `run_options_setup.py` | `--validate-tradier / --download-orats / --validate-consistency / --daily-poll` |
| `run_orats_full_download.py` | Orats historical, 6 endpoints, ~9K calls, ~60-90 min |
| `run_tradier_edge_case_test.py` | 16-ticker edge-case harness |
| `validate_tradier_orats.py` | 4-phase empirical Tradier-vs-Orats validation (HTML report) |
| `explore_tradier_options.py` | Pre-deployment API connectivity check |
| `merge_tradier_daily_to_iv_panel.py` | Merge daily Tradier parquet → panel pickle |
| `run_implied_div_validation.py` | Implied-dividend proxy vs Orats annIdiv |
| `dashboard_app.py` | Dash interactive app (6 tabs: Overview/Portfolio/Performance/Research/Test Suite/Time Travel) |
| `snapshot_baseline.py` | Immutable baseline capture (`--label X`) + drift validate |
| `clean_stale_caches.py` | Archive/restore stale caches w/ manifest |
| `compare_ml_caches.py` | Pearson correlation across ML cache payloads |
| `run_overnight.sh` | Chains yfinance baseline → isolated-retest CZ suite |

---

## 14. Results directory

**Root CSVs** (written every run): `tearsheet.csv`, `annual_returns.csv`, `monthly_returns.csv`, `wealth_growth.csv`, `current_holdings.csv`, `sector_allocation.csv`, `feature_importance.csv`, `regime_performance.csv`, `stress_test.csv`, `oos_tearsheet.csv`, `fama_french.csv`, `factor_correlation.csv`, `factor_decay.csv`, `bootstrap_ci.csv`, `permutation_test.csv`, `capacity_analysis.csv`, `dashboard.html`, `weights_history.parquet`.

**Standard tearsheet columns** (29 metrics): CAGR, Total Return, SPY CAGR, Alpha, Ann Vol, Sharpe, Sortino, Calmar, IR, Max DD, Avg DD Duration, Daily/Monthly Win Rate, Profit Factor, Tail Ratio, Tx Costs, Beta, PSR, DSR, Expected Max SR, DSR n_trials, After-Tax CAGR, Effective Tax Rate, ST/LT Fraction, Tax Drag, TLH Benefit, Alpha Hurdle, Account Recommendation (+ HIFO fields if `--use-tax-aware`).

**Top-20 feature-importance winners** (current): idiovol_21d_sn, alt_dxy_beta_signal, rvol_63d, rank_price_52w_high, rvol_63d_csz, rvol_21d_sn, idiovol_21d, rvol_10d, rvol_21d, semi_beta_P, z_rvol_21d_sn, rank_market_beta, semi_beta_Mminus_csz, rvol_21d_sni_csz, semi_beta_Mplus_csz, mom_12_1_csz, ix_beta_x_vol, tail_risk_5pct_csz, z_idiovol_21d_sn, semi_beta_P_csz. **Note**: vol and idiovol dominate (~60% of gain); this is the "low-vol anomaly" exposure.

**Archived experiments** (`results/archive/`):
- SWEEP1 (N=20–75), S2/S2U (quality 30–75%), S3 (leverage), S4 (fwd window)
- Fixed0–Fixed4 (baseline + SRAF + FINRA + lazy + LM)
- D0–D6 (Phase D Leave-one-out)
- B0–B4 (Phase B flag flips)
- iso_CZ_* (all_signals/price_only/accounting_only/coskew_only/xfin_only/net_payout_only)
- iso_OPTIONS_* (baseline/with_cz)
- iso_PHASE_D_* (eodhd_only/eodhd_sraf/full/no_sraf)
- Stack_sraf_finra[_tax]
- _baseline_recovery_v2, _repro_fixed2_v1-v3 (debugging)
- A0_fresh_pruned, R1–R4 (revert/cost_model/meta_gate/max_train_days)

---

## 15. Options stack (Tradier ⇢ Orats)

**Plan** (memory/options_data_plan.md): Orats $399 historical (one-time) + Tradier free API (live). Tradier embeds Orats-computed Greeks via `greek_smv_vol` field — train/serve-skew-free.

**Flow**:
```
Tradier chain  →  chain_to_smv_summary()  →  Orats /cores schema
                     │
                     ├─ cmiv_interpolator  → iv30d/iv60d/iv90d
                     ├─ tradier_orats_adapter → slope + dlt5/25/75/95
                     ├─ implied_borrow → borrow30 (clip [-0.05, 0.50])
                     └─ implied_dividend_proxy → annIdiv (PCP)
                                  │
                                  ▼
                  build_options_signals → 15 ranked signals → alt_features
```

**Cron**: `TradierDailyPoll` @ 16:30 ET daily → `run_options_setup.py --daily-poll --max-tickers 1500` (~5 min, ~6000 API calls, free tier). Writes `data/cache/options/tradier_daily/{YYYY-MM-DD}.parquet`.

**Expected value**: +250–450 bps CAGR. Validation (`validate_tradier_orats.py`): 4 phases, ship/blend/drop decision on rank-corr ≥0.70/0.50. Deploy post-Tradier approval.

---

## 16. Known gotchas & recent fixes

### Still active
- **_DENY_LIST may over-prune** (**101 entries** as of 2026-04-17 audit; comment at model.py:852 notes "Phase E bottom-80 prune entries REMOVED to recover 25.67% baseline" — partial revert): test with `EMPTY_DENY_LIST=1` to recover toward 25.67%
- **SPY must be force-loaded** (run_strategy.py:643): regime detector falls back to weak EW proxy without it
- **Investable mask** (run_strategy.py:925): features ranked within top-1500 by ADV, not full 3000
- **Vol-target default 0.40** in CLI vs docstring says 0.16 — inconsistency, check flag
- **Options signals lookahead** (options_signals.py): `LAG_OPTIONS_SIGNALS=1` mandatory to offset EOD timestamp
- **Wavelet FFT** (features.py:1454): chunked 500 tickers/chunk or OOM (was 30GB, now 1.4GB/chunk)
- **Dividend cash flow** off when CZ payout_yield on (dup) — `DISABLE_DIVIDEND_YIELD=1`
- **FRED fallback**: if T10Y2Y/HY spread/VIX/DFF fail, reconstructs from yfinance proxies
- **Form 4 insider**: volume only, no direction (just filing dates)
- **Short interest + institutional holdings**: both are yfinance snapshots, broadcast-across-time → lookahead risk, OFF by default

### Recent commits
- `dab62f7`: test_suite universe=3000 to match run_strategy (fixes cache miss)
- `d19313f`: quality_signal→quality_filter duplicate kwarg fix
- `abeceda`: Tier 10 portfolio quality/risk fixes
- `90d6c3a`: wavelet chunking, ranked_signals f32, size_neutralize f32 (3 hidden memory spikes)
- `9903ea2`: float32 construction + numpy panel + window cap

### Historical lessons
- EODHD fundamentals **disabled in load_all_api_data** (diluted 25.67→17.02 CAGR); EDGAR free is the authoritative source
- FINRA bi-monthly must pivot on `publication_date` (settlement+8 BDay), NOT `settlement_date` (2-day lookahead on 74% of events)
- `lazy_prices_features.py:236`: Windows backslash path normalization — without this, every filing returned None → zero-feature panel silently
- `dividend_yield` key vs `dividend_yield_signal`: TTM-rolling rewrite broke cache-key match, degraded feature importance 357→414
- Risk-adjusted labels killed momentum edge (strategy_diagnosis_2026.md)
- Cache-key bug (pre-test_suite_design.md fix): deny-list changes didn't invalidate feature panel → phantom reproducibility

---

## 17. Strategy diagnosis & priorities

**Root causes of SPY underperformance** (memory/strategy_diagnosis_2026.md):
- Beta ~0.53 drag: cost ~6% annual in bull regimes (but current beta is 0.90-1.33, partially fixed)
- 21d labels but IC peaks at 5-10d (halflife_optimization memory: `signal_smooth_halflife = peak_IC_horizon`, NOT /2; hl=10 locked, 7/12/15 untested)
- Vol-bucketing forced small-cap tilt (disabled, `--use-vol-buckets=False` default)
- Anti-profitability loading (RMW -0.35 → fixed via quality tilt + SWEEP-2U quality percentile 70%)

**Default assumption**: when something regresses, suspect beta/horizon/factor-exposure BEFORE model quality.

**EODHD ROI** (memory/eodhd_roi_evaluation.md): $60/mo tier NOT ROI-positive below $180K AUM. Only valuable feature = 25yr earnings calendar. Sentiment=zero importance. Fundamentals=redundant vs free EDGAR. Insider=1yr history only (need SEC EDGAR for historical).

**C&Z buildable** (memory/cz_research_results.md): top-10 novel signals with published IC_IR: coskewness (0.198), XFIN (0.182), net_payout_yield (0.176), payout (0.146), mom_season (0.146) — all implemented. 5 high-IC options signals (dCPVolSpread 0.375, SmileSlope 0.369) deferred until $50K AUM.

**Sprint 1 revival plan** (memory/sprint1_feature_reduction_revival.md): 22 factor-momentum + lead-lag features regressed -99 bps. Prune low-importance first, then retest. Defer until Phase 2 baseline locks.

---

## 18. Academic canon (memory/academic_factor_canon.md)

Reference papers when making factor decisions:
- **JKP** (Jensen-Kelly-Pedersen 2023): validates momentum/profitability/low-vol
- **HXZ** (Hou-Xue-Zhang q-factor): investment+profitability
- **McLean-Pontiff**: ~35% post-publication decay
- **GKX** (Gu-Kelly-Xiu 2020): ML in finance; long-only underwhelms long-short; quarterly retrain; non-stationary favors 4-5y sample decay
- **Moreira-Muir 2017**: vol management adds ~0.2 Sharpe
- **Novy-Marx 2013**: gross profitability
- **Ball et al. 2016**: cash-based op profitability beats accruals
- **Asness 2018 QMJ**: quality minus junk
- **Ehsani-Linnainmaa 2022**: factor momentum
- **Baz 2015, Barroso & Santa-Clara 2015**: momentum horizon & vol scaling
- **AFML (Lopez de Prado 2018)**: embargo, uniqueness weights, meta-labeling, DSR

---

## 19. Quick-reference commands

```bash
# Full production pipeline
python run_strategy.py --start 2013-01-01 --end 2026-03-01 --capital 100000

# Fast iteration (skip robustness, reuse cache)
python run_strategy.py --skip-robustness

# Force fresh run (new cache keys)
python run_strategy.py --no-cache

# Honest reproducible baseline: 22.58% CAGR / 0.715 Sharpe (~3h 15min, 17 GB RAM peak)
rm -f data/cache/feature_panel_*.pkl data/cache/ml_predictions_*.pkl
EMPTY_DENY_LIST=1 python run_strategy.py \
  --skip-robustness \
  --use-finra-short-interest \
  --results-dir results/_baseline

# Ablation: clear deny list only
EMPTY_DENY_LIST=1 python run_strategy.py

# Isolate single signal
EXTRA_DENY_FEATURES=coskewness,xfin python run_strategy.py --use-cz-signals

# Options backtest (requires iv_panels_orats.pkl)
python run_strategy.py --use-options-signals

# Tax-aware with HIFO
python run_strategy.py --use-tax-aware

# Phase 5 production (locked config)
python run_phase5_production.py

# Isolated retest batch
python run_isolated_retest.py --group cz

# Horizon study (~50 min)
python poc_horizons.py

# Full Orats download (~90 min, $399 subscription)
python run_orats_full_download.py --max-tickers 1500

# Daily Tradier poll (runs via cron)
python run_options_setup.py --daily-poll

# Interactive dashboard
python dashboard_app.py        # then open http://localhost:8050

# Snapshot / drift check
python snapshot_baseline.py --label pre_experiment_X
python snapshot_baseline.py --validate pre_experiment_X

# Cache cleanup (dry-run first)
python clean_stale_caches.py --list
python clean_stale_caches.py --archive --keep abc123,def456 --yes
```

---

## 20. Working-style notes

- **User context**: personal capital (~$15-50K AUM), solo developer, Windows + Git Bash, monthly rebalance target. Build for that scale; do NOT propose hedge-fund infrastructure.
- **When regressions appear**: suspect beta/horizon/factor-exposure before model quality. Check memory/strategy_diagnosis_2026.md.
- **When adding signals**: use `run_isolated_retest.py` with `EXTRA_DENY_FEATURES` to isolate. Check cache hash actually changed (test_suite_design.md).
- **When baselines disagree**: read [REPRODUCTION.md](REPRODUCTION.md). Honest baseline is **22.58% CAGR / 0.715 Sharpe** with `EMPTY_DENY_LIST=1`. 18.81% is the master-default (deny-list active) lower bound. 25.67% is the Fixed2 aspiration, -309 bps gap remaining under investigation.
- **Halflife rule**: `signal_smooth_halflife ≈ peak_IC_horizon`. hl=10 locked; 7/12/15 untested.
- **Options economics**: wait for $50K AUM (memory/options_data_plan.md). Meanwhile Tradier validation prep (free tier).
- **Paths**: always forward-slash inside Python. Windows backslash → pathlib fails silently (lazy_prices_features.py:236).

---

---

# PART II — Improvement Levers (added 2026-04-17, second pass)

> This half is for *finding CAGR/Sharpe wins*. Part I is the lay of the land. Part II is the hit list.

---

## 21. Horizon truth & factor decay

From `results/factor_decay.csv` — IC vs forward window (strategy's pooled composite signal):

| fwd | mean IC | IR | % positive |
|---|---|---|---|
| **1d** | 0.0289 | 0.141 | 55% |
| **5d** | 0.0518 | 0.267 | 62% |
| **10d** | **0.0611** | **0.332** | **64%** ← PEAK |
| **21d** | 0.0547 | 0.308 | 63% |

**Current default `--forward-window=7`** → ~10% below peak. **IC peaks at 10d, not 7 or 21.**

**Lever**: set `--forward-window=10` and re-test. Expected IR uplift +5-10%. Confirms memory/halflife_optimization_2026_04.md rule: `signal_smooth_halflife ≈ peak_IC_horizon`. hl=10 already locked → horizon=10 is consistent.

---

## 22. Feature importance truth table

From `results/feature_importance.csv` (LightGBM gain).

### Top 10 (≥ 60% of total gain)
1. `idiovol_21d_sn` (3835) · size-neutral 21-day idiosyncratic vol
2. `alt_dxy_beta_signal` (3193) · 36-mo β to ΔDXY — **only macro feature in top 3**
3. `rvol_63d` (3059)
4. `rank_price_52w_high` (2395)
5. `rvol_63d_csz` (2314)
6. `rvol_21d_sn` (2154)
7. `idiovol_21d` (2010)
8. `rvol_10d` (1998)
9. `rvol_21d` (1988)
10. `semi_beta_P` (1902)

**Pattern**: volatility + semi-beta own the model. `rvol_*` and `idiovol_*` variants occupy 8 of top 25. This is the low-vol anomaly captured 9 ways. **Implication**: adding *more* volatility variants is a dead-end; orthogonal signals needed.

### Bottom decile (drop candidates — gains 366-480, audited exact values)
`vol_of_vol` (479), `alt_dividend_yield` (477), `rank_long_momentum` (473), `ma200_distance_csz` (469), `alt_oil_beta_x_oil_momentum_csz` (464), `rank_amihud` (402), `alt_dividend_yield_csz` (398), `alt_credit_beta_signal_sni_csz` (394), `ix_mom_x_vol` (387), `ix_beta_x_yield_curve` (366).

**Lever**: add these 10 to `EXTRA_DENY_FEATURES` → tighten regularization, test Δ Sharpe. Low risk, potentially +0.02–0.05 Sharpe from noise reduction.

### Lead-lag features: ZERO importance
None of the 15 `leadlag_*` features from `cross_asset_leadlag.py` appear in top-500. **Delete.** Set `EXTRA_DENY_FEATURES="leadlag_*"` or remove the call in `alt_features.py`. Saves 2% feature count and reduces fit-noise.

### _DENY_LIST (93 entries) — still under investigation
Session 2026-04-17 is testing `EMPTY_DENY_LIST=1` to see if the deny list was the cause of the 25.67% → 18.81% regression. No strong re-enable candidates from importance data (deny-listed features had zero gain in prior runs), but `_sn`/`_sni` variants of momentum may have been culled too aggressively.

---

## 23. C&Z signal IC in *our* universe

From `results/_cz_research/cz_signal_ic.csv` — Chen-Zimmermann 209 signals tested on our 2013-2026 universe:

### Top 20 by IC_IR (our data)
| Rank | Signal | IC_IR | mean IC | coverage | status |
|---|---|---|---|---|---|
| 1 | **dCPVolSpread** | **0.375** | 0.0149 | 39% | ❌ options-blocked |
| 2 | **SmileSlope** | **0.369** | 0.0176 | 39% | ❌ options-blocked |
| 3 | OScore | 0.286 | 0.0189 | 18% | ≈ chs_distress_signal |
| 4 | **AnnouncementReturn** | **0.254** | 0.0142 | 41% | ≈ EAR (check overlap) |
| 5 | ShareIss5Y | 0.250 | 0.0183 | 36% | = net_share_issuance |
| 6 | **EarningsStreak** | **0.245** | 0.0142 | 33% | ❌ **NOT IMPLEMENTED** |
| 7 | ShareIss1Y | 0.237 | 0.0192 | 41% | = net_share_issuance |
| 8 | **ShareRepurchase** | **0.225** | 0.0130 | 42% | ❌ **NOT IMPLEMENTED** |
| 9 | CBOperProf | 0.202 | 0.0150 | 29% | ≈ cash_based_op_prof |
| 10 | FirmAgeMom | 0.200 | 0.0375 | 4% | ❌ data-blocked (sparse) |
| 11 | Coskewness | 0.198 | 0.0194 | 42% | ✅ implemented |
| 12 | XFIN | 0.182 | 0.0125 | 40% | ✅ implemented |
| 13 | NetEquityFinance | 0.178 | 0.0149 | 40% | ❌ not implemented |
| 14 | NetPayoutYield | 0.176 | 0.0187 | 28% | ✅ implemented |
| 15-20 | CPVolSpread, AnalystRevision, PayoutYield, MomSeason, ChangeInRecommendation, CustomerMomentum | 0.143-0.152 | — | — | mixed |

### Verified WRONG-sign / wrong-horizon
- **Mom12m IC = -0.010 (NEGATIVE)** in our universe despite being a canonical signal. Check label window; our `mom_12_1` uses skip-21d JT convention — may not match C&Z definition.

### Top 3 immediate build candidates (data-feasible now)
1. **EarningsStreak** (IC_IR 0.245) — consecutive earnings beats from EODHD. ~2-3 hrs to build. Expected Sharpe +0.05-0.10 if orthogonal. Low correlation with existing momentum.
2. **ShareRepurchase** (IC_IR 0.225) — buyback-only from EDGAR CF statement. Partial overlap with net_payout_yield; needs 1-feature-out test.
3. **NetEquityFinance** (IC_IR 0.178) — equity issuance - repurchase from EDGAR. Complements XFIN.

### Top 2 deferred (blocked)
- **dCPVolSpread** (IC_IR 0.375) — needs options; already coded in `options_signals.py` awaiting validated Tradier-Orats panel.
- **SmileSlope** (IC_IR 0.369) — same.

---

## 24. Unused data surfaces (deletion targets) — AUDITED 2026-04-17

Earlier pass had multiple wrong "unused" claims. Re-audit via grep corrected below.

### ACTUALLY unused columns (confirmed zero callers in src/ or root *.py)
- **FRED `NAPM`** (ISM PMI) — mapped but no feature consumer. Delete from `_FRED_SERIES`.
- **FRED `ICSA`** (initial jobless claims) — mapped but no consumer.
- **FRED `PERMIT`** (building permits) — mapped but no consumer.
- **VIX `VIX9D`**, **`VIX6M`** — raw levels loaded but only `term_slope`, `vix_percentile`, `vix_change_5d` (derived from VIX and VIX3M) consumed.
- **`institutionCount`** from `load_institutional_holders` — loaded at alt_data_loader.py:1298 but zero grep hits.
- **`targetMeanPrice`, `currentPrice`, `numberOfAnalysts`** from `load_earnings_estimates` — stored but not extracted as features.
- **`eps_estimate`, `eps_actual`, `eps_surprise`** (non-`_pct` variants) from `load_earnings_calendar` — loaded but no feature builder consumes them.

### Previously claimed unused but ACTUALLY USED (claims corrected)
- ~~`load_insider_transactions`~~ — **IS called** at run_strategy.py:713; loaded as `insider_dict` (but its output then flows to `build_insider_signals` which uses other inputs, so the CALL is used even if the insider_dict itself isn't consumed downstream — re-verify independently).
- ~~`load_earnings_calendar_eodhd`~~ — **IS called** at run_strategy.py:710 when `--use-eodhd-earnings` (default True).
- ~~`load_fred_macro`~~ — **IS called** at run_strategy.py:689.
- ~~FRED `UNRATE`~~ — **IS used**: mapped to `unemployment` feature in alt_features.py:401.
- ~~FRED `T10YIE`~~ — **IS used**: exposed via standalone `load_breakeven_inflation()`.
- ~~FRED `VIXCLS`, `DFF`~~ — USED as fallback proxies in alt_data_loader.py:593-625.
- ~~VIX `VIX3M`~~ — **IS used** in `term_slope = VIX / VIX3M` at alt_data_loader.py:694.
- ~~`before_after_market`~~ from earnings calendar — **IS used** at alt_features.py:637-638 (EAR timing).
- ~~EODHD sentiment `count`~~ — **IS used** at api_data.py:779-791 to create `eodhd_news_volume` signal.

### Feature hypotheses still valid (from truly unused data)
- `unemployment_z` × price momentum → recession hedge (already have unemployment, this is a NEW interaction, not unused data reclamation)
- `(VIX6M - VIX3M)` term-structure inversion — crisis early-warning
- `institutionCount_z` — crowding indicator
- `estimate_accuracy = (eps_actual - eps_estimate) / |eps_estimate|` — analyst-skill ranking
- NAPM/ICSA/PERMIT z-score interactions — business-cycle regime dummies

---

## 25. Sweep surface map (where winners live, where to test next)

From `results/sweep_*.csv`, `phase_d_bisect.csv`:

| Sweep | Winner | At-edge? | Test next |
|---|---|---|---|
| **vol target** | `V_vt035` (Sharpe 0.741, CAGR 21.82%) | `vt040` edge wins CAGR (23.85%) | vt043, vt045 |
| **quality upper** | `S2U_quality70` (Sharpe 0.716) | concave peak | fine grid 68, 70, 72 |
| **quality lower** | `S2_quality45` (Sharpe 0.601) | monotone up | extend 50, 55 |
| **leverage × fwd** | `S4_fwd7` (Sharpe 0.734) | fwd10 drops sharply | fwd 5, 6, 8, 9, 10 + lev 26, 28 |
| **N positions** | `SWEEP1_N30` (Sharpe 0.685) | peak ~30 | 28, 30, 32, 35 |
| **quality × vol 2D** | — never run | — | 3×3 grid vol[030,035,040] × qual[65,70,75] |
| **Phase D bisect** | D0 (no phase D) = D5 (all phase D) — **all identical** | — | **CACHE-KEY BUG** (see §16). Fix first. |

**Critical finding**: phase_d_bisect and iso_CZ_* produce IDENTICAL tearsheets across 6+ signal variants. This is the cache-key collision bug documented in `docs/test_suite_design.md`. **All isolation experiments before that fix are void.** Re-run with `EXTRA_DENY_FEATURES` + `DENY_LIST_HASH_SUFFIX` to force cache miss.

---

## 26. Regime attribution — where strategy lives and dies

From `results/regime_performance.csv`:

| Regime | Days | % time | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|
| **bull_calm** | 2461 | 74.3% | 27.4% | 0.94 | -30.8% |
| **bull_volatile** | 253 | 7.6% | **86.5%** | **1.97** | -13.4% |
| **bear_calm** | 332 | **10.0%** | **-54.9%** | **-2.00** | **-71.1%** |
| **bear_volatile** | 164 | 5.0% | 40.9% | 0.92 | -23.6% |

**Strategy is a bull-volatile monster (1.97 Sharpe!) and a bear-calm catastrophe (-71% DD).**

### Stress periods (from `stress_test.csv`)
- COVID Feb-Mar 2020: -31.8% total, +1.6% vs SPY (marginal)
- 2022 rate-hike bear: -35.0% total, -10.9% vs SPY (lost)
- **2025 tariff shock**: -9.5% in 8 days, -3.3% vs SPY (worst alpha deterioration)
- Post-COVID rally: +126.8%, +73.4% vs SPY (huge win)

### Levers
1. **Bear-calm kill-switch**: when rolling 60d DD > -25%, hedge or go to cash. Bear-calm is 10% of time and ~accounts for nearly all DD.
2. **Upsize in bull-volatile** (already have `--bsc-scaling` for own-vol target; consider regime-gated leverage boost 1.3 → 1.6 in bull_volatile)
3. **Expand credit overlay coverage**: current NFCI + HY-OAS thresholds (0.0, 5.0%) didn't catch 2025 tariff shock. Tighten to 0.0, 4.5% OR add VIX percentile trigger.
4. **Down-weight bear_calm longs**: regime-conditioned position sizing. Memory/strategy_diagnosis_2026.md hints this already proposed.

---

## 27. Annual return attribution

From `results/annual_returns.csv`:

| Year | Strat | SPY | Alpha |
|---|---|---|---|
| 2013 | 131.6% | 29.0% | **+102.6%** ⭐ |
| 2014 | -24.3% | 13.5% | **-37.8%** ☠ |
| 2015 | 4.5% | 1.2% | +3.3% |
| 2016 | 6.0% | 12.0% | -6.0% |
| 2017 | 81.6% | 21.7% | **+59.9%** ⭐ |
| 2018 | -23.6% | -4.6% | -19.1% |
| 2019 | 31.0% | 31.2% | -0.3% |
| 2020 | 118.5% | 18.3% | **+100.2%** ⭐ |
| 2021 | 19.9% | 28.7% | -8.8% |
| 2022 | -30.8% | -18.2% | **-12.6%** ☠ |
| 2023 | 17.8% | 26.2% | -8.4% |
| 2024 | 23.1% | 24.9% | -1.7% |
| 2025 | 29.8% | 17.7% | +12.0% |
| 2026 | 20.1% | 0.6% | +19.5% |

**Lost to SPY in 7 of 14 years.** Wins are massive (2013, 2017, 2020 crisis/recovery years). Losses concentrated in stable low-vol years (2016, 2019, 2021, 2023, 2024) — exactly the years where being a "small-cap high-vol monster" hurts.

### Month-of-year pattern
Sept chronically weak (Sep 2020 -1.9%, 2021 -5.0%, 2022 -12.0%, 2023 -5.7%). Oct volatile both ways. Strategy underperforms SPY's Nov strength.

### Levers
- **September hedge overlay** (short SPY puts or VIX spread in Sep)
- **Rate-regime filter** (2018/2022 both rate-hike years — down-weight in rising-DGS10 regime)
- **Low-vol year defensive mode** (VIX < 15 for 60d → shrink to SPY-core 30%)

---

## 28. Archive experiments — wins, losses, patterns

From `results/archive/*.csv`:

### Top 5 winners (but see §28.1 caveat)
| Config | CAGR | Sharpe | Δ vs baseline |
|---|---|---|---|
| **Fixed2_finra** | 25.67% | 0.785 | +138 bps ⚠ unreproducible |
| Stack_sraf_finra | 25.20% | 0.772 | +91 bps |
| Fixed1_sraf | 24.98% | 0.765 | +69 bps |
| Fixed0_baseline | 24.29% | 0.751 | baseline ⚠ see §28.1 |
| iso_PHASE_D_full | 21.77% | 0.700 | -252 bps vs contaminated baseline |

### Bottom 5 losers (avoid re-running)
- `SWEEP1_N20` — 13.4% CAGR, 0.53 Sharpe → **too concentrated**
- `S2_quality30/40` — 14.5-14.7% → **quality gate too low**
- `SWEEP1_N75` — 15.2% → **too diffuse, no alpha**
- `iso_OPTIONS_baseline` — 17.6% / 0.59 → **options don't work yet** (coverage bug, see §15)

### §28.1 Caveat — Fixed0/Fixed2 baselines are NOT reproducible
Memory session_20260417: the 24-26% numbers were on *uncommitted code* with lookahead-bias bugs. Today's honest baseline is **18.81%**. All archive winners shown may not hold after the cache-key + lookahead fixes in REPRODUCTION.md. Use archive *relatively* (which knob helped?), not *absolutely* (what CAGR will we see?).

### Patterns in winners
1. **Multi-source fusion beats isolation** — SRAF + FINRA stacked beats either alone
2. **N=30 sweet spot** — above = dilution, below = concentration risk
3. **Quality 65-70 upper** — below hurts, above marginal
4. **Vol target 0.035-0.040** — extends grid unlocked CAGR, Sharpe flat

### PHASE_D isolation result (if cache-key fix validated):
- PHASE_D_full (+475 bps) > PHASE_D_no_sraf (+389 bps) > PHASE_D_eodhd_only (+348 bps)
- **EODHD + SRAF synergize** — keep both

---

## 29. Walk-forward mechanics

`WalkForwardModel.fit_predict()` in `src/model.py:2776-3137`:

- **Window**: expanding, capped by `max_train_days=756` (3 years)
- **Retrain**: every 63 days (quarterly, GKX 2020)
- **First fit**: at fold 126 (~6 months in → late June 2013)
- **Folds over 2013-2026**: ≈ 52 windows
- **Embargo**: 5 days (AFML)
- **Sample weights stacked** (multiplied, in order):
  1. Sample decay: `exp(-ln(2)/630 × (max_idx - sample_idx))` → 630-day halflife
  2. Per-date normalization: `1/count_on_date`
  3. Uniqueness (AFML): `1/concurrency`
  4. Magnitude (opt-in): `|y - median_y|`
- **Early stop**: LightGBM only, Spearman-IC feval, 50 rounds patience

### Levers
1. **Monthly retrain (21d vs 63d)** → +3× compute but +5-15 bps IC in regime shifts
2. **Adaptive max_train_days** → 504d in volatile (2020, 2022), 1260d in stable
3. **Per-head early stop** → currently only LGBM; Ridge trains full 500 models, likely overfitting tail

---

## 30. Label construction

`src/model.py::build_labels()` (lines 924-1113):

**Default**: run_strategy.py CLI `--forward-window=7` passes to WalkForwardModel; WalkForwardModel's own constructor default (model.py:1321) is `forward_window=5` if invoked directly. Production pipeline always goes through the CLI so effective horizon is **7d**. 6-grade LambdaRank bucketing:
```
Grade 0: bottom 20%     (worst)
Grade 1: 20-40%
Grade 2: 40-60%         (median)
Grade 3: 60-80%
Grade 4: 80-95%
Grade 5: top 5%         (superstar)
```
Cuts: `pd.cut(rank, bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 0.95, 1.001])`

**Alternative modes**:
- `--use-rank-targets`: continuous percentile [0,1]
- `--sector-rank-weight w`: blend `(1-w)×univ_rank + w×sector_rank`
- `--beta-neutral-labels`: residualize forward return vs market via 60d rolling β
- `--forward-windows 1,5,21`: multi-horizon ensemble (avg of percentile ranks per horizon)
- `--risk-adjust-labels`: `fwd_return / 21d_vol × sqrt(252)` — **OFF by default, memory note: kills momentum edge**

### Levers
1. **forward_window=10** → align with IC peak (see §21) — single highest-expected-value knob
2. **sector_rank_weight=0.3** → force broader stock-picking, reduce mega-cap dependency
3. **median-demean labels** → `(fwd - median)/MAD` robust alternative; test in fat-tail regimes

---

## 31. Cost, turnover, capacity

- **Annualized turnover**: 375% (without 70% overlap constraint → ~900%)
- **Total cost**: $40K over 13.2y on ~$800K avg AUM → **38 bps/year**
- **Cost tier distribution**: estimated 60-70% notional in 1-3 bps tiers, 30-40% in 8 bps tier
- **Holding overlap (70%) IS binding** — halves turnover
- **Capacity**: viable to ~$100M; above $250M cost drag > returns (see `capacity_analysis.csv`)

### Levers
1. **Raise min-holding-overlap to 80%** → 375% → 250-300% turnover, save ~35 bps
2. **Hard turnover cap at 200%** → saves ~20 bps but adds concentration risk
3. **Tier-aware position sizing**: larger sizes in mega/large-cap (1-3 bps), smaller in small-cap (8 bps). Currently uniform → 15-20% notional in wrong tier.

---

## 32. Fama-French attribution

From `results/fama_french.csv` and `factor_correlation.csv`:

| Factor | Coef | t-stat | p | Verdict |
|---|---|---|---|---|
| Alpha | 0.000293 | 1.42 | 0.155 | marginal (p>0.10) |
| Mkt-RF | 1.235 | 7.9 | <.001 | strong |
| **SMB** | **0.913** | **19.3** | <.001 | **massive small-cap tilt** |
| HML | -0.194 | -2.1 | .036 | short value |
| **RMW** | **-0.172** | **-2.3** | .020 | **anti-profitability drag** |
| CMA | 0.073 | 1.04 | .298 | neutral |
| Mom | 0.351 | 5.1 | <.001 | momentum loading |

**Annualized α = 7.37%, R² = 0.692, OOS Sharpe 1.045 > IS Sharpe 0.680** (OOS BETTER — no overfitting evidence).

### Levers
1. **Fix RMW anti-factor**: quality_tilt at 0.35 is not enough. Test 0.50 or add hard ROE floor (12%+). Expected +30-50 bps.
2. **Reduce SMB tilt**: partial — min_adv_for_selection + max_selection_pool limit small-caps. Could tighten further by raising min_adv from $5M to $20M.
3. **Exploit positive Mom loading**: 0.35 Mom t=5.1 is real. Monthly retrain (see §29) doubles down on momentum capture.

---

## 33. NLP / cross-asset attribution (what text adds, what macro adds)

### Text features (default: OFF)
From Fixed1-4 archive tearsheets (pre-reproduction; treat as directional only):
- **SRAF sentiment**: +130 bps / +0.029 Sharpe (Fixed1) — **best text signal**
- **LM sentiment**: +61 bps / +0.015 Sharpe (Fixed4)
- **Lazy prices**: 0 / 0 (Fixed3) — matched baseline → no lift

**SRAF dispatch fails with "index must be monotonic" error** — needs sort-before-dispatch fix in `alt_features.py`. Once fixed, SRAF is the clear next add.

### Macro / cross-asset
- **DXY beta = #3 feature overall** — currency exposure the single best macro lever
- **Credit beta, oil×regime, Cu/Au cyclicality** = moderate contributors (rank 44-54)
- **Lead-lag features = ZERO importance** — drop them all
- **ix_ interactions beat raw broadcasts** — β×vol, β×yield_curve earn 366-1289 gain vs broadcast scalars at 0

### Levers
1. **Fix SRAF dispatch, enable `--use-sraf-sentiment`** → if Fixed1 holds, +130 bps
2. **Drop all `leadlag_*` features** → cleaner model, no alpha loss
3. **Add VIX regime dummies** → categorical (quiet <12, normal, elevated, distress >30) × signal-sign-flips. Expected +0.05-0.10 Sharpe.
4. **EPS revision diffusion × DXY beta** → macro-to-micro bridge (earnings cuts follow FX stress by ~3 months)

---

## 34. Options stack — blocked

From validation smoke test 2026-04-17:
- **Phase 0**: done (Tradier embeds Orats greeks — confirmed)
- **Phase 1**: **FAILED** — IV30d median diff +2.5 vol pts vs 1.0 pt threshold. MSFT borrow diff +20.5 pts. All delta-bucket IVs fail.
- **Phases 2-4**: blocked on Phase 1

**Current caches**: `iv_panels_orats.pkl` (2.2 GB, 2026-04-16) + 1 Tradier daily parquet. **Cron IS registered**: Windows Task Scheduler shows `TradierDailyPoll` (State: Ready, trigger 16:30 ET) — CONTEXT was stale; corrected 2026-04-17.

**Root cause to investigate**: Tradier `borrow30` appears to come from a different source than Orats /cores. Field mapping in `tradier_orats_adapter.py` may need a second pass, or we need a proper market-hours capture (not off-hours).

**Expected upside if validated**: +250-450 bps CAGR (top signals dCPVolSpread 0.375 IR, SmileSlope 0.369 IR).

**Known historical pitfalls** (options_signals.py:46-70, contributors to 809 bps past drop):
1. 1-day lookahead from EOD options timestamps (fixed via `LAG_OPTIONS_SIGNALS=1`)
2. Universe mismatch: 73% of tickers have no options → ranking dilution (fixed via `OPTIONS_MIN_COVERAGE=0.30`)
3. Vol outliers not winsorized (fixed via `OPTIONS_WINSORIZE=0.01`)
4. Research IC measured on 39%-coverage subset, not full universe
5. Feature dilution (15 signals displace baseline) — mitigate via `OPTIONS_SIGNAL_SET="validated"` (top 5 only)

---

## 35. Cache & memory bottlenecks

### Disk — consolidation targets (~5-6 GB reclaimable)
- `edgar_fundamentals_v2raw.pkl` (1.8 GB) — intermediate, v2 is downstream. Archive.
- 6× `eodhd_prices_*.pkl` (5.2 GB total) — multiple nearly-identical universe snapshots. Keep 2 (current + one extended range), archive 4.
- `edgar_fundamentals_extra_*_v1.pkl` (2× 547 MB) — 10K vs 3K ticker variants. Grep usage; keep only one.
- Old v1 caches (`analyst_actions v1`, `fred_macro v1`) — superseded by v2 Apr 5.
- `feature_panel_c30c1db5cd03.pkl` (26 GB) — LIVE, keep.

### Memory — remaining bottlenecks
1. **`features.py:1466` sliding_window_view for wavelets** — creates (1248 × 500 × 252) float32 temporary ≈ 157 GB before chunk GC. Use cumsum for rolling mean instead. **Est. -12 GB peak.**
2. **`size_neutralize()` Ridge** — sklearn Ridge may upcast float32 → float64 during fit. Verify dtype preservation or downcast residuals. **Est. -4-8 GB.**
3. **`lgb.Dataset` per fold** — each walk-forward window creates new Dataset copies. Use `free_raw_data=False` or shared pool. **Est. -6-10 GB.**

---

## 36. Log / runtime observations

From `logs/` (104+ files):
- Most pipelines cache-bound → <5 min wall time when caches warm
- **Recurring warning**: `sraf_sentiment dispatch failed: index must be monotonic` — loses 3 features. Fix by sorting index before dispatch.
- Hundreds of HTTP 404s on delisted tickers — non-blocking but noisy
- No 429 rate limits observed (caching absorbs API load)
- Pandas FutureWarnings on `pct_change`, `concat`, `setitem with None` — not fatal but should fix before pandas 3.0

---

## 37. Prioritized improvement backlog

Ranked by (expected impact × low effort). Execute top-down.

| # | Lever | Expected Δ | Effort | File refs |
|---|---|---|---|---|
| 1 | **Set `--forward-window=10`** (IC peak) | +5-10 bps IR → ~+100 bps CAGR | 1 flag | §21 |
| 2 | **Resolve _DENY_LIST investigation** (EMPTY_DENY_LIST test in flight) | ±500 bps, sign unknown | wait for 17:40 result | §22 |
| 3 | **Drop bottom-10 low-gain features** (vol_of_vol, alt_dividend_yield, leadlag_*, etc.) | +0.02-0.05 Sharpe | set `EXTRA_DENY_FEATURES` | §22 |
| 4 | **Fix SRAF dispatch + enable `--use-sraf-sentiment`** | +130 bps / +0.03 Sharpe | fix sort in `alt_features.py` | §33 |
| 5 | **Build EarningsStreak (IC_IR 0.245)** | +0.05-0.10 Sharpe | 2-3 hrs, EODHD earnings | §23 |
| 6 | **Extend vol-target sweep to 0.043-0.045** | +100-200 bps CAGR | run `run_sweep_vol.py` extended | §25 |
| 7 | **Raise `quality_tilt` to 0.50 or add hard ROE floor** (fix RMW -0.17) | +30-50 bps α | 1 flag | §32 |
| 8 | **Bear-calm kill-switch** (60d DD > -25% → hedge) | -71% DD → ~-40% | 1 block in portfolio.py | §26 |
| 9 | **Cache-key fix verify** (phase_d, iso_CZ all produced identical tearsheets) | enables every signal test | check test_suite_design.md | §25, §15 |
| 10 | **Build ShareRepurchase + NetEquityFinance** (C&Z top 10) | +0.05-0.08 Sharpe combined | 3-4 hrs, EDGAR CF | §23 |
| 11 | **Tighter min-holding-overlap 70→80%** | +35 bps (cost savings) | 1 flag | §31 |
| 12 | **September hedge overlay** | reduce chronic Sep losses | credit_overlay extension | §27 |
| 13 | **Tier-aware position sizing** | +5-10 bps (cost) | portfolio.py weighting | §31 |
| 14 | **Monthly retrain** (21d vs 63d) | +5-15 bps IC | `--retrain-freq=21` | §29 |
| 15 | **Memory: cumsum rolling for wavelets** | -12 GB peak, enables larger universe | features.py:1466 rewrite | §35 |
| 16 | **Delete unused loaders** (`load_insider_transactions`, unused FRED series, VIX9D/3M/6M raw) | faster cache build, clearer lineage | run_strategy.py + alt_data_loader.py | §24 |
| 17 | **Options: fix Phase 1 validation failures** | +250-450 bps (blocked!) | debug tradier_orats_adapter | §34 |
| 18 | **sector_rank_weight=0.3** | broader selection | 1 flag | §30 |

### Hardest-to-reverse decisions (discuss before committing)
- **Raising min_adv from $5M → $20M** (SMB tilt fix) — reduces universe by ~30%, may hurt alpha if small-caps are the edge
- **Deleting lead-lag features** — cheap to add back, but the code is there and someone may care
- **Dropping Finnhub entirely** — 4 features, all auto-disabled anyway; just delete

### DO NOT repeat (from chronology, §14)
- Risk-adjusted labels (kills momentum)
- Lazy prices alone (sparse coverage → no lift)
- Numerai techniques (era-weighting, rank targets, meta-labeling) — wrong regime for 30-stock long-only
- Vol-bucketing for cross-sectional neutralization (forces small-cap tilt)

---

## 38. Quick-check commands (diagnostics)

```bash
# 1. Verify deny-list ablation (in-flight test)
EMPTY_DENY_LIST=1 python run_strategy.py --no-cache

# 2. Test forward-window=10 (IC peak)
python run_strategy.py --forward-window 10 --no-cache

# 3. Apply bottom-10 feature drop
EXTRA_DENY_FEATURES=vol_of_vol,alt_dividend_yield,rank_long_momentum,ma200_distance_csz,rank_amihud,ix_mom_x_vol,ix_beta_x_yield_curve,alt_dividend_yield_csz,alt_credit_beta_signal_sni_csz,alt_oil_beta_x_oil_momentum_csz python run_strategy.py

# 4. Drop all lead-lag (separate test)
EXTRA_DENY_FEATURES=$(python -c "import re; print(','.join([f'leadlag_{a}_{h}d' for a in ['oil','copper','gold','dxy','hy_oas'] for h in [5,10,20]]))") python run_strategy.py

# 5. Vol-target extended sweep
for vt in 0.035 0.040 0.043 0.045; do
  python run_strategy.py --vol-target $vt --results-dir results/_v2_sweep/vt${vt} &
done; wait

# 6. Cache archive dry-run before cleanup
python clean_stale_caches.py --list | less

# 7. Confirm cache-key isolation works (set a unique suffix)
DENY_LIST_HASH_SUFFIX=$(date +%s) python run_strategy.py --use-cz-signals --cz-only coskewness

# 8. Memory profile (Linux only — PowerShell equivalent: Measure-Command)
/usr/bin/time -v python run_strategy.py 2>&1 | grep "Maximum resident"

# 9. Quick feature-importance diff across runs
python compare_ml_caches.py
```

---

## 39. Data flow in one page (ASCII)

```
                    RAW SOURCES                         FEATURES                   MODEL                 PORTFOLIO
──────────────────────────────────┬─────────────────────────────┬─────────────────────────┬───────────────────────
EODHD prices $20/mo  ┐             │                             │                         │
  (+$60 for bulk     ├─► data_loader.load_prices ─► OHLCV  ────┐ │                         │
   earnings only)    │   universe_builder          (MultiIdx) │ │                         │
                     │   sector_mapper (EDGAR SIC)            │ │                         │
                     │                                         ▼ │                         │
Free SEC EDGAR  ─────┼─► alt_data_loader.load_edgar_*  ──► alt_features.py   ──►         │
                     │   (15 XBRL concepts, 10-K/10-Q text)    │    build_*_signals       │
                     │                                         │    ~80 alt signals        │
Free FRED API  ──────┼─► load_fred_macro,                     │                           │
 (9 macro, 4 used)   │   load_vix_term_structure, DXY, VVIX, │                           │
                     │   oil, Cu/Au, OAS, EBP, treasuries     │                           │
Free CBOE VIX URLs ──┤                                         │                           │
                     │                                         │                           │
yfinance (fallback) ─┼─► earnings_calendar, analyst_*,        │                           │
                     │   institutional, short_interest         │                           │
                     │                                         │                           │
Chen-Zimmermann ─────┼─► cz_signals.py  ─► 10 anomaly signals ┘                           │
 (1.3 GB parquet)    │                                                                     │
                     │                                                                     │
Tradier API (free) ──┼─► tradier_client + tradier_orats_adapter                           │
 + Orats $399 (1x)   │   → chain_to_smv_summary                                            │
                     │   → orats_loader → options_signals.py (15 signals, currently OFF)   │
                     │                                         │                           │
SRAF / LM dicts   ───┼─► sraf_sentiment, lm_sentiment,        │                           │
 (optional)          │   lazy_prices_features                  │                           │
                     │                                         ▼                           │
                                                        features.py (core 24)
                                                        + alt_features (~80)
                                                        + cz_signals (10)
                                                        + options (15 gated)
                                                        + factor_mom (6)
                                                        + cross_asset_leadlag (15 — drop!)
                                                        ───────────────────────
                                                        ~350 candidate features
                                                              │
                                                        model._DENY_LIST (93 cut)
                                                              │
                                                              ▼
                                                        ~257 features → build_feature_matrix
                                                              │
                                                              ▼
                                                        build_labels (6-grade LambdaRank, fwd=7)
                                                              │
                                                              ▼
                                                        WalkForwardModel (52 folds, 63d retrain)
                                                              │
                                                        ┌─────┴─────┬──────┬─────────┐
                                                        │           │      │         │
                                                       LGBM       XGB   Ridge     (+ MLP, Huber, Quantile)
                                                        │           │      │         │
                                                        └─── rank average + era weights ──┘
                                                              │
                                                        meta-label reject gate
                                                              │                         ▼
                                                      70% ML + 30% 12-1 mom ──► portfolio.build_monthly_portfolio
                                                                                        │ quality gate, sector caps,
                                                                                        │ SPY core, mega-cap force,
                                                                                        │ vol target, BSC, credit overlay
                                                                                        │
                                                                                        ▼
                                                                                 backtest.run_backtest
                                                                                 T+1, tiered costs 1/3/8/20 bps
                                                                                        │
                                                                                        ▼
                                                                                 tearsheet + dashboard.html
```

---

---

# PART III — Complete Feature Catalog

> Every predictor that can feed the model, where it comes from, what it means, how to turn it on/off. Part III is a reference for the curious or the refactoring.

## Conventions used throughout Part III

- **Sign**: `↑` = higher value → higher expected return; `↓` = inverted/negated before ranking. The pipeline expects `↑` convention after ranking.
- **Lag**: `shift(k)` = predictor at T uses data up to T-k (lookahead guard).
- **Window**: rolling window size in days. `min_periods` ≈ window/2 unless noted.
- **Gate**: CLI flag or env var controlling whether feature is built/included.
- **Filing lag**: EDGAR features lag raw data 45 days to approximate 10-Q/10-K filing delay.
- **Publication lag**: Form 4/SRAF features shift raw data forward by publication_lag_days BDay.
- **Variants auto-produced by the pipeline** (see §41): `_sn` (size-neutralized), `_sni` (sector-neutralized), `_csz` (cross-sectional z-score), `z_` prefix (early z-score), `ix_*` (interaction). These multiply the base feature count ~3-4×.

## Totals

- **Core / features.py**: ~70 base predictors (24 ranked composites, 12 tail/higher-moment, 10 breadth/wavelet, ~24 in the ranking basket, 7 liquidity/beta/price)
- **Alt / alt_features.py**: ~80 base predictors (fundamentals, macro, distress, insider, analyst, earnings, FINRA)
- **C&Z / cz_signals.py**: 10 signals (3 price + 7 accounting)
- **Factor momentum**: 7 (1 composite + 6 per-factor)
- **Cross-asset lead-lag**: 15 (5 assets × 3 horizons) — currently ZERO importance, drop candidates
- **Text / NLP**: 14 (3 lazy prices + 7 LM + 4 SRAF)
- **Options**: 15 (5 C&Z + 7 academic + 3 Orats-bonus) — blocked on data validation
- **Interactions + variants**: ~120 (11 `ix_*` + ~15 `_sn` + ~12 `_sni` + ~50 `_csz` + early `z_`)
- **After _DENY_LIST (93 entries)**: ~257 features enter the model
- **Before deny-list**: ~350 candidate features

---

## 40. Pipeline order of operations

From `model.py::build_feature_matrix` (line 174) and `run_strategy.py`:

```
STAGE 1 — Raw construction (features.py + alt_features.py + cz_signals.py + ...)
STAGE 2 — Signal smoothing: EMA halflife=10 (portfolio.py:27)
STAGE 3 — Size-neutralize: regress feature ~ log_mcap_z per date, keep residual → _sn suffix (model.py:131-172)
STAGE 4 — Sector-neutralize: subtract sector mean per date → _sni suffix (features.py:922-963)
STAGE 5 — Winsorize: 1st/99th percentile cross-sectional clip (model.py:762-768)
STAGE 6 — z_ prefix: early z-score for 4 features (rvol_21d, idiovol_21d, amihud_21d, mkt_beta_63d) (model.py:499-501)
STAGE 7 — Macro × stock interactions: 6 ix_* features (model.py:681-694)
STAGE 8 — Stock × stock interactions: 5 ix_* features (model.py:706-714)
STAGE 9 — _csz: cross-sectional z-score (±4 clip) applied to ALL continuous features (model.py:770-794)
STAGE 10 — Deny-list filter: 93 entries removed (model.py:800, 859-868)
STAGE 11 — Coverage gate: features with >10% non-null per-date required (model.py:652, 659)
STAGE 12 — Cache save: feature_panel_{hash}.pkl (model.py:918)
```

---

## 41. Transformation system (variants auto-produced)

| Suffix/Prefix | Construction | Clip | Scope | Line |
|---|---|---|---|---|
| **`_sn`** (size-neutralized) | Per-date OLS: `feature - (α + β×log_mcap_z)`; skipped if <10 tickers or zero log_mcap variance | — | 15 base features get `_sn` variants | model.py:131-172 |
| **`_sni`** (sector-neutralized) | Per-date subtract sector mean; skip sectors with <3 members | — | 13 base features get `_sni` variants | features.py:922-945 |
| **`_csz`** (cross-sec z-score) | Per-date `(x - row_mean) / row_std` | ±4 | ALL continuous features | model.py:770-794 |
| **`z_` prefix** | Early z-score (before sector/interactions) | ±4 | 4 features: `rvol_21d`, `idiovol_21d`, `amihud_21d`, `mkt_beta_63d` | model.py:499-501 |
| **`ix_` prefix** | Pairwise multiply + per-date z-score | ±4 | 11 interactions (6 macro×stock + 5 stock×stock) | model.py:681-714 |

### The 15 `_sn` variants
`mom_21d_sn`, `mom_63d_sn`, `mom_126d_sn`, `mom_12_1_sn`, `z_mom_63d_sn`, `z_mom_126d_sn`, `z_mom_12_1_sn`, `rvol_21d_sn`, `idiovol_21d_sn`, `z_rvol_21d_sn`, `z_idiovol_21d_sn`, `mkt_beta_63d_sn`, `z_mkt_beta_63d_sn`, `amihud_21d_sn`, `z_amihud_21d_sn`.

### The 12 `_sni` variants (audited — amihud is NOT sector-neutralized)
`mom_63d_sni`, `mom_126d_sni`, `mom_12_1_sni`, `z_mom_63d_sni`, `z_mom_126d_sni`, `z_mom_12_1_sni`, `idiovol_21d_sni`, `rvol_21d_sni`, `z_idiovol_21d_sni`, `z_rvol_21d_sni`, `mkt_beta_63d_sni`, `z_mkt_beta_63d_sni`. Plus fundamentals `_sni` (12 more) and distress `_sni` (3 more) when `sector_neutralize_fundamentals=True` and `use_sni_variants=True`.

### The 11 `ix_*` interactions
**Macro × stock (6)**: `ix_beta_x_yield_curve`, `ix_idiovol_x_hy_spread`, `ix_mom_x_vix`, `ix_rvol_x_vix_pctile`, `ix_longmom_x_yield_curve`, `ix_amihud_x_hy_spread`.
**Stock × stock (5)**: `ix_mom_x_vol`, `ix_mom_x_hurst`, `ix_idiovol_x_illiq`, `ix_beta_x_vol`, `ix_high52w_x_mom`.

---

## 42. Core momentum (features.py)

| Feature | Formula | Builder:line | Window | Lag | Sign | Gate | Notes |
|---|---|---|---|---|---|---|---|
| `mom_5d` | `close[T-1]/close[T-6] - 1` | features.py:31 | 5d | shift(1) | ↑ | always | part of short_horizon_momentum composite |
| `mom_10d` | `close[T-1]/close[T-11] - 1` | features.py:31 | 10d | shift(1) | ↑ | always | IC decays at very short horizon |
| `mom_21d` | `close[T-1]/close[T-22] - 1` | features.py:31 | 21d | shift(1) | ↑ | always | in _DENY_LIST |
| `mom_63d` | `close[T-21]/close[T-84] - 1` | features.py:58 | 63d | shift(21) | ↑ | always | JT skip-21d; in many interactions |
| `mom_126d` | `close[T-21]/close[T-147] - 1` | features.py:58 | 126d | shift(21) | ↑ | always | strongest continuation horizon |
| `mom_12_1` | 252d return skip-21d | features.py:58 | 252d | shift(21) | ↑ | always | Carhart 1997 standard |
| `residual_mom_21d` | `sum(r_i - r_mkt, 21d)` | features.py:355 | 21d | shift(1) | ↑ | always | market-neutral momentum; in _DENY_LIST |
| `price_accel_21d` | `mom(T) - mom(T-21)` | features.py:374 | 21d | shift(1) | ↑ | always | in _DENY_LIST |
| `efficiency_21d` | `|close[T]-close[T-21]| / Σ|Δclose|` | features.py:410 | 21d | — | ↑ | always | Kaufman ER [0,1]; in _DENY_LIST |
| `short_term_reversal` (1d/2d) | `-Σ returns[T-w:T-1]` | features.py:478 | 1-2d | shift(1) | ↓(→↑) | always | Jegadeesh 1990; in _DENY_LIST |
| `volume_confirmed_reversal` | `-ret × clip(vol/avg_vol, 0.5, 3)` | features.py:497 | 20d | shift(1) | ↓(→↑) | always | exhaustion signal; in _DENY_LIST |
| `sector_rel_mom_63d` | `stock_cum_return - sector_avg_cum_return` | features.py:521 | 63d | shift(1) | ↑ | `sector_map` present | skip sectors <3 tickers |

---

## 43. Mean reversion (features.py)

| Feature | Formula | Builder:line | Window | Lag | Sign | Gate | Notes |
|---|---|---|---|---|---|---|---|
| `zscore_reversion` | `-(close - rolling_mean)/rolling_std` | features.py:79 | 20d | — | ↓(→↑) | always | oversold → high score |
| `rsi` | `50 - RSI(close, 14)` using Wilder smoothing | features.py:93 | 14d | — | ↓(→↑) | always | α=1/14 EWM |
| `composite_mean_reversion` | `0.5 × rank(zscore_rev) + 0.5 × rank(rsi)` | features.py:109 | 14/20d | — | ↑ | always | already ranked inside |
| `bollinger_pct_b` | `-(close - lower)/(upper - lower)` | features.py:335 | 20d, 2σ | — | ↓(→↑) | always | lower band (0) → high score |
| `ma_distance` | `-(close - MA50)/MA50` | features.py:428 | 50d | — | ↓(→↑) | always | also called ma50_distance |
| `macd_signal` | `EMA12 - EMA26 - EMA9(MACD)` | features.py:233 | 12/26/9 | — | ↑ | always | histogram; trend complement |

---

## 44. Volatility core (features.py)

| Feature | Formula | Builder:line | Window | Lag | Sign | Gate | Notes |
|---|---|---|---|---|---|---|---|
| `rvol_10d`, `rvol_21d`, `rvol_63d` | `returns.rolling(w).std() × √252` | features.py:125 | 10/21/63d | — | ↑ | always | vol-premium family, tops feature importance |
| `vol_regime` | `-short_vol(10) / long_vol(60)` | features.py:133 | 10/60d | — | ↓(→↑) | always | calm regime → high score |
| `idiovol_21d` | `std(r_i - r_mkt_ew)` | features.py:211 | 21d | — | ↓(implicit) | always | **#1 importance feature (sn variant)**; EW market proxy |
| `vol_of_vol` | `-std(rvol_5, 21d)` | features.py:445 | 5/21d | — | ↓(→↑) | always | drop candidate (bottom decile) |
| `mkt_beta_63d` | `Cov(r_i, r_mkt)/Var(r_mkt)` | features.py:290 | 63d | — | ↓(BAB implicit) | always | Frazzini-Pedersen; rank_market_beta uses |

---

## 45. Tail / semi-beta / lottery (features.py)

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `max_return` | `-max(daily_ret[T-21:T-1])` | features.py:169 | 21d | ↓(→↑) | always | Bali-Cakici-Whitelaw lottery avoidance |
| `tail_risk` | `-5th_percentile(returns, 21d)` | features.py:462 | 21d | ↓(→↑) | always | crash-aversion premium |
| `realized_skewness` | `-skew(returns, 21d)` | features.py:970 | 21d | ↓(→↑) | always | Amaya et al. 2015 |
| `co_skewness` | `-E[ε_i · ε_m²]/(σ_i σ_m²)` 252d | features.py:990 | 252d | ↓(→↑) | `DISABLE_CO_SKEW_252` env | Harvey-Siddique 2000; cumsum-optimized |
| `semi_beta_N` | `Σ r_i⁻·r_m⁻ / Σ r_m²` | features.py:1081 | 252d | ↑ | always | concordant negative — strongest premium |
| `semi_beta_P` | `Σ r_i⁺·r_m⁺ / Σ r_m²` | features.py:1081 | 252d | ↑ | always | concordant positive; top-10 importance |
| `semi_beta_Mplus` | `Σ r_i⁺·r_m⁻ / Σ r_m²` | features.py:1081 | 252d | ↑ | always | mixed (stock up, market down) |
| `semi_beta_Mminus` | `Σ r_i⁻·r_m⁺ / Σ r_m²` | features.py:1081 | 252d | ↑ | always | mixed (stock down, market up) |
| `tail_dep_lower_252` | `P(r_i ≤ q10 AND r_m ≤ q10)` | features.py:1149 | 252d, q=0.1 | ↑ | always | Chabi-Yo-Ruenzi-Weigert 2018 |
| `signed_jump_21` | `Σ r²·sign(r)·1{|r|>4σ}` | features.py:1177 | 21d, threshold=4σ | ↑ | always | Jiang-Yao 2013; in _DENY_LIST |
| `downside_upside_vol_ratio_63` | `√(Σr²|r<0) / √(Σr²|r>0)` | features.py:1200 | 63d | ↑ | always | left-skew indicator |
| `downside_beta_spread_252` | `β⁻ - β⁺` conditional on market regime | features.py:1219 | 252d | ↑ | always | Ang-Chen-Xing 2006 |
| `kumar_lottery_21` | `-(z_ivol + z_iskew - z_price)` | features.py:1266 | 21d | ↓(→↑) | always | Kumar 2009 lottery composite |

---

## 46. Liquidity, beta, size (features.py + model.py)

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `amihud_21d` | `mean(|r_t|/(P_t·V_t), 21d)` | features.py:187 | 21d | ↑ (illiq premium) | always | dollar volume denominator |
| `hurst_63d` | `0.5 × log₂(var_ratio)` clipped [0,1] | features.py:562 | 63d, lag=16 | ↑ trending | always | variance ratio approximation |
| `price_52w_high` | `close[T-1]/max(close[T-252:T-1])` | features.py:151 | 252d | ↑ | always | George-Hwang 2004 reference point |
| `log_mcap_z` | z-score of `log(close×shares_out)` | model.py:553 | cross-sec | ↑ | ≥20% shares coverage | clipped ±4σ |
| `log_liquidity_z` | z-score of `log(close×volume_21d_avg)` | model.py:535 | 21d + cross-sec | ↑ | always | fallback when shares unavailable |
| `log_price_z` | z-score of `log(close)` | model.py:540 | cross-sec | ↑ | close>0 | separate from log_mcap_z |

---

## 47. Volume signals (features.py)

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `obv_signal` | z-score of cumsum(sign(ret)·volume) | features.py:317 | 60d z-window | ↑ | always | On-Balance Volume |
| `volume_price_trend` (VPT) | z-score of Σ(pct_change × volume, 20d) | features.py:272 | 20d/60d | ↑ | always | price-weighted volume |
| `volume_trend` | z-score of 21d pct_change of volume MA | features.py:392 | 21d/63d | ↑ | always | participation growth |
| `volume_spike` | `volume / MA(volume, 20d)` | features.py:263 | 20d | raw ratio | always | feeds volume_confirmed_reversal |
| `volume_confirmed_reversal` | `-ret_{T-1} × clip(vol_ratio, 0.5, 3)` | features.py:497 | 20d | ↓(→↑) | always | high-vol sell-off reverses stronger |

---

## 48. Breadth + wavelet (features.py)

All breadth signals are **time-series scalars broadcast across all tickers** (per-date constant). Wavelet features are per-ticker.

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `pct_above_200ma` | `(close > MA200).mean(axis=1)` | features.py:1311 | 200d, min=100 | ↑ | `--use-breadth-wavelet` | broadcast, in _DENY_LIST |
| `new_highs_minus_lows` | `(#52w_highs - #52w_lows)/N` | features.py:1327 | 252d, min=126 | ↑ | `--use-breadth-wavelet` | broadcast, in _DENY_LIST |
| `advance_decline_ratio` | `#(r>0)/#(r<0)` | features.py:1349 | daily | ↑ | `--use-breadth-wavelet` | broadcast, in _DENY_LIST |
| `breadth_z` | composite z-score of 3 above | features.py:1363 | 252d | ↑ | `--use-breadth-wavelet` | broadcast |
| `wavelet_intra_week_energy` | FFT power <5d periods | features.py:1392 | 256d FFT | ↑ normalized | `--use-breadth-wavelet` | chunked 500 tickers |
| `wavelet_weekly_energy` | FFT power 5-21d | features.py:1392 | 256d FFT | ↑ | same | |
| `wavelet_monthly_energy` | FFT power 21-63d | features.py:1392 | 256d FFT | ↑ | same | |
| `wavelet_quarterly_energy` | FFT power 63-256d | features.py:1392 | 256d FFT | ↑ | same | |
| `wavelet_dominant_frequency` | argmax of power spectrum | features.py:1392 | 256d FFT | — | same | in _DENY_LIST |
| `wavelet_spectral_entropy` | `-Σp·log(p)` over bands | features.py:1392 | 256d FFT | — | same | noise/randomness measure |

---

## 49. Ranked composites — the "rule-based" 30% signal (features.py::build_composite_signal)

24 raw signals are cross-sectionally ranked to `[0,1]` and weighted into a single composite. The composite becomes the `rank_*` features in the model AND the 30% leg of the 70/30 ML-plus-momentum blend.

| Signal | Weight | Underlying | |
|---|---|---|---|
| `rank_idiovol` | 0.08 | `idiosyncratic_volatility(21d)` | |
| `rank_amihud` | 0.06 | `amihud_illiquidity(21d)` | |
| `rank_long_momentum` | 0.08 | `long_horizon_momentum` (63/126/12-1) | |
| `rank_market_beta` | 0.06 | `market_beta(63d)` | |
| `rank_residual_momentum` | 0.05 | `residual_momentum(21d)` | |
| `rank_reversal_1d` | 0.05 | `short_term_reversal(1d)` | strongest short signal |
| `rank_reversal_2d` | 0.03 | `short_term_reversal(2d)` | |
| `rank_vol_reversal` | 0.03 | `volume_confirmed_reversal` | |
| `rank_sector_rel_mom` | 0.04 (if map) | `sector_relative_momentum(63d)` | else 0 |
| `rank_macd` | 0.05 | `macd_signal` | |
| `rank_price_52w_high` | 0.06 | `price_to_52w_high` | |
| `rank_efficiency_ratio` | 0.03 | `efficiency_ratio(21d)` | |
| `rank_price_acceleration` | 0.04 | `price_acceleration(21d)` | |
| `rank_mean_reversion` | 0.06 | `composite_mean_reversion` | |
| `rank_bollinger_pct_b` | 0.04 | `bollinger_pct_b` | |
| `rank_ma_distance` | 0.03 | `ma_distance(50d)` | |
| `rank_max_effect` | 0.04 | `max_return(21d)` | lottery |
| `rank_tail_risk` | 0.04 | `tail_risk(21d)` | |
| `rank_vol_of_vol` | 0.03 | `vol_of_vol(5/21)` | drop candidate |
| `rank_obv` | 0.03 | `obv_signal` | |
| `rank_volume_trend` | 0.02 | `volume_trend(21d)` | |
| `rank_vpt` | 0.01 | `volume_price_trend(20d)` | |
| `rank_vol_regime` | 0.02 | `volatility_regime(10/60)` | |
| `rank_short_momentum` | 0.01 | `short_horizon_momentum` (5/10/21) | |

**IC-weighted option**: `build_ic_weighted_composite()` (features.py:638) — 63d rolling Spearman-IC per signal vs 21d forward return; negative IC clipped to 0; equal-weight fallback. Enabled via `use_ic_weights=True` on `build_composite_signal()`.

---

## 50. Fundamentals (alt_features.py::build_fundamental_signals, EDGAR)

All use 45-day filing lag + quarterly→daily forward-fill. Default ON. Each generates a `_sni` variant when `sector_neutralize_fundamentals=True`.

| Feature | Formula | Raw EDGAR | Clip | Sign | Notes |
|---|---|---|---|---|---|
| `book_to_market_signal` | StockholdersEquity / Assets | equity, assets | — | ↑ | cheap valuation |
| `roe_signal` | NetIncomeLoss / StockholdersEquity | NI, equity | — | ↑ | |
| `gross_margin_signal` | GrossProfit / Revenues | GP, revenues | ±1 | ↑ | |
| `asset_growth_signal` | `-YoY_pct_change(Assets, 252)` | assets | [-1,2] | ↓(→↑) | contrarian (low growth good) |
| `leverage_signal` | `-Liabilities / Assets` | liab, assets | [0,5] | ↓(→↑) | low debt → good |
| `eps_trend_signal` | `EPS[t] - EPS[t-252]` | EPS basic | — | ↑ | |
| `gross_profitability_signal` | GrossProfit / Assets | GP, assets | [-1,2] | ↑ | Novy-Marx 2013 |
| `accruals_signal` | `-(NI - OCF) / Assets` | NI, OCF, assets | ±2 | ↓(→↑) | Sloan 1996 |
| `net_operating_assets_signal` | `-((A-Cash) - (L-LTDebt))/A` | assets, cash, liab, LT debt | [-2,5] | ↓(→↑) | Hirshleifer 2004 |
| `net_share_issuance_signal` | `-log(Shares[t]/Shares[t-252])` | shares_out | ±1 | ↓(→↑) | Pontiff-Woodgate 2008 |
| `operating_profitability_signal` | (Rev - GP - SGA - IntExp)/Equity | rev, GP, SGA, IntExp, equity | [-2,5] | ↑ | FF5 RMW |
| `current_ratio_chg_signal` | diff(CurA/CurL, 252d) | currentA, currentL | ±2 | ↑ | Piotroski F component |

+ 12 `_sni` variants when flag on. Raw EDGAR concepts cached in `edgar_fundamentals_*_v2raw.pkl`.

---

## 51. Tier 3 academic value/quality (alt_features.py, lines 231-356)

Gated by `--use-tier3-academic` (default True). All require `market_cap` (close × shares_out).

| Feature | Formula | Clip | Gate | Notes |
|---|---|---|---|---|
| `earnings_yield_signal` | NetIncome / MarketCap | ±5 | always on | |
| `sales_yield_signal` | Revenue / MarketCap | ±50 | always on | wider clip for scale |
| `fcf_yield_signal` | OCF / MarketCap | ±5 | always on | proxy for FCF (capex unavailable) |
| `ebit_ev_signal` | (NI + IntExp) / (MC + LTDebt - Cash) | ±5 | always on | EV normalization |
| `value_composite_signal` | mean-rank of {earn_y, sales_y, fcf_y, ebit_ev} | — | always on | Asness-Frazzini 2013 |
| `cash_based_op_prof_signal` | (Rev - COGS - SGA - ΔWC) / Assets | ±5 | `DISABLE_CASH_BASED_OP_PROF=0` | Ball et al. 2016 |
| `piotroski_f_score_signal` | sum of 9 binary flags (f1-f9), then rank | [0,9] | always on | Piotroski 2000 |

---

## 52. Earnings signals (alt_features.py::build_earnings_signals)

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `earnings_surprise_signal` | rank(eps_surprise_pct), ffill 60d post-ann | alt_features.py:602 | 60d carry | ↑ | always on | yfinance or EODHD source |
| `days_to_earnings_signal` | `-days_to_next_earnings`, clip window=30 | alt_features.py:606 | 30d | ↓(→↑) | always on | pre-earnings drift |
| `earnings_ann_return_signal` (EAR) | 3-day announcement return, ffill 60d | alt_features.py:656 | [t,t+2] AMC or [t-1,t+1] BMO | ↑ | `--use-tier3-academic` | Brandt 2008; uses `before_after_market` |
| `earnings_beat_streak_signal` | consec + surprise count capped 8 | alt_features.py:684 | rolling | ↑ | `ENABLE_PHASE_D_SIGNALS=1` | Phase D |
| `earnings_surprise_consistency_signal` | `-CoV(surprise, 8Q rolling)` | alt_features.py:716 | 8 quarters | ↓(→↑) | `ENABLE_PHASE_D_SIGNALS=1` | Phase D |

---

## 53. Distress (alt_features.py lines 1056-1209)

| Feature | Formula | Builder:line | Window | Sign | Gate | Notes |
|---|---|---|---|---|---|---|
| `chs_distress_signal` | 8-var Campbell-Hilscher-Szilagyi logit | alt_features.py:1056 | 252d rolling | ↑ as distress (→ short) | `use_chs=True` | cumsum-opt; + `_sni` variant |
| `naive_dtd_signal` | Bharath-Shumway Distance-to-Default | alt_features.py:1131 | 252d + T=1y | ↑ safety | `use_dtd=True` | + `_sni` |
| `altman_z_signal` | Z = 1.2WC/TA + 1.4RE/TA + 3.3EBIT/TA + 0.6ME/L + 1.0S/TA | alt_features.py:1169 | PIT | ↑ safety | `use_altman_z=True` | clip ±20; + `_sni` |

---

## 54. Macro broadcasts — EXCLUDED from raw input, only used for interactions

All z-scored (252d rolling, clip ±4), shifted 1 BDay. **Excluded at model.py:636-641.** Only enter via `ix_*` interactions.

| Feature | Series ID | Source | Notes |
|---|---|---|---|
| `yield_curve` | T10Y2Y | FRED | used in `ix_beta_x_yield_curve`, `ix_longmom_x_yield_curve` |
| `hy_spread` | BAMLH0A0HYM2 | FRED | used in `ix_idiovol_x_hy_spread`, `ix_amihud_x_hy_spread` |
| `vix_macro` | VIXCLS | FRED (or ^VIX fallback) | used in `ix_mom_x_vix` |
| `fed_funds` | DFF | FRED | **loaded but unused** — candidate for deletion |
| `unemployment` | UNRATE | FRED | **loaded but unused** |
| `breakeven_inflation` | T10YIE | FRED | **loaded but unused** |
| `ism_pmi` | NAPM | FRED | **loaded but unused** |
| `initial_claims` | ICSA | FRED | **loaded but unused** |
| `building_permits` | PERMIT | FRED | **loaded but unused** |
| `yield_curve_chg5d` | 5d diff(T10Y2Y) | FRED | curve momentum |
| `vix_term_slope` | VIX/VIX3M | CBOE | used in `ix_rvol_x_vix_pctile` context |
| `vix_percentile` | rolling 252d rank of VIX | CBOE | used in `ix_rvol_x_vix_pctile` |
| `vix_change_5d` | 5d pct change of VIX | CBOE | |
| `vvix_vix_ratio` | VVIX/VIX | CBOE | broadcast |
| `vvix_vix_ratio_chg5d` | 5d diff | CBOE | broadcast |

---

## 55. Macro cross-sectional betas (alt_features.py lines 1230-1510)

All gated by `use_macro_cross_section=True` (default). Rolling regression of stock returns on macro series; per-stock β cross-sectionally ranked.

| Feature | Formula | Window | Sign | Notes |
|---|---|---|---|---|
| `dxy_beta_signal` | `-rolling_β(r_i, Δdxy, 756d)` | 36mo | ↓(→↑) | **#2 feature importance overall** |
| `dxy_beta_x_dxy_mom_regime` | β × sign(63d dxy momentum) | 36mo × 63d | regime-dep | interaction |
| `pc2_slope_beta_signal` | β to yield-curve PC2 (slope) | 60mo (weekly) | ↑ | PCA on Treasury curve |
| `cu_au_zscore` | z-score of Cu/Au ratio | 252d | broadcast | not per-stock |
| `cu_au_roc_3m` | 63d ROC of Cu/Au | 63d | broadcast | |
| `cyclicality_cu_au_signal` | β to Δ(Cu/Au), 60mo | 1260d | ↑ | per-stock cyclicality |
| `credit_beta_signal` | β to Δ(HY-IG OAS), 36mo | 756d | ↑ | credit-cycle sensitivity |
| `ebp_beta_signal` | β to ΔExcessBondPremium, 36mo | 756d | ↑ | |
| `mom_spy_12_1` | SPY 12-1 momentum | 252d | broadcast | |
| `mom_tlt_12_1` | TLT 12-1 | 252d | broadcast | |
| `mom_dbc_12_1` | DBC 12-1 | 252d | broadcast | |
| `mom_uup_12_1` | UUP 12-1 | 252d | broadcast | |
| `mom_gld_12_1` | GLD 12-1 | 252d | broadcast | |
| `mom_hyg_12_1` | HYG 12-1 | 252d | broadcast | in _DENY_LIST |
| `inflation_sensitivity_signal` | β to Δbreakeven × regime, 60mo | 1260d | regime-dep | |
| `oil_beta_signal` | `-β(r_i, Δoil, 60mo)` | 1260d | ↓(→↑) | |
| `oil_beta_x_oil_momentum` | β × sign(21d oil momentum) | 1260d × 21d | regime-dep | interaction |
| `sector_oas_momentum_signal` | -21d OAS change → sector bucket | 21d | ↓(→↑) | BBB defensives, BB cyclicals |

---

## 56. Cross-asset lead-lag (cross_asset_leadlag.py) — **DROP CANDIDATES**

15 features: 5 assets × 3 horizons. All show ZERO feature importance in top-500. Proposed deletion via `EXTRA_DENY_FEATURES="leadlag_*"`.

| Asset | Horizons | Formula | Sign | Builder |
|---|---|---|---|---|
| oil | 5d, 10d, 20d | `β(stock, oil, 252d) × Σ(oil_r[t-lag:t-1])` | ↑ | cross_asset_leadlag.py:92 |
| copper | 5d, 10d, 20d | same | ↑ | same |
| gold | 5d, 10d, 20d | same | ↑ | same |
| dxy | 5d, 10d, 20d | `-β × Σ(dxy_r)` | ↓ (invert) | same |
| hy_oas | 5d, 10d, 20d | `-β × Σ(Δhy_oas)` | ↓ (invert) | same |

Hong-Stein 1999 / Wang 2023 IRF — theoretically sound, empirically zero lift in our universe.

---

## 57. Insider / analyst / institutional / short interest (alt_features.py)

| Feature | Formula | Builder:line | Sign | Gate | Notes |
|---|---|---|---|---|---|
| `insider_activity_signal` | rolling 60d Form 4 count | alt_features.py:744 | ↑ | always | Lakonishok-Lee 2001 |
| `insider_net_buy_signal` | rolling 60d buys - sells | alt_features.py:744 | ↑ | always | |
| `insider_cluster_buy_signal` | binary if ≥3 buys / 30d | alt_features.py:816 | ↑ | always | Hameed-Lee 2010 |
| `opportunistic_insider_signal` | CMP-classified net $, EWMA hl=60d | opportunistic_insider_signal.py:148 | ↑ | always | Cohen-Malloy-Pomorski 2012; T+2 lag |
| `analyst_revision_signal` | rolling 60d (upgrades - downgrades) | alt_features.py:859 | ↑ | always | Womack 1996 |
| `analyst_coverage_signal` | rolling 60d action count | alt_features.py:859 | ↑ | always | coverage intensity |
| `target_upside_signal` | **snapshot** rank(targetUpside) broadcast | alt_features.py:932 | ↑ | always | **lookahead risk** |
| `analyst_rating_signal` | **snapshot** -rank(recommendationMean) | alt_features.py:960 | ↑ | always | **lookahead risk** |
| `institutional_ownership_signal` | **snapshot** rank(pctHeldByInstitutions) | alt_features.py:978 | ↑ | `use_institutional_holdings` (OFF) | snapshot bias |
| `insider_ownership_signal` | **snapshot** rank(pctHeldByInsiders) | alt_features.py:1005 | ↑ | `use_institutional_holdings` (OFF) | snapshot bias |
| `short_ratio_signal` | **snapshot** rank(short_ratio) | alt_features.py:476 | ↑ bearish | `use_short_interest` (OFF) | yfinance snapshot only |
| `short_pct_float_signal` | **snapshot** rank(short_pct_float) | alt_features.py:496 | ↑ bearish | `use_short_interest` (OFF) | same |
| `finra_delta_si_signal` | rank(ΔSI, 21d) | alt_features.py:1821 | ↑ bearish | `use_finra_short_interest=True` | Boehmer-Jones-Zhang 2008 |
| `finra_delta_si_3m_signal` | rank(ΔSI, 63d) | alt_features.py:1827 | ↑ bearish | same | Rapach-Ringgenberg-Zhou 2016 |
| `finra_si_level_signal` | rank(SI/shares_out) | alt_features.py:1833 | ↑ bearish | same | level, not change |

---

## 58. Chen-Zimmermann signals (cz_signals.py)

Gated by `--use-cz-signals`, with `--cz-only f1,f2` / `--cz-exclude f3` for subsets.

| Feature | Formula | Builder:line | Window | Sign | Published IC_IR | Raw data |
|---|---|---|---|---|---|---|
| `coskewness_signal` | `-E[r_i·r_m²]/(σ_i·σ_m²)` monthly | cz_signals.py:48 | 60 months | ↓(→↑) | 0.20 | monthly returns + SPY |
| `coskew_acx_signal` | same daily (Ang-Chen-Xing) | cz_signals.py:114 | 252d | ↓(→↑) | 0.12 | daily |
| `mom_season_signal` | avg return in same calendar month, yrs 6-10 | cz_signals.py:159 | 6-10yrs back | ↑ | 0.15 | Heston-Sadka 2008 |
| `payout_yield_signal` | (div+buybacks)/MC_lagged | cz_signals.py:238 | 45d filing lag, 6mo MC lag | ↑ | 0.15 | edgar_extra raw |
| `net_payout_yield_signal` | (div+buybacks-issuance)/MC | cz_signals.py:259 | same | ↑ | 0.18 | |
| `xfin_signal` | `-(sstk-dvc-prstkc+dltis-dltr)/Assets` | cz_signals.py:282 | 45d filing lag | ↓(→↑) | 0.18 | Bradshaw-Richardson-Sloan 2006 |
| `cfp_signal` | OCF / MarketCap | cz_signals.py:312 | 45d lag | ↑ | 0.11 | Desai et al. 2004 |
| `operprof_rd_signal` | (Rev - COGS - (SGA - R&D)) / Assets | cz_signals.py:326 | 45d lag | ↑ | 0.13 | Ball et al. 2016 |
| `tax_signal` | taxes / (0.21 × NI) | cz_signals.py:350 | 45d lag | ↑ | 0.12 | Lev-Nissim 2004 |
| `deldrc_signal` | ΔDeferred Revenue / avg(Assets) | cz_signals.py:369 | 45d lag, 252d diff | ↑ | 0.14 | Prakash-Sinha 2013 |

---

## 59. Factor momentum (factor_momentum_features.py)

Gated by `use_factor_momentum=True`. All z-scored cross-sectionally, clipped ±2.

| Feature | Formula | Builder:line | Window | Notes |
|---|---|---|---|---|
| `factor_mom_composite` | `Σ_k β_{i,k,t} × factor_mom_{k,t}` | factor_momentum_features.py:85 | 252d β + 12-1 mom shifted 21d | Ehsani-Linnainmaa 2022 |
| `factor_mom_SMB` | β_{i,SMB} × SMB_mom | :137 | same | size-factor mom |
| `factor_mom_HML` | β_{i,HML} × HML_mom | :137 | same | value |
| `factor_mom_RMW` | β_{i,RMW} × RMW_mom | :137 | same | profitability |
| `factor_mom_CMA` | β_{i,CMA} × CMA_mom | :137 | same | investment |
| `factor_mom_Mom` | β_{i,Mom} × Mom_mom | :137 | same | momentum of momentum |

Sourced from `factor_momentum_data.py::load_ff6_factors()` (Ken French library) + `compute_factor_momentum()` 12-1 log-cumsum.

---

## 60. Text / NLP (lazy_prices_features.py + lm_sentiment.py + sraf_sentiment.py)

All DEFAULT OFF (`--use-lazy-prices`, `--use-lm-sentiment`, `--use-sraf-sentiment`). Publication lag 2 BDay, forward-fill 180d, cross-section z-scored `±3`.

| Feature | Formula | Builder | Sign | Notes |
|---|---|---|---|---|
| `lazy_prices_item7` | TF-IDF cosine (1-2 grams) on MD&A vs prior filing | lazy_prices_features.py:420 | ↑ similarity = bullish | Cohen-Malloy-Nguyen 2020 |
| `lazy_prices_item1a` | same on Risk Factors | :420 | ↑ | |
| `lazy_prices_avg` | NaN-aware mean of both | :464 | ↑ | |
| `lm_pos_ratio` | pos_words / total | lm_sentiment.py:334 | ↑ | Loughran-McDonald dict |
| `lm_neg_ratio` | neg / total | :334 | ↓ | not exposed in feature_map |
| `lm_net_tone` | (pos-neg) / total | :334 | ↑ | core LM signal |
| `lm_uncertainty_ratio` | uncertainty / total | :334 | ↓ (→↑ negated) | |
| `lm_litigious_ratio` | litigious / total | :334 | ↓ (→↑) | |
| `lm_constraining_ratio` | constraining / total | :334 | ↓ (→↑) | not exposed |
| `lm_tone_change` | net_tone(t) - net_tone(t-1) per ticker | :518 | ↑ | |
| `sraf_net_tone` | pre-computed SRAF CSV | sraf_sentiment.py:139 | ↑ | no text parse |
| `sraf_uncertainty_ratio` | SRAF N_Uncertainty/N_Words | :139 | ↓ (→↑) | |
| `sraf_litigious_ratio` | SRAF N_Litigious/N_Words | :139 | ↓ (→↑) | |
| `sraf_tone_change` | net_tone diff per ticker | :157 | ↑ | |
| `sraf_filing_length` | log1p(N_Words) | :207 | — | complexity proxy |

**Known bug**: `sraf_sentiment` dispatch fails with "index must be monotonic" — sort index before dispatch in `alt_features.py`. Fix unlocks +130 bps per Fixed1 archive.

---

## 61. Options signals (options_signals.py) — 15 features, currently BLOCKED

Gated by `--use-options-signals` + `OPTIONS_MIN_COVERAGE=0.30` + `OPTIONS_WINSORIZE=0.01` + `OPTIONS_SIGNAL_SET={all, validated}` + `LAG_OPTIONS_SIGNALS=1` (prevents EOD lookahead).

| Feature | Formula | Builder:line | Raw data | Sign | Pub IC_IR | Tier |
|---|---|---|---|---|---|---|
| `dcpvolspread_signal` | `diff(call_IV - put_IV, 5d)` | options_signals.py:133 | Orats dlt25/75 | ↑ | 0.375 | validated |
| `smileslope_signal` | Orats `slope` (put skew) | :142 | Orats slope | ↑ | 0.369 | validated |
| `cpvolspread_signal` | level call_IV - put_IV | :152 | Orats deltas | ↑ | 0.155 | all |
| `dvolcall_signal` | `diff(log(call_vol), 5d)` | :160 | Orats cVolu | ↑ | 0.112 | all |
| `dvolput_signal` | `diff(log(put_vol), 5d)` | :171 | Orats pVolu | ↑ | 0.111 | all |
| `iv_rank_signal` | `-pct_rank(iv30, 252d)` | :186 | Orats iv30d | ↓(→↑) | — | academic |
| `rv_iv_spread_signal` | RV30 - IV30 | :195 | returns + iv30d | ↑ | — | academic |
| `variance_premium_signal` | `-(IV² - RV²)_30d` | :207 | same | ↓(→↑) | — | Han-Zhou 2012 |
| `iv_term_slope_signal` | iv60 - iv30 | :219 | Orats iv60d/30d | ↑ | — | Vasquez 2017 |
| `risk_reversal_25d_signal` | `-(25Δcall_IV - 25Δput_IV)` | :382 | Orats dlt25/75 | ↓(→↑) | — | Xing-Zhang-Zhao 2010 |
| `crash_risk_signal` | 5Δput_IV - ATM_IV | :393 | Orats dlt95/iv30d | ↑ | — | Kelly-Jiang |
| `oi_concentration_signal` | call_OI / (call_OI + put_OI) | :406 | Orats cOi/pOi | ↑ | — | positioning |
| `borrow30_signal` | rank(borrow30) | :682 | Orats borrow30 | ↑ | — | squeeze proxy |
| `dividend_surprise_signal` | rank(annIdiv - annActDiv) | :692 | Orats annIdiv | ↑ | — | earnings proxy |
| `etf_skew_relative_signal` | rank(-etfSlopeRatio) | :703 | Orats etfSlopeRatio | ↓(→↑) | — | sector-rel |

**Blocker 2026-04-17**: Phase 1 Tradier-vs-Orats validation failed (IV30d diff +2.5 vs 1.0 pt threshold, MSFT borrow +20.5 pt). Investigate `tradier_orats_adapter` field mapping before enabling.

---

## 62. Loader → feature map (data lineage in one table)

Every loader in `api_data.py` and `alt_data_loader.py` that feeds the feature panel:

| Raw source | Loader fn | Cache file | Features produced | Unused columns |
|---|---|---|---|---|
| EODHD prices | `data_loader.load_prices` | `eodhd_prices_*.pkl` | all of features.py (~70) | — |
| EODHD fundamentals | `api_data.load_eodhd_fundamentals` | `api/eodhd_fundamentals_v2.pkl` | (disabled — hurts CAGR) | all |
| EODHD sentiment | `api_data.load_eodhd_sentiment` | `api/eodhd_sentiment_v2.pkl` | `eodhd_sentiment`, `eodhd_sentiment_chg`, `eodhd_news_volume` | `count` |
| EODHD dividends | `api_data.load_eodhd_dividends` | `api/eodhd_dividends_v2.pkl` | `dividend_yield` (gated) | — |
| EODHD earnings bulk | `alt_data_loader.load_earnings_calendar_eodhd` | `earnings_calendar_eodhd_v2.pkl` | `earnings_surprise_signal`, EAR, `before_after_market` gates Phase D | |
| Finnhub | `api_data.load_finnhub_data` | `api/finnhub_all_data.pkl` | `fh_recommendation_score`, `fh_target_upside`, `fh_eps_surprise`, `fh_news_sentiment` (auto-disabled <1000) | — |
| SEC EDGAR (XBRL) | `alt_data_loader.load_edgar_fundamentals` | `edgar_fundamentals_v2raw.pkl` | 12 fundamentals + Tier 3 + distress + C&Z accounting | ~50% of concepts |
| SEC EDGAR (extra) | `alt_data_loader.load_edgar_fundamentals_extra` | `edgar_fundamentals_extra_v1.pkl` | C&Z payout/xfin/tax/deldrc inputs | |
| SEC EDGAR (text) | `lazy_prices_downloader.bulk_download_filings` | `edgar_text/*.json` | lazy prices, LM sentiment (if enabled) | — |
| SEC EDGAR (form 4) | `alt_data_loader.load_insider_transactions` | `insider_subs_v2.pkl` | **NONE consumed — unused loader!** | all |
| SEC EDGAR (CIK) | `sector_mapper.get_sectors_from_edgar` | `edgar_cik_map_v2.pkl`, `edgar_sic_codes_v2.pkl` | sector_map (feeds `_sni` + sector_rel_mom) | |
| EODHD insider | `insider_eodhd_loader.load_insider_transactions_v3` | `insider_eodhd_v3_*.pkl` | `insider_activity`, `insider_net_buy`, `insider_cluster_buy`, `opportunistic_insider` | — |
| yfinance earnings | `alt_data_loader.load_earnings_calendar` | `earnings_calendar_v2.pkl` | `earnings_surprise_signal` (fallback) | `eps_estimate`, `eps_actual`, `eps_surprise` raw |
| yfinance analyst | `alt_data_loader.load_analyst_actions` | `analyst_actions_v2.pkl` | `analyst_revision_signal`, `analyst_coverage_signal` | grade columns |
| yfinance estimates | `alt_data_loader.load_earnings_estimates` | `earnings_estimates_v2.pkl` | `target_upside_signal`, `analyst_rating_signal` | `targetMeanPrice`, `currentPrice`, `numberOfAnalysts` |
| yfinance institutional | `alt_data_loader.load_institutional_holders` | `institutional_holders_v2.pkl` | `institutional_ownership`, `insider_ownership` (OFF) | `institutionCount` |
| yfinance short (legacy) | `alt_data_loader.load_short_interest` | `short_interest_snapshot_v2.pkl` | `short_ratio`, `short_pct_float` (OFF) | — |
| FINRA short | `finra_short_interest.load_finra_short_interest` | `finra_si_v1.pkl` | `finra_delta_si`, `finra_delta_si_3m`, `finra_si_level` | — |
| FRED macro | `alt_data_loader.load_fred_macro` | `fred_macro_v2.pkl` | `yield_curve`, `hy_spread`, `vix_macro` (interactions only) | `fed_funds`, `UNRATE`, `NAPM`, `ICSA`, `PERMIT`, `T10YIE` |
| CBOE VIX term | `alt_data_loader.load_vix_term_structure` | `vix_term_v2.pkl` | `vix_term_slope`, `vix_percentile`, `vix_change_5d` (interactions) | VIX9D/3M/6M raw levels |
| FRED treasury curve | `alt_data_loader.load_treasury_yield_curve` | `treasury_yield_curve_v2.pkl` | `pc2_slope_beta_signal` | partially (8 maturities → PC2 only) |
| yfinance DXY | `alt_data_loader.load_dxy` | `dxy_v2.pkl` | `dxy_beta_signal` (**#2 importance**) | — |
| FRED breakeven | `alt_data_loader.load_breakeven_inflation` | `breakeven_v2.pkl` | `inflation_sensitivity_signal` | — |
| yfinance oil WTI | `alt_data_loader.load_oil_wti` | `oil_wti_v2.pkl` | `oil_beta_signal`, `oil_beta_x_oil_momentum` | — |
| yfinance Cu/Au | `alt_data_loader.load_copper_gold` | `copper_gold_v2.pkl` | `cu_au_zscore`, `cu_au_roc_3m`, `cyclicality_cu_au_signal` | — |
| yfinance cross-asset ETFs | `alt_data_loader.load_cross_asset_etf_panel` | `cross_asset_panel_v2.pkl` | `mom_spy/tlt/dbc/uup/gld/hyg_12_1` | — |
| FRED IG OAS | `alt_data_loader.load_ig_oas` | `ig_oas_v2.pkl` | feeds `credit_beta_signal` | — |
| FRED sector OAS | `alt_data_loader.load_sector_oas` | `sector_oas_v2.pkl` | `sector_oas_momentum_signal` | partial (BBB, BB used) |
| Fed Reserve EBP | `alt_data_loader.load_excess_bond_premium` | `ebp_v2.pkl` | `ebp_beta_signal` | gz_spread column |
| yfinance VVIX | `alt_data_loader.load_vvix` | `vvix_v2.pkl` | `vvix_vix_ratio`, `vvix_vix_ratio_chg5d` | — |
| Ken French | `factor_momentum_data.load_ff6_factors` | `ff6_factors_v1.pkl` | 6 factor_mom features | RF (risk-free) |
| Chen-Zimmermann | (external) | `chen_zimmermann/signed_predictors_dl_wide.parquet` | research-only (`run_cz_research.py`), not fed to live model | 200+ cz signals researched but only 10 built |
| Orats (historical) | `orats_loader.bulk_download_historical` | `options/iv_panels_orats.pkl` | 15 options signals (if enabled) | — |
| Tradier (live) | `tradier_client.fetch_universe_smv_panel` | `options/tradier_daily/YYYY-MM-DD.parquet` | feeds options signals via adapter | — |
| SRAF | (pre-built CSV) | `sraf/lm_10x_summaries_1993_2025.csv` | 5 SRAF sentiment features (OFF) | |

---

## 63. Environment-variable & flag matrix (complete)

| Flag / Env | Default | Effect | Features toggled |
|---|---|---|---|
| `--use-tier3-academic` | True | enable value/quality + EAR tier | earnings_yield, sales_yield, fcf_yield, ebit_ev, value_composite, piotroski, EAR, beat_streak (with env), surprise_consistency (with env) |
| `--sector-neutralize-fundamentals` | True | add `_sni` to fundamentals | 12 `_sni` variants |
| `--use-sni-variants` | True | add `_sni` to distress & macro | chs/dtd/altman `_sni`, macro-beta `_sni` |
| `--use-cz-signals` | False | 10 C&Z signals | coskew, coskew_acx, mom_season, payout_yield, net_payout_yield, xfin, cfp, operprof_rd, tax, deldrc |
| `--cz-only f1,f2,...` | — | keep only listed | subset |
| `--cz-exclude f1,f2,...` | — | exclude listed | subset |
| `--use-lazy-prices` | False | text similarity | lazy_prices_item7/item1a/avg |
| `--use-lm-sentiment` | False | LM dict sentiment | 7 LM features |
| `--use-sraf-sentiment` | False | SRAF pre-parsed | 5 SRAF features (currently broken — fix sort) |
| `--use-finra-short-interest` | True | FINRA bi-monthly | 3 finra features |
| `--use-eodhd-earnings` | True | bulk EODHD earnings | enables `before_after_market` → EAR timing |
| `--use-factor-momentum` | False | FF6 factor momentum | 6 factor_mom features |
| `--use-macro-cross-section` | True | macro betas | 17 macro-β features |
| `--use-options-signals` | False | options signals | 15 options features |
| `--use-higher-moment` | True | tail/semi-beta | 12 tail features |
| `--use-breadth-wavelet` | True | breadth + FFT | 10 features |
| `--use-institutional-holdings` | False | snapshot-based | institutional_ownership, insider_ownership |
| `--use-short-interest` | False | yfinance snapshot | short_ratio, short_pct_float |
| `--size-neutralize` | True | generate `_sn` | 15 `_sn` variants |
| `--sector-neutralize-features` | True | generate `_sni` | 13 `_sni` variants |
| `--winsorize` | True | 1st/99th clip | applied to all continuous |
| `--cs-zscore-all` | True | generate `_csz` | ~50 `_csz` variants |
| `--forward-window N` | 7 | label horizon | no direct feature effect, but retrain IC |
| `ENABLE_PHASE_D_SIGNALS` | 1 | earnings beat/consistency | beat_streak, surprise_consistency, EAR |
| `DISABLE_DIVIDEND_YIELD` | 0 | skip div feature when CZ on | dividend_yield |
| `DISABLE_CASH_BASED_OP_PROF` | 0 | skip CBOP (redundant with operprof_rd) | cash_based_op_prof_signal |
| `DISABLE_CO_SKEW_252` | 0 | memory saver | co_skewness |
| `EXTRA_DENY_FEATURES` | "" | comma-separated add to deny | runtime exclusion |
| `EMPTY_DENY_LIST` | 0 | clear 93-entry deny | ablation test |
| `DENY_LIST_HASH_SUFFIX` | "" | force cache-key change | isolation tests |
| `LAG_OPTIONS_SIGNALS` | 1 | options shift days | all 15 options features |
| `OPTIONS_MIN_COVERAGE` | **0.0** (intent 0.30, see options_signals.py:504-506 comment) | drop sparse tickers | **audit: code default is 0.0 — must set explicitly** |
| `OPTIONS_WINSORIZE` | **0.0** (intent 0.01, see options_signals.py:509 comment) | per-date 1%/99% clip | **audit: code default is 0.0 — must set explicitly** |
| `OPTIONS_SIGNAL_SET` | "all" | {"all","validated"} | validated = top 5 only |
| `FINNHUB_MIN_COVERAGE` | (doc-only, **not wired**) | — | no code consumer found in audit |

---

## 64. Known sign-convention gotchas

Every predictor must satisfy: **higher feature value → higher expected return** (post-ranking). The following are negated inside their builders to enforce this — DO NOT double-negate:

- `asset_growth_signal`, `leverage_signal`, `accruals_signal`, `net_operating_assets_signal`, `net_share_issuance_signal`, `days_to_earnings_signal`, `earnings_surprise_consistency_signal` — fundamental/earnings negations
- `short_term_reversal`, `volume_confirmed_reversal`, `zscore_reversion`, `rsi` (as `50-RSI`), `bollinger_pct_b`, `ma_distance` — mean-reversion group
- `max_return`, `tail_risk`, `realized_skewness`, `co_skewness`, `vol_of_vol`, `kumar_lottery_composite` — lottery/tail group
- `cash_based_op_prof_signal`, `earnings_surprise_consistency_signal` — via CoV inversion
- `market_beta` (features.py:311 explicit `-beta`), `dxy_beta_signal` (alt_features.py:1247 explicit `-beta`) — BAB/low-X anomaly
- ⚠️ `oil_beta_signal` is **NOT negated** in current code (alt_features.py:1403 returns raw β). Either flip the sign in the builder or accept the signal is conceptually opposite to dxy_beta. Audit 2026-04-17 flagged this as a real code/doc inconsistency.
- `xfin_signal`, `coskewness`, `coskew_acx` — C&Z convention
- `risk_reversal_25d_signal`, `iv_rank_signal`, `variance_premium_signal`, `etf_skew_relative_signal` — options: high crash-vol → mean-reversion signal
- FINRA short interest signals: higher SI = bearish (ranked ↑ predicts ↓ returns in LGBM rank training)

---

## 65. Feature-count sanity check

- `features.py` raw: ~42 (12 momentum, 6 mean-rev, 7 vol, 12 tail/semi-β, 3 liquidity, 5 volume)
- `features.py` ranked composites: +24 (`rank_*`)
- `features.py` breadth + wavelet: +10
- `model.py` size features: +3 (`log_mcap_z`, `log_liquidity_z`, `log_price_z`)
- `alt_features.py` fundamentals: +12 (+12 `_sni`)
- `alt_features.py` tier3 academic: +7
- `alt_features.py` earnings: +5
- `alt_features.py` distress: +3 (+3 `_sni`)
- `alt_features.py` macro betas: +17
- `alt_features.py` insider/analyst/institutional: +10
- `alt_features.py` FINRA short interest: +3
- `cross_asset_leadlag.py`: +15 (ZERO importance, drop)
- `cz_signals.py`: +10 (gated)
- `factor_momentum_features.py`: +7 (gated)
- `lazy_prices_features.py`: +3 (gated OFF)
- `lm_sentiment.py`: +7 (gated OFF)
- `sraf_sentiment.py`: +5 (gated OFF, broken)
- `options_signals.py`: +15 (gated OFF)
- Transformation variants: `_sn` (+15), `_sni` (+13 core + 12 fundamentals + 3 distress + few macro), `z_` (+4 early), `ix_` (+11), `_csz` (applied to ~half)

**Before deny-list**: ~350 candidate features when all gates ON.
**After deny-list (93)**: ~257 features enter `build_feature_matrix`.
**After coverage gate (>10%)**: varies, typically ~200-240 in model.

---

*End of CONTEXT.md. Part III (feature catalog) added 2026-04-17. Together Parts I-III provide: lay of the land (I), improvement levers (II), complete predictor lineage (III). When adding a new feature, append to the relevant §42-§61 table and cross-reference in §62 loader map. When removing, update §22 (importance) and §37 (backlog).*

---

## 40. Signal Research Library - C&Z + Orats Academic Reference

> Consolidated 2026-04-17 from 20 parallel research agents. Each subsection cites the
> originating academic paper, the formula, the sign convention, our implementation file/lines,
> and any data-source notes (cached vs live). Use this when designing new signals or auditing
> existing ones - do NOT reinvent. Source markdown lives at `docs/research_chunks/*.md`.


### 40.1 C&Z (Open Source Asset Pricing) - 209-Signal Replication Framework


#### C&Z (Open Source Asset Pricing) - Project Overview

##### The Chen–Zimmermann Open Source Asset Pricing Project

The Open Source Asset Pricing (OSAP) project, maintained by **Andrew Y. Chen** (Federal Reserve Board) and **Tom Zimmermann** (University of Cologne), is the most comprehensive public replication of the cross-sectional stock-return predictability literature. The project's working paper, Chen and Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing," appeared in the *Critical Finance Review* 11(2), 207–264, with code and data hosted at [openassetpricing.com](https://www.openassetpricing.com/) and on [GitHub](https://github.com/OpenSourceAP/CrossSection).

##### Scope and Outputs

The October 2025 release ships **212 long–short predictor portfolios and 209 firm-level signals** in signed form, where each signal is oriented so that higher values forecast higher subsequent returns. Two artifacts anchor the dataset:

- **`SignalDoc.csv`** — metadata for every signal, including the originating publication, sample period, signal category (accounting, trading, event, analyst, etc.), construction notes, and the predicted sign.
- **`signed_predictors_dl_wide.csv`** — a firm-month wide panel (≈1.6 GB zipped) of all 209 signed characteristics, currently extending through December 2024 (option-implied predictors end December 2022).

Daily and monthly long–short return files, plus Python (`pip install openassetpricing`) and R packages, accompany the raw signal panel.

##### Replication Methodology

For each predictor, Chen and Zimmermann re-implement the original paper's construction directly from CRSP and Compustat (with auxiliary IBES, OptionMetrics, and 13F inputs where applicable), then sign the variable so the long leg is the predicted-high-return tail. Reproduction quality is strong: among the 161 characteristics deemed "clearly significant" in the original studies, 98% of OSAP long–short portfolios produce |t| > 1.96, and a regression of reproduced on original t-statistics yields a slope near 0.90 with R² ≈ 0.83.

##### Relevance and Caveats

For an ML-driven equity strategy, OSAP supplies a vetted, citation-grounded universe of candidate signals — eliminating the engineering cost of re-deriving each predictor from primary sources and providing a defensible benchmark for any in-house factor. The dataset must, however, be used with the documented post-publication-decay literature in mind: McLean and Pontiff (2016, *Journal of Finance* 71(1), 5–32) report that anomaly returns fall ~26% out-of-sample and ~58% post-publication, attributable to a mixture of statistical bias (data mining) and arbitrage by informed traders. Compounding this, the breadth of the OSAP zoo raises acute multiple-testing and false-discovery concerns (Harvey, Liu, and Zhu, 2016), so signals should be evaluated under stringent OOS protocols, deflated t-thresholds, or hierarchical/Bayesian shrinkage rather than taken at their published face value.


#### Coskewness (Harvey-Siddique 2000)

##### Origin and Theoretical Motivation

Coskewness extends the mean-variance CAPM into a third-moment framework. The seminal treatment is (Harvey & Siddique 2000, Journal of Finance), "Conditional Skewness in Asset Pricing Tests," vol. 55(3), pp. 1263-1295. Their thesis: if the cross-section of returns exhibits systematic skewness, risk-averse investors with non-quadratic utility will demand compensation for holding assets that exacerbate left-tail co-movement with the market. They document a coskewness risk premium of roughly 3.6 percent per year and show that part of the momentum anomaly loads on systematic skewness.

The standardized coskewness measure is

$$\text{coskew}_i = \frac{\mathbb{E}[\varepsilon_i \cdot \varepsilon_m^2]}{\sqrt{\mathbb{E}[\varepsilon_i^2]} \cdot \mathbb{E}[\varepsilon_m^2]}$$

where $\varepsilon_i$ and $\varepsilon_m$ are demeaned stock and market returns. Names with **negative** coskewness pay off poorly precisely when the market is volatile (large $\varepsilon_m^2$), so they should command **higher** expected returns. This yields the Chen-Zimmermann CSK and CSR signals (both negate the raw measure so that "high signal = predicted high return").

(Ang, Chen & Xing 2006, Review of Financial Studies), "Downside Risk," vol. 19(4), pp. 1191-1239, refines this by isolating downside co-movement explicitly, reporting a downside-risk premium near 6 percent per annum that is not subsumed by size, value, momentum, liquidity, or unconditional coskewness. We implement their daily-window variant as `coskew_acx`.

##### Empirical Performance in Our Universe

Per `results/cz_signal_ic.csv`, **Coskewness IC_IR = 0.198** and **CoskewACX IC_IR = 0.124** in our 2013-2026 backtest. Both signals clear the Chen-Zimmermann replication threshold and rank among the strongest price-only predictors we have surfaced.

##### Implementations

Two non-redundant implementations live in the codebase:

- `src/features.py:co_skewness()` (lines 990-1078) — daily 252-day rolling window using a cumsum-optimized rolling mean over $\varepsilon_i \cdot \varepsilon_m^2$, $\varepsilon_i^2$, and $\varepsilon_m^2$. Loaded into the main feature pipeline; gated by `DISABLE_CO_SKEW_252` env var to avoid double-counting.
- `src/cz_signals.py:build_coskewness_signal()` (lines 48-107) — monthly 60-month window matching the original Harvey-Siddique horizon, activated under `--use-cz-signals`. Forward-filled to daily and cross-sectionally ranked to $[0, 1]$.

The two specifications are **statistically orthogonal** (Spearman rho = 0.041): the daily 252-day version captures conditional tail co-movement at business-cycle frequency, while the monthly 60-month version captures the slow-moving unconditional moment. Stacking both adds incremental information.

##### Strategic Note

Coskewness is **price-only**: inputs are stock returns and a market proxy (SPY). It does **not** depend on options-implied moments, EDGAR filings, or any paid alt-data feed. Cancelling the Orats subscription (or any options vendor) preserves both CSK and CSR signals in full. This makes coskewness a high-conviction, zero-marginal-cost component of the alpha stack.

##### Sources

- [Conditional Skewness in Asset Pricing Tests (Harvey & Siddique 2000) - Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00247)
- [Conditional Skewness in Asset Pricing Tests - Duke working paper PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P56_Conditional_skewness_in.pdf)
- [Downside Risk (Ang, Chen & Xing 2006) - Oxford Academic](https://academic.oup.com/rfs/article-abstract/19/4/1191/1580531)
- [Downside Risk - NBER working paper w11824](https://www.nber.org/papers/w11824)


#### XFIN - External Financing (Bradshaw-Richardson-Sloan 2006)

##### Origin

The composite external financing measure was introduced by Bradshaw, Richardson & Sloan (2006), "The Relation Between Corporate Financing Activities, Analysts' Forecasts and Stock Returns," *Journal of Accounting and Economics* 42(1-2), 53-85. The authors aggregate equity and debt issuance/retirement into a single net-financing variable and document that its statistical association with future returns is stronger than that of any individual financing channel studied previously. Pontiff & Woodgate (2008), "Share Issuance and Cross-Sectional Returns," *Journal of Finance* 63(2), 921-945, corroborate the equity-only leg, showing that share issuance subsumes size, book-to-market, and momentum in post-1970 Fama-MacBeth regressions.

##### Construction

XFIN is computed as a flow-to-stock ratio:

```
XFIN_t = (sstk + dltis - dvc - prstkc - dltr) / total_assets_t
```

where `sstk` is equity issuance, `dltis` long-term debt issuance, `dvc` cash dividends, `prstkc` equity repurchases, and `dltr` long-term debt repayment (Compustat tags). The numerator captures net cash raised externally over the trailing fiscal period; scaling by total assets yields a unit-free intensity measure comparable across firm size.

##### Sign and Economic Interpretation

The empirical sign is **negative** — high XFIN predicts low subsequent returns. The dominant explanation is market timing: managers issue equity and debt when their securities are overvalued and retire them when undervalued (Baker-Wurgler-style timing combined with Stein 1996 catering). Bradshaw et al. (2006) show analyst over-optimism rises with XFIN, supporting a misvaluation rather than risk-based interpretation.

##### Implementation and IC

Built in `src/cz_signals.py:build_xfin_signal()` (lines 282-309). The function pulls the five raw EDGAR fields, fills missing components with zero, ratios to total assets with a `(-2, 2)` winsorization, then negates the cross-sectional rank so that long-leg exposure is consistent with the documented sign. A 45-day filing lag is enforced upstream in the EDGAR loader for point-in-time safety. Per `cz_signal_ic.csv`, the signal achieves IC_IR = 0.181 with 40% panel coverage — among the strongest of the C&Z fundamental novelties.

##### Relationship to Sister Signals

NetEquityFinance, defined as (issuance - repurchase) / market cap, is the equity-only sub-component normalized by market value rather than book assets; it is mathematically the negation of NetPayoutYield. We obtain that exposure through `build_net_payout_yield_signal()` and therefore omit a standalone NetEquityFinance build to avoid collinear factor loadings.

Sources:
- [Bradshaw, Richardson & Sloan (2006) — IDEAS/RePEc](https://ideas.repec.org/a/eee/jaecon/v42y2006i1-2p53-85.html)
- [Bradshaw, Richardson & Sloan (2006) — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=904226)
- [Pontiff & Woodgate (2008) — Journal of Finance / Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01335.x)
- [Pontiff & Woodgate (2008) — author PDF](https://www2.bc.edu/jeffrey-pontiff/Documents/11_pontiff-woodgate.pdf)


#### PayoutYield + NetPayoutYield (Boudoukh-Michaely-Richardson-Roberts 2007)

##### Origin

Boudoukh, Michaely, Richardson, and Roberts (2007), "On the Importance of Measuring Payout Yield: Implications for Empirical Asset Pricing," *Journal of Finance* 62(2), 877-915. The paper challenges the traditional dividend-yield literature (Fama and French 1988; Campbell and Shiller 1988) by demonstrating that the secular decline in dividend payouts during the 1980s-1990s reflects not a weakening of the cash-flow-to-price relationship but a *substitution* of share repurchases for dividends. Once buybacks (and net issuance) are incorporated, the time-series predictability of aggregate returns is restored, and the cross-sectional spread is materially stronger than that of dividend yield alone.

##### Key Insight

Dividend yield understates total cash returned to shareholders. Modern firms increasingly prefer repurchases for tax efficiency (capital-gains rate < ordinary dividend rate, particularly pre-2003 JGTRRA), payout-flexibility (no implicit "dividend smoothing" commitment), and management compensation (option-overhang absorption). Measuring only dividends therefore truncates the true payout signal and produces deteriorating predictive power post-1980. The fix is conceptually trivial but empirically decisive: replace D/P with (D+R)/P, and optionally subtract issuance to capture *net* shareholder cash flow.

##### Formulas

- **PayoutYield** = (dividends + share repurchases) / market_cap
- **NetPayoutYield** = (dividends + repurchases - issuance) / market_cap

The net version subtracts seasoned-equity issuance and option-related share creation, capturing genuine dilution-adjusted shareholder yield.

##### Sign and IC

Sign is **positive** (classic value channel: high payout yield -> higher forward returns). Per `cz_signal_ic.csv`, **PayoutYield IC_IR = 0.146** and **NetPayoutYield IC_IR = 0.176**. The net version's superior IR confirms Boudoukh et al.'s thesis: dilution materially erodes the gross-payout signal, and adjusting for it produces a cleaner shareholder-yield measure.

##### Implementation

- `src/cz_signals.py:build_payout_yield_signal()` lines 238-256
- `src/cz_signals.py:build_net_payout_yield_signal()` lines 259-279

Both signals use a 6-month lag (`market_cap.shift(126)`) on the denominator to avoid look-ahead via simultaneous price updates, clip raw ratios to `[-1, 5]`, and emit cross-sectional ranks via `_cs_rank()`. Both are gated behind `--use-cz-signals` (default OFF).

##### Data Dependencies

`edgar_extra` fields: `raw_dividends_paid`, `raw_buybacks`, `raw_stock_issuance` (sourced from EDGAR XBRL cash-flow statements with standard filing-lag protection in `_raw_panel_from_edgar`). `market_cap` panel from the standard pricing/shares-outstanding pipeline.

##### Relationship to Existing `dividend_yield`

The legacy `dividend_yield` feature (in `src/api_data.py`) is a strict **subset** of PayoutYield: it captures only the dividend leg and ignores the now-dominant repurchase channel. When `--use-cz-signals` is enabled, set `DISABLE_DIVIDEND_YIELD=1` to suppress the legacy feature and avoid signal duplication / multicollinearity in the LightGBM stack. NetPayoutYield further dominates by netting issuance, making the legacy feature strictly redundant under the C&Z stack.

##### Theory

Total return to shareholders - not the accounting label attached to it - drives valuation. Substitution between dividends and buybacks is a tax-and-flexibility decision, not a cash-flow signal; conditioning on the gross channel reintroduces the predictability that vanished from D/P after 1980.


#### Earnings Announcement Return / EAR (Brandt et al. 2008)

##### Earnings Announcement Return (EAR / AnnouncementReturn)

##### Origin and Theoretical Motivation

The Earnings Announcement Return signal traces to Brandt, Kishore, Santa-Clara, and Venkatachalam (2008), *"Earnings Announcements are Full of Surprises"* (SSRN 909563). The authors argue that the price reaction in a tight window around the earnings release is a sufficient statistic for the *total* information content of the announcement — not only the headline EPS surprise versus consensus (SUE), but also unexpected information about sales, margins, guidance, capex, and the qualitative tone of the call. They show that a long-short portfolio sorted on EAR earns a 7.55% annual abnormal return, exceeding a SUE-sorted strategy by 1.3 ppt and remaining largely orthogonal to it. The signal is intellectually adjacent to Frazzini and Lamont (2007) *"The Earnings Announcement Premium and Trading Volume"*, who document a related event-window premium.

##### Construction

EAR is the cumulative return on a 3-day (sometimes 4-day) window centered on the announcement date `t`:

- Pre-market / before-open releases: `[t-1, t+1]`
- After-market-close (AMC) releases: `[t, t+2]` — because the price impact lands in the next trading session

The sign is **positive**: large positive announcement returns predict continued outperformance over the following 30–60 trading days. The economic content is a generalised post-earnings-announcement-drift (PEAD) — markets fail to fully impound event-window information, leaving a slow diffusion that arbitrageurs can harvest.

##### Empirical Performance in Our Stack

Per `cz_signal_ic.csv`, `AnnouncementReturn` produces an **IC information ratio of 0.254**, our 3rd-highest novel signal in the Chen-Zimmermann research sweep — strong enough to justify inclusion despite EAR's relatively narrow universe (event-conditional).

##### Implementation Notes

- Code: `src/alt_features.py:earnings_ann_return_signal` (lines 608–656).
- Window logic (Phase D fix, April 2026): the `before_after_market` field returned by EODHD selects the window — `"amc"` triggers `[t, t+2]`, otherwise the symmetric `[-1, +1]` window applies.
- **Latent bug discovered 2026-04-17**: the EODHD response field is snake_case `before_after_market`, but earlier parser code in `src/alt_data_loader.py:load_earnings_calendar_eodhd` had silently defaulted every record to `"unknown"`, collapsing the AMC branch and degrading the signal to the symmetric default. Fixed at lines 1005–1013, with an automatic cache-migration probe (lines 895–927) that detects pre-fix caches and forces one-time re-fetch.
- Data dependency: EODHD bulk earnings calendar (`/api/calendar/earnings`), 281 tickers cached, 2013 → 2026 (`load_earnings_calendar_eodhd`).
- Carry: signal value is held forward for `surprise_carry_days` (currently 5–10 trading days, tuned in Phase D) — matching the empirical PEAD half-life from Bernard-Thomas (1989) and the Brandt et al. drift window.

##### Sources

- [Earnings Announcements are Full of Surprises — SSRN 909563](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=909563)
- [Quantpedia mirror PDF](https://quantpedia.com/www/Earnings_Announcements_are_Full_of_Surprises.pdf)
- [Semantic Scholar entry](https://www.semanticscholar.org/paper/Earnings-Announcements-are-Full-of-Surprises-Kishore-Brandt/3cb316e8f28bc359cbb92fe985c9011a2998198f)


#### Earnings Streak + Surprise Consistency

##### Theme and Theoretical Motivation

Repeated positive earnings surprises predict subsequent abnormal returns through two reinforcing channels. First, the post-earnings-announcement drift (PEAD) literature, founded by (Foster, Olsen & Shevlin 1984, The Accounting Review), "Earnings Releases, Anomalies, and the Behavior of Security Returns," vol. 59(4), pp. 574-603, documents that prices underreact to earnings news and continue to drift in the surprise's direction for roughly sixty trading days. When surprises occur in succession, the underreaction compounds: each beat adds fresh drift before the prior beat's drift has been fully absorbed.

Second, persistent beats re-anchor analyst expectations. (Bartov, Givoly & Hayn 2002, Journal of Accounting and Economics), "The Rewards to Meeting or Beating Earnings Expectations," vol. 33(2), pp. 173-204, show that firms which consistently meet or beat consensus enjoy a premium *over and above* the magnitude of the surprise itself, consistent with managerial signaling and reduced uncertainty about the firm's earnings process. (DeFond & Park 2001, The Accounting Review), "The Reversal of Abnormal Accruals and the Market Valuation of Earnings Surprises," vol. 76(3), pp. 375-404, further document that the market only partially distinguishes accrual-driven from cash-driven surprises, leaving a tradable mispricing in firms whose surprise stream is "clean."

##### Variants

Two complementary daily signals are constructed from quarterly EPS surprises, defined as $(\text{eps}_{\text{actual}} - \text{eps}_{\text{est}}) / |\text{eps}_{\text{est}}|$ and clipped to $[-2, 2]$:

- **Earnings Streak** — count of consecutive quarters with positive surprise, capped at 8. Resets to zero on any miss. Sign: **positive**.
- **Surprise Consistency** — reciprocal of the rolling 8-quarter coefficient of variation of surprises ($1 / (\sigma / |\mu|)$). Low CV implies a predictable beat process. Sign: **positive** (we negate CV before ranking).

##### Empirical Performance

Per `results/cz_signal_ic.csv`, **EarningsStreak IC_IR = 0.245**, our #4-ranked novel Chen-Zimmermann signal in the 2013-2026 backtest. The consistency variant is correlated but contributes incremental information at the tails (very low CV firms behave more like a quality-of-earnings proxy).

##### Implementations

- `src/alt_features.py:earnings_beat_streak_signal` (lines 590+) — iterates per ticker, builds the capped streak series at each event date, then forward-fills to daily.
- `src/alt_features.py:earnings_surprise_consistency_signal` (lines 620+) — rolling 8-quarter CV with $|\mu| > 10^{-9}$ guard, also forward-filled.

Both are Phase D additions, gated by the `ENABLE_PHASE_D_SIGNALS=1` environment variable (default ON). Earnings data resolves from yfinance by default or from the EODHD bulk earnings endpoint when `--use-eodhd-earnings` is passed. The signal value carries forward roughly 5-15 trading days post-event before the next quarterly print refreshes it, which aligns with the bulk of the PEAD window documented in Foster-Olsen-Shevlin.

##### Sources

- [Earnings Releases, Anomalies, and the Behavior of Security Returns (Foster, Olsen & Shevlin 1984) - JSTOR](https://www.jstor.org/stable/247321)
- [The Reversal of Abnormal Accruals and the Market Valuation of Earnings Surprises (DeFond & Park 2001) - JSTOR](https://www.jstor.org/stable/3068959)
- [The Rewards to Meeting or Beating Earnings Expectations (Bartov, Givoly & Hayn 2002) - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165410102000451)


#### Cash-Based Operating Profitability (Ball et al. 2016)

##### Cash-Based Operating Profitability (CBOperProf) and OperProfRD

##### Origin

Ball, Gerakos, Linnainmaa & Nikolaev (2016), "Accruals, Cash Flows, and Operating Profitability in the Cross Section of Stock Returns," *Journal of Financial Economics* 121(1), 28-45. The authors decompose accounting earnings into a cash component and an accruals component, then ask which piece carries the cross-sectional return signal. Their headline result: a cash-based operating profitability measure that strips accruals out of operating profit *subsumes* both the standard profitability factor (Novy-Marx 2013) and the accruals anomaly (Sloan 1996). Adding a single cash-based profitability factor to the opportunity set raises the maximum attainable Sharpe ratio more than adding the accruals and accrual-laden profitability factors jointly.

##### Core insight

Accruals are management discretion. Revenue recognized before cash is collected, expenses deferred via capitalization, and working-capital movements all inject noise (and outright manipulation) into reported operating profit. Removing the accrual component yields a *cleaner* measure of recurring economic profitability, which in turn predicts returns more reliably because (i) cash earnings are harder to manage, and (ii) cash earnings persist longer than accrual earnings.

##### Formulas

- Standard operating profitability: `OperProf = (Revenue - COGS - SGA) / Total Assets`
- Cash-based variant: `CBOperProf = (Revenue - COGS - SGA - DeltaWC) / Total Assets`, where `DeltaWC` is the year-over-year change in non-cash working capital (the accrual wedge).
- R&D-adjusted variant: `OperProfRD = (Revenue - COGS - (SGA - R&D)) / Total Assets`, recapitalizing R&D as investment rather than period expense (Ball et al. discuss this robustness check; it also appears in Chen-Zimmermann's signal library).

All three carry a **positive** sign: high profitability predicts high subsequent returns.

##### Empirical strength in our pipeline

Per `cz_signal_ic.csv`, CBOperProf delivers IC_IR = 0.200 - one of the strongest single-name fundamentals we measure - while OperProfRD delivers IC_IR = 0.125. The gap confirms the BGLN result that the working-capital adjustment matters more than the R&D adjustment for return prediction.

##### Implementation

- `src/alt_features.py:cash_based_op_prof_signal` constructs CBOperProf and is enabled by default under `use_tier3_academic`.
- `src/cz_signals.py:build_operprof_rd_signal()` constructs OperProfRD and only activates under `--use-cz-signals`.
- The two are *related but non-redundant*: CBOperProf adjusts for `DeltaWC`, OperProfRD adjusts for R&D capitalization. Running both simultaneously can dilute the cleaner CBOperProf signal, so the env var `DISABLE_CASH_BASED_OP_PROF` lets the alt-features version step aside when the CZ OperProfRD path is active.

Sources:
- [Ball, Gerakos, Linnainmaa & Nikolaev (2016) - Tuck PDF](https://faculty.tuck.dartmouth.edu/images/uploads/faculty/joseph-gerakos/Ball,_Gerakos,_Linnainmaa,_et_al._2016.pdf)
- [SSRN listing for the paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2587199)
- [ScienceDirect - JFE 121(1) 28-45](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300307)
- [Alpha Architect summary](https://alphaarchitect.com/2016/01/14/value-investing-accruals-cash-flows-and-operating-profitability/)


#### Momentum Seasonality (Heston-Sadka 2008)

##### Momentum Seasonality (MomSeason)

##### Origin

The momentum seasonality signal originates with Heston and Sadka (2008), "Seasonality in the Cross-Section of Stock Returns," *Journal of Financial Economics* 87(2), 418–445. Their core empirical finding is striking: stocks that historically delivered high (or low) returns in a given calendar month tend to repeat that pattern in the same calendar month of subsequent years. The autocorrelation pattern is statistically significant at annual lags of 12, 24, 36, ..., and persists out to roughly 20 annual lags, superimposed on the well-known short-horizon momentum and long-horizon reversal patterns. Keloharju, Linnainmaa, and Nyberg (2016), "Return Seasonalities," *Journal of Finance*, generalize the result to non-calendar-month seasonalities (day-of-week, intra-month) and argue the effect reflects persistent firm-level expected-return components rather than risk-factor exposure.

##### Economic Insight

If Apple has outperformed in April for the past several Aprils, the signal predicts April outperformance again — independent of size, industry, earnings, dividends, or fiscal year (Heston-Sadka show all of these as non-explanations). The mechanism is debated; candidates include seasonal liquidity, tax-driven trading, and seasonal cash-flow news. The effect is subtle because it operates per-month rather than per-day, which is likely why it remains less arbitraged than 12-1 momentum.

##### Construction

Following the Heston-Sadka specification, we use the **years 6–10 lookback** window, deliberately skipping years 1–5 to orthogonalize the seasonal signal from conventional momentum and short-horizon reversal effects. For each (date *t*, ticker *i*):

$$\text{MomSeason}_{i,t} = \frac{1}{5} \sum_{k=6}^{10} r_{i, t - 12k \text{ months}}$$

where $r$ is the calendar-month return. The expected sign is **positive**: higher historical same-month return predicts higher current same-month return. We then apply a cross-sectional rank to [0, 1].

##### Implementation

`src/cz_signals.py:build_mom_season_signal()` (lines 159–205) computes monthly compounded returns via `resample("ME").prod()`, then for each month-end date averages returns from years t−6 through t−10 in the same calendar month, requires at least 2 historical observations, and forward-fills to daily frequency (limit=25 days).

##### Performance

Per `cz_signal_ic.csv`, MomSeason achieves **IC_IR = 0.146**, ranking among the strongest novel signals in our research panel. The data dependency is minimal: only the daily returns panel is required — no fundamentals, no options, no alternative data. This makes it a pure price-based signal with zero incremental data cost.

##### Sources

- [Seasonality in the cross-section of stock returns (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X0700195X)
- [Heston & Sadka working paper (NYU Stern)](https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/063f-sadka.pdf)
- [SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=687022)
- [RePEc entry](https://ideas.repec.org/a/eee/jfinec/v87y2008i2p418-445.html)


#### CFP, Tax, DelDRC - Minor Accounting Signals

This chunk documents three minor accounting signals from the Chen & Zimmermann
open-source replication universe that are integrated into our pipeline behind
the `--use-cz-signals` flag. Each signal is individually modest
(IC_IR ~0.11-0.14 in our 2013-2026 EDGAR-derived panel), but together they
add diversification across orthogonal accounting dimensions: cash-based
valuation, tax-based earnings quality, and revenue-recognition timing.

##### CFP (Cash Flow to Price)

**Origin.** Lakonishok, Shleifer, and Vishny (1994), "Contrarian Investment,
Extrapolation, and Risk," *Journal of Finance* 49(5), 1541-1578. LSV
demonstrate that simple value strategies built on cash-based fundamentals
outperform glamour stocks, attributing the premium to behavioral
extrapolation rather than risk compensation.

**Formula.** `CFP_i,t = operating_cash_flow_i,t / market_cap_i,t`. Operating
cash flow is preferred to earnings because it is harder to manipulate via
accruals.

**Sign.** Positive — high CFP firms are undervalued relative to their cash
generation and earn higher subsequent returns.

**Implementation.** `src/cz_signals.py::build_cfp_signal()` (lines 312-323),
clipped to [-5, 5] before cross-sectional ranking. **IC_IR: 0.111**.

##### Tax (Tax Expense Surprise)

**Origin.** Lev and Nissim (2004), "Taxable Income, Future Earnings, and
Equity Values," *The Accounting Review* 79(4), 1039-1074. Lev and Nissim
show that the tax-to-book income ratio predicts five-year earnings growth
and subsequent stock returns, capturing information about earnings quality
that book income alone misses.

**Formula.** `Tax_i,t = tax_expense_i,t / net_income_i,t` (an effective tax
rate proxy). High book-tax conformity implies less aggressive accrual
management and more durable earnings.

**Sign.** Positive — firms reporting tax expense closer to their book
income exhibit higher earnings persistence and outperform firms with low
effective tax rates that signal earnings inflation.

**Implementation.** `src/cz_signals.py::build_tax_signal()` (lines 350-366),
normalized against the 21% post-TCJA federal statutory rate and clipped to
[-2, 5]. **IC_IR: 0.123**.

##### DelDRC (Change in Deferred Revenue / Cash)

**Origin.** Prakash and Sinha (2013) and the broader deferred-revenue
literature, which identifies unearned revenue as a leading indicator of
future top-line growth.

**Formula.** `DelDRC_i,t = Δ(deferred_revenue_i,t) / avg(assets_t, assets_t-1)`,
computed as a year-over-year (252-day) change scaled by average assets.

**Sign.** Positive — rising deferred revenue represents cash already
collected but not yet recognized; it locks in future revenue and predicts
higher subsequent returns as the liability unwinds into the income
statement.

**Implementation.** `src/cz_signals.py::build_deldrc_signal()`
(lines 369-386), clipped to [-1, 1] before ranking. **IC_IR: 0.136**.

---

**Aggregate Role.** These three signals are each below the typical
single-factor inclusion threshold of IC_IR > 0.15, so they are deployed only
inside the C&Z accounting composite. Their value is diversification: CFP
spans cash valuation, Tax spans earnings-quality conformity, and DelDRC
spans revenue-timing accruals — three weakly correlated accounting
dimensions that each contribute marginal alpha when combined with the
stronger C&Z signals (Coskewness, XFIN, NetPayoutYield).

Sources:
- [Lakonishok, Shleifer, and Vishny (1994), Contrarian Investment, Extrapolation, and Risk - Journal of Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1994.tb04772.x)
- [Lakonishok, Shleifer, and Vishny (1994) - NBER Working Paper 4360](https://www.nber.org/papers/w4360)
- [Lev and Nissim (2004), Taxable Income, Future Earnings, and Equity Values - The Accounting Review](https://meridian.allenpress.com/accounting-review/article-abstract/79/4/1039/53457/Taxable-Income-Future-Earnings-and-Equity-Values)
- [Lev and Nissim (2004) - Columbia working paper PDF](http://www.columbia.edu/~dn75/taxableincome.pdf)


#### C&Z Implementation Roadmap (built / overlap / missing)

Reference: `results/_cz_research/cz_signal_ic.csv` (209 signals scored), `cz_overlap_map.csv` (28 overlaps), `src/cz_signals.py` (10 implementations). Universe: monthly IC over 143 months on our live US equity panel.

##### A. Currently Built — 10 signals in `src/cz_signals.py`

| Signal | IC_IR | One-line formula | Builder |
|---|---|---|---|
| `coskewness_signal` | 0.198 | `−E[r_i_dm · r_m_dm²] / (σ_i · σ_m²)` over 60 monthly obs | `build_coskewness_signal` |
| `coskew_acx_signal` | 0.124 | Same as above, daily 252d window | `build_coskew_acx_signal` |
| `mom_season_signal` | 0.146 | Avg same-calendar-month return, years t−6 … t−10 | `build_mom_season_signal` |
| `payout_yield_signal` | 0.146 | `(divs + buybacks) / mcap.shift(126)` | `build_payout_yield_signal` |
| `net_payout_yield_signal` | 0.176 | `(divs + buybacks − issuance) / mcap.shift(126)` | `build_net_payout_yield_signal` |
| `xfin_signal` | 0.182 | `−(sstk − dvc − prstkc + dltis − dltr) / assets` | `build_xfin_signal` |
| `cfp_signal` | 0.111 | `operating_cf / market_cap` | `build_cfp_signal` |
| `operprof_rd_signal` | 0.106 (OperProfRD=0.125) | `(rev − cogs − (sga − rd)) / assets` | `build_operprof_rd_signal` |
| `tax_signal` | 0.123 | `taxes_paid / (0.21 · net_income)` | `build_tax_signal` |
| `deldrc_signal` | 0.135 | `Δ(deferred_revenue, 252d) / avg(assets_t, t−1)` | `build_deldrc_signal` |

All return cs-ranked `[0,1]` (date × ticker), sign-oriented so higher = bullish.

##### B. Already Covered via Equivalents (top ~10 of 28 in `cz_overlap_map.csv`)

| C&Z | Our equivalent | C&Z IC_IR |
|---|---|---|
| OScore | `chs_distress_signal` | 0.286 |
| ShareIss5Y | `net_share_issuance_signal` | 0.250 |
| ShareIss1Y | `net_share_issuance_signal` | 0.237 |
| OperProf | `operating_profitability_signal` | 0.141 |
| NOA | `net_operating_assets_signal` | 0.119 |
| GP | `gross_profitability_signal` | 0.103 |
| EP | `earnings_yield_signal` | 0.103 |
| EarningsSurprise | `earnings_surprise_signal` | 0.067 |
| ForecastDispersion | `analyst_coverage_signal` | 0.065 |
| Leverage | `leverage_signal` | 0.078 |
| BM | `book_to_market_signal` | 0.036 |
| Illiquidity | `amihud_21d` | −0.105 (sign-aligned) |
| ShortInterest | `finra_si_level_signal`, `short_ratio_signal` | 0.030 |
| Mom6m / Mom12m | `mom_126d`, `mom_12_1` | 0.047 / −0.011 |

Do NOT rebuild these — verify our impls have correct sign and lag, then move on.

##### C. MISSING with IC_IR > 0.10 — Build Candidates (ranked)

| Signal | IC_IR | Source needed | Difficulty |
|---|---|---|---|
| `FirmAgeMom` | 0.200 | price-only (CRSP age proxy = days since first non-NaN price) × Mom12m interaction | trivial |
| `AnalystRevision` | 0.152 | Finnhub / EODHD analyst estimates Δ EPS | medium (paid API) |
| `ChangeInRecommendation` | 0.143 | Finnhub recommendation_trends, Δ mean rating | medium |
| `RDAbility` | 0.136 | EDGAR `raw_rd` 5y avg, residual on size/BM | medium |
| `RoE` | 0.132 | EDGAR `raw_net_income / raw_equity` (annual, distinct from `q_roe` quarterly) | trivial |
| `EarningsConsistency` | 0.127 | EDGAR `raw_eps`, std of trailing 8q earnings growth | easy |
| `UpRecomm` | 0.135 | Finnhub: count of upgrades last 30d | medium |
| `std_turn` | 0.123 | price+volume: 36-month std of monthly turnover | trivial |
| `CitationsRD` | 0.119 | NBER patent citations per R&D dollar — external dataset | hard |
| `DownRecomm` | 0.114 | Finnhub: downgrades last 30d | medium |
| `sfe` | 0.113 | Finnhub: scaled forecast EPS (FEPS / price) | medium |

Suggested build order: `FirmAgeMom` → `RoE` (annual) → `std_turn` → `EarningsConsistency` (all price/EDGAR, no new API). Then batch the four Finnhub recommendation/forecast signals (`AnalystRevision`, `ChangeInRecommendation`, `UpRecomm`, `DownRecomm`, `sfe`) in a single PR. `CitationsRD` requires NBER patent dataset — defer.

##### D. Test Discipline

- **Cache-key fix in `src/model.py`**: `EXTRA_DENY_FEATURES` is now folded into the panel-hash, so per-signal isolation A/B tests no longer hit a stale cache. Earlier sweeps were silently re-using identical panels.
- **CLI flags**: `--cz-only=<csv>` (allowlist) and `--cz-exclude=<csv>` (denylist) on `run_strategy.py` for proper per-signal A/B. Both flags participate in the cache hash.
- **Test harness**: `run_signal_test_suite.py` runs each candidate signal in isolation against the locked baseline, asserts cache-hash uniqueness across runs, and emits `results/_signal_test/<signal>__metrics.csv`.

##### E. Known Gotchas

- **Sibling-feature leakage**: deny-lists must include sibling families. The "net_payout_only" iso test leaked `dividend_yield` because it shares the EDGAR `raw_dividends_paid` panel — exclude entire family, not just the named feature.
- **McLean-Pontiff (2016) decay**: published anomalies lose ~35% of in-sample IC out-of-sample. C&Z IC_IR figures are in-sample on their universe; expect ~0.65× on our 143-month panel.
- **Multiple-testing**: 209 signals × 7d-fwd → naive p<0.05 yields ~10 false positives. Require **IC_IR > 0.15** on our panel before promoting to production; signals in the 0.10–0.15 band ship only if they survive a 5-fold time-series CV with stable sign.


### 40.2 Orats Options Data + Tradier Migration


#### Orats - Company, Delayed Data API, SMV Methodology

##### Orats: Company and Data Stack

Option Research and Technology Services (Orats, [orats.com](https://orats.com)) is a Chicago-based options analytics vendor founded in 2014 by Matt Amberson, a former Cboe market maker who previously hired statistically-trained traders to operate on the floor and built proprietary research tooling to support that desk. Amberson (Kellogg MBA, CPA, CFA Level I) remains principal and drives the firm's product and quantitative roadmap. The firm productized its internal market-making analytics into three commercial offerings: a **Delayed Data API** ($99/month, 15-minute lag, full historical and live coverage), a **Live Data API** ($399/month plus per-call pricing, real-time), and **Datashop**, a one-time bulk download channel used for research-grade historical pulls.

##### Endpoints Consumed

This project ingests six endpoints from the `datav2/hist` namespace. `/cores` returns a 340-column daily snapshot per ticker spanning the full IV surface, term structure, skew, and earnings-adjusted variants. `/summaries` returns a denser 128-column slice that includes proprietary derived fields such as `impliedMove`. `/ivrank` provides the 1-month and 1-year IV percentile rank in 8 columns. `/dailies` exposes daily OHLCV (substitutable from any equity vendor). `/hvs` returns the historical realized volatility surface across multiple lookback horizons, including ex-earnings variants critical for de-noising event windows. `/earnings` provides an options-aware earnings calendar; the `anncTod` field encodes announcement time as HHMM, enabling correct point-in-time alignment of pre- versus post-announcement implied volatility.

##### SMV (Smoothed Market Value) Methodology

The analytical core of every Orats endpoint is the **Smoothed Market Value** engine, an arbitrage-free IV surface fitter. SMV cleans raw exchange quotes, solves for a residual yield using put-call parity and dividend assumptions, feeds the cleaned inputs to a modified binomial pricing engine, and fits a smooth, non-arbitrageable curve through strike implied volatilities. This filters wide bid-ask "joke quotes," arbitrage-violating prices, end-of-day artifacts, and the noisy wings of low-premium out-of-the-money strikes; for illiquid contracts, SMV blends in historical priors when current confidence is low. The published `smv_vol` field, together with SMV-derived Greeks (delta, gamma, theta, vega, rho), produces theoretical values inside the bid-ask over 99% of the time.

A practical consequence is **train/serve consistency**: the same SMV engine generates this project's cached historical parquets and is also embedded inside Tradier's free options chain endpoint (Tradier explicitly credits Orats for Greeks and IV). Live polls and historical training data therefore share an identical numerical generator, eliminating the surface-fit skew that typically arises when historical and live vendors differ.

##### Project Coverage

The local cache spans 1,414 tickers from 2013-01-01 through 2026-04-15, materialized as 8,061 parquet files totaling roughly 11 GB. This depth supports point-in-time IV-surface, skew, term-structure, and earnings-move features without survivorship bias.


#### Orats /cores Endpoint - 340-Column Schema

The Orats `hist/cores` endpoint returns a single end-of-day record per `(ticker, tradeDate)` containing 340 derived option-surface fields. This is the densest of the Orats historical endpoints and is the primary feed for our options-signal stack. Field names below are verified against the live `/datav2/hist/cores` payload (snapshot 2026-04-15) and against the parquet sample `data/cache/options/orats_raw/cores_AAPL_2013-01-01_2026-04-15.parquet`. Inventory source: `results/validation/orats_cores_field_inventory.csv`.

##### A. Identifiers / metadata (~10)

| Field | Notes |
|---|---|
| `ticker`, `tradeDate`, `updatedAt` | primary key + freshness stamp |
| `assetType`, `sector`, `sectorName` | classification |
| `pxCls`, `priorCls`, `pxAtmIv`, `mktCap` | spot reference + size |

##### B. Constant-maturity ATM IV term structure (~10)

| Field | Maturity |
|---|---|
| `iv10d`, `iv20d`, `iv30d`, `iv60d`, `iv90d`, `iv6m`, `iv1yr` | 10d → 1y |
| `atmIvM1`, `atmIvM2`, `atmIvM3`, `atmIvM4` | first 4 listed monthly expiries |
| `atmFitIvM1..4`, `atmFcstIvM1..4`, `dtExM1..4` | fit/forecast variants + DTE |

##### C. Smile / delta-bucketed IV matrix (~80)

Cross of 5 delta strikes (`dlt5`, `dlt25`, `dlt50`, `dlt75`, `dlt95`) × 7 maturities (`Iv10d`, `Iv20d`, `Iv30d`, `Iv60d`, `Iv90d`, `Iv6m`, `Iv1y`). Convention: `dlt5` = 5-delta call (deep OTM upside), `dlt95` = 95-delta call ≡ 5-delta put (crash strike). Each cell is also republished in the earnings-stripped form (`exErnDltXIvY`), doubling the bucket count.

##### D. Slope / skew metrics (~10)

| Field | Meaning |
|---|---|
| `slope`, `slopeInf`, `slopeFcst`, `slopeFcstInf` | smile slope (current / asymptotic / forecast) |
| `deriv`, `derivInf`, `derivFcst`, `derivFcstInf` | smile curvature (2nd derivative) |
| `slopepctile`, `slopeavg1m`, `slopeavg1y`, `slopeStdv1y` | rank/normalisation context |

##### E. Volume / Open Interest (~12)

`cVolu`, `pVolu`, `cOi`, `pOi`, `oi`, `stkVolu`, `avgOptVolu20d`, `cAddPrem`, `pAddPrem`, plus implicit put/call ratios derived downstream.

##### F. Realized volatility (~30)

| Family | Windows |
|---|---|
| `orHv*` (open-to-close) | 1d, 5d, 10d, 20d, 60d, 90d, 120d, 252d, 500d, 1000d |
| `clsHv*` (close-to-close) | 5d → 1000d (same windows) |
| `orHvXern*`, `clsHvXern*` | earnings-stripped variants |

##### G. Forward IV curve / forecasts (~12)

`fwd30_20`, `fwd60_30`, `fwd90_60`, `fwd180_90`, `fwd90_30` plus `f`-prefixed forecast and `fb`-prefixed beta-adjusted versions. Quality fields: `fcstR2`, `fcstR2Imp`, `confidence`, `error`, `orFcst20d`, `orIvFcst20d`, `orFcstInf`.

##### H. Earnings effects (~30)

| Family | Fields |
|---|---|
| Calendar | `nextErn`, `lastErn`, `daysToNextErn`, `wksNextErn`, `ernMnth`, `nextErnTod`, `lastErnTod` |
| Historical 12-quarter array | `ernDate1..12`, `ernMv1..12`, `ernStraPct1..12`, `ernEffct1..12` |
| Implied move | `impErnMv`, `impErnMv90d`, `impMth2ErnMv`, `impliedEarningsMove`, `absAvgErnMv`, `ernMvStdv`, `fcstErnEffct`, `impliedIee`, `impliedEe`, `ivEarnReturn` |

##### I. Earnings-stripped IV (~12)

`exErnIv10d`, `exErnIv20d`, `exErnIv30d`, `exErnIv60d`, `exErnIv90d`, `exErnIv6m`, `exErnIv1yr`, plus `fexErn*` forward and `ffexErn*` filtered-forward variants — the cleanest cross-period vol comparison surface.

##### J. Dividend / borrow (~10)

`divFreq`, `divYield`, `divGrwth`, `divDate`, `divAmt`, `nextDiv`, `impliedNextDiv`, `annActDiv`, `annIdiv`, `borrow30`, `borrow2yr`. The `annIdiv` − `annActDiv` gap is a dividend-surprise proxy; `borrow30` flags hard-to-borrow short squeezes.

##### K. Sector / ETF reference (~10)

`bestEtf`, `etfIncl`, `correlSpy1m/1y`, `correlEtf1m/1y`, `beta1m`, `beta1y`, `ivSpyRatio` (+1m/1y avg + stdv), `ivEtfRatio` (+1m/1y avg + stdv), `etfSlopeRatio`, `etfIvHvXernRatio`.

##### L. Misc / aggregates (~30)

`px1kGam` (gamma per $1k notional), `volOfVol`, `volOfIvol`, `iv200Ma`, `contango`, `iRate5wk`, `iRateLt`, `mktWidthVol`, `rip`, `tkOver`, `hiHedge`/`loHedge`, `ivPctile1m/1y/Spy/Etf`, `ivStdvMean`/`Stdv1y`, `straPxM1/M2`, `smoothStraPxM1/M2`, `fcstStraPxM1/M2`, `loStrikeM1/M2`, `hiStrikeM1/M2`, `fairVol90d`, `fairXieeVol90d`, `fairMth2XieeVol90d`, `impliedR2`.

##### Our extraction: 27-field core slice

`src/orats_loader.py` pivots only the fields needed for the C&Z options signal stack. Selected (priority): ATM term structure (`iv30d`, `iv60d`, `iv90d`), the 30d delta smile (`dlt5Iv30d`, `dlt25Iv30d`, `dlt75Iv30d`, `dlt95Iv30d`), earnings-stripped 30d skew (`exErnIv30d`, `exErnDlt25Iv30d`, `exErnDlt75Iv30d`, `exErnDlt95Iv30d`), `slope`, call/put `cVolu`/`pVolu`/`cOi`/`pOi`, `borrow30`, `annIdiv`, `annActDiv`, plus realized-vol anchors (`orHv20d`, `orHv60d`), forward IV decomposition (`fwd30_20`, `fwd60_30`, `fwd90_30`), and earnings calendar (`daysToNextErn`, `impliedEarningsMove`). One derived column is built downstream: `cp_vol_spread_proxy = dlt25Iv30d − dlt75Iv30d` (call-skew proxy for `dCPVolSpread`).

That is ~8% of the 340-column surface; the other 92% (full smile-maturity matrix, 12-quarter earnings history, second-order Greeks, sector-relative ratios) remains parquet-resident and addressable for future signal research without re-pulling the API.


#### Other Orats Endpoints (/summaries, /ivrank, /dailies, /hvs, /earnings)

Reference for the five auxiliary Orats endpoints (`/summaries`, `/ivrank`, `/dailies`, `/hvs`, `/earnings`) and what each contributes beyond the primary `/cores` surface. Source facts: `results/validation/orats_other_endpoints_inventory.csv` (187 fields audited).

##### /summaries (128 cols)

Despite carrying fewer columns than `/cores`, `/summaries` exposes **18 fields not present in /cores**. The bulk are redundant (IVs, smile slices, forward variances all derivable from a chain), but a handful are genuinely unique and not reproducible from a live Tradier chain:

- **`impliedMove`** — single-number expected percentage move to next expiry, calibrated from the full vol surface. Not derivable cheaply.
- **`impliedEarningsMove`** — earnings-event-specific implied jump.
- **`exErnIv{10d,20d,30d,60d,90d,6m,1y}`** — earnings-stripped ATM IV per maturity. Critical for separating event vol from baseline vol on names with binary catalysts.
- **`rDrv30/2y`, `rSlp30/2y`, `rVol30/2y`** — Orats' proprietary residual surface diagnostics; not reconstructible.

**Recommendation:** keep cached snapshot for `impliedMove`, `impliedEarningsMove`, and the `exErnIv*` curve. Drop the rest as redundant with `/cores`.

##### /ivrank (8 cols)

Five unique fields: `iv`, `ivRank1m`, `ivPct1m`, `ivRank1y`, `ivPct1y`. These are 21-day and 252-day rolling IV percentile ranks. Mathematically derivable from a `/cores` `iv30d` history but expensive to compute daily across 3000 names.

**Recommendation:** retain historical cache; recompute fresh from Tradier `iv30d` history going forward (one-pass `pd.Series.rank(pct=True).rolling(252)`).

##### /dailies (13 cols)

Adjusted and unadjusted OHLCV plus stock volume. Every field is replaceable from any equities vendor — we already pay for EODHD daily bars with corporate-action adjustments.

**Recommendation:** **DROP**. Zero unique value.

##### /hvs (48 cols)

Realized volatility across **11 horizons (1d → 1000d) × 3 variants** (open-to-close `orHv*`, close-to-close `clsHv*`, ex-earnings `*Xern*`). Standard textbook HV — fully derivable from daily prices via `np.log(close).diff().rolling(N).std() * np.sqrt(252)`. Ex-earnings variants need only an earnings-date mask we already build.

**Recommendation:** **DROP**, compute in-house.

##### /earnings (4 cols)

Sparse (~150 rows per ticker over 13 years). The only field with marginal unique value is **`anncTod`** (announcement time as `HHMM`, e.g. `1630` for after-close). EODHD earnings calendar covers the same dates; before/after-market flag can be inferred from EODHD's `before_after_market` column.

**Recommendation:** **DROP**.

##### Consolidation Table

| Endpoint   | Cols | Unique Fields | Decision        | Replacement                        |
|------------|------|---------------|-----------------|------------------------------------|
| /summaries | 128  | 18            | Cache subset    | Keep `impliedMove`, `exErnIv*` only |
| /ivrank    | 8    | 5             | Cache + recompute | Tradier `iv30d` rolling rank       |
| /dailies   | 13   | 13 (trivial)  | **DROP**        | EODHD adjusted OHLCV               |
| /hvs       | 48   | 11 (trivial)  | **DROP**        | `np.log(close).diff().rolling(N)`  |
| /earnings  | 4    | 1             | **DROP**        | EODHD earnings calendar            |

**Net:** post-cancellation we operate on `/cores` (primary live surface) + a frozen `/summaries` cache for `impliedMove` and earnings-stripped IV curves. The other four endpoints contribute nothing that is both unique and worth the API spend.


#### impliedMove + Earnings IV Crush (Beber-Brandt 2010)

##### Definition

The **implied move** is the option market's expectation of the absolute return of the underlying over the life of a near-the-money straddle. For a stock at spot $S$ with at-the-money call and put mid-prices $C$ and $P$ at roughly 30 days to expiry,

$$\text{impliedMove} = \frac{C_{\text{ATM}} + P_{\text{ATM}}}{S}$$

Under standard Black-Scholes assumptions this approximates $\sigma_{\text{IV}} \cdot \sqrt{T} \cdot \sqrt{2/\pi}$, so the straddle-to-spot ratio is a clean, model-light proxy for ex-ante 30-day return uncertainty.

##### Origin in the Literature

Patell & Wolfson (1979, 1981, *Journal of Accounting Research*) were the first to document that equity-option implied volatilities rise systematically into scheduled earnings announcements and collapse on the release date — the original "IV crush" observation. (Beber & Brandt 2010, *Review of Finance*) generalize the idea to macro releases, showing that the ex-ante level of implied vol is a measure of resolvable uncertainty and that the post-release IV change is a tradeable signal of how much uncertainty was actually retired. (Dubinsky, Johannes, Kaeck & Seeger 2019, *Review of Financial Studies*, vol. 32(2), pp. 646-687) provide the modern reduced-form model: they decompose option prices into a "normal" diffusive component and an earnings-announcement jump component, estimate the announcement variance from straddles, and demonstrate it forecasts realized announcement-window volatility out of sample.

##### The IV Crush Trade

Pre-announcement, options on names with imminent earnings price in event-jump variance, so 30-day IV spikes (typically 30-60 percent above the no-event baseline) and ATM straddles become rich. On release, the jump risk is realized and IV mean-reverts violently — the **vol crush**. The textbook trade is therefore *short premium into the print* (sell the straddle, delta-hedge), capturing the variance risk premium plus the crush. The mirror trade — *long the straddle pre-print* — bets on the post-earnings-announcement drift and is the gamma-positive cousin documented in (Dubinsky et al. 2019).

##### Our Implementations

Two cross-sectional signals operationalize this literature:

- `opt_iv_crush_signal` (Beber-Brandt-style): rank `pre_iv30 - post_iv30` across the universe on each earnings date. Built in `src/earnings_iv_crush.py:build_iv_crush_signal()`, keyed by the EODHD earnings calendar; consumes cached Orats `iv30d` historically and Tradier daily polls live.
- `opt_implied_move_signal`: rank `-impliedMove` so that names with the **most expensive** straddles are predicted to mean-revert. Built in `src/options_signals.py:build_implied_move_signal()`. The straddle-to-spot computation lives in `src/options_adapter/chain_to_smv_summary.py`.

In-sample IC_IR for the crush signal lands in the **0.12-0.18** range, consistent with the Dubinsky et al. evidence that announcement variance is partially predictable.

##### Strategic Note

Tradier exposes per-contract bid/ask for free, so impliedMove is computable live with **zero marginal data cost** post-deployment — the Orats subscription is required only for the 2013-2026 historical training window.

##### Sources

- [Resolving Macroeconomic Uncertainty in Stock and Bond Markets (Beber & Brandt 2010) - Oxford Academic](https://academic.oup.com/rof/article/13/1/1/1589546)
- [Resolving Macroeconomic Uncertainty - NBER w12270 PDF](https://www.nber.org/system/files/working_papers/w12270/w12270.pdf)
- [Option Pricing of Earnings Announcement Risks (Dubinsky, Johannes, Kaeck & Seeger 2019) - Oxford Academic](https://academic.oup.com/rfs/article-abstract/32/2/646/5001193)
- [Option Pricing of Earnings Announcement Risks - VU Amsterdam PDF](https://research.vu.nl/ws/portalfiles/portal/108247883/Option_Pricing_of_Earnings_Announcement_Risks.pdf)


#### Variance Risk Premium + IV-RV Spread (Han-Zhou, Bali-Hovakimian)

The variance risk premium (VRP) and the implied-realized volatility spread are two closely related, options-derived signals that monetize the systematic gap between risk-neutral and physical variance forecasts. Both rank among the highest-IC signals in our `cz_signal_ic.csv` and were the #2 and #3 most important features in the `iso_OPTIONS_baseline` run, while requiring only Tradier-sourced 30-day implied volatility (CMIV) plus standard price history.

##### Variance Risk Premium (Bollerslev, Tauchen & Zhou, 2009)

Bollerslev, Tauchen & Zhou (2009, *RFS*) introduced the index-level VRP as the difference between option-implied variance and subsequently realized variance, demonstrating that the resulting wedge predicts a non-trivial fraction of post-1990 aggregate equity returns, with predictability concentrated at the quarterly horizon and dominating P/E, default spread, and consumption-wealth ratio benchmarks. Han & Zhou (2012) extended the construct cross-sectionally, sorting individual stocks on firm-level VRP and reporting long-short Sharpe ratios of roughly 0.7-1.0 with annualized spreads near 10%.

We compute VRP in annualized variance units as `iv30**2 - rv**2` (`src/options_signals.py:build_variance_premium_signal`). The economic sign is **negative for return prediction**: a high VRP indicates that variance is expensively priced, so variance sellers are over-compensated and the underlying tends to *underperform*. Our implementation flips the sign accordingly. In `cz_signal_ic.csv` the resulting `variance_premium` signal achieves IC_IR ≈ 0.10.

##### IV-RV Spread (Bali & Hovakimian, 2009)

Bali & Hovakimian (2009, *Management Science*) work in volatility (level) rather than variance (squared) space, defining the realized-implied spread as `RV_30d - IV_30d`. Using portfolio sorts and Fama-MacBeth regressions, they show that while raw RV and IV levels are not priced, the spread between them is, capturing a volatility-risk component distinct from the call-put implied-vol spread that proxies jump risk. Stocks where realized has overshot implied earn higher subsequent returns, consistent with vol-sellers being on the wrong side and the equity being temporarily underpriced.

Our `build_rv_iv_spread_signal` implements `RV - IV` directly with **positive sign**, achieving IC_IR = 0.155 in `cz_signal_ic.csv`. Because both signals require only IV30 and trailing returns, they remain fully buildable on the Tradier-only stack going forward, with no Orats subscription required for live serving.

Sources:
- [Expected Stock Returns and Variance Risk Premia | Oxford Academic](https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787)
- [Bollerslev, Tauchen & Zhou (2009) RFS PDF](https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf)
- [Expected Stock Returns and Variance Risk Premia | SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=948309)
- [Volatility Spreads and Expected Stock Returns | SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1443848)
- [Volatility Spreads and Expected Stock Returns | Management Science](https://dl.acm.org/doi/10.1287/mnsc.1090.1063)
- [Volatility Spreads and Expected Stock Returns | IDEAS/RePEc](https://ideas.repec.org/a/inm/ormnsc/v55y2009i11p1797-1812.html)


#### IV Term Structure: Slope, Convexity, Forward Curves (Vasquez 2017)

##### Foundational Literature

Vasquez (2017), "Equity Volatility Term Structures and the Cross Section of Option Returns," *Journal of Financial and Quantitative Analysis* 52(6), 2727-2754, establishes the canonical result: the **slope of the equity implied-volatility term structure is positively related to future option returns**. Sorting firms on the difference between long- and short-tenor at-the-money implied volatilities yields a monotonic cross-section, with steep upward-sloping curves preceding higher straddle returns. The economic interpretation is that an upward-sloping IV curve implies that short-dated volatility is depressed relative to its long-run anchor and is expected to mean-revert upward — a state that benefits long-vol option positions and, through the leverage and gamma channels, predicts cross-sectional dispersion in the underlying equity returns as well.

##### Implemented Slope Variants

We instrument three points on the curve to separate short-, mid-, and long-end information:

- **`iv_term_slope`** = `iv60d - iv30d` — the canonical Vasquez slope. Sign is **positive**: steep upward slope implies future vol expansion through mean reversion, which predicts higher equity returns.
- **`iv_short_term_slope`** = `iv30d - iv7d` — captures sub-two-week IV expectations, isolating event-driven kinks (earnings, macro prints) at the front of the curve that the 30/60d slope obscures.
- **`iv_long_term_slope`** = `iv365d - iv90d` — long-end "carry" measure capturing structural vol-risk-premium loadings independent of near-term event noise.

##### Convexity Extension

Beyond the linear slope, we measure **`iv_term_convexity`** = `iv30 - 2*iv60 + iv90`, a discrete second derivative. Sign is **negative**: pronounced negative convexity (mid-tenor expensive relative to the wings) signals pre-event positioning concentrated in 60d strikes that subsequently unwinds. This extends Vasquez's first-moment slope with a curvature channel.

##### Forward IV Curves (Variance Additivity)

By the no-arbitrage additivity of variance across non-overlapping intervals,

```
fwd_iv(t1 -> t2) = sqrt((iv_t2^2 * t2 - iv_t1^2 * t1) / (t2 - t1))
```

Forward IVs are **pure math from any constant-maturity IV grid** — Orats publishes `fwd30_20`, `fwd60_30`, `fwd90_60` as proprietary fields, but they are derivable for free from any vendor that supplies multiple maturities. Our implementation lives in `src/options_adapter/forward_curve.py:compute_forward_iv()` and reproduces Orats' published forwards within **0.5 vol points on 99.85% of 16,385 historical samples**. From this we construct three forward-curve signals:

- **`fwd_curve_slope`** = `-(fwd60_90 - fwd30_60)` — steep forward slope flags vol overpriced for the future window.
- **`realized_vs_forward`** = `RV30 - fwd30_60` — when realized exceeds the forward, vol-sellers were wrong and the market mean-reverts.
- **`long_forward_premium`** = `-(fwd90_180 - iv30)` — long-end vol crowding indicator.

##### Implementation Notes

Convexity and forward slope encode similar curvature information; we **decorrelate at the model stage rather than dropping** either, preserving non-linear interactions. Critically, every signal in this chunk is **Tradier-only viable**: Tradier returns option chains across multiple maturities, our CMIV layer interpolates constant-maturity points, and the forward curve falls out of variance additivity — no proprietary feed required.

Sources:
- [Equity Volatility Term Structures and the Cross-Section of Option Returns (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1944298)
- [Vasquez 2017 JFQA listing (IDEAS/RePEc)](https://ideas.repec.org/a/cup/jfinqa/v52y2017i06p2727-2754_00.html)
- [Vasquez 2017 (EconPapers)](https://econpapers.repec.org/article/cupjfinqa/v_3a52_3ay_3a2017_3ai_3a06_3ap_3a2727-2754_5f00.htm)
- [Vasquez full paper (EFMA 2015 working version)](https://efmaefm.org/0efmameetings/efma%20annual%20meetings/2015-Amsterdam/papers/EFMA2015_0530_fullpaper.pdf)


#### Risk Reversal, SmileSlope, Crash Risk (Xing-Zhang-Zhao, Bali-Murray, Kelly-Jiang)

The implied-volatility surface of single-name equity options encodes forward-looking, risk-neutral information about the conditional return distribution that the historical realised time series cannot recover. Three closely related but theoretically distinct signals — Risk Reversal (RR25), Smile Slope, and Crash Risk — exploit different sections of the 30-day delta-IV curve to extract complementary cross-sectional return predictors.

##### Risk Reversal (RR25)

Xing, Zhang and Zhao (2010), in "What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns?" (JFQA 45(3): 641-662), document that stocks whose volatility smirk — the gap between OTM-put IV and ATM-call IV — is steepest underperform stocks with the flattest smirks by roughly 10.9% per year on a risk-adjusted basis. The mechanism is informational: traders with negative private information preferentially purchase OTM puts, bidding up downside IV before the equity market incorporates the news. We define `RR25 = dlt25Iv30d - dlt75Iv30d` in Orats's call-delta convention (equivalent to 25-delta call IV minus 25-delta put IV, since a 75-delta call ≈ a 25-delta put). The signal carries a **negative** loading: a deeply negative RR25 (high put skew) flags an oversold name with positive expected return. Implementation: `src/options_signals.py:build_risk_reversal_25d_signal()`; in-sample IC_IR ≈ 0.155.

##### Smile Slope

Bali and Murray (2013), "Does Risk-Neutral Skewness Predict the Cross Section of Equity Option Portfolio Returns?" (JFQA 48(4): 1145-1171), construct delta- and vega-neutral skewness assets and find a robust negative pricing of risk-neutral skewness, consistent with skewness preference. We operationalise the slope directly as `dlt75Iv30d - dlt25Iv30d`, replacing an earlier OLS-fit method whose rank correlation against the Orats reference was only 0.063; the closed-form bucket difference recovers a rank correlation of 0.753. Per `cz_signal_ic.csv`, SmileSlope posts an IC_IR of 0.369, the second-highest options signal in our research panel. The sign is **positive** — counterintuitively, names with high crash fear are oversold and earn the skewness risk premium ex post. Implementation: `src/options_signals.py:build_smileslope_signal()`.

##### Crash Risk

Kelly and Jiang (2014), "Tail Risk and Asset Prices" (RFS 27(10): 2841-2871), establish that tail-risk exposure carries a positive risk premium in the cross section. We adapt their insight to the single-name IV surface as `CrashRisk = dlt95Iv30d - iv30d`, the deep-OTM-put (5-delta put ≈ 95-delta call) premium over ATM volatility. Loading is **positive**: names paying the highest crash insurance premium compensate holders with higher subsequent returns. Implementation: `src/options_signals.py:build_crash_risk_signal()`.

All three signals are derivable from Tradier's delta-bucket IV chain via PCHIP interpolation (`dlt5/25/75/95Iv30d`), eliminating any forward dependence on an Orats subscription for production scoring.

Sources:
- [Xing, Zhang & Zhao (2010), JFQA — author preprint](https://www.ruf.rice.edu/~yxing/option-skew-FINAL.pdf)
- [Xing, Zhang & Zhao (2010) — Cambridge Core](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/ECFD16BA9ACBDC8D577D1BD866FBEA72/S0022109010000220a.pdf/div-class-title-what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns-div.pdf)
- [Bali & Murray (2013), JFQA — UNL repository](https://digitalcommons.unl.edu/context/financefacpub/article/1029/viewcontent/Murray_JFQA_2013_Does_Risk_Neutral_Skewness__Cambridge_UP.pdf)
- [Bali & Murray (2013) — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1572827)
- [Kelly & Jiang (2014), RFS — Oxford Academic](https://academic.oup.com/rfs/article-abstract/27/10/2841/1607080)
- [Kelly & Jiang (2014) — NBER WP 19375](https://www.nber.org/system/files/working_papers/w19375/w19375.pdf)


#### IV Rank, Vol-of-Vol, OI Concentration (Goyal-Saretto, Cremers, Pan-Poteshman)

This chunk documents three lower-IC but diversifying options-derived signals layered on top of the SmileSlope and VRP cores: implied-volatility rank, volatility-of-volatility, and open-interest concentration. All three are sourced live from Tradier chains and aggregated across the multi-expiration term structure.

##### IV Rank (Goyal & Saretto, 2009)

Goyal and Saretto, "Cross-Section of Option Returns and Volatility," *Journal of Financial Economics* 94(2), 310-326 (2009), show that equity-option returns are strongly predictable from the gap between historical realized volatility and ATM implied volatility — a reduced-form expression of mean reversion in implied vol. Long-short option strategies that sell expensive vol (high IV percentile) and buy cheap vol (low IV percentile) earn 50%+ annualized in their cross-section. We translate this into an equity-side signal as the rolling percentile of current 30-day IV in its trailing 252-day window, scaled to [0, 1]. Sign is **negative**: a high IV-rank stock has expensive options, vol is expected to mean-revert downward, and underlying-equity drift tends to disappoint relative to the expected-return wedge implied by an inflated risk premium. Implementation: `src/options_signals.py:build_iv_rank_signal()` (line 186) — `iv30.rolling(252, min_periods=60).rank(pct=True)`, then cross-sectionally re-ranked with a sign flip.

##### Volatility-of-Volatility (Cremers, Halling & Weinbaum, 2015)

Cremers, Halling, and Weinbaum, "Aggregate Jump and Volatility Risk in the Cross-Section of Stock Returns," *Journal of Finance* 70(2) (2015), document a robust **negative** equity-return premium for vol-of-vol exposure: stocks whose IV itself is unstable carry a discount, but realize lower returns going forward (compensation is dwarfed by ex-post crash drag). We proxy with the rolling 30-day standard deviation of `iv30`. High vol-of-vol = unstable, hard-to-hedge regime — avoid.

##### Open Interest Concentration (Pan & Poteshman, 2006)

Pan and Poteshman, "The Information in Option Volume for Future Stock Prices," *Review of Financial Studies* 19(3), 871-908 (2006), show that option order flow contains private information about underlying stocks: low put/call ratios precede next-day excess returns of 40+ bps and ~1% over a week. We adapt the open-interest analogue: `call_oi / (call_oi + put_oi)`, summed across the full chain. Sign is **positive** — call-heavy positioning reflects informed bullish demand. Implementation: `src/options_signals.py:build_oi_concentration_signal()` (line 406).

##### Implementation & Performance Notes

Tradier's chain endpoint returns volume, OI, and per-contract IV across every listed expiration; we aggregate across the multi-exp surface and maintain a forward IV history via daily cron polls (no Orats backfill, so the IV-rank window builds out prospectively). Standalone IC_IR for each of the three falls in the **0.10-0.15** range — materially below SmileSlope or VRP — but their factor exposures are nearly orthogonal to the vol-premium core, so they earn their seat through diversification rather than raw signal strength.

Sources:
- [Goyal & Saretto (2009), Cross-Section of Option Returns and Volatility, JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X09001251)
- [Goyal & Saretto (2009), preprint PDF](https://personal.utdallas.edu/~axs125732/CrossOptionsJFE.pdf)
- [Pan & Poteshman (2006), The Information in Option Volume for Future Stock Prices, RFS](https://academic.oup.com/rfs/article-abstract/19/3/871/1646711)
- [Pan & Poteshman, NBER w10925](https://www.nber.org/papers/w10925)


#### BKM Implied Moments - Risk-Neutral Skew & Kurtosis (Bakshi-Kapadia-Madan 2003)

##### Bakshi-Kapadia-Madan (BKM) Implied Moments — Risk-Neutral Skewness and Kurtosis from the Full Smile

##### Origin

Bakshi, Kapadia & Madan (2003), "Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options," *Review of Financial Studies* 16(1), 101-143, derive a model-free spanning result that recovers the entire risk-neutral distribution of returns from a cross-section of European option prices. The same machinery underlies the CBOE SKEW index. Whereas ATM-only or 25-delta-only volatility metrics collapse the smile to a single number, BKM reads the **full** OTM strike continuum and decomposes the implied distribution into its second, third, and fourth moments.

##### Key Insight

A traded option's payoff is a Dirac function in strike space; an integral of OTM call and put prices weighted by appropriate strike-dependent kernels reconstructs the risk-neutral expectation of any twice-differentiable payoff. Applying this to powers of log-returns yields:

- Risk-neutral variance σ²_RN
- Risk-neutral skewness — typically **negative** for individual equities (downside fear premium)
- Risk-neutral kurtosis — typically **>3** (fat-tail premium for jump risk)

##### Discretized Formulas (trapezoidal integration over OTM strikes)

```
V(t,τ) = ∫ [2(1 − ln(K/S)) / K²] C(K) dK
       + ∫ [2(1 + ln(S/K)) / K²] P(K) dK              (variance)

W(t,τ) = ∫ [(6 ln(K/S) − 3 ln²(K/S)) / K²] C(K) dK
       + (corresponding put term)                        (cubic)

X(t,τ) = ∫ [(12 ln²(K/S) − 4 ln³(K/S)) / K²] C(K) dK
       + (corresponding put term)                        (quartic)

μ      = e^(rτ) − 1 − e^(rτ)·V/2 − e^(rτ)·W/6 − e^(rτ)·X/24
σ²_RN  = e^(rτ)·V − μ²
skew   = (e^(rτ)·W − 3μ·e^(rτ)·V + 2μ³) / σ³_RN
kurt   = (e^(rτ)·X − 4μ·e^(rτ)·W + 6 e^(rτ)·μ²·V − 3μ⁴) / σ⁴_RN
```

##### Empirical Findings

Conrad, Dittmar & Ghysels (2013), "Ex Ante Skewness and Expected Stock Returns," *Journal of Finance* 68(1), 85-124, demonstrate that risk-neutral skewness predicts the cross-section of stock returns: **less-negative (more right-shifted) skew forecasts higher subsequent returns**, even after controlling for systematic co-moments. Cremers & Driessen (2008) and the broader crash-risk-premium literature show that risk-neutral kurtosis carries an analogous tail-fear premium that mean-reverts.

##### Our Implementation

`src/options_adapter/implied_moments.py:compute_bkm_moments()` implements full BKM with trapezoidal integration over a liquidity-filtered OTM smile:

- Liquidity filter: bid > 0 and relative spread ≤ 1.5
- `min_otm_per_side = 5` strikes (else NaN)
- Outputs: `implied_var_30d`, `implied_skew_30d`, `implied_kurt_30d`

Live AAPL validation:

- BKM/ATM variance ratio = 1.018 (consistent — smile correction modest for liquid names)
- `implied_skew = −1.297` (sharply negative — downside fear priced in)
- `implied_kurt = 9.51` (well above 3 — fat tails confirmed)

##### Two New Signals

- `build_implied_skew_signal()` — sign **POSITIVE** per Conrad-Dittmar-Ghysels: less-negative skew → outperformance
- `build_implied_kurt_signal()` — sign **POSITIVE**: high kurtosis = priced crash fear that mean-reverts, paying the seller of tail insurance

##### Tradier-Only Viability

Critically, BKM is **viable on the free Tradier feed** without paid Orats subscription: the chain endpoint exposes `greek_smv_vol` per contract, giving the full smile. BKM integrates that smile directly, so we recover risk-neutral skew and kurt without any historical-Orats dependency — train/serve consistent and zero marginal data cost.

Sources:
- [Bakshi, Kapadia & Madan (2003) RFS PDF](https://people.umass.edu/nkapadia/docs/Bakshi_Kapadia_Madan_2003_RFS.pdf)
- [Bakshi, Kapadia & Madan (2003) — Oxford Academic](https://academic.oup.com/rfs/article-abstract/16/1/101/1615098)
- [Conrad, Dittmar & Ghysels (2013) — Journal of Finance](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2012.01795.x)
- [Conrad, Dittmar & Ghysels (2013) — UNC Repository PDF](https://cdr.lib.unc.edu/downloads/pr76fc54k)
- [BKM Gram-Charlier extension — Review of Derivatives Research](https://link.springer.com/article/10.1007/s11147-022-09187-x)


#### Tradier Migration - Architecture, What's Lost, $0/mo Going Forward

##### Tradier Migration: What We Lose, What We Keep, What's Free

##### Why Tradier
Tradier's market data API is **free with any brokerage account** — no separate $99/mo subscription, no per-call metering for normal polling cadence. The decisive feature for our stack: Tradier's `/markets/options/chains` endpoint **embeds Orats SMV Greeks** (per Tradier docs), meaning IV, delta, gamma, theta, vega come from the same SMV (Stochastic-Model Volatility) engine that produced our cached Orats historical data. This delivers **train/serve consistency**: the 13 years of Orats Delayed Data we already cached and the daily Tradier polls going forward are produced by the same vol surface model. No methodology drift between train and inference.

##### Architecture
- **`src/tradier_client.py`** — `TradierClient` class with rate-limit retry (exponential backoff on 429) and `ThreadPool` batch fetching across tickers/expirations.
- **`src/options_adapter/chain_to_smv_summary.py`** — orchestrator that takes a raw Tradier chain and emits Orats `/cores`-equivalent rows (~27 fields per ticker per day).
- Five sub-adapters under `src/options_adapter/`:
  - `cmiv_interpolator.py` — variance-time interpolation across expiries → `iv7d/14d/30d/60d/90d/180d/365d`
  - `tradier_orats_adapter.py` — PCHIP smile fit → `slope`, `dlt5/25/75/95Iv30d`
  - `implied_borrow.py` — put-call parity → `borrow30`
  - `implied_dividend_proxy.py` — PCP-based dividend yield with EODHD fallback
  - `forward_curve.py` — variance additivity → `fwd30_60`, `fwd60_90`, `fwd90_180`
  - `implied_moments.py` — BKM trapezoidal integration → `implied_var/skew/kurt`

##### Daily Polling
- **`cron/poll_tradier_daily.bat`** — Windows Task Scheduler entry, runs 16:30 ET.
- **`run_options_setup.py:cmd_daily_poll`** — fetches 5 expirations × N tickers, runs through SMV adapter, writes `data/cache/options/tradier_daily/<date>.parquet`.
- Wires the EODHD dividend yield map for accurate `borrow30` + `annIdiv` solving.
- 17 fields per ticker per day land in the daily snapshot.

##### Merger Pipeline
**`merge_tradier_daily_to_iv_panel.py`** rolls daily snapshots into per-field date×ticker panels at `data/cache/options/iv_panels_tradier.pkl`, consumed directly by `src/options_signals.py`.

##### Validation (Phase 1, N=30 large-caps, 2026-04-17)
- All 6 IV/delta fields rank-correlate **≥ 0.88 vs Orats**.
- `slope`: improved from **0.063 → 0.753** after methodology fix (replaced OLS regression with `dlt75 - dlt25` differential).
- `borrow30` + `annIdiv` remain **weak rank-corr (~0.2)** — these drive only two low-IC bonus signals, so the residual error is tolerable.

##### What's Permanently Lost by Cancelling Orats
- **`fcstSlope`** — Orats's proprietary IV forecast model. Not used by any of our signals, so zero impact.
- **Real-time `exErnIv*` updates** — workaround: 13 years cached + forward proxy via earnings calendar.
- **Real-time `impliedMove` from `/summaries`** — workaround: compute ourselves from straddle inside the adapter.

##### Coverage
- **Orats**: 1,414 tickers cached.
- **Tradier**: 80–86% of full backtest universe (5,077 tickers) — **broader coverage than Orats** going forward.

##### Cost Timeline
- Pre: **$99/mo** Orats Delayed Data.
- Post: **$0/mo** Tradier (free, requires brokerage account).
- Savings: **$1,188/yr indefinitely**, with no signal-quality regression on the fields that drive our IC.


---

### 40.3 Master Signal Priority - Quick Reference

Sorted by IC_IR (per `results/_cz_research/cz_signal_ic.csv`). All signals built via `--use-cz-signals` (C&Z) or `--use-options-signals` (Orats/Tradier-derived) flags. Validated subset for production: `OPTIONS_SIGNAL_SET=validated`.

| IC_IR | Signal | Source | Built? | Tradier-only OK? |
|---|---|---|---|---|
| 0.375 | dCPVolSpread | Options /cores delta-bucket spread d5d | Yes | YES |
| 0.369 | SmileSlope | Bali-Murray 2013, dlt75-dlt25 | Yes (post-fix rank corr 0.753) | YES |
| 0.286 | OScore | Campbell-Hilscher-Szilagyi (covered via `chs_distress_signal`) | Yes | n/a |
| 0.254 | AnnouncementReturn | Brandt et al. 2008 | Yes | YES (via Tradier+EODHD calendar) |
| 0.250 | ShareIss5Y | Pontiff-Woodgate 2008 (covered via `net_share_issuance_signal`) | Yes | n/a |
| 0.245 | EarningsStreak | Bartov-Givoly-Hayn 2002 | Yes | YES |
| 0.225 | ShareRepurchase | Grullon-Michaely 2004 (covered via `q_buyback_yield`) | Yes | n/a |
| 0.200 | CBOperProf | Ball et al. 2016 | Yes (`alt_cash_based_op_prof_signal`) | n/a |
| 0.200 | FirmAgeMom | (firm-age momentum) | **NO** - easy build | YES (price-only) |
| 0.198 | Coskewness | Harvey-Siddique 2000 | Yes (price-only, 2 horizons) | YES (price-only) |
| 0.181 | XFIN | Bradshaw-Richardson-Sloan 2006 | Yes (`build_xfin_signal`) | n/a |
| 0.178 | NetEquityFinance | Sign-flipped NetPayoutYield | Yes (covered) | n/a |
| 0.176 | NetPayoutYield | Boudoukh et al. 2007 | Yes (`build_net_payout_yield_signal`) | n/a |
| 0.155 | rv_iv_spread / RR25 | Bali-Hovakimian / Xing-Zhang-Zhao | Yes | YES |
| 0.152 | AnalystRevision | (analyst signals - Finnhub data needed) | **NO** - Finnhub | YES (need Finnhub) |
| 0.146 | PayoutYield | Boudoukh et al. 2007 | Yes | n/a |
| 0.146 | MomSeason | Heston-Sadka 2008 | Yes (price-only) | YES (price-only) |
| 0.143 | ChangeInRecommendation | Finnhub recommendations | **NO** - Finnhub | YES |
| 0.136 | DelDRC | Prakash-Sinha 2013 | Yes | n/a |
| 0.135 | RDAbility | (R&D efficiency) | **NO** - EDGAR | YES (need EDGAR) |
| 0.132 | RoE | (direct ROE) | **NO** - distinct from `q_roe` | YES (EDGAR) |
| 0.127 | EarningsConsistency | (CV of surprises) | Yes (`earnings_surprise_consistency`) | YES |
| 0.125 | OperProfRD | Ball et al. 2016 R&D variant | Yes (`build_operprof_rd_signal`) | n/a |
| 0.124 | CoskewACX | Ang-Chen-Xing 2006 downside variant | Yes | YES (price-only) |
| 0.123 | Tax | Lev-Nissim 2004 | Yes | n/a |
| 0.123 | std_turn | (idiosyncratic turnover) | **NO** | YES (price+volume) |
| 0.119 | NOA | Hirshleifer et al. (covered via `net_operating_assets_signal`) | Yes | n/a |
| 0.119 | CitationsRD | (NBER patent dataset) | **NO** - patents | YES (NBER patents) |
| 0.113 | sfe | (sector FX exposure) | **NO** | YES (cross-asset) |
| 0.111 | CFP | Lakonishok-Shleifer-Vishny 1994 | Yes | n/a |
| 0.103 | GP | Novy-Marx (covered via `gross_profitability_signal`) | Yes | n/a |
| 0.103 | EP | (earnings/price - covered via `earnings_yield_signal`) | Yes | n/a |

**Net status (post Tradier migration + 5-agent extraction):**
- 26+ functional options signals from Tradier alone (was 15, with 4 dead)
- Coskewness confirmed PRICE-ONLY (no Orats dependency)
- 11 high-IC C&Z signals MISSING (FirmAgeMom, AnalystRevision, RoE, RDAbility, etc.) - buildable from existing data sources
- ~$1,188/yr saved by cancelling Orats once 30-day Tradier archive accumulates

### 40.4 Cross-References

- **Source chunks**: `docs/research_chunks/01_cz_project.md` ... `20_orats_tradier_migration.md`
- **C&Z research outputs**: `results/_cz_research/cz_signal_ic.csv`, `cz_overlap_map.csv`
- **Orats field inventories**: `results/validation/orats_cores_field_inventory.csv`, `orats_other_endpoints_inventory.csv`
- **Phase 1 Tradier validation**: `results/validation/phase1_n30_post_fix_2026-04-17.csv`
- **Validated signal subset**: `OPTIONS_SIGNAL_SET=validated` keeps top-5 (dCPVolSpread, SmileSlope, variance_premium, rv_iv_spread, iv_term_slope)
- **Test discipline**: cache-key fix in `src/model.py:234-256` includes `EXTRA_DENY_FEATURES` for proper isolation; new CLI flags `--cz-only=<csv>` and `--cz-exclude=<csv>` enable per-signal A/B tests
