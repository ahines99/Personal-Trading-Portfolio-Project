# Research Plan — Optimizing ML Model for CAGR/Sharpe

> Compiled 2026-04-17 from 30 parallel research agents (5 data cache health + 5 feature cache analysis + 10 literature review + 10 ML model review). Subsequently audited by 10 plan-review agents. Targeting improvements over the 18.81% honest baseline.

---

## Critical Findings (Must Read Before Any Experiment)

### 🔴 BUG 1 — Label Leakage from Insufficient Embargo
`forward_window=7d` + `embargo_days=5d` → the last 2 days of training labels overlap the first test day's feature window. AFML Ch 7.4 explicitly requires `embargo ≥ label_span + buffer`. **Estimated inflation of historical IC: 1-3%; may explain a meaningful fraction of the 25.67%→18.81% reproduction gap.**

**Fix**: set `embargo_days ≥ forward_window + 2` (i.e., 9d when fw=7, 12d when fw=10). One-line change in `model.py` constructor default.

### 🔴 BUG 2 — Manual `_DENY_LIST` is a Substitution-Effect Disaster
93-101 features manually deny-listed based on LGBM **gain importance**. Gain cannot split credit between correlated features — one feature's twin wins the coin-flip and the other is dropped, removing real signal. This is the textbook Lopez de Prado warning (AFML Ch 8). Very likely the root cause of post-April-15 regression.

**Fix**: Delete `_DENY_LIST`. Replace with:
1. HRP feature clustering (already have the code at `hrp_feature_clusters`).
2. Clustered **permutation importance** on OOS folds.
3. Drop whole clusters below threshold, never a single feature whose twin still sits in the model.

### 🔴 BUG 3 — Phase D Data is 0% Populated
Cache audit found `eps_surprise_pct` and `before_after_market` both 0% non-null in earnings estimates cache. `earnings_beat_streak_signal`, `earnings_surprise_consistency_signal`, `earnings_ann_return_signal` are all training on **pure NaN**. Phase D's claimed +134 bps / +475 bps contributions are fiction — the features aren't loaded.

**Fix**: Re-fetch EODHD earnings with verified response parsing. Cross-check that `api_data.py` actually extracts `percent` → `eps_surprise_pct` and `before_after_market`. Validate 1 known ticker (AAPL) manually before re-running.

### 🔴 BUG 4 — Deflated Sharpe Ratio Over-Credited
DSR 0.957 was computed with `n_trials` = number of walk-forward folds (52), not the true search cardinality. Actual `n_trials` should count:
- ~52 archived `_iso_*`/`_phase_*` result folders
- Optuna trials (default 40, possibly many runs)
- ~80 CLI flags × combinations tested
Honest `n_trials > 500`. Real DSR probably <0.5. Strategy **fails Harvey-Liu t>3 threshold** (α t=1.42, p=0.155).

**Fix**: Re-compute DSR with honest `n_trials` from git history + log directory count. If DSR < 0.5, flag the strategy as statistically suspect and require OOS validation on held-out 2026+ data before any capital allocation.

### 🔴 BUG 5 — VIX9D Missing From Macro Cache
Short-end of VIX term structure is absent. `vix_change_5d` is computable but any VIX9D-based signal (9d vs 30d term-slope) silently uses NaN.

**Fix**: Backfill from CBOE historical API; regenerate `vix_term_v2.pkl`.

### 🔴 BUG 6 — EDGAR Fundamentals v2raw Deserialization Broken
`edgar_fundamentals_*_v2raw.pkl` fails to load due to pandas 3.x / Python 3.13 StringDtype pickle incompatibility. We've been using v1 (8 concepts) instead of v2raw (15 concepts). **Most fundamental signals may be incomplete.**

**Fix**: Regenerate pickle with current pandas or migrate to parquet.

### 🔴 BUG 7 — `institutional_holders_v2.pkl` Loads as Empty
Same StringDtype bug. Zero impact today (feature is OFF), but blocks any future re-enable.

### 🟠 BUG 8 — SRAF Monotonic-Index Dispatch Failure
Known issue in `alt_features.py` — throws on non-sorted index, loses 3-5 SRAF sentiment features. Memory notes suggest `--use-sraf-sentiment` could add +130 bps / +0.03 Sharpe (Fixed1 archive).

**Fix**: `df.sort_index()` before dispatch. 1-line fix.

### 🟠 BUG 9 — `forward_window` Constructor/CLI Mismatch
CLI default 7, `WalkForwardModel` constructor default 5. Production path uses CLI so effective = 7, but any direct instantiation silently uses 5. Fragile.

**Fix**: Align both to 7 (or 10 per IC peak analysis).

---

## Tier 1 — Quick Wins (≤1 day each, high expected ROI)

| # | Change | Expected Δ | File refs | Rationale |
|---|---|---|---|---|
| T1.1 | **Fix embargo_days to `max(9, forward_window+2)`** | Corrects label leakage; CAGR/Sharpe may *drop* honestly but true alpha will survive | `model.py` constructor | AFML Ch 7.4; root cause of part of repro gap |
| T1.2 | **Replace `_DENY_LIST` with clustered permutation importance on OOS** | Potentially recovers up to 500+ bps toward 25.67% baseline if §25 bisect bug + substitution effect were the real cause | `model.py:800-868` → new `select_features_clustered_oof()` | Lopez de Prado AFML Ch 8; fixes root cause of repro gap |
| T1.3 | **Set `--forward-window=10`** (IC peak) | +5-10% IR → ~+50-100 bps CAGR | 1 CLI flag | `factor_decay.csv` peak at 10d |
| T1.4 | **Drop bottom-10 low-gain features** | +0.02-0.05 Sharpe | `EXTRA_DENY_FEATURES` env | §22 audit — gains 366-480 |
| T1.5 | **Drop all 15 `leadlag_*` features** (zero importance) | Reduce overfit, +0.01-0.03 Sharpe | `EXTRA_DENY_FEATURES="leadlag_*"` | §56 confirmed zero importance |
| T1.6 | **Fix SRAF dispatch (sort_index before call)** | +130 bps CAGR / +0.03 Sharpe if Fixed1 archive holds | `alt_features.py` | 1-line fix, tested-positive feature |
| T1.7 | **Seed-bag LGBM at 5 seeds** | +0.05-0.15 Sharpe (Numerai/Kaggle standard) | `model.py`: set `lgbm_num_seeds=5` | Variance reduction ~1/√5 |
| T1.8 | **Sample-weight halflife 630d → 1260d** | +50-100 bps Sharpe; less late-2022 regime bias | `model.py` `sample_weight_decay_days` | GKX 2020 / AFML best practice for 13y training |
| T1.9 | **Un-clip earnings `eps_surprise_pct` from ±2 → winsorize cross-sectionally 1/99** | +30-80 bps; preserves right-tail PEAD signal | `alt_data_loader.py:1003` | Ball-Brown right-tail carries most alpha |
| T1.10 | **Shorten PEAD carry 60d → 15d with hl=7d exponential decay** | +20-50 bps; frees capital for next earnings cohort | `alt_features.py:602` | Bernard-Thomas post-ann drift concentration |

**Tier 1 combined expected lift: +200-800 bps CAGR / +0.3-0.6 Sharpe** (range wide because T1.1-T1.2 may reveal the baseline was over-stated).

---

## Tier 2 — New Features (1-3 days each, literature-backed)

| # | Feature | Source | Expected IR | Data req |
|---|---|---|---|---|
| T2.1 | **EarningsStreak** (C&Z, IC_IR 0.245) | Chen-Zimmermann | +0.05-0.10 Sharpe | Already have EODHD |
| T2.2 | **ShareRepurchase + NetEquityFinance** (C&Z, 0.22/0.18) | EDGAR CF statement | +0.05-0.08 Sharpe | Have EDGAR extra |
| T2.3 | **Intangible-adjusted B/M (iHML)** | Arnott-Harvey-Kalesnik-Linnainmaa 2021 | Sharpe 0.55 vs 0.30 plain B/M | EDGAR R&D + SG&A (PIT cap) |
| T2.4 | **5Y composite share issuance** (Daniel-Titman) | DT 2006, Pontiff-Woodgate | Sharpe 0.70 US | EDGAR shares outstanding |
| T2.5 | **QMJ Safety sleeve**: earnings vol + ROE vol (5Y) | Asness-Frazzini-Pedersen 2019 | Long-only Sharpe ~0.45 | EDGAR quarterly |
| T2.6 | **EBITDA/EV + Payout Yield composite** | Loughran-Wellman 2011 | Sharpe 0.61 | Already have inputs |
| T2.7 | **SUE (time-series, Foster-Olsen-Shevlin)** standardized by 8Q rolling stdev | FOS 1984 | IR 0.6-0.9 | Already have EODHD history |
| T2.8 | **Pre-earnings drift signal** (T-10 to T-2) | Frazzini-Lamont 2007 | IR ~0.3 | Earnings dates |
| T2.9 | **Barroso-Santa Clara vol scaling on mom_12_1** | B-SC 2015 | +0.3-0.4 Sharpe (single largest mom improvement) | Already have rvol |
| T2.10 | **Novy-Marx intermediate momentum (t-12 to t-7)** | Novy-Marx 2012 | IC ~0.02, orthogonal to 12-1 | Already have prices |
| T2.11 | **Target-price momentum** (3-mo Δ median target/price) | Da-Schaumburg 2011 | IR ~0.4-0.5 | EODHD analyst (verify pulled) |
| T2.12 | **Forecast dispersion** (std/|mean| of analyst EPS) | Diether-Malloy-Scherbina 2002 | IR ~0.3 | Analyst estimates |
| T2.13 | **Earnings-call Q&A FinBERT tone** | Huang-Wang-Yang 2023 | 1.5-3× LM dict lift | EDGAR 8-K Exhibit 99 |
| T2.14 | **FinBERT replacement for LM dict on 10-K Item 7** | Huang et al. 2023 | +1-2 IC points | Have 10-K text |
| T2.15 | **VIX term structure (VIX3M/VIX1M) × defensive sector interaction** | Johnson 2017 | +0.05-0.10 Sharpe | FRED CBOE |
| T2.16 | **Breakeven × B/M interaction** (growth-duration trade) | — | +0.03-0.05 Sharpe | Existing broadcasts |
| T2.17 | **Credit β × firm leverage** (not sector-level) | Campbell-Hilscher-Szilagyi | +0.05 Sharpe | Existing credit β + EDGAR |

**Tier 2 combined expected lift: +100-400 bps CAGR if 8-10 of 17 hit published IR.**

Prioritize by effort × confidence: T2.1 (EarningsStreak) > T2.9 (BSC vol-scaling) > T2.7 (SUE) > T2.10 (intermediate mom) > T2.3 (iHML) > T2.2 (ShareRepurchase).

---

## Tier 3 — Architecture Upgrades (1-3 days each)

| # | Upgrade | Expected Δ | Risk | Rationale |
|---|---|---|---|---|
| T3.1 | **OOF stacking with non-negative Ridge over {LGBM, XGB, Ridge, MLP, rule-composite}** | +0.15-0.30 Sharpe | Med | Replaces fixed 70/30; GKX/Chen-Pelger-Zhu standard |
| T3.2 | **Add CatBoost head at 12% weight** (already installed, docstring claims it!) | +0.05-0.10 Sharpe | Low | True diversity vs LGBM's leaf-wise |
| T3.3 | **HMM regime detection + weighted-blend (not hard switch)** | +0.3-0.5 Sharpe (mostly loss avoidance in bear_calm) | Med | Kritzman-Page-Turkington 2012 |
| T3.4 | **Bear_calm kill-switch**: 60d DD<-15% + HMM P(bear)>0.6 → gross 25%, SPY hedge to β-neutral | Reduces -71% DD to ~-25% | Med | Addresses 10%-of-time disaster regime |
| T3.5 | **Daily TLH loop** (currently passive) + vol-scaled loss threshold `3% × σ/σ_median` | +40-80 bps after-tax | Low | Chaudhuri-Burnham-Lo 2020 |
| T3.6 | **Multi-horizon labels [1,5,21] weighted [0.2,0.5,0.3]** | +50-150 bps IR | Low | Already in code as `--forward-windows`, off by default |
| T3.7 | **Continuous rank targets (20-bin qcut) replacing 6-grade bucketing** | +30-80 bps | Low | Lim-Zohren 2020 LambdaMART |
| T3.8 | **Native LGBM categorical_feature=['sector']** + cat_smooth=10, cat_l2=10 | +0.02-0.05 Sharpe | Low | Replaces one-hot/target-encode |
| T3.9 | **Raise LambdaRank truncation 20→30 with exponential label_gain `[0,1,3,7,15,...]`** | +0.03-0.08 Sharpe | Low | Better alignment with top-decile P&L |
| T3.10 | **Widen Optuna search** (min_data_in_leaf 50-2000, λ1/λ2 up to 500, feature_fraction 0.1-0.8) | +50-150 bps if Numerai range is closer to optimum | Med | Our ranges are too conservative |

**Tier 3 combined expected lift: +200-500 bps CAGR / +0.5-1.2 Sharpe.**

---

## Tier 4 — Statistical Validation (ongoing)

| # | Task | Why |
|---|---|---|
| T4.1 | **Implement PBO (Probability of Backtest Overfitting)** on 52 folds with all candidate configs | Not currently computed; with ~500 trials, PBO likely >0.3 |
| T4.2 | **Re-compute DSR with honest n_trials** (count archives + Optuna + flags) | Current 0.957 is inflated; true value probably <0.5 |
| T4.3 | **White Reality Check / Hansen SPA** across tier experiments | Data-snooping correction |
| T4.4 | **McLean-Pontiff 35% haircut** on any anomaly-sourced feature | Set production expectation = 0.65 × backtest |
| T4.5 | **Point-in-time liquidity filter audit**: verify top-3000 is rebuilt per-date | Survivorship via filter re-introduction |
| T4.6 | **Audit 4 `np.broadcast_to` sites** in features.py for PIT correctness | One leaking broadcast contaminates every ticker |
| T4.7 | **Feature stability selection across 52 folds** | Keep only features in ≥60% of folds' top-K by permutation importance |
| T4.8 | **Log effective sample size** `(Σw)²/Σw²` per year | If <0.3 in early years, decay too aggressive |

---

## Tier 5 — Deferred (blocked on data or AUM)

| # | Task | Blocker |
|---|---|---|
| T5.1 | **Options signals** (15, +250-450 bps claim) | Tradier-Orats Phase 1 validation failing (IV30d diff +2.5 pts) |
| T5.2 | **Semi-beta N up-weight** (drop P/M+/M-) | Needs proper BPQ 2022 regime-conditioning |
| T5.3 | **dCPVolSpread + SmileSlope** (IC_IR 0.375, 0.369) | Options data pipeline |
| T5.4 | **13F ΔAUM HF-only** (IR ~0.25) | Needs HF-identifier mapping |
| T5.5 | **Supply-chain / patent novelty** | Requires new data source |

---

## Data Quality Roadmap (prerequisites for above)

1. **Fix v2raw pickle deserialization** — regenerate with compatible pandas or migrate to parquet. Unlocks 7 additional EDGAR concepts.
2. **Backfill VIX9D** from CBOE historical API into `vix_term_v2.pkl`.
3. **Re-fetch earnings calendar** to populate `eps_surprise_pct` + `before_after_market` (Phase D is bricked without this).
4. **Purge delisted tickers from hot cache** — separate archive. 42.7% of 4,948 tickers are delisted; current cache bloats RAM.
5. **Resume Tradier daily poll** — only 1 parquet collected so far; cron IS registered but hasn't accumulated days.

---

## Testing Methodology

Every Tier 1-3 change must:

1. **Run with `DENY_LIST_HASH_SUFFIX=$(uuidgen)` to force cache miss** (prevents §25 cache-key collision bug masking isolation results).
2. **Compare against a frozen reference run** (snapshot via `snapshot_baseline.py`).
3. **Report Δ CAGR / Δ Sharpe / Δ Max DD with confidence interval** via 1000-sample block bootstrap.
4. **Check OOS fold IC stability** (mean + std across 52 folds).
5. **Validate PBO < 0.3** if change is a config tweak (hyperparameter or feature selection).
6. **Manual inspection of top-5 picks per rebalance date** — look for obvious contamination (delisted tickers, penny stocks, earnings-blackout violations).

---

## Execution Order

Week 1 (critical bugs, mostly single-line fixes):
- Day 1 AM: T1.1 (embargo), T1.3 (fw=10), T1.9 (un-clip SUE), T1.10 (PEAD carry)
- Day 1 PM: T1.5 (drop leadlag), T1.6 (SRAF fix), T1.7 (seed-bag=5), T1.8 (hl=1260)
- Day 2: T1.2 (deny-list replacement — biggest single effort of Tier 1)
- Day 3: BUG 3 fix (re-fetch earnings), BUG 5 fix (VIX9D), BUG 6 fix (v2raw pickle)
- Day 4-5: Run baseline reproduction with all above. Target: recover toward 25.67% honest, not via contamination.

Week 2-3 (Tier 2 features, sequenced by effort):
- Week 2: T2.1, T2.9, T2.7, T2.10 (~2 days each, all use existing data)
- Week 3: T2.2, T2.3, T2.5, T2.6, T2.11, T2.12 (EDGAR + analyst features)

Week 4 (Tier 3 architecture):
- T3.6 (multi-horizon labels ON by default)
- T3.2 (add CatBoost)
- T3.8 (native categorical sector)
- T3.10 (Optuna widen)

Week 5 (advanced):
- T3.1 (OOF stacking)
- T3.3-T3.4 (HMM + kill-switch)

Week 6+ (validation + NLP/options):
- Tier 4 stats validation
- T2.13-T2.14 (FinBERT, if environment allows)
- T5.x as data/AUM permits

---

## Target Outcomes

Conservative (Tier 0 bug fixes + half of Tier 1):
- Baseline 18.81% → **honest 20-22% CAGR** (after removing leakage-inflated numbers and restoring correctly-processed features).
- Sharpe 0.622 → **0.75-0.85**.
- Max DD -59% → -40% to -50% (via kill-switch).

Aggressive (all Tier 1-3 hit):
- CAGR **25-30%**
- Sharpe **0.95-1.2**
- Max DD **-30% to -35%**
- OOS vs IS gap closes (OOS currently 1.045 vs IS 0.680 — already good)

Honest expectation after McLean-Pontiff 35% haircut for live trading:
- Live CAGR ≈ 0.65 × backtest → **16-18% realistic, 19-22% if optimistic**.
- Target beat: SPY 13.99% → meaningful real α of 2-8%.

---

## Risks / What Could Go Wrong

1. **T1.1 (embargo fix) may reveal baseline was 15-17%, not 18.81%**, if current baseline already benefits from small leak. Accept this — honesty is the goal.
2. **T1.2 (deny-list replacement) may regress** if the deny-list was accidentally removing genuinely noisy features (not substitution-effect victims). Have a rollback plan.
3. **T2.x new features may not hit published IR in our universe** (McLean-Pontiff decay, universe differences, implementation mistakes). Budget only 30-50% of published IR as realistic.
4. **T3.3 (HMM regime) needs enough regime observations to fit**; 52 folds × 4 regimes = 13 obs each — borderline. Use regime probabilities as features, not model partitions, to preserve sample.
5. **PBO > 0.5 on T4.1** would invalidate years of optimization work. Prepare for this outcome mentally.
6. **DSR < 0.5 on T4.2** means strategy lacks statistical edge post-multiple-testing. Would require either (a) accepting a lower-complexity baseline or (b) gathering fresh OOS data 2026-2027.

---

## Research References (selected)

- **Lopez de Prado 2018** — *Advances in Financial Machine Learning* — Ch 4 (weights), Ch 7 (CV), Ch 8 (feature importance)
- **Gu-Kelly-Xiu 2020 RFS** — "Empirical Asset Pricing via Machine Learning"
- **Bailey-Lopez de Prado 2014** — DSR / PBO
- **JKP 2023** (Jensen-Kelly-Pedersen) — "Is There A Replication Crisis In Finance?" JF
- **Harvey-Liu-Zhu 2016** — "...and the Cross-Section of Expected Returns" (t>3)
- **Harvey-Liu 2020** — "False (and Missed) Discoveries in Financial Economics"
- **Novy-Marx-Medhat 2025** — Profitability Retrospective
- **Arnott-Harvey-Kalesnik-Linnainmaa 2021** — Intangible-adjusted value
- **Barroso-Santa Clara 2015 JFE** — Momentum vol scaling
- **Ball-Gerakos-Linnainmaa-Nikolaev 2016 JAR** — Cash-based operating profitability
- **Cohen-Malloy-Nguyen 2020 JF** — Lazy Prices
- **Huang-Wang-Yang 2023 CAR** — FinBERT
- **Chen-Zimmermann Open Source Asset Pricing** — signal catalog
- **Cremers-Weinbaum 2010, Xing-Zhang-Zhao 2010** — Options signals
- **Kritzman-Page-Turkington 2012 FAJ** — Regime shifts
- **Chaudhuri-Burnham-Lo 2020** — Tax-loss harvesting alpha
- **Kelly-Malamud-Zhou 2023** — Virtue of Complexity
- **Lim-Zohren 2020** — LambdaMART for cross-sectional systematic
- **Ehsani-Linnainmaa 2022 JF** — Factor momentum

---

*Plan compiled from 30 research agents + 10 audit agents. Execute Tier 0-1 first (all should take ≤1 week). Validate at each step with PBO and blocked OOS. Expect honest improvement rather than chasing Fixed2 25.67% mirage.*

---

# APPENDIX A — Audit Corrections (Post-Review, 2026-04-17)

10 plan-audit agents cross-checked the original plan against actual code and data. Corrections below supersede the main body where they conflict.

## A.1 Bug Claims — Status After Audit

| Bug | Original claim | Audit verdict | Action |
|---|---|---|---|
| **BUG 1** (embargo leakage) | Real, high-priority | ✅ **CONFIRMED** — exact timing analysis proves last training label at T-6 uses returns to T+1, leaking into test window that starts at T+5. 5-day overlap confirmed. | Keep. Fix to `embargo_days = max(9, forward_window + 2)` is correct. Expected honest CAGR drop 1-3% from removing leak. |
| **BUG 2** (deny-list substitution-effect) | Root cause of repro gap | ⚠️ **SOFTENED** — Theoretically sound (gain-based selection is vulnerable) but empirically the 93-101 entries are dominated by `_sn/_sni/_csz/z_` transformed variants (90% of entries), NOT substitution victims of raw signals. Clustered permutation importance infrastructure exists (hrp_feature_clusters lines 1172-1240) but is unused. | **Revised approach**: Run `EMPTY_DENY_LIST=1` ablation FIRST. Only implement clustered permutation if Run-Delta > 200 bps. Don't assume this is the repro gap source. |
| **BUG 3** (Phase D data 0% null) | Critical data bug | ❌ **FALSE** — Audit found `eps_surprise_pct` is 96.6% populated (yfinance) / 75.3% populated (EODHD). `before_after_market` is 100%. Data IS correct. | **RETRACTED**. If Phase D signals are underperforming, root cause is downstream in alt_features.py processing or in the _DENY_LIST filtering, NOT data loading. |
| **BUG 4** (DSR inflated) | Real | ✅ **CONFIRMED** — run_strategy.py:1327 hardcodes `n_trials = args.optuna_trials if args.optimize_ml else 1`. All recent tearsheets report DSR at `n_trials=1`. Honest count (56 archive dirs + ~40 Optuna + ~100 CLI variants) = **200-500+**. At n_trials=300, DSR drops from ~0.92 → **0.36**. Strategy already fails Harvey-Liu t>3 (α t=1.42). | Keep. Implement honest n_trials counter. Prepare for DSR <0.5 reality. |
| **BUG 5** (VIX9D missing) | Blocks VIX term signals | ✅ **CONFIRMED** — VIX9D absent from vix_term_v2.pkl. | Keep. 1-day backfill from CBOE. |
| **BUG 6** (v2raw pickle broken) | StringDtype incompatibility | ✅ **CONFIRMED** — Python 3.13 + pandas 3.x. Currently using v1 (8 concepts) instead of v2raw (15). | Keep. Regenerate with compatible pandas or migrate to parquet. |
| **BUG 7** (institutional_holders empty) | Same pickle issue | ✅ **CONFIRMED** | Fix with BUG 6 batch. |
| **BUG 8** (SRAF monotonic-index) | Real | ✅ **CONFIRMED** | Keep. 1-line fix. |
| **BUG 9** (forward_window 5 vs 7 mismatch) | Fragile | ✅ **CONFIRMED** | Keep. Align constructor default to CLI. |
| **NEW BUG 10** (point-in-time liquidity filter) | Not in original | 🔴 **NEW** — `build_top_liquidity_universe()` runs ONCE in `load_prices()` (data_loader.py:572), returning a static ticker list used for ALL backtest dates. This re-introduces survivorship via filter. | **ADD TO TIER 0**. Move inside rebalance loop. |
| **NEW BUG 11** (cache-key collision, §25) | Mentioned but not prioritized | 🔴 **PROMOTED** — If current baseline benefits from cached features with different deny-list hash, all numbers are invalid. Must audit before any Tier 1 work. | **ADD TO TIER 0**. Run with `DENY_LIST_HASH_SUFFIX=$(uuidgen)` once, lock baseline. |
| **NEW BUG 12** (CatBoost promised but not wired) | Found by T3.2 audit | 🟠 **NEW** — model.py:1300 docstring claims "LGBM + CatBoost + Ridge" but only LGBM + XGB + Ridge implemented. Lying docstring. | Fix when implementing T3.2. |

## A.2 Estimate Corrections

| Claim | Audit adjustment |
|---|---|
| Tier 1 combined **+200-800 bps** | Too wide; additive math is wrong. Realistic: **+150-400 bps CAGR / +0.15-0.35 Sharpe** after non-linear stacking penalties. Fixed1→Fixed2 empirical delta was only +138 bps with multiple fixes combined. |
| T1.6 SRAF "+130 bps" | Understates Fixed1 delta (+407 bps vs no_sraf) but that includes other fixes. SRAF in isolation likely **+30-100 bps**. Test alone. |
| T1.7 seed-bag "+0.05-0.15 Sharpe" | Math holds (~1/√5 variance reduction). Numerai claim is training-folklore, not literature. Keep but lower confidence. |
| Aggressive target **25-30% CAGR / 0.95-1.2 Sharpe** | **Unrealistic.** Requires all Tier 1-3 to hit published IR; historical implementation gap is 40-50%. Hard-cap stretch target at **22-24% backtest**. Live (post McLean-Pontiff): **14-17% realistic.** |
| McLean-Pontiff 35% haircut flat | Too coarse. Apply **25% to EDGAR/price factors, 40% to ML-learned signals, 20% to SRAF/alt data**, weighted ≈ 30%. Sharpe decays slower than return: **15-20% Sharpe haircut**, not 30%. |
| PBO < 0.3 threshold | **Too strict.** Consensus is 0.3-0.5 acceptable; >0.5 is alarm. With 52 folds × ~500 trials, realistic PBO is 0.4-0.6. Flag as "overfitting-prone" not "validates." |

## A.3 Tier 2 Feature Corrections

| Feature | Correction |
|---|---|
| **T2.1 EarningsStreak** | ❌ **DUPLICATE** — Already implemented at alt_features.py:684 as `earnings_beat_streak_signal`. Gating: `ENABLE_PHASE_D_SIGNALS=1` (default ON). Investigate why it's not earning importance, not "add new". |
| **T2.5 QMJ Safety** | Was listed partially-blocked; audit confirms **90.2% of tickers have ≥20 quarters** of EPS history. **Fully buildable in 1 day.** Upgrade priority. |
| **T2.13 Earnings-call Q&A FinBERT** | ❌ **DATA BLOCKED** — edgar_text cache has zero 8-K files (only 10-K, 10-Q). Requires new 8-K Exhibit 99 scraper (3-5 days). Drop from Tier 2 to Tier 5. |
| **T2.14 FinBERT on 10-K Item 7** | Feasible: have 24K 10-Ks, GPU (RTX 4070 Super + CUDA 12.4), torch 2.6. Need `pip install transformers==4.40`. Batch inference ~13-20 hours on GPU for backfill. Keep in Tier 2. |
| **T2.7 SUE** | Blocked only on T1.9 (un-clip surprise). Not blocked on Phase D data (which is fine per A.1). |

## A.4 Missing Items — Added

| New Item | Reason |
|---|---|
| **T1.11 Earnings blackout filter (±2d)** | Avoids selection bias on announcement day. One-line feature mask. Must pair with PEAD signal (T1.10). |
| **T1.12 Ledoit-Wolf Ridge shrinkage = 0.3** | 5-min code change at stage-2 meta-learner. Typical +20-40 bps on noisy panels. Do in Week 1. |
| **T1.13 HIFO cost-basis routing** | Zero-cost tax optimization. Already have tax_aware.py infrastructure. +30-60 bps after-tax before TLH. |
| **T2.18 Simple SPY 12-mo momentum macro overlay** | If SPY 12m return > 0 → 100% equity; else → 50% equity + 50% cash (no short). Captures 80%+ of 2008/2020 DD reduction. +0.1-0.2 Sharpe; ~1 day to implement. |
| **T3.11 Adaptive retraining on volatility regime** | Trigger retrain when rolling 20d rvol crosses 1.5× median. Regime shifts precede vol spikes. ~2 days. |
| **T4.9 Reserve 2026-01 to 2026-04 as sealed OOS** | **Critical** — do this NOW. Do not touch for Tier 2-3 validation. Hold for post-everything final test. If strategy passes only IS, OOS proves/disproves. |

## A.5 Execution Order — Corrected Sequencing

Original plan bundled 8 one-liners into Day 1. Audit flagged this as experimental-isolation violation. **Revised (strictly sequential Week 1)**:

**Pre-work (Day 0)**:
- Snapshot current baseline via `snapshot_baseline.py --label audit_baseline_20260417`
- Run BUG 11 isolation: `DENY_LIST_HASH_SUFFIX=$(uuidgen) python run_strategy.py --no-cache`. Compare tearsheet to snapshot. If >100 bps different, cache-key collision is real and must be fixed first.
- **Seal 2026-01 to 2026-04 as OOS**. Move / flag that date range in data_loader.

**Day 1**: T1.1 embargo fix **alone**. Measure Δ CAGR/Sharpe. (Critical moment: does baseline regress?)

**Day 2**: T1.2 — run `EMPTY_DENY_LIST=1` first. If <200 bps delta, **don't** do full clustered permutation replacement; the deny-list is approximately correct. If >200 bps, proceed with clustered permutation (~2 days).

**Day 3**: Data fixes in parallel — BUG 5 (VIX9D), BUG 6 (v2raw pickle), BUG 10 (PIT liquidity filter), BUG 11 (if not already fixed Day 0). Can run data fetches in background.

**Day 4**: T1.3 (fw=10), T1.9 (un-clip surprise), T1.10 (shorten carry), T1.11 (earnings blackout), T1.12 (Ledoit-Wolf), T1.13 (HIFO). **Orthogonal batches OK** since they touch different subsystems. Run as 2 groups with measurement in between.

**Day 5**: T1.5 (drop leadlag), T1.6 (SRAF fix), T1.7 (seed-bag=5), T1.8 (hl=1260). Measure combined lift. Lock honest baseline.

**Week 2-3**: Tier 2 features, sequenced by audit-cleared priority:
1. T2.9 (BSC vol scaling) — 1 day, data ready
2. T2.7 (SUE proper) — 1 day post T1.9
3. T2.10 (intermediate momentum) — 1 day, no data needed
4. T2.2 (ShareRepurchase / NetEquityFinance) — 1 day, have buybacks/issuance
5. T2.5 (QMJ Safety) — 1 day, 90% coverage confirmed
6. T2.6 (EBITDA/EV + payout) — 1 day
7. T2.3 (iHML) — 2 days (need SG&A pipeline extension)
8. T2.11 (target-price momentum) — 2 days (API validation)
9. T2.18 (SPY 12-mo macro overlay) — 1 day

**Week 4**: Tier 3 architecture (T3.6 → T3.2 → T3.8 → T3.10 → T3.5 → T3.3+T3.4 pair → T3.1). Start with 3 quick wins (T3.6, T3.2, T3.8).

**Week 5-6**: Tier 4 validation. Run PBO, honest DSR, SPA. Apply McLean-Pontiff haircut to all reports.

## A.6 Target Outcomes — Honest Re-baseline

| Scenario | Backtest CAGR | Live CAGR (after haircut) | Sharpe | Max DD |
|---|---|---|---|---|
| Post-Tier-0 (bug fixes only) | **16-19%** (likely drop from 18.81% is honest removal of leakage) | 12-15% | 0.55-0.70 | -55% to -45% |
| Tier 0 + Tier 1 | **19-22%** | 14-17% | 0.70-0.85 | -45% to -40% |
| Tier 0-3 hitting 60% of published IR | **22-26%** | 16-19% | 0.85-1.05 | -35% to -30% |
| Stretch (all Tier hits at 80%) | **25-28%** | 18-21% | 1.0-1.15 | -30% |

**Real live target after McLean-Pontiff + slippage + taxes**: **16-19% CAGR** on $15-50K AUM, **2-5% real alpha vs SPY**, Sharpe **0.75-0.95**.

## A.7 Additional Risks — Expanded

1. **Survivorship in feature engineering** (not just universe): EDGAR text reads "as of today" can contain 2025 data in 2020 rows. Audit `shift()` on all 10-K-based features.
2. **Regime shift 2027+**: HMM fits on 2013-2026 (bull-heavy). If stagflation / persistent drawdown, correlations flip. Budget ≥-40% DD, not -30%.
3. **Vendor risk**: EODHD free tier may sunset. Tradier API may break (already partial Phase 1 failure). No sourced alternative.
4. **Execution risk at scale**: At $15-50K AUM slippage is <1 bp; at $5M it's 15-20 bps. Scale gradually.
5. **Tax law risk**: ST 32.8% / LT 18.8% could change post-2026 elections. TLH assumes current rates.
6. **Model.py refactor introduces bugs**: 3500 lines of walk-forward logic; any Tier 2+ change must include regression tests against frozen snapshot.

---

*Appendix A compiled from 10 audit agents. Supersedes main-body claims where they conflict. When in doubt, trust A.X over the original.*
