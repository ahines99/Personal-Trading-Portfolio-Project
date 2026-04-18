# Reproducing the Honest Baseline

Last verified: 2026-04-17. Reference tearsheet: [results/_repro_fixed2_v3/tearsheet.csv](results/_repro_fixed2_v3/tearsheet.csv)

## Honest baseline

**22.58% CAGR / 0.715 Sharpe / -54.76% MaxDD / +8.60% alpha vs SPY / Beta 1.491 / DSR 0.947 / PSR 0.995**

Reproduction command (clean rebuild — purge caches, then run with `EMPTY_DENY_LIST=1`):

```bash
rm -f data/cache/feature_panel_*.pkl data/cache/ml_predictions_*.pkl
EMPTY_DENY_LIST=1 python run_strategy.py \
  --skip-robustness \
  --use-finra-short-interest \
  --results-dir results/_repro_fixed2_v3
```

Expected runtime: ~3h 15m on Python 3.13, peak 19 GB RAM. Reproduces within ±20 bps.

**Cache-purge is required.** `EMPTY_DENY_LIST` is not currently part of the feature-panel cache hash ([src/model.py:238](src/model.py#L238)), so a stale cache from a master-default run will shadow the result. Purging `feature_panel_*.pkl` forces a fresh rebuild.

## Master default (18.81%) — why the gap

Running `python run_strategy.py --skip-robustness --use-finra-short-interest` without `EMPTY_DENY_LIST=1` applies a hardcoded 101-entry `_DENY_LIST` in [src/model.py:800](src/model.py#L800) that aggressively prunes features flagged as zero-importance in earlier runs plus several SNI momentum/vol/beta variants. This over-prunes legitimate alpha: **−377 bps CAGR, −0.093 Sharpe** vs the honest baseline.

The deny-list was introduced post-Fixed2 (2026-04-15) as a research-isolation tool and is too broad for production. Clear it via `EMPTY_DENY_LIST=1`.

## Environment controls

| Env var | Default | Effect |
|---|---|---|
| `EMPTY_DENY_LIST` | `"0"` | `"1"` clears the 101-entry deny-list. Required for the 22.58% baseline. |
| `ENABLE_PHASE_D_SIGNALS` | `"1"` | `"0"` disables 4 Phase D features (Δ ≈ −134 bps CAGR). |
| `EXTRA_DENY_FEATURES` | `""` | Comma-separated feature names to additionally exclude (isolation tests). |
| `DENY_LIST_HASH_SUFFIX` | `""` | Append to feature-panel hash to force cache isolation when doing A/B tests. |
| `DISABLE_CO_SKEW_252` | `"0"` | Memory saver; `"1"` disables daily coskewness (keeps C&Z monthly variant). |
| `DISABLE_DIVIDEND_YIELD` | `"0"` | Skip the redundant dividend_yield signal when C&Z `payout_yield` is on. |
| `DISABLE_CASH_BASED_OP_PROF` | `"0"` | Skip the redundant cash_based_op_prof signal when C&Z `operprof_rd` is on. |

Phase D features gated by `ENABLE_PHASE_D_SIGNALS`:
- `q_buyback_yield` — Grullon-Michaely buyback signal ([src/api_data.py:622](src/api_data.py#L622))
- `hmlint_signal` — Arnott-Harvey intangibles-adjusted value ([src/api_data.py:639](src/api_data.py#L639))
- `earnings_beat_streak_signal` — consecutive positive EPS surprises ([src/alt_features.py:660](src/alt_features.py#L660))
- `earnings_surprise_consistency_signal` — rolling 8-quarter CV ([src/alt_features.py:688](src/alt_features.py#L688))

## Known gap to Fixed2 target (−309 bps)

The Fixed2 artifact (2026-04-15) recorded 25.67% CAGR on uncommitted code; today we reach 22.58%. The remaining gap likely comes from:

- **Feature-column-order non-determinism**: dict iterations at [src/model.py:359](src/model.py#L359) and [src/alt_features.py:173](src/alt_features.py#L173) lack `sorted()`. LightGBM split decisions change if column order shifts between runs; the feature-panel hash does not account for this.
- **XGBoost version drift**: [requirements.txt](requirements.txt) pins `xgboost<3.0.0`, but 3.2.0 is installed. 3.x has breaking-change algorithm updates. Candidate for the largest portion of the gap.
- **MLP unseeded shuffle**: `torch.randperm` at [src/model.py:1937](src/model.py#L1937) has no explicit seed (~5-10 bps variance; small since `mlp_weight` is low).
- **Feature-panel delta**: the 2026-04-15 panel had mom_dbc/gld/hyg_12_1 + SRAF + alt_initial_claims + vol_regime + breadth (cluster of ~10) that today's panel lacks; today's panel has 18 `alt_q_*` + `beat_streak` + `surprise_consistency` + `hmlint` + `days_to_earnings` (cluster of ~22) that Fixed2 lacked. Net additions, but the subtractions matter.

## Disproven theories (do not repeat)

- ❌ **EODHD sentiment v1 lookahead bias** — v1 applied `shift(1)` in the consumer; v2 moved it to the loader. Both correct for their formats. Fixed2 ran on v2 cache (built 2026-04-06, untouched since). The v1 pickle exists on disk but is never read.
- ❌ **Fixed2 ran without EODHD fundamentals** — `load_eodhd_fundamentals()` in [src/api_data.py:944](src/api_data.py#L944) is commented out in current code and was commented out then too. Today's `q_*` features come from free SEC EDGAR via [src/alt_features.py](src/alt_features.py), not EODHD.

## Cache state assumptions

API caches (auto-rebuilt if missing):
- `data/cache/api/eodhd_sentiment_v2_2013-01-02_2026-02-27.pkl` — 22.4 MB, `shift(1)` applied at load
- `data/cache/api/eodhd_fundamentals_v2.pkl` — 638 MB (loader currently commented out; cache retained for future use)
- `data/cache/api/eodhd_dividends_2013-01-01_v2.pkl` — 15.3 MB

Purge feature-panel and ML-prediction caches (`data/cache/feature_panel_*.pkl`, `data/cache/ml_predictions_*.pkl`) before the baseline run to guarantee reproducibility.

## Test harness delta policy

When evaluating new signals, compare **delta vs this 22.58% baseline** (with `EMPTY_DENY_LIST=1`), not absolute performance. A signal is additive iff **CAGR delta ≥ +25 bps AND Sharpe delta ≥ +0.02** on a clean rebuild (purged feature/ML caches). Each isolation test must produce a distinct feature-panel hash via `DENY_LIST_HASH_SUFFIX` — several prior `iso_CZ_*` runs produced identical hashes across 6 subset configurations, which is a cache-key bug that must be fixed before relying on isolation results.
