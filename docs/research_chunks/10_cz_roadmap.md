# C&Z Signals — Roadmap: What We Have, What's Missing, What's Next

Reference: `results/_cz_research/cz_signal_ic.csv` (209 signals scored), `cz_overlap_map.csv` (28 overlaps), `src/cz_signals.py` (10 implementations). Universe: monthly IC over 143 months on our live US equity panel.

### A. Currently Built — 10 signals in `src/cz_signals.py`

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

### B. Already Covered via Equivalents (top ~10 of 28 in `cz_overlap_map.csv`)

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

### C. MISSING with IC_IR > 0.10 — Build Candidates (ranked)

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

### D. Test Discipline

- **Cache-key fix in `src/model.py`**: `EXTRA_DENY_FEATURES` is now folded into the panel-hash, so per-signal isolation A/B tests no longer hit a stale cache. Earlier sweeps were silently re-using identical panels.
- **CLI flags**: `--cz-only=<csv>` (allowlist) and `--cz-exclude=<csv>` (denylist) on `run_strategy.py` for proper per-signal A/B. Both flags participate in the cache hash.
- **Test harness**: `run_signal_test_suite.py` runs each candidate signal in isolation against the locked baseline, asserts cache-hash uniqueness across runs, and emits `results/_signal_test/<signal>__metrics.csv`.

### E. Known Gotchas

- **Sibling-feature leakage**: deny-lists must include sibling families. The "net_payout_only" iso test leaked `dividend_yield` because it shares the EDGAR `raw_dividends_paid` panel — exclude entire family, not just the named feature.
- **McLean-Pontiff (2016) decay**: published anomalies lose ~35% of in-sample IC out-of-sample. C&Z IC_IR figures are in-sample on their universe; expect ~0.65× on our 143-month panel.
- **Multiple-testing**: 209 signals × 7d-fwd → naive p<0.05 yields ~10 false positives. Require **IC_IR > 0.15** on our panel before promoting to production; signals in the 0.10–0.15 band ship only if they survive a 5-fold time-series CV with stable sign.
