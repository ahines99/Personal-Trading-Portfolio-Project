# Orats Non-/cores Endpoints — Unique Value Audit

Reference for the five auxiliary Orats endpoints (`/summaries`, `/ivrank`, `/dailies`, `/hvs`, `/earnings`) and what each contributes beyond the primary `/cores` surface. Source facts: `results/validation/orats_other_endpoints_inventory.csv` (187 fields audited).

### /summaries (128 cols)

Despite carrying fewer columns than `/cores`, `/summaries` exposes **18 fields not present in /cores**. The bulk are redundant (IVs, smile slices, forward variances all derivable from a chain), but a handful are genuinely unique and not reproducible from a live Tradier chain:

- **`impliedMove`** — single-number expected percentage move to next expiry, calibrated from the full vol surface. Not derivable cheaply.
- **`impliedEarningsMove`** — earnings-event-specific implied jump.
- **`exErnIv{10d,20d,30d,60d,90d,6m,1y}`** — earnings-stripped ATM IV per maturity. Critical for separating event vol from baseline vol on names with binary catalysts.
- **`rDrv30/2y`, `rSlp30/2y`, `rVol30/2y`** — Orats' proprietary residual surface diagnostics; not reconstructible.

**Recommendation:** keep cached snapshot for `impliedMove`, `impliedEarningsMove`, and the `exErnIv*` curve. Drop the rest as redundant with `/cores`.

### /ivrank (8 cols)

Five unique fields: `iv`, `ivRank1m`, `ivPct1m`, `ivRank1y`, `ivPct1y`. These are 21-day and 252-day rolling IV percentile ranks. Mathematically derivable from a `/cores` `iv30d` history but expensive to compute daily across 3000 names.

**Recommendation:** retain historical cache; recompute fresh from Tradier `iv30d` history going forward (one-pass `pd.Series.rank(pct=True).rolling(252)`).

### /dailies (13 cols)

Adjusted and unadjusted OHLCV plus stock volume. Every field is replaceable from any equities vendor — we already pay for EODHD daily bars with corporate-action adjustments.

**Recommendation:** **DROP**. Zero unique value.

### /hvs (48 cols)

Realized volatility across **11 horizons (1d → 1000d) × 3 variants** (open-to-close `orHv*`, close-to-close `clsHv*`, ex-earnings `*Xern*`). Standard textbook HV — fully derivable from daily prices via `np.log(close).diff().rolling(N).std() * np.sqrt(252)`. Ex-earnings variants need only an earnings-date mask we already build.

**Recommendation:** **DROP**, compute in-house.

### /earnings (4 cols)

Sparse (~150 rows per ticker over 13 years). The only field with marginal unique value is **`anncTod`** (announcement time as `HHMM`, e.g. `1630` for after-close). EODHD earnings calendar covers the same dates; before/after-market flag can be inferred from EODHD's `before_after_market` column.

**Recommendation:** **DROP**.

### Consolidation Table

| Endpoint   | Cols | Unique Fields | Decision        | Replacement                        |
|------------|------|---------------|-----------------|------------------------------------|
| /summaries | 128  | 18            | Cache subset    | Keep `impliedMove`, `exErnIv*` only |
| /ivrank    | 8    | 5             | Cache + recompute | Tradier `iv30d` rolling rank       |
| /dailies   | 13   | 13 (trivial)  | **DROP**        | EODHD adjusted OHLCV               |
| /hvs       | 48   | 11 (trivial)  | **DROP**        | `np.log(close).diff().rolling(N)`  |
| /earnings  | 4    | 1             | **DROP**        | EODHD earnings calendar            |

**Net:** post-cancellation we operate on `/cores` (primary live surface) + a frozen `/summaries` cache for `impliedMove` and earnings-stripped IV curves. The other four endpoints contribute nothing that is both unique and worth the API spend.
