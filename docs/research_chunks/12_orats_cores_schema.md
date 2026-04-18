# Orats `/cores` 340-Column Schema Reference

The Orats `hist/cores` endpoint returns a single end-of-day record per `(ticker, tradeDate)` containing 340 derived option-surface fields. This is the densest of the Orats historical endpoints and is the primary feed for our options-signal stack. Field names below are verified against the live `/datav2/hist/cores` payload (snapshot 2026-04-15) and against the parquet sample `data/cache/options/orats_raw/cores_AAPL_2013-01-01_2026-04-15.parquet`. Inventory source: `results/validation/orats_cores_field_inventory.csv`.

### A. Identifiers / metadata (~10)

| Field | Notes |
|---|---|
| `ticker`, `tradeDate`, `updatedAt` | primary key + freshness stamp |
| `assetType`, `sector`, `sectorName` | classification |
| `pxCls`, `priorCls`, `pxAtmIv`, `mktCap` | spot reference + size |

### B. Constant-maturity ATM IV term structure (~10)

| Field | Maturity |
|---|---|
| `iv10d`, `iv20d`, `iv30d`, `iv60d`, `iv90d`, `iv6m`, `iv1yr` | 10d → 1y |
| `atmIvM1`, `atmIvM2`, `atmIvM3`, `atmIvM4` | first 4 listed monthly expiries |
| `atmFitIvM1..4`, `atmFcstIvM1..4`, `dtExM1..4` | fit/forecast variants + DTE |

### C. Smile / delta-bucketed IV matrix (~80)

Cross of 5 delta strikes (`dlt5`, `dlt25`, `dlt50`, `dlt75`, `dlt95`) × 7 maturities (`Iv10d`, `Iv20d`, `Iv30d`, `Iv60d`, `Iv90d`, `Iv6m`, `Iv1y`). Convention: `dlt5` = 5-delta call (deep OTM upside), `dlt95` = 95-delta call ≡ 5-delta put (crash strike). Each cell is also republished in the earnings-stripped form (`exErnDltXIvY`), doubling the bucket count.

### D. Slope / skew metrics (~10)

| Field | Meaning |
|---|---|
| `slope`, `slopeInf`, `slopeFcst`, `slopeFcstInf` | smile slope (current / asymptotic / forecast) |
| `deriv`, `derivInf`, `derivFcst`, `derivFcstInf` | smile curvature (2nd derivative) |
| `slopepctile`, `slopeavg1m`, `slopeavg1y`, `slopeStdv1y` | rank/normalisation context |

### E. Volume / Open Interest (~12)

`cVolu`, `pVolu`, `cOi`, `pOi`, `oi`, `stkVolu`, `avgOptVolu20d`, `cAddPrem`, `pAddPrem`, plus implicit put/call ratios derived downstream.

### F. Realized volatility (~30)

| Family | Windows |
|---|---|
| `orHv*` (open-to-close) | 1d, 5d, 10d, 20d, 60d, 90d, 120d, 252d, 500d, 1000d |
| `clsHv*` (close-to-close) | 5d → 1000d (same windows) |
| `orHvXern*`, `clsHvXern*` | earnings-stripped variants |

### G. Forward IV curve / forecasts (~12)

`fwd30_20`, `fwd60_30`, `fwd90_60`, `fwd180_90`, `fwd90_30` plus `f`-prefixed forecast and `fb`-prefixed beta-adjusted versions. Quality fields: `fcstR2`, `fcstR2Imp`, `confidence`, `error`, `orFcst20d`, `orIvFcst20d`, `orFcstInf`.

### H. Earnings effects (~30)

| Family | Fields |
|---|---|
| Calendar | `nextErn`, `lastErn`, `daysToNextErn`, `wksNextErn`, `ernMnth`, `nextErnTod`, `lastErnTod` |
| Historical 12-quarter array | `ernDate1..12`, `ernMv1..12`, `ernStraPct1..12`, `ernEffct1..12` |
| Implied move | `impErnMv`, `impErnMv90d`, `impMth2ErnMv`, `impliedEarningsMove`, `absAvgErnMv`, `ernMvStdv`, `fcstErnEffct`, `impliedIee`, `impliedEe`, `ivEarnReturn` |

### I. Earnings-stripped IV (~12)

`exErnIv10d`, `exErnIv20d`, `exErnIv30d`, `exErnIv60d`, `exErnIv90d`, `exErnIv6m`, `exErnIv1yr`, plus `fexErn*` forward and `ffexErn*` filtered-forward variants — the cleanest cross-period vol comparison surface.

### J. Dividend / borrow (~10)

`divFreq`, `divYield`, `divGrwth`, `divDate`, `divAmt`, `nextDiv`, `impliedNextDiv`, `annActDiv`, `annIdiv`, `borrow30`, `borrow2yr`. The `annIdiv` − `annActDiv` gap is a dividend-surprise proxy; `borrow30` flags hard-to-borrow short squeezes.

### K. Sector / ETF reference (~10)

`bestEtf`, `etfIncl`, `correlSpy1m/1y`, `correlEtf1m/1y`, `beta1m`, `beta1y`, `ivSpyRatio` (+1m/1y avg + stdv), `ivEtfRatio` (+1m/1y avg + stdv), `etfSlopeRatio`, `etfIvHvXernRatio`.

### L. Misc / aggregates (~30)

`px1kGam` (gamma per $1k notional), `volOfVol`, `volOfIvol`, `iv200Ma`, `contango`, `iRate5wk`, `iRateLt`, `mktWidthVol`, `rip`, `tkOver`, `hiHedge`/`loHedge`, `ivPctile1m/1y/Spy/Etf`, `ivStdvMean`/`Stdv1y`, `straPxM1/M2`, `smoothStraPxM1/M2`, `fcstStraPxM1/M2`, `loStrikeM1/M2`, `hiStrikeM1/M2`, `fairVol90d`, `fairXieeVol90d`, `fairMth2XieeVol90d`, `impliedR2`.

### Our extraction: 27-field core slice

`src/orats_loader.py` pivots only the fields needed for the C&Z options signal stack. Selected (priority): ATM term structure (`iv30d`, `iv60d`, `iv90d`), the 30d delta smile (`dlt5Iv30d`, `dlt25Iv30d`, `dlt75Iv30d`, `dlt95Iv30d`), earnings-stripped 30d skew (`exErnIv30d`, `exErnDlt25Iv30d`, `exErnDlt75Iv30d`, `exErnDlt95Iv30d`), `slope`, call/put `cVolu`/`pVolu`/`cOi`/`pOi`, `borrow30`, `annIdiv`, `annActDiv`, plus realized-vol anchors (`orHv20d`, `orHv60d`), forward IV decomposition (`fwd30_20`, `fwd60_30`, `fwd90_30`), and earnings calendar (`daysToNextErn`, `impliedEarningsMove`). One derived column is built downstream: `cp_vol_spread_proxy = dlt25Iv30d − dlt75Iv30d` (call-skew proxy for `dCPVolSpread`).

That is ~8% of the 340-column surface; the other 92% (full smile-maturity matrix, 12-quarter earnings history, second-order Greeks, sector-relative ratios) remains parquet-resident and addressable for future signal research without re-pulling the API.
