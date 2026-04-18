# Phase 5 Production Summary

- Generated: 20260411_095752
- Mode: DRY RUN (re-processed existing results)
- Config phase: 1.8d (locked)

## Headline Metrics

- **CAGR**: 14.24%
- **SPY CAGR**: 13.99%
- **Alpha vs SPY**: +0.25%
- **Annualized Volatility**: 20.48%
- **Sharpe Ratio**: 0.655
- **Sortino Ratio**: 0.909
- **Calmar Ratio**: 0.389
- **Max Drawdown**: -36.57%
- **Beta to Market**: 0.904
- **Probabilistic Sharpe (PSR)**: 0.992
- **Deflated Sharpe (DSR)**: 0.919
- **DSR n_trials**: 1

## Multi-Trial Deflated Sharpe (Bailey & Lopez de Prado 2014)

- Trials assumed across all phases: **15**
- DSR @ n_trials=1 (reported): **0.919**
- DSR @ n_trials=15 (corrected): **0.7385**
- Sample length T = 3411 days

## Fama-French 6-Factor Regression

| Factor | Coef | t-stat | p-value |
|---|---:|---:|---:|
| Alpha | 0.0001 | 1.03 | 0.3030 |
| Mkt-RF | 0.8371 | 8.51 | 0.0000 |
| SMB | 0.6765 | 17.53 | 0.0000 |
| HML | -0.1392 | -2.36 | 0.0185 |
| RMW | -0.1223 | -2.56 | 0.0104 |
| CMA | 0.0776 | 1.60 | 0.1093 |
| Mom | 0.2879 | 6.26 | 0.0000 |

## OOS vs In-Sample

- **In-Sample**: Sharpe=0.554, Ann Ret=11.92%, MaxDD=-36.57%, Beta=0.886
- **Out-of-Sample**: Sharpe=1.181, Ann Ret=26.75%, MaxDD=-20.31%, Beta=1.006

## Stress Tests

- **COVID crash (2020)** [2020-02-19 -> 2020-03-23]: ret=-25.4%, dd=-26.3%, vs bench=+8.0%
- **Post-COVID rally (2020)** [2020-03-23 -> 2020-08-31]: ret=43.2%, dd=-6.2%, vs bench=-10.1%
- **Rate hike cycle (2022)** [2022-01-01 -> 2022-12-31]: ret=-20.9%, dd=-27.7%, vs bench=-2.8%
- **GFC peak-to-trough** [ -> ]: ret=, dd=, vs bench=
- **Dot-com bust** [ -> ]: ret=, dd=, vs bench=
- **2018 Q4 selloff** [2018-09-20 -> 2018-12-24]: ret=-22.0%, dd=-23.0%, vs bench=-3.3%
- **COVID recovery rally** [2020-04-01 -> 2021-01-31]: ret=82.6%, dd=-8.0%, vs bench=+37.2%
- **2022 bear market** [2022-01-03 -> 2022-10-12]: ret=-24.5%, dd=-27.7%, vs bench=-0.4%
- **SVB/regional bank crisis** [2023-03-08 -> 2023-03-24]: ret=-8.4%, dd=-8.9%, vs bench=-8.1%
- **2025 tariff shock** [2025-04-01 -> 2025-04-10]: ret=-5.1%, dd=-8.9%, vs bench=+1.1%

## Regime Performance

- **bear_calm** (10.0% of time): Sharpe=-2.18, ann_ret=-42.8%, active=-11.7%
- **bear_volatile** (5.0% of time): Sharpe=-0.42, ann_ret=-12.2%, active=2.6%
- **bull_calm** (74.3% of time): Sharpe=0.97, ann_ret=20.9%, active=-0.3%
- **bull_volatile** (7.6% of time): Sharpe=2.35, ann_ret=68.2%, active=5.1%
- **unknown** (3.1% of time): Sharpe=3.55, ann_ret=86.8%, active=36.0%

## Bootstrap 90% Confidence Intervals

- **Sharpe Ratio**: observed=0.401, p5=-0.037, p95=0.833, P(>0)=93.8%
- **Sortino Ratio**: observed=0.545, p5=-0.048, p95=1.162, P(>0)=93.8%
- **Annualized Return**: observed=0.07, p5=0.004, p95=0.138, P(>0)=95.4%

## Production Readiness Checklist

| # | Check | Pass | Detail |
|---|---|:---:|---|
| 1 | CAGR > SPY | Y | 14.24% vs SPY 13.99% |
| 2 | Sharpe > 0.60 | Y | 0.655 |
| 3 | DSR (n=1) > 0.95 | N | 0.919 |
| 4 | DSR (n=15 corrected) > 0.80 | N | 0.739 |
| 5 | Alpha p-value < 0.10 | N | p=0.3030 |
| 6 | MaxDD > -40% | Y | -36.57% |
| 7 | OOS Sharpe > Full-sample Sharpe | Y | IS=0.554 OOS=1.181 Full=0.655 |
| 8 | FF6 RMW loading > -0.20 | Y | -0.122 |

**Score: 5/8 (62%)**

**Verdict: CAUTION — marginal, review failures**

## Reproduction

```
"C:\Users\Alex Hines\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\Alex Hines\OneDrive\Documents\Personal Projects\Personal-Trading-Portfolio-Project\run_strategy.py" --start 2013-01-01 --forward-window 21 --signal-smooth-halflife 10.0 --quality-percentile 0.0 --max-stock-vol 1.0 --min-adv-for-selection 0 --min-holding-overlap 0.5 --n-positions 50 --ml-blend 0.3 --vol-target 0.3 --max-leverage 1.8 --vol-ceiling 0.4 --quality-tilt 0.35 --cash-in-bear 0.3 --no-cs-zscore-all --sample-weight-halflife-years 4.5
```
