### Momentum Seasonality (MomSeason)

### Origin

The momentum seasonality signal originates with Heston and Sadka (2008), "Seasonality in the Cross-Section of Stock Returns," *Journal of Financial Economics* 87(2), 418–445. Their core empirical finding is striking: stocks that historically delivered high (or low) returns in a given calendar month tend to repeat that pattern in the same calendar month of subsequent years. The autocorrelation pattern is statistically significant at annual lags of 12, 24, 36, ..., and persists out to roughly 20 annual lags, superimposed on the well-known short-horizon momentum and long-horizon reversal patterns. Keloharju, Linnainmaa, and Nyberg (2016), "Return Seasonalities," *Journal of Finance*, generalize the result to non-calendar-month seasonalities (day-of-week, intra-month) and argue the effect reflects persistent firm-level expected-return components rather than risk-factor exposure.

### Economic Insight

If Apple has outperformed in April for the past several Aprils, the signal predicts April outperformance again — independent of size, industry, earnings, dividends, or fiscal year (Heston-Sadka show all of these as non-explanations). The mechanism is debated; candidates include seasonal liquidity, tax-driven trading, and seasonal cash-flow news. The effect is subtle because it operates per-month rather than per-day, which is likely why it remains less arbitraged than 12-1 momentum.

### Construction

Following the Heston-Sadka specification, we use the **years 6–10 lookback** window, deliberately skipping years 1–5 to orthogonalize the seasonal signal from conventional momentum and short-horizon reversal effects. For each (date *t*, ticker *i*):

$$\text{MomSeason}_{i,t} = \frac{1}{5} \sum_{k=6}^{10} r_{i, t - 12k \text{ months}}$$

where $r$ is the calendar-month return. The expected sign is **positive**: higher historical same-month return predicts higher current same-month return. We then apply a cross-sectional rank to [0, 1].

### Implementation

`src/cz_signals.py:build_mom_season_signal()` (lines 159–205) computes monthly compounded returns via `resample("ME").prod()`, then for each month-end date averages returns from years t−6 through t−10 in the same calendar month, requires at least 2 historical observations, and forward-fills to daily frequency (limit=25 days).

### Performance

Per `cz_signal_ic.csv`, MomSeason achieves **IC_IR = 0.146**, ranking among the strongest novel signals in our research panel. The data dependency is minimal: only the daily returns panel is required — no fundamentals, no options, no alternative data. This makes it a pure price-based signal with zero incremental data cost.

### Sources

- [Seasonality in the cross-section of stock returns (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X0700195X)
- [Heston & Sadka working paper (NYU Stern)](https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/063f-sadka.pdf)
- [SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=687022)
- [RePEc entry](https://ideas.repec.org/a/eee/jfinec/v87y2008i2p418-445.html)
