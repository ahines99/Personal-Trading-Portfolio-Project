## Variance Risk Premium (VRP) and IV-RV Spread

The variance risk premium (VRP) and the implied-realized volatility spread are two closely related, options-derived signals that monetize the systematic gap between risk-neutral and physical variance forecasts. Both rank among the highest-IC signals in our `cz_signal_ic.csv` and were the #2 and #3 most important features in the `iso_OPTIONS_baseline` run, while requiring only Tradier-sourced 30-day implied volatility (CMIV) plus standard price history.

### Variance Risk Premium (Bollerslev, Tauchen & Zhou, 2009)

Bollerslev, Tauchen & Zhou (2009, *RFS*) introduced the index-level VRP as the difference between option-implied variance and subsequently realized variance, demonstrating that the resulting wedge predicts a non-trivial fraction of post-1990 aggregate equity returns, with predictability concentrated at the quarterly horizon and dominating P/E, default spread, and consumption-wealth ratio benchmarks. Han & Zhou (2012) extended the construct cross-sectionally, sorting individual stocks on firm-level VRP and reporting long-short Sharpe ratios of roughly 0.7-1.0 with annualized spreads near 10%.

We compute VRP in annualized variance units as `iv30**2 - rv**2` (`src/options_signals.py:build_variance_premium_signal`). The economic sign is **negative for return prediction**: a high VRP indicates that variance is expensively priced, so variance sellers are over-compensated and the underlying tends to *underperform*. Our implementation flips the sign accordingly. In `cz_signal_ic.csv` the resulting `variance_premium` signal achieves IC_IR ≈ 0.10.

### IV-RV Spread (Bali & Hovakimian, 2009)

Bali & Hovakimian (2009, *Management Science*) work in volatility (level) rather than variance (squared) space, defining the realized-implied spread as `RV_30d - IV_30d`. Using portfolio sorts and Fama-MacBeth regressions, they show that while raw RV and IV levels are not priced, the spread between them is, capturing a volatility-risk component distinct from the call-put implied-vol spread that proxies jump risk. Stocks where realized has overshot implied earn higher subsequent returns, consistent with vol-sellers being on the wrong side and the equity being temporarily underpriced.

Our `build_rv_iv_spread_signal` implements `RV - IV` directly with **positive sign**, achieving IC_IR = 0.155 in `cz_signal_ic.csv`. Because both signals require only IV30 and trailing returns, they remain fully buildable on the Tradier-only stack going forward, with no Orats subscription required for live serving.

Sources:
- [Expected Stock Returns and Variance Risk Premia | Oxford Academic](https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787)
- [Bollerslev, Tauchen & Zhou (2009) RFS PDF](https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf)
- [Expected Stock Returns and Variance Risk Premia | SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=948309)
- [Volatility Spreads and Expected Stock Returns | SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1443848)
- [Volatility Spreads and Expected Stock Returns | Management Science](https://dl.acm.org/doi/10.1287/mnsc.1090.1063)
- [Volatility Spreads and Expected Stock Returns | IDEAS/RePEc](https://ideas.repec.org/a/inm/ormnsc/v55y2009i11p1797-1812.html)
