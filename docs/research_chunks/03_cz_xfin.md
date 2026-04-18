## XFIN — External Financing

### Origin

The composite external financing measure was introduced by Bradshaw, Richardson & Sloan (2006), "The Relation Between Corporate Financing Activities, Analysts' Forecasts and Stock Returns," *Journal of Accounting and Economics* 42(1-2), 53-85. The authors aggregate equity and debt issuance/retirement into a single net-financing variable and document that its statistical association with future returns is stronger than that of any individual financing channel studied previously. Pontiff & Woodgate (2008), "Share Issuance and Cross-Sectional Returns," *Journal of Finance* 63(2), 921-945, corroborate the equity-only leg, showing that share issuance subsumes size, book-to-market, and momentum in post-1970 Fama-MacBeth regressions.

### Construction

XFIN is computed as a flow-to-stock ratio:

```
XFIN_t = (sstk + dltis - dvc - prstkc - dltr) / total_assets_t
```

where `sstk` is equity issuance, `dltis` long-term debt issuance, `dvc` cash dividends, `prstkc` equity repurchases, and `dltr` long-term debt repayment (Compustat tags). The numerator captures net cash raised externally over the trailing fiscal period; scaling by total assets yields a unit-free intensity measure comparable across firm size.

### Sign and Economic Interpretation

The empirical sign is **negative** — high XFIN predicts low subsequent returns. The dominant explanation is market timing: managers issue equity and debt when their securities are overvalued and retire them when undervalued (Baker-Wurgler-style timing combined with Stein 1996 catering). Bradshaw et al. (2006) show analyst over-optimism rises with XFIN, supporting a misvaluation rather than risk-based interpretation.

### Implementation and IC

Built in `src/cz_signals.py:build_xfin_signal()` (lines 282-309). The function pulls the five raw EDGAR fields, fills missing components with zero, ratios to total assets with a `(-2, 2)` winsorization, then negates the cross-sectional rank so that long-leg exposure is consistent with the documented sign. A 45-day filing lag is enforced upstream in the EDGAR loader for point-in-time safety. Per `cz_signal_ic.csv`, the signal achieves IC_IR = 0.181 with 40% panel coverage — among the strongest of the C&Z fundamental novelties.

### Relationship to Sister Signals

NetEquityFinance, defined as (issuance - repurchase) / market cap, is the equity-only sub-component normalized by market value rather than book assets; it is mathematically the negation of NetPayoutYield. We obtain that exposure through `build_net_payout_yield_signal()` and therefore omit a standalone NetEquityFinance build to avoid collinear factor loadings.

Sources:
- [Bradshaw, Richardson & Sloan (2006) — IDEAS/RePEc](https://ideas.repec.org/a/eee/jaecon/v42y2006i1-2p53-85.html)
- [Bradshaw, Richardson & Sloan (2006) — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=904226)
- [Pontiff & Woodgate (2008) — Journal of Finance / Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01335.x)
- [Pontiff & Woodgate (2008) — author PDF](https://www2.bc.edu/jeffrey-pontiff/Documents/11_pontiff-woodgate.pdf)
