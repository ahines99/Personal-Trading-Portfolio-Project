### Bakshi-Kapadia-Madan (BKM) Implied Moments — Risk-Neutral Skewness and Kurtosis from the Full Smile

### Origin

Bakshi, Kapadia & Madan (2003), "Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options," *Review of Financial Studies* 16(1), 101-143, derive a model-free spanning result that recovers the entire risk-neutral distribution of returns from a cross-section of European option prices. The same machinery underlies the CBOE SKEW index. Whereas ATM-only or 25-delta-only volatility metrics collapse the smile to a single number, BKM reads the **full** OTM strike continuum and decomposes the implied distribution into its second, third, and fourth moments.

### Key Insight

A traded option's payoff is a Dirac function in strike space; an integral of OTM call and put prices weighted by appropriate strike-dependent kernels reconstructs the risk-neutral expectation of any twice-differentiable payoff. Applying this to powers of log-returns yields:

- Risk-neutral variance σ²_RN
- Risk-neutral skewness — typically **negative** for individual equities (downside fear premium)
- Risk-neutral kurtosis — typically **>3** (fat-tail premium for jump risk)

### Discretized Formulas (trapezoidal integration over OTM strikes)

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

### Empirical Findings

Conrad, Dittmar & Ghysels (2013), "Ex Ante Skewness and Expected Stock Returns," *Journal of Finance* 68(1), 85-124, demonstrate that risk-neutral skewness predicts the cross-section of stock returns: **less-negative (more right-shifted) skew forecasts higher subsequent returns**, even after controlling for systematic co-moments. Cremers & Driessen (2008) and the broader crash-risk-premium literature show that risk-neutral kurtosis carries an analogous tail-fear premium that mean-reverts.

### Our Implementation

`src/options_adapter/implied_moments.py:compute_bkm_moments()` implements full BKM with trapezoidal integration over a liquidity-filtered OTM smile:

- Liquidity filter: bid > 0 and relative spread ≤ 1.5
- `min_otm_per_side = 5` strikes (else NaN)
- Outputs: `implied_var_30d`, `implied_skew_30d`, `implied_kurt_30d`

Live AAPL validation:

- BKM/ATM variance ratio = 1.018 (consistent — smile correction modest for liquid names)
- `implied_skew = −1.297` (sharply negative — downside fear priced in)
- `implied_kurt = 9.51` (well above 3 — fat tails confirmed)

### Two New Signals

- `build_implied_skew_signal()` — sign **POSITIVE** per Conrad-Dittmar-Ghysels: less-negative skew → outperformance
- `build_implied_kurt_signal()` — sign **POSITIVE**: high kurtosis = priced crash fear that mean-reverts, paying the seller of tail insurance

### Tradier-Only Viability

Critically, BKM is **viable on the free Tradier feed** without paid Orats subscription: the chain endpoint exposes `greek_smv_vol` per contract, giving the full smile. BKM integrates that smile directly, so we recover risk-neutral skew and kurt without any historical-Orats dependency — train/serve consistent and zero marginal data cost.

Sources:
- [Bakshi, Kapadia & Madan (2003) RFS PDF](https://people.umass.edu/nkapadia/docs/Bakshi_Kapadia_Madan_2003_RFS.pdf)
- [Bakshi, Kapadia & Madan (2003) — Oxford Academic](https://academic.oup.com/rfs/article-abstract/16/1/101/1615098)
- [Conrad, Dittmar & Ghysels (2013) — Journal of Finance](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2012.01795.x)
- [Conrad, Dittmar & Ghysels (2013) — UNC Repository PDF](https://cdr.lib.unc.edu/downloads/pr76fc54k)
- [BKM Gram-Charlier extension — Review of Derivatives Research](https://link.springer.com/article/10.1007/s11147-022-09187-x)
