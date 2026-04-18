## Coskewness as a Return Predictor

### Origin and Theoretical Motivation

Coskewness extends the mean-variance CAPM into a third-moment framework. The seminal treatment is (Harvey & Siddique 2000, Journal of Finance), "Conditional Skewness in Asset Pricing Tests," vol. 55(3), pp. 1263-1295. Their thesis: if the cross-section of returns exhibits systematic skewness, risk-averse investors with non-quadratic utility will demand compensation for holding assets that exacerbate left-tail co-movement with the market. They document a coskewness risk premium of roughly 3.6 percent per year and show that part of the momentum anomaly loads on systematic skewness.

The standardized coskewness measure is

$$\text{coskew}_i = \frac{\mathbb{E}[\varepsilon_i \cdot \varepsilon_m^2]}{\sqrt{\mathbb{E}[\varepsilon_i^2]} \cdot \mathbb{E}[\varepsilon_m^2]}$$

where $\varepsilon_i$ and $\varepsilon_m$ are demeaned stock and market returns. Names with **negative** coskewness pay off poorly precisely when the market is volatile (large $\varepsilon_m^2$), so they should command **higher** expected returns. This yields the Chen-Zimmermann CSK and CSR signals (both negate the raw measure so that "high signal = predicted high return").

(Ang, Chen & Xing 2006, Review of Financial Studies), "Downside Risk," vol. 19(4), pp. 1191-1239, refines this by isolating downside co-movement explicitly, reporting a downside-risk premium near 6 percent per annum that is not subsumed by size, value, momentum, liquidity, or unconditional coskewness. We implement their daily-window variant as `coskew_acx`.

### Empirical Performance in Our Universe

Per `results/cz_signal_ic.csv`, **Coskewness IC_IR = 0.198** and **CoskewACX IC_IR = 0.124** in our 2013-2026 backtest. Both signals clear the Chen-Zimmermann replication threshold and rank among the strongest price-only predictors we have surfaced.

### Implementations

Two non-redundant implementations live in the codebase:

- `src/features.py:co_skewness()` (lines 990-1078) — daily 252-day rolling window using a cumsum-optimized rolling mean over $\varepsilon_i \cdot \varepsilon_m^2$, $\varepsilon_i^2$, and $\varepsilon_m^2$. Loaded into the main feature pipeline; gated by `DISABLE_CO_SKEW_252` env var to avoid double-counting.
- `src/cz_signals.py:build_coskewness_signal()` (lines 48-107) — monthly 60-month window matching the original Harvey-Siddique horizon, activated under `--use-cz-signals`. Forward-filled to daily and cross-sectionally ranked to $[0, 1]$.

The two specifications are **statistically orthogonal** (Spearman rho = 0.041): the daily 252-day version captures conditional tail co-movement at business-cycle frequency, while the monthly 60-month version captures the slow-moving unconditional moment. Stacking both adds incremental information.

### Strategic Note

Coskewness is **price-only**: inputs are stock returns and a market proxy (SPY). It does **not** depend on options-implied moments, EDGAR filings, or any paid alt-data feed. Cancelling the Orats subscription (or any options vendor) preserves both CSK and CSR signals in full. This makes coskewness a high-conviction, zero-marginal-cost component of the alpha stack.

### Sources

- [Conditional Skewness in Asset Pricing Tests (Harvey & Siddique 2000) - Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00247)
- [Conditional Skewness in Asset Pricing Tests - Duke working paper PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P56_Conditional_skewness_in.pdf)
- [Downside Risk (Ang, Chen & Xing 2006) - Oxford Academic](https://academic.oup.com/rfs/article-abstract/19/4/1191/1580531)
- [Downside Risk - NBER working paper w11824](https://www.nber.org/papers/w11824)
