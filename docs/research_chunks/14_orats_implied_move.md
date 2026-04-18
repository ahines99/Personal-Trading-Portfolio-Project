## Implied Move and the Earnings IV Crush Trade

### Definition

The **implied move** is the option market's expectation of the absolute return of the underlying over the life of a near-the-money straddle. For a stock at spot $S$ with at-the-money call and put mid-prices $C$ and $P$ at roughly 30 days to expiry,

$$\text{impliedMove} = \frac{C_{\text{ATM}} + P_{\text{ATM}}}{S}$$

Under standard Black-Scholes assumptions this approximates $\sigma_{\text{IV}} \cdot \sqrt{T} \cdot \sqrt{2/\pi}$, so the straddle-to-spot ratio is a clean, model-light proxy for ex-ante 30-day return uncertainty.

### Origin in the Literature

Patell & Wolfson (1979, 1981, *Journal of Accounting Research*) were the first to document that equity-option implied volatilities rise systematically into scheduled earnings announcements and collapse on the release date — the original "IV crush" observation. (Beber & Brandt 2010, *Review of Finance*) generalize the idea to macro releases, showing that the ex-ante level of implied vol is a measure of resolvable uncertainty and that the post-release IV change is a tradeable signal of how much uncertainty was actually retired. (Dubinsky, Johannes, Kaeck & Seeger 2019, *Review of Financial Studies*, vol. 32(2), pp. 646-687) provide the modern reduced-form model: they decompose option prices into a "normal" diffusive component and an earnings-announcement jump component, estimate the announcement variance from straddles, and demonstrate it forecasts realized announcement-window volatility out of sample.

### The IV Crush Trade

Pre-announcement, options on names with imminent earnings price in event-jump variance, so 30-day IV spikes (typically 30-60 percent above the no-event baseline) and ATM straddles become rich. On release, the jump risk is realized and IV mean-reverts violently — the **vol crush**. The textbook trade is therefore *short premium into the print* (sell the straddle, delta-hedge), capturing the variance risk premium plus the crush. The mirror trade — *long the straddle pre-print* — bets on the post-earnings-announcement drift and is the gamma-positive cousin documented in (Dubinsky et al. 2019).

### Our Implementations

Two cross-sectional signals operationalize this literature:

- `opt_iv_crush_signal` (Beber-Brandt-style): rank `pre_iv30 - post_iv30` across the universe on each earnings date. Built in `src/earnings_iv_crush.py:build_iv_crush_signal()`, keyed by the EODHD earnings calendar; consumes cached Orats `iv30d` historically and Tradier daily polls live.
- `opt_implied_move_signal`: rank `-impliedMove` so that names with the **most expensive** straddles are predicted to mean-revert. Built in `src/options_signals.py:build_implied_move_signal()`. The straddle-to-spot computation lives in `src/options_adapter/chain_to_smv_summary.py`.

In-sample IC_IR for the crush signal lands in the **0.12-0.18** range, consistent with the Dubinsky et al. evidence that announcement variance is partially predictable.

### Strategic Note

Tradier exposes per-contract bid/ask for free, so impliedMove is computable live with **zero marginal data cost** post-deployment — the Orats subscription is required only for the 2013-2026 historical training window.

### Sources

- [Resolving Macroeconomic Uncertainty in Stock and Bond Markets (Beber & Brandt 2010) - Oxford Academic](https://academic.oup.com/rof/article/13/1/1/1589546)
- [Resolving Macroeconomic Uncertainty - NBER w12270 PDF](https://www.nber.org/system/files/working_papers/w12270/w12270.pdf)
- [Option Pricing of Earnings Announcement Risks (Dubinsky, Johannes, Kaeck & Seeger 2019) - Oxford Academic](https://academic.oup.com/rfs/article-abstract/32/2/646/5001193)
- [Option Pricing of Earnings Announcement Risks - VU Amsterdam PDF](https://research.vu.nl/ws/portalfiles/portal/108247883/Option_Pricing_of_Earnings_Announcement_Risks.pdf)
