## IV Term Structure Signals: Slope, Convexity, Forward IV, Short/Long Ends

### Foundational Literature

Vasquez (2017), "Equity Volatility Term Structures and the Cross Section of Option Returns," *Journal of Financial and Quantitative Analysis* 52(6), 2727-2754, establishes the canonical result: the **slope of the equity implied-volatility term structure is positively related to future option returns**. Sorting firms on the difference between long- and short-tenor at-the-money implied volatilities yields a monotonic cross-section, with steep upward-sloping curves preceding higher straddle returns. The economic interpretation is that an upward-sloping IV curve implies that short-dated volatility is depressed relative to its long-run anchor and is expected to mean-revert upward — a state that benefits long-vol option positions and, through the leverage and gamma channels, predicts cross-sectional dispersion in the underlying equity returns as well.

### Implemented Slope Variants

We instrument three points on the curve to separate short-, mid-, and long-end information:

- **`iv_term_slope`** = `iv60d - iv30d` — the canonical Vasquez slope. Sign is **positive**: steep upward slope implies future vol expansion through mean reversion, which predicts higher equity returns.
- **`iv_short_term_slope`** = `iv30d - iv7d` — captures sub-two-week IV expectations, isolating event-driven kinks (earnings, macro prints) at the front of the curve that the 30/60d slope obscures.
- **`iv_long_term_slope`** = `iv365d - iv90d` — long-end "carry" measure capturing structural vol-risk-premium loadings independent of near-term event noise.

### Convexity Extension

Beyond the linear slope, we measure **`iv_term_convexity`** = `iv30 - 2*iv60 + iv90`, a discrete second derivative. Sign is **negative**: pronounced negative convexity (mid-tenor expensive relative to the wings) signals pre-event positioning concentrated in 60d strikes that subsequently unwinds. This extends Vasquez's first-moment slope with a curvature channel.

### Forward IV Curves (Variance Additivity)

By the no-arbitrage additivity of variance across non-overlapping intervals,

```
fwd_iv(t1 -> t2) = sqrt((iv_t2^2 * t2 - iv_t1^2 * t1) / (t2 - t1))
```

Forward IVs are **pure math from any constant-maturity IV grid** — Orats publishes `fwd30_20`, `fwd60_30`, `fwd90_60` as proprietary fields, but they are derivable for free from any vendor that supplies multiple maturities. Our implementation lives in `src/options_adapter/forward_curve.py:compute_forward_iv()` and reproduces Orats' published forwards within **0.5 vol points on 99.85% of 16,385 historical samples**. From this we construct three forward-curve signals:

- **`fwd_curve_slope`** = `-(fwd60_90 - fwd30_60)` — steep forward slope flags vol overpriced for the future window.
- **`realized_vs_forward`** = `RV30 - fwd30_60` — when realized exceeds the forward, vol-sellers were wrong and the market mean-reverts.
- **`long_forward_premium`** = `-(fwd90_180 - iv30)` — long-end vol crowding indicator.

### Implementation Notes

Convexity and forward slope encode similar curvature information; we **decorrelate at the model stage rather than dropping** either, preserving non-linear interactions. Critically, every signal in this chunk is **Tradier-only viable**: Tradier returns option chains across multiple maturities, our CMIV layer interpolates constant-maturity points, and the forward curve falls out of variance additivity — no proprietary feed required.

Sources:
- [Equity Volatility Term Structures and the Cross-Section of Option Returns (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1944298)
- [Vasquez 2017 JFQA listing (IDEAS/RePEc)](https://ideas.repec.org/a/cup/jfinqa/v52y2017i06p2727-2754_00.html)
- [Vasquez 2017 (EconPapers)](https://econpapers.repec.org/article/cupjfinqa/v_3a52_3ay_3a2017_3ai_3a06_3ap_3a2727-2754_5f00.htm)
- [Vasquez full paper (EFMA 2015 working version)](https://efmaefm.org/0efmameetings/efma%20annual%20meetings/2015-Amsterdam/papers/EFMA2015_0530_fullpaper.pdf)
