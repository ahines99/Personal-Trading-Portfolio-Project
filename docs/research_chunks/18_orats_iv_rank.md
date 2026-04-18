# IV Rank, Volatility Mean Reversion, and OI Concentration

This chunk documents three lower-IC but diversifying options-derived signals layered on top of the SmileSlope and VRP cores: implied-volatility rank, volatility-of-volatility, and open-interest concentration. All three are sourced live from Tradier chains and aggregated across the multi-expiration term structure.

### IV Rank (Goyal & Saretto, 2009)

Goyal and Saretto, "Cross-Section of Option Returns and Volatility," *Journal of Financial Economics* 94(2), 310-326 (2009), show that equity-option returns are strongly predictable from the gap between historical realized volatility and ATM implied volatility — a reduced-form expression of mean reversion in implied vol. Long-short option strategies that sell expensive vol (high IV percentile) and buy cheap vol (low IV percentile) earn 50%+ annualized in their cross-section. We translate this into an equity-side signal as the rolling percentile of current 30-day IV in its trailing 252-day window, scaled to [0, 1]. Sign is **negative**: a high IV-rank stock has expensive options, vol is expected to mean-revert downward, and underlying-equity drift tends to disappoint relative to the expected-return wedge implied by an inflated risk premium. Implementation: `src/options_signals.py:build_iv_rank_signal()` (line 186) — `iv30.rolling(252, min_periods=60).rank(pct=True)`, then cross-sectionally re-ranked with a sign flip.

### Volatility-of-Volatility (Cremers, Halling & Weinbaum, 2015)

Cremers, Halling, and Weinbaum, "Aggregate Jump and Volatility Risk in the Cross-Section of Stock Returns," *Journal of Finance* 70(2) (2015), document a robust **negative** equity-return premium for vol-of-vol exposure: stocks whose IV itself is unstable carry a discount, but realize lower returns going forward (compensation is dwarfed by ex-post crash drag). We proxy with the rolling 30-day standard deviation of `iv30`. High vol-of-vol = unstable, hard-to-hedge regime — avoid.

### Open Interest Concentration (Pan & Poteshman, 2006)

Pan and Poteshman, "The Information in Option Volume for Future Stock Prices," *Review of Financial Studies* 19(3), 871-908 (2006), show that option order flow contains private information about underlying stocks: low put/call ratios precede next-day excess returns of 40+ bps and ~1% over a week. We adapt the open-interest analogue: `call_oi / (call_oi + put_oi)`, summed across the full chain. Sign is **positive** — call-heavy positioning reflects informed bullish demand. Implementation: `src/options_signals.py:build_oi_concentration_signal()` (line 406).

### Implementation & Performance Notes

Tradier's chain endpoint returns volume, OI, and per-contract IV across every listed expiration; we aggregate across the multi-exp surface and maintain a forward IV history via daily cron polls (no Orats backfill, so the IV-rank window builds out prospectively). Standalone IC_IR for each of the three falls in the **0.10-0.15** range — materially below SmileSlope or VRP — but their factor exposures are nearly orthogonal to the vol-premium core, so they earn their seat through diversification rather than raw signal strength.

Sources:
- [Goyal & Saretto (2009), Cross-Section of Option Returns and Volatility, JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X09001251)
- [Goyal & Saretto (2009), preprint PDF](https://personal.utdallas.edu/~axs125732/CrossOptionsJFE.pdf)
- [Pan & Poteshman (2006), The Information in Option Volume for Future Stock Prices, RFS](https://academic.oup.com/rfs/article-abstract/19/3/871/1646711)
- [Pan & Poteshman, NBER w10925](https://www.nber.org/papers/w10925)
