## Earnings Streak and Surprise Consistency

### Theme and Theoretical Motivation

Repeated positive earnings surprises predict subsequent abnormal returns through two reinforcing channels. First, the post-earnings-announcement drift (PEAD) literature, founded by (Foster, Olsen & Shevlin 1984, The Accounting Review), "Earnings Releases, Anomalies, and the Behavior of Security Returns," vol. 59(4), pp. 574-603, documents that prices underreact to earnings news and continue to drift in the surprise's direction for roughly sixty trading days. When surprises occur in succession, the underreaction compounds: each beat adds fresh drift before the prior beat's drift has been fully absorbed.

Second, persistent beats re-anchor analyst expectations. (Bartov, Givoly & Hayn 2002, Journal of Accounting and Economics), "The Rewards to Meeting or Beating Earnings Expectations," vol. 33(2), pp. 173-204, show that firms which consistently meet or beat consensus enjoy a premium *over and above* the magnitude of the surprise itself, consistent with managerial signaling and reduced uncertainty about the firm's earnings process. (DeFond & Park 2001, The Accounting Review), "The Reversal of Abnormal Accruals and the Market Valuation of Earnings Surprises," vol. 76(3), pp. 375-404, further document that the market only partially distinguishes accrual-driven from cash-driven surprises, leaving a tradable mispricing in firms whose surprise stream is "clean."

### Variants

Two complementary daily signals are constructed from quarterly EPS surprises, defined as $(\text{eps}_{\text{actual}} - \text{eps}_{\text{est}}) / |\text{eps}_{\text{est}}|$ and clipped to $[-2, 2]$:

- **Earnings Streak** — count of consecutive quarters with positive surprise, capped at 8. Resets to zero on any miss. Sign: **positive**.
- **Surprise Consistency** — reciprocal of the rolling 8-quarter coefficient of variation of surprises ($1 / (\sigma / |\mu|)$). Low CV implies a predictable beat process. Sign: **positive** (we negate CV before ranking).

### Empirical Performance

Per `results/cz_signal_ic.csv`, **EarningsStreak IC_IR = 0.245**, our #4-ranked novel Chen-Zimmermann signal in the 2013-2026 backtest. The consistency variant is correlated but contributes incremental information at the tails (very low CV firms behave more like a quality-of-earnings proxy).

### Implementations

- `src/alt_features.py:earnings_beat_streak_signal` (lines 590+) — iterates per ticker, builds the capped streak series at each event date, then forward-fills to daily.
- `src/alt_features.py:earnings_surprise_consistency_signal` (lines 620+) — rolling 8-quarter CV with $|\mu| > 10^{-9}$ guard, also forward-filled.

Both are Phase D additions, gated by the `ENABLE_PHASE_D_SIGNALS=1` environment variable (default ON). Earnings data resolves from yfinance by default or from the EODHD bulk earnings endpoint when `--use-eodhd-earnings` is passed. The signal value carries forward roughly 5-15 trading days post-event before the next quarterly print refreshes it, which aligns with the bulk of the PEAD window documented in Foster-Olsen-Shevlin.

### Sources

- [Earnings Releases, Anomalies, and the Behavior of Security Returns (Foster, Olsen & Shevlin 1984) - JSTOR](https://www.jstor.org/stable/247321)
- [The Reversal of Abnormal Accruals and the Market Valuation of Earnings Surprises (DeFond & Park 2001) - JSTOR](https://www.jstor.org/stable/3068959)
- [The Rewards to Meeting or Beating Earnings Expectations (Bartov, Givoly & Hayn 2002) - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165410102000451)
