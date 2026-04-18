# Risk Reversal and Smile Slope Signals

The implied-volatility surface of single-name equity options encodes forward-looking, risk-neutral information about the conditional return distribution that the historical realised time series cannot recover. Three closely related but theoretically distinct signals — Risk Reversal (RR25), Smile Slope, and Crash Risk — exploit different sections of the 30-day delta-IV curve to extract complementary cross-sectional return predictors.

### Risk Reversal (RR25)

Xing, Zhang and Zhao (2010), in "What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns?" (JFQA 45(3): 641-662), document that stocks whose volatility smirk — the gap between OTM-put IV and ATM-call IV — is steepest underperform stocks with the flattest smirks by roughly 10.9% per year on a risk-adjusted basis. The mechanism is informational: traders with negative private information preferentially purchase OTM puts, bidding up downside IV before the equity market incorporates the news. We define `RR25 = dlt25Iv30d - dlt75Iv30d` in Orats's call-delta convention (equivalent to 25-delta call IV minus 25-delta put IV, since a 75-delta call ≈ a 25-delta put). The signal carries a **negative** loading: a deeply negative RR25 (high put skew) flags an oversold name with positive expected return. Implementation: `src/options_signals.py:build_risk_reversal_25d_signal()`; in-sample IC_IR ≈ 0.155.

### Smile Slope

Bali and Murray (2013), "Does Risk-Neutral Skewness Predict the Cross Section of Equity Option Portfolio Returns?" (JFQA 48(4): 1145-1171), construct delta- and vega-neutral skewness assets and find a robust negative pricing of risk-neutral skewness, consistent with skewness preference. We operationalise the slope directly as `dlt75Iv30d - dlt25Iv30d`, replacing an earlier OLS-fit method whose rank correlation against the Orats reference was only 0.063; the closed-form bucket difference recovers a rank correlation of 0.753. Per `cz_signal_ic.csv`, SmileSlope posts an IC_IR of 0.369, the second-highest options signal in our research panel. The sign is **positive** — counterintuitively, names with high crash fear are oversold and earn the skewness risk premium ex post. Implementation: `src/options_signals.py:build_smileslope_signal()`.

### Crash Risk

Kelly and Jiang (2014), "Tail Risk and Asset Prices" (RFS 27(10): 2841-2871), establish that tail-risk exposure carries a positive risk premium in the cross section. We adapt their insight to the single-name IV surface as `CrashRisk = dlt95Iv30d - iv30d`, the deep-OTM-put (5-delta put ≈ 95-delta call) premium over ATM volatility. Loading is **positive**: names paying the highest crash insurance premium compensate holders with higher subsequent returns. Implementation: `src/options_signals.py:build_crash_risk_signal()`.

All three signals are derivable from Tradier's delta-bucket IV chain via PCHIP interpolation (`dlt5/25/75/95Iv30d`), eliminating any forward dependence on an Orats subscription for production scoring.

Sources:
- [Xing, Zhang & Zhao (2010), JFQA — author preprint](https://www.ruf.rice.edu/~yxing/option-skew-FINAL.pdf)
- [Xing, Zhang & Zhao (2010) — Cambridge Core](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/ECFD16BA9ACBDC8D577D1BD866FBEA72/S0022109010000220a.pdf/div-class-title-what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns-div.pdf)
- [Bali & Murray (2013), JFQA — UNL repository](https://digitalcommons.unl.edu/context/financefacpub/article/1029/viewcontent/Murray_JFQA_2013_Does_Risk_Neutral_Skewness__Cambridge_UP.pdf)
- [Bali & Murray (2013) — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1572827)
- [Kelly & Jiang (2014), RFS — Oxford Academic](https://academic.oup.com/rfs/article-abstract/27/10/2841/1607080)
- [Kelly & Jiang (2014) — NBER WP 19375](https://www.nber.org/system/files/working_papers/w19375/w19375.pdf)
