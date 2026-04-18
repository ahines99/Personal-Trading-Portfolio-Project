### Orats: Company and Data Stack

Option Research and Technology Services (Orats, [orats.com](https://orats.com)) is a Chicago-based options analytics vendor founded in 2014 by Matt Amberson, a former Cboe market maker who previously hired statistically-trained traders to operate on the floor and built proprietary research tooling to support that desk. Amberson (Kellogg MBA, CPA, CFA Level I) remains principal and drives the firm's product and quantitative roadmap. The firm productized its internal market-making analytics into three commercial offerings: a **Delayed Data API** ($99/month, 15-minute lag, full historical and live coverage), a **Live Data API** ($399/month plus per-call pricing, real-time), and **Datashop**, a one-time bulk download channel used for research-grade historical pulls.

### Endpoints Consumed

This project ingests six endpoints from the `datav2/hist` namespace. `/cores` returns a 340-column daily snapshot per ticker spanning the full IV surface, term structure, skew, and earnings-adjusted variants. `/summaries` returns a denser 128-column slice that includes proprietary derived fields such as `impliedMove`. `/ivrank` provides the 1-month and 1-year IV percentile rank in 8 columns. `/dailies` exposes daily OHLCV (substitutable from any equity vendor). `/hvs` returns the historical realized volatility surface across multiple lookback horizons, including ex-earnings variants critical for de-noising event windows. `/earnings` provides an options-aware earnings calendar; the `anncTod` field encodes announcement time as HHMM, enabling correct point-in-time alignment of pre- versus post-announcement implied volatility.

### SMV (Smoothed Market Value) Methodology

The analytical core of every Orats endpoint is the **Smoothed Market Value** engine, an arbitrage-free IV surface fitter. SMV cleans raw exchange quotes, solves for a residual yield using put-call parity and dividend assumptions, feeds the cleaned inputs to a modified binomial pricing engine, and fits a smooth, non-arbitrageable curve through strike implied volatilities. This filters wide bid-ask "joke quotes," arbitrage-violating prices, end-of-day artifacts, and the noisy wings of low-premium out-of-the-money strikes; for illiquid contracts, SMV blends in historical priors when current confidence is low. The published `smv_vol` field, together with SMV-derived Greeks (delta, gamma, theta, vega, rho), produces theoretical values inside the bid-ask over 99% of the time.

A practical consequence is **train/serve consistency**: the same SMV engine generates this project's cached historical parquets and is also embedded inside Tradier's free options chain endpoint (Tradier explicitly credits Orats for Greeks and IV). Live polls and historical training data therefore share an identical numerical generator, eliminating the surface-fit skew that typically arises when historical and live vendors differ.

### Project Coverage

The local cache spans 1,414 tickers from 2013-01-01 through 2026-04-15, materialized as 8,061 parquet files totaling roughly 11 GB. This depth supports point-in-time IV-surface, skew, term-structure, and earnings-move features without survivorship bias.
