# Personal Trading Portfolio — ML-Ranked Concentrated Long-Only

A systematic ML-driven personal equity portfolio. Ranks 3,000+ US stocks using 80+ features and a walk-forward LightGBM/Ridge ensemble, then concentrates capital in the top 20 picks each month. Designed to actually trade with real money.

## Strategy Overview

**Core idea:** Use machine learning to identify the 20 best stocks each month from a universe of 3,000+ US equities. Buy them, hold for one month, rebalance. No shorting, no hedging, no daily monitoring.

**Why this works at personal scale:**
- Concentrated positions (20 stocks at ~5% each) — each pick matters
- Small-cap access that institutions can't touch (capacity-constrained alpha)
- $0 commissions on major brokers, monthly rebalance = negligible costs
- Signal-weighted positions overweight highest-conviction picks
- Momentum pre-filter avoids value traps (never buy a downtrending stock)

**Target:** Beat SPY by 5-10%+ annualized. That compounds to 2-3x the wealth over 20 years.

## Architecture

```
Price/Volume Data (yfinance, 3,000+ tickers)
        |
        v
+-------------------+     +-------------------------+
| Alpha Signals (24) |     | Alternative Data        |
| - Momentum (5 win) |     | SEC EDGAR fundamentals  |
| - Mean reversion   |     | FRED macro series       |
| - Volatility       |     | CBOE VIX term structure |
| - Liquidity        |     | SEC Form 4 insider txns |
| - Reversal         |     | Analyst upgrades        |
+-------------------+     | Earnings surprise       |
        |                  +-------------------------+
        v                           |
+-----------------------------------------------+
| Feature Matrix (80+ features)                 |
| - Momentum, mean-reversion, vol, liquidity    |
| - Cross-sectional ranks of all factor signals |
| - Macro x stock interaction features          |
| - Fundamental + event-driven signals          |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| Walk-Forward ML Ensemble                      |
| - LightGBM LambdaRank (60%) + Ridge (40%)    |
| - 126-day min training, 21-day retrain cycle  |
| - 21-day forward labels (monthly horizon)     |
| - Lookahead-safe: gap labels near cutoff      |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| Long-Only Portfolio Construction              |
| 1. ML signal smoothing (5d rolling mean)      |
| 2. Momentum pre-filter (63d positive trend)   |
| 3. Select top 20 stocks by ML rank            |
| 4. Sector cap (max 35% per sector)            |
| 5. Signal-weighted sizing (concentration=1.0) |
| 6. Position caps (2% min, 10% max)            |
| 7. Monthly rebalance (1st trading day)        |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| Backtest Engine                               |
| - T+1 fill at open prices                    |
| - Buy-and-hold between monthly rebalances     |
| - Personal-scale costs: 3bp spread, $0 comm   |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| Analytics & Dashboard                         |
| - CAGR, Alpha vs SPY, Sharpe, Sortino        |
| - Annual & monthly returns (calendar heatmap) |
| - Wealth growth comparison vs SPY             |
| - Current holdings & sector breakdown         |
| - Fama-French regression, regime performance  |
| - Bootstrap confidence intervals              |
| - Interactive Plotly dashboard                |
+-----------------------------------------------+
```

## Key Design Decisions

### Why 20 Positions?
- Fewer than 10 = too much idiosyncratic risk (one bad pick kills the month)
- More than 30 = dilutes alpha (starts looking like an index fund)
- 20 is the sweet spot: enough diversification, enough concentration per pick

### Why Monthly Rebalance?
- Weekly = too much turnover, tax-inefficient (short-term cap gains)
- Quarterly = too stale (signals decay)
- Monthly = matches the 21-day forward label horizon. Tax-friendly (some positions held 12+ months for long-term gains)

### Why Signal Weighting?
Equal weighting treats the #1 and #20 pick identically. Signal weighting concentrates more capital in the model's highest-conviction picks. The ML model has demonstrated real predictive power (statistically significant alpha), so we should trust its conviction levels.

### Why Momentum Pre-Filter?
Never buy a stock in a clear downtrend. Even if the model likes the fundamentals, falling price often signals information the model hasn't captured (insider selling, accounting issues, sector rotation). Requiring positive 63-day momentum eliminates value traps.

### Why the Largest Possible Universe?
More stocks = more opportunities for the ML model to find mispriced names. Small-cap alpha is especially lucrative at personal scale because institutions can't trade these names without moving the market. We run 3,000+ stocks with a $1M ADV filter.

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| Yahoo Finance | OHLCV prices, 3,000+ tickers | Free API |
| SEC EDGAR XBRL | 10-K fundamentals (7 accounting concepts) | Company Facts API |
| SEC EDGAR | Form 4 insider transactions (buy/sell) | Submissions API + XML |
| FRED | Yield curve, HY spread, Fed Funds, unemployment | Free CSV API |
| CBOE | VIX term structure (9D/30D/3M/6M) | Free CSV |
| Yahoo Finance | Analyst upgrades/downgrades | Free API |
| Yahoo Finance | Earnings calendar + EPS surprise | Free API |

All data is cached locally after first download.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build ticker universe (3000+ stocks from S&P 1500 + Russell 1000/2000)
python src/universe_builder.py

# Run the full pipeline (first run downloads data, ~30-60 min)
python run_strategy.py

# Subsequent runs use cached data (~5-10 min)
python run_strategy.py

# Custom parameters
python run_strategy.py --capital 50000 --n-positions 15
python run_strategy.py --weighting equal           # equal weight instead of signal
python run_strategy.py --skip-ml --skip-alt-data   # fast rule-based only
python run_strategy.py --start 2015-01-01 --end 2025-01-01

# View interactive dashboard
open results/dashboard.html
```

## Output Files

| File | Description |
|------|-------------|
| `results/dashboard.html` | Interactive Plotly dashboard |
| `results/tearsheet.csv` | Strategy performance metrics |
| `results/annual_returns.csv` | Year-by-year returns vs SPY |
| `results/monthly_returns.csv` | Calendar monthly returns (%) |
| `results/wealth_growth.csv` | Cumulative wealth: strategy vs SPY |
| `results/current_holdings.csv` | Latest portfolio holdings |
| `results/sector_allocation.csv` | Sector weights over time |
| `results/oos_tearsheet.csv` | In-sample vs out-of-sample |
| `results/fama_french.csv` | Factor regression results |
| `results/feature_importance.csv` | ML feature importances |
| `results/regime_performance.csv` | Performance by market regime |
| `results/stress_test.csv` | Crisis period analysis |
| `results/bootstrap_ci.csv` | Bootstrap confidence intervals |

## How to Actually Trade This

1. Run the pipeline monthly (first trading day of the month)
2. Open `results/current_holdings.csv` — these are your 20 stocks with target weights
3. In your broker (IBKR, Schwab, Fidelity): sell positions not in the list, buy new ones at target weights
4. ~10-15 trades per month, takes 15 minutes

## Project Structure

```
run_strategy.py          Main pipeline entry point
src/
  data_loader.py         Price/volume data, universe construction
  universe_builder.py    Ticker universe from S&P + Russell indices
  features.py            24 alpha factors + composite signal
  alt_data_loader.py     EDGAR, FRED, VIX, insider, analyst data
  alt_features.py        Alternative data feature engineering
  model.py               LightGBM/Ridge ensemble, walk-forward trainer
  portfolio.py           Long-only concentrated portfolio construction
  backtest.py            Event-driven backtest engine
  metrics.py             Performance analytics, factor regressions
  regime.py              Market regime detection
  robustness.py          Bootstrap confidence intervals
  dashboard.py           Interactive Plotly dashboard
```
