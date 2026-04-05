# Personal Trading Portfolio — ML-Ranked Concentrated Long-Only

A systematic ML-driven personal equity portfolio. Ranks 6,000-7,600 US stocks (survivorship-bias-free, including 28K delisted names) using 150+ features and a walk-forward LightGBM/XGBoost/Ridge ensemble, then concentrates capital in the top 20 picks each month. Designed to actually trade with real money.

## Strategy Overview

**Core idea:** Use machine learning to identify the 20 best stocks each month from a universe of 6,000-7,600 US equities. Buy them, hold for one month, rebalance. No shorting, no hedging, no daily monitoring.

**Why this works at personal scale:**
- Concentrated positions (20 stocks at ~5% each) — each pick matters
- Small-cap access that institutions can't touch (capacity-constrained alpha)
- $0 commissions on major brokers, monthly rebalance = negligible costs
- Signal-weighted positions overweight highest-conviction picks
- Momentum pre-filter avoids value traps (never buy a downtrending stock)

**Target:** Beat SPY on a risk-adjusted basis. Core+satellite construction keeps beta near 1.0 while the ML sleeve hunts for alpha.

## Architecture

```
Price/Volume Data (EODHD primary, 7,600+ tickers w/ 28K delisted; yfinance fallback)
        |
        v
+----------------------+     +-------------------------+
| Alpha Signals (150+) |     | Alternative Data        |
| - Momentum           |     | SEC EDGAR fundamentals  |
| - Mean reversion     |     | FRED macro series       |
| - Volatility         |     | CBOE VIX / VVIX         |
| - Higher moments     |     | EODHD sentiment         |
|   (skew, co-skew,    |     | Insider (Form 4)        |
|    semi-betas, tail  |     | Analyst upgrades        |
|    dep, jump intens, |     | Earnings surprise / EAR |
|    Kumar lottery)    |     | Finnhub (free-gated)    |
| - Value composite    |     +-------------------------+
| - Quality (Piotroski,|              |
|   cash-based op prof)|              |
| - Distress (CHS, DtD,|              |
|   Altman Z)          |              |
| - Macro (DXY beta,   |              |
|   yield PCA, credit, |              |
|   EBP, cross-asset)  |              |
| - Breadth (%>200MA,  |              |
|   NH-NL, A/D line)   |              |
+----------------------+              |
        |                             |
        v                             v
+-----------------------------------------------+
| Feature Matrix (150+ features)                |
| - Cross-sectional z-scoring + winsorization   |
| - Size-neutralized momentum/vol/liquidity     |
| - Sector/macro interaction features           |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| Walk-Forward ML Ensemble                      |
| - LightGBM + XGBoost + Ridge (configurable)   |
|   + optional MLP / Huber / Quantile heads     |
| - IC-based early stopping, monotone constr.   |
| - Sample-concurrency weights, seed bagging    |
| - Embargo between train/test folds            |
| - 5-day beta-neutral forward labels           |
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
- Monthly = tax-friendly (some positions held 12+ months for long-term gains). Labels themselves are 5-day forward returns to align with short-horizon signal decay, but holdings churn monthly.

### Why Signal Weighting?
Equal weighting treats the #1 and #20 pick identically. Signal weighting concentrates more capital in the model's highest-conviction picks, tempered by vol targeting and position caps.

### Why Momentum Pre-Filter?
Never buy a stock in a clear downtrend. Even if the model likes the fundamentals, falling price often signals information the model hasn't captured (insider selling, accounting issues, sector rotation). Requiring positive 63-day momentum eliminates value traps.

### Why the Largest Possible Universe?
More stocks = more opportunities for the ML model to find mispriced names. Small-cap alpha is especially lucrative at personal scale because institutions can't trade these names without moving the market. We run 6,000-7,600 stocks (post ETF-filter) with an ADV liquidity floor, including ~28K delisted tickers for survivorship-bias-free backtesting.

## Strategy Improvements (Tier 1-7)

- **Beta drag fixes:** size-neutralized momentum/vol/liquidity features, SPY core+satellite construction, forced mega-cap inclusion, vol targeting with leverage cap
- **Horizon alignment:** forward_window 21 -> 5 days, no risk-adjust on labels, beta-neutral label option
- **Academic factors:** multi-factor value composite, Piotroski F-score, cash-based operating profitability, earnings announcement return (EAR), CHS distress, naive DtD, Altman Z, semi-betas, macro cross-asset momentum
- **Model hygiene:** cross-sectional winsorization + z-scoring, IC-based early stopping, monotone constraints, sample-concurrency weights, seed bagging, train/test embargo
- **Data integrity:** ETF filtering, sentinel-price filter, proper sector mapping, survivorship-bias-free delisted tickers

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| EODHD | OHLCV prices (primary), 7,600+ tickers incl. 28K delisted | Paid API |
| SEC EDGAR XBRL | 10-K fundamentals (accounting concepts) | Company Facts API |
| SEC EDGAR | Form 4 insider transactions (buy/sell) | Submissions API + XML |
| FRED | Yield curve, HY spread, Fed Funds, unemployment, DXY | Free CSV API |
| CBOE | VIX term structure (9D/30D/3M/6M), VVIX | Free CSV |
| EODHD | News sentiment, analyst, earnings alt data | Paid API |
| Finnhub | Alt data (free-tier gated) | Free API |
| yfinance | SPY benchmark, earnings calendar (fallback only) | Free API |

All data is cached locally after first download.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build ticker universe (6,000-7,600 stocks from EODHD incl. delisted)
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

# Tier 1-7 flags
python run_strategy.py --beta-neutral-labels --spy-core 0.4 --force-mega-caps \
                      --vol-target 0.16 --max-leverage 1.3

# Ensemble weighting (LightGBM + XGBoost + Ridge + optional heads)
python run_strategy.py --huber-weight 0.2 --quantile-weight 0.2

# Live dashboard (Dash app)
python dashboard_app.py

# Parameter sweep harness
python test_suite.py

# Static dashboard
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
dashboard_app.py         Dash live app (interactive)
test_suite.py            Parameter sweep harness
src/
  data_loader.py         Price/volume data (EODHD primary), universe
  universe_builder.py    Ticker universe (EODHD, incl. delisted)
  features.py            150+ alpha factors + composite signal
  alt_data_loader.py     EDGAR, FRED, VIX, insider, analyst, sentiment
  alt_features.py        Alternative data feature engineering
  model.py               LightGBM/XGBoost/Ridge/MLP/Huber/Quantile ensemble
  portfolio.py           Long-only concentrated + SPY core/satellite
  backtest.py            Event-driven backtest engine
  metrics.py             Performance analytics, factor regressions
  regime.py              Market regime detection
  robustness.py          Bootstrap confidence intervals
  dashboard.py           Interactive Plotly dashboard
```
