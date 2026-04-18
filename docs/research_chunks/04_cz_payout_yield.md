## Net Payout Yield + Payout Yield

### Origin

Boudoukh, Michaely, Richardson, and Roberts (2007), "On the Importance of Measuring Payout Yield: Implications for Empirical Asset Pricing," *Journal of Finance* 62(2), 877-915. The paper challenges the traditional dividend-yield literature (Fama and French 1988; Campbell and Shiller 1988) by demonstrating that the secular decline in dividend payouts during the 1980s-1990s reflects not a weakening of the cash-flow-to-price relationship but a *substitution* of share repurchases for dividends. Once buybacks (and net issuance) are incorporated, the time-series predictability of aggregate returns is restored, and the cross-sectional spread is materially stronger than that of dividend yield alone.

### Key Insight

Dividend yield understates total cash returned to shareholders. Modern firms increasingly prefer repurchases for tax efficiency (capital-gains rate < ordinary dividend rate, particularly pre-2003 JGTRRA), payout-flexibility (no implicit "dividend smoothing" commitment), and management compensation (option-overhang absorption). Measuring only dividends therefore truncates the true payout signal and produces deteriorating predictive power post-1980. The fix is conceptually trivial but empirically decisive: replace D/P with (D+R)/P, and optionally subtract issuance to capture *net* shareholder cash flow.

### Formulas

- **PayoutYield** = (dividends + share repurchases) / market_cap
- **NetPayoutYield** = (dividends + repurchases - issuance) / market_cap

The net version subtracts seasoned-equity issuance and option-related share creation, capturing genuine dilution-adjusted shareholder yield.

### Sign and IC

Sign is **positive** (classic value channel: high payout yield -> higher forward returns). Per `cz_signal_ic.csv`, **PayoutYield IC_IR = 0.146** and **NetPayoutYield IC_IR = 0.176**. The net version's superior IR confirms Boudoukh et al.'s thesis: dilution materially erodes the gross-payout signal, and adjusting for it produces a cleaner shareholder-yield measure.

### Implementation

- `src/cz_signals.py:build_payout_yield_signal()` lines 238-256
- `src/cz_signals.py:build_net_payout_yield_signal()` lines 259-279

Both signals use a 6-month lag (`market_cap.shift(126)`) on the denominator to avoid look-ahead via simultaneous price updates, clip raw ratios to `[-1, 5]`, and emit cross-sectional ranks via `_cs_rank()`. Both are gated behind `--use-cz-signals` (default OFF).

### Data Dependencies

`edgar_extra` fields: `raw_dividends_paid`, `raw_buybacks`, `raw_stock_issuance` (sourced from EDGAR XBRL cash-flow statements with standard filing-lag protection in `_raw_panel_from_edgar`). `market_cap` panel from the standard pricing/shares-outstanding pipeline.

### Relationship to Existing `dividend_yield`

The legacy `dividend_yield` feature (in `src/api_data.py`) is a strict **subset** of PayoutYield: it captures only the dividend leg and ignores the now-dominant repurchase channel. When `--use-cz-signals` is enabled, set `DISABLE_DIVIDEND_YIELD=1` to suppress the legacy feature and avoid signal duplication / multicollinearity in the LightGBM stack. NetPayoutYield further dominates by netting issuance, making the legacy feature strictly redundant under the C&Z stack.

### Theory

Total return to shareholders - not the accounting label attached to it - drives valuation. Substitution between dividends and buybacks is a tax-and-flexibility decision, not a cash-flow signal; conditioning on the gross channel reintroduces the predictability that vanished from D/P after 1980.
