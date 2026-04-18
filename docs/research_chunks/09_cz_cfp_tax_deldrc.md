# C&Z Minor Accounting Signals: CFP, Tax, DelDRC

This chunk documents three minor accounting signals from the Chen & Zimmermann
open-source replication universe that are integrated into our pipeline behind
the `--use-cz-signals` flag. Each signal is individually modest
(IC_IR ~0.11-0.14 in our 2013-2026 EDGAR-derived panel), but together they
add diversification across orthogonal accounting dimensions: cash-based
valuation, tax-based earnings quality, and revenue-recognition timing.

### CFP (Cash Flow to Price)

**Origin.** Lakonishok, Shleifer, and Vishny (1994), "Contrarian Investment,
Extrapolation, and Risk," *Journal of Finance* 49(5), 1541-1578. LSV
demonstrate that simple value strategies built on cash-based fundamentals
outperform glamour stocks, attributing the premium to behavioral
extrapolation rather than risk compensation.

**Formula.** `CFP_i,t = operating_cash_flow_i,t / market_cap_i,t`. Operating
cash flow is preferred to earnings because it is harder to manipulate via
accruals.

**Sign.** Positive — high CFP firms are undervalued relative to their cash
generation and earn higher subsequent returns.

**Implementation.** `src/cz_signals.py::build_cfp_signal()` (lines 312-323),
clipped to [-5, 5] before cross-sectional ranking. **IC_IR: 0.111**.

### Tax (Tax Expense Surprise)

**Origin.** Lev and Nissim (2004), "Taxable Income, Future Earnings, and
Equity Values," *The Accounting Review* 79(4), 1039-1074. Lev and Nissim
show that the tax-to-book income ratio predicts five-year earnings growth
and subsequent stock returns, capturing information about earnings quality
that book income alone misses.

**Formula.** `Tax_i,t = tax_expense_i,t / net_income_i,t` (an effective tax
rate proxy). High book-tax conformity implies less aggressive accrual
management and more durable earnings.

**Sign.** Positive — firms reporting tax expense closer to their book
income exhibit higher earnings persistence and outperform firms with low
effective tax rates that signal earnings inflation.

**Implementation.** `src/cz_signals.py::build_tax_signal()` (lines 350-366),
normalized against the 21% post-TCJA federal statutory rate and clipped to
[-2, 5]. **IC_IR: 0.123**.

### DelDRC (Change in Deferred Revenue / Cash)

**Origin.** Prakash and Sinha (2013) and the broader deferred-revenue
literature, which identifies unearned revenue as a leading indicator of
future top-line growth.

**Formula.** `DelDRC_i,t = Δ(deferred_revenue_i,t) / avg(assets_t, assets_t-1)`,
computed as a year-over-year (252-day) change scaled by average assets.

**Sign.** Positive — rising deferred revenue represents cash already
collected but not yet recognized; it locks in future revenue and predicts
higher subsequent returns as the liability unwinds into the income
statement.

**Implementation.** `src/cz_signals.py::build_deldrc_signal()`
(lines 369-386), clipped to [-1, 1] before ranking. **IC_IR: 0.136**.

---

**Aggregate Role.** These three signals are each below the typical
single-factor inclusion threshold of IC_IR > 0.15, so they are deployed only
inside the C&Z accounting composite. Their value is diversification: CFP
spans cash valuation, Tax spans earnings-quality conformity, and DelDRC
spans revenue-timing accruals — three weakly correlated accounting
dimensions that each contribute marginal alpha when combined with the
stronger C&Z signals (Coskewness, XFIN, NetPayoutYield).

Sources:
- [Lakonishok, Shleifer, and Vishny (1994), Contrarian Investment, Extrapolation, and Risk - Journal of Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1994.tb04772.x)
- [Lakonishok, Shleifer, and Vishny (1994) - NBER Working Paper 4360](https://www.nber.org/papers/w4360)
- [Lev and Nissim (2004), Taxable Income, Future Earnings, and Equity Values - The Accounting Review](https://meridian.allenpress.com/accounting-review/article-abstract/79/4/1039/53457/Taxable-Income-Future-Earnings-and-Equity-Values)
- [Lev and Nissim (2004) - Columbia working paper PDF](http://www.columbia.edu/~dn75/taxableincome.pdf)
