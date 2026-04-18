### Cash-Based Operating Profitability (CBOperProf) and OperProfRD

### Origin

Ball, Gerakos, Linnainmaa & Nikolaev (2016), "Accruals, Cash Flows, and Operating Profitability in the Cross Section of Stock Returns," *Journal of Financial Economics* 121(1), 28-45. The authors decompose accounting earnings into a cash component and an accruals component, then ask which piece carries the cross-sectional return signal. Their headline result: a cash-based operating profitability measure that strips accruals out of operating profit *subsumes* both the standard profitability factor (Novy-Marx 2013) and the accruals anomaly (Sloan 1996). Adding a single cash-based profitability factor to the opportunity set raises the maximum attainable Sharpe ratio more than adding the accruals and accrual-laden profitability factors jointly.

### Core insight

Accruals are management discretion. Revenue recognized before cash is collected, expenses deferred via capitalization, and working-capital movements all inject noise (and outright manipulation) into reported operating profit. Removing the accrual component yields a *cleaner* measure of recurring economic profitability, which in turn predicts returns more reliably because (i) cash earnings are harder to manage, and (ii) cash earnings persist longer than accrual earnings.

### Formulas

- Standard operating profitability: `OperProf = (Revenue - COGS - SGA) / Total Assets`
- Cash-based variant: `CBOperProf = (Revenue - COGS - SGA - DeltaWC) / Total Assets`, where `DeltaWC` is the year-over-year change in non-cash working capital (the accrual wedge).
- R&D-adjusted variant: `OperProfRD = (Revenue - COGS - (SGA - R&D)) / Total Assets`, recapitalizing R&D as investment rather than period expense (Ball et al. discuss this robustness check; it also appears in Chen-Zimmermann's signal library).

All three carry a **positive** sign: high profitability predicts high subsequent returns.

### Empirical strength in our pipeline

Per `cz_signal_ic.csv`, CBOperProf delivers IC_IR = 0.200 - one of the strongest single-name fundamentals we measure - while OperProfRD delivers IC_IR = 0.125. The gap confirms the BGLN result that the working-capital adjustment matters more than the R&D adjustment for return prediction.

### Implementation

- `src/alt_features.py:cash_based_op_prof_signal` constructs CBOperProf and is enabled by default under `use_tier3_academic`.
- `src/cz_signals.py:build_operprof_rd_signal()` constructs OperProfRD and only activates under `--use-cz-signals`.
- The two are *related but non-redundant*: CBOperProf adjusts for `DeltaWC`, OperProfRD adjusts for R&D capitalization. Running both simultaneously can dilute the cleaner CBOperProf signal, so the env var `DISABLE_CASH_BASED_OP_PROF` lets the alt-features version step aside when the CZ OperProfRD path is active.

Sources:
- [Ball, Gerakos, Linnainmaa & Nikolaev (2016) - Tuck PDF](https://faculty.tuck.dartmouth.edu/images/uploads/faculty/joseph-gerakos/Ball,_Gerakos,_Linnainmaa,_et_al._2016.pdf)
- [SSRN listing for the paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2587199)
- [ScienceDirect - JFE 121(1) 28-45](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300307)
- [Alpha Architect summary](https://alphaarchitect.com/2016/01/14/value-investing-accruals-cash-flows-and-operating-profitability/)
