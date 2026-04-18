### The Chen–Zimmermann Open Source Asset Pricing Project

The Open Source Asset Pricing (OSAP) project, maintained by **Andrew Y. Chen** (Federal Reserve Board) and **Tom Zimmermann** (University of Cologne), is the most comprehensive public replication of the cross-sectional stock-return predictability literature. The project's working paper, Chen and Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing," appeared in the *Critical Finance Review* 11(2), 207–264, with code and data hosted at [openassetpricing.com](https://www.openassetpricing.com/) and on [GitHub](https://github.com/OpenSourceAP/CrossSection).

### Scope and Outputs

The October 2025 release ships **212 long–short predictor portfolios and 209 firm-level signals** in signed form, where each signal is oriented so that higher values forecast higher subsequent returns. Two artifacts anchor the dataset:

- **`SignalDoc.csv`** — metadata for every signal, including the originating publication, sample period, signal category (accounting, trading, event, analyst, etc.), construction notes, and the predicted sign.
- **`signed_predictors_dl_wide.csv`** — a firm-month wide panel (≈1.6 GB zipped) of all 209 signed characteristics, currently extending through December 2024 (option-implied predictors end December 2022).

Daily and monthly long–short return files, plus Python (`pip install openassetpricing`) and R packages, accompany the raw signal panel.

### Replication Methodology

For each predictor, Chen and Zimmermann re-implement the original paper's construction directly from CRSP and Compustat (with auxiliary IBES, OptionMetrics, and 13F inputs where applicable), then sign the variable so the long leg is the predicted-high-return tail. Reproduction quality is strong: among the 161 characteristics deemed "clearly significant" in the original studies, 98% of OSAP long–short portfolios produce |t| > 1.96, and a regression of reproduced on original t-statistics yields a slope near 0.90 with R² ≈ 0.83.

### Relevance and Caveats

For an ML-driven equity strategy, OSAP supplies a vetted, citation-grounded universe of candidate signals — eliminating the engineering cost of re-deriving each predictor from primary sources and providing a defensible benchmark for any in-house factor. The dataset must, however, be used with the documented post-publication-decay literature in mind: McLean and Pontiff (2016, *Journal of Finance* 71(1), 5–32) report that anomaly returns fall ~26% out-of-sample and ~58% post-publication, attributable to a mixture of statistical bias (data mining) and arbitrage by informed traders. Compounding this, the breadth of the OSAP zoo raises acute multiple-testing and false-discovery concerns (Harvey, Liu, and Zhu, 2016), so signals should be evaluated under stringent OOS protocols, deflated t-thresholds, or hierarchical/Bayesian shrinkage rather than taken at their published face value.
