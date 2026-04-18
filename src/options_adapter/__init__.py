"""Options adapter package — turns Tradier raw chain → Orats-equivalent SMV summary.

After the Orats subscription ends (~May 16, 2026), this adapter lets us continue
producing the same options-derived signals using only Tradier's free brokerage
API (which embeds Orats Greeks per-contract).

Public API:
    chain_to_smv_summary(chain, underlying, risk_free_rates, dividend_yield, asof)
        → dict with iv30d, iv60d, slope, dlt5/25/75/95Iv30d, borrow30, annIdiv

Architecture:
    Tradier chain DataFrame
        ├─ cmiv_interpolator.compute_constant_maturity_iv() → iv30d/60d/90d
        ├─ tradier_orats_adapter.compute_cores_row() → slope, dlt buckets
        ├─ implied_borrow.compute_implied_borrow_rate() → borrow30
        └─ implied_dividend_proxy.compute_implied_annual_dividend() → annIdiv
                ↓
        SMV summary dict (Orats-compatible schema)
                ↓
        Append to data/cache/options/iv_panels_orats.pkl
                ↓
        options_signals.py (no changes) consumes the panels
"""
from .chain_to_smv_summary import (
    chain_to_smv_summary,
    chain_to_smv_summary_batch,
    DEFAULT_TARGETS,
)

__all__ = [
    "chain_to_smv_summary",
    "chain_to_smv_summary_batch",
    "DEFAULT_TARGETS",
]
