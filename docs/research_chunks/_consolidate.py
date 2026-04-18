"""Consolidate 20 research chunks into a new section of CONTEXT.md."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = ROOT / "docs" / "research_chunks"
CONTEXT = ROOT / "CONTEXT.md"

CZ_CHUNKS = [
    ("01_cz_project.md",            "C&Z (Open Source Asset Pricing) - Project Overview"),
    ("02_cz_coskewness.md",         "Coskewness (Harvey-Siddique 2000)"),
    ("03_cz_xfin.md",               "XFIN - External Financing (Bradshaw-Richardson-Sloan 2006)"),
    ("04_cz_payout_yield.md",       "PayoutYield + NetPayoutYield (Boudoukh-Michaely-Richardson-Roberts 2007)"),
    ("05_cz_announcement_return.md","Earnings Announcement Return / EAR (Brandt et al. 2008)"),
    ("06_cz_earnings_streak.md",    "Earnings Streak + Surprise Consistency"),
    ("07_cz_cb_op_prof.md",         "Cash-Based Operating Profitability (Ball et al. 2016)"),
    ("08_cz_mom_season.md",         "Momentum Seasonality (Heston-Sadka 2008)"),
    ("09_cz_cfp_tax_deldrc.md",     "CFP, Tax, DelDRC - Minor Accounting Signals"),
    ("10_cz_roadmap.md",            "C&Z Implementation Roadmap (built / overlap / missing)"),
]

ORATS_CHUNKS = [
    ("11_orats_company.md",            "Orats - Company, Delayed Data API, SMV Methodology"),
    ("12_orats_cores_schema.md",       "Orats /cores Endpoint - 340-Column Schema"),
    ("13_orats_other_endpoints.md",    "Other Orats Endpoints (/summaries, /ivrank, /dailies, /hvs, /earnings)"),
    ("14_orats_implied_move.md",       "impliedMove + Earnings IV Crush (Beber-Brandt 2010)"),
    ("15_orats_vrp.md",                "Variance Risk Premium + IV-RV Spread (Han-Zhou, Bali-Hovakimian)"),
    ("16_orats_term_structure.md",     "IV Term Structure: Slope, Convexity, Forward Curves (Vasquez 2017)"),
    ("17_orats_risk_reversal.md",      "Risk Reversal, SmileSlope, Crash Risk (Xing-Zhang-Zhao, Bali-Murray, Kelly-Jiang)"),
    ("18_orats_iv_rank.md",            "IV Rank, Vol-of-Vol, OI Concentration (Goyal-Saretto, Cremers, Pan-Poteshman)"),
    ("19_orats_bkm.md",                "BKM Implied Moments - Risk-Neutral Skew & Kurtosis (Bakshi-Kapadia-Madan 2003)"),
    ("20_orats_tradier_migration.md",  "Tradier Migration - Architecture, What's Lost, $0/mo Going Forward"),
]


def demote(text: str) -> str:
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,4})(\s+.*)$", line)
        if m:
            hashes, rest = m.group(1), m.group(2)
            new_hashes = "#" * min(6, len(hashes) + 2)
            out.append(new_hashes + rest)
        else:
            out.append(line)
    return "\n".join(out)


def load_chunk(filename: str) -> str:
    path = CHUNK_DIR / filename
    if not path.exists():
        return f"\n_(missing chunk: {filename})_\n"
    text = path.read_text(encoding="utf-8").strip()
    text = re.sub(r"^(#{1,2})\s+.*?\n+", "", text, count=1)
    return demote(text)


parts = []
parts.append("\n---\n\n## 40. Signal Research Library - C&Z + Orats Academic Reference\n")
parts.append("> Consolidated 2026-04-17 from 20 parallel research agents. Each subsection cites the\n"
             "> originating academic paper, the formula, the sign convention, our implementation file/lines,\n"
             "> and any data-source notes (cached vs live). Use this when designing new signals or auditing\n"
             "> existing ones - do NOT reinvent. Source markdown lives at `docs/research_chunks/*.md`.\n")

parts.append("\n### 40.1 C&Z (Open Source Asset Pricing) - 209-Signal Replication Framework\n")
for fname, title in CZ_CHUNKS:
    parts.append(f"\n#### {title}\n")
    parts.append(load_chunk(fname))
    parts.append("")

parts.append("\n### 40.2 Orats Options Data + Tradier Migration\n")
for fname, title in ORATS_CHUNKS:
    parts.append(f"\n#### {title}\n")
    parts.append(load_chunk(fname))
    parts.append("")

parts.append("""
---

### 40.3 Master Signal Priority - Quick Reference

Sorted by IC_IR (per `results/_cz_research/cz_signal_ic.csv`). All signals built via `--use-cz-signals` (C&Z) or `--use-options-signals` (Orats/Tradier-derived) flags. Validated subset for production: `OPTIONS_SIGNAL_SET=validated`.

| IC_IR | Signal | Source | Built? | Tradier-only OK? |
|---|---|---|---|---|
| 0.375 | dCPVolSpread | Options /cores delta-bucket spread d5d | Yes | YES |
| 0.369 | SmileSlope | Bali-Murray 2013, dlt75-dlt25 | Yes (post-fix rank corr 0.753) | YES |
| 0.286 | OScore | Campbell-Hilscher-Szilagyi (covered via `chs_distress_signal`) | Yes | n/a |
| 0.254 | AnnouncementReturn | Brandt et al. 2008 | Yes | YES (via Tradier+EODHD calendar) |
| 0.250 | ShareIss5Y | Pontiff-Woodgate 2008 (covered via `net_share_issuance_signal`) | Yes | n/a |
| 0.245 | EarningsStreak | Bartov-Givoly-Hayn 2002 | Yes | YES |
| 0.225 | ShareRepurchase | Grullon-Michaely 2004 (covered via `q_buyback_yield`) | Yes | n/a |
| 0.200 | CBOperProf | Ball et al. 2016 | Yes (`alt_cash_based_op_prof_signal`) | n/a |
| 0.200 | FirmAgeMom | (firm-age momentum) | **NO** - easy build | YES (price-only) |
| 0.198 | Coskewness | Harvey-Siddique 2000 | Yes (price-only, 2 horizons) | YES (price-only) |
| 0.181 | XFIN | Bradshaw-Richardson-Sloan 2006 | Yes (`build_xfin_signal`) | n/a |
| 0.178 | NetEquityFinance | Sign-flipped NetPayoutYield | Yes (covered) | n/a |
| 0.176 | NetPayoutYield | Boudoukh et al. 2007 | Yes (`build_net_payout_yield_signal`) | n/a |
| 0.155 | rv_iv_spread / RR25 | Bali-Hovakimian / Xing-Zhang-Zhao | Yes | YES |
| 0.152 | AnalystRevision | (analyst signals - Finnhub data needed) | **NO** - Finnhub | YES (need Finnhub) |
| 0.146 | PayoutYield | Boudoukh et al. 2007 | Yes | n/a |
| 0.146 | MomSeason | Heston-Sadka 2008 | Yes (price-only) | YES (price-only) |
| 0.143 | ChangeInRecommendation | Finnhub recommendations | **NO** - Finnhub | YES |
| 0.136 | DelDRC | Prakash-Sinha 2013 | Yes | n/a |
| 0.135 | RDAbility | (R&D efficiency) | **NO** - EDGAR | YES (need EDGAR) |
| 0.132 | RoE | (direct ROE) | **NO** - distinct from `q_roe` | YES (EDGAR) |
| 0.127 | EarningsConsistency | (CV of surprises) | Yes (`earnings_surprise_consistency`) | YES |
| 0.125 | OperProfRD | Ball et al. 2016 R&D variant | Yes (`build_operprof_rd_signal`) | n/a |
| 0.124 | CoskewACX | Ang-Chen-Xing 2006 downside variant | Yes | YES (price-only) |
| 0.123 | Tax | Lev-Nissim 2004 | Yes | n/a |
| 0.123 | std_turn | (idiosyncratic turnover) | **NO** | YES (price+volume) |
| 0.119 | NOA | Hirshleifer et al. (covered via `net_operating_assets_signal`) | Yes | n/a |
| 0.119 | CitationsRD | (NBER patent dataset) | **NO** - patents | YES (NBER patents) |
| 0.113 | sfe | (sector FX exposure) | **NO** | YES (cross-asset) |
| 0.111 | CFP | Lakonishok-Shleifer-Vishny 1994 | Yes | n/a |
| 0.103 | GP | Novy-Marx (covered via `gross_profitability_signal`) | Yes | n/a |
| 0.103 | EP | (earnings/price - covered via `earnings_yield_signal`) | Yes | n/a |

**Net status (post Tradier migration + 5-agent extraction):**
- 26+ functional options signals from Tradier alone (was 15, with 4 dead)
- Coskewness confirmed PRICE-ONLY (no Orats dependency)
- 11 high-IC C&Z signals MISSING (FirmAgeMom, AnalystRevision, RoE, RDAbility, etc.) - buildable from existing data sources
- ~$1,188/yr saved by cancelling Orats once 30-day Tradier archive accumulates

### 40.4 Cross-References

- **Source chunks**: `docs/research_chunks/01_cz_project.md` ... `20_orats_tradier_migration.md`
- **C&Z research outputs**: `results/_cz_research/cz_signal_ic.csv`, `cz_overlap_map.csv`
- **Orats field inventories**: `results/validation/orats_cores_field_inventory.csv`, `orats_other_endpoints_inventory.csv`
- **Phase 1 Tradier validation**: `results/validation/phase1_n30_post_fix_2026-04-17.csv`
- **Validated signal subset**: `OPTIONS_SIGNAL_SET=validated` keeps top-5 (dCPVolSpread, SmileSlope, variance_premium, rv_iv_spread, iv_term_slope)
- **Test discipline**: cache-key fix in `src/model.py:234-256` includes `EXTRA_DENY_FEATURES` for proper isolation; new CLI flags `--cz-only=<csv>` and `--cz-exclude=<csv>` enable per-signal A/B tests
""")

with open(CONTEXT, "a", encoding="utf-8") as f:
    f.write("\n".join(parts))

new_lines = len(CONTEXT.read_text(encoding="utf-8").splitlines())
print(f"Appended Section 40 to CONTEXT.md")
print(f"  new total: {new_lines} lines")
print(f"  20 chunks + master priority table consolidated")
