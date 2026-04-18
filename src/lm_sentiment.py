"""
Loughran-McDonald sentiment features from EDGAR 10-K/10-Q text.

Reuses Sprint 2's text corpus (data/cache/edgar_text/) to compute financial-
domain sentiment features per Loughran & McDonald (2011, JF). Tone of
management discussion (Item 7) predicts future earnings surprises and
cross-sectional returns (~30-60 bps/yr, long-only).

Pipeline:
    1. Load LM dictionary (pysentiment2 -> SRAF download -> pickle cache -> fallback)
    2. For each cached filing, score Item 7 (MD&A) text
    3. Align to trading dates with publication lag + forward fill
    4. Cross-sectional z-score -> float32 features

Features produced:
    lm_net_tone     : (positive - negative) / total_words         (HIGH = bullish)
    lm_uncertainty  : uncertainty / total_words                   (HIGH = bearish)
    lm_litigious    : litigious / total_words                     (HIGH = bearish)
    lm_tone_change  : net_tone - prior_filing_net_tone            (NEG shock = bearish)
"""

from __future__ import annotations

import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fallback dictionary (offline, ~50 words/category).
# Low-quality shim so the module works without network access. Real LM master
# dict has thousands of words per category -- prefer the SRAF download.
# ---------------------------------------------------------------------------
_LM_FALLBACK: Dict[str, set] = {
    "positive": {
        "able", "abundance", "achieve", "achieved", "achievement", "advance",
        "advances", "advantage", "advantages", "beneficial", "benefit",
        "benefits", "best", "better", "boost", "boosted", "boosts", "confident",
        "constructive", "effective", "efficient", "efficiency", "enhance",
        "enhanced", "enhancement", "enjoy", "enjoyed", "excellent", "excelled",
        "exceptional", "favorable", "gain", "gained", "gains", "good",
        "great", "greater", "greatest", "grew", "growth", "improve", "improved",
        "improvement", "improves", "improving", "increase", "increased",
        "increases", "leadership", "leading", "positive", "profitable",
        "profitability", "progress", "prosper", "prosperous", "record",
        "rewarding", "robust", "satisfactory", "solid", "strength", "strong",
        "stronger", "strongest", "success", "successful", "successfully",
        "superior", "surpass", "surpassed", "upside", "win",
    },
    "negative": {
        "adverse", "adversely", "against", "bad", "bankrupt", "bankruptcy",
        "breach", "claim", "claims", "close", "closed", "closing", "concern",
        "concerns", "contraction", "crisis", "critical", "damage", "damaged",
        "damages", "decline", "declined", "declines", "declining", "default",
        "defaulted", "defaults", "deficiency", "deficit", "deteriorate",
        "deteriorated", "deterioration", "difficult", "difficulty", "disaster",
        "disclose", "doubt", "downturn", "drop", "dropped", "failed", "failure",
        "fall", "fell", "force", "fraud", "harm", "harmful", "hurt", "impair",
        "impaired", "impairment", "impairments", "inability", "indicted",
        "inefficient", "infringement", "injury", "insolvent", "investigation",
        "late", "lawsuit", "lawsuits", "litigation", "lose", "loss", "losses",
        "lost", "material", "misleading", "mistake", "negative", "negatively",
        "poor", "problem", "problems", "recall", "recession", "reduce",
        "reduced", "reductions", "restated", "restructure", "restructuring",
        "risk", "risks", "severe", "shortage", "slowdown", "suffered",
        "terminate", "terminated", "termination", "threat", "trouble", "unable",
        "unfavorable", "violation", "volatile", "volatility", "weak",
        "weakened", "weakness", "worse", "worst", "writedown", "wrong",
    },
    "uncertainty": {
        "almost", "ambiguity", "anticipate", "anticipated", "anticipates",
        "anticipating", "anticipation", "appear", "appeared", "appears",
        "approximate", "approximately", "approximated", "assume", "assumed",
        "assumes", "assuming", "assumption", "assumptions", "believe",
        "believed", "believes", "cautious", "conditional", "contingency",
        "contingent", "could", "depend", "depended", "dependence", "dependent",
        "depending", "depends", "doubt", "doubts", "estimate", "estimated",
        "estimates", "estimating", "exposure", "fluctuate", "fluctuated",
        "fluctuates", "fluctuating", "fluctuation", "fluctuations", "hidden",
        "hinges", "imprecise", "likelihood", "may", "maybe", "might",
        "nearly", "occasionally", "perhaps", "possibility", "possible",
        "possibly", "precaution", "predict", "predicted", "predicting",
        "prediction", "predictions", "probability", "probable", "probably",
        "random", "randomize", "reassess", "reassessed", "reassessment",
        "revise", "revised", "rough", "roughly", "seems", "seldom", "somewhat",
        "speculate", "speculated", "speculating", "speculation", "speculative",
        "sudden", "suddenly", "suggest", "suggested", "suggests", "susceptible",
        "tentative", "tentatively", "uncertain", "uncertainly", "uncertainties",
        "uncertainty", "unclear", "unconfirmed", "undecided", "unfamiliar",
        "unforeseen", "unknown", "unplanned", "unpredictable", "unproven",
        "untested", "unusual", "vague", "variability", "variable", "variably",
        "vary", "varying", "volatile", "volatility",
    },
    "litigious": {
        "accusation", "accusations", "acquit", "acquittal", "acquitted",
        "adjudicate", "adjudicated", "allegation", "allegations", "allege",
        "alleged", "allegedly", "alleges", "appeal", "appealed", "appealing",
        "appeals", "appellate", "arbitrate", "arbitrated", "arbitration",
        "arbitrator", "attorney", "attorneys", "bailiff", "claim", "claimant",
        "claimed", "claims", "codified", "complainant", "complaint",
        "complaints", "contempt", "convict", "convicted", "conviction",
        "counsel", "counterclaim", "counterclaims", "court", "courts",
        "criminal", "criminally", "damages", "defend", "defendant",
        "defendants", "defended", "defense", "deposition", "discovery",
        "dismissed", "docket", "enjoin", "enjoined", "felony", "forbearance",
        "foreclosed", "foreclosure", "fraud", "fraudulent", "hearing",
        "indicted", "indictment", "infringe", "infringed", "infringement",
        "injunction", "judge", "judges", "judgment", "judicial", "juries",
        "jurisdiction", "juror", "jurors", "jury", "justice", "law", "lawful",
        "lawsuit", "lawsuits", "lawyer", "lawyers", "legal", "legally",
        "liability", "liable", "libel", "litigant", "litigate", "litigated",
        "litigation", "mediate", "mediation", "misconduct", "motion",
        "negligence", "negligent", "offense", "overrule", "overruled",
        "penalties", "penalty", "petition", "petitioned", "plaintiff",
        "plaintiffs", "plea", "pleaded", "pleading", "prosecute", "prosecuted",
        "prosecution", "regulate", "regulated", "regulation", "regulations",
        "regulatory", "remand", "settle", "settled", "settlement",
        "settlements", "statute", "statutes", "statutory", "subpoena",
        "sue", "sued", "sues", "suing", "summon", "suspend", "suspended",
        "testify", "testified", "testimony", "tort", "tribunal", "trial",
        "trials", "verdict", "violate", "violated", "violation", "violations",
        "witness", "witnesses",
    },
    "constraining": {
        "abide", "abiding", "binding", "bound", "commitment", "commitments",
        "committed", "compel", "compelled", "compelling", "compliance",
        "compliant", "comply", "conditional", "conformance", "conformity",
        "constrain", "constrained", "constraining", "constraint", "constraints",
        "covenant", "covenants", "demand", "demanded", "demanding", "directive",
        "disallow", "disallowed", "enforce", "enforced", "enforcement",
        "forbid", "forbidden", "force", "forced", "forcing", "imposed",
        "imposes", "imposing", "indenture", "indentures", "limit", "limitation",
        "limitations", "limited", "limiting", "mandate", "mandated", "mandates",
        "mandating", "mandatory", "must", "necessitate", "necessitated",
        "necessity", "obligate", "obligated", "obligation", "obligations",
        "obliged", "oblige", "precluded", "precludes", "precluding",
        "prohibit", "prohibited", "prohibition", "prohibits", "require",
        "required", "requirement", "requirements", "requires", "requiring",
        "restrain", "restrained", "restraint", "restricted", "restriction",
        "restrictions", "restrictive", "strict", "strictly", "stringent",
        "subject", "unconditional", "violate", "violated", "violation",
    },
    "strong_modal": {
        "always", "best", "clearly", "definitely", "definitively",
        "highest", "lowest", "must", "never", "strongly", "undisputed",
        "undoubtedly", "unequivocally", "uniquely", "unparalleled",
        "unsurpassed", "will",
    },
    "weak_modal": {
        "almost", "apparently", "appeared", "appears", "conceivable",
        "could", "depend", "depended", "dependent", "depending", "may",
        "maybe", "might", "nearly", "occasionally", "perhaps", "possibly",
        "seldom", "sometimes", "somewhat", "suggest", "suggested",
    },
}


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------
_SRAF_MASTER_URL = (
    "https://sraf.nd.edu/wp-content/uploads/2022/09/"
    "Loughran-McDonald_MasterDictionary_1993-2021.csv"
)

_CATEGORY_COLUMNS = {
    "positive": "Positive",
    "negative": "Negative",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
    "constraining": "Constraining",
    "strong_modal": "Strong_Modal",
    "weak_modal": "Weak_Modal",
}


def _try_pysentiment2() -> Optional[Dict[str, set]]:
    """Attempt to load LM dict from the pysentiment2 pip package."""
    try:
        import pysentiment2 as ps  # type: ignore
    except Exception:
        return None
    try:
        lm = ps.LM()
        # pysentiment2 exposes .lexicon dict mapping category -> list of words
        lex = getattr(lm, "lexicon", None)
        if not lex:
            return None
        out: Dict[str, set] = {}
        mapping = {
            "Positive": "positive",
            "Negative": "negative",
            "Uncertainty": "uncertainty",
            "Litigious": "litigious",
            "Constraining": "constraining",
            "StrongModal": "strong_modal",
            "WeakModal": "weak_modal",
        }
        for src_key, dst_key in mapping.items():
            words = lex.get(src_key) or lex.get(src_key.lower()) or []
            out[dst_key] = {str(w).lower().strip() for w in words if w}
        if all(len(v) > 0 for v in out.values()):
            return out
    except Exception as exc:
        warnings.warn(f"pysentiment2 load failed: {exc}")
    return None


def _try_sraf_download() -> Optional[Dict[str, set]]:
    """Attempt to download the SRAF master dictionary CSV."""
    try:
        import io
        import urllib.request
    except Exception:
        return None
    try:
        req = urllib.request.Request(
            _SRAF_MASTER_URL,
            headers={"User-Agent": "lm_sentiment/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        warnings.warn(f"SRAF download failed: {exc}")
        return None

    try:
        word_col = None
        for cand in ("Word", "word", "WORD"):
            if cand in df.columns:
                word_col = cand
                break
        if word_col is None:
            return None
        words = df[word_col].astype(str).str.lower().str.strip()

        out: Dict[str, set] = {}
        for dst_key, src_col in _CATEGORY_COLUMNS.items():
            if src_col not in df.columns:
                out[dst_key] = set()
                continue
            flag = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
            mask = flag > 0
            out[dst_key] = set(words[mask].tolist())
        if sum(len(v) for v in out.values()) == 0:
            return None
        return out
    except Exception as exc:
        warnings.warn(f"SRAF parse failed: {exc}")
        return None


def load_lm_dictionary(
    use_cache: bool = True,
    cache_dir: str = "data/cache",
) -> Dict[str, set]:
    """
    Load the Loughran-McDonald financial sentiment dictionary.

    Strategy: cache pickle -> pysentiment2 -> SRAF download -> embedded fallback.
    """
    cache_path = Path(cache_dir) / "lm_dictionary.pkl"

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                obj = pickle.load(fh)
            if isinstance(obj, dict) and "positive" in obj:
                return {k: set(v) for k, v in obj.items()}
        except Exception as exc:
            warnings.warn(f"LM cache load failed: {exc}")

    # Try pysentiment2
    result = _try_pysentiment2()

    # Try SRAF
    if result is None:
        result = _try_sraf_download()

    # Fallback
    if result is None:
        warnings.warn(
            "Using embedded LM fallback dictionary (~50 words/category). "
            "For full quality, install pysentiment2 or ensure SRAF is reachable."
        )
        result = {k: set(v) for k, v in _LM_FALLBACK.items()}

    # Ensure all expected keys exist
    for key in _CATEGORY_COLUMNS.keys():
        result.setdefault(key, set())

    # Cache result
    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump({k: set(v) for k, v in result.items()}, fh)
        except Exception as exc:
            warnings.warn(f"LM cache write failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


_EMPTY_SCORE = {
    "pos_ratio": np.nan,
    "neg_ratio": np.nan,
    "net_tone": np.nan,
    "uncertainty_ratio": np.nan,
    "litigious_ratio": np.nan,
    "constraining_ratio": np.nan,
    "total_words": 0,
}


def score_text_sentiment(
    text: str,
    lm_dict: Dict[str, set],
) -> Dict[str, float]:
    """
    Score a document using Loughran-McDonald categories.

    Short/empty texts (<100 words) return NaN ratios.
    """
    tokens = _tokenize(text)
    total = len(tokens)
    if total < 100:
        out = dict(_EMPTY_SCORE)
        out["total_words"] = total
        return out

    pos_set = lm_dict.get("positive", set())
    neg_set = lm_dict.get("negative", set())
    unc_set = lm_dict.get("uncertainty", set())
    lit_set = lm_dict.get("litigious", set())
    con_set = lm_dict.get("constraining", set())

    pos = neg = unc = lit = con = 0
    for tok in tokens:
        if tok in pos_set:
            pos += 1
        if tok in neg_set:
            neg += 1
        if tok in unc_set:
            unc += 1
        if tok in lit_set:
            lit += 1
        if tok in con_set:
            con += 1

    inv = 1.0 / float(total)
    return {
        "pos_ratio": pos * inv,
        "neg_ratio": neg * inv,
        "net_tone": (pos - neg) * inv,
        "uncertainty_ratio": unc * inv,
        "litigious_ratio": lit * inv,
        "constraining_ratio": con * inv,
        "total_words": total,
    }


# ---------------------------------------------------------------------------
# Panel computation
# ---------------------------------------------------------------------------
def _load_filing_item(cache_path: Path, item: str) -> Optional[str]:
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    text = blob.get(item)
    if text is None:
        # common alternates
        for alt in (item.replace("_", ""), item.upper()):
            text = blob.get(alt)
            if text is not None:
                break
    if text is None:
        return None
    return str(text)


def compute_sentiment_panel(
    filings_index: pd.DataFrame,
    item: str = "item_7",
    lm_dict: Optional[Dict[str, set]] = None,
) -> pd.DataFrame:
    """
    Score each filing in the index on the specified item text.
    """
    if lm_dict is None:
        lm_dict = load_lm_dictionary()

    cols = [
        "ticker", "filing_date", "form",
        "pos_ratio", "neg_ratio", "net_tone",
        "uncertainty_ratio", "litigious_ratio", "constraining_ratio",
        "total_words",
    ]
    if filings_index is None or len(filings_index) == 0:
        return pd.DataFrame(columns=cols)

    req = {"ticker", "filing_date", "cache_path"}
    if not req.issubset(filings_index.columns):
        warnings.warn(
            f"filings_index missing required columns {req - set(filings_index.columns)}"
        )
        return pd.DataFrame(columns=cols)

    rows: List[Dict] = []
    for _, r in filings_index.iterrows():
        cache_path = Path(str(r["cache_path"]))
        text = _load_filing_item(cache_path, item) if cache_path.exists() else None
        if not text:
            scores = dict(_EMPTY_SCORE)
        else:
            scores = score_text_sentiment(text, lm_dict)

        rows.append({
            "ticker": r["ticker"],
            "filing_date": pd.to_datetime(r["filing_date"]),
            "form": r.get("form", ""),
            **scores,
        })

    df = pd.DataFrame(rows, columns=cols)
    return df


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def _xsec_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.astype(np.float32)


def build_lm_sentiment_features(
    cache_dir: str = "data/cache/edgar_text",
    trading_dates: Optional[pd.DatetimeIndex] = None,
    universe: Optional[List[str]] = None,
    forward_fill_days: int = 180,
    publication_lag_days: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end LM sentiment feature build. Returns dict of z-scored float32 panels.
    """
    cache_root = Path(cache_dir)
    index_path = cache_root / "filings_index.csv"

    if not index_path.exists():
        warnings.warn(f"LM sentiment: no filings_index at {index_path}")
        return {}

    try:
        idx = pd.read_csv(index_path)
    except Exception as exc:
        warnings.warn(f"LM sentiment: failed reading filings_index: {exc}")
        return {}

    if len(idx) == 0:
        warnings.warn("LM sentiment: empty filings_index")
        return {}

    if universe is not None:
        idx = idx[idx["ticker"].isin(set(universe))].copy()
        if len(idx) == 0:
            warnings.warn("LM sentiment: no filings for requested universe")
            return {}

    # Normalize cache_path to absolute (relative to cache_root if needed)
    def _resolve(p: str) -> str:
        pp = Path(p)
        if not pp.is_absolute():
            pp = cache_root / pp
        return str(pp)

    if "cache_path" in idx.columns:
        idx["cache_path"] = idx["cache_path"].astype(str).map(_resolve)
    else:
        warnings.warn("LM sentiment: filings_index missing cache_path column")
        return {}

    lm_dict = load_lm_dictionary()
    panel = compute_sentiment_panel(idx, item="item_7", lm_dict=lm_dict)
    if len(panel) == 0:
        return {}

    panel = panel.dropna(subset=["net_tone"])
    if len(panel) == 0:
        warnings.warn("LM sentiment: all filings scored NaN (empty/short Item 7)")
        return {}

    # Sort & compute prior-filing tone change per ticker
    panel = panel.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    panel["prior_net_tone"] = panel.groupby("ticker")["net_tone"].shift(1)
    panel["tone_change"] = panel["net_tone"] - panel["prior_net_tone"]

    # Publication-lagged effective date
    lag = pd.Timedelta(days=int(publication_lag_days))
    panel["effective_date"] = pd.to_datetime(panel["filing_date"]) + lag

    # Build trading-date index
    if trading_dates is None:
        start = panel["effective_date"].min()
        end = panel["effective_date"].max() + pd.Timedelta(days=forward_fill_days + 5)
        trading_dates = pd.bdate_range(start=start, end=end)
    trading_dates = pd.DatetimeIndex(trading_dates)

    tickers = sorted(panel["ticker"].unique().tolist())

    feature_map = {
        "lm_net_tone": "net_tone",
        "lm_uncertainty": "uncertainty_ratio",
        "lm_litigious": "litigious_ratio",
        "lm_tone_change": "tone_change",
    }

    raw_panels: Dict[str, pd.DataFrame] = {}
    for feat_name, col in feature_map.items():
        wide = pd.DataFrame(
            index=trading_dates, columns=tickers, dtype=np.float32
        )

        for tkr, grp in panel.groupby("ticker"):
            grp = grp.dropna(subset=[col]).sort_values("effective_date")
            if len(grp) == 0 or tkr not in wide.columns:
                continue
            s = pd.Series(
                grp[col].astype(np.float32).values,
                index=pd.DatetimeIndex(grp["effective_date"].values),
            )
            s = s[~s.index.duplicated(keep="last")]
            # Reindex to trading dates, forward-fill up to N business days
            aligned = s.reindex(trading_dates, method=None)
            aligned = aligned.ffill(limit=int(forward_fill_days))
            wide[tkr] = aligned.astype(np.float32)

        raw_panels[feat_name] = wide

    # Cross-sectional z-score
    out: Dict[str, pd.DataFrame] = {}
    for feat_name, wide in raw_panels.items():
        out[feat_name] = _xsec_zscore(wide)

    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: Fallback dictionary loads
    lm_dict = load_lm_dictionary(use_cache=False)
    print(f"LM dictionary: {list(lm_dict.keys())}")
    for cat, words in lm_dict.items():
        print(f"  {cat}: {len(words)} words")

    # Test 2: Score a positive vs negative text
    positive_text = """
    The company achieved strong growth in revenue and improved profitability.
    Market leadership is an advantage and we successfully expanded our customer base.
    Operational efficiency gains drove margin expansion.
    """ * 10

    negative_text = """
    The company faces adverse market conditions and is at risk of impairment.
    Weak demand and litigation exposure have caused a decline in our results.
    Bankruptcy risk remains a material concern.
    """ * 10

    pos_scores = score_text_sentiment(positive_text, lm_dict)
    neg_scores = score_text_sentiment(negative_text, lm_dict)

    print(f"\nPositive text scores:")
    print(f"  net_tone: {pos_scores['net_tone']:.4f}")
    print(f"  pos_ratio: {pos_scores['pos_ratio']:.4f}")
    print(f"  neg_ratio: {pos_scores['neg_ratio']:.4f}")

    print(f"\nNegative text scores:")
    print(f"  net_tone: {neg_scores['net_tone']:.4f}")
    print(f"  pos_ratio: {neg_scores['pos_ratio']:.4f}")
    print(f"  neg_ratio: {neg_scores['neg_ratio']:.4f}")

    assert pos_scores['net_tone'] > neg_scores['net_tone'], \
        "Positive text should have higher net_tone than negative text"

    # Test 3: build_lm_sentiment_features with no cache
    features = build_lm_sentiment_features("/nonexistent/dir")
    print(f"\nNo cache features: {len(features)} dict entries (should be 0)")

    print("\nALL TESTS PASSED")
