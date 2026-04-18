"""
Lazy Prices features — Cohen, Malloy, Nguyen (2020 JF).

Hypothesis: when management COPIES last year's 10-K / 10-Q text with few
changes (high similarity), the stock OUTPERFORMS over the following 3-6
months. When text changes substantially (low similarity), the stock
UNDERPERFORMS.

Sign convention (LONG-ONLY TILT):
    HIGH similarity  -->  HIGH z-score  -->  BULLISH
    LOW  similarity  -->  LOW  z-score  -->  BEARISH

This module consumes the filings cache produced by
`src/lazy_prices_downloader.py` (built by a separate agent). If the cache
does not yet exist, every public function degrades gracefully (empty
DataFrames / empty dicts with a warning) rather than crashing.

Output features (all float32, date x ticker cross-sectional z-scores):
    lazy_prices_item7   -- Item 7  (MD&A) similarity signal
    lazy_prices_item1a  -- Item 1A (Risk Factors) similarity signal
    lazy_prices_avg     -- average of the two
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Minimum text length (characters) — filings below this are treated as unreadable.
_MIN_TEXT_CHARS = 100

# Boilerplate phrases that appear verbatim in nearly every SEC filing and
# therefore inflate similarity without carrying signal. Stripped pre-similarity.
_BOILERPLATE_PATTERNS = [
    r"united states securities and exchange commission",
    r"washington,?\s*d\.?\s*c\.?\s*20549",
    r"form\s+10-?k",
    r"form\s+10-?q",
    r"annual report pursuant to section 13 or 15\(d\)",
    r"quarterly report pursuant to section 13 or 15\(d\)",
    r"of the securities exchange act of 1934",
    r"indicate by check mark whether the registrant",
    r"this report on form 10-?k",
    r"this report on form 10-?q",
    r"forward[- ]looking statements",
    r"safe harbor",
    r"table of contents",
]

# Small extra Loughran-McDonald-style stopwords on top of sklearn's english list.
# Kept short and purely additive so sklearn's stop_words='english' remains valid.
_LM_EXTRA_STOPWORDS = {
    "company", "companys", "corporation", "inc", "llc", "ltd",
    "fiscal", "quarter", "quarterly", "annual", "annually",
    "report", "reports", "reported", "reporting",
    "filing", "filed", "files", "filings",
    "registrant", "registrants",
    "thereof", "herein", "hereby", "hereof", "hereunder",
}


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #

def _preprocess_text(text: str) -> str:
    """Lowercase, strip numbers/punctuation, remove SEC boilerplate."""
    if not isinstance(text, str) or len(text) == 0:
        return ""

    t = text.lower()

    # Remove boilerplate phrases before other transforms so patterns still match.
    for pat in _BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t)

    # Strip numbers.
    t = re.sub(r"\d+", " ", t)
    # Strip punctuation (keep word chars and whitespace).
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace.
    t = re.sub(r"\s+", " ", t).strip()

    # Drop the LM extra stopwords.
    if t:
        tokens = [tok for tok in t.split() if tok not in _LM_EXTRA_STOPWORDS]
        t = " ".join(tokens)

    return t


# --------------------------------------------------------------------------- #
# Public: similarity primitive
# --------------------------------------------------------------------------- #

def compute_text_similarity(
    text_a: str,
    text_b: str,
    method: str = "tfidf_cosine",
) -> float:
    """
    Compute similarity between two text documents.

    Methods
    -------
    tfidf_cosine   : TF-IDF vectorization (1-2 grams) + cosine similarity.
                     The Loughran-McDonald / Cohen-Malloy-Nguyen standard.
    jaccard        : word-set Jaccard index. Simple and robust.
    character_ngram: 4-gram character overlap (Kogan et al. 2009).

    Returns
    -------
    float in [0, 1]. 1.0 = identical, 0.0 = no overlap.
    NaN if either side is empty or shorter than _MIN_TEXT_CHARS.
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return float("nan")
    if len(text_a) < _MIN_TEXT_CHARS or len(text_b) < _MIN_TEXT_CHARS:
        return float("nan")

    a = _preprocess_text(text_a)
    b = _preprocess_text(text_b)
    if not a or not b:
        return float("nan")

    if method == "tfidf_cosine":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as e:
            warnings.warn(f"scikit-learn not available: {e}")
            return float("nan")

        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=1,  # pair-wise: min_df=2 would drop every token
        )
        try:
            mat = vec.fit_transform([a, b])
        except ValueError:
            # Empty vocabulary after stopword removal.
            return float("nan")
        sim = cosine_similarity(mat[0], mat[1])[0, 0]
        return float(np.clip(sim, 0.0, 1.0))

    if method == "jaccard":
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return float("nan")
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return float(inter / union) if union else float("nan")

    if method == "character_ngram":
        n = 4

        def _ngrams(s: str) -> set:
            s = s.replace(" ", "")
            if len(s) < n:
                return set()
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ng_a = _ngrams(a)
        ng_b = _ngrams(b)
        if not ng_a or not ng_b:
            return float("nan")
        inter = len(ng_a & ng_b)
        union = len(ng_a | ng_b)
        return float(inter / union) if union else float("nan")

    raise ValueError(f"Unknown similarity method: {method!r}")


# --------------------------------------------------------------------------- #
# Public: index loader
# --------------------------------------------------------------------------- #

def load_filings_index(
    cache_dir: str = "data/cache/edgar_text",
) -> pd.DataFrame:
    """
    Load the master filings index written by the downloader.

    Expected columns:
        ['ticker', 'cik', 'filing_date', 'form', 'accession', 'cache_path']

    Returns an empty DataFrame (with the right columns) if the cache or
    the index file does not exist yet.
    """
    columns = ["ticker", "cik", "filing_date", "form", "accession", "cache_path"]
    cache_path = Path(cache_dir)
    index_path = cache_path / "filings_index.csv"

    if not cache_path.exists() or not index_path.exists():
        warnings.warn(
            f"Lazy Prices cache not found at {index_path}. "
            f"Returning empty filings index."
        )
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(index_path)
    except Exception as e:
        warnings.warn(f"Failed to read filings_index.csv: {e}")
        return pd.DataFrame(columns=columns)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        warnings.warn(
            f"filings_index.csv missing columns {missing}. Returning empty."
        )
        return pd.DataFrame(columns=columns)

    df = df[columns].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["ticker", "filing_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    # BUG FIX: normalize Windows backslash paths to forward slashes.
    # The downloader writes cache_path with OS-native separators; on Bash/Git
    # Bash under Windows, pathlib.Path() fails to resolve backslash paths,
    # silently returning None for every filing and producing a zero-feature
    # panel that matches the baseline bit-for-bit.
    df["cache_path"] = df["cache_path"].astype(str).str.replace("\\", "/", regex=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Internal: safe JSON read
# --------------------------------------------------------------------------- #

def _read_filing_text(cache_path: str, item: str) -> Optional[str]:
    """Return the requested item's text from a cached filing JSON, or None."""
    try:
        # BUG FIX: belt-and-suspenders path normalization. Even if some caller
        # passes a raw Windows-style path (backslashes), convert to forward
        # slashes so pathlib resolves correctly on Unix/Bash environments.
        cache_path = str(cache_path).replace("\\", "/")
        p = Path(cache_path)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    txt = data.get(item) if isinstance(data, dict) else None
    if not isinstance(txt, str) or len(txt) < _MIN_TEXT_CHARS:
        return None
    return txt


# --------------------------------------------------------------------------- #
# Public: similarity panel
# --------------------------------------------------------------------------- #

def compute_similarity_panel(
    filings_index: pd.DataFrame,
    item: str = "item_7",
    method: str = "tfidf_cosine",
    min_prior_days: int = 60,
) -> pd.DataFrame:
    """
    For each ticker, compute similarity between each filing and its PRIOR filing.

    Returns DataFrame with columns:
        ['ticker', 'filing_date', 'similarity_<itemtag>',
         'prior_filing_date', 'prior_days']
    """
    item_tag = item.replace("_", "")  # item_7 -> item7
    sim_col = f"similarity_{item_tag}"
    out_cols = ["ticker", "filing_date", sim_col, "prior_filing_date", "prior_days"]

    if filings_index is None or len(filings_index) == 0:
        return pd.DataFrame(columns=out_cols)

    required = {"ticker", "filing_date", "cache_path"}
    if not required.issubset(filings_index.columns):
        warnings.warn(
            f"compute_similarity_panel: filings_index missing {required - set(filings_index.columns)}"
        )
        return pd.DataFrame(columns=out_cols)

    rows: List[dict] = []
    idx = filings_index.copy()
    idx["filing_date"] = pd.to_datetime(idx["filing_date"], errors="coerce")
    idx = idx.dropna(subset=["filing_date"])

    for ticker, grp in idx.groupby("ticker"):
        grp = grp.sort_values("filing_date").reset_index(drop=True)
        if len(grp) < 2:
            continue

        prev_text: Optional[str] = None
        prev_date: Optional[pd.Timestamp] = None

        for _, row in grp.iterrows():
            cur_text = _read_filing_text(row["cache_path"], item)
            cur_date = row["filing_date"]

            if prev_text is None or cur_text is None or prev_date is None:
                # Advance the "previous" pointer if we have usable text now.
                if cur_text is not None:
                    prev_text = cur_text
                    prev_date = cur_date
                continue

            prior_days = int((cur_date - prev_date).days)
            if prior_days < min_prior_days:
                # Too close: skip but still advance pointer.
                prev_text = cur_text
                prev_date = cur_date
                continue

            sim = compute_text_similarity(prev_text, cur_text, method=method)
            rows.append(
                {
                    "ticker": ticker,
                    "filing_date": cur_date,
                    sim_col: sim,
                    "prior_filing_date": prev_date,
                    "prior_days": prior_days,
                }
            )
            prev_text = cur_text
            prev_date = cur_date

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows, columns=out_cols)
    out = out.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Internal: align panel to trading dates
# --------------------------------------------------------------------------- #

def _align_to_trading_dates(
    panel: pd.DataFrame,
    similarity_column: str,
    trading_dates: pd.DatetimeIndex,
    universe: List[str],
    forward_fill_days: int,
    publication_lag_days: int,
) -> pd.DataFrame:
    """
    Shift filing dates by `publication_lag_days`, forward-fill up to
    `forward_fill_days`, cross-section z-score, clip to [-3, 3].
    """
    if panel is None or len(panel) == 0 or similarity_column not in panel.columns:
        return pd.DataFrame(
            index=trading_dates, columns=universe, dtype=np.float32
        )

    tdates = pd.DatetimeIndex(trading_dates).sort_values().unique()
    universe = [str(t).upper() for t in universe]

    p = panel.copy()
    p["ticker"] = p["ticker"].astype(str).str.upper()
    p = p[p["ticker"].isin(set(universe))]
    if p.empty:
        return pd.DataFrame(index=tdates, columns=universe, dtype=np.float32)

    p["filing_date"] = pd.to_datetime(p["filing_date"])

    # Publication lag in business days.
    p["effective_date"] = p["filing_date"] + pd.tseries.offsets.BDay(
        max(0, int(publication_lag_days))
    )

    # Snap each effective_date to the next available trading date.
    td_arr = tdates.values
    eff_arr = p["effective_date"].values.astype("datetime64[ns]")
    insert_pos = np.searchsorted(td_arr, eff_arr, side="left")
    valid = insert_pos < len(td_arr)
    p = p.loc[valid].copy()
    snapped = td_arr[insert_pos[valid]]
    p["aligned_date"] = pd.to_datetime(snapped)

    # Most recent aligned filing per (date, ticker) wins.
    p = p.sort_values(["ticker", "aligned_date"])
    raw = p.pivot_table(
        index="aligned_date",
        columns="ticker",
        values=similarity_column,
        aggfunc="last",
    )
    raw = raw.reindex(index=tdates, columns=universe)

    # Forward-fill with a decay limit.
    ffill_limit = max(1, int(forward_fill_days))
    filled = raw.ffill(limit=ffill_limit)

    # Cross-sectional z-score per date.
    row_mean = filled.mean(axis=1, skipna=True)
    row_std = filled.std(axis=1, skipna=True).replace(0.0, np.nan)
    z = filled.sub(row_mean, axis=0).div(row_std, axis=0)
    z = z.clip(lower=-3.0, upper=3.0)

    return z.astype(np.float32)


# --------------------------------------------------------------------------- #
# Public: cross-sectional signal
# --------------------------------------------------------------------------- #

def build_lazy_prices_signal(
    filings_index: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    universe: List[str],
    similarity_column: str = "similarity_item7",
    forward_fill_days: int = 180,
    publication_lag_days: int = 2,
) -> pd.DataFrame:
    """
    Build a (trading_date x ticker) z-score DataFrame of the Lazy Prices
    signal using the requested `similarity_column`. Sign convention:
    HIGH similarity -> HIGH z-score -> BULLISH.
    """
    if filings_index is None or len(filings_index) == 0:
        return pd.DataFrame(
            index=pd.DatetimeIndex(trading_dates),
            columns=list(universe),
            dtype=np.float32,
        )

    # Map the column name back to an item key.
    if similarity_column == "similarity_item7":
        item = "item_7"
    elif similarity_column == "similarity_item1a":
        item = "item_1a"
    else:
        # Fallback: trust the caller and assume item_7.
        item = "item_7"

    panel = compute_similarity_panel(filings_index, item=item)
    return _align_to_trading_dates(
        panel=panel,
        similarity_column=similarity_column,
        trading_dates=pd.DatetimeIndex(trading_dates),
        universe=list(universe),
        forward_fill_days=forward_fill_days,
        publication_lag_days=publication_lag_days,
    )


# --------------------------------------------------------------------------- #
# Public: end-to-end feature builder
# --------------------------------------------------------------------------- #

def build_lazy_prices_features(
    cache_dir: str = "data/cache/edgar_text",
    trading_dates: Optional[pd.DatetimeIndex] = None,
    universe: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end: load filings cache, compute similarity panels, produce
    aligned cross-sectional z-score features.

    Produces:
        lazy_prices_item7   (Item 7 — MD&A)
        lazy_prices_item1a  (Item 1A — Risk Factors)
        lazy_prices_avg     (mean of the two)

    Returns an empty dict if no cached filings are available.
    """
    filings_index = load_filings_index(cache_dir)
    if filings_index is None or len(filings_index) == 0:
        warnings.warn(
            "build_lazy_prices_features: no cached filings available — "
            "returning empty feature dict."
        )
        return {}

    if trading_dates is None or universe is None or len(universe) == 0:
        warnings.warn(
            "build_lazy_prices_features: trading_dates / universe not "
            "supplied — returning empty feature dict."
        )
        return {}

    tdates = pd.DatetimeIndex(trading_dates)
    uni = [str(t).upper() for t in universe]

    feat_item7 = build_lazy_prices_signal(
        filings_index=filings_index,
        trading_dates=tdates,
        universe=uni,
        similarity_column="similarity_item7",
    )
    feat_item1a = build_lazy_prices_signal(
        filings_index=filings_index,
        trading_dates=tdates,
        universe=uni,
        similarity_column="similarity_item1a",
    )

    # Average (nan-aware): mean of item7 and item1a where available.
    stacked = np.stack(
        [feat_item7.values.astype(np.float32), feat_item1a.values.astype(np.float32)],
        axis=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_arr = np.nanmean(stacked, axis=0).astype(np.float32)
    feat_avg = pd.DataFrame(
        avg_arr, index=feat_item7.index, columns=feat_item7.columns, dtype=np.float32
    )

    return {
        "lazy_prices_item7": feat_item7.astype(np.float32),
        "lazy_prices_item1a": feat_item1a.astype(np.float32),
        "lazy_prices_avg": feat_avg.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile  # noqa: F401 — referenced in spec

    # Test 1: text similarity functions
    text1 = "The company faces competition from established players. " * 50
    text2 = "The company faces competition from established players. " * 50  # identical
    text3 = "Completely different text about apples and oranges." * 50

    sim_identical = compute_text_similarity(text1, text2, method="tfidf_cosine")
    sim_different = compute_text_similarity(text1, text3, method="tfidf_cosine")
    print(f"Identical texts: {sim_identical:.4f} (should be ~1.0)")
    print(f"Different texts: {sim_different:.4f} (should be < 0.5)")
    assert sim_identical > 0.95, "Identical texts should have similarity ~1.0"
    assert sim_different < 0.5, "Different texts should have low similarity"

    # Test 2: Jaccard
    sim_jac = compute_text_similarity(text1, text2, method="jaccard")
    print(f"Jaccard identical: {sim_jac:.4f}")

    # Test 2b: character n-gram sanity
    sim_cng = compute_text_similarity(text1, text2, method="character_ngram")
    print(f"Char-4gram identical: {sim_cng:.4f}")

    # Test 3: Mock filings_index (not actually consumed — sanity construct only)
    dates = pd.date_range("2020-01-01", periods=5, freq="3MS")
    mock_index = pd.DataFrame(
        {
            "ticker": ["TEST"] * 5,
            "cik": ["0000001"] * 5,
            "filing_date": dates,
            "form": ["10-K", "10-Q", "10-Q", "10-Q", "10-K"],
            "accession": [f"0000001-{i}-000000" for i in range(5)],
            "cache_path": [f"/nonexistent/{i}.json" for i in range(5)],
        }
    )
    assert len(mock_index) == 5

    # load_filings_index should return empty when no cache exists
    missing = load_filings_index("/nonexistent/dir")
    print(f"Missing cache returns: {len(missing)} rows (should be 0)")
    assert len(missing) == 0

    # build_lazy_prices_features should also degrade gracefully
    features = build_lazy_prices_features("/nonexistent/dir")
    print(f"No cache features: {len(features)} dict entries (should be 0)")
    assert len(features) == 0

    print("\nALL TESTS PASSED")
