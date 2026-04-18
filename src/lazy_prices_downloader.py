"""
SEC EDGAR 10-K/10-Q full-text downloader for the Lazy Prices signal.

Cohen-Malloy-Nguyen 2020 JF "Lazy Prices": text similarity between consecutive
10-K/10-Q filings predicts 150-250 bps/yr long-only alpha. This module is the
DATA LAYER — it fetches filings from SEC EDGAR and extracts Item 1A (Risk
Factors) and Item 7 (MD&A) sections for downstream similarity computation.

SEC EDGAR is free but rate-limited to 10 req/sec per IP. We throttle to ~8
req/sec for safety and cache aggressively so we never re-fetch a filing.

Usage:
    # Smoke test (3 tickers, ~15 filings)
    python src/lazy_prices_downloader.py

    # Bulk download from another script
    from src.lazy_prices_downloader import bulk_download_filings
    idx = bulk_download_filings(tickers=[...], max_tickers=50)
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import re
import warnings
from pathlib import Path
from typing import Optional, Dict, List
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "Personal Research alex.hines@example.com"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": None,  # set per-request
}

RATE_LIMIT_SLEEP = 0.125  # ~8 req/sec (SEC allows 10)
TIMEOUT = 30

CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, retries: int = 3) -> Optional[requests.Response]:
    """GET with retry/backoff on 429 and 5xx."""
    host = url.split("/")[2]
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": host}
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                time.sleep(RATE_LIMIT_SLEEP)
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                warnings.warn(f"HTTP {resp.status_code} on {url}, retry in {delay}s")
                time.sleep(delay)
                delay *= 2
                continue
            # 404 / permanent errors
            warnings.warn(f"HTTP {resp.status_code} on {url}, giving up")
            return None
        except requests.RequestException as e:
            warnings.warn(f"Request error on {url}: {e}, retry in {delay}s")
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# CIK map
# ---------------------------------------------------------------------------

def get_cik_map(
    use_cache: bool = True,
    cache_dir: str = "data/cache",
) -> Dict[str, str]:
    """
    Load or fetch the SEC EDGAR ticker -> CIK mapping.

    Source: https://www.sec.gov/files/company_tickers.json
    Returns dict: {TICKER: "0000001234"} (CIK as 10-digit zero-padded string).
    """
    cache_path = Path(cache_dir) / "edgar_text" / "cik_map.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    resp = _http_get(CIK_MAP_URL)
    if resp is None:
        raise RuntimeError(f"Failed to fetch CIK map from {CIK_MAP_URL}")

    raw = resp.json()
    # raw is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    cik_map: Dict[str, str] = {}
    for _, row in raw.items():
        ticker = str(row["ticker"]).upper()
        cik = str(row["cik_str"]).zfill(10)
        # If duplicate, keep the first (usually primary listing)
        if ticker not in cik_map:
            cik_map[ticker] = cik

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cik_map, f)

    return cik_map


# ---------------------------------------------------------------------------
# Filing index
# ---------------------------------------------------------------------------

def fetch_filing_index(
    cik: str,
    forms: List[str] = ["10-K", "10-Q"],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch list of filings for a single CIK from SEC EDGAR submissions API.

    Returns DataFrame with columns:
        ['filing_date', 'accession_number', 'form', 'primary_document', 'primary_doc_url']
    """
    cik_padded = str(cik).zfill(10)
    url = SUBMISSIONS_URL.format(cik=cik_padded)
    resp = _http_get(url)
    if resp is None:
        return pd.DataFrame(columns=["filing_date", "accession_number", "form",
                                     "primary_document", "primary_doc_url"])

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(columns=["filing_date", "accession_number", "form",
                                     "primary_document", "primary_doc_url"])

    df = pd.DataFrame({
        "filing_date": recent.get("filingDate", []),
        "accession_number": recent.get("accessionNumber", []),
        "form": recent.get("form", []),
        "primary_document": recent.get("primaryDocument", []),
    })

    if df.empty:
        return df

    df = df[df["form"].isin(forms)].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    if start is not None:
        df = df[df["filing_date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["filing_date"] <= pd.to_datetime(end)]

    df = df.sort_values("filing_date").reset_index(drop=True)

    cik_int = str(int(cik_padded))
    df["primary_doc_url"] = df.apply(
        lambda r: ARCHIVE_URL.format(
            cik_int=cik_int,
            accession_nodash=str(r["accession_number"]).replace("-", ""),
            primary_doc=r["primary_document"],
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Raw filing fetch
# ---------------------------------------------------------------------------

def fetch_filing_text(
    cik: str,
    accession: str,
    primary_doc: str,
    retries: int = 3,
) -> Optional[str]:
    """
    Download the raw HTML/TXT of a specific filing.

    Returns raw HTML string, or None on failure.
    """
    cik_int = str(int(str(cik).lstrip("0") or "0"))
    accession_nodash = str(accession).replace("-", "")
    url = ARCHIVE_URL.format(
        cik_int=cik_int,
        accession_nodash=accession_nodash,
        primary_doc=primary_doc,
    )
    resp = _http_get(url, retries=retries)
    if resp is None:
        return None
    try:
        return resp.text
    except Exception as e:
        warnings.warn(f"Failed to decode {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Item extraction
# ---------------------------------------------------------------------------

_ITEM_PATTERN = re.compile(
    r"\bItem\s+(\d+[A-Z]?)\b[\.\:\s]",
    flags=re.IGNORECASE,
)


def _html_to_text(raw_html: str) -> str:
    """BeautifulSoup parse -> plain text, normalized whitespace."""
    # Try lxml, fall back to html.parser
    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")

    # Drop scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Normalize entities + whitespace
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _item_order(item: str) -> tuple:
    """Convert '1A' -> (1, 'A'); '7' -> (7, '')."""
    m = re.match(r"(\d+)([A-Z]?)", item.upper())
    if not m:
        return (999, "")
    return (int(m.group(1)), m.group(2))


def extract_items(
    raw_html: str,
    item_numbers: List[str] = ["1A", "7"],
) -> Dict[str, str]:
    """
    Extract Item 1A (Risk Factors) and Item 7 (MD&A) from a 10-K/10-Q.

    Returns dict like {"item_1a": "...", "item_7": "..."}.
    On failure for a given item, returns empty string.
    """
    out = {f"item_{it.lower()}": "" for it in item_numbers}
    if not raw_html:
        return out

    try:
        text = _html_to_text(raw_html)
    except Exception as e:
        warnings.warn(f"HTML parse failure: {e}")
        return out

    if not text:
        return out

    # Find all item header positions
    matches = list(_ITEM_PATTERN.finditer(text))
    if not matches:
        return out

    # Build list of (item_key, start_pos)
    positions = [(m.group(1).upper(), m.start()) for m in matches]

    for target in item_numbers:
        target_u = target.upper()
        target_ord = _item_order(target_u)

        # Filings often contain a table-of-contents reference AND the actual
        # section. The real section is the LAST occurrence (ToC appears first).
        target_hits = [p for p in positions if p[0] == target_u]
        if not target_hits:
            continue

        start_pos = target_hits[-1][1]

        # Find the next item header with a strictly larger ordinal
        end_pos = len(text)
        for item_key, pos in positions:
            if pos <= start_pos:
                continue
            if _item_order(item_key) > target_ord:
                end_pos = pos
                break

        section = text[start_pos:end_pos].strip()
        # Cap section at 500K chars just to avoid pathological filings
        if len(section) > 500_000:
            section = section[:500_000]
        out[f"item_{target.lower()}"] = section

    return out


# ---------------------------------------------------------------------------
# Per-ticker download
# ---------------------------------------------------------------------------

def _filing_cache_path(cache_dir: Path, ticker: str, filing_date: str,
                       form: str, accession: str) -> Path:
    safe_accession = str(accession).replace("/", "_")
    safe_form = str(form).replace("/", "_")
    fname = f"{filing_date}_{safe_form}_{safe_accession}.json"
    return cache_dir / ticker / fname


def download_filings_for_ticker(
    ticker: str,
    cik_map: Dict[str, str],
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    cache_dir: str = "data/cache/edgar_text",
) -> pd.DataFrame:
    """
    End-to-end: for a ticker, fetch filing index, download each filing,
    extract items, save to cache.
    """
    ticker = ticker.upper()
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    ticker_dir = cache_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    cik = cik_map.get(ticker)
    if cik is None:
        warnings.warn(f"{ticker}: no CIK in map, skipping")
        return pd.DataFrame()

    idx = fetch_filing_index(cik, forms=["10-K", "10-Q"], start=start, end=end)
    if idx.empty:
        return pd.DataFrame()

    rows = []
    for _, row in idx.iterrows():
        filing_date = row["filing_date"].strftime("%Y-%m-%d")
        form = row["form"]
        accession = row["accession_number"]
        primary_doc = row["primary_document"]

        out_path = _filing_cache_path(cache_root, ticker, filing_date, form, accession)

        if out_path.exists():
            rows.append({
                "ticker": ticker,
                "cik": cik,
                "filing_date": filing_date,
                "form": form,
                "accession": accession,
                "cache_path": str(out_path),
            })
            continue

        raw = fetch_filing_text(cik, accession, primary_doc)
        if raw is None:
            warnings.warn(f"{ticker} {filing_date} {form}: fetch failed")
            continue

        items = extract_items(raw, item_numbers=["1A", "7"])

        payload = {
            "ticker": ticker,
            "cik": cik,
            "filing_date": filing_date,
            "form": form,
            "accession": accession,
            "primary_doc": primary_doc,
            "item_1a": items.get("item_1a", ""),
            "item_7": items.get("item_7", ""),
        }

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"{ticker} {filing_date}: write failed: {e}")
            continue

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "filing_date": filing_date,
            "form": form,
            "accession": accession,
            "cache_path": str(out_path),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bulk download
# ---------------------------------------------------------------------------

def bulk_download_filings(
    tickers: List[str],
    start: str = "2013-01-01",
    end: str = "2026-03-01",
    cache_dir: str = "data/cache/edgar_text",
    max_tickers: Optional[int] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Bulk downloader across the ticker universe. Resume-aware.

    Returns master filings_index DataFrame.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    master_path = cache_root / "filings_index.csv"

    existing_master = pd.DataFrame()
    if resume and master_path.exists():
        try:
            existing_master = pd.read_csv(master_path, dtype=str)
        except Exception:
            existing_master = pd.DataFrame()

    cik_map = get_cik_map(cache_dir=str(cache_root.parent))

    tickers = [t.upper() for t in tickers]
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    all_rows = []
    for i, ticker in enumerate(tickers):
        try:
            df = download_filings_for_ticker(
                ticker=ticker,
                cik_map=cik_map,
                start=start,
                end=end,
                cache_dir=str(cache_root),
            )
            if not df.empty:
                all_rows.append(df)
            print(f"[{i+1}/{len(tickers)}] {ticker}: {len(df)} filings")
        except Exception as e:
            warnings.warn(f"{ticker}: bulk download error: {e}")
            continue

    if all_rows:
        new_master = pd.concat(all_rows, ignore_index=True)
    else:
        new_master = pd.DataFrame(
            columns=["ticker", "cik", "filing_date", "form", "accession", "cache_path"]
        )

    if not existing_master.empty:
        combined = pd.concat([existing_master, new_master], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["ticker", "filing_date", "form", "accession"], keep="last"
        )
    else:
        combined = new_master

    try:
        combined.to_csv(master_path, index=False)
    except Exception as e:
        warnings.warn(f"Failed to write master index: {e}")

    return new_master


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: CIK map
    cik_map = get_cik_map()
    print(f"CIK map loaded: {len(cik_map)} tickers")
    print(f"AAPL CIK: {cik_map.get('AAPL')}")

    # Step 2: Small bulk download (3 tickers, 2020-2024)
    index_df = bulk_download_filings(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start="2020-01-01",
        end="2024-12-31",
        max_tickers=3,
    )
    print(f"\nDownloaded {len(index_df)} filings")
    if not index_df.empty:
        print(index_df.head())

    # Step 3: Verify item extraction on one filing
    if not index_df.empty:
        first_path = index_df.iloc[0]["cache_path"]
        with open(first_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\nSample filing: {index_df.iloc[0]['ticker']} "
              f"{index_df.iloc[0]['filing_date']}")
        print(f"Item 1A length: {len(data.get('item_1a', ''))} chars")
        print(f"Item 7  length: {len(data.get('item_7', ''))} chars")
        if data.get("item_1a"):
            print(f"Item 1A preview: {data['item_1a'][:200]}...")
