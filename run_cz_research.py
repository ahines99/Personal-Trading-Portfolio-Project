"""Chen & Zimmermann Open Source Asset Pricing — Signal Discovery Research.

Standalone research script (NOT wired into the live pipeline).
Downloads the C&Z signed_predictors dataset, maps CRSP permnos to tickers,
computes rank IC for all 209 signals against our forward returns, and
reports which novel signals are worth building from live data sources.

Usage:
    python run_cz_research.py                    # full run
    python run_cz_research.py --skip-download    # reuse cached C&Z data
    python run_cz_research.py --top-n 30         # show top 30 signals
"""
import argparse
import os
import sys
import zipfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CZ_DIR = Path("data/cache/chen_zimmermann")
CZ_ZIP = CZ_DIR / "signed_predictors_dl_wide.zip"
CZ_CSV = CZ_DIR / "signed_predictors_dl_wide.csv"
CZ_PARQUET = CZ_DIR / "signed_predictors_dl_wide.parquet"
PERMNO_MAP_CACHE = CZ_DIR / "permno_ticker_map.parquet"
RESULTS_DIR = Path("results/_cz_research")
RESULTS_FILE = RESULTS_DIR / "cz_signal_ic.csv"
OVERLAP_FILE = RESULTS_DIR / "cz_overlap_map.csv"
NOVEL_FILE = RESULTS_DIR / "cz_novel_signals.csv"

# ---------------------------------------------------------------------------
# Known overlap: C&Z signal name -> our feature name(s)
# Built from manual comparison of C&Z SignalDoc categories vs our features.
# Conservative: only map clear 1:1 matches.
# ---------------------------------------------------------------------------
CZ_TO_OUR_FEATURES = {
    # Momentum
    "Mom12m": ["mom_126d", "mom_12_1"],
    "Mom6m": ["mom_126d"],
    "Mom1m": ["mom_21d"],
    "MomRev": ["ret_reversal_1d", "ret_reversal_2d"],
    "STreversal": ["ret_reversal_1d", "zscore_rev_20d"],
    # Value
    "BM": ["book_to_market_signal"],
    "BM_ia": ["book_to_market_signal"],
    "EP": ["earnings_yield_signal"],
    "CFP": ["fcf_yield_signal"],
    "SP": ["sales_yield_signal"],
    "EBM": ["ebit_ev_signal"],
    # Profitability
    "GP": ["gross_profitability_signal"],
    "ROE": ["roe_signal"],
    "ROA": ["q_roa"],
    "PM": ["q_net_margin"],
    "OperProf": ["operating_profitability_signal"],
    "GrGMToGrSales": ["gross_margin_signal"],
    "cash": ["cash_based_op_prof_signal"],
    "roaq": ["q_roe"],
    # Investment / Growth
    "AssetGrowth": ["asset_growth_signal"],
    "Investment": ["asset_growth_signal"],
    "ShareIss1Y": ["net_share_issuance_signal"],
    "ShareIss5Y": ["net_share_issuance_signal"],
    "NOA": ["net_operating_assets_signal"],
    # Accruals
    "Accruals": ["q_accruals"],
    "PctAcc": ["q_accruals"],
    # Leverage / Distress
    "Leverage": ["leverage_signal"],
    "OScore": ["chs_distress_signal"],
    "ZScore": ["altman_z_signal"],
    # Volatility / Risk
    "IdioVol": ["idiovol_21d"],
    "BetaArbitrage": ["mkt_beta_63d"],
    "VolSD": ["rvol_21d", "vol_of_vol"],
    "betaVIX": ["mkt_beta_63d"],
    # Size
    "Size": ["log_mcap_z"],
    # Earnings Surprise
    "SUE": ["earnings_surprise_signal"],
    "EarningsSurprise": ["earnings_surprise_signal"],
    # Dividends
    "DivYield": ["dividend_yield_signal"],
    "DivInit": ["dividend_yield_signal"],
    # Short Interest
    "ShortInterest": ["finra_si_level_signal", "short_ratio_signal"],
    # Analyst
    "NumEarnIncrease": ["analyst_revision_signal"],
    "ForecastDispersion": ["analyst_coverage_signal"],
    # Piotroski
    "Fscore": ["piotroski_f_score_signal"],
    # Liquidity
    "Illiquidity": ["amihud_21d"],
    # Skewness
    "IdioSkew": ["realized_skew_21"],
    "Skew": ["realized_skew_63"],
}


def parse_args():
    p = argparse.ArgumentParser(description="C&Z Signal Discovery Research")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download, use cached C&Z data")
    p.add_argument("--top-n", type=int, default=40,
                   help="Number of top novel signals to report")
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--forward-window", type=int, default=7,
                   help="Forward return horizon in trading days")
    p.add_argument("--min-coverage", type=float, default=0.50,
                   help="Min fraction of our universe matched per date (0-1)")
    return p.parse_args()


# ===================================================================
# STEP 1: Download C&Z data
# ===================================================================
def download_cz_data():
    """Download signed_predictors_dl_wide.zip from Google Drive."""
    CZ_DIR.mkdir(parents=True, exist_ok=True)

    if CZ_PARQUET.exists():
        print(f"[cz] Parquet cache exists: {CZ_PARQUET}")
        return

    if not CZ_ZIP.exists() and not CZ_CSV.exists():
        print("[cz] === MANUAL DOWNLOAD REQUIRED ===")
        print("[cz] The C&Z dataset (~1.6 GB) must be downloaded manually from:")
        print("[cz]   https://www.openassetpricing.com/data/")
        print("[cz] Download 'signed_predictors_dl_wide.zip' and place it at:")
        print(f"[cz]   {CZ_ZIP.resolve()}")
        print("[cz]")
        print("[cz] (Google Drive blocks automated downloads for large files)")
        sys.exit(1)

    if CZ_ZIP.exists() and not CZ_CSV.exists():
        print(f"[cz] Extracting {CZ_ZIP} ...")
        with zipfile.ZipFile(CZ_ZIP, "r") as zf:
            zf.extractall(CZ_DIR)
        print("[cz] Extraction complete.")

    if CZ_CSV.exists() and not CZ_PARQUET.exists():
        print("[cz] Converting CSV to Parquet for faster reloads ...")
        df = pd.read_csv(CZ_CSV, low_memory=False)
        df.to_parquet(CZ_PARQUET, index=False)
        print(f"[cz] Saved {CZ_PARQUET} ({len(df):,} rows, {len(df.columns)} cols)")


def load_cz_data(start: str, end: str) -> pd.DataFrame:
    """Load C&Z data, filter to date range, return DataFrame."""
    print("[cz] Loading C&Z data ...")
    if CZ_PARQUET.exists():
        df = pd.read_parquet(CZ_PARQUET)
    elif CZ_CSV.exists():
        df = pd.read_csv(CZ_CSV, low_memory=False)
    else:
        raise FileNotFoundError(f"No C&Z data found at {CZ_DIR}")

    # Convert yyyymm to datetime (end of month)
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    signal_cols = [c for c in df.columns if c not in ("permno", "yyyymm", "date")]
    print(f"[cz] Loaded: {len(df):,} rows, {len(signal_cols)} signals, "
          f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

    return df


# ===================================================================
# STEP 2: Build permno -> ticker crosswalk
# ===================================================================
def build_permno_ticker_map_via_std_security_code() -> pd.DataFrame:
    """Build permno→ticker via the Std_Security_Code 3-step chain.

    Chain: permno → gvkey (via gvkey_permco_permno.pq)
         → ISIN  (via isin/gvkey.pq)
         → ticker (via isin/ticker.pq)
    """
    import io, requests

    base = "https://github.com/Wenzhi-Ding/Std_Security_Code/raw/main"
    files = {
        "permno_gvkey": f"{base}/other/gvkey_permco_permno.pq",
        "gvkey_isin":   f"{base}/isin/gvkey.pq",
        "isin_ticker":  f"{base}/isin/ticker.pq",
    }

    parts = {}
    for name, url in files.items():
        print(f"[cz]   Downloading {name} ...")
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            print(f"[cz]   FAILED {name}: HTTP {resp.status_code}")
            return pd.DataFrame(columns=["permno", "ticker"])
        df = pd.read_parquet(io.BytesIO(resp.content))
        df.columns = [c.lower() for c in df.columns]
        print(f"[cz]     {len(df):,} rows, cols={list(df.columns)}")
        parts[name] = df

    # Step 1: permno → gvkey
    pg = parts["permno_gvkey"]
    # Columns: gvkey, linkprim, liid, linktype, lpermno, lpermco, linkdt, linkenddt
    pg = pg[pg["linktype"].isin(["LU", "LC"])]
    pg = pg[pg["linkprim"].isin(["P", "C"])]
    # Keep most recent link per permno
    if "linkenddt" in pg.columns:
        pg = pg.sort_values("linkenddt", ascending=False, na_position="first")
    pg = pg.drop_duplicates(subset=["lpermno"], keep="first")
    pg = pg[["lpermno", "gvkey"]].rename(columns={"lpermno": "permno"})
    print(f"[cz]   permno→gvkey: {len(pg):,} unique permnos")

    # Step 2: gvkey → isin
    gi = parts["gvkey_isin"]
    # Keep most recent isin per gvkey
    if "update" in gi.columns:
        gi = gi.sort_values("update", ascending=False)
    gi = gi.drop_duplicates(subset=["gvkey"], keep="first")
    gi = gi[["gvkey", "isin"]]
    print(f"[cz]   gvkey→isin: {len(gi):,} unique gvkeys")

    # Step 3: isin → ticker
    it = parts["isin_ticker"]
    if "update" in it.columns:
        it = it.sort_values("update", ascending=False)
    # Filter to US tickers (no exchange suffix or .US)
    it["ticker"] = it["ticker"].astype(str).str.upper().str.strip()
    # Prefer tickers without dots (US format) and remove .US suffix
    it["ticker"] = it["ticker"].str.replace(".US", "", regex=False)
    it = it.drop_duplicates(subset=["isin"], keep="first")
    it = it[["isin", "ticker"]]
    print(f"[cz]   isin→ticker: {len(it):,} unique isins")

    # Chain together
    m1 = pg.merge(gi, on="gvkey", how="inner")
    m2 = m1.merge(it, on="isin", how="inner")
    m2 = m2[["permno", "ticker"]].drop_duplicates(subset=["permno"], keep="first")
    print(f"[cz]   Final permno→ticker chain: {len(m2):,} mappings")

    return m2


def build_permno_ticker_map_via_returns(
    cz_df: pd.DataFrame,
    our_returns: pd.DataFrame,
    min_correlation: float = 0.92,
    min_overlap_months: int = 24,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Fallback: match permnos to tickers via 12-month return correlation.

    Uses C&Z's Mom12m signal (12-month cumulative return, raw not z-scored)
    as the fingerprint. Matches it against rolling 12-month returns from our
    daily price data.
    """
    print(f"[cz]   Computing rolling 12-month return panels ...")
    # C&Z Mom12m panel (already 12-month cumulative returns)
    cz = cz_df[["permno", "yyyymm", "Mom12m"]].dropna(subset=["Mom12m"]).copy()
    cz["yyyymm"] = cz["yyyymm"].astype(int)
    cz_panel = cz.pivot_table(index="yyyymm", columns="permno",
                              values="Mom12m", aggfunc="first")

    # Our daily returns -> rolling 12-month cumulative, sampled month-end
    dr = our_returns.copy()
    dr.index = pd.to_datetime(dr.index)
    monthly_simple = (1.0 + dr).resample("ME").prod() - 1.0
    # Compute 12-month rolling cumulative return: (1+r1)*(1+r2)*...*(1+r12) - 1
    log_monthly = np.log1p(monthly_simple)
    rolling_12m = log_monthly.rolling(window=12).sum()
    monthly = np.expm1(rolling_12m)
    monthly.index = monthly.index.year * 100 + monthly.index.month
    monthly.index.name = "yyyymm"

    common_idx = cz_panel.index.intersection(monthly.index)
    if len(common_idx) < min_overlap_months:
        return pd.DataFrame(columns=["permno", "ticker", "correlation", "n_overlap_months"])

    cz_panel = cz_panel.loc[common_idx]
    our_panel = monthly.loc[common_idx]

    # Replace inf/-inf with NaN, then fill NaN with 0 for matrix math
    cz_panel = cz_panel.replace([np.inf, -np.inf], np.nan)
    our_panel = our_panel.replace([np.inf, -np.inf], np.nan)

    permnos = cz_panel.columns.to_numpy()
    tickers = our_panel.columns.to_numpy()
    T = len(common_idx)
    print(f"[cz]   {len(permnos):,} permnos x {len(tickers):,} tickers, {T} months")

    # Build numpy arrays + masks
    # X: our returns (T, K). Y: C&Z returns (T, P)
    # NaN values become 0 in X/Y (so they contribute nothing to dot products)
    Mx = (~our_panel.isna()).to_numpy().astype(np.float32)   # (T, K)
    My = (~cz_panel.isna()).to_numpy().astype(np.float32)    # (T, P)
    X = our_panel.fillna(0).to_numpy().astype(np.float32)    # (T, K)
    Y = cz_panel.fillna(0).to_numpy().astype(np.float32)     # (T, P)
    X2 = X * X
    Y2 = Y * Y

    # Use BOTH correlation AND mean-absolute-difference as match score.
    # For same-stock matches, MAD should be near zero AND correlation near 1.
    rows = []
    for start in range(0, len(tickers), batch_size):
        end = min(start + batch_size, len(tickers))
        Xb = X[:, start:end]
        X2b = X2[:, start:end]
        Mxb = Mx[:, start:end]

        # Per-pair sums (K_batch, P)
        N_kp   = Mxb.T @ My
        Sx_kp  = Xb.T @ My
        Sy_kp  = Mxb.T @ Y
        Sxy_kp = Xb.T @ Y
        Sx2_kp = X2b.T @ My
        Sy2_kp = Mxb.T @ Y2

        # Sum of squared differences = sum(x^2) + sum(y^2) - 2*sum(x*y)
        # MSE per pair = SSD / N
        SSD_kp = Sx2_kp + Sy2_kp - 2 * Sxy_kp
        with np.errstate(divide="ignore", invalid="ignore"):
            mse_kp = SSD_kp / np.maximum(N_kp, 1)
            rmse_kp = np.sqrt(np.maximum(mse_kp, 0))

            mu_x = Sx_kp / np.maximum(N_kp, 1)
            mu_y = Sy_kp / np.maximum(N_kp, 1)
            cov = Sxy_kp / np.maximum(N_kp, 1) - mu_x * mu_y
            var_x = Sx2_kp / np.maximum(N_kp, 1) - mu_x ** 2
            var_y = Sy2_kp / np.maximum(N_kp, 1) - mu_y ** 2
            denom = np.sqrt(np.maximum(var_x * var_y, 1e-20))
            corr = np.clip(cov / denom, -1.0, 1.0)

        # For SAME-stock matches:
        #   - corr should be very high (>0.85)
        #   - RMSE on 12m returns should be small (<0.10 = 10pp annualized)
        # Composite score: prefer matches with high corr AND low RMSE
        # Score = corr - rmse (high corr, low rmse is best)
        score = corr - rmse_kp

        # Mask
        score = np.where(N_kp >= min_overlap_months, score, -np.inf)

        # For each ticker, find the best permno
        best_p = score.argmax(axis=1)
        best_score = score[np.arange(end - start), best_p]
        best_corr = corr[np.arange(end - start), best_p]
        best_rmse = rmse_kp[np.arange(end - start), best_p]
        best_n = N_kp[np.arange(end - start), best_p]

        for i in range(end - start):
            c = best_corr[i]
            r = best_rmse[i]
            n = best_n[i]
            # Accept if either: very high correlation, OR good corr + low RMSE
            if (c >= 0.95) or (c >= min_correlation and r < 0.20):
                rows.append((int(permnos[best_p[i]]), str(tickers[start + i]),
                             float(c), float(r), int(n)))

        if (start // batch_size) % 5 == 0:
            print(f"[cz]     Batch {start}/{len(tickers)} done")

    out = pd.DataFrame(rows, columns=["permno", "ticker", "correlation", "rmse", "n_overlap_months"])
    # Composite score: correlation - 0.1 * rmse (allows some RMSE for high-vol stocks)
    out["score"] = out["correlation"] - 0.1 * out["rmse"]
    out = (out.sort_values("score", ascending=False)
              .drop_duplicates("permno", keep="first")
              .drop_duplicates("ticker", keep="first")
              .drop(columns=["score"])
              .reset_index(drop=True))
    print(f"[cz]   Returns-based matching: {len(out):,} matches "
          f"(min corr={min_correlation:.2f})")
    return out


def build_permno_ticker_map(cz_df=None, our_returns=None) -> pd.DataFrame:
    """Build permno-to-ticker mapping with two-stage approach:
       1. Std_Security_Code chain (fast, ground-truth via gvkey/ISIN)
       2. Returns-correlation matching (fills gaps using Mom1m fingerprint)
    """
    if PERMNO_MAP_CACHE.exists():
        mapping = pd.read_parquet(PERMNO_MAP_CACHE)
        print(f"[cz] Permno map cache: {len(mapping):,} mappings")
        return mapping

    CZ_DIR.mkdir(parents=True, exist_ok=True)
    print("[cz] === Stage 1: Std_Security_Code chain ===")
    try:
        m_chain = build_permno_ticker_map_via_std_security_code()
    except Exception as e:
        print(f"[cz]   Stage 1 failed: {e}")
        m_chain = pd.DataFrame(columns=["permno", "ticker"])

    # Filter chain mapping to our universe
    if our_returns is not None and len(m_chain) > 0:
        our_set = set(our_returns.columns)
        m_chain_in_universe = m_chain[m_chain["ticker"].isin(our_set)]
        print(f"[cz]   Stage 1 in our universe: {len(m_chain_in_universe):,}/{len(m_chain):,}")
    else:
        m_chain_in_universe = m_chain

    # Stage 2: Returns matching for unmapped permnos
    print("\n[cz] === Stage 2: Returns-correlation matching ===")
    if cz_df is not None and our_returns is not None:
        try:
            mapped_permnos = set(m_chain_in_universe["permno"].astype(int).values) \
                             if len(m_chain_in_universe) > 0 else set()
            mapped_tickers = set(m_chain_in_universe["ticker"].values) \
                             if len(m_chain_in_universe) > 0 else set()

            # Only run on unmapped permnos and unmapped tickers
            cz_unmapped = cz_df[~cz_df["permno"].isin(mapped_permnos)] \
                          if len(mapped_permnos) > 0 else cz_df
            our_unmapped = our_returns[
                [c for c in our_returns.columns if c not in mapped_tickers]
            ]
            print(f"[cz]   Stage 2 working set: {cz_unmapped['permno'].nunique():,} permnos, "
                  f"{len(our_unmapped.columns):,} tickers")

            m_returns = build_permno_ticker_map_via_returns(cz_unmapped, our_unmapped)
        except Exception as e:
            print(f"[cz]   Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
            m_returns = pd.DataFrame(columns=["permno", "ticker"])
    else:
        m_returns = pd.DataFrame(columns=["permno", "ticker"])

    # Combine: chain results take priority
    if len(m_chain_in_universe) > 0 and len(m_returns) > 0:
        m_chain_in_universe = m_chain_in_universe[["permno", "ticker"]].copy()
        m_chain_in_universe["source"] = "chain"
        m_returns_min = m_returns[["permno", "ticker"]].copy()
        m_returns_min["source"] = "returns"
        mapping = pd.concat([m_chain_in_universe, m_returns_min], ignore_index=True)
        mapping = mapping.drop_duplicates(subset=["permno"], keep="first")
    elif len(m_chain_in_universe) > 0:
        mapping = m_chain_in_universe[["permno", "ticker"]].copy()
        mapping["source"] = "chain"
    elif len(m_returns) > 0:
        mapping = m_returns[["permno", "ticker"]].copy()
        mapping["source"] = "returns"
    else:
        # Last-resort: manual override
        manual = CZ_DIR / "permno_ticker_manual.csv"
        if manual.exists():
            mapping = pd.read_csv(manual)
            mapping["source"] = "manual"
        else:
            mapping = pd.DataFrame(columns=["permno", "ticker", "source"])

    if len(mapping) > 0:
        mapping["permno"] = mapping["permno"].astype(int)
        mapping["ticker"] = mapping["ticker"].astype(str).str.upper().str.strip()
        mapping.to_parquet(PERMNO_MAP_CACHE, index=False)
        print(f"\n[cz] Saved permno map cache: {PERMNO_MAP_CACHE}")
        print(f"[cz] Total mappings: {len(mapping):,}")
        if "source" in mapping.columns:
            print(f"[cz] By source: {mapping['source'].value_counts().to_dict()}")

    return mapping


# ===================================================================
# STEP 3: Load our price data & compute forward returns
# ===================================================================
def load_our_returns(start: str, end: str) -> pd.DataFrame:
    """Load daily returns from our pipeline's data_loader."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_loader import load_prices

    print("[cz] Loading price data from our pipeline ...")
    prices = load_prices(
        start=start, end=end,
        dynamic_universe=True, universe_size=3000,
        min_price=5.0, min_adv=500_000,
    )
    returns = prices["Returns"]
    print(f"[cz] Returns: {returns.shape[0]} dates x {returns.shape[1]} tickers")
    return returns


def compute_forward_returns(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute forward cumulative returns (sum of log returns)."""
    fwd = returns.shift(-window).rolling(window).sum()
    fwd.iloc[-window:] = np.nan
    return fwd


# ===================================================================
# STEP 4: Align C&Z signals to our universe & compute IC
# ===================================================================
def align_cz_to_universe(
    cz_df: pd.DataFrame,
    our_tickers: list,
    permno_map: pd.DataFrame,
) -> dict:
    """Map C&Z permno-indexed signals to our ticker-indexed panels.

    Returns dict: {signal_name: DataFrame(date x ticker)}
    """
    # Build permno -> ticker lookup
    pmap = permno_map.set_index("permno")["ticker"].to_dict()

    # Map permnos to tickers in C&Z data
    cz_df = cz_df.copy()
    cz_df["ticker"] = cz_df["permno"].map(pmap)
    matched = cz_df["ticker"].notna().sum()
    total = len(cz_df)
    print(f"[cz] Permno match rate: {matched:,}/{total:,} "
          f"({matched/total*100:.1f}%)")

    # Filter to our universe tickers
    our_set = set(our_tickers)
    cz_df = cz_df[cz_df["ticker"].isin(our_set)]
    unique_matched = cz_df["ticker"].nunique()
    print(f"[cz] Tickers in our universe: {unique_matched:,}/{len(our_set):,} "
          f"({unique_matched/len(our_set)*100:.1f}%)")

    # Pivot each signal to (date x ticker)
    signal_cols = [c for c in cz_df.columns
                   if c not in ("permno", "yyyymm", "date", "ticker")]

    signals = {}
    for col in signal_cols:
        pivot = cz_df.pivot_table(
            index="date", columns="ticker", values=col, aggfunc="first"
        )
        # Drop signals with very few observations
        coverage = pivot.notna().mean().mean()
        if coverage > 0.05:  # At least 5% non-NaN
            signals[col] = pivot

    print(f"[cz] Signals with >5% coverage: {len(signals)}/{len(signal_cols)}")
    return signals


def compute_signal_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    min_stocks: int = 30,
) -> dict:
    """Compute Spearman rank IC between a monthly signal and daily fwd returns.

    Since C&Z signals are monthly, we align each month-end signal date to the
    corresponding fwd return date.
    """
    ics = []
    signal_dates = signal.index

    for date in signal_dates:
        # Find closest trading day in fwd_returns
        if date not in fwd_returns.index:
            # Find nearest trading day (within 5 days)
            mask = (fwd_returns.index >= date - pd.Timedelta(days=5)) & \
                   (fwd_returns.index <= date + pd.Timedelta(days=5))
            candidates = fwd_returns.index[mask]
            if len(candidates) == 0:
                continue
            trade_date = candidates[candidates >= date][0] if any(candidates >= date) \
                else candidates[-1]
        else:
            trade_date = date

        s = signal.loc[date].dropna()
        r = fwd_returns.loc[trade_date].dropna() if trade_date in fwd_returns.index \
            else pd.Series(dtype=float)

        common = s.index.intersection(r.index)
        if len(common) < min_stocks:
            continue

        ic, _ = spearmanr(s[common].values, r[common].values)
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) < 12:  # Need at least 12 months
        return {"mean_IC": np.nan, "IC_IR": np.nan, "pct_positive": np.nan,
                "n_months": len(ics)}

    ic_arr = np.array(ics)
    return {
        "mean_IC": ic_arr.mean(),
        "IC_IR": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
        "pct_positive": (ic_arr > 0).mean(),
        "n_months": len(ics),
    }


# ===================================================================
# STEP 5: Classify signals as overlapping vs novel
# ===================================================================
def classify_signals(signal_names: list) -> tuple:
    """Split C&Z signals into overlapping (we already have) and novel."""
    overlapping = {}
    novel = []
    for name in signal_names:
        if name in CZ_TO_OUR_FEATURES:
            overlapping[name] = CZ_TO_OUR_FEATURES[name]
        else:
            novel.append(name)
    return overlapping, novel


# ===================================================================
# Main
# ===================================================================
def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Chen & Zimmermann Signal Discovery Research")
    print("=" * 70)

    # --- Step 1: Download / load C&Z data ---
    if not args.skip_download:
        download_cz_data()
    cz_df = load_cz_data(args.start, args.end)

    # --- Step 2: Load our returns (needed for permno mapping fallback) ---
    returns = load_our_returns(args.start, args.end)
    fwd_returns = compute_forward_returns(returns, args.forward_window)

    # --- Step 3: Build permno-ticker mapping (needs both C&Z + our returns) ---
    permno_map = build_permno_ticker_map(cz_df=cz_df, our_returns=returns)
    if len(permno_map) == 0:
        print("[cz] FATAL: No permno-ticker mapping available. Exiting.")
        sys.exit(1)

    # --- Step 4: Align C&Z signals to our universe ---
    signals = align_cz_to_universe(cz_df, list(returns.columns), permno_map)

    # --- Step 5: Classify overlap vs novel ---
    overlapping, novel = classify_signals(list(signals.keys()))
    print(f"\n[cz] Signal classification:")
    print(f"  Overlapping with our features: {len(overlapping)}")
    print(f"  Novel (we don't have):         {len(novel)}")
    print(f"  Total with coverage:           {len(signals)}")

    # Save overlap map
    overlap_rows = [{"cz_signal": k, "our_features": ", ".join(v)}
                    for k, v in overlapping.items()]
    pd.DataFrame(overlap_rows).to_csv(OVERLAP_FILE, index=False)
    print(f"[cz] Overlap map saved: {OVERLAP_FILE}")

    # --- Step 6: Compute IC for ALL signals ---
    print(f"\n[cz] Computing rank IC for {len(signals)} signals "
          f"(fwd={args.forward_window}d) ...")
    results = []
    t0 = time.time()

    for i, (name, panel) in enumerate(signals.items()):
        ic_stats = compute_signal_ic(panel, fwd_returns)
        ic_stats["signal"] = name
        ic_stats["is_novel"] = name in novel
        ic_stats["is_overlap"] = name in overlapping
        if name in overlapping:
            ic_stats["our_equivalent"] = ", ".join(overlapping[name])
        else:
            ic_stats["our_equivalent"] = ""

        # Coverage: avg fraction of our universe with non-NaN signal
        coverage = panel.reindex(columns=returns.columns).notna().mean().mean()
        ic_stats["coverage"] = coverage

        results.append(ic_stats)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(signals) - i - 1)
            print(f"  [{i+1}/{len(signals)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"[cz] IC computation done in {elapsed:.0f}s")

    # --- Step 7: Build results table ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("IC_IR", ascending=False)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"[cz] Full IC results saved: {RESULTS_FILE}")

    # --- Step 8: Report top novel signals ---
    novel_df = results_df[results_df["is_novel"]].copy()
    novel_df = novel_df[novel_df["mean_IC"].notna()]
    novel_df = novel_df[novel_df["coverage"] >= args.min_coverage]
    novel_df = novel_df.sort_values("IC_IR", ascending=False).head(args.top_n)
    novel_df.to_csv(NOVEL_FILE, index=False)

    print(f"\n{'='*70}")
    print(f"TOP {min(args.top_n, len(novel_df))} NOVEL SIGNALS (not in our pipeline)")
    print(f"{'='*70}")
    print(f"{'Signal':<30} {'IC':>8} {'IC_IR':>8} {'%pos':>8} {'Cov':>6} {'N_mo':>5}")
    print("-" * 70)
    for _, row in novel_df.iterrows():
        print(f"{row['signal']:<30} {row['mean_IC']:>8.4f} {row['IC_IR']:>8.3f} "
              f"{row['pct_positive']:>7.1%} {row['coverage']:>5.1%} {row['n_months']:>5.0f}")

    # --- Step 9: Report overlapping signal IC comparison ---
    overlap_df = results_df[results_df["is_overlap"]].copy()
    overlap_df = overlap_df[overlap_df["mean_IC"].notna()]
    overlap_df = overlap_df.sort_values("IC_IR", ascending=False)

    print(f"\n{'='*70}")
    print(f"OVERLAPPING SIGNALS (validate our implementations)")
    print(f"{'='*70}")
    print(f"{'C&Z Signal':<25} {'IC':>8} {'IC_IR':>8} {'Our Feature':<35}")
    print("-" * 80)
    for _, row in overlap_df.head(30).iterrows():
        print(f"{row['signal']:<25} {row['mean_IC']:>8.4f} {row['IC_IR']:>8.3f} "
              f"{row['our_equivalent']:<35}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total C&Z signals tested:  {len(results_df)}")
    print(f"Overlapping with ours:     {len(overlapping)}")
    print(f"Novel signals:             {len(novel_df)} with coverage >= {args.min_coverage:.0%}")
    print(f"Novel with IC_IR > 0.10:   {(novel_df['IC_IR'] > 0.10).sum()}")
    print(f"Novel with IC_IR > 0.20:   {(novel_df['IC_IR'] > 0.20).sum()}")
    print(f"")
    print(f"Next step: review {NOVEL_FILE}")
    print(f"For each high-IC novel signal, check SignalDoc.csv to understand")
    print(f"what data it needs, then build it from our live sources (EDGAR/EODHD/price).")
    print(f"")
    print(f"Results: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
