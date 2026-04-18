"""
run_implied_div_validation.py
-----------------------------
Compares our Tradier+EODHD implied-dividend proxy to Orats `annIdiv` over
the 30-day overlap window before Orats subscription cancellation.

Usage:
    python run_implied_div_validation.py \
        --start 2026-04-16 --end 2026-05-15 \
        --universe data/universe.csv

Decision (printed at end):
    rank_corr_cs >= 0.70  -> ship the proxy
    0.50 <= rank_corr_cs < 0.70 -> blend at half weight
    rank_corr_cs <  0.50  -> drop dividend_surprise signal
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.implied_dividend_proxy import (
    build_dividend_surprise_panel,
    validate_against_orats,
)
from src.orats_loader import OratsLoader
from src.tradier_client import TradierClient
from src.api_data import load_eodhd_dividends


def load_rf_curve(date: pd.Timestamp) -> Dict[int, float]:
    """Load FRED treasury yields for `date` and return {dte_days: rate}.
    Falls back to flat 4.5% if FRED cache missing."""
    fred_cache = Path("data/cache/api/fred_yields.parquet")
    if not fred_cache.exists():
        return {30: 0.045, 90: 0.046, 180: 0.047, 365: 0.048}
    df = pd.read_parquet(fred_cache)
    if date not in df.index:
        date = df.index[df.index <= date].max()
    row = df.loc[date]
    # Map FRED series codes -> approx DTE
    tenor_map = {"DGS1MO": 30, "DGS3MO": 90, "DGS6MO": 180, "DGS1": 365,
                 "DGS2": 730, "DGS5": 1825}
    out = {}
    for code, dte in tenor_map.items():
        if code in row.index and pd.notna(row[code]):
            out[dte] = float(row[code]) / 100.0
    return out or {30: 0.045, 90: 0.046, 180: 0.047, 365: 0.048}


def fetch_full_chain(client: TradierClient, ticker: str,
                     min_dte: int = 30, max_dte: int = 270) -> pd.DataFrame:
    """Fetch ALL expirations between min_dte..max_dte (not just one DTE)."""
    exps = client.get_expirations(ticker)
    if not exps:
        return pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    rows = []
    for exp in exps:
        dte = (pd.Timestamp(exp) - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        ch = client.get_chain(ticker, exp, greeks=False)
        if not ch.empty:
            ch["expiration_date"] = exp
            rows.append(ch)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fetch_spot(client: TradierClient, ticker: str) -> float:
    """Get last quote price."""
    try:
        data = client._get("/markets/quotes", {"symbols": ticker})
        q = data.get("quotes", {}).get("quote", {})
        if isinstance(q, list):
            q = q[0] if q else {}
        return float(q.get("last") or q.get("close") or 0.0)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="overlap start YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="overlap end YYYY-MM-DD")
    ap.add_argument("--universe", required=True,
                    help="CSV with column 'ticker' for universe to validate")
    ap.add_argument("--max-tickers", type=int, default=200,
                    help="limit for fast iteration (default 200)")
    ap.add_argument("--out", default="results/implied_div_validation.json")
    args = ap.parse_args()

    # 1) Load Orats annIdiv panel (target)
    loader = OratsLoader(api_token=os.environ.get("ORATS_TOKEN"))
    panels = loader.load_iv_panel()
    if "annIdiv" not in panels:
        print("ERROR: annIdiv panel missing from Orats cache. Run "
              "run_orats_full_download.py first.")
        sys.exit(1)
    orats = panels["annIdiv"]
    orats = orats.loc[args.start:args.end]
    print(f"[validation] Orats annIdiv: {orats.shape}")

    # 2) Pick universe (intersection of file + Orats columns)
    uni = pd.read_csv(args.universe)["ticker"].tolist()
    uni = [t for t in uni if t in orats.columns][: args.max_tickers]
    print(f"[validation] Universe: {len(uni)} tickers")

    # 3) Load EODHD dividends once
    divs = load_eodhd_dividends(uni, start="2023-01-01", use_cache=True)

    # 4) For each overlap date, snapshot Tradier chains + spots
    #    NOTE: Tradier only serves CURRENT chains. So in practice this script
    #    must be RUN DAILY during the overlap to accumulate snapshots; we
    #    stash each day's panel under data/cache/implied_div_proxy/.
    snap_dir = Path("data/cache/implied_div_proxy")
    snap_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    snap_path = snap_dir / f"snapshot_{today.strftime('%Y%m%d')}.parquet"

    if not snap_path.exists():
        client = TradierClient(token=os.environ["TRADIER_TOKEN"])
        rf = load_rf_curve(today)
        chains: Dict[str, pd.DataFrame] = {}
        spots: Dict[str, float] = {}
        for i, t in enumerate(uni):
            chains[t] = fetch_full_chain(client, t)
            spots[t] = fetch_spot(client, t)
            if (i + 1) % 25 == 0:
                print(f"  [tradier] {i+1}/{len(uni)} chains pulled")
        s = build_dividend_surprise_panel(
            chains, divs, spots, rf, as_of=today, target_min_dte=60,
        )
        s.to_frame(name=str(today.date())).to_parquet(snap_path)
        print(f"[validation] Snapshot written: {snap_path}")
    else:
        print(f"[validation] Reusing snapshot: {snap_path}")

    # 5) Aggregate every snapshot we have so far into ours_panel
    snaps = []
    for f in sorted(snap_dir.glob("snapshot_*.parquet")):
        snaps.append(pd.read_parquet(f))
    ours = pd.concat(snaps, axis=1).T
    ours.index = pd.to_datetime(ours.index)
    ours = ours.sort_index()
    print(f"[validation] Ours panel: {ours.shape} ({len(snaps)} snapshots)")

    # 6) Validate
    result = validate_against_orats(ours, orats,
                                    ship_threshold=0.70, blend_threshold=0.50)
    print("\n=== Validation Result ===")
    for k, v in result.to_dict().items():
        print(f"  {k:18s}: {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\n[validation] Wrote {args.out}")
    print(f"\nDECISION: {result.decision.upper()}")


if __name__ == "__main__":
    main()
