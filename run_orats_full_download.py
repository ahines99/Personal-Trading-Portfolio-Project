"""Full Orats historical download — runs all 6 available endpoints.

API budget: ~9,000 calls (45% of 20K monthly limit)
Wall clock: ~60-90 min (rate limit is 1000/min, we sleep 0.6s between)

Endpoints downloaded:
  /hist/cores      — 213-field SMV summary (most signals)
  /hist/summaries  — alt summary fields (borrow, dividend, IV term structure)
  /hist/ivrank     — IV rank pre-computed
  /hist/dailies    — daily OHLCV with options-aware adjustments
  /hist/hvs        — multi-window historical volatility
  /hist/earnings   — earnings dates with implied moves

Skipped:
  /hist/divs       — 403 not in $99 plan (use EODHD dividend data instead)

Usage:
    export ORATS_TOKEN="..."
    python run_orats_full_download.py [--max-tickers 1500] [--endpoints cores,ivrank]
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
import pandas_compat  # noqa: F401

from data_loader import load_prices
from orats_loader import OratsLoader

# Endpoints confirmed working with $99 Delayed Data plan
ENDPOINTS = ["cores", "summaries", "ivrank", "dailies", "hvs", "earnings"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tickers", type=int, default=1500,
                   help="Top N tickers by ADV to download (default 1500)")
    p.add_argument("--endpoints", default=",".join(ENDPOINTS),
                   help="Comma-separated endpoints to download")
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--budget", type=int, default=18000,
                   help="Max API calls (safety: leave 2K buffer of 20K)")
    p.add_argument("--sleep-sec", type=float, default=0.65,
                   help="Sleep between calls (~92 calls/min, well under 1000/min limit)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-cached tickers per endpoint")
    args = p.parse_args()

    if not os.environ.get("ORATS_TOKEN"):
        print("ERROR: Set ORATS_TOKEN env var first")
        sys.exit(1)

    endpoints = [e.strip() for e in args.endpoints.split(",")]
    print("=" * 70)
    print(f"ORATS FULL DOWNLOAD")
    print("=" * 70)
    print(f"Endpoints: {endpoints}")
    print(f"Universe: top {args.max_tickers} tickers by ADV")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Budget cap: {args.budget} calls (of 20K monthly)")
    print()

    # Load universe
    print("[1/3] Loading universe from data_loader...")
    prices = load_prices(
        start=args.start, end=args.end,
        dynamic_universe=True, universe_size=3000,
        min_price=5.0, min_adv=500_000,
    )
    # Get top N by 60d median dollar volume
    close = prices["Close"]
    volume = prices["Volume"]
    dollar_vol = (close * volume).rolling(60).mean()
    avg_dv = dollar_vol.mean(axis=0).dropna().sort_values(ascending=False)
    tickers = avg_dv.head(args.max_tickers).index.tolist()
    print(f"   Top {len(tickers)} tickers by 60d ADV")
    print(f"   Sample top 5: {tickers[:5]}")

    # Init loader
    loader = OratsLoader()

    # Track progress
    manifest_path = Path("data/cache/options/orats_download_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"   Resuming from manifest: {sum(len(v) for v in manifest.values())} entries")
    else:
        manifest = {ep: [] for ep in endpoints}

    # Calculate budget
    total_to_do = sum(
        len([t for t in tickers if t not in manifest.get(ep, [])])
        for ep in endpoints
    )
    print(f"   Total calls needed: {total_to_do} (budget {args.budget})")
    if total_to_do > args.budget:
        print(f"   WARNING: exceeds budget. Will stop at {args.budget}.")

    # Download
    print()
    print("[2/3] Downloading...")
    api_calls_made = 0
    t0 = time.time()

    for ep in endpoints:
        print(f"\n--- Endpoint: /hist/{ep} ---")
        ep_done = set(manifest.get(ep, []))
        ep_pending = [t for t in tickers if t not in ep_done]
        print(f"   {len(ep_pending)} tickers pending (skip {len(ep_done)} already done)")

        for i, ticker in enumerate(ep_pending):
            if api_calls_made >= args.budget:
                print(f"   STOPPING: budget exhausted ({api_calls_made}/{args.budget})")
                break
            df = loader.fetch_batch_history(
                [ticker], endpoint=ep,
                start_date=args.start, end_date=args.end,
            )
            api_calls_made += 1
            if not df.empty:
                manifest[ep].append(ticker)
            time.sleep(args.sleep_sec)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                eta = elapsed / api_calls_made * (total_to_do - api_calls_made)
                print(f"   [{i+1}/{len(ep_pending)}] {ticker}: {len(df)} rows | "
                      f"calls={api_calls_made}/{args.budget} | "
                      f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
                # Save manifest every 25 tickers
                manifest_path.write_text(json.dumps(manifest, indent=2))

        # Final manifest save per endpoint
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Build unified IV panels
    print()
    print("[3/3] Building unified IV panels...")
    panels = loader.load_iv_panel()
    if panels:
        for field, panel in list(panels.items())[:10]:
            print(f"   {field}: {panel.shape}")
        cache_path = Path("data/cache/options/iv_panels_orats.pkl")
        pd.to_pickle(panels, cache_path)
        print(f"\n   Saved {len(panels)} panels to {cache_path}")

    print()
    print("=" * 70)
    print(f"DONE. Made {api_calls_made} API calls in {time.time()-t0:.0f}s")
    print(f"Manifest saved to {manifest_path}")
    print("=" * 70)
    print(f"\nNext step: enable in backtest with --use-options-signals")


if __name__ == "__main__":
    main()
