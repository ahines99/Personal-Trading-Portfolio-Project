"""
run_options_setup.py
--------------------
One-shot script for the options data pipeline setup.

Run this AFTER:
  1. Tradier brokerage account is approved + funded
  2. TRADIER_TOKEN env var set
  3. Orats subscribed (one-time $399 with Tradier discount)
  4. ORATS_TOKEN env var set OR ORATS_FTP_DIR points to bulk CSV download

Usage:
    # Step 1: Verify Tradier returns Greeks
    python run_options_setup.py --validate-tradier

    # Step 2: Download Orats historical (one-time, run during 30-day window)
    python run_options_setup.py --download-orats

    # Step 3: Validate consistency between Orats and Tradier (Phase 1 of validation)
    python run_options_setup.py --validate-consistency

    # Step 4: Daily polling of Tradier (run via cron after Orats cancellation)
    python run_options_setup.py --daily-poll
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_validate_tradier(args):
    """Step 1: Verify Tradier API returns Orats-supplied Greeks."""
    from tradier_client import TradierClient
    print("=" * 60)
    print("Step 1: Validate Tradier returns Greeks (with smv_vol field)")
    print("=" * 60)

    client = TradierClient()
    test_tickers = ["AAPL", "MSFT", "SPY"]

    for ticker in test_tickers:
        print(f"\n--- {ticker} ---")
        exp = client.find_target_expiration(ticker, target_dte=30)
        print(f"  Target expiration: {exp}")
        if exp is None:
            print(f"  FAIL: no valid expiration found")
            continue
        chain = client.get_chain(ticker, exp, greeks=True)
        if chain.empty:
            print(f"  FAIL: empty chain")
            continue
        print(f"  Chain shape: {chain.shape}")

        # Check for Orats signature fields
        orats_fields = ["greek_smv_vol", "greek_mid_iv", "greek_delta",
                         "greek_gamma", "greek_theta", "greek_vega"]
        found = [f for f in orats_fields if f in chain.columns]
        missing = [f for f in orats_fields if f not in chain.columns]
        print(f"  Orats fields found: {found}")
        if missing:
            print(f"  WARNING: missing fields: {missing}")

        if "greek_smv_vol" in chain.columns:
            sample = chain["greek_smv_vol"].dropna().head(3).tolist()
            print(f"  Sample smv_vol values: {sample}")
            print(f"  ✅ Orats SMV embedding CONFIRMED")
        elif "greek_mid_iv" in chain.columns:
            sample = chain["greek_mid_iv"].dropna().head(3).tolist()
            print(f"  Sample mid_iv values: {sample}")
            print(f"  ⚠️  No smv_vol (Orats signature), but mid_iv present")


def cmd_download_orats(args):
    """Step 2: One-time bulk download of Orats historical data."""
    from orats_loader import OratsLoader
    from data_loader import load_prices

    print("=" * 60)
    print("Step 2: Download Orats historical data")
    print("=" * 60)

    print("\nLoading our universe...")
    prices = load_prices(
        start=args.start, end=args.end,
        dynamic_universe=True, universe_size=args.universe_size,
        min_price=5.0, min_adv=500_000,
    )
    tickers = list(prices["Close"].columns)
    print(f"Universe: {len(tickers)} tickers")

    if args.ftp_dir:
        loader = OratsLoader(ftp_dir=args.ftp_dir)
        n = loader.import_csv_dump()
        print(f"\nImported {n:,} records from FTP/CSV dump")
    else:
        loader = OratsLoader()  # uses ORATS_TOKEN env var
        results = loader.bulk_download_historical(
            tickers=tickers, start_date=args.start, end_date=args.end,
        )
        print(f"\nDownloaded {len(results)} tickers")

    # Build IV panel and save
    print("\nBuilding unified IV panel...")
    panels = loader.load_iv_panel()
    if panels:
        for field, panel in panels.items():
            print(f"  {field}: {panel.shape}")
        # Save combined panel for fast reload
        cache_path = Path("data/cache/options/iv_panels_orats.pkl")
        pd.to_pickle(panels, cache_path)
        print(f"\nSaved {len(panels)} panels to {cache_path}")


def cmd_validate_consistency(args):
    """Step 3: Phase 1 validation — same-day overlap between Orats and Tradier."""
    from tradier_client import TradierClient
    from orats_loader import OratsLoader

    print("=" * 60)
    print("Step 3: Validate Orats-Tradier consistency (Phase 1)")
    print("=" * 60)

    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "JPM", "WMT", "JNJ", "GE", "AMZN"]
    print(f"\nTest tickers: {test_tickers}")

    # Pull live from Tradier
    print("\nPulling Tradier live snapshots...")
    tradier = TradierClient()
    tradier_data = {}
    for ticker in test_tickers:
        summary = tradier.fetch_ticker_summary(ticker, target_dte=30)
        if summary:
            tradier_data[ticker] = summary

    print(f"  Got {len(tradier_data)} ticker summaries")

    # Load Orats most-recent date for comparison
    print("\nLoading Orats panels (most recent date)...")
    loader = OratsLoader(api_token=os.environ.get("ORATS_TOKEN"))
    panels = loader.load_iv_panel()

    if not panels:
        print("  No Orats data cached — run --download-orats first")
        return

    # Compare iv30 (or atmIvM1) for the most recent date
    iv_field = "iv30" if "iv30" in panels else "atmIvM1"
    iv_panel = panels[iv_field]
    last_date = iv_panel.index.max()
    print(f"\nMost recent Orats date: {last_date.date()}")

    print(f"\n{'Ticker':<8} {'Orats IV30':<12} {'Tradier IV30':<14} {'Diff (vol pts)':<14}")
    print("-" * 50)
    diffs = []
    for ticker in test_tickers:
        orats_iv = iv_panel.loc[last_date, ticker] if ticker in iv_panel.columns else None
        tradier_iv = tradier_data.get(ticker, {}).get("iv30_atm")
        if orats_iv is not None and tradier_iv is not None:
            diff = abs(orats_iv - tradier_iv) * 100  # vol points
            diffs.append(diff)
            print(f"{ticker:<8} {orats_iv:<12.4f} {tradier_iv:<14.4f} {diff:<14.2f}")
        else:
            print(f"{ticker:<8} {str(orats_iv):<12} {str(tradier_iv):<14} N/A")

    if diffs:
        print(f"\nMean diff: {np.mean(diffs):.2f} vol pts")
        print(f"Max diff: {np.max(diffs):.2f} vol pts")
        print(f"\nPass criteria: median diff < 1.0 vol pt, max < 3.0 vol pts")
        if np.median(diffs) < 1.0 and np.max(diffs) < 3.0:
            print("✅ PASS")
        else:
            print("❌ FAIL — investigate methodology divergence")


def cmd_daily_poll(args):
    """Step 4: Daily polling of Tradier (run via cron after Orats cancellation)."""
    from tradier_client import TradierClient
    from data_loader import load_prices

    print("=" * 60)
    print(f"Step 4: Daily Tradier poll [{pd.Timestamp.now()}]")
    print("=" * 60)

    print("\nLoading universe...")
    prices = load_prices(
        start=args.start, end=args.end,
        dynamic_universe=True, universe_size=args.universe_size,
        min_price=5.0, min_adv=500_000,
    )
    tickers = list(prices["Close"].columns)[:args.max_tickers]
    print(f"Polling {len(tickers)} tickers...")

    # Build per-ticker EODHD TTM dividend yield map for accurate borrow30
    # solving and annIdiv fallback (otherwise borrow absorbs the dividend
    # signal as negative borrow → clip floor; annIdiv proxy returns 0 for
    # low-yield names → dividend_surprise signal is dead).
    div_yield_map: dict = {}
    if not getattr(args, "legacy_summary", False):
        try:
            from api_data import load_eodhd_dividends
            divs = load_eodhd_dividends(tickers, use_cache=True)
            close_panel = prices["Close"]
            for tk, ddf in (divs or {}).items():
                if tk not in close_panel.columns:
                    continue
                if ddf is None or ddf.empty or "value" not in ddf.columns:
                    continue
                last_year_div = ddf["value"].tail(4).sum()  # ~TTM (4 quarterly)
                spot = close_panel[tk].dropna().iloc[-1] if close_panel[tk].notna().any() else None
                if spot is not None and spot > 0:
                    div_yield_map[tk] = float(last_year_div / spot)
            print(f"  EODHD dividend yields wired for {len(div_yield_map)}/{len(tickers)} tickers")
        except Exception as e:
            print(f"  WARN: could not load EODHD dividends ({e}); borrow30/annIdiv "
                  f"may be degraded")

    client = TradierClient()
    t0 = time.time()
    # Legacy mode: fast (1 chain/ticker), flat summary → merger must remap
    # SMV mode (default): fetches 3 expirations per ticker and runs through
    # chain_to_smv_summary for a full Orats /cores-equivalent row. Slower
    # (3-4x API calls) but produces all 11 fields options_signals.py needs.
    if getattr(args, "legacy_summary", False):
        panel = client.fetch_universe_panel(
            tickers, target_dte=30, max_workers=8,
            on_progress=lambda d, n: print(
                f"  {d}/{n} ({(time.time()-t0)/d*(n-d):.0f}s remaining)"),
        )
    else:
        panel = client.fetch_universe_smv_panel(
            tickers, target_dtes=[10, 30, 60, 120, 270], max_workers=8,
            dividend_yields=div_yield_map,
            on_progress=lambda d, n: print(
                f"  {d}/{n} ({(time.time()-t0)/d*(n-d):.0f}s remaining)"),
        )
    elapsed = time.time() - t0
    if len(tickers) > 0:
        print(f"\nDone in {elapsed:.0f}s ({elapsed/len(tickers):.2f}s/ticker)")
    print(f"Panel shape: {panel.shape}")
    if not panel.empty:
        # Report field coverage — helps spot when the adapter is degrading
        # (e.g. sparse-strike tickers missing delta buckets).
        fields = ["iv30d", "iv60d", "iv90d", "slope",
                  "dlt5Iv30d", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d",
                  "borrow30", "annIdiv"]
        print("Field coverage:")
        for f in fields:
            if f in panel.columns:
                pct = 100.0 * panel[f].notna().mean()
                print(f"    {f:14s} {pct:5.1f}% of tickers")

    # Save daily snapshot
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    cache_dir = Path("data/cache/options/tradier_daily")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{today}.parquet"
    # Sanitize object-dtype columns (e.g. derivation_notes list) for parquet
    for col in panel.columns:
        if panel[col].dtype == object:
            try:
                panel[col] = pd.to_numeric(panel[col], errors="ignore")
            except Exception:
                pass
        if panel[col].dtype == object:
            panel[col] = panel[col].astype(str)
    panel.to_parquet(out_path)
    print(f"\nSaved snapshot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate-tradier", action="store_true",
                   help="Verify Tradier returns Orats-supplied Greeks")
    p.add_argument("--download-orats", action="store_true",
                   help="Bulk download Orats historical data (one-time)")
    p.add_argument("--validate-consistency", action="store_true",
                   help="Phase 1 validation — Orats vs Tradier same-day")
    p.add_argument("--daily-poll", action="store_true",
                   help="Daily Tradier polling (cron job)")

    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--universe-size", type=int, default=3000)
    p.add_argument("--max-tickers", type=int, default=2000,
                   help="Cap for daily polling (saves API calls)")
    p.add_argument("--legacy-summary", action="store_true",
                   help="Use old flat Tradier-legacy summary (1 chain/ticker) "
                        "instead of the SMV adapter (3 chains/ticker but full "
                        "Orats-equivalent schema). Default: SMV mode.")
    p.add_argument("--ftp-dir", default=None,
                   help="Path to Orats CSV bulk dump (alternative to API)")

    args = p.parse_args()

    if args.validate_tradier:
        cmd_validate_tradier(args)
    elif args.download_orats:
        cmd_download_orats(args)
    elif args.validate_consistency:
        cmd_validate_consistency(args)
    elif args.daily_poll:
        cmd_daily_poll(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
