"""
validate_tradier_orats.py
-------------------------
Validation harness for the Tradier -> Orats /cores adapter.

Empirically tests whether our derived metrics from raw Tradier chains match
Orats's published /datav2/cores values within acceptable tolerances during
the ~30-day Orats subscription overlap (subscription ends ~2026-05-16).

Phases:
  1. Single-day single-ticker spot check (AAPL today)
  2. Single-day cross-section (200 tickers across liquidity tiers)
  3. 5-day rolling test (Phase 2 repeated daily, drift detection)
  4. Edge case probing (high-vol, small-cap, pre-earnings, pre-dividend)

Usage:
    export ORATS_TOKEN=...
    export TRADIER_TOKEN=...
    python validate_tradier_orats.py --tickers AAPL,MSFT,SPY --date 2026-04-17
    python validate_tradier_orats.py --phase 2 --universe sp500_top200.csv \
        --date 2026-04-17 --report results/validation/2026-04-17.html

Decision tree (after Phase 2/3):
  - All metrics PASS  -> ship adapter, cancel Orats with confidence
  - Mixed PASS/FAIL   -> ship the passing signals, drop the failing ones
  - Most FAIL         -> investigate methodology bugs OR keep Orats subscribed

Output:
  results/validation/{date}_diffs.csv        per-ticker per-metric
  results/validation/{date}_acceptance.csv   pass/fail summary
  results/validation/{date}_report.html      human-readable report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Reuse existing infrastructure.
sys.path.insert(0, str(Path(__file__).parent / "src"))
from tradier_client import TradierClient                       # noqa: E402
from tradier_orats_adapter import compute_cores_row, _normalize  # noqa: E402


ORATS_API_BASE = "https://api.orats.io/datav2"
RESULTS_DIR = Path(__file__).parent / "results" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Recommended thresholds (vol points or rank correlation).
DEFAULT_THRESHOLDS = {
    # vol-point absolute diff (pass if median < pass, fail if median > fail)
    "iv30d":      {"kind": "abs", "pass_med": 0.010, "fail_med": 0.030, "pass_max": 0.030, "fail_max": 0.060},
    "iv60d":      {"kind": "abs", "pass_med": 0.010, "fail_med": 0.030, "pass_max": 0.030, "fail_max": 0.060},
    # rank-correlation thresholds
    "slope":      {"kind": "rank_corr", "pass": 0.95, "fail": 0.85},
    "dlt25Iv30d": {"kind": "rank_corr", "pass": 0.95, "fail": 0.85},
    "dlt75Iv30d": {"kind": "rank_corr", "pass": 0.95, "fail": 0.85},
    "dlt95Iv30d": {"kind": "rank_corr", "pass": 0.95, "fail": 0.85},
    "borrow30":   {"kind": "rank_corr", "pass": 0.70, "fail": 0.40},
    "annIdiv":    {"kind": "rank_corr", "pass": 0.50, "fail": 0.20},
}

METRICS = list(DEFAULT_THRESHOLDS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Orats fetch (one ticker, one date)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_orats_cores(ticker: str, date: pd.Timestamp, token: str) -> Optional[dict]:
    """Pull /datav2/cores row for one ticker on one trade date."""
    url = f"{ORATS_API_BASE}/cores"
    params = {"token": token, "ticker": ticker, "tradeDate": date.strftime("%Y-%m-%d")}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            # /cores is intraday; fall back to /hist/cores for past dates
            url2 = f"{ORATS_API_BASE}/hist/cores"
            r = requests.get(url2, params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("data", [])
        for row in data:
            row_date = pd.Timestamp(row.get("tradeDate", "")).normalize()
            if row_date == date.normalize():
                return row
        return data[0] if data else None
    except Exception as e:
        print(f"  [orats] {ticker} {date.date()} failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tradier fetch + adapter (one ticker, one date)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_tradier_full_chain(
    client: TradierClient, ticker: str, asof: pd.Timestamp,
    max_dte: int = 120, min_dte: int = 5,
) -> Tuple[pd.DataFrame, float]:
    """Pull all near-term expiries (<=max_dte) and concat into one chain.

    Returns (combined_chain, spot). Raises if no quotes returned.
    """
    exps = client.get_expirations(ticker)
    keep = []
    for e in exps:
        try:
            dte = (pd.Timestamp(e).normalize() - asof.normalize()).days
        except Exception:
            continue
        if min_dte <= dte <= max_dte:
            keep.append(e)
    if not keep:
        return pd.DataFrame(), float("nan")

    parts = []
    for exp in keep:
        df = client.get_chain(ticker, exp, greeks=True)
        if df.empty:
            continue
        df["expiration_date"] = exp
        parts.append(df)
    if not parts:
        return pd.DataFrame(), float("nan")
    chain = pd.concat(parts, ignore_index=True)

    # Spot via /markets/quotes (more accurate than ATM-strike proxy).
    spot = float("nan")
    try:
        r = client._get("/markets/quotes", {"symbols": ticker, "greeks": "false"})
        q = r.get("quotes", {}).get("quote") if r else None
        if isinstance(q, list):
            q = q[0]
        if q and "last" in q and q["last"]:
            spot = float(q["last"])
    except Exception:
        pass
    if not math.isfinite(spot):
        spot = float(chain["strike"].median())
    return chain, spot


def _atm_pair_for_dte(chain_norm: pd.DataFrame, dte: int, spot: float) -> Optional[Tuple[float, float, float]]:
    """For one DTE, return (K, call_mid, put_mid) at strike closest to spot."""
    leg = chain_norm[chain_norm["dte"] == dte].dropna(subset=["bid", "ask"])
    leg = leg[(leg["bid"] > 0) & (leg["ask"] > 0)]
    if leg.empty:
        return None
    leg = leg.assign(_mid=(leg["bid"] + leg["ask"]) / 2.0)
    cand = []
    for k, g in leg.groupby("strike"):
        if {"c", "p"}.issubset(set(g["option_type"])):
            c = g[g["option_type"] == "c"]["_mid"].iloc[0]
            p = g[g["option_type"] == "p"]["_mid"].iloc[0]
            cand.append((abs(k - spot), float(k), float(c), float(p)))
    if not cand:
        return None
    cand.sort()
    _, k, c, p = cand[0]
    return (k, c, p)


def _atm_iv_for_dte(chain_norm: pd.DataFrame, dte: int, spot: float) -> Optional[float]:
    leg = chain_norm[chain_norm["dte"] == dte]
    if leg.empty:
        return None
    ivs = []
    for side in ("c", "p"):
        sub = leg[leg["option_type"] == side]
        if sub.empty:
            continue
        iv = sub.iloc[(sub["strike"] - spot).abs().argsort()[:1]]["iv"].iloc[0]
        if pd.notna(iv) and iv > 0:
            ivs.append(float(iv))
    return float(np.mean(ivs)) if ivs else None


def _variance_interp(t1: float, iv1: float, t2: float, iv2: float, t_target: float) -> float:
    if t1 == t2:
        return iv1
    v1 = iv1 ** 2 * t1
    v2 = iv2 ** 2 * t2
    w = (t_target - t1) / (t2 - t1)
    v = max(v1 + w * (v2 - v1), 1e-12)
    return math.sqrt(v / t_target)


def _bracket(dtes: List[int], target: int) -> Optional[Tuple[int, int]]:
    if not dtes:
        return None
    s = sorted(set(dtes))
    if len(s) == 1:
        return (s[0], s[0])
    below = [d for d in s if d <= target]
    above = [d for d in s if d >= target]
    if below and above:
        return (max(below), min(above))
    return (s[0], s[1]) if not below else (s[-2], s[-1])


def _implied_carry(chain_norm: pd.DataFrame, spot: float, rf: float = 0.045) -> Tuple[Optional[float], Optional[float]]:
    """Solve put-call parity for per-expiry carry q; split into (borrow30, annIdiv).

    Method: q(T) = -ln((C - P + K e^{-rT}) / S) / T. annIdiv = median q at long
    expiries (dte>=60); borrow30 = max(0, q_30 - annIdiv).
    """
    qs: Dict[int, float] = {}
    for dte in sorted(chain_norm["dte"].unique()):
        pair = _atm_pair_for_dte(chain_norm, int(dte), spot)
        if pair is None:
            continue
        k, c, p = pair
        T = max(int(dte), 1) / 365.0
        try:
            arg = (c - p + k * math.exp(-rf * T)) / spot
            if arg <= 0:
                continue
            qs[int(dte)] = -math.log(arg) / T
        except (ValueError, ZeroDivisionError):
            continue
    if not qs:
        return (None, None)
    if len(qs) == 1:
        return (float(next(iter(qs.values()))), 0.0)
    long_qs = [q for d, q in qs.items() if d >= 60]
    ann_idiv = float(np.median(long_qs)) if long_qs else float(min(qs.values()))
    dtes_sorted = sorted(qs)
    qs_sorted = [qs[d] for d in dtes_sorted]
    q30 = float(np.interp(30, dtes_sorted, qs_sorted))
    return (max(q30 - ann_idiv, 0.0), ann_idiv)


def derive_from_tradier(
    chain: pd.DataFrame, spot: float, asof: pd.Timestamp, rf: float = 0.045,
) -> Dict[str, Optional[float]]:
    """Run the full adapter: cores_row + iv30d/iv60d + borrow30 + annIdiv."""
    out: Dict[str, Optional[float]] = {m: None for m in METRICS}
    if chain.empty:
        return out

    # Slope + delta buckets: existing adapter
    try:
        cr = compute_cores_row(chain, target_dte=30, asof=asof)
        for k in ("slope", "dlt25Iv30d", "dlt75Iv30d", "dlt95Iv30d"):
            v = cr.get(k)
            out[k] = float(v) if v is not None and np.isfinite(v) else None
    except Exception as e:
        print(f"  [adapter] cores_row failed: {e}")

    # iv30d / iv60d via variance-interp on ATM IVs per expiry
    norm = _normalize(chain)
    if "expiry" in norm.columns and not norm.empty:
        norm["dte"] = (norm["expiry"] - asof.normalize()).dt.days.clip(lower=1)
        atm = {int(d): _atm_iv_for_dte(norm, int(d), spot) for d in norm["dte"].unique()}
        atm = {d: v for d, v in atm.items() if v is not None}
        for target, key in ((30, "iv30d"), (60, "iv60d")):
            br = _bracket(list(atm.keys()), target)
            if br is None:
                continue
            lo, hi = br
            if lo == hi:
                out[key] = atm[lo]
            else:
                out[key] = _variance_interp(lo / 365.0, atm[lo], hi / 365.0, atm[hi], target / 365.0)

        borrow, ann_idiv = _implied_carry(norm, spot, rf=rf)
        out["borrow30"] = borrow
        out["annIdiv"] = ann_idiv

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: single-day single-ticker
# ─────────────────────────────────────────────────────────────────────────────
def validate_single_day(
    ticker: str, date: pd.Timestamp, orats_token: str, tradier_token: str,
) -> Dict[str, Dict[str, float]]:
    """Returns {metric: {'orats': X, 'ours': Y, 'diff': X-Y, 'pct_diff': (X-Y)/X}}."""
    date = pd.Timestamp(date).normalize()
    orats_row = fetch_orats_cores(ticker, date, orats_token) or {}
    client = TradierClient(token=tradier_token)
    chain, spot = _fetch_tradier_full_chain(client, ticker, date)
    ours = derive_from_tradier(chain, spot, date)

    out: Dict[str, Dict[str, float]] = {}
    for m in METRICS:
        o = orats_row.get(m)
        u = ours.get(m)
        try:
            o = float(o) if o is not None else None
        except (TypeError, ValueError):
            o = None
        diff = (o - u) if (o is not None and u is not None) else None
        pct = (diff / o) if (diff is not None and o not in (None, 0)) else None
        out[m] = {"orats": o, "ours": u, "diff": diff, "pct_diff": pct}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: cross-section
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_tradier_for_ticker(client, ticker, date):
    try:
        chain, spot = _fetch_tradier_full_chain(client, ticker, date)
        return ticker, derive_from_tradier(chain, spot, date)
    except Exception as e:
        print(f"  [tradier] {ticker} failed: {e}")
        return ticker, {m: None for m in METRICS}


def _fetch_orats_for_ticker(ticker, date, token):
    try:
        return ticker, fetch_orats_cores(ticker, date, token) or {}
    except Exception as e:
        print(f"  [orats] {ticker} failed: {e}")
        return ticker, {}


def validate_cross_section(
    tickers: List[str], date: pd.Timestamp, orats_token: str, tradier_token: str,
    max_workers: int = 6,
) -> pd.DataFrame:
    """One row per ticker, columns: orats_<m>, ours_<m>, diff_<m>."""
    date = pd.Timestamp(date).normalize()
    client = TradierClient(token=tradier_token)
    rows: Dict[str, dict] = {t: {} for t in tickers}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_fetch_orats_for_ticker, t, date, orats_token)
                                 for t in tickers]):
            t, row = fut.result()
            for m in METRICS:
                v = row.get(m)
                try:
                    rows[t][f"orats_{m}"] = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    rows[t][f"orats_{m}"] = np.nan

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_fetch_tradier_for_ticker, client, t, date)
                                 for t in tickers]):
            t, derived = fut.result()
            for m in METRICS:
                v = derived.get(m)
                rows[t][f"ours_{m}"] = float(v) if v is not None else np.nan

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    for m in METRICS:
        df[f"diff_{m}"] = df[f"orats_{m}"] - df[f"ours_{m}"]
    df["date"] = date
    return df.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: 5-day rolling
# ─────────────────────────────────────────────────────────────────────────────
def validate_rolling(
    tickers: List[str], end_date: pd.Timestamp, n_days: int,
    orats_token: str, tradier_token: str,
) -> pd.DataFrame:
    """Run cross-section for last n_days trading days; concat results."""
    cal = pd.bdate_range(end=end_date, periods=n_days)
    parts = []
    for d in cal:
        print(f"\n[rolling] {d.date()}  ({len(parts)+1}/{n_days})")
        parts.append(validate_cross_section(tickers, d, orats_token, tradier_token))
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: edge case probing
# ─────────────────────────────────────────────────────────────────────────────
EDGE_BUCKETS = {
    "high_vol":       ["VIX", "GME", "AMC", "MSTR", "SMCI", "TSLA"],
    "low_liquidity":  ["FORD", "TWST", "BLNK", "VICR", "AAOI", "IIPR"],
    "pre_earnings":   [],   # populated dynamically from earnings calendar
    "pre_dividend":   [],   # populated dynamically from div calendar
    "etfs":           ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLK"],
}


def validate_edge_cases(
    date: pd.Timestamp, orats_token: str, tradier_token: str,
    extra_pre_earnings: Optional[List[str]] = None,
    extra_pre_dividend: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run cross-section per edge bucket and return dict of DataFrames."""
    buckets = {k: list(v) for k, v in EDGE_BUCKETS.items()}
    if extra_pre_earnings:
        buckets["pre_earnings"] = extra_pre_earnings
    if extra_pre_dividend:
        buckets["pre_dividend"] = extra_pre_dividend

    out: Dict[str, pd.DataFrame] = {}
    for name, tickers in buckets.items():
        if not tickers:
            continue
        print(f"\n[edge] {name}: {tickers}")
        out[name] = validate_cross_section(tickers, date, orats_token, tradier_token)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance scoring
# ─────────────────────────────────────────────────────────────────────────────
def compute_acceptance_report(
    validation_df: pd.DataFrame,
    thresholds: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    """Returns {metric: {'verdict': PASS|FAIL|MARGINAL, 'stat': float, 'n': int, ...}}."""
    th = thresholds or DEFAULT_THRESHOLDS
    report: Dict[str, dict] = {}
    for m, cfg in th.items():
        o = pd.to_numeric(validation_df.get(f"orats_{m}"), errors="coerce")
        u = pd.to_numeric(validation_df.get(f"ours_{m}"), errors="coerce")
        if o is None or u is None:
            report[m] = {"verdict": "NO_DATA", "stat": None, "n": 0}
            continue
        mask = o.notna() & u.notna()
        n = int(mask.sum())
        if n < 5:
            report[m] = {"verdict": "NO_DATA", "stat": None, "n": n}
            continue
        if cfg["kind"] == "abs":
            d = (o[mask] - u[mask]).abs()
            med, mx = float(d.median()), float(d.max())
            ok_med = med <= cfg["pass_med"]
            bad_med = med >= cfg["fail_med"]
            ok_max = mx <= cfg["pass_max"]
            bad_max = mx >= cfg["fail_max"]
            if ok_med and ok_max:
                v = "PASS"
            elif bad_med or bad_max:
                v = "FAIL"
            else:
                v = "MARGINAL"
            report[m] = {
                "verdict": v, "stat": med, "max": mx, "n": n,
                "pass_med": cfg["pass_med"], "fail_med": cfg["fail_med"],
            }
        else:  # rank_corr
            rc = float(o[mask].rank().corr(u[mask].rank()))
            if rc >= cfg["pass"]:
                v = "PASS"
            elif rc < cfg["fail"]:
                v = "FAIL"
            else:
                v = "MARGINAL"
            report[m] = {
                "verdict": v, "stat": rc, "n": n,
                "pass": cfg["pass"], "fail": cfg["fail"],
            }
    return report


def decision_from_report(report: Dict[str, dict]) -> Tuple[str, Dict[str, str]]:
    """Map per-metric verdicts to ship/drop decisions."""
    per_signal = {}
    for m, r in report.items():
        v = r.get("verdict", "NO_DATA")
        if v == "PASS":
            per_signal[m] = "SHIP"
        elif v == "MARGINAL":
            per_signal[m] = "SHIP_WITH_MONITORING"
        elif v == "FAIL":
            per_signal[m] = "DROP_OR_KEEP_ORATS"
        else:
            per_signal[m] = "INSUFFICIENT_DATA"

    n_pass = sum(1 for v in per_signal.values() if v == "SHIP")
    n_fail = sum(1 for v in per_signal.values() if v == "DROP_OR_KEEP_ORATS")
    if n_fail == 0 and n_pass >= len(per_signal) - 1:
        overall = "ALL_GREEN_CANCEL_ORATS"
    elif n_pass >= n_fail:
        overall = "PARTIAL_SHIP_DROP_FAILING_SIGNALS"
    else:
        overall = "MOSTLY_FAIL_INVESTIGATE_OR_KEEP_ORATS"
    return overall, per_signal


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tradier-Orats Validation {date}</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:2em;max-width:1100px;color:#222}}
h1{{border-bottom:2px solid #444}} h2{{margin-top:1.5em;color:#333}}
table{{border-collapse:collapse;width:100%;margin:0.5em 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
th{{background:#f4f4f4;text-align:center}}
.PASS{{background:#d4edda;color:#155724;font-weight:bold}}
.MARGINAL{{background:#fff3cd;color:#856404;font-weight:bold}}
.FAIL{{background:#f8d7da;color:#721c24;font-weight:bold}}
.NO_DATA{{background:#e2e3e5;color:#383d41}}
.banner{{padding:1em;border-radius:4px;font-size:1.1em;margin:1em 0}}
.banner-green{{background:#d4edda;color:#155724}}
.banner-yellow{{background:#fff3cd;color:#856404}}
.banner-red{{background:#f8d7da;color:#721c24}}
small{{color:#666}}
</style></head><body>
<h1>Tradier -> Orats Adapter Validation</h1>
<p><b>Date:</b> {date} &nbsp; <b>Tickers:</b> {n_tickers} &nbsp;
   <b>Phase:</b> {phase}</p>
<div class="banner banner-{banner_color}"><b>Decision:</b> {overall}</div>

<h2>Acceptance summary</h2>
{acceptance_table}

<h2>Per-signal ship/drop</h2>
{decision_table}

<h2>Spot-check sample (first 25 tickers)</h2>
{sample_table}

<h2>Methodology notes</h2>
<ul>
<li><b>iv30d / iv60d:</b> variance-time interpolation (Demeterfi-Derman-Kamal)
between the two expiries bracketing 30d/60d, using ATM IV per expiry.</li>
<li><b>dlt25/dlt75/dlt95Iv30d:</b> PCHIP interpolation in (call-delta, IV) space
within each expiry, then variance-interp across expiries to 30d.</li>
<li><b>slope:</b> dlt75Iv30d - dlt25Iv30d (Orats call-delta convention).</li>
<li><b>borrow30 / annIdiv:</b> put-call parity carry decomposition.
annIdiv = median carry at expiries >=60d (assumed roughly flat in T);
borrow30 = max(0, q_30 - annIdiv). Coarse split — expected noisier than other metrics.</li>
</ul>

<small>Generated {timestamp}. Thresholds: {threshold_json}</small>
</body></html>
"""


def _verdict_table(report: Dict[str, dict]) -> str:
    rows = []
    for m, r in report.items():
        v = r.get("verdict", "NO_DATA")
        stat = r.get("stat")
        stat_s = f"{stat:.4f}" if isinstance(stat, (int, float)) else "—"
        extra = ""
        if "max" in r:
            extra = f" (max diff {r['max']:.4f})"
        rows.append(f'<tr><td style="text-align:left">{m}</td>'
                    f'<td>{stat_s}{extra}</td><td>{r.get("n", 0)}</td>'
                    f'<td class="{v}">{v}</td></tr>')
    return ('<table><tr><th>Metric</th><th>Statistic</th><th>N</th><th>Verdict</th></tr>'
            + "".join(rows) + '</table>')


def _decision_table(per_signal: Dict[str, str]) -> str:
    rows = "".join(
        f'<tr><td style="text-align:left">{m}</td><td>{d}</td></tr>'
        for m, d in per_signal.items()
    )
    return f'<table><tr><th>Signal</th><th>Action</th></tr>{rows}</table>'


def _sample_table(df: pd.DataFrame, n: int = 25) -> str:
    cols = ["ticker"] + [f"{p}_{m}" for m in METRICS for p in ("orats", "ours")]
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].head(n).round(4)
    return sample.to_html(index=False, classes="sample", border=0)


def write_report(
    df: pd.DataFrame, report: Dict[str, dict], overall: str,
    per_signal: Dict[str, str], date: pd.Timestamp, phase: str, out_path: Path,
) -> None:
    color = {"ALL_GREEN_CANCEL_ORATS": "green",
             "PARTIAL_SHIP_DROP_FAILING_SIGNALS": "yellow",
             "MOSTLY_FAIL_INVESTIGATE_OR_KEEP_ORATS": "red"}.get(overall, "yellow")
    html = HTML_TEMPLATE.format(
        date=pd.Timestamp(date).date(),
        n_tickers=df["ticker"].nunique() if "ticker" in df.columns else 0,
        phase=phase,
        banner_color=color,
        overall=overall,
        acceptance_table=_verdict_table(report),
        decision_table=_decision_table(per_signal),
        sample_table=_sample_table(df),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        threshold_json=json.dumps({k: v for k, v in DEFAULT_THRESHOLDS.items()}),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _load_universe(path: str) -> List[str]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        return df[col].astype(str).str.upper().tolist()
    return [s.strip().upper() for s in p.read_text().splitlines() if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Tradier->Orats adapter vs Orats actuals")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                    help="1=spot, 2=cross-section, 3=rolling, 4=edge cases")
    ap.add_argument("--tickers", default="AAPL",
                    help="Comma-separated tickers (Phase 1/2/4 if no --universe)")
    ap.add_argument("--universe", default=None, help="CSV/TXT file with tickers (Phase 2)")
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--n-days", type=int, default=5, help="Phase 3 trading days")
    ap.add_argument("--report", default=None, help="HTML output path (auto if omitted)")
    ap.add_argument("--orats-token", default=os.environ.get("ORATS_TOKEN"))
    ap.add_argument("--tradier-token", default=os.environ.get("TRADIER_TOKEN"))
    args = ap.parse_args()

    if not args.orats_token or not args.tradier_token:
        print("ERROR: set ORATS_TOKEN and TRADIER_TOKEN env vars (or pass --*-token).")
        return 2

    date = pd.Timestamp(args.date).normalize()
    tickers = (_load_universe(args.universe) if args.universe
               else [t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    print(f"[validate] phase={args.phase} date={date.date()} n_tickers={len(tickers)}")

    out_csv = RESULTS_DIR / f"{date.date()}_phase{args.phase}_diffs.csv"
    out_acc = RESULTS_DIR / f"{date.date()}_phase{args.phase}_acceptance.csv"
    out_html = Path(args.report) if args.report else (
        RESULTS_DIR / f"{date.date()}_phase{args.phase}_report.html")

    if args.phase == 1:
        ticker = tickers[0]
        result = validate_single_day(ticker, date, args.orats_token, args.tradier_token)
        rows = []
        for m, vals in result.items():
            rows.append({"ticker": ticker, "metric": m, **vals})
        df_long = pd.DataFrame(rows)
        df_long.to_csv(out_csv, index=False)
        print(f"\n{ticker} on {date.date()}:")
        print(df_long.to_string(index=False))
        # build wide single-row frame for the same reporting path
        wide = {"ticker": [ticker]}
        for m, vals in result.items():
            wide[f"orats_{m}"] = [vals["orats"]]
            wide[f"ours_{m}"] = [vals["ours"]]
            wide[f"diff_{m}"] = [vals["diff"]]
        df = pd.DataFrame(wide)
    elif args.phase == 2:
        df = validate_cross_section(tickers, date, args.orats_token, args.tradier_token)
        df.to_csv(out_csv, index=False)
    elif args.phase == 3:
        df = validate_rolling(tickers, date, args.n_days,
                              args.orats_token, args.tradier_token)
        df.to_csv(out_csv, index=False)
    else:  # phase 4
        per_bucket = validate_edge_cases(date, args.orats_token, args.tradier_token)
        df = pd.concat([d.assign(_bucket=b) for b, d in per_bucket.items()],
                       ignore_index=True)
        df.to_csv(out_csv, index=False)

    report = compute_acceptance_report(df)
    overall, per_signal = decision_from_report(report)

    acc_rows = [{"metric": m, **r, "action": per_signal.get(m, "")}
                for m, r in report.items()]
    pd.DataFrame(acc_rows).to_csv(out_acc, index=False)

    print("\n=== Acceptance ===")
    for m, r in report.items():
        stat = r.get("stat")
        stat_s = f"{stat:.4f}" if isinstance(stat, (int, float)) else "—"
        print(f"  {m:12s}  {r['verdict']:10s}  stat={stat_s}  n={r.get('n', 0)}")
    print(f"\n=== Overall: {overall} ===")
    for m, d in per_signal.items():
        print(f"  {m:12s} -> {d}")

    write_report(df, report, overall, per_signal, date,
                 phase=f"Phase {args.phase}", out_path=out_html)
    print(f"\n[csv] diffs:      {out_csv}")
    print(f"[csv] acceptance: {out_acc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
