# Test Suite Design and Cache-Key Fix

**Status:** Operating manual for the next test session.
**Last updated:** 2026-04-17
**Owner:** Alex Hines
**Scope:** ML US-equity portfolio research repo. Applies to any signal-isolation / A-B test that touches the feature panel.

This document describes:

1. Why the previous round of `iso_CZ_*` isolation tests was silently broken.
2. What was fixed in the cache layer, the CZ-signal gating layer, and the alt-data loader.
3. The new test harness (`run_signal_test_suite.py`) and its supporting tools.
4. The recommended day-to-day workflow for proposing, isolating, and committing new signals.

Read this end-to-end before launching another test sweep. The pitfalls section in particular has bitten us already.

---

## 1. The Problem (background)

### 1.1 Symptom

Six independent isolation tests — each run with a different `EXTRA_DENY_FEATURES` set targeting one C&Z signal at a time — produced **bit-identical** equity curves and metrics:

| Test                       | CAGR (observed) | Sharpe | Notes                           |
|----------------------------|-----------------|--------|---------------------------------|
| `iso_CZ_coskew`            | 20.83%          | 0.91   | suspicious                      |
| `iso_CZ_xfin`              | 20.83%          | 0.91   | suspicious                      |
| `iso_CZ_netpayoutyield`    | 20.83%          | 0.91   | suspicious                      |
| `iso_CZ_payoutyield`       | 20.83%          | 0.91   | suspicious                      |
| `iso_CZ_momseason`         | 20.83%          | 0.91   | suspicious                      |
| `iso_CZ_cashbasedopprof`   | 20.83%          | 0.91   | suspicious                      |

Six different deny lists. One result. That is not a coincidence — it is a tell that the deny list was being applied **after** the cache hit, so every run loaded the same precomputed feature panel.

### 1.2 Root causes (three layers, all contributing)

**Layer 1 — feature_panel cache key (the big one).**
`src/model.py` lines 234-242 (pre-fix) computed an MD5 cache key over close-price index, ranked-signal keys, alt-feature keys, sector-map flag, normalization flags, etc. — but **not** over `EXTRA_DENY_FEATURES`. The deny list was honored *during* feature construction, but the cache lookup happened first, so a previous panel built without any deny list satisfied the lookup and the deny list was effectively ignored.

```
old key  = md5(dates + signals + alt_features + flags)
new key  = md5(dates + signals + alt_features + flags + deny_list + deny_hash_suffix)
```

**Layer 2 — per-signal `enable` dict in `src/cz_signals.py`.**
The module had a clean `enable={"co_skew_252": True, ...}` dict structure, but it was never plumbed through to the CLI. There was no way to disable a single CZ signal short of editing source.

**Layer 3 — sibling-feature leakage.**
Several CZ signals have near-duplicates living in `src/alt_features.py` or other loaders (e.g. `co_skew_252` ↔ legacy `co_skew`, `dividend_yield` from CZ vs. `dividend_yield` from EODHD fundamentals, `cash_based_op_prof` ↔ accruals/quality features). Denying the CZ entry alone did **not** remove the sibling, so the model still received an information-equivalent feature and the test was not actually isolating the signal.

### 1.3 Why this matters

If isolation tests cannot isolate, every "this signal is worth +X bps" claim from that round is unverifiable. We need a harness that *forces* distinct caches per test and a deny-list discipline that catches sibling leakage before metrics are reported.

---

## 2. The Fixes Applied

### 2.1 `src/model.py` — cache key includes deny-list

The feature-panel cache key now folds in a normalized deny-list and an optional hash suffix:

```python
_deny_raw = os.environ.get("EXTRA_DENY_FEATURES", "") or ""
_deny_norm = ",".join(sorted({s.strip().lower()
                              for s in _deny_raw.split(",") if s.strip()}))
_deny_hash_suffix = os.environ.get("DENY_LIST_HASH_SUFFIX", "") or ""

_cache_key = hashlib.md5(
    f"{close.index[0]}_{close.index[-1]}_{close.shape}_"
    f"{sorted(ranked_signals.keys())}_"
    f"{sorted(alt_features.keys()) if alt_features else []}_"
    f"sector={bool(sector_map)}_sn={bool(size_neutralize)}_..."
    f"_deny={_deny_norm}_dhs={_deny_hash_suffix}".encode()
).hexdigest()[:12]
```

Behavior:

- The deny-list is **lowercased, deduped, sorted** before hashing — so `"a,b"` and `"B, A"` collapse to the same key (intentional).
- `DENY_LIST_HASH_SUFFIX` is a free-form string you can set to force a fresh cache key without changing the deny list itself (useful for "I changed the upstream feature, blow the cache" scenarios).
- A log line `[cache] deny-list active: N entries → hash includes deny suffix` prints whenever a deny list is in effect, so you can confirm at a glance.

Reference: `src/model.py` lines 233-256.

### 2.2 `run_strategy.py` — new CLI flags

| Flag                          | Effect                                                                                  |
|-------------------------------|-----------------------------------------------------------------------------------------|
| `--cz-only NAMES`             | Comma-separated CZ-signal allowlist. All other CZ signals are auto-denied for this run. |
| `--cz-exclude NAMES`          | Comma-separated CZ-signal blocklist. Listed signals are denied; everything else loads.  |
| `--no-cache`                  | Bypass feature-panel cache for this run. Still writes a fresh cache afterward.          |
| `--clear-cache`               | Delete `data/cache/feature_panel_*.pkl` before running.                                 |

These flags compose with `EXTRA_DENY_FEATURES`; the union of all sources becomes the effective deny list (and is what gets folded into the cache key).

### 2.3 Three duplicate signals gated by env vars

To eliminate sibling leakage on the three known overlap pairs, the legacy / loader-side versions are now opt-out via env:

| Env var                        | Disables                                                       |
|--------------------------------|----------------------------------------------------------------|
| `DISABLE_CO_SKEW_252=1`        | The legacy `co_skew_252` feeder so only the CZ version is used |
| `DISABLE_DIVIDEND_YIELD=1`     | The EODHD-fundamentals `dividend_yield` sibling                |
| `DISABLE_CASH_BASED_OP_PROF=1` | The quality-bucket cash-based op profitability sibling         |

Always set the matching `DISABLE_*` env var when isolating any of these three CZ signals, otherwise your test still sees the sibling and the result is meaningless.

### 2.4 EAR snake_case bug fix

`src/alt_data_loader.py` previously aliased the EODHD earnings-announcement-return field as `EAR` while the model expected `ear` (snake_case lower). The merge silently dropped it. Fixed: the loader now emits `ear` and the column flows into the panel as expected.

---

## 3. The New Test Harness

### 3.1 `run_signal_test_suite.py`

A small orchestrator that:

1. Reads a Python-defined test matrix (list of dicts: `name`, `extra_deny`, `env`, `cli_args`).
2. For each test:
   - Sets env vars (`EXTRA_DENY_FEATURES`, `DENY_LIST_HASH_SUFFIX`, `DISABLE_*`).
   - Optionally clears the feature-panel cache.
   - Shells out to `python run_strategy.py ...` with the test-specific CLI args.
   - Captures the resulting tearsheet row, cache hash printed in the log, and run duration.
3. Writes a comparison CSV (`results/_test_suite/<timestamp>/comparison.csv`) with one row per test.
4. Asserts at the end that **every test produced a distinct cache hash** — and screams loudly if two tests collided (which now indicates a real bug, not the silent cache-key leak we just fixed).

### 3.2 Cache discipline

Because `EXTRA_DENY_FEATURES` is now folded into the key, each entry in the test matrix produces its own pickle at `data/cache/feature_panel_<hash>.pkl`. There are two operating modes:

- **Default — keep caches.** Re-running the same test matrix is fast (cache hits). Good for iterating on downstream logic without paying the feature-build cost.
- **`--clear-cache-between-tests` — ultra-clean.** Wipes the feature-panel cache between every test. Slow but immune to "did I leave a stale cache from yesterday?" doubts. Use this for the final confirmation run before committing a signal.

### 3.3 Cache-hash collision check

At the end of every suite run the harness emits:

```
[suite] 6 tests run, 6 distinct cache hashes — OK
```

or

```
[suite] !!! CACHE HASH COLLISION !!!
        iso_CZ_coskew  → ab12cd34ef56
        iso_CZ_xfin    → ab12cd34ef56  (same as iso_CZ_coskew)
        Investigate immediately.
```

If you ever see a collision again, do not trust the metrics. Stop and debug.

---

## 4. Reproducible Baseline Locking

### 4.1 `snapshot_baseline.py`

Captures a JSON manifest pinning everything needed to reproduce a baseline run:

```json
{
  "label": "baseline_2567_cagr",
  "timestamp": "2026-04-17T14:32:11",
  "git": {
    "commit": "dab62f7",
    "branch": "master",
    "dirty": false,
    "untracked": ["..."]
  },
  "env": {
    "ENABLE_PHASE_D_SIGNALS": "1",
    "EXTRA_DENY_FEATURES": "",
    "DENY_LIST_HASH_SUFFIX": "",
    "DISABLE_CO_SKEW_252": "",
    "DISABLE_DIVIDEND_YIELD": "",
    "DISABLE_CASH_BASED_OP_PROF": ""
  },
  "cache_hashes": {
    "feature_panel": "9f3a1c7e2b08",
    "alt_features":  "44b1d09a771c",
    "regime":        "0e7b2f15c3aa"
  },
  "metrics": {
    "cagr": 0.2567,
    "sharpe": 1.12,
    "max_dd": -0.184,
    "turnover": 0.83
  },
  "feature_count": 142,
  "feature_list_md5": "c1a09f..."
}
```

### 4.2 Why bother

The 25.67% CAGR baseline was lost once (the iso-test contamination episode) and recovering it cost a week. Locking the manifest means:

- The exact env + git commit + cache hash that produced the number is on disk.
- Any future change can be measured against `metrics.cagr - baseline.cagr` with no ambiguity.
- If a cache file vanishes, the manifest tells you which feature set to rebuild.

Snapshot the baseline **once it is recovered** and commit the manifest to `results/baselines/` (this directory is git-tracked, the cache pickles are not).

---

## 5. Cache Hygiene

### 5.1 `clean_stale_caches.py`

Lists and archives — **not deletes** — feature-panel caches that are no longer referenced by any baseline manifest or recent test run.

```
$ python clean_stale_caches.py --list
data/cache/feature_panel_9f3a1c7e2b08.pkl   1.4 GB  IN USE (baseline_2567_cagr)
data/cache/feature_panel_ab12cd34ef56.pkl   1.4 GB  STALE  (last touched 12 days ago)
data/cache/feature_panel_77e0091bbf3a.pkl   1.4 GB  STALE  (last touched 18 days ago)
```

```
$ python clean_stale_caches.py --archive
moved 2 files → data/cache/_archive/2026-04-17/
```

Archived files are still on disk under `data/cache/_archive/<date>/`; deletion is a manual step done only when disk pressure forces it. This is intentional — losing a cache pickle that backed a known-good baseline is much more painful than a few extra GB.

### 5.2 Why archive instead of delete

- A "stale" cache today might be the baseline you want to bisect against tomorrow.
- Archive gives a 30-day grace window before any irreversible action.
- The archive directory is gitignored, so it does not bloat the repo.

---

## 6. Recommended Workflow (step-by-step)

### Step 1 — Rebuild a clean baseline

```bash
# Clean slate
python clean_stale_caches.py --archive

# Rebuild from current code with no overrides
python run_strategy.py --no-cache

# Snapshot it
python snapshot_baseline.py --label baseline_$(date +%Y%m%d) --out results/baselines/
```

Confirm CAGR / Sharpe match expectations before snapshotting. The manifest is the source of truth from this point on.

### Step 2 — Define the test matrix

Edit the `TESTS` list in `run_signal_test_suite.py`. Minimum fields per test:

```python
TESTS = [
    dict(
        name="iso_CZ_coskew",
        cli_args=["--cz-only", "co_skew_252"],
        env={
            "DISABLE_CO_SKEW_252": "1",      # kill the sibling
            "DENY_LIST_HASH_SUFFIX": "v2",   # force fresh hash if needed
        },
    ),
    dict(
        name="iso_CZ_dividend_yield",
        cli_args=["--cz-only", "dividend_yield"],
        env={"DISABLE_DIVIDEND_YIELD": "1"},
    ),
    # ...
]
```

Rule of thumb: every CZ signal in the matrix that has a known sibling **must** carry the matching `DISABLE_*` env var. If you forget, the harness will still run, but the result will be silently contaminated.

### Step 3 — Run with `--clear-cache-between-tests`

```bash
python run_signal_test_suite.py \
    --matrix TESTS \
    --clear-cache-between-tests \
    --out results/_test_suite/
```

This is the slow path (rebuild every panel from scratch) but it gives you the strongest possible isolation guarantee. Use it for every test sweep that is going to drive a commit decision.

For quick iteration during development, drop `--clear-cache-between-tests` — caches will be reused per unique deny list, which is now safe.

### Step 4 — Review the comparison CSV

```bash
python -c "import pandas as pd; \
    print(pd.read_csv('results/_test_suite/<ts>/comparison.csv') \
            .sort_values('delta_sharpe', ascending=False))"
```

Columns to look at:

| Column          | Meaning                                          |
|-----------------|--------------------------------------------------|
| `name`          | Test name from the matrix                        |
| `cache_hash`    | Feature-panel cache hash — must be unique        |
| `cagr`          | Compound annual growth rate                      |
| `sharpe`        | Annualized Sharpe                                |
| `delta_cagr`    | `cagr - baseline.cagr` (signed bps)              |
| `delta_sharpe`  | `sharpe - baseline.sharpe`                       |
| `turnover`      | For sanity-checking — should not move much       |
| `runtime_sec`   | Useful for spotting cache misses / stalls        |

### Step 5 — Commit decision rule

Promote a signal into the production feature set only if:

- `delta_sharpe ≥ 0.05` **OR**
- `delta_cagr ≥ 100 bps` (i.e. ≥ 1.00%)

AND

- `turnover` does not blow up by more than 10% relative
- The signal does not collide on cache hash with any other (i.e. the harness suite-end check passed)

Anything below that bar is statistical noise at our backtest length and should not be committed.

---

## 7. Common Pitfalls

### 7.1 Cache contamination from leftover stale caches

Even with the fixed cache-key logic, a `data/cache/feature_panel_*.pkl` file from a buggy older run can still be loaded if its key matches. Mitigations:

- Always run `clean_stale_caches.py --list` before a meaningful sweep.
- For the final confirmation run, use `--clear-cache-between-tests`.
- If a cache hash matches a known-bad commit (check `git log --oneline data/cache/`), archive it.

### 7.2 Sibling-feature leakage

When you deny a CZ-only feature, sibling features in `src/alt_features.py` or other loaders may still feed the model with an information-equivalent column. The classic offenders:

| CZ signal                | Sibling lives in                              | Required disable env var       |
|--------------------------|-----------------------------------------------|--------------------------------|
| `co_skew_252`            | `src/alt_features.py` (legacy `co_skew_252`)  | `DISABLE_CO_SKEW_252=1`        |
| `dividend_yield`         | EODHD fundamentals loader                     | `DISABLE_DIVIDEND_YIELD=1`     |
| `cash_based_op_prof`     | quality bucket in alt_features                | `DISABLE_CASH_BASED_OP_PROF=1` |

If you add a new CZ signal that overlaps an existing feature, add the sibling to this table **and** add a corresponding `DISABLE_*` env-var gate.

### 7.3 Forgetting to set `DISABLE_*` env vars

The harness does not auto-infer which siblings need disabling. If your test matrix omits the env var, the cache hash will still be unique (so the suite-end check passes) but the metrics are contaminated. Treat the table above as a checklist; review it every time a CZ signal appears in a test.

### 7.4 Tests passing the cache-hash uniqueness check but still being wrong

Uniqueness only catches the cache-key bug. It does not catch:

- Sibling leakage (see above).
- Bugs in the deny-list parsing in `src/model.py` (always grep the run log for the `[cache] deny-list active: N entries` line and confirm `N` matches what you set).
- Upstream changes (e.g. an EODHD field rename) that silently change feature content under a stable cache key — use `DENY_LIST_HASH_SUFFIX` to force a re-hash when you suspect this.

### 7.5 Mixing `--cz-only` and `--cz-exclude`

These flags are mutually exclusive. The harness should reject the combination, but if you bypass the harness and run `run_strategy.py` directly, double-check that you are using one or the other, not both.

### 7.6 Snapshot manifest drifting silently

The snapshot manifest pins a git commit. If you snapshot while the working tree is dirty, the `dirty: true` flag goes into the manifest — but the manifest cannot reproduce uncommitted changes. Always commit (or stash + clean) before snapshotting a baseline you intend to defend.

---

## 8. Reference: env vars and their effects

| Env var                          | Type   | Default | Effect                                                                                              |
|----------------------------------|--------|---------|-----------------------------------------------------------------------------------------------------|
| `ENABLE_PHASE_D_SIGNALS`         | bool   | `0`     | Loads the Phase D experimental signal set into the alt-features pipeline.                           |
| `EXTRA_DENY_FEATURES`            | csv    | empty   | Comma-separated feature names to drop from the panel **after** construction. Folded into cache key. |
| `DENY_LIST_HASH_SUFFIX`          | string | empty   | Free-form suffix folded into the cache key. Use to force a rebuild without changing the deny list.  |
| `DISABLE_CO_SKEW_252`            | bool   | `0`     | Disables the legacy `co_skew_252` feeder so only the CZ version reaches the model.                  |
| `DISABLE_DIVIDEND_YIELD`         | bool   | `0`     | Disables the EODHD-fundamentals `dividend_yield` sibling.                                           |
| `DISABLE_CASH_BASED_OP_PROF`     | bool   | `0`     | Disables the quality-bucket cash-based op profitability sibling.                                    |

### 8.1 Bool conventions

Any value other than empty string, `0`, `false`, `no` is treated as truthy (case-insensitive). When in doubt, use `1`. Use empty string (or unset the var entirely) to disable.

### 8.2 Setting env vars in the harness

The harness sets env vars per-test in a child-process environment, so they do not bleed between tests:

```python
env = os.environ.copy()
env.update(test["env"])
subprocess.run(["python", "run_strategy.py", *test["cli_args"]], env=env, check=True)
```

Do **not** `os.environ[...] = ...` in the parent process — that leaks across tests and re-introduces the silent-contamination class of bug we just fixed.

### 8.3 Inspecting the live env at run time

`run_strategy.py` prints an `[env]` block near the top of every run:

```
[env] ENABLE_PHASE_D_SIGNALS    = 1
[env] EXTRA_DENY_FEATURES       = co_skew_252,dividend_yield   (2 entries)
[env] DENY_LIST_HASH_SUFFIX     = v2
[env] DISABLE_CO_SKEW_252       = 1
[env] DISABLE_DIVIDEND_YIELD    = 1
[env] DISABLE_CASH_BASED_OP_PROF= (unset)
[cache] deny-list active: 2 entries → hash includes deny suffix
[cache] feature_panel cache hash = ab12cd34ef56
```

Always grep the log for `[cache] feature_panel cache hash` and confirm it matches what the harness's collision check reports. If the two disagree, something is wrong with how the env is being passed to the child process.

---

## Appendix A — Files touched by these fixes

| File                            | Change                                                                                  |
|---------------------------------|-----------------------------------------------------------------------------------------|
| `src/model.py`                  | Cache key now folds in `EXTRA_DENY_FEATURES` + `DENY_LIST_HASH_SUFFIX` (lines 233-256). |
| `run_strategy.py`               | New CLI flags `--cz-only`, `--cz-exclude`, `--no-cache`, `--clear-cache`.               |
| `src/cz_signals.py`             | Per-signal `enable` dict now wired through to the CLI flags above.                      |
| `src/alt_data_loader.py`        | EAR field aliased to snake-case `ear` so the model picks it up.                         |
| `src/alt_features.py`           | `DISABLE_CO_SKEW_252`, `DISABLE_DIVIDEND_YIELD`, `DISABLE_CASH_BASED_OP_PROF` honored.  |
| `run_signal_test_suite.py`      | New. Orchestrates A/B tests, enforces cache-hash uniqueness.                            |
| `snapshot_baseline.py`          | New. Produces JSON manifest pinning git + env + cache hashes + metrics.                 |
| `clean_stale_caches.py`         | New. Lists and archives (not deletes) stale `feature_panel_*.pkl` files.                |

## Appendix B — Quick reference: one-liners

```bash
# List caches, show which are in use vs stale
python clean_stale_caches.py --list

# Archive stale caches
python clean_stale_caches.py --archive

# Rebuild baseline from scratch and snapshot it
python run_strategy.py --no-cache && \
    python snapshot_baseline.py --label baseline_$(date +%Y%m%d)

# Run the full test suite, ultra-clean
python run_signal_test_suite.py --matrix TESTS --clear-cache-between-tests

# Run one isolation test by hand (smoke test)
EXTRA_DENY_FEATURES="co_skew_252" \
DISABLE_CO_SKEW_252=1 \
DENY_LIST_HASH_SUFFIX="smoke" \
python run_strategy.py --no-cache

# Inspect the latest comparison CSV
python -c "import pandas as pd, glob; \
    p = sorted(glob.glob('results/_test_suite/*/comparison.csv'))[-1]; \
    print(pd.read_csv(p).sort_values('delta_sharpe', ascending=False).to_string())"
```

## Appendix C — Decision log

| Date       | Decision                                                                                  |
|------------|-------------------------------------------------------------------------------------------|
| 2026-04-17 | Cache key now includes `EXTRA_DENY_FEATURES` + `DENY_LIST_HASH_SUFFIX`. Six iso-tests re-run. |
| 2026-04-17 | Three sibling-feature env-var gates added (`DISABLE_CO_SKEW_252`, `_DIVIDEND_YIELD`, `_CASH_BASED_OP_PROF`). |
| 2026-04-17 | EAR snake_case bug fixed in `src/alt_data_loader.py`.                                     |
| 2026-04-17 | `run_signal_test_suite.py`, `snapshot_baseline.py`, `clean_stale_caches.py` introduced.   |
| 2026-04-17 | This doc written as the operating manual for the next test session.                       |
