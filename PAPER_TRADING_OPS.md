# Paper Trading Operational Runbook

Last updated: 2026-04-23

This runbook covers the implemented paper-trading operator flow: Stage 1 shadow,
Stage 2 broker read-only, Stage 3 manual approval, and Stage 4 controlled auto.
Times are Eastern Time unless noted. All commands run from the repo root.

## Current Implementation Status

- Implemented daily data runner: `cron/poll_tradier_daily.bat`, registered by
  `cron/register_cron.bat` as `TradierDailyPoll`.
- Implemented Stage 1 scheduled wrapper: `automation/run_paper_shadow.ps1`.
- Implemented Phase A CLI: `python run_paper_phase_a.py --config config/paper_trading.yaml --as-of-date YYYY-MM-DD`.
- Implemented approval CLI: `python -m src.paper.approval_cli <bundle_dir>\intents.json --approve --operator <name>`.
- Implemented Phase B CLI: `python run_paper_phase_b.py --config config/paper_trading.yaml --bundle-dir <bundle_dir>`.
- Implemented daily Phase A/Phase B orchestrator: `python run_paper_daily.py --mode auto`.
- Implemented daily PowerShell wrapper: `automation/run_paper_daily.ps1`.

No CPU-heavy dependencies are required for operator setup, broker smoke,
approval, or Phase A/B docs/config work. Use the existing project environment;
do not add CUDA, torch, numba, or other heavy local installs for these ops paths.

## Quick Reference

Store a Tradier sandbox token in Windows Credential Manager:

```powershell
python -m src.paper.secrets --store sandbox PAPER-ACCOUNT-ID
```

Smoke the Tradier sandbox connection without submitting orders:

```powershell
python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml
```

Optionally preview a one-share sandbox order without placing it:

```powershell
python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml --preview-order --preview-symbol SPY --preview-qty 1
```

Capture a read-only broker snapshot:

```powershell
python -m src.paper.tools.broker_snapshot --config config/paper_trading.yaml --output paper_trading/current/broker_snapshot.sandbox.json
```

Run Phase A. This generates intents and never submits orders:

```powershell
python run_paper_phase_a.py --config config/paper_trading.yaml --as-of-date YYYY-MM-DD
```

Approve or reject the Phase A bundle:

```powershell
python -m src.paper.approval_cli <bundle_dir>\intents.json --approve --operator <name>
python -m src.paper.approval_cli <bundle_dir>\intents.json --reject --operator <name> --comment "reason"
```

Run Phase B against an approved bundle:

```powershell
python run_paper_phase_b.py --config config/paper_trading.yaml --bundle-dir <bundle_dir>
```

Run the daily orchestrator. In normal `auto` mode it runs Phase A only and
stops with an informational status once the bundle is `APPROVED`, preserving the
T+1 contract. Use a separate `phase-b` run at the next-open execution window:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_paper_daily.ps1 -Mode auto -Config config/paper_trading.yaml
```

If you explicitly need same-process chaining for a sandbox drill, require the
override flag:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_paper_daily.ps1 -Mode auto -Config config/paper_trading.yaml -ChainApprovedPhaseB
```

Engage the kill switch:

```powershell
New-Item -Path "paper_trading\state\KILL_SWITCH" -ItemType File -Force
Add-Content -Path "paper_trading\state\KILL_SWITCH" -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), reason: <brief reason>"
```

Clear the kill switch only after root-cause analysis and operator sign-off:

```powershell
Add-Content -Path "paper_trading\state\KILL_SWITCH_LOG.txt" -Value "Cleared at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); RCA: <root cause>"
Remove-Item "paper_trading\state\KILL_SWITCH" -Force
```

## Daily Data Runner

`cron/poll_tradier_daily.bat` is the implemented Tradier options poll. It skips
weekends, runs `python run_options_setup.py --daily-poll --max-tickers 1500`,
writes logs to `logs/tradier_daily/<YYYY-MM-DD>.log`, and is intended to produce
`data/cache/options/tradier_daily/<YYYY-MM-DD>.parquet`.

Register it from an elevated shell:

```cmd
cron\register_cron.bat
```

Verify or run it on demand:

```cmd
schtasks /Query /TN TradierDailyPoll /V /FO LIST
schtasks /Run /TN TradierDailyPoll
```

## Phase A: Intent Generation

Phase A loads the configured baseline, builds the target book, diffs current
broker or mock holdings, applies order policy, previews orders, and writes a
timestamped bundle under `results_dir`.

Default artifact shape:

```text
results/
  _paper_phase_a_<timestamp>/
    intents.json
    approval.template.json
    intended_trades.csv
    target_weights.csv
    prior_weights.csv
    manifest.json
    checksums.sha256
logs/
  paper_phase_a/
    <YYYY-MM-DD>.log
paper_trading/
  blotter/
    orders.jsonl
```

Phase A is the only place Stage 4 auto approval can be written. Auto approval
requires `stage: 4`, `stage4_mode: auto`, passing preflight, and clean previews.
If `stage4_mode: manual`, Phase A still writes an awaiting-approval bundle.

## Approval Flow

Stage 3 requires a human approval record before Phase B. The approval CLI edits
the same `intents.json` written by Phase A and embeds an immutable
`approval_record`. Rejections require a comment.

Use explicit bundle paths. Although `approval_cli` has a default path of
`paper_trading/current/intents.json`, Phase A currently writes timestamped
bundles under `results_dir`, so operators should pass `<bundle_dir>\intents.json`
directly.

## Phase B: Submission and Reconciliation

Phase B loads an approved Phase A bundle, checks the kill switch, verifies the
config hash, checks prior-day reconciliation, pings the broker, checks overnight
drift, submits approved orders, polls fills, and writes reconciliation artifacts.

Important guards:

- `paper_trading/state/KILL_SWITCH` exists: Phase B halts before broker action.
- Bundle config hash differs from current config: Phase B halts.
- Bundle is not approved: Phase B halts.
- Broker ping fails: Phase B halts.
- Overnight weight drift exceeds `overnight_drift_halt_pct`: Phase B halts.

Primary outputs:

```text
paper_trading/
  blotter/
    orders.jsonl
    fills.jsonl
  reports/
    <YYYY-MM-DD>/
      reconciliation.md
    reconciliation_summary.jsonl
results/
  _paper_phase_a_<timestamp>/
    phase_b_summary.json
    reconciliation_report.json
logs/
  paper_phase_b/
    <YYYY-MM-DD>.log
```

## Stage Operations

Stage 1 shadow:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_paper_shadow.ps1 -DryRun
python run_paper_shadow.py --config config/paper_trading.yaml --as-of-date YYYY-MM-DD --dry-run
```

Stage 2 broker read-only:

```powershell
python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml
python -m src.paper.tools.broker_snapshot --config config/paper_trading.yaml --output paper_trading/current/broker_snapshot.sandbox.json
```

Stage 3 manual paper:

```powershell
python run_paper_phase_a.py --config config/paper_trading.yaml --as-of-date YYYY-MM-DD
python -m src.paper.approval_cli <bundle_dir>\intents.json --approve --operator <name>
python run_paper_phase_b.py --config config/paper_trading.yaml --bundle-dir <bundle_dir>
```

Stage 4 controlled auto:

Set `stage: 4`. Use `stage4_mode: manual` until the Stage 3 gates pass. Switch
to `stage4_mode: auto` only after sign-off, and use `canary_mode: true` for the
first 10 trading days. Even in Stage 4 auto, keep the normal scheduler split:
Phase A on day T and Phase B on day T+1. The same-process chain flag is for
explicit drills only.

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_paper_daily.ps1 -Mode auto -Config config/paper_trading.yaml
```

## Kill Switch

Canonical path: `paper_trading/state/KILL_SWITCH`.

File presence means halt. Phase B checks this path before loading or submitting
orders. Stage 4 preflight also checks this path. Do not clear the file until the
root cause, mitigation, evidence, and operator sign-off have been appended to
`paper_trading/state/KILL_SWITCH_LOG.txt`.

## Go-Live Gates

Live trading is not enabled by the current paper stages. Promotion requires all
contract gates plus an explicit live config and environment confirmation.

Minimum gates before Stage 4 auto:

- At least 10 Stage 3 trading days with zero red alerts.
- Clean reconciliation within configured quantity, price, and drift tolerances.
- Turnover and estimated costs within policy caps.
- No manual overrides or kill-switch activations during the gate window.
- Canary mode configured for first 10 Stage 4 trading days.

Minimum gates before live:

- PAPER_TRADING_CONTRACT gates PASS for 6 consecutive full-threshold months
  after the provisional window.
- No red operational alerts for at least 20 consecutive trading days.
- Factor loadings and tracking error remain within documented thresholds.
- Operator sign-off is recorded.
- Config uses `stage: 4`, `broker: tradier_live`, `capital_mode: live`.
- `CONFIRM_LIVE_TRADING=true` or explicit programmatic confirmation is present.

## Failure Handling

Broker smoke fails: verify token storage with `src.paper.secrets`, confirm
`account_id`, check internet access, then rerun `tradier_sandbox_smoke` and the
broker snapshot.

Approval deadline expired: rerun Phase A for a fresh bundle rather than editing
timestamps manually.

Phase B reports config hash mismatch: rerun Phase A with the current config, or
restore the exact approved config before attempting Phase B.

Unexpected broker or reconciliation error: engage the kill switch, preserve
logs, and do not rerun submission until the issue is root-caused.

## References

- `config/paper_trading.example.yaml`
- `docs/SECRETS_SETUP.md`
- `paper_trading/DEPLOYMENT_CHECKLIST.md`
- `PAPER_TRADING_CONTRACT.md`
- `FEATURE_CONTRACT.md`
