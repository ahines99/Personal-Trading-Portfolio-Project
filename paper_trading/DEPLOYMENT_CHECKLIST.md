# Paper Trading Deployment Checklist

Version: v1.1
Date: 2026-04-23
Operator: ______________________

## Config and Secrets

- [ ] Confirm current promoted baseline is recorded in `CURRENT_BASELINE.md`
- [ ] Validate `config/paper_trading.yaml` loads cleanly
- [ ] Confirm target stage is intentional: Stage 2 read-only, Stage 3 manual approval, or Stage 4 controlled auto
- [ ] Confirm `broker=tradier_sandbox` and `capital_mode=paper` until live approval is granted
- [ ] Confirm `account_id` matches the token stored with `python -m src.paper.secrets --store sandbox <account_id>`
- [ ] Verify `stage4_mode` is set intentionally (`manual` or `auto`)
- [ ] Verify `intents_only` and `dry_run` defaults are understood before any submission test
- [ ] Confirm `kill_switch_path` is `paper_trading/state/KILL_SWITCH`
- [ ] Confirm `alert_log_path` is writable
- [ ] Confirm `paper_trading/state/` exists
- [ ] Confirm no real token is stored in `.env.example`, committed YAML, docs, or logs
- [ ] Confirm no new CPU-heavy dependencies were added for operator setup

## Implemented Smoke Checks

- [ ] Run mock config load or mock broker snapshot successfully
- [ ] Run Tradier sandbox smoke:
  `python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml`
- [ ] If needed, run Tradier sandbox preview-only smoke:
  `python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml --preview-order --preview-symbol SPY --preview-qty 1`
- [ ] Run Tradier sandbox read-only broker snapshot:
  `python -m src.paper.tools.broker_snapshot --config config/paper_trading.yaml --output paper_trading/current/broker_snapshot.sandbox.json`
- [ ] Run Phase A without submission:
  `python run_paper_phase_a.py --config config/paper_trading.yaml --as-of-date YYYY-MM-DD`
- [ ] Approve or reject the emitted bundle explicitly:
  `python -m src.paper.approval_cli <bundle_dir>\intents.json --approve --operator <name>`
- [ ] Run Phase B only against an approved bundle:
  `python run_paper_phase_b.py --config config/paper_trading.yaml --bundle-dir <bundle_dir>`
- [ ] Run daily orchestrator in safe mode:
  `powershell -ExecutionPolicy Bypass -File .\automation\run_paper_daily.ps1 -Mode auto -Config config/paper_trading.yaml`
- [ ] Keep Phase B scheduled separately for the T+1 execution window; do not use `-ChainApprovedPhaseB` outside explicit sandbox drills
- [ ] Confirm prior reconciliation status is CLEAN or within tolerance
- [ ] Confirm no CRITICAL or FATAL alerts in the last 6 hours

## Scheduling and Safety

- [ ] If using the implemented Tradier daily poll, register or verify `TradierDailyPoll` from `cron/register_cron.bat`
- [ ] Confirm `cron/poll_tradier_daily.bat` writes to `data/cache/options/tradier_daily/<YYYY-MM-DD>.parquet`
- [ ] If using Task Scheduler for paper trading, point it at `automation/run_paper_daily.ps1` and document the local task name
- [ ] Test kill-switch activation
- [ ] Test kill-switch clear path
- [ ] Verify canary mode is enabled for first 10 trading days
- [ ] Verify canary caps match the approved config
- [ ] Verify drawdown thresholds match the contract
- [ ] Verify alerting channels are tested

## Go / No-Go

- [ ] Stage 2: at least 5 clean read-only sandbox snapshots
- [ ] Stage 3: at least 10 clean manual-approval paper runs before Stage 4
- [ ] Stage 4: canary enabled for first 10 trading days
- [ ] Live: all PAPER_TRADING_CONTRACT gates PASS for 6 consecutive full-threshold months after the provisional window
- [ ] Live: no red alerts for 20+ consecutive trading days
- [ ] Live: operator sign-off recorded; `CONFIRM_LIVE_TRADING=true` is set only for approved live deployment
- [ ] Run one Stage 4 dry path end to end before enabling auto approval
- [ ] Review auto-approval record format
- [ ] Review reconciliation markdown output
- [ ] Review operator rollback path to Stage 3
- [ ] Record final go/no-go decision and timestamp

Decision: ______________________
Timestamp: _____________________
