# Paper Trading Secrets Setup

Tradier API tokens are loaded by `src.paper.secrets` and should be stored in
Windows Credential Manager through `keyring`. Account IDs are not secrets and may
live in `config/paper_trading.yaml` or local `PAPER_TRADING_ACCOUNT_ID`.

## Preferred Path

Store the sandbox token before Stage 2 or higher:

```powershell
python -m src.paper.secrets --store sandbox PAPER-ACCOUNT-ID
```

Store a live token only after live-trading approval:

```powershell
python -m src.paper.secrets --store live LIVE-ACCOUNT-ID
```

Use `sandbox` for `broker: tradier_sandbox` and `live` for
`broker: tradier_live`. The keyring lookup key is mode plus account ID, so the
`account_id` in `config/paper_trading.yaml` must match the account used here.

## Emergency Fallback

If `keyring` is unavailable, the broker factory can read a token from the
environment:

```powershell
$env:TRADIER_API_TOKEN = '...'
$env:TRADIER_TOKEN_SANDBOX = '...'
$env:TRADIER_API_TOKEN_SANDBOX = '...'
$env:TRADIER_TOKEN_LIVE = '...'
$env:TRADIER_API_TOKEN_LIVE = '...'
```

This path prints a warning and is not approved for unattended operation. Do not
write real tokens to `.env.example`, committed config, docs, or logs.

## Sandbox Smoke

After `config/paper_trading.yaml` is set to `stage: 2`,
`broker: tradier_sandbox`, `capital_mode: paper`, and the sandbox `account_id`,
run the implemented sandbox smoke:

```powershell
python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml
```

Expected behavior: the command pings Tradier, fetches profile, balances, and
positions, and prints a structured JSON result. It does not submit orders.

Optional preview-only order smoke:

```powershell
python -m src.paper.tools.tradier_sandbox_smoke --config config/paper_trading.yaml --preview-order --preview-symbol SPY --preview-qty 1
```

This sends Tradier's preview request only and records `would_place_order=false`
in the result. For a persisted read-only account snapshot, run:

```powershell
python -m src.paper.tools.broker_snapshot --config config/paper_trading.yaml --output paper_trading/current/broker_snapshot.sandbox.json
```

No extra CPU-heavy dependencies are required for this smoke path; use the
existing project environment and the dependencies already listed for paper ops.
