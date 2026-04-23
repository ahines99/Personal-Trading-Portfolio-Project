# Paper Trading State

This directory is the on-disk state root for the paper-trading runtime.

Stage 1 writes only shadow artifacts:

- `history/blotter.jsonl`
  - Append-only intent history.
  - One JSON object per shadow run.
- `current/state.json`
  - Atomic snapshot of the latest intended portfolio state.
- `state/KILL_SWITCH`
  - Reserved canonical kill-switch path for later stages.

Daily shadow bundles are written separately under `results/_paper_shadow_<timestamp>/`.

Retention:

- History is append-only by design.
- Stage 1 does not prune bundles or blotter history automatically.
- If retention rules are added later, they should archive or rotate data without rewriting prior JSONL entries.
