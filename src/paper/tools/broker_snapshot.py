from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.paper.brokerage.factory import create_broker_client
from src.paper.loader import DEFAULT_CONFIG_PATH, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a normalized read-only broker snapshot."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Paper trading YAML config path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def build_snapshot_payload(config: Any) -> dict[str, Any]:
    client = create_broker_client(config)
    if client is None:
        raise RuntimeError("Broker factory did not return a client.")

    profile = client.get_profile()
    balances = client.get_balances()
    positions = client.get_positions()
    ping = client.ping()
    snapshot_payload = None
    position_records = positions
    if hasattr(client, "get_broker_snapshot"):
        snapshot = client.get_broker_snapshot(balances=balances, positions=positions)
        snapshot_payload = snapshot.header_payload()
        position_records = snapshot.position_records()
    return {
        "ping": ping,
        "profile": profile,
        "balances": balances,
        "positions": positions,
        "snapshot": snapshot_payload,
        "position_records": position_records,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    payload = build_snapshot_payload(config)
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
