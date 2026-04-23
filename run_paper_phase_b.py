from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.phase_b_executor import PhaseBExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 Phase B paper-trading executor")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON paper config path")
    parser.add_argument("--bundle-dir", default=None, help="Explicit Phase A bundle dir")
    parser.add_argument("--approval-path", default=None, help="Explicit approval record path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    config = _load_config(args.config)
    log_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    _configure_logging(repo_root, log_date)
    logger = logging.getLogger("paper.stage3.phase_b.cli")

    try:
        executor = PhaseBExecutor(config=config, repo_root=repo_root)
        result = executor.run(bundle_dir=args.bundle_dir, approval_path=args.approval_path)
    except Exception:
        logger.exception("paper phase B failed")
        return 1

    print(json.dumps(result, indent=2))
    logger.info(
        "paper phase B complete bundle_dir=%s executed_trade_count=%s unfilled=%s",
        result["bundle_dir"],
        result["executed_trade_count"],
        result["unfilled_order_count"],
    )
    return 0


def _configure_logging(repo_root: Path, log_date: str) -> None:
    log_dir = repo_root / "logs" / "paper_phase_b"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_date}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        from src.paper.loader import load_config as external_load_config

        loaded = external_load_config(path)
        if hasattr(loaded, "model_dump"):
            return dict(loaded.model_dump())
        if hasattr(loaded, "dict"):
            return dict(loaded.dict())
    except Exception:
        pass
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _coerce_scalar(value.strip())
    return config


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


if __name__ == "__main__":
    raise SystemExit(main())

