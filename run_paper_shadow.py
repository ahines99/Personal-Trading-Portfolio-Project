from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.controller import PaperTradingController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 paper-trading shadow runner")
    parser.add_argument("--config", default=None, help="Optional JSON or simple YAML config path")
    parser.add_argument("--as-of-date", default=None, help="Override as-of date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without persistence side effects")
    parser.add_argument("--results-dir", default=None, help="Override bundle output root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config = _load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True
    if args.results_dir:
        config["results_dir"] = args.results_dir

    as_of_date = args.as_of_date
    log_date = pd.Timestamp(as_of_date).strftime("%Y-%m-%d") if as_of_date else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    _configure_logging(repo_root, log_date)
    logger = logging.getLogger("paper.stage1.cli")

    try:
        controller = PaperTradingController(config=config, repo_root=repo_root)
        result = controller.run_daily(as_of_date=as_of_date)
    except Exception:
        logger.exception("paper shadow run failed")
        return 1

    print(json.dumps({
        "bundle_dir": result["bundle_dir"],
        "n_intents": result["n_intents"],
    }, indent=2))
    logger.info(
        "paper shadow run complete bundle_dir=%s n_intents=%s",
        result["bundle_dir"],
        result["n_intents"],
    )
    return 0


def _configure_logging(repo_root: Path, log_date: str) -> None:
    log_dir = repo_root / "logs" / "paper_shadow"
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
