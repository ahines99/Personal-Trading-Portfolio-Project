from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

try:  # Supports both `python -m src.paper...` and tests importing `paper...`.
    from src.paper.brokerage.keyring_manager import load_token_for_config
    from src.paper.brokerage.tradier import TradierBrokerClient
    from src.paper.loader import DEFAULT_CONFIG_PATH, load_config
except ModuleNotFoundError:  # pragma: no cover - import style depends on entrypoint
    from paper.brokerage.keyring_manager import load_token_for_config
    from paper.brokerage.tradier import TradierBrokerClient
    from paper.loader import DEFAULT_CONFIG_PATH, load_config


TokenLoader = Callable[[Any], str]


@dataclass(frozen=True)
class PreviewOrderRequest:
    symbol: str = "SPY"
    quantity: float = 1.0
    side: str = "buy"
    order_type: str = "market"
    limit_price: float | None = None


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SmokeResult:
    ok: bool
    broker: str | None
    account_id: str | None
    checks: list[SmokeCheck]
    preview: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_tradier_sandbox_smoke(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    preview_order: bool = False,
    preview_request: PreviewOrderRequest | None = None,
    token_loader: TokenLoader = load_token_for_config,
    session: requests.Session | None = None,
    timeout: float = 15.0,
    max_retries: int = 2,
) -> SmokeResult:
    """Verify Tradier sandbox connectivity and optionally preview an order.

    The optional order path uses Tradier's preview flag only. This helper has no
    order placement path by design.
    """

    checks: list[SmokeCheck] = []
    config: Any | None = None
    broker: str | None = None
    account_id: str | None = None

    try:
        config = load_config(config_path)
        broker = str(_read_attr(config, "broker") or "").strip().lower()
        account_id = str(_read_attr(config, "account_id") or "").strip()
        checks.append(SmokeCheck("config", True, f"loaded {Path(config_path)}"))
    except Exception as exc:
        checks.append(SmokeCheck("config", False, _safe_error(exc)))
        return SmokeResult(False, broker, account_id, checks)

    guard_error = _validate_sandbox_config(config)
    if guard_error:
        checks.append(SmokeCheck("sandbox_guard", False, guard_error))
        return SmokeResult(False, broker, account_id, checks)
    checks.append(SmokeCheck("sandbox_guard", True, "broker=tradier_sandbox capital_mode=paper"))

    try:
        token = token_loader(config)
        if not str(token).strip():
            raise RuntimeError("Tradier token loader returned an empty token.")
        checks.append(SmokeCheck("credentials", True, "token loaded"))
    except Exception as exc:
        checks.append(SmokeCheck("credentials", False, _safe_error(exc)))
        return SmokeResult(False, broker, account_id, checks)

    client = TradierBrokerClient(
        account_id=account_id or "",
        token=token,
        broker="tradier_sandbox",
        timeout=timeout,
        max_retries=max_retries,
        session=session,
    )

    for check_name, call in (
        ("account_ping", client.ping),
        ("profile", client.get_profile),
        ("balances", client.get_balances),
        ("positions", client.get_positions),
    ):
        try:
            payload = call()
            if check_name == "account_ping" and payload is not True:
                checks.append(
                    SmokeCheck(check_name, False, "account endpoint returned false")
                )
                return SmokeResult(False, broker, account_id, checks)
            checks.append(SmokeCheck(check_name, True, _summarize_payload(check_name, payload)))
        except Exception as exc:
            checks.append(SmokeCheck(check_name, False, _safe_error(exc)))
            return SmokeResult(False, broker, account_id, checks)

    preview_payload: dict[str, Any] | None = None
    if preview_order:
        request = preview_request or PreviewOrderRequest()
        try:
            preview_payload = client.preview_equity_order(
                request.symbol,
                request.quantity,
                request.side,
                request.order_type,
                limit_price=request.limit_price,
            )
            preview_payload["would_place_order"] = False
            checks.append(
                SmokeCheck(
                    "order_preview",
                    True,
                    (
                        f"previewed {request.side} {request.quantity:g} "
                        f"{request.symbol}; no order placed"
                    ),
                )
            )
        except Exception as exc:
            checks.append(SmokeCheck("order_preview", False, _safe_error(exc)))
            return SmokeResult(
                False, broker, account_id, checks, preview=preview_payload
            )
    else:
        checks.append(
            SmokeCheck(
                "order_preview",
                True,
                "skipped; use --preview-order to preview only",
            )
        )

    return SmokeResult(True, broker, account_id, checks, preview=preview_payload)


def require_explicit_write_flag(*, allow_write: bool) -> None:
    """Guard for any future write path.

    Keep write operations out of this smoke helper unless callers explicitly opt in.
    The current CLI never places orders.
    """

    if not allow_write:
        raise PermissionError("write path requires explicit --allow-write")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Tradier sandbox read-only connectivity. By default this never "
            "previews or places orders."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Paper trading YAML config path.",
    )
    parser.add_argument(
        "--preview-order",
        action="store_true",
        help="Preview a tiny sandbox equity order with Tradier preview=true; does not place it.",
    )
    parser.add_argument("--preview-symbol", default="SPY", help="Preview symbol.")
    parser.add_argument("--preview-qty", type=float, default=1.0, help="Preview quantity.")
    parser.add_argument(
        "--preview-side",
        choices=("buy", "sell"),
        default="buy",
        help="Preview side.",
    )
    parser.add_argument(
        "--preview-type",
        choices=("market", "limit"),
        default="market",
        help="Preview order type.",
    )
    parser.add_argument(
        "--preview-limit-price",
        type=float,
        default=None,
        help="Limit price when --preview-type=limit.",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="HTTP timeout seconds."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count for retryable reads.",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="Reserved safety flag for future write paths; this CLI still never places orders.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.allow_write:
        # Intentional no-op today: the CLI has no write path.
        require_explicit_write_flag(allow_write=True)

    if args.preview_type == "limit" and args.preview_limit_price is None:
        parser.error("--preview-limit-price is required when --preview-type=limit")

    request = PreviewOrderRequest(
        symbol=args.preview_symbol,
        quantity=args.preview_qty,
        side=args.preview_side,
        order_type=args.preview_type,
        limit_price=args.preview_limit_price,
    )
    result = run_tradier_sandbox_smoke(
        config_path=args.config,
        preview_order=args.preview_order,
        preview_request=request,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _validate_sandbox_config(config: Any) -> str | None:
    broker = str(_read_attr(config, "broker") or "").strip().lower()
    capital_mode = str(_read_attr(config, "capital_mode") or "").strip().lower()
    if broker != "tradier_sandbox":
        return f"expected broker=tradier_sandbox, got {broker or '<missing>'}"
    if capital_mode != "paper":
        return f"expected capital_mode=paper for sandbox smoke, got {capital_mode or '<missing>'}"
    return None


def _read_attr(config: Any, field_name: str) -> Any:
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _safe_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _summarize_payload(check_name: str, payload: Any) -> str:
    if check_name == "account_ping":
        return (
            "account endpoint reachable"
            if payload is True
            else "account endpoint returned false"
        )
    if check_name == "profile" and isinstance(payload, dict):
        status = payload.get("status") or "unknown"
        return f"profile status={status}"
    if check_name == "balances" and isinstance(payload, dict):
        equity = payload.get("equity") or payload.get("total_equity")
        return f"balances equity={equity}"
    if check_name == "positions" and isinstance(payload, list):
        return f"positions count={len(payload)}"
    return f"received {type(payload).__name__}"


if __name__ == "__main__":
    raise SystemExit(main())
