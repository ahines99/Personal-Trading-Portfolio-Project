from __future__ import annotations

import argparse
import getpass
import os
import sys


SERVICE_NAME = "tradier"
_ENV_FALLBACK_TEMPLATE_CANDIDATES = (
    "TRADIER_API_TOKEN",
    "TRADIER_TOKEN_{mode}",
    "TRADIER_API_TOKEN_{mode}",
)

try:
    import keyring as _keyring
except Exception:  # pragma: no cover - runtime dependency may be absent
    _keyring = None


def load_tradier_token(mode: str, account_id: str) -> str:
    normalized_mode = _normalize_mode(mode)
    account_key = _account_key(normalized_mode, account_id)
    token = _load_from_keyring(account_key)
    if token:
        return token

    fallback_token, fallback_var = _load_from_env(normalized_mode)
    if fallback_token:
        print(
            "WARNING: using Tradier token from environment fallback "
            f"({fallback_var}); move it to keyring.",
            file=sys.stderr,
        )
        return fallback_token

    raise RuntimeError(
        "No Tradier token found in keyring or environment fallback for "
        f"mode={mode} account_id={account_id}."
    )


def store_tradier_token(mode: str, account_id: str, token: str) -> None:
    if _keyring is None:
        raise RuntimeError(
            "keyring is unavailable; cannot store Tradier token in Credential Manager."
        )

    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Tradier token must be non-empty.")

    normalized_mode = _normalize_mode(mode)
    _keyring.set_password(
        SERVICE_NAME,
        _account_key(normalized_mode, account_id),
        cleaned,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tradier token helper")
    parser.add_argument(
        "--store",
        nargs=2,
        metavar=("MODE", "ACCOUNT_ID"),
        help="Store a Tradier token in keyring for MODE and ACCOUNT_ID.",
    )
    parser.add_argument(
        "--token",
        help="Optional token value. If omitted, the CLI prompts securely.",
    )
    args = parser.parse_args(argv)

    if not args.store:
        parser.error("No action requested. Use --store MODE ACCOUNT_ID.")

    mode, account_id = args.store
    token = args.token or getpass.getpass("Tradier token: ")
    store_tradier_token(mode, account_id, token)
    print(
        "Stored Tradier token in keyring for "
        f"{mode}/{account_id}.",
        file=sys.stderr,
    )
    return 0


def _load_from_keyring(account_key: str) -> str | None:
    if _keyring is None:
        return None
    token = _keyring.get_password(SERVICE_NAME, account_key)
    if token:
        return token.strip() or None
    return None


def _load_from_env(mode: str) -> tuple[str | None, str | None]:
    mode_upper = str(mode).upper()
    for template in _ENV_FALLBACK_TEMPLATE_CANDIDATES:
        env_name = template.format(mode=mode_upper)
        token = os.getenv(env_name)
        if token and token.strip():
            return token.strip(), env_name
    return None, None


def _account_key(mode: str, account_id: str) -> str:
    return f"{mode}_{account_id}"


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized.startswith("tradier_"):
        normalized = normalized.split("_", 1)[1]
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
