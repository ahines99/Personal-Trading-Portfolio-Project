from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def compute_config_hash(config: Any) -> str:
    normalized = _normalize_for_hash(config)
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_rebalance_approval(config_hash: str, rebalance_entry: Any) -> bool:
    approved_hash = _extract_config_hash(rebalance_entry)
    return bool(approved_hash) and approved_hash == config_hash


def validate_rebalance_approval(config_hash: str, rebalance_entry: Any) -> str:
    approved_hash = _extract_config_hash(rebalance_entry)
    if not approved_hash:
        raise ValueError("Rebalance approval is missing required config_hash.")
    if approved_hash != config_hash:
        raise ValueError(
            "Rebalance approval config hash mismatch: "
            f"current={config_hash} approved={approved_hash}"
        )
    return approved_hash


def validate_manifest_config_hash(config: Any, manifest: Any) -> str:
    current_hash = compute_config_hash(config)
    manifest_hash = _extract_config_hash(manifest)
    if not manifest_hash:
        raise ValueError("Manifest is missing required config_hash.")
    if manifest_hash != current_hash:
        raise ValueError(
            "Manifest config hash mismatch: "
            f"current={current_hash} manifest={manifest_hash}"
        )
    return current_hash


def verify_approval_freshness(bundle: Any, approval_entry: Any | None = None) -> bool:
    try:
        validate_approval_freshness(bundle, approval_entry)
    except (TypeError, ValueError):
        return False
    return True


def validate_approval_freshness(
    bundle: Any,
    approval_entry: Any | None = None,
) -> datetime:
    bundle_status = _extract_bundle_status(bundle)
    if bundle_status not in {"APPROVED", "REJECTED"}:
        raise ValueError(
            "Approval freshness validation requires bundle status APPROVED or REJECTED."
        )

    entry = approval_entry if approval_entry is not None else _extract_approval_entry(bundle)
    if entry is None:
        raise ValueError("Approval freshness validation requires an approval record.")

    approval_status = _extract_approval_status(entry)
    if approval_status != bundle_status:
        raise ValueError(
            "Approval record status mismatch: "
            f"bundle={bundle_status} approval_record={approval_status or 'MISSING'}"
        )

    approved_at = _extract_approved_at(entry)
    if approved_at is None:
        raise ValueError("Approval record is missing required approved_at.")

    generated_at = _extract_bundle_datetime(bundle, "generated_at")
    if generated_at is None:
        raise ValueError("Bundle is missing required generated_at.")

    approval_deadline = _extract_bundle_datetime(bundle, "approval_deadline")
    if approval_deadline is None:
        raise ValueError("Bundle is missing required approval_deadline.")

    if approved_at < generated_at:
        raise ValueError(
            "Approval timestamp predates bundle generation: "
            f"generated_at={generated_at.isoformat()} approved_at={approved_at.isoformat()}"
        )
    if approved_at > approval_deadline:
        raise ValueError(
            "Approval timestamp exceeds bundle approval deadline: "
            f"approval_deadline={approval_deadline.isoformat()} "
            f"approved_at={approved_at.isoformat()}"
        )

    return approved_at


def _extract_config_hash(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        direct = obj.get("config_hash")
        if direct:
            return str(direct)
        for nested_key in ("approval_record", "manifest", "rebalance_entry"):
            nested = obj.get(nested_key)
            nested_hash = _extract_config_hash(nested)
            if nested_hash:
                return nested_hash
        return None

    if hasattr(obj, "config_hash"):
        value = getattr(obj, "config_hash")
        if value:
            return str(value)

    for nested_attr in ("approval_record", "manifest", "rebalance_entry"):
        if hasattr(obj, nested_attr):
            nested_hash = _extract_config_hash(getattr(obj, nested_attr))
            if nested_hash:
                return nested_hash

    return None


def _extract_approval_entry(obj: Any) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        direct = obj.get("approval_record")
        if direct is not None:
            return direct
        nested = obj.get("bundle")
        if nested is not None:
            return _extract_approval_entry(nested)
        return None

    if hasattr(obj, "approval_record"):
        value = getattr(obj, "approval_record")
        if value is not None:
            return value

    if hasattr(obj, "bundle"):
        return _extract_approval_entry(getattr(obj, "bundle"))
    return None


def _extract_bundle_status(obj: Any) -> str | None:
    value = _extract_bundle_field(obj, "status")
    return _normalize_status(value)


def _extract_approval_status(obj: Any) -> str | None:
    value = _extract_direct_field(obj, "status")
    normalized = _normalize_status(value)
    if normalized:
        return normalized

    approved = _extract_direct_field(obj, "approved")
    normalized_bool = _normalize_bool(approved)
    if normalized_bool is None:
        return None
    return "APPROVED" if normalized_bool else "REJECTED"


def _extract_approved_at(obj: Any) -> datetime | None:
    value = _extract_direct_field(obj, "approved_at")
    if value is None:
        return None
    return _normalize_datetime(value)


def _extract_bundle_datetime(obj: Any, field_name: str) -> datetime | None:
    value = _extract_bundle_field(obj, field_name)
    if value is None:
        return None
    return _normalize_datetime(value)


def _extract_bundle_field(obj: Any, field_name: str) -> Any | None:
    value = _extract_direct_field(obj, field_name)
    if value is not None:
        return value

    nested = _extract_direct_field(obj, "bundle")
    if nested is not None:
        return _extract_bundle_field(nested, field_name)
    return None


def _extract_direct_field(obj: Any, field_name: str) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(field_name)
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    return None


def _normalize_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported datetime value: {type(value)!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().upper()
    return text or None


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None
    return bool(value)


def _normalize_for_hash(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_hash(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, set):
        return [_normalize_for_hash(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if hasattr(value, "model_dump"):
        return _normalize_for_hash(
            value.model_dump(mode="json", exclude_none=False)
        )
    if hasattr(value, "dict") and callable(value.dict):
        return _normalize_for_hash(value.dict())
    if hasattr(value, "__dict__"):
        public_items = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        return _normalize_for_hash(public_items)
    return str(value)
