"""Identify and SAFELY archive stale feature_panel_*.pkl / ml_predictions_*.pkl caches.

Stale caches accumulate (15-20 GB each) and obscure which one was the baseline
for any given backtest. This utility never deletes; it only moves caches to an
archive directory under data/cache/archive/<timestamp>/, with a manifest.json
recording original paths, sizes and reason. A --restore flag puts a single
hash back where it came from.

Usage:
    python clean_stale_caches.py --list
    python clean_stale_caches.py --suggest-keep
    python clean_stale_caches.py --archive --keep <hash1>,<hash2> --yes
    python clean_stale_caches.py --restore <hash>

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
ARCHIVE_ROOT = CACHE_DIR / "archive"
LOGS_DIR = ROOT / "logs"

# Cache-file patterns we manage. Both follow <prefix>_<12-hex>.pkl.
CACHE_PATTERNS = ("feature_panel_*.pkl", "ml_predictions_*.pkl")
HASH_RE = re.compile(r"(?:feature_panel|ml_predictions)_([0-9a-f]{8,})\.pkl", re.I)
RECENT_LOG_DAYS = 7
GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    path: Path
    hash: str
    size_bytes: int
    mtime: float
    origin_log: str | None = None
    origin_cagr: str | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / GB

    @property
    def mtime_str(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")

    def origin_str(self) -> str:
        if not self.origin_log:
            return "(no matching log)"
        cagr = f" -> {self.origin_cagr}" if self.origin_cagr else ""
        return f"{self.origin_log}{cagr}"


@dataclass
class LogIndex:
    """Maps cache hash -> (log_name, cagr_string) using the most recent log."""
    by_hash: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    recent_hashes: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_caches() -> list[CacheEntry]:
    entries: list[CacheEntry] = []
    for pattern in CACHE_PATTERNS:
        for p in CACHE_DIR.glob(pattern):
            m = HASH_RE.match(p.name)
            if not m:
                continue
            stat = p.stat()
            entries.append(
                CacheEntry(
                    path=p,
                    hash=m.group(1),
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )
    entries.sort(key=lambda e: e.mtime, reverse=True)
    return entries


def index_logs() -> LogIndex:
    """Scan logs/* for cache-hash references and CAGR results.

    For each hash we keep the most recent log that mentioned it. We also flag
    hashes referenced by any log modified within RECENT_LOG_DAYS.
    """
    idx = LogIndex()
    if not LOGS_DIR.exists():
        return idx
    cagr_re = re.compile(r"\bCAGR\s+([0-9.]+%)", re.I)
    cutoff = time.time() - RECENT_LOG_DAYS * 86400
    # Sort logs newest-first so the first hit wins.
    logs = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hashes_here = {m.group(1) for m in HASH_RE.finditer(text)}
        if not hashes_here:
            continue
        cagr_match = cagr_re.search(text)
        cagr = cagr_match.group(1) if cagr_match else None
        rel = f"logs/{log.name}"
        for h in hashes_here:
            idx.by_hash.setdefault(h, (rel, cagr))
            if log.stat().st_mtime >= cutoff:
                idx.recent_hashes.add(h)
    return idx


def annotate(entries: list[CacheEntry], idx: LogIndex) -> None:
    for e in entries:
        hit = idx.by_hash.get(e.hash)
        if hit:
            e.origin_log, e.origin_cagr = hit


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def fmt_table(entries: Iterable[CacheEntry]) -> str:
    rows = list(entries)
    if not rows:
        return "(no feature_panel_*.pkl or ml_predictions_*.pkl caches found)"
    header = f"{'HASH':<14}{'SIZE_GB':>9}  {'MTIME':<18}  ORIGIN_LOG (if known)"
    sep = "-" * len(header)
    out = [header, sep]
    for e in rows:
        out.append(
            f"{e.hash[:12]:<14}{e.size_gb:>9.2f}  {e.mtime_str:<18}  {e.origin_str()}"
        )
    return "\n".join(out)


def short(h: str) -> str:
    return h[:12]


def matches_keep(entry_hash: str, keep: set[str]) -> bool:
    """A keep token matches if it is a prefix of the entry hash (or vice-versa)."""
    for k in keep:
        if entry_hash.startswith(k) or k.startswith(entry_hash):
            return True
    return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_list(_args: argparse.Namespace) -> int:
    entries = discover_caches()
    idx = index_logs()
    annotate(entries, idx)
    print(fmt_table(entries))
    total_gb = sum(e.size_gb for e in entries)
    print(f"\nTotal: {len(entries)} caches, {total_gb:.2f} GB")
    return 0


def cmd_suggest_keep(_args: argparse.Namespace) -> int:
    entries = discover_caches()
    idx = index_logs()
    annotate(entries, idx)
    suggested = [e for e in entries if e.hash in idx.recent_hashes]
    if not suggested:
        print(
            f"No cache hashes referenced by logs modified in the last "
            f"{RECENT_LOG_DAYS} days. Inspect --list manually."
        )
        return 0
    print(
        f"Suggested --keep (referenced by logs modified in last {RECENT_LOG_DAYS} days):"
    )
    print(fmt_table(suggested))
    keep_arg = ",".join(short(e.hash) for e in suggested)
    print(f"\nUse:  --keep {keep_arg}")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    if not args.keep:
        print("ERROR: --archive requires --keep <hash[,hash...]> (refusing to archive ALL).",
              file=sys.stderr)
        return 2
    keep = {k.strip().lower() for k in args.keep.split(",") if k.strip()}
    if not keep:
        print("ERROR: --keep is empty after parsing.", file=sys.stderr)
        return 2

    entries = discover_caches()
    idx = index_logs()
    annotate(entries, idx)
    if not entries:
        print("Nothing to archive: no caches discovered.")
        return 0

    keepers = [e for e in entries if matches_keep(e.hash, keep)]
    movers = [e for e in entries if not matches_keep(e.hash, keep)]

    unknown = {k for k in keep if not any(matches_keep(e.hash, {k}) for e in entries)}
    if unknown:
        print(f"ERROR: --keep references unknown hash(es): {sorted(unknown)}",
              file=sys.stderr)
        print("       Run --list to see available hashes.", file=sys.stderr)
        return 2

    if not movers:
        print("Nothing to archive: every cache matched --keep.")
        return 0

    print("Will KEEP in place:")
    print(fmt_table(keepers))
    print("\nWill ARCHIVE (move, not delete):")
    print(fmt_table(movers))
    free_gb = sum(e.size_gb for e in movers)
    print(f"\nBytes that will be freed from data/cache/: {free_gb:.2f} GB "
          f"({len(movers)} files)")

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to actually move files.")
        return 0

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = ARCHIVE_ROOT / ts
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {
        "archived_at": ts,
        "reason": args.reason or "stale-cache cleanup",
        "kept_hashes": sorted({e.hash for e in keepers}),
        "entries": [],
    }
    for e in movers:
        target = dest / e.path.name
        shutil.move(str(e.path), str(target))
        manifest["entries"].append({
            "hash": e.hash,
            "original_path": str(e.path),
            "archived_path": str(target),
            "size_bytes": e.size_bytes,
            "mtime": e.mtime,
            "origin_log": e.origin_log,
            "origin_cagr": e.origin_cagr,
        })
        print(f"  moved {e.path.name} -> {target.relative_to(ROOT)}")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {(dest / 'manifest.json').relative_to(ROOT)}")
    print(f"Archived {len(movers)} files ({free_gb:.2f} GB).")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    target_hash = args.restore.strip().lower()
    if not target_hash:
        print("ERROR: --restore requires a hash.", file=sys.stderr)
        return 2
    if not ARCHIVE_ROOT.exists():
        print(f"ERROR: no archive directory at {ARCHIVE_ROOT}.", file=sys.stderr)
        return 2

    matches: list[tuple[Path, dict]] = []
    for manifest_path in sorted(ARCHIVE_ROOT.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for entry in manifest.get("entries", []):
            h = entry.get("hash", "")
            if h.startswith(target_hash) or target_hash.startswith(h):
                matches.append((manifest_path, entry))

    if not matches:
        print(f"No archived cache matches hash '{target_hash}'.", file=sys.stderr)
        return 1
    if len(matches) > 1 and not args.yes:
        print(f"Hash '{target_hash}' matches {len(matches)} archived entries:")
        for _, entry in matches:
            print(f"  {entry['hash']}  {entry['archived_path']}")
        print("Re-run with a longer prefix, or pass --yes to restore the most recent.")
        return 2

    matches.sort(key=lambda m: m[1].get("mtime", 0), reverse=True)
    _, entry = matches[0]
    src = Path(entry["archived_path"])
    dst = Path(entry["original_path"])
    if not src.exists():
        print(f"ERROR: archived file is missing on disk: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"ERROR: destination already exists, refusing to overwrite: {dst}",
              file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"Restored {src.name} -> {dst}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safely archive stale feature_panel/ml_predictions caches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if "Usage:" in __doc__ else "",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="List all caches with size, mtime and origin log.")
    g.add_argument("--suggest-keep", action="store_true",
                   help=f"Suggest --keep hashes referenced by logs modified in "
                        f"the last {RECENT_LOG_DAYS} days.")
    g.add_argument("--archive", action="store_true",
                   help="Move caches NOT in --keep to data/cache/archive/<ts>/.")
    g.add_argument("--restore", metavar="HASH",
                   help="Move an archived cache back to data/cache/.")
    p.add_argument("--keep", metavar="H1,H2,...",
                   help="Comma-separated hash prefixes to keep when archiving.")
    p.add_argument("--reason", help="Free-text reason recorded in manifest.json.")
    p.add_argument("--yes", action="store_true",
                   help="Confirm: actually move files (otherwise dry-run).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not CACHE_DIR.exists():
        print(f"ERROR: cache directory not found: {CACHE_DIR}", file=sys.stderr)
        return 2
    if args.list:
        return cmd_list(args)
    if args.suggest_keep:
        return cmd_suggest_keep(args)
    if args.archive:
        return cmd_archive(args)
    if args.restore:
        return cmd_restore(args)
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
