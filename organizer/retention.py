"""Safe archive derivatives and retention bookkeeping.

Originals are never removed by the web cleanup flow.  Explicit compaction is
available only after a derivative has been created and verified.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_for(row) -> tuple[str, str]:
    if row and (row["source_is_card"] or row["human_correction"] or row["review_status"] == "approved"):
        return "gold", "card or human-reviewed source"
    return "ordinary", "unmodified working source"


def prepare_archive(archive_root: Path, rows: Iterable, *, max_size: int = 1800) -> dict:
    """Create verified JPEG derivatives and a manifest without deleting sources."""
    input_root = archive_root / "input"
    derivative_root = archive_root / "derivatives"
    derivative_root.mkdir(parents=True, exist_ok=True)
    by_name = {Path(str(row["source_path"])).name: row for row in rows} if rows else {}
    manifest = []

    for source in sorted(input_root.iterdir(), key=lambda item: item.name.lower()):
        if not source.is_file() or source.name.startswith("."):
            continue
        row = by_name.get(source.name)
        retention_class, reason = _class_for(row)
        derivative = derivative_root / f"{source.stem}.jpg"
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                image.save(derivative, "JPEG", quality=82, optimize=True)
        except Exception as exc:
            manifest.append({"source": str(source.relative_to(archive_root)), "status": "error", "error": str(exc)})
            continue
        manifest.append({
            "source": str(source.relative_to(archive_root)),
            "derivative": str(derivative.relative_to(archive_root)),
            "retention_class": retention_class,
            "reason": reason,
            "source_sha256": sha256(source),
            "source_size_bytes": source.stat().st_size,
            "derivative_sha256": sha256(derivative),
            "derivative_size_bytes": derivative.stat().st_size,
            "status": "verified",
        })

    manifest_path = archive_root / "retention-manifest.json"
    manifest_path.write_text(json.dumps({"version": 1, "items": manifest}, indent=2) + "\n")
    return {"manifest": manifest_path, "items": len(manifest), "verified": sum(item.get("status") == "verified" for item in manifest)}


def persist_manifest(conn: sqlite3.Connection, manifest_path: Path, root: Path) -> None:
    """Mirror the archive manifest into the central database for later tooling."""
    payload = json.loads(manifest_path.read_text())
    now = datetime.now().isoformat(timespec="seconds")
    for item in payload.get("items", []):
        if item.get("status") != "verified":
            continue
        relative = str((manifest_path.parent / item["source"]).relative_to(root))
        derivative = str((manifest_path.parent / item["derivative"]).relative_to(root))
        conn.execute(
            """INSERT INTO retention_items
            (relative_path, retention_class, derivative_path, source_sha256, source_size_bytes,
             derivative_sha256, derivative_size_bytes, status, created_at, updated_at, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'verified', ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
            derivative_path=excluded.derivative_path, derivative_sha256=excluded.derivative_sha256,
            derivative_size_bytes=excluded.derivative_size_bytes, status='verified',
            updated_at=excluded.updated_at, reason=excluded.reason""",
            (relative, item["retention_class"], derivative, item["source_sha256"],
             item["source_size_bytes"], item["derivative_sha256"], item["derivative_size_bytes"],
             now, now, item.get("reason", "")),
        )


def compact_archive(archive_root: Path, *, apply: bool = False) -> dict:
    """Report or remove only ordinary originals with a verified derivative."""
    manifest_path = archive_root / "retention-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    planned = removed = skipped = 0
    if apply:
        shutil.copy2(manifest_path, archive_root / "retention-manifest.before-compaction.json")
    for item in payload.get("items", []):
        if item.get("status") != "verified" or item.get("retention_class") != "ordinary":
            continue
        source = archive_root / item["source"]
        derivative = archive_root / item["derivative"]
        if not source.exists() or not derivative.exists() or sha256(derivative) != item["derivative_sha256"]:
            skipped += 1
            continue
        planned += 1
        if apply:
            source.unlink()
            removed += 1
    return {"planned": planned, "removed": removed, "skipped": skipped, "apply": apply}
