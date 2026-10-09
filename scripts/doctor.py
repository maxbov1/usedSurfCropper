#!/usr/bin/env python3
"""Check that the local UsedSurf installation is ready to run."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organizer.config import paths  # noqa: E402
from organizer.db import connect  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    directories = paths(ROOT)
    db_path = directories["data"] / "usedsurf.sqlite3"
    try:
        with connect(db_path) as conn:
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for required in {"boards", "photos", "runs", "artifacts", "retention_items"}:
                if required not in tables:
                    errors.append(f"database table missing: {required}")
            stale = conn.execute("SELECT COUNT(*) AS count FROM runs WHERE status='running'").fetchone()["count"]
            if stale:
                warnings.append(f"{stale} run(s) still marked running; inspect data/runs in the database")
    except Exception as exc:
        errors.append(f"database check failed: {exc}")

    if not (ROOT / "yolo11n.pt").exists():
        warnings.append("yolo11n.pt is not present; YOLO-assisted paths may use fallback detection")
    for module in ("PIL", "cv2", "pytesseract", "ultralytics"):
        try:
            __import__(module)
        except ImportError:
            warnings.append(f"runtime module unavailable: {module}")
    if shutil.which("tesseract") is None:
        warnings.append("native Tesseract executable unavailable; card OCR cannot run")
    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gb < 2:
        warnings.append(f"low free disk space: {free_gb:.1f} GB")

    if not args.quiet:
        print(f"UsedSurf doctor: {ROOT}")
        for message in warnings:
            print(f"WARN  {message}")
        for message in errors:
            print(f"ERROR {message}")
        if not errors:
            print("OK    local runtime is ready")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
