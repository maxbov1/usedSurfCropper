#!/usr/bin/env python3
"""Safely compact generated crop history while retaining review evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def is_reviewed(run_root: Path) -> bool:
    report_path = run_root / "report.json"
    annotations_path = run_root / "annotations.json"
    try:
        report = json.loads(report_path.read_text())
        annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not report.get("complete"):
        return False
    items = [item for item in report.get("files", report.get("cases", [])) if not item.get("exclude_from_listing")]
    return bool(items) and all(annotations.get(item.get("source", item.get("file", "")), {}).get("decision") in {"good", "manual"} for item in items)


def file_targets(crop_root: Path) -> tuple[list[Path], dict]:
    run_root = crop_root / "runs"
    normal_runs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    reviewed = sorted((path for path in normal_runs if is_reviewed(path)), key=lambda path: path.stat().st_mtime, reverse=True)
    keep_run = reviewed[0] if reviewed else None
    if keep_run is None:
        latest_pointer = crop_root / "latest-run.txt"
        if latest_pointer.exists():
            candidate = run_root / latest_pointer.read_text().strip()
            if candidate.is_dir():
                # A run with no saved annotation file is still the user's
                # current review surface; protect it rather than guessing.
                keep_run = candidate
    targets: list[Path] = []
    categories: dict[str, int] = {"normal_generated": 0, "true_generated": 0}

    for path in normal_runs:
        if keep_run and path == keep_run:
            continue
        for folder_name in ("proposals", "overlays"):
            folder = path / folder_name
            if folder.exists():
                targets.extend(item for item in folder.rglob("*") if item.is_file())
                categories["normal_generated"] += sum(1 for item in folder.rglob("*") if item.is_file())

    true_runs = crop_root / "true-test-set" / "runs"
    if true_runs.exists():
        for path in true_runs.iterdir():
            if not path.is_dir():
                continue
            for folder_name in ("model-proposals", "overlays"):
                folder = path / folder_name
                if folder.exists():
                    targets.extend(item for item in folder.rglob("*") if item.is_file())
                    categories["true_generated"] += sum(1 for item in folder.rglob("*") if item.is_file())
    summary = {"keep_reviewed_run": str(keep_run.relative_to(ROOT)) if keep_run else None, "files": len(targets), "bytes": sum(path.stat().st_size for path in targets), **categories}
    return targets, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="remove the generated files listed by the dry run")
    args = parser.parse_args()
    crop_root = ROOT / "data" / "crop-eval"
    targets, summary = file_targets(crop_root)
    print(json.dumps(summary, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to remove these generated proposal/overlay files.")
        return 0

    manifest = crop_root / f"retention-cleanup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    manifest.write_text(json.dumps({"summary": summary, "files": [str(path.relative_to(ROOT)) for path in targets]}, indent=2) + "\n")
    removed = 0
    for path in targets:
        if path.is_file():
            path.unlink()
            removed += 1
    for directory in sorted({path.parent for path in targets}, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    print(f"Removed {removed} generated files. Cleanup manifest: {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
