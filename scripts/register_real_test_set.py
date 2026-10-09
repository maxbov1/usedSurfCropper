#!/usr/bin/env python3
"""Register an untouched input batch and create separate crop proposals.

The source photos stay in input/. This creates a manifest and proposal JPEGs
only; it never edits or replaces the originals.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organizer.crop import suggested_crop  # noqa: E402
from organizer.ingest import SUPPORTED, read_image, scan_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default="real-batch-2026-10-05")
    args = parser.parse_args()
    root = args.root.resolve()
    test_root = root / "data" / "test-sets" / args.name
    proposals_root = test_root / "proposals"
    proposals_root.mkdir(parents=True, exist_ok=True)

    scanned = [item for item in scan_files(root / "input") if "error" not in item]
    records = []
    for item in scanned:
        source = item["path"]
        image, capture = read_image(source)
        x, y, width, height, method = suggested_crop(image, "full_board")
        proposal_path = proposals_root / f"{source.stem}.jpg"
        image.crop((x, y, x + width, y + height)).save(proposal_path, "JPEG", quality=95, subsampling=0)
        records.append(
            {
                "source": str(source.relative_to(root)),
                "source_hash": item["hash"],
                "capture_time": capture,
                "source_format": image.format or "JPEG",
                "orientation_normalized_size": {"width": image.width, "height": image.height},
                "is_card_prediction": item["is_card"],
                "ocr_identifier": item["identifier"],
                "proposal": {"x": x, "y": y, "width": width, "height": height},
                "method": method,
                "proposal_path": str(proposal_path.relative_to(root)),
                "review_status": "unreviewed",
            }
        )

    manifest = {
        "name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "originals remain in input and are never modified",
        "manual_reference_policy": "place matching approved crops in approved-crops/ using the same basenames",
        "processing": "full_board proposal pass; shot type remains subject to review",
        "files": records,
    }
    test_root.mkdir(parents=True, exist_ok=True)
    (test_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Registered {len(records)} originals")
    print(f"Manifest: {test_root.relative_to(root)}/manifest.json")
    print(f"Proposals: {proposals_root.relative_to(root)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
