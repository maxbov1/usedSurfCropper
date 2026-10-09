#!/usr/bin/env python3
"""Run the behavioral crop fixtures and emit reviewable proposals.

This is intentionally not an accuracy benchmark yet: the downloaded pack has
no pixel-aligned original/finished annotations. It verifies fixture coverage,
runs the current crop proposer, and creates preview rectangles for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organizer.crop import suggested_crop  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    fixture_path = root / "fixtures" / "crop-fixtures.json"
    fixture = json.loads(fixture_path.read_text())
    sample_root = root / "sample-pack" / "usedsurf-crop-sample-pack"
    sample_manifest = {item["file"]: item for item in json.loads((sample_root / "manifest.json").read_text())}
    output = root / "data" / "crop-eval"
    previews = output / "previews"
    crops = output / "crops"
    previews.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)

    files = []
    seen = set()
    for case in fixture["cases"]:
        for relative in case.get("files", []):
            if relative in seen:
                continue
            seen.add(relative)
            source = sample_root / relative
            result = {"file": relative, "cases": [], "status": "missing"}
            if source.is_file():
                board = sample_manifest.get(relative, {}).get("board", "unassigned")
                board_dir = safe_name(board)
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
                # Use the fixture case's primary type when there is one; the
                # specific type is recorded below for review.
                case_for_crop = next(c for c in fixture["cases"] if relative in c.get("files", []))
                shot_type = case_for_crop["shot_type"]
                if shot_type == "detail_or_accessory":
                    shot_type = "fins_closeup"
                crop = suggested_crop(image, shot_type)
                x, y, width, height, method = crop
                preview = image.copy()
                draw = ImageDraw.Draw(preview)
                draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#ff5a36", width=max(3, image.width // 300))
                preview.thumbnail((900, 1200))
                board_preview_dir = previews / board_dir
                board_crop_dir = crops / board_dir
                board_preview_dir.mkdir(parents=True, exist_ok=True)
                board_crop_dir.mkdir(parents=True, exist_ok=True)
                preview_path = board_preview_dir / (Path(relative).stem + ".jpg")
                crop_path = board_crop_dir / (Path(relative).stem + ".jpg")
                preview.save(preview_path, "JPEG", quality=90)
                image.crop((x, y, x + width, y + height)).save(crop_path, "JPEG", quality=95, subsampling=0)
                result.update({"status": "proposal_generated", "board": board, "board_source": "fixture_manifest_ground_truth", "width": image.width, "height": image.height, "shot_type": shot_type, "crop": {"x": x, "y": y, "width": width, "height": height}, "method": method, "model_available": not method.startswith("OpenCV rejected"), "review": "needs_manual_annotation", "annotated_preview": str(preview_path.relative_to(root)), "crop_preview": str(crop_path.relative_to(root))})
            result["cases"] = [c["id"] for c in fixture["cases"] if relative in c.get("files", [])]
            files.append(result)

    report = {
        "fixture": str(fixture_path.relative_to(root)),
        "ground_truth_status": fixture["ground_truth_status"],
        "board_assignment_source": "fixture_manifest_ground_truth_only",
        "source_files": len(files),
        "proposals_generated": sum(item["status"] == "proposal_generated" for item in files),
        "missing_files": [item["file"] for item in files if item["status"] == "missing"],
        "notes": [
            "Rectangles are proposals only; annotate expected rectangles before measuring accuracy.",
            "Optional fins_closeup has no required source file and is not a failure when absent.",
        ],
        "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("source_files", "proposals_generated", "missing_files", "ground_truth_status")}, indent=2))
    print(f"annotated originals: {previews.relative_to(root)}")
    print(f"cropped previews: {crops.relative_to(root)}")
    print(f"report: {(output / 'latest.json').relative_to(root)}")
    return 0 if not report["missing_files"] else 1


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_") or "unassigned"


if __name__ == "__main__":
    raise SystemExit(main())
