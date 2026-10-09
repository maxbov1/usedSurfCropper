from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .crop import save_crop

SHOT_ORDER = {
    "card": -1,
    "deck_full_light": 10,
    "deck_low_light": 20,
    "bottom_full_light": 30,
    "bottom_low_light": 40,
    "rail_full_light": 50,
    "fins_closeup": 60,
    "detail": 70,
    "accessory": 80,
    "keep_original": 90,
}


def safe_name(value: str, fallback: str = "unknown") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip(".-_")
    return value or fallback


def export_board(root: Path, board, photos) -> Path:
    base = root / "output"
    stem = "-".join(safe_name(v) for v in (board["shaper"], board["model"], board["sku"]))
    destination = base / stem
    if destination.exists():
        manifest = destination / "manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if existing.get("board_id") == board["id"]:
                return destination
        suffix = 2
        while (base / f"{stem}-{suffix}").exists():
            suffix += 1
        destination = base / f"{stem}-{suffix}"
    temp = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=base))
    try:
        exported = []
        number = 1
        ordered_photos = sorted(photos, key=lambda photo: (SHOT_ORDER.get(photo["shot_type"], 999), photo["capture_time"] or "9999", photo["source_path"]))
        for photo in ordered_photos:
            if photo["source_is_card"] or photo["shot_type"] == "card":
                continue
            output_name = f"{number:03d}.jpg"
            crop = (photo["crop_x"], photo["crop_y"], photo["crop_w"], photo["crop_h"])
            save_crop(root / photo["source_path"], temp / output_name, crop)
            exported.append({"source": photo["source_path"], "export": output_name, "shot_type": photo["shot_type"], "crop": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]}, "review_status": photo["review_status"]})
            number += 1
        manifest = {"board_id": board["id"], "sku": board["sku"], "shaper": board["shaper"], "model": board["model"], "fin_system": board["fin_system"], "fins_included": board["fins_included"], "status": "approved", "processing_version": board["processing_version"], "photos": exported}
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temp, destination)
        return destination
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
