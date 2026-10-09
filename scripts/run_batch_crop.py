#!/usr/bin/env python3
"""Generate a fresh crop-review run from the current reviewed grouping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from organizer.config import runtime_root  # noqa: E402
from organizer.crop import classify_shot, rotated_crop_proposal  # noqa: E402
from organizer.db import connect, record_artifact, set_run_worker, update_run  # noqa: E402
from organizer.ingest import read_image  # noqa: E402
from organizer.export import safe_name  # noqa: E402

ROOT = runtime_root()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    db = connect(ROOT / "data" / "usedsurf.sqlite3")
    update_run(db, args.run_id, "running")
    set_run_worker(db, args.run_id, pid=os.getpid(), log_path=str((ROOT / "data" / "crop-eval" / f"{args.run_id}.log").relative_to(ROOT)))
    db.close()
    run_root = ROOT / "data" / "crop-eval" / "runs" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "proposals").mkdir()
    (run_root / "overlays").mkdir()
    (run_root / "annotations.json").write_text("{}\n")

    conn = sqlite3.connect(ROOT / "data" / "usedsurf.sqlite3")
    conn.row_factory = sqlite3.Row
    photos = conn.execute(
        "SELECT p.*, b.label AS board_label FROM photos p JOIN boards b ON b.id=p.board_id "
        "WHERE p.source_path LIKE 'input/%' AND p.source_is_card=0 "
        "ORDER BY p.board_id, COALESCE(p.capture_time, '9999'), p.source_path"
    ).fetchall()
    conn.close()
    report_path = run_root / "report.json"
    results = []

    def write_progress(complete: bool = False) -> None:
        payload = {"run_id": args.run_id, "complete": complete, "files": results}
        temporary = run_root / "report.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, report_path)
        (run_root / "progress.json").write_text(json.dumps({
            "run_id": args.run_id, "done": len(results), "total": len(photos),
            "listing_done": len(results), "listing_total": len(photos), "complete": complete,
        }) + "\n")
        db = connect(ROOT / "data" / "usedsurf.sqlite3")
        update_run(db, args.run_id, "complete" if complete else "running")
        db.close()

    write_progress()
    for photo in photos:
        source = ROOT / photo["source_path"]
        image, _ = read_image(source)
        classification = classify_shot(image)
        shot_type = str(classification.get("shot_type", "detail"))
        proposal = rotated_crop_proposal(image, shot_type=shot_type, classification=classification)
        working = proposal["image"]
        x, y, width, height = proposal["crop"]
        board = safe_name(photo["board_label"] or f"board-{photo['board_id']}")
        stem = Path(photo["source_path"]).stem
        crop_path = run_root / "proposals" / board / f"{stem}.jpg"
        overlay_path = run_root / "overlays" / board / f"{stem}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        working.crop((x, y, x + width, y + height)).save(crop_path, "JPEG", quality=95, subsampling=0)
        preview = working.copy()
        preview.thumbnail((1000, 1000))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(preview)
        sx, sy = preview.width / working.width, preview.height / working.height
        draw.rectangle((x * sx, y * sy, (x + width) * sx, (y + height) * sy), outline="#25a866", width=6)
        preview.save(overlay_path, "JPEG", quality=90)
        results.append({
            "file": photo["source_path"], "source": photo["source_path"], "board": photo["board_label"],
            "status": "proposal_generated", "model_available": bool(classification.get("yolo_detection")) or shot_type not in {"full_board", "side_profile"},
            "width": image.width, "height": image.height, "source_size": {"width": image.width, "height": image.height},
            "shot_type": shot_type, "evaluated_shot_type": shot_type,
            "crop": {"x": x, "y": y, "width": width, "height": height},
            "proposal": {"x": x, "y": y, "width": width, "height": height},
            "method": proposal.get("reason", ""), "classification": classification,
            "rotation": {"angle": proposal.get("angle", 0.0), "applied": proposal.get("rotation_applied", False)},
            "review": True, "crop_preview": str(crop_path.relative_to(ROOT)),
            "overlay_path": str(overlay_path.relative_to(ROOT)),
        })
        write_progress()
    write_progress(complete=True)
    latest_path = ROOT / "data" / "crop-eval" / "latest-run.txt"
    latest_path.write_text(args.run_id + "\n")
    db = connect(ROOT / "data" / "usedsurf.sqlite3")
    for artifact, kind in (
        (report_path, "crop_report"),
        (run_root / "progress.json", "crop_progress"),
        (run_root / "annotations.json", "crop_annotations"),
        (latest_path, "latest_run_pointer"),
    ):
        record_artifact(db, args.run_id, artifact, ROOT, kind)
    for artifact in list((run_root / "proposals").rglob("*.jpg")) + list((run_root / "overlays").rglob("*.jpg")):
        record_artifact(db, args.run_id, artifact, ROOT, "crop_image")
    update_run(db, args.run_id, "complete")
    db.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if "--run-id" in sys.argv:
            try:
                run_id = sys.argv[sys.argv.index("--run-id") + 1]
                db = connect(ROOT / "data" / "usedsurf.sqlite3")
                update_run(db, run_id, "failed", error=str(exc))
                db.close()
            except Exception:
                pass
        raise
