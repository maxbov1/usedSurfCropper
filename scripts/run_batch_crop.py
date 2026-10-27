#!/usr/bin/env python3
"""Generate a fresh crop-review run from the current reviewed grouping."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from organizer.config import runtime_root  # noqa: E402
from organizer.crop import _full_board_crop, classify_shot, recover_outlier_full_board_box, rotated_crop_proposal, shared_full_board_padding  # noqa: E402
from organizer.db import connect, record_artifact, set_run_worker, update_run  # noqa: E402
from organizer.ingest import read_image  # noqa: E402
from organizer.export import safe_name  # noqa: E402
from organizer.segmentation import validate_board_box  # noqa: E402

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
        "SELECT p.*, b.label AS board_label, b.fins_included AS board_fins_included, b.status AS board_status "
        "FROM photos p JOIN boards b ON b.id=p.board_id "
        "WHERE p.source_path LIKE 'input/%' AND p.source_is_card=0 "
        "ORDER BY p.board_id, COALESCE(p.capture_time, '9999'), p.source_path"
    ).fetchall()
    conn.close()

    def fins_confirmed_for(photo) -> bool:
        # boards.fins_included starts as an unreviewed OCR guess (status
        # 'unreviewed') and only becomes trustworthy once a human has saved
        # or approved the board (status 'reviewing'/'approved'). Reading it
        # before that risks a bad auto-extraction silently driving a crop
        # decision - the same misclassify-then-propagate failure mode as
        # the full_board/side_profile mislabels found elsewhere today.
        if photo["board_status"] == "unreviewed":
            return False
        return str(photo["board_fins_included"] or "").strip().lower() == "yes"
    report_path = run_root / "report.json"
    results = []

    def stored_shot_type(photo) -> str:
        value = str(photo["shot_type"] or "")
        return value if value in {"full_board", "side_profile", "fin_detail", "detail"} else ""

    # Group padding only needs each photo's board-boundary box and
    # dimensions, both already sitting in the photos table from grouping
    # (organizer.segmentation.board_perimeter). Computing it from that
    # stored metadata means every board's padding - and so its first crop -
    # is ready before a single image gets decoded, instead of waiting on a
    # full priming pass over every photo in the batch.
    padding_by_board: dict[str, dict] = {}
    for board_id in {str(photo["board_id"]) for photo in photos}:
        cases = []
        for photo in photos:
            if str(photo["board_id"]) != board_id:
                continue
            shot_type = stored_shot_type(photo) or "detail"
            boundary = json.loads(photo["board_perimeter"])["box"] if photo["board_perimeter"] else None
            cases.append({"shot_type": shot_type, "classification": {"opencv_boundary": boundary} if boundary else {}, "image_size": (photo["width"], photo["height"])})
        padding_by_board[board_id] = shared_full_board_padding(cases)

    # A board-boundary detection can be confidently wrong, not just noisy:
    # measured on a real photo where YOLO (and so the SAM2 mask it guided)
    # locked onto only the right ~60% of the board every time the pipeline
    # ran fresh - and removing YOLO guidance didn't fix it either, it just
    # swapped the failure for a mask bleeding into nearly the whole frame.
    # Neither automated path is trustworthy on a photo like that.
    # validate_board_box's sibling-consistency layer catches it where
    # disagreement-based checks can't, because here every detector agrees
    # on the same wrong answer. Checked for every shot type that can have
    # more than one photo per board (today: full_board; side_profile and
    # fin_detail are capped at one each, so there's never a sibling to
    # compare against for those).
    board_photos_by_id: dict[str, list] = {}
    for photo in photos:
        board_photos_by_id.setdefault(str(photo["board_id"]), []).append(photo)

    geometry_outliers: set[str] = set()
    for board_id, board_photos in board_photos_by_id.items():
        for shot_type_key in {stored_shot_type(p) for p in board_photos}:
            if shot_type_key != "full_board":
                continue
            same_type = [p for p in board_photos if stored_shot_type(p) == shot_type_key and p["board_perimeter"]]
            boxes_by_path = {p["source_path"]: json.loads(p["board_perimeter"])["box"] for p in same_type}
            for p in same_type:
                box = boxes_by_path[p["source_path"]]
                siblings = [b for path, b in boxes_by_path.items() if path != p["source_path"]]
                if not validate_board_box(box, shot_type_key, (p["width"], p["height"]), sibling_boxes=siblings):
                    geometry_outliers.add(p["source_path"])

    # shot_type is assigned by capture position, not content, so it can be
    # wrong even when the board grouping itself is right (a real board had
    # a plain full-deck photo stuck with shot_type="fin_detail"). Box
    # geometry alone can't safely auto-correct this, though: on another
    # real board, a genuine fin close-up's SAM 2 box matched its
    # full_board siblings' width just as closely as the actually-mislabeled
    # photo did on the first board - both passed the same sibling-width
    # check. So this only flags for human review; it never reassigns
    # shot_type on its own, unlike the full_board outlier recovery above.
    #
    # Deliberately fin_detail only, not side_profile: a genuine side-profile
    # photographs the board edge-on, a physically narrower cross-section
    # than any full-deck view of the same board, so its box can never
    # legitimately match full_board sibling widths - any match there is SAM
    # 2 mask bleed inflating a narrow box, not a real mislabel (confirmed on
    # two real side-profile photos whose stored, pre-bleed-fix boxes both
    # coincidentally matched their board's full_board siblings). That
    # failure mode is already handled by the YOLO cross-check above, not
    # this one.
    possible_mislabels: set[str] = set()
    for board_id, board_photos in board_photos_by_id.items():
        full_board_boxes = [
            json.loads(p["board_perimeter"])["box"]
            for p in board_photos
            if stored_shot_type(p) == "full_board" and p["board_perimeter"]
        ]
        if len(full_board_boxes) < 2:
            continue
        for p in board_photos:
            if stored_shot_type(p) != "fin_detail" or not p["board_perimeter"]:
                continue
            box = json.loads(p["board_perimeter"])["box"]
            if validate_board_box(box, "full_board", (p["width"], p["height"]), sibling_boxes=full_board_boxes):
                possible_mislabels.add(p["source_path"])

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
        # classify_shot() runs GrabCut plus 1-3 YOLO passes per photo.
        # Grouping already paid that cost and persisted the result (see
        # web.py's /process loop) - reuse it instead of recomputing from
        # scratch here. Only photos from before this was added fall back
        # to a fresh call.
        if photo["classification_snapshot"]:
            classification = json.loads(photo["classification_snapshot"])
        else:
            classification = classify_shot(image)
        # The photo's stored shot_type already reflects grouping-time
        # correction (SAM2's board-region measurement overriding a wrong
        # OpenCV full_board/side_profile call) and any human review
        # correction. A fresh classify_shot() call here knows neither, so a
        # standing side-rail shot would silently get cropped as full_board
        # again. Trust the stored value when it's a real classification.
        shot_type = stored_shot_type(photo) or str(classification.get("shot_type", "detail"))
        classification["shot_type"] = shot_type
        # Grouping already ran SAM 2 on this photo and persisted the result
        # (see organizer.segmentation.board_perimeter) - reuse it instead of
        # predicting a fresh mask here.
        if photo["board_perimeter"]:
            classification["stored_perimeter"] = json.loads(photo["board_perimeter"])
        group_padding = padding_by_board[str(photo["board_id"])]
        declined = False
        possible_mislabel = photo["source_path"] in possible_mislabels
        if photo["source_path"] in geometry_outliers:
            # A width mismatch against full_board siblings has two possible
            # causes: the shot is genuinely mislabeled (it's really a
            # side-profile), or the detector just failed narrow on a real
            # full-board photo. Box shape alone can't tell these apart - a
            # botched full_board detection that only finds half the board
            # produces the exact same narrow geometry as a real profile
            # shot (verified on a real photo: IMG_6239, a confirmed wide
            # deck-view board, failed detection down to a 584px sliver that
            # passed every side-profile shape check just as cleanly as a
            # genuine profile would). The one signal that does distinguish
            # them: a board only ever has one real side-profile shot. If
            # this board already has a different photo holding that slot,
            # this one can't also be a genuine side-profile - it's a failed
            # detection, full stop. Recovery is only safe when the slot is
            # still open.
            slot_taken = any(
                stored_shot_type(p) == "side_profile" and p["source_path"] != photo["source_path"]
                for p in board_photos_by_id[str(photo["board_id"])]
            )
            outlier_box = json.loads(photo["board_perimeter"])["box"] if photo["board_perimeter"] else None
            if outlier_box is None and classification.get("yolo_detection"):
                yolo = classification["yolo_detection"]
                outlier_box = (yolo["x"], yolo["y"], yolo["width"], yolo["height"])
            looks_like_side_profile = (
                not slot_taken
                and outlier_box is not None
                and validate_board_box(outlier_box, "side_profile", (image.width, image.height))
            )
            classification["review"] = True
            if looks_like_side_profile:
                shot_type = "side_profile"
                classification["shot_type"] = shot_type
                proposal = rotated_crop_proposal(image, shot_type=shot_type, classification=classification, fins_confirmed=fins_confirmed_for(photo))
                db = connect(ROOT / "data" / "usedsurf.sqlite3")
                db.execute("UPDATE photos SET shot_type='side_profile' WHERE id=?", (photo["id"],))
                db.commit()
                db.close()
            elif outlier_box is not None and len(
                sibling_boxes := [
                    (json.loads(p["board_perimeter"])["box"], p["width"])
                    for p in board_photos_by_id[str(photo["board_id"])]
                    if stored_shot_type(p) == "full_board" and p["source_path"] != photo["source_path"] and p["board_perimeter"]
                ]
            ) >= 2:
                # Ruled out side-profile above, and this photo's own box
                # disagrees with its full_board siblings - but the siblings
                # themselves agree with each other (that's what "outlier"
                # means here), so their median width AND center are far more
                # trustworthy than this photo's own undershot detection. See
                # recover_outlier_full_board_box()'s docstring for the real
                # case (IMG_6481) this is built from.
                corrected_box = recover_outlier_full_board_box(outlier_box, sibling_boxes, image.width)
                crop = _full_board_crop(*corrected_box, image.width, image.height, float(group_padding["vertical"]), float(group_padding["horizontal"]))
                classification["opencv_boundary"] = list(corrected_box)
                classification["boundary_source"] = "sibling-width recovery"
                proposal = {
                    "image": image, "crop": crop, "angle": 0.0, "rotation_applied": False,
                    "reason": f"board width inconsistent with sibling photos ({round(outlier_box[2])}px vs {corrected_box[2]}px sibling-recovered) - recovered using sibling-consistent box, flagged for review",
                }
            else:
                declined = True
                proposal = {"image": image, "crop": (0, 0, image.width, image.height), "angle": 0.0, "rotation_applied": False, "reason": "board width inconsistent with sibling photos; manual crop required"}
        else:
            proposal = rotated_crop_proposal(
                image,
                shot_type=shot_type,
                padding=float(group_padding["vertical"]),
                rail_padding=float(group_padding["horizontal"]),
                classification=classification,
                fins_confirmed=fins_confirmed_for(photo),
            )
        if possible_mislabel:
            proposal["review"] = True
            classification["review"] = True
            classification["possible_shot_type_mislabel"] = True
            proposal["reason"] = f"{proposal.get('reason', '')} - shot_type '{shot_type}' may be wrong: detected box closely matches this board's full_board photos, flagged for review".lstrip(" -")
        classification["group_padding"] = group_padding
        working = proposal["image"]
        x, y, width, height = proposal["crop"]
        manual_required = declined or "manual crop required" in str(proposal.get("reason", "")).lower()
        board = safe_name(photo["board_label"] or f"board-{photo['board_id']}")
        stem = Path(photo["source_path"]).stem
        crop_path = run_root / "proposals" / board / f"{stem}.jpg"
        # Tagged on the overlay only, never the original/proposal: this is
        # the "mutated" debug image (final crop box drawn on it), so its
        # filename alone should say what it is at a glance when pasted
        # somewhere without the surrounding report.json for context.
        debug_status = "review" if classification.get("review") else "ok"
        overlay_path = run_root / "overlays" / board / f"{stem}__{shot_type}__{debug_status}.jpg"
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
            "status": "proposal_generated", "model_available": not manual_required, "manual_required": manual_required,
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
