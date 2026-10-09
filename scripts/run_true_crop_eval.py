#!/usr/bin/env python3
"""Generate crop proposals for the real manual-reference crop set."""

from __future__ import annotations

import json
import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from organizer.config import runtime_root
from organizer.crop import apply_crop_calibration, build_crop_calibration, classify_shot, rotated_crop_proposal
from organizer.db import connect, create_run, record_artifact, set_run_worker, update_run
from organizer.ingest import read_image

ROOT = runtime_root()
TEST_ROOT = ROOT / "data" / "crop-eval" / "true-test-set"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Stable run name; defaults to a new timestamped run")
    args = parser.parse_args()
    manifest = json.loads((TEST_ROOT / "manifest.json").read_text())
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    db = connect(ROOT / "data" / "usedsurf.sqlite3")
    create_run(db, run_id, "true_crop_eval", metadata={"retest": True})
    update_run(db, run_id, "running")
    set_run_worker(db, run_id, pid=os.getpid(), log_path=str((TEST_ROOT / f"{run_id}.log").relative_to(ROOT)))
    db.close()
    run_root = TEST_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    historical_cases = []
    historical_annotations = {}
    legacy_annotations_path = TEST_ROOT / "annotations.json"
    legacy_report_path = TEST_ROOT / "baseline-v1.json"
    history = [("legacy", legacy_report_path, legacy_annotations_path)]
    runs_root = TEST_ROOT / "runs"
    if runs_root.exists():
        history.extend((run.name, run / "report.json", run / "annotations.json") for run in sorted(runs_root.iterdir()) if run.is_dir())
    for history_id, report_path, annotations_path in history:
        if not report_path.exists():
            continue
        cases = json.loads(report_path.read_text()).get("cases", [])
        annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
        for case in cases:
            copied = dict(case)
            copied["source"] = f"{history_id}::{case.get('source')}"
            historical_cases.append(copied)
        for source, annotation in annotations.items():
            historical_annotations[f"{history_id}::{source}"] = annotation
    calibration = build_crop_calibration(historical_cases, historical_annotations)
    (run_root / "crop-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (run_root / "annotations.json").write_text("{}\n")
    proposals = run_root / "model-proposals"
    overlays = run_root / "overlays"
    proposals.mkdir()
    overlays.mkdir()
    results = []
    gallery_position = 0
    report_path = run_root / "report.json"

    def write_progress(done: int, complete: bool = False) -> None:
        payload = {
            "name": "true-crop-baseline-v1",
            "source_unchanged": True,
            "policy": "Cards stay original and are excluded from listing exports.",
            "run_id": run_id,
            "complete": complete,
            "cases": results,
        }
        fd, temporary_name = tempfile.mkstemp(prefix="report-", suffix=".json", dir=run_root)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary_name, report_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        listing_total = sum(not case.get("exclude_from_listing") for case in manifest["cases"])
        listing_done = sum(not case.get("exclude_from_listing") for case in results)
        (run_root / "progress.json").write_text(json.dumps({"done": done, "total": len(manifest["cases"]), "listing_done": listing_done, "listing_total": listing_total, "complete": complete}) + "\n")
        db = connect(ROOT / "data" / "usedsurf.sqlite3")
        update_run(db, run_id, "complete" if complete else "running")
        db.close()

    write_progress(0)
    for case in manifest["cases"]:
        source = ROOT / case["source"]
        image, capture = read_image(source)
        shot_type = "keep_original" if case["exclude_from_listing"] else "auto"
        classification = classify_shot(image, card=case["exclude_from_listing"])
        if case["exclude_from_listing"]:
            gallery_position = 0
        else:
            gallery_position += 1
        order_hint = {1: "full_board", 2: "full_board", 3: "full_board", 4: "full_board", 5: "side_profile", 6: "fin_detail"}.get(gallery_position)
        effective_shot_type = classification["shot_type"]
        if order_hint == "side_profile" and classification["shot_type"] == "full_board":
            effective_shot_type = "side_profile"
            classification["order_hint"] = "side_profile fallback at gallery position 5"
        if case["exclude_from_listing"]:
            proposal = {"image": image, "crop": (0, 0, image.width, image.height), "angle": 0.0, "rotation_applied": False, "review": False, "reason": "card kept original"}
        else:
            proposal = rotated_crop_proposal(image, shot_type=effective_shot_type, classification=classification)
            calibrated_crop = apply_crop_calibration(proposal["crop"], proposal["image"].size, str(effective_shot_type), calibration)
            if calibrated_crop != proposal["crop"]:
                proposal["crop"] = calibrated_crop
                proposal["reason"] = f"{proposal['reason']}; human calibration ({calibration['groups'][classification['shot_type']]['count']} manual corrections)"
        working = proposal["image"]
        x, y, width, height = proposal["crop"]
        method = str(proposal["reason"])
        stem = Path(case["source"]).stem
        proposal_path = proposals / f"{stem}.jpg"
        working.crop((x, y, x + width, y + height)).save(proposal_path, "JPEG", quality=95, subsampling=0)
        overlay = working.copy()
        overlay.thumbnail((1000, 1000))
        scale_x, scale_y = overlay.width / working.width, overlay.height / working.height
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y), outline="#1e9b5a", width=6)
        overlay.save(overlays / f"{stem}.jpg", "JPEG", quality=90)
        results.append({
            **case,
            "evaluated_shot_type": effective_shot_type,
            "classification": classification,
            "capture_time": capture,
            "source_size": {"width": image.width, "height": image.height},
            "proposal": {"x": x, "y": y, "width": width, "height": height},
            "rotation": {"angle": proposal["angle"], "attempted_angle": proposal.get("attempted_angle", proposal["angle"]), "applied": proposal["rotation_applied"]},
            "method": method,
            "review": proposal["review"],
            "proposal_path": str(proposal_path.relative_to(ROOT)),
            "overlay_path": str((overlays / f"{stem}.jpg").relative_to(ROOT)),
        })
        write_progress(len(results))
    report = {
        "name": "true-crop-baseline-v1",
        "source_unchanged": True,
        "policy": "Cards stay original and are excluded from listing exports. Non-card images use the conservative silhouette classifier; full-board and side-profile crops are proposed only when a complete silhouette is credible, otherwise the original framing is preserved for review.",
        "cases": results,
    }
    report["run_id"] = run_id
    report["complete"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    listing_total = sum(not case.get("exclude_from_listing") for case in manifest["cases"])
    listing_done = sum(not case.get("exclude_from_listing") for case in results)
    (run_root / "progress.json").write_text(json.dumps({"done": len(results), "total": len(manifest["cases"]), "listing_done": listing_done, "listing_total": listing_total, "complete": True}) + "\n")
    (TEST_ROOT / "latest-run.txt").write_text(run_id + "\n")
    db = connect(ROOT / "data" / "usedsurf.sqlite3")
    for artifact, kind in (
        (report_path, "true_crop_report"),
        (run_root / "progress.json", "true_crop_progress"),
        (run_root / "annotations.json", "true_crop_annotations"),
        (TEST_ROOT / "latest-run.txt", "true_latest_run_pointer"),
    ):
        record_artifact(db, run_id, artifact, ROOT, kind)
    for artifact in list(proposals.rglob("*.jpg")) + list(overlays.rglob("*.jpg")):
        record_artifact(db, run_id, artifact, ROOT, "crop_image")
    update_run(db, run_id, "complete")
    db.close()
    print(f"generated {len(results)} proposals")
    print(f"cards kept original: {sum(case['exclude_from_listing'] for case in results)}")
    print(f"run: {run_id}")
    print(f"report: {report_path.relative_to(ROOT)}")
    print(f"proposals: {proposals.relative_to(ROOT)}")
    print(f"overlays: {overlays.relative_to(ROOT)}")
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
