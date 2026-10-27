#!/usr/bin/env python3
"""Compare SAM 2 checkpoints on photos the current pipeline handled poorly.

Report generator only - never mutates production data or the live
checkpoint env vars. Candidate photos are selected from an existing
crop-eval batch report.json by OUTCOME (disagreed with YOLO, flagged for
review, ...), not a hardcoded file list, so the same command works on
whatever the model is currently struggling with, today or in a future run.

Each --checkpoint entry is independently configured, including whether it
uses centerline point prompts - so the same invocation can A/B box-only vs
box+points on one checkpoint, compare two checkpoints, or both at once.

Example (box-only vs point-prompted, same tiny checkpoint):
    .venv-sam2/bin/python scripts/compare_sam2_checkpoints.py \\
        --checkpoint tiny-box=sam2.1_hiera_tiny.pt:configs/sam2.1/sam2.1_hiera_t.yaml \\
        --checkpoint tiny-points=sam2.1_hiera_tiny.pt:configs/sam2.1/sam2.1_hiera_t.yaml:points
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from organizer.config import runtime_root  # noqa: E402
from organizer.crop import usable_yolo_prompt_box, yolo_cross_check_box  # noqa: E402
from organizer.ingest import read_image  # noqa: E402
from organizer.segmentation import Sam2Segmenter, _centerline_points, mask_crop, validate_board_box  # noqa: E402

ROOT = runtime_root()


_POINT_MODES = {"none", "points", "points-fg"}


@dataclass
class CheckpointSpec:
    label: str
    checkpoint: Path
    config: str
    point_mode: str = "none"


def _parse_checkpoint(spec: str) -> CheckpointSpec:
    label, _, rest = spec.partition("=")
    checkpoint, _, rest = rest.partition(":")
    config, _, flag = rest.partition(":")
    if not label or not checkpoint or not config:
        raise argparse.ArgumentTypeError(f"expected label=checkpoint_path:config[:points|:points-fg], got {spec!r}")
    point_mode = flag or "none"
    if point_mode not in _POINT_MODES:
        raise argparse.ArgumentTypeError(f"unknown checkpoint flag {flag!r} (expected 'points' or 'points-fg')")
    return CheckpointSpec(label, Path(checkpoint), config, point_mode=point_mode)


def _latest_report() -> Path:
    runs_root = ROOT / "data" / "crop-eval" / "runs"
    candidates = sorted((p / "report.json" for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise SystemExit(f"no report.json found under {runs_root}")
    return candidates[-1]


def _select_photos(report: dict, select: str, shot_types: set[str]) -> list[dict]:
    chosen = []
    for entry in report.get("files", []):
        classification = entry.get("classification") or {}
        shot_type = entry.get("shot_type")
        if shot_types and shot_type not in shot_types:
            continue
        if select == "mismatch" and not classification.get("boundary_mismatch"):
            continue
        if select == "review" and not classification.get("review"):
            continue
        if not classification.get("yolo_detection"):
            continue  # nothing to prompt SAM 2 with or cross-check against
        chosen.append(entry)
    return chosen


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="crop-eval report.json (default: latest run under data/crop-eval/runs)")
    parser.add_argument("--select", choices=["mismatch", "review", "all"], default="mismatch", help="which photos to test (default: boundary_mismatch cases - where SAM 2 and YOLO disagreed)")
    parser.add_argument("--shot-type", action="append", choices=["full_board", "side_profile"], help="restrict to one or more shot types (default: both)")
    parser.add_argument("--checkpoint", action="append", required=True, type=_parse_checkpoint, help="label=checkpoint_path:config, repeatable - compare any number of checkpoints")
    parser.add_argument("--output", type=Path, help="where to write the JSON report (default: data/evaluation/sam2-checkpoint-comparison-<timestamp>.json)")
    args = parser.parse_args()

    report_path = args.report or _latest_report()
    report = json.loads(report_path.read_text())
    shot_types = set(args.shot_type) if args.shot_type else set()
    photos = _select_photos(report, args.select, shot_types)
    if not photos:
        print(f"no photos matched --select {args.select}" + (f" --shot-type {sorted(shot_types)}" if shot_types else ""))
        return 1

    # Dedupe by (checkpoint, config) so two labels testing the same weights
    # (e.g. box-only vs +points) share one loaded model instead of loading
    # it twice.
    segmenter_cache: dict[tuple[str, str], Sam2Segmenter] = {}
    specs = {spec.label: spec for spec in args.checkpoint}
    for spec in args.checkpoint:
        key = (str(spec.checkpoint), spec.config)
        if key not in segmenter_cache:
            segmenter_cache[key] = Sam2Segmenter(spec.checkpoint, spec.config)
    totals = {label: {"score_sum": 0.0, "iou_sum": 0.0, "valid": 0, "n": 0} for label in specs}
    rows = []

    for entry in photos:
        image, _ = read_image(ROOT / entry["source"])
        width, height = image.size
        shot_type = entry["shot_type"]
        yolo = entry["classification"]["yolo_detection"]
        prompt = usable_yolo_prompt_box(yolo, width, height, shot_type)
        if prompt is None:
            prompt = (yolo["x"], yolo["y"], yolo["x"] + yolo["width"], yolo["y"] + yolo["height"])
        cross_check = yolo_cross_check_box(yolo, width, height, shot_type)

        row = {"file": entry["file"], "shot_type": shot_type}
        line = [entry["file"]]
        for label, spec in specs.items():
            segmenter = segmenter_cache[(str(spec.checkpoint), spec.config)]
            points = None
            if spec.point_mode != "none":
                foreground, background = _centerline_points(prompt, (width, height))
                points = (foreground, background if spec.point_mode == "points" else [])
            prediction = segmenter.predict(image, prompt, points=points)
            result = {"available": prediction.mask is not None, "score": prediction.score}
            if prediction.mask is not None:
                box = mask_crop(prediction.mask, width, height, padding=0.0)
                if box is not None:
                    box_corners = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                    iou = _iou(box_corners, cross_check) if cross_check else None
                    valid = validate_board_box(box, shot_type, (width, height))
                    result.update({"box": box, "iou_vs_yolo": round(iou, 3) if iou is not None else None, "validate_board_box": valid})
                    totals[label]["iou_sum"] += iou or 0.0
                    totals[label]["valid"] += bool(valid)
                totals[label]["score_sum"] += prediction.score or 0.0
                totals[label]["n"] += 1
            else:
                result["reason"] = prediction.reason
            row[label] = result
            line.append(f"{label}: score={result.get('score')} iou={result.get('iou_vs_yolo')} valid={result.get('validate_board_box')}")
        rows.append(row)
        print(" | ".join(line))

    summary = {}
    for label, agg in totals.items():
        n = max(1, agg["n"])
        summary[label] = {
            "n": agg["n"],
            "mean_score": round(agg["score_sum"] / n, 3),
            "mean_iou_vs_yolo": round(agg["iou_sum"] / n, 3),
            "validate_pass_rate": round(agg["valid"] / n, 3),
        }

    output = args.output or (ROOT / "data" / "evaluation" / f"sam2-checkpoint-comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    report_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_report": str(report_path),
        "select": args.select,
        "shot_types": sorted(shot_types) if shot_types else "all",
        "checkpoints": {spec.label: {"checkpoint": str(spec.checkpoint), "config": spec.config, "point_mode": spec.point_mode} for spec in args.checkpoint},
        "summary": summary,
        "photos": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_out, indent=2) + "\n", encoding="utf-8")

    print()
    for label, stat in summary.items():
        print(f"{label}: {stat}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
