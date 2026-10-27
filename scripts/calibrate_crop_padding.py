#!/usr/bin/env python3
"""Report how well current crop padding constants match human corrections.

Report generator only - never edits organizer/crop.py or any padding
constant itself. Walks every batch run's saved manual corrections,
reconstructs the padding percentage each one implies (how much margin the
human's crop added around the same raw detected box the model padded), and
compares it to today's default per shot type. A human reads the report and
decides whether/how to update a constant; this script never writes one -
same "report generator, not a training command" rule already used by
evaluate_six_field_candidate.py. Not wired into any UI - backend/CLI only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from organizer.config import runtime_root  # noqa: E402
from organizer.crop import DEFAULT_HORIZONTAL_PADDING, DEFAULT_PROFILE_HORIZONTAL_PADDING, DEFAULT_PROFILE_PADDING, DEFAULT_VERTICAL_PADDING  # noqa: E402

ROOT = runtime_root()

CURRENT_HORIZONTAL = {"full_board": DEFAULT_HORIZONTAL_PADDING, "side_profile": DEFAULT_PROFILE_HORIZONTAL_PADDING}
CURRENT_VERTICAL = {"full_board": DEFAULT_VERTICAL_PADDING, "side_profile": DEFAULT_PROFILE_PADDING}
# Below this many corrected examples in a shot type, a derived median isn't
# trustworthy - same principle as the sibling-width-consistency check in
# segmentation.py, which also refuses to trust a median from under 2
# siblings. 5 is a deliberately low bar, not a statistically rigorous one -
# it exists to suppress reports built from 1-2 noisy points, not to claim
# real confidence at 5.
MIN_SAMPLES_TO_TRUST = 5
# A case whose implied percentage is more than 2x (or under 0.5x) its own
# stratum's median gets flagged, not silently averaged in - real example
# (IMG_6472, 2026-10-18): a correction that looked like a padding signal
# was actually a different bug (the SAM2/YOLO override had picked the
# wrong box entirely), and pooling it with genuine padding corrections
# would have quietly skewed the result.
OUTLIER_RATIO = 2.0


def _collect_corrections() -> list[dict]:
    """One row per corrected photo, deduped to its most recently saved correction."""
    by_source: dict[str, dict] = {}
    runs_root = ROOT / "data" / "crop-eval" / "runs"
    if not runs_root.exists():
        return []
    for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime):
        report_path = run_dir / "report.json"
        annotations_path = run_dir / "annotations.json"
        if not report_path.exists() or not annotations_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
            annotations = json.loads(annotations_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        by_file = {item.get("file", item.get("source")): item for item in report.get("files", [])}
        for source, annotation in annotations.items():
            if annotation.get("decision") != "manual":
                continue
            item = by_file.get(source)
            expected = annotation.get("expected_crop")
            if not item or not expected:
                continue
            classification = item.get("classification", {})
            raw_box = classification.get("opencv_boundary")
            if not raw_box or len(raw_box) < 4:
                continue
            # A fully declined photo (model shipped the whole original
            # frame because nothing was usable) isn't a padding correction
            # - there's no padding ratio to imply when the model proposed
            # nothing to pad in the first place.
            model_crop = annotation.get("model_crop") or {}
            if (model_crop.get("x"), model_crop.get("y"), model_crop.get("width"), model_crop.get("height")) == (0, 0, item.get("width"), item.get("height")):
                continue
            by_source[source] = {
                "source": source,
                "run": run_dir.name,
                "shot_type": item.get("evaluated_shot_type") or item.get("shot_type"),
                "boundary_source": classification.get("boundary_source", ""),
                "raw_box": raw_box,
                "expected_crop": expected,
            }
    return list(by_source.values())


def _analyze(corrections: list[dict]) -> dict:
    by_shot_type: dict[str, list[dict]] = defaultdict(list)
    for row in corrections:
        x, y, w, h = row["raw_box"][:4]
        ex = row["expected_crop"]
        if not w or not h:
            continue
        by_shot_type[row["shot_type"]].append({
            **row,
            "horizontal_pct": round(((ex["width"] - w) / 2) / w, 3),
            "vertical_pct": round(((ex["height"] - h) / 2) / h, 3),
        })

    report = {}
    for shot_type, rows in by_shot_type.items():
        h_vals = [r["horizontal_pct"] for r in rows]
        h_median = statistics.median(h_vals)
        outliers = [r["source"] for r in rows if h_median and not (0.5 <= r["horizontal_pct"] / h_median <= OUTLIER_RATIO)] if h_median else []
        report[shot_type] = {
            "n": len(rows),
            "trustworthy": len(rows) >= MIN_SAMPLES_TO_TRUST,
            "current_horizontal_padding": CURRENT_HORIZONTAL.get(shot_type),
            "implied_horizontal_padding_median": round(h_median, 3),
            "implied_horizontal_padding_mean": round(sum(h_vals) / len(h_vals), 3),
            "current_vertical_padding": CURRENT_VERTICAL.get(shot_type),
            "implied_vertical_padding_median": round(statistics.median(r["vertical_pct"] for r in rows), 3),
            "outliers_flagged": outliers,
            "cases": [{"source": r["source"], "boundary_source": r["boundary_source"], "horizontal_pct": r["horizontal_pct"], "vertical_pct": r["vertical_pct"]} for r in rows],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "crop-padding-calibration.json")
    args = parser.parse_args()
    corrections = _collect_corrections()
    report = _analyze(corrections)
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "total_corrections_considered": len(corrections), "min_samples_to_trust": MIN_SAMPLES_TO_TRUST, "by_shot_type": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"considered {len(corrections)} manual corrections across all runs\n")
    for shot_type, stats in report.items():
        trust = "enough samples" if stats["trustworthy"] else f"NOT ENOUGH SAMPLES YET (need {MIN_SAMPLES_TO_TRUST}+)"
        print(f"{shot_type}: n={stats['n']} ({trust})")
        print(f"  horizontal: current={stats['current_horizontal_padding']}  implied_median={stats['implied_horizontal_padding_median']}  implied_mean={stats['implied_horizontal_padding_mean']}")
        print(f"  vertical:   current={stats['current_vertical_padding']}  implied_median={stats['implied_vertical_padding_median']}")
        if stats["outliers_flagged"]:
            print(f"  flagged (check before trusting the median, e.g. a different bug masquerading as a padding signal): {stats['outliers_flagged']}")
        print()
    print(f"full report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
