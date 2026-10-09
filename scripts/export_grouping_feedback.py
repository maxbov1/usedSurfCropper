#!/usr/bin/env python3
"""Export grouping predictions, human corrections, and exclusions for review."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "usedsurf.sqlite3"
    output = root / "data" / "grouping-eval" / "real-batch-2026-10-05.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    records = []
    for row in conn.execute(
        """SELECT p.source_path, p.content_hash, p.capture_time,
                  gp.predicted_board_id, pb.label predicted_label,
                  p.board_id corrected_board_id, cb.label corrected_label,
                  gc.corrected_at, p.human_correction
           FROM photos p
           LEFT JOIN grouping_predictions gp ON gp.photo_id=p.id
           LEFT JOIN grouping_corrections gc ON gc.photo_id=p.id
           LEFT JOIN boards pb ON pb.id=gp.predicted_board_id
           LEFT JOIN boards cb ON cb.id=p.board_id
           WHERE p.source_path LIKE 'input/IMG_%'
           ORDER BY COALESCE(p.capture_time, '9999'), p.source_path"""
    ):
        records.append(dict(row))
    exclusions = [dict(row) for row in conn.execute("SELECT * FROM excluded_sources ORDER BY source_path")]
    feedback = [dict(row) for row in conn.execute(
        """SELECT f.*, b.status, b.pipeline_version, b.created_at batch_created_at, b.saved_at batch_saved_at
           FROM grouping_feedback f
           JOIN grouping_feedback_batches b ON b.run_id=f.run_id
           WHERE b.status='saved'
           ORDER BY b.created_at, f.source_path"""
    )]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch": "real-batch-2026-10-05",
        "rules_under_test": [
            "capture sequence is ordering evidence, not board identity",
            "inventory card may occur anywhere within a board shoot",
            "six photos is standard including the card; seven is allowed for fins",
            "underfilled groups require an OCR/neighbor review before acceptance",
        ],
        "photos": records,
        "saved_grouping_feedback": feedback,
        "excluded": exclusions,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {len(records)} photo records and {len(exclusions)} exclusions to {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
