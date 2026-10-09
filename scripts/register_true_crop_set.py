#!/usr/bin/env python3
"""Register manually cropped reference images against untouched input sources."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "data" / "crop-eval" / "true-test-set"


def source_stem(reference: Path) -> str:
    stem = reference.stem
    if stem.endswith(" 2"):
        stem = stem[:-2]
    return stem.upper()


def main() -> int:
    source_by_stem = {path.stem.upper(): path for path in (ROOT / "input").iterdir() if path.is_file()}
    conn = sqlite3.connect(ROOT / "data" / "usedsurf.sqlite3")
    conn.row_factory = sqlite3.Row
    card_sources = {row["source_path"] for row in conn.execute("SELECT source_path FROM photos WHERE source_is_card=1")}
    cases = []
    missing = []
    for reference in sorted((TEST_ROOT / "reference-crops").iterdir()):
        if reference.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        stem = source_stem(reference)
        source = source_by_stem.get(stem)
        if source is None:
            missing.append(reference.name)
            continue
        source_relative = str(source.relative_to(ROOT))
        cases.append({
            "source": source_relative,
            "reference_crop": str(reference.relative_to(ROOT)),
            "shot_type": "card" if source_relative in card_sources else "needs_label",
            "exclude_from_listing": source_relative in card_sources,
        })
    payload = {
        "name": "true-crop-test-set",
        "version": 1,
        "source_root": "input",
        "reference_root": "data/crop-eval/true-test-set/reference-crops",
        "cases": cases,
        "missing_sources": missing,
    }
    (TEST_ROOT / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"registered {len(cases)} reference crops")
    print(f"card references excluded from listing: {sum(case['exclude_from_listing'] for case in cases)}")
    print(f"shot types needing labels: {sum(case['shot_type'] == 'needs_label' for case in cases)}")
    if missing:
        print("missing source matches:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
