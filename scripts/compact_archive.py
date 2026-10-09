#!/usr/bin/env python3
"""Safely compact an archived batch after reviewing its retention manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organizer.retention import compact_archive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="archive directory, for example data/archive/cleanup-...")
    parser.add_argument("--apply", action="store_true", help="delete only verified ordinary originals")
    args = parser.parse_args()
    result = compact_archive(args.archive.resolve(), apply=args.apply)
    mode = "removed" if args.apply else "eligible"
    print(f"{mode}: {result['removed'] if args.apply else result['planned']}; skipped: {result['skipped']}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing retention-manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
