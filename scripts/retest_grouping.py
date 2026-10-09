#!/usr/bin/env python3
"""Run a non-destructive grouping retest against the saved human corrections.

This is an evaluation pass, not a training pass and not a database migration.
The candidate grouping uses capture timestamps only to present the batch in
shoot order. It does not treat the first detected card as a hard boundary.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organizer.crop import _find_board_candidate
from organizer.ingest import read_image, scan_files
from organizer.visual import group_visual_check, signature, similarity

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


MIN_PHOTOS = 5
PREFERRED_PHOTOS = 6
MAX_PHOTOS = 7
VISUAL_WEIGHT = 0.6


@dataclass
class Candidate:
    start: int
    end: int
    score: float
    card_count: int


def segment_score(items: list[dict], start: int, end: int) -> Candidate:
    group = items[start:end]
    length = len(group)
    cards = sum(bool(item.get("is_card")) for item in group)
    # Six is the normal shoot, seven is the optional fin/detail extension,
    # and five is allowed only so the retest can expose an underfilled group.
    score = {5: 1.5, 6: 0.0, 7: 0.35}[length]
    # A card can occur anywhere in a board shoot. Missing cards are reviewable,
    # but multiple card candidates in one window are a stronger warning.
    score += {0: 0.8, 1: 0.0}.get(cards, 4.0 + cards)
    return Candidate(start, end, score, cards)


def revision_cost(items: list[dict], candidate: Candidate) -> float:
    group = items[candidate.start : candidate.end]
    score = segment_score(items, candidate.start, candidate.end).score
    visual = group_visual_check(group)["adjacent_similarity"]
    if visual is not None:
        score += (1.0 - visual) * VISUAL_WEIGHT
    return score


def window_cost(items: list[dict], candidates: list[Candidate]) -> float:
    cost = sum(revision_cost(items, candidate) for candidate in candidates)
    for left, right in zip(candidates, candidates[1:]):
        left_items = items[left.start : left.end]
        right_items = items[right.start : right.end]
        # A card at the very end of a group with no card in the next group is
        # a boundary ambiguity. Prefer the alternative where that card starts
        # the next shoot, but only as soft evidence.
        if left_items and left_items[-1].get("is_card") and not any(item.get("is_card") for item in right_items):
            cost += 0.35
    return cost


def compositions(total: int, count: int) -> list[list[int]]:
    if count == 0:
        return [[]] if total == 0 else []
    result = []
    for length in range(MIN_PHOTOS, MAX_PHOTOS + 1):
        for rest in compositions(total - length, count - 1):
            result.append([length, *rest])
    return result


def revise_groups(items: list[dict], groups: list[Candidate]) -> list[Candidate]:
    """Make only local, visually supported boundary revisions.

    The baseline partition remains authoritative unless a 3- or 4-group
    neighborhood gets a lower combined cost. This permits coordinated shifts
    such as moving one frame through several adjacent boundaries without
    letting visual similarity globally rearrange the shoot.
    """
    revised = list(groups)
    for window_size in (4, 3):
        start_group = 0
        while start_group + window_size <= len(revised):
            window = revised[start_group : start_group + window_size]
            start = window[0].start
            end = window[-1].end
            baseline_cost = window_cost(items, window)
            best_cost = baseline_cost
            best_lengths = None
            for lengths in compositions(end - start, window_size):
                candidates = []
                cursor = start
                valid = True
                for length in lengths:
                    candidate = segment_score(items, cursor, cursor + length)
                    if candidate.card_count > 1:
                        valid = False
                        break
                    candidates.append(candidate)
                    cursor += length
                if not valid:
                    continue
                cost = window_cost(items, candidates)
                if cost < best_cost - 0.02:
                    best_cost = cost
                    best_lengths = lengths
            if best_lengths:
                cursor = start
                replacement = []
                for length in best_lengths:
                    replacement.append(segment_score(items, cursor, cursor + length))
                    cursor += length
                revised[start_group : start_group + window_size] = replacement
            start_group += 1
    return revised


def partition(items: list[dict]) -> list[Candidate]:
    """Find a low-penalty partition into 5–7-photo shoots.

    This deliberately has no visual similarity or learned component. It is a
    small baseline to test the user's known shoot contract before adding a
    heavier detector.
    """
    n = len(items)
    best: list[tuple[float, list[Candidate]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for end in range(1, n + 1):
        for length in range(MIN_PHOTOS, MAX_PHOTOS + 1):
            start = end - length
            if start < 0 or best[start] is None:
                continue
            candidate = segment_score(items, start, end)
            # One inventory card identifies at most one board. Never allow a
            # candidate window to combine two cards, even when its size is 5–7.
            if candidate.card_count > 1:
                continue
            previous_score, previous = best[start]
            proposal = (previous_score + candidate.score, previous + [candidate])
            if best[end] is None or proposal[0] < best[end][0]:
                best[end] = proposal
    if best[n] is None:
        raise RuntimeError(f"could not partition {n} photos into {MIN_PHOTOS}-{MAX_PHOTOS}-photo groups")
    return best[n][1]


def load_reference(root: Path) -> tuple[dict[str, str], set[str]]:
    conn = sqlite3.connect(root / "data" / "usedsurf.sqlite3")
    conn.row_factory = sqlite3.Row
    reference = {}
    for row in conn.execute(
        """SELECT source_path, board_id FROM photos
           WHERE source_path LIKE 'input/IMG_%' AND board_id IS NOT NULL"""
    ):
        reference[row["source_path"]] = str(row["board_id"])
    excluded = {row["source_path"] for row in conn.execute("SELECT source_path FROM excluded_sources")}
    conn.close()
    return reference, excluded


def likely_fin_or_detail(item: dict) -> bool:
    """Return a review hint for a possible optional close-up.

    This is not a shot-type prediction. A normal listing image should contain
    a tall board-shaped candidate; when it does not, the image is worth
    checking as a fin/detail shot. The original remains unchanged.
    """
    if cv2 is None or np is None or item.get("is_card"):
        return False
    try:
        image, _ = read_image(item["path"])
        image.thumbnail((900, 900))
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return _find_board_candidate(array, "full_board") is None
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/grouping-eval/real-batch-2026-10-05-retest.json", help="report path relative to the project root")
    args = parser.parse_args()
    root = ROOT
    reference, excluded = load_reference(root)
    scanned = [item for item in scan_files(root / "input") if str(item["path"].relative_to(root)) not in excluded]
    for item in scanned:
        item["visual_signature"] = signature(item["path"])
    baseline_groups = partition(scanned)
    groups = revise_groups(scanned, baseline_groups)

    proposed_by_source = {}
    output_groups = []
    for index, group in enumerate(groups, start=1):
        sources = [str(item["path"].relative_to(root)) for item in scanned[group.start : group.end]]
        group_items = scanned[group.start : group.end]
        visual_check = group_visual_check(group_items)
        card_items = [item for item in group_items if item.get("is_card")]
        fin_candidates = [
            str(item["path"].relative_to(root))
            for item in group_items
            if likely_fin_or_detail(item)
        ]
        proposal_id = f"retest-{index:02d}"
        for source in sources:
            proposed_by_source[source] = proposal_id
        output_groups.append({
            "id": proposal_id,
            "photo_count": len(sources),
            "card_candidates": [str(item["path"].relative_to(root)) for item in card_items],
            "card_identifiers": [{"source": str(item["path"].relative_to(root)), "identifier": item.get("identifier", {})} for item in card_items],
            "possible_fin_or_detail": fin_candidates,
            "visual_check": visual_check,
            "seven_photo_fin_check_passed": len(sources) != 7 or bool(fin_candidates),
            "needs_review": len(sources) < PREFERRED_PHOTOS or (len(sources) == 7 and not fin_candidates),
            "sources": sources,
        })

    evaluated = []
    for source, expected in sorted(reference.items()):
        if source in excluded or source not in proposed_by_source:
            continue
        evaluated.append({
            "source": source,
            "reference_board_id": expected,
            "proposed_group": proposed_by_source[source],
        })

    reference_sets: dict[str, set[str]] = {}
    proposal_sets: dict[str, set[str]] = {}
    for row in evaluated:
        reference_sets.setdefault(row["reference_board_id"], set()).add(row["source"])
        proposal_sets.setdefault(row["proposed_group"], set()).add(row["source"])
    exact_matches = 0
    for sources in proposal_sets.values():
        if sources in reference_sets.values():
            exact_matches += 1

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch": "real-batch-2026-10-05",
        "method": {
            "name": "contract_partition_v1",
            "description": "Capture order is presentation order; groups are selected as 5–7-photo windows with six preferred. Card candidates may occur anywhere, but each group may contain at most one inventory card.",
            "visual_layer": "Local OpenCV thumbnail signature using grayscale structure, HSV histogram, edges, and board-region position. Used only by a bounded 3–4-group revisor after the deterministic partition; it does not deduplicate views or override hard rules.",
            "not_training": True,
        },
        "excluded": sorted(excluded),
        "groups": output_groups,
        "evaluation": {
            "reference_photo_count": len(evaluated),
            "reference_group_count": len(reference_sets),
            "proposed_group_count": len(proposal_sets),
            "exact_photo_set_matches": exact_matches,
            "groups_needing_size_or_fin_review": sum(group["needs_review"] for group in output_groups),
            "photo_assignments": evaluated,
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {len(scanned)} photos into {len(groups)} proposed groups")
    print(f"exact photo-set matches: {exact_matches}/{len(reference_sets)} reference groups")
    print(f"groups needing size or fin review: {payload['evaluation']['groups_needing_size_or_fin_review']}")
    print(f"saved {output.relative_to(root)}")
    for group in output_groups:
        cards = ", ".join(Path(source).name for source in group["card_candidates"]) or "none detected"
        fins = ", ".join(Path(source).name for source in group["possible_fin_or_detail"]) or "none detected"
        review = " REVIEW" if group["needs_review"] else ""
        print(f"{group['id']}: {group['photo_count']} photos; cards: {cards}; possible fin/detail: {fins}{review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
