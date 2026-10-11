"""Small, local, invisible learning loop for board grouping.

The human's saved shuffleboard is the label.  This module deliberately keeps
the first learner boring and inspectable: it stores pairwise board-identity
distances and learns the relative weight of color/layout/edge/geometry/shot
terms.  It never copies the original photos and never requires a training
button or a new user workflow.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

from .ingest import board_identity_comparison


FEATURE_NAMES = ("color", "layout", "edges", "geometry", "shot")
MIN_POSITIVE_PAIRS = 20
MIN_NEGATIVE_PAIRS = 20
MAX_POSITIVE_PAIRS = 250
MAX_NEGATIVE_PAIRS = 500


def _paths(data_root: Path) -> dict[str, Path]:
    root = data_root / "grouping-learning"
    return {
        "root": root,
        "runs": root / "runs",
        "examples": root / "examples.jsonl",
        "batches": root / "batches.jsonl",
        "weights": root / "weights.json",
    }


def write_feature_snapshot(data_root: Path, run_id: str, items: list[dict]) -> Path | None:
    """Persist compact board features for a grouping run.

    The snapshot is intentionally made during processing, while the image is
    already in memory.  Saving later can therefore use exactly the values the
    grouper saw, without rereading 129 originals or duplicating them.
    """
    records = []
    for item in items:
        signature = item.get("board_identity_signature")
        if not signature or item.get("is_card") or "error" in item:
            continue
        source = Path(item.get("path", ""))
        try:
            source_name = str(source.relative_to(data_root.parent))
        except ValueError:
            source_name = str(source)
        records.append({
            "source": source_name,
            "content_hash": item.get("hash", ""),
            "shot_type": item.get("shot_type", ""),
            "signature": [round(float(value), 6) for value in signature],
        })
    if not records:
        return None
    paths = _paths(data_root)
    try:
        paths["runs"].mkdir(parents=True, exist_ok=True)
        path = paths["runs"] / f"{run_id}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return path
    except OSError:
        return None


def _load_features(data_root: Path, run_id: str) -> dict[str, dict]:
    path = _paths(data_root)["runs"] / f"{run_id}.jsonl"
    result = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            result[row["source"]] = row
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return result


def _pair_id(run_id: str, left: dict, right: dict, label: int) -> str:
    sources = sorted((left["source"], right["source"]))
    raw = "|".join((run_id, str(label), *sources)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _pair_record(run_id: str, left: dict, right: dict, label: int, hard_negative: bool = False) -> dict | None:
    comparison = board_identity_comparison(left, right)
    if comparison.get("status") != "ok":
        return None
    return {
        "id": _pair_id(run_id, left, right, label),
        "kind": "pair",
        "run_id": run_id,
        "label": label,
        "hard_negative": hard_negative,
        "left": left["source"],
        "right": right["source"],
        "features": {name: round(float(comparison[name]), 6) for name in FEATURE_NAMES},
    }


def _read_examples(path: Path) -> list[dict]:
    examples = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "pair" and row.get("features"):
                examples.append(row)
    except OSError:
        pass
    return examples


def fit_weights(data_root: Path) -> dict:
    """Fit transparent weights from all saved human pair labels.

    This is a conservative metric learner: it gives weight to terms whose
    average distance is lower for human-positive pairs than human-negative
    pairs, scaled by observed spread.  It is deterministic and can be safely
    replaced by a larger embedding model later without changing the UX/data
    contract.
    """
    paths = _paths(data_root)
    examples = _read_examples(paths["examples"])
    positives = [row for row in examples if int(row.get("label", 0)) == 1]
    negatives = [row for row in examples if int(row.get("label", 0)) == 0]
    result = {
        "version": 1,
        "ready": len(positives) >= MIN_POSITIVE_PAIRS and len(negatives) >= MIN_NEGATIVE_PAIRS,
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "weights": {name: default for name, default in zip(FEATURE_NAMES, (0.30, 0.35, 0.25, 0.05, 0.05))},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if result["ready"]:
        scores = {}
        for name in FEATURE_NAMES:
            positive_mean = sum(float(row["features"][name]) for row in positives) / len(positives)
            negative_mean = sum(float(row["features"][name]) for row in negatives) / len(negatives)
            spread = max(positive_mean, negative_mean, 0.01)
            scores[name] = max(0.0, (negative_mean - positive_mean) / spread)
        total = sum(scores.values()) or 1.0
        result["weights"] = {name: round(scores[name] / total, 6) for name in FEATURE_NAMES}
    try:
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["weights"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return result


def load_weights(data_root: Path) -> dict | None:
    try:
        result = json.loads(_paths(data_root)["weights"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return result if result.get("ready") and isinstance(result.get("weights"), dict) else None


def record_human_save(data_root: Path, run_id: str, feedback_rows: list[dict], assignments: dict) -> dict:
    """Turn one human save into bounded positive/negative pair examples."""
    features = _load_features(data_root, run_id)
    items = []
    for row in feedback_rows:
        source = row.get("source_path", "")
        feature = features.get(source) or features.get(str(Path(source)))
        if not feature:
            continue
        final_group = assignments.get(str(row["photo_id"]), row.get("final_board_id"))
        initial_group = row.get("initial_board_id")
        items.append({
            **feature,
            "board_identity_signature": feature.get("signature", ()),
            "photo_id": row["photo_id"],
            "final_group": str(final_group),
            "initial_group": str(initial_group),
        })
    positive_candidates = []
    negative_candidates = []
    for left, right in combinations(items, 2):
        same = left["final_group"] == right["final_group"]
        hard = not same and left["initial_group"] == right["initial_group"]
        pair = _pair_record(run_id, left, right, 1 if same else 0, hard_negative=hard)
        if not pair:
            continue
        (positive_candidates if same else negative_candidates).append(pair)
    positive_candidates.sort(key=lambda row: row["id"])
    # Hard negatives are the most useful mistakes; keep them before ordinary negatives.
    negative_candidates.sort(key=lambda row: (not row["hard_negative"], row["id"]))
    selected = positive_candidates[:MAX_POSITIVE_PAIRS] + negative_candidates[:MAX_NEGATIVE_PAIRS]
    paths = _paths(data_root)
    existing = {row.get("id") for row in _read_examples(paths["examples"])}
    new_rows = [row for row in selected if row["id"] not in existing]
    try:
        paths["root"].mkdir(parents=True, exist_ok=True)
        with paths["examples"].open("a", encoding="utf-8") as stream:
            for row in new_rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        summary = {
            "run_id": run_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "photos_with_features": len(items),
            "positive_pairs": sum(row["label"] == 1 for row in new_rows),
            "negative_pairs": sum(row["label"] == 0 for row in new_rows),
            "hard_negatives": sum(row["hard_negative"] for row in new_rows),
        }
        with paths["batches"].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, separators=(",", ":")) + "\n")
    except OSError:
        summary = {"run_id": run_id, "photos_with_features": len(items), "error": "learning storage unavailable"}
    summary["model"] = fit_weights(data_root)
    return summary
