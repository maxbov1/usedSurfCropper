"""Small, inspectable visual features for shot-type classification."""

from __future__ import annotations

from PIL import Image

from .crop import classify_shot

LABELS = ["full_board", "side_profile", "fin_detail"]


def composition_features(image: Image.Image) -> tuple[list[float], dict[str, object]]:
    """Return geometry features and a human-readable composition hint.

    The feature vector deliberately uses framing evidence only. It does not
    claim to recognize a fin as an object; a close-up/detail hint is supporting
    evidence and stays review-required unless the learned model agrees.
    """
    width, height = image.size
    result = classify_shot(image)
    candidate = result.get("candidate")
    if candidate:
        x, y, w, h = [float(value) for value in candidate]
        box = [x / width, y / height, w / width, h / height]
        area = (w * h) / max(width * height, 1)
        aspect = w / max(h, 1.0)
    else:
        box = [0.0, 0.0, 0.0, 0.0]
        area = 0.0
        aspect = 0.0
    hint = str(result.get("shot_type", "detail"))
    hint_label = {
        "full_board": "full_board",
        "side_profile": "side_profile",
        "detail": "fin_detail",
    }.get(hint, "fin_detail")
    one_hot = [float(hint_label == label) for label in LABELS]
    features = [
        float(width / max(height, 1)),
        *box,
        float(area),
        float(aspect),
        float(result.get("confidence", 0.0)),
        float(bool(result.get("review", True))),
        *one_hot,
    ]
    metadata = {
        "hint": hint_label,
        "hint_confidence": float(result.get("confidence", 0.0)),
        "hint_review": bool(result.get("review", True)),
        "reason": result.get("reason", "unknown"),
        "candidate": candidate,
    }
    return features, metadata


def composition_matrix(cases: list[dict], root) -> tuple[object, list[dict]]:
    """Extract features in case order, returning a NumPy matrix and metadata."""
    import numpy as np

    rows, metadata = [], []
    for case in cases:
        with Image.open(root / case["local_path"]) as image:
            row, info = composition_features(image.convert("RGB"))
        rows.append(row)
        metadata.append(info)
    return np.asarray(rows, dtype=np.float32), metadata
