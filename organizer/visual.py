"""Small, local visual signatures for grouping support.

This is intentionally not an object-recognition model. The signature is useful
for evidence such as "these neighboring frames look unusually different", but
it must not deduplicate deck/bottom or light/low-light views.
"""

from __future__ import annotations

from pathlib import Path

from .crop import _find_board_candidate
from .ingest import read_image

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def signature(path: Path) -> dict | None:
    if cv2 is None or np is None:
        return None
    try:
        image, _ = read_image(path)
        image.thumbnail((320, 320))
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        small_gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        small_gray = (small_gray - small_gray.mean()) / max(float(small_gray.std()), 1.0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
        histogram /= max(float(np.linalg.norm(histogram)), 1.0)
        edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 50, 130)
        edge_small = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        candidate = _find_board_candidate(bgr, "full_board")
        height, width = bgr.shape[:2]
        silhouette = None
        if candidate:
            x, y, w, h = candidate
            silhouette = [round((x + w / 2) / width, 4), round(y / height, 4), round(w / width, 4), round(h / height, 4)]
        return {
            "gray": small_gray.flatten().round(4).tolist(),
            "histogram": histogram.round(4).tolist(),
            "edges": edge_small.flatten().round(4).tolist(),
            "silhouette": silhouette,
        }
    except Exception:
        return None


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator))) if denominator else 0.0


def similarity(left: dict | None, right: dict | None) -> float | None:
    if not left or not right or cv2 is None or np is None:
        return None
    structure = _cosine(left["gray"], right["gray"])
    edges = _cosine(left["edges"], right["edges"])
    color = _cosine(left["histogram"], right["histogram"])
    silhouette_bonus = 0.0
    if left.get("silhouette") and right.get("silhouette"):
        silhouette_bonus = max(0.0, 1.0 - sum(abs(a - b) for a, b in zip(left["silhouette"], right["silhouette"])) / 2.0)
    # Structure is deliberately dominant; color is weak because lighting and
    # deck/bottom views can change substantially.
    return round(0.45 * structure + 0.25 * edges + 0.20 * color + 0.10 * silhouette_bonus, 4)


def group_visual_check(items: list[dict]) -> dict:
    usable = [item for item in items if not item.get("is_card") and item.get("visual_signature")]
    scores = [similarity(left["visual_signature"], right["visual_signature"]) for left, right in zip(usable, usable[1:])]
    scores = [score for score in scores if score is not None]
    return {
        "non_card_photos": len(usable),
        "adjacent_similarity": round(sum(scores) / len(scores), 4) if scores else None,
        "supporting_only": True,
    }
