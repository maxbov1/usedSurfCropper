from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .card_ocr import extract_card_identifier, extract_card_identifier_with_confidence
from .board_evidence import evidence_fields_for_shot

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

SUPPORTED = {".jpg", ".jpeg", ".png", ".heic"}
PROCESSING_VERSION = "0.24.0-board-region-evidence"
MAX_PHOTOS_PER_BOARD = 7
STANDARD_BOARD_PHOTOS = 6
SHOT_LIMITS = {"full_board": 4, "side_profile": 1, "fin_detail": 1, "card": 1}
BOARD_SIMILARITY_BOUNDARY = 0.10
LOCAL_NEIGHBOR_COUNT = 8
LOCAL_TIME_WINDOW_SECONDS = 10 * 60
CARD_MATCH_MAX_MINUTES = 8
CARD_MATCH_MAX_VISUAL_DISTANCE = 0.35
BOUNDARY_REFINEMENT_RADIUS = 3
IDENTITY_ASSIGNMENT_MAX_DISTANCE = 0.42
IDENTITY_ASSIGNMENT_MIN_MARGIN = 0.035
IDENTITY_TRUSTED_DISTANCE = 0.27
IDENTITY_RETRY_ROUNDS = 3
IDENTITY_GROUP_MAX_DISTANCE = IDENTITY_ASSIGNMENT_MAX_DISTANCE
IDENTITY_LOCAL_MAX_DISTANCE = 0.22
QUARANTINE_CLUSTER_MAX_DISTANCE = IDENTITY_ASSIGNMENT_MAX_DISTANCE
QUARANTINE_CLUSTER_MIN_SIZE = 2
_LEARNED_BOARD_IDENTITY_WEIGHTS: dict[str, float] | None = None


def is_card_anchor(item: dict) -> bool:
    """Whether an item is a standalone card frame that anchors a group.

    ``contains_inventory_card`` is deliberately not enough: many board
    photos contain a small yellow panel, and treating every such photo as a
    new group inflated a 19-board batch into 77 anchors. The contained card
    remains available as identity evidence, but only a standalone card frame
    defines the expected group count.
    """
    return bool(item.get("is_card") or item.get("source_is_card"))


def set_board_identity_weights(weights: dict[str, float] | None) -> None:
    """Set the optional locally learned metric used by future grouping runs."""
    global _LEARNED_BOARD_IDENTITY_WEIGHTS
    if not weights:
        _LEARNED_BOARD_IDENTITY_WEIGHTS = None
        return
    _LEARNED_BOARD_IDENTITY_WEIGHTS = {
        name: max(0.0, float(weights.get(name, 0.0)))
        for name in ("color", "layout", "edges", "geometry", "shot")
    }


def visual_signature(image: Image.Image) -> tuple[float, ...]:
    """Return a small appearance signature for grouping support.

    This is deliberately not a board recognizer. In this controlled room the
    center of the frame carries enough board color/shape information to catch
    an obvious board change while remaining tolerant of light versus low-light
    paired views. It is compared within a small capture-time neighborhood,
    not against the entire batch.
    """
    if cv2 is None or np is None:
        return ()
    working = image.copy()
    working.thumbnail((480, 640))
    # PIL-backed arrays can be read-only.  Downstream image normalization and
    # masking are allowed to mutate their working buffers, so make ownership
    # explicit instead of relying on the decoder's buffer flags.
    array = np.asarray(working.convert("RGB")).copy()
    height, width = array.shape[:2]
    roi = array[round(height * 0.08):round(height * 0.92), round(width * 0.16):round(width * 0.84)]
    if roi.size == 0:
        return ()
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    # Normalize luminance so the light/low-light pair does not look like two
    # boards, but retain chroma because different board colors are meaningful.
    lab[:, :, 0] = (lab[:, :, 0] - float(lab[:, :, 0].mean())) / max(float(lab[:, :, 0].std()), 0.04)
    small = cv2.resize(lab, (12, 18), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = (gray - float(gray.mean())) / max(float(gray.std()), 0.08)
    small_gray = cv2.resize(gray, (12, 18), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny((np.clip(gray, -2.0, 2.0) * 40.0 + 128.0).astype(np.uint8), 40, 120)
    small_edges = cv2.resize(edges, (12, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    summary = np.concatenate([
        lab.mean(axis=(0, 1)),
        lab.std(axis=(0, 1)),
        small[:, :, 1:].reshape(-1),
        small_gray.reshape(-1),
        small_edges.reshape(-1),
    ])
    return tuple(float(value) for value in summary)


def board_identity_signature(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None = None,
    shot_type: str = "",
    card_regions: list[tuple[int, int, int, int]] | None = None,
) -> tuple[float, ...]:
    """Describe the board itself after normalizing away the room.

    The grouping fingerprint is intentionally made from the detected board
    crop, not the whole camera frame. It combines color distribution, a
    normalized grayscale layout, edge/graphic layout, and coarse geometry.
    Geometry is kept as a small final section so a side profile can still be
    compared to the same board's full-board views without dominating identity.
    """
    if cv2 is None or np is None:
        return ()
    working = image.convert("RGB")
    width, height = working.size
    geometry = (0.0, 0.0, 0.0, 0.0)
    if candidate and len(candidate) == 4:
        x, y, box_width, box_height = (int(value) for value in candidate)
        if box_width > 2 and box_height > 2:
            # Identity must describe the board, not the room around it. The
            # candidate is the board-bound rectangle; trim inward only to
            # avoid background leaking through rounded board corners. Never
            # expand outside the candidate.
            inner_x = round(box_width * 0.06)
            inner_y = round(box_height * 0.03)
            left, top = max(0, x + inner_x), max(0, y + inner_y)
            right = min(width, x + box_width - inner_x)
            bottom = min(height, y + box_height - inner_y)
            working = working.crop((left, top, right, bottom))
            geometry = (
                box_width / max(width, 1),
                box_height / max(height, 1),
                (x + box_width / 2) / max(width, 1),
                (y + box_height / 2) / max(height, 1),
            )
    else:
        # Keep the fallback compatible with scan-only callers. Production
        # grouping replaces this with the classifier's board candidate.
        left, right = round(width * 0.16), round(width * 0.84)
        top, bottom = round(height * 0.08), round(height * 0.92)
        working = working.crop((left, top, right, bottom))
    working.thumbnail((480, 640))
    # ``np.asarray(PIL.Image)`` may expose a read-only decoder buffer.  Card
    # masking below writes into this array, so always use a writable copy.
    array = np.asarray(working).copy()
    if array.size == 0:
        return ()
    # Inventory cards are identity anchors, not board identity. Mask their
    # pixels after the board crop so a yellow card cannot make two unrelated
    # boards look alike (or make the same board look different by lighting).
    if card_regions:
        for region in card_regions:
            rx, ry, rw, rh = (int(value) for value in region)
            if candidate and len(candidate) == 4:
                cx, cy, _, _ = (int(value) for value in candidate)
                rx, ry = rx - cx - round(int(candidate[2]) * 0.06), ry - cy - round(int(candidate[3]) * 0.03)
            scale_x = array.shape[1] / max(working.width, 1)
            scale_y = array.shape[0] / max(working.height, 1)
            left = max(0, round(rx * scale_x))
            top = max(0, round(ry * scale_y))
            right = min(array.shape[1], round((rx + rw) * scale_x))
            bottom = min(array.shape[0], round((ry + rh) * scale_y))
            if right <= left or bottom <= top:
                continue
            patch = array[max(0, top - 2):min(array.shape[0], bottom + 2), max(0, left - 2):min(array.shape[1], right + 2)]
            fill = np.median(patch.reshape(-1, 3), axis=0) if patch.size else np.median(array.reshape(-1, 3), axis=0)
            array[top:bottom, left:right] = fill.astype(np.uint8)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
    histogram /= max(float(np.linalg.norm(histogram)), 1.0)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = (gray - float(gray.mean())) / max(float(gray.std()), 0.08)
    gray_small = cv2.resize(gray, (20, 32), interpolation=cv2.INTER_AREA).reshape(-1)
    edges = cv2.Canny((np.clip(gray, -2.0, 2.0) * 40.0 + 128.0).astype(np.uint8), 40, 120)
    edge_small = (cv2.resize(edges, (20, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).reshape(-1)
    # A one-hot shot marker prevents profiles from being mistaken for full
    # boards solely because their room background is similar, while identity
    # comparison still relies primarily on the board crop features.
    shot_marker = (float(shot_type == "full_board"), float(shot_type == "side_profile"), float(shot_type == "fin_detail"))
    return tuple(float(value) for value in (*histogram, *gray_small, *edge_small, *geometry, *shot_marker))


def board_local_signature(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None = None,
    shot_type: str = "",
    card_regions: list[tuple[int, int, int, int]] | None = None,
) -> tuple[float, ...]:
    """Capture local board evidence that whole-board averages erase.

    The global signature is useful for silhouette and layout, but it can make
    unrelated white boards look alike. These fixed board-relative patches keep
    rail color, nose/deck markings, tail traction, fin-box layout, and local
    edge texture separate. Profiles intentionally do not get this signature.
    """
    if cv2 is None or np is None or shot_type == "side_profile" or not candidate:
        return ()
    x, y, width, height = (int(value) for value in candidate)
    if width <= 2 or height <= 2:
        return ()
    source = np.asarray(image.convert("RGB")).copy()
    board = source[max(0, y):min(source.shape[0], y + height), max(0, x):min(source.shape[1], x + width)]
    if board.size == 0:
        return ()
    board = cv2.resize(board, (48, 128), interpolation=cv2.INTER_AREA)
    if card_regions:
        for rx, ry, rw, rh in card_regions:
            left = max(0, round((int(rx) - x) * 48 / max(width, 1)))
            top = max(0, round((int(ry) - y) * 128 / max(height, 1)))
            right = min(48, round((int(rx) + int(rw) - x) * 48 / max(width, 1)))
            bottom = min(128, round((int(ry) + int(rh) - y) * 128 / max(height, 1)))
            if right > left and bottom > top:
                patch = board[max(0, top - 2):min(128, bottom + 2), max(0, left - 2):min(48, right + 2)]
                board[top:bottom, left:right] = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    regions = (
        (0, 0, 10, 128),       # left rail
        (38, 0, 48, 128),      # right rail
        (14, 0, 34, 32),       # nose/deck markings
        (14, 32, 34, 96),      # center/deck markings
        (8, 96, 40, 128),      # tail/traction/fin area
    )
    values: list[float] = []
    for left, top, right, bottom in regions:
        color = hsv[top:bottom, left:right]
        gray_patch = gray[top:bottom, left:right]
        if color.size == 0:
            values.extend([0.0] * 7)
            continue
        hue = color[:, :, 0]
        saturation = color[:, :, 1] / 255.0
        value = color[:, :, 2] / 255.0
        edges = cv2.Canny(gray_patch, 45, 130)
        values.extend([
            float(np.mean(np.cos(hue * np.pi / 90.0))),
            float(np.mean(np.sin(hue * np.pi / 90.0))),
            float(np.mean(saturation)),
            float(np.std(saturation)),
            float(np.mean(value)),
            float(np.std(value)),
            float(np.count_nonzero(edges) / max(edges.size, 1)),
        ])
    return tuple(values)


def _board_identity_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Compare board identity sections with explicit, inspectable weights."""
    components = _board_identity_components(left, right)
    return components.get("total", 0.0)


def _board_identity_components(left: tuple[float, ...], right: tuple[float, ...]) -> dict[str, float | str]:
    """Return the terms used by board identity distance for diagnostics."""
    histogram_end = 48
    layout_end = histogram_end + 640
    edge_end = layout_end + 640
    geometry_end = edge_end + 4
    if len(left) != len(right) or len(left) < geometry_end + 3:
        # An unavailable or incompatible representation is not evidence of
        # identity. Returning zero here made malformed signatures look like
        # perfect matches and allowed arbitrary same-angle assignments.
        return {"status": "incompatible_signature", "total": 1.0}
    first, second = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    color = float(np.mean(np.abs(first[:histogram_end] - second[:histogram_end])))
    layout = float(np.mean(np.abs(first[histogram_end:layout_end] - second[histogram_end:layout_end])))
    edges = float(np.mean(np.abs(first[layout_end:edge_end] - second[layout_end:edge_end])))
    geometry = float(np.mean(np.abs(first[edge_end:geometry_end] - second[edge_end:geometry_end])))
    shot = float(np.mean(np.abs(first[geometry_end:] - second[geometry_end:])))
    weights = _LEARNED_BOARD_IDENTITY_WEIGHTS or {
        "color": 0.30, "layout": 0.35, "edges": 0.25, "geometry": 0.05, "shot": 0.05,
    }
    total = sum(weights[name] * value for name, value in {
        "color": color, "layout": layout, "edges": edges, "geometry": geometry, "shot": shot,
    }.items())
    return {
        "status": "ok",
        "color": color,
        "layout": layout,
        "edges": edges,
        "geometry": geometry,
        "shot": shot,
        "total": total,
    }


def board_identity_comparison(left: dict, right: dict) -> dict[str, float | str]:
    """Explain one board-value comparison without exposing the full vector."""
    first, second = left.get("board_identity_signature", ()), right.get("board_identity_signature", ())
    if not first or not second:
        return {"status": "missing_signature", "total": 0.0}
    result = _board_identity_components(first, second)
    left_local = left.get("board_local_signature", ())
    right_local = right.get("board_local_signature", ())
    if left_local and right_local and len(left_local) == len(right_local) and result.get("status") == "ok":
        local = float(np.mean(np.abs(np.asarray(left_local, dtype=np.float32) - np.asarray(right_local, dtype=np.float32))))
        result["local"] = local
        result["total"] = float(result["total"]) * 0.68 + local * 0.32
    return result


def board_identity_diagnostics(items: list[dict]) -> dict[str, object]:
    """Produce a compact, human-readable capture and comparison report."""
    photos = [item for item in items if not item.get("is_card")]
    captured = []
    for item in photos:
        signature = item.get("board_identity_signature", ())
        values = np.asarray(signature, dtype=np.float32) if signature and np is not None else None
        sections = {}
        if values is not None and len(values) >= 1335:
            for name, start, end in (
                ("color", 0, 48),
                ("layout", 48, 688),
                ("edges", 688, 1328),
                ("geometry", 1328, 1332),
                ("shot", 1332, 1335),
            ):
                section = values[start:end]
                sections[name] = {
                    "count": int(len(section)),
                    "mean": round(float(section.mean()), 6),
                    "std": round(float(section.std()), 6),
                }
        captured.append({
            "source": str(item.get("path", "")),
            "shot_type": item.get("shot_type", ""),
            "contains_inventory_card": bool(item.get("contains_inventory_card")),
            "card_regions": item.get("card_regions") or [],
            "card_field_evidence": item.get("card_field_evidence") or {},
            "candidate": item.get("board_candidate"),
            "processing_error": item.get("processing_error", ""),
            "signature_status": "ready" if signature else "missing",
            "value_count": len(signature),
            "local_signature_status": "ready" if item.get("board_local_signature") else "missing",
            "local_value_count": len(item.get("board_local_signature", ())),
            "board_evidence": item.get("board_evidence") or {},
            "value_mean": round(float(values.mean()), 6) if values is not None else None,
            "value_std": round(float(values.std()), 6) if values is not None else None,
            "value_min": round(float(values.min()), 6) if values is not None else None,
            "value_max": round(float(values.max()), 6) if values is not None else None,
            "sections": sections,
        })
    comparisons = []
    for left, right in zip(photos, photos[1:]):
        result = board_identity_comparison(left, right)
        comparisons.append({
            "left": str(left.get("path", "")),
            "right": str(right.get("path", "")),
            **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in result.items()},
        })
    return {"photos": captured, "adjacent_comparisons": comparisons}


def board_group_diagnostics(groups: list[list[dict]]) -> list[dict[str, object]]:
    """Explain each proposed board group and its nearest competing evidence."""
    result = []
    for number, group in enumerate(groups, start=1):
        photos = [item for item in group if not item.get("is_card")]
        card = next((item for item in group if is_card_anchor(item)), None)
        bridge = next((item for item in photos if item.get("identity_assignment_reason") == "card_bridge_seed"), None)
        internal = []
        for index, left in enumerate(photos):
            for right in photos[index + 1:]:
                comparison = board_identity_comparison(left, right)
                if comparison.get("status") != "ok":
                    continue
                internal.append({
                    "left": str(left.get("path", "")),
                    "right": str(right.get("path", "")),
                    **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in comparison.items()},
                })
        internal.sort(key=lambda row: row.get("total", 0.0), reverse=True)
        weakest = internal[0] if internal else None
        photo_summaries = [{
            "source": str(item.get("path", "")),
            "shot_type": item.get("shot_type", ""),
            "signature_status": "ready" if item.get("board_identity_signature") else "missing",
            "value_count": len(item.get("board_identity_signature", ())),
            "processing_error": item.get("processing_error", ""),
            "assignment_reason": item.get("identity_assignment_reason", ""),
            "retry_round": item.get("identity_retry_round"),
            "retry_score": item.get("identity_retry_score"),
            "assignment_conflicts": item.get("identity_assignment_conflicts", []),
        } for item in group]
        result.append({
            "group_number": number,
            "photo_count": len(group),
            "board_photo_count": len(photos),
            "card_count": sum(1 for item in group if is_card_anchor(item)),
            "card_source": str(card.get("path", "")) if card else None,
            "card_bridge_source": str(bridge.get("path", "")) if bridge else None,
            "unassigned_photo": len(group) == 1 and bool(group[0].get("_unassigned_photo")),
            "unassigned_cluster": any(item.get("_unassigned_cluster") for item in group),
            "card_affinity": round(float(card_visual_affinity(card, photos)), 6) if card and card_visual_affinity(card, photos) is not None else None,
            "photos": photo_summaries,
            "internal_comparisons": internal,
            "weakest_internal_match": weakest,
            "internal_match_count": len(internal),
        })
    for current in result:
        strongest = None
        current_sources = {photo["source"] for photo in current["photos"] if photo["shot_type"] != "card"}
        for other in result:
            if other is current:
                continue
            for left in current["photos"]:
                if left["source"] not in current_sources:
                    continue
                left_item = next((item for item in groups[current["group_number"] - 1] if str(item.get("path", "")) == left["source"]), None)
                for right in other["photos"]:
                    right_item = next((item for item in groups[other["group_number"] - 1] if str(item.get("path", "")) == right["source"]), None)
                    if not left_item or not right_item:
                        continue
                    comparison = board_identity_comparison(left_item, right_item)
                    if comparison.get("status") == "ok":
                        candidate = {
                            "other_group": other["group_number"],
                            "left": left["source"],
                            "right": right["source"],
                            **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in comparison.items()},
                        }
                        if strongest is None or candidate.get("total", 1.0) < strongest.get("total", 1.0):
                            strongest = candidate
        current["strongest_competing_match"] = strongest
    return result


def visual_distance(left: dict, right: dict) -> float:
    left_identity = left.get("board_identity_signature", ())
    right_identity = right.get("board_identity_signature", ())
    if left_identity and right_identity:
        base = _board_identity_distance(left_identity, right_identity)
        left_local = left.get("board_local_signature", ())
        right_local = right.get("board_local_signature", ())
        if left_local and right_local and len(left_local) == len(right_local):
            local = float(np.mean(np.abs(np.asarray(left_local, dtype=np.float32) - np.asarray(right_local, dtype=np.float32))))
            return base * 0.68 + local * 0.32
        return base
    first, second = left.get("visual_signature", ()), right.get("visual_signature", ())
    if not first or not second or len(first) != len(second):
        return 0.0
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def card_visual_affinity(card: dict, photos: list[dict]) -> float | None:
    """Measure card similarity only after board-photo clusters are formed."""
    distances = sorted(
        visual_distance(card, photo)
        for photo in photos
        if photo.get("visual_signature")
    )
    if not distances:
        return None
    nearest = distances[: min(3, len(distances))]
    return sum(nearest) / len(nearest)


def _capture_seconds(item: dict) -> float | None:
    capture = item.get("capture")
    if not capture:
        return None
    try:
        return datetime.fromisoformat(str(capture)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _local_context(photos: list[dict], boundary: int, direction: int) -> list[dict]:
    """Return up to eight nearby photos on one side of a proposed boundary."""
    if direction < 0:
        candidates = list(reversed(photos[max(0, boundary - LOCAL_NEIGHBOR_COUNT):boundary]))
    else:
        candidates = photos[boundary:min(len(photos), boundary + LOCAL_NEIGHBOR_COUNT)]
    anchor = _capture_seconds(photos[boundary - 1] if direction < 0 else photos[boundary])
    if anchor is None:
        return candidates
    return [
        item for item in candidates
        if (item_capture := _capture_seconds(item)) is None
        or abs(item_capture - anchor) <= LOCAL_TIME_WINDOW_SECONDS
    ]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _context_distance(left: list[dict], right: list[dict]) -> float:
    return _median([visual_distance(before, after) for before in left for after in right])


def _context_cohesion(context: list[dict]) -> float:
    return _median([
        visual_distance(context[left], context[right])
        for left in range(len(context))
        for right in range(left + 1, len(context))
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exif_capture_time(image: Image.Image) -> str | None:
    exif = image.getexif()
    raw = exif.get(36867) or exif.get(36868) or exif.get(306)
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
    except ValueError:
        return None


def read_image(path: Path) -> tuple[Image.Image, str | None]:
    with Image.open(path) as opened:
        capture = exif_capture_time(opened)
        return ImageOps.exif_transpose(opened).convert("RGB"), capture


def ocr_runtime_status() -> dict[str, str | bool]:
    """Report whether both pytesseract and the native Tesseract binary work."""
    if pytesseract is None:
        return {"available": False, "status": "python_package_missing", "error": "pytesseract is not installed"}
    _configure_tesseract_binary()
    try:
        version = str(pytesseract.get_tesseract_version()).splitlines()[0]
        return {"available": True, "status": "ready", "version": version}
    except Exception as exc:
        return {"available": False, "status": "binary_missing_or_unusable", "error": f"{type(exc).__name__}: {exc}"}


def local_ocr(image: Image.Image, region: tuple[int, int, int, int] | None = None) -> tuple[str, dict[str, str], str]:
    if pytesseract is None:
        return "", {}, "pytesseract is not installed"
    _configure_tesseract_binary()
    try:
        variants = []
        if region:
            x, y, width, height = region
            pad_x, pad_y = round(width * 0.08), round(height * 0.08)
            left, top = max(0, x - pad_x), max(0, y - pad_y)
            right, bottom = min(image.width, x + width + pad_x), min(image.height, y + height + pad_y)
            raw_card = image.crop((left, top, right, bottom))
            cards = [_deskew_yellow_card(raw_card), _rectify_card(image, (left, top, right - left, bottom - top))]
            variants = []
            for card in cards:
                card.thumbnail((1800, 1800))
                variants.extend([card, card.crop((round(card.width * 0.47), 0, card.width, round(card.height * 0.62)))])
        else:
            image = image.copy()
            image.thumbnail((1200, 1200))
            width, height = image.size
            variants = [
                image,
                image.crop((0, round(height * 0.18), width, round(height * 0.86))),
                image.crop((round(width * 0.10), round(height * 0.22), round(width * 0.90), round(height * 0.78))),
            ]
        texts = []
        for variant in variants:
            texts.append(pytesseract.image_to_string(variant, config="--psm 6"))
            texts.append(pytesseract.image_to_string(ImageOps.autocontrast(ImageOps.grayscale(variant)), config="--psm 11"))
        text = max(texts, key=_ocr_signal_score, default="")
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"
    return text, extract_card_identifier(texts), ""


def board_ocr(image: Image.Image, candidate: tuple[int, int, int, int] | None = None) -> tuple[str, dict[str, str], str]:
    """Read board markings from board-relative crops, not the room.

    Full-frame OCR mostly sees wall seams and produces no useful brand/model
    evidence. The board candidate is enlarged first, then OCR sees the whole
    board, the nose/deck half, and the tail/traction half. This is intentionally
    separate from card OCR because the card rectifier is wrong for logos.
    """
    if pytesseract is None:
        return "", {}, "pytesseract is not installed"
    _configure_tesseract_binary()
    try:
        if candidate and len(candidate) == 4:
            x, y, width, height = (int(value) for value in candidate)
            crop = image.crop((max(0, x), max(0, y), min(image.width, x + width), min(image.height, y + height)))
        else:
            crop = image.copy()
        crop.thumbnail((1800, 2400))
        width, height = crop.size
        variants = [
            crop,
            crop.crop((0, 0, width, round(height * 0.58))),
            crop.crop((0, round(height * 0.42), width, height)),
            ImageEnhance.Contrast(crop).enhance(1.7).filter(ImageFilter.SHARPEN),
        ]
        texts = [pytesseract.image_to_string(variant, config="--psm 11") for variant in variants]
        text = max(texts, key=_ocr_signal_score, default="")
        return text, extract_card_identifier(texts), ""
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"


def inventory_card_regions(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Find yellow inventory-card candidates anywhere in the frame.

    This is deliberately broader than ``card_candidate``. A card may be
    taped low on a board, partially covered by a hand, rotated, or adjacent
    to a second UsedSurf panel. False positives are acceptable here because
    OCR and field structure validate each region later.
    """
    if cv2 is None or np is None:
        return []
    working = image.copy()
    working.thumbnail((1200, 1200))
    array = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(array, np.array([8, 75, 55], np.uint8), np.array([48, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_x, scale_y = image.width / max(working.width, 1), image.height / max(working.height, 1)
    regions: list[tuple[int, int, int, int]] = []
    frame_area = float(mask.shape[0] * mask.shape[1])
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height / max(frame_area, 1.0)
        ratio = width / max(height, 1)
        fill = cv2.contourArea(contour) / max(width * height, 1)
        if not 0.004 <= area <= 0.45 or not 0.18 <= ratio <= 4.0 or fill < 0.18:
            continue
        pad_x, pad_y = round(width * 0.04), round(height * 0.04)
        regions.append((
            max(0, round((x - pad_x) * scale_x)),
            max(0, round((y - pad_y) * scale_y)),
            min(image.width, round((width + 2 * pad_x) * scale_x)),
            min(image.height, round((height + 2 * pad_y) * scale_y)),
        ))
    # Color is deliberately not sufficient. Warm boards and yellow graphics
    # can dominate the mask, while a real card may be shadowed or partly
    # covered. Look for large four-sided, text-dense regions independently.
    gray = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 35, 120), np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edge_regions: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = cv2.contourArea(contour) / max(frame_area, 1.0)
        if not 0.008 <= area <= 0.42:
            continue
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(quad) != 4:
            continue
        x, y, width, height = cv2.boundingRect(quad)
        ratio = width / max(height, 1)
        if not 0.25 <= ratio <= 3.8:
            continue
        roi_hsv = array[max(0, y):min(array.shape[0], y + height), max(0, x):min(array.shape[1], x + width)]
        roi_edges = edges[max(0, y):min(edges.shape[0], y + height), max(0, x):min(edges.shape[1], x + width)]
        yellow_fraction = float(np.count_nonzero(cv2.inRange(roi_hsv, np.array([5, 35, 45], np.uint8), np.array([55, 255, 255], np.uint8)))) / max(roi_hsv.shape[0] * roi_hsv.shape[1], 1)
        text_density = float(np.count_nonzero(roi_edges)) / max(roi_edges.size, 1)
        if yellow_fraction < 0.08 and text_density < 0.10:
            continue
        score = yellow_fraction + min(text_density, 0.35) * 0.8 + area * 0.25
        edge_regions.append((score, (
            max(0, round(x * scale_x)), max(0, round(y * scale_y)),
            min(image.width, round(width * scale_x)), min(image.height, round(height * scale_y)),
        )))
    for _, region in sorted(edge_regions, reverse=True)[:8]:
        regions.append(region)
    # Keep the largest non-overlapping proposals. This handles the two yellow
    # panels in a single photo without flooding OCR with near-duplicates.
    regions.sort(key=lambda region: region[2] * region[3], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for region in regions:
        x, y, width, height = region
        overlaps = False
        for ox, oy, ow, oh in selected:
            intersection = max(0, min(x + width, ox + ow) - max(x, ox)) * max(0, min(y + height, oy + oh) - max(y, oy))
            if intersection / max(min(width * height, ow * oh), 1) > 0.65:
                overlaps = True
                break
        if not overlaps:
            selected.append(region)
        if len(selected) >= 3:
            break
    return selected


def _configure_tesseract_binary() -> None:
    """Make Finder-launched apps find Homebrew's Tesseract without a shell PATH."""
    if pytesseract is None:
        return
    configured = Path(str(getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")))
    if configured.is_file():
        return
    candidates = [shutil.which("tesseract"), "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    binary = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if binary:
        pytesseract.pytesseract.tesseract_cmd = binary


def _ocr_signal_score(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(word) for word in ("item", "brand", "model", "board")) * 10 + sum(character.isdigit() for character in text)


def _deskew_yellow_card(image: Image.Image) -> Image.Image:
    if cv2 is None or np is None:
        return image
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([10, 120, 70], np.uint8), np.array([45, 255, 255], np.uint8))
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 20:
        return image
    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1.5:
        return image
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(rgb, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).flatten()
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
    return ordered


def _rectify_card(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    """Perspective-warp the strongest card-like quadrilateral in a rough region."""
    if cv2 is None or np is None:
        return _deskew_yellow_card(image.crop((region[0], region[1], region[0] + region[2], region[1] + region[3])))
    x, y, width, height = region
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    region_area = max(width * height, 1)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < region_area * 0.03 or area > region_area * 0.95:
            continue
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(quad) != 4:
            continue
        qx, qy, qw, qh = cv2.boundingRect(quad)
        if qx + qw < x or qy + qh < y or qx > x + width or qy > y + height:
            continue
        ratio = qw / max(qh, 1)
        if not 0.35 <= ratio <= 3.0:
            continue
        overlap = max(0, min(qx + qw, x + width) - max(qx, x)) * max(0, min(qy + qh, y + height) - max(qy, y))
        candidates.append((area * (overlap / region_area), quad))
    if not candidates:
        return _deskew_yellow_card(image.crop((x, y, x + width, y + height)))
    _, quad = max(candidates, key=lambda value: value[0])
    source = _order_quad(quad)
    top_width = np.linalg.norm(source[1] - source[0])
    bottom_width = np.linalg.norm(source[2] - source[3])
    left_height = np.linalg.norm(source[3] - source[0])
    right_height = np.linalg.norm(source[2] - source[1])
    target_width = max(800, round(max(top_width, bottom_width)))
    target_height = max(600, round(max(left_height, right_height)))
    target = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(rgb, matrix, (target_width, target_height), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def card_candidate(image: Image.Image) -> tuple[int, int, int, int, float] | None:
    """Find likely yellow inventory-card regions on a small working image.

    This is a card detector, not a board-color rule. It is intentionally
    allowed to be uncertain: a candidate can start a reviewable group even
    when OCR cannot read its identifiers.
    """
    if cv2 is None or np is None:
        return None
    original_width, original_height = image.size
    working = image.copy()
    working.thumbnail((900, 900))
    array = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
    # UsedSurf's yellow card is high-saturation and warm. The broad hue range
    # tolerates the warm/cool lighting seen across the batch.
    mask = cv2.inRange(hsv, np.array([10, 145, 75], np.uint8), np.array([45, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    working_height, working_width = mask.shape
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height / float(working_width * working_height)
        ratio = width / max(height, 1)
        if not 0.025 <= area <= 0.36 or not 0.35 <= ratio <= 2.4:
            continue
        fill = cv2.contourArea(contour) / max(width * height, 1)
        if fill < 0.35:
            continue
        # Cards tend to contain many dark text edges; this helps reject broad
        # warm lighting on a board while keeping the detector lightweight.
        roi = cv2.cvtColor(array[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
        edge_density = float(np.count_nonzero(cv2.Canny(roi, 50, 130))) / max(roi.size, 1)
        score = area * 2.0 + min(edge_density, 0.25) * 2.0 + fill
        candidates.append((score, x, y, width, height))
    if not candidates:
        points = cv2.findNonZero(mask)
        if points is not None and len(points) >= 40:
            x, y, width, height = cv2.boundingRect(points)
            area = width * height / float(working_width * working_height)
            if 0.01 <= area <= 0.70:
                scale_x = original_width / working_width
                scale_y = original_height / working_height
                return round(x * scale_x), round(y * scale_y), round(width * scale_x), round(height * scale_y), 0.25
        # Yellow can bleed into a warm board or wall. Strong four-sided card
        # edges are a useful fallback when the color contour is unusable.
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 45, 135), np.ones((3, 3), np.uint8), iterations=1)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_candidates = []
        for contour in edge_contours:
            area = cv2.contourArea(contour) / float(working_width * working_height)
            if not 0.015 <= area <= 0.45:
                continue
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(quad) != 4:
                continue
            ex, ey, ew, eh = cv2.boundingRect(quad)
            ratio = ew / max(eh, 1)
            if not 0.35 <= ratio <= 3.0:
                continue
            edge_candidates.append((area, ex, ey, ew, eh))
        if edge_candidates:
            _, x, y, width, height = max(edge_candidates)
            scale_x = original_width / working_width
            scale_y = original_height / working_height
            return round(x * scale_x), round(y * scale_y), round(width * scale_x), round(height * scale_y), 0.20
        return None
    _, x, y, width, height = max(candidates)
    scale_x = original_width / working_width
    scale_y = original_height / working_height
    return round(x * scale_x), round(y * scale_y), round(width * scale_x), round(height * scale_y), float(max(candidates)[0])


def yellow_card_score(image: Image.Image) -> float:
    """Return the proportion of strongly yellow pixels in a thumbnail."""
    if cv2 is None or np is None:
        return 0.0
    working = image.copy()
    working.thumbnail((300, 400))
    array = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(array, np.array([10, 145, 75], np.uint8), np.array([45, 255, 255], np.uint8))
    return float(np.count_nonzero(mask)) / max(mask.size, 1)


def card_signal(text: str, identifier: dict[str, str]) -> bool:
    words = text.lower()
    if identifier.get("sku"):
        return True
    fields = ("item", "brand", "model")
    # One OCR word is not enough: board graphics and wall texture frequently
    # produce isolated false words. Multiple inventory-field terms are the
    # automatic boundary threshold; possible visual cards remain reviewable.
    return sum(field in words for field in fields) >= 2


def scan_files(input_dir: Path, cache: dict[str, dict] | None = None) -> list[dict]:
    found = []
    ocr_status = ocr_runtime_status()
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            content_hash = sha256(path)
            cached = (cache or {}).get(content_hash)
            if cached and cached.get("cache_version") == PROCESSING_VERSION and cached.get("complete"):
                restored = dict(cached)
                restored.pop("cache_version", None)
                restored.pop("complete", None)
                restored["path"] = path
                restored["hash"] = content_hash
                restored["_processing_cache_hit"] = True
                for key in ("visual_signature", "board_identity_signature", "board_local_signature"):
                    if isinstance(restored.get(key), list):
                        restored[key] = tuple(restored[key])
                for key in ("card_candidate", "board_candidate"):
                    if isinstance(restored.get(key), list):
                        restored[key] = tuple(restored[key])
                found.append(restored)
                continue
            image, capture = read_image(path)
            candidate = card_candidate(image)
            # Broad edge proposals can accidentally describe the entire
            # board/frame. They are useful diagnostics, but not card regions:
            # an inventory card is a compact panel inside the image.
            raw_card_regions = inventory_card_regions(image)
            card_regions = [
                region for region in raw_card_regions
                if (
                    0.004 <= (region[2] * region[3]) / max(image.width * image.height, 1) <= 0.45
                    and region[0] > image.width * 0.02
                    and region[1] > image.height * 0.02
                    and region[0] + region[2] < image.width * 0.98
                    and region[1] + region[3] < image.height * 0.98
                )
            ]
            yellow_score = yellow_card_score(image)
            candidate_is_card = False
            if candidate:
                cx, cy, cw, ch, candidate_score = candidate
                # A real card is a substantial interior rectangle. Long warm
                # floor/rail edges and yellow board graphics usually touch an
                # image edge or are too short to satisfy this test.
                candidate_is_card = (
                    candidate_score >= 1.05
                    and 0.08 * image.width < cx < 0.78 * image.width
                    and 0.18 * image.height < cy < 0.70 * image.height
                    and 0.12 * image.width < cw < 0.70 * image.width
                    and 0.16 * image.height < ch < 0.62 * image.height
                    and cy + ch < 0.92 * image.height
                )
            # Avoid running Tesseract on every warm wall edge or board logo.
            # OCR is reserved for a plausible interior card or a strong yellow
            # region; the lightweight geometry still records the card candidate
            # even when OCR cannot read it.
            region = candidate[:4] if candidate else None
            if region is None and yellow_score >= 0.12:
                # A yellow card can merge into a yellow board in the color
                # mask. Give OCR a centered card-sized search window instead
                # of sending the entire board to Tesseract.
                region = (
                    round(image.width * 0.08),
                    round(image.height * 0.18),
                    round(image.width * 0.84),
                    round(image.height * 0.68),
                )
            likely_card = candidate_is_card or bool(card_regions) or (
                yellow_score >= 0.12 and candidate is not None and candidate_score >= 1.05
            )
            if likely_card:
                text, identifier, ocr_error = local_ocr(image, region)
                card_texts = [text] if text else []
                for card_region in card_regions:
                    region_text, region_identifier, region_error = local_ocr(image, card_region)
                    if region_text:
                        card_texts.append(region_text)
                    if not identifier and region_identifier:
                        identifier = region_identifier
                    else:
                        identifier.update({key: value for key, value in region_identifier.items() if key not in identifier})
                    if region_error and not ocr_error:
                        ocr_error = region_error
                field_evidence = extract_card_identifier_with_confidence(card_texts)
                per_file_ocr_status = "read" if identifier else ("empty" if not ocr_error else "error")
            else:
                text, identifier, ocr_error, field_evidence = "", {}, "", {}
                per_file_ocr_status = "not_run"
            largest_card_fraction = max((region[2] * region[3] for region in card_regions), default=0) / max(image.width * image.height, 1)
            has_card_evidence = bool(card_regions and (
                identifier
                or any(word in text.lower() for word in ("board information", "fin system", "brand", "model", "item"))
                or largest_card_fraction >= 0.015
            )) or candidate_is_card
            # OCR from a full board frequently finds words such as "model"
            # or "brand" in logos and graphics. It is only standalone-card
            # evidence when paired with a compact card proposal.
            compact_card_ocr = bool(card_regions and largest_card_fraction <= 0.45 and card_signal(text, identifier))
            standalone_card = candidate_is_card or compact_card_ocr
            # A readable card mounted on a board is an anchor, not a separate
            # card frame. Only promote a frame to ``is_card`` when the card
            # occupies most of the image or the strict detector says it is a
            # standalone card.
            if has_card_evidence and largest_card_fraction < 0.40 and not candidate_is_card and not compact_card_ocr:
                standalone_card = False
            found.append({
                "path": path, "hash": content_hash, "capture": capture, "width": image.width,
                "height": image.height, "ocr": text, "identifier": identifier,
                "card_candidate": candidate[:4] if candidate else None,
                "card_confidence": candidate[4] if candidate else 0.0,
                "card_regions": card_regions,
                "contains_inventory_card": has_card_evidence,
                "card_field_evidence": field_evidence,
                "yellow_card_score": yellow_score,
                "ocr_status": per_file_ocr_status,
                "ocr_error": ocr_error,
                "ocr_runtime": ocr_status,
                "visual_signature": visual_signature(image),
                "is_card": standalone_card,
            })
        except Exception as exc:
            found.append({"path": path, "error": str(exc), "capture": None, "width": None, "height": None, "ocr": "", "identifier": {}, "ocr_status": "scan_error", "ocr_error": "", "ocr_runtime": ocr_status, "is_card": False, "visual_signature": ()})
    # Stable tie-breaker is filename, but capture time remains primary.
    return sorted(found, key=lambda item: (item["capture"] is None, item["capture"] or "", item["path"].name.lower()))


def group_scanned(files: list[dict]) -> list[list[dict]]:
    """Group photos by visual runs, using cards as identity anchors.

    The card count is a validation signal, not a reason to manufacture a
    merge or split. A weak card/cluster match remains a visible review item.
    """
    if not files:
        return []
    for index, item in enumerate(files):
        item.setdefault("_sequence_index", index)
    cards = [item for item in files if is_card_anchor(item)]
    board_photos = [item for item in files if not item.get("is_card")]
    if not cards:
        return _cluster_board_photos(board_photos)
    # Once the classifier has produced board-identity values, do not let the
    # old chronological lane splitter own a photo. Cards still determine the
    # expected number of groups, but identity evidence makes the assignment
    # globally. The visual-signature fallback keeps older scan-only callers
    # and small fixtures compatible.
    if any(item.get("board_identity_signature") for item in board_photos):
        return _group_from_global_identity(files, cards, board_photos)
    return _group_from_card_lanes(files, cards, board_photos)


def _seed_card_lanes(cards: list[dict], photos: list[dict]) -> list[list[dict]]:
    """Create soft initial seeds; these are never treated as ownership."""
    ordered_cards = sorted(cards, key=lambda item: item.get("_sequence_index", 0))
    ordered_photos = sorted(photos, key=lambda item: item.get("_sequence_index", 0))
    card_keys = [_card_order_key(card) for card in ordered_cards]
    photo_keys = [_card_order_key(photo) for photo in ordered_photos]
    boundaries = []
    for left, right in zip(card_keys, card_keys[1:]):
        midpoint = (left + right) / 2.0
        # A boundary is a count of photos on the left, not the index of the
        # nearest photo. Using the nearest index split immediately before a
        # photo that was just left of the midpoint, which moved the final
        # side/profile frame into the following board.
        midpoint_boundary = sum(1 for key in photo_keys if key <= midpoint)
        boundaries.append(_refine_lane_boundary(ordered_photos, midpoint_boundary))
    lanes = []
    start = 0
    for boundary in sorted(max(0, min(len(ordered_photos), value)) for value in boundaries):
        boundary = max(start, boundary)
        lanes.append(ordered_photos[start:boundary])
        start = boundary
    lanes.append(ordered_photos[start:])
    return lanes[:len(ordered_cards)] + [[] for _ in range(max(0, len(ordered_cards) - len(lanes)))]


def _card_bridge_seeds(cards: list[dict], lanes: list[list[dict]]) -> list[dict | None]:
    """Choose one nearby full-board bridge for each immutable card slot.

    An inventory card is not a full-board image. It is an identity anchor with
    a partial deck view. The first full-board frame in that card's local lane
    is therefore the safest visual bridge; later grouping must grow from that
    bridge instead of asking a generic board cluster to claim the card after
    the fact. A visual tie-breaker is useful, but capture order remains the
    primary protocol signal because the card is normally photographed before
    its board views.
    """
    ordered_cards = sorted(cards, key=lambda item: item.get("_sequence_index", 0))
    bridges: list[dict | None] = []
    for card, lane in zip(ordered_cards, lanes):
        candidates = [
            photo for photo in lane
            if not photo.get("is_card")
            and photo.get("shot_type") == "full_board"
            and photo.get("board_identity_signature")
            and not _photo_identity_conflicts(photo, card, ordered_cards)
        ]
        if not candidates:
            bridges.append(None)
            continue
        card_index = card.get("_sequence_index")
        def bridge_key(photo: dict) -> tuple[float, float, int]:
            photo_index = photo.get("_sequence_index")
            sequence_distance = abs((photo_index or 0) - (card_index or 0))
            before_penalty = 0.0 if photo_index is None or card_index is None or photo_index >= card_index else 0.5
            affinity = card_visual_affinity(card, [photo])
            return before_penalty, sequence_distance + (affinity or 0.0) * 0.25, photo_index or 0
        bridge = min(candidates, key=bridge_key)
        bridge["identity_assignment_reason"] = "card_bridge_seed"
        bridge["card_bridge_source"] = str(card.get("path", ""))
        bridges.append(bridge)
    return bridges


def _identity_representatives(cluster: list[dict], limit: int = 4) -> list[dict]:
    if len(cluster) <= limit:
        return cluster
    ranked = sorted(
        cluster,
        key=lambda item: _median([visual_distance(item, other) for other in cluster]),
    )
    return ranked[:limit]


def _identity_text_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _identity_value_matches(expected: object, actual: object) -> bool:
    expected_key = _identity_text_key(expected)
    actual_key = _identity_text_key(actual)
    return bool(expected_key and actual_key and (expected_key in actual_key or actual_key in expected_key))


def _side_profile_neighbor_lanes(photo: dict, cards: list[dict], count: int) -> list[int]:
    """Return the card-backed lanes immediately before and after a profile."""
    if not cards or count <= 0:
        return []
    photo_key = _card_order_key(photo)
    ordered = [(_card_order_key(card), index) for index, card in enumerate(cards[:count])]
    before = [index for key, index in ordered if key <= photo_key]
    after = [index for key, index in ordered if key > photo_key]
    candidates = []
    if before:
        candidates.append(before[-1])
    if after:
        candidates.append(after[0])
    return candidates


def _photo_identity_conflicts(photo: dict, card: dict, all_cards: list[dict]) -> list[str]:
    """Return only explicit contradictions; missing fields are never negative."""
    expected = card.get("identifier") or {}
    evidence = photo.get("identity_evidence") or {}
    conflicts: list[str] = []
    eligible = evidence_fields_for_shot(str(photo.get("shot_type", "")))
    for field in ("brand", "model", "fin_system", "fin_box_count", "fins_included", "fins_present_count"):
        if field not in eligible:
            continue
        wanted = expected.get(field)
        observed = (evidence.get(field) or {}).get("value")
        if wanted and observed and not _identity_value_matches(wanted, observed):
            conflicts.append(field)

    # Some board photos expose a logo/model to OCR without the `BRAND:` or
    # `MODEL:` label. Treat another card's explicit value as a contradiction,
    # but only when the target card's value is not also present in the text.
    text = _identity_text_key(photo.get("photo_ocr_text", ""))
    if text:
        target_values = [expected.get(field, "") for field in ("brand", "model") if field in eligible]
        target_seen = any(_identity_text_key(value) in text for value in target_values if value)
        if not target_seen:
            for other in all_cards:
                if other is card:
                    continue
                other_identifier = other.get("identifier") or {}
                for field in ("brand", "model"):
                    if field not in eligible:
                        continue
                    value = other_identifier.get(field, "")
                    key = _identity_text_key(value)
                    if key and len(key) >= 4 and key in text:
                        conflicts.append(f"{field}:other-card")
                        break
    return sorted(set(conflicts))


def _identity_candidate_key(item: dict) -> tuple[int, float, int]:
    """Prefer reliable board views, then stable capture order."""
    shot_type = str(item.get("shot_type") or "")
    # Full-board views expose the most identity information. Other views are
    # still useful retries, but must not define the first prototype.
    shot_priority = {"full_board": 0, "fin_detail": 1, "side_profile": 2}.get(shot_type, 3)
    return shot_priority, _card_order_key(item), int(item.get("_sequence_index", 0))


def _identity_medoid(items: list[dict]) -> dict | None:
    """Return the most central item without averaging unlike board views."""
    if not items:
        return None
    return min(
        items,
        key=lambda candidate: _median([
            visual_distance(candidate, other)
            for other in items
            if other is not candidate
        ]),
    )


def _trusted_identity_members(lane: list[dict]) -> list[dict]:
    """Select a small clean seed set from a soft timestamp lane.

    The lane is only a search window. It is never itself a group. A prototype
    is made from the central, mutually coherent full-board views, so an
    adjacent board cannot poison the group's identity during the first pass.
    """
    candidates = [
        item for item in lane
        if not item.get("is_card") and item.get("board_identity_signature")
    ]
    if not candidates:
        return []
    full_boards = [item for item in candidates if item.get("shot_type") == "full_board"]
    pool = full_boards or candidates
    medoid = _identity_medoid(pool)
    if medoid is None:
        return []
    ranked = sorted(
        ((visual_distance(item, medoid), item) for item in pool),
        key=lambda pair: (pair[0], _identity_candidate_key(pair[1])),
    )
    trusted = []
    seen_signatures: set[tuple[object, ...]] = set()
    for distance, item in ranked:
        signature_key = (
            str(item.get("shot_type") or ""),
            tuple(item.get("board_identity_signature") or ()),
        )
        # Identical same-angle values do not provide independent evidence. A
        # single seed prevents an ambiguous duplicate at a card boundary from
        # becoming falsely trusted simply because it is visually identical.
        if signature_key in seen_signatures:
            continue
        seen_signatures.add(signature_key)
        if distance <= IDENTITY_TRUSTED_DISTANCE:
            trusted.append(item)
        if len(trusted) >= 4:
            break
    # A single unusually framed board may not have two close views. One
    # central full-board view is still a valid seed; it is marked weak and is
    # never enough to override an explicit identity contradiction.
    if not trusted:
        trusted = [ranked[0][1]]
    return trusted


def _identity_representative_score(photo: dict, representatives: list[dict]) -> float:
    """Score against trusted views, preferring the closest compatible view."""
    if not representatives:
        return 1.0
    distances = sorted(visual_distance(photo, representative) for representative in representatives)
    # The closest view handles a painted back/rail/fin angle. The median keeps
    # a single accidental near-match from dominating when more views exist.
    closest = distances[0]
    median = _median(distances[: min(3, len(distances))])
    return closest * 0.65 + median * 0.35


def _identity_admission(
    photo: dict,
    cluster: list[dict],
    representatives: list[dict],
) -> tuple[bool, float, str]:
    """Apply the hard board-identity admission rule.

    A candidate is never accepted merely because it is the least-bad option.
    It must be compatible with the frozen representatives of the group. The
    candidate is also checked against the group's central member so a single
    accidental near-match cannot poison the group.
    """
    if not representatives:
        return False, 1.0, "no_trusted_representative"
    distances = sorted(visual_distance(photo, representative) for representative in representatives)
    score = _identity_representative_score(photo, representatives)
    medoid = _identity_medoid(representatives)
    medoid_distance = visual_distance(photo, medoid) if medoid else 1.0
    # Side profiles are an intentional protocol exception: their narrow
    # silhouette is not a reliable appearance signature. They are admitted
    # only from the neighboring chronological card lanes, whose temporal
    # routing is enforced by _retry_group_candidates(). Full-board and fin
    # photos still require visual identity compatibility.
    if photo.get("shot_type") == "side_profile":
        return True, 0.0, "side_profile_temporal_neighbor"
    if photo.get("shot_type") == "full_board":
        admitted = (
            distances[0] <= IDENTITY_GROUP_MAX_DISTANCE
            and medoid_distance <= IDENTITY_GROUP_MAX_DISTANCE
        )
        local_signature = photo.get("board_local_signature") or ()
        if local_signature:
            local_distances = [
                float(np.mean(np.abs(
                    np.asarray(local_signature, dtype=np.float32)
                    - np.asarray(item.get("board_local_signature"), dtype=np.float32)
                )))
                for item in representatives
                if item.get("board_local_signature") and len(item["board_local_signature"]) == len(local_signature)
            ]
            if local_distances:
                admitted = admitted and min(local_distances) <= IDENTITY_LOCAL_MAX_DISTANCE
                if not admitted:
                    return False, score, "local_board_identity_contradiction"
    else:
        admitted = distances[0] <= IDENTITY_GROUP_MAX_DISTANCE
    if not admitted:
        return False, score, "board_identity_contradiction"
    return True, score, "board_identity_compatible"


def _identity_retry_score(
    photo: dict,
    group_index: int,
    representatives: list[dict],
    clusters: list[list[dict]],
    cards: list[dict],
) -> tuple[float, str]:
    """Return an evidence score and an explainable retry strategy."""
    shot_type = str(photo.get("shot_type") or "")
    visual = _identity_representative_score(photo, representatives)
    photo_time = _capture_seconds(photo)
    group_time = _cluster_time_center(clusters[group_index])
    temporal = 0.0
    if photo_time is not None and group_time is not None:
        temporal = min(abs(photo_time - group_time) / 480.0, 1.0)

    if shot_type == "side_profile":
        # Side profiles are deliberately weak identity evidence. Their retry
        # route is the controlled capture protocol: choose among adjacent
        # card lanes by time, then use visual evidence only to reject a clear
        # contradiction.
        # Do not let the narrow side crop overturn the capture protocol. The
        # visual term is retained only for diagnostics; assignment is based on
        # the neighboring group's mean time.
        return temporal * 0.20, "side_profile_temporal_neighbor"
    if shot_type == "fin_detail":
        return visual * 0.90 + temporal * 0.10, "fin_detail_identity_retry"
    if shot_type == "full_board":
        return visual * 0.96 + temporal * 0.04, "full_board_visual_retry"
    return visual * 0.90 + temporal * 0.10, "generic_identity_retry"


def _retry_group_candidates(photo: dict, cards: list[dict], count: int) -> list[int]:
    """Limit expensive retries to plausible card-neighbor groups."""
    if not cards:
        return list(range(count))
    if photo.get("shot_type") != "side_profile":
        return list(range(count))
    return _side_profile_neighbor_lanes(photo, cards, count)


def _global_identity_clusters(
    photos: list[dict],
    seed_lanes: list[list[dict]],
    count: int,
    cards: list[dict] | None = None,
) -> list[list[dict]]:
    """Assign photos using trusted seeds followed by targeted retries.

    Timestamp lanes are candidate windows only. The old implementation copied
    every lane into its initial representative, so one boundary mistake became
    permanent evidence for the wrong board. This version locks a small,
    coherent set of high-confidence views first, then retries the remainder
    against those frozen/robust representatives.
    """
    ordered_cards = sorted(cards or [], key=lambda item: item.get("_sequence_index", 0))
    candidate_lanes = [list(seed_lanes[index]) if index < len(seed_lanes) else [] for index in range(count)]
    clusters: list[list[dict]] = [[] for _ in range(count)]
    # These seeds are the identity anchor for the entire pass.  They must not
    # be rebuilt from later assignments, otherwise one bad boundary photo can
    # gradually redefine the group and then validate itself.
    frozen_representatives: list[list[dict]] = [[] for _ in range(count)]
    locked: set[int] = set()
    last_unassigned: list[dict] = []
    if not photos:
        return clusters
    bridge_seeds = _card_bridge_seeds(ordered_cards, candidate_lanes) if ordered_cards else [None] * count
    # Phase 1: accumulate only trusted guesses. The soft lanes narrow the
    # search, but ownership is earned by centrality and mutual similarity.
    for group_index, lane in enumerate(candidate_lanes):
        bridge = bridge_seeds[group_index] if group_index < len(bridge_seeds) else None
        if bridge is not None:
            clusters[group_index].append(bridge)
            frozen_representatives[group_index] = [bridge]
            locked.add(id(bridge))
        remaining_lane = [item for item in lane if id(item) not in locked]
        trusted = _trusted_identity_members(remaining_lane)
        accepted_trusted = []
        for photo in trusted:
            if id(photo) in locked:
                continue
            if bridge is not None and (
                tuple(photo.get("board_identity_signature") or ())
                == tuple(bridge.get("board_identity_signature") or ())
            ):
                # An identical signature is not independent evidence. Keep
                # the frame unresolved so a duplicate-looking boundary shot
                # cannot silently inflate the anchored group.
                continue
            conflicts = (
                _photo_identity_conflicts(photo, ordered_cards[group_index], ordered_cards)
                if group_index < len(ordered_cards) else []
            )
            if conflicts:
                photo["identity_assignment_conflicts"] = conflicts
                continue
            if bridge is not None:
                admitted, _, reason = _identity_admission(photo, [bridge], [bridge])
                if not admitted:
                    photo["identity_assignment_reason"] = reason
                    continue
            clusters[group_index].append(photo)
            accepted_trusted.append(photo)
            locked.add(id(photo))
            photo["identity_assignment_reason"] = "trusted_seed"
        if not frozen_representatives[group_index]:
            frozen_representatives[group_index] = accepted_trusted

    # Phase 2: expensive retries. Prototypes are rebuilt only from accepted
    # members, never from unresolved or quarantined photos.
    for retry_round in range(IDENTITY_RETRY_ROUNDS):
        representatives = [_identity_representatives(cluster) for cluster in clusters]
        unresolved = [photo for photo in photos if id(photo) not in locked]
        if not unresolved:
            break
        assignments: list[tuple[float, int, dict, str]] = []
        still_unassigned: list[dict] = []
        for photo in unresolved:
            candidate_indices = _retry_group_candidates(photo, ordered_cards, count)
            scored = []
            conflicts_by_cluster = {}
            # Prefer the neighboring chronological lanes first, but retain
            # bounded global backtracking for a card photographed slightly
            # late or early. Ordering is a prior, never an assignment.
            photo_time = _capture_seconds(photo)
            def temporal_candidate_key(index: int) -> tuple[float, int]:
                group_time = _cluster_time_center(clusters[index])
                if photo_time is None or group_time is None:
                    return float("inf"), index
                return abs(group_time - photo_time), index

            candidate_indices = sorted(candidate_indices, key=temporal_candidate_key)
            for group_index in candidate_indices:
                conflicts = (
                    _photo_identity_conflicts(photo, ordered_cards[group_index], ordered_cards)
                    if group_index < len(ordered_cards) else []
                )
                conflicts_by_cluster[group_index] = conflicts
                if conflicts:
                    continue
                score, reason = _identity_retry_score(
                    photo, group_index, representatives[group_index], clusters, ordered_cards,
                )
                admitted, admission_score, admission_reason = _identity_admission(
                    photo,
                    clusters[group_index],
                    frozen_representatives[group_index] or representatives[group_index],
                )
                if not admitted:
                    conflicts_by_cluster[group_index] = [admission_reason]
                    continue
                score = max(score, admission_score)
                scored.append((score, group_index, reason))
            if not scored:
                photo["identity_assignment_reason"] = "explicit_identity_conflict"
                photo["identity_assignment_conflicts"] = conflicts_by_cluster
                still_unassigned.append(photo)
                continue
            scored.sort(key=lambda row: (row[0], row[1]))
            best_score, winner, reason = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else None
            margin = (second_score - best_score) if second_score is not None else None
            # Side profiles have a protocol-driven route and are accepted on
            # the local time decision. Other shots need both a plausible
            # distance and a meaningful lead over the next group.
            side_profile = photo.get("shot_type") == "side_profile"
            acceptable = (
                side_profile
                or (
                    best_score <= IDENTITY_ASSIGNMENT_MAX_DISTANCE
                    and (margin is None or margin >= IDENTITY_ASSIGNMENT_MIN_MARGIN)
                )
            )
            if not acceptable:
                photo["identity_assignment_reason"] = "ambiguous_identity" if margin is not None and margin < IDENTITY_ASSIGNMENT_MIN_MARGIN else "weak_identity_match"
                photo["identity_assignment_scores"] = [
                    {"group_index": index, "distance": round(float(score), 6)}
                    for score, index, _ in scored
                ]
                still_unassigned.append(photo)
                continue
            # A retry round should not bulk-absorb every remaining photo. Lock
            # only the strongest half of each group, then rebuild prototypes;
            # later rounds handle the harder angles with better evidence.
            assignments.append((best_score, winner, photo, reason))

        by_group: dict[int, list[tuple[float, dict, str]]] = {}
        for score, winner, photo, reason in assignments:
            by_group.setdefault(winner, []).append((score, photo, reason))
        for winner, rows in by_group.items():
            rows.sort(key=lambda row: (row[0], _identity_candidate_key(row[1])))
            # Round one is intentionally conservative; subsequent rounds may
            # admit more, but each group still gets its strongest evidence.
            quota = max(1, min(len(rows), 2 if retry_round == 0 else len(rows)))
            for score, photo, reason in rows[:quota]:
                clusters[winner].append(photo)
                locked.add(id(photo))
                photo["identity_assignment_reason"] = reason
                photo["identity_retry_round"] = retry_round + 1
                photo["identity_retry_score"] = round(float(score), 6)
            for _, photo, _ in rows[quota:]:
                still_unassigned.append(photo)
        last_unassigned = still_unassigned

    # Final contamination pass. A group is only legitimate if every member
    # still agrees with the group's trusted board identity. Any late member
    # that fails is removed and returned to quarantine instead of being shown
    # as part of a mixed board group.
    contaminated: list[dict] = []
    for cluster_index, cluster in enumerate(clusters):
        representatives = frozen_representatives[cluster_index] or _identity_representatives(cluster)
        retained = []
        for photo in cluster:
            admitted, score, reason = _identity_admission(photo, cluster, representatives)
            if admitted:
                retained.append(photo)
                continue
            photo["identity_assignment_reason"] = reason
            photo["identity_retry_score"] = round(float(score), 6)
            photo["identity_assignment_conflicts"] = ["final_group_coherence"]
            contaminated.append(photo)
        clusters[cluster_index] = retained

    quarantined = [photo for photo in photos if id(photo) not in locked]
    quarantined.extend(photo for photo in contaminated if id(photo) not in {id(item) for item in quarantined})
    for photo in quarantined:
        photo.setdefault("identity_assignment_reason", "weak_identity_match")
    if quarantined:
        assigned = {id(photo) for cluster in clusters for photo in cluster}
        quarantined = [photo for photo in quarantined if id(photo) not in assigned]
        for photo in quarantined:
            photo.setdefault("identity_assignment_reason", "weak_identity_match")
    return clusters + _split_quarantine_photos(quarantined)


def _quarantine_cluster_coherence(cluster: list[dict]) -> float:
    """Return the worst pairwise identity distance in a remainder cluster."""
    if len(cluster) < 2:
        return 0.0
    return max(
        visual_distance(left, right)
        for index, left in enumerate(cluster)
        for right in cluster[index + 1:]
    )


def _split_quarantine_photos(photos: list[dict]) -> list[list[dict]]:
    """Split unassigned photos before exposing them as review groups."""
    if not photos:
        return []
    remaining = list(photos)
    clusters: list[list[dict]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        while remaining:
            ranked = sorted(
                (
                    _median([visual_distance(candidate, member) for member in cluster]),
                    index,
                )
                for index, candidate in enumerate(remaining)
            )
            if not ranked or ranked[0][0] > QUARANTINE_CLUSTER_MAX_DISTANCE:
                break
            _, candidate_index = ranked[0]
            candidate = remaining[candidate_index]
            proposed = cluster + [candidate]
            if _quarantine_cluster_coherence(proposed) > QUARANTINE_CLUSTER_MAX_DISTANCE:
                break
            cluster.append(candidate)
            remaining.pop(candidate_index)
        clusters.append(cluster)

    result: list[list[dict]] = []
    for cluster in clusters:
        if len(cluster) < QUARANTINE_CLUSTER_MIN_SIZE or _quarantine_cluster_coherence(cluster) > QUARANTINE_CLUSTER_MAX_DISTANCE:
            for photo in cluster:
                photo["_unassigned_photo"] = True
                photo.setdefault("identity_assignment_reason", "no_coherent_quarantine_cluster")
                result.append([photo])
            continue
        for photo in cluster:
            photo["_unassigned_cluster"] = True
        result.append(cluster)
    return result


def _group_from_global_identity(files: list[dict], cards: list[dict], photos: list[dict]) -> list[list[dict]]:
    seed_lanes = _seed_card_lanes(cards, photos)
    clusters = _global_identity_clusters(photos, seed_lanes, len(cards), cards)
    # The final cluster is a cardless quarantine bucket. It must not be
    # passed through card matching or the review photo becomes falsely owned
    # by whichever card happens to be left over.
    owned_clusters = clusters[:len(cards)]
    review_clusters = clusters[len(cards):]
    # Card labels are identity anchors, not free labels. Keep the one-to-one
    # visual match, but require the cluster to remain within the capture-time
    # neighborhood of its card. Disabling this guard swapped the Stewart card
    # onto the neighboring Fire Wire cluster in real batches.
    return _attach_cards_to_anchored_clusters(cards, owned_clusters) + review_clusters


def _attach_cards_to_anchored_clusters(cards: list[dict], clusters: list[list[dict]]) -> list[list[dict]]:
    """Attach cards to their pre-created slots, never to a best-fit cluster.

    ``_global_identity_clusters`` creates one slot per sorted card. Re-running
    card-to-cluster matching here would undo that ownership and can attach a
    card to a neighboring board merely because both are visually generic.
    """
    ordered_cards = sorted(cards, key=lambda item: item.get("_sequence_index", 0))
    groups = []
    for index, cluster in enumerate(clusters[:len(ordered_cards)]):
        card = ordered_cards[index]
        already_present = any(item is card for item in cluster)
        groups.append(([] if already_present else [card]) + list(cluster))
    for card in ordered_cards[len(groups):]:
        groups.append([card])
    return groups


def _group_from_card_lanes(files: list[dict], cards: list[dict], photos: list[dict]) -> list[list[dict]]:
    """Make card-anchored chronological lanes, then refine their boundaries.

    The initial lane is the nearest-card time partition. Each boundary may
    move only within the local eight-photo neighborhood and only when a sharp
    visual change supports the move. This keeps cards from competing across an
    entire batch while still handling a card photographed slightly late.
    """
    ordered_cards = sorted(cards, key=lambda item: item.get("_sequence_index", 0))
    ordered_photos = sorted(photos, key=lambda item: item.get("_sequence_index", 0))
    card_keys = [_card_order_key(card) for card in ordered_cards]
    photo_keys = [_card_order_key(photo) for photo in ordered_photos]
    initial_boundaries = []
    for left, right in zip(card_keys, card_keys[1:]):
        initial_boundaries.append((left + right) / 2.0)
    boundaries = []
    for initial in initial_boundaries:
        midpoint_boundary = sum(1 for key in photo_keys if key <= initial)
        boundaries.append(_refine_lane_boundary(ordered_photos, midpoint_boundary))
    boundaries = sorted(max(0, min(len(ordered_photos), boundary)) for boundary in boundaries)
    lanes = []
    start = 0
    for boundary in boundaries:
        boundary = max(start, boundary)
        lanes.append(ordered_photos[start:boundary])
        start = boundary
    lanes.append(ordered_photos[start:])
    grouped: list[list[dict]] = []
    for index, lane in enumerate(lanes):
        subclusters = _cluster_board_photos(lane)
        if not subclusters:
            grouped.append([ordered_cards[index]])
            continue
        # The card can only match a visual run inside its own chronological
        # lane. It can never pull a photo across another card's lane.
        grouped.extend(_match_cards_to_clusters([ordered_cards[index]], subclusters))
    return grouped


def _card_order_key(item: dict) -> float:
    capture = _capture_seconds(item)
    if capture is not None:
        return capture
    return float(item.get("_sequence_index", 0))


def _boundary_change_score(photos: list[dict], boundary: int) -> float:
    if boundary < 2 or len(photos) - boundary < 2:
        return 0.0
    left = photos[max(0, boundary - 3):boundary]
    right = photos[boundary:min(len(photos), boundary + 3)]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    across = _context_distance(left, right)
    within = max(_context_cohesion(left), _context_cohesion(right))
    return across - within


def _refine_lane_boundary(photos: list[dict], initial: int) -> int:
    """Find the strongest supported change point near a card-time boundary."""
    candidates = list(range(
        max(2, initial - BOUNDARY_REFINEMENT_RADIUS),
        min(len(photos) - 1, initial + BOUNDARY_REFINEMENT_RADIUS) + 1,
    ))
    # The capture protocol ends each board with a side profile immediately
    # before the next inventory card. If that profile is in the local search
    # window, every boundary at or before it is invalid. Merely removing the
    # exact boundary before the profile is insufficient: the visual scorer can
    # otherwise choose one frame earlier and produce the same one-photo shift.
    profiles_in_window = [
        index for index in range(max(0, initial - BOUNDARY_REFINEMENT_RADIUS), min(len(photos), initial + BOUNDARY_REFINEMENT_RADIUS + 1))
        if photos[index].get("shot_type") == "side_profile"
    ]
    if profiles_in_window:
        candidates = [boundary for boundary in candidates if boundary > max(profiles_in_window)]
    minimum_boundary = max(profiles_in_window, default=-1) + 1
    scored = sorted(
        (
            (_boundary_change_score(photos, boundary), boundary)
            for boundary in candidates
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < BOARD_SIMILARITY_BOUNDARY / 3:
        # Preserve the protocol invariant even when visual evidence is weak:
        # the final side profile belongs to the preceding card lane.
        return minimum_boundary if profiles_in_window else initial
    return scored[0][1]


def _cluster_board_photos(photos: list[dict]) -> list[list[dict]]:
    """Create visual board clusters without letting cards contaminate them."""
    if not photos:
        return []
    clusters = _split_by_board_similarity([photos])
    return clusters


def _cluster_distance(left: list[dict], right: list[dict]) -> float:
    return _median([
        visual_distance(before, after)
        for before in left[-LOCAL_NEIGHBOR_COUNT:]
        for after in right[:LOCAL_NEIGHBOR_COUNT]
    ])


def _cluster_time_center(cluster: list[dict]) -> float | None:
    captures = [value for value in (_capture_seconds(item) for item in cluster) if value is not None]
    return sum(captures) / len(captures) if captures else None


def _card_cluster_score(card: dict, cluster: list[dict], *, global_match: bool = False) -> float:
    visual = card_visual_affinity(card, cluster)
    if visual is None:
        visual = 1.0
    card_time, cluster_time = _capture_seconds(card), _cluster_time_center(cluster)
    if global_match and card_time is not None:
        photo_times = [value for value in (_capture_seconds(item) for item in cluster) if value is not None]
        temporal = min((abs(card_time - value) for value in photo_times), default=0.0) / 60.0
        return visual * 0.25 + min(temporal, CARD_MATCH_MAX_MINUTES) * 0.05
    if card_time is None or cluster_time is None:
        temporal = 0.5
    else:
        temporal = abs(card_time - cluster_time) / 60.0
    sequence = 0.0
    if card_time is None:
        card_position = card.get("_sequence_index")
        positions = [item.get("_sequence_index") for item in cluster if item.get("_sequence_index") is not None]
        if card_position is not None and positions:
            start, end = min(positions), max(positions)
            sequence = min(abs(card_position - start), abs(card_position - end)) * 0.1
            # A card between two runs conventionally starts the following
            # board's photo set. Prefer that run over the completed run just
            # before it when visual evidence is otherwise tied.
            if end < card_position:
                sequence += 0.5
    # Time is a useful anchor for the controlled photo protocol; visual
    # evidence breaks ties but cannot rescue a distant card match.
    return temporal + visual * 0.25 + sequence


def _match_cards_to_clusters(cards: list[dict], clusters: list[list[dict]], *, enforce_time: bool = True) -> list[list[dict]]:
    """Attach cards only to a nearby compatible cluster, one-to-one."""
    if not clusters:
        return [[card] for card in cards]
    assignments: dict[int, int] = {}
    pairs = sorted(
        (_card_cluster_score(card, cluster, global_match=not enforce_time), card_index, cluster_index)
        for card_index, card in enumerate(cards)
        for cluster_index, cluster in enumerate(clusters)
    )
    used_cards: set[int] = set()
    used_clusters: set[int] = set()
    max_minutes = CARD_MATCH_MAX_MINUTES
    for score, card_index, cluster_index in pairs:
        if card_index in used_cards or cluster_index in used_clusters:
            continue
        card_time = _capture_seconds(cards[card_index])
        cluster_time = _cluster_time_center(clusters[cluster_index])
        visual = card_visual_affinity(cards[card_index], clusters[cluster_index])
        if not enforce_time:
            assignments[cluster_index] = card_index
            used_cards.add(card_index)
            used_clusters.add(cluster_index)
            continue
        if card_time is None or cluster_time is None:
            if visual is not None and visual > CARD_MATCH_MAX_VISUAL_DISTANCE:
                continue
        if card_time is not None and cluster_time is not None:
            if abs(card_time - cluster_time) > max_minutes * 60:
                continue
        assignments[cluster_index] = card_index
        used_cards.add(card_index)
        used_clusters.add(cluster_index)
    groups = []
    for cluster_index, cluster in enumerate(clusters):
        card_index = assignments.get(cluster_index)
        card = cards[card_index] if card_index is not None else None
        # A board photo can also contain its inventory card. It is already in
        # the visual cluster, so do not duplicate it as a synthetic leading
        # card item in the output group.
        already_present = card is not None and any(item is card for item in cluster)
        groups.append(([] if card is None or already_present else [card]) + list(cluster))
    groups.extend([cards[index:index + 1] for index in range(len(cards)) if index not in used_cards])
    return groups


def _split_oversized_groups(groups: list[list[dict]], include_card: bool = True) -> list[list[dict]]:
    """Apply only the physical seven-photo cap; shot types remain review data."""
    limited: list[list[dict]] = []
    limit = MAX_PHOTOS_PER_BOARD if include_card else STANDARD_BOARD_PHOTOS
    for group in groups:
        remaining = list(group)
        while len(remaining) > limit:
            limited.append(remaining[:limit])
            remaining = remaining[limit:]
        if remaining:
            limited.append(remaining)
    return limited


def _split_by_board_similarity(groups: list[list[dict]]) -> list[list[dict]]:
    """Split mixed sessions by board appearance, then attach cards to a cluster.

    Cards provide identity data, but the board photos determine membership.
    Card frames are excluded from the boundary calculation because their
    paper/background layout is intentionally different from board frames.
    """
    partitioned: list[list[dict]] = []
    for group in groups:
        photos = [item for item in group if not item.get("is_card")]
        cards = [item for item in group if item.get("is_card")]
        if len(cards) > 1 or len(photos) < 4:
            partitioned.append(group)
            continue
        candidates: list[tuple[float, int]] = []
        for index in range(1, len(photos)):
            left_context = _local_context(photos, index, -1)
            right_context = _local_context(photos, index, 1)
            # Three immediately adjacent frames are enough to establish a
            # change point. Requiring four on each side caused a board at a
            # card boundary to borrow photos from its neighbor.
            left_context = left_context[:3]
            right_context = right_context[:3]
            if index < 2 or len(photos) - index < 2 or len(left_context) < 2 or len(right_context) < 2:
                continue
            across = _context_distance(left_context, right_context)
            within = max(_context_cohesion(left_context), _context_cohesion(right_context))
            if across >= BOARD_SIMILARITY_BOUNDARY and across >= within + BOARD_SIMILARITY_BOUNDARY / 3:
                candidates.append((across - within, index))
        boundaries: list[int] = []
        for _, index in sorted(candidates, reverse=True):
            if all(abs(index - selected) >= 2 for selected in boundaries):
                boundaries.append(index)
        boundaries.sort()
        if not boundaries:
            partitioned.append(group)
            continue
        clusters: list[list[dict]] = []
        start = 0
        for boundary in boundaries:
            clusters.append(photos[start:boundary])
            start = boundary
        clusters.append(photos[start:])
        # Cards are deliberately not inserted here. Keeping them out of the
        # visual-run pass prevents a card's yellow paper/background from
        # changing the board boundary calculation; matching happens once all
        # runs are known.
        partitioned.extend(cluster for cluster in clusters if cluster)
    return partitioned


def shot_bucket(item: dict) -> str | None:
    """Map a scanned/database photo to the composition bucket it consumes."""
    if item.get("is_card") or item.get("source_is_card"):
        return "card"
    shot_type = str(item.get("shot_type") or "").lower()
    if shot_type in SHOT_LIMITS:
        return shot_type
    return None


def group_limit_violations(items: list[dict]) -> dict[str, int]:
    """Return only hard composition-limit violations for one board group."""
    counts: dict[str, int] = {}
    for item in items:
        bucket = shot_bucket(item)
        if bucket:
            counts[bucket] = counts.get(bucket, 0) + 1
    violations = {bucket: count for bucket, count in counts.items() if count > SHOT_LIMITS[bucket]}
    if len(items) > MAX_PHOTOS_PER_BOARD:
        violations["total"] = len(items)
    return violations


def _enforce_shot_limits(groups: list[list[dict]]) -> list[list[dict]]:
    """Split proposed groups before a known shot-type limit is exceeded."""
    limited: list[list[dict]] = []
    for group in groups:
        current: list[dict] = []
        for item in group:
            candidate = current + [item]
            if current and group_limit_violations(candidate):
                limited.append(current)
                current = []
            current.append(item)
        if current:
            limited.append(current)
    return limited
