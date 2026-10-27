from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .card_ocr import extract_card_identifier, extract_card_identifier_with_confidence, ocr_quality
from .board_evidence import evidence_fields_for_shot
from .segmentation import board_perimeter, board_prompt_box, get_segmenter, mask_crop

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
PROCESSING_VERSION = "1.3.0-embedding-boundary-tiebreak"
MAX_PHOTOS_PER_BOARD = 7
STANDARD_BOARD_PHOTOS = 6
SHOT_LIMITS = {"full_board": 4, "side_profile": 1, "fin_detail": 1, "card": 1}
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
    board_mask: "np.ndarray | None" = None,
) -> tuple[float, ...]:
    """Describe the board itself after normalizing away the room.

    The grouping fingerprint is intentionally made from the detected board
    crop, not the whole camera frame. It combines color distribution, a
    normalized grayscale layout, edge/graphic layout, and coarse geometry.
    Geometry is kept as a small final section so a side profile can still be
    compared to the same board's full-board views without dominating identity.

    ``board_mask`` (e.g. from SAM 2), if given, is a boolean array the same
    size as ``image``. The candidate rectangle alone is not reliable: an
    off-center board can leave the rectangle mostly background, and that
    background then dominates the signature, so two unrelated boards shot
    against the same studio wall score as near-identical. The mask, when
    present, excludes non-board pixels from the crop before any feature is
    computed.
    """
    if cv2 is None or np is None:
        return ()
    working = image.convert("RGB")
    width, height = working.size
    geometry = (0.0, 0.0, 0.0, 0.0)
    crop_box: tuple[int, int, int, int] | None = None
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
            crop_box = (left, top, right, bottom)
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
        crop_box = (left, top, right, bottom)
    working.thumbnail((480, 640))
    # ``np.asarray(PIL.Image)`` may expose a read-only decoder buffer.  Card
    # masking below writes into this array, so always use a writable copy.
    array = np.asarray(working).copy()
    if array.size == 0:
        return ()
    if board_mask is not None and crop_box is not None and tuple(board_mask.shape[:2]) == (height, width):
        left, top, right, bottom = crop_box
        cropped_mask = board_mask[top:bottom, left:right]
        if cropped_mask.size:
            resized_mask = cv2.resize(
                cropped_mask.astype(np.uint8), (array.shape[1], array.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            # A real board can be a thin sliver of a loose/off-center crop
            # (an edge-on full-board shot easily covers under 5% of its own
            # candidate box), so the floor here is an absolute pixel count,
            # not a share of the crop. Only an essentially-empty or
            # essentially-everything mask is distrusted as a bad prediction.
            if max(100, int(resized_mask.size * 0.002)) <= int(resized_mask.sum()) < resized_mask.size * 0.98:
                array[~resized_mask] = 128
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
    board_mask: "np.ndarray | None" = None,
) -> tuple[float, ...]:
    """Capture local board evidence that whole-board averages erase.

    The global signature is useful for silhouette and layout, but it can make
    unrelated white boards look alike. These fixed board-relative patches keep
    rail color, nose/deck markings, tail traction, fin-box layout, and local
    edge texture separate. Profiles intentionally do not get this signature.

    ``board_mask`` behaves as in ``board_identity_signature``: when given, it
    blanks non-board pixels before the rail/nose/center/tail regions are
    sampled, so a loose candidate box that is mostly background wall does not
    get reported as this board's rail or deck color.
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
    if board_mask is not None and tuple(board_mask.shape[:2]) == source.shape[:2]:
        cropped_mask = board_mask[max(0, y):min(source.shape[0], y + height), max(0, x):min(source.shape[1], x + width)]
        if cropped_mask.size:
            resized_mask = cv2.resize(cropped_mask.astype(np.uint8), (48, 128), interpolation=cv2.INTER_NEAREST).astype(bool)
            if 0.05 <= float(resized_mask.mean()) < 0.98:
                board[~resized_mask] = 128
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


def board_region_mask(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None,
    shot_type: str,
) -> "np.ndarray | None":
    """Return a full-image boolean board mask from SAM 2, or None.

    Lazy and optional: with no SAM 2 checkpoint configured (``get_segmenter``
    returns None, the default today) this is a no-op, and callers fall back
    to the candidate rectangle alone — identical to behavior before this
    function existed. When SAM 2 is configured, its mask is what lets
    ``board_identity_signature``/``board_local_signature`` describe the
    board itself rather than whatever a loose, off-center rectangle happened
    to include (see their docstrings for why that matters).
    """
    result = board_perimeter(image, shot_type)
    return result["mask"] if result else None


def board_mask_width_fraction(mask: "np.ndarray") -> float | None:
    """Return the true board's width as a fraction of the frame width.

    Uses the largest connected component of the mask, not its raw bounding
    box: a few stray false-positive pixels elsewhere in the frame (a pipe,
    a shadow) can otherwise make a genuinely narrow, standing-edge-on board
    look as wide as the whole image.
    """
    if cv2 is None or np is None or mask is None:
        return None
    height, width = mask.shape[:2]
    if not width:
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(stats[index, cv2.CC_STAT_WIDTH]) / width


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


def exif_capture_metadata(image: Image.Image) -> tuple[str | None, str]:
    """Return the capture time and the exact metadata source used.

    File ordering and filesystem mtimes are not capture evidence.  Keeping the
    source beside the parsed value makes a missing/ambiguous timestamp visible
    in diagnostics instead of silently turning it into a temporal prior.
    """
    exif = image.getexif()
    for tag, source in (
        (36867, "exif_datetime_original"),
        (36868, "exif_datetime_digitized"),
        (306, "exif_datetime"),
    ):
        raw = exif.get(tag)
        if not raw:
            continue
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat(sep=" "), source
        except ValueError:
            return None, f"{source}_invalid"
    return None, "missing"


def exif_capture_time(image: Image.Image) -> str | None:
    return exif_capture_metadata(image)[0]


def read_image(path: Path) -> tuple[Image.Image, str | None]:
    with Image.open(path) as opened:
        capture, source = exif_capture_metadata(opened)
        converted = ImageOps.exif_transpose(opened).convert("RGB")
        # Preserve this without changing the long-standing read_image API.
        converted.info["usedsurf_capture_time_source"] = source
        return converted, capture


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


def local_ocr(image: Image.Image, region: tuple[int, int, int, int] | None = None, fast: bool = False) -> tuple[str, dict[str, str], str]:
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
                variants.append(card)
                if not fast:
                    variants.append(card.crop((round(card.width * 0.47), 0, card.width, round(card.height * 0.62))))
        else:
            image = image.copy()
            image.thumbnail((1200, 1200))
            width, height = image.size
            variants = [image]
            if not fast:
                variants.extend([
                    image.crop((0, round(height * 0.18), width, round(height * 0.86))),
                    image.crop((round(width * 0.10), round(height * 0.22), round(width * 0.90), round(height * 0.78))),
                ])
        texts = []
        for variant in variants:
            if fast:
                texts.append(pytesseract.image_to_string(variant, config="--psm 11"))
            else:
                texts.append(pytesseract.image_to_string(variant, config="--psm 6"))
                texts.append(pytesseract.image_to_string(ImageOps.autocontrast(ImageOps.grayscale(variant)), config="--psm 11"))
        text = max(texts, key=_ocr_signal_score, default="")
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"
    return text, extract_card_identifier(texts), ""


def board_ocr(image: Image.Image, candidate: tuple[int, int, int, int] | None = None, fast: bool = False) -> tuple[str, dict[str, str], str]:
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
        variants = [crop]
        if not fast:
            variants.extend([
                crop.crop((0, 0, width, round(height * 0.58))),
                crop.crop((0, round(height * 0.42), width, height)),
                ImageEnhance.Contrast(crop).enhance(1.7).filter(ImageFilter.SHARPEN),
            ])
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


def _densest_text_cluster(
    boxes: list[tuple[int, int, int, int]], pad: float, min_words: int = 4,
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of the largest cluster of nearby word boxes.

    Pure geometry, no OCR — kept separate from ``locate_card_region_by_text``
    so the clustering itself is directly testable without pytesseract. Two
    boxes (padded by ``pad``) join the same cluster when they overlap; the
    cluster with the most words wins. A handful of stray hits (a sponsor
    logo word, a wall scuff read as a character) is noise, not a card, so a
    cluster smaller than ``min_words`` is rejected rather than returned.
    """
    if len(boxes) < min_words:
        return None
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[0] + a[2] + pad, a[1] + a[3] + pad
        bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[0] + b[2] + pad, b[1] + b[3] + pad
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if overlaps(boxes[i], boxes[j]):
                union(i, j)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(find(index), []).append(box)
    best = max(groups.values(), key=len)
    if len(best) < min_words:
        return None
    x0 = min(box[0] for box in best)
    y0 = min(box[1] for box in best)
    x1 = max(box[0] + box[2] for box in best)
    y1 = max(box[1] + box[3] for box in best)
    return x0, y0, x1 - x0, y1 - y0


def locate_card_region_by_text(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find the inventory card by where the dense printed text is, not by
    color.

    The color-based detector (``card_candidate``) looks for a yellow,
    saturated patch — but it can't tell a small yellow card apart from a
    yellow board it's taped to; the whole frame reads as one yellow blob.
    A full-board photo has exactly one object with dense printed text,
    though: the card. This clusters OCR word boxes from a full-frame pass
    (see ``_densest_text_cluster``) and returns the bounding box of the
    densest one. It is color-blind by construction, so it works the same
    whether the board underneath is yellow, white, or anything else.
    """
    if pytesseract is None:
        return None
    _configure_tesseract_binary()
    try:
        working = image.convert("RGB")
        scale = min(1.0, 1200 / max(working.width, working.height))
        if scale < 1.0:
            working = working.resize((round(working.width * scale), round(working.height * scale)))
        data = pytesseract.image_to_data(working, config="--psm 11", output_type=pytesseract.Output.DICT)
    except Exception:
        return None
    boxes: list[tuple[int, int, int, int]] = []
    for text, conf, x, y, w, h in zip(
        data.get("text", []), data.get("conf", []), data.get("left", []),
        data.get("top", []), data.get("width", []), data.get("height", []),
    ):
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = -1.0
        if len((text or "").strip()) < 2 or confidence < 40 or w <= 0 or h <= 0:
            continue
        boxes.append((x, y, w, h))
    cluster = _densest_text_cluster(boxes, pad=max(working.width, working.height) * 0.02)
    if cluster is None:
        return None
    x0, y0, cw, ch = cluster
    inv_scale = 1.0 / scale if scale < 1.0 else 1.0
    return (
        max(0, round(x0 * inv_scale)),
        max(0, round(y0 * inv_scale)),
        round(cw * inv_scale),
        round(ch * inv_scale),
    )


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


def scan_files(input_dir: Path, progress=None) -> list[dict]:
    """Scan and analyze every photo in ``input_dir`` fresh, every time.

    No cache: this app's real usage never reprocesses the same batch twice
    (a batch is grouped, reviewed, and archived once), so a hash-keyed
    cache only added complexity and a real failure mode — a stale entry
    can survive a file rename and silently hand back another photo's
    analysis under the new name.
    """
    found = []
    ocr_status = ocr_runtime_status()
    paths = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        try:
            content_hash = sha256(path)
            image, capture = read_image(path)
            capture_source = str(image.info.get("usedsurf_capture_time_source", "missing"))
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
            # Color-based card detection fails outright when the board
            # itself is the card's color (a yellow board makes the whole
            # frame one yellow blob, same as the card taped to it). Text
            # density doesn't have that blind spot: the card is the only
            # object in a full-board photo with dense printed fields. When
            # SAM 2 is configured its mask tightens this into a precise
            # card boundary; otherwise the raw text cluster is still a far
            # better OCR crop than guessing from color.
            text_region = locate_card_region_by_text(image)
            if text_region is not None:
                segmenter = get_segmenter()
                if segmenter is not None:
                    tx, ty, tw, th = text_region
                    result = segmenter.predict(image, (float(tx), float(ty), float(tx + tw), float(ty + th)))
                    if result.mask is not None:
                        tightened = mask_crop(result.mask, image.width, image.height, padding=0.08)
                        if tightened is not None:
                            text_region = tightened
                card_regions = card_regions + [text_region]
            yellow_score = yellow_card_score(image)
            candidate_is_card = False
            if candidate:
                cx, cy, cw, ch, candidate_score = candidate
                # A real card is a substantial interior rectangle. Long warm
                # floor/rail edges and yellow board graphics usually touch an
                # image edge or are too short to satisfy this test.
                within_frame_horizontally = (
                    0.08 * image.width < cx < 0.78 * image.width
                    and 0.12 * image.width < cw < 0.70 * image.width
                )
                within_frame_vertically = (
                    0.18 * image.height < cy < 0.70 * image.height
                    and 0.16 * image.height < ch < 0.62 * image.height
                    and cy + ch < 0.92 * image.height
                )
                # A card held close to the camera can clip the top or bottom
                # margin on its own; an overwhelming yellow fraction (nothing
                # else in this capture protocol produces that much yellow
                # coverage) is strong enough evidence to skip that specific
                # check rather than silently drop a well-read card.
                candidate_is_card = (
                    candidate_score >= 1.05
                    and within_frame_horizontally
                    and (within_frame_vertically or yellow_score >= 0.35)
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
                quality = ocr_quality(text, identifier)
                per_file_ocr_status = "read" if quality["status"] == "usable" else ("empty" if not ocr_error else "error")
            else:
                text, identifier, ocr_error, field_evidence = "", {}, "", {}
                per_file_ocr_status = "not_run"
            largest_card_fraction = max((region[2] * region[3] for region in card_regions), default=0) / max(image.width * image.height, 1)
            has_card_evidence = bool(
                identifier
                or any(word in text.lower() for word in ("board information", "fin system", "brand", "model", "item"))
                or (card_regions and largest_card_fraction >= 0.015)
            ) or candidate_is_card
            # OCR from a full board frequently finds words such as "model" or
            # "brand" in logos and graphics, so a bare keyword hit alone is
            # not standalone-card evidence. `card_signal` is already the
            # strict gate against that false positive (a SKU, or 2+ of
            # item/brand/model) and must not additionally depend on the
            # separate contour-based `card_regions` detector agreeing — a
            # card photographed at an angle or cropped at the frame edge can
            # read perfectly (SKU and all) while that detector finds nothing,
            # which previously meant a clearly-identified card was silently
            # demoted to full_board/side_profile and never anchored a group.
            compact_card_ocr = card_signal(text, identifier) and (not card_regions or largest_card_fraction <= 0.45)
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
                "capture_source": capture_source,
                "card_candidate": candidate[:4] if candidate else None,
                "card_confidence": candidate[4] if candidate else 0.0,
                "card_regions": card_regions,
                "contains_inventory_card": has_card_evidence,
                "card_field_evidence": field_evidence,
                "yellow_card_score": yellow_score,
                "ocr_status": per_file_ocr_status,
                "ocr_quality": quality,
                "ocr_error": ocr_error,
                "ocr_runtime": ocr_status,
                "visual_signature": visual_signature(image),
                "is_card": standalone_card,
            })
        except Exception as exc:
            found.append({"path": path, "error": str(exc), "capture": None, "width": None, "height": None, "ocr": "", "identifier": {}, "ocr_status": "scan_error", "ocr_error": "", "ocr_runtime": ocr_status, "is_card": False, "visual_signature": ()})
        if progress:
            progress(index, total, path.name, cached=False)
    # Stable tie-breaker is filename, but capture time remains primary.
    return sorted(found, key=lambda item: (item["capture"] is None, item["capture"] or "", item["path"].name.lower()))


def group_scanned(files: list[dict]) -> list[list[dict]]:
    """Group photos by capture order, anchored to inventory cards.

    The capture protocol is fixed and the input is human-curated: a card
    photo, then that board's photos, in that order, until the next card.
    Trusting that order *is* the algorithm. Earlier versions tried to
    confirm or override it with visual similarity (color histograms, then
    masked histograms, then SAM 2 masks) — every version of that produced
    real cross-board contamination in production, because a confused visual
    match only had to be marginally more confident than the right answer to
    win, not actually correct. No visual signal is used to second-guess
    capture order here.

    A run of photos with no card at all (before the first detected card, or
    an entire batch with no cards) stays one group for a human to assign,
    rather than being auto-split by a visual-similarity guess.
    """
    if not files:
        return []
    for index, item in enumerate(files):
        item.setdefault("_sequence_index", index)
    ordered = sorted(files, key=lambda item: item.get("_sequence_index", 0))
    groups: list[list[dict]] = []
    current: list[dict] | None = None
    for item in ordered:
        if is_card_anchor(item):
            current = [item]
            groups.append(current)
            continue
        if current is None:
            if not groups:
                groups.append([])
            current = groups[0]
        current.append(item)
    return groups


def assign_shot_types_by_position(groups: list[list[dict]]) -> None:
    """Assign shot_type by position in each card-anchored, ordered group.

    The capture protocol is fixed: up to ``SHOT_LIMITS["full_board"]`` full
    board views, then one side profile, then optionally one fin detail, in
    that order. Position already tells us the type — guessing it from pixel
    shape (a silhouette's width, an aspect ratio) is where real
    misclassifications came from: a standing side-rail shot whose detected
    box happened to measure wide was called full_board and ended up
    contaminating an unrelated board's group. Mutates items in place. A
    photo beyond the known protocol slots (real, if rare: an extra shot) is
    left as full_board rather than guessed at or dropped.
    """
    slots = (
        ["full_board"] * SHOT_LIMITS.get("full_board", 4)
        + ["side_profile"] * SHOT_LIMITS.get("side_profile", 1)
        + ["fin_detail"] * SHOT_LIMITS.get("fin_detail", 1)
    )
    for group in groups:
        position = 0
        for item in group:
            if is_card_anchor(item):
                item["shot_type"] = "card"
                continue
            item["shot_type"] = slots[position] if position < len(slots) else "full_board"
            item["shot_type_source"] = "capture_position"
            position += 1


def next_shot_type_for_occupancy(counts: dict[str, int]) -> str:
    """Return the next protocol slot type given a board's current counts.

    Same slot order as assign_shot_types_by_position (full_board, then
    side_profile, then fin_detail), but keyed by what a board already has
    rather than raw capture position - for a photo manually moved onto an
    existing board after the fact, whose own original position belonged to
    a different group's sequence and says nothing about this board's
    actual gap. Cards never pass through here.
    """
    for shot_type in ("full_board", "side_profile", "fin_detail"):
        if counts.get(shot_type, 0) < SHOT_LIMITS.get(shot_type, 0):
            return shot_type
    return "full_board"


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


def embedding_distance(left: "np.ndarray | None", right: "np.ndarray | None") -> float | None:
    """Cosine distance between two pooled SAM 2 embeddings, or None if
    either is unavailable."""
    if left is None or right is None:
        return None
    return float(1.0 - np.dot(left, right))


def locate_missed_card_boundary(distances: list[float]) -> int | None:
    """Given consecutive-pair embedding distances within an oversized,
    card-anchored group (in capture order), return the index to split
    after if one gap clearly stands out as a likely missed card, else
    None.

    This is a tie-breaker, not a decision-maker. Capture order already
    assigned every photo in this group to one board; this only ever runs
    on a group that's already structurally oversized (more photos than
    the fixed protocol allows for one board), and it only acts when the
    evidence for a second board inside it is strong: one gap both
    absolutely large and clearly the odd one out among the group's own
    other adjacent gaps. A merely large group with no such standout gap
    is left alone for a human to resolve, rather than force-split on a
    weak signal.

    The thresholds below are a starting point, not a value tuned against
    labeled examples — there was no accessible "this group really does
    contain 2 boards, split exactly here" ground truth to calibrate
    against yet. Revisit once real oversized groups with a known correct
    split point are available to check this against.
    """
    if len(distances) < 2:
        return None
    ranked = sorted(distances, reverse=True)
    top, runner_up = ranked[0], ranked[1]
    if top < 0.3 or top < runner_up * 1.5:
        return None
    return distances.index(top) + 1


def split_oversized_groups_by_embedding(groups: list[list[dict]]) -> list[list[dict]]:
    """Split an oversized group at a likely missed-card boundary, using
    SAM 2's own image embedding as a tie-breaker (see
    ``locate_missed_card_boundary``).

    Lazy and optional: with no SAM 2 checkpoint configured this is a
    no-op and every group passes through unchanged. Only groups already
    larger than the standard protocol size are even considered — a
    correctly sized group never touches SAM 2 or an embedding here.
    """
    segmenter = get_segmenter()
    if segmenter is None:
        return groups
    result: list[list[dict]] = []
    for group in groups:
        photos = [item for item in group if not is_card_anchor(item)]
        if len(photos) <= STANDARD_BOARD_PHOTOS:
            result.append(group)
            continue
        embeddings = []
        for item in photos:
            try:
                image, _ = read_image(item["path"])
            except Exception:
                embeddings.append(None)
                continue
            embeddings.append(segmenter.embed(image))
        distances = [
            embedding_distance(left, right) or 0.0
            for left, right in zip(embeddings, embeddings[1:])
        ]
        split_at = locate_missed_card_boundary(distances)
        if split_at is None:
            result.append(group)
            continue
        card = next((item for item in group if is_card_anchor(item)), None)
        second_half = photos[split_at:]
        for item in second_half:
            item["identity_assignment_reason"] = "sam2_embedding_boundary_split"
        result.append(([card] if card else []) + photos[:split_at])
        result.append(second_half)
    return result
