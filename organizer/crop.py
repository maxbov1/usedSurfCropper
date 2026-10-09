from __future__ import annotations

import math
import statistics
from pathlib import Path

from PIL import Image

from .segmentation import board_prompt_box, get_segmenter, mask_crop
from .yolo_detector import PROFILE_ROI_SCALE, detect_board

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Product framing is board-relative: every side gets the same fixed margin
# measured from the detected board height.  This makes full-board shots
# stackable across a board's light/dark deck and bottom views.
DEFAULT_VERTICAL_PADDING = 0.10
DEFAULT_HORIZONTAL_PADDING = 0.10
# Compatibility names retained for callers and saved settings.
DEFAULT_END_PADDING = DEFAULT_VERTICAL_PADDING
DEFAULT_RAIL_PADDING = DEFAULT_HORIZONTAL_PADDING
# Side profiles use the same fixed board-relative margin.  Their width remains
# natural; racks and fins never determine the profile crop center.
# Profiles need slightly tighter nose/tail framing than deck/bottom shots;
# horizontal padding remains independently controlled below.
DEFAULT_PROFILE_PADDING = 0.08
DEFAULT_ROI_SEARCH_PADDING = 0.08
PROFILE_YOLO_CONFIDENCE = 0.10
FULL_BOARD_ROI_YOLO_CONFIDENCE = 0.05
# Backward-compatible name: the primary padding setting is the equal
# nose/tail margin. Rail padding is intentionally separate.
DEFAULT_CROP_PADDING = DEFAULT_END_PADDING


def build_crop_calibration(cases: list[dict], annotations: dict) -> dict[str, object]:
    """Learn conservative framing deltas from human manual crop corrections."""
    by_source = {case.get("source"): case for case in cases}
    samples: dict[str, list[tuple[float, float, float, float]]] = {}
    for source, annotation in annotations.items():
        if annotation.get("decision") != "manual":
            continue
        case = by_source.get(source)
        model = annotation.get("model_crop") or (case.get("proposal") if case else None)
        user = annotation.get("user_crop") or annotation.get("expected_crop")
        if not case or not model or not user:
            continue
        shot = annotation.get("shot_type") or case.get("evaluated_shot_type")
        if shot not in {"full_board", "side_profile"}:
            continue
        width = max(1, case.get("source_size", {}).get("width", 1))
        height = max(1, case.get("source_size", {}).get("height", 1))
        samples.setdefault(shot, []).append(((user["x"] - model["x"]) / width, (user["y"] - model["y"]) / height, user["width"] / max(1, model["width"]) - 1.0, user["height"] / max(1, model["height"]) - 1.0))
    groups = {}
    for shot, values in samples.items():
        if len(values) < 3:
            continue
        medians = [statistics.median(v[index] for v in values) for index in range(4)]
        deviations = [statistics.median(abs(v[index] - medians[index]) for v in values) for index in range(4)]
        # Corrections that represent different failure modes must not become
        # one global shift. Wait for a more consistent set instead.
        if deviations[0] > 0.04 or deviations[1] > 0.04 or deviations[2] > 0.06 or deviations[3] > 0.06:
            continue
        groups[shot] = {"count": len(values), "dx": round(max(-0.06, min(0.06, medians[0])), 5), "dy": round(max(-0.06, min(0.06, medians[1])), 5), "dw": round(max(-0.12, min(0.12, medians[2])), 5), "dh": round(max(-0.12, min(0.12, medians[3])), 5), "mad": [round(value, 5) for value in deviations]}
    return {"version": 1, "minimum_samples": 3, "groups": groups}


def apply_crop_calibration(crop: tuple[int, int, int, int], image_size: tuple[int, int], shot_type: str, calibration: dict[str, object] | None) -> tuple[int, int, int, int]:
    """Apply a bounded median correction learned from manual crops."""
    group = (calibration or {}).get("groups", {}).get(shot_type)
    if not group or group.get("count", 0) < 3:
        return crop
    image_width, image_height = image_size
    x, y, width, height = crop
    width = round(width * (1.0 + group["dw"]))
    height = round(height * (1.0 + group["dh"]))
    x += round(group["dx"] * image_width)
    y += round(group["dy"] * image_height)
    return _clamp_crop(x, y, width, height, image_width, image_height)


def classify_shot(image: Image.Image, card: bool = False) -> dict[str, object]:
    """Classify a normalized source using controlled-room geometry.

    This is deliberately a conservative framing classifier, not an object
    recognizer. A complete upright silhouette is a full-board shot; a long,
    shallow silhouette is a rail/profile shot; close-ups without either are
    preserved as details. ``review`` is true when the evidence is weak.
    """
    width, height = image.size
    if card:
        return {"shot_type": "card", "confidence": 1.0, "review": False, "reason": "inventory card supplied by OCR/card detector"}
    if cv2 is None or np is None:
        return {"shot_type": "detail", "confidence": 0.0, "review": True, "reason": "OpenCV unavailable"}
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    vertical = (_grabcut_candidate(array, orientation="vertical")
                or _grabcut_candidate(array, orientation="vertical", focused=True)
                or _find_silhouette_candidate(array, orientation="vertical"))
    horizontal = _grabcut_candidate(array, orientation="horizontal") or _find_silhouette_candidate(array, orientation="horizontal")
    profile_edge = _bright_profile_candidate(array)
    profile_seed = profile_edge
    if profile_seed is None and vertical is not None:
        _, vertical_box = vertical
        if vertical_box[2] / max(width, 1) < 0.20:
            profile_seed = vertical_box
    # YOLO is supporting evidence only. OpenCV's controlled-room geometry
    # remains responsible for the shot class and final crop boundary.
    yolo_detection = detect_board(image)
    # Edge-on boards occupy very few pixels and routinely fall below the
    # normal COCO detection threshold. A second, lower-threshold proposal is
    # useful only as profile evidence; full-board validation remains strict.
    if yolo_detection is None:
        yolo_detection = detect_board(image, confidence=PROFILE_YOLO_CONFIDENCE)
    roi_seed = profile_seed
    full_board_roi = False
    if yolo_detection is None and vertical is not None:
        # Full-board appearance can be far outside COCO's training domain
        # (for example, a heavily painted deck). OpenCV's complete silhouette
        # is a safe search seed even when YOLO sees no object at all.
        _, vertical_box = vertical
        if (vertical_box[2] / max(width, 1) >= 0.20
                and vertical_box[3] / max(height, 1) >= 0.48):
            roi_seed = vertical_box
            full_board_roi = True
    if roi_seed is not None:
        roi_confidence = FULL_BOARD_ROI_YOLO_CONFIDENCE if full_board_roi else PROFILE_YOLO_CONFIDENCE
        roi_detection = detect_board(image, confidence=roi_confidence, roi=roi_seed, scale=PROFILE_ROI_SCALE)
        if roi_detection is not None and (yolo_detection is None or roi_detection["confidence"] > yolo_detection["confidence"]):
            yolo_detection = roi_detection

    def with_yolo(result: dict[str, object]) -> dict[str, object]:
        if yolo_detection is not None:
            result["yolo_detection"] = yolo_detection
        return result

    # The detector's aspect ratio is a useful second opinion for rocker shots:
    # a board photographed across the frame produces a wide box, while an edge
    # view produces an unusually thin footprint. This is intentionally gated
    # by confidence and decisive ratios; OpenCV still supplies the crop.
    if yolo_detection is not None:
        yolo_width = float(yolo_detection["width"])
        yolo_height = float(yolo_detection["height"])
        yolo_ratio = yolo_width / max(yolo_height, 1.0)
        profile_ratio_limit = 0.18 if yolo_detection.get("inference_roi") is not None else 0.12
        if (yolo_detection["confidence"] >= 0.20
                and yolo_ratio <= profile_ratio_limit
                and yolo_detection["height"] >= height * 0.65
                and yolo_detection["y"] <= height * 0.12
                and yolo_detection["y"] + yolo_detection["height"] >= height * 0.82):
            return with_yolo({
                "shot_type": "side_profile",
                "confidence": round(min(0.98, max(0.72, float(yolo_detection["confidence"]))), 3),
                "review": bool(yolo_detection["confidence"] < 0.62),
                "reason": f"YOLO board footprint indicates rocker/side profile (box ratio {yolo_ratio:.3f})",
                "candidate": (int(yolo_detection["x"]), int(yolo_detection["y"]), int(yolo_detection["width"]), int(yolo_detection["height"])),
            })

    if horizontal is not None and (vertical is None or horizontal[0] >= vertical[0] * 0.90):
        score, box = horizontal
        confidence = float(min(0.98, max(0.45, score)))
        return with_yolo({"shot_type": "side_profile", "confidence": round(confidence, 3), "review": bool(confidence < 0.62), "reason": "wide shallow board silhouette", "candidate": box})
    if vertical is not None:
        score, box = vertical
        if (profile_edge is not None
                and box[2] / max(width, 1) >= 0.25
                and profile_edge[2] / max(box[2], 1) < 0.24
                and profile_edge[3] / max(height, 1) >= 0.65):
            confidence = float(min(0.98, max(0.55, float(score))))
            top_fraction = box[1] / max(height, 1)
            bottom_fraction = (box[1] + box[3]) / max(height, 1)
            partial_board = top_fraction > 0.16 or bottom_fraction < 0.87
            if partial_board:
                return with_yolo({"shot_type": "fin_detail", "confidence": round(confidence, 3), "review": True, "reason": "partial board with prominent appendage region; preserve original framing", "candidate": box})
            if profile_edge[2] / max(box[2], 1) < 0.17:
                return with_yolo({"shot_type": "side_profile", "confidence": round(confidence, 3), "review": bool(confidence < 0.62), "reason": "long narrow board edge with contaminated wide contour", "candidate": box})
        confidence = float(min(0.98, max(0.45, score)))
        narrow = box[2] / max(width, 1) < 0.20
        shot_type = "side_profile" if narrow else "full_board"
        reason = "narrow vertical rail silhouette" if narrow else "tall complete board silhouette"
        return with_yolo({"shot_type": shot_type, "confidence": round(confidence, 3), "review": bool(confidence < 0.62), "reason": reason, "candidate": box})
    return with_yolo({"shot_type": "detail", "confidence": 0.45, "review": True, "reason": "no complete board silhouette; preserve original framing"})


def shared_full_board_padding(cases: list[dict], desired_vertical: float = DEFAULT_VERTICAL_PADDING, desired_horizontal: float = DEFAULT_HORIZONTAL_PADDING) -> dict[str, float | int | bool]:
    """Choose one safe padding ratio for all full-board photos in a group.

    The tightest usable frame sets the group target. Each sibling therefore
    receives the same board-relative padding without any crop inventing pixels.
    """
    vertical_support: list[float] = []
    horizontal_support: list[float] = []
    for case in cases:
        if case.get("shot_type") != "full_board":
            continue
        boundary = case.get("classification", {}).get("opencv_boundary") or case.get("classification", {}).get("candidate")
        image_size = case.get("image_size") or (0, 0)
        if not boundary or len(boundary) < 4 or not image_size[0] or not image_size[1]:
            continue
        x, y, width, height = [float(value) for value in boundary[:4]]
        image_width, image_height = [float(value) for value in image_size]
        vertical_support.append(min(y, max(0.0, image_height - (y + height))) / max(height, 1.0))
        horizontal_support.append(min(x, max(0.0, image_width - (x + width))) / max(height, 1.0))
    if not vertical_support or not horizontal_support:
        return {"available": False, "vertical": desired_vertical, "horizontal": desired_horizontal, "count": 0}
    return {
        "available": True,
        "vertical": max(0.0, min(desired_vertical, min(vertical_support))),
        "horizontal": max(0.0, min(desired_horizontal, min(horizontal_support))),
        "count": len(vertical_support),
    }


def suggested_crop(image: Image.Image, shot_type: str = "full_board", padding: float = DEFAULT_CROP_PADDING, classification: dict[str, object] | None = None, rail_padding: float | None = None) -> tuple[int, int, int, int, str]:
    """Return an orientation-normalized pixel crop. No pixel is invented."""
    width, height = image.size
    candidate = None
    profile_edge = None
    boundary_source = "classifier"
    if shot_type == "auto":
        classification = classification or classify_shot(image)
        shot_type = str(classification["shot_type"])
        candidate = classification.get("candidate")
    elif shot_type in {"full_board", "side_profile"} and cv2 is not None and np is not None:
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        classification = classification or classify_shot(image)
        candidate = classification.get("candidate")
        yolo = (classification or {}).get("yolo_detection")
        usable_yolo = isinstance(yolo, dict) and _usable_yolo_box(yolo, width, height, shot_type)
        # A partial/frame-touching profile box is useful evidence, but it is
        # not safe to use as the crop extent. Let the OpenCV silhouette carry
        # the full nose-to-tail boundary in that case.
        if shot_type == "side_profile" and isinstance(yolo, dict):
            usable_yolo = usable_yolo and _complete_profile_yolo_box(yolo, width, height)
        if not usable_yolo:
            if classification is not None:
                classification["boundary_source"] = "OpenCV profile fallback" if shot_type == "side_profile" else "OpenCV full-board fallback"
                classification["review"] = True
            boundary_source = "OpenCV profile fallback" if shot_type == "side_profile" else "OpenCV full-board fallback"
        if isinstance(yolo, dict):
            refined = _contour_board_candidate_in_roi(array, yolo, min_width_fraction=0.03 if shot_type == "side_profile" else 0.18)
            if refined is not None:
                candidate = refined
                boundary_source = "YOLO ROI + OpenCV contour"
            elif usable_yolo:
                candidate = (int(yolo["x"]), int(yolo["y"]), int(yolo["width"]), int(yolo["height"]))
                boundary_source = "YOLO coarse box fallback"
        if shot_type == "side_profile":
            profile_edge = _bright_profile_candidate(array)
            if profile_edge is not None and candidate is not None:
                # Use the long board edge for horizontal centering, while
                # retaining the detected vertical extent so fins and tail are
                # not lost merely because the edge is narrow.
                candidate = (profile_edge[0], candidate[1], profile_edge[2], candidate[3])
        if shot_type == "full_board":
            # Use YOLO only to narrow the search region. The final boundary is
            # still the OpenCV contour, so this does not turn a coarse detector
            # box into a product crop or add any pixels.
            contour_box = None
            if isinstance(yolo, dict):
                contour_box = _contour_board_candidate_in_roi(array, yolo, min_width_fraction=0.18)
                if contour_box is not None:
                    boundary_source = "YOLO surfboard ROI + external contour"
                elif _usable_yolo_box(yolo, width, height):
                    contour_box = (int(yolo["x"]), int(yolo["y"]), int(yolo["width"]), int(yolo["height"]))
                    boundary_source = "YOLO coarse box fallback"
            if contour_box is None:
                contour_box = _contour_board_candidate(array)
            # A traced external contour is the primary boundary. A narrow contour
            # is usually the stringer, a shadow, or an internal logo edge.
            if contour_box is not None and contour_box[2] / max(image.width, 1) >= 0.25:
                candidate = contour_box
                if not boundary_source.startswith("YOLO"):
                    boundary_source = "external contour"
            elif candidate is None or candidate[2] / max(image.width, 1) < 0.25:
                extrema_box = _extrema_board_candidate(array)
                if extrema_box is not None and extrema_box[2] / max(image.width, 1) >= 0.25:
                    candidate = extrema_box
                    boundary_source = "silhouette extrema fallback"
                else:
                    broader = _find_board_candidate(array, "full_board")
                    if broader is not None and broader[2] / max(image.width, 1) >= 0.25:
                        candidate = broader
                        boundary_source = "broad candidate fallback"
    if shot_type in {"full_board", "side_profile"} and candidate:
            if shot_type == "side_profile" and cv2 is not None and np is not None:
                array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                if profile_edge is None:
                    candidate = _profile_body_candidate(array, candidate)
            x, y, w, h = candidate
            if classification is not None:
                classification["opencv_boundary"] = [x, y, w, h]
                classification["boundary_source"] = boundary_source
            if shot_type == "side_profile":
                crop = _profile_crop(x, y, w, h, width, height, max(padding, DEFAULT_PROFILE_PADDING))
                confidence = classification["confidence"] if classification else "contour"
                return (*crop, f"OpenCV shot classifier: side-profile silhouette ({confidence})")
            crop = _full_board_crop(x, y, w, h, width, height, padding, DEFAULT_RAIL_PADDING if rail_padding is None else rail_padding)
            confidence = classification["confidence"] if classification else "contour"
            return (*crop, f"OpenCV {boundary_source}: full-board boundary ({confidence})")
    if shot_type == "fin_detail":
        return 0, 0, width, height, "fin-detail framing preserved"
    if shot_type in {"detail", "accessory", "fins_closeup", "card", "keep_original"}:
        return 0, 0, width, height, "original framing"
    segmenter = get_segmenter()
    if segmenter is not None:
        prompt = board_prompt_box(width, height, shot_type)
        result = segmenter.predict(image, prompt)
        if result.mask is not None:
            crop = mask_crop(result.mask, width, height, padding)
            if crop is not None:
                return (*crop, f"SAM 2 mask (score {result.score:.3f})")
            return 0, 0, width, height, "SAM 2 rejected mask; manual crop required"
        return 0, 0, width, height, f"{result.method}; manual crop required"
    if cv2 is None:
        return 0, 0, width, height, "OpenCV unavailable; review full-frame suggestion"
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    candidate = _find_board_candidate(array, shot_type)
    if candidate is None:
        return 0, 0, width, height, "OpenCV found no board-shaped candidate; manual crop required"
    x, y, w, h = candidate
    if shot_type == "side_profile":
        crop = _profile_crop(x, y, w, h, width, height, padding)
        return (*crop, "OpenCV grayscale/edge profile candidate")
    crop = _full_board_crop(x, y, w, h, width, height, padding)
    return (*crop, "OpenCV grayscale/edge board candidate")


def rotated_crop_proposal(image: Image.Image, shot_type: str = "auto", padding: float = DEFAULT_CROP_PADDING, classification: dict[str, object] | None = None, rail_padding: float | None = None) -> dict[str, object]:
    """Propose a small board-angle correction followed by a crop.

    The rotated image uses an expanded canvas, but a rotated crop is accepted
    only when every output pixel maps to the source. Otherwise the original
    orientation/crop is returned and the caller can send it to review.
    """
    classification = classification or classify_shot(image)
    resolved_type = classification["shot_type"] if shot_type == "auto" else shot_type
    if resolved_type not in {"full_board", "side_profile"}:
        crop = suggested_crop(image, str(resolved_type), padding, classification=classification, rail_padding=rail_padding)
        reason = crop[4]
        return {"image": image, "crop": crop[:4], "shot_type": resolved_type, "angle": 0.0, "rotation_applied": False, "review": bool(classification.get("review", False)), "reason": reason}
    # Detect, refine, and pad in the original coordinate system first. The
    # rotation is deliberately the final transform; detector boxes must never
    # be reused against an image whose pixels have already moved.
    base_crop = suggested_crop(image, str(resolved_type), padding, classification=classification, rail_padding=rail_padding)
    angle = estimate_board_angle(image, str(resolved_type))
    if abs(angle) < 0.75:
        return {"image": image, "crop": base_crop[:4], "shot_type": resolved_type, "angle": 0.0, "rotation_applied": False, "review": bool(classification.get("review", False)), "reason": f"board already nearly upright; {base_crop[4]}"}
    rotated = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    rotated_crop = _rotate_rect(base_crop[:4], image.size, rotated.size, angle)
    if resolved_type == "full_board" and isinstance(classification.get("opencv_boundary"), (tuple, list)):
        boundary = tuple(classification["opencv_boundary"])
        if len(boundary) == 4:
            rotated_boundary = _rotate_rect(boundary, image.size, rotated.size, angle)
            rotated_crop = _equal_vertical_padding_crop(rotated_crop, rotated_boundary, rotated.size)
    if not _crop_is_source_valid(image.size, angle, rotated.size, rotated_crop):
        return {"image": image, "crop": base_crop[:4], "shot_type": resolved_type, "angle": 0.0, "attempted_angle": round(angle, 3), "rotation_applied": False, "review": True, "reason": f"rotation not applied; source crop kept; {base_crop[4]}"}
    return {"image": rotated, "crop": rotated_crop, "shot_type": resolved_type, "angle": round(angle, 3), "rotation_applied": True, "review": bool(classification.get("review", False)), "reason": f"OpenCV crop + padding before final board-axis rotation; {base_crop[4]}"}


def _rotate_rect(crop: tuple[int, int, int, int], source_size: tuple[int, int], rotated_size: tuple[int, int], angle: float) -> tuple[int, int, int, int]:
    """Transform an original-image crop rectangle into an expanded rotated canvas."""
    width, height = source_size
    rotated_width, rotated_height = rotated_size
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    center_x, center_y = width / 2.0, height / 2.0
    output_center_x, output_center_y = rotated_width / 2.0, rotated_height / 2.0
    x, y, crop_width, crop_height = crop
    points = ((x, y), (x + crop_width, y), (x + crop_width, y + crop_height), (x, y + crop_height))
    transformed = []
    for point_x, point_y in points:
        dx, dy = point_x - center_x, point_y - center_y
        transformed.append((output_center_x + dx * cosine - dy * sine, output_center_y + dx * sine + dy * cosine))
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return _clamp_crop(round(min(xs)), round(min(ys)), round(max(xs) - min(xs)), round(max(ys) - min(ys)), rotated_width, rotated_height)


def _equal_vertical_padding_crop(crop: tuple[int, int, int, int], boundary: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Force equal board-axis padding above and below a full-board crop."""
    crop_x, _, crop_width, _ = crop
    boundary_x, boundary_y, boundary_width, boundary_height = boundary
    image_width, image_height = image_size
    requested = max(0, min(crop[1] - boundary_y, (crop[1] + crop[3]) - (boundary_y + boundary_height)))
    available = min(boundary_y, max(0, image_height - (boundary_y + boundary_height)))
    padding = min(requested, available)
    return _clamp_crop(crop_x, boundary_y - padding, crop_width, boundary_height + 2 * padding, image_width, image_height)


def estimate_board_angle(image: Image.Image, shot_type: str) -> float:
    """Estimate a conservative rotation; full boards use the center stringer."""
    if cv2 is None or np is None:
        return 0.0
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if shot_type == "full_board":
        # The outline's PCA axis is not the product centerline: rocker,
        # shadows, and fins can pull it several degrees off. If the center
        # stringer is not found confidently, leave the original orientation.
        return _estimate_stringer_angle(array)
    mask = _silhouette_mask(array)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        wf, hf = w / width, h / height
        if shot_type == "side_profile":
            valid = hf >= 0.48 and wf <= 0.45
        else:
            valid = hf >= 0.48 and wf >= 0.12 and wf <= 0.82
        if not valid:
            continue
        candidates.append((w * h, contour))
    if not candidates:
        return 0.0
    _, contour = max(candidates, key=lambda item: item[0])
    points = contour.reshape(-1, 2).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    vx, vy = eigenvectors[0]
    angle = math.degrees(math.atan2(float(vx), float(-vy)))
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return float(-angle)


def _estimate_stringer_angle(array: np.ndarray) -> float:
    """Find the long centerline and return a small rotation toward vertical."""
    candidate = _grabcut_candidate(array, "vertical", focused=True)
    if candidate is None:
        return 0.0
    _, (x, y, board_width, board_height) = candidate
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 640.0 / max(array.shape[:2]))
    small = cv2.resize(gray, (max(80, round(array.shape[1] * scale)), max(80, round(array.shape[0] * scale))))
    sx, sy, sw, sh = [round(value * scale) for value in (x, y, board_width, board_height)]
    center = sx + sw / 2
    left = max(0, round(center - sw * 0.50))
    right = min(small.shape[1], round(center + sw * 0.50))
    top = max(0, sy)
    bottom = min(small.shape[0], sy + sh)
    roi = small[top:bottom, left:right]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=max(30, round(sh * 0.18)), maxLineGap=30)
    choices = []
    if lines is not None:
        # OpenCV returns either (N, 1, 4) or a squeezed variant depending on
        # the build/input. Normalize it before unpacking so one malformed
        # shape cannot abort the entire batch.
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            midpoint_x = (x1 + x2) / 2 + left
            if abs(dy) <= abs(dx) * 1.1 or abs(midpoint_x - center) > sw * 0.35:
                continue
            angle = math.degrees(math.atan2(dx, -dy))
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            choices.append((length, angle))
    if len(choices) < 4:
        return 0.0
    angles = [angle for _, angle in choices]
    angle = statistics.median(angles)
    dispersion = statistics.median(abs(value - angle) for value in angles)
    # A few wall seams or rail edges can be longer than the stringer. Require
    # consensus, and decline rotation entirely when the photo is materially
    # tilted instead of applying a conspicuous automatic correction.
    if dispersion > 2.0 or abs(angle) > 2.0:
        return 0.0
    return round(float(-angle), 3)


def _crop_is_source_valid(source_size: tuple[int, int], angle: float, rotated_size: tuple[int, int], crop: tuple[int, int, int, int]) -> bool:
    """Reject rotated crops that include expanded-canvas fill pixels."""
    if cv2 is None or np is None:
        return False
    source_width, source_height = source_size
    rotated_width, rotated_height = rotated_size
    mask = np.full((source_height, source_width), 255, dtype=np.uint8)
    rotated_mask = Image.fromarray(mask).rotate(angle, expand=True, resample=Image.Resampling.NEAREST, fillcolor=0)
    x, y, width, height = crop
    patch = np.asarray(rotated_mask)[y:y + height, x:x + width]
    return patch.shape == (height, width) and bool(np.all(patch > 0))


def _silhouette_mask(array: np.ndarray) -> np.ndarray:
    """Build a color-agnostic foreground mask from the gray controlled wall."""
    height, width = array.shape[:2]
    small = cv2.resize(array, (min(640, width), min(640, round(height * min(640, width) / width))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sh, sw = blurred.shape
    border = np.concatenate([
        blurred[: max(2, int(sh * .12)), :].ravel(),
        blurred[int(sh * .88):, :].ravel(),
        blurred[:, : max(2, int(sw * .07))].ravel(),
        blurred[:, int(sw * .93):].ravel(),
    ])
    background = float(np.median(border))
    variation = float(np.median(np.abs(border - background)))
    threshold = max(12.0, variation * 3.0 + 9.0)
    mask = (np.abs(blurred.astype(np.float32) - background) > threshold).astype(np.uint8) * 255
    # Edges retain dark rails and colored boards that have similar luminance.
    edges = cv2.Canny(blurred, 24, 90)
    mask = cv2.bitwise_or(mask, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    mask[: max(2, int(sh * .015)), :] = 0
    mask[int(sh * .985):, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _grabcut_candidate(array: np.ndarray, orientation: str, focused: bool = False) -> tuple[float, tuple[int, int, int, int]] | None:
    """Use one small GrabCut pass as a foreground proposal, not a final mask."""
    result = _grabcut_foreground_mask(array, focused=focused)
    if result is None:
        return None
    foreground, scale = result
    height, width = foreground.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[index]
    wf, hf = w / width, h / height
    if orientation == "vertical":
        # A side-profile board can occupy only a few percent of portrait
        # width. Keep the lower bound low, then classify it as profile below.
        if not (.02 <= wf <= .78 and hf >= .64 and area / float(width * height) >= .025):
            return None
        score = .55 * min(1.0, hf / .90) + .30 * min(1.0, area / (width * height) / .35) + .15
    else:
        if not (wf >= .62 and .06 <= hf <= .48 and area / float(width * height) >= .08):
            return None
        score = .55 * min(1.0, wf / .85) + .30 * min(1.0, area / (width * height) / .25) + .15
    box = (round(x / scale), round(y / scale), round(w / scale), round(h / scale))
    return min(.98, score), box


def _grabcut_foreground_mask(array: np.ndarray, focused: bool = False) -> tuple[np.ndarray, float] | None:
    """Return the small GrabCut foreground mask used by proposal logic."""
    original_height, original_width = array.shape[:2]
    scale = min(1.0, 220.0 / max(original_height, original_width))
    width, height = max(80, round(original_width * scale)), max(80, round(original_height * scale))
    small = cv2.resize(array, (width, height))
    mask = np.zeros((height, width), np.uint8)
    margin_x = max(4, round(width * (.25 if focused else .04)))
    margin_y = max(4, round(height * .02))
    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        # GrabCut can otherwise vary slightly across repeated calls on macOS,
        # which makes the same source produce different crop bounds.
        cv2.setRNGSeed(0)
        cv2.grabCut(small, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    foreground = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return foreground, scale


def _find_silhouette_candidate(array: np.ndarray, orientation: str) -> tuple[float, tuple[int, int, int, int]] | None:
    """Return a scored vertical or horizontal board-like silhouette."""
    original_height, original_width = array.shape[:2]
    mask = _silhouette_mask(array)
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        wf, hf = w / width, h / height
        area = (w * h) / float(width * height)
        center_x, center_y = (x + w / 2) / width, (y + h / 2) / height
        if orientation == "vertical":
            if hf < .48 or wf < .03 or wf > .82 or w / max(h, 1) > .78:
                continue
            shape = min(1.0, hf / .86) * min(1.0, wf / .34)
        else:
            if wf < .42 or hf < .045 or hf > .34 or w / max(h, 1) < 2.0:
                continue
            shape = min(1.0, wf / .72) * min(1.0, .24 / max(hf, .01))
        # A contour touching the frame is usually the wall/floor boundary or
        # a crop that already lost part of the board. It is unsafe to promote
        # that into an automatic crop.
        if x < width * .025 or y < height * .015 or x + w > width * .975 or y + h > height * .985:
            continue
        if area > .72 or abs(center_x - .5) > .34 or center_y < .22 or center_y > .82:
            continue
        # A silhouette should be materially larger than isolated wall/floor edges.
        score = .45 * shape + .30 * min(1.0, area / (.25 if orientation == "vertical" else .12)) + .25 * (1.0 - min(1.0, abs(center_x - .5) * 2.0))
        box = (round(x * original_width / width), round(y * original_height / height), round(w * original_width / width), round(h * original_height / height))
        if best is None or score > best[0]:
            best = (score, box)
    return best


def _find_board_candidate(array: np.ndarray, shot_type: str) -> tuple[int, int, int, int] | None:
    """Find a board-like rectangle without assuming the board's color.

    The background estimate is grayscale only. Edges are included because a
    colored or dark board can have little luminance distance from the wall.
    Framing and aspect-ratio constraints do the surfboard-specific work.
    """
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    border_width = max(8, round(width * 0.025))
    border_height = max(8, round(height * 0.025))
    # Avoid using the floor as the sole background reference. The wall sides
    # remain useful even when the board is dark or strongly colored.
    wall_border = np.concatenate(
        [
            blurred[: int(height * 0.70), :border_width].ravel(),
            blurred[: int(height * 0.70), -border_width:].ravel(),
            blurred[:border_height, :].ravel(),
        ]
    )
    background = float(np.median(wall_border))
    noise = float(np.median(np.abs(wall_border - background)))
    # A fixed low threshold absorbs wall texture and the floor seam. Scale it
    # from the measured wall variation instead; this remains color-agnostic.
    threshold = max(18.0, min(120.0, noise * 4.5 + 12.0))
    distance_mask = (np.abs(blurred.astype(np.float32) - background) > threshold).astype(np.uint8) * 255
    # Prefer the grayscale foreground mask. An all-edge mask tends to connect
    # wall cracks and the floor seam into a false frame-sized contour.
    base_mask = _clean_candidate_mask(distance_mask, width, height)
    candidate = _best_candidate(base_mask, shot_type, width, height)
    if candidate is not None:
        return candidate
    edges = cv2.Canny(blurred, 24, 78)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edge_mask = _clean_candidate_mask(cv2.bitwise_or(distance_mask, edges), width, height)
    return _best_candidate(edge_mask, shot_type, width, height)


def _contour_board_candidate(array: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find robust crop bounds from the board's traced contour.

    Raw single-pixel extrema are too sensitive to shadows and wall cracks. A
    convex hull plus one-percent quantiles keeps the board's rails while
    rejecting isolated contour noise. This is the primary full-board crop
    source; the older candidate detector remains the fallback.
    """
    result = _contour_box_from_mask(_silhouette_mask(array), array.shape[:2])
    if result is not None:
        return result
    # A centered seed prevents wall and floor regions from joining the board
    # when their shadows touch the board edge.
    grabcut = _grabcut_foreground_mask(array, focused=True)
    if grabcut is None:
        return None
    foreground, scale = grabcut
    return _contour_box_from_mask(foreground, array.shape[:2], scale=scale)


def _contour_board_candidate_in_roi(array: np.ndarray, detection: dict[str, object], min_width_fraction: float = 0.18) -> tuple[int, int, int, int] | None:
    """Trace a board contour inside a padded YOLO box and translate it back.

    YOLO boxes can be tight around the board, so the ROI is expanded before
    OpenCV sees it. If the expanded ROI touches the source edge, the result is
    rejected rather than inventing a reliable-looking crop.
    """
    image_height, image_width = array.shape[:2]
    x = int(detection.get("x", 0))
    y = int(detection.get("y", 0))
    w = int(detection.get("width", 0))
    h = int(detection.get("height", 0))
    if w <= 0 or h <= 0:
        return None
    pad_x = max(16, round(w * DEFAULT_ROI_SEARCH_PADDING))
    pad_y = max(16, round(h * DEFAULT_ROI_SEARCH_PADDING))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(image_width, x + w + pad_x), min(image_height, y + h + pad_y)
    if x1 - x0 < image_width * 0.16 or y1 - y0 < image_height * 0.45:
        return None
    roi_box = _contour_board_candidate(array[y0:y1, x0:x1])
    if roi_box is None:
        return None
    rx, ry, rw, rh = roi_box
    translated = (x0 + rx, y0 + ry, rw, rh)
    # The ROI should remain a meaningful board-shaped result, not a tiny
    # contour or a nearly full-frame rectangle.
    if translated[2] < image_width * min_width_fraction or translated[3] < image_height * 0.48:
        return None
    return translated


def _usable_yolo_box(detection: dict[str, object], image_width: int, image_height: int, shot_type: str = "full_board") -> bool:
    """Reject coarse boxes that touch the frame or are not board-shaped."""
    x, y = int(detection.get("x", 0)), int(detection.get("y", 0))
    w, h = int(detection.get("width", 0)), int(detection.get("height", 0))
    if w <= 0 or h <= 0:
        return False
    ratio = w / max(h, 1)
    if shot_type == "side_profile":
        # A profile detector may see only part of the board and may touch a
        # frame edge. OpenCV must corroborate the result before it is used.
        return h >= image_height * 0.30 and ratio <= 0.35
    if x <= image_width * 0.02 or y <= image_height * 0.01 or x + w >= image_width * 0.98 or y + h >= image_height * 0.99:
        return False
    return h >= image_height * 0.45 and ratio <= 1.0


def _complete_profile_yolo_box(detection: dict[str, object], image_width: int, image_height: int) -> bool:
    """Return whether a profile box spans enough of the board to crop from."""
    x, y = int(detection.get("x", 0)), int(detection.get("y", 0))
    w, h = int(detection.get("width", 0)), int(detection.get("height", 0))
    return (
        x > image_width * 0.02
        and x + w < image_width * 0.98
        and y <= image_height * 0.12
        and y + h >= image_height * 0.82
    )


def _contour_box_from_mask(mask: np.ndarray, original_shape: tuple[int, int], scale: float = 1.0) -> tuple[int, int, int, int] | None:
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        hull = cv2.convexHull(contour).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(hull.reshape(-1, 1, 2))
        if h < height * 0.48 or w < width * 0.12 or w > width * 0.86:
            continue
        if x < width * 0.02 or y < height * 0.01 or x + w > width * 0.98 or y + h > height * 0.99:
            continue
        area = (w * h) / float(width * height)
        candidates.append((h * w + area * width * height, hull))
    if not candidates:
        return None
    _, hull = max(candidates, key=lambda item: item[0])
    left, right = np.quantile(hull[:, 0], [0.01, 0.99])
    top, bottom = np.quantile(hull[:, 1], [0.01, 0.99])
    scale_x = original_shape[1] / width
    scale_y = original_shape[0] / height
    return (round(left * scale_x), round(top * scale_y), max(1, round((right - left) * scale_x)), max(1, round((bottom - top) * scale_y)))


def _extrema_board_candidate(array: np.ndarray) -> tuple[int, int, int, int] | None:
    """Use foreground extrema when an external contour cannot be trusted.

    This fallback deliberately derives all four sides from the foreground
    pixels. It is not a fixed-width crop and is only accepted for a plausible
    upright board-sized component; otherwise the caller keeps the classifier's
    candidate and marks the case for review.
    """
    mask = _silhouette_mask(array)
    height, width = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if h < height * 0.48 or w < width * 0.12 or w > width * 0.86:
            continue
        if x < width * 0.02 or y < height * 0.01 or x + w > width * 0.98 or y + h > height * 0.99:
            continue
        candidates.append((int(area), index))
    if not candidates:
        return None
    _, index = max(candidates)
    ys, xs = np.where(labels == index)
    if len(xs) < 20:
        return None
    left, right = np.quantile(xs, [0.01, 0.99])
    top, bottom = np.quantile(ys, [0.01, 0.99])
    scale_x = array.shape[1] / width
    scale_y = array.shape[0] / height
    return (
        round(left * scale_x),
        round(top * scale_y),
        max(1, round((right - left) * scale_x)),
        max(1, round((bottom - top) * scale_y)),
    )


def _clean_candidate_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    # Do not let the frame boundary or wall/floor seam become the object.
    mask = mask.copy()
    mask[: max(2, int(height * 0.02)), :] = 0
    mask[int(height * 0.975) :, :] = 0
    mask[:, : max(2, int(width * 0.10))] = 0
    mask[:, int(width * 0.90) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 9), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _best_candidate(mask: np.ndarray, shot_type: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    min_height = height * (0.42 if shot_type == "side_profile" else 0.48)
    max_width_fraction = 0.42 if shot_type == "side_profile" else 0.65
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < min_height or w <= 2:
            continue
        if w / width > max_width_fraction or w / max(h, 1) > (0.75 if shot_type == "side_profile" else 0.85):
            continue
        area_fraction = (w * h) / float(width * height)
        if area_fraction > 0.72:
            continue
        center = (x + w / 2) / width
        center_penalty = abs(center - 0.5)
        height_score = min(1.0, h / height)
        area_score = min(1.0, area_fraction / 0.22)
        score = height_score * 3.0 + area_score - center_penalty * 1.5
        candidates.append((score, (x, y, w, h)))
    if not candidates:
        return None
    candidates.sort(key=lambda value: value[0], reverse=True)
    return candidates[0][1]


def _full_board_crop(x: int, y: int, w: int, h: int, image_width: int, image_height: int, padding: float, rail_padding: float = DEFAULT_RAIL_PADDING) -> tuple[int, int, int, int]:
    """Crop with fixed board-relative margins on all four sides.

    Both settings are proportions of detected board height.  There are no
    image-frame minimums: the board boundary, not the camera frame, controls
    the crop.  Clamping only occurs when the requested crop reaches the source
    edge, and never invents pixels.
    """
    end_pad = min(max(1, round(h * padding)), max(0, y), max(0, image_height - (y + h)))
    rail_pad = min(max(1, round(h * rail_padding)), max(0, x), max(0, image_width - (x + w)))
    center = (x + w / 2) / image_width
    target_width = w + 2 * rail_pad
    target_height = h + 2 * end_pad
    left = round(center * image_width - target_width / 2)
    top = y - end_pad
    return _clamp_crop(left, top, target_width, target_height, image_width, image_height)


def _profile_crop(x: int, y: int, w: int, h: int, image_width: int, image_height: int, padding: float) -> tuple[int, int, int, int]:
    vertical_pad = min(max(1, round(h * padding)), max(0, y), max(0, image_height - (y + h)))
    horizontal_pad = min(max(1, round(h * DEFAULT_HORIZONTAL_PADDING)), max(0, x), max(0, image_width - (x + w)))
    # Center the output on the detected board body. The source frame can be
    # deliberately off-center because of a rack, neighboring board, or floor
    # space; those objects must not steer the listing crop.
    center = (x + w / 2) / image_width
    target_width = w + 2 * horizontal_pad
    target_height = h + 2 * vertical_pad
    left = round(center * image_width - target_width / 2)
    top = y - vertical_pad
    return _clamp_crop(left, top, target_width, target_height, image_width, image_height)


def _profile_body_candidate(array: np.ndarray, candidate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Use the main board body for profile centering, excluding lower fins/rack."""
    result = _grabcut_foreground_mask(array, focused=True)
    if result is None:
        return candidate
    mask, scale = result
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return candidate
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, component_y, _, component_height, _ = stats[index]
    ys, xs = np.where(labels == index)
    keep = (ys >= component_y + component_height * 0.05) & (ys <= component_y + component_height * 0.78)
    if int(keep.sum()) < 20:
        return candidate
    left, right = np.quantile(xs[keep], [0.02, 0.98])
    original_x = round(left / scale)
    original_right = round(right / scale)
    original_y, original_height = candidate[1], candidate[3]
    return original_x, original_y, max(1, original_right - original_x), original_height


def _bright_profile_candidate(array: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a long narrow board edge for side-profile centering.

    The rack and floor are usually dark, while the board edge spans most of
    the frame vertically. This is supporting evidence only; if no stable edge
    is found, the GrabCut candidate remains in charge.
    """
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    best = None
    for threshold in (210, 190, 170, 150):
        mask = np.uint8(gray >= threshold)
        mask[: max(2, round(height * 0.05)), :] = 0
        mask[round(height * 0.80) :, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Frame-edge wall/light is a common false positive. A usable
            # edge-on board must be central enough and materially narrower
            # than the frame; otherwise fall back to the silhouette path.
            if h < height * 0.45 or w > width * 0.16 or w < 8:
                continue
            if x < width * 0.10 or x + w > width * 0.90:
                continue
            score = (h / height) - (w / width) * 0.35
            if best is None or score > best[0]:
                best = (score, (x, y, w, h))
    return best[1] if best else None


def _clamp_crop(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    width = min(width, image_width)
    height = min(height, image_height)
    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))
    return x, y, width, height


def save_crop(source: Path, target: Path, crop: tuple[int, int, int, int]) -> None:
    from .ingest import read_image
    image, _ = read_image(source)
    x, y, width, height = crop
    image.crop((x, y, x + width, y + height)).save(target, "JPEG", quality=95, subsampling=0)
