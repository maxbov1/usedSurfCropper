from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from .card_ocr import extract_card_identifier

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
PROCESSING_VERSION = "0.10.0-board-similarity-grouping"
MAX_PHOTOS_PER_BOARD = 7
STANDARD_BOARD_PHOTOS = 6
SHOT_LIMITS = {"full_board": 4, "side_profile": 1, "fin_detail": 1, "card": 1}
BOARD_SIMILARITY_BOUNDARY = 0.10


def visual_signature(image: Image.Image) -> tuple[float, ...]:
    """Return a small appearance signature for grouping support.

    This is deliberately not a board recognizer. In this controlled room the
    center of the frame carries enough board color/shape information to catch
    an obvious board change while remaining tolerant of light versus low-light
    paired views. It is only used as supporting evidence around timestamp/card
    boundaries.
    """
    if cv2 is None or np is None:
        return ()
    working = image.copy()
    working.thumbnail((480, 640))
    array = np.asarray(working.convert("RGB"))
    height, width = array.shape[:2]
    roi = array[round(height * 0.08):round(height * 0.92), round(width * 0.16):round(width * 0.84)]
    if roi.size == 0:
        return ()
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    # Normalize luminance so the light/low-light pair does not look like two
    # boards, but retain chroma because different board colors are meaningful.
    lab[:, :, 0] = (lab[:, :, 0] - float(lab[:, :, 0].mean())) / max(float(lab[:, :, 0].std()), 0.04)
    small = cv2.resize(lab, (12, 18), interpolation=cv2.INTER_AREA)
    summary = np.concatenate([
        lab.mean(axis=(0, 1)),
        lab.std(axis=(0, 1)),
        small[:, :, 1:].reshape(-1),
    ])
    return tuple(float(value) for value in summary)


def visual_distance(left: dict, right: dict) -> float:
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


def scan_files(input_dir: Path) -> list[dict]:
    found = []
    ocr_status = ocr_runtime_status()
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            image, capture = read_image(path)
            candidate = card_candidate(image)
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
            likely_card = candidate_is_card or yellow_score >= 0.12
            if likely_card:
                text, identifier, ocr_error = local_ocr(image, region)
                per_file_ocr_status = "read" if identifier else ("empty" if not ocr_error else "error")
            else:
                text, identifier, ocr_error = "", {}, ""
                per_file_ocr_status = "not_run"
            found.append({
                "path": path, "hash": sha256(path), "capture": capture, "width": image.width,
                "height": image.height, "ocr": text, "identifier": identifier,
                "card_candidate": candidate[:4] if candidate else None,
                "card_confidence": candidate[4] if candidate else 0.0,
                "yellow_card_score": yellow_score,
                "ocr_status": per_file_ocr_status,
                "ocr_error": ocr_error,
                "ocr_runtime": ocr_status,
                "visual_signature": visual_signature(image),
                "is_card": card_signal(text, identifier) or candidate_is_card or yellow_score >= 0.35,
            })
        except Exception as exc:
            found.append({"path": path, "error": str(exc), "capture": None, "width": None, "height": None, "ocr": "", "identifier": {}, "ocr_status": "scan_error", "ocr_error": "", "ocr_runtime": ocr_status, "is_card": False, "visual_signature": ()})
    # Stable tie-breaker is filename, but capture time remains primary.
    return sorted(found, key=lambda item: (item["capture"] is None, item["capture"] or "", item["path"].name.lower()))


def group_scanned(files: list[dict]) -> list[list[dict]]:
    """Group a capture-ordered batch using the known capture protocol.

    A normal board session is six board views, optionally accompanied by one
    identity card and one fin/detail view. A card belongs to the contiguous
    session in which it was photographed; the next card closes that session.
    Visual/ML signals do not choose an arbitrary timestamp-based cut point.
    """
    if not files:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    for item in files:
        has_card = any(photo.get("is_card") for photo in current)
        card_closes_current = bool(current) and item.get("is_card") and not has_card
        # A second card is the next session marker only after the current
        # session already has its identity card. A late card therefore joins
        # the preceding board instead of becoming a singleton group.
        if current and item.get("is_card") and has_card:
            groups.append(current)
            current = []
            has_card = False
        # Six board views is the standard cardless session. A card arriving at
        # this boundary is still allowed to join it as its identity anchor;
        # another board view starts the next session.
        elif current and not has_card and len(current) >= STANDARD_BOARD_PHOTOS and not item.get("is_card"):
            groups.append(current)
            current = []
        if current and len(current) >= MAX_PHOTOS_PER_BOARD:
            groups.append(current)
            current = []
        current.append(item)
        if card_closes_current:
            groups.append(current)
            current = []
            continue
        if len(current) >= MAX_PHOTOS_PER_BOARD:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    visually_grouped = _split_by_board_similarity([group for group in groups if group])
    return _enforce_shot_limits(visually_grouped)


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
        boundaries = [
            index
            for index in range(1, len(photos))
            if index >= 2
            and len(photos) - index >= 2
            and visual_distance(photos[index - 1], photos[index]) >= BOARD_SIMILARITY_BOUNDARY
        ]
        if not boundaries:
            partitioned.append(group)
            continue
        clusters: list[list[dict]] = []
        start = 0
        for boundary in boundaries:
            clusters.append(photos[start:boundary])
            start = boundary
        clusters.append(photos[start:])
        card = cards[0] if cards else None
        if card:
            first_photo_index = next(index for index, item in enumerate(group) if not item.get("is_card"))
            last_photo_index = max(index for index, item in enumerate(group) if not item.get("is_card"))
            card_index = group.index(card)
            if card_index <= first_photo_index:
                card_cluster = 0
            elif card_index >= last_photo_index:
                card_cluster = len(clusters) - 1
            else:
                before = card_visual_affinity(card, clusters[0])
                after = card_visual_affinity(card, clusters[-1])
                card_cluster = 0 if before is not None and (after is None or before <= after) else len(clusters) - 1
            clusters[card_cluster].append(card)
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
