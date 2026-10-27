"""Field-level board evidence, kept separate from grouping ownership.

Grouping answers "which photos belong together?" This module answers "what
does that board appear to be?" It never turns missing evidence into a
negative and keeps the inventory card as a source, not an unquestioned truth.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .card_ocr import extract_card_identifier, normalize_ocr_text

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


FIELDS = ("brand", "model", "fin_system", "fin_box_count", "fins_included", "fins_present_count")
BOARD_EVIDENCE_VERSION = "0.2-text-patches"
SHOT_FIELD_ELIGIBILITY = {
    "card": set(FIELDS),
    # Full-board/deck views are the useful angle for markings and boxes.
    "full_board": {"brand", "model", "fin_system", "fin_box_count"},
    # A profile can show whether fins are physically installed, but not
    # reliably identify the board's markings or box count.
    "side_profile": {"fins_present_count"},
    "fin_detail": {"fin_system", "fin_box_count", "fins_present_count"},
}
_FIN_SYSTEM_RE = re.compile(r"\b(FCS\s*(?:II|I|2)?|FUTURES?)\b", re.I)
_VISIBLE_FIN_COUNT_RE = re.compile(
    r"\b(?:FINS?|FIN\s+COUNT|NUMBER\s+OF\s+FINS?)\s*(?:PRESENT|VISIBLE|ON\s+BOARD|COUNT)?\s*[:=\-]?\s*([0-9]+)\b",
    re.I,
)


def board_evidence_regions(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None,
) -> dict[str, tuple[int, int, int, int]]:
    """Return board-relative crops for the markings that identify a board.

    These are deliberately overlapping regions. Branding may be near the nose
    or center, while model text and traction frequently sit lower on the deck.
    The coordinates are always clipped to the detected board, never the room.
    """
    if not candidate or len(candidate) != 4:
        return {}
    x, y, width, height = (int(value) for value in candidate)
    if width <= 2 or height <= 2:
        return {}
    regions = {
        "logo_brand": (0.08, 0.02, 0.92, 0.48),
        "model_marking": (0.08, 0.20, 0.92, 0.78),
        "tail_fin": (0.04, 0.62, 0.96, 1.00),
    }
    result = {}
    for name, (left, top, right, bottom) in regions.items():
        px = max(0, min(image.width - 1, x + round(width * left)))
        py = max(0, min(image.height - 1, y + round(height * top)))
        rx = max(px + 1, min(image.width, x + round(width * right)))
        by = max(py + 1, min(image.height, y + round(height * bottom)))
        result[name] = (px, py, rx - px, by - py)
    return result


def _crop_for_evidence(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = box
    crop = image.crop((x, y, x + width, y + height)).convert("RGB")
    # Text on a board is often only a few pixels high in the source frame.
    scale = min(3.0, max(1.0, 1800 / max(crop.width, crop.height, 1)))
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    return crop


def board_text_patch_candidates(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None,
    max_patches: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Find compact text/logo-like patches strictly inside the board box.

    This is a proposal step, not a text detector. MSER is useful here because
    lettering and logos create stable high-contrast connected regions even
    when the board itself is white. Every returned box is clipped to the
    detected board, so room/background pixels cannot become identity evidence.
    """
    if cv2 is None or np is None or not candidate or len(candidate) != 4:
        return []
    x, y, width, height = (int(value) for value in candidate)
    if width <= 4 or height <= 4:
        return []
    board = image.crop((x, y, x + width, y + height)).convert("RGB")
    board.thumbnail((1400, 2200))
    array = np.asarray(board)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    scale_x = width / max(board.width, 1)
    scale_y = height / max(board.height, 1)
    area = float(board.width * board.height)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
    except (AttributeError, cv2.error):
        regions = []
    for points in regions:
        if len(points) < 5:
            continue
        px, py, pw, ph = cv2.boundingRect(points.reshape(-1, 1, 2))
        fraction = (pw * ph) / max(area, 1.0)
        if not 0.00002 <= fraction <= 0.08:
            continue
        if pw < 3 or ph < 3 or pw > board.width * 0.75 or ph > board.height * 0.22:
            continue
        # Expand just enough to capture the complete glyph/logo stroke.
        pad_x, pad_y = max(3, round(pw * 0.35)), max(3, round(ph * 0.55))
        left, top = max(0, px - pad_x), max(0, py - pad_y)
        right, bottom = min(board.width, px + pw + pad_x), min(board.height, py + ph + pad_y)
        patch = (
            max(x, x + round(left * scale_x)),
            max(y, y + round(top * scale_y)),
            max(1, round((right - left) * scale_x)),
            max(1, round((bottom - top) * scale_y)),
        )
        crop = gray[top:bottom, left:right]
        if crop.size == 0:
            continue
        contrast = float(crop.std()) / 255.0
        edges = cv2.Canny(crop, 45, 140)
        edge_density = float(np.mean(edges > 0))
        # Text-like patches are compact and contrasty. Prefer patches away
        # from the extreme board boundary, where rails and shadows dominate.
        interior = 1.0 if 0.03 < (px + pw / 2) / board.width < 0.97 and 0.02 < (py + ph / 2) / board.height < 0.98 else 0.35
        score = (contrast * 0.55 + min(edge_density, 0.45) * 0.45) * interior
        candidates.append((score, patch))
    candidates.sort(key=lambda row: row[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, patch in candidates:
        px, py, pw, ph = patch
        overlaps = False
        for ox, oy, ow, oh in selected:
            intersection = max(0, min(px + pw, ox + ow) - max(px, ox)) * max(0, min(py + ph, oy + oh) - max(py, oy))
            if intersection / max(min(pw * ph, ow * oh), 1) > 0.55:
                overlaps = True
                break
        if not overlaps:
            selected.append(patch)
        if len(selected) >= max_patches:
            break
    return selected


def _patch_metrics(crop: Image.Image, region: str) -> dict[str, float | int | str]:
    """Describe a patch without pretending the heuristic is a detector."""
    if cv2 is None or np is None:
        return {"status": "unavailable", "region": region}
    array = np.asarray(crop.convert("RGB")).copy()
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    # Dark/contrasty marks are useful for comparing the same deck, but are not
    # sufficient to claim a logo or a fin is present.
    dark = gray < max(55, int(float(gray.mean()) * 0.62))
    metrics: dict[str, float | int | str] = {
        "status": "ready",
        "region": region,
        "mean_saturation": round(float(hsv[:, :, 1].mean() / 255.0), 4),
        "mean_value": round(float(hsv[:, :, 2].mean() / 255.0), 4),
        "edge_density": round(float(np.mean(edges > 0)), 4),
        "dark_mark_density": round(float(np.mean(dark)), 4),
    }
    if region == "tail_fin":
        # Candidate count only: components are filtered by geometry and must
        # be reviewed alongside the crop. This avoids asserting "3 fins" from
        # shadows, slots, or graphics.
        contours, _ = cv2.findContours(dark.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        area = max(1, gray.shape[0] * gray.shape[1])
        for contour in contours:
            cx, cy, width, height = cv2.boundingRect(contour)
            fraction = (width * height) / area
            if 0.002 <= fraction <= 0.12 and height >= max(8, width * 0.7):
                candidates.append((cx, cy, width, height))
        metrics["fin_box_candidate_count"] = len(candidates)
        metrics["fin_box_candidates"] = len(candidates)
    return metrics


def extract_board_evidence(
    image: Image.Image,
    candidate: tuple[int, int, int, int] | None,
    shot_type: str,
    ocr_runner=None,
    use_text_patches: bool = False,
) -> dict[str, object]:
    """Extract high-resolution, board-relative evidence for diagnostics.

    ``ocr_runner`` is injected by the ingest layer so this module does not
    own Tesseract configuration. Each crop gets the normal OCR preprocessing
    plus sharpened and high-contrast variants through the runner.
    """
    regions = board_evidence_regions(image, candidate)
    if not regions or shot_type not in {"full_board", "fin_detail"}:
        return {"version": BOARD_EVIDENCE_VERSION, "status": "not_applicable", "regions": {}}
    evidence_regions = {}
    all_text = []
    errors = []
    text_patches = board_text_patch_candidates(image, candidate) if use_text_patches else []
    patch_records = []
    if use_text_patches and ocr_runner is not None:
        for box in text_patches:
            crop = _crop_for_evidence(image, box)
            try:
                text, _, error = ocr_runner(crop)
                if text:
                    all_text.append(text)
                if error:
                    errors.append(f"text_patch: {error}")
                patch_records.append({"box": list(box), "ocr_text": text, "ocr_passes": 1, "status": "error" if error else "ready"})
            except Exception as exc:
                errors.append(f"text_patch: {type(exc).__name__}: {exc}")
                patch_records.append({"box": list(box), "ocr_text": "", "ocr_passes": 1, "status": "error"})
    for name, box in regions.items():
        crop = _crop_for_evidence(image, box)
        texts = []
        pass_count = 0
        if ocr_runner is not None and not use_text_patches:
            variants = [crop, ImageOps.autocontrast(ImageOps.grayscale(crop)), ImageEnhance.Contrast(crop).enhance(1.8).filter(ImageFilter.SHARPEN)]
            for variant in variants:
                pass_count += 1
                try:
                    text, _, error = ocr_runner(variant)
                    if text:
                        texts.append(text)
                    if error:
                        errors.append(f"{name}: {error}")
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        text = max(texts, key=lambda value: len(re.findall(r"[A-Za-z0-9]", value)), default="")
        if not use_text_patches:
            all_text.append(text)
        evidence_regions[name] = {
            "box": list(box),
            "ocr_text": text,
            "ocr_passes": pass_count,
            "metrics": _patch_metrics(crop, name),
        }
    merged_text = "\n".join(value for value in all_text if value)
    return {
        "version": BOARD_EVIDENCE_VERSION,
        "status": "ready",
        "regions": evidence_regions,
        "ocr_text": merged_text,
        "ocr_passes": len(patch_records) if use_text_patches else sum(int(item["ocr_passes"]) for item in evidence_regions.values()),
        "text_patches": patch_records,
        "errors": errors[:10],
    }


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _matches(expected: str, text: str) -> bool:
    expected_key = _compact(expected)
    text_key = _compact(text)
    if not expected_key or not text_key:
        return False
    if expected_key in text_key:
        return True
    return SequenceMatcher(None, expected_key, text_key).ratio() >= 0.86


def extract_photo_evidence(text: str, shot_type: str = "") -> dict[str, dict]:
    """Extract only explicit text evidence from a board photo.

    A fin/profile frame is useful context, but its mere existence does not
    prove that fins are included. That distinction is intentional.
    """
    identifier = extract_card_identifier(text) if text else {}
    fin_match = _FIN_SYSTEM_RE.search(normalize_ocr_text(text)) if text else None
    if fin_match and "fin_system" not in identifier:
        identifier["fin_system"] = "FCS II" if fin_match.group(1).upper().startswith("FCS") else "Futures"
    evidence = {}
    eligible = SHOT_FIELD_ELIGIBILITY.get(shot_type, set())
    for field, value in identifier.items():
        if field in eligible and value:
            evidence[field] = {"value": value, "source": "photo_ocr", "shot_type": shot_type, "confidence": 0.62}
    if text and "fins_present_count" in eligible:
        visible_match = _VISIBLE_FIN_COUNT_RE.search(normalize_ocr_text(text))
        if visible_match:
            evidence["fins_present_count"] = {
                "value": visible_match.group(1),
                "source": "photo_ocr",
                "shot_type": shot_type,
                "confidence": 0.62,
            }
    return evidence


def evidence_fields_for_shot(shot_type: str) -> set[str]:
    """Return fields this angle is allowed to assert."""
    return set(SHOT_FIELD_ELIGIBILITY.get(shot_type, set()))


def detect_side_profile_fins(image: Image.Image, candidate: tuple[int, int, int, int] | None = None) -> dict[str, object]:
    """Keep side-profile fin count unknown until actual fins are visible.

    A side silhouette, rail, rack, shadow, or board graphic cannot reliably
    establish either the number of fin boxes or the number of installed fins.
    The previous contour heuristic converted those shapes into false
    ``fins yes`` evidence, so this angle is now explicitly non-authoritative.
    """
    return {
        "value": "",
        "status": "unknown",
        "confidence": 0.0,
        "source": "side_profile_visual",
        "reason": "side_profile_cannot_count_installed_fins",
    }


def _photo_text(item: dict) -> str:
    return str(item.get("photo_ocr_text") or item.get("ocr") or "")


def board_evidence_diagnostics(groups: list[list[dict]]) -> list[dict]:
    """Compare card fields with all grouped photo evidence."""
    reports = []
    for number, group in enumerate(groups, start=1):
        card = next((item for item in group if item.get("is_card") or item.get("source_is_card")), None)
        card_identifier = dict((card or {}).get("identifier") or {})
        fields = {}
        for field in FIELDS:
            expected = str(card_identifier.get(field, ""))
            direct_values = []
            matched_photos = []
            conflicting_photos = []
            for item in group:
                if item.get("is_card") or item.get("source_is_card"):
                    continue
                shot_type = str(item.get("shot_type", "")) or "full_board"
                if field not in evidence_fields_for_shot(shot_type):
                    continue
                evidence = item.get("identity_evidence") or {}
                direct = evidence.get(field, {}).get("value", "")
                if direct:
                    direct_values.append((str(direct), str(item.get("path", "")), "explicit"))
                text = _photo_text(item)
                if expected and _matches(expected, text):
                    matched_photos.append(str(item.get("path", "")))
                if expected and direct and not _matches(expected, direct):
                    conflicting_photos.append({"source": str(item.get("path", "")), "value": direct})
            if expected:
                if conflicting_photos:
                    status = "conflicting"
                elif matched_photos or any(_matches(expected, value) for value, _, _ in direct_values):
                    status = "confirmed"
                else:
                    status = "card_only"
            elif direct_values:
                winner, count = Counter(value for value, _, _ in direct_values).most_common(1)[0]
                status = "photo_consensus" if count >= 2 else "needs_review"
                expected = winner
            else:
                status = "unknown"
            fields[field] = {
                "value": expected,
                "status": status,
                "card_value": str(card_identifier.get(field, "")),
                "photo_matches": matched_photos,
                "photo_values": [{"value": value, "source": source} for value, source, _ in direct_values],
                "conflicts": conflicting_photos,
            }
        reports.append({
            "group_number": number,
            "card_source": str(card.get("path", "")) if card else None,
            "card_identifier": card_identifier,
            "photo_evidence": [{
                "source": str(item.get("path", "")),
                "shot_type": item.get("shot_type", ""),
                "ocr_status": item.get("photo_ocr_status", "not_run"),
                "ocr_error": item.get("photo_ocr_error", ""),
                "ocr_text": _photo_text(item),
                "board_evidence": item.get("board_evidence") or {},
                "eligible_fields": sorted(evidence_fields_for_shot(str(item.get("shot_type", "")) or "full_board")),
                "evidence": item.get("identity_evidence") or {},
            } for item in group if not item.get("is_card")],
            "fields": fields,
        })
    return reports
