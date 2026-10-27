"""Optional local SAM 2 segmentation backend.

SAM 2 is deliberately lazy-loaded. The base app remains runnable without
PyTorch, SAM 2, or a checkpoint; callers receive an explicit unavailable
status instead of a fake crop.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:
    mask: np.ndarray | None
    score: float | None
    method: str
    reason: str = ""


class Sam2Segmenter:
    def __init__(self, checkpoint: Path, config: str, device: str | None = None):
        self.checkpoint = checkpoint
        self.config = config
        self.device = device or _device()
        self._predictor = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.is_file():
            self._load_error = f"checkpoint not found: {self.checkpoint}"
            return None
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self.config, str(self.checkpoint), device=self.device, apply_postprocessing=False)
            self._predictor = SAM2ImagePredictor(model)
            return self._predictor
        except Exception as exc:  # Import/build errors are surfaced in review metadata.
            self._load_error = f"SAM 2 unavailable: {exc}"
            return None

    def predict(
        self,
        image: Image.Image,
        box: tuple[float, float, float, float],
        points: tuple[list[tuple[float, float]], list[tuple[float, float]]] | None = None,
    ) -> SegmentationResult:
        """``points`` is (foreground, background) image-coordinate pairs.

        A box alone tells SAM 2 roughly where to look but not which pixels
        inside it are board vs. background - measured on a real batch, SAM
        2's mask disagreed with the same box's own detector on 41% of
        side_profile photos even with a tight YOLO box prompt. Foreground
        points on the board's centerline plus background points just
        outside the box's left/right edges give SAM 2 an explicit answer
        instead of making it guess.
        """
        predictor = self._load()
        if predictor is None:
            return SegmentationResult(None, None, "sam2_unavailable", self._load_error or "unknown load error")
        try:
            # np.asarray(PIL.Image) may expose a read-only decoder buffer;
            # SAM 2's internal torchvision preprocessing needs a writable
            # array (PyTorch warns of undefined behavior otherwise).
            array = np.asarray(image.convert("RGB")).copy()
            predictor.set_image(array)
            point_coords, point_labels = _point_prompt_arrays(points) if points is not None else (None, None)
            masks, scores, _ = predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            return SegmentationResult(masks[best].astype(bool), float(scores[best]), "sam2", "")
        except Exception as exc:
            return SegmentationResult(None, None, "sam2_error", str(exc))

    def embed(self, image: Image.Image) -> np.ndarray | None:
        """Return a single pooled feature vector describing the whole image.

        Reuses SAM 2's own already-loaded image encoder — no extra model or
        dependency. This is deliberately not board-masked or trained for
        identity matching; it is only ever used as a tie-breaker to locate
        a likely missed-card boundary inside an already-oversized group
        (see ``locate_missed_card_boundary`` in ``organizer.ingest``), not
        to decide board identity on its own.
        """
        predictor = self._load()
        if predictor is None:
            return None
        try:
            # np.asarray(PIL.Image) may expose a read-only decoder buffer;
            # SAM 2's internal torchvision preprocessing needs a writable
            # array (PyTorch warns of undefined behavior otherwise).
            array = np.asarray(image.convert("RGB")).copy()
            predictor.set_image(array)
            embedding = predictor.get_image_embedding()
            vector = embedding.mean(dim=(-2, -1)).flatten().detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            return vector
        except Exception:
            return None


def _centerline_points(
    box: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Foreground points along a box's centerline, background just outside its edges.

    ``box`` is a confident external detection (YOLO), trusted for *where*
    the board's rails are even though the mask prompted from it alone can
    still drift - see Sam2Segmenter.predict()'s docstring. Background
    points sit just past the box's left/right edges specifically because
    the measured failure mode is horizontal bleed into the wall, not
    vertical bleed past the nose/tail.
    """
    x1, y1, x2, y2 = box
    width, height = image_size
    center_x = (x1 + x2) / 2
    foreground = [(center_x, y1 + (y2 - y1) * fraction) for fraction in (0.15, 0.5, 0.85)]
    margin = max(4.0, (x2 - x1) * 0.04)
    mid_y = (y1 + y2) / 2
    background = [
        (max(0.0, x1 - margin), mid_y),
        (min(float(width - 1), x2 + margin), mid_y),
    ]
    return foreground, background


def _point_prompt_arrays(
    points: tuple[list[tuple[float, float]], list[tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    foreground, background = points
    coords = [*foreground, *background]
    if not coords:
        return None, None
    labels = [1] * len(foreground) + [0] * len(background)
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


_SEGMENTER: Sam2Segmenter | None = None

# Below this score a mask tends to bleed into the wall/background rather
# than tracking the board (measured on a real 65-photo batch: scores under
# ~0.72 produced raw mask boxes covering 59-67% of the frame vs. the normal
# 20-50%). Shared so grouping and cropping apply the same bar to the same
# mask instead of computing it twice with different standards.
SAM2_MASK_MIN_SCORE = 0.70


def get_segmenter() -> Sam2Segmenter | None:
    global _SEGMENTER
    checkpoint = os.environ.get("USEDSURF_SAM2_CHECKPOINT")
    config = os.environ.get("USEDSURF_SAM2_CONFIG")
    if not checkpoint or not config:
        return None
    if _SEGMENTER is None or str(_SEGMENTER.checkpoint) != checkpoint or _SEGMENTER.config != config:
        _SEGMENTER = Sam2Segmenter(Path(checkpoint), config)
    return _SEGMENTER


# Aspect ratio (width/height) bounds per shot type. Deliberately wider than
# the measured real-data cluster below, so normal board-to-board variation
# (a wider fish shape, a narrower shortboard) is never falsely rejected -
# these only catch shapes that couldn't plausibly be the named shot type.
#   full_board measured across a real 65-photo batch: 0.24-0.46 typical.
#   side_profile measured across 14 real photos: 0.07-0.30 typical, one
#   real failure case at 0.67 (SAM 2 bled to near-full-frame width) that
#   these bounds exist specifically to catch.
_ASPECT_RATIO_BOUNDS = {"full_board": (0.15, 0.60), "side_profile": (0.03, 0.40)}
_MIN_FRAME_FRACTION = 0.02
_MAX_FRAME_FRACTION = 0.85
# A sibling's width must fall within this fraction of the group's median
# width to count as consistent. Measured on a real board: an outlier at
# 65% of its siblings' median width was a genuine detection failure (SAM 2
# and YOLO both found only the right ~60% of the board); every correct
# photo in every board measured so far sits within 10-15% of its siblings.
_SIBLING_WIDTH_TOLERANCE = (0.7, 1.4)


def validate_board_box(
    box: tuple[float, float, float, float] | None,
    shot_type: str,
    image_size: tuple[int, int],
    sibling_boxes: list[tuple[float, float, float, float]] | None = None,
) -> bool:
    """Decide whether a detected board box actually looks like a real board.

    This is the one place that answers that question, regardless of which
    detector (SAM 2, YOLO, self-refined or not) produced the box and
    regardless of which code path is asking. It replaces what used to be
    four separate, duplicated, differently-thresholded checks scattered
    across segmentation.py, crop.py, and run_batch_crop.py - a real failure
    mode in itself, since the duplicates didn't stay in sync.

    Two layers, both grounded in measured real data, not guesses:
      - Absolute shape: aspect ratio and frame-fraction bounds per shot
        type (see the module-level constants above).
      - Sibling consistency, only when ``sibling_boxes`` (other boxes of
        the same shot type from the same board) are supplied: this is the
        layer that catches a box that looks fine in isolation but doesn't
        match its own board's other photos - the harder, more common real
        failure (one real case had an aspect ratio just inside the normal
        full_board range on its own, but was 35% narrower than every one
        of its siblings).
    """
    if not box or len(box) < 4:
        return False
    _, _, box_width, box_height = box[:4]
    if box_width <= 0 or box_height <= 0:
        return False
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return False
    ratio_bounds = _ASPECT_RATIO_BOUNDS.get(shot_type)
    if ratio_bounds is not None:
        ratio = box_width / box_height
        if not (ratio_bounds[0] <= ratio <= ratio_bounds[1]):
            return False
    frame_fraction = (box_width * box_height) / (image_width * image_height)
    if not (_MIN_FRAME_FRACTION <= frame_fraction <= _MAX_FRAME_FRACTION):
        return False
    if sibling_boxes:
        sibling_widths = [sb[2] for sb in sibling_boxes if sb and len(sb) >= 3 and sb[2] > 0]
        if len(sibling_widths) >= 2:  # too few siblings to trust a median over real variation
            median_width = statistics.median(sibling_widths)
            if median_width > 0 and not (_SIBLING_WIDTH_TOLERANCE[0] * median_width <= box_width <= _SIBLING_WIDTH_TOLERANCE[1] * median_width):
                return False
    return True


def board_perimeter(image: Image.Image, shot_type: str, prompt_box: tuple[float, float, float, float] | None = None) -> dict | None:
    """Run SAM 2 once and return its mask plus a raw (unpadded) bounding box.

    This is the single board-boundary computation both grouping (board
    identity-signature diagnostics) and cropping (the final listing crop)
    should use, so SAM 2 only runs once per photo across the whole
    pipeline instead of each phase predicting its own mask independently.
    Returns None when SAM 2 isn't configured.

    ``prompt_box`` lets a caller hand SAM 2 a tight, already-localized region
    (a confident YOLO detection, as (x1, y1, x2, y2)) instead of the generic
    ``board_prompt_box`` region. Measured on a real batch: with the generic
    prompt SAM 2's mask regularly bled several hundred pixels into the wall
    on one side even at a high score; a tight YOLO box as the prompt keeps
    the mask anchored to the actual board instead of searching the whole
    controlled-setup region.
    """
    segmenter = get_segmenter()
    if segmenter is None:
        return None
    width, height = image.size
    yolo_guided = prompt_box is not None
    box = prompt_box if prompt_box is not None else board_prompt_box(width, height, shot_type)
    points = None
    if yolo_guided and shot_type == "side_profile":
        # Measured on real photos (compare_sam2_checkpoints.py): foreground
        # points alone raised mean IoU-vs-YOLO on side_profile (0.761 ->
        # 0.777, 17 photos) with every score staying above
        # SAM2_MASK_MIN_SCORE. Adding background points on top of them
        # looked like it should help more but measured worse on both axes
        # (lower IoU, and confidence dropped enough to cross the score gate
        # on the one case it fixed best) - foreground-only is not a
        # simplification, it's the one that actually won. full_board was
        # already at 0.95+ IoU with no headroom to gain and real downside
        # (confidence cratered on several already-correct photos, e.g.
        # IMG_6419 0.99 -> 0.63), so this stays side_profile-only.
        foreground, _ = _centerline_points(box, (width, height))
        points = (foreground, [])
    result = segmenter.predict(image, box, points=points)
    self_refined = False
    if not yolo_guided and result.mask is not None:
        # No confident external detector to anchor the prompt, so this pass
        # used the generic wide search region - which gives SAM 2 room to
        # bleed into the wall (measured on a real side-profile photo: the
        # mask overshot the true board's right edge by 587px this way,
        # even though it found the left edge correctly). Re-prompt with
        # this pass's own tight bbox, same principle as YOLO-guidance, just
        # self-supplied when nothing external is usable. This is the
        # computation grouping persists and cropping reuses, so refining it
        # here - not just in crop.py's live fallback, which calls this same
        # function rather than duplicating it - is what reaches most photos.
        first_pass_box = mask_crop(result.mask, width, height, padding=0.0)
        if first_pass_box is not None:
            rx, ry, rw, rh = first_pass_box
            refined = segmenter.predict(image, (float(rx), float(ry), float(rx + rw), float(ry + rh)))
            if refined.mask is not None:
                result = refined
                self_refined = True
    perimeter_box = mask_crop(result.mask, width, height, padding=0.0) if result.mask is not None else None
    # Single-image plausibility only - no sibling data exists yet at this
    # point (grouping processes photos individually; a board's full set of
    # siblings isn't known until all of them have been classified). The
    # sibling-aware layer runs later, in run_batch_crop.py, once a board's
    # complete sibling set is available from the database.
    if perimeter_box is not None and not validate_board_box(perimeter_box, shot_type, (width, height)):
        perimeter_box = None
    return {"mask": result.mask, "box": perimeter_box, "score": result.score, "self_refined": self_refined}


def board_prompt_box(width: int, height: int, shot_type: str) -> tuple[float, float, float, float]:
    """Initial prompt for the controlled upright setup; it is not a crop."""
    if shot_type in {"rail_full_light", "side_profile"}:
        return width * 0.20, height * 0.05, width * 0.80, height * 0.95
    return width * 0.15, height * 0.03, width * 0.85, height * 0.97


def mask_crop(mask: np.ndarray, width: int, height: int, padding: float = 0.05) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) < max(100, int(width * height * 0.002)):
        return None
    x, right = int(xs.min()), int(xs.max()) + 1
    y, bottom = int(ys.min()), int(ys.max()) + 1
    box_area = (right - x) * (bottom - y)
    if box_area / float(width * height) > 0.90:
        return None
    pad_x = round((right - x) * padding)
    pad_y = round((bottom - y) * padding)
    x = max(0, x - pad_x); y = max(0, y - pad_y)
    right = min(width, right + pad_x); bottom = min(height, bottom + pad_y)
    return x, y, right - x, bottom - y


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
