"""Optional local SAM 2 segmentation backend.

SAM 2 is deliberately lazy-loaded. The base app remains runnable without
PyTorch, SAM 2, or a checkpoint; callers receive an explicit unavailable
status instead of a fake crop.
"""

from __future__ import annotations

import os
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

    def predict(self, image: Image.Image, box: tuple[float, float, float, float]) -> SegmentationResult:
        predictor = self._load()
        if predictor is None:
            return SegmentationResult(None, None, "sam2_unavailable", self._load_error or "unknown load error")
        try:
            array = np.asarray(image.convert("RGB"))
            predictor.set_image(array)
            masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
            best = int(np.argmax(scores))
            return SegmentationResult(masks[best].astype(bool), float(scores[best]), "sam2", "")
        except Exception as exc:
            return SegmentationResult(None, None, "sam2_error", str(exc))


_SEGMENTER: Sam2Segmenter | None = None


def get_segmenter() -> Sam2Segmenter | None:
    global _SEGMENTER
    checkpoint = os.environ.get("USEDSURF_SAM2_CHECKPOINT")
    config = os.environ.get("USEDSURF_SAM2_CONFIG")
    if not checkpoint or not config:
        return None
    if _SEGMENTER is None or str(_SEGMENTER.checkpoint) != checkpoint or _SEGMENTER.config != config:
        _SEGMENTER = Sam2Segmenter(Path(checkpoint), config)
    return _SEGMENTER


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
    pad_x = round((bottom - y) * padding)
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
