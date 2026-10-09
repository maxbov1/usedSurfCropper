"""Small, optional YOLO board detector used as coarse evidence for cropping.

YOLO is deliberately not the cropper. Its COCO ``surfboard`` box is usually
too coarse for product framing, so OpenCV still chooses the final silhouette
and padding. The box is useful for constraining that search away from racks,
walls, and other high-contrast objects.
"""

from __future__ import annotations

from typing import Any

from .config import model_path, runtime_root

_MODEL = None
_MODEL_ERROR: str | None = None
_DEVICE: str | None = None
PROFILE_ROI_SCALE = 2.5
PROFILE_ROI_PADDING = 0.20


def _device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_built() and mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def _load_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        return None
    try:
        from ultralytics import YOLO

        # This is intentionally the small COCO detector requested for the POC.
        # Ultralytics downloads yolo11n.pt once, then keeps inference local.
        configured_model = model_path(runtime_root())
        # Keep Ultralytics' normal first-run downloader for development. A
        # packaged app supplies the copied model at an explicit external path.
        model = YOLO(str(configured_model) if configured_model.exists() else "yolo11n.pt")
        model.to(_device())
        _MODEL = model
        return _MODEL
    except Exception as exc:  # OpenCV remains a working fallback.
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _surfboard_ids(model: Any) -> list[int]:
    names = getattr(model, "names", {})
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [int(index) for index, name in items if str(name).lower() == "surfboard"]


def detect_board(
    image: Any,
    confidence: float = 0.25,
    roi: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    """Return the best COCO surfboard box in source-image pixels, if any.

    Profile mode can run on an upscaled, padded ROI. The returned coordinates
    are always mapped back to the original image, so downstream crop code does
    not need a second coordinate system.
    """
    model = _load_model()
    if model is None:
        return None
    class_ids = _surfboard_ids(model)
    if not class_ids:
        return None
    try:
        source = image
        offset_x = offset_y = 0
        if roi is not None:
            image_width, image_height = image.size
            roi_x, roi_y, roi_width, roi_height = [int(value) for value in roi]
            pad_x = max(16, round(roi_width * PROFILE_ROI_PADDING))
            pad_y = max(16, round(roi_height * PROFILE_ROI_PADDING))
            x0 = max(0, roi_x - pad_x)
            y0 = max(0, roi_y - pad_y)
            x1 = min(image_width, roi_x + roi_width + pad_x)
            y1 = min(image_height, roi_y + roi_height + pad_y)
            source = image.crop((x0, y0, x1, y1))
            offset_x, offset_y = x0, y0
            if scale != 1.0:
                source = source.resize((round(source.width * scale), round(source.height * scale)))
        results = model.predict(
            source=source,
            device=_device(),
            classes=class_ids,
            conf=confidence,
            verbose=False,
        )
        best: dict[str, Any] | None = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            for coords, score in zip(xyxy, scores):
                x1, y1, x2, y2 = [int(round(value / scale)) for value in coords]
                candidate = {
                    "x": max(0, offset_x + x1),
                    "y": max(0, offset_y + y1),
                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1),
                    "confidence": round(float(score), 4),
                    "inference_confidence": confidence,
                    "device": _device(),
                    "model": "yolo11n.pt",
                    "inference_roi": roi,
                    "inference_scale": scale,
                }
                if candidate["width"] > 0 and candidate["height"] > 0 and (best is None or candidate["confidence"] > best["confidence"]):
                    best = candidate
        return best
    except Exception:
        return None


def detector_status() -> dict[str, Any]:
    """Small diagnostic payload for the UI and test reports."""
    return {"model": "yolo11n.pt", "device": _device(), "loaded": _MODEL is not None, "error": _MODEL_ERROR}
