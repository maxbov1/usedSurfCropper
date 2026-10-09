import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    configured = os.environ.get("USED_SURF_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path.home() / "Library" / "Application Support" / "UsedSurf"
    return project_root()


def model_path(root: Path) -> Path:
    configured = os.environ.get("USED_SURF_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    external = root / "models" / "yolo11n.pt"
    return external if external.exists() else root / "yolo11n.pt"


def paths(root: Path) -> dict[str, Path]:
    result = {name: root / name for name in ("input", "output", "data")}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result
