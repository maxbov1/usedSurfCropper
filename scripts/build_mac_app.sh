#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" && -x "$ROOT_DIR/.venv-build/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv-build/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
APP_ARCH="${APP_ARCH:-$(uname -m)}"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$ROOT_DIR/.pyinstaller}"
mkdir -p "$PYINSTALLER_CONFIG_DIR"
MODEL_PATH="$ROOT_DIR/yolo11n.pt"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Downloading YOLO11 Nano weights for the bundle..."
  "$PYTHON" -c 'from ultralytics import YOLO; YOLO("yolo11n.pt")'
fi

rm -rf "$ROOT_DIR/build" "$DIST_DIR"
"$PYTHON" -m PyInstaller --noconfirm --clean \
  --workpath "$ROOT_DIR/build" \
  --distpath "$DIST_DIR" \
  "$ROOT_DIR/packaging/UsedSurf.spec"

mkdir -p "$DIST_DIR"
ditto -c -k --sequesterRsrc --keepParent "$DIST_DIR/UsedSurf.app" "$DIST_DIR/UsedSurf-macOS-${APP_ARCH}.zip"
echo "Built: $DIST_DIR/UsedSurf.app"
echo "Archive: $DIST_DIR/UsedSurf-macOS-${APP_ARCH}.zip"
