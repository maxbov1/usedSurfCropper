#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv-sam2/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv-sam2/bin/python"
else
  echo "Creating the local UsedSurf Python environment…"
  python3 -m venv "$ROOT_DIR/.venv"
  PYTHON="$ROOT_DIR/.venv/bin/python"
fi

if ! "$PYTHON" -c 'import pytesseract' >/dev/null 2>&1; then
  echo "Installing OCR and runtime dependencies…"
  "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi
if ! command -v tesseract >/dev/null 2>&1 && command -v brew >/dev/null 2>&1; then
  echo "Installing the native Tesseract OCR engine…"
  brew install tesseract
fi

"$ROOT_DIR/doctor.sh"
cd "$ROOT_DIR"
exec "$PYTHON" app.py "$@"
