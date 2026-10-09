#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]] && "$ROOT_DIR/.venv/bin/python" -c 'import pytesseract' >/dev/null 2>&1; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv-sam2/bin/python" ]] && "$ROOT_DIR/.venv-sam2/bin/python" -c 'import pytesseract' >/dev/null 2>&1; then
  PYTHON="$ROOT_DIR/.venv-sam2/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
exec "$PYTHON" "$ROOT_DIR/scripts/doctor.py" "$@"
