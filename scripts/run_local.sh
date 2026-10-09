#!/bin/zsh
set -e

cd "${0:A:h}/.."
export USED_SURF_PORT="${USED_SURF_PORT:-5051}"

# Keep the Mac awake while local OCR/cropping work is running.
exec caffeinate -dimsu .venv-sam2/bin/python app.py
