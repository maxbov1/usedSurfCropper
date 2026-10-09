#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${USED_SURF_REPO_URL:-https://github.com/maxbov1/usedSurfCropper.git}"
APP_NAME="UsedSurf.app"
INSTALL_DIR="${USED_SURF_INSTALL_DIR:-$HOME/Applications/$APP_NAME}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/usedsurf-install.XXXXXX")"
REPO_DIR="$TMP_ROOT/usedSurfCropper"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "UsedSurf's one-command installer only supports macOS." >&2
  exit 1
fi

for command_name in git python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    echo "Install Xcode Command Line Tools, then run this installer again." >&2
    exit 1
  fi
done

if ! command -v tesseract >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing the native Tesseract OCR engine…"
    brew install tesseract
  else
    echo "Warning: Homebrew/Tesseract is not installed. OCR card fields will be unavailable." >&2
    echo "Install Homebrew from https://brew.sh, then run this installer again." >&2
  fi
fi
if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract OCR is required for card identity extraction. Install Homebrew and rerun this installer." >&2
  exit 1
fi

echo "Downloading UsedSurf…"
git clone --depth 1 "$REPO_URL" "$REPO_DIR" >/dev/null
cd "$REPO_DIR"

echo "Installing local build dependencies…"
python3 -m venv .venv-build
.venv-build/bin/python -m pip install --upgrade pip >/dev/null
.venv-build/bin/python -m pip install -r requirements.txt -r requirements-build.txt

echo "Building the UsedSurf app for $(uname -m)…"
bash scripts/build_mac_app.sh

echo "Installing $APP_NAME in $INSTALL_DIR…"
mkdir -p "$(dirname "$INSTALL_DIR")"
if [[ -e "$INSTALL_DIR" ]]; then
  rm -rf "$INSTALL_DIR"
fi
ditto --rsrc --extattr "$REPO_DIR/dist/$APP_NAME" "$INSTALL_DIR"

# Remove the quarantine bit when present so Finder can launch this locally-built
# app without treating it as an untrusted downloaded bundle.
xattr -dr com.apple.quarantine "$INSTALL_DIR" 2>/dev/null || true

echo "Installed: $INSTALL_DIR"
echo "Your photos and review history will live in ~/Library/Application Support/UsedSurf."
open -R "$INSTALL_DIR"
open "$INSTALL_DIR"
