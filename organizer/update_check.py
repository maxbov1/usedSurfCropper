"""Best-effort GitHub update information for the local app."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
import sys


DEFAULT_REPOSITORY = "maxbov1/usedSurfCropper"


def _write_status(root: Path, payload: dict) -> None:
    try:
        status_path = root / "data" / "update-status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError:
        # Update checking is diagnostic only. A read-only or restricted data
        # directory must never prevent the local cropper from starting.
        return


def bundled_commit() -> str:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    info_path = bundle_root / "build-info.json"
    try:
        return json.loads(info_path.read_text()).get("commit", "")
    except (OSError, json.JSONDecodeError):
        return ""


def check_github_async(root: Path, on_update=None) -> None:
    repository = os.environ.get("USED_SURF_GITHUB_REPO", DEFAULT_REPOSITORY).strip().strip("/")
    if not repository or "/" not in repository:
        _write_status(root, {"status": "not_configured", "checked_at": datetime.now().isoformat(timespec="seconds")})
        return

    def check() -> None:
        url = f"https://api.github.com/repos/{repository}/commits/HEAD"
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "UsedSurf"})
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            latest_sha = payload.get("sha", "")
            current_sha = bundled_commit()
            update_available = bool(current_sha and latest_sha and not latest_sha.startswith(current_sha))
            _write_status(root, {
                "status": "update_available" if update_available else "up_to_date",
                "repository": repository,
                "current_sha": current_sha[:12],
                "latest_sha": latest_sha[:12],
                "latest_message": (payload.get("commit", {}).get("message", "").splitlines() or [""])[0],
                "release_url": payload.get("html_url", ""),
                "checked_at": datetime.now().isoformat(timespec="seconds"),
            })
            if update_available and on_update:
                on_update(repository, latest_sha)
        except Exception as exc:
            _write_status(root, {"status": "offline", "repository": repository, "error": str(exc), "checked_at": datetime.now().isoformat(timespec="seconds")})

    threading.Thread(target=check, name="usedsurf-update-check", daemon=True).start()
