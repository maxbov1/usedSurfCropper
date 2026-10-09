"""Best-effort GitHub update information for the local app."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime
from pathlib import Path


def _write_status(root: Path, payload: dict) -> None:
    status_path = root / "data" / "update-status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2) + "\n")


def check_github_async(root: Path) -> None:
    repository = os.environ.get("USED_SURF_GITHUB_REPO", "").strip().strip("/")
    if not repository or "/" not in repository:
        _write_status(root, {"status": "not_configured", "checked_at": datetime.now().isoformat(timespec="seconds")})
        return

    def check() -> None:
        url = f"https://api.github.com/repos/{repository}/commits/HEAD"
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "UsedSurf"})
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            _write_status(root, {
                "status": "ok",
                "repository": repository,
                "latest_sha": payload.get("sha", "")[:12],
                "latest_message": (payload.get("commit", {}).get("message", "").splitlines() or [""])[0],
                "release_url": payload.get("html_url", ""),
                "checked_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception as exc:
            _write_status(root, {"status": "offline", "repository": repository, "error": str(exc), "checked_at": datetime.now().isoformat(timespec="seconds")})

    threading.Thread(target=check, name="usedsurf-update-check", daemon=True).start()
