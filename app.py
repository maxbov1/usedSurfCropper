import os
import socket
import subprocess
import sys
import threading
import shutil
import shlex
import time
import traceback
from pathlib import Path


def startup_log_path() -> Path:
    if getattr(sys, "frozen", False) or sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "UsedSurf" / "logs" / "startup.log"
    return Path(__file__).resolve().parent / "data" / "startup.log"


def write_startup_log(message: str) -> None:
    try:
        path = startup_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def show_fatal_startup_error(phase: str, error: BaseException) -> None:
    """Make windowless app failures visible instead of only bouncing in the Dock."""
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    log_path = startup_log_path()
    write_startup_log(f"Fatal startup failure during {phase}:\n{details}")
    message = (
        f"UsedSurf could not start during {phase}.\n\n"
        f"{type(error).__name__}: {error}\n\n"
        f"A full diagnostic was saved to:\n{log_path}"
    )
    if sys.platform == "darwin":
        try:
            def applescript_string(value: str) -> str:
                return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

            subprocess.run(
                [
                    "/usr/bin/osascript",
                    "-e",
                    "display alert " + applescript_string("UsedSurf failed to start") +
                    " message " + applescript_string(message) +
                    " as critical buttons {\"OK\"}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return
        except Exception:
            pass
    # This is useful when launching from Terminal or a non-macOS environment.
    print(message, file=sys.stderr)


try:
    from organizer.config import runtime_root
    from organizer.update_check import check_github_async
    from organizer.web import create_app
except Exception:
    error = sys.exc_info()[1]
    show_fatal_startup_error("importing the application", error or RuntimeError("unknown import error"))
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent


def prepare_runtime() -> Path:
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False):
        bundled_model = Path(getattr(sys, "_MEIPASS", ROOT)) / "models" / "yolo11n.pt"
        target = root / "models" / "yolo11n.pt"
        if bundled_model.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_model, target)
        if target.exists():
            os.environ.setdefault("USED_SURF_MODEL_PATH", str(target))
    return root


try:
    RUNTIME_ROOT = prepare_runtime()
    app = create_app(RUNTIME_ROOT)
    write_startup_log(f"App initialized; frozen={getattr(sys, 'frozen', False)}; runtime={RUNTIME_ROOT}")
except Exception:
    error = sys.exc_info()[1]
    show_fatal_startup_error("initializing the application", error or RuntimeError("unknown initialization error"))
    raise SystemExit(1)


def open_browser_when_ready(url: str, port: int) -> None:
    """Open the local app after Flask is listening, with a durable failure log."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    else:
        write_startup_log(f"Browser launch skipped: server did not listen on port {port}")
        return
    try:
        if sys.platform == "darwin":
            result = subprocess.run(["/usr/bin/open", url], capture_output=True, text=True, timeout=10)
            write_startup_log(f"Browser launch: /usr/bin/open {url}; exit={result.returncode}; stderr={result.stderr.strip()!r}")
        else:
            import webbrowser
            opened = webbrowser.open(url)
            write_startup_log(f"Browser launch: webbrowser.open({url}); result={opened!r}")
    except Exception:
        write_startup_log("Browser launch failure:\n" + traceback.format_exc())


def offer_update(repository: str, latest_sha: str) -> None:
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return
    try:
        script = RUNTIME_ROOT / "logs" / "update-usedsurf.command"
        installer_url = f"https://raw.githubusercontent.com/{repository}/main/scripts/install_mac.sh?update={latest_sha[:12]}"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "#!/bin/bash\nset -e\n"
            f"curl -fsSL {shlex.quote(installer_url)} -o \"$TMPDIR/usedsurf-update-installer.sh\"\n"
            "bash \"$TMPDIR/usedsurf-update-installer.sh\"\n"
            "",
            encoding="utf-8",
        )
        script.chmod(0o700)
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", "display dialog \"A newer UsedSurf version is available. Update now?\" buttons {\"Later\", \"Update\"} default button \"Update\" with title \"UsedSurf Update\""],
            capture_output=True, text=True, timeout=30,
        )
        write_startup_log(f"Update check: current app is behind {latest_sha[:12]}; prompt_exit={result.returncode}")
        if result.returncode == 0 and "Update" in result.stdout:
            subprocess.Popen(["/usr/bin/open", "-a", "Terminal", str(script)])
            write_startup_log(f"Update installer opened: {script}")
    except Exception:
        write_startup_log("Update offer failure:\n" + traceback.format_exc())


def run_worker_mode() -> int:
    if "--worker-batch" in sys.argv:
        from scripts.run_batch_crop import main
        sys.argv = ["run_batch_crop.py", "--run-id", sys.argv[sys.argv.index("--run-id") + 1]]
        return main()
    if "--worker-true-crop" in sys.argv:
        from scripts.run_true_crop_eval import main
        sys.argv = ["run_true_crop_eval.py", "--run-id", sys.argv[sys.argv.index("--run-id") + 1]]
        return main()
    return -1

def main() -> None:
    worker_result = run_worker_mode()
    if worker_result >= 0:
        return_code = worker_result
        if return_code:
            raise RuntimeError(f"worker exited with status {return_code}")
        return
    # Keep the local server reachable during long OCR/cropping runs. The
    # assertion is tied to this process and disappears when the app exits.
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
    port = int(os.environ.get("USED_SURF_PORT", "5050"))
    open_browser = os.environ.get("USED_SURF_OPEN_BROWSER", "1" if getattr(sys, "frozen", False) else "0")
    write_startup_log(f"Starting server on 127.0.0.1:{port}; open_browser={open_browser!r}")
    if open_browser == "1":
        threading.Thread(target=open_browser_when_ready, args=(f"http://127.0.0.1:{port}/upload-photos", port), daemon=True).start()
    check_github_async(RUNTIME_ROOT, on_update=offer_update)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        show_fatal_startup_error("starting the local server", error)
        raise SystemExit(1)
