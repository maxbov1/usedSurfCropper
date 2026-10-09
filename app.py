import os
import subprocess
import sys
import threading
import webbrowser
import shutil
from pathlib import Path
from organizer.config import runtime_root
from organizer.update_check import check_github_async
from organizer.web import create_app

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


RUNTIME_ROOT = prepare_runtime()
app = create_app(RUNTIME_ROOT)


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

if __name__ == "__main__":
    worker_result = run_worker_mode()
    if worker_result >= 0:
        raise SystemExit(worker_result)
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
    if open_browser == "1":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}/upload-photos")).start()
    check_github_async(RUNTIME_ROOT)
    app.run(host="127.0.0.1", port=port, debug=False)
