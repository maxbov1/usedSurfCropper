from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
import traceback
import zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image

from .config import model_path, paths
from .crop import _silhouette_mask
from .card_ocr import ocr_quality
from .board_evidence import board_evidence_diagnostics, detect_side_profile_fins, extract_board_evidence, extract_photo_evidence
from .db import connect, create_run, record_artifact, recover_stale_runs, set_run_worker, update_run
from .export import export_board, safe_name
from .ingest import MAX_PHOTOS_PER_BOARD, PROCESSING_VERSION, SHOT_LIMITS, STANDARD_BOARD_PHOTOS, _enforce_shot_limits, assign_shot_types_by_position, board_group_diagnostics, board_identity_diagnostics, board_identity_signature, board_local_signature, board_ocr, group_limit_violations, group_scanned, is_card_anchor, local_ocr, next_shot_type_for_occupancy, ocr_runtime_status, read_image, scan_files, set_board_identity_weights, split_oversized_groups_by_embedding
from .segmentation import board_perimeter
from .grouping_learning import load_weights, record_human_save, write_feature_snapshot
from .retention import compact_archive, persist_manifest, prepare_archive

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def _ordered_shot_label(position: int | None) -> str | None:
    if position is None:
        return None
    if 1 <= position <= 4:
        return "full_board"
    if position == 5:
        return "side_profile"
    if position == 6:
        return "fin_detail"
    return None


def create_app(root: Path) -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(project_root / "templates"), static_folder=str(project_root / "static"))
    app.secret_key = "usedsurf-local-only"
    directories = paths(root)
    db_path = directories["data"] / "usedsurf.sqlite3"
    upload_log_path = directories["data"] / "logs" / "upload.log"
    grouping_progress_path = directories["data"] / "grouping-progress.json"
    grouping_log_path = directories["data"] / "logs" / "grouping.log"

    def write_upload_log(message: str) -> None:
        try:
            upload_log_path.parent.mkdir(parents=True, exist_ok=True)
            with upload_log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
        except OSError:
            pass

    def write_grouping_progress(run_id: str, phase: str, message: str, done: int = 0, total: int = 0, status: str = "running", error: str = "") -> None:
        """Publish progress outside the request so the browser can poll it."""
        stages = {
            "scanning": (1, "Reading photos"),
            "grouping": (2, "Finding groups"),
            "classifying": (3, "Analyzing boards"),
            "matching": (4, "Comparing identities"),
            "enriching": (5, "Checking details"),
            "saving": (6, "Saving groups"),
            "complete": (6, "Complete"),
        }
        stage, stage_label = stages.get(phase, (0, "Preparing"))
        phase_percent = round(100 * min(1.0, max(0.0, done / total))) if total > 0 else None
        payload = {
            "run_id": run_id,
            "status": status,
            "phase": phase,
            "stage": stage,
            "stage_count": 6,
            "stage_label": stage_label,
            "message": message,
            "done": done,
            "total": total,
            "phase_percent": phase_percent,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "error": error,
        }
        try:
            grouping_progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = grouping_progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, grouping_progress_path)
            with grouping_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def db():
        return connect(db_path)

    @app.errorhandler(500)
    def grouping_server_error(error):
        """Make synchronous grouping failures visible to the polling UI."""
        try:
            progress = json.loads(grouping_progress_path.read_text())
            run_id = str(progress.get("run_id", ""))
            if run_id:
                write_grouping_progress(run_id, progress.get("phase", "error"), "Grouping failed.", progress.get("done", 0), progress.get("total", 0), status="failed", error=str(error))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"ok": False, "error": "Grouping failed. Open the grouping log for details."}), 500
        return error

    def worker_command(kind: str, run_id: str) -> list[str]:
        if getattr(sys, "frozen", False):
            flag = "--worker-true-crop" if kind == "true_crop_eval" else "--worker-batch"
            return [sys.executable, flag, "--run-id", run_id]
        script = "run_true_crop_eval.py" if kind == "true_crop_eval" else "run_batch_crop.py"
        return [sys.executable, str(project_root / "scripts" / script), "--run-id", run_id]

    def input_signature() -> str:
        digest = hashlib.sha256()
        digest.update(PROCESSING_VERSION.encode())
        for path in sorted(directories["input"].iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            stat = path.stat()
            digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def current_grouping_state(conn, signature: str, active_count: int) -> tuple[bool, bool]:
        """Return (already grouped, OCR newly available) for the active batch."""
        if not active_count:
            return False, False
        previous = conn.execute("SELECT value FROM app_meta WHERE key='active_input_signature'").fetchone()
        if not previous or previous["value"] != signature:
            return False, False
        previous_grouping = conn.execute(
            "SELECT pipeline_version FROM grouping_feedback_batches WHERE input_signature=? ORDER BY created_at DESC LIMIT 1",
            (signature,),
        ).fetchone()
        if not previous_grouping or previous_grouping["pipeline_version"] != PROCESSING_VERSION:
            return False, False
        try:
            saved_ocr_runtime = json.loads((directories["data"] / "ocr-status.json").read_text()).get("runtime", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            saved_ocr_runtime = {}
        ocr_recovered = not saved_ocr_runtime.get("available", False) and ocr_runtime_status().get("available", False)
        return not ocr_recovered, ocr_recovered

    def crop_run_paths(true_mode: bool, run_id: str | None = None) -> tuple[Path, Path, str | None]:
        if not true_mode:
            if not run_id:
                latest = directories["data"] / "crop-eval" / "latest-run.txt"
                if latest.is_file():
                    run_id = latest.read_text().strip() or None
            if run_id:
                run_root = directories["data"] / "crop-eval" / "runs" / run_id
                return run_root / "report.json", run_root / "annotations.json", run_id
            return directories["data"] / "crop-eval" / "latest.json", directories["data"] / "crop-eval" / "annotations.json", None
        root = directories["data"] / "crop-eval" / "true-test-set"
        selected = run_id
        latest = root / "latest-run.txt"
        if not selected and latest.exists():
            selected = latest.read_text().strip()
        if selected:
            run_root = root / "runs" / selected
            return run_root / "report.json", run_root / "annotations.json", selected
        return root / "baseline-v1.json", root / "annotations.json", "legacy"

    def manual_crop_required(item: dict) -> bool:
        """Recognize failed proposals in both current and older crop reports."""
        if item.get("manual_required") is True:
            return True
        reason = str(item.get("method", "")).lower()
        if "manual crop required" in reason:
            return True
        return item.get("model_available") is False and not reason

    def reviewed_crop(item: dict, annotation: dict) -> bool:
        decision = annotation.get("decision")
        if decision not in {"good", "manual"}:
            return False
        if not manual_crop_required(item):
            return True
        if decision != "manual":
            return False
        source_size = item.get("source_size") or item
        crop = annotation.get("expected_crop") or {}
        return bool(annotation.get("rotation") or crop != {"x": 0, "y": 0, "width": source_size.get("width"), "height": source_size.get("height")})

    def active_retest_path() -> Path | None:
        """Choose the current grouping report without letting an experiment win.

        A saved edit is an explicit human decision. Prefer the report with the
        most recently saved edits; only fall back to report mtime when no retest
        has edits. This prevents a later OCR experiment from silently replacing
        the last reviewed grouping in the shuffleboard.
        """
        retest_dir = directories["data"] / "grouping-eval"
        candidates = [
            path for path in retest_dir.glob("real-batch-*-retest*.json")
            if not path.name.endswith("-edits.json")
        ]
        edited = []
        for path in candidates:
            edits = path.with_name(f"{path.stem}-edits.json")
            if edits.exists():
                edited.append((edits.stat().st_mtime, path))
        if edited:
            return max(edited, key=lambda item: item[0])[1]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    @app.get("/")
    def index():
        return render_template("home.html")

    @app.get("/upload-photos")
    def upload_photos():
        uploaded_files = [
            path for path in sorted(directories["input"].iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
        current_input_signature = input_signature() if uploaded_files else ""
        with db() as conn:
            stored_signature_row = conn.execute("SELECT value FROM app_meta WHERE key='active_input_signature'").fetchone()
            library_ready = bool(uploaded_files and stored_signature_row and stored_signature_row["value"] == current_input_signature)
            already_grouped, _ = current_grouping_state(conn, current_input_signature, len(uploaded_files))
            if library_ready:
                all_boards = conn.execute("SELECT * FROM boards ORDER BY id").fetchall()
                active_board_ids = {row["board_id"] for row in conn.execute("SELECT DISTINCT board_id, source_path FROM photos") if (root / row["source_path"]).is_file()}
                boards = [board for board in all_boards if board["id"] in active_board_ids]
                counts = {row["board_id"]: row["n"] for row in conn.execute("SELECT board_id, COUNT(*) n FROM photos GROUP BY board_id") if row["board_id"] in active_board_ids}
                thumbnails = {}
                for row in conn.execute("SELECT board_id, source_path FROM photos ORDER BY COALESCE(capture_time, '9999'), source_path"):
                    if row["board_id"] not in active_board_ids:
                        continue
                    thumbnails.setdefault(row["board_id"], []).append(row["source_path"])
                needs = conn.execute("SELECT COUNT(*) n FROM photos WHERE review_status != 'approved'").fetchone()["n"]
            else:
                boards, counts, thumbnails, needs = [], {}, {}, 0
        uploaded_files = [str(path.relative_to(root)) for path in uploaded_files]
        return render_template("index.html", boards=boards, counts=counts, thumbnails=thumbnails, needs=needs, uploaded_files=uploaded_files, library_ready=library_ready, already_grouped=already_grouped, input_dir=directories["input"], cv2_available=_cv2_available())

    @app.get("/rules")
    def rules():
        return render_template("info.html", title="Rules", sections=[
            ("Your originals", "Upload JPEG photos. The software keeps the original files unchanged."),
            ("What it assumes", "A normal board set usually includes one inventory card and about five or six board photos. The card may appear out of timestamp order or may be missed, so check the groups."),
            ("What it detects", "The software reads the inventory card for SKU, brand, model, and fin information when possible. It uses computer vision to suggest board crops, but the suggestions are not guaranteed."),
            ("What needs review", "Check the board groups and every listing image. Inventory cards are kept for reference and are not included in listing exports."),
        ])

    @app.get("/how-to")
    def how_to():
        return render_template("info.html", title="How To", sections=[
            ("1 · Upload", "Drop your JPEG photos into the upload area, then choose Continue."),
            ("2 · Check groups", "Make sure each board's photos are together. Drag a photo to another row if it is in the wrong group."),
            ("3 · Review photos", "Go through every listing photo. If the suggested crop looks good, move to the next photo. That automatically accepts the suggestion. If it is wrong, draw the crop yourself."),
            ("4 · Prepare upload", "When every photo has been reviewed, choose Go to export. From there, drag individual images to BigCommerce or select a board and download its folder."),
        ])

    @app.get("/runs")
    def runs():
        now = datetime.now()
        with db() as conn:
            rows = conn.execute(
                """SELECT r.*, COUNT(a.id) AS artifact_count
                   FROM runs r LEFT JOIN artifacts a ON a.run_id=r.run_id
                   GROUP BY r.run_id ORDER BY r.created_at DESC LIMIT 100"""
            ).fetchall()
        run_data = []
        for row in rows:
            item = dict(row)
            item["stale"] = False
            if item["status"] == "running" and item.get("started_at"):
                try:
                    item["stale"] = (now - datetime.fromisoformat(item["started_at"])).total_seconds() > 120 * 60
                except ValueError:
                    pass
            run_data.append(item)
        return render_template("runs.html", runs=run_data)

    @app.get("/debug")
    def debug_center():
        with db() as conn:
            schema = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
            active = conn.execute("SELECT COUNT(*) AS count FROM photos WHERE source_path LIKE 'input/%'").fetchone()["count"]
            failed = conn.execute("SELECT COUNT(*) AS count FROM runs WHERE status='failed'").fetchone()["count"]
            running = conn.execute("SELECT COUNT(*) AS count FROM runs WHERE status='running'").fetchone()["count"]
            artifacts = conn.execute("SELECT COUNT(*) AS count FROM artifacts").fetchone()["count"]
        archives = sorted((directories["data"] / "archive").glob("*")) if (directories["data"] / "archive").exists() else []
        disk = shutil.disk_usage(root)
        update_status = {"status": "not checked"}
        update_path = directories["data"] / "update-status.json"
        if update_path.exists():
            try:
                update_status = json.loads(update_path.read_text())
            except (OSError, json.JSONDecodeError):
                update_status = {"status": "unreadable"}
        ocr_status = {"status": "not run"}
        ocr_status_path = directories["data"] / "ocr-status.json"
        if ocr_status_path.exists():
            try:
                ocr_status = json.loads(ocr_status_path.read_text())
            except (OSError, json.JSONDecodeError):
                ocr_status = {"status": "unreadable"}
        # The saved batch report can come from an older process/interpreter.
        # Always show the runtime serving this diagnostics page as the source
        # of truth for whether a new grouping pass can use OCR.
        ocr_status["runtime"] = ocr_runtime_status()
        ocr_status["runtime_python"] = sys.executable
        identity_report = {}
        identity_path = directories["data"] / "board-identity.json"
        if identity_path.exists():
            try:
                identity_report = json.loads(identity_path.read_text())
            except (OSError, json.JSONDecodeError):
                identity_report = {}
        return render_template("debug.html", schema_version=schema["value"] if schema else "unknown",
                               active_photos=active, failed_runs=failed, running_runs=running,
                               artifact_count=artifacts, archive_count=len(archives),
                               free_gb=disk.free / (1024 ** 3), cv2_available=_cv2_available(),
                               yolo_available=model_path(root).exists(), db_path=db_path, update_status=update_status, ocr_status=ocr_status,
                               identity_report=identity_report)

    @app.get("/debug/retention")
    def retention_review():
        archive_dir = directories["data"] / "archive"
        archives = []
        for path in (sorted(archive_dir.glob("*")) if archive_dir.exists() else []):
            manifest_path = path / "retention-manifest.json"
            if not path.is_dir() or not manifest_path.exists():
                continue
            try:
                items = json.loads(manifest_path.read_text()).get("items", [])
            except (OSError, json.JSONDecodeError):
                items = []
            archives.append({
                "name": path.name,
                "items": len(items),
                "eligible": sum(item.get("retention_class") == "ordinary" and item.get("status") == "verified" for item in items),
                "protected": sum(item.get("retention_class") != "ordinary" for item in items),
                "compacted": sum(not (path / item.get("source", "")).exists() for item in items if item.get("status") == "verified"),
            })
        return render_template("retention.html", archives=archives)

    @app.get("/debug/upload-log")
    def upload_log():
        if not upload_log_path.is_file():
            return app.response_class("No upload events have been logged yet.\n", mimetype="text/plain")
        return app.response_class(upload_log_path.read_text(errors="replace")[-30000:], mimetype="text/plain")

    @app.get("/debug/grouping-log")
    def grouping_log():
        if not grouping_log_path.is_file():
            return app.response_class("No grouping events have been logged yet.\n", mimetype="text/plain")
        return app.response_class(grouping_log_path.read_text(errors="replace")[-50000:], mimetype="text/plain")

    @app.get("/process-status/<run_id>")
    def process_status(run_id):
        """Return live grouping progress while the synchronous process request runs."""
        if run_id != Path(run_id).name:
            return jsonify({"status": "failed", "error": "Invalid grouping run id."}), 400
        try:
            progress = json.loads(grouping_progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            progress = {"run_id": run_id, "status": "starting", "phase": "starting", "stage": 0, "stage_count": 6, "stage_label": "Preparing", "message": "Starting grouping…", "phase_percent": None, "done": 0, "total": 0}
        if progress.get("run_id") != run_id:
            progress = {"run_id": run_id, "status": "starting", "phase": "starting", "stage": 0, "stage_count": 6, "stage_label": "Preparing", "message": "Starting grouping…", "phase_percent": None, "done": 0, "total": 0}
        return jsonify(progress)

    @app.get("/debug/board-identity")
    def board_identity_log():
        diagnostics_path = directories["data"] / "board-identity.json"
        if not diagnostics_path.is_file():
            return app.response_class('{"status":"not_available","message":"Run grouping first."}\n', mimetype="application/json")
        return app.response_class(diagnostics_path.read_text(errors="replace"), mimetype="application/json")

    @app.get("/debug/board-evidence")
    def board_evidence_log():
        diagnostics_path = directories["data"] / "board-evidence.json"
        if not diagnostics_path.is_file():
            return app.response_class('{"status":"not_available","message":"Run grouping first."}\n', mimetype="application/json")
        return app.response_class(diagnostics_path.read_text(errors="replace"), mimetype="application/json")

    @app.post("/debug/client-log")
    def client_log():
        payload = request.get_json(silent=True) or {}
        event = str(payload.get("event", "unknown"))[:120]
        details = str(payload.get("details", ""))[:1000]
        write_upload_log(f"browser event={event!r} details={details!r} user_agent={request.user_agent.string!r}")
        return jsonify({"ok": True})

    @app.post("/debug/retention/<archive_name>/compact")
    def compact_retention(archive_name):
        if archive_name != Path(archive_name).name:
            return "Invalid archive", 400
        archive_root = directories["data"] / "archive" / archive_name
        if not archive_root.is_dir():
            return "Archive not found", 404
        backup_dir = directories["data"] / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"retention-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        backup_path = backup_dir / f"pre-compaction-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.sqlite3"
        with db() as conn:
            create_run(conn, run_id, "retention_compaction", pipeline_version=PROCESSING_VERSION,
                       metadata={"archive": archive_name}, status="running")
        shutil.copy2(db_path, backup_path)
        result = compact_archive(archive_root, apply=True)
        with db() as conn:
            manifest = json.loads((archive_root / "retention-manifest.json").read_text())
            for item in manifest.get("items", []):
                if item.get("retention_class") == "ordinary" and item.get("status") == "verified":
                    relative = str((archive_root / item["source"]).relative_to(root))
                    conn.execute("UPDATE retention_items SET status='compacted', updated_at=? WHERE relative_path=?", (datetime.now().isoformat(timespec="seconds"), relative))
            record_artifact(conn, run_id, backup_path, root, "database_backup")
            record_artifact(conn, run_id, archive_root / "retention-manifest.json", root, "retention_manifest")
            update_run(conn, run_id, "complete", metadata=result)
        flash(f"Compacted {result['removed']} ordinary original(s); protected sources were kept.", "success")
        return redirect(url_for("retention_review"))

    @app.get("/runs/<run_id>")
    def run_detail(run_id):
        with db() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            artifacts = conn.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY kind, relative_path", (run_id,)).fetchall()
        if not run:
            return "Run not found", 404
        return render_template("run_detail.html", run=run, artifacts=artifacts)

    @app.get("/runs/<run_id>/log")
    def run_log(run_id):
        with db() as conn:
            run = conn.execute("SELECT log_path, kind FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            return "Run not found", 404
        log_path = root / run["log_path"] if run["log_path"] else None
        if not log_path or not log_path.is_file() or root not in log_path.resolve().parents:
            return "No log is available for this run.", 404
        return app.response_class(log_path.read_text(errors="replace")[-12000:], mimetype="text/plain")

    @app.post("/runs/recover")
    def recover_runs():
        with db() as conn:
            recovered = recover_stale_runs(conn)
        flash(f"Marked {len(recovered)} abandoned run(s) as failed." if recovered else "No abandoned runs found.", "success")
        return redirect(url_for("runs"))

    @app.post("/runs/<run_id>/recover")
    def recover_run(run_id):
        with db() as conn:
            row = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row and row["status"] == "running":
                update_run(conn, run_id, "failed", error="Marked failed manually; worker may have stopped.")
                flash(f"Recovered run {run_id}.", "success")
            else:
                flash("That run is no longer active.", "success")
        return redirect(url_for("runs"))

    @app.post("/runs/<run_id>/restart")
    def restart_run(run_id):
        with db() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row or row["kind"] not in {"crop", "true_crop_eval"}:
                flash("Only crop runs can be restarted from here.", "error")
                return redirect(url_for("runs"))
            if row["status"] not in {"failed", "complete"}:
                flash("Recover the active run before restarting it.", "error")
                return redirect(url_for("runs"))
            retry_base = f"{row['kind']}-retry-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            retry_id = retry_base
            retry_number = 2
            output_root = directories["data"] / "crop-eval" / ("true-test-set" if row["kind"] == "true_crop_eval" else "") / "runs" / retry_id
            while conn.execute("SELECT 1 FROM runs WHERE run_id=?", (retry_id,)).fetchone() or output_root.exists():
                retry_id = f"{retry_base}-{retry_number}"
                output_root = directories["data"] / "crop-eval" / ("true-test-set" if row["kind"] == "true_crop_eval" else "") / "runs" / retry_id
                retry_number += 1
            create_run(conn, retry_id, row["kind"], pipeline_version=row["pipeline_version"],
                       input_signature=row["input_signature"], metadata={"retry_of": run_id})
        true_mode = row["kind"] == "true_crop_eval"
        if true_mode:
            log_path = directories["data"] / "crop-eval" / "true-test-set" / f"{retry_id}.log"
            script = project_root / "scripts" / "run_true_crop_eval.py"
        else:
            log_path = directories["data"] / "crop-eval" / f"{retry_id}.log"
            script = project_root / "scripts" / "run_batch_crop.py"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w")
        try:
            worker = subprocess.Popen(worker_command(row["kind"], retry_id), cwd=project_root,
                                       stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log_stream.close()
            with db() as conn:
                update_run(conn, retry_id, "failed", error=str(exc))
            flash(f"Could not restart run: {exc}", "error")
            return redirect(url_for("runs"))
        finally:
            log_stream.close()
        with db() as conn:
            set_run_worker(conn, retry_id, pid=worker.pid, log_path=str(log_path.relative_to(root)))
        flash(f"Restarted as {retry_id}.", "success")
        return redirect(url_for("runs"))

    @app.post("/process")
    def process():
        signature = input_signature()
        with db() as conn:
            active_count = sum(
                1 for path in directories["input"].iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            )
            already_grouped, ocr_recovered = current_grouping_state(conn, signature, active_count)
        if already_grouped:
            if request.accept_mimetypes.best == "application/json":
                return jsonify({"ok": True, "already_grouped": True, "redirect": url_for("shuffleboard")})
            return redirect(url_for("shuffleboard"))
        if ocr_recovered:
            flash("OCR is now available; rescanning this batch for board identity fields.", "success")
        requested_run_id = str(request.headers.get("X-Grouping-Progress-Id", "")).strip()
        group_run_id = requested_run_id if requested_run_id and requested_run_id.replace("-", "").isalnum() else datetime.now().strftime("group-%Y%m%d-%H%M%S-%f")
        write_grouping_progress(group_run_id, "starting", f"Preparing {active_count} photos…", total=active_count)
        with db() as conn:
            create_run(conn, group_run_id, "grouping", pipeline_version=PROCESSING_VERSION,
                       input_signature=signature, metadata={"input_count": active_count}, status="running")
        write_grouping_progress(group_run_id, "scanning", f"Reading {active_count} photos…", total=active_count)
        def scan_progress(done, total, filename, cached=False):
            write_grouping_progress(
                group_run_id,
                "scanning",
                f"Read photo {done} of {total}: {filename}",
                done,
                total,
            )

        scanned = scan_files(directories["input"], progress=scan_progress)
        write_grouping_progress(group_run_id, "grouping", "Finding board groups from capture order and image evidence…")
        # Grouping runs first and decides shot_type by position in each
        # card-anchored, capture-ordered group (see group_scanned /
        # assign_shot_types_by_position) — not by guessing a photo's shape.
        # The per-photo pass below only needs to know the type that
        # decision already produced; it never recomputes or overrides it.
        groups = group_scanned(scanned)
        # Tie-breaker only: a group this oversized already means capture
        # order couldn't fully explain it (most likely a missed card). SAM 2
        # embeddings get one chance to find a confident internal boundary
        # before shot types are assigned; a group it can't confidently split
        # (or with no SAM 2 configured at all) passes through unchanged for
        # _enforce_shot_limits' plain arithmetic split further down.
        groups = split_oversized_groups_by_embedding(groups)
        assign_shot_types_by_position(groups)
        from .crop import classify_shot, usable_yolo_prompt_box
        total_scanned = len(scanned)
        write_grouping_progress(group_run_id, "classifying", f"Analyzing {total_scanned} photos…", total=total_scanned)
        for item_index, item in enumerate(scanned, start=1):
            if "error" in item or item.get("is_card"):
                write_grouping_progress(group_run_id, "classifying", f"Checked photo {item_index} of {total_scanned}", item_index, total_scanned)
                continue
            try:
                image, _ = read_image(item["path"])
                # classify_shot() is used here only for its candidate box
                # (a useful region hint for OCR cropping below) — never for
                # its own shot_type guess, which is where real
                # misclassifications used to come from.
                classification = classify_shot(image)
                item["board_candidate"] = classification.get("candidate")
                # classify_shot() runs GrabCut plus 1-3 YOLO passes - real
                # cost, and the crop worker (a separate later process) would
                # otherwise redo all of it from scratch for every photo.
                # Persist the raw result now so cropping can skip straight
                # to using it.
                item["classification_snapshot"] = dict(classification)
                # SAM 2 is optional and lazy (see board_perimeter): with no
                # checkpoint configured this is None and the signatures below
                # fall back to the candidate rectangle alone, unchanged. The
                # box/score are persisted so the crop worker (a separate
                # process, run later) reuses this instead of predicting again.
                yolo_prompt = usable_yolo_prompt_box(classification.get("yolo_detection"), image.width, image.height, item["shot_type"])
                perimeter = board_perimeter(image, item["shot_type"], prompt_box=yolo_prompt)
                region_mask = perimeter["mask"] if perimeter else None
                item["board_perimeter"] = (
                    {"box": list(perimeter["box"]), "score": perimeter["score"]}
                    if perimeter and perimeter.get("box") and perimeter.get("score") is not None
                    else None
                )
                item["board_identity_signature"] = board_identity_signature(
                    image,
                    classification.get("candidate"),
                    item["shot_type"],
                    item.get("card_regions") or [],
                    region_mask,
                )
                item["board_local_signature"] = board_local_signature(
                    image,
                    classification.get("candidate"),
                    item["shot_type"],
                    item.get("card_regions") or [],
                    region_mask,
                )
                if item["shot_type"] == "side_profile":
                    # These views are poor OCR sources. Card OCR, when a
                    # contained card was detected during scanning, is already
                    # available in ``ocr`` and does not need a second pass.
                    photo_text = item.get("ocr", "") if item.get("contains_inventory_card") else ""
                    photo_ocr_error = ""
                    item["photo_ocr_status"] = "skipped_angle"
                elif item["shot_type"] == "fin_detail":
                    # Fin-detail OCR is deferred until grouping proves that
                    # this group actually needs fin/box evidence.
                    photo_text, photo_ocr_error = "", ""
                    item["photo_ocr_status"] = "deferred_angle"
                elif item["shot_type"] == "full_board":
                    # Board signatures are sufficient for the first grouping
                    # pass. Expensive multi-crop OCR is selected after groups
                    # exist, only where it can resolve a real uncertainty.
                    photo_text, photo_ocr_error = "", ""
                    item["photo_ocr_status"] = "deferred_identity"
                else:
                    photo_text, photo_ocr_error = "", ""
                    item["photo_ocr_status"] = "deferred_unclassified"
                item["photo_ocr_text"] = photo_text
                item["photo_ocr_error"] = photo_ocr_error
                photo_quality = ocr_quality(photo_text)
                item["photo_ocr_quality"] = photo_quality
                if item.get("photo_ocr_status") not in {
                    "skipped_angle", "deferred_angle", "deferred_identity", "deferred_unclassified",
                } and not photo_ocr_error:
                    item["photo_ocr_status"] = "usable" if photo_quality.get("status") == "usable" else (
                        "empty" if photo_quality.get("status") == "empty" else "junk"
                    )
                item["identity_evidence"] = (
                    extract_photo_evidence(photo_text, item["shot_type"])
                    if photo_quality.get("status") == "usable" else {}
                )
                if item["shot_type"] == "side_profile":
                    visual_fins = detect_side_profile_fins(image, classification.get("candidate"))
                    if visual_fins.get("value"):
                        item["identity_evidence"]["fins_included"] = visual_fins
            except Exception as exc:
                # shot_type is left as grouping already decided it (by
                # position) — this failure is about OCR/signature
                # enrichment, not about which board or slot the photo is.
                item["photo_ocr_text"] = ""
                item["processing_error"] = f"{type(exc).__name__}: {exc}"
                item["photo_ocr_error"] = item["processing_error"]
                item["photo_ocr_status"] = "error"
                item["photo_ocr_quality"] = {"status": "error"}
                item["identity_evidence"] = {}
            write_grouping_progress(group_run_id, "classifying", f"Processed photo {item_index} of {total_scanned}", item_index, total_scanned)
        write_grouping_progress(group_run_id, "matching", "Comparing board identity evidence…")
        with db() as conn:
            excluded = {row["source_path"] for row in conn.execute("SELECT source_path FROM excluded_sources")}
            current_sources = {str(item["path"].relative_to(root)) for item in scanned if "error" not in item}
            current_by_source = {
                str(item["path"].relative_to(root)): item
                for item in scanned if "error" not in item
            }
            # A new grouping pass must not inherit stale *unreviewed* board
            # ownership. That was the hidden data leak behind reruns looking
            # unchanged. Approved boards and explicit human moves are kept;
            # otherwise the old proposed folders are rebuilt from the current
            # visual/card evidence while grouping history remains in SQLite.
            active_rows = conn.execute(
                "SELECT p.id, p.source_path, p.content_hash, p.human_correction, p.review_status, b.status "
                "FROM photos p JOIN boards b ON b.id=p.board_id WHERE p.source_path LIKE 'input/%'"
            ).fetchall()
            changed_or_missing_ids = [
                row["id"] for row in active_rows
                if row["source_path"] not in current_sources
                or row["content_hash"] != current_by_source[row["source_path"]]["hash"]
            ]
            if changed_or_missing_ids:
                placeholders = ",".join("?" for _ in changed_or_missing_ids)
                conn.execute(f"DELETE FROM grouping_predictions WHERE photo_id IN ({placeholders})", changed_or_missing_ids)
                conn.execute(f"DELETE FROM grouping_corrections WHERE photo_id IN ({placeholders})", changed_or_missing_ids)
                conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", changed_or_missing_ids)
                active_rows = [row for row in active_rows if row["id"] not in changed_or_missing_ids]
            has_locked_work = any(
                row["status"] == "approved"
                or row["review_status"] == "approved"
                or row["human_correction"]
                for row in active_rows
                if row["source_path"] in current_sources
            )
            if not has_locked_work and current_sources:
                stale_ids = [row["id"] for row in active_rows if row["source_path"] in current_sources]
                if stale_ids:
                    placeholders = ",".join("?" for _ in stale_ids)
                    conn.execute(f"DELETE FROM grouping_predictions WHERE photo_id IN ({placeholders})", stale_ids)
                    conn.execute(f"DELETE FROM photos WHERE id IN ({placeholders})", stale_ids)
                    conn.execute("DELETE FROM boards WHERE status != 'approved' AND id NOT IN (SELECT DISTINCT board_id FROM photos)")
            learned_weights = load_weights(directories["data"])
            set_board_identity_weights((learned_weights or {}).get("weights"))
            # Composition limits are model invariants. Keep the visual
            # proposal reviewable by splitting an over-limit cluster before
            # it reaches SQLite or the shuffleboard. A human may still
            # deliberately override this arrangement in the shuffleboard.
            # ``groups`` was already computed above (capture-order grouping,
            # then shot_type assigned by position) — only the overflow-
            # splitting safety net runs here.
            groups = _enforce_shot_limits(groups)
            # Expensive board/fin OCR is demand-driven. The first pass uses
            # card OCR plus board signatures; only incomplete or conflicted
            # groups receive representative enrichment.
            deferred_targets = []
            for group in groups:
                card = next((item for item in group if is_card_anchor(item)), None)
                identifier = (card or {}).get("identifier") or {}
                needs_identity = (
                    card is None
                    or not identifier.get("brand")
                    or not identifier.get("model")
                    or any(item.get("identity_assignment_conflicts") for item in group)
                    or len(group) < STANDARD_BOARD_PHOTOS
                )
                needs_fin = (
                    card is None
                    or not identifier.get("fin_system")
                    or not identifier.get("fin_box_count")
                    or not any(item.get("shot_type") == "fin_detail" for item in group)
                )
                selected = []
                if needs_identity:
                    # Two representative full-board views are enough to
                    # resolve a missing/conflicting identity. The remaining
                    # views stay cheap unless a later review explicitly
                    # requests deeper evidence.
                    selected.extend(
                        item for item in group if item.get("shot_type") == "full_board"
                    )
                    selected = selected[:2]
                if needs_fin:
                    selected.extend(
                        item for item in group if item.get("shot_type") == "fin_detail"
                    )
                seen = set()
                for item in selected:
                    if id(item) not in seen:
                        deferred_targets.append(item)
                        seen.add(id(item))
            if deferred_targets:
                write_grouping_progress(group_run_id, "enriching", f"Checking {len(deferred_targets)} targeted board views…", 0, len(deferred_targets))
            for target_index, item in enumerate(deferred_targets, start=1):
                try:
                    image, _ = read_image(item["path"])
                    if item.get("shot_type") == "fin_detail":
                        item["board_evidence"] = extract_board_evidence(
                            image, item.get("board_candidate"), item["shot_type"],
                            ocr_runner=lambda patch: local_ocr(patch, fast=True),
                            use_text_patches=True,
                        )
                        photo_text = item["board_evidence"].get("ocr_text", "")
                    else:
                        photo_text, _, photo_ocr_error = board_ocr(image, item.get("board_candidate"), fast=True)
                        item["board_evidence"] = extract_board_evidence(
                            image, item.get("board_candidate"), item["shot_type"],
                            ocr_runner=lambda patch: local_ocr(patch, fast=True),
                            use_text_patches=True,
                        )
                        regional_text = item["board_evidence"].get("ocr_text", "")
                        if regional_text:
                            photo_text = "\n".join(value for value in (photo_text, regional_text) if value)
                    item["photo_ocr_text"] = photo_text
                    item["photo_ocr_quality"] = ocr_quality(photo_text)
                    item["photo_ocr_status"] = "usable" if item["photo_ocr_quality"].get("status") == "usable" else ("junk" if photo_text.strip() else "empty")
                    item["identity_evidence"] = extract_photo_evidence(photo_text, item.get("shot_type", "")) if item["photo_ocr_quality"].get("status") == "usable" else {}
                except Exception as exc:
                    item["photo_ocr_status"] = "error"
                    item["photo_ocr_error"] = f"{type(exc).__name__}: {exc}"
                write_grouping_progress(group_run_id, "enriching", f"Checked targeted view {target_index} of {len(deferred_targets)}", target_index, len(deferred_targets))
            write_grouping_progress(group_run_id, "saving", f"Preparing {len(groups)} proposed groups…", total=len(groups))
            learning_snapshot = write_feature_snapshot(directories["data"], group_run_id, scanned)
            evidence_report = {
                "pipeline_version": PROCESSING_VERSION,
                "run_id": group_run_id,
                "groups": board_evidence_diagnostics(groups),
            }
            try:
                (directories["data"] / "board-evidence.json").write_text(json.dumps(evidence_report, indent=2) + "\n")
            except OSError:
                pass
            identity_report = board_identity_diagnostics(scanned)
            identity_report.update({
                "pipeline_version": PROCESSING_VERSION,
                "run_id": group_run_id,
                "proposed_groups": len(groups),
                "grouping_method": "global-card-count-board-identity" if any(item.get("board_identity_signature") for item in scanned if not item.get("is_card")) else "legacy-visual-run-fallback",
                "learned_metric_active": bool(learned_weights),
                "groups": board_group_diagnostics(groups),
            })
            for group_report, evidence in zip(identity_report["groups"], evidence_report["groups"]):
                group_report["field_evidence"] = evidence
            try:
                (directories["data"] / "board-identity.json").write_text(json.dumps(identity_report, indent=2) + "\n")
            except OSError:
                pass
            ocr_runtime = next((item.get("ocr_runtime") for item in scanned if item.get("ocr_runtime")), {"available": False, "status": "not_checked"})
            ocr_summary = {
                "runtime": ocr_runtime,
                "files_scanned": len(scanned),
                "cards_detected": sum(1 for item in scanned if is_card_anchor(item)),
                "expected_groups": sum(1 for item in scanned if is_card_anchor(item)),
                "proposed_groups": len(groups),
                "identifiers_extracted": sum(1 for item in scanned if item.get("identifier")),
                "ocr_attempts": sum(1 for item in scanned if item.get("ocr_status") in {"read", "empty", "error"}) + sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") in {"empty", "nonempty", "error"}),
                "photo_ocr_attempts": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") in {"empty", "usable", "junk", "error"}),
                "photo_ocr_usable": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") == "usable"),
                "photo_ocr_junk": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") == "junk"),
                "photo_ocr_nonempty": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") in {"usable", "junk"}),
                "photo_ocr_empty": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") == "empty"),
                "photo_ocr_usable_fields": sum(1 for item in scanned if not item.get("is_card") and item.get("identity_evidence")),
                "photo_ocr_errors": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") == "error"),
                "photo_ocr_skipped_angles": sum(1 for item in scanned if not item.get("is_card") and item.get("photo_ocr_status") == "skipped_angle"),
                "ocr_errors": [
                    {"source": str(item["path"].relative_to(root)), "error": item.get("ocr_error") or item.get("photo_ocr_error")}
                    for item in scanned if item.get("ocr_error") or item.get("photo_ocr_error")
                ][:20],
                "checked_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                ocr_status_path = directories["data"] / "ocr-status.json"
                ocr_status_path.write_text(json.dumps(ocr_summary, indent=2) + "\n")
            except OSError:
                pass
            initial_grouping = {}
            for group_number, group in enumerate(groups, start=1):
                if not group:
                    write_grouping_progress(group_run_id, "saving", f"Prepared group {group_number} of {len(groups)}", group_number, len(groups))
                    continue
                card = next((item for item in group if is_card_anchor(item)), None)
                ident = (card or {}).get("identifier", {})
                card_count = sum(1 for item in group if is_card_anchor(item))
                unassigned_photo = len(group) == 1 and bool(group[0].get("_unassigned_photo"))
                unassigned_cluster = any(item.get("_unassigned_cluster") for item in group)
                label = ident.get("sku") or f"group-{group_number:02d}-needs-review"
                if unassigned_photo:
                    label = f"unassigned-photo-{group_number:02d}"
                elif unassigned_cluster:
                    label = f"group-{group_number:02d}-unassigned-cluster"
                elif card is None:
                    label = f"group-{group_number:02d}-needs-review-boundary"
                if card_count > 1:
                    label = f"group-{group_number:02d}-card-conflict"
                refresh_label = label if unassigned_photo or unassigned_cluster else ident.get("sku", "")
                existing = None
                # Source ownership is the idempotency key: corrections may have changed
                # the board label since the previous scan.
                for item in group:
                    source = str(item["path"].relative_to(root))
                    existing = conn.execute("SELECT b.id FROM boards b JOIN photos p ON p.board_id=b.id WHERE p.source_path=?", (source,)).fetchone()
                    if existing:
                        break
                if not existing:
                    existing = conn.execute("SELECT id FROM boards WHERE label=? AND id IN (SELECT DISTINCT board_id FROM photos WHERE source_path LIKE 'input/%')", (label,)).fetchone()
                board_id = existing["id"] if existing else conn.execute("INSERT INTO boards(label, shaper, model, sku, fin_system, fin_box_count, fins_included, fins_present_count, status, created_at, processing_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (label, ident.get("brand", ""), ident.get("model", ""), ident.get("sku", ""), ident.get("fin_system", ""), ident.get("fin_box_count", ""), ident.get("fins_included", ""), ident.get("fins_present_count", ""), "unreviewed", datetime.now().isoformat(timespec="seconds"), PROCESSING_VERSION)).lastrowid
                if existing and (ident.get("brand") or ident.get("model") or ident.get("sku") or ident.get("fin_system") or ident.get("fin_box_count") or ident.get("fins_included") or ident.get("fins_present_count") or refresh_label):
                    # Populate only empty predictions; human corrections remain
                    # authoritative on later grouping runs.
                    conn.execute("UPDATE boards SET shaper=?, model=?, sku=?, fin_system=?, fin_box_count=?, fins_included=?, fins_present_count=?, label=CASE WHEN label LIKE 'group-%' OR label LIKE 'unassigned-photo-%' THEN COALESCE(NULLIF(?,''), label) ELSE label END, processing_version=? WHERE id=? AND status='unreviewed'", (ident.get("brand", ""), ident.get("model", ""), ident.get("sku", ""), ident.get("fin_system", ""), ident.get("fin_box_count", ""), ident.get("fins_included", ""), ident.get("fins_present_count", ""), refresh_label, PROCESSING_VERSION, board_id))
                for item in group:
                    if "error" in item:
                        continue
                    if str(item["path"].relative_to(root)) in excluded:
                        continue
                    source_path = str(item["path"].relative_to(root))
                    initial_grouping[source_path] = {"board_id": board_id, "board_label": label}
                    old = conn.execute("SELECT id FROM photos WHERE source_path=?", (source_path,)).fetchone()
                    if old:
                        # A rerun with the OCR-enabled environment refreshes
                        # prior blank card predictions without touching human
                        # grouping/crop corrections.
                        persisted_ocr = item.get("ocr") or item.get("photo_ocr_text", "")
                        perimeter_json = json.dumps(item["board_perimeter"]) if item.get("board_perimeter") else ""
                        classification_json = json.dumps(item["classification_snapshot"]) if item.get("classification_snapshot") else ""
                        conn.execute("UPDATE photos SET source_is_card=?, shot_type=CASE WHEN ? <> '' THEN ? ELSE shot_type END, ocr_text=CASE WHEN ? <> '' THEN ? ELSE ocr_text END, card_fins_included=CASE WHEN ? <> '' THEN ? ELSE card_fins_included END, card_fin_system=CASE WHEN ? <> '' THEN ? ELSE card_fin_system END, board_perimeter=CASE WHEN ? <> '' THEN ? ELSE board_perimeter END, classification_snapshot=CASE WHEN ? <> '' THEN ? ELSE classification_snapshot END WHERE id=?", (int(item["is_card"]), item.get("shot_type", ""), item.get("shot_type", ""), persisted_ocr, persisted_ocr, item.get("identifier", {}).get("fins_included", ""), item.get("identifier", {}).get("fins_included", ""), item.get("identifier", {}).get("fin_system", ""), item.get("identifier", {}).get("fin_system", ""), perimeter_json, perimeter_json, classification_json, classification_json, old["id"]))
                        continue
                    # Grouping is deliberately separate from cropping. Keep a
                    # full-frame placeholder until this group is approved.
                    shot = item.get("shot_type") or ("card" if item["is_card"] else "unclassified")
                    persisted_ocr = item.get("ocr") or item.get("photo_ocr_text", "")
                    conn.execute("INSERT INTO photos(board_id, source_path, content_hash, capture_time, width, height, shot_type, original_prediction, crop_x, crop_y, crop_w, crop_h, source_is_card, ocr_text, card_fins_included, card_fin_system, board_perimeter, classification_snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (board_id, source_path, item["hash"], item["capture"], item["width"], item["height"], shot, "grouping only; crop not run", 0, 0, item["width"], item["height"], int(item["is_card"]), persisted_ocr, item.get("identifier", {}).get("fins_included", ""), item.get("identifier", {}).get("fin_system", ""), json.dumps(item["board_perimeter"]) if item.get("board_perimeter") else "", json.dumps(item["classification_snapshot"]) if item.get("classification_snapshot") else ""))
                write_grouping_progress(group_run_id, "saving", f"Prepared group {group_number} of {len(groups)}", group_number, len(groups))
            conn.execute("INSERT OR REPLACE INTO grouping_feedback_batches(run_id, input_signature, pipeline_version, created_at, status) VALUES (?,?,?,?,?)", (group_run_id, signature, PROCESSING_VERSION, datetime.now().isoformat(timespec="seconds"), "proposed"))
            for source_path, initial in initial_grouping.items():
                photo = conn.execute("SELECT id, content_hash FROM photos WHERE source_path=?", (source_path,)).fetchone()
                if photo:
                    conn.execute("INSERT OR REPLACE INTO grouping_feedback(run_id, photo_id, source_path, content_hash, initial_board_id, initial_board_label) VALUES (?,?,?,?,?,?)", (group_run_id, photo["id"], source_path, photo["content_hash"], initial["board_id"], initial["board_label"]))
            write_grouping_progress(group_run_id, "saving", "Committing proposed groups…")
        conn.commit()
        with db() as conn:
            conn.execute("INSERT INTO app_meta(key,value) VALUES('active_input_signature',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (signature,))
            conn.execute("INSERT INTO app_meta(key,value) VALUES('active_grouping_run_id',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (group_run_id,))
            update_run(conn, group_run_id, "complete", metadata={"scanned": len(scanned), "groups": len(groups), "cards": sum(1 for item in scanned if item.get("is_card")), "ocr": ocr_summary, "learning": {"active": bool(learned_weights), "snapshot": bool(learning_snapshot), "weights_version": (learned_weights or {}).get("version")}})
            conn.commit()
            card_count = sum(1 for item in scanned if is_card_anchor(item))
        write_grouping_progress(group_run_id, "complete", f"Grouped {len(scanned)} photos into {len(groups)} groups.", total=total_scanned, done=total_scanned, status="complete")
        warning = " OCR/card detection found no card boundaries; review the group before editing." if card_count == 0 and scanned else ""
        if card_count and len(groups) != card_count:
            short_groups = sum(1 for group in groups if len(group) < STANDARD_BOARD_PHOTOS)
            warning += f" Card/group review required: detected {card_count} inventory cards but proposed {len(groups)} groups ({short_groups} under six photos)."
        if not ocr_runtime.get("available"):
            warning += f" OCR unavailable ({ocr_runtime.get('error') or ocr_runtime.get('status')}); install/check Tesseract before relying on card identity."
        elif ocr_summary["cards_detected"] and not ocr_summary["identifiers_extracted"]:
            warning += " Cards were detected but OCR extracted no identifiers; open Advanced debug and inspect OCR status."
        flash(f"Grouped {len(scanned)} originals into {len(groups)} proposed folder(s), six standard or seven with fins. Cropping has not run.{warning}", "success")
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"ok": True, "run_id": group_run_id, "redirect": url_for("shuffleboard")})
        return redirect(url_for("shuffleboard"))

    @app.post("/upload")
    def upload():
        """Accept valid JPEGs and skip every other dropped file."""
        uploaded = request.files.getlist("photos")
        write_upload_log(
            f"request content_length={request.content_length!r} files_received={len(uploaded)} "
            f"user_agent={request.user_agent.string!r}"
        )
        accepted, rejected = [], []
        for item in uploaded:
            original_name = (item.filename or "").strip()
            suffix = Path(original_name).suffix.lower()
            if suffix not in {".jpg", ".jpeg"}:
                rejected.append(f"{original_name or 'unnamed file'} — JPEG only")
                write_upload_log(f"rejected name={original_name!r} reason=extension")
                continue
            stem = safe_name(Path(original_name).stem, fallback="photo")
            temporary = directories["input"] / f".{stem}.uploading"
            try:
                item.save(temporary)
                with Image.open(temporary) as image:
                    if image.format != "JPEG":
                        raise ValueError("file contents are not JPEG")
                    image.verify()
                destination = directories["input"] / f"{stem}{suffix}"
                counter = 2
                while destination.exists():
                    destination = directories["input"] / f"{stem}-{counter}{suffix}"
                    counter += 1
                os.replace(temporary, destination)
                accepted.append(destination.name)
                write_upload_log(f"accepted name={original_name!r} stored={destination.name!r}")
                with db() as conn:
                    conn.execute("DELETE FROM app_meta WHERE key='active_input_signature'")
                    conn.commit()
            except Exception as exc:
                rejected.append(f"{original_name} — {exc}")
                write_upload_log(f"rejected name={original_name!r} reason={exc!r}\n{traceback.format_exc()}")
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        if not uploaded:
            write_upload_log("request completed with no files; likely browser drag/drop or form-data failure")
        write_upload_log(f"request complete accepted={len(accepted)} rejected={len(rejected)}")
        return jsonify({"ok": True, "accepted": accepted, "rejected": rejected, "message": f"Added {len(accepted)} JPEG" + ("s" if len(accepted) != 1 else "") + " to this batch.", "diagnostic_log_url": url_for("upload_log")})

    @app.post("/cleanup")
    def cleanup():
        """Archive the active working batch and reset only unapproved UI state."""
        cleanup_id = datetime.now().strftime("cleanup-%Y%m%d-%H%M%S")
        archive_root = directories["data"] / "archive" / cleanup_id
        archive_input = archive_root / "input"
        archive_root.mkdir(parents=True, exist_ok=False)
        archive_input.mkdir(parents=True, exist_ok=True)
        db_snapshot = archive_root / "usedsurf-before-cleanup.sqlite3"
        with db() as conn:
            retention_rows = conn.execute(
                """SELECT p.source_path, p.source_is_card, p.human_correction,
                   p.review_status FROM photos p WHERE p.source_path LIKE 'input/%'"""
            ).fetchall()
            create_run(conn, cleanup_id, "cleanup", pipeline_version=PROCESSING_VERSION,
                       metadata={"archive": str(archive_root.relative_to(root))}, status="running")
        shutil.copy2(db_path, db_snapshot)

        archived = 0
        for source in sorted(directories["input"].iterdir(), key=lambda path: path.name.lower()):
            if not source.is_file() or source.name.startswith("."):
                continue
            destination = archive_input / source.name
            counter = 2
            while destination.exists():
                destination = archive_input / f"{source.stem}-{counter}{source.suffix}"
                counter += 1
            os.replace(source, destination)
            archived += 1

        retention = prepare_archive(archive_root, retention_rows)
        with db() as conn:
            persist_manifest(conn, retention["manifest"], root)
            record_artifact(conn, cleanup_id, retention["manifest"], root, "retention_manifest")
            record_artifact(conn, cleanup_id, db_snapshot, root, "database_backup")

        with db() as conn:
            working_boards = [row["id"] for row in conn.execute("SELECT id FROM boards WHERE status != 'approved'")]
            if working_boards:
                placeholders = ",".join("?" for _ in working_boards)
                photo_ids = [row["id"] for row in conn.execute(f"SELECT id FROM photos WHERE board_id IN ({placeholders})", working_boards)]
                if photo_ids:
                    photo_placeholders = ",".join("?" for _ in photo_ids)
                    conn.execute(f"DELETE FROM grouping_predictions WHERE photo_id IN ({photo_placeholders})", photo_ids)
                    conn.execute(f"DELETE FROM grouping_corrections WHERE photo_id IN ({photo_placeholders})", photo_ids)
                    conn.execute(f"DELETE FROM photos WHERE id IN ({photo_placeholders})", photo_ids)
                conn.execute(f"DELETE FROM boards WHERE id IN ({placeholders})", working_boards)
            conn.execute("DELETE FROM app_meta WHERE key='active_input_signature'")
            update_run(conn, cleanup_id, "complete", metadata={"archived": archived, "verified_derivatives": retention["verified"]})
            conn.commit()

        latest_run = directories["data"] / "crop-eval" / "true-test-set" / "latest-run.txt"
        if latest_run.exists():
            os.replace(latest_run, archive_root / "latest-run.txt")
        flash(f"Archived {archived} photos and created {retention['verified']} verified derivatives. The active workspace is ready for a new batch; originals remain in data/archive.", "success")
        return redirect(url_for("upload_photos"))

    @app.get("/board/<int:board_id>")
    def board(board_id):
        with db() as conn:
            board_row = conn.execute("SELECT * FROM boards WHERE id=?", (board_id,)).fetchone()
            photos = conn.execute("SELECT * FROM photos WHERE board_id=? ORDER BY COALESCE(capture_time, '9999'), source_path", (board_id,)).fetchall()
        if not board_row:
            return "Not found", 404
        has_card = any(photo["source_is_card"] for photo in photos)
        return render_template("board.html", board=board_row, photos=photos, has_card=has_card, cv2_available=_cv2_available())

    @app.get("/shuffleboard")
    def shuffleboard():
        retest = None
        # Saved evaluation reports are opt-in. Automatically selecting the
        # newest report made a real upload look like the old fixture whenever
        # camera filenames happened to match (for example IMG_5930.JPG).
        retest_requested = request.args.get("retest", "").lower() in {"1", "true", "yes"}
        retest_path = active_retest_path() if retest_requested else None
        retest_edits = {}
        retest_edits_path = retest_path.with_name(f"{retest_path.stem}-edits.json") if retest_path else None
        if retest_path and retest_path.exists():
            try:
                retest = json.loads(retest_path.read_text())
            except (OSError, json.JSONDecodeError):
                retest = None
        if retest_edits_path and retest_edits_path.exists():
            try:
                retest_edits = json.loads(retest_edits_path.read_text()).get("assignments", {})
            except (OSError, json.JSONDecodeError):
                retest_edits = {}
        with db() as conn:
            active_grouping = conn.execute("SELECT value FROM app_meta WHERE key='active_grouping_run_id'").fetchone()
            grouping_run_id = active_grouping["value"] if active_grouping else ""
            all_photos = conn.execute("SELECT p.*, b.label AS board_label FROM photos p JOIN boards b ON b.id=p.board_id WHERE p.source_path LIKE 'input/%' ORDER BY COALESCE(p.capture_time, '9999'), p.source_path").fetchall()
            current_sources = {row["source_path"] for row in all_photos}
            if retest:
                retest_sources = {source for group in retest.get("groups", []) for source in group.get("sources", [])}
                # A retest belongs to one exact batch. Never show its empty
                # board rows for a newer upload with different source files.
                if retest_sources != current_sources:
                    retest = None
                    retest_edits = {}
            if retest:
                by_source = {row["source_path"]: dict(row) for row in all_photos}
                photo_data = []
                boards = []
                for group in retest.get("groups", []):
                    retest_id = group["id"]
                    # Experiment/retest IDs stay in the data layer; the review UI
                    # should present ordinary board folders instead.
                    boards.append({"id": retest_id, "label": f"Board {len(boards) + 1:02d}", "sku": ""})
                    for source in group["sources"]:
                        photo = by_source.get(source)
                        if not photo:
                            continue
                        photo["board_id"] = retest_edits.get(str(photo["id"]), retest_id)
                        photo["board_label"] = retest_id
                        photo_data.append(photo)
                predictions = {str(photo["id"]): {"predicted_board_id": next((group["id"] for group in retest["groups"] if source in group["sources"]), "")} for source, photo in by_source.items()}
                retest_mode = True
            else:
                photos = all_photos
                boards = conn.execute("SELECT * FROM boards WHERE id IN (SELECT DISTINCT board_id FROM photos WHERE source_path LIKE 'input/%') ORDER BY id").fetchall()
                now = datetime.now().isoformat(timespec="seconds")
                for photo in photos:
                    # Predictions belong to the current grouping pass. The
                    # old INSERT OR IGNORE left prior-run board IDs in place,
                    # so a fresh grouping appeared to contain human moves
                    # even when the current model had just assigned the
                    # photo there.
                    conn.execute(
                        "INSERT INTO grouping_predictions(photo_id, predicted_board_id, predicted_label, predicted_at) VALUES (?,?,?,?) "
                        "ON CONFLICT(photo_id) DO UPDATE SET predicted_board_id=excluded.predicted_board_id, predicted_label=excluded.predicted_label, predicted_at=excluded.predicted_at",
                        (photo["id"], photo["board_id"], photo["board_label"], now),
                    )
                predictions = {str(row["photo_id"]): dict(row) for row in conn.execute("SELECT * FROM grouping_predictions")}
                photo_data = [dict(row) for row in photos]
                retest_mode = False
            conn.commit()
        group_counts = {}
        for photo in photo_data:
            group_counts[str(photo["board_id"])] = group_counts.get(str(photo["board_id"]), 0) + 1
        oversized_groups = {group_id: count for group_id, count in group_counts.items() if count > MAX_PHOTOS_PER_BOARD}
        composition_violations = {
            str(board["id"]): group_limit_violations([photo for photo in photo_data if str(photo["board_id"]) == str(board["id"])])
            for board in boards
        }
        composition_violations = {board_id: violations for board_id, violations in composition_violations.items() if violations}
        identity_missing = bool(photo_data) and bool(boards) and all(
            not any(board[name] for name in ("sku", "shaper", "model") if name in board.keys())
            for board in boards
        )
        ocr_status = {"runtime": {"available": False, "status": "not_run"}, "files_scanned": 0, "cards_detected": 0, "identifiers_extracted": 0}
        ocr_status_path = directories["data"] / "ocr-status.json"
        if ocr_status_path.exists():
            try:
                ocr_status.update(json.loads(ocr_status_path.read_text()))
            except (OSError, json.JSONDecodeError):
                ocr_status["runtime"] = {"available": False, "status": "status_unreadable"}
        # The persisted report describes the interpreter used for the last
        # grouping pass. Replace only its runtime probe with the interpreter
        # serving this page so an old failure cannot masquerade as current.
        ocr_status["runtime"] = ocr_runtime_status()
        return render_template("shuffleboard.html", boards=boards, photos=photo_data, predictions=predictions, retest=retest, retest_mode=retest_mode, retest_name=retest_path.name if retest_path else "", grouping_run_id=grouping_run_id, max_photos_per_board=MAX_PHOTOS_PER_BOARD, shot_limits=SHOT_LIMITS, oversized_groups=oversized_groups, composition_violations=composition_violations, ocr_status=ocr_status, identity_missing=identity_missing)

    @app.post("/shuffleboard/save")
    def save_shuffleboard():
        payload = request.get_json(silent=True) or {}
        assignments = payload.get("assignments") or {}
        if payload.get("mode") == "retest":
            with db() as conn:
                source_by_id = {str(row["id"]): row["source_path"] for row in conn.execute("SELECT id, source_path FROM photos WHERE source_path LIKE 'input/IMG_%'")}
            saved = {photo_id: str(board_id) for photo_id, board_id in assignments.items() if photo_id in source_by_id}
            active_report = active_retest_path()
            if not active_report:
                return jsonify({"error": "No retest report is available."}), 400
            path = active_report.with_name(f"{active_report.stem}-edits.json")
            path.write_text(json.dumps({"saved_at": datetime.now().isoformat(timespec="seconds"), "assignments": saved}, indent=2) + "\n")
            run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
            with db() as conn:
                create_run(conn, run_id, "true_crop_eval", pipeline_version=PROCESSING_VERSION,
                           metadata={"retest": True})
            crop_log = directories["data"] / "crop-eval" / "true-test-set" / f"{run_id}.log"
            crop_log.parent.mkdir(parents=True, exist_ok=True)
            log_stream = crop_log.open("w")
            try:
                worker = subprocess.Popen(
                    worker_command("true_crop_eval", run_id),
                    cwd=project_root, stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
            except OSError as exc:
                with db() as conn:
                    update_run(conn, run_id, "failed", error=str(exc))
                log_stream.close()
                return jsonify({"error": f"Could not start crop worker: {exc}"}), 500
            finally:
                log_stream.close()
            with db() as conn:
                set_run_worker(conn, run_id, pid=worker.pid, log_path=str(crop_log.relative_to(root)))
            return jsonify({"ok": True, "moved": len(saved), "retest": True, "crop_run": run_id, "redirect": url_for("annotate", set="true", run=run_id)})
        with db() as conn:
            allowed_photos = {str(row["id"]): row for row in conn.execute("SELECT id, board_id, shot_type, source_is_card FROM photos WHERE source_path LIKE 'input/%'")}
            allowed_boards = {row["id"] for row in conn.execute("SELECT id FROM boards")}
            new_board_ids = {str(value) for value in (payload.get("new_board_ids") or [])}
            new_board_map = {}
            for new_id in sorted(new_board_ids):
                if not new_id.startswith("new-"):
                    continue
                label = f"manual-group-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(new_board_map) + 1:02d}"
                created = conn.execute(
                    "INSERT INTO boards(label, status, created_at, processing_version) VALUES (?,?,?,?,?)",
                    (label, "unreviewed", datetime.now().isoformat(timespec="seconds"), PROCESSING_VERSION),
                )
                new_board_map[new_id] = int(created.lastrowid)
                allowed_boards.add(int(created.lastrowid))
            feedback_run_id = str(payload.get("grouping_run_id") or "")
            if not feedback_run_id:
                active_grouping = conn.execute("SELECT value FROM app_meta WHERE key='active_grouping_run_id'").fetchone()
                feedback_run_id = active_grouping["value"] if active_grouping else ""
            feedback = {str(row["photo_id"]): row for row in conn.execute("SELECT * FROM grouping_feedback WHERE run_id=?", (feedback_run_id,))} if feedback_run_id else {}
            final_assignments = payload.get("final_assignments") or payload.get("assignments") or {}
            normalized_assignments = {
                str(photo_id): new_board_map.get(str(board_id), board_id)
                for photo_id, board_id in final_assignments.items()
            }
            groups_for_validation: dict[int, list[dict]] = {}
            for photo_id, photo in allowed_photos.items():
                try:
                    board_id = int(normalized_assignments.get(photo_id, photo["board_id"]))
                except (TypeError, ValueError):
                    continue
                groups_for_validation.setdefault(board_id, []).append(dict(photo))
            composition_violations = {
                board_id: group_limit_violations(items)
                for board_id, items in groups_for_validation.items()
                if group_limit_violations(items)
            }
            model_limit_warnings = []
            if composition_violations:
                limits = {"total": MAX_PHOTOS_PER_BOARD, **SHOT_LIMITS}
                for board_id, violations in composition_violations.items():
                    details = ", ".join(f"{bucket.replace('_', ' ')}={count} (up to {limits.get(bucket, count)})" for bucket, count in violations.items())
                    model_limit_warnings.append(f"group {board_id}: {details}")
            counts = {}
            for photo_id, board_id in normalized_assignments.items():
                if photo_id in allowed_photos:
                    try:
                        counts[int(board_id)] = counts.get(int(board_id), 0) + 1
                    except (TypeError, ValueError):
                        continue
            oversized = {board_id: count for board_id, count in counts.items() if count > MAX_PHOTOS_PER_BOARD}
            for board_id, count in oversized.items():
                warning = f"group {board_id}: total={count} (up to {MAX_PHOTOS_PER_BOARD})"
                if warning not in model_limit_warnings:
                    model_limit_warnings.append(warning)
            now = datetime.now().isoformat(timespec="seconds")
            moved = 0
            # A photo moved onto a different board no longer matches the
            # capture-position slot it was assigned when grouping ran -
            # that position belonged to its old group's sequence, not this
            # one. Re-derive its shot_type from the TARGET board's existing
            # occupancy (by type, not raw order) so a moved-in photo fills
            # whichever protocol slot that board is actually missing,
            # instead of silently keeping a stale label from its old spot.
            target_board_shot_counts: dict[int, dict[str, int]] = {}

            def next_shot_type_slot(board_id: int) -> str:
                counts = target_board_shot_counts.get(board_id)
                if counts is None:
                    counts = {"full_board": 0, "side_profile": 0, "fin_detail": 0}
                    for row in conn.execute("SELECT shot_type FROM photos WHERE board_id=? AND source_is_card=0", (board_id,)):
                        if row["shot_type"] in counts:
                            counts[row["shot_type"]] += 1
                    target_board_shot_counts[board_id] = counts
                bucket = next_shot_type_for_occupancy(counts)
                counts[bucket] = counts.get(bucket, 0) + 1
                return bucket

            for photo_id, board_id in normalized_assignments.items():
                try:
                    board_id_int = int(board_id)
                except (TypeError, ValueError):
                    continue
                if str(photo_id) not in allowed_photos or board_id_int not in allowed_boards:
                    continue
                photo = allowed_photos[str(photo_id)]
                target = board_id_int
                initial = feedback.get(str(photo_id))
                initial_board_id = initial["initial_board_id"] if initial else photo["board_id"]
                if target != initial_board_id:
                    prediction = conn.execute("SELECT predicted_board_id FROM grouping_predictions WHERE photo_id=?", (int(photo_id),)).fetchone()
                    predicted_board_id = prediction["predicted_board_id"] if prediction else initial_board_id
                    conn.execute("INSERT INTO grouping_corrections(photo_id, predicted_board_id, corrected_board_id, corrected_at) VALUES (?,?,?,?) ON CONFLICT(photo_id) DO UPDATE SET corrected_board_id=excluded.corrected_board_id, corrected_at=excluded.corrected_at", (int(photo_id), predicted_board_id, target, now))
                if photo["board_id"] != target:
                    if photo["source_is_card"]:
                        conn.execute("UPDATE photos SET board_id=?, human_correction=? WHERE id=?", (target, "grouping_manual" if target != initial_board_id else None, int(photo_id)))
                    else:
                        new_shot_type = next_shot_type_slot(target)
                        conn.execute("UPDATE photos SET board_id=?, shot_type=?, human_correction=? WHERE id=?", (target, new_shot_type, "grouping_manual" if target != initial_board_id else None, int(photo_id)))
                    moved += 1
                if initial and target != initial_board_id:
                    target_row = conn.execute("SELECT label FROM boards WHERE id=?", (target,)).fetchone()
                    conn.execute("UPDATE grouping_feedback SET final_board_id=?, final_board_label=?, corrected=1, saved_at=? WHERE run_id=? AND photo_id=?", (target, target_row["label"] if target_row else "", now, feedback_run_id, int(photo_id)))
                elif initial:
                    conn.execute("UPDATE grouping_feedback SET final_board_id=initial_board_id, final_board_label=initial_board_label, corrected=0, saved_at=? WHERE run_id=? AND photo_id=?", (now, feedback_run_id, int(photo_id)))
            if feedback_run_id:
                conn.execute("UPDATE grouping_feedback_batches SET status='saved', saved_at=? WHERE run_id=?", (now, feedback_run_id))
            conn.commit()
        # Learning is deliberately downstream of the human save.  It is best
        # effort and never blocks the crop the user explicitly approved.
        learning_summary = {}
        if feedback_run_id:
            try:
                with db() as conn:
                    saved_feedback = [dict(row) for row in conn.execute(
                        "SELECT * FROM grouping_feedback WHERE run_id=?", (feedback_run_id,)
                    ).fetchall()]
                learning_summary = record_human_save(
                    directories["data"], feedback_run_id, saved_feedback,
                    {str(photo_id): board_id for photo_id, board_id in normalized_assignments.items()},
                )
            except Exception as exc:
                # A model update must never turn an approved crop into a
                # failed save. The next save can retry from the same snapshot.
                learning_summary = {"run_id": feedback_run_id, "error": str(exc)}
        run_id = datetime.now().strftime("batch-%Y%m%d-%H%M%S")
        with db() as conn:
            create_run(conn, run_id, "crop", pipeline_version=PROCESSING_VERSION,
                       input_signature=input_signature(), metadata={"moved": moved, "model_limit_warnings": model_limit_warnings, "learning": learning_summary})
        crop_root = directories["data"] / "crop-eval"
        crop_root.mkdir(parents=True, exist_ok=True)
        log_stream = (crop_root / f"{run_id}.log").open("w")
        try:
            worker = subprocess.Popen(
                worker_command("crop", run_id),
                cwd=project_root, stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError as exc:
            with db() as conn:
                update_run(conn, run_id, "failed", error=str(exc))
            log_stream.close()
            return jsonify({"error": f"Could not start crop worker: {exc}"}), 500
        finally:
            log_stream.close()
        with db() as conn:
            set_run_worker(conn, run_id, pid=worker.pid, log_path=str((crop_root / f"{run_id}.log").relative_to(root)))
        return jsonify({"ok": True, "moved": moved, "model_limit_warnings": model_limit_warnings, "learning": {"pairs_added": learning_summary.get("positive_pairs", 0) + learning_summary.get("negative_pairs", 0), "model_ready": learning_summary.get("model", {}).get("ready", False)}, "crop_run": run_id, "redirect": url_for("annotate", run=run_id)})

    @app.get("/crop-status/<run_id>")
    def crop_status(run_id):
        crop_root = directories["data"] / "crop-eval"
        true_mode = not (crop_root / "runs" / run_id).exists()
        if true_mode:
            crop_root = crop_root / "true-test-set"
        progress_path = crop_root / "runs" / run_id / "progress.json"
        if not progress_path.exists():
            with db() as conn:
                run = conn.execute("SELECT status, error FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run and run["status"] == "failed":
                return jsonify({"status": "failed", "error": run["error"] or "The crop worker failed before writing progress.", "done": 0, "total": 0, "complete": False})
            return jsonify({"status": "starting", "done": 0, "total": 0, "complete": False})
        try:
            progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            return jsonify({"status": "starting", "done": 0, "total": 0, "complete": False})
        log_path = crop_root / f"{run_id}.log"
        if not progress.get("complete") and log_path.exists():
            try:
                log_tail = log_path.read_text(errors="replace")[-4000:]
            except OSError:
                log_tail = ""
            if "Traceback (most recent call last)" in log_tail:
                progress["status"] = "failed"
                progress["error"] = "The crop worker stopped; start the crop run again."
                return jsonify(progress)
        report_path = crop_root / "runs" / run_id / "report.json"
        annotations_path = crop_root / "runs" / run_id / "annotations.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
                progress["items"] = []
                for item in report.get("cases", report.get("files", [])):
                    if item.get("exclude_from_listing"):
                        continue
                    source = item.get("source", item.get("file"))
                    progress["items"].append({
                        "source": source,
                        "source_url": url_for("media", relative=source),
                        "board": item.get("board", Path(source).parent.name),
                        "model_available": not manual_crop_required(item),
                        "manual_required": manual_crop_required(item),
                        "width": item["source_size"]["width"],
                        "height": item["source_size"]["height"],
                        "proposal": item.get("proposal", item.get("crop")),
                        "method": item.get("method", ""),
                        "classification": item.get("classification", {}),
                        "shot_type": item.get("evaluated_shot_type", item.get("shot_type", "")),
                        "rotation": item.get("rotation", {"angle": 0.0, "applied": False}),
                        "annotation": annotations.get(source, {}),
                        "dataset": f"{'true' if true_mode else 'batch'}:{run_id}",
                    })
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        progress["status"] = "complete" if progress.get("complete") else "running"
        return jsonify(progress)

    @app.get("/annotate")
    def annotate():
        true_mode = request.args.get("set") == "true"
        requested_run = request.args.get("run")
        report_path, annotations_path, run_id = crop_run_paths(true_mode, requested_run)
        if not report_path.exists():
            return render_template("annotate.html", items=[], board_groups={}, true_mode=true_mode, processing_run=(run_id if run_id and run_id != "legacy" else None), error=None if run_id and run_id != "legacy" else "No crop run is available yet.")
        report = json.loads(report_path.read_text())
        annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
        items = []
        report_items = report.get("cases", report.get("files", []))
        for item in report_items:
            if not true_mode and item["status"] != "proposal_generated":
                continue
            if true_mode and item.get("exclude_from_listing"):
                continue
            source = item.get("source", item.get("file"))
            saved = annotations.get(source, {})
            board = Path(source).parent.name if true_mode else item.get("board", "unassigned")
            crop_relative = saved.get("annotated_crop") or (item.get("proposal_path", "") if true_mode else item.get("crop_preview", ""))
            # Older annotations stored the rectangle but not a crop JPEG. Materialize
            # that crop on read so the Original/Cropped toggle reflects the saved edit.
            if saved.get("expected_crop") and not saved.get("annotated_crop"):
                source_path = root / source if (true_mode or source.startswith("input/")) else root / "sample-pack" / "usedsurf-crop-sample-pack" / source
                if source_path.is_file():
                    annotated_dir = directories["data"] / "crop-eval" / ("true-test-set/annotated-crops" if true_mode else f"annotated-crops/{safe_name(board)}")
                    annotated_dir.mkdir(parents=True, exist_ok=True)
                    annotated_path = annotated_dir / f"{Path(source).stem}.jpg"
                    image, _ = read_image(source_path)
                    crop = saved["expected_crop"]
                    image.crop((crop["x"], crop["y"], crop["x"] + crop["width"], crop["y"] + crop["height"])).save(annotated_path, "JPEG", quality=95, subsampling=0)
                    crop_relative = str(annotated_path.relative_to(root))
            items.append({
                "source": source,
                "source_url": url_for("media", relative=source if true_mode or source.startswith("input/") else f"sample-pack/usedsurf-crop-sample-pack/{source}"),
                "crop_url": url_for("media", relative=crop_relative),
                "board": board,
                "model_available": not manual_crop_required(item),
                "manual_required": manual_crop_required(item),
                "width": item.get("source_size", {"width": item.get("width")})["width"], "height": item.get("source_size", {"height": item.get("height")})["height"], "proposal": item.get("proposal", item.get("crop")),
                "method": item.get("method", ""), "classification": item.get("classification", {}), "shot_type": item.get("evaluated_shot_type", item.get("shot_type", "")), "cases": item.get("cases", []), "rotation": item.get("rotation", {"angle": 0.0, "applied": False}), "annotation": saved, "dataset": f"true:{run_id}" if true_mode else f"batch:{run_id or 'latest'}",
                # Debug reference for pasting into a bug report: the overlay
                # is the "mutated" image (crop box drawn on it), distinct
                # from source_url (the untouched original) - see
                # run_batch_crop.py's overlay filename tagging.
                "overlay_url": url_for("media", relative=item["overlay_path"]) if item.get("overlay_path") else None,
            })
        board_groups = {}
        for item in items:
            item["board"] = item.get("board", "unassigned")
            board_groups.setdefault(item["board"], []).append(item)
        all_reviewed = bool(items) and all(reviewed_crop(item, item["annotation"]) for item in items)
        return render_template("annotate.html", items=items, board_groups=board_groups, true_mode=true_mode, processing_run=(run_id if true_mode and run_id != "legacy" else None), all_reviewed=all_reviewed, error=None)

    @app.get("/bigcommerce")
    def bigcommerce():
        """Present reviewed listing crops in board folders for marketplace upload."""
        true_mode = request.args.get("set") == "true"
        report_path, annotations_path, run_id = crop_run_paths(true_mode, request.args.get("run"))
        if not report_path.exists():
            return render_template("bigcommerce.html", groups=[], complete=False, reviewed=0, total=0, true_mode=true_mode, run_id=run_id, error="No crop run is available yet.")
        try:
            report = json.loads(report_path.read_text())
            annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            return render_template("bigcommerce.html", groups=[], complete=False, reviewed=0, total=0, true_mode=true_mode, run_id=run_id, error="The crop run could not be read.")
        groups = {}
        reviewed = 0
        total = 0
        report_items = report.get("cases", report.get("files", []))
        for item in report_items:
            if true_mode and item.get("exclude_from_listing"):
                continue
            if not true_mode and item.get("status") != "proposal_generated":
                continue
            source = item.get("source", item.get("file"))
            annotation = annotations.get(source, {})
            total += 1
            if not reviewed_crop(item, annotation):
                continue
            reviewed += 1
            crop_relative = annotation.get("annotated_crop") or (item.get("proposal_path", "") if true_mode else item.get("crop_preview", ""))
            crop_path = root / crop_relative
            if not crop_relative or not crop_path.is_file():
                continue
            board = Path(source).parent.name if true_mode else item.get("board", "unassigned")
            group = groups.setdefault(board, {"name": board, "items": []})
            group["items"].append({
                "id": f"{board}:{source}",
                "source": source,
                "filename": Path(crop_relative).name,
                "image_url": url_for("media", relative=crop_relative),
                "download_path": crop_relative,
                "shot_type": item.get("evaluated_shot_type", item.get("shot_type", "listing photo")),
            })
        group_list = sorted(groups.values(), key=lambda group: group["name"].lower())
        complete = total > 0 and reviewed == total
        return render_template("bigcommerce.html", groups=group_list, complete=complete, reviewed=reviewed, total=total, true_mode=true_mode, run_id=run_id, error=None)

    @app.post("/bigcommerce/download")
    def bigcommerce_download():
        """Download selected reviewed crops with board folders preserved."""
        payload = request.get_json(silent=True) or {}
        selected = payload.get("selected") or []
        if not isinstance(selected, list) or not selected:
            return jsonify({"error": "Select at least one image."}), 400
        archive = BytesIO()
        added = 0
        crop_root = (directories["data"] / "crop-eval").resolve()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for item in selected:
                if not isinstance(item, dict):
                    continue
                relative = str(item.get("path", ""))
                board = safe_name(str(item.get("board", "unassigned")))
                source = (root / relative).resolve()
                if crop_root not in source.parents or not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg"}:
                    continue
                bundle.write(source, f"{board}/{safe_name(source.name)}")
                added += 1
        if not added:
            return jsonify({"error": "No valid reviewed crops were selected."}), 400
        archive.seek(0)
        return send_file(archive, mimetype="application/zip", as_attachment=True, download_name="usedsurf-bigcommerce-images.zip")

    @app.get("/crop-stack")
    def crop_stack():
        """Show proposed listing crops together for framing consistency review."""
        true_mode = request.args.get("set", "true") == "true"
        full_only = request.args.get("shot") == "full_board"
        report_path, _, run_id = crop_run_paths(true_mode, request.args.get("run"))
        if not report_path.exists():
            return render_template("crop_stack.html", items=[], true_mode=true_mode, full_only=full_only, error="Run the crop evaluator first.")
        report = json.loads(report_path.read_text())
        report_items = report["cases"] if true_mode else report["files"]
        items = []
        group_number = 0
        for item in report_items:
            if true_mode:
                if item.get("exclude_from_listing"):
                    group_number += 1
                    continue
                if not item.get("proposal_path"):
                    continue
                shot_type = item.get("evaluated_shot_type", item.get("shot_type", "unknown"))
                crop_relative = item["proposal_path"]
                source = item["source"]
            else:
                if item.get("status") != "proposal_generated":
                    continue
                shot_type = item.get("shot_type", "unknown")
                crop_relative = item.get("crop_preview", "")
                source = item["file"]
            if full_only and shot_type != "full_board":
                continue
            items.append({"source": source, "shot_type": shot_type, "image_url": url_for("media", relative=crop_relative), "review": bool(item.get("review", False)), "group": group_number if true_mode else "all"})
        return render_template("crop_stack.html", items=items, true_mode=true_mode, full_only=full_only, error=None)

    @app.get("/crop-stack/silhouette.png")
    def silhouette_overlay():
        """Overlay only full-board silhouettes in normalized crop coordinates."""
        if cv2 is None or np is None:
            return jsonify({"error": "OpenCV is unavailable."}), 503
        true_mode = request.args.get("set", "true") == "true"
        try:
            group_number = int(request.args.get("group", "0"))
        except ValueError:
            return jsonify({"error": "Invalid board group."}), 400
        report_path, _, _ = crop_run_paths(true_mode, request.args.get("run"))
        if not report_path.exists():
            return jsonify({"error": "Run the crop evaluator first."}), 404
        report = json.loads(report_path.read_text())
        selected = []
        current_group = 0
        for item in report["cases"] if true_mode else report["files"]:
            if true_mode and item.get("exclude_from_listing"):
                current_group += 1
                continue
            shot_type = item.get("evaluated_shot_type", item.get("shot_type", ""))
            if current_group == group_number and shot_type == "full_board" and item.get("proposal"):
                selected.append(item)
        canvas_width, canvas_height = 460, 760
        canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
        colors = [(31, 120, 180, 120), (218, 112, 38, 120), (45, 155, 86, 120), (145, 76, 170, 120)]
        for index, item in enumerate(selected):
            source_path = root / item["source"]
            if not source_path.is_file():
                continue
            image, _ = read_image(source_path)
            array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            mask = _silhouette_mask(array)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            candidate = item.get("classification", {}).get("candidate") or item["proposal"]
            center_x = (candidate[0] + candidate[2] / 2) / image.width
            center_y = (candidate[1] + candidate[3] / 2) / image.height
            mask_h, mask_w = mask.shape
            def contour_score(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = (x + w / 2) / mask_w, (y + h / 2) / mask_h
                return h * w - abs(cx - center_x) * image.width * image.height * 0.15 - abs(cy - center_y) * image.width * image.height * 0.05
            contour = max(contours, key=contour_score)
            points = contour.reshape(-1, 2).astype(float)
            crop = item["proposal"]
            points[:, 0] = points[:, 0] * image.width / mask_w
            points[:, 1] = points[:, 1] * image.height / mask_h
            points[:, 0] = (points[:, 0] - crop["x"]) / max(1, crop["width"]) * canvas_width
            points[:, 1] = (points[:, 1] - crop["y"]) / max(1, crop["height"]) * canvas_height
            points = np.round(points).astype(int).tolist()
            if len(points) >= 3:
                layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(layer)
                color = colors[index % len(colors)]
                draw.polygon(points, fill=color)
                draw.line(points + [points[0]], fill=(color[0], color[1], color[2], 230), width=3, joint="curve")
                canvas = Image.alpha_composite(canvas, layer)
        output = BytesIO()
        canvas.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png", download_name=f"group-{group_number:02d}-silhouette.png")

    @app.get("/labeler")
    def labeler():
        manifest_path = directories["data"] / "shot-labels" / "manifest.json"
        if not manifest_path.exists():
            return render_template("labeler.html", cases=[], labeled=0, total=0, error="Run the local UsedSurf importer first.")
        try:
            manifest = json.loads(manifest_path.read_text())
            all_cases = manifest.get("cases", [])
            prediction_path = directories["data"] / "shot-labels" / "visual-predictions.json"
            if prediction_path.exists():
                try:
                    visual_predictions = json.loads(prediction_path.read_text()).get("predictions", [])
                    by_id = {item.get("id"): item for item in visual_predictions}
                    for case in all_cases:
                        prediction = by_id.get(case.get("id"))
                        if prediction:
                            case["model_prediction"] = prediction.get("model_prediction")
                            case["model_confidence"] = prediction.get("model_confidence")
                            case["composition_hint"] = prediction.get("composition_hint")
                            case["needs_review"] = prediction.get("needs_review", True)
                            case["auto_label_eligible"] = prediction.get("auto_label_eligible", False)
                except (OSError, json.JSONDecodeError):
                    pass
            queue = request.args.get("queue", "")
            cases = [case for case in all_cases if case.get("needs_review") and case.get("label_source") != "human"] if queue == "needs_review" else all_cases
            labeled = sum(1 for case in all_cases if case.get("label_source") == "human" or case.get("status") == "labeled")
            queue_label = "Needs Review" if queue == "needs_review" else "All photos"
        except (OSError, json.JSONDecodeError):
            return render_template("labeler.html", cases=[], labeled=0, total=0, error="The shot-label manifest is not valid JSON.")
        return render_template("labeler.html", cases=cases, labeled=labeled, total=len(all_cases), queue_label=queue_label, needs_review_count=sum(1 for case in all_cases if case.get("needs_review") and case.get("label_source") != "human"), error=None)

    @app.post("/labeler/save")
    def save_shot_label():
        payload = request.get_json(silent=True) or {}
        manifest_path = directories["data"] / "shot-labels" / "manifest.json"
        if not manifest_path.exists():
            return jsonify({"error": "Shot-label manifest is missing."}), 400
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return jsonify({"error": "Shot-label manifest is invalid."}), 400
        allowed = {"full_board", "side_profile", "fin_detail", "skip"}
        case = next((item for item in manifest.get("cases", []) if item.get("id") == payload.get("id")), None)
        if not case:
            return jsonify({"error": "Unknown label case."}), 404
        label = payload.get("label")
        if label is not None and label not in allowed:
            return jsonify({"error": "Unknown shot label."}), 400
        if label is None:
            position = case.get("gallery_position")
            label = _ordered_shot_label(int(position)) if position else None
            case["label_source"] = "gallery_order" if label else None
            case["status"] = "auto_labeled" if label else "unlabeled"
            case.pop("label_decision", None)
        else:
            case["label_source"] = "human"
            case["status"] = "labeled"
            case["label_decision"] = payload.get("decision", "corrected")
        case["label_updated_at"] = datetime.now().isoformat(timespec="seconds")
        case["label"] = label
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return jsonify({"ok": True, "label": case.get("label"), "label_source": case.get("label_source"), "labeled": sum(1 for item in manifest.get("cases", []) if item.get("label_source") == "human" or item.get("status") == "labeled")})

    @app.post("/annotate/save")
    def save_annotation():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source", "")
        dataset = payload.get("dataset", "")
        # Older review tabs received "true:batch-*" from crop_status even
        # though those photos came from a batch run. Keep their saves working.
        true_mode = (dataset.startswith("true:") or dataset == "true") and not dataset.startswith("true:batch-")
        if dataset.startswith("true:"):
            run_id = dataset.split(":", 1)[1]
        elif dataset.startswith("batch:"):
            token = dataset.split(":", 1)[1]
            if token and token != "latest":
                run_id = token
            else:
                latest_pointer = directories["data"] / "crop-eval" / "latest-run.txt"
                run_id = latest_pointer.read_text().strip() if latest_pointer.is_file() else None
        else:
            run_id = None
        report_path, annotations_path, _ = crop_run_paths(true_mode, run_id)
        if not report_path.exists() and not true_mode:
            # A review tab can outlive its crop run. Recover against the latest
            # batch only when it contains the same source photo below.
            report_path, annotations_path, _ = crop_run_paths(False)
        if not report_path.exists():
            return jsonify({"error": "No crop report is available. Start crop review from the grouping page."}), 409
        report = json.loads(report_path.read_text())
        allowed = {item["source"]: item for item in report["cases"]} if true_mode else {item["file"]: item for item in report["files"]}
        if source not in allowed and not true_mode:
            latest_report, latest_annotations, _ = crop_run_paths(False)
            if latest_report != report_path and latest_report.is_file():
                latest_data = json.loads(latest_report.read_text())
                latest_allowed = {item["file"]: item for item in latest_data["files"]}
                if source in latest_allowed:
                    report_path, annotations_path, report, allowed = latest_report, latest_annotations, latest_data, latest_allowed
        if source not in allowed:
            return jsonify({"error": "This photo is not in the current crop run. Reopen crop review and try again."}), 409
        item = allowed[source]
        crop = payload.get("expected_crop") or {}
        try:
            values = {key: int(crop[key]) for key in ("x", "y", "width", "height")}
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Expected crop must include integer x, y, width, and height."}), 400
        # "source" is already a real repo-relative path ("input/...") for
        # both the true-crop-eval flow and real batch runs - only the
        # legacy sample-pack fixture flow needs that extra prefix (same
        # signal annotate() uses to build source_url).
        source_path = root / source if (true_mode or source.startswith("input/")) else root / "sample-pack" / "usedsurf-crop-sample-pack" / source
        if not source_path.is_file():
            return jsonify({"error": "Crop source is missing."}), 400
        rotation = float(payload.get("rotation", 0.0) or 0.0)
        decision = payload.get("decision", "manual")
        if manual_crop_required(item) and decision != "manual":
            return jsonify({"error": "The automatic crop failed. Draw and save a manual crop for this photo."}), 409
        source_size = item.get("source_size") or item
        if manual_crop_required(item) and not rotation and source_size.get("width") and source_size.get("height") and values == {"x": 0, "y": 0, "width": int(source_size["width"]), "height": int(source_size["height"])}:
            return jsonify({"error": "Draw a smaller crop so the full photo is not exported unchanged."}), 400
        model_crop = item["proposal"] if true_mode else item["crop"]
        # "Good" with the model's own unchanged rectangle just re-derives a
        # file the batch crop worker already produced - re-reading the
        # multi-megapixel source, re-doing its EXIF transpose, and
        # re-encoding at quality=95/subsampling=0 (the slowest JPEG mode)
        # on every click is pure duplicate work for far and away the most
        # common click in this review flow. Point at that existing file
        # instead whenever nothing actually changed.
        # Not gated on decision=="good": the canvas also auto-fires a
        # "manual" save on any click/drag (see annotate.html's pointerup
        # handler), including ones that just re-trace the model's own green
        # box without actually changing it. Whether the request arrived via
        # "Good" or "Save my crop", identical values mean nothing actually
        # changed, so the same fast path applies either way.
        existing_crop = root / (item.get("proposal_path") if true_mode else item.get("crop_preview", "")) if (true_mode and item.get("proposal_path")) or (not true_mode and item.get("crop_preview")) else None
        reuse_existing = (
            not rotation
            and model_crop
            and values == {key: int(model_crop[key]) for key in ("x", "y", "width", "height")}
            and existing_crop is not None
            and existing_crop.is_file()
        )
        if reuse_existing:
            annotated_path = existing_crop
        else:
            image, _ = read_image(source_path)
            # Batch and fixture proposals both use expanded rotated-image
            # coordinates when rotation is applied. Save against that same
            # canvas or a valid manual crop can be rejected or misplaced.
            working = image.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC) if rotation else image
            if values["x"] < 0 or values["y"] < 0 or values["width"] < 1 or values["height"] < 1 or values["x"] + values["width"] > working.width or values["y"] + values["height"] > working.height:
                return jsonify({"error": "Expected crop must stay inside the source image."}), 400
            board = Path(source).parent.name if true_mode else item.get("board", "unassigned")
            annotated_dir = directories["data"] / "crop-eval" / ("true-test-set/annotated-crops" if true_mode else f"annotated-crops/{safe_name(board)}")
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = annotated_dir / f"{Path(source).stem}.jpg"
            working.crop((values["x"], values["y"], values["x"] + values["width"], values["y"] + values["height"])).save(annotated_path, "JPEG", quality=95, subsampling=0)
        annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
        annotations[source] = {"source": source, "expected_crop": values, "model_crop": model_crop, "user_crop": None if decision == "good" else values, "decision": decision, "rotation": rotation, "shot_type": payload.get("shot_type") or item.get("evaluated_shot_type", ""), "failure_labels": payload.get("failure_labels", []), "notes": payload.get("notes", "").strip(), "updated_at": datetime.now().isoformat(timespec="seconds"), "proposed_crop": model_crop, "annotated_crop": str(annotated_path.relative_to(root))}
        fd, temp_name = tempfile.mkstemp(prefix="annotations-", suffix=".json", dir=annotations_path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(annotations, stream, indent=2)
                stream.write("\n")
            os.replace(temp_name, annotations_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return jsonify({"ok": True, "saved": len(annotations)})

    @app.post("/board/<int:board_id>")
    def save_board(board_id):
        with db() as conn:
            conn.execute("UPDATE boards SET shaper=?, model=?, sku=?, fin_system=?, fin_box_count=?, fins_included=?, fins_present_count=?, label=?, status='reviewing' WHERE id=?", (request.form.get("shaper", "").strip(), request.form.get("model", "").strip(), request.form.get("sku", "").strip(), request.form.get("fin_system", "").strip(), request.form.get("fin_box_count", "").strip(), request.form.get("fins_included", "").strip(), request.form.get("fins_present_count", "").strip(), request.form.get("sku", "needs-review").strip() or "needs-review", board_id))
            for photo in conn.execute("SELECT id, width, height FROM photos WHERE board_id=?", (board_id,)).fetchall():
                prefix = f"photo_{photo['id']}_"
                if prefix + "x" not in request.form:
                    continue
                shot = request.form.get(prefix + "shot_type", "full_board")
                crop = [int(float(request.form.get(prefix + key, 0))) for key in ("x", "y", "w", "h")]
                crop[0] = max(0, min(crop[0], photo["width"] or crop[0])); crop[1] = max(0, min(crop[1], photo["height"] or crop[1]))
                crop[2] = max(1, min(crop[2], (photo["width"] or crop[2]) - crop[0])); crop[3] = max(1, min(crop[3], (photo["height"] or crop[3]) - crop[1]))
                conn.execute("UPDATE photos SET shot_type=?, crop_x=?, crop_y=?, crop_w=?, crop_h=?, human_correction=?, review_status='reviewed' WHERE id=?", (shot, *crop, "manual" if request.form.get(prefix + "changed") else None, photo["id"]))
            conn.commit()
        flash("Board review saved. It remains unapproved until you export it.", "success")
        return redirect(url_for("board", board_id=board_id))

    @app.post("/board/<int:board_id>/approve")
    def approve(board_id):
        with db() as conn:
            board_row = conn.execute("SELECT * FROM boards WHERE id=?", (board_id,)).fetchone()
            photos = conn.execute("SELECT * FROM photos WHERE board_id=? ORDER BY COALESCE(capture_time, '9999'), source_path", (board_id,)).fetchall()
            if not board_row or not board_row["shaper"] or not board_row["model"] or not board_row["sku"]:
                flash("Enter shaper, model, and SKU before approving; use 'unknown' explicitly if needed.", "error")
                return redirect(url_for("board", board_id=board_id))
            destination = export_board(root, board_row, photos)
            conn.execute("UPDATE boards SET status='approved', approved_at=? WHERE id=?", (datetime.now().isoformat(timespec="seconds"), board_id))
            conn.execute("UPDATE photos SET review_status='approved' WHERE board_id=?", (board_id,))
            conn.commit()
        flash(f"Approved and exported to {destination.relative_to(root)}", "success")
        return redirect(url_for("upload_photos"))

    @app.get("/media/<path:relative>")
    def media(relative):
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            return "Not found", 404
        return send_file(path)

    @app.get("/media-thumb/<path:relative>")
    def media_thumb(relative):
        """Serve a small cached preview so folder views never decode originals."""
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            return "Not found", 404
        cache_root = directories["data"] / "media-thumbs"
        cache_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{path}:{path.stat().st_mtime_ns}".encode()).hexdigest()
        cached = cache_root / f"{key}.jpg"
        if not cached.exists():
            image, _ = read_image(path)
            image.thumbnail((360, 480))
            image.convert("RGB").save(cached, "JPEG", quality=82, optimize=True)
        return send_file(cached, mimetype="image/jpeg", max_age=86400)

    return app


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False
