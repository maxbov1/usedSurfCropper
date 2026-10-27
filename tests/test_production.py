import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from organizer.db import CURRENT_SCHEMA_VERSION, connect, create_run, recover_stale_runs, set_run_worker, update_run
from organizer.retention import compact_archive, prepare_archive
from organizer.ingest import PROCESSING_VERSION
from organizer.web import create_app


class ProductionLifecycleTests(unittest.TestCase):
    def test_failed_machine_crop_requires_manual_edit_before_export(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = create_app(root)
            Image.new("RGB", (64, 80), "white").save(root / "input" / "board.jpg")
            run = root / "data" / "crop-eval" / "runs" / "batch-1"
            run.mkdir(parents=True)
            (run / "progress.json").write_text(json.dumps({"complete": True, "done": 1, "total": 1}))
            (run / "report.json").write_text(json.dumps({"files": [{"file": "input/board.jpg", "status": "proposal_generated", "board": "board-1", "source_size": {"width": 64, "height": 80}, "crop": {"x": 0, "y": 0, "width": 64, "height": 80}, "model_available": True, "method": "no usable board boundary found; manual crop required"}]}))
            client = app.test_client()
            self.assertFalse(client.get("/crop-status/batch-1").json["items"][0]["model_available"])
            self.assertIn('"manual_required": true', client.get("/annotate?run=batch-1").get_data(as_text=True))
            payload = {"source": "input/board.jpg", "dataset": "batch:batch-1", "expected_crop": {"x": 0, "y": 0, "width": 64, "height": 80}}
            self.assertEqual(client.post("/annotate/save", json={**payload, "decision": "good"}).status_code, 409)
            self.assertEqual(client.post("/annotate/save", json={**payload, "decision": "manual"}).status_code, 400)
            (run / "annotations.json").write_text(json.dumps({"input/board.jpg": {"decision": "manual", "rotation": 0, "expected_crop": payload["expected_crop"]}}))
            self.assertIn("0 of 1", client.get("/bigcommerce?run=batch-1").get_data(as_text=True))
            payload["expected_crop"] = {"x": 5, "y": 6, "width": 48, "height": 60}
            self.assertEqual(client.post("/annotate/save", json={**payload, "decision": "manual"}).status_code, 200)
            self.assertIn("All 1 listing images", client.get("/bigcommerce?run=batch-1").get_data(as_text=True))

    def test_polled_batch_photo_and_old_tab_save_to_their_own_run(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = create_app(root)
            photo = root / "input" / "board.jpg"
            Image.new("RGB", (64, 64), "white").save(photo)
            crop_root = root / "data" / "crop-eval"
            old_run = crop_root / "runs" / "batch-old"
            old_run.mkdir(parents=True)
            (old_run / "progress.json").write_text(json.dumps({"done": 1, "total": 1, "complete": True}))
            (old_run / "report.json").write_text(json.dumps({"files": [{"file": "input/board.jpg", "board": "board-1", "source_size": {"width": 64, "height": 64}, "crop": {"x": 0, "y": 0, "width": 64, "height": 64}}]}))
            (crop_root / "latest-run.txt").write_text("batch-new\n")
            client = app.test_client()
            polled = client.get("/crop-status/batch-old")
            self.assertEqual(polled.status_code, 200)
            self.assertEqual(polled.json["items"][0]["dataset"], "batch:batch-old")
            # A page opened before the poller fix still sends this old tag.
            saved = client.post("/annotate/save", json={"source": "input/board.jpg", "dataset": "true:batch-old", "expected_crop": {"x": 4, "y": 5, "width": 40, "height": 41}, "decision": "manual"})
            self.assertEqual(saved.status_code, 200, saved.json)
            self.assertEqual(json.loads((old_run / "annotations.json").read_text())["input/board.jpg"]["decision"], "manual")

    def test_manual_crop_saves_against_latest_run_when_review_tab_has_no_run(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = create_app(root)
            photo = root / "input" / "board.jpg"
            Image.new("RGB", (64, 64), "white").save(photo)
            crop_root = root / "data" / "crop-eval"
            run = crop_root / "runs" / "batch-1"
            run.mkdir(parents=True)
            (crop_root / "latest-run.txt").write_text("batch-1\n")
            (run / "report.json").write_text(json.dumps({"files": [{"file": "input/board.jpg", "board": "board-1", "crop": {"x": 0, "y": 0, "width": 64, "height": 64}}]}))
            client = app.test_client()
            for dataset in ("", "batch:missing-run"):
                result = client.post("/annotate/save", json={"source": "input/board.jpg", "dataset": dataset, "expected_crop": {"x": 4, "y": 5, "width": 40, "height": 41}, "decision": "manual"})
                self.assertEqual(result.status_code, 200, result.json)
                self.assertTrue(result.json["ok"])
            saved = json.loads((run / "annotations.json").read_text())["input/board.jpg"]
            self.assertEqual(saved["decision"], "manual")
            self.assertEqual(saved["expected_crop"], {"x": 4, "y": 5, "width": 40, "height": 41})
            self.assertTrue((root / saved["annotated_crop"]).is_file())

    def test_rotated_batch_crop_uses_the_rotated_canvas(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = create_app(root)
            photo = root / "input" / "board.jpg"
            Image.new("RGB", (64, 64), "white").save(photo)
            run = root / "data" / "crop-eval" / "runs" / "batch-1"
            run.mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"files": [{"file": "input/board.jpg", "board": "board-1", "crop": {"x": 0, "y": 0, "width": 64, "height": 64}}]}))
            rotated = Image.open(photo).rotate(6, expand=True, resample=Image.Resampling.BICUBIC)
            self.assertGreater(rotated.width, 64)
            crop = {"x": 64, "y": 20, "width": 4, "height": 12}
            result = app.test_client().post("/annotate/save", json={"source": "input/board.jpg", "dataset": "batch:batch-1", "expected_crop": crop, "rotation": 6, "decision": "manual"})
            self.assertEqual(result.status_code, 200, result.json)
            saved = json.loads((run / "annotations.json").read_text())["input/board.jpg"]
            with Image.open(root / saved["annotated_crop"]) as output:
                self.assertEqual(output.size, (4, 12))

    def test_current_batch_opens_review_without_starting_another_grouping(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = create_app(root)
            photo = root / "input" / "board.jpg"
            Image.new("RGB", (64, 64), "white").save(photo)
            stat = photo.stat()
            digest = hashlib.sha256()
            digest.update(PROCESSING_VERSION.encode())
            digest.update(f"{photo.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
            signature = digest.hexdigest()
            with connect(root / "data" / "usedsurf.sqlite3") as conn:
                conn.execute("INSERT INTO app_meta(key,value) VALUES('active_input_signature',?)", (signature,))
                conn.execute("INSERT INTO grouping_feedback_batches(run_id,input_signature,pipeline_version,created_at,status) VALUES(?,?,?,?,?)", ("group-1", signature, PROCESSING_VERSION, "2026-10-19", "proposed"))
                conn.commit()
            (root / "data" / "ocr-status.json").write_text(json.dumps({"runtime": {"available": True}}))
            client = app.test_client()
            page = client.get("/upload-photos")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Review groups", page.data)
            self.assertNotIn(b'id="continue-form"', page.data)
            result = client.post("/process", headers={"Accept": "application/json"})
            self.assertEqual(result.json, {"ok": True, "already_grouped": True, "redirect": "/shuffleboard"})
            with connect(root / "data" / "usedsurf.sqlite3") as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)

    def test_run_records_capture_worker_and_recover_dead_process(self):
        with TemporaryDirectory() as folder:
            db = connect(Path(folder) / "usedsurf.sqlite3")
            create_run(db, "crop-1", "crop", status="running")
            set_run_worker(db, "crop-1", pid=999999, log_path="data/crop.log")
            self.assertEqual(recover_stale_runs(db), ["crop-1"])
            row = db.execute("SELECT status, worker_pid, log_path FROM runs WHERE run_id='crop-1'").fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["worker_pid"], 999999)
            self.assertEqual(row["log_path"], "data/crop.log")

    def test_metadata_and_schema_version_are_persisted(self):
        with TemporaryDirectory() as folder:
            db = connect(Path(folder) / "usedsurf.sqlite3")
            create_run(db, "group-1", "grouping", status="running")
            update_run(db, "group-1", "complete", metadata={"groups": 3})
            self.assertEqual(db.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0], str(CURRENT_SCHEMA_VERSION))
            self.assertEqual(db.execute("SELECT metadata_json FROM runs WHERE run_id='group-1'").fetchone()[0], '{"groups": 3}')

    def test_retention_dry_run_preserves_source_until_apply(self):
        with TemporaryDirectory() as folder:
            archive = Path(folder) / "archive"
            (archive / "input").mkdir(parents=True)
            Image.new("RGB", (2400, 1200), "white").save(archive / "input" / "board.jpg", quality=95)
            result = prepare_archive(archive, [])
            self.assertEqual(result["verified"], 1)
            self.assertEqual(compact_archive(archive)["planned"], 1)
            self.assertTrue((archive / "input" / "board.jpg").exists())
            self.assertEqual(compact_archive(archive, apply=True)["removed"], 1)
            self.assertFalse((archive / "input" / "board.jpg").exists())


if __name__ == "__main__":
    unittest.main()
