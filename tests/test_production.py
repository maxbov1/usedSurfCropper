import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from organizer.db import connect, create_run, recover_stale_runs, set_run_worker, update_run
from organizer.retention import compact_archive, prepare_archive


class ProductionLifecycleTests(unittest.TestCase):
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
            self.assertEqual(db.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0], "3")
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
