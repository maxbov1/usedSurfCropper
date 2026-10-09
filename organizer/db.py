from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


CURRENT_SCHEMA_VERSION = 3


SCHEMA = """
CREATE TABLE IF NOT EXISTS boards (
  id INTEGER PRIMARY KEY, label TEXT NOT NULL, shaper TEXT DEFAULT '', model TEXT DEFAULT '',
  sku TEXT DEFAULT '', fin_system TEXT DEFAULT '', fins_included TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'unreviewed', created_at TEXT NOT NULL,
  approved_at TEXT, processing_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
  id INTEGER PRIMARY KEY, board_id INTEGER NOT NULL REFERENCES boards(id), source_path TEXT NOT NULL,
  content_hash TEXT NOT NULL, capture_time TEXT, width INTEGER, height INTEGER, shot_type TEXT NOT NULL,
  original_prediction TEXT, human_correction TEXT, crop_x INTEGER, crop_y INTEGER, crop_w INTEGER, crop_h INTEGER,
  crop_padding REAL NOT NULL DEFAULT 0.10, review_status TEXT NOT NULL DEFAULT 'needs_review',
  source_is_card INTEGER NOT NULL DEFAULT 0, ocr_text TEXT DEFAULT '', card_fins_included TEXT DEFAULT '', card_fin_system TEXT DEFAULT '', unique(source_path)
);
CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS grouping_predictions (
  photo_id INTEGER PRIMARY KEY REFERENCES photos(id),
  predicted_board_id INTEGER NOT NULL,
  predicted_label TEXT NOT NULL,
  predicted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grouping_corrections (
  photo_id INTEGER PRIMARY KEY REFERENCES photos(id),
  predicted_board_id INTEGER NOT NULL,
  corrected_board_id INTEGER NOT NULL,
  corrected_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grouping_feedback_batches (
  run_id TEXT PRIMARY KEY,
  input_signature TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  saved_at TEXT,
  status TEXT NOT NULL DEFAULT 'proposed'
);
CREATE TABLE IF NOT EXISTS grouping_feedback (
  run_id TEXT NOT NULL REFERENCES grouping_feedback_batches(run_id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL,
  source_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  initial_board_id INTEGER NOT NULL,
  initial_board_label TEXT NOT NULL,
  final_board_id INTEGER,
  final_board_label TEXT,
  corrected INTEGER NOT NULL DEFAULT 0,
  saved_at TEXT,
  PRIMARY KEY (run_id, photo_id)
);
CREATE TABLE IF NOT EXISTS excluded_sources (
  source_path TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  excluded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  pipeline_version TEXT NOT NULL DEFAULT '',
  input_signature TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  error TEXT DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  worker_pid INTEGER,
  log_path TEXT DEFAULT '',
  last_heartbeat_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  sha256 TEXT DEFAULT '',
  size_bytes INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(run_id, relative_path)
);
CREATE TABLE IF NOT EXISTS retention_items (
  relative_path TEXT PRIMARY KEY,
  retention_class TEXT NOT NULL,
  derivative_path TEXT DEFAULT '',
  source_sha256 TEXT NOT NULL,
  source_size_bytes INTEGER NOT NULL DEFAULT 0,
  derivative_sha256 TEXT DEFAULT '',
  derivative_size_bytes INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'planned',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  reason TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    existing_version = None
    try:
        existing_version = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        pass
    if db_path.exists() and db_path.stat().st_size and (existing_version is None or int(existing_version["value"]) < CURRENT_SCHEMA_VERSION):
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"pre-migration-{datetime.now().strftime('%Y%m%d-%H%M%S')}.sqlite3"
        if not backup_path.exists():
            shutil.copy2(db_path, backup_path)
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
    board_columns = {row["name"] for row in conn.execute("PRAGMA table_info(boards)")}
    if "fin_system" not in board_columns:
        conn.execute("ALTER TABLE boards ADD COLUMN fin_system TEXT DEFAULT ''")
    if "fins_included" not in board_columns:
        conn.execute("ALTER TABLE boards ADD COLUMN fins_included TEXT DEFAULT ''")
    if "card_fins_included" not in columns:
        conn.execute("ALTER TABLE photos ADD COLUMN card_fins_included TEXT DEFAULT ''")
    if "card_fin_system" not in columns:
        conn.execute("ALTER TABLE photos ADD COLUMN card_fin_system TEXT DEFAULT ''")
    run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    for name, definition in (("worker_pid", "INTEGER"), ("log_path", "TEXT DEFAULT ''"), ("last_heartbeat_at", "TEXT")):
        if name not in run_columns:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
    version = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    if version is None:
        # Tables are created with IF NOT EXISTS above, so this also upgrades
        # databases created before schema versioning was introduced.
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(CURRENT_SCHEMA_VERSION),),
        )
    elif int(version["value"]) < CURRENT_SCHEMA_VERSION:
        conn.execute(
            "UPDATE schema_meta SET value=? WHERE key='schema_version'",
            (str(CURRENT_SCHEMA_VERSION),),
        )
    conn.commit()
    return conn


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_run(conn: sqlite3.Connection, run_id: str, kind: str, *,
               pipeline_version: str = "", input_signature: str = "",
               metadata: dict | None = None, status: str = "queued") -> None:
    conn.execute(
        """INSERT OR IGNORE INTO runs
        (run_id, kind, status, pipeline_version, input_signature, created_at, started_at,
         last_heartbeat_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, kind, status, pipeline_version, input_signature, now_text(),
         now_text() if status == "running" else None,
         now_text() if status == "running" else None,
         json.dumps(metadata or {}, sort_keys=True)),
    )
    conn.commit()


def update_run(conn: sqlite3.Connection, run_id: str, status: str, *, error: str = "",
               metadata: dict | None = None) -> None:
    timestamp = now_text()
    conn.execute(
        """UPDATE runs SET status=?, error=?, last_heartbeat_at=?,
        metadata_json=COALESCE(?, metadata_json),
        started_at=CASE WHEN ?='running' AND started_at IS NULL THEN ? ELSE started_at END,
        completed_at=CASE WHEN ? IN ('complete', 'failed') THEN ? ELSE completed_at END
        WHERE run_id=?""",
        (status, error, timestamp, json.dumps(metadata, sort_keys=True) if metadata is not None else None,
         status, timestamp, status, timestamp, run_id),
    )
    conn.commit()


def set_run_worker(conn: sqlite3.Connection, run_id: str, *, pid: int | None = None, log_path: str = "") -> None:
    conn.execute(
        "UPDATE runs SET worker_pid=?, log_path=?, last_heartbeat_at=? WHERE run_id=?",
        (pid, log_path, now_text(), run_id),
    )
    conn.commit()


def recover_stale_runs(conn: sqlite3.Connection, *, max_age_minutes: int = 120) -> list[str]:
    """Mark abandoned workers failed, without touching completed output."""
    now = datetime.now()
    stale: list[str] = []
    for row in conn.execute("SELECT run_id, started_at, worker_pid FROM runs WHERE status='running'").fetchall():
        if not row["started_at"]:
            continue
        try:
            started = datetime.fromisoformat(row["started_at"])
        except ValueError:
            continue
        worker_dead = False
        if row["worker_pid"]:
            try:
                os.kill(int(row["worker_pid"]), 0)
            except (ProcessLookupError, PermissionError):
                worker_dead = True
        if worker_dead or (now - started).total_seconds() > max_age_minutes * 60:
            stale.append(row["run_id"])
    for run_id in stale:
        update_run(conn, run_id, "failed", error=f"Worker stopped or exceeded {max_age_minutes} minutes; marked recoverable.")
    return stale


def record_artifact(conn: sqlite3.Connection, run_id: str, path: Path, root: Path, kind: str,
                    *, metadata: dict | None = None) -> None:
    if not path.exists() or not path.is_file():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    conn.execute(
        """INSERT INTO artifacts
        (run_id, kind, relative_path, sha256, size_bytes, created_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, relative_path) DO UPDATE SET
        kind=excluded.kind, sha256=excluded.sha256, size_bytes=excluded.size_bytes,
        created_at=excluded.created_at, metadata_json=excluded.metadata_json""",
        (run_id, kind, str(path.relative_to(root)), digest, path.stat().st_size,
         now_text(), json.dumps(metadata or {}, sort_keys=True)),
    )
    conn.commit()
