"""
db.py
=====
SQLite connection factory and schema init for SafeFrame.

Schema design: key lookup columns are stored as proper SQL columns;
the full record dict is also stored as a JSON blob so load_*/add_*
functions can reconstruct complete dicts without any field loss —
regardless of whether a record is an image upload, video upload,
API-submitted scan, or legacy JSON-migrated entry.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "safeframe.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # safe concurrent reads
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
    -- Users table ─────────────────────────────────────────────────────────────
    -- Stores every scalar field directly so queries/updates stay plain SQL.
    -- 'extra_json' holds any additional fields (e.g. priority) that don't
    -- have a dedicated column, so the full user dict is always round-trippable.
    CREATE TABLE IF NOT EXISTS users (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        email        TEXT UNIQUE NOT NULL COLLATE NOCASE,
        password     TEXT NOT NULL,
        role         TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'active',
        priority     INTEGER NOT NULL DEFAULT 0,
        platform_type TEXT NOT NULL DEFAULT 'standard',
        created_at   TEXT NOT NULL,
        extra_json   TEXT NOT NULL DEFAULT '{}'
    );

    -- Uploads table ────────────────────────────────────────────────────────────
    -- owner_id + created_at are indexed so uploads_for_user() is O(log n).
    -- data_json holds the complete record dict (handles image/video/API
    -- records that each carry a different set of optional fields).
    CREATE TABLE IF NOT EXISTS uploads (
        id           TEXT PRIMARY KEY,
        owner_id     TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        data_json    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_uploads_owner ON uploads (owner_id);

    -- Requests table ───────────────────────────────────────────────────────────
    -- customer_id, owner_id, status are indexed for the three main query shapes.
    -- data_json holds the complete request dict (score_breakdown, ai_summary, etc.)
    CREATE TABLE IF NOT EXISTS requests (
        id            TEXT PRIMARY KEY,
        customer_id   TEXT NOT NULL,
        owner_id      TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'pending',
        created_at    TEXT NOT NULL,
        data_json     TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_requests_customer ON requests (customer_id);
    CREATE INDEX IF NOT EXISTS ix_requests_owner    ON requests (owner_id);
    CREATE INDEX IF NOT EXISTS ix_requests_status   ON requests (status);
    """)
    conn.commit()
    conn.close()
