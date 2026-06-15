"""
migrate.py
==========
One-shot migration from JSON flat-files (data_store/*.json) to SQLite.

Safe to run multiple times — uses INSERT OR REPLACE so re-running
after a partial failure doesn't produce duplicates.

Run: python migrate.py
"""

import json
import sys
from pathlib import Path

# Bootstrap DB before importing store_sqlite (init_db is called on import)
from db import init_db, DB_PATH
import store_sqlite as sql_store

STORE_DIR = Path(__file__).parent / "data_store"

USERS_FILE    = STORE_DIR / "users.json"
UPLOADS_FILE  = STORE_DIR / "uploads.json"
REQUESTS_FILE = STORE_DIR / "requests.json"


def _load_json(path: Path, default):
    if not path.exists():
        print(f"  [MIGRATE] {path.name} not found — using default")
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  [MIGRATE] WARNING: could not parse {path.name}: {exc}")
        return default


def migrate() -> None:
    print("[MIGRATE] Initialising SQLite schema …")
    init_db()

    # ── Users ──────────────────────────────────────────────────────────────────
    users = _load_json(USERS_FILE, list(sql_store.SEED_USERS))
    u_count = 0
    for u in users:
        try:
            sql_store.save_user(u)
            u_count += 1
        except Exception as exc:
            print(f"  [MIGRATE] WARNING: skipped user {u.get('email')}: {exc}")
    print(f"[MIGRATE] Users migrated:   {u_count}")

    # ── Uploads ────────────────────────────────────────────────────────────────
    uploads = _load_json(UPLOADS_FILE, [])
    up_count = 0
    for up in uploads:
        if "id" not in up or "owner_id" not in up:
            print(f"  [MIGRATE] WARNING: skipped upload missing id/owner_id: {up.get('filename')}")
            continue
        try:
            from db import get_connection
            import json as _json
            conn = get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO uploads (id, owner_id, created_at, data_json) VALUES (?, ?, ?, ?)",
                (up["id"], up["owner_id"], up.get("created_at", sql_store.now_iso()),
                 _json.dumps(up, default=str)),
            )
            conn.commit()
            conn.close()
            up_count += 1
        except Exception as exc:
            print(f"  [MIGRATE] WARNING: skipped upload {up.get('id')}: {exc}")
    print(f"[MIGRATE] Uploads migrated: {up_count}")

    # ── Requests ───────────────────────────────────────────────────────────────
    requests = _load_json(REQUESTS_FILE, [])
    rq_count = 0
    for r in requests:
        if "id" not in r or "customer_id" not in r or "owner_id" not in r:
            print(f"  [MIGRATE] WARNING: skipped request missing required fields: {r.get('id')}")
            continue
        try:
            from db import get_connection
            import json as _json
            conn = get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO requests
                   (id, customer_id, owner_id, status, created_at, data_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r["id"], r["customer_id"], r["owner_id"],
                 r.get("status", "pending"), r.get("created_at", sql_store.now_iso()),
                 _json.dumps(r, default=str)),
            )
            conn.commit()
            conn.close()
            rq_count += 1
        except Exception as exc:
            print(f"  [MIGRATE] WARNING: skipped request {r.get('id')}: {exc}")
    print(f"[MIGRATE] Requests migrated: {rq_count}")

    print(f"[MIGRATE] Done. Database: {DB_PATH}")
    print(f"[MIGRATE] Backup: {STORE_DIR}/ (JSON files untouched)")


if __name__ == "__main__":
    migrate()
    sys.exit(0)
