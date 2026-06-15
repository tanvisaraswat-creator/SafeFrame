"""
store_sqlite.py
===============
SQLite-backed implementation of the SafeFrame store.
Exports IDENTICAL function names and return types as store_json.py
so the router in store.py can swap between them with a single flag.

Every function that returns a record returns a plain Python dict —
never a sqlite3.Row — so callers never need to change.
"""

import json
import uuid
from datetime import datetime, timezone

from db import get_connection, init_db

# ── Constants (same as store_json) ─────────────────────────────────────────────
APPROVED_STATUSES = ("approved",)
DENIED_STATUSES   = ("denied",)

SEED_USERS = [
    {"id": "u-admin",  "name": "SafeFrame Admin", "email": "admin@safeframe.com",
     "password": "admin123", "role": "admin",    "status": "active", "priority": False,
     "created_at": "2025-01-10T09:00:00+00:00"},
    {"id": "u-brand1", "name": "Brand One",       "email": "brand1@test.com",
     "password": "brand123", "role": "brand",    "status": "active", "priority": False,
     "created_at": "2025-03-02T09:00:00+00:00"},
    {"id": "u-brand2", "name": "Brand Two",       "email": "brand2@test.com",
     "password": "brand123", "role": "brand",    "status": "active", "priority": False,
     "created_at": "2025-06-18T09:00:00+00:00"},
    {"id": "u-cust1",  "name": "Customer One",    "email": "customer@test.com",
     "password": "cust123",  "role": "customer", "status": "active", "priority": False,
     "created_at": "2026-04-20T09:00:00+00:00"},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _user_row_to_dict(row) -> dict:
    """Convert a users table row to the canonical user dict."""
    d = {
        "id":            row["id"],
        "name":          row["name"],
        "email":         row["email"],
        "password":      row["password"],
        "role":          row["role"],
        "status":        row["status"],
        "priority":      bool(row["priority"]),
        "platform_type": row["platform_type"],
        "created_at":    row["created_at"],
    }
    # Merge any extra fields stored in extra_json (future-proof)
    try:
        extra = json.loads(row["extra_json"] or "{}")
        d.update(extra)
    except (json.JSONDecodeError, TypeError):
        pass
    return d


def _upload_row_to_dict(row) -> dict:
    """Reconstruct a full upload dict from the data_json blob."""
    try:
        return json.loads(row["data_json"])
    except (json.JSONDecodeError, TypeError):
        return {}


def _request_row_to_dict(row) -> dict:
    """Reconstruct a full request dict from the data_json blob."""
    try:
        return json.loads(row["data_json"])
    except (json.JSONDecodeError, TypeError):
        return {}


# ── Users ──────────────────────────────────────────────────────────────────────

def load_users() -> list:
    """Return all users as a list of dicts (matches store_json.load_users)."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
    conn.close()
    return [_user_row_to_dict(r) for r in rows]


def save_users(users: list) -> None:
    """Replace all users (used by migration; prefer update_user for single-field edits)."""
    conn = get_connection()
    conn.execute("DELETE FROM users")
    for u in users:
        _upsert_user(conn, u)
    conn.commit()
    conn.close()


def save_user(user: dict) -> None:
    """Insert or replace a single user dict."""
    conn = get_connection()
    _upsert_user(conn, user)
    conn.commit()
    conn.close()


def _upsert_user(conn, user: dict) -> None:
    known = {"id", "name", "email", "password", "role",
             "status", "priority", "platform_type", "created_at"}
    extra = {k: v for k, v in user.items() if k not in known}
    conn.execute(
        """INSERT OR REPLACE INTO users
           (id, name, email, password, role, status, priority, platform_type, created_at, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user.get("id", uuid.uuid4().hex[:12]),
            user["name"],
            user["email"],
            user["password"],
            user["role"],
            user.get("status", "active"),
            1 if user.get("priority") else 0,
            user.get("platform_type", "standard"),
            user.get("created_at", now_iso()),
            json.dumps(extra),
        ),
    )


def find_user_by_email(email: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)
    ).fetchone()
    conn.close()
    return _user_row_to_dict(row) if row else None


def find_user_by_id(user_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_row_to_dict(row) if row else None


def update_user(user_id: str, **changes) -> bool:
    """Update one or more fields on a user. Returns True if the user existed."""
    if not changes:
        return False

    # Map Python field names to SQL column names; anything else goes to extra_json
    sql_cols = {"name", "email", "password", "role", "status", "platform_type", "created_at"}

    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        return False

    # Handle priority separately (stored as INTEGER 0/1)
    if "priority" in changes:
        conn.execute(
            "UPDATE users SET priority = ? WHERE id = ?",
            (1 if changes.pop("priority") else 0, user_id),
        )

    sql_changes   = {k: v for k, v in changes.items() if k in sql_cols}
    extra_changes = {k: v for k, v in changes.items() if k not in sql_cols}

    if sql_changes:
        set_clause = ", ".join(f"{col} = ?" for col in sql_changes)
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?",
            (*sql_changes.values(), user_id),
        )

    if extra_changes:
        try:
            existing_extra = json.loads(row["extra_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            existing_extra = {}
        existing_extra.update(extra_changes)
        conn.execute(
            "UPDATE users SET extra_json = ? WHERE id = ?",
            (json.dumps(existing_extra), user_id),
        )

    conn.commit()
    conn.close()
    return True


# ── Uploads ────────────────────────────────────────────────────────────────────

def load_uploads() -> list:
    """Return all uploads newest-first (matches store_json.load_uploads)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT data_json FROM uploads ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [_upload_row_to_dict(r) for r in rows]


def save_uploads(uploads: list) -> None:
    """Replace all uploads (used by migration)."""
    conn = get_connection()
    conn.execute("DELETE FROM uploads")
    for u in uploads:
        _insert_upload(conn, u["owner_id"], u["created_at"], u)
    conn.commit()
    conn.close()


def add_upload(owner_id: str, record: dict) -> dict:
    """Create a new upload entry and return the saved dict."""
    entry = {
        "id":         uuid.uuid4().hex[:12],
        "owner_id":   owner_id,
        "private":    True,
        "created_at": now_iso(),
        **record,
    }
    conn = get_connection()
    _insert_upload(conn, owner_id, entry["created_at"], entry)
    conn.commit()
    conn.close()
    return entry


def _insert_upload(conn, owner_id: str, created_at: str, entry: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO uploads (id, owner_id, created_at, data_json) VALUES (?, ?, ?, ?)",
        (entry["id"], owner_id, created_at, json.dumps(entry, default=str)),
    )


def uploads_for_user(owner_id: str) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT data_json FROM uploads WHERE owner_id = ? ORDER BY created_at DESC",
        (owner_id,),
    ).fetchall()
    conn.close()
    return [_upload_row_to_dict(r) for r in rows]


def find_upload_by_id(upload_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT data_json FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    conn.close()
    return _upload_row_to_dict(row) if row else None


def delete_upload(upload_id: str):
    """Remove an upload and return its dict, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT data_json FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return None
    record = _upload_row_to_dict(row)
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()
    return record


# ── Access Requests ────────────────────────────────────────────────────────────

def load_requests() -> list:
    """Return all requests newest-first (matches store_json.load_requests)."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT data_json FROM requests ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [_request_row_to_dict(r) for r in rows]


def save_requests(reqs: list) -> None:
    """Replace all requests (used by migration)."""
    conn = get_connection()
    conn.execute("DELETE FROM requests")
    for r in reqs:
        _insert_request(conn, r)
    conn.commit()
    conn.close()


def add_request(customer_id: str, owner_id: str, reason: str, rd_result: dict) -> dict:
    entry = {
        "id":             uuid.uuid4().hex[:12],
        "customer_id":    customer_id,
        "owner_id":       owner_id,
        "reason":         reason,
        "status":         "pending",
        "trust_score":    rd_result["total"],
        "score_breakdown": rd_result["breakdown"],
        "risk_level":     rd_result["risk_level"],
        "recommendation": rd_result["recommendation"],
        "ai_summary":     rd_result["summary"],
        "admin_note":     "",
        "deny_reason":    "",
        "created_at":     now_iso(),
    }
    conn = get_connection()
    _insert_request(conn, entry)
    conn.commit()
    conn.close()
    return entry


def _insert_request(conn, entry: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO requests
           (id, customer_id, owner_id, status, created_at, data_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            entry["id"],
            entry["customer_id"],
            entry["owner_id"],
            entry.get("status", "pending"),
            entry["created_at"],
            json.dumps(entry, default=str),
        ),
    )


def set_request_status(request_id: str, status: str, note: str = "") -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT data_json FROM requests WHERE id = ?", (request_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return False
    record = _request_row_to_dict(row)
    record["status"]     = status
    record["decided_at"] = now_iso()
    if status == "denied":
        record["deny_reason"] = note
    else:
        record["admin_note"] = note
    conn.execute(
        "UPDATE requests SET status = ?, data_json = ? WHERE id = ?",
        (status, json.dumps(record, default=str), request_id),
    )
    conn.commit()
    conn.close()
    return True


def requests_for_owner(owner_id: str, status: str = None) -> list:
    conn = get_connection()
    if status:
        rows = conn.execute(
            "SELECT data_json FROM requests WHERE owner_id = ? AND status = ? ORDER BY created_at DESC",
            (owner_id, status),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT data_json FROM requests WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
    conn.close()
    return [_request_row_to_dict(r) for r in rows]


def requests_by_customer(customer_id: str) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT data_json FROM requests WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    ).fetchall()
    conn.close()
    return [_request_row_to_dict(r) for r in rows]


def count_denials(customer_id: str) -> int:
    conn = get_connection()
    n = conn.execute(
        "SELECT COUNT(*) FROM requests WHERE customer_id = ? AND status = 'denied'",
        (customer_id,),
    ).fetchone()[0]
    conn.close()
    return n


def approved_owner_ids(customer_id: str) -> set:
    conn = get_connection()
    rows = conn.execute(
        "SELECT data_json FROM requests WHERE customer_id = ? AND status = 'approved'",
        (customer_id,),
    ).fetchall()
    conn.close()
    return {_request_row_to_dict(r)["owner_id"] for r in rows}


# ── Initialise DB + seed on first import (all functions defined above) ─────────
def _ensure_seeded():
    """Insert SEED_USERS if the users table is empty (first-run only)."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    if count == 0:
        for u in SEED_USERS:
            save_user(u)


init_db()
_ensure_seeded()
