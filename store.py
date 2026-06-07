"""
store.py
========
Tiny JSON-file "database" for SafeFrame's multi-user system.
No real database needed — users, uploads and access-requests all
live in simple JSON files under data_store/.

WHAT: load/save helpers + seed data for users, uploads, requests
WHY:  keeps app.py simple — all persistence logic lives in one place
"""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

STORE_DIR     = Path(__file__).parent / "data_store"
USERS_FILE    = STORE_DIR / "users.json"
UPLOADS_FILE  = STORE_DIR / "uploads.json"
REQUESTS_FILE = STORE_DIR / "requests.json"

STORE_DIR.mkdir(exist_ok=True)

# ── Seed accounts (created once, on first run) ────────────────────────────────
SEED_USERS = [
    {"id": "u-admin",  "name": "SafeFrame Admin", "email": "admin@safeframe.com",
     "password": "admin123", "role": "admin",    "status": "active", "priority": False},
    {"id": "u-brand1", "name": "Brand One",       "email": "brand1@test.com",
     "password": "brand123", "role": "brand",    "status": "active", "priority": False},
    {"id": "u-brand2", "name": "Brand Two",       "email": "brand2@test.com",
     "password": "brand123", "role": "brand",    "status": "active", "priority": False},
    {"id": "u-cust1",  "name": "Customer One",    "email": "customer@test.com",
     "password": "cust123",  "role": "customer", "status": "active", "priority": False},
]


def _load(path: Path, default):
    # WHAT: read a JSON file, returning a default value if it doesn't exist yet
    # WHY:  avoids crashing on first run before any data has been saved
    # IN:   path (Path to the json file), default (value to use if missing)
    # OUT:  parsed JSON content (list or dict)
    if not path.exists():
        _save(path, default)
        return default
    with open(path, "r") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return default


def _save(path: Path, data) -> None:
    # WHAT: write data to a JSON file with nice formatting
    # WHY:  every store function needs to persist changes the same way
    # IN:   path (Path to write to), data (list or dict to serialise)
    # OUT:  nothing — file is written to disk
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def now_iso() -> str:
    # WHAT: current UTC time as an ISO string
    # WHY:  every record needs a consistent timestamp format
    # IN:   nothing
    # OUT:  ISO-8601 string, e.g. "2026-06-07T10:30:00+00:00"
    return datetime.now(timezone.utc).isoformat()


# ── Users ──────────────────────────────────────────────────────────────────────
def load_users() -> list:
    return _load(USERS_FILE, list(SEED_USERS))


def save_users(users: list) -> None:
    _save(USERS_FILE, users)


def find_user_by_email(email: str):
    # WHAT: look up one user by email (case-insensitive)
    # WHY:  login needs to match the typed email against stored accounts
    # IN:   email string
    # OUT:  user dict or None
    email = email.strip().lower()
    for u in load_users():
        if u["email"].lower() == email:
            return u
    return None


def find_user_by_id(user_id: str):
    for u in load_users():
        if u["id"] == user_id:
            return u
    return None


def update_user(user_id: str, **changes) -> bool:
    # WHAT: change fields on one user (e.g. status, priority) and save
    # WHY:  admin actions like Activate/Deactivate/Make Priority need this
    # IN:   user_id string, keyword fields to update
    # OUT:  True if a user was found and updated, else False
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            u.update(changes)
            save_users(users)
            return True
    return False


# ── Uploads ────────────────────────────────────────────────────────────────────
def load_uploads() -> list:
    return _load(UPLOADS_FILE, [])


def save_uploads(uploads: list) -> None:
    _save(UPLOADS_FILE, uploads)


def add_upload(owner_id: str, record: dict) -> dict:
    # WHAT: store a new upload owned by a brand user
    # WHY:  every brand-dashboard upload must be tracked + kept private
    # IN:   owner_id (user id), record (moderation result fields to store)
    # OUT:  the saved upload dict (with id, owner_id, private flag added)
    uploads = load_uploads()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "owner_id": owner_id,
        "private": True,
        "created_at": now_iso(),
        **record,
    }
    uploads.insert(0, entry)
    save_uploads(uploads)
    return entry


def uploads_for_user(owner_id: str) -> list:
    return [u for u in load_uploads() if u["owner_id"] == owner_id]


# ── Access Requests ────────────────────────────────────────────────────────────
def load_requests() -> list:
    return _load(REQUESTS_FILE, [])


def save_requests(reqs: list) -> None:
    _save(REQUESTS_FILE, reqs)


def add_request(customer_id: str, owner_id: str, upload_id: str = None) -> dict:
    # WHAT: create a pending access request from a customer to a brand user
    # WHY:  uploads are private — customers must ask before viewing them
    # IN:   customer_id, owner_id (whose content is being requested), upload_id (optional)
    # OUT:  the saved request dict
    reqs = load_requests()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "customer_id": customer_id,
        "owner_id": owner_id,
        "upload_id": upload_id,
        "status": "pending",          # pending | approved | denied
        "created_at": now_iso(),
    }
    reqs.insert(0, entry)
    save_requests(reqs)
    return entry


def set_request_status(request_id: str, status: str) -> bool:
    reqs = load_requests()
    for r in reqs:
        if r["id"] == request_id:
            r["status"] = status
            r["decided_at"] = now_iso()
            save_requests(reqs)
            return True
    return False


def requests_for_owner(owner_id: str, status: str = None) -> list:
    reqs = [r for r in load_requests() if r["owner_id"] == owner_id]
    if status:
        reqs = [r for r in reqs if r["status"] == status]
    return reqs
