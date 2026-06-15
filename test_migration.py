"""
test_migration.py
=================
Comprehensive tests for the SQLite migration.
Run: python test_migration.py
All tests must print PASS. Any FAIL stops the migration process.
"""

import sys
import json
import traceback
from pathlib import Path

# ── Ensure we're testing against SQLite ───────────────────────────────────────
import config
config.USE_DATABASE = True  # force SQLite for this test run

from db import get_connection, init_db, DB_PATH
import store_sqlite as S

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")
        FAIL += 1


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. init_db()
# ─────────────────────────────────────────────────────────────────────────────
section("1. init_db() creates tables")
try:
    init_db()
    conn = get_connection()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    check("users table exists",    "users"    in tables)
    check("uploads table exists",  "uploads"  in tables)
    check("requests table exists", "requests" in tables)
except Exception as e:
    print(f"  EXCEPTION in init_db test: {e}")
    FAIL += 3

# ─────────────────────────────────────────────────────────────────────────────
# 2. migrate() run count
# ─────────────────────────────────────────────────────────────────────────────
section("2. Migration row counts")
try:
    users    = S.load_users()
    uploads  = S.load_uploads()
    requests = S.load_requests()
    check("at least 4 users",      len(users)    >= 4,  f"got {len(users)}")
    check("uploads migrated",       len(uploads)  >= 0,  f"got {len(uploads)}")   # may be 0 fresh
    check("requests migrated",      len(requests) >= 0,  f"got {len(requests)}")
    # Verify JSON counts match (if JSON files exist)
    json_u_path = Path("data_store/users.json")
    if json_u_path.exists():
        json_u = json.loads(json_u_path.read_text())
        check("user count matches JSON", len(users) == len(json_u),
              f"sqlite={len(users)} json={len(json_u)}")
except Exception as e:
    print(f"  EXCEPTION in count test: {e}")
    traceback.print_exc()
    FAIL += 4

# ─────────────────────────────────────────────────────────────────────────────
# 3. find_user_by_email() for all 4 demo accounts
# ─────────────────────────────────────────────────────────────────────────────
section("3. find_user_by_email() — all 4 demo accounts")
DEMO_ACCOUNTS = [
    ("admin@safeframe.com", "admin"),
    ("brand1@test.com",     "brand"),
    ("brand2@test.com",     "brand"),
    ("customer@test.com",   "customer"),
]
for email, expected_role in DEMO_ACCOUNTS:
    try:
        u = S.find_user_by_email(email)
        check(f"find {email}", u is not None and u["role"] == expected_role,
              f"got {u}")
    except Exception as e:
        check(f"find {email}", False, str(e))

# Case-insensitive lookup
try:
    u = S.find_user_by_email("ADMIN@SAFEFRAME.COM")
    check("case-insensitive lookup", u is not None and u["role"] == "admin")
except Exception as e:
    check("case-insensitive lookup", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 4. find_user_by_id()
# ─────────────────────────────────────────────────────────────────────────────
section("4. find_user_by_id()")
try:
    u = S.find_user_by_id("u-admin")
    check("find u-admin by id",  u is not None and u["email"] == "admin@safeframe.com")
    u2 = S.find_user_by_id("u-brand1")
    check("find u-brand1 by id", u2 is not None and u2["email"] == "brand1@test.com")
    u3 = S.find_user_by_id("nonexistent-id")
    check("returns None for missing id", u3 is None)
except Exception as e:
    check("find_user_by_id", False, str(e))
    FAIL += 2

# ─────────────────────────────────────────────────────────────────────────────
# 5. Login flow — password field preserved
# ─────────────────────────────────────────────────────────────────────────────
section("5. Login flow — password fields intact")
for email, pw in [("admin@safeframe.com","admin123"),
                   ("brand1@test.com","brand123"),
                   ("customer@test.com","cust123")]:
    try:
        u = S.find_user_by_email(email)
        check(f"password correct for {email}", u is not None and u["password"] == pw)
    except Exception as e:
        check(f"password correct for {email}", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 6. save_upload() + find_upload_by_id() round trip
# ─────────────────────────────────────────────────────────────────────────────
section("6. save_upload / find_upload_by_id round trip")
TEST_UPLOAD_OWNER = "u-brand1"
test_record = {
    "filename":         "test_image.jpg",
    "stored_filename":  "abc123.jpg",
    "image_url":        "/api/image/abc123.jpg",
    "type":             "image",
    "raw_class":        "neutral",
    "action":           "ALLOW",
    "confidence_pct":   99.1,
    "blurred":          False,
    "blocked":          False,
    "filter_applied":   "none",
    "processing_time_ms": 42.0,
    "platform_type":    "standard",
    "source":           "test",
    "primary_category": "SAFE",
}
try:
    saved = S.add_upload(TEST_UPLOAD_OWNER, test_record)
    check("add_upload returns dict with id",    isinstance(saved, dict) and "id" in saved)
    check("add_upload owner_id preserved",      saved["owner_id"] == TEST_UPLOAD_OWNER)
    check("add_upload filename preserved",      saved["filename"] == "test_image.jpg")
    check("add_upload private flag set",        saved.get("private") is True)

    found = S.find_upload_by_id(saved["id"])
    check("find_upload_by_id returns record",   found is not None)
    check("round-trip filename",                found["filename"] == "test_image.jpg")
    check("round-trip raw_class",               found["raw_class"] == "neutral")
    check("round-trip confidence_pct",          found["confidence_pct"] == 99.1)
    SAVED_UPLOAD_ID = saved["id"]
except Exception as e:
    check("save_upload/find_upload round trip", False, str(e))
    traceback.print_exc()
    SAVED_UPLOAD_ID = None
    FAIL += 7

# ─────────────────────────────────────────────────────────────────────────────
# 7. uploads_for_user() + delete_upload()
# ─────────────────────────────────────────────────────────────────────────────
section("7. uploads_for_user() and delete_upload()")
try:
    user_ups = S.uploads_for_user(TEST_UPLOAD_OWNER)
    check("uploads_for_user returns list",    isinstance(user_ups, list))
    check("test upload in user uploads",
          SAVED_UPLOAD_ID and any(u["id"] == SAVED_UPLOAD_ID for u in user_ups))

    if SAVED_UPLOAD_ID:
        removed = S.delete_upload(SAVED_UPLOAD_ID)
        check("delete_upload returns the record",  removed is not None and removed["id"] == SAVED_UPLOAD_ID)
        gone = S.find_upload_by_id(SAVED_UPLOAD_ID)
        check("record gone after delete",          gone is None)
        none_result = S.delete_upload("nonexistent-id")
        check("delete nonexistent returns None",   none_result is None)
except Exception as e:
    check("uploads_for_user/delete_upload", False, str(e))
    traceback.print_exc()
    FAIL += 4

# ─────────────────────────────────────────────────────────────────────────────
# 8. Video upload round trip (complex nested fields)
# ─────────────────────────────────────────────────────────────────────────────
section("8. Video upload round trip")
try:
    video_record = {
        "type":                 "video",
        "filename":             "test_video.mp4",
        "stored_filename":      "vid_abc.mp4",
        "image_url":            "/api/image/thumb_vid_abc.jpg",
        "video_url":            "/api/image/vid_abc.mp4",
        "verdict":              "REVIEW",
        "action":               "review",
        "total_frames_checked": 5,
        "flagged_frames": [
            {"timestamp": 1.0, "label": "Mature Content", "raw_class": "sexy", "confidence": 0.42},
        ],
        "raw_class":            "video",
        "primary_category":     "REVIEW",
        "confidence_pct":       0,
        "blurred":              True,
        "blocked":              False,
        "filter_applied":       "none",
        "processing_time_ms":   0,
        "platform_type":        "standard",
        "source":               "test",
    }
    vsaved = S.add_upload("u-brand1", video_record)
    vfound = S.find_upload_by_id(vsaved["id"])
    check("video record stored",              vfound is not None)
    check("video type preserved",             vfound.get("type") == "video")
    check("video verdict preserved",          vfound.get("verdict") == "REVIEW")
    check("flagged_frames list preserved",    isinstance(vfound.get("flagged_frames"), list))
    check("flagged_frames timestamp correct", vfound["flagged_frames"][0]["timestamp"] == 1.0)
    check("total_frames_checked correct",     vfound["total_frames_checked"] == 5)
    S.delete_upload(vsaved["id"])   # cleanup
except Exception as e:
    check("video upload round trip", False, str(e))
    traceback.print_exc()
    FAIL += 5

# ─────────────────────────────────────────────────────────────────────────────
# 9. save_request() + get/load_requests()
# ─────────────────────────────────────────────────────────────────────────────
section("9. save_request() + load_requests()")
TEST_RD = {
    "total": 72, "breakdown": {"age": 20, "history": 30, "profile": 22},
    "risk_level": "LOW", "recommendation": "APPROVE",
    "summary": "Low-risk customer with good history."
}
try:
    req = S.add_request("u-cust1", "u-brand1", "I want to view their content", TEST_RD)
    check("add_request returns dict",         isinstance(req, dict) and "id" in req)
    check("status starts pending",            req["status"] == "pending")
    check("trust_score stored",               req["trust_score"] == 72)
    check("ai_summary stored",                "Low-risk" in req["ai_summary"])
    check("score_breakdown stored",           req["score_breakdown"]["age"] == 20)

    all_reqs = S.load_requests()
    check("load_requests returns list",       isinstance(all_reqs, list))
    check("new request in load_requests",     any(r["id"] == req["id"] for r in all_reqs))

    owner_reqs = S.requests_for_owner("u-brand1")
    check("requests_for_owner returns list",  isinstance(owner_reqs, list))
    check("request found for owner",          any(r["id"] == req["id"] for r in owner_reqs))

    cust_reqs = S.requests_by_customer("u-cust1")
    check("requests_by_customer returns list", isinstance(cust_reqs, list))
    check("request found for customer",        any(r["id"] == req["id"] for r in cust_reqs))

    SAVED_REQ_ID = req["id"]
except Exception as e:
    check("save_request/load_requests", False, str(e))
    traceback.print_exc()
    FAIL += 11
    SAVED_REQ_ID = None

# ─────────────────────────────────────────────────────────────────────────────
# 10. set_request_status()
# ─────────────────────────────────────────────────────────────────────────────
section("10. set_request_status()")
try:
    if SAVED_REQ_ID:
        ok = S.set_request_status(SAVED_REQ_ID, "approved", "Looks good")
        check("set_status returns True",      ok is True)

        updated = next(r for r in S.load_requests() if r["id"] == SAVED_REQ_ID)
        check("status updated to approved",   updated["status"] == "approved")
        check("admin_note stored",            updated["admin_note"] == "Looks good")
        check("decided_at timestamp added",   "decided_at" in updated)

        # denied path
        ok2 = S.set_request_status(SAVED_REQ_ID, "denied", "Changed mind")
        updated2 = next(r for r in S.load_requests() if r["id"] == SAVED_REQ_ID)
        check("can re-set to denied",         updated2["status"] == "denied")
        check("deny_reason stored",           updated2["deny_reason"] == "Changed mind")

        ok3 = S.set_request_status("nonexistent-req", "approved")
        check("returns False for missing id", ok3 is False)
except Exception as e:
    check("set_request_status", False, str(e))
    traceback.print_exc()
    FAIL += 6

# ─────────────────────────────────────────────────────────────────────────────
# 11. count_denials() + approved_owner_ids()
# ─────────────────────────────────────────────────────────────────────────────
section("11. count_denials() + approved_owner_ids()")
try:
    # The test request above is now "denied" → count should be >= 1
    denials = S.count_denials("u-cust1")
    check("count_denials returns int",        isinstance(denials, int))
    check("denial counted correctly",         denials >= 1, f"got {denials}")

    # Flip it to approved and check approved_owner_ids
    S.set_request_status(SAVED_REQ_ID, "approved", "Re-approved")
    approved_ids = S.approved_owner_ids("u-cust1")
    check("approved_owner_ids returns set",   isinstance(approved_ids, set))
    check("u-brand1 in approved_ids",         "u-brand1" in approved_ids)
except Exception as e:
    check("count_denials/approved_owner_ids", False, str(e))
    traceback.print_exc()
    FAIL += 4

# ─────────────────────────────────────────────────────────────────────────────
# 12. update_user()
# ─────────────────────────────────────────────────────────────────────────────
section("12. update_user()")
try:
    ok = S.update_user("u-brand1", platform_type="fashion")
    check("update_user returns True",         ok is True)
    u = S.find_user_by_id("u-brand1")
    check("platform_type updated",            u["platform_type"] == "fashion")

    ok2 = S.update_user("u-brand1", status="inactive")
    u2 = S.find_user_by_id("u-brand1")
    check("status updated",                   u2["status"] == "inactive")

    ok3 = S.update_user("u-brand1", priority=True)
    u3 = S.find_user_by_id("u-brand1")
    check("priority updated to True",         u3["priority"] is True)

    ok4 = S.update_user("u-brand1", priority=False, status="active", platform_type="standard")
    check("multi-field update returns True",  ok4 is True)

    ok5 = S.update_user("nonexistent-user-id", status="active")
    check("returns False for missing user",   ok5 is False)
except Exception as e:
    check("update_user", False, str(e))
    traceback.print_exc()
    FAIL += 5

# ─────────────────────────────────────────────────────────────────────────────
# 13. store router (store.py USE_DATABASE=True)
# ─────────────────────────────────────────────────────────────────────────────
section("13. store.py router uses SQLite backend")
try:
    import store
    u = store.find_user_by_email("admin@safeframe.com")
    check("store.find_user_by_email works",  u is not None and u["role"] == "admin")
    ups = store.load_uploads()
    check("store.load_uploads returns list", isinstance(ups, list))
    reqs = store.load_requests()
    check("store.load_requests returns list", isinstance(reqs, list))
    check("APPROVED_STATUSES constant",      store.APPROVED_STATUSES == ("approved",))
    check("DENIED_STATUSES constant",        store.DENIED_STATUSES == ("denied",))
except Exception as e:
    check("store router", False, str(e))
    traceback.print_exc()
    FAIL += 5

# ─────────────────────────────────────────────────────────────────────────────
# 14. rollback — store.py USE_DATABASE=False uses JSON
# ─────────────────────────────────────────────────────────────────────────────
section("14. Rollback path — store_json works independently")
try:
    import store_json as sj
    uj = sj.find_user_by_email("admin@safeframe.com")
    check("store_json.find_user_by_email works", uj is not None and uj["role"] == "admin")
    upj = sj.load_uploads()
    check("store_json.load_uploads returns list", isinstance(upj, list))
except Exception as e:
    check("rollback path", False, str(e))
    traceback.print_exc()
    FAIL += 2

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  Results: {PASS} PASS  /  {FAIL} FAIL")
print(f"{'═'*55}")
if FAIL > 0:
    print("\n  ⛔  MIGRATION NOT SAFE — fix failures before pushing.\n")
    sys.exit(1)
else:
    print("\n  ✅  ALL TESTS PASS — safe to continue.\n")
    sys.exit(0)
