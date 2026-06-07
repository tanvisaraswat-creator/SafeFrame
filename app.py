"""
app.py
======
Flask backend for the AI Image Content Moderation System.

Routes
------
GET  /                      → Main upload UI
GET  /admin                 → Admin dashboard
POST /api/upload            → Upload + moderate an image
GET  /api/image/<filename>  → Serve stored image
GET  /api/stats             → Aggregate moderation statistics
GET  /api/logs              → Full moderation log (paginated)
GET  /api/report/<image_id> → Single moderation report
DELETE /api/image/<image_id>→ Remove image + log entry (admin)
"""

import os
import json
import uuid
import logging
import cv2
import functools
from pathlib import Path
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, abort, session, redirect, url_for,
)
from moderation_engine import ModerationEngine, report_to_dict, CONTEXT_PROFILES
from moderate import apply_blur, apply_mask
import store

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
UPLOAD_DIR      = BASE_DIR / "static" / "uploads"
LOG_FILE        = BASE_DIR / "logs" / "moderation_log.json"
TEMPLATES_DIR   = BASE_DIR / "templates"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16 MB

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

# ─── App Initialisation ────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "moderation-dev-secret-2024")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Warm-up the model at startup (non-blocking in dev, blocking in prod) ──────
engine: ModerationEngine = None

def get_engine() -> ModerationEngine:
    global engine
    if engine is None:
        engine = ModerationEngine.get_instance()
    return engine


# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


def save_log(entries: list[dict]) -> None:
    with open(LOG_FILE, "w") as fh:
        json.dump(entries, fh, indent=2, default=str)


def append_log(entry: dict) -> None:
    entries = load_log()
    entries.insert(0, entry)          # newest first
    entries = entries[:5000]           # cap at 5 000 entries
    save_log(entries)


def compute_stats(entries: list[dict]) -> dict:
    total = len(entries)
    if total == 0:
        return {
            "total": 0, "safe": 0, "sensitive": 0,
            "restricted": 0, "blocked": 0, "review": 0,
            "safe_pct": 0, "unsafe_pct": 0,
            "avg_processing_ms": 0,
            "category_breakdown": {},
        }

    cats = {}
    actions = {"ALLOW": 0, "BLUR_WARNING": 0, "RESTRICT": 0,
                "BLOCK": 0, "FLAG_FOR_REVIEW": 0}
    total_ms = 0.0

    for e in entries:
        cat = e.get("primary_category", "UNKNOWN")
        cats[cat] = cats.get(cat, 0) + 1
        action = e.get("action", "UNKNOWN")
        if action in actions:
            actions[action] += 1
        total_ms += e.get("processing_time_ms", 0)

    safe_count = actions["ALLOW"]
    unsafe_count = total - safe_count

    return {
        "total": total,
        "safe": safe_count,
        "sensitive": actions["BLUR_WARNING"],
        "restricted": actions["RESTRICT"],
        "blocked": actions["BLOCK"],
        "review": actions["FLAG_FOR_REVIEW"],
        "safe_pct": round(safe_count / total * 100, 1),
        "unsafe_pct": round(unsafe_count / total * 100, 1),
        "avg_processing_ms": round(total_ms / total, 1),
        "category_breakdown": cats,
    }


# ─── Auth helpers ─────────────────────────────────────────────────────────────
def current_user():
    # WHAT: fetch the logged-in user's record from the session
    # WHY:  every protected route needs to know who is asking
    # IN:   nothing (reads Flask session)
    # OUT:  user dict or None
    uid = session.get("user_id")
    if not uid:
        return None
    return store.find_user_by_id(uid)


def login_required(view):
    # WHAT: decorator that redirects to /login if nobody is signed in
    # WHY:  /dashboard, /admin, /upload etc. must not be reachable anonymously
    # IN:   view function to wrap
    # OUT:  wrapped view function
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    # WHAT: decorator that only lets role == "admin" through
    # WHY:  the admin dashboard and approval actions are admin-only
    # IN:   view function to wrap
    # OUT:  wrapped view function
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for("login"))
        if user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


# ─── Page Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if current_user():
            user = current_user()
            return redirect(url_for("admin") if user["role"] == "admin" else url_for("dashboard"))
        return render_template("login.html")

    email    = request.form.get("email", "")
    password = request.form.get("password", "")
    user = store.find_user_by_email(email)

    if not user or user["password"] != password:
        return render_template("login.html", error="Invalid email or password."), 401
    if user.get("status") == "inactive":
        return render_template("login.html", error="This account has been deactivated."), 403

    session["user_id"] = user["id"]
    logger.info("[LOGIN] %s (%s) signed in", user["email"], user["role"])
    return redirect(url_for("admin") if user["role"] == "admin" else url_for("dashboard"))


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    if user["role"] == "admin":
        return redirect(url_for("admin"))

    my_uploads = store.uploads_for_user(user["id"])
    incoming   = store.requests_for_owner(user["id"], status="pending")
    customers  = {u["id"]: u for u in store.load_users()}

    # attach requester names + per-upload pending request lists
    incoming_view = []
    for r in incoming:
        cust = customers.get(r["customer_id"], {})
        incoming_view.append({**r, "customer_name": cust.get("name", "Unknown"),
                              "customer_email": cust.get("email", "")})

    return render_template("user_dashboard.html", user=user,
                           uploads=my_uploads, requests=incoming_view)


@app.route("/admin")
@admin_required
def admin():
    user      = current_user()
    users     = [u for u in store.load_users() if u["role"] != "admin"]
    requests_ = store.load_requests()
    uploads   = store.load_uploads()
    by_id     = {u["id"]: u for u in store.load_users()}

    # enrich users with their upload counts
    upload_counts = {}
    for up in uploads:
        upload_counts[up["owner_id"]] = upload_counts.get(up["owner_id"], 0) + 1
    for u in users:
        u["upload_count"] = upload_counts.get(u["id"], 0)

    pending_requests = [r for r in requests_ if r["status"] == "pending"]
    requests_view = []
    for r in pending_requests:
        cust  = by_id.get(r["customer_id"], {})
        owner = by_id.get(r["owner_id"], {})
        requests_view.append({**r, "customer_name": cust.get("name", "Unknown"),
                              "owner_name": owner.get("name", "Unknown")})

    flagged = [u for u in uploads if u.get("blocked") or u.get("blurred")]

    stats = {
        "total_users":   len(users),
        "total_uploads": len(uploads),
        "pending":       len(pending_requests),
        "flagged":       len(flagged),
    }

    recent = uploads[:10]
    for r in recent:
        owner = by_id.get(r["owner_id"], {})
        r["owner_name"] = owner.get("name", "Unknown")

    return render_template("admin.html", user=user, users=users,
                           requests=requests_view, stats=stats, recent=recent)


# ─── Brand-user upload (private, tracked per-account) ─────────────────────────
@app.route("/upload", methods=["POST"])
@login_required
def user_upload():
    user = current_user()
    if user["role"] != "brand":
        abort(403)

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 415

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_DIR / unique_name

    try:
        file.save(str(save_path))
        eng = get_engine()
        report = eng.moderate(str(save_path), file.filename, context="standard")
        result = report_to_dict(report)

        if report.blurred:
            image_bgr = cv2.imread(str(save_path))
            if report.raw_class in ("porn", "hentai"):
                cv2.imwrite(str(save_path), apply_mask(image_bgr))
                result["filter_applied"] = "masked"
            else:
                cv2.imwrite(str(save_path), apply_blur(image_bgr))
                result["filter_applied"] = "blurred"
        else:
            result["filter_applied"] = "none"
    except Exception as exc:
        logger.error("Brand upload failed: %s", exc)
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Upload/moderation failed.", "detail": str(exc)}), 500

    record = {
        "filename": file.filename,
        "stored_filename": unique_name,
        "image_url": f"/api/image/{unique_name}",
        "primary_category": result["primary_category"],
        "raw_class": result["raw_class"],
        "action": result["action"],
        "confidence_pct": result["confidence_pct"],
        "blurred": result["blurred"],
        "blocked": result["blocked"],
        "filter_applied": result["filter_applied"],
        "processing_time_ms": result["processing_time_ms"],
    }
    saved = store.add_upload(user["id"], record)
    logger.info("[UPLOAD] %s by %s -> %s (%s)", file.filename, user["email"],
                saved["raw_class"], saved["action"])
    return jsonify({"success": True, "upload": saved}), 200


# ─── Access requests (customer <-> brand user, or admin on their behalf) ──────
@app.route("/request-access", methods=["POST"])
@login_required
def request_access():
    user = current_user()
    if user["role"] != "customer":
        abort(403)

    owner_id  = request.form.get("owner_id", "")
    upload_id = request.form.get("upload_id") or None
    if not store.find_user_by_id(owner_id):
        return jsonify({"error": "Brand not found."}), 404

    entry = store.add_request(user["id"], owner_id, upload_id)
    logger.info("[ACCESS] %s requested access to %s's content", user["email"], owner_id)
    return jsonify({"success": True, "request": entry}), 200


@app.route("/approve-request", methods=["POST"])
@login_required
def approve_request():
    user       = current_user()
    request_id = request.form.get("request_id", "")
    decision   = request.form.get("decision", "approved")   # approved | denied

    reqs = store.load_requests()
    target = next((r for r in reqs if r["id"] == request_id), None)
    if not target:
        return jsonify({"error": "Request not found."}), 404

    # only the content owner or an admin may decide
    if user["role"] != "admin" and user["id"] != target["owner_id"]:
        abort(403)

    store.set_request_status(request_id, decision)
    logger.info("[ACCESS] request %s -> %s (decided by %s)", request_id, decision, user["email"])
    return jsonify({"success": True, "status": decision}), 200


# ─── Admin: manage brand/customer accounts ────────────────────────────────────
@app.route("/admin/user-action", methods=["POST"])
@admin_required
def admin_user_action():
    target_id = request.form.get("user_id", "")
    action    = request.form.get("action", "")

    target = store.find_user_by_id(target_id)
    if not target:
        return jsonify({"error": "User not found."}), 404

    if action == "toggle-status":
        new_status = "inactive" if target.get("status") == "active" else "active"
        store.update_user(target_id, status=new_status)
    elif action == "toggle-priority":
        store.update_user(target_id, priority=not target.get("priority", False))
    else:
        return jsonify({"error": "Unknown action."}), 400

    logger.info("[ADMIN] %s -> %s", target["email"], action)
    return jsonify({"success": True}), 200


# ─── API: Upload & Moderate ───────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # Save with unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_DIR / unique_name

    try:
        file.save(str(save_path))
    except Exception as exc:
        logger.error("File save failed: %s", exc)
        return jsonify({"error": "Failed to save uploaded file."}), 500

    # ── Run AI moderation (our own ResNet50 — no external APIs) ────────────────
    try:
        eng = get_engine()
        context = request.form.get("context", "standard")
        report = eng.moderate(str(save_path), file.filename, context=context)
        result = report_to_dict(report)
        result["stored_filename"] = unique_name
        result["image_url"] = f"/api/image/{unique_name}"
    except Exception as exc:
        logger.error("Moderation failed: %s", exc)
        # Clean up the saved file on error
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "AI moderation engine error.", "detail": str(exc)}), 500

    # ── Apply OpenCV blur/mask directly on the served file if unsafe ───────────
    # porn / hentai → hard black mask | sexy → heavy Gaussian blur | else → untouched
    try:
        if report.blurred:
            image_bgr = cv2.imread(str(save_path))
            if report.raw_class in ("porn", "hentai"):
                filtered = apply_mask(image_bgr)
                result["filter_applied"] = "masked"
            else:
                filtered = apply_blur(image_bgr)
                result["filter_applied"] = "blurred"
            cv2.imwrite(str(save_path), filtered)
            logger.info("[MODERATE] %s → %s %.1f%% → %s",
                        file.filename, report.raw_class, report.confidence_pct, result["filter_applied"])
        else:
            result["filter_applied"] = "none"
    except Exception as exc:
        logger.warning("OpenCV filtering failed (image still served unfiltered): %s", exc)
        result["filter_applied"] = "none"

    # ── Persist log ────────────────────────────────────────────────────────────
    append_log(result)

    return jsonify(result), 200


# ─── API: Serve Image ─────────────────────────────────────────────────────────
@app.route("/api/image/<filename>")
def serve_image(filename: str):
    safe_name = secure_filename(filename)
    if not safe_name:
        abort(404)
    return send_from_directory(str(UPLOAD_DIR), safe_name)


# ─── API: Stats ───────────────────────────────────────────────────────────────
@app.route("/api/contexts")
def get_contexts():
    """Return all available platform context profiles."""
    return jsonify(CONTEXT_PROFILES), 200


@app.route("/api/stats")
def get_stats():
    entries = load_log()
    return jsonify(compute_stats(entries)), 200


# ─── API: Full Log ────────────────────────────────────────────────────────────
@app.route("/api/logs")
def get_logs():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    category = request.args.get("category", "")    # optional filter
    action   = request.args.get("action", "")

    entries = load_log()

    if category:
        entries = [e for e in entries if e.get("primary_category") == category.upper()]
    if action:
        entries = [e for e in entries if e.get("action") == action.upper()]

    total = len(entries)
    start = (page - 1) * per_page
    end   = start + per_page

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "items": entries[start:end],
    }), 200


# ─── API: Single Report ───────────────────────────────────────────────────────
@app.route("/api/report/<image_id>")
def get_report(image_id: str):
    entries = load_log()
    for entry in entries:
        if entry.get("image_id") == image_id:
            return jsonify(entry), 200
    return jsonify({"error": "Report not found."}), 404


# ─── API: Delete Image ────────────────────────────────────────────────────────
@app.route("/api/image/<image_id>", methods=["DELETE"])
def delete_image(image_id: str):
    entries = load_log()
    target  = next((e for e in entries if e.get("image_id") == image_id), None)

    if not target:
        return jsonify({"error": "Image not found."}), 404

    # Remove file
    stored = target.get("stored_filename", "")
    if stored:
        path = UPLOAD_DIR / secure_filename(stored)
        path.unlink(missing_ok=True)

    # Remove from log
    entries = [e for e in entries if e.get("image_id") != image_id]
    save_log(entries)

    return jsonify({"success": True, "deleted_id": image_id}), 200


# ─── Error Handlers ───────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Resource not found."}), 404


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load model so first request isn't slow
    logger.info("Pre-loading moderation engine …")
    get_engine()
    logger.info("🚀 Starting moderation server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
