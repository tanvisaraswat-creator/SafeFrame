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
from pathlib import Path
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, abort,
)
from moderation_engine import ModerationEngine, report_to_dict, CONTEXT_PROFILES

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


# ─── Page Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


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

    # ── Run AI moderation ──────────────────────────────────────────────────────
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
