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
from moderate import apply_blur, apply_mask, moderate_video
from config import (PLATFORM_THRESHOLDS, PLATFORM_TYPE_LABELS, DEFAULT_PLATFORM_TYPE,
                    SUPPORTED_VIDEO_FORMATS, MAX_VIDEO_SIZE, SAFEFRAME_API_KEY)
import store
import auto_rd

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
UPLOAD_DIR      = BASE_DIR / "static" / "uploads"
LOG_FILE        = BASE_DIR / "logs" / "moderation_log.json"
TEMPLATES_DIR   = BASE_DIR / "templates"

ALLOWED_EXTENSIONS       = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "webm"}
MAX_CONTENT_LENGTH       = 100 * 1024 * 1024   # 100 MB (covers video uploads)

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

def allowed_video(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def _extract_thumbnail(video_path: Path, thumb_path: Path) -> bool:
    """Pull the first readable frame from a video and save it as a JPEG thumbnail."""
    cap = cv2.VideoCapture(str(video_path))
    saved = False
    for _ in range(30):           # try up to the 30th frame in case early ones are black
        ret, frame = cap.read()
        if not ret:
            break
        if frame.mean() > 8:      # skip nearly-black frames
            cv2.imwrite(str(thumb_path), frame)
            saved = True
            break
    cap.release()
    return saved


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


STATUS_LABELS = {
    "pending":  "Pending Admin Decision",
    "approved": "Approved by Admin",
    "denied":   "Denied by Admin",
}


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    if user["role"] == "admin":
        return redirect(url_for("admin"))

    if user["role"] == "customer":
        return _customer_dashboard(user)

    # ── Brand dashboard ────────────────────────────────────────────────────
    my_uploads = store.uploads_for_user(user["id"])
    approved   = store.requests_for_owner(user["id"])
    by_id      = {u["id"]: u for u in store.load_users()}

    approved_view = []
    for r in approved:
        if r["status"] not in store.APPROVED_STATUSES:
            continue
        cust = by_id.get(r["customer_id"], {})
        approved_view.append({**r, "customer_name": cust.get("name", "Unknown"),
                              "customer_email": cust.get("email", "")})

    platform_type = user.get("platform_type", DEFAULT_PLATFORM_TYPE)
    return render_template("user_dashboard.html", user=user,
                           uploads=my_uploads, approved_customers=approved_view,
                           status_labels=STATUS_LABELS,
                           platform_type=platform_type,
                           platform_types=PLATFORM_TYPE_LABELS)


def _customer_dashboard(user):
    # WHAT: build everything the customer dashboard template needs
    # WHY:  customers never upload — they browse brands & track access requests
    # IN:   user (logged-in customer dict)
    # OUT:  rendered customer_dashboard.html
    by_id        = {u["id"]: u for u in store.load_users()}
    my_requests  = store.requests_by_customer(user["id"])
    brands       = [u for u in store.load_users() if u["role"] == "brand"]
    uploads      = store.load_uploads()

    upload_counts = {}
    for up in uploads:
        upload_counts[up["owner_id"]] = upload_counts.get(up["owner_id"], 0) + 1

    # latest request status per brand (for Browse Brands button states)
    latest_status = {}
    for r in my_requests:
        if r["owner_id"] not in latest_status:
            latest_status[r["owner_id"]] = r["status"]

    requests_view = []
    for r in my_requests:
        owner = by_id.get(r["owner_id"], {})
        requests_view.append({
            **r,
            "brand_name":  owner.get("name", "Unknown"),
            "status_label": STATUS_LABELS.get(r["status"], r["status"]),
        })

    brand_cards = []
    for b in brands:
        brand_cards.append({
            "id":            b["id"],
            "name":          b["name"],
            "upload_count":  upload_counts.get(b["id"], 0),
            "request_status": latest_status.get(b["id"]),   # None | pending | approved | denied
        })

    approved_owner_ids = store.approved_owner_ids(user["id"])
    approved_brands    = [b for b in brands if b["id"] in approved_owner_ids]
    approved_uploads   = [u for u in uploads if u["owner_id"] in approved_owner_ids]
    for u in approved_uploads:
        u["owner_name"] = by_id.get(u["owner_id"], {}).get("name", "Unknown")

    return render_template("customer_dashboard.html", user=user,
                           requests=requests_view, brands=brand_cards,
                           approved_brands=approved_brands, approved_uploads=approved_uploads,
                           status_labels=STATUS_LABELS)


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

    def _enrich(r):
        cust  = by_id.get(r["customer_id"], {})
        owner = by_id.get(r["owner_id"], {})
        return {**r, "customer_name": cust.get("name", "Unknown"),
                "customer_email": cust.get("email", ""),
                "owner_name": owner.get("name", "Unknown")}

    # Single source of truth: every request the admin still needs to decide on.
    # Auto R&D never decides — it only attaches a ready-made report (trust score,
    # risk level, recommendation, signal breakdown, AI summary) to each one.
    pending_requests = [_enrich(r) for r in requests_ if r["status"] == "pending"]
    decided_today = len([r for r in requests_
                         if r["status"] in ("approved", "denied")
                         and r.get("decided_at", "")[:10] == datetime.now(timezone.utc).date().isoformat()])

    flagged = [u for u in uploads if u.get("blocked") or u.get("blurred")]

    # ── chart data: uploads per day (last 7 days) + safe/sensitive/blocked donut ──
    from datetime import timedelta
    today_d = datetime.now(timezone.utc).date()
    day_labels = [(today_d - timedelta(days=i)) for i in range(6, -1, -1)]
    day_counts = {d.isoformat(): 0 for d in day_labels}
    for up in uploads:
        key = up.get("created_at", "")[:10]
        if key in day_counts:
            day_counts[key] += 1
    max_day = max(day_counts.values()) or 1
    chart_days = [{"label": d.strftime("%a"), "count": day_counts[d.isoformat()],
                   "pct": round(day_counts[d.isoformat()] / max_day * 100)} for d in day_labels]

    safe_n  = len([u for u in uploads if u.get("action") == "ALLOW"])
    sens_n  = len([u for u in uploads if u.get("action") in ("BLUR_WARNING", "FLAG_FOR_REVIEW")])
    block_n = len([u for u in uploads if u.get("action") in ("BLOCK", "RESTRICT")])
    donut_total = max(safe_n + sens_n + block_n, 1)

    stats = {
        "total_users":          len(users),
        "total_uploads":        len(uploads),
        "pending_manual":       len(pending_requests),
        "decided_today":        decided_today,
        "flagged":              len(flagged),
    }

    recent = uploads[:10]
    for r in recent:
        owner = by_id.get(r["owner_id"], {})
        r["owner_name"] = owner.get("name", "Unknown")

    donut = {
        "safe":  {"count": safe_n,  "pct": round(safe_n / donut_total * 100)},
        "sens":  {"count": sens_n,  "pct": round(sens_n / donut_total * 100)},
        "block": {"count": block_n, "pct": round(block_n / donut_total * 100)},
    }

    return render_template("admin.html", user=user, users=users,
                           pending_requests=pending_requests, stats=stats, recent=recent,
                           status_labels=STATUS_LABELS, chart_days=chart_days, donut=donut)


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
        platform_type = user.get("platform_type", DEFAULT_PLATFORM_TYPE)
        report = eng.moderate(str(save_path), file.filename, context="standard",
                              platform_type=platform_type)
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


@app.route("/upload-video", methods=["POST"])
@login_required
def upload_video():
    # WHAT: accept a brand video upload, moderate it frame-by-frame, store result
    # WHY:  video can hide unsafe content in any frame — scanning the whole clip
    #       before it goes anywhere protects downstream viewers
    # IN:   multipart "video" file field (mp4/avi/mov/webm, max 100 MB)
    # OUT:  JSON {success, upload} where upload includes verdict + flagged_frames
    user = current_user()
    if user["role"] != "brand":
        abort(403)

    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if not allowed_video(file.filename):
        return jsonify({"error": "Unsupported video format. Allowed: mp4, avi, mov, webm."}), 415

    # Check file size against MAX_VIDEO_SIZE (Content-Length header is a hint)
    content_len = request.content_length
    if content_len and content_len > MAX_VIDEO_SIZE:
        return jsonify({"error": "Video too large. Maximum size is 100 MB."}), 413

    ext         = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path   = UPLOAD_DIR / unique_name

    try:
        file.save(str(save_path))
    except Exception as exc:
        logger.error("Video save failed: %s", exc)
        return jsonify({"error": "Failed to save video file."}), 500

    # Double-check saved size
    if save_path.stat().st_size > MAX_VIDEO_SIZE:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Video too large. Maximum size is 100 MB."}), 413

    try:
        eng     = get_engine()
        result  = moderate_video(str(save_path), eng.model, eng.device)
    except Exception as exc:
        logger.error("Video moderation failed: %s", exc)
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Video moderation failed.", "detail": str(exc)}), 500

    # Extract a thumbnail from the first non-black frame
    thumb_name = f"thumb_{unique_name.rsplit('.', 1)[0]}.jpg"
    thumb_path = UPLOAD_DIR / thumb_name
    thumb_url  = f"/api/image/{thumb_name}" if _extract_thumbnail(save_path, thumb_path) else None

    platform_type = user.get("platform_type", DEFAULT_PLATFORM_TYPE)
    record = {
        "type":                  "video",
        "filename":              file.filename,
        "stored_filename":       unique_name,
        "image_url":             thumb_url,            # thumbnail used in gallery
        "video_url":             f"/api/image/{unique_name}",
        "platform_type":         platform_type,
        "verdict":               result["verdict"],
        "action":                result["action"],
        "total_frames_checked":  result["total_frames_checked"],
        "flagged_frames":        result["flagged_frames"],
        # keep image-upload compat fields so the store/templates don't break
        "raw_class":             "video",
        "primary_category":      result["verdict"],
        "confidence_pct":        0,
        "blurred":               result["action"] in ("review", "block"),
        "blocked":               result["action"] == "block",
        "filter_applied":        "none",
        "processing_time_ms":    0,
    }
    saved = store.add_upload(user["id"], record)
    logger.info("[VIDEO] %s by %s → verdict=%s flagged=%d/%d frames",
                file.filename, user["email"],
                result["verdict"], len(result["flagged_frames"]),
                result["total_frames_checked"])
    return jsonify({"success": True, "upload": saved}), 200


@app.route("/delete-upload", methods=["POST"])
@login_required
def delete_upload():
    # WHAT: permanently remove an uploaded image — its file AND its record
    # WHY:  brands need to be able to clean up their own gallery; admins can
    #       remove any image (e.g. moderation cleanup)
    # IN:   upload_id (form field)
    # OUT:  JSON success / error
    user      = current_user()
    upload_id = request.form.get("upload_id", "").strip()
    if not upload_id:
        return jsonify({"error": "upload_id is required."}), 400

    target = store.find_upload_by_id(upload_id)
    if not target:
        return jsonify({"error": "Upload not found."}), 404

    # Only the owning brand — or an admin — may delete
    if user["role"] != "admin" and target["owner_id"] != user["id"]:
        abort(403)

    removed = store.delete_upload(upload_id)
    if not removed:
        return jsonify({"error": "Upload not found."}), 404

    stored_name = removed.get("stored_filename")
    if stored_name:
        file_path = UPLOAD_DIR / stored_name
        try:
            file_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("[DELETE] could not remove file %s: %s", file_path, exc)

    logger.info("[DELETE] upload %s (%s) removed by %s", upload_id,
                removed.get("filename"), user["email"])
    return jsonify({"success": True, "deleted_id": upload_id}), 200


@app.route("/set-platform-type", methods=["POST"])
@login_required
def set_platform_type():
    # WHAT: a brand picks their "Platform Type" (Fashion, Standard, Children's,
    #       Medical, Enterprise) — saved to their user profile in users.json
    # WHY:  every future upload from this brand is then moderated against
    #       that platform's PLATFORM_THRESHOLDS instead of generic defaults —
    #       e.g. a fashion catalogue tolerates more skin than a children's app
    # IN:   platform_type (form field — must be a known PLATFORM_THRESHOLDS key)
    # OUT:  JSON success / error
    user = current_user()
    if user["role"] != "brand":
        abort(403)

    platform_type = request.form.get("platform_type", "").strip().lower()
    if platform_type not in PLATFORM_THRESHOLDS:
        return jsonify({"error": "Unknown platform type."}), 400

    store.update_user(user["id"], platform_type=platform_type)
    logger.info("[PLATFORM] %s set platform_type -> %s", user["email"], platform_type)
    return jsonify({"success": True, "platform_type": platform_type}), 200


# ─── Access requests — customer asks, Auto R&D decides instantly ──────────────
@app.route("/request-access", methods=["POST"])
@login_required
def request_access():
    user = current_user()
    if user["role"] != "customer":
        abort(403)

    owner_id = request.form.get("owner_id", "")
    reason   = request.form.get("reason", "").strip()
    owner    = store.find_user_by_id(owner_id)
    if not owner:
        return jsonify({"error": "Brand not found."}), 404

    word_count = len(reason.split())
    if word_count < 15:
        return jsonify({"error": "Please explain your reason in at least 15 words."}), 400

    # ── Auto R&D INVESTIGATES instantly — it never decides. The request always
    #    lands as "pending" in front of the admin, with a full report attached. ──
    past = store.requests_by_customer(user["id"])
    rd_result = auto_rd.evaluate_request(
        customer=user,
        reason=reason,
        past_request_count=len(past),
        past_denials=store.count_denials(user["id"]),
    )
    entry = store.add_request(user["id"], owner_id, reason, rd_result)
    logger.info("[AUTO-RD] investigated %s -> %s | score=%s | risk=%s | recommendation=%s (admin will decide)",
                user["email"], owner["email"], rd_result["total"], rd_result["risk_level"], rd_result["recommendation"])

    return jsonify({
        "success": True,
        "request": entry,
        "trust_score": rd_result["total"],
        "risk_level": rd_result["risk_level"],
        "recommendation": rd_result["recommendation"],
        "summary": rd_result["summary"],
        "breakdown": rd_result["breakdown"],
    }), 200


@app.route("/approve-request", methods=["POST"])
@admin_required
def approve_request():
    # WHAT: admin makes the FINAL (and only) decision on an access request
    # WHY:  Auto R&D only investigates and recommends — the admin always decides
    # IN:   request_id, decision ("approved" | "denied"), note (optional reason)
    # OUT:  JSON success / error
    user       = current_user()
    request_id = request.form.get("request_id", "")
    decision   = request.form.get("decision", "")
    note       = request.form.get("note", "").strip()

    if decision not in ("approved", "denied"):
        return jsonify({"error": "Invalid decision."}), 400

    target = next((r for r in store.load_requests() if r["id"] == request_id), None)
    if not target:
        return jsonify({"error": "Request not found."}), 404

    store.set_request_status(request_id, decision, note)
    logger.info("[ADMIN] request %s -> %s (decided by %s)%s", request_id, decision, user["email"],
                f" — {note}" if note else "")
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
