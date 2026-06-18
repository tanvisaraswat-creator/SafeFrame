# config.py
# All project settings live here. No values are hardcoded anywhere else.

from pathlib import Path

# ── Storage backend ────────────────────────────────────────────────────────────
# True  → SQLite via store_sqlite.py  (default for v1.1+)
# False → JSON files via store_json.py (instant rollback if anything breaks)
USE_DATABASE = True

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
UPLOADS_DIR    = BASE_DIR / "static" / "uploads"
OUTPUTS_DIR    = BASE_DIR / "outputs"
RESULTS_DIR    = OUTPUTS_DIR / "results"
FLAGS_DIR      = OUTPUTS_DIR / "flags"
MODEL_PATH     = BASE_DIR / "safeframe_model.pth"

# Create folders if they don't exist
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FLAGS_DIR.mkdir(exist_ok=True)

# ── Model Settings ─────────────────────────────────────────────────────────────
NUM_CLASSES    = 5
IMAGE_SIZE     = 224          # ResNet50 expects 224x224
DROPOUT_RATE   = 0.4
HIDDEN_DIM     = 512          # Linear(2048 → 512 → 5)

# ── Class Names (index order matters — must match training labels) ──────────────
# IMPORTANT: torchvision's ImageFolder assigns label indices ALPHABETICALLY by
# folder name during training. The order below MUST match that exactly, or
# predictions will be silently mislabelled (verified bug — fixed 2026-06-07).
# Alphabetical order of our 5 folders: drawings, hentai, neutral, porn, sexy
CLASS_NAMES    = ["drawings", "hentai", "neutral", "porn", "sexy"]

# SAFE classes: model will PASS these
SAFE_CLASSES   = ["neutral", "drawings"]

# UNSAFE classes: model will BLUR / FLAG these
UNSAFE_CLASSES = ["sexy", "porn", "hentai"]

# ── Platform Context Thresholds ────────────────────────────────────────────────
# WHAT: per-platform sensitivity profiles for brand accounts — different
#       businesses have very different baselines for what counts as "normal".
#       A fashion catalogue shows skin as a matter of course; a children's
#       platform needs to flag the same image instantly.
# WHY:  lets each brand pick a "Platform Type" on their profile and have every
#       upload they make moderated against thresholds that fit their context,
#       instead of one-size-fits-all global thresholds.
# Keys map onto our model's raw classes:
#   "mature"   -> raw_class == "sexy"   (suggestive / lightly revealing)
#   "explicit" -> raw_class == "porn"   (explicit real-photo content)
#   "hentai"   -> raw_class == "hentai" (explicit illustrated content)
# A LOWER threshold = stricter (flags at lower model confidence).
PLATFORM_THRESHOLDS = {
    "fashion": {
        "mature":   0.75,   # fashion brands show skin normally
        "explicit": 0.50,
        "hentai":   0.75,
    },
    "standard": {
        "mature":   0.35,   # default
        "explicit": 0.25,
        "hentai":   0.65,
    },
    "children": {
        "mature":   0.15,   # very strict
        "explicit": 0.10,
        "hentai":   0.50,
    },
    "medical": {
        "mature":   0.90,   # medical images need to show body
        "explicit": 0.60,
        "hentai":   0.75,
    },
    "enterprise": {
        "mature":   0.55,   # professional environment, conservative but not strict
        "explicit": 0.35,
        "hentai":   0.65,
    },
}

DEFAULT_PLATFORM_TYPE = "standard"

# ── Video Moderation Settings ──────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS    = [".mp4", ".avi", ".mov", ".webm"]
MAX_VIDEO_SIZE             = 100 * 1024 * 1024   # 100 MB
VIDEO_FRAME_INTERVAL       = 1                   # sample every N seconds
VIDEO_UNSAFE_FRAME_THRESHOLD = 3                 # frames at/above this → block
VIDEO_CLIP_FRAMES            = 16               # frames to sample for temporal ensemble
VIDEO_TEMPORAL_THRESHOLD     = 0.45             # averaged unsafe-class probability → escalate

# ── API Access ─────────────────────────────────────────────────────────────────
import os as _os
SAFEFRAME_API_KEY = _os.environ.get("SAFEFRAME_API_KEY", "sf-dev-key-imb360")

# Display labels + emoji for the brand "Platform Type" selector / badge
PLATFORM_TYPE_LABELS = {
    "fashion":    {"label": "Fashion & Apparel",    "emoji": "👗", "badge": "Fashion Mode"},
    "standard":   {"label": "Standard Business",    "emoji": "🏪", "badge": "Standard Mode"},
    "children":   {"label": "Children's Platform",  "emoji": "👶", "badge": "Children's Mode"},
    "medical":    {"label": "Medical & Health",     "emoji": "🏥", "badge": "Medical Mode"},
    "enterprise": {"label": "Enterprise",           "emoji": "🏢", "badge": "Enterprise Mode"},
}

# ── Decision Thresholds ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50   # below this → FLAG FOR REVIEW, don't auto-blur
BLUR_THRESHOLD       = 0.35   # >= this + unsafe class → apply Gaussian blur (lowered — catches lower-confidence hits, e.g. explicit content diluted inside screenshots)
FLAG_THRESHOLD       = 0.25   # >= this → save a JSON flag file (lowered to match — act on weaker signals too)

# ── OpenCV Blur Settings ───────────────────────────────────────────────────────
BLUR_KERNEL    = (51, 51)     # Gaussian blur kernel size (must be odd numbers)
MASK_COLOR     = (0, 0, 0)    # Black rectangle for hard mask (BGR format)

# ── Image Preprocessing (must match what ResNet50 was trained on) ──────────────
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# NOTE: Sightengine API has been completely removed (2026-06-07).
# SafeFrame now runs 100% on our own trained ResNet50 model — no
# third-party APIs, no external network calls, no API keys to manage.

# ── Demo Settings ──────────────────────────────────────────────────────────────
DEMO_IMAGE_COUNT  = 5         # how many images to pick from uploads for demo
DEMO_REPORT_FILE  = OUTPUTS_DIR / "demo_report.csv"
