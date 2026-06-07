# config.py
# All project settings live here. No values are hardcoded anywhere else.

from pathlib import Path

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
