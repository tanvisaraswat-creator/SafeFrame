"""
moderation_engine.py
====================
AI moderation engine for SafeFrame — powered ENTIRELY by our own
trained ResNet50 model (safeframe_model.pth). No third-party APIs.

Pipeline:
  Image -> our ResNet50 (5-class: neutral/sexy/porn/hentai/drawings)
        -> mapped to report taxonomy (SAFE/ADULT/SENSITIVE)
        -> Context-aware threshold decision -> Moderation Report
"""

import os
import uuid
import json
import logging
import hashlib
import time
from collections import Counter
import torch
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import load_model

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Context-Aware Threshold Profiles ────────────────────────────────────────
CONTEXT_PROFILES = {
    "children": {
        "label":          "Children's Platform",
        "description":    "Maximum strictness. Any hint of unsafe content is flagged immediately.",
        "emoji":          "👶",
        "hard_block":     0.40,
        "sensitive_flag": 0.25,
        "hate_block":     0.25,
    },
    "standard": {
        "label":          "Standard Social Media",
        "description":    "Balanced moderation for general social platforms.",
        "emoji":          "📱",
        "hard_block":     0.75,
        "sensitive_flag": 0.55,
        "hate_block":     0.55,
    },
    "journalism": {
        "label":          "News & Journalism",
        "description":    "Higher tolerance for graphic content in a journalistic context.",
        "emoji":          "📰",
        "hard_block":     0.85,
        "sensitive_flag": 0.72,
        "hate_block":     0.60,
    },
    "medical": {
        "label":          "Medical & Healthcare",
        "description":    "Allows clinical imagery. Hate and violence still moderated.",
        "emoji":          "🏥",
        "hard_block":     0.88,
        "sensitive_flag": 0.75,
        "hate_block":     0.50,
    },
    "enterprise": {
        "label":          "Enterprise / Workplace",
        "description":    "Professional environment. Conservative moderation.",
        "emoji":          "🏢",
        "hard_block":     0.65,
        "sensitive_flag": 0.40,
        "hate_block":     0.35,
    },
}

DEFAULT_CONTEXT = "standard"

# ─── Screenshot / Patch-Ensemble Detection ───────────────────────────────────
# WHY: a single whole-image inference on a browser screenshot gets diluted by
# tabs, chrome, taskbars and surrounding thumbnails — explicit content tucked
# in one corner can score as "neutral" overall. To catch it, large images are
# additionally sliced into overlapping patches; every image (large or not)
# also gets two zoomed-in centre crops, since explicit content is usually
# centred in a frame. The single most-unsafe finding across all of these wins.
SCREENSHOT_MIN_W   = 1200      # treat as a screenshot if wider than this …
SCREENSHOT_MIN_H   = 900       # … or taller than this
PATCH_UNSAFE_CONF  = 0.45      # a patch counts as a "hit" at/above this confidence
UNSAFE_RAW_CLASSES = {"sexy", "porn", "hentai"}
_UNSAFE_RANK       = {"porn": 3, "hentai": 3, "sexy": 1, "neutral": 0, "drawings": 0}


def _grid_patches(image: Image.Image, rows: int = 3, cols: int = 3) -> list:
    """Slice an image into a rows x cols grid of non-overlapping crops."""
    w, h = image.size
    patches = []
    for r in range(rows):
        for c in range(cols):
            left, upper   = int(c * w / cols), int(r * h / rows)
            right, lower  = int((c + 1) * w / cols), int((r + 1) * h / rows)
            if right > left and lower > upper:
                patches.append(image.crop((left, upper, right, lower)))
    return patches


def _overlapping_center_patches(image: Image.Image) -> list:
    """Four half-sized crops straddling the image centre from different angles —
    these overlap the 3x3 grid lines, so content sitting on a seam isn't missed."""
    w, h = image.size
    pw, ph = max(1, w // 2), max(1, h // 2)
    cx, cy = w // 2, h // 2
    offsets = [(-pw // 4, -ph // 4), (pw // 4, -ph // 4),
               (-pw // 4,  ph // 4), (pw // 4,  ph // 4)]
    patches = []
    for ox, oy in offsets:
        left  = max(0, min(w - pw, cx - pw // 2 + ox))
        upper = max(0, min(h - ph, cy - ph // 2 + oy))
        patches.append(image.crop((left, upper, left + pw, upper + ph)))
    return patches


def _center_crop(image: Image.Image, frac: float) -> Image.Image:
    """A zoomed-in crop of the given fraction of width/height, centred."""
    w, h = image.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    left, upper = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, upper, left + cw, upper + ch))

def get_profile(context: str) -> dict:
    return CONTEXT_PROFILES.get(context, CONTEXT_PROFILES[DEFAULT_CONTEXT])

# Legacy flat THRESHOLDS for backward compat
THRESHOLDS = CONTEXT_PROFILES[DEFAULT_CONTEXT]

# ─── Report Label Taxonomy ────────────────────────────────────────────────────
# The web dashboard / templates speak this 5-label vocabulary. Our ResNet50
# predicts a different 5-class set (neutral/sexy/porn/hentai/drawings), so we
# map one onto the other below — this keeps the rest of the app unchanged.
LABEL_NAMES = ["SAFE", "ADULT", "VIOLENT", "SENSITIVE", "HATE"]

# Maps our trained model's classes onto the report taxonomy above.
# neutral/drawings -> SAFE | sexy -> SENSITIVE | porn/hentai -> ADULT
CLASS_TO_LABEL = {
    "neutral":  "SAFE",
    "drawings": "SAFE",
    "sexy":     "SENSITIVE",
    "porn":     "ADULT",
    "hentai":   "ADULT",
}

ACTION_MAP = {
    "SAFE":      "ALLOW",
    "SENSITIVE": "BLUR_WARNING",
    "ADULT":     "RESTRICT",
    "VIOLENT":   "RESTRICT",
    "HATE":      "BLOCK",
    "REVIEW":    "FLAG_FOR_REVIEW",
}

REASONING_MAP = {
    "ALLOW":          "Content passed all safety checks. No harmful material detected.",
    "BLUR_WARNING":   "Content may contain sensitive material. Blurred by default — viewer discretion advised.",
    "RESTRICT":       "High-confidence detection of adult content. Restricted and blurred. Age confirmation required.",
    "BLOCK":          "High-confidence detection of prohibited content. Content blocked.",
    "FLAG_FOR_REVIEW":"Ambiguous signals detected. Flagged for human review and temporarily blurred.",
}


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class CategoryScore:
    label: str
    score: float
    percentage: float


@dataclass
class ModerationReport:
    image_id: str
    filename: str
    file_hash: str
    timestamp: str
    primary_category: str
    raw_class: str                # actual ResNet50 prediction: neutral/sexy/porn/hentai/drawings
    action: str
    confidence: float
    confidence_pct: float
    all_scores: list
    reasoning: str
    model_used: str
    processing_time_ms: float
    blocked: bool
    blurred: bool
    requires_age_gate: bool
    flagged_for_review: bool
    context: str
    context_label: str


# ─── Image Preprocessing (must match training exactly) ────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Moderation Engine ────────────────────────────────────────────────────────
class ModerationEngine:
    """
    Singleton engine running entirely on our own trained ResNet50 model.
    No external APIs, no network calls — fully self-contained inference.
    """

    _instance: Optional["ModerationEngine"] = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SafeFrame ResNet50 model on %s ...", self.device)
        self.model = load_model(self.device)
        self.model_name = "safeframe-resnet50-v1"
        logger.info("✅ SafeFrame model ready — using our own trained ResNet50 (no external APIs).")

    @classmethod
    def get_instance(cls) -> "ModerationEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    # ── Inference ──────────────────────────────────────────────────────────────
    def _classify(self, image: Image.Image) -> tuple:
        """Run image through our ResNet50, return (class_name, confidence, all 5-class probs)."""
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)
        confidence, idx = probs.max(dim=0)
        return CLASS_NAMES[idx.item()], confidence.item(), probs.tolist()

    # ── Screenshot-aware ensemble inference ────────────────────────────────────
    def _classify_ensemble(self, image: Image.Image) -> tuple:
        """
        Run multiple crops through the model and return the single most-unsafe
        finding, instead of trusting one whole-image pass that browser chrome /
        thumbnails / surrounding "neutral" pixels can dilute.

        Always checked: the full image, plus a 60% and a 40% centre crop
        (explicit content tends to sit centred in a frame).
        Large images ("screenshots" — wider than 1200px or taller than 900px)
        additionally get a 3x3 grid (9 patches) plus 4 overlapping centre
        patches — 13 extra crops — run through the model too.

        Voting:
          • if 2+ crops agree on the same unsafe class -> that class wins
            (highest-confidence agreeing crop is reported)
          • else if any single crop is unsafe at >= PATCH_UNSAFE_CONF -> the
            most-unsafe (highest rank, then highest confidence) one wins
          • else -> fall back to the whole-image result (so far, all neutral)

        Returns (class_name, confidence, raw_probs) — same shape as _classify().
        """
        w, h = image.size
        candidates = [self._classify(image)]                      # 1) whole image

        for frac in (0.6, 0.4):                                   # 2) centre zooms
            candidates.append(self._classify(_center_crop(image, frac)))

        is_screenshot = w > SCREENSHOT_MIN_W or h > SCREENSHOT_MIN_H
        if is_screenshot:                                          # 3) patch grid
            patches = _grid_patches(image, 3, 3) + _overlapping_center_patches(image)
            for patch in patches:
                if patch.size[0] >= 8 and patch.size[1] >= 8:
                    candidates.append(self._classify(patch))
            logger.info("[ENSEMBLE] screenshot mode (%dx%d) — checked %d crops total",
                        w, h, len(candidates))

        unsafe_hits = [c for c in candidates if c[0] in UNSAFE_RAW_CLASSES]
        if not unsafe_hits:
            return candidates[0]

        # Rule A — two or more crops agree on the same unsafe class
        votes = Counter(cls for cls, _, _ in unsafe_hits)
        agreed = [cls for cls, n in votes.items() if n >= 2]
        if agreed:
            best_cls = max(agreed, key=lambda c: _UNSAFE_RANK.get(c, 0))
            return max((c for c in unsafe_hits if c[0] == best_cls), key=lambda c: c[1])

        # Rule B — any single crop crosses the unsafe-confidence floor
        strong = [c for c in unsafe_hits if c[1] >= PATCH_UNSAFE_CONF]
        if strong:
            return max(strong, key=lambda c: (_UNSAFE_RANK.get(c[0], 0), c[1]))

        # Nothing strong enough — trust the whole-image pass
        return candidates[0]

    @staticmethod
    def _to_label_probs(raw_probs: list) -> list:
        """Convert our 5-class probs into the [SAFE,ADULT,VIOLENT,SENSITIVE,HATE] vector."""
        safe      = raw_probs[CLASS_NAMES.index("neutral")] + raw_probs[CLASS_NAMES.index("drawings")]
        adult     = raw_probs[CLASS_NAMES.index("porn")]    + raw_probs[CLASS_NAMES.index("hentai")]
        sensitive = raw_probs[CLASS_NAMES.index("sexy")]
        return [safe, adult, 0.0, sensitive, 0.0]

    # ── Decision Logic ─────────────────────────────────────────────────────────
    @staticmethod
    def _make_decision(probs: list, profile: dict) -> tuple:
        safe_p, adult_p, violent_p, sensitive_p, hate_p = probs
        hard_unsafe = max(adult_p, violent_p, hate_p)
        hard_block  = profile["hard_block"]
        sens_flag   = profile["sensitive_flag"]
        hate_block  = profile["hate_block"]

        if hate_p >= hate_block:
            return "HATE", "BLOCK"
        if hard_unsafe >= hard_block:
            return ("ADULT", "RESTRICT") if adult_p >= violent_p else ("VIOLENT", "RESTRICT")
        if hard_unsafe >= sens_flag:
            return ("ADULT", "BLUR_WARNING") if adult_p >= violent_p else ("VIOLENT", "BLUR_WARNING")
        if sensitive_p >= sens_flag:
            return "SENSITIVE", "BLUR_WARNING"
        return "SAFE", "ALLOW"

    # ── Public API ─────────────────────────────────────────────────────────────
    def moderate(
        self,
        image_path: str,
        original_filename: str,
        context: str = DEFAULT_CONTEXT,
    ) -> ModerationReport:

        t0 = time.perf_counter()
        profile = get_profile(context)

        image = Image.open(image_path).convert("RGB")
        with open(image_path, "rb") as fh:
            file_hash = hashlib.sha256(fh.read()).hexdigest()[:16]

        # ── Run inference with our own trained model ──────────────────────────
        # Screenshot-aware ensemble: checks the whole image, zoomed centre crops,
        # and (for large/screenshot-sized images) a grid of patches — so explicit
        # content tucked into a corner of a browser screenshot can't hide behind
        # surrounding "neutral" chrome and thumbnails diluting a single pass.
        raw_class, raw_confidence, raw_probs = self._classify_ensemble(image)
        probs = self._to_label_probs(raw_probs)

        # ── Decision ──────────────────────────────────────────────────────────
        category, action = self._make_decision(probs, profile)
        primary_index = LABEL_NAMES.index(category)
        confidence    = probs[primary_index]

        all_scores = [
            {"label": LABEL_NAMES[i], "score": round(probs[i], 4),
             "percentage": round(probs[i] * 100, 1)}
            for i in range(len(LABEL_NAMES))
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        report = ModerationReport(
            image_id=str(uuid.uuid4()),
            filename=original_filename,
            file_hash=file_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            primary_category=category,
            raw_class=raw_class,
            action=action,
            confidence=round(confidence, 4),
            confidence_pct=round(confidence * 100, 1),
            all_scores=all_scores,
            reasoning=REASONING_MAP.get(action, "No reasoning available."),
            model_used=self.model_name,
            processing_time_ms=round(elapsed_ms, 1),
            blocked=(action == "BLOCK"),
            blurred=(action in {"BLUR_WARNING", "RESTRICT", "BLOCK"}),
            requires_age_gate=(action == "RESTRICT"),
            flagged_for_review=(action == "FLAG_FOR_REVIEW"),
            context=context,
            context_label=profile["label"],
        )

        logger.info(
            "Moderated '%s' [%s] → raw=%s | %s (%s) | conf=%.1f%% | %.1f ms",
            original_filename, self.model_name, raw_class, category, action,
            confidence * 100, elapsed_ms,
        )
        return report


# ─── Report Serialisation ─────────────────────────────────────────────────────
def report_to_dict(report: ModerationReport) -> dict:
    return asdict(report)


# ─── Quick CLI test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python moderation_engine.py <image_path>")
        sys.exit(1)
    engine = ModerationEngine.get_instance()
    rep = engine.moderate(sys.argv[1], os.path.basename(sys.argv[1]))
    print(json.dumps(report_to_dict(rep), indent=2))
