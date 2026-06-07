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
        raw_class, raw_confidence, raw_probs = self._classify(image)
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
