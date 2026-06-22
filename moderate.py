# moderate.py
# Takes an image, classifies it, applies OpenCV blur/mask, saves output + flag.

import cv2
import torch
import json
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from torchvision import transforms
from datetime import datetime
from pathlib import Path

from config import (
    CLASS_NAMES, SAFE_CLASSES, BLUR_KERNEL, MASK_COLOR,
    BLUR_THRESHOLD, FLAG_THRESHOLD, RESULTS_DIR, FLAGS_DIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    VIDEO_CLIP_FRAMES, VIDEO_TEMPORAL_THRESHOLD,
    VIDEO_START_DURATION, VIDEO_END_DURATION,
    VIDEO_SAMPLE_INTERVAL, VIDEO_THREAD_WORKERS,
)
from model import load_model

# Image transform — must match what ResNet50 was trained on
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def classify_image(image_path: Path, model: torch.nn.Module, device: torch.device) -> tuple:
    # WHAT: Run image through ResNet50, return predicted class + confidence
    # WHY:  Core inference step — all decisions flow from this
    # IN:   image_path (Path), model, device
    # OUT:  (class_name: str, confidence: float, all_probs: list)

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    confidence, idx = probs.max(dim=0)
    return CLASS_NAMES[idx.item()], confidence.item(), probs.tolist()


def apply_blur(image_bgr: np.ndarray) -> np.ndarray:
    # WHAT: Apply heavy Gaussian blur to the entire image
    # WHY:  Makes unsafe content unviewable while preserving image dimensions
    # IN:   image_bgr (numpy array in BGR format from OpenCV)
    # OUT:  blurred image (same shape)
    return cv2.GaussianBlur(image_bgr, BLUR_KERNEL, 0)


def apply_mask(image_bgr: np.ndarray) -> np.ndarray:
    # WHAT: Draw a solid black rectangle over the centre 80% of the image
    # WHY:  Hard mask for highest-confidence unsafe content (porn, hentai)
    # IN:   image_bgr (numpy array)
    # OUT:  masked image (same shape)
    h, w  = image_bgr.shape[:2]
    x1, y1 = int(w * 0.1), int(h * 0.1)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    result = image_bgr.copy()
    cv2.rectangle(result, (x1, y1), (x2, y2), MASK_COLOR, thickness=-1)
    return result


def save_flag(image_path: Path, class_name: str, confidence: float, action: str) -> None:
    # WHAT: Write a JSON file recording what was flagged and why
    # WHY:  Audit trail — every unsafe detection is logged for review
    # IN:   image_path, class_name, confidence (0–1), action string
    # OUT:  None (writes file to FLAGS_DIR)
    flag = {
        "filename":   image_path.name,
        "class":      class_name,
        "confidence": round(confidence * 100, 2),
        "action":     action,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    }
    out = FLAGS_DIR / f"{image_path.stem}_flag.json"
    out.write_text(json.dumps(flag, indent=2))


def moderate_image(image_path: str, model=None, device=None) -> dict:
    # WHAT: Full pipeline — classify → decide action → blur/mask → save → log
    # WHY:  Single function that demo.py and main.py call for each image
    # IN:   image_path (str), optional pre-loaded model + device
    # OUT:  result dict with filename, class, confidence, action, output_path

    path = Path(image_path)
    if not path.exists():
        print(f"[MODERATE] ERROR — file not found: {image_path}")
        return {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(device)

    class_name, confidence, _ = classify_image(path, model, device)
    is_unsafe = class_name not in SAFE_CLASSES

    # Decide action based on class and confidence
    if is_unsafe and confidence >= BLUR_THRESHOLD:
        action = "masked" if class_name in ["porn", "hentai"] else "blurred"
    elif is_unsafe and confidence >= FLAG_THRESHOLD:
        action = "flagged"
    else:
        action = "passed"

    # Apply OpenCV filter
    image_bgr = cv2.imread(str(path))
    if action == "masked":
        output_bgr = apply_mask(image_bgr)
    elif action == "blurred":
        output_bgr = apply_blur(image_bgr)
    else:
        output_bgr = image_bgr

    # Save output image
    out_path = RESULTS_DIR / path.name
    cv2.imwrite(str(out_path), output_bgr)

    # Save flag file for any unsafe detection
    if is_unsafe and confidence >= FLAG_THRESHOLD:
        save_flag(path, class_name, confidence, action)

    tag = "[SAFE]" if action == "passed" else ("[BLOCKED]" if action == "masked" else "[UNSAFE]")
    print(f"[MODERATE] {path.name:<30} | {class_name:<10} {confidence*100:.1f}%  | {action} {tag}")

    return {
        "filename":    path.name,
        "class":       class_name,
        "confidence":  round(confidence * 100, 1),
        "action":      action,
        "output_path": str(out_path),
    }


def classify_frame_array(frame_bgr: np.ndarray,
                         model: torch.nn.Module,
                         device: torch.device) -> tuple:
    # WHAT: classify a raw OpenCV BGR frame (no disk I/O needed)
    # WHY:  moderate_video() has frames as numpy arrays, not file paths,
    #       so we convert in-memory instead of writing a temp file
    # IN:   frame_bgr (H×W×3 uint8 numpy array), model, device
    # OUT:  (class_name, confidence, all_probs) — same shape as classify_image()
    image  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)
    confidence, idx = probs.max(dim=0)
    return CLASS_NAMES[idx.item()], confidence.item(), probs.tolist()


def moderate_video_temporal(video_path: str,
                            model: torch.nn.Module,
                            device: torch.device) -> dict:
    # WHAT: Late-fusion temporal ensemble — extract VIDEO_CLIP_FRAMES evenly
    #       spaced frames, run each through ResNet50, average the probability
    #       vectors, decide on the pooled result.
    # WHY:  Stage 1 (per-second frame scan) catches clear-cut cases fast.
    #       For borderline REVIEW cases (1–2 flagged frames) the raw counts
    #       are noisy — a single unusual frame can flip the verdict.
    #       Averaging 16 probability vectors across the whole clip smooths
    #       that noise: if unsafe signal is real it survives the average;
    #       if it was a one-off artefact it gets diluted.
    # IN:   video_path (str), model, device
    # OUT:  {verdict, confidence, stage, method, avg_probs, flagged_frames}
    #         stage   "temporal"  — always, so callers can log which path ran
    #         method  "late_fusion_ensemble"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "verdict": "REVIEW", "confidence": 0.0,
            "stage": "temporal", "method": "late_fusion_ensemble",
            "error": "Could not re-open video for temporal pass",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Choose VIDEO_CLIP_FRAMES evenly spaced indices across the clip
    n_samples = min(VIDEO_CLIP_FRAMES, max(1, total_frames))
    if n_samples == 1:
        indices = [0]
    else:
        step    = (total_frames - 1) / (n_samples - 1)
        indices = [int(round(i * step)) for i in range(n_samples)]

    cap = cv2.VideoCapture(str(video_path))
    prob_accumulator = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    frames_read = 0
    flagged_temporal = []

    _display = {
        "sexy":   "Mature Content",
        "porn":   "Explicit Content",
        "hentai": "Illustrated Explicit",
    }

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        _, _, probs = classify_frame_array(frame, model, device)
        prob_vec = np.array(probs, dtype=np.float64)
        prob_accumulator += prob_vec
        frames_read += 1

        # Track per-frame unsafe hits for the flagged_frames list
        cls_idx    = int(np.argmax(prob_vec))
        cls_name   = CLASS_NAMES[cls_idx]
        cls_conf   = float(prob_vec[cls_idx])
        if cls_name not in SAFE_CLASSES:
            flagged_temporal.append({
                "timestamp":  round(target_idx / fps, 1),
                "label":      _display.get(cls_name, cls_name),
                "raw_class":  cls_name,
                "confidence": round(cls_conf, 4),
            })

    cap.release()

    if frames_read == 0:
        return {
            "verdict": "REVIEW", "confidence": 0.0,
            "stage": "temporal", "method": "late_fusion_ensemble",
            "flagged_frames": [],
        }

    avg_probs = (prob_accumulator / frames_read).tolist()

    # Unsafe probability = sum of all unsafe class averages
    unsafe_indices = [CLASS_NAMES.index(c) for c in CLASS_NAMES if c not in SAFE_CLASSES]
    unsafe_prob    = sum(avg_probs[i] for i in unsafe_indices)
    top_idx        = int(np.argmax(avg_probs))
    top_class      = CLASS_NAMES[top_idx]
    top_conf       = avg_probs[top_idx]

    if unsafe_prob >= VIDEO_TEMPORAL_THRESHOLD:
        verdict = "UNSAFE"
        action  = "block"
    else:
        verdict = "SAFE"
        action  = "pass"

    return {
        "verdict":        verdict,
        "action":         action,
        "confidence":     round(unsafe_prob * 100, 1),
        "top_class":      top_class,
        "top_confidence": round(top_conf * 100, 1),
        "avg_probs":      [round(p, 4) for p in avg_probs],
        "frames_sampled": frames_read,
        "flagged_frames": flagged_temporal,
        "stage":          "temporal",
        "method":         "late_fusion_ensemble",
    }


_INFERENCE_LOCK = threading.Lock()

_DISPLAY_LABELS = {
    "sexy":   "Mature Content",
    "porn":   "Explicit Content",
    "hentai": "Illustrated Explicit",
}

_UNSAFE_INDICES = [i for i, c in enumerate(CLASS_NAMES) if c not in SAFE_CLASSES]


def _read_frame_at(video_path: str, frame_idx: int) -> tuple[int, np.ndarray | None]:
    # WHAT: open a private VideoCapture, seek to frame_idx, return it
    # WHY:  cv2.VideoCapture is NOT thread-safe — each worker needs its own
    #       handle. The model inference step is serialised separately via lock.
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame_idx, (frame if ret and frame is not None else None)


def _classify_frame_locked(frame_bgr: np.ndarray,
                            model: torch.nn.Module,
                            device: torch.device) -> list:
    # WHAT: thread-safe model inference — acquires _INFERENCE_LOCK before
    #       calling the model so concurrent threads don't race on GPU state
    # WHY:  PyTorch model.forward() is not re-entrant across threads; the GIL
    #       alone is insufficient because CUDA ops release it internally
    image  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with _INFERENCE_LOCK:
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)
    return probs.tolist()


def moderate_video_smart(video_path: str,
                         model: torch.nn.Module,
                         device: torch.device) -> dict:
    # WHAT: Smart video moderation — samples only key sections, processes frames
    #       in parallel, and averages probability vectors per section.
    # WHY:  Scanning every frame of a long video is slow. Most harmful content
    #       appears in the intro, outro, or in regular intervals — not randomly
    #       distributed. Sampling those sections gives fast, accurate coverage.
    #
    # Sampling strategy:
    #   • First VIDEO_START_DURATION seconds  (intro check)
    #   • Every VIDEO_SAMPLE_INTERVAL seconds in the middle
    #   • Last VIDEO_END_DURATION seconds     (outro check)
    #
    # Parallel execution:
    #   • VIDEO_THREAD_WORKERS threads handle frame decode (cv2, CPU-bound)
    #   • Model inference is serialised via _INFERENCE_LOCK (GPU-safe)
    #
    # Decision:
    #   • Explicit class (porn/hentai) detected → action "block"
    #   • Mature class (sexy) detected          → action "blur"
    #   • Neither                               → action "pass"
    #
    # IN:   video_path (str), model, device
    # OUT:  {verdict, action, flagged_sections, total_frames_checked,
    #        flagged_frames, stage, method}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "verdict": "ERROR", "error": "Could not open video file",
            "total_frames_checked": 0, "flagged_frames": [],
            "flagged_sections": [], "action": "block",
            "stage": "smart", "method": "smart_section_sampling",
        }

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps
    cap.release()

    # ── Build the list of (section_label, frame_idx) pairs to check ──────────
    frame_indices: list[tuple[str, int]] = []

    start_end_f = min(int(VIDEO_START_DURATION * fps), total_frames)
    # 1 frame per second within start window
    for fi in range(0, start_end_f, max(1, int(fps))):
        frame_indices.append(("start", fi))

    # Mid-video: every VIDEO_SAMPLE_INTERVAL seconds between start and end windows
    mid_start_s = VIDEO_START_DURATION
    mid_end_s   = max(VIDEO_START_DURATION, duration_s - VIDEO_END_DURATION)
    t = mid_start_s + VIDEO_SAMPLE_INTERVAL
    while t < mid_end_s:
        frame_indices.append(("middle", int(t * fps)))
        t += VIDEO_SAMPLE_INTERVAL

    end_start_f = max(0, total_frames - int(VIDEO_END_DURATION * fps))
    for fi in range(end_start_f, total_frames, max(1, int(fps))):
        if ("end", fi) not in frame_indices:
            frame_indices.append(("end", fi))

    # Deduplicate (start/end windows can overlap on very short clips)
    seen: set[int] = set()
    unique_pairs: list[tuple[str, int]] = []
    for label, fi in frame_indices:
        if fi not in seen:
            seen.add(fi)
            unique_pairs.append((label, fi))

    # ── Parallel frame read (CPU) + serialised inference (GPU) ───────────────
    # Submit all frame reads in parallel; inference runs one at a time inside
    # _classify_frame_locked, but the decode/resize/toTensor prep overlaps.
    results_by_idx: dict[int, tuple[str, list]] = {}  # frame_idx → (section, probs)

    def _process(label: str, fi: int) -> tuple[int, str, list | None]:
        _, frame = _read_frame_at(video_path, fi)
        if frame is None:
            return fi, label, None
        probs = _classify_frame_locked(frame, model, device)
        return fi, label, probs

    with ThreadPoolExecutor(max_workers=VIDEO_THREAD_WORKERS) as pool:
        futures = {pool.submit(_process, lbl, fi): (lbl, fi)
                   for lbl, fi in unique_pairs}
        for fut in as_completed(futures):
            fi, label, probs = fut.result()
            if probs is not None:
                results_by_idx[fi] = (label, probs)

    # ── Section-level averaging ───────────────────────────────────────────────
    section_probs: dict[str, list[list]] = {"start": [], "middle": [], "end": []}
    for fi, (label, probs) in results_by_idx.items():
        section_probs[label].append(probs)

    flagged_sections: list[dict] = []
    flagged_frames:   list[dict] = []
    has_explicit = False
    has_mature   = False

    for section_name in ("start", "middle", "end"):
        frames = section_probs[section_name]
        if not frames:
            continue
        avg = np.mean(frames, axis=0)  # shape: (num_classes,)
        top_idx  = int(np.argmax(avg))
        top_cls  = CLASS_NAMES[top_idx]
        top_conf = float(avg[top_idx])

        if top_cls not in SAFE_CLASSES:
            unsafe_sum = float(sum(avg[i] for i in _UNSAFE_INDICES))
            # Map section to representative timestamp for reporting
            if section_name == "start":
                ts = 0.0
            elif section_name == "end":
                ts = round(duration_s - VIDEO_END_DURATION, 1)
            else:
                ts = round(VIDEO_START_DURATION + VIDEO_SAMPLE_INTERVAL, 1)

            flagged_sections.append({
                "section":    section_name,
                "timestamp":  ts,
                "label":      _DISPLAY_LABELS.get(top_cls, top_cls),
                "raw_class":  top_cls,
                "confidence": round(top_conf, 4),
                "unsafe_sum": round(unsafe_sum, 4),
            })
            if top_cls in ("porn", "hentai"):
                has_explicit = True
            elif top_cls == "sexy":
                has_mature = True

    # Individual unsafe frames (for flagged_frames list in store record)
    for fi in sorted(results_by_idx):
        label, probs = results_by_idx[fi]
        top_idx  = int(np.argmax(probs))
        top_cls  = CLASS_NAMES[top_idx]
        top_conf = probs[top_idx]
        if top_cls not in SAFE_CLASSES:
            flagged_frames.append({
                "timestamp":  round(fi / fps, 1),
                "label":      _DISPLAY_LABELS.get(top_cls, top_cls),
                "raw_class":  top_cls,
                "confidence": round(top_conf, 4),
                "section":    results_by_idx[fi][0],
            })

    # ── Final verdict ─────────────────────────────────────────────────────────
    if has_explicit:
        verdict, action = "UNSAFE", "block"
    elif has_mature:
        verdict, action = "REVIEW", "blur"
    elif flagged_sections:
        verdict, action = "REVIEW", "review"
    else:
        verdict, action = "SAFE", "pass"

    return {
        "verdict":               verdict,
        "action":                action,
        "flagged_sections":      flagged_sections,
        "flagged_frames":        flagged_frames,
        "total_frames_checked":  len(results_by_idx),
        "stage":                 "smart",
        "method":                "smart_section_sampling",
    }


def moderate_video(video_path: str,
                   model: torch.nn.Module,
                   device: torch.device) -> dict:
    # WHAT: sample one frame per second, classify each, tally unsafe hits
    # WHY:  a single unsafe frame can hide anywhere in a video — checking
    #       every second gives coverage without being prohibitively slow
    # IN:   video_path (str), model, device
    # OUT:  {verdict, total_frames_checked, flagged_frames, action}
    #         verdict  SAFE   → 0 unsafe frames   → action "pass"
    #                  REVIEW → 1–2 unsafe frames  → action "review"
    #                  UNSAFE → 3+ unsafe frames   → action "block"

    from config import VIDEO_FRAME_INTERVAL, VIDEO_UNSAFE_FRAME_THRESHOLD

    _display = {
        "sexy":   "Mature Content",
        "porn":   "Explicit Content",
        "hentai": "Illustrated Explicit",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "verdict": "ERROR", "error": "Could not open video file",
            "total_frames_checked": 0, "flagged_frames": [], "action": "block",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_step   = max(1, int(round(fps * VIDEO_FRAME_INTERVAL)))
    total_checked = 0
    flagged_frames = []
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            cls, conf, _ = classify_frame_array(frame, model, device)
            if cls not in SAFE_CLASSES:
                flagged_frames.append({
                    "timestamp":  round(frame_idx / fps, 1),
                    "label":      _display.get(cls, cls),
                    "raw_class":  cls,
                    "confidence": round(conf, 4),
                })
            total_checked += 1
        frame_idx += 1

    cap.release()

    n = len(flagged_frames)
    if n == 0:
        verdict, action = "SAFE", "pass"
    elif n < VIDEO_UNSAFE_FRAME_THRESHOLD:
        verdict, action = "REVIEW", "review"
    else:
        verdict, action = "UNSAFE", "block"

    # Stage 2: temporal ensemble for uncertain REVIEW cases only.
    # SAFE and UNSAFE are decided by Stage 1 alone — no need to spend
    # extra compute when the frame scan is already conclusive.
    stage = "frame_detection"
    temporal_result = None

    if verdict == "REVIEW":
        temporal_result = moderate_video_temporal(video_path, model, device)
        if "error" not in temporal_result:
            verdict = temporal_result["verdict"]
            action  = temporal_result["action"]
            stage   = "temporal"
            # Use the temporal flagged_frames list (16-frame sample) so the
            # timestamps align with the ensemble pass, not the per-second scan
            flagged_frames = temporal_result.get("flagged_frames", flagged_frames)

    result = {
        "verdict":               verdict,
        "total_frames_checked":  total_checked,
        "flagged_frames":        flagged_frames,
        "action":                action,
        "stage":                 stage,
    }
    if temporal_result and "error" not in temporal_result:
        result["temporal_confidence"] = temporal_result.get("confidence", 0)
        result["frames_sampled"]      = temporal_result.get("frames_sampled", 0)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python moderate.py <image_path>")
        sys.exit(1)
    moderate_image(sys.argv[1])
