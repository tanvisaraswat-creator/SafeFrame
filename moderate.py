# moderate.py
# Takes an image, classifies it, applies OpenCV blur/mask, saves output + flag.

import cv2
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime
from pathlib import Path

from config import (
    CLASS_NAMES, SAFE_CLASSES, BLUR_KERNEL, MASK_COLOR,
    BLUR_THRESHOLD, FLAG_THRESHOLD, RESULTS_DIR, FLAGS_DIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python moderate.py <image_path>")
        sys.exit(1)
    moderate_image(sys.argv[1])
