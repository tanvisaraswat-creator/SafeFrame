# demo.py
# Runs the SafeFrame moderation pipeline on 5 sample images.
# Prints a clean results table and saves a CSV report.

import csv
import torch
from pathlib import Path
from datetime import datetime
from config import UPLOADS_DIR, DEMO_IMAGE_COUNT, DEMO_REPORT_FILE
from model import load_model
from moderate import moderate_image

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_images(folder: Path, count: int) -> list:
    # WHAT: Pick up to `count` image files from the uploads folder
    # WHY:  Demo needs real images — uses what is already uploaded
    # IN:   folder (Path), count (int)
    # OUT:  list of Path objects

    images = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in SUPPORTED]
    if not images:
        print(f"[DEMO]     No images found in {folder}")
    return images[:count]


def print_table(results: list) -> None:
    # WHAT: Print a formatted results table to the terminal
    # WHY:  Clean visual output — looks impressive at a demo / presentation
    # IN:   results (list of dicts from moderate_image)
    # OUT:  None

    col = [28, 10, 12, 12]
    divider = "+" + "+".join("-" * (c + 2) for c in col) + "+"
    header  = "| {:<28} | {:<10} | {:<12} | {:<12} |".format(
                "Image", "Class", "Confidence", "Action")

    print(f"\n[DEMO]     Results")
    print(f"[DEMO]     " + divider)
    print(f"[DEMO]     " + header)
    print(f"[DEMO]     " + divider)

    for r in results:
        name  = r["filename"][:28]
        cls   = r["class"][:10]
        conf  = f"{r['confidence']:.1f}%"
        act   = r["action"]
        tag   = "[SAFE]" if act == "passed" else ("[BLOCKED]" if act == "masked" else "[UNSAFE]")
        line  = "| {:<28} | {:<10} | {:<12} | {:<12} |".format(name, cls, conf, f"{act} {tag}")
        print(f"[DEMO]     " + line)

    print(f"[DEMO]     " + divider + "\n")


def save_csv(results: list) -> None:
    # WHAT: Write results to a CSV file for sharing / reporting
    # WHY:  Gives something tangible to show the mentor — open in Excel
    # IN:   results (list of dicts)
    # OUT:  None (writes to DEMO_REPORT_FILE)

    fieldnames = ["filename", "class", "confidence", "action", "output_path"]
    with open(DEMO_REPORT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"[DEMO]     Report saved to {DEMO_REPORT_FILE.name}")


def run_demo():
    # WHAT: Full demo pipeline — load model, moderate 5 images, print table, save CSV
    # WHY:  Single function called by main.py for the Monday presentation
    # IN:   None
    # OUT:  None

    print(f"\n[DEMO]     SafeFrame — NSFW Content Moderation Demo")
    print(f"[DEMO]     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    images = pick_images(UPLOADS_DIR, DEMO_IMAGE_COUNT)
    if not images:
        print("[DEMO]     Add images to static/uploads/ and try again.")
        return

    print(f"[DEMO]     Running on {len(images)} images...\n")

    results = []
    for img_path in images:
        result = moderate_image(str(img_path), model=model, device=device)
        if result:
            results.append(result)

    print_table(results)
    save_csv(results)

    safe    = sum(1 for r in results if r["action"] == "passed")
    unsafe  = len(results) - safe
    print(f"[DEMO]     Summary: {safe} safe | {unsafe} unsafe out of {len(results)} images")
    print(f"[DEMO]     Output images saved to outputs/results/\n")


if __name__ == "__main__":
    run_demo()
