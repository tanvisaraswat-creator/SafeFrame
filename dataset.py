# dataset.py
# Loads NSFW image dataset from a folder, splits into train / val / test.
# Expects this folder structure (ImageFolder format):
#   data/
#     neutral/    (safe images)
#     sexy/       (suggestive)
#     porn/       (explicit)
#     hentai/     (animated explicit)
#     drawings/   (safe art / illustrations)

import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"          # point at any folder with 5 class subfolders

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE  = 32
NUM_WORKERS = 0     # 0 = safe default on Windows (avoids multiprocessing errors)


def get_transforms() -> tuple:
    # WHAT: Define image augmentations for train set and plain resize for val/test
    # WHY:  Augmentation prevents overfitting; val/test must not be augmented
    # IN:   None
    # OUT:  (train_transform, val_transform)

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, val_tf


def load_datasets(data_dir: str = str(DATA_DIR)) -> tuple:
    # WHAT: Load full dataset from folder, split 70/15/15, return DataLoaders
    # WHY:  One function to get all three splits ready for train.py
    # IN:   data_dir (str) — path to folder with 5 class subfolders
    # OUT:  (train_loader, val_loader, test_loader, class_names)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[DATASET]  ERROR — data folder not found: {data_path}")
        print("[DATASET]  Create it with: python dataset.py --download")
        return None, None, None, []

    train_tf, val_tf = get_transforms()

    # Load full dataset with train transforms first (we re-apply val_tf below)
    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_tf)
    class_names  = full_dataset.classes

    # Verify expected classes are present
    missing = [c for c in CLASS_NAMES if c not in class_names]
    if missing:
        print(f"[DATASET]  WARNING — missing class folders: {missing}")

    total      = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size   = int(total * VAL_RATIO)
    test_size  = total - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply val/test transform (no augmentation) to val and test splits
    val_set.dataset.transform  = val_tf
    test_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"[DATASET]  Loaded {total:,} images across {len(class_names)} classes")
    print(f"[DATASET]  Split  — Train: {train_size:,} | Val: {val_size:,} | Test: {test_size:,}")
    print(f"[DATASET]  Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick check — run this to verify your dataset folder is set up correctly
    print("[DATASET]  Scanning data folder...")
    train_loader, val_loader, test_loader, classes = load_datasets()
    if train_loader:
        print(f"[DATASET]  Batch shape check...")
        images, labels = next(iter(train_loader))
        print(f"[DATASET]  Image batch: {images.shape} | Label batch: {labels.shape}")
