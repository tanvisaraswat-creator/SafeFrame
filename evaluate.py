# evaluate.py
# Evaluates the trained model on the test set.
# Reports: accuracy, per-class F1 score, confusion matrix.

import torch
from config import CLASS_NAMES, MODEL_PATH
from model import load_model
from dataset import load_datasets


def run_evaluation(model, test_loader, device) -> tuple:
    # WHAT: Run model on every test image, collect predictions and true labels
    # WHY:  Need the full prediction list to calculate F1 and confusion matrix
    # IN:   model, test_loader (DataLoader), device
    # OUT:  (all_preds: list, all_labels: list)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def compute_f1(all_preds: list, all_labels: list, num_classes: int) -> list:
    # WHAT: Compute per-class F1 score without sklearn
    # WHY:  sklearn blocked by security policy — pure Python replacement
    # IN:   all_preds, all_labels (lists of int), num_classes (int)
    # OUT:  list of F1 scores, one per class

    f1_scores = []
    for c in range(num_classes):
        tp = sum(p == c and l == c for p, l in zip(all_preds, all_labels))
        fp = sum(p == c and l != c for p, l in zip(all_preds, all_labels))
        fn = sum(p != c and l == c for p, l in zip(all_preds, all_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return f1_scores


def print_report(all_preds: list, all_labels: list) -> None:
    # WHAT: Print accuracy, per-class F1, and confusion matrix to terminal
    # WHY:  Gives a full picture of where the model is strong or weak
    # IN:   all_preds (list of int), all_labels (list of int)
    # OUT:  None (prints to terminal)

    n = len(CLASS_NAMES)
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) * 100
    print(f"\n[EVALUATE] Test Accuracy: {accuracy:.1f}%  ({correct}/{len(all_labels)} correct)\n")

    f1_scores = compute_f1(all_preds, all_labels, n)
    print(f"[EVALUATE] {'Class':<12} {'F1 Score':>10}")
    print(f"[EVALUATE] {'-'*24}")
    for name, f1 in zip(CLASS_NAMES, f1_scores):
        print(f"[EVALUATE] {name:<12} {f1:>10.3f}")
    avg_f1 = sum(f1_scores) / n
    print(f"[EVALUATE] {'avg / total':<12} {avg_f1:>10.3f}\n")

    print("[EVALUATE] Confusion matrix (rows=actual, cols=predicted):")
    cm = [[0] * n for _ in range(n)]
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>10}" + "".join(f"{v:>10}" for v in row))
    print()


def evaluate():
    # WHAT: Load saved model + test set, run evaluation, print full report
    # WHY:  Main entry point — one command gives complete model performance stats
    # IN:   None (reads MODEL_PATH and data folder from config.py)
    # OUT:  None (prints report to terminal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        print(f"[EVALUATE] No trained model found at {MODEL_PATH.name}")
        print("[EVALUATE] Run train.py first to generate a model checkpoint.")
        return

    _, _, test_loader, _ = load_datasets()
    if test_loader is None:
        print("[EVALUATE] No dataset found. See dataset.py instructions.")
        return

    model = load_model(device)
    print(f"[EVALUATE] Running on {len(test_loader.dataset):,} test images...")

    all_preds, all_labels = run_evaluation(model, test_loader, device)
    print_report(all_preds, all_labels)


if __name__ == "__main__":
    evaluate()
