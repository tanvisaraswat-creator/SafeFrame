# train.py
# Trains the SafeFrame ResNet50 model on the NSFW dataset.
# Optimisations: AdamW, mixed precision (AMP), early stopping, LR scheduler.

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from config import MODEL_PATH, NUM_CLASSES
from model import build_model, print_model_summary
from dataset import load_datasets

# ── Hyperparameters (all tunable here) ────────────────────────────────────────
EPOCHS        = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 4      # stop if val accuracy doesn't improve for 4 epochs


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    # WHAT: Run one full pass over training data, return average loss
    # WHY:  Separated so train loop stays clean and readable
    # IN:   model, dataloader, loss fn, optimizer, AMP scaler, device
    # OUT:  average loss (float)

    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device) -> tuple:
    # WHAT: Run model on val/test set, return loss and accuracy
    # WHY:  Tracks generalisation — if val acc stops improving, we stop training
    # IN:   model, dataloader, loss fn, device
    # OUT:  (avg_loss: float, accuracy: float as percentage)

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return total_loss / len(loader), accuracy


def train():
    # WHAT: Full training loop with early stopping and model checkpoint saving
    # WHY:  Main entry point — ties together data, model, optimiser, and logging
    # IN:   None (reads all config from config.py and dataset.py)
    # OUT:  None (saves best model to MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN]    Device: {device}")

    train_loader, val_loader, _, class_names = load_datasets()
    if train_loader is None:
        print("[TRAIN]    Aborting — no dataset found. See dataset.py instructions.")
        return

    model     = build_model(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_val_acc  = 0.0
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"[TRAIN]    Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[TRAIN]    Saved best model ({val_acc:.1f}%) to {MODEL_PATH.name}")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[TRAIN]    Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
                break

    print(f"[TRAIN]    Done. Best val accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    train()
