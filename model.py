# model.py
# Builds the SafeFrame classifier: pretrained ResNet50 + custom 5-class head.

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DROPOUT_RATE, HIDDEN_DIM, MODEL_PATH


def build_model(device: torch.device) -> nn.Module:
    # WHAT: Load pretrained ResNet50, freeze early layers, attach custom head
    # WHY:  Transfer learning — ImageNet weights already understand shapes/textures
    # IN:   device (cpu or cuda)
    # OUT:  model ready for inference (or fine-tuning)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze ALL layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks (layer3 + layer4) so they can adapt
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final FC layer with our custom head
    # ResNet50 outputs 2048 features → we map to 5 classes
    model.fc = nn.Sequential(
        nn.Linear(2048, HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(HIDDEN_DIM, NUM_CLASSES),
    )

    model = model.to(device)
    return model


def load_model(device: torch.device) -> nn.Module:
    # WHAT: Load model weights from disk if available, else return fresh model
    # WHY:  Lets moderate.py work with a saved checkpoint after training
    # IN:   device (cpu or cuda)
    # OUT:  model (loaded from file OR fresh pretrained ResNet50)

    model = build_model(device)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[MODEL]    Loaded weights from {MODEL_PATH.name}")
    else:
        print("[MODEL]    No saved weights found — using pretrained ResNet50 backbone")
        print("[MODEL]    (Run train.py to fine-tune on NSFW data)")

    model.eval()
    return model


def print_model_summary(model: nn.Module) -> None:
    # WHAT: Count and display trainable vs frozen parameters
    # WHY:  Confirms the freeze worked — important for debugging transfer learning
    # IN:   model (nn.Module)
    # OUT:  None (prints to terminal)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"[MODEL]    ResNet50 ready — {trainable/1e6:.1f}M trainable params "
          f"| {frozen/1e6:.1f}M frozen | {total/1e6:.1f}M total")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL]    Device: {device}")
    model = load_model(device)
    print_model_summary(model)
