"""
Training Script for Spatial ResNet18 Deepfake Detection
======================================================
Trains the spatial CNN model on FaceForensics++ face crops.

Key Features:
- Pretrained ResNet18 with fine-tuning / backbone freezing option.
- CrossEntropyLoss with AdamW optimizer.
- Validation monitoring, learning rate scheduling and early stopping.
- Best model checkpoint saving with complete metadata.
- Reproducible random seeds.
- Safe handling when datasets are awaiting preparation.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.spatial_model import build_spatial_model, get_default_device
from src.config import (
    CHECKPOINTS_DIR,
    CLASS_LABELS,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
)
from src.dataset_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for a single epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation data. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def train_model(
    train_csv: Path = TRAIN_SPLIT_FILE,
    val_csv: Path = VAL_SPLIT_FILE,
    batch_size: int = 32,
    lr: float = 1e-4,
    epochs: int = 25,
    patience: int = 5,
    freeze_backbone: bool = False,
    fine_tune_layer4: bool = False,
    save_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict:
    """Run full model training pipeline."""
    set_seed(seed)

    if save_path is None:
        save_path = CHECKPOINTS_DIR / "resnet18_best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    logger.info("Using device: %s", device)

    # 1. Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
    )

    if len(train_loader.dataset) == 0:
        logger.warning(
            "Training dataset is empty (%s has 0 records).\n"
            "Please ensure raw videos are placed in data/raw/ and run:\n"
            "    python scripts/extract_frames.py\n"
            "    python scripts/prepare_dataset.py\n"
            "before executing model training.",
            train_csv,
        )
        return {"status": "empty_dataset", "records": 0}

    logger.info(
        "Loaded %d training samples and %d validation samples.",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # 2. Build Model
    model, _ = build_spatial_model(
        pretrained=True,
        freeze_backbone=freeze_backbone,
        fine_tune_layer4=fine_tune_layer4,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info("Starting training for %d epochs (Early stopping patience = %d)...", epochs, patience)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch,
            epochs,
            train_loss,
            train_acc * 100,
            val_loss,
            val_acc * 100,
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint with complete metadata
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "architecture": "resnet18",
                "num_classes": 2,
                "class_mapping": CLASS_LABELS,
                "input_size": (224, 224),
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "training_metadata": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                },
            }
            torch.save(checkpoint, save_path)
            logger.info("--> Best model saved to: %s (Val Loss: %.4f)", save_path, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Save training history
    history_file = save_path.parent / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best Epoch: %d | Best Val Loss: %.4f | Best Val Acc: %.2f%%", best_epoch, best_val_loss, best_val_acc * 100)
    return {
        "status": "success",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "checkpoint_path": str(save_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 spatial deepfake detector.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet18 backbone.")
    parser.add_argument("--fine-tune-layer4", action="store_true", help="Freeze conv1-layer3, fine-tune layer4 + classifier.")
    parser.add_argument("--save-path", type=str, default=None, help="Output checkpoint path.")
    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        fine_tune_layer4=args.fine_tune_layer4,
        save_path=Path(args.save_path) if args.save_path else None,
    )
