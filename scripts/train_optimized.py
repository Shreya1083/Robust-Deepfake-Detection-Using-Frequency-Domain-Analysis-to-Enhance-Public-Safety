"""
Optimized Model Training & Validation Threshold Calibration
===========================================================
Trains ResNet18 on authentic FaceForensics++ C23 dataset using:
- Pretrained ImageNet weights with frozen early layers (conv1 - layer3)
- Fine-tuned layer4 with low learning rate (2e-5)
- Trainable classifier head with adapted learning rate (5e-4) and 0.5 dropout
- AdamW optimizer with 1e-3 weight decay
- Checkpoint selection based on validation accuracy and loss
- Validation threshold sweep to minimize False Positive Rate (FPR)
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, TRAIN_SPLIT_FILE, VAL_SPLIT_FILE
from src.dataset_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainOptimized")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs_fake = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_probs_fake.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc, np.array(all_probs_fake), np.array(all_labels)


def find_optimal_threshold(probs_fake: np.ndarray, labels: np.ndarray) -> Dict:
    """Sweep threshold tau in [0.2, 0.8] to find threshold minimizing FPR while preserving high Recall."""
    best_tau = 0.50
    best_score = -1.0
    best_metrics = {}

    for tau in np.linspace(0.20, 0.80, 61):
        preds = (probs_fake >= tau).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        tpr = tp / max(tp + fn, 1)  # Recall on deepfakes
        fpr = fp / max(fp + tn, 1)  # False positive rate on real photos
        precision = tp / max(tp + fp, 1)
        acc = (tp + tn) / max(len(labels), 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-6)

        # Objective: High accuracy and balanced F1 with low FPR penalty
        score = f1 - 0.5 * fpr

        if score > best_score:
            best_score = score
            best_tau = round(float(tau), 3)
            best_metrics = {
                "threshold": best_tau,
                "accuracy": round(acc, 4),
                "precision": round(precision, 4),
                "recall": round(tpr, 4),
                "fpr": round(fpr, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

    return best_metrics


def main():
    set_seed(42)
    device = get_default_device()
    logger.info("Device for training: %s", device)

    train_loader, val_loader, _ = create_dataloaders(
        train_csv=TRAIN_SPLIT_FILE,
        val_csv=VAL_SPLIT_FILE,
        batch_size=16,
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    # Build model: Pretrained ResNet18 with 0.5 dropout
    model = SpatialResNet18(pretrained=True, num_classes=2, dropout_rate=0.5)
    model.freeze_early_layers()  # Freeze conv1 through layer3, keep layer4 + classifier trainable
    model.to(device)

    # Differential learning rates
    optimizer = optim.AdamW([
        {"params": model.backbone.layer4.parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 4e-4},
    ], weight_decay=1e-3)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    save_path = CHECKPOINTS_DIR / "resnet18_best.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info("Starting training for 12 epochs...")

    for epoch in range(1, 13):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, val_probs_fake, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        logger.info(
            "Epoch [%2d/12] Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%",
            epoch, t_loss, t_acc * 100, v_loss, v_acc * 100,
        )

        # Improvement criteria: higher accuracy, or equal accuracy with lower loss
        is_best = (v_acc > best_val_acc) or (v_acc == best_val_acc and v_loss < best_val_loss)
        if is_best:
            best_val_acc = v_acc
            best_val_loss = v_loss
            best_epoch = epoch

            # Calculate optimal threshold on validation predictions
            thresh_metrics = find_optimal_threshold(val_probs_fake, val_labels)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "val_acc": v_acc,
                "optimal_threshold": thresh_metrics.get("threshold", 0.50),
                "threshold_metrics": thresh_metrics,
                "classes": ["REAL", "DEEPFAKE"],
                "dataset": "FaceForensics++ C23 Authentic",
            }, save_path)
            logger.info("--> Saved BEST model at Epoch %d (Val Acc: %.2f%%, Optimal Tau: %.2f)", epoch, v_acc * 100, thresh_metrics.get("threshold", 0.50))

    # Save training history to json
    history_file = CHECKPOINTS_DIR / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to: %s", history_file)

    logger.info("Training complete. Best Epoch: %d (Val Acc: %.2f%%, Val Loss: %.4f)", best_epoch, best_val_acc * 100, best_val_loss)


if __name__ == "__main__":
    main()
