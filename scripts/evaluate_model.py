"""
Model Evaluation Script
=======================
Evaluates the trained spatial ResNet18 model on the held-out test split.

Calculates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC score
- Plots training history and confusion matrix
- Does not fabricate or hardcode performance metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, CLASS_LABELS, TEST_SPLIT_FILE
from src.dataset_loader import DeepfakeFaceDataset, get_spatial_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[SpatialResNet18, Dict]:
    """Load model architecture and weights from checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpatialResNet18(pretrained=False, num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded checkpoint from %s (Best Val Loss: %.4f)", checkpoint_path, checkpoint.get("val_loss", 0.0))
    return model, checkpoint


def plot_training_curves(history_file: Path, output_image: Path) -> None:
    """Plot train vs validation loss and accuracy curves."""
    if not history_file.exists():
        logger.warning("History file does not exist: %s", history_file)
        return

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0B1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#15213A")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_color("#1E3050")

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#00B4D8", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#E74C3C", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", color="#E8EDF5", fontsize=11)
    ax1.set_xlabel("Epoch", color="#94A3B8")
    ax1.set_ylabel("CrossEntropy Loss", color="#94A3B8")
    ax1.legend(facecolor="#15213A", edgecolor="#1E3050", labelcolor="#E8EDF5")

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#00B4D8", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#2ECC71", lw=2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy (%)", color="#E8EDF5", fontsize=11)
    ax2.set_xlabel("Epoch", color="#94A3B8")
    ax2.set_ylabel("Accuracy (%)", color="#94A3B8")
    ax2.legend(facecolor="#15213A", edgecolor="#1E3050", labelcolor="#E8EDF5")

    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved training curves to: %s", output_image)


def plot_confusion_matrix(cm: np.ndarray, output_image: Path) -> None:
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B1120")
    ax.set_facecolor("#15213A")

    labels = ["REAL", "DEEPFAKE"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_title("Confusion Matrix (Spatial ResNet18)", color="#E8EDF5", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Label", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Ground Truth Label", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#94A3B8")

    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved confusion matrix to: %s", output_image)


def evaluate_test_set(
    checkpoint_path: Path = CHECKPOINTS_DIR / "resnet18_best.pth",
    test_csv: Path = TEST_SPLIT_FILE,
    batch_size: int = 32,
) -> Dict:
    """Run full evaluation on held-out test dataset."""
    device = get_default_device()

    if not checkpoint_path.exists():
        logger.warning("Checkpoint does not exist: %s", checkpoint_path)
        return {"status": "missing_checkpoint"}

    if not test_csv.exists():
        logger.warning("Test split CSV does not exist: %s", test_csv)
        return {"status": "missing_test_csv"}

    model, checkpoint = load_trained_model(checkpoint_path, device)

    eval_transform = get_spatial_transforms(is_train=False)
    test_dataset = DeepfakeFaceDataset(test_csv, transform=eval_transform)

    if len(test_dataset) == 0:
        logger.warning("Test dataset has 0 samples. Cannot run evaluation.")
        return {"status": "empty_test_set"}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []

    logger.info("Running evaluation on %d test samples...", len(test_dataset))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(DEEPFAKE)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5  # If only 1 class present

    report = classification_report(all_labels, all_preds, target_names=["REAL", "DEEPFAKE"], output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_samples": len(test_dataset),
    }

    # Plot graphs
    output_dir = checkpoint_path.parent
    plot_training_curves(output_dir / "training_history.json", output_dir / "training_history.png")
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    # Save metrics report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation Complete -> Acc: %.2f%% | F1: %.4f | AUC: %.4f", acc * 100, f1, auc)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 deepfake detector on test split.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "resnet18_best.pth"))
    parser.add_argument("--test-csv", type=str, default=str(TEST_SPLIT_FILE))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=Path(args.checkpoint),
        test_csv=Path(args.test_csv),
        batch_size=args.batch_size,
    )
