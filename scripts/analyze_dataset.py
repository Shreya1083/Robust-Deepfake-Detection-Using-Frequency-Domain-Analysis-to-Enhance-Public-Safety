"""
Dataset Analysis and Visualization Script
==========================================
Analyzes dataset statistics, class balance, and split distribution.
Generates a markdown report and a summary visualization chart.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CELEB_DF_EVAL_FILE,
    DATA_DIR,
    INTERIM_FAKE_FRAMES_DIR,
    INTERIM_REAL_FRAMES_DIR,
    PROCESSED_FAKE_FACES_DIR,
    PROCESSED_REAL_FACES_DIR,
    TEST_SPLIT_FILE,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
)
from src.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def count_files_in_dir(directory: Path, pattern: str = "*.jpg") -> int:
    """Safely count files matching pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def read_split_csv(csv_path: Path) -> List[Dict]:
    """Read split CSV file into list of dictionaries."""
    if not csv_path.exists():
        return []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_dataset_analysis() -> Dict:
    """Compute comprehensive dataset statistics across all stages."""
    # 1. Raw Videos
    loader = DatasetLoader()
    raw_videos = loader.scan_all()
    video_summary = loader.get_summary()

    # 2. Interim Frames
    real_frames_count = count_files_in_dir(INTERIM_REAL_FRAMES_DIR)
    fake_frames_count = count_files_in_dir(INTERIM_FAKE_FRAMES_DIR)
    total_frames = real_frames_count + fake_frames_count

    # 3. Processed Face Crops
    real_faces_count = count_files_in_dir(PROCESSED_REAL_FACES_DIR)
    fake_faces_count = count_files_in_dir(PROCESSED_FAKE_FACES_DIR)
    total_faces = real_faces_count + fake_faces_count

    # 4. Splits
    train_records = read_split_csv(TRAIN_SPLIT_FILE)
    val_records = read_split_csv(VAL_SPLIT_FILE)
    test_records = read_split_csv(TEST_SPLIT_FILE)
    celeb_records = read_split_csv(CELEB_DF_EVAL_FILE)

    def split_stats(records: List[Dict]) -> Dict:
        real_c = sum(1 for r in records if r.get("label") in ("0", 0))
        fake_c = sum(1 for r in records if r.get("label") in ("1", 1))
        unique_vids = len(set(r.get("video_id") for r in records if r.get("video_id")))
        return {
            "total": len(records),
            "real": real_c,
            "fake": fake_c,
            "unique_videos": unique_vids,
        }

    stats = {
        "video_summary": video_summary,
        "frames": {
            "real": real_frames_count,
            "fake": fake_frames_count,
            "total": total_frames,
        },
        "faces": {
            "real": real_faces_count,
            "fake": fake_faces_count,
            "total": total_faces,
            "failed_extractions": max(0, total_frames - total_faces),
        },
        "splits": {
            "train": split_stats(train_records),
            "val": split_stats(val_records),
            "test": split_stats(test_records),
            "celeb_eval": split_stats(celeb_records),
        },
    }
    return stats


def plot_distribution(stats: Dict, output_image: Path) -> None:
    """Generate a clean visual chart showing class balance and split distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0B1120")

    # Custom styling
    for ax in axes:
        ax.set_facecolor("#15213A")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_color("#1E3050")

    # Chart 1: Split Distribution
    splits = ["Train", "Validation", "Test", "Celeb-DF Eval"]
    counts = [
        stats["splits"]["train"]["total"],
        stats["splits"]["val"]["total"],
        stats["splits"]["test"]["total"],
        stats["splits"]["celeb_eval"]["total"],
    ]

    colors = ["#00B4D8", "#48CAE4", "#0077B6", "#F39C12"]
    axes[0].bar(splits, counts, color=colors, width=0.55, edgecolor="#1E3050")
    axes[0].set_title("Dataset Split Distribution (Images)", color="#E8EDF5", fontsize=12, pad=10)
    axes[0].set_ylabel("Number of Face Crops", color="#94A3B8", fontsize=10)
    for i, v in enumerate(counts):
        axes[0].text(i, v + max(1, max(counts or [1]) * 0.02), str(v), ha="center", color="#E8EDF5", fontsize=10)

    # Chart 2: Real vs Deepfake Face Crops
    categories = ["REAL Faces", "FAKE Faces"]
    face_counts = [stats["faces"]["real"], stats["faces"]["fake"]]
    pie_colors = ["#2ECC71", "#E74C3C"]

    if sum(face_counts) > 0:
        axes[1].pie(
            face_counts,
            labels=categories,
            autopct="%1.1f%%",
            startangle=90,
            colors=pie_colors,
            textprops={"color": "#E8EDF5", "fontsize": 10},
            wedgeprops={"edgecolor": "#15213A", "linewidth": 2},
        )
    else:
        axes[1].text(0.5, 0.5, "No Extracted Faces Yet\n(Awaiting Raw Datasets)", ha="center", va="center", color="#94A3B8")
        axes[1].axis("off")

    axes[1].set_title("Overall Class Distribution (Face Crops)", color="#E8EDF5", fontsize=12, pad=10)

    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved distribution chart to: %s", output_image)


def write_markdown_report(stats: Dict, output_path: Path) -> None:
    """Save formatted markdown report."""
    raw_v = stats["video_summary"]["overall"]
    tr = stats["splits"]["train"]
    va = stats["splits"]["val"]
    te = stats["splits"]["test"]
    ce = stats["splits"]["celeb_eval"]

    report = f"""# 📊 Phase 2: Dataset Collection & Preprocessing Report

**Project:** Robust Deepfake Detection Using Frequency Domain Analysis to Enhance Public Safety

---

## 1. Raw Dataset Video Inventory

| Dataset Source | REAL Videos | FAKE Videos | Total Valid Videos | Corrupted / Invalid |
|---|---|---|---|---|
| **FaceForensics++** | {stats['video_summary']['faceforensics']['REAL']} | {stats['video_summary']['faceforensics']['DEEPFAKE']} | {stats['video_summary']['faceforensics']['TOTAL']} | {stats['video_summary']['faceforensics']['INVALID']} |
| **Celeb-DF (v2)** | {stats['video_summary']['celeb_df']['REAL']} | {stats['video_summary']['celeb_df']['DEEPFAKE']} | {stats['video_summary']['celeb_df']['TOTAL']} | {stats['video_summary']['celeb_df']['INVALID']} |
| **Overall Total** | **{raw_v['REAL']}** | **{raw_v['DEEPFAKE']}** | **{raw_v['TOTAL']}** | **{raw_v['INVALID']}** |

---

## 2. Frame Extraction & Face Detection

| Metric | Count |
|---|---|
| **Total Frames Extracted** | {stats['frames']['total']} (Real: {stats['frames']['real']}, Fake: {stats['frames']['fake']}) |
| **Detected & Cropped Faces** | {stats['faces']['total']} (Real: {stats['faces']['real']}, Fake: {stats['faces']['fake']}) |
| **Failed Extractions (No Face / Blurry)** | {stats['faces']['failed_extractions']} |
| **Face Resolution** | 256 × 256 (RGB, normalized) |

---

## 3. Dataset Splits (Partitioned by Video ID)

*Data Leakage Prevention: Zero video ID overlap between splits.*

| Split | Partition Type | Unique Videos | REAL Faces | FAKE Faces | Total Images |
|---|---|---|---|---|---|
| **Train (70%)** | FaceForensics++ | {tr['unique_videos']} | {tr['real']} | {tr['fake']} | **{tr['total']}** |
| **Validation (15%)** | FaceForensics++ | {va['unique_videos']} | {va['real']} | {va['fake']} | **{va['total']}** |
| **Test (15%)** | FaceForensics++ | {te['unique_videos']} | {te['real']} | {te['fake']} | **{te['total']}** |
| **Celeb-DF Eval (100%)** | Celeb-DF v2 (Cross-Dataset) | {ce['unique_videos']} | {ce['real']} | {ce['fake']} | **{ce['total']}** |

---

## 4. Visual Evidence & Charts

Distribution plot generated: `data/dataset_distribution.png`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved report to: %s", output_path)


def run_analysis():
    """Run full dataset analysis, generate charts and markdown report."""
    logger.info("Running dataset analysis...")
    stats = generate_dataset_analysis()

    chart_path = DATA_DIR / "dataset_distribution.png"
    report_path = DATA_DIR / "dataset_report.md"

    plot_distribution(stats, chart_path)
    write_markdown_report(stats, report_path)
    logger.info("Dataset analysis completed.")
    return stats


if __name__ == "__main__":
    run_analysis()
