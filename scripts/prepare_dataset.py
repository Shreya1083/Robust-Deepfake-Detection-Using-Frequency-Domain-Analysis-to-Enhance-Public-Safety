"""
Dataset Preparation and Splitting Script
========================================
Orchestrates face extraction, preprocessing and video-level dataset splitting.

Critical Guarantees:
- Split strictly by source video ID to prevent data leakage.
- Celeb-DF v2 is preserved as an independent, unseen cross-dataset evaluation set.
- Generates train.csv, validation.csv, test.csv, and celeb_df_eval.csv.
- Verifies zero video ID overlap across train, validation and test splits.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CELEB_DF_EVAL_FILE,
    INTERIM_FAKE_FRAMES_DIR,
    INTERIM_REAL_FRAMES_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_FAKE_FACES_DIR,
    PROCESSED_REAL_FACES_DIR,
    SPLITS_DIR,
    TEST_SPLIT_FILE,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
)
from src.face_extraction import FaceCropResult, FaceExtractor
from src.preprocessing import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_frame_metadata_from_filename(filename: str) -> Optional[Dict]:
    """
    Parse metadata from standardized filename:
    <dataset>_<label>_<video_id>_frame<N>.jpg
    or
    <dataset>_<label>_<video_id>_f<N>.jpg
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    dataset = parts[0]
    class_name = parts[1].upper()  # "REAL" or "FAKE"
    label = 0 if class_name == "REAL" else 1

    # video_id can have underscores; last part is frame index
    frame_part = parts[-1]
    video_id = "_".join(parts[2:-1])

    # Extract integer frame index
    digits = "".join(filter(str.isdigit, frame_part))
    frame_idx = int(digits) if digits else 0

    return {
        "dataset": dataset,
        "class_name": "REAL" if label == 0 else "DEEPFAKE",
        "label": label,
        "video_id": video_id,
        "frame_idx": frame_idx,
    }


def extract_faces_from_interim_frames(
    interim_real_dir: Path = INTERIM_REAL_FRAMES_DIR,
    interim_fake_dir: Path = INTERIM_FAKE_FRAMES_DIR,
    processed_real_dir: Path = PROCESSED_REAL_FACES_DIR,
    processed_fake_dir: Path = PROCESSED_FAKE_FACES_DIR,
) -> List[Dict]:
    """
    Detect and crop faces from extracted frames and save to processed directories.
    Returns list of metadata dicts for all successfully extracted faces.
    """
    extractor = FaceExtractor()
    preprocessor = ImagePreprocessor()

    processed_real_dir.mkdir(parents=True, exist_ok=True)
    processed_fake_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    failures = 0

    # Collect all frame files
    frames_to_process: List[Tuple[Path, int]] = []
    if interim_real_dir.exists():
        for f in interim_real_dir.glob("*.jpg"):
            frames_to_process.append((f, 0))
    if interim_fake_dir.exists():
        for f in interim_fake_dir.glob("*.jpg"):
            frames_to_process.append((f, 1))

    if not frames_to_process:
        logger.warning("No interim frames found in %s or %s.", interim_real_dir, interim_fake_dir)
        return records

    logger.info("Extracting faces from %d interim frames...", len(frames_to_process))

    for frame_path, label in tqdm(frames_to_process, desc="Face extraction"):
        meta = parse_frame_metadata_from_filename(frame_path.name)
        if meta is None:
            # Fallback metadata
            meta = {
                "dataset": "faceforensics",
                "class_name": "REAL" if label == 0 else "DEEPFAKE",
                "label": label,
                "video_id": frame_path.stem,
                "frame_idx": 0,
            }

        target_dir = processed_real_dir if label == 0 else processed_fake_dir
        dest_filename = f"{meta['dataset']}_{meta['class_name'].lower()}_{meta['video_id']}_f{meta['frame_idx']:04d}.jpg"
        dest_path = target_dir / dest_filename

        # If already extracted and processed, record directly
        if dest_path.exists():
            records.append({
                "filepath": str(dest_path.relative_to(ROOT_DIR)),
                "label": meta["label"],
                "class_name": meta["class_name"],
                "dataset": meta["dataset"],
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"],
            })
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            failures += 1
            continue

        result = extractor.extract_face(frame)
        if not result.success or result.face_image is None:
            failures += 1
            continue

        # Standardize face crop dimensions to 256x256
        resized_face = preprocessor.resize(result.face_image)
        cv2.imwrite(str(dest_path), resized_face)

        records.append({
            "filepath": str(dest_path.relative_to(ROOT_DIR)),
            "label": meta["label"],
            "class_name": meta["class_name"],
            "dataset": meta["dataset"],
            "video_id": meta["video_id"],
            "frame_idx": meta["frame_idx"],
        })

    logger.info(
        "Face extraction complete: %d valid faces saved (%d frames had no face or failed).",
        len(records),
        failures,
    )
    return records


def split_dataset_by_video_id(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset strictly by video_id to prevent data leakage.

    Rules:
    - FaceForensics++ is split into Train (70%), Validation (15%), Test (15%).
    - Celeb-DF v2 is kept 100% separate as an independent evaluation set.
    """
    ff_records = [r for r in records if r["dataset"] == "faceforensics"]
    celeb_records = [r for r in records if r["dataset"] == "celeb_df"]

    if not ff_records:
        return [], [], [], celeb_records

    # Unique video IDs and their labels
    video_labels: Dict[str, int] = {}
    for r in ff_records:
        video_labels[r["video_id"]] = r["label"]

    unique_videos = np.array(list(video_labels.keys()))
    unique_labels = np.array([video_labels[vid] for vid in unique_videos])

    # First split: Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(unique_videos, unique_labels, groups=unique_videos))

    train_videos = set(unique_videos[train_idx])
    temp_videos = unique_videos[temp_idx]
    temp_labels = unique_labels[temp_idx]

    # Second split: Val vs Test
    test_sub_ratio = test_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_sub_ratio, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_videos, temp_labels, groups=temp_videos))

    val_videos = set(temp_videos[val_idx])
    test_videos = set(temp_videos[test_idx])

    # Assert zero leakage
    assert len(train_videos & val_videos) == 0, "DATA LEAKAGE: Overlap between Train and Val videos!"
    assert len(train_videos & test_videos) == 0, "DATA LEAKAGE: Overlap between Train and Test videos!"
    assert len(val_videos & test_videos) == 0, "DATA LEAKAGE: Overlap between Val and Test videos!"

    train_records = [r for r in ff_records if r["video_id"] in train_videos]
    val_records = [r for r in ff_records if r["video_id"] in val_videos]
    test_records = [r for r in ff_records if r["video_id"] in test_videos]

    logger.info(
        "Split complete (by video ID): Train=%d videos (%d faces), Val=%d videos (%d faces), Test=%d videos (%d faces).",
        len(train_videos),
        len(train_records),
        len(val_videos),
        len(val_records),
        len(test_videos),
        len(test_records),
    )
    return train_records, val_records, test_records, celeb_records


def write_split_csv(records: List[Dict], output_file: Path) -> None:
    """Write records to CSV with standard schema."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "class_name", "dataset", "video_id", "frame_idx"]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Saved %d records to %s", len(records), output_file)


def run_pipeline():
    """Execute the full dataset preparation and splitting pipeline."""
    logger.info("Starting Phase 2 Dataset Preparation Pipeline...")

    # 1. Face Extraction
    records = extract_faces_from_interim_frames()

    if not records:
        logger.warning(
            "No processed faces available. If you have raw videos, run:\n"
            "    python scripts/extract_frames.py\n"
            "then rerun prepare_dataset.py."
        )
        # Initialize empty CSVs with headers so files exist cleanly
        write_split_csv([], TRAIN_SPLIT_FILE)
        write_split_csv([], VAL_SPLIT_FILE)
        write_split_csv([], TEST_SPLIT_FILE)
        write_split_csv([], CELEB_DF_EVAL_FILE)
        return

    # 2. Video-Level Splitting
    train_rec, val_rec, test_rec, celeb_rec = split_dataset_by_video_id(records)

    # 3. Save CSV Splits
    write_split_csv(train_rec, TRAIN_SPLIT_FILE)
    write_split_csv(val_rec, VAL_SPLIT_FILE)
    write_split_csv(test_rec, TEST_SPLIT_FILE)
    write_split_csv(celeb_rec, CELEB_DF_EVAL_FILE)

    logger.info("Dataset preparation finished successfully.")


if __name__ == "__main__":
    run_pipeline()
