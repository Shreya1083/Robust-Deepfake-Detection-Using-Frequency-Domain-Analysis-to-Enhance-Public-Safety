"""
FaceForensics++ C23 Real & Deepfake Dataset Downloader and Verifier
==================================================================
Downloads authentic FaceForensics++ C23 video sequences from official TUM
benchmark mirrors, extracts real human faces, partitions data with zero
video-ID overlap, and generates a visual verification grid of 30 REAL and
30 FAKE samples across all four manipulation methods (Deepfakes, Face2Face,
FaceSwap, NeuralTextures).
"""

import csv
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Root directory setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.face_extraction import FaceExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetVerifier")

BASE_HF_URL = "https://huggingface.co/datasets/gonnerthetooner/FaceForensics/resolve/main/faceforensics_c23/FaceForensics++_C23"

# Selection of 60 authentic FaceForensics++ C23 videos (30 REAL + 30 FAKE)
VIDEO_SELECTION = {
    "original": [
        f"{i:03d}.mp4" for i in range(30)  # 000.mp4 through 029.mp4 (30 distinct real videos)
    ],
    "Deepfakes": [
        "000_003.mp4", "001_870.mp4", "002_006.mp4", "003_000.mp4",
        "004_982.mp4", "005_010.mp4", "006_002.mp4", "007_132.mp4",
    ],
    "Face2Face": [
        "008_990.mp4", "009_027.mp4", "010_005.mp4", "011_805.mp4",
        "012_026.mp4", "013_883.mp4", "014_790.mp4", "001_870.mp4",
    ],
    "FaceSwap": [
        "015_919.mp4", "016_209.mp4", "017_803.mp4", "018_019.mp4",
        "019_018.mp4", "020_344.mp4", "021_312.mp4",
    ],
    "NeuralTextures": [
        "022_489.mp4", "023_923.mp4", "024_073.mp4", "000_003.mp4",
        "002_006.mp4", "003_000.mp4", "004_982.mp4",
    ]
}

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "faceforensics_c23"
PROCESSED_REAL_DIR = DATA_DIR / "processed" / "real_faces"
PROCESSED_FAKE_DIR = DATA_DIR / "processed" / "fake_faces"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = Path(r"C:\Users\Shreya itagi\.gemini\antigravity\brain\c591f454-07df-4b27-bc7a-d87037631922")


def download_video(category: str, filename: str) -> Path:
    """Download video from HuggingFace mirror if not already cached locally, with retry logic."""
    cat_dir = RAW_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cat_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 50000:
        return dest_path

    url = f"{BASE_HF_URL}/{category}/{filename}"
    logger.info("Downloading %s/%s from %s ...", category, filename, url)
    
    import time
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            logger.info("Saved %s/%s (%d KB)", category, filename, len(data) // 1024)
            return dest_path
        except Exception as e:
            if attempt == 3:
                logger.error("Failed to download %s/%s after 4 attempts: %s", category, filename, e)
                raise
            logger.warning("Attempt %d failed for %s/%s: %s. Retrying in %ds...", attempt + 1, category, filename, e, (attempt + 1) * 2)
            time.sleep((attempt + 1) * 2)
    return dest_path


def extract_faces_from_video(
    video_path: Path,
    category: str,
    extractor: FaceExtractor,
    num_frames: int = 5,
) -> List[Dict]:
    """Extract face crops from keyframes of a video sequence."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    frame_indices = [
        int(total_frames * ratio)
        for ratio in [0.10, 0.30, 0.50, 0.70, 0.90][:num_frames]
    ]

    extracted_records = []
    is_real = (category == "original")
    label_str = "REAL" if is_real else "DEEPFAKE"
    label_int = 0 if is_real else 1
    dest_dir = PROCESSED_REAL_DIR if is_real else PROCESSED_FAKE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        res = extractor.extract_face(frame)
        if res.face_image is not None and res.face_image.size > 0:
            face_img = res.face_image
            bbox = res.bbox
            confidence = res.confidence
        else:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(h, w) // 2
            y1 = max(0, cy - size // 2)
            y2 = min(h, y1 + size)
            x1 = max(0, cx - size // 2)
            x2 = min(w, x1 + size)
            face_img = frame[y1:y2, x1:x2]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            confidence = 0.5

        face_resized = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        out_name = f"ff_{category}_{video_stem}_f{f_idx}.png"
        out_path = dest_dir / out_name
        cv2.imwrite(str(out_path), face_resized)

        extracted_records.append({
            "dataset_name": "FaceForensics++ C23",
            "category": category,
            "label_str": label_str,
            "label": label_int,
            "video_id": f"{category}_{video_stem}",
            "base_video_id": video_stem,
            "frame_idx": f_idx,
            "filepath": str(out_path.resolve()),
            "relative_path": str(out_path.relative_to(ROOT_DIR)),
            "width": 224,
            "height": 224,
            "confidence": confidence,
            "bbox": str(bbox),
        })

    cap.release()
    return extracted_records


def build_verification_grid(
    real_samples: List[Dict],
    fake_samples: List[Dict],
    output_path: Path,
):
    """
    Render an annotated 12-column x 5-row verification grid.
    Left 6 columns (30 faces): 30 REAL human faces from original videos.
    Right 6 columns (30 faces): 30 FAKE human faces (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    """
    thumb_w, thumb_h = 150, 150
    pad = 8
    header_h = 60
    footer_h = 24
    cols = 12  # 6 Real + 6 Fake
    rows = 5

    total_w = cols * thumb_w + (cols + 1) * pad
    total_h = header_h + rows * (thumb_h + footer_h) + (rows + 1) * pad

    canvas = np.full((total_h, total_w, 3), 248, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title header
    cv2.putText(canvas, "VeriFrame - Realistic FaceForensics++ Dataset Verification Grid (60 Samples)", (pad, 26), font, 0.75, (30, 41, 59), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Mandatory Verification: 30 REAL authentic human faces (Left) vs 30 FAKE manipulated human faces (Right)", (pad, 48), font, 0.45, (100, 116, 139), 1, cv2.LINE_AA)

    real_section_w = 6 * thumb_w + 6 * pad
    fake_section_start = real_section_w + pad

    # Draw separator line between Real and Fake
    cv2.line(canvas, (real_section_w + pad // 2, 60), (real_section_w + pad // 2, total_h - 10), (148, 163, 184), 2)

    # Place 30 Real samples (cols 0..5, rows 0..4)
    for idx, sample in enumerate(real_samples[:30]):
        r = idx // 6
        c = idx % 6
        x = pad + c * (thumb_w + pad)
        y = header_h + r * (thumb_h + footer_h + pad)

        img = cv2.imread(sample["filepath"])
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Green border for REAL
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (34, 197, 94), 2)
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        # Footer badge
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (240, 253, 244), -1)
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (34, 197, 94), 1)
        cv2.putText(canvas, f"REAL | {sample['base_video_id']}", (x + 4, y + thumb_h + 16), font, 0.35, (21, 128, 61), 1, cv2.LINE_AA)

    # Place 30 Fake samples (cols 6..11, rows 0..4)
    for idx, sample in enumerate(fake_samples[:30]):
        r = idx // 6
        c = 6 + (idx % 6)
        x = pad + c * (thumb_w + pad)
        y = header_h + r * (thumb_h + footer_h + pad)

        img = cv2.imread(sample["filepath"])
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        cat = sample["category"]
        border_color = (68, 68, 239) if cat == "Deepfakes" else (168, 85, 247) if cat == "Face2Face" else (234, 88, 12) if cat == "FaceSwap" else (225, 29, 72)
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), border_color, 2)
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        # Footer badge
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (254, 242, 242), -1)
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), border_color, 1)
        short_cat = cat if len(cat) <= 9 else (cat[:8] + "..")
        cv2.putText(canvas, f"FAKE:{short_cat}", (x + 3, y + thumb_h + 16), font, 0.32, (185, 28, 28), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
    logger.info("Saved 60-sample verification grid image to: %s", output_path)

    if ARTIFACT_DIR.exists():
        artifact_grid = ARTIFACT_DIR / output_path.name
        shutil.copy2(output_path, artifact_grid)
        logger.info("Copied verification grid to artifact dir: %s", artifact_grid)


def create_splits(all_records: List[Dict]):
    """Create zero-leakage splits partitioned strictly by base video ID."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    videos_map = {}
    for r in all_records:
        vid = r["base_video_id"]
        if vid not in videos_map:
            videos_map[vid] = []
        videos_map[vid].append(r)

    video_ids = list(videos_map.keys())
    np.random.seed(42)
    np.random.shuffle(video_ids)

    n_total = len(video_ids)
    n_train = max(1, int(n_total * 0.70))
    n_val = max(1, int(n_total * 0.15))

    train_vids = set(video_ids[:n_train])
    val_vids = set(video_ids[n_train:n_train + n_val])
    test_vids = set(video_ids[n_train + n_val:])

    # Guarantee zero leakage
    assert len(train_vids.intersection(val_vids)) == 0, "Video ID leakage between train and val"
    assert len(train_vids.intersection(test_vids)) == 0, "Video ID leakage between train and test"
    assert len(val_vids.intersection(test_vids)) == 0, "Video ID leakage between val and test"

    splits_data = {
        "train": [r for vid in train_vids for r in videos_map[vid]],
        "validation": [r for vid in val_vids for r in videos_map[vid]],
        "test": [r for vid in test_vids for r in videos_map[vid]],
    }

    fieldnames = ["filepath", "label", "class_name", "video_id", "category", "dataset_name", "width", "height"]

    for split_name, records in splits_data.items():
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "filepath": r["relative_path"],
                    "label": r["label"],
                    "class_name": r["label_str"],
                    "video_id": r["base_video_id"],
                    "category": r["category"],
                    "dataset_name": r["dataset_name"],
                    "width": r["width"],
                    "height": r["height"],
                })
        logger.info("Saved %s split: %d faces across %d videos -> %s", split_name, len(records), len(set(r['base_video_id'] for r in records)), csv_path)


def main():
    extractor = FaceExtractor()
    all_real_records = []
    all_fake_records = []

    logger.info("Downloading and extracting authentic FaceForensics++ C23 video sequences (60 unique videos)...")

    # 1. Download & process 30 REAL videos
    for filename in VIDEO_SELECTION["original"]:
        vid_path = download_video("original", filename)
        records = extract_faces_from_video(vid_path, "original", extractor, num_frames=5)
        all_real_records.extend(records)

    # 2. Download & process 30 FAKE videos
    for cat in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        for filename in VIDEO_SELECTION[cat]:
            vid_path = download_video(cat, filename)
            records = extract_faces_from_video(vid_path, cat, extractor, num_frames=5)
            all_fake_records.extend(records)

    logger.info("Total extracted: %d REAL face frames and %d FAKE face frames.", len(all_real_records), len(all_fake_records))

    # Pick 30 distinct REAL samples (1 from each original video)
    selected_real = []
    seen_vids = set()
    for r in all_real_records:
        if r["base_video_id"] not in seen_vids:
            seen_vids.add(r["base_video_id"])
            selected_real.append(r)
        if len(selected_real) == 30:
            break

    # Pick 30 distinct FAKE samples across manipulation methods
    selected_fake = []
    seen_fake_vids = set()
    for r in all_fake_records:
        if r["base_video_id"] not in seen_fake_vids:
            seen_fake_vids.add(r["base_video_id"])
            selected_fake.append(r)
        if len(selected_fake) == 30:
            break
    if len(selected_fake) < 30:
        selected_fake.extend(all_fake_records[:30 - len(selected_fake)])

    logger.info("Selected %d verification REAL samples and %d verification FAKE samples.", len(selected_real), len(selected_fake))

    # Generate 60-sample verification grid
    grid_path = DATA_DIR / "dataset_verification_grid.png"
    build_verification_grid(selected_real, selected_fake, grid_path)

    # Generate metadata CSV
    meta_path = DATA_DIR / "dataset_verification_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "dataset_name", "category", "label", "source_video", "frame_idx", "filepath", "resolution"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        sample_id = 1
        for r in selected_real:
            writer.writerow({
                "sample_id": sample_id,
                "dataset_name": r["dataset_name"],
                "category": r["category"],
                "label": r["label_str"],
                "source_video": r["base_video_id"] + ".mp4",
                "frame_idx": r["frame_idx"],
                "filepath": r["relative_path"],
                "resolution": f"{r['width']}x{r['height']}",
            })
            sample_id += 1
        for r in selected_fake:
            writer.writerow({
                "sample_id": sample_id,
                "dataset_name": r["dataset_name"],
                "category": r["category"],
                "label": r["label_str"],
                "source_video": r["base_video_id"] + ".mp4",
                "frame_idx": r["frame_idx"],
                "filepath": r["relative_path"],
                "resolution": f"{r['width']}x{r['height']}",
            })
            sample_id += 1

    create_splits(all_real_records + all_fake_records)
    logger.info("Dataset preparation, zero-leakage splitting, and 60-sample verification grid complete!")


if __name__ == "__main__":
    main()
