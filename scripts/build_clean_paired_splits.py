"""
Build Clean Paired Zero-Leakage Dataset Splits & Verification Grid
=================================================================
Partitions authentic FaceForensics++ C23 face crops by primary actor ID:
- Train: Source IDs 000 - 019 (20 unique source identities, paired Real & Fake)
- Validation: Source IDs 020 - 024 (5 unique source identities, paired Real & Fake)
- Test: Source IDs 025 - 029 (5 unique source identities, paired Real & Fake)
Zero identity or video leakage across splits.
"""

import csv
import glob
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = Path(r"C:\Users\Shreya itagi\.gemini\antigravity\brain\c591f454-07df-4b27-bc7a-d87037631922")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildSplits")


def get_source_id(filepath: Path) -> str:
    stem = filepath.stem
    parts = stem.split("_")
    for p in parts[2:]:
        if len(p) == 3 and p.isdigit():
            return p
    return parts[2]


def get_category(filepath: Path) -> str:
    stem = filepath.stem
    parts = stem.split("_")
    # ff_<cat>_...
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    real_paths = sorted(list(Path(DATA_DIR / "processed" / "real_faces").glob("*.png")))
    fake_paths = sorted(list(Path(DATA_DIR / "processed" / "fake_faces").glob("*.png")))

    logger.info("Found %d Real face images and %d Fake face images on disk.", len(real_paths), len(fake_paths))

    all_records = []

    for p in real_paths:
        src_id = get_source_id(p)
        cat = get_category(p)
        all_records.append({
            "filepath": str(p.relative_to(ROOT_DIR)),
            "label": 0,
            "class_name": "REAL",
            "source_id": src_id,
            "category": cat,
            "dataset_name": "FaceForensics++ C23",
            "width": 224,
            "height": 224,
            "full_path": str(p),
        })

    for p in fake_paths:
        src_id = get_source_id(p)
        cat = get_category(p)
        all_records.append({
            "filepath": str(p.relative_to(ROOT_DIR)),
            "label": 1,
            "class_name": "DEEPFAKE",
            "source_id": src_id,
            "category": cat,
            "dataset_name": "FaceForensics++ C23",
            "width": 224,
            "height": 224,
            "full_path": str(p),
        })

    # Partition IDs
    train_ids = {f"{i:03d}" for i in range(20)}       # 000 - 019
    val_ids = {f"{i:03d}" for i in range(20, 25)}     # 020 - 024
    test_ids = {f"{i:03d}" for i in range(25, 30)}    # 025 - 029

    train_recs = [r for r in all_records if r["source_id"] in train_ids]
    val_recs = [r for r in all_records if r["source_id"] in val_ids]
    test_recs = [r for r in all_records if r["source_id"] in test_ids]

    fieldnames = ["filepath", "label", "class_name", "video_id", "category", "dataset_name", "width", "height"]

    for name, recs in [("train", train_recs), ("validation", val_recs), ("test", test_recs)]:
        out_csv = SPLITS_DIR / f"{name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in recs:
                writer.writerow({
                    "filepath": r["filepath"],
                    "label": r["label"],
                    "class_name": r["class_name"],
                    "video_id": r["source_id"],
                    "category": r["category"],
                    "dataset_name": r["dataset_name"],
                    "width": r["width"],
                    "height": r["height"],
                })
        n_real = sum(1 for r in recs if r["label"] == 0)
        n_fake = sum(1 for r in recs if r["label"] == 1)
        logger.info("%s.csv: Total=%d | Real=%d | Fake=%d | Unique IDs=%d", name, len(recs), n_real, n_fake, len(set(r['source_id'] for r in recs)))

    # Verification Grid: 30 REAL (one from each of 000-029) and 30 FAKE (one from each of 000-029)
    selected_real = []
    selected_fake = []

    for sid in sorted([f"{i:03d}" for i in range(30)]):
        reals_for_sid = [r for r in all_records if r["label"] == 0 and r["source_id"] == sid]
        fakes_for_sid = [r for r in all_records if r["label"] == 1 and r["source_id"] == sid]
        if reals_for_sid:
            selected_real.append(reals_for_sid[0])
        if fakes_for_sid:
            selected_fake.append(fakes_for_sid[0])

    logger.info("Verification sample selection: %d Real, %d Fake.", len(selected_real), len(selected_fake))

    # Build 12x5 montage
    thumb_w, thumb_h = 150, 150
    pad = 8
    header_h = 60
    footer_h = 24
    cols = 12
    rows = 5

    total_w = cols * thumb_w + (cols + 1) * pad
    total_h = header_h + rows * (thumb_h + footer_h) + (rows + 1) * pad

    canvas = np.full((total_h, total_w, 3), 248, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, "VeriFrame - Realistic FaceForensics++ Dataset Verification Grid (60 Samples)", (pad, 26), font, 0.75, (30, 41, 59), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Mandatory Verification: 30 REAL authentic human faces (Left) vs 30 FAKE manipulated human faces (Right)", (pad, 48), font, 0.45, (100, 116, 139), 1, cv2.LINE_AA)

    real_section_w = 6 * thumb_w + 6 * pad
    cv2.line(canvas, (real_section_w + pad // 2, 60), (real_section_w + pad // 2, total_h - 10), (148, 163, 184), 2)

    for idx, sample in enumerate(selected_real[:30]):
        r = idx // 6
        c = idx % 6
        x = pad + c * (thumb_w + pad)
        y = header_h + r * (thumb_h + footer_h + pad)

        img = cv2.imread(sample["full_path"])
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (34, 197, 94), 2)
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (240, 253, 244), -1)
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (34, 197, 94), 1)
        cv2.putText(canvas, f"REAL | {sample['source_id']}", (x + 4, y + thumb_h + 16), font, 0.35, (21, 128, 61), 1, cv2.LINE_AA)

    for idx, sample in enumerate(selected_fake[:30]):
        r = idx // 6
        c = 6 + (idx % 6)
        x = pad + c * (thumb_w + pad)
        y = header_h + r * (thumb_h + footer_h + pad)

        img = cv2.imread(sample["full_path"])
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        cat = sample["category"]
        border_color = (68, 68, 239) if cat == "Deepfakes" else (168, 85, 247) if cat == "Face2Face" else (234, 88, 12) if cat == "FaceSwap" else (225, 29, 72)
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), border_color, 2)
        canvas[y:y + thumb_h, x:x + thumb_w] = thumb

        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), (254, 242, 242), -1)
        cv2.rectangle(canvas, (x, y + thumb_h + 1), (x + thumb_w, y + thumb_h + footer_h), border_color, 1)
        short_cat = cat if len(cat) <= 9 else (cat[:8] + "..")
        cv2.putText(canvas, f"FAKE:{short_cat}", (x + 3, y + thumb_h + 16), font, 0.32, (185, 28, 28), 1, cv2.LINE_AA)

    grid_path = DATA_DIR / "dataset_verification_grid.png"
    cv2.imwrite(str(grid_path), canvas)
    logger.info("Saved verification grid image to: %s", grid_path)

    if ARTIFACT_DIR.exists():
        import shutil
        artifact_grid = ARTIFACT_DIR / "dataset_verification_grid.png"
        shutil.copy2(grid_path, artifact_grid)
        logger.info("Copied grid to artifact dir: %s", artifact_grid)

    meta_csv = DATA_DIR / "dataset_verification_metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "dataset_name", "category", "label", "source_video", "resolution", "filepath"])
        writer.writeheader()
        sample_id = 1
        for r in selected_real:
            writer.writerow({
                "sample_id": sample_id,
                "dataset_name": r["dataset_name"],
                "category": r["category"],
                "label": r["class_name"],
                "source_video": r["source_id"] + ".mp4",
                "resolution": f"{r['width']}x{r['height']}",
                "filepath": r["filepath"],
            })
            sample_id += 1
        for r in selected_fake:
            writer.writerow({
                "sample_id": sample_id,
                "dataset_name": r["dataset_name"],
                "category": r["category"],
                "label": r["class_name"],
                "source_video": r["source_id"] + ".mp4",
                "resolution": f"{r['width']}x{r['height']}",
                "filepath": r["filepath"],
            })
            sample_id += 1
    logger.info("Saved metadata CSV: %s", meta_csv)


if __name__ == "__main__":
    main()
