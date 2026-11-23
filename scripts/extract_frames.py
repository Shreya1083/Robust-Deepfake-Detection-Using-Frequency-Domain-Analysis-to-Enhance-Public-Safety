"""
Frame Extraction Script
=======================
Extracts representative video frames from raw video datasets using OpenCV.

Features:
- Configurable sampling rate (e.g., every 10th frame) to avoid temporal redundancy.
- Maximum frames cap per video to ensure class and video balance.
- Video ID and frame index preservation.
- Organized interim frame storage.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FRAME_SAMPLE_RATE,
    INTERIM_FAKE_FRAMES_DIR,
    INTERIM_REAL_FRAMES_DIR,
    MAX_FRAMES_PER_VIDEO,
)
from src.dataset_loader import DatasetLoader, VideoMetadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    dataset_name: str,
    label: int,
    video_id: str,
    sample_rate: int = FRAME_SAMPLE_RATE,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> List[Path]:
    """
    Extract evenly sampled frames from a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    output_dir : Path
        Directory where frames will be saved.
    dataset_name : str
        Dataset origin ("faceforensics" or "celeb_df").
    label : int
        0: REAL, 1: DEEPFAKE.
    video_id : str
        Unique video identifier.
    sample_rate : int
        Extract 1 frame every `sample_rate` frames.
    max_frames : int
        Maximum number of frames to extract from this video.

    Returns
    -------
    List[Path]
        List of saved frame filepaths.
    """
    saved_frames: List[Path] = []

    if not video_path.exists():
        logger.warning("Video does not exist: %s", video_path)
        return saved_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return saved_frames

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return saved_frames

    output_dir.mkdir(parents=True, exist_ok=True)
    label_str = "real" if label == 0 else "fake"

    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frame_filename = (
                f"{dataset_name}_{label_str}_{video_id}_frame{frame_idx:05d}.jpg"
            )
            save_path = output_dir / frame_filename

            # Save frame if not already extracted
            if not save_path.exists():
                cv2.imwrite(str(save_path), frame)

            saved_frames.append(save_path)
            extracted_count += 1

            if extracted_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_frames


def run_extraction(
    sample_rate: int = FRAME_SAMPLE_RATE,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    dataset_filter: Optional[str] = None,
) -> Dict[str, int]:
    """Extract frames for all discovered videos."""
    loader = DatasetLoader()
    all_videos = loader.scan_all()

    if not all_videos:
        logger.warning(
            "No raw video files found. Please ensure raw datasets are placed in data/raw/."
        )
        return {"total_videos": 0, "extracted_frames": 0}

    valid_videos = [v for v in all_videos if v.is_valid]
    if dataset_filter:
        valid_videos = [v for v in valid_videos if v.dataset_name == dataset_filter]

    logger.info("Extracting frames from %d valid videos...", len(valid_videos))

    total_frames = 0
    for video in tqdm(valid_videos, desc="Extracting frames"):
        target_dir = (
            INTERIM_REAL_FRAMES_DIR if video.label == 0 else INTERIM_FAKE_FRAMES_DIR
        )
        frames = extract_frames_from_video(
            video_path=video.filepath,
            output_dir=target_dir,
            dataset_name=video.dataset_name,
            label=video.label,
            video_id=video.video_id,
            sample_rate=sample_rate,
            max_frames=max_frames,
        )
        total_frames += len(frames)

    logger.info(
        "Frame extraction complete: %d frames extracted from %d videos.",
        total_frames,
        len(valid_videos),
    )
    return {"total_videos": len(valid_videos), "extracted_frames": total_frames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames for deepfake detection.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=FRAME_SAMPLE_RATE,
        help="Extract 1 frame every N frames (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES_PER_VIDEO,
        help="Max frames per video (default: 20).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["faceforensics", "celeb_df"],
        help="Filter extraction by dataset name.",
    )
    args = parser.parse_args()

    run_extraction(
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        dataset_filter=args.dataset,
    )
