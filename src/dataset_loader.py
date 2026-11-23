"""
Dataset Loader Module
=====================
Discovers, validates and catalogues video files from FaceForensics++ and
Celeb-DF v2 raw datasets.

Labels are strictly derived from verified directory structures and metadata:
    0: REAL (authentic)
    1: DEEPFAKE (synthesized / manipulated)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2

from src.config import (
    CDF_FAKE_DIR,
    CDF_REAL_DIR,
    CELEB_DF_DIR,
    CLASS_LABELS,
    FACEFORENSICS_DIR,
    FF_FAKE_DIR,
    FF_REAL_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Stores verified metadata for a single video file."""

    video_id: str
    filepath: Path
    dataset_name: str  # "faceforensics" or "celeb_df"
    label: int  # 0: REAL, 1: DEEPFAKE
    class_name: str  # "REAL" or "DEEPFAKE"
    is_valid: bool
    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    error_message: Optional[str] = None


class DatasetLoader:
    """Discovers, validates, and tracks raw video files across datasets."""

    def __init__(
        self,
        ff_real_dir: Optional[Path] = None,
        ff_fake_dir: Optional[Path] = None,
        cdf_real_dir: Optional[Path] = None,
        cdf_fake_dir: Optional[Path] = None,
    ):
        self.ff_real_dir = Path(ff_real_dir or FF_REAL_DIR)
        self.ff_fake_dir = Path(ff_fake_dir or FF_FAKE_DIR)
        self.cdf_real_dir = Path(cdf_real_dir or CDF_REAL_DIR)
        self.cdf_fake_dir = Path(cdf_fake_dir or CDF_FAKE_DIR)

        self.videos: List[VideoMetadata] = []
        self.corrupted_files: List[Tuple[Path, str]] = []

    @staticmethod
    def extract_video_id(filepath: Path) -> str:
        """
        Derive a clean, deterministic video ID from the file path.
        Preserves original stem and parent directory context if applicable.
        """
        return filepath.stem

    @classmethod
    def validate_video_file(cls, filepath: Path) -> Tuple[bool, Dict, Optional[str]]:
        """
        Validate whether a video file exists, has a supported format,
        and can be decoded by OpenCV.
        """
        if not filepath.exists():
            return False, {}, f"File does not exist: {filepath}"

        if filepath.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return (
                False,
                {},
                f"Unsupported format '{filepath.suffix}'. Expected one of {SUPPORTED_VIDEO_EXTENSIONS}",
            )

        if filepath.stat().st_size == 0:
            return False, {}, "File is empty (0 bytes)."

        try:
            cap = cv2.VideoCapture(str(filepath))
            if not cap.isOpened():
                return False, {}, "OpenCV VideoCapture failed to open file."

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if frame_count <= 0 or width <= 0 or height <= 0:
                return False, {}, f"Invalid video dimensions or frame count: {width}x{height}, frames={frame_count}"

            duration = (frame_count / fps) if fps > 0 else 0.0
            metadata = {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_sec": round(duration, 2),
            }
            return True, metadata, None

        except Exception as e:
            return False, {}, f"Exception while reading video: {str(e)}"

    def _scan_directory(
        self,
        directory: Path,
        dataset_name: str,
        label: int,
    ) -> List[VideoMetadata]:
        """Scan a directory for video files matching the specified label."""
        discovered: List[VideoMetadata] = []

        if not directory.exists():
            logger.debug("Directory not found (skipped): %s", directory)
            return discovered

        class_name = CLASS_LABELS.get(label, "UNKNOWN")

        # Search recursively for supported video formats
        found_paths: Set[Path] = set()
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            found_paths.update(directory.glob(f"*{ext}"))
            found_paths.update(directory.glob(f"**/*{ext}"))

        for path in sorted(found_paths):
            video_id = self.extract_video_id(path)
            is_valid, props, error_msg = self.validate_video_file(path)

            if is_valid:
                meta = VideoMetadata(
                    video_id=video_id,
                    filepath=path,
                    dataset_name=dataset_name,
                    label=label,
                    class_name=class_name,
                    is_valid=True,
                    frame_count=props["frame_count"],
                    fps=props["fps"],
                    width=props["width"],
                    height=props["height"],
                    duration_sec=props["duration_sec"],
                )
                discovered.append(meta)
            else:
                meta = VideoMetadata(
                    video_id=video_id,
                    filepath=path,
                    dataset_name=dataset_name,
                    label=label,
                    class_name=class_name,
                    is_valid=False,
                    error_message=error_msg,
                )
                self.corrupted_files.append((path, error_msg or "Unknown error"))
                logger.warning("Corrupted or invalid video: %s (%s)", path.name, error_msg)

        return discovered

    def scan_all(self, validate_integrity: bool = True) -> List[VideoMetadata]:
        """
        Scan all raw dataset directories (FaceForensics++ and Celeb-DF v2)
        for both real and fake videos.
        """
        self.videos.clear()
        self.corrupted_files.clear()

        # 1. FaceForensics++ Real
        ff_real = self._scan_directory(self.ff_real_dir, "faceforensics", 0)
        self.videos.extend(ff_real)

        # 2. FaceForensics++ Fake
        ff_fake = self._scan_directory(self.ff_fake_dir, "faceforensics", 1)
        self.videos.extend(ff_fake)

        # 3. Celeb-DF v2 Real
        cdf_real = self._scan_directory(self.cdf_real_dir, "celeb_df", 0)
        self.videos.extend(cdf_real)

        # 4. Celeb-DF v2 Fake
        cdf_fake = self._scan_directory(self.cdf_fake_dir, "celeb_df", 1)
        self.videos.extend(cdf_fake)

        logger.info(
            "Scan complete: %d valid videos found across datasets (%d corrupted/invalid).",
            len([v for v in self.videos if v.is_valid]),
            len(self.corrupted_files),
        )
        return self.videos

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """Return a structured dictionary of video counts per dataset and label."""
        summary: Dict[str, Dict[str, int]] = {
            "faceforensics": {"REAL": 0, "DEEPFAKE": 0, "TOTAL": 0, "INVALID": 0},
            "celeb_df": {"REAL": 0, "DEEPFAKE": 0, "TOTAL": 0, "INVALID": 0},
            "overall": {"REAL": 0, "DEEPFAKE": 0, "TOTAL": 0, "INVALID": 0},
        }

        for v in self.videos:
            ds = v.dataset_name
            if ds not in summary:
                summary[ds] = {"REAL": 0, "DEEPFAKE": 0, "TOTAL": 0, "INVALID": 0}

            if v.is_valid:
                cls_key = v.class_name
                summary[ds][cls_key] += 1
                summary[ds]["TOTAL"] += 1
                summary["overall"][cls_key] += 1
                summary["overall"]["TOTAL"] += 1
            else:
                summary[ds]["INVALID"] += 1
                summary["overall"]["INVALID"] += 1

        return summary
