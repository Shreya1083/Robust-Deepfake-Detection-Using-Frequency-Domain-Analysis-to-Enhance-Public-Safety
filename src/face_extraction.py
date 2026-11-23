"""
Face Detection and Extraction Module
====================================
Detects and crops facial regions from video frames using OpenCV.

Features:
- Offline OpenCV Haar-Cascade detector (guaranteed available).
- Configurable bounding box padding to capture facial boundaries.
- Graceful handling of frames with 0 faces or multiple faces.
- Preservation of original video IDs, frame indices and class labels.
- Logging and tracking of extraction statistics and failures.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    FACE_MIN_SIZE,
    FACE_PADDING_RATIO,
    IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class FaceCropResult:
    """Stores the result of a face detection attempt."""

    success: bool
    face_image: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    confidence: float = 1.0
    error_reason: Optional[str] = None


class FaceExtractor:
    """Detects, crops and saves facial regions from video frames."""

    def __init__(
        self,
        padding_ratio: float = FACE_PADDING_RATIO,
        min_size: Tuple[int, int] = FACE_MIN_SIZE,
        cascade_path: Optional[str] = None,
    ):
        self.padding_ratio = padding_ratio
        self.min_size = min_size

        # Load OpenCV's built-in frontal face Haar Cascade
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade from: %s", cascade_path)
            raise RuntimeError(f"Could not load face cascade from {cascade_path}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in a single frame.
        Returns a list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale for Haar detection
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Equalize histogram for contrast enhancement
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return [tuple(map(int, box)) for box in faces]

    def crop_face_with_padding(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Crop face region with proportional padding around edges to include
        jawline, forehead and blending boundaries.
        """
        h_img, w_img = frame.shape[:2]
        x, y, w, h = bbox

        # Compute padding
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        # Clamped coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        return frame[y1:y2, x1:x2].copy()

    def extract_face(
        self,
        frame: np.ndarray,
        strategy: str = "largest",
    ) -> FaceCropResult:
        """
        Detect and crop a facial region from a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array.
        strategy : str
            "largest": select the face with the largest area (default for deepfake benchmarks).
            "first": select the first detected face.

        Returns
        -------
        FaceCropResult
        """
        if frame is None or frame.size == 0:
            return FaceCropResult(success=False, error_reason="Frame is None or empty.")

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return FaceCropResult(success=False, error_reason="No face detected in frame.")

        if strategy == "largest":
            # Select face with maximum area (w * h)
            best_bbox = max(faces, key=lambda b: b[2] * b[3])
        else:
            best_bbox = faces[0]

        try:
            cropped = self.crop_face_with_padding(frame, best_bbox)
            if cropped.size == 0:
                return FaceCropResult(success=False, error_reason="Cropped face region is empty.")

            return FaceCropResult(
                success=True,
                face_image=cropped,
                bbox=best_bbox,
                confidence=1.0,
            )
        except Exception as e:
            return FaceCropResult(success=False, error_reason=f"Cropping exception: {str(e)}")

    def extract_and_save(
        self,
        frame: np.ndarray,
        output_dir: Path,
        dataset_name: str,
        label: int,
        video_id: str,
        frame_idx: int,
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Detect face in frame and save crop to output_dir with standardized naming:
        <dataset>_<label>_<video_id>_f<frame_idx>.jpg
        """
        result = self.extract_face(frame)
        if not result.success:
            return False, None, result.error_reason

        label_name = "real" if label == 0 else "fake"
        filename = f"{dataset_name}_{label_name}_{video_id}_f{frame_idx:04d}.jpg"
        save_path = output_dir / filename

        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), result.face_image)
        return True, save_path, None
