"""
Phase 2 Unit and Integration Tests
==================================
Tests dataset loading, frame extraction, face detection, image preprocessing,
and video-level data leakage prevention.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.extract_frames import extract_frames_from_video
from scripts.prepare_dataset import (
    parse_frame_metadata_from_filename,
    split_dataset_by_video_id,
    write_split_csv,
)
from src.dataset_loader import DatasetLoader, VideoMetadata
from src.face_extraction import FaceExtractor
from src.preprocessing import ImagePreprocessor


def create_synthetic_face_image(width: int = 200, height: int = 200) -> np.ndarray:
    """Create a synthetic image with a face-like pattern that Haar cascades can detect."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 180

    # Draw face oval
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (60, 80), 0, 0, 360, (140, 140, 140), -1)

    # Eyes
    cv2.circle(img, (center[0] - 25, center[1] - 20), 10, (40, 40, 40), -1)
    cv2.circle(img, (center[0] + 25, center[1] - 20), 10, (40, 40, 40), -1)

    # Nose
    cv2.line(img, (center[0], center[1] - 5), (center[0], center[1] + 15), (60, 60, 60), 3)

    # Mouth
    cv2.ellipse(img, (center[0], center[1] + 35), (25, 10), 0, 0, 180, (40, 40, 40), 3)
    return img


def create_synthetic_video(video_path: Path, num_frames: int = 15, fps: int = 10) -> None:
    """Generate a valid dummy video file using OpenCV VideoWriter."""
    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.full((h, w, 3), fill_value=i * 10 % 255, dtype=np.uint8)
        # Put synthetic pattern
        cv2.circle(frame, (160, 120), 40, (255, 255, 255), -1)
        out.write(frame)

    out.release()


class TestDatasetLoader(unittest.TestCase):
    """Test video validation and directory scanning."""

    def test_invalid_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Non-existent file
            is_valid, _, err = DatasetLoader.validate_video_file(tmp_path / "ghost.mp4")
            self.assertFalse(is_valid)
            self.assertIn("does not exist", err)

            # 2. Unsupported extension
            text_file = tmp_path / "notes.txt"
            text_file.write_text("not a video")
            is_valid, _, err = DatasetLoader.validate_video_file(text_file)
            self.assertFalse(is_valid)
            self.assertIn("Unsupported format", err)

            # 3. Empty 0-byte video
            empty_video = tmp_path / "empty.mp4"
            empty_video.write_bytes(b"")
            is_valid, _, err = DatasetLoader.validate_video_file(empty_video)
            self.assertFalse(is_valid)
            self.assertIn("empty", err)

    def test_valid_video_metadata_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "sample_video.mp4"
            create_synthetic_video(video_file, num_frames=20, fps=10)

            is_valid, props, err = DatasetLoader.validate_video_file(video_file)
            self.assertTrue(is_valid, f"Validation failed: {err}")
            self.assertEqual(props["frame_count"], 20)
            self.assertEqual(props["width"], 320)
            self.assertEqual(props["height"], 240)


class TestFrameExtraction(unittest.TestCase):
    """Test frame sampling and extraction."""

    def test_extract_frames_from_video(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test_vid.mp4"
            create_synthetic_video(video_file, num_frames=30, fps=10)

            out_dir = tmp_path / "frames"
            extracted = extract_frames_from_video(
                video_path=video_file,
                output_dir=out_dir,
                dataset_name="faceforensics",
                label=0,
                video_id="001",
                sample_rate=5,
                max_frames=4,
            )

            # Should extract 4 frames capped by max_frames
            self.assertEqual(len(extracted), 4)
            for f in extracted:
                self.assertTrue(f.exists())
                self.assertIn("faceforensics_real_001_frame", f.name)


class TestFaceExtraction(unittest.TestCase):
    """Test OpenCV face detection and graceful failure handling."""

    def setUp(self):
        self.extractor = FaceExtractor()

    def test_no_face_detection(self):
        # Blank black image has no face
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        result = self.extractor.extract_face(blank)
        self.assertFalse(result.success)
        self.assertIn("No face detected", result.error_reason)

    def test_crop_face_with_padding(self):
        # Test bbox padding calculation
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        cropped = self.extractor.crop_face_with_padding(frame, bbox)
        # Bbox 100x100 with 20% pad -> 140x140
        self.assertEqual(cropped.shape[0], 140)
        self.assertEqual(cropped.shape[1], 140)


class TestPreprocessing(unittest.TestCase):
    """Test image resizing, normalization, and tensor conversion."""

    def setUp(self):
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))

    def test_resize_and_normalize(self):
        raw_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess(raw_img)

        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (256, 256, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0.0 <= processed.min() <= 1.0)
        self.assertTrue(0.0 <= processed.max() <= 1.0)

    def test_to_tensor_conversion(self):
        raw_img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        tensor = self.preprocessor.preprocess_to_tensor(raw_img)

        self.assertIsNotNone(tensor)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)


class TestDataLeakagePrevention(unittest.TestCase):
    """Verify that train/val/test splits strictly partition by video ID."""

    def test_split_by_video_id_zero_overlap(self):
        # Generate mock dataset records representing 20 FaceForensics videos
        # with 5 frames per video
        records = []
        for v in range(20):
            video_id = f"vid_{v:03d}"
            label = v % 2  # Alternate real and fake
            for f in range(5):
                records.append({
                    "filepath": f"processed/faces/{video_id}_f{f}.jpg",
                    "label": label,
                    "class_name": "REAL" if label == 0 else "DEEPFAKE",
                    "dataset": "faceforensics",
                    "video_id": video_id,
                    "frame_idx": f,
                })

        # Add 5 Celeb-DF videos
        for v in range(5):
            video_id = f"celeb_vid_{v:03d}"
            label = v % 2
            for f in range(5):
                records.append({
                    "filepath": f"processed/faces/{video_id}_f{f}.jpg",
                    "label": label,
                    "class_name": "REAL" if label == 0 else "DEEPFAKE",
                    "dataset": "celeb_df",
                    "video_id": video_id,
                    "frame_idx": f,
                })

        train_r, val_r, test_r, celeb_r = split_dataset_by_video_id(records, random_state=42)

        # 1. Total records check
        self.assertEqual(len(train_r) + len(val_r) + len(test_r), 20 * 5)
        self.assertEqual(len(celeb_r), 5 * 5)

        # 2. Extract video IDs per split
        train_vids = set(r["video_id"] for r in train_r)
        val_vids = set(r["video_id"] for r in val_r)
        test_vids = set(r["video_id"] for r in test_r)
        celeb_vids = set(r["video_id"] for r in celeb_r)

        # 3. CRITICAL: ZERO LEAKAGE ASSERTIONS
        self.assertEqual(len(train_vids & val_vids), 0, "Leakage: Train and Val share video IDs!")
        self.assertEqual(len(train_vids & test_vids), 0, "Leakage: Train and Test share video IDs!")
        self.assertEqual(len(val_vids & test_vids), 0, "Leakage: Val and Test share video IDs!")
        self.assertEqual(len(train_vids & celeb_vids), 0, "Leakage: Celeb-DF not isolated!")

    def test_csv_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_split.csv"
            records = [{
                "filepath": "path/to/face.jpg",
                "label": 0,
                "class_name": "REAL",
                "dataset": "faceforensics",
                "video_id": "vid01",
                "frame_idx": 5,
            }]
            write_split_csv(records, csv_path)
            self.assertTrue(csv_path.exists())

            # Read back and verify header
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("filepath,label,class_name,dataset,video_id,frame_idx", content)
            self.assertIn("vid01", content)


if __name__ == "__main__":
    unittest.main()
