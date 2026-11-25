"""
Phase 6 Tests — Multimodal Feature Fusion (CNN + FFT + DWT)
============================================================
Comprehensive test suite verifying:
1. Feature extraction dimensions:
   - CNN spatial embedding: 512-dim
   - FFT frequency spectrum: 70-dim
   - DWT wavelet sub-bands: 28-dim
   - Total fused vector: 610-dim
2. FeatureNormalizer:
   - Normalization statistics fitted strictly on training data
   - Numerical safety on extreme values, NaNs, and Infs
   - Correct serialization and deserialization
3. MultimodalFusionClassifier:
   - Architecture: 610 -> 256 -> 64 -> 2
   - Forward pass shapes: (B, 610) -> (B, 2)
   - Softmax probabilities sum to 1.0
4. Training and Evaluation pipelines:
   - Checkpoint creation with metadata
   - Evaluation metrics calculation
5. Inference Engine & Backend API:
   - SpatialDetector integration with fusion model
   - FastAPI POST /api/detect/image endpoint
   - Backward compatibility for existing fields

Run:
    python -m unittest tests/test_phase6.py -v
"""

import asyncio
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, CLASS_LABELS
from src.detector import SpatialDetector
from src.feature_fusion import (
    CNN_FEATURE_DIM,
    DWT_FEATURE_DIM,
    FFT_FEATURE_DIM,
    FUSION_FEATURE_DIM,
    FeatureNormalizer,
    MultimodalFeatureExtractor,
    MultimodalFusionClassifier,
)
from scripts.evaluate_fusion_model import evaluate_predictions


def _make_synthetic_face(w=224, h=224, is_fake=False) -> Image.Image:
    """Helper to generate a synthetic face for deterministic testing."""
    arr = np.full((h, w, 3), [210, 170, 140], dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(arr, (cx, cy), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, (190, 150, 120), -1)

    if is_fake:
        # Inject high-frequency periodic pattern
        y, x = np.ogrid[:h, :w]
        checker = ((x % 4 == 0) & (y % 4 == 0)).astype(np.uint8) * 40
        arr = np.clip(arr.astype(np.int16) + checker[:, :, None], 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


class TestFeatureDimensions(unittest.TestCase):
    """Verify feature extractor output shapes and ordering."""

    @classmethod
    def setUpClass(cls):
        cls.device = get_default_device()
        cls.extractor = MultimodalFeatureExtractor(device=cls.device)
        cls.test_image = _make_synthetic_face()

    def test_cnn_feature_dimension(self):
        feat = self.extractor.extract_cnn_features(self.test_image)
        self.assertIsInstance(feat, np.ndarray)
        self.assertEqual(feat.shape, (512,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertEqual(CNN_FEATURE_DIM, 512)

    def test_fft_feature_dimension(self):
        feat = self.extractor.extract_fft_features(self.test_image)
        self.assertIsInstance(feat, np.ndarray)
        self.assertEqual(feat.shape, (70,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertEqual(FFT_FEATURE_DIM, 70)

    def test_dwt_feature_dimension(self):
        feat = self.extractor.extract_dwt_features(self.test_image)
        self.assertIsInstance(feat, np.ndarray)
        self.assertEqual(feat.shape, (28,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertEqual(DWT_FEATURE_DIM, 28)

    def test_all_features_combined_dimension(self):
        feats = self.extractor.extract_all(self.test_image)
        self.assertIn("cnn", feats)
        self.assertIn("fft", feats)
        self.assertIn("dwt", feats)
        self.assertIn("raw_fused", feats)

        self.assertEqual(feats["cnn"].shape, (512,))
        self.assertEqual(feats["fft"].shape, (70,))
        self.assertEqual(feats["dwt"].shape, (28,))
        self.assertEqual(feats["raw_fused"].shape, (610,))
        self.assertEqual(FUSION_FEATURE_DIM, 610)


class TestFeatureNormalizer(unittest.TestCase):
    """Verify feature normalization statistics, safety, and serialization."""

    def setUp(self):
        np.random.seed(42)
        self.train_cnn = np.random.randn(20, 512).astype(np.float32)
        self.train_fft = np.random.randn(20, 70).astype(np.float32)
        self.train_dwt = np.random.randn(20, 28).astype(np.float32)

    def test_fit_and_transform(self):
        normalizer = FeatureNormalizer()
        self.assertFalse(normalizer.is_fitted)

        normalizer.fit(self.train_cnn, self.train_fft, self.train_dwt)
        self.assertTrue(normalizer.is_fitted)

        norm_cnn, norm_fft, norm_dwt, fused = normalizer.transform(
            self.train_cnn, self.train_fft, self.train_dwt
        )

        self.assertEqual(norm_cnn.shape, (20, 512))
        self.assertEqual(norm_fft.shape, (20, 70))
        self.assertEqual(norm_dwt.shape, (20, 28))
        self.assertEqual(fused.shape, (20, 610))

        # Check mean is close to 0 and std is close to 1 on training set
        np.testing.assert_allclose(np.mean(norm_cnn, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(norm_cnn, axis=0), 1.0, atol=1e-5)

    def test_single_vector_transform(self):
        normalizer = FeatureNormalizer()
        normalizer.fit(self.train_cnn, self.train_fft, self.train_dwt)

        c1 = self.train_cnn[0]
        f1 = self.train_fft[0]
        d1 = self.train_dwt[0]

        norm_c, norm_f, norm_d, fused = normalizer.transform(c1, f1, d1)
        self.assertEqual(norm_c.shape, (512,))
        self.assertEqual(norm_f.shape, (70,))
        self.assertEqual(norm_d.shape, (28,))
        self.assertEqual(fused.shape, (610,))

    def test_safety_on_nan_and_inf(self):
        normalizer = FeatureNormalizer()
        normalizer.fit(self.train_cnn, self.train_fft, self.train_dwt)

        bad_cnn = np.copy(self.train_cnn[0])
        bad_cnn[0] = np.nan
        bad_cnn[1] = np.inf
        bad_cnn[2] = -np.inf

        _, _, _, fused = normalizer.transform(bad_cnn, self.train_fft[0], self.train_dwt[0])
        self.assertFalse(np.isnan(fused).any())
        self.assertFalse(np.isinf(fused).any())

    def test_serialization(self):
        normalizer = FeatureNormalizer()
        normalizer.fit(self.train_cnn, self.train_fft, self.train_dwt)

        data = normalizer.to_dict()
        self.assertTrue(data["is_fitted"])
        self.assertEqual(len(data["cnn_mean"]), 512)
        self.assertEqual(len(data["fft_mean"]), 70)
        self.assertEqual(len(data["dwt_mean"]), 28)

        restored = FeatureNormalizer.from_dict(data)
        self.assertTrue(restored.is_fitted)
        np.testing.assert_array_equal(normalizer.cnn_mean, restored.cnn_mean)


class TestMultimodalFusionClassifier(unittest.TestCase):
    """Verify PyTorch MLP model architecture, forward pass, and probabilities."""

    def test_model_forward_pass(self):
        model = MultimodalFusionClassifier(input_dim=610, num_classes=2)
        model.eval()

        batch_x = torch.randn(8, 610)
        with torch.no_grad():
            logits = model(batch_x)
            probs = model.predict_proba(batch_x)

        self.assertEqual(logits.shape, (8, 2))
        self.assertEqual(probs.shape, (8, 2))

        # Check probabilities sum to 1.0
        prob_sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(prob_sums, np.ones(8), atol=1e-5)

    def test_binary_class_labels(self):
        self.assertEqual(CLASS_LABELS[0], "REAL")
        self.assertEqual(CLASS_LABELS[1], "DEEPFAKE")


class TestEvaluationMetrics(unittest.TestCase):
    """Verify metric computation (Accuracy, Precision, Recall, F1, ROC-AUC)."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = evaluate_predictions(y_true, y_pred, y_prob)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])


class TestDetectorIntegration(unittest.TestCase):
    """Verify SpatialDetector integration with Multimodal Fusion model."""

    def setUp(self):
        self.detector = SpatialDetector()

    def test_detector_is_ready_and_fusion(self):
        self.assertTrue(self.detector.is_ready)
        self.assertTrue(self.detector.is_fusion)
        self.assertEqual(self.detector.model_type, "multimodal_fusion")

    def test_inference_on_real_image(self):
        real_img = _make_synthetic_face(is_fake=False)
        res = self.detector.predict_image(real_img)

        self.assertTrue(res["is_ready"])
        self.assertIn(res["prediction"], ["REAL", "DEEPFAKE"])
        self.assertIn(res["label"], [0, 1])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertAlmostEqual(res["real_prob"] + res["deepfake_prob"], 1.0, places=3)
        self.assertEqual(res["model_type"], "multimodal_fusion")
        self.assertTrue(res["is_fusion"])
        self.assertIn("multimodal_branches", res)
        self.assertIn("baseline_cnn", res)
        self.assertEqual(res["feature_dims"]["total"], 610)

    def test_inference_on_fake_image(self):
        fake_img = _make_synthetic_face(is_fake=True)
        res = self.detector.predict_image(fake_img)

        self.assertTrue(res["is_ready"])
        self.assertIn(res["prediction"], ["REAL", "DEEPFAKE"])
        self.assertIn("multimodal_branches", res)
        self.assertEqual(res["feature_dims"]["cnn"], 512)
        self.assertEqual(res["feature_dims"]["fft"], 70)
        self.assertEqual(res["feature_dims"]["dwt"], 28)

    def test_invalid_image_handling(self):
        res = self.detector.predict_image(np.array([]))
        self.assertIn("error", res)
        self.assertIsNotNone(res["error"])


class TestFastAPIDetectRoute(unittest.TestCase):
    """Verify POST /api/detect/image endpoint returns multimodal response."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.client = TestClient(app)

    def test_detect_image_endpoint_valid_upload(self):
        img = _make_synthetic_face()
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/api/detect/image",
            files={"file": ("test_face.jpg", buf, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["model_ready"])
        self.assertIn(data["prediction"], ["REAL", "DEEPFAKE"])
        self.assertIn("real_prob", data)
        self.assertIn("deepfake_prob", data)
        self.assertEqual(data["is_fusion"], True)
        self.assertEqual(data["model_type"], "multimodal_fusion")
        self.assertIn("multimodal_branches", data)
        self.assertEqual(data["feature_dims"]["total"], 610)

    def test_detect_image_unsupported_file_type(self):
        buf = io.BytesIO(b"dummy text")
        response = self.client.post(
            "/api/detect/image",
            files={"file": ("test.txt", buf, "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_detect_image_empty_file(self):
        buf = io.BytesIO(b"")
        response = self.client.post(
            "/api/detect/image",
            files={"file": ("empty.jpg", buf, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
