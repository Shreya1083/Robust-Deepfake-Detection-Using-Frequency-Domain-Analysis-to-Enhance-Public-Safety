"""
Phase 5 Tests — DWT (Discrete Wavelet Transform) Frequency Analysis
===================================================================
Tests real 2D Discrete Wavelet Transform computation using PyWavelets,
feature extraction, sub-band decomposition, backend API endpoint behavior,
and regression checks for FFT analysis and CNN spatial detection.

Run:
    python -m unittest tests/test_phase5.py -v
"""

import asyncio
import base64
import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pywt
from src.wavelet import (
    DWTAnalyzer,
    DWTFeatures,
    DWTResult,
    SubbandFeatures,
    run_dwt_analysis,
)
from src.frequency import run_fft_analysis
from backend.routes.frequency_routes import dwt_analysis, fft_analysis


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_pil_random(w=224, h=224) -> Image.Image:
    """Create a random-pixel PIL image."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_real_face_pil(w=224, h=224) -> Image.Image:
    """
    Synthetic real-face simulation:
    Smooth skin-toned backdrop + soft oval facial contour with natural low-frequency roll-off.
    """
    arr = np.full((h, w, 3), [215, 175, 145], dtype=np.uint8)
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            norm_dist = ((x - cx) / (w * 0.42)) ** 2 + ((y - cy) / (h * 0.48)) ** 2
            if norm_dist < 1.0:
                # Smooth, soft skin shading
                arr[y, x] = [
                    int(190 - 30 * norm_dist),
                    int(150 - 25 * norm_dist),
                    int(120 - 20 * norm_dist),
                ]
    # Smooth bilateral-like blur to emulate natural photography
    blurred = cv2.GaussianBlur(arr, (7, 7), 2.0)
    return Image.fromarray(blurred, "RGB")


def _make_deepfake_face_pil(w=224, h=224) -> Image.Image:
    """
    Synthetic deepfake face simulation:
    Face structure + added high-frequency synthesis noise artifacts.
    """
    face = np.array(_make_real_face_pil(w, h))
    # Inject high-frequency noise typical of GAN / diffusion artifact residual
    noise = np.random.RandomState(42).randint(-30, 30, face.shape, dtype=np.int16)
    noisy = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, "RGB")


def _is_valid_b64_jpeg(b64_str: str) -> bool:
    """Decode base64 string and verify it produces a valid cv2 image."""
    if not b64_str:
        return False
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestDWTAnalyzerCore(unittest.TestCase):
    """Test core 2D DWT computation, sub-band shapes, and feature extraction."""

    def setUp(self):
        self.analyzer = DWTAnalyzer(input_size=(224, 224), wavelet="haar")

    def test_dwt_on_valid_real_face_image(self):
        """DWT on a valid real face image succeeds and extracts valid features."""
        real_face = _make_real_face_pil()
        res = self.analyzer.analyze(real_face, generate_visuals=False)
        self.assertTrue(res.success, msg=res.error)
        self.assertIsNotNone(res.features)
        self.assertEqual(res.wavelet_type, "haar")
        self.assertEqual(res.image_size, (224, 224))
        self.assertEqual(res.subband_shape, (112, 112))

    def test_dwt_on_valid_deepfake_face_image(self):
        """DWT on a deepfake face with synthetic artifacts succeeds and captures high-frequency energy."""
        fake_face = _make_deepfake_face_pil()
        res = self.analyzer.analyze(fake_face, generate_visuals=False)
        self.assertTrue(res.success, msg=res.error)
        self.assertIsNotNone(res.features)

        # Deepfake with grid artifacts must exhibit elevated high-frequency detail energy
        real_res = self.analyzer.analyze(_make_real_face_pil(), generate_visuals=False)
        self.assertGreater(
            res.features.high_freq_energy_ratio,
            real_res.features.high_freq_energy_ratio,
            "Deepfake with checkerboard noise should have higher relative detail energy than smooth real face.",
        )

    def test_all_four_subbands_generated(self):
        """Verify LL, LH, HL, and HH sub-bands are all computed."""
        res = self.analyzer.analyze(_make_pil_random(), generate_visuals=False)
        self.assertTrue(res.success)
        f = res.features
        for band_name, band_obj in [("LL", f.ll), ("LH", f.lh), ("HL", f.hl), ("HH", f.hh)]:
            self.assertIsInstance(band_obj, SubbandFeatures, f"Band {band_name} must be SubbandFeatures")
            self.assertGreaterEqual(band_obj.energy, 0.0)
            self.assertGreaterEqual(band_obj.max_abs, 0.0)
            self.assertGreaterEqual(band_obj.mean_abs, 0.0)

    def test_numerical_features_and_energy_calculated(self):
        """Verify numerical features: mean, std, energy, mean_abs, max_abs, energy ratios."""
        res = self.analyzer.analyze(_make_real_face_pil(), generate_visuals=False)
        self.assertTrue(res.success)
        f = res.features

        # Check total energy equals sum of subband energies
        subband_sum = f.ll.energy + f.lh.energy + f.hl.energy + f.hh.energy
        self.assertAlmostEqual(f.total_energy, subband_sum, delta=1e-3)

        # Check energy distribution sums to ~1.0
        ratio_sum = f.ll.energy_ratio + f.lh.energy_ratio + f.hl.energy_ratio + f.hh.energy_ratio
        self.assertAlmostEqual(ratio_sum, 1.0, delta=1e-5)

        # Check LL dominates in photographic image (typically >85%)
        self.assertGreater(f.ll.energy_ratio, 0.80)

        # Shannon entropy is non-negative and <= 1.0
        self.assertGreaterEqual(f.wavelet_entropy, 0.0)
        self.assertLessEqual(f.wavelet_entropy, 1.0 + 1e-6)

    def test_feature_vector_dimensionality(self):
        """Feature vector must be exactly 28-dimensional float32."""
        res = self.analyzer.analyze(_make_pil_random(), generate_visuals=False)
        self.assertTrue(res.success)
        vec = res.features.to_vector()
        self.assertEqual(vec.shape, (28,), f"Expected 28-dim, got {vec.shape}")
        self.assertEqual(vec.dtype, np.float32)

    def test_feature_dict_structure(self):
        """Verify to_dict() returns correct keys matching backend contract."""
        res = self.analyzer.analyze(_make_real_face_pil(), generate_visuals=False)
        d = res.features.to_dict()
        self.assertIn("subbands", d)
        for band in ("LL", "LH", "HL", "HH"):
            self.assertIn(band, d["subbands"])
            for metric in ("mean", "std", "energy", "mean_abs", "max_abs", "energy_ratio"):
                self.assertIn(metric, d["subbands"][band])
        self.assertIn("energy_distribution", d)
        self.assertIn("total_energy", d)
        self.assertIn("detail_to_approx_ratio", d)
        self.assertIn("high_freq_energy_ratio", d)
        self.assertIn("wavelet_entropy", d)
        self.assertEqual(d["feature_dim"], 28)

    def test_values_are_deterministic_and_not_hardcoded(self):
        """Two different images produce distinct DWT features, identical inputs produce identical features."""
        img1 = _make_real_face_pil()
        img2 = _make_deepfake_face_pil()

        r1 = self.analyzer.analyze(img1, generate_visuals=False)
        r2 = self.analyzer.analyze(img2, generate_visuals=False)
        r1_repeat = self.analyzer.analyze(img1, generate_visuals=False)

        # Distinct inputs -> distinct features
        self.assertNotEqual(r1.features.hh.energy, r2.features.hh.energy)
        # Identical inputs -> identical features
        np.testing.assert_array_equal(r1.features.to_vector(), r1_repeat.features.to_vector())


class TestDWTVisualizations(unittest.TestCase):
    """Test generation of base64 JPEG visualizations for all 4 sub-bands and 2x2 grid."""

    def setUp(self):
        self.analyzer = DWTAnalyzer()

    def test_visualizations_generated_when_requested(self):
        img = _make_real_face_pil()
        res = self.analyzer.analyze(img, generate_visuals=True)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.original_b64)
        self.assertIsNotNone(res.ll_b64)
        self.assertIsNotNone(res.lh_b64)
        self.assertIsNotNone(res.hl_b64)
        self.assertIsNotNone(res.hh_b64)
        self.assertIsNotNone(res.combined_b64)

    def test_all_visuals_are_valid_decodable_jpegs(self):
        img = _make_real_face_pil()
        res = self.analyzer.analyze(img, generate_visuals=True)
        self.assertTrue(_is_valid_b64_jpeg(res.original_b64), "original_b64 invalid")
        self.assertTrue(_is_valid_b64_jpeg(res.ll_b64), "ll_b64 invalid")
        self.assertTrue(_is_valid_b64_jpeg(res.lh_b64), "lh_b64 invalid")
        self.assertTrue(_is_valid_b64_jpeg(res.hl_b64), "hl_b64 invalid")
        self.assertTrue(_is_valid_b64_jpeg(res.hh_b64), "hh_b64 invalid")
        self.assertTrue(_is_valid_b64_jpeg(res.combined_b64), "combined_b64 invalid")

    def test_no_visuals_when_flag_false(self):
        res = self.analyzer.analyze(_make_pil_random(), generate_visuals=False)
        self.assertTrue(res.success)
        self.assertIsNone(res.original_b64)
        self.assertIsNone(res.ll_b64)
        self.assertIsNone(res.lh_b64)
        self.assertIsNone(res.hl_b64)
        self.assertIsNone(res.hh_b64)
        self.assertIsNone(res.combined_b64)


class TestDWTEdgeCases(unittest.TestCase):
    """Test degenerate, empty, corrupted, and unusual inputs."""

    def setUp(self):
        self.analyzer = DWTAnalyzer()

    def test_too_small_image_returns_error(self):
        tiny = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        res = self.analyzer.analyze(tiny, generate_visuals=False)
        self.assertFalse(res.success)
        self.assertIn("too small", res.error.lower())

    def test_invalid_bytes_returns_error(self):
        res = self.analyzer.analyze(b"corrupted_invalid_data", generate_visuals=False)
        self.assertFalse(res.success)
        self.assertIsNotNone(res.error)

    def test_uniform_blank_image(self):
        """A uniform flat gray image: all detail sub-bands should have ~0 energy."""
        blank = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        res = self.analyzer.analyze(blank, generate_visuals=False)
        self.assertTrue(res.success)
        f = res.features
        self.assertAlmostEqual(f.ll.energy_ratio, 1.0, delta=1e-4)
        self.assertAlmostEqual(f.high_freq_energy_ratio, 0.0, delta=1e-4)

    def test_grayscale_numpy_input(self):
        """Grayscale 2D numpy array is handled seamlessly."""
        gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        res = self.analyzer.analyze(gray, generate_visuals=False)
        self.assertTrue(res.success)

    def test_bgr_numpy_input(self):
        """BGR 3D numpy array is converted properly."""
        bgr = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)
        res = self.analyzer.analyze(bgr, generate_visuals=False)
        self.assertTrue(res.success)


class MockUploadFile:
    def __init__(self, content: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        return self.content


class TestDWTBackendRoute(unittest.TestCase):
    """Test backend FastAPI endpoint handler POST /api/frequency/dwt."""

    def _make_upload_file(self, content: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
        return MockUploadFile(content=content, filename=filename, content_type=content_type)

    def test_valid_image_upload(self):
        img = np.full((224, 224, 3), 150, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        upload = self._make_upload_file(buf.tobytes())

        res = asyncio.run(dwt_analysis(upload))
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["features"])
        self.assertEqual(res["wavelet_type"], "haar")
        self.assertIsNotNone(res["ll_b64"])
        self.assertIsNotNone(res["lh_b64"])
        self.assertIsNotNone(res["hl_b64"])
        self.assertIsNotNone(res["hh_b64"])
        self.assertIsNotNone(res["combined_b64"])

    def test_unsupported_file_type_rejected(self):
        upload = self._make_upload_file(b"text content", filename="doc.txt", content_type="text/plain")
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(dwt_analysis(upload))
        self.assertEqual(ctx.exception.status_code, 415)

    def test_empty_file_rejected(self):
        upload = self._make_upload_file(b"", filename="empty.jpg", content_type="image/jpeg")
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(dwt_analysis(upload))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_corrupted_image_rejected(self):
        upload = self._make_upload_file(b"corrupted_non_jpeg_bytes_123456", filename="corrupt.jpg", content_type="image/jpeg")
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(dwt_analysis(upload))
        self.assertEqual(ctx.exception.status_code, 422)


class TestFFTAndCNNRegressions(unittest.TestCase):
    """Ensure FFT frequency analysis and CNN spatial model remain 100% operational."""

    def test_fft_analysis_still_functional(self):
        """FFT analysis runs and returns valid 70-dim feature vector."""
        img = _make_real_face_pil()
        res = run_fft_analysis(img, generate_visuals=True)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.features)
        self.assertEqual(len(res.features.to_vector()), 70)
        self.assertIsNotNone(res.magnitude_b64)
        self.assertIsNotNone(res.log_spectrum_b64)

    def test_spatial_detector_regression(self):
        """CNN SpatialDetector initializes without breaking."""
        from src.detector import SpatialDetector
        detector = SpatialDetector()
        self.assertIn(detector.is_ready, [True, False])
        self.assertIsNotNone(detector.device)
        self.assertIsNotNone(detector.face_extractor)

    def test_spatial_resnet_forward_pass(self):
        """SpatialResNet18 architecture produces expected tensor dimensions."""
        from models.spatial_model import SpatialResNet18
        import torch
        model = SpatialResNet18(pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
            emb = model.extract_features(dummy)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(emb.shape, (2, 512))

    def test_fft_and_dwt_coexist_on_same_image(self):
        """Both FFT and DWT can analyze the exact same image instance independently."""
        face = _make_real_face_pil()
        dwt_res = run_dwt_analysis(face, generate_visuals=True)
        fft_res = run_fft_analysis(face, generate_visuals=True)

        self.assertTrue(dwt_res.success)
        self.assertTrue(fft_res.success)

        dwt_vec = dwt_res.features.to_vector()
        fft_vec = fft_res.features.to_vector()

        self.assertEqual(dwt_vec.shape, (28,))
        self.assertEqual(fft_vec.shape, (70,))
        # Future feature fusion in Phase 6: concatenated 28 + 70 + 512 = 610 dim vector
        fused = np.concatenate([dwt_vec, fft_vec])
        self.assertEqual(fused.shape, (98,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
