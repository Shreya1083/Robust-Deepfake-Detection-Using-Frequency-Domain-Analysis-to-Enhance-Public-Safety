"""
Phase 4 Tests — FFT Frequency Analysis
=======================================
Tests real FFT computation, feature extraction, API endpoint behaviour,
and regression checks on the existing CNN endpoint.

Run:
    .venv\Scripts\python.exe -m unittest tests/test_phase4.py -v
"""

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

from src.frequency import FFTAnalyzer, FFTFeatures, FFTResult, run_fft_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil(w=128, h=128, mode="RGB") -> Image.Image:
    """Create a random-pixel PIL image."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_face_pil(w=224, h=224) -> Image.Image:
    """Synthetic face-like image: skin-tone background + darker oval."""
    arr = np.full((h, w, 3), [210, 170, 140], dtype=np.uint8)
    # Draw face oval
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            if ((x - cx) / (w * 0.4)) ** 2 + ((y - cy) / (h * 0.45)) ** 2 < 1:
                arr[y, x] = [180, 140, 110]
    return Image.fromarray(arr, "RGB")


def _make_bgr_numpy(w=200, h=200) -> np.ndarray:
    """Return a random BGR uint8 numpy array."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestFFTAnalyzerCore(unittest.TestCase):
    """Test FFT computation correctness and output structure."""

    def setUp(self):
        self.analyzer = FFTAnalyzer(input_size=(224, 224))

    def test_pil_input_returns_success(self):
        img = _make_pil()
        result = self.analyzer.analyze(img, generate_visuals=False)
        self.assertTrue(result.success, msg=result.error)
        self.assertIsNotNone(result.features)

    def test_numpy_bgr_input_returns_success(self):
        arr = _make_bgr_numpy()
        result = self.analyzer.analyze(arr, generate_visuals=False)
        self.assertTrue(result.success, msg=result.error)

    def test_feature_vector_dimensionality(self):
        """Feature vector must be 70-dimensional (6 scalars + 64 radial)."""
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        self.assertTrue(result.success)
        vec = result.features.to_vector()
        self.assertEqual(vec.shape, (70,), f"Expected 70-dim, got {vec.shape}")

    def test_radial_spectrum_length(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        self.assertTrue(result.success)
        self.assertEqual(len(result.features.radial_spectrum), 64)

    def test_energy_bands_sum_to_approx_one(self):
        """Low + mid + high energy fractions should cover most of the spectrum."""
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        self.assertTrue(result.success)
        f = result.features
        total = f.low_energy + f.mid_energy + f.high_energy
        self.assertAlmostEqual(total, 1.0, delta=0.05,
                               msg=f"Energy bands sum={total:.4f}, expected ~1.0")

    def test_energy_values_are_non_negative(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        f = result.features
        self.assertGreaterEqual(f.low_energy,  0.0)
        self.assertGreaterEqual(f.mid_energy,  0.0)
        self.assertGreaterEqual(f.high_energy, 0.0)

    def test_spectral_entropy_in_range(self):
        """Normalised Shannon entropy must be in [0, 1]."""
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        f = result.features
        self.assertGreaterEqual(f.spectral_entropy, 0.0)
        self.assertLessEqual(f.spectral_entropy, 1.0 + 1e-6)

    def test_radial_spectrum_normalised(self):
        """Radial spectrum should be normalised to [0, 1]."""
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        rs = result.features.radial_spectrum
        self.assertGreaterEqual(min(rs), 0.0 - 1e-6)
        self.assertLessEqual(max(rs), 1.0 + 1e-6)

    def test_values_are_real_not_hardcoded(self):
        """Two different images should produce different FFT features."""
        r1 = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        r2 = self.analyzer.analyze(_make_pil(), generate_visuals=False)  # different random seed
        self.assertTrue(r1.success and r2.success)
        # At minimum, spectral entropy should differ
        # (identical for edge-case pure-noise with same seed — use face image)
        r3 = self.analyzer.analyze(_make_face_pil(), generate_visuals=False)
        self.assertNotAlmostEqual(r1.features.mean_magnitude,
                                  r3.features.mean_magnitude, places=3)

    def test_feature_dict_keys(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        d = result.features.to_dict()
        for key in ("mean_magnitude", "std_magnitude", "low_energy",
                    "mid_energy", "high_energy", "spectral_entropy",
                    "radial_spectrum", "feature_dim"):
            self.assertIn(key, d, msg=f"Missing key: {key}")


class TestFFTVisualizations(unittest.TestCase):
    """Test that visualization outputs are valid base64 JPEGs."""

    def setUp(self):
        self.analyzer = FFTAnalyzer()

    def _is_valid_b64_jpeg(self, b64_str: str) -> bool:
        """Decode base64 and verify it's a loadable JPEG."""
        import base64
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0

    def test_visuals_generated_when_requested(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=True)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.original_b64)
        self.assertIsNotNone(result.magnitude_b64)
        self.assertIsNotNone(result.log_spectrum_b64)

    def test_original_b64_is_valid_jpeg(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=True)
        self.assertTrue(self._is_valid_b64_jpeg(result.original_b64))

    def test_magnitude_b64_is_valid_jpeg(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=True)
        self.assertTrue(self._is_valid_b64_jpeg(result.magnitude_b64))

    def test_log_spectrum_b64_is_valid_jpeg(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=True)
        self.assertTrue(self._is_valid_b64_jpeg(result.log_spectrum_b64))

    def test_no_visuals_when_flag_false(self):
        result = self.analyzer.analyze(_make_pil(), generate_visuals=False)
        self.assertTrue(result.success)
        self.assertIsNone(result.original_b64)
        self.assertIsNone(result.magnitude_b64)
        self.assertIsNone(result.log_spectrum_b64)


class TestFFTEdgeCases(unittest.TestCase):
    """Invalid / degenerate inputs."""

    def setUp(self):
        self.analyzer = FFTAnalyzer()

    def test_too_small_image_returns_error(self):
        tiny = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        result = self.analyzer.analyze(tiny, generate_visuals=False)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_invalid_bytes_returns_error(self):
        result = self.analyzer.analyze(b"not_an_image_at_all", generate_visuals=False)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_grayscale_pil_input(self):
        """Grayscale PIL image should work fine."""
        gray_pil = Image.fromarray(np.random.randint(0, 256, (128, 128), dtype=np.uint8), "L")
        result = self.analyzer.analyze(gray_pil.convert("RGB"), generate_visuals=False)
        self.assertTrue(result.success)

    def test_uniform_image(self):
        """Uniform (blank) image — edge case for FFT (all energy at DC)."""
        blank = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        result = self.analyzer.analyze(blank, generate_visuals=False)
        self.assertTrue(result.success)
        # For a uniform image, all energy should be in the DC (low) band
        f = result.features
        self.assertGreater(f.low_energy, 0.5,
                           msg="Uniform image should have high low-band energy")


class TestRunFFTConvenienceFunction(unittest.TestCase):
    """Test the module-level run_fft_analysis() helper."""

    def test_pil_image(self):
        result = run_fft_analysis(_make_pil())
        self.assertTrue(result.success)
        self.assertIsNotNone(result.features)

    def test_numpy_bgr(self):
        result = run_fft_analysis(_make_bgr_numpy())
        self.assertTrue(result.success)

    def test_feature_vector_is_float32(self):
        result = run_fft_analysis(_make_pil(), generate_visuals=False)
        vec = result.features.to_vector()
        self.assertEqual(vec.dtype, np.float32)


class TestFFTVsCNNRegression(unittest.TestCase):
    """
    Verify the existing CNN endpoint (SpatialDetector) is unaffected by Phase 4.
    We only test import and instantiation (no checkpoint required).
    """

    def test_spatial_detector_still_importable(self):
        from src.detector import SpatialDetector
        detector = SpatialDetector()
        self.assertFalse(detector.is_ready)   # no checkpoint — expected
        self.assertIsNotNone(detector.device)

    def test_spatial_model_still_importable(self):
        from models.spatial_model import SpatialResNet18
        model = SpatialResNet18(pretrained=False)
        import torch
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 2))

    def test_fft_and_cnn_imports_coexist(self):
        """Both modules must import without conflict."""
        from src.frequency import run_fft_analysis
        from src.detector import SpatialDetector
        self.assertTrue(callable(run_fft_analysis))


if __name__ == "__main__":
    unittest.main(verbosity=2)
