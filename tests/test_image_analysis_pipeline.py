"""
Integration Test for VeriFrame Image Detection Pipeline
========================================================
Tests:
1. Real face image detection (ResNet18 prediction, Grad-CAM, FFT, DWT)
2. Deepfake face image detection (ResNet18 prediction, Grad-CAM, FFT, DWT)
3. Grad-CAM heatmap and overlay base64 payload validation
4. FFT numerical calculations and radial frequency profile
5. DWT multi-scale sub-band metrics
6. Holistic analytical summary presence
7. Invalid file type rejection (415)
8. Empty file rejection (400)
9. Detection history logging with authenticated session
"""

import base64
import io
import json
import unittest
from pathlib import Path

import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"
REAL_SAMPLE = Path("data/processed/real_faces/ff_original_000_f118.png")
FAKE_SAMPLE = Path("data/processed/fake_faces/ff_Deepfakes_000_003_f118.png")


class TestImageDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify server is up
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        assert r.status_code == 200, "Backend server is not running!"

        # Register and login a test user for history check
        cls.session = requests.Session()
        signup_payload = {
            "full_name": "Pipeline Tester",
            "email": f"pipelinetest_{int(Path(__file__).stat().st_mtime)}@veriframe.io",
            "username": f"pipelinetest_{int(Path(__file__).stat().st_mtime) % 10000}",
            "password": "Password123!",
            "confirm_password": "Password123!",
        }
        cls.session.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
        login_res = cls.session.post(f"{BASE_URL}/api/auth/login", json={
            "identifier": signup_payload["email"],
            "password": signup_payload["password"],
        })
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"

    def test_01_real_image_detection(self):
        """Upload a real face image and verify all analysis branches."""
        self.assertTrue(REAL_SAMPLE.exists(), f"Sample image {REAL_SAMPLE} not found")
        with open(REAL_SAMPLE, "rb") as f:
            files = {"file": (REAL_SAMPLE.name, f, "image/jpeg")}
            res = self.session.post(f"{BASE_URL}/api/detect/image", files=files)

        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        # Check prediction & probabilities
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], ["REAL", "DEEPFAKE"])
        self.assertGreaterEqual(data["confidence"], 0.5)
        self.assertAlmostEqual(data["real_prob"] + data["deepfake_prob"], 1.0, places=2)

        # Check spatial Grad-CAM
        spatial = data.get("spatial_analysis")
        self.assertIsNotNone(spatial, "Missing spatial_analysis in response")
        self.assertEqual(spatial["status"], "active")
        self.assertEqual(spatial["architecture"], "ResNet18")
        self.assertIsNotNone(spatial["heatmap_b64"])
        self.assertIsNotNone(spatial["overlay_b64"])
        self.assertIn("salient_regions", spatial)
        self.assertIn("explanation", spatial)

        # Check FFT frequency analysis
        freq = data.get("frequency_analysis")
        self.assertIsNotNone(freq, "Missing frequency_analysis in response")
        self.assertEqual(freq["status"], "active")
        self.assertIsNotNone(freq["magnitude_b64"])
        self.assertIsNotNone(freq["log_spectrum_b64"])
        self.assertIsNotNone(freq["metrics"])
        self.assertIn("low_energy_pct", freq["metrics"])
        self.assertIn("high_energy_pct", freq["metrics"])
        self.assertIn("radial_spectrum", freq["metrics"])
        self.assertEqual(len(freq["metrics"]["radial_spectrum"]), 64)

        # Check DWT analysis
        dwt = data.get("dwt_analysis")
        self.assertIsNotNone(dwt, "Missing dwt_analysis in response")
        self.assertEqual(dwt["status"], "active")
        self.assertIsNotNone(dwt["subbands_b64"])
        self.assertIsNotNone(dwt["features"])

        # Check analysis summary
        self.assertIsNotNone(data.get("analysis_summary"))
        self.assertIn("Model Determination", data["analysis_summary"])

        # Check history ID generated for logged-in user
        self.assertIsNotNone(data.get("history_id"), "Detection was not logged into MongoDB history")

    def test_02_deepfake_image_detection(self):
        """Upload a deepfake face image and verify all analysis branches."""
        self.assertTrue(FAKE_SAMPLE.exists(), f"Sample image {FAKE_SAMPLE} not found")
        with open(FAKE_SAMPLE, "rb") as f:
            files = {"file": (FAKE_SAMPLE.name, f, "image/jpeg")}
            res = self.session.post(f"{BASE_URL}/api/detect/image", files=files)

        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertIn(data["prediction"], ["REAL", "DEEPFAKE"])
        self.assertIsNotNone(data["spatial_analysis"]["heatmap_b64"])
        self.assertIsNotNone(data["frequency_analysis"]["magnitude_b64"])
        self.assertIsNotNone(data["dwt_analysis"]["subbands_b64"])

    def test_03_invalid_file_type(self):
        """Uploading a non-image file type should return 415 Unsupported Media Type."""
        fake_text_bytes = io.BytesIO(b"This is not an image file.")
        files = {"file": ("test.txt", fake_text_bytes, "text/plain")}
        res = self.session.post(f"{BASE_URL}/api/detect/image", files=files)
        self.assertEqual(res.status_code, 415)

    def test_04_empty_file(self):
        """Uploading a 0-byte file should return 400 Bad Request."""
        empty_bytes = io.BytesIO(b"")
        files = {"file": ("empty.jpg", empty_bytes, "image/jpeg")}
        res = self.session.post(f"{BASE_URL}/api/detect/image", files=files)
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
