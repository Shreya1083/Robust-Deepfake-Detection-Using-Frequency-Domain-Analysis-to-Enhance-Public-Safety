"""
FFT Frequency Analysis Module
==============================
Real Fast Fourier Transform (FFT) based frequency-domain feature extraction
for deepfake detection.

Key Features:
- 2D FFT with centered shift (DC at centre).
- Log-scaled magnitude spectrum for visualization.
- Numerical feature extraction: energy bands, mean, std, distribution.
- Base64-encoded spectrum image for API transport.
- Modular design — FFT feature vector reusable in Phase 6 feature fusion.

Class Mapping (informational only — FFT alone does not classify REAL/DEEPFAKE):
  REAL     = 0
  DEEPFAKE = 1
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFT Analysis Constants
# ---------------------------------------------------------------------------

# Input image size for FFT (resize before analysis for consistency)
FFT_INPUT_SIZE: Tuple[int, int] = (224, 224)

# Radial frequency band boundaries (fraction of Nyquist radius 0..1)
#   low  : 0.00 – 0.10  (global structure, illumination)
#   mid  : 0.10 – 0.40  (coarse textures, facial features)
#   high : 0.40 – 1.00  (fine details, edges, GAN artefacts)
LOW_BAND_RATIO:  Tuple[float, float] = (0.00, 0.10)
MID_BAND_RATIO:  Tuple[float, float] = (0.10, 0.40)
HIGH_BAND_RATIO: Tuple[float, float] = (0.40, 1.00)

# Number of radial bins for the azimuthally averaged power spectrum
NUM_RADIAL_BINS: int = 64


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FFTFeatures:
    """
    Numerical FFT feature vector extracted from one face image.
    Designed for direct concatenation with CNN spatial features in Phase 6.

    Total feature dimensionality: 6 scalar + 64 radial = 70 values.
    """
    # Scalar summary statistics
    mean_magnitude:   float = 0.0
    std_magnitude:    float = 0.0
    low_energy:       float = 0.0  # fraction of total energy in low-freq band
    mid_energy:       float = 0.0  # fraction of total energy in mid-freq band
    high_energy:      float = 0.0  # fraction of total energy in high-freq band
    spectral_entropy: float = 0.0  # Shannon entropy of the normalised power spectrum

    # Azimuthally averaged radial power spectrum (64 bins, normalised)
    radial_spectrum: List[float] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Return flat numpy array [6 scalars + 64 radial] = 70-dim."""
        scalars = np.array([
            self.mean_magnitude,
            self.std_magnitude,
            self.low_energy,
            self.mid_energy,
            self.high_energy,
            self.spectral_entropy,
        ], dtype=np.float32)
        radial = np.array(self.radial_spectrum, dtype=np.float32)
        return np.concatenate([scalars, radial])

    def to_dict(self) -> Dict:
        return {
            "mean_magnitude":   round(float(self.mean_magnitude),   6),
            "std_magnitude":    round(float(self.std_magnitude),     6),
            "low_energy":       round(float(self.low_energy),        6),
            "mid_energy":       round(float(self.mid_energy),        6),
            "high_energy":      round(float(self.high_energy),       6),
            "spectral_entropy": round(float(self.spectral_entropy),  6),
            "radial_spectrum":  [round(float(v), 6) for v in self.radial_spectrum],
            "feature_dim":      len(self.to_vector()),
        }


@dataclass
class FFTResult:
    """Full result returned by FFTAnalyzer.analyze()."""
    success:          bool = False
    features:         Optional[FFTFeatures] = None
    # Base64-encoded JPEG images for API / frontend display
    original_b64:     Optional[str] = None   # resized grayscale input
    magnitude_b64:    Optional[str] = None   # raw log-magnitude spectrum
    log_spectrum_b64: Optional[str] = None   # colourised (BONE colormap) spectrum
    image_size:       Optional[Tuple[int, int]] = None  # (H, W) of FFT input
    error:            Optional[str] = None


# ---------------------------------------------------------------------------
# Core Analyser
# ---------------------------------------------------------------------------

class FFTAnalyzer:
    """
    Performs real 2D FFT analysis on face images.

    Usage
    -----
    analyzer = FFTAnalyzer()
    result   = analyzer.analyze(pil_image_or_numpy_bgr)
    features = result.features.to_vector()   # 70-dim float32 for Phase 6
    """

    def __init__(self, input_size: Tuple[int, int] = FFT_INPUT_SIZE):
        self.input_size = input_size  # (H, W)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_input,
        generate_visuals: bool = True,
    ) -> FFTResult:
        """
        Run FFT analysis on an image.

        Parameters
        ----------
        image_input : PIL.Image | np.ndarray (BGR or RGB) | file path str
        generate_visuals : bool
            When True, produce base64-encoded spectrum images.

        Returns
        -------
        FFTResult
        """
        # 1. Decode to grayscale numpy array
        gray = self._to_grayscale(image_input)
        if gray is None:
            return FFTResult(success=False, error="Could not decode image to grayscale.")

        H, W = gray.shape
        if H < 8 or W < 8:
            return FFTResult(success=False, error=f"Image too small ({W}×{H}). Minimum 8×8.")

        # 2. Resize to fixed input size for consistent FFT bins
        gray_resized = cv2.resize(gray, (self.input_size[1], self.input_size[0]),
                                  interpolation=cv2.INTER_AREA)

        # 3. 2D FFT
        fft2d       = np.fft.fft2(gray_resized.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft2d)          # DC at centre

        # 4. Magnitude spectrum
        magnitude   = np.abs(fft_shifted)              # linear scale
        log_mag     = np.log1p(magnitude)              # log scale (1 + |F|)

        # 5. Extract numerical features
        features = self._extract_features(magnitude, log_mag)

        # 6. Optionally build visualisation images
        orig_b64 = mag_b64 = log_b64 = None
        if generate_visuals:
            orig_b64 = self._ndarray_to_b64(gray_resized, cmap=None)
            mag_b64  = self._spectrum_to_b64(log_mag, colormap=cv2.COLORMAP_BONE)
            log_b64  = self._spectrum_to_b64(log_mag, colormap=cv2.COLORMAP_JET)

        return FFTResult(
            success=True,
            features=features,
            original_b64=orig_b64,
            magnitude_b64=mag_b64,
            log_spectrum_b64=log_b64,
            image_size=(self.input_size[0], self.input_size[1]),
            error=None,
        )

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        magnitude: np.ndarray,
        log_mag:   np.ndarray,
    ) -> FFTFeatures:
        """Extract scalar and radial spectrum features from FFT magnitude."""

        H, W = magnitude.shape
        cy, cx = H // 2, W // 2  # centre (DC component after shift)
        max_radius = min(cy, cx)

        # Build coordinate grids
        y_idx, x_idx = np.indices((H, W))
        radius_map = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

        # --- Scalar statistics on log magnitude --------------------------
        mean_mag = float(np.mean(log_mag))
        std_mag  = float(np.std(log_mag))

        # --- Radial energy bands -----------------------------------------
        total_energy = float(np.sum(magnitude ** 2)) + 1e-12  # prevent /0

        def band_energy(r_min_frac, r_max_frac) -> float:
            r_min = r_min_frac * max_radius
            r_max = r_max_frac * max_radius
            mask  = (radius_map >= r_min) & (radius_map < r_max)
            return float(np.sum((magnitude[mask]) ** 2)) / total_energy

        low_e  = band_energy(*LOW_BAND_RATIO)
        mid_e  = band_energy(*MID_BAND_RATIO)
        high_e = band_energy(*HIGH_BAND_RATIO)

        # --- Spectral entropy (Shannon entropy of normalised power) -------
        power = magnitude ** 2
        power_norm = power / (power.sum() + 1e-12)
        # Clip to avoid log(0)
        p = power_norm[power_norm > 0]
        spectral_entropy = float(-np.sum(p * np.log2(p)))
        # Normalise by log2(N) so it is in [0, 1]
        spectral_entropy /= np.log2(float(H * W))

        # --- Azimuthally averaged radial power spectrum -------------------
        bin_edges   = np.linspace(0, max_radius, NUM_RADIAL_BINS + 1)
        radial_bins = np.zeros(NUM_RADIAL_BINS, dtype=np.float64)

        for i in range(NUM_RADIAL_BINS):
            mask = (radius_map >= bin_edges[i]) & (radius_map < bin_edges[i + 1])
            if mask.any():
                radial_bins[i] = np.mean(log_mag[mask])

        # Normalise radial spectrum to [0, 1]
        r_min, r_max = radial_bins.min(), radial_bins.max()
        if r_max > r_min:
            radial_bins = (radial_bins - r_min) / (r_max - r_min)
        else:
            radial_bins[:] = 0.0

        return FFTFeatures(
            mean_magnitude=mean_mag,
            std_magnitude=std_mag,
            low_energy=low_e,
            mid_energy=mid_e,
            high_energy=high_e,
            spectral_entropy=spectral_entropy,
            radial_spectrum=radial_bins.tolist(),
        )

    # ------------------------------------------------------------------
    # Image Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(image_input) -> Optional[np.ndarray]:
        """Convert any supported input type to uint8 grayscale numpy array."""
        try:
            if isinstance(image_input, np.ndarray):
                arr = image_input
                if arr.ndim == 3 and arr.shape[2] == 3:
                    # Could be BGR or RGB — convert either way to gray
                    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                elif arr.ndim == 2:
                    return arr.astype(np.uint8)
                return None

            if isinstance(image_input, Image.Image):
                return np.array(image_input.convert("L"))

            if isinstance(image_input, (str,)):
                from pathlib import Path
                path = Path(image_input)
                if not path.exists():
                    return None
                img = cv2.imread(str(path))
                if img is None:
                    return None
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # bytes / BytesIO
            if isinstance(image_input, (bytes, io.BytesIO)):
                data = image_input if isinstance(image_input, bytes) else image_input.getvalue()
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                return img

        except Exception as exc:
            logger.warning("FFTAnalyzer._to_grayscale error: %s", exc)
        return None

    @staticmethod
    def _spectrum_to_b64(log_mag: np.ndarray, colormap: int = cv2.COLORMAP_BONE) -> str:
        """Normalise log-magnitude spectrum → 8-bit colourised → base64 JPEG."""
        norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)
        coloured = cv2.applyColorMap(norm_u8, colormap)
        _, buf = cv2.imencode(".jpg", coloured, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _ndarray_to_b64(gray: np.ndarray, cmap=None) -> str:
        """Encode grayscale uint8 numpy array as base64 JPEG."""
        _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
fft_analyzer = FFTAnalyzer()


def run_fft_analysis(image_input, generate_visuals: bool = True) -> FFTResult:
    """
    Convenience function. Calls the shared FFTAnalyzer instance.

    Parameters
    ----------
    image_input : PIL.Image | np.ndarray (BGR) | str (file path) | bytes
    generate_visuals : bool

    Returns
    -------
    FFTResult
    """
    return fft_analyzer.analyze(image_input, generate_visuals=generate_visuals)
