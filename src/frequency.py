"""
FFT Frequency Analysis Module
==============================
Real Fast Fourier Transform (FFT) based frequency-domain feature extraction
for deepfake detection.

Key Features:
- 2D FFT with centered shift (DC zero-frequency at centre).
- Log-scaled magnitude spectrum for visualization.
- Multi-band decomposition: low (0-10%), mid (10-40%), high (40-100%).
- Azimuthally averaged radial power spectrum (64 frequency bins).
- Numerical metrics: total energy, low/mid/high ratios, spectral entropy.
- Dynamic scientific explanation tailored to computed spectral measurements.
- Base64-encoded spectrum images for seamless API transport.
- Preserves 70-dim feature vector for multi-modal model compatibility.
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

FFT_INPUT_SIZE: Tuple[int, int] = (224, 224)

# Radial frequency band boundaries (fraction of Nyquist radius 0..1)
LOW_BAND_RATIO: Tuple[float, float] = (0.00, 0.10)
MID_BAND_RATIO: Tuple[float, float] = (0.10, 0.40)
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
    Total feature vector dimensionality: 6 scalar + 64 radial = 70 values.
    """
    mean_magnitude: float = 0.0
    std_magnitude: float = 0.0
    low_energy: float = 0.0      # fraction of total energy in low-freq band
    mid_energy: float = 0.0      # fraction of total energy in mid-freq band
    high_energy: float = 0.0     # fraction of total energy in high-freq band
    spectral_entropy: float = 0.0  # Shannon entropy of normalised power spectrum
    total_energy: float = 0.0    # unnormalized sum of squared magnitude
    high_energy_ratio: float = 0.0  # high energy / (low + mid + high)
    explanation: str = ""

    # Azimuthally averaged radial power spectrum (64 bins, normalised to [0, 1])
    radial_spectrum: List[float] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Return flat numpy array [6 scalars + 64 radial] = 70-dim for model compatibility."""
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
            "mean_magnitude": round(float(self.mean_magnitude), 6),
            "std_magnitude": round(float(self.std_magnitude), 6),
            "low_energy": round(float(self.low_energy), 6),
            "mid_energy": round(float(self.mid_energy), 6),
            "high_energy": round(float(self.high_energy), 6),
            "low_energy_pct": round(float(self.low_energy * 100), 2),
            "mid_energy_pct": round(float(self.mid_energy * 100), 2),
            "high_energy_pct": round(float(self.high_energy * 100), 2),
            "total_energy": round(float(self.total_energy), 2),
            "high_energy_ratio": round(float(self.high_energy_ratio), 4),
            "spectral_entropy": round(float(self.spectral_entropy), 6),
            "radial_spectrum": [round(float(v), 6) for v in self.radial_spectrum],
            "feature_dim": len(self.to_vector()),
            "explanation": self.explanation,
        }


@dataclass
class FFTResult:
    """Full result returned by FFTAnalyzer.analyze()."""
    success: bool = False
    features: Optional[FFTFeatures] = None
    original_b64: Optional[str] = None      # input grayscale image
    magnitude_b64: Optional[str] = None     # log-magnitude spectrum (BONE colormap)
    log_spectrum_b64: Optional[str] = None  # colorized spectrum (JET colormap)
    image_size: Optional[Tuple[int, int]] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core Analyser
# ---------------------------------------------------------------------------

class FFTAnalyzer:
    """Performs 2D FFT analysis on face images."""

    def __init__(self, input_size: Tuple[int, int] = FFT_INPUT_SIZE):
        self.input_size = input_size

    def analyze(self, image_input, generate_visuals: bool = True) -> FFTResult:
        # 1. Decode to grayscale numpy array
        gray = self._to_grayscale(image_input)
        if gray is None:
            return FFTResult(success=False, error="Could not decode image to grayscale.")

        H, W = gray.shape
        if H < 8 or W < 8:
            return FFTResult(success=False, error=f"Image too small ({W}×{H}). Minimum 8×8.")

        # 2. Resize to fixed input size for consistent FFT frequency bins
        gray_resized = cv2.resize(
            gray,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_AREA,
        )

        # 3. 2D Fast Fourier Transform
        fft2d = np.fft.fft2(gray_resized.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft2d)  # shift DC zero-frequency to centre

        # 4. Magnitude spectrum & Logarithmic scaling
        magnitude = np.abs(fft_shifted)
        log_mag = np.log1p(magnitude)  # log(1 + |F(u, v)|)

        # 5. Extract numerical features
        features = self._extract_features(magnitude, log_mag)

        # 6. Generate dynamic explanation
        explanation = self._generate_explanation(features)
        features.explanation = explanation

        # 7. Visualisations
        orig_b64 = mag_b64 = log_b64 = None
        if generate_visuals:
            orig_b64 = self._ndarray_to_b64(gray_resized)
            mag_b64 = self._spectrum_to_b64(log_mag, colormap=cv2.COLORMAP_BONE)
            log_b64 = self._spectrum_to_b64(log_mag, colormap=cv2.COLORMAP_JET)

        return FFTResult(
            success=True,
            features=features,
            original_b64=orig_b64,
            magnitude_b64=mag_b64,
            log_spectrum_b64=log_b64,
            image_size=(self.input_size[0], self.input_size[1]),
            explanation=explanation,
            error=None,
        )

    def _extract_features(self, magnitude: np.ndarray, log_mag: np.ndarray) -> FFTFeatures:
        H, W = magnitude.shape
        cy, cx = H // 2, W // 2  # DC component center
        max_radius = min(cy, cx)

        y_idx, x_idx = np.indices((H, W))
        radius_map = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

        mean_mag = float(np.mean(log_mag))
        std_mag = float(np.std(log_mag))

        total_power = float(np.sum(magnitude ** 2))
        total_energy = total_power + 1e-12

        def band_energy(r_min_frac, r_max_frac) -> float:
            r_min = r_min_frac * max_radius
            r_max = r_max_frac * max_radius
            mask = (radius_map >= r_min) & (radius_map < r_max)
            return float(np.sum((magnitude[mask]) ** 2)) / total_energy

        low_e = band_energy(*LOW_BAND_RATIO)
        mid_e = band_energy(*MID_BAND_RATIO)
        high_e = band_energy(*HIGH_BAND_RATIO)

        denom = low_e + mid_e + high_e
        high_ratio = (high_e / denom) if denom > 0 else 0.0

        # Spectral entropy (Shannon entropy of normalized power)
        power = magnitude ** 2
        power_norm = power / total_energy
        p = power_norm[power_norm > 0]
        spectral_entropy = float(-np.sum(p * np.log2(p)))
        spectral_entropy /= np.log2(float(H * W))

        # Azimuthally averaged radial power spectrum (64 bins)
        bin_edges = np.linspace(0, max_radius, NUM_RADIAL_BINS + 1)
        radial_bins = np.zeros(NUM_RADIAL_BINS, dtype=np.float64)

        for i in range(NUM_RADIAL_BINS):
            mask = (radius_map >= bin_edges[i]) & (radius_map < bin_edges[i + 1])
            if mask.any():
                radial_bins[i] = np.mean(log_mag[mask])

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
            total_energy=total_power,
            high_energy_ratio=high_ratio,
            radial_spectrum=radial_bins.tolist(),
        )

    @staticmethod
    def _generate_explanation(features: FFTFeatures) -> str:
        """Generate a factual, scientific explanation of computed frequency metrics."""
        low_pct = features.low_energy * 100
        mid_pct = features.mid_energy * 100
        high_pct = features.high_energy * 100

        if high_pct > 25.0:
            pattern = (
                f"Elevated high-frequency spectral energy ({high_pct:.1f}%) observed. "
                "In deepfake detection, high-frequency spikes or slow spectral decay often correspond to "
                "transposed-convolution upsampling grids, checkerboard artifacts, or synthetic blending seams."
            )
        elif high_pct < 5.0:
            pattern = (
                f"Strong low-frequency dominance ({low_pct:.1f}%) with subdued high frequencies ({high_pct:.1f}%). "
                "This distribution indicates heavy smoothing or natural optical low-pass decay characteristic of standard photography."
            )
        else:
            pattern = (
                f"Balanced energy progression: {low_pct:.1f}% low (structural), {mid_pct:.1f}% mid (texture), "
                f"and {high_pct:.1f}% high (edge/detail) frequency distribution."
            )

        return (
            f"Frequency analysis computes the 2D Fast Fourier Transform to inspect the radial energy decay. {pattern} "
            "Frequency distribution serves as an analytical descriptor and not a standalone classifier."
        )

    @staticmethod
    def _to_grayscale(image_input) -> Optional[np.ndarray]:
        try:
            if isinstance(image_input, np.ndarray):
                arr = image_input
                if arr.ndim == 3 and arr.shape[2] == 3:
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
        norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)
        coloured = cv2.applyColorMap(norm_u8, colormap)
        _, buf = cv2.imencode(".jpg", coloured, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _ndarray_to_b64(gray: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")


fft_analyzer = FFTAnalyzer()


def run_fft_analysis(image_input, generate_visuals: bool = True) -> FFTResult:
    return fft_analyzer.analyze(image_input, generate_visuals=generate_visuals)
