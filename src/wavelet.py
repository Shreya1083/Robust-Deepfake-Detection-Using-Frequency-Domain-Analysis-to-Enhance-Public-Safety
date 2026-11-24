"""
DWT (Discrete Wavelet Transform) Frequency Analysis Module
==========================================================
Real 2D Discrete Wavelet Transform (DWT) multi-scale frequency decomposition
for deepfake detection using PyWavelets.

Key Features:
- 2D DWT using pywt.dwt2() with Haar wavelet (default).
- Sub-band decomposition into LL (Approximation), LH (Horizontal detail),
  HL (Vertical detail), and HH (Diagonal detail).
- Extract rigorous numerical statistical and energy features for each sub-band.
- Sub-band energy distribution and relative energy ratios.
- Base64-encoded visualisations for individual sub-bands and combined 2x2 grid.
- Independent and reusable feature vector for Phase 6 multimodal fusion.

Class Mapping (informational only — DWT alone does not classify REAL/DEEPFAKE):
  REAL     = 0
  DEEPFAKE = 1
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import pywt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DWT Analysis Constants
# ---------------------------------------------------------------------------

# Input image size for DWT (resize before analysis for consistent sub-band sizes)
DWT_INPUT_SIZE: Tuple[int, int] = (224, 224)

# Default wavelet family
DEFAULT_WAVELET: str = "haar"

# Sub-band identifiers
SUBBANDS: Tuple[str, str, str, str] = ("LL", "LH", "HL", "HH")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubbandFeatures:
    """Numerical features for a single DWT sub-band."""
    mean: float = 0.0
    std: float = 0.0
    energy: float = 0.0             # sum of squared coefficients
    mean_abs: float = 0.0          # mean absolute coefficient
    max_abs: float = 0.0           # max absolute coefficient
    energy_ratio: float = 0.0      # fraction of total wavelet energy in this band

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(float(self.mean), 6),
            "std": round(float(self.std), 6),
            "energy": round(float(self.energy), 6),
            "mean_abs": round(float(self.mean_abs), 6),
            "max_abs": round(float(self.max_abs), 6),
            "energy_ratio": round(float(self.energy_ratio), 6),
        }


@dataclass
class DWTFeatures:
    """
    Comprehensive numerical DWT feature vector extracted from one face image.
    Designed for direct concatenation with CNN and FFT features in Phase 6.

    Feature Structure:
    - 4 sub-bands x 5 metrics (mean, std, energy, mean_abs, max_abs) = 20 values
    - 4 relative energy ratios (LL, LH, HL, HH) = 4 values
    - Summary metrics: total_energy, detail_to_approx_ratio,
      high_freq_energy_ratio, wavelet_entropy = 4 values
    Total vector dimensionality = 28 values.
    """
    ll: SubbandFeatures = field(default_factory=SubbandFeatures)
    lh: SubbandFeatures = field(default_factory=SubbandFeatures)
    hl: SubbandFeatures = field(default_factory=SubbandFeatures)
    hh: SubbandFeatures = field(default_factory=SubbandFeatures)

    total_energy: float = 0.0
    detail_to_approx_ratio: float = 0.0
    high_freq_energy_ratio: float = 0.0
    wavelet_entropy: float = 0.0
    wavelet_type: str = DEFAULT_WAVELET

    def to_vector(self) -> np.ndarray:
        """
        Return flat numpy array [28-dimensional float32 vector].
        Feature ordering is deterministic and guaranteed stable across runs.
        """
        vals: List[float] = []
        # 1. Per-band metrics (4 * 5 = 20)
        for band in (self.ll, self.lh, self.hl, self.hh):
            vals.extend([
                band.mean,
                band.std,
                band.energy,
                band.mean_abs,
                band.max_abs,
            ])
        # 2. Relative energy ratios (4)
        vals.extend([
            self.ll.energy_ratio,
            self.lh.energy_ratio,
            self.hl.energy_ratio,
            self.hh.energy_ratio,
        ])
        # 3. Global summary features (4)
        vals.extend([
            self.total_energy,
            self.detail_to_approx_ratio,
            self.high_freq_energy_ratio,
            self.wavelet_entropy,
        ])
        return np.array(vals, dtype=np.float32)

    def to_dict(self) -> Dict:
        return {
            "subbands": {
                "LL": self.ll.to_dict(),
                "LH": self.lh.to_dict(),
                "HL": self.hl.to_dict(),
                "HH": self.hh.to_dict(),
            },
            "energy_distribution": {
                "ll_ratio": round(float(self.ll.energy_ratio), 6),
                "lh_ratio": round(float(self.lh.energy_ratio), 6),
                "hl_ratio": round(float(self.hl.energy_ratio), 6),
                "hh_ratio": round(float(self.hh.energy_ratio), 6),
            },
            "total_energy": round(float(self.total_energy), 6),
            "detail_to_approx_ratio": round(float(self.detail_to_approx_ratio), 6),
            "high_freq_energy_ratio": round(float(self.high_freq_energy_ratio), 6),
            "wavelet_entropy": round(float(self.wavelet_entropy), 6),
            "wavelet_type": self.wavelet_type,
            "feature_dim": len(self.to_vector()),
        }


@dataclass
class DWTResult:
    """Full result returned by DWTAnalyzer.analyze()."""
    success: bool = False
    features: Optional[DWTFeatures] = None
    # Base64-encoded JPEG images for API / frontend display
    original_b64: Optional[str] = None   # Resized grayscale input (224x224)
    ll_b64: Optional[str] = None         # Approximation sub-band
    lh_b64: Optional[str] = None         # Horizontal detail sub-band
    hl_b64: Optional[str] = None         # Vertical detail sub-band
    hh_b64: Optional[str] = None         # Diagonal detail sub-band
    combined_b64: Optional[str] = None   # 2x2 combined decomposition grid
    subband_shape: Optional[Tuple[int, int]] = None
    image_size: Optional[Tuple[int, int]] = None
    wavelet_type: str = DEFAULT_WAVELET
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core Analyzer
# ---------------------------------------------------------------------------

class DWTAnalyzer:
    """
    Performs real 2D Discrete Wavelet Transform (DWT) on face images.

    Usage
    -----
    analyzer = DWTAnalyzer(wavelet='haar')
    result   = analyzer.analyze(pil_image_or_numpy_bgr)
    features = result.features.to_vector()   # 28-dim float32 for Phase 6
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = DWT_INPUT_SIZE,
        wavelet: str = DEFAULT_WAVELET,
    ):
        self.input_size = input_size
        self.wavelet = wavelet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes, io.BytesIO],
        generate_visuals: bool = True,
        wavelet: Optional[str] = None,
    ) -> DWTResult:
        """
        Run 2D DWT analysis on an image.

        Parameters
        ----------
        image_input : PIL.Image | np.ndarray (BGR or RGB) | file path str | bytes
        generate_visuals : bool
            When True, produce base64-encoded sub-band and combined images.
        wavelet : str, optional
            Wavelet name (default is self.wavelet, usually 'haar').

        Returns
        -------
        DWTResult
        """
        wavelet_name = wavelet or self.wavelet

        # Validate wavelet family
        if wavelet_name not in pywt.wavelist(kind="discrete"):
            return DWTResult(
                success=False,
                error=f"Invalid discrete wavelet '{wavelet_name}'. Choose from pywt discrete wavelets (e.g., 'haar', 'db2').",
            )

        # 1. Decode to grayscale numpy array
        gray = self._to_grayscale(image_input)
        if gray is None:
            return DWTResult(success=False, error="Could not decode image to grayscale.")

        H, W = gray.shape
        if H < 8 or W < 8:
            return DWTResult(success=False, error=f"Image too small ({W}x{H}). Minimum 8x8.")

        # 2. Resize to fixed input size for consistent decomposition
        gray_resized = cv2.resize(
            gray,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_AREA,
        )

        # 3. 2D Discrete Wavelet Transform (1-level)
        try:
            coeffs2 = pywt.dwt2(gray_resized.astype(np.float64), wavelet_name)
            LL, (LH, HL, HH) = coeffs2
        except Exception as exc:
            logger.error("DWT computation error: %s", exc)
            return DWTResult(success=False, error=f"Wavelet decomposition failed: {exc}")

        subband_shape = (int(LL.shape[0]), int(LL.shape[1]))

        # 4. Extract numerical features
        features = self._extract_features(LL, LH, HL, HH, wavelet_name)

        # 5. Optionally build visualisations
        orig_b64 = ll_b64 = lh_b64 = hl_b64 = hh_b64 = combined_b64 = None
        if generate_visuals:
            orig_b64 = self._ndarray_to_b64(gray_resized)
            ll_b64 = self._coeff_to_b64(LL, is_approx=True)
            lh_b64 = self._coeff_to_b64(LH, is_approx=False)
            hl_b64 = self._coeff_to_b64(HL, is_approx=False)
            hh_b64 = self._coeff_to_b64(HH, is_approx=False)
            combined_b64 = self._build_combined_grid_b64(LL, LH, HL, HH)

        return DWTResult(
            success=True,
            features=features,
            original_b64=orig_b64,
            ll_b64=ll_b64,
            lh_b64=lh_b64,
            hl_b64=hl_b64,
            hh_b64=hh_b64,
            combined_b64=combined_b64,
            subband_shape=subband_shape,
            image_size=(self.input_size[0], self.input_size[1]),
            wavelet_type=wavelet_name,
            error=None,
        )

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        LL: np.ndarray,
        LH: np.ndarray,
        HL: np.ndarray,
        HH: np.ndarray,
        wavelet_name: str,
    ) -> DWTFeatures:
        """Calculate statistical and energy features for each sub-band."""

        def compute_subband_stats(band: np.ndarray) -> Tuple[float, float, float, float, float]:
            mean_val = float(np.mean(band))
            std_val = float(np.std(band))
            energy_val = float(np.sum(band ** 2))
            mean_abs_val = float(np.mean(np.abs(band)))
            max_abs_val = float(np.max(np.abs(band)))
            return mean_val, std_val, energy_val, mean_abs_val, max_abs_val

        ll_m, ll_s, ll_e, ll_ma, ll_mx = compute_subband_stats(LL)
        lh_m, lh_s, lh_e, lh_ma, lh_mx = compute_subband_stats(LH)
        hl_m, hl_s, hl_e, hl_ma, hl_mx = compute_subband_stats(HL)
        hh_m, hh_s, hh_e, hh_ma, hh_mx = compute_subband_stats(HH)

        # Total wavelet energy
        total_e = ll_e + lh_e + hl_e + hh_e
        safe_total_e = total_e if total_e > 1e-12 else 1e-12

        # Relative energy ratios
        ll_ratio = ll_e / safe_total_e
        lh_ratio = lh_e / safe_total_e
        hl_ratio = hl_e / safe_total_e
        hh_ratio = hh_e / safe_total_e

        # Detail-to-approximation ratio
        detail_e = lh_e + hl_e + hh_e
        detail_to_approx = detail_e / (ll_e + 1e-12)
        high_freq_ratio = detail_e / safe_total_e

        # Wavelet Shannon entropy across the 4 sub-band energy proportions
        ratios = np.array([ll_ratio, lh_ratio, hl_ratio, hh_ratio], dtype=np.float64)
        positive_ratios = ratios[ratios > 1e-12]
        # Shannon entropy normalized by log2(4) = 2.0 to bound in [0, 1]
        raw_entropy = -np.sum(positive_ratios * np.log2(positive_ratios))
        wavelet_entropy = float(raw_entropy / 2.0)

        ll_feat = SubbandFeatures(mean=ll_m, std=ll_s, energy=ll_e, mean_abs=ll_ma, max_abs=ll_mx, energy_ratio=ll_ratio)
        lh_feat = SubbandFeatures(mean=lh_m, std=lh_s, energy=lh_e, mean_abs=lh_ma, max_abs=lh_mx, energy_ratio=lh_ratio)
        hl_feat = SubbandFeatures(mean=hl_m, std=hl_s, energy=hl_e, mean_abs=hl_ma, max_abs=hl_mx, energy_ratio=hl_ratio)
        hh_feat = SubbandFeatures(mean=hh_m, std=hh_s, energy=hh_e, mean_abs=hh_ma, max_abs=hh_mx, energy_ratio=hh_ratio)

        return DWTFeatures(
            ll=ll_feat,
            lh=lh_feat,
            hl=hl_feat,
            hh=hh_feat,
            total_energy=total_e,
            detail_to_approx_ratio=detail_to_approx,
            high_freq_energy_ratio=high_freq_ratio,
            wavelet_entropy=wavelet_entropy,
            wavelet_type=wavelet_name,
        )

    # ------------------------------------------------------------------
    # Image Helpers & Visualizations
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(image_input) -> Optional[np.ndarray]:
        """Convert any supported input type to uint8 grayscale numpy array."""
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
            logger.warning("DWTAnalyzer._to_grayscale error: %s", exc)
        return None

    @staticmethod
    def _normalize_coeff(band: np.ndarray, is_approx: bool = False) -> np.ndarray:
        """
        Normalize coefficient array to uint8 [0, 255] for visual representation.
        - Approximation (LL): Direct linear min-max stretch.
        - Details (LH, HL, HH): Absolute values scaled to [0, 255] with high-contrast boost
          so subtle edge artifacts are clearly perceptible.
        """
        if is_approx:
            b_min, b_max = np.min(band), np.max(band)
            if b_max > b_min:
                norm = ((band - b_min) / (b_max - b_min) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(band, dtype=np.uint8)
            return norm
        else:
            # Detail band: visualize absolute magnitude
            abs_band = np.abs(band)
            mx = np.max(abs_band)
            if mx > 1e-12:
                # Square-root gamma stretch to enhance weak edge/noise details
                stretched = np.sqrt(abs_band / mx) * 255.0
                return np.clip(stretched, 0, 255).astype(np.uint8)
            return np.zeros_like(band, dtype=np.uint8)

    @classmethod
    def _coeff_to_b64(cls, band: np.ndarray, is_approx: bool = False) -> str:
        """Convert a 2D sub-band to normalized base64 JPEG."""
        u8_norm = cls._normalize_coeff(band, is_approx=is_approx)
        # Apply colormap to detail bands for richer forensic clarity
        if not is_approx:
            vis = cv2.applyColorMap(u8_norm, cv2.COLORMAP_VIRIDIS)
        else:
            # Approximation band shown in clean grayscale
            vis = cv2.cvtColor(u8_norm, cv2.COLOR_GRAY2BGR)

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @classmethod
    def _build_combined_grid_b64(
        cls,
        LL: np.ndarray,
        LH: np.ndarray,
        HL: np.ndarray,
        HH: np.ndarray,
    ) -> str:
        """
        Build the standard 2x2 DWT decomposition quad-grid:
        [ LL (Approximation) | HL (Vertical Details)   ]
        [ LH (Horizontal)    | HH (Diagonal Details)   ]
        With subtle quadrant separation lines.
        """
        ll_u8 = cls._normalize_coeff(LL, is_approx=True)
        lh_u8 = cls._normalize_coeff(LH, is_approx=False)
        hl_u8 = cls._normalize_coeff(HL, is_approx=False)
        hh_u8 = cls._normalize_coeff(HH, is_approx=False)

        # Convert all to BGR
        ll_bgr = cv2.cvtColor(ll_u8, cv2.COLOR_GRAY2BGR)
        lh_bgr = cv2.applyColorMap(lh_u8, cv2.COLORMAP_VIRIDIS)
        hl_bgr = cv2.applyColorMap(hl_u8, cv2.COLORMAP_VIRIDIS)
        hh_bgr = cv2.applyColorMap(hh_u8, cv2.COLORMAP_VIRIDIS)

        # Concatenate rows
        top_row = np.hstack([ll_bgr, hl_bgr])
        bot_row = np.hstack([lh_bgr, hh_bgr])
        grid = np.vstack([top_row, bot_row])

        # Draw thin dividing borders
        h, w = grid.shape[:2]
        ch, cw = h // 2, w // 2
        grid[ch - 1:ch + 2, :] = [255, 255, 255]  # horizontal divider line
        grid[:, cw - 1:cw + 2] = [255, 255, 255]  # vertical divider line

        _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _ndarray_to_b64(gray: np.ndarray) -> str:
        """Encode grayscale uint8 numpy array as base64 JPEG."""
        _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
dwt_analyzer = DWTAnalyzer()


def run_dwt_analysis(
    image_input: Union[Image.Image, np.ndarray, str, bytes, io.BytesIO],
    generate_visuals: bool = True,
    wavelet: Optional[str] = None,
) -> DWTResult:
    """
    Convenience function. Calls the shared DWTAnalyzer instance.

    Parameters
    ----------
    image_input : PIL.Image | np.ndarray (BGR) | str (file path) | bytes
    generate_visuals : bool
    wavelet : str, optional

    Returns
    -------
    DWTResult
    """
    return dwt_analyzer.analyze(
        image_input,
        generate_visuals=generate_visuals,
        wavelet=wavelet,
    )
