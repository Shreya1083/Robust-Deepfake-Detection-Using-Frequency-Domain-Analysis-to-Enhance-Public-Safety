"""
Multimodal Feature Fusion Module (CNN + FFT + DWT)
===================================================
Combines three complementary forensic feature representations into a single
unified representation for deepfake detection:

1. CNN Spatial Features (512-dim):
   Penultimate average pooling embedding from pretrained ResNet18 capturing
   fine facial blend boundaries, dermal texture abnormalities, and spatial artifacts.

2. FFT Frequency Domain Features (70-dim):
   Spectral summary statistics (6 scalar values) and azimuthally averaged radial
   power spectrum (64 radial bins) from 2D Fast Fourier Transform capturing
   frequency roll-off and periodic checkerboard artefacts.

3. DWT Wavelet Features (28-dim):
   Sub-band coefficient statistics (LL, LH, HL, HH: 20 values), relative energy
   ratios (4 values), and global wavelet metrics (4 values) from 2D Discrete Wavelet
   Transform capturing localized spatial-frequency discontinuities.

Total Fused Feature Dimensionality = 512 + 70 + 28 = 610.

Key Components:
- MultimodalFusionClassifier: PyTorch MLP (610 -> 256 -> 64 -> 2).
- FeatureNormalizer: Independent branch-level z-score normalization computed
  strictly on training data.
- MultimodalFeatureExtractor: Deterministic extraction from raw or face-cropped images.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, CLASS_LABELS
from src.dataset_loader import get_spatial_transforms
from src.frequency import run_fft_analysis
from src.wavelet import run_dwt_analysis

logger = logging.getLogger(__name__)

# Feature dimensionality constants
CNN_FEATURE_DIM: int = 512
FFT_FEATURE_DIM: int = 70
DWT_FEATURE_DIM: int = 28
FUSION_FEATURE_DIM: int = CNN_FEATURE_DIM + FFT_FEATURE_DIM + DWT_FEATURE_DIM  # 610


class MultimodalFusionClassifier(nn.Module):
    """
    Multilayer Perceptron (MLP) for classification on fused 610-dimensional features.

    Architecture:
      Input (610)
        -> Linear(610, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
        -> Linear(256, 64)  -> BatchNorm1d(64)  -> ReLU -> Dropout(0.2)
        -> Linear(64, 2)
    """

    def __init__(
        self,
        input_dim: int = FUSION_FEATURE_DIM,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        num_classes: int = 2,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized classification logits.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 610)

        Returns
        -------
        torch.Tensor of shape (B, 2)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probability distributions (B, 2)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class FeatureNormalizer:
    """
    Independent branch-level z-score normalizer:
        z = (x - mean) / (std + eps)

    Guarantees:
    - Normalization parameters (mean, std) must be computed ONLY on training data.
    - Missing or non-finite values (NaN / Inf) are sanitized safely.
    - Independent normalization preserves relative scale of each domain
      (spatial embeddings, power spectrum, wavelet coefficients).
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.is_fitted = False

        self.cnn_mean: Optional[np.ndarray] = None
        self.cnn_std: Optional[np.ndarray] = None

        self.fft_mean: Optional[np.ndarray] = None
        self.fft_std: Optional[np.ndarray] = None

        self.dwt_mean: Optional[np.ndarray] = None
        self.dwt_std: Optional[np.ndarray] = None

    def fit(
        self,
        cnn_feats: np.ndarray,
        fft_feats: np.ndarray,
        dwt_feats: np.ndarray,
    ) -> "FeatureNormalizer":
        """
        Compute normalization statistics strictly on the training set.

        Parameters
        ----------
        cnn_feats : np.ndarray of shape (N, 512)
        fft_feats : np.ndarray of shape (N, 70)
        dwt_feats : np.ndarray of shape (N, 28)
        """
        cnn_arr = np.nan_to_num(np.asarray(cnn_feats, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
        fft_arr = np.nan_to_num(np.asarray(fft_feats, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
        dwt_arr = np.nan_to_num(np.asarray(dwt_feats, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)

        self.cnn_mean = np.mean(cnn_arr, axis=0)
        self.cnn_std = np.std(cnn_arr, axis=0)

        self.fft_mean = np.mean(fft_arr, axis=0)
        self.fft_std = np.std(fft_arr, axis=0)

        self.dwt_mean = np.mean(dwt_arr, axis=0)
        self.dwt_std = np.std(dwt_arr, axis=0)

        self.is_fitted = True
        return self

    def transform(
        self,
        cnn_feats: np.ndarray,
        fft_feats: np.ndarray,
        dwt_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize each branch and concatenate into fused feature array.

        Returns
        -------
        Tuple of (norm_cnn, norm_fft, norm_dwt, fused_vector)
        fused_vector has shape (N, 610) or (610,) for 1D input.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer must be fitted on training data before transform.")

        is_1d = (cnn_feats.ndim == 1)
        cnn = np.atleast_2d(np.asarray(cnn_feats, dtype=np.float32))
        fft = np.atleast_2d(np.asarray(fft_feats, dtype=np.float32))
        dwt = np.atleast_2d(np.asarray(dwt_feats, dtype=np.float32))

        cnn = np.nan_to_num(cnn, nan=0.0, posinf=1e4, neginf=-1e4)
        fft = np.nan_to_num(fft, nan=0.0, posinf=1e4, neginf=-1e4)
        dwt = np.nan_to_num(dwt, nan=0.0, posinf=1e4, neginf=-1e4)

        norm_cnn = (cnn - self.cnn_mean) / (self.cnn_std + self.eps)
        norm_fft = (fft - self.fft_mean) / (self.fft_std + self.eps)
        norm_dwt = (dwt - self.dwt_mean) / (self.dwt_std + self.eps)

        # Clip extreme outliers to prevent gradient explosion
        norm_cnn = np.clip(norm_cnn, -10.0, 10.0)
        norm_fft = np.clip(norm_fft, -10.0, 10.0)
        norm_dwt = np.clip(norm_dwt, -10.0, 10.0)

        fused = np.concatenate([norm_cnn, norm_fft, norm_dwt], axis=1)

        if is_1d:
            return norm_cnn[0], norm_fft[0], norm_dwt[0], fused[0]
        return norm_cnn, norm_fft, norm_dwt, fused

    def to_dict(self) -> Dict:
        """Serialize parameters to JSON-serializable dictionary."""
        return {
            "eps": self.eps,
            "is_fitted": self.is_fitted,
            "cnn_mean": self.cnn_mean.tolist() if self.cnn_mean is not None else None,
            "cnn_std": self.cnn_std.tolist() if self.cnn_std is not None else None,
            "fft_mean": self.fft_mean.tolist() if self.fft_mean is not None else None,
            "fft_std": self.fft_std.tolist() if self.fft_std is not None else None,
            "dwt_mean": self.dwt_mean.tolist() if self.dwt_mean is not None else None,
            "dwt_std": self.dwt_std.tolist() if self.dwt_std is not None else None,
            "feature_dims": {
                "cnn": CNN_FEATURE_DIM,
                "fft": FFT_FEATURE_DIM,
                "dwt": DWT_FEATURE_DIM,
                "fusion": FUSION_FEATURE_DIM,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureNormalizer":
        """Reconstruct normalizer from dictionary."""
        norm = cls(eps=data.get("eps", 1e-7))
        norm.is_fitted = data.get("is_fitted", False)
        if data.get("cnn_mean") is not None:
            norm.cnn_mean = np.array(data["cnn_mean"], dtype=np.float32)
            norm.cnn_std = np.array(data["cnn_std"], dtype=np.float32)
        if data.get("fft_mean") is not None:
            norm.fft_mean = np.array(data["fft_mean"], dtype=np.float32)
            norm.fft_std = np.array(data["fft_std"], dtype=np.float32)
        if data.get("dwt_mean") is not None:
            norm.dwt_mean = np.array(data["dwt_mean"], dtype=np.float32)
            norm.dwt_std = np.array(data["dwt_std"], dtype=np.float32)
        return norm

    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalization parameters to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved FeatureNormalizer parameters to: %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureNormalizer":
        """Load normalization parameters from JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MultimodalFeatureExtractor:
    """
    Extracts CNN spatial (512-dim), FFT frequency (70-dim), and DWT wavelet (28-dim)
    features from image inputs.
    """

    def __init__(
        self,
        cnn_model: Optional[SpatialResNet18] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.cnn_transform = get_spatial_transforms(is_train=False, input_size=(224, 224))

        if cnn_model is not None:
            self.cnn_model = cnn_model
        else:
            self.cnn_model = SpatialResNet18(pretrained=True)
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

    def _ensure_pil(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Standardize input to RGB PIL Image."""
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            return Image.open(path).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
            elif img_input.shape[2] == 3:
                # Assume BGR if from cv2, else standard uint8
                rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported numpy image shape: {img_input.shape}")
            return Image.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

    def extract_cnn_features(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract 512-dimensional spatial embedding from ResNet18 avgpool layer.

        Returns
        -------
        np.ndarray of shape (512,), dtype float32.
        """
        tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_tensor = self.cnn_model.extract_features(tensor)  # (1, 512)
            feat_arr = feat_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        if feat_arr.shape != (CNN_FEATURE_DIM,):
            raise ValueError(f"Expected CNN feature shape ({CNN_FEATURE_DIM},), got {feat_arr.shape}")
        return feat_arr

    def extract_fft_features(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract 70-dimensional FFT frequency-domain feature vector.

        Returns
        -------
        np.ndarray of shape (70,), dtype float32.
        """
        rgb_arr = np.array(pil_image)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        fft_res = run_fft_analysis(bgr_arr, generate_visuals=False)
        vec = fft_res.features.to_vector().astype(np.float32)

        if vec.shape != (FFT_FEATURE_DIM,):
            raise ValueError(f"Expected FFT feature shape ({FFT_FEATURE_DIM},), got {vec.shape}")
        return vec

    def extract_dwt_features(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract 28-dimensional DWT wavelet feature vector.

        Returns
        -------
        np.ndarray of shape (28,), dtype float32.
        """
        rgb_arr = np.array(pil_image)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        dwt_res = run_dwt_analysis(bgr_arr, wavelet="haar", generate_visuals=False)
        vec = dwt_res.features.to_vector().astype(np.float32)

        if vec.shape != (DWT_FEATURE_DIM,):
            raise ValueError(f"Expected DWT feature shape ({DWT_FEATURE_DIM},), got {vec.shape}")
        return vec

    def extract_all(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, np.ndarray]:
        """
        Extract all three forensic feature vectors (CNN, FFT, DWT).

        Returns
        -------
        Dict with keys:
          - "cnn": (512,)
          - "fft": (70,)
          - "dwt": (28,)
          - "raw_fused": (610,)
        """
        pil_img = self._ensure_pil(image_input)

        cnn_feat = self.extract_cnn_features(pil_img)
        fft_feat = self.extract_fft_features(pil_img)
        dwt_feat = self.extract_dwt_features(pil_img)

        raw_fused = np.concatenate([cnn_feat, fft_feat, dwt_feat], axis=0)

        return {
            "cnn": cnn_feat,
            "fft": fft_feat,
            "dwt": dwt_feat,
            "raw_fused": raw_fused,
        }
