"""
Unified Deepfake Detector Module (Spatial CNN + Multimodal Fusion)
==================================================================
Inference engine that loads either:
1. Trained Multimodal Fusion Model (fusion_best.pth) combining CNN (512),
   FFT (70), and DWT (28) features into a unified 610-dim MLP classifier.
2. Baseline Spatial ResNet18 (resnet18_best.pth) as fallback.

Key Features:
- Automatic face detection and padded cropping via FaceExtractor.
- Exact training preprocessing (224x224, ImageNet normalization).
- Deterministic extraction: CNN embedding (512), FFT spectrum (70), DWT sub-bands (28).
- Branch-level z-score normalization using saved training statistics.
- Valid class probabilities (summing to 1.0) and confidence scoring.
- Comprehensive response schema supporting both single-domain and multimodal inspection.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

from models.spatial_model import SpatialResNet18, get_default_device
from src.config import CHECKPOINTS_DIR, CLASS_LABELS
from src.dataset_loader import get_spatial_transforms
from src.face_extraction import FaceExtractor
from src.feature_fusion import (
    CNN_FEATURE_DIM,
    DWT_FEATURE_DIM,
    FFT_FEATURE_DIM,
    FUSION_FEATURE_DIM,
    FeatureNormalizer,
    MultimodalFeatureExtractor,
    MultimodalFusionClassifier,
)

logger = logging.getLogger(__name__)


class SpatialDetector:
    """
    Unified deepfake detector that prioritizes the trained Multimodal Fusion Model
    (CNN + FFT + DWT) when available, while preserving seamless fallback to
    the CNN-only baseline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        fusion_checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()

        # Checkpoint paths
        self.fusion_checkpoint_path = Path(
            fusion_checkpoint_path or (CHECKPOINTS_DIR / "fusion_best.pth")
        )
        self.cnn_checkpoint_path = Path(
            checkpoint_path or (CHECKPOINTS_DIR / "resnet18_best.pth")
        )

        # Primary checkpoint pointer for backward-compatibility
        self.checkpoint_path = (
            self.fusion_checkpoint_path
            if self.fusion_checkpoint_path.exists()
            else self.cnn_checkpoint_path
        )

        # Models & preprocessors
        self.face_extractor = FaceExtractor()
        self.transform = get_spatial_transforms(is_train=False, input_size=(224, 224))

        self.model: Optional[SpatialResNet18] = None
        self.fusion_model: Optional[MultimodalFusionClassifier] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        self.feature_extractor: Optional[MultimodalFeatureExtractor] = None

        self.checkpoint_meta: Optional[Dict] = None
        self.model_type: str = "none"
        self.is_ready: bool = False
        self.is_fusion: bool = False

        self._load_models()

    def _load_models(self) -> bool:
        """Attempt to load models with priority on Multimodal Fusion."""
        # 1. First, always load or initialize the spatial CNN model (used by feature extractor or baseline)
        if self.cnn_checkpoint_path.exists():
            try:
                cnn_ckpt = torch.load(self.cnn_checkpoint_path, map_location=self.device)
                self.model = SpatialResNet18(
                    pretrained=False,
                    num_classes=cnn_ckpt.get("num_classes", 2),
                )
                self.model.load_state_dict(cnn_ckpt["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                self.checkpoint_meta = cnn_ckpt
                self.model_type = "spatial_resnet18"
                self.is_ready = True
                logger.info("Loaded baseline CNN checkpoint from: %s", self.cnn_checkpoint_path)
            except Exception as e:
                logger.warning("Could not load CNN checkpoint %s: %s", self.cnn_checkpoint_path, e)
                self.model = SpatialResNet18(pretrained=True).to(self.device)
                self.model.eval()
        else:
            try:
                self.model = SpatialResNet18(pretrained=True).to(self.device)
                self.model.eval()
            except Exception:
                self.model = SpatialResNet18(pretrained=False).to(self.device)
                self.model.eval()

        # Initialize multimodal feature extractor with CNN model
        self.feature_extractor = MultimodalFeatureExtractor(
            cnn_model=self.model,
            device=self.device,
        )

        # 2. Check if Multimodal Fusion checkpoint exists
        if self.fusion_checkpoint_path.exists():
            try:
                fusion_ckpt = torch.load(self.fusion_checkpoint_path, map_location=self.device)
                self.fusion_model = MultimodalFusionClassifier(
                    input_dim=fusion_ckpt.get("input_dim", FUSION_FEATURE_DIM),
                    hidden_dim1=256,
                    hidden_dim2=64,
                    num_classes=2,
                )
                self.fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
                self.fusion_model.to(self.device)
                self.fusion_model.eval()

                # Load normalizer
                norm_dict = fusion_ckpt.get("normalization")
                if norm_dict is not None:
                    self.normalizer = FeatureNormalizer.from_dict(norm_dict)
                else:
                    norm_file = self.fusion_checkpoint_path.parent / "fusion_normalization.json"
                    if norm_file.exists():
                        self.normalizer = FeatureNormalizer.load(norm_file)

                self.checkpoint_meta = fusion_ckpt
                self.checkpoint_path = self.fusion_checkpoint_path
                self.model_type = "multimodal_fusion"
                self.is_fusion = True
                self.is_ready = True
                logger.info("Successfully loaded Multimodal Fusion model from: %s", self.fusion_checkpoint_path)
                return True
            except Exception as e:
                logger.error("Failed to load fusion checkpoint %s: %s", self.fusion_checkpoint_path, e)

        # If fusion model is not loaded, verify baseline CNN is ready
        if self.model is not None and self.cnn_checkpoint_path.exists():
            self.model_type = "spatial_resnet18"
            self.is_fusion = False
            self.is_ready = True
            return True

        if self.model is not None:
            # Baseline evaluation mode without trained weights
            self.model_type = "spatial_resnet18_untrained"
            self.is_ready = True
            return True

        self.is_ready = False
        return False

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict:
        """
        Run deepfake inference on an uploaded image.
        Uses Multimodal Fusion (CNN + FFT + DWT) if available, with CNN baseline comparison.

        Parameters
        ----------
        image_input : file path, numpy array (BGR or RGB), or PIL Image.

        Returns
        -------
        Dict with complete forensic results.
        """
        if not self.is_ready:
            return {
                "is_ready": False,
                "error": f"Model checkpoints not found at {self.fusion_checkpoint_path} or {self.cnn_checkpoint_path}. Please train the model first.",
            }

        # 1. Standardize input to numpy BGR array
        np_frame: Optional[np.ndarray] = None

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {"is_ready": True, "error": f"File does not exist: {path}"}
            np_frame = cv2.imread(str(path))
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            np_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                np_frame = None
            elif image_input.ndim == 2:
                np_frame = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                np_frame = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                np_frame = image_input

        if np_frame is None or np_frame.size == 0:
            return {"is_ready": True, "error": "Invalid or empty image provided."}

        # 2. Face Detection & Padded Cropping
        face_result = self.face_extractor.extract_face(np_frame)
        face_detected = face_result.success and face_result.face_image is not None

        if face_detected:
            face_bgr = face_result.face_image
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
        else:
            face_rgb = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

        # Compute CNN baseline prediction
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_logits = self.model(tensor)
            cnn_probs = torch.softmax(cnn_logits, dim=1).squeeze(0).cpu().numpy()

        cnn_real = float(cnn_probs[0])
        cnn_fake = float(cnn_probs[1])
        cnn_pred_label = int(np.argmax(cnn_probs))
        cnn_pred = CLASS_LABELS.get(cnn_pred_label, "UNKNOWN")
        cnn_conf = float(cnn_probs[cnn_pred_label])

        baseline_cnn = {
            "prediction": cnn_pred,
            "label": cnn_pred_label,
            "confidence": round(cnn_conf, 4),
            "real_prob": round(cnn_real, 4),
            "deepfake_prob": round(cnn_fake, 4),
        }

        # 3. Multimodal Fusion Inference if model & normalizer available
        if self.is_fusion and self.fusion_model is not None and self.normalizer is not None:
            try:
                # Extract all 3 feature representations
                features = self.feature_extractor.extract_all(pil_image)
                cnn_feat = features["cnn"]
                fft_feat = features["fft"]
                dwt_feat = features["dwt"]

                # Apply training-set normalization
                _, _, _, fused_vector = self.normalizer.transform(cnn_feat, fft_feat, dwt_feat)
                fused_tensor = torch.from_numpy(fused_vector).unsqueeze(0).float().to(self.device)

                # Forward pass through fusion classifier MLP
                with torch.no_grad():
                    fusion_logits = self.fusion_model(fused_tensor)
                    fusion_probs = torch.softmax(fusion_logits, dim=1).squeeze(0).cpu().numpy()

                real_prob = float(fusion_probs[0])
                deepfake_prob = float(fusion_probs[1])
                label = int(np.argmax(fusion_probs))
                prediction = CLASS_LABELS.get(label, "UNKNOWN")
                confidence = float(fusion_probs[label])

                return {
                    "is_ready": True,
                    "model_type": "multimodal_fusion",
                    "is_fusion": True,
                    "prediction": prediction,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "real_prob": round(real_prob, 4),
                    "deepfake_prob": round(deepfake_prob, 4),
                    "face_detected": face_detected,
                    "cropped_face": face_rgb if face_detected else None,
                    "feature_dims": {
                        "cnn": CNN_FEATURE_DIM,
                        "fft": FFT_FEATURE_DIM,
                        "dwt": DWT_FEATURE_DIM,
                        "total": FUSION_FEATURE_DIM,
                    },
                    "multimodal_branches": {
                        "cnn": {
                            "status": "active",
                            "name": "CNN Spatial Features",
                            "dim": CNN_FEATURE_DIM,
                            "desc": "ResNet18 penultimate average pooling embedding",
                        },
                        "fft": {
                            "status": "active",
                            "name": "FFT Frequency Features",
                            "dim": FFT_FEATURE_DIM,
                            "desc": "Spectral summary statistics + 64 radial frequency bins",
                        },
                        "dwt": {
                            "status": "active",
                            "name": "DWT Wavelet Features",
                            "dim": DWT_FEATURE_DIM,
                            "desc": "Haar wavelet sub-bands (LL, LH, HL, HH) & energy ratios",
                        },
                    },
                    "baseline_cnn": baseline_cnn,
                    "device": str(self.device),
                    "error": None,
                }
            except Exception as e:
                logger.error("Error during multimodal fusion inference: %s, falling back to CNN", e)

        # Fallback to CNN spatial prediction
        return {
            "is_ready": True,
            "model_type": self.model_type,
            "is_fusion": False,
            "prediction": cnn_pred,
            "label": cnn_pred_label,
            "confidence": round(cnn_conf, 4),
            "real_prob": round(cnn_real, 4),
            "deepfake_prob": round(cnn_fake, 4),
            "face_detected": face_detected,
            "cropped_face": face_rgb if face_detected else None,
            "feature_dims": {"cnn": CNN_FEATURE_DIM, "total": CNN_FEATURE_DIM},
            "multimodal_branches": {
                "cnn": {
                    "status": "active",
                    "name": "CNN Spatial Features",
                    "dim": CNN_FEATURE_DIM,
                    "desc": "ResNet18 penultimate average pooling embedding",
                },
                "fft": {"status": "inactive", "name": "FFT Frequency Features", "dim": 0},
                "dwt": {"status": "inactive", "name": "DWT Wavelet Features", "dim": 0},
            },
            "baseline_cnn": baseline_cnn,
            "device": str(self.device),
            "error": None,
        }
