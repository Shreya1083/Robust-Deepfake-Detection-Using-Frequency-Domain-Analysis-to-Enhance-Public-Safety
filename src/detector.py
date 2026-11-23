"""
Spatial Deepfake Detector Module
================================
Inference engine that loads the trained ResNet18 model and performs
spatial-domain deepfake detection on input images.

Key Features:
- Automatic face detection and padded cropping via FaceExtractor.
- Exact training preprocessing (224x224, ImageNet normalization).
- Deterministic inference using model.eval() and torch.no_grad().
- Valid class probabilities (summing to 1.0) and confidence scoring.
- Safe fallback when checkpoint is not yet trained.
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

logger = logging.getLogger(__name__)


class SpatialDetector:
    """Performs CNN-based spatial deepfake inference on single images."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.checkpoint_path = Path(checkpoint_path or (CHECKPOINTS_DIR / "resnet18_best.pth"))
        self.model: Optional[SpatialResNet18] = None
        self.checkpoint_meta: Optional[Dict] = None
        self.face_extractor = FaceExtractor()
        self.transform = get_spatial_transforms(is_train=False, input_size=(224, 224))
        self.is_ready = False

        self._load_model()

    def _load_model(self) -> bool:
        """Attempt to load trained model checkpoint."""
        if not self.checkpoint_path.exists():
            logger.info("Spatial model checkpoint not found at: %s", self.checkpoint_path)
            self.is_ready = False
            return False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model = SpatialResNet18(
                pretrained=False,
                num_classes=checkpoint.get("num_classes", 2),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.checkpoint_meta = checkpoint
            self.is_ready = True
            logger.info("SpatialDetector successfully initialized from: %s", self.checkpoint_path)
            return True
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", self.checkpoint_path, e)
            self.is_ready = False
            return False

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict:
        """
        Run inference on an uploaded image.

        Parameters
        ----------
        image_input : file path, numpy array (BGR), or PIL Image.

        Returns
        -------
        Dict with prediction results:
            - is_ready: bool
            - prediction: "REAL" or "DEEPFAKE"
            - label: int (0 or 1)
            - confidence: float (0.0 to 1.0)
            - real_prob: float
            - deepfake_prob: float
            - face_detected: bool
            - cropped_face: Optional[np.ndarray] (RGB for display)
            - device: str
            - error: Optional[str]
        """
        if not self.is_ready or self.model is None:
            return {
                "is_ready": False,
                "error": f"Model checkpoint not found at {self.checkpoint_path}. Please train the model first.",
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
            # Fall back to center region / whole image if no face detected
            face_rgb = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

        # 3. Apply ResNet18 Preprocessing
        tensor = self.transform(pil_image)
        batch = tensor.unsqueeze(0).to(self.device)  # Shape (1, 3, 224, 224)

        # 4. Model Inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        real_prob = float(probs[0])
        deepfake_prob = float(probs[1])

        label = int(np.argmax(probs))
        prediction = CLASS_LABELS.get(label, "UNKNOWN")
        confidence = float(probs[label])

        return {
            "is_ready": True,
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "real_prob": round(real_prob, 4),
            "deepfake_prob": round(deepfake_prob, 4),
            "face_detected": face_detected,
            "cropped_face": face_rgb if face_detected else None,
            "device": str(self.device),
            "error": None,
        }
