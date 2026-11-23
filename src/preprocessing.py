"""
Image Preprocessing Module
===========================
Prepares extracted face images for model training and inference.

Features:
- Format validation and corrupt image detection.
- High-quality bicubic / area resizing to standard input dimensions (256x256).
- Color-space conversion (BGR -> RGB).
- Floating-point normalization to [0.0, 1.0] or ImageNet standardization.
- Conversion to PyTorch (C, H, W) tensors.
- Safe error handling that never produces silent failures or corrupted tensors.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.config import IMAGE_SIZE

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePreprocessor:
    """Preprocesses cropped facial images for CNN and frequency-domain analysis."""

    def __init__(
        self,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        apply_imagenet_norm: bool = False,
    ):
        self.target_size = target_size  # (height, width)
        self.apply_imagenet_norm = apply_imagenet_norm

    @staticmethod
    def load_image(image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """
        Load an image from filepath or validate an existing numpy array.
        Returns BGR numpy array or None if corrupted/unreadable.
        """
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                logger.warning("Empty numpy image array provided.")
                return None
            return image_input

        path = Path(image_input)
        if not path.exists():
            logger.warning("Image file does not exist: %s", path)
            return None

        try:
            img = cv2.imread(str(path))
            if img is None or img.size == 0:
                logger.warning("cv2.imread failed to decode image: %s", path)
                return None
            return img
        except Exception as e:
            logger.warning("Exception loading image %s: %s", path, str(e))
            return None

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target_size (height, width) with appropriate interpolation.
        Uses INTER_AREA when downscaling, INTER_LINEAR when upscaling.
        """
        h_target, w_target = self.target_size
        h_src, w_src = image.shape[:2]

        if h_src == h_target and w_src == w_target:
            return image

        if h_target < h_src or w_target < w_src:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        return cv2.resize(image, (w_target, h_target), interpolation=interpolation)

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert uint8 [0, 255] RGB image to float32 [0.0, 1.0].
        Optionally apply ImageNet channel normalization.
        """
        img_float = image.astype(np.float32) / 255.0

        if self.apply_imagenet_norm:
            img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD

        return img_float

    def preprocess(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Full preprocessing pipeline:
        1. Load & validate image
        2. Resize to target size (256x256)
        3. Convert BGR -> RGB
        4. Normalize to float32 [0.0, 1.0]

        Returns:
            np.ndarray of shape (height, width, 3), float32, or None if failed.
        """
        img = self.load_image(image_input)
        if img is None:
            return None

        # Resize
        resized = self.resize(img)

        # Convert to RGB
        rgb = self.bgr_to_rgb(resized)

        # Normalize
        normalized = self.normalize(rgb)

        return normalized

    def preprocess_to_tensor(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Optional[torch.Tensor]:
        """
        Preprocess image and return as PyTorch tensor of shape (3, H, W).
        """
        processed = self.preprocess(image_input)
        if processed is None:
            return None

        # Transpose from (H, W, C) to (C, H, W)
        transposed = np.transpose(processed, (2, 0, 1))
        return torch.from_numpy(transposed)
