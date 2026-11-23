"""
Model Service — Singleton wrapper around SpatialDetector.
Loaded once at startup; all routes share the same instance.
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import SpatialDetector

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton that initialises SpatialDetector once and exposes it."""

    def __init__(self):
        logger.info("Initializing ModelService …")
        self._detector = SpatialDetector()
        logger.info("ModelService ready=%s", self._detector.is_ready)

    # Proxy commonly-needed attributes so callers can inspect without
    # reaching into the private detector object.
    @property
    def is_ready(self) -> bool:
        return self._detector.is_ready

    @property
    def checkpoint_path(self) -> Path:
        return self._detector.checkpoint_path

    @property
    def device(self):
        return self._detector.device

    @property
    def checkpoint_meta(self):
        return self._detector.checkpoint_meta

    def predict_image(self, image_input) -> dict:
        return self._detector.predict_image(image_input)


# Single module-level instance — imported by all routes.
model_service = ModelService()
