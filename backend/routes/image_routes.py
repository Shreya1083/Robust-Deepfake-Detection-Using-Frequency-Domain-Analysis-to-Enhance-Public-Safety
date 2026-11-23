"""
Image Detection Routes
======================
POST /api/detect/image — receive an uploaded image, run CNN inference,
return REAL/DEEPFAKE prediction with probabilities.
"""

import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Image Detection"])

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image and receive deepfake detection results.

    Returns
    -------
    JSON with:
    - prediction : "REAL" | "DEEPFAKE"
    - label      : 0 | 1
    - confidence : float
    - real_prob  : float
    - deepfake_prob : float
    - face_detected : bool
    - cropped_face_b64 : base64-encoded JPEG of detected face (or null)
    - model_ready : bool
    - device      : str
    - error       : str | null
    """
    # --- Validate content type ------------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG or PNG.",
        )

    # --- Read bytes -----------------------------------------------------------
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Decode to PIL Image --------------------------------------------------
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    # --- Run inference --------------------------------------------------------
    logger.info("Running inference on uploaded image (size=%s)", pil_image.size)
    result = model_service.predict_image(pil_image)

    # --- Encode cropped face as base64 for JSON transport --------------------
    cropped_b64 = None
    if result.get("cropped_face") is not None:
        try:
            face_arr = result["cropped_face"]  # numpy RGB
            face_pil = Image.fromarray(face_arr.astype(np.uint8))
            buf = io.BytesIO()
            face_pil.save(buf, format="JPEG", quality=90)
            cropped_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            cropped_b64 = None

    return {
        "model_ready": result.get("is_ready", False),
        "prediction": result.get("prediction"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "real_prob": result.get("real_prob"),
        "deepfake_prob": result.get("deepfake_prob"),
        "face_detected": result.get("face_detected"),
        "cropped_face_b64": cropped_b64,
        "device": result.get("device"),
        "error": result.get("error"),
        "filename": file.filename,
    }
