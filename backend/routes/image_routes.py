"""
Image Detection Routes
======================
POST /api/detect/image — receive an uploaded image, run ResNet18 CNN inference,
Grad-CAM spatial feature analysis, FFT frequency decomposition, and DWT wavelets.
Automatically persists detection result in user's MongoDB history if authenticated.
"""

import base64
import io
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from backend.database import db_manager
from backend.routes.auth_routes import get_current_user_optional
from backend.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Image Detection"])

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Upload an image and receive full forensic deepfake detection results:
    - Final Prediction (REAL vs DEEPFAKE) & Calibrated Probabilities
    - Spatial Feature Analysis (CNN – Grad-CAM heatmap & overlay)
    - FFT Frequency Domain Analysis (magnitude spectrum & radial profile)
    - DWT Wavelet Sub-band Analysis
    - Holistic Analytical Summary
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
    logger.info("Running forensic inference on uploaded image '%s' (size=%s)", file.filename, pil_image.size)
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

    spatial_analysis = result.get("spatial_analysis")
    frequency_analysis = result.get("frequency_analysis")
    dwt_analysis = result.get("dwt_analysis")
    analysis_summary = result.get("analysis_summary")

    # --- Persist to MongoDB history if authenticated -------------------------
    history_id = None
    if current_user and result.get("prediction"):
        try:
            connected, _ = await db_manager.check_connection()
            if connected:
                db = await db_manager.get_db()
                doc = {
                    "user_id": current_user["id"],
                    "filename": file.filename or "uploaded_image.jpg",
                    "media_type": "image",
                    "prediction": result.get("prediction"),
                    "confidence": result.get("confidence"),
                    "real_prob": result.get("real_prob"),
                    "deepfake_prob": result.get("deepfake_prob"),
                    "face_detected": result.get("face_detected"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "file_size_bytes": len(raw_bytes),
                    "details": {
                        "model_ready": result.get("is_ready", False),
                        "model_type": result.get("model_type", "spatial_resnet18"),
                        "device": str(result.get("device")),
                        "label": result.get("label"),
                        "salient_regions": spatial_analysis.get("salient_regions") if spatial_analysis else None,
                        "fft_high_energy_ratio": (
                            frequency_analysis.get("metrics", {}).get("high_energy_ratio")
                            if frequency_analysis and frequency_analysis.get("metrics")
                            else None
                        ),
                        "analysis_summary": analysis_summary,
                    },
                }
                insert_res = await db["detection_history"].insert_one(doc)
                history_id = str(insert_res.inserted_id)
                logger.info("Saved image detection record %s for user %s", history_id, current_user['id'])
        except Exception as exc:
            logger.warning("Could not persist image detection history: %s", exc)

    return {
        "model_ready": result.get("is_ready", False),
        "model_type": result.get("model_type", "spatial_resnet18"),
        "prediction": result.get("prediction"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "real_prob": result.get("real_prob"),
        "deepfake_prob": result.get("deepfake_prob"),
        "face_detected": result.get("face_detected"),
        "cropped_face_b64": cropped_b64,
        "spatial_analysis": spatial_analysis,
        "frequency_analysis": frequency_analysis,
        "dwt_analysis": dwt_analysis,
        "analysis_summary": analysis_summary,
        "device": result.get("device"),
        "error": result.get("error"),
        "filename": file.filename,
        "history_id": history_id,
    }
