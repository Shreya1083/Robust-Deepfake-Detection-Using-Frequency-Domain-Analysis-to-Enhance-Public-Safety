"""
FFT Frequency Analysis Route
=============================
POST /api/frequency/fft — Accept an uploaded image, run real 2D FFT analysis,
return numerical features + base64-encoded spectrum visualisations.
"""

import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.frequency import run_fft_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/frequency", tags=["Frequency Analysis"])

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/fft")
async def fft_analysis(file: UploadFile = File(...)):
    """
    Upload a face image and receive real FFT frequency analysis.

    Returns
    -------
    JSON with:
    - success          : bool
    - features         : dict of numerical FFT measurements
    - original_b64     : base64 JPEG — resized grayscale input (224×224)
    - magnitude_b64    : base64 JPEG — log-magnitude FFT spectrum (BONE colormap)
    - log_spectrum_b64 : base64 JPEG — colourised log spectrum (JET colormap)
    - image_size       : [H, W] of FFT input
    - filename         : uploaded filename
    - error            : str | null
    """
    # --- Validate content type -------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG, PNG or BMP.",
        )

    # --- Read bytes -------------------------------------------------------
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    # --- Decode to PIL Image (validate it's a real image) ----------------
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot decode image — file may be corrupted: {exc}",
        ) from exc

    # --- Run FFT analysis -------------------------------------------------
    logger.info(
        "Running FFT analysis on '%s' (size=%s, %d bytes)",
        file.filename, pil_image.size, len(raw_bytes),
    )
    result = run_fft_analysis(pil_image, generate_visuals=True)

    if not result.success:
        # Analysis-level failure (e.g. image too small)
        return {
            "success": False,
            "features": None,
            "original_b64": None,
            "magnitude_b64": None,
            "log_spectrum_b64": None,
            "image_size": None,
            "filename": file.filename,
            "error": result.error,
        }

    return {
        "success": True,
        "features": result.features.to_dict(),
        "original_b64": result.original_b64,
        "magnitude_b64": result.magnitude_b64,
        "log_spectrum_b64": result.log_spectrum_b64,
        "image_size": list(result.image_size) if result.image_size else None,
        "filename": file.filename,
        "error": None,
    }
