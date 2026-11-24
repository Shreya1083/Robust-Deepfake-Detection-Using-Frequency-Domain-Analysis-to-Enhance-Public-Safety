"""
Frequency Analysis Routes (FFT & DWT)
=====================================
POST /api/frequency/fft — Accept uploaded image, run 2D FFT, return spectrum.
POST /api/frequency/dwt — Accept uploaded image, run 2D DWT, return sub-bands & features.
"""

import asyncio
import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.frequency import run_fft_analysis
from src.wavelet import run_dwt_analysis

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
    read_res = file.read()
    raw_bytes = await read_res if asyncio.iscoroutine(read_res) else read_res
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


@router.post("/dwt")
async def dwt_analysis(file: UploadFile = File(...)):
    """
    Upload a face image and receive real 2D Discrete Wavelet Transform (DWT) analysis.

    Returns
    -------
    JSON with:
    - success       : bool
    - features      : dict with LL, LH, HL, HH metrics, energy distribution, total energy, entropy
    - original_b64  : base64 JPEG — resized grayscale input (224x224)
    - ll_b64        : base64 JPEG — LL approximation sub-band
    - lh_b64        : base64 JPEG — LH horizontal detail sub-band
    - hl_b64        : base64 JPEG — HL vertical detail sub-band
    - hh_b64        : base64 JPEG — HH diagonal detail sub-band
    - combined_b64  : base64 JPEG — 2x2 combined quad-grid
    - subband_shape : [H, W] of sub-bands (e.g., [112, 112])
    - image_size    : [H, W] of DWT input
    - wavelet_type  : name of wavelet family used (default: "haar")
    - filename      : uploaded filename
    - error         : str | null
    """
    # --- Validate content type -------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG, PNG or BMP.",
        )

    # --- Read bytes -------------------------------------------------------
    read_res = file.read()
    raw_bytes = await read_res if asyncio.iscoroutine(read_res) else read_res
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

    # --- Run DWT analysis -------------------------------------------------
    logger.info(
        "Running DWT analysis on '%s' (size=%s, %d bytes)",
        file.filename, pil_image.size, len(raw_bytes),
    )
    result = run_dwt_analysis(pil_image, generate_visuals=True, wavelet="haar")

    if not result.success:
        return {
            "success": False,
            "features": None,
            "original_b64": None,
            "ll_b64": None,
            "lh_b64": None,
            "hl_b64": None,
            "hh_b64": None,
            "combined_b64": None,
            "subband_shape": None,
            "image_size": None,
            "wavelet_type": "haar",
            "filename": file.filename,
            "error": result.error,
        }

    return {
        "success": True,
        "features": result.features.to_dict(),
        "original_b64": result.original_b64,
        "ll_b64": result.ll_b64,
        "lh_b64": result.lh_b64,
        "hl_b64": result.hl_b64,
        "hh_b64": result.hh_b64,
        "combined_b64": result.combined_b64,
        "subband_shape": list(result.subband_shape) if result.subband_shape else None,
        "image_size": list(result.image_size) if result.image_size else None,
        "wavelet_type": result.wavelet_type,
        "filename": file.filename,
        "error": None,
    }
