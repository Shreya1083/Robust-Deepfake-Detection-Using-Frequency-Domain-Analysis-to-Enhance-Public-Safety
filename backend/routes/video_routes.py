"""
Video Detection Routes
======================
Endpoint for video-based deepfake detection.
Accurately reflects current system implementation status without simulated results.
"""

from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/detect", tags=["Video Detection"])


@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Video deepfake detection endpoint.
    Reports operational readiness status transparently.
    """
    return {
        "model_ready": False,
        "prediction": None,
        "error": "Video deepfake detection is currently in development. Temporal frame analysis will be enabled in an upcoming system update.",
        "filename": file.filename,
        "status": "in_development",
    }
