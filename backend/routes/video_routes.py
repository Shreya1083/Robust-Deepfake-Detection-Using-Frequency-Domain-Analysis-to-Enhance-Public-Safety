"""
Video Detection Routes (Phase 5 placeholder)
=============================================
Video detection is not yet implemented; endpoint returns a clear status.
"""

from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/detect", tags=["Video Detection"])


@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Video detection placeholder — Phase 5.
    """
    return {
        "model_ready": False,
        "prediction": None,
        "error": "Video detection will be implemented in Phase 5.",
        "filename": file.filename,
        "phase": "Phase 5 — coming soon",
    }
