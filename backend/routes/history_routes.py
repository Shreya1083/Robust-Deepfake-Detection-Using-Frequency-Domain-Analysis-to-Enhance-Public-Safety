"""
Detection History Routes
========================
Provides REST API endpoints to retrieve, inspect, filter, and delete
user-specific detection records stored in MongoDB.
Each user can only access their own records.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.database import db_manager
from backend.routes.auth_routes import get_current_user
from backend.schemas.history_schemas import (
    DetectionHistoryItem,
    DetectionHistoryResponse,
    DetectionStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["Detection History"])


@router.get("", response_model=DetectionHistoryResponse)
async def list_history(
    search: Optional[str] = Query(None, description="Filter by filename substring"),
    media_type: Optional[str] = Query("all", description="'image', 'video', or 'all'"),
    prediction: Optional[str] = Query("all", description="'REAL', 'DEEPFAKE', or 'all'"),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List detection history records for the authenticated user with optional filtering."""
    db = await db_manager.get_db()
    collection = db["detection_history"]

    query = {"user_id": current_user["id"]}

    if isinstance(search, str) and search.strip():
        # Case-insensitive substring match on filename
        query["filename"] = {"$regex": search.strip(), "$options": "i"}

    if isinstance(media_type, str) and media_type.lower() in ("image", "video"):
        query["media_type"] = media_type.lower()

    if isinstance(prediction, str) and prediction.upper() in ("REAL", "DEEPFAKE"):
        query["prediction"] = prediction.upper()

    skip_val = int(skip) if isinstance(skip, (int, str)) and str(skip).isdigit() else 0
    limit_val = int(limit) if isinstance(limit, (int, str)) and str(limit).isdigit() else 20

    total = await collection.count_documents(query)

    cursor = collection.find(query).sort("created_at", -1).skip(skip_val).limit(limit_val)
    items = []
    async for doc in cursor:
        items.append(
            DetectionHistoryItem(
                id=str(doc["_id"]),
                user_id=doc.get("user_id", current_user["id"]),
                filename=doc.get("filename", "unknown_media"),
                media_type=doc.get("media_type", "image"),
                prediction=doc.get("prediction"),
                confidence=doc.get("confidence"),
                real_prob=doc.get("real_prob"),
                deepfake_prob=doc.get("deepfake_prob"),
                face_detected=doc.get("face_detected"),
                created_at=doc.get("created_at", ""),
                file_size_bytes=doc.get("file_size_bytes"),
                details=doc.get("details"),
            )
        )

    return DetectionHistoryResponse(success=True, total=total, items=items)


@router.get("/stats", response_model=DetectionStatsResponse)
async def get_history_stats(current_user: dict = Depends(get_current_user)):
    """Calculate aggregated detection summary metrics from actual stored user history."""
    db = await db_manager.get_db()
    collection = db["detection_history"]
    user_id = current_user["id"]

    total = await collection.count_documents({"user_id": user_id})
    images_count = await collection.count_documents({"user_id": user_id, "media_type": "image"})
    videos_count = await collection.count_documents({"user_id": user_id, "media_type": "video"})
    real_count = await collection.count_documents({"user_id": user_id, "prediction": "REAL"})
    deepfake_count = await collection.count_documents({"user_id": user_id, "prediction": "DEEPFAKE"})

    # Activity in last 7 days
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    recent_count = await collection.count_documents({
        "user_id": user_id,
        "created_at": {"$gte": seven_days_ago},
    })

    return DetectionStatsResponse(
        success=True,
        total_analyzed=total,
        images_analyzed=images_count,
        videos_analyzed=videos_count,
        real_count=real_count,
        deepfake_count=deepfake_count,
        recent_activity_count=recent_count,
    )


@router.get("/{history_id}", response_model=DetectionHistoryItem)
async def get_history_record(history_id: str, current_user: dict = Depends(get_current_user)):
    """Retrieve details of a specific detection record owned by the authenticated user."""
    db = await db_manager.get_db()
    collection = db["detection_history"]

    try:
        obj_id = ObjectId(history_id)
        doc = await collection.find_one({"_id": obj_id, "user_id": current_user["id"]})
    except Exception:
        doc = await collection.find_one({"_id": history_id, "user_id": current_user["id"]})

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection record not found or unauthorized.",
        )

    return DetectionHistoryItem(
        id=str(doc["_id"]),
        user_id=doc.get("user_id", current_user["id"]),
        filename=doc.get("filename", "unknown_media"),
        media_type=doc.get("media_type", "image"),
        prediction=doc.get("prediction"),
        confidence=doc.get("confidence"),
        real_prob=doc.get("real_prob"),
        deepfake_prob=doc.get("deepfake_prob"),
        face_detected=doc.get("face_detected"),
        created_at=doc.get("created_at", ""),
        file_size_bytes=doc.get("file_size_bytes"),
        details=doc.get("details"),
    )


@router.delete("/{history_id}")
async def delete_history_record(history_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a detection record owned by the authenticated user."""
    db = await db_manager.get_db()
    collection = db["detection_history"]

    try:
        obj_id = ObjectId(history_id)
        res = await collection.delete_one({"_id": obj_id, "user_id": current_user["id"]})
    except Exception:
        res = await collection.delete_one({"_id": history_id, "user_id": current_user["id"]})

    if res.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection record not found or unauthorized to delete.",
        )

    return {"success": True, "message": "Detection record deleted successfully."}
