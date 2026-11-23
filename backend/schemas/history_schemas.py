"""
Pydantic Schemas for Detection History
======================================
Schemas for storing, querying, and serializing user detection history records.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DetectionHistoryItem(BaseModel):
    """Single detection history record returned to user."""
    id: str
    user_id: str
    filename: str
    media_type: str = Field(..., description="'image' or 'video'")
    prediction: Optional[str] = Field(None, description="'REAL', 'DEEPFAKE', or None")
    confidence: Optional[float] = None
    real_prob: Optional[float] = None
    deepfake_prob: Optional[float] = None
    face_detected: Optional[bool] = None
    created_at: str
    file_size_bytes: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class DetectionHistoryResponse(BaseModel):
    """Paginated list of detection history records."""
    success: bool = True
    total: int
    items: List[DetectionHistoryItem]


class DetectionStatsResponse(BaseModel):
    """Summary metrics of user's past detection activity."""
    success: bool = True
    total_analyzed: int
    images_analyzed: int
    videos_analyzed: int
    real_count: int
    deepfake_count: int
    recent_activity_count: int  # Scans in last 7 days
