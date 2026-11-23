"""
FastAPI Backend for Robust Deepfake Detection
=============================================
Serves CNN-based deepfake detection via REST API.
Reuses existing SpatialDetector from src/detector.py.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so src/ and models/ are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.image_routes import router as image_router
from backend.routes.video_routes import router as video_router
from backend.routes.frequency_routes import router as frequency_router
from backend.routes.auth_routes import router as auth_router
from backend.services.model_service import model_service
from backend.database import db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Robust Deepfake Detection API",
    description="CNN-based spatial deepfake detection using ResNet18.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server on :5173
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                    "http://localhost:5174", "http://127.0.0.1:5174",
                    "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth_router,      prefix="/api")
app.include_router(image_router,     prefix="/api")
app.include_router(video_router,     prefix="/api")
app.include_router(frequency_router, prefix="/api")


# ---------------------------------------------------------------------------
# Health / Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok", "service": "deepfake-detection-api"}


@app.get("/api/status")
async def model_status():
    """Return model readiness and metadata."""
    return {
        "model_ready": model_service.is_ready,
        "checkpoint_path": str(model_service.checkpoint_path),
        "device": str(model_service.device),
        "checkpoint_meta": {
            k: v
            for k, v in (model_service.checkpoint_meta or {}).items()
            if k != "model_state_dict"
        } if model_service.checkpoint_meta else None,
    }


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup complete. Model ready: %s", model_service.is_ready)
    # Test MongoDB connectivity non-blockingly
    connected, err = await db_manager.check_connection()
    if connected:
        logger.info("MongoDB connected successfully. Ensuring indexes.")
        await db_manager.ensure_indexes()
    else:
        logger.warning("MongoDB not connected on startup: %s", err)


@app.on_event("shutdown")
async def shutdown_event():
    db_manager.close()
    logger.info("MongoDB connection closed.")
