"""
Backend Configuration Module
=============================
Loads configuration settings from environment variables with sensible defaults.
Supports .env files via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root or backend directory if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# MongoDB Settings
# ---------------------------------------------------------------------------
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "deepfake_detection_db")

# ---------------------------------------------------------------------------
# JWT Authentication Settings
# ---------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "deepguard-secure-auth-secret-key-32-bytes-minimum-robust-deepfake-detection",
)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Cookie Settings
COOKIE_NAME = "access_token"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # True in production HTTPS
COOKIE_SAMESITE = "lax"
COOKIE_MAX_AGE = ACCESS_TOKEN_EXPIRE_MINUTES * 60
