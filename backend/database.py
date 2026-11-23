"""
Database Connection Manager
===========================
Provides asynchronous MongoDB connection pooling using motor.
Includes graceful error detection when MongoDB is offline or misconfigured.
"""

import asyncio
import logging
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from backend.config import MONGODB_URL, MONGODB_DB_NAME

logger = logging.getLogger(__name__)


def _is_server_reachable(url: str, timeout: float = 0.5) -> bool:
    """Fast non-blocking TCP socket check before attempting Motor handshake."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 27017
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            return True
    except Exception:
        return False


class DatabaseManager:
    """Manages the lifecycle of the AsyncIOMotorClient."""

    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._is_connected: bool = False
        self._is_mock: bool = False
        self._last_error: Optional[str] = None

    def connect(self, custom_client=None):
        """Initialize motor client with a short serverSelectionTimeoutMS for fast fallback."""
        if custom_client is not None:
            self._client = custom_client
            self._db = self._client[MONGODB_DB_NAME]
            self._is_connected = True
            self._is_mock = True
            return

        self._is_mock = False
        try:
            self._client = AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=1000,
                connectTimeoutMS=1000,
            )
            self._db = self._client[MONGODB_DB_NAME]
            logger.info("MongoDB client created for URL: %s", MONGODB_URL)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Failed to initialize MongoDB client: %s", exc)

    async def get_db(self) -> AsyncIOMotorDatabase:
        """Get the active database instance."""
        if self._db is None:
            self.connect()
        return self._db

    async def check_connection(self) -> Tuple[bool, Optional[str]]:
        """Ping MongoDB with strict timeout to verify active connectivity."""
        if getattr(self, "_is_mock", False):
            return True, None

        # 1. Fast socket pre-check (returns in <1ms if offline)
        if not _is_server_reachable(MONGODB_URL, timeout=0.3):
            self._is_connected = False
            msg = (
                f"Could not connect to MongoDB at '{MONGODB_URL}'. "
                "Ensure MongoDB service is running (e.g. 'net start MongoDB' or mongod), "
                "or update MONGODB_URL in your .env file."
            )
            self._last_error = msg
            return False, msg

        if self._client is None:
            self.connect()
        try:
            # Quick ping strictly bounded by 1.0 second timeout
            await asyncio.wait_for(self._client.admin.command("ping"), timeout=1.0)
            self._is_connected = True
            self._last_error = None
            return True, None
        except Exception as exc:
            self._is_connected = False
            self._last_error = str(exc)
            return False, str(exc)

    async def ensure_indexes(self):
        """Create indexes on users and detection_history collections."""
        if getattr(self, "_is_mock", False):
            return
        try:
            db = await self.get_db()
            users = db["users"]
            await asyncio.wait_for(users.create_index("email", unique=True), timeout=1.0)
            await asyncio.wait_for(users.create_index("username", unique=True), timeout=1.0)
            
            history = db["detection_history"]
            await asyncio.wait_for(history.create_index([("user_id", 1), ("created_at", -1)]), timeout=1.0)
            logger.info("MongoDB user and detection_history indexes ensured.")
        except Exception as exc:
            logger.warning("Could not ensure MongoDB indexes: %s", exc)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._is_connected = False


# Module singleton
db_manager = DatabaseManager()
