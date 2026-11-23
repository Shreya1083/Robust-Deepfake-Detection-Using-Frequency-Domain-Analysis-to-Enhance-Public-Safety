"""
Authentication Routes
=====================
REST API endpoints for user registration, login, logout, and profile retrieval.
Supports both HttpOnly cookies and Bearer Authorization headers.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, status
from pymongo.errors import DuplicateKeyError

from backend.config import (
    COOKIE_NAME,
    COOKIE_SECURE,
    COOKIE_SAMESITE,
    COOKIE_MAX_AGE,
    MONGODB_URL,
    MONGODB_DB_NAME,
)
from backend.database import db_manager
from backend.schemas.auth_schemas import (
    UserSignUp,
    UserLogin,
    UserResponse,
    AuthSuccessResponse,
    AuthStatusResponse,
)
from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# Helper: Extract Token from Cookie or Header
# ---------------------------------------------------------------------------

async def get_current_user(
    request: Request,
    access_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
    authorization: Optional[str] = Header(None),
) -> dict:
    """
    Dependency that resolves the authenticated user from:
    1. HttpOnly 'access_token' cookie
    2. 'Authorization: Bearer <token>' header
    """
    token = None
    if access_token:
        token = access_token
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid token. Please sign in again.",
        )

    user_id = payload["sub"]

    # Retrieve user from database
    is_connected, err = await db_manager.check_connection()
    if not is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error: {err}",
        )

    db = await db_manager.get_db()
    try:
        user = await db["users"].find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = await db["users"].find_one({"_id": user_id})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
        )

    return {
        "id": str(user["_id"]),
        "full_name": user.get("full_name", ""),
        "email": user.get("email", ""),
        "username": user.get("username", ""),
        "created_at": user.get("created_at", ""),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=AuthStatusResponse)
async def auth_status():
    """Report database connectivity status and configuration guidelines."""
    connected, error = await db_manager.check_connection()
    return AuthStatusResponse(
        database_connected=connected,
        database_url=MONGODB_URL,
        database_name=MONGODB_DB_NAME,
        error=error,
    )


@router.post("/signup", response_model=AuthSuccessResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserSignUp, response: Response):
    """Register a new user account with hashed password and unique email/username."""
    connected, err = await db_manager.check_connection()
    if not connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB is not available: {err}",
        )

    db = await db_manager.get_db()
    users = db["users"]

    # Check duplicate email or username
    existing_email = await users.find_one({"email": payload.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    existing_username = await users.find_one({"username": payload.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken. Please choose another.",
        )

    # Hash password & insert
    hashed = hash_password(payload.password)
    now_iso = datetime.now(timezone.utc).isoformat()

    doc = {
        "full_name": payload.full_name,
        "email": payload.email,
        "username": payload.username,
        "password_hash": hashed,
        "created_at": now_iso,
    }

    try:
        res = await users.insert_one(doc)
        user_id = str(res.inserted_id)
    except DuplicateKeyError as dke:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already in use.",
        ) from dke

    # Issue JWT access token
    token = create_access_token({"sub": user_id, "username": payload.username})

    # Set HttpOnly Cookie
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        path="/",
    )

    user_resp = UserResponse(
        id=user_id,
        full_name=payload.full_name,
        email=payload.email,
        username=payload.username,
        created_at=now_iso,
    )

    return AuthSuccessResponse(
        message="Account created successfully.",
        access_token=token,
        user=user_resp,
    )


@router.post("/login", response_model=AuthSuccessResponse)
async def login(payload: UserLogin, response: Response):
    """Authenticate via email or username + password. Sets HttpOnly JWT cookie."""
    connected, err = await db_manager.check_connection()
    if not connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB is not available: {err}",
        )

    db = await db_manager.get_db()
    users = db["users"]

    # Match by email or username
    user = await users.find_one({
        "$or": [
            {"email": payload.identifier},
            {"username": payload.identifier},
        ]
    })

    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password.",
        )

    user_id = str(user["_id"])
    token = create_access_token({"sub": user_id, "username": user.get("username")})

    # Set secure HttpOnly cookie
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        path="/",
    )

    user_resp = UserResponse(
        id=user_id,
        full_name=user.get("full_name", ""),
        email=user.get("email", ""),
        username=user.get("username", ""),
        created_at=user.get("created_at"),
    )

    return AuthSuccessResponse(
        message="Signed in successfully.",
        access_token=token,
        user=user_resp,
    )


@router.post("/logout")
async def logout(response: Response):
    """Sign out by clearing the HttpOnly access token cookie."""
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        samesite=COOKIE_SAMESITE,
    )
    return {"success": True, "message": "Signed out successfully."}


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    """Return currently authenticated user profile."""
    return UserResponse(**current_user)
