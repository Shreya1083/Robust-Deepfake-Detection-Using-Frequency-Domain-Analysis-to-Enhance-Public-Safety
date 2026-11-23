"""
Pydantic Schemas for Authentication
===================================
Input validation and response serialization schemas for Sign In, Sign Up, and User profiles.
"""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class UserSignUp(BaseModel):
    """Registration request payload."""
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name of user")
    email: str = Field(..., description="Valid email address")
    username: str = Field(..., min_length=3, max_length=30, description="Unique username")
    password: str = Field(..., min_length=6, max_length=128, description="Account password")
    confirm_password: str = Field(..., min_length=6, max_length=128, description="Password confirmation")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, clean):
            raise ValueError("Invalid email address format.")
        return clean

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[a-zA-Z0-9_.-]+$", clean):
            raise ValueError("Username may only contain letters, numbers, underscores, dots, or hyphens.")
        return clean

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 2:
            raise ValueError("Full name must be at least 2 characters long.")
        return clean

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return v


class UserLogin(BaseModel):
    """Login request payload."""
    identifier: str = Field(..., min_length=2, description="Email address or username")
    password: str = Field(..., min_length=1, description="Account password")

    @field_validator("identifier")
    @classmethod
    def clean_identifier(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("Email or username is required.")
        return clean


class UserResponse(BaseModel):
    """User profile response (safe, excluding password hash)."""
    id: str
    full_name: str
    email: str
    username: str
    created_at: Optional[str] = None


class AuthSuccessResponse(BaseModel):
    """Successful authentication response payload."""
    success: bool = True
    message: str
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class AuthStatusResponse(BaseModel):
    """MongoDB and auth system connectivity report."""
    database_connected: bool
    database_url: str
    database_name: str
    error: Optional[str] = None
