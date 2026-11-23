"""
Phase 3B Tests — User Authentication & MongoDB Integration
===========================================================
Tests all 8 required authentication scenarios:
1. New user registration
2. Duplicate email rejection (409)
3. Duplicate username rejection (409)
4. Successful login (with email and with username)
5. Incorrect password rejection (401)
6. Protected user profile access (/api/auth/me) with token
7. Logout (cookie cleared)
8. Accessing protected endpoint without token / after logout (401)

Run:
    .venv\\Scripts\\python.exe -m unittest tests/test_phase3b_auth.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bson import ObjectId
from fastapi import HTTPException, Response
from pymongo.errors import DuplicateKeyError

from backend.database import db_manager
from backend.routes.auth_routes import signup, login, logout, get_current_user, auth_status
from backend.schemas.auth_schemas import UserSignUp, UserLogin
from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)


# ---------------------------------------------------------------------------
# Pure Python Async In-Memory MongoDB Mock for Tests
# ---------------------------------------------------------------------------

class MockCollection:
    def __init__(self):
        self.docs = []
        self.indexes = []

    async def create_index(self, key, unique=False):
        self.indexes.append((key, unique))

    async def find_one(self, query):
        for doc in self.docs:
            match = True
            if "$or" in query:
                match = any(
                    all(doc.get(k) == v for k, v in subquery.items())
                    for subquery in query["$or"]
                )
            else:
                for k, v in query.items():
                    if doc.get(k) != v:
                        match = False
                        break
            if match:
                return doc.copy()
        return None

    async def insert_one(self, doc):
        for field, unique in self.indexes:
            if unique and field in doc:
                if any(existing.get(field) == doc[field] for existing in self.docs):
                    raise DuplicateKeyError(f"Duplicate key error on index: {field}")
        d = doc.copy()
        if "_id" not in d:
            d["_id"] = ObjectId()
        self.docs.append(d)

        class InsertResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        return InsertResult(d["_id"])


class MockDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]


class MockClient:
    def __init__(self):
        self.databases = {}

    def __getitem__(self, name):
        if name not in self.databases:
            self.databases[name] = MockDatabase()
        return self.databases[name]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class TestAuthSecurityPrimitives(unittest.TestCase):
    """Test bcrypt hashing and JWT token cryptographic operations."""

    def test_bcrypt_hashing_and_verification(self):
        pwd = "SecretPassword123!"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(hashed.startswith("$2b$") or hashed.startswith("$2a$"))
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword123!", hashed))

    def test_jwt_token_creation_and_decoding(self):
        claims = {"sub": "user_id_12345", "username": "alice"}
        token = create_access_token(claims)
        self.assertIsInstance(token, str)
        decoded = decode_access_token(token)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "user_id_12345")
        self.assertEqual(decoded["username"], "alice")

    def test_invalid_jwt_decoding_returns_none(self):
        self.assertIsNone(decode_access_token("invalid.jwt.token"))


class TestAuthenticationEndpoints(unittest.IsolatedAsyncioTestCase):
    """Integration test suite testing full FastAPI auth endpoints with in-memory async mock."""

    async def asyncSetUp(self):
        self.mock_client = MockClient()
        db_manager.connect(custom_client=self.mock_client)
        # Ensure collection indexes on mock
        users_col = self.mock_client["deepfake_detector"]["users"]
        await users_col.create_index("email", unique=True)
        await users_col.create_index("username", unique=True)

    async def asyncTearDown(self):
        db_manager.close()

    async def test_1_new_user_registration(self):
        """Scenario 1: New user registration with valid details."""
        payload = UserSignUp(
            full_name="Alice Smith",
            email="alice@example.com",
            username="alicesmith",
            password="Password123!",
            confirm_password="Password123!",
        )
        resp = Response()
        res = await signup(payload, resp)

        self.assertTrue(res.success)
        self.assertEqual(res.user.email, "alice@example.com")
        self.assertEqual(res.user.username, "alicesmith")
        self.assertEqual(res.user.full_name, "Alice Smith")
        self.assertIsNotNone(res.access_token)

        # Verify HttpOnly cookie was set in headers
        set_cookie = resp.headers.get("set-cookie", "")
        self.assertIn("access_token=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)

    async def test_2_duplicate_email_registration(self):
        """Scenario 2: Duplicate email registration rejection (HTTP 409)."""
        payload1 = UserSignUp(
            full_name="Bob Original",
            email="bob@example.com",
            username="boborig",
            password="Password123!",
            confirm_password="Password123!",
        )
        await signup(payload1, Response())

        payload2 = UserSignUp(
            full_name="Bob Duplicate",
            email="bob@example.com",  # Duplicate email
            username="bobnew",
            password="Password123!",
            confirm_password="Password123!",
        )
        with self.assertRaises(HTTPException) as ctx:
            await signup(payload2, Response())

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("email", ctx.exception.detail.lower())

    async def test_3_duplicate_username_registration(self):
        """Scenario 3: Duplicate username registration rejection (HTTP 409)."""
        payload1 = UserSignUp(
            full_name="Charlie Original",
            email="charlie1@example.com",
            username="charlie",
            password="Password123!",
            confirm_password="Password123!",
        )
        await signup(payload1, Response())

        payload2 = UserSignUp(
            full_name="Charlie Duplicate",
            email="charlie2@example.com",
            username="charlie",  # Duplicate username
            password="Password123!",
            confirm_password="Password123!",
        )
        with self.assertRaises(HTTPException) as ctx:
            await signup(payload2, Response())

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("username", ctx.exception.detail.lower())

    async def test_4_successful_login(self):
        """Scenario 4: Successful login with email and with username."""
        signup_payload = UserSignUp(
            full_name="Diana Prince",
            email="diana@themyscira.gov",
            username="wonderdiana",
            password="GoldenLasso999!",
            confirm_password="GoldenLasso999!",
        )
        await signup(signup_payload, Response())

        # Login via Email
        resp_email = Response()
        res_email = await login(
            UserLogin(identifier="diana@themyscira.gov", password="GoldenLasso999!"),
            resp_email,
        )
        self.assertTrue(res_email.success)
        self.assertEqual(res_email.user.username, "wonderdiana")
        self.assertIn("access_token=", resp_email.headers.get("set-cookie", ""))

        # Login via Username
        resp_user = Response()
        res_user = await login(
            UserLogin(identifier="wonderdiana", password="GoldenLasso999!"),
            resp_user,
        )
        self.assertTrue(res_user.success)
        self.assertEqual(res_user.user.email, "diana@themyscira.gov")

    async def test_5_incorrect_password(self):
        """Scenario 5: Incorrect password returns HTTP 401."""
        signup_payload = UserSignUp(
            full_name="Evan Wright",
            email="evan@example.com",
            username="evanw",
            password="CorrectPassword123!",
            confirm_password="CorrectPassword123!",
        )
        await signup(signup_payload, Response())

        with self.assertRaises(HTTPException) as ctx:
            await login(
                UserLogin(identifier="evan@example.com", password="WrongPassword456!"),
                Response(),
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid", ctx.exception.detail)

    async def test_6_protected_dashboard_access(self):
        """Scenario 6: Accessing protected /api/auth/me with valid session."""
        signup_payload = UserSignUp(
            full_name="Fiona Gallagher",
            email="fiona@example.com",
            username="fionag",
            password="FionaPassword123!",
            confirm_password="FionaPassword123!",
        )
        signup_res = await signup(signup_payload, Response())
        token = signup_res.access_token

        # Mock incoming request
        mock_req = MagicMock()

        # Access with Bearer token
        user_from_header = await get_current_user(
            request=mock_req,
            access_token=None,
            authorization=f"Bearer {token}",
        )
        self.assertEqual(user_from_header["email"], "fiona@example.com")
        self.assertEqual(user_from_header["username"], "fionag")
        self.assertEqual(user_from_header["full_name"], "Fiona Gallagher")

        # Access with Cookie
        user_from_cookie = await get_current_user(
            request=mock_req,
            access_token=token,
            authorization=None,
        )
        self.assertEqual(user_from_cookie["email"], "fiona@example.com")

    async def test_7_logout(self):
        """Scenario 7: Logout clears HttpOnly cookie."""
        resp = Response()
        res = await logout(resp)
        self.assertTrue(res["success"])
        set_cookie = resp.headers.get("set-cookie", "")
        self.assertIn("access_token", set_cookie)
        self.assertTrue("Max-Age=0" in set_cookie or "expires=" in set_cookie.lower())

    async def test_8_access_protected_without_token(self):
        """Scenario 8: Unauthenticated access returns HTTP 401."""
        mock_req = MagicMock()

        # No credentials provided
        with self.assertRaises(HTTPException) as ctx:
            await get_current_user(request=mock_req, access_token=None, authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Authentication required", ctx.exception.detail)

        # Invalid token provided
        with self.assertRaises(HTTPException) as ctx:
            await get_current_user(request=mock_req, access_token="bogus.jwt.token", authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_auth_status_endpoint(self):
        """Test GET /api/auth/status reporting."""
        status_res = await auth_status()
        self.assertTrue(status_res.database_connected)
        self.assertIsNotNone(status_res.database_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
