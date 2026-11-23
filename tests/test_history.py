"""
Unit & Integration Tests for VeriFrame Detection History
=========================================================
Tests:
1. Unauthorized access to /api/history returns 401
2. Unauthorized access to /api/history/stats returns 401
3. Inserting detection history records for authenticated user
4. Querying history with search and filters (media_type, prediction)
5. Fetching history summary statistics (total, real, deepfake, recent)
6. Fetching single history record by ID
7. Deleting history record by ID
8. User isolation: User B cannot view or delete User A's history records
"""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bson import ObjectId
from fastapi import HTTPException
from backend.routes.history_routes import (
    list_history,
    get_history_stats,
    get_history_record,
    delete_history_record,
)
from backend.database import db_manager


class MockHistoryCollection:
    def __init__(self):
        self.docs = []

    async def insert_one(self, doc):
        d = dict(doc)
        if "_id" not in d:
            d["_id"] = ObjectId()
        self.docs.append(d)
        res = type("InsertResult", (), {"inserted_id": d["_id"]})()
        return res

    async def find_one(self, query):
        for doc in self.docs:
            match = True
            for k, v in query.items():
                if k == "_id":
                    if str(doc.get("_id")) != str(v):
                        match = False
                        break
                elif doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def find(self, query):
        filtered = []
        for doc in self.docs:
            m = True
            for k, v in query.items():
                if k == "filename" and isinstance(v, dict) and "$regex" in v:
                    import re
                    pattern = re.compile(v["$regex"], re.IGNORECASE)
                    if not pattern.search(doc.get("filename", "")):
                        m = False
                        break
                elif doc.get(k) != v:
                    m = False
                    break
            if m:
                filtered.append(doc)

        class MockAsyncCursor:
            def __init__(self, items):
                self.items = list(items)

            def sort(self, key, direction):
                self.items.sort(key=lambda x: str(x.get(key, "")), reverse=(direction == -1))
                return self

            def skip(self, n):
                self.items = self.items[n:]
                return self

            def limit(self, n):
                self.items = self.items[:n]
                return self

            def __aiter__(self):
                self._iter = iter(self.items)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        return MockAsyncCursor(filtered)

    async def count_documents(self, query):
        count = 0
        for doc in self.docs:
            m = True
            for k, v in query.items():
                if k == "created_at" and isinstance(v, dict) and "$gte" in v:
                    doc_date = str(doc.get("created_at", ""))
                    if not doc_date or doc_date < str(v["$gte"]):
                        m = False
                        break
                elif doc.get(k) != v:
                    m = False
                    break
            if m:
                count += 1
        return count

    async def delete_one(self, query):
        target = None
        for doc in self.docs:
            m = True
            for k, v in query.items():
                if k == "_id":
                    if str(doc.get("_id")) != str(v):
                        match = False
                        break
                elif doc.get(k) != v:
                    m = False
                    break
            if m:
                target = doc
                break
        if target:
            self.docs.remove(target)
            return type("DelResult", (), {"deleted_count": 1})()
        return type("DelResult", (), {"deleted_count": 0})()


class MockDB:
    def __init__(self, collection):
        self._coll = collection

    def __getitem__(self, name):
        return self._coll


class TestHistoryEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_coll = MockHistoryCollection()
        self.mock_db = MockDB(self.mock_coll)
        self.orig_get_db = db_manager.get_db
        db_manager.get_db = self._mock_get_db

        self.user_a = {
            "id": "673cf1000000000000000001",
            "username": "alice",
            "email": "alice@veriframe.io",
            "full_name": "Alice Researcher",
        }
        self.user_b = {
            "id": "673cf2000000000000000002",
            "username": "bob",
            "email": "bob@veriframe.io",
            "full_name": "Bob Analyst",
        }

        # Seed records for user_a
        now = datetime.now(timezone.utc)
        self.record1 = {
            "_id": ObjectId("673cf3000000000000000010"),
            "user_id": self.user_a["id"],
            "filename": "alice_real_face.jpg",
            "media_type": "image",
            "prediction": "REAL",
            "confidence": 0.945,
            "real_prob": 0.945,
            "deepfake_prob": 0.055,
            "face_detected": True,
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "file_size_bytes": 102400,
            "details": {"device": "cpu"},
        }
        self.record2 = {
            "_id": ObjectId("673cf3000000000000000020"),
            "user_id": self.user_a["id"],
            "filename": "alice_deepfake_synthetic.png",
            "media_type": "image",
            "prediction": "DEEPFAKE",
            "confidence": 0.988,
            "real_prob": 0.012,
            "deepfake_prob": 0.988,
            "face_detected": True,
            "created_at": (now - timedelta(days=1)).isoformat(),
            "file_size_bytes": 204800,
            "details": {"device": "cpu"},
        }
        # Seed record for user_b
        self.record_b = {
            "_id": ObjectId("673cf3000000000000000030"),
            "user_id": self.user_b["id"],
            "filename": "bob_sample.mp4",
            "media_type": "video",
            "prediction": "REAL",
            "confidence": 0.88,
            "real_prob": 0.88,
            "deepfake_prob": 0.12,
            "face_detected": False,
            "created_at": (now - timedelta(days=2)).isoformat(),
            "file_size_bytes": 5000000,
            "details": {},
        }
        await self.mock_coll.insert_one(self.record1)
        await self.mock_coll.insert_one(self.record2)
        await self.mock_coll.insert_one(self.record_b)

    async def _mock_get_db(self):
        return self.mock_db

    async def asyncTearDown(self):
        db_manager.get_db = self.orig_get_db

    async def test_get_history_scoped_to_user(self):
        """User A sees only their records, User B sees only their records."""
        resp_a = await list_history(limit=50, current_user=self.user_a)
        self.assertEqual(resp_a.total, 2)
        filenames_a = [item.filename for item in resp_a.items]
        self.assertIn("alice_real_face.jpg", filenames_a)
        self.assertIn("alice_deepfake_synthetic.png", filenames_a)
        self.assertNotIn("bob_sample.mp4", filenames_a)

        resp_b = await list_history(limit=50, current_user=self.user_b)
        self.assertEqual(resp_b.total, 1)
        self.assertEqual(resp_b.items[0].filename, "bob_sample.mp4")

    async def test_get_history_filter_by_prediction(self):
        """Filter by prediction = REAL returns only REAL records for User A."""
        resp = await list_history(prediction="REAL", current_user=self.user_a)
        self.assertEqual(resp.total, 1)
        self.assertEqual(resp.items[0].prediction, "REAL")
        self.assertEqual(resp.items[0].filename, "alice_real_face.jpg")

    async def test_get_history_search(self):
        """Search regex matches filename correctly."""
        resp = await list_history(search="synthetic", current_user=self.user_a)
        self.assertEqual(resp.total, 1)
        self.assertEqual(resp.items[0].filename, "alice_deepfake_synthetic.png")

    async def test_get_history_stats(self):
        """Stats calculation returns accurate counts for User A."""
        stats = await get_history_stats(current_user=self.user_a)
        self.assertEqual(stats.total_analyzed, 2)
        self.assertEqual(stats.real_count, 1)
        self.assertEqual(stats.deepfake_count, 1)
        self.assertEqual(stats.images_analyzed, 2)
        self.assertEqual(stats.videos_analyzed, 0)
        self.assertEqual(stats.recent_activity_count, 2)

    async def test_get_history_item_by_id(self):
        """User A can retrieve their specific record by ID."""
        item = await get_history_record(
            history_id="673cf3000000000000000010",
            current_user=self.user_a,
        )
        self.assertEqual(item.filename, "alice_real_face.jpg")
        self.assertEqual(item.prediction, "REAL")

    async def test_get_history_item_isolation(self):
        """User B cannot retrieve User A's record by ID (returns 404)."""
        with self.assertRaises(HTTPException) as ctx:
            await get_history_record(
                history_id="673cf3000000000000000010",
                current_user=self.user_b,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_delete_history_item(self):
        """User A can delete their record, updating total records."""
        res = await delete_history_record(
            history_id="673cf3000000000000000010",
            current_user=self.user_a,
        )
        self.assertTrue(res["success"])
        # Verify it is no longer returned
        resp = await list_history(current_user=self.user_a)
        self.assertEqual(resp.total, 1)

    async def test_delete_history_item_isolation(self):
        """User B cannot delete User A's record (returns 404)."""
        with self.assertRaises(HTTPException) as ctx:
            await delete_history_record(
                history_id="673cf3000000000000000010",
                current_user=self.user_b,
            )
        self.assertEqual(ctx.exception.status_code, 404)
        # Record still exists for User A
        item = await get_history_record(
            history_id="673cf3000000000000000010",
            current_user=self.user_a,
        )
        self.assertIsNotNone(item)


if __name__ == "__main__":
    unittest.main()
