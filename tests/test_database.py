"""Tests for signal_viewer.core.database."""

import os
import tempfile
import unittest
from unittest.mock import patch

from signal_viewer.core.database import Database, _get_user


class _DBTestCase(unittest.TestCase):
    """Base class that creates a temporary database for each test."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "test.db3")
        self.db = Database(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestSchemaCreation(_DBTestCase):
    """Auto-create schema on init."""

    def test_tables_exist(self):
        conn = self.db._connect()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        for t in ("favourites", "comments", "lists", "list_files"):
            self.assertIn(t, tables)

    def test_idempotent_init(self):
        """Calling init twice doesn't raise."""
        db2 = Database(self.db_path)
        self.assertTrue(db2.db_path.exists())


class TestFavourites(_DBTestCase):

    def test_add_and_check(self):
        self.assertFalse(self.db.is_favourite("/a/b.h5"))
        self.db.set_favourite("/a/b.h5", True)
        self.assertTrue(self.db.is_favourite("/a/b.h5"))

    def test_remove(self):
        self.db.set_favourite("/a/b.h5", True)
        self.db.set_favourite("/a/b.h5", False)
        self.assertFalse(self.db.is_favourite("/a/b.h5"))

    def test_duplicate_add_ok(self):
        self.db.set_favourite("/a/b.h5", True)
        self.db.set_favourite("/a/b.h5", True)  # no error
        self.assertTrue(self.db.is_favourite("/a/b.h5"))

    def test_user_isolation(self):
        self.db.set_favourite("/a/b.h5", True)
        with patch("signal_viewer.core.database._get_user", return_value="other"):
            self.assertFalse(self.db.is_favourite("/a/b.h5"))


class TestComments(_DBTestCase):

    def test_private_comment_upsert(self):
        cid = self.db.save_private_comment("/f.h5", "hello")
        self.assertIsInstance(cid, int)
        # Update same
        cid2 = self.db.save_private_comment("/f.h5", "updated")
        self.assertEqual(cid, cid2)
        data = self.db.get_comments("/f.h5")
        self.assertEqual(data["private"]["content"], "updated")

    def test_public_comment_create(self):
        cid = self.db.save_public_comment("/f.h5", "public note")
        data = self.db.get_comments("/f.h5")
        self.assertEqual(len(data["public"]), 1)
        self.assertEqual(data["public"][0]["content"], "public note")
        self.assertEqual(data["public"][0]["id"], cid)

    def test_public_comment_update(self):
        cid = self.db.save_public_comment("/f.h5", "v1")
        self.db.save_public_comment("/f.h5", "v2", comment_id=cid)
        data = self.db.get_comments("/f.h5")
        self.assertEqual(data["public"][0]["content"], "v2")

    def test_delete_comment(self):
        cid = self.db.save_public_comment("/f.h5", "bye")
        self.assertTrue(self.db.delete_comment(cid))
        data = self.db.get_comments("/f.h5")
        self.assertEqual(len(data["public"]), 0)

    def test_private_hidden_from_other_user(self):
        self.db.save_private_comment("/f.h5", "secret")
        with patch("signal_viewer.core.database._get_user", return_value="other"):
            data = self.db.get_comments("/f.h5")
            self.assertIsNone(data["private"])

    def test_no_comments_returns_empty(self):
        data = self.db.get_comments("/empty.h5")
        self.assertIsNone(data["private"])
        self.assertEqual(data["public"], [])


class TestLists(_DBTestCase):

    def test_create_and_get(self):
        lid = self.db.create_list("My List")
        self.assertIsNotNone(lid)
        lists = self.db.get_lists()
        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0]["name"], "My List")

    def test_duplicate_name_returns_none(self):
        self.db.create_list("dup")
        self.assertIsNone(self.db.create_list("dup"))

    def test_delete_list(self):
        lid = self.db.create_list("rm me")
        self.assertTrue(self.db.delete_list(lid))
        self.assertEqual(len(self.db.get_lists()), 0)

    def test_add_file_to_list(self):
        lid = self.db.create_list("batch")
        self.assertTrue(self.db.add_file_to_list(lid, "/a.h5"))
        self.assertEqual(self.db.get_list_files(lid), ["/a.h5"])

    def test_remove_file_from_list(self):
        lid = self.db.create_list("batch")
        self.db.add_file_to_list(lid, "/a.h5")
        self.assertTrue(self.db.remove_file_from_list(lid, "/a.h5"))
        self.assertEqual(self.db.get_list_files(lid), [])

    def test_duplicate_file_in_list_returns_false(self):
        lid = self.db.create_list("batch")
        self.db.add_file_to_list(lid, "/a.h5")
        self.assertFalse(self.db.add_file_to_list(lid, "/a.h5"))

    def test_get_lists_for_file(self):
        lid1 = self.db.create_list("L1")
        lid2 = self.db.create_list("L2")
        self.db.add_file_to_list(lid1, "/a.h5")
        self.db.add_file_to_list(lid2, "/a.h5")
        result = self.db.get_lists_for_file("/a.h5")
        self.assertEqual(len(result), 2)

    def test_cascade_delete_removes_files(self):
        lid = self.db.create_list("cascade")
        self.db.add_file_to_list(lid, "/a.h5")
        self.db.delete_list(lid)
        # list_files should be gone due to FK cascade
        conn = self.db._connect()
        rows = conn.execute(
            "SELECT * FROM list_files WHERE list_id=?", (lid,)
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 0)

    def test_public_list_visible_to_other_user(self):
        self.db.create_list("shared", is_public=True)
        with patch("signal_viewer.core.database._get_user", return_value="other"):
            lists = self.db.get_lists()
            self.assertEqual(len(lists), 1)
            self.assertEqual(lists[0]["name"], "shared")


class TestGetUser(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_get_user(), str)

    def test_fallback_on_error(self):
        with patch("os.getlogin", side_effect=OSError):
            user = _get_user()
            self.assertIsInstance(user, str)


if __name__ == "__main__":
    unittest.main()
