"""
SQLite database for signal viewer user data: favourites, comments, and lists.

Auto-creates the schema on first run.  Uses os.getlogin() (with env fallback)
to identify the current user so that private data is scoped per OS account.

The database file is a standard SQLite .db3 located next to the project root
(configurable via config.DATABASE_PATH).
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_user() -> str:
    """Return current OS username (best-effort)."""
    try:
        return os.getlogin()
    except Exception:
        return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


class Database:
    """Thread-safe SQLite database for favourites, comments, and lists."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS favourites (
                id        INTEGER PRIMARY KEY,
                user      TEXT    NOT NULL,
                file_path TEXT    NOT NULL,
                created_at TEXT   DEFAULT (datetime('now')),
                UNIQUE(user, file_path)
            );

            CREATE TABLE IF NOT EXISTS comments (
                id         INTEGER PRIMARY KEY,
                user       TEXT    NOT NULL,
                file_path  TEXT    NOT NULL,
                content    TEXT    NOT NULL DEFAULT '',
                is_public  INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    DEFAULT (datetime('now')),
                updated_at TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS lists (
                id         INTEGER PRIMARY KEY,
                user       TEXT    NOT NULL,
                name       TEXT    NOT NULL,
                is_public  INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    DEFAULT (datetime('now')),
                UNIQUE(user, name)
            );

            CREATE TABLE IF NOT EXISTS list_files (
                id         INTEGER PRIMARY KEY,
                list_id    INTEGER NOT NULL,
                file_path  TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (list_id) REFERENCES lists(id) ON DELETE CASCADE,
                UNIQUE(list_id, file_path)
            );

            CREATE INDEX IF NOT EXISTS idx_comments_file
                ON comments(file_path);
            CREATE INDEX IF NOT EXISTS idx_list_files_file
                ON list_files(file_path);
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Favourites
    # ------------------------------------------------------------------

    def is_favourite(self, file_path: str) -> bool:
        user = _get_user()
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM favourites WHERE user=? AND file_path=? LIMIT 1",
            (user, file_path),
        ).fetchone()
        conn.close()
        return row is not None

    def set_favourite(self, file_path: str, active: bool) -> bool:
        """Toggle favourite on/off. Returns the new state."""
        user = _get_user()
        conn = self._connect()
        if active:
            try:
                conn.execute(
                    "INSERT INTO favourites(user, file_path) VALUES(?,?)",
                    (user, file_path),
                )
            except sqlite3.IntegrityError:
                pass  # already exists
        else:
            conn.execute(
                "DELETE FROM favourites WHERE user=? AND file_path=?",
                (user, file_path),
            )
        conn.commit()
        conn.close()
        return active

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    def get_comments(self, file_path: str) -> Dict[str, Any]:
        """Return private + public comments for the file.

        Returns dict with keys ``private`` (str) and ``public`` (list of dicts).
        Private comment is the calling user's own private comment (at most one
        per user/file pair — stored as a single row).
        Public comments are all rows with ``is_public=1`` for the file.
        """
        user = _get_user()
        conn = self._connect()

        priv = conn.execute(
            "SELECT id, content, updated_at FROM comments "
            "WHERE file_path=? AND user=? AND is_public=0 "
            "ORDER BY updated_at DESC LIMIT 1",
            (file_path, user),
        ).fetchone()

        pubs = conn.execute(
            "SELECT id, user, content, created_at, updated_at FROM comments "
            "WHERE file_path=? AND is_public=1 "
            "ORDER BY created_at ASC",
            (file_path,),
        ).fetchall()

        conn.close()
        return {
            "private": {
                "id": priv["id"],
                "content": priv["content"],
                "updated_at": priv["updated_at"],
            }
            if priv
            else None,
            "public": [dict(r) for r in pubs],
        }

    def save_private_comment(self, file_path: str, content: str) -> int:
        """Upsert the calling user's private comment for a file."""
        user = _get_user()
        conn = self._connect()
        existing = conn.execute(
            "SELECT id FROM comments WHERE file_path=? AND user=? AND is_public=0 LIMIT 1",
            (file_path, user),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE comments SET content=?, updated_at=datetime('now') WHERE id=?",
                (content, existing["id"]),
            )
            cid = existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO comments(user, file_path, content, is_public) VALUES(?,?,?,0)",
                (user, file_path, content),
            )
            cid = cur.lastrowid
        conn.commit()
        conn.close()
        return cid

    def save_public_comment(self, file_path: str, content: str,
                            comment_id: Optional[int] = None) -> int:
        """Create or update a public comment. Returns comment id."""
        user = _get_user()
        conn = self._connect()
        if comment_id:
            conn.execute(
                "UPDATE comments SET content=?, updated_at=datetime('now') WHERE id=?",
                (content, comment_id),
            )
            cid = comment_id
        else:
            cur = conn.execute(
                "INSERT INTO comments(user, file_path, content, is_public) VALUES(?,?,?,1)",
                (user, file_path, content),
            )
            cid = cur.lastrowid
        conn.commit()
        conn.close()
        return cid

    def delete_comment(self, comment_id: int) -> bool:
        conn = self._connect()
        conn.execute("DELETE FROM comments WHERE id=?", (comment_id,))
        conn.commit()
        n = conn.total_changes
        conn.close()
        return n > 0

    # ------------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------------

    def get_lists(self) -> List[Dict[str, Any]]:
        """Return all lists visible to the calling user (own + public)."""
        user = _get_user()
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, user, name, is_public, created_at FROM lists "
            "WHERE user=? OR is_public=1 ORDER BY name",
            (user,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def create_list(self, name: str, is_public: bool = False) -> Optional[int]:
        user = _get_user()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO lists(user, name, is_public) VALUES(?,?,?)",
                (user, name, 1 if is_public else 0),
            )
            conn.commit()
            lid = cur.lastrowid
        except sqlite3.IntegrityError:
            lid = None
        conn.close()
        return lid

    def delete_list(self, list_id: int) -> bool:
        user = _get_user()
        conn = self._connect()
        conn.execute(
            "DELETE FROM lists WHERE id=? AND user=?", (list_id, user)
        )
        conn.commit()
        n = conn.total_changes
        conn.close()
        return n > 0

    def get_current_user(self) -> str:
        """Return the current OS username."""
        return _get_user()

    def update_list_public(self, list_id: int, is_public: bool) -> bool:
        """Toggle a list's public/private visibility. Only the owner can change it."""
        user = _get_user()
        conn = self._connect()
        conn.execute(
            "UPDATE lists SET is_public=? WHERE id=? AND user=?",
            (1 if is_public else 0, list_id, user),
        )
        conn.commit()
        n = conn.total_changes
        conn.close()
        return n > 0

    def get_list_files(self, list_id: int) -> List[str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT file_path FROM list_files WHERE list_id=? ORDER BY created_at",
            (list_id,),
        ).fetchall()
        conn.close()
        return [r["file_path"] for r in rows]

    def add_file_to_list(self, list_id: int, file_path: str) -> bool:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO list_files(list_id, file_path) VALUES(?,?)",
                (list_id, file_path),
            )
            conn.commit()
            ok = True
        except sqlite3.IntegrityError:
            ok = False
        conn.close()
        return ok

    def remove_file_from_list(self, list_id: int, file_path: str) -> bool:
        conn = self._connect()
        conn.execute(
            "DELETE FROM list_files WHERE list_id=? AND file_path=?",
            (list_id, file_path),
        )
        conn.commit()
        n = conn.total_changes
        conn.close()
        return n > 0

    def get_favourite_paths(self) -> List[str]:
        """Return file paths favourited by the current user."""
        user = _get_user()
        conn = self._connect()
        rows = conn.execute(
            "SELECT file_path FROM favourites WHERE user=? ORDER BY created_at",
            (user,),
        ).fetchall()
        conn.close()
        return [r["file_path"] for r in rows]

    def get_lists_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Return lists that contain the given file (visible to current user)."""
        user = _get_user()
        conn = self._connect()
        rows = conn.execute(
            "SELECT l.id, l.user, l.name, l.is_public "
            "FROM lists l JOIN list_files lf ON l.id=lf.list_id "
            "WHERE lf.file_path=? AND (l.user=? OR l.is_public=1) "
            "ORDER BY l.name",
            (file_path, user),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
