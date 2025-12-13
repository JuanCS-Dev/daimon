"""
Tests for memory/db_utils.py

Scientific tests covering:
- Database initialization
- Schema creation via callable
"""

import sqlite3
from pathlib import Path

import pytest

from memory.db_utils import init_sqlite_db


class TestInitSqliteDb:
    """Tests for init_sqlite_db function."""

    def test_init_creates_db(self, tmp_path: Path) -> None:
        """Test that init creates the database file."""
        db_path = tmp_path / "test.db"

        def schema(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

        init_sqlite_db(db_path, schema)
        assert db_path.exists()

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that init creates parent directories."""
        db_path = tmp_path / "nested" / "dirs" / "test.db"

        def schema(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE test (id INTEGER)")

        init_sqlite_db(db_path, schema)
        assert db_path.exists()

    def test_init_runs_schema(self, tmp_path: Path) -> None:
        """Test that schema initializer is executed."""
        db_path = tmp_path / "test.db"

        def schema(conn: sqlite3.Connection) -> None:
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """)
            conn.execute("INSERT INTO users (name) VALUES ('test')")

        init_sqlite_db(db_path, schema)

        # Verify schema was created
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM users")
            row = cursor.fetchone()
            assert row[0] == "test"

    def test_init_commits_changes(self, tmp_path: Path) -> None:
        """Test that changes are committed."""
        db_path = tmp_path / "test.db"
        calls = []

        def schema(conn: sqlite3.Connection) -> None:
            calls.append("schema")
            conn.execute("CREATE TABLE test (id INTEGER)")

        init_sqlite_db(db_path, schema)
        assert "schema" in calls

        # Verify table persists after connection closed
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
            )
            assert cursor.fetchone() is not None
