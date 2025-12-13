"""
DAIMON Database Utilities.

Common utilities for SQLite database initialization.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable


def init_sqlite_db(
    db_path: Path,
    schema_initializer: Callable[[sqlite3.Connection], None],
) -> None:
    """
    Initialize SQLite database with given schema.

    Args:
        db_path: Path to the database file
        schema_initializer: Function that creates tables/indexes given a connection
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        schema_initializer(conn)
        conn.commit()
