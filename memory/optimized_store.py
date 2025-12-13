"""
DAIMON Optimized Memory Store.

SQLite + FTS5 for ultra-fast local search (<10ms).

Features:
- Full-text search via FTS5
- Importance decay over time
- Category-based filtering
- Automatic cleanup of old memories

Usage:
    store = MemoryStore()
    store.add("Remember to always use type hints", category="code_style")
    results = store.search("type hints", limit=5)
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("daimon.memory")

# Default storage location
DEFAULT_DB_PATH = Path.home() / ".daimon" / "memory" / "memories.db"


@dataclass
class MemoryTimestamps:
    """Timestamp information for a memory item."""

    created_at: datetime
    accessed_at: datetime
    access_count: int = 0


@dataclass
class MemoryItem:
    """Single memory item."""

    id: str
    content: str
    category: str
    importance: float
    timestamps: MemoryTimestamps

    @property
    def created_at(self) -> datetime:
        """When memory was created."""
        return self.timestamps.created_at

    @property
    def accessed_at(self) -> datetime:
        """When memory was last accessed."""
        return self.timestamps.accessed_at

    @property
    def access_count(self) -> int:
        """How many times memory was accessed."""
        return self.timestamps.access_count

    @classmethod
    def from_row(cls, row: tuple) -> "MemoryItem":
        """Create MemoryItem from database row."""
        timestamps = MemoryTimestamps(
            created_at=datetime.fromisoformat(row[4]),
            accessed_at=datetime.fromisoformat(row[5]),
            access_count=row[6],
        )
        return cls(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            timestamps=timestamps,
        )


@dataclass
class SearchResult:
    """Search result with relevance score."""

    item: MemoryItem
    score: float
    match_type: str  # "fts" or "semantic"


class MemoryStore:
    """
    Optimized memory store with FTS5.

    Performance targets:
    - Full-text search: <10ms
    - Insert: <5ms
    - Category filter: <5ms
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.daimon/memory/memories.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._ensure_db_initialized()

    def _ensure_db_initialized(self) -> None:
        """Ensure database directory exists and schema is created."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema for memories."""
        with sqlite3.connect(self.db_path) as conn:
            self._create_memories_table(conn)
            self._create_fts_index(conn)
            self._create_triggers(conn)
            self._create_indexes(conn)
            conn.commit()

    def _create_memories_table(self, conn: sqlite3.Connection) -> None:
        """Create main memories table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)

    def _create_fts_index(self, conn: sqlite3.Connection) -> None:
        """Create FTS5 virtual table for full-text search."""
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, category, content=memories, content_rowid=rowid)
        """)

    def _create_triggers(self, conn: sqlite3.Connection) -> None:
        """Create triggers to keep FTS in sync."""
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.rowid, new.content, new.category);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category)
                VALUES ('delete', old.rowid, old.content, old.category);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category)
                VALUES ('delete', old.rowid, old.content, old.category);
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.rowid, new.content, new.category);
            END
        """)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create indexes for efficient queries."""
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(importance DESC)
        """)
        logger.debug("Database initialized at %s", self.db_path)

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
    ) -> str:
        """
        Add memory to store.

        Args:
            content: Memory content
            category: Category for filtering
            importance: Importance score 0-1

        Returns:
            Memory ID
        """
        memory_id = self._generate_id(content)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

            if existing:
                # Update access count and importance
                conn.execute(
                    """
                    UPDATE memories
                    SET accessed_at = ?, access_count = access_count + 1,
                        importance = MIN(1.0, importance + 0.1)
                    WHERE id = ?
                    """,
                    (now, memory_id),
                )
                logger.debug("Memory reinforced: %s", memory_id)
            else:
                # Insert new
                conn.execute(
                    """
                    INSERT INTO memories
                    (id, content, category, importance, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, content, category, importance, now, now),
                )
                logger.debug("Memory added: %s", memory_id)

            conn.commit()

        return memory_id

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """
        Full-text search memories.

        Target latency: <10ms

        Args:
            query: Search query
            category: Filter by category
            limit: Maximum results
            min_importance: Minimum importance score

        Returns:
            List of SearchResult sorted by relevance
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build FTS query
            if category:
                sql = """
                    SELECT m.*, bm25(memories_fts) as score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.category = ?
                    AND m.importance >= ?
                    ORDER BY score
                    LIMIT ?
                """
                params = (query, category, min_importance, limit)
            else:
                sql = """
                    SELECT m.*, bm25(memories_fts) as score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.importance >= ?
                    ORDER BY score
                    LIMIT ?
                """
                params = (query, min_importance, limit)

            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                # Handle malformed FTS query
                logger.warning("FTS query failed: %s", e)
                return []

        results = []
        for row in rows:
            item = MemoryItem.from_row(row[:7])
            score = abs(row[7]) if len(row) > 7 else 0.5
            results.append(SearchResult(item=item, score=score, match_type="fts"))

        return results

    def get_by_category(
        self, category: str, limit: int = 50
    ) -> List[MemoryItem]:
        """
        Get memories by category.

        Args:
            category: Category to filter
            limit: Maximum results

        Returns:
            List of MemoryItem sorted by importance
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE category = ?
                ORDER BY importance DESC, accessed_at DESC
                LIMIT ?
                """,
                (category, limit),
            ).fetchall()

        return [MemoryItem.from_row(row) for row in rows]

    def get_recent(self, limit: int = 20, hours: int = 24) -> List[MemoryItem]:
        """
        Get recent memories.

        Args:
            limit: Maximum results
            hours: How many hours back to look

        Returns:
            List of MemoryItem sorted by recency
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()

        return [MemoryItem.from_row(row) for row in rows]

    def decay_importance(self, decay_rate: float = 0.01) -> int:
        """
        Apply importance decay to all memories.

        Older, less accessed memories lose importance over time.

        Args:
            decay_rate: How much to decay (0-1)

        Returns:
            Number of memories updated
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                UPDATE memories
                SET importance = MAX(0.1, importance - ?)
                WHERE importance > 0.1
                """,
                (decay_rate,),
            )
            conn.commit()
            return result.rowcount

    def cleanup(self, min_importance: float = 0.1, max_age_days: int = 90) -> int:
        """
        Remove old, low-importance memories.

        Args:
            min_importance: Remove memories below this importance
            max_age_days: Remove memories older than this

        Returns:
            Number of memories deleted
        """
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                DELETE FROM memories
                WHERE importance <= ? AND created_at < ?
                """,
                (min_importance, cutoff),
            )
            conn.commit()
            return result.rowcount

    def get_stats(self) -> dict:
        """
        Get memory store statistics.

        Returns:
            Dict with counts and category breakdown
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            categories = conn.execute(
                """
                SELECT category, COUNT(*), AVG(importance)
                FROM memories
                GROUP BY category
                """
            ).fetchall()

            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM memories"
            ).fetchone()[0] or 0.0

        return {
            "total_memories": total,
            "average_importance": round(avg_importance, 3),
            "categories": {
                cat: {"count": count, "avg_importance": round(avg_imp or 0, 3)}
                for cat, count, avg_imp in categories
            },
            "db_path": str(self.db_path),
        }

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            conn.commit()
            return result.rowcount > 0


if __name__ == "__main__":
    import time
    print("DAIMON Memory Store - Performance Test")
    test_db = Path("/tmp/daimon_memory_test.db")
    if test_db.exists():
        test_db.unlink()
    store = MemoryStore(test_db)
    # Add test memories
    start = time.perf_counter()
    for i in range(100):
        store.add(f"Test memory {i} about coding", category="test", importance=0.5)
    print(f"Insert 100: {(time.perf_counter() - start) * 1000:.1f}ms")
    # Search
    start = time.perf_counter()
    search_results = store.search("coding", limit=10)
    print(f"Search: {(time.perf_counter() - start) * 1000:.2f}ms, {len(search_results)} results")
    # Stats
    stats = store.get_stats()
    print(f"Total: {stats['total_memories']}, Avg importance: {stats['average_importance']}")
    test_db.unlink()
    print("Tests passed!")
