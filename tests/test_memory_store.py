"""
Tests for memory/optimized_store.py

Scientific tests covering:
- Database initialization
- Memory CRUD operations
- FTS5 search functionality
- Category filtering
- Importance decay
- Edge cases and error handling
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memory.optimized_store import (
    MemoryStore,
    MemoryItem,
    MemoryTimestamps,
    SearchResult,
)


class TestMemoryTimestamps:
    """Tests for MemoryTimestamps dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating timestamps with default access_count."""
        now = datetime.now()
        ts = MemoryTimestamps(created_at=now, accessed_at=now)
        assert ts.created_at == now
        assert ts.accessed_at == now
        assert ts.access_count == 0

    def test_create_with_access_count(self) -> None:
        """Test creating timestamps with custom access_count."""
        now = datetime.now()
        ts = MemoryTimestamps(created_at=now, accessed_at=now, access_count=5)
        assert ts.access_count == 5


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_create_memory_item(self) -> None:
        """Test creating a MemoryItem."""
        now = datetime.now()
        ts = MemoryTimestamps(created_at=now, accessed_at=now, access_count=3)
        item = MemoryItem(
            id="test123",
            content="Test content",
            category="test",
            importance=0.8,
            timestamps=ts,
        )
        assert item.id == "test123"
        assert item.content == "Test content"
        assert item.category == "test"
        assert item.importance == 0.8
        assert item.created_at == now
        assert item.accessed_at == now
        assert item.access_count == 3

    def test_from_row(self) -> None:
        """Test creating MemoryItem from database row."""
        now = datetime.now().isoformat()
        row = ("id1", "content", "category", 0.7, now, now, 2)
        item = MemoryItem.from_row(row)
        assert item.id == "id1"
        assert item.content == "content"
        assert item.category == "category"
        assert item.importance == 0.7
        assert item.access_count == 2


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """Test creating a SearchResult."""
        now = datetime.now()
        ts = MemoryTimestamps(created_at=now, accessed_at=now)
        item = MemoryItem(
            id="test",
            content="Test",
            category="test",
            importance=0.5,
            timestamps=ts,
        )
        result = SearchResult(item=item, score=0.9, match_type="fts")
        assert result.item == item
        assert result.score == 0.9
        assert result.match_type == "fts"


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    def test_init_creates_db(self, tmp_path: Path) -> None:
        """Test that initialization creates the database."""
        db_path = tmp_path / "test.db"
        store = MemoryStore(db_path)
        assert db_path.exists()
        assert store.db_path == db_path

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that initialization creates parent directories."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        MemoryStore(db_path)
        assert db_path.exists()

    def test_init_creates_tables(self, tmp_path: Path) -> None:
        """Test that initialization creates required tables."""
        db_path = tmp_path / "test.db"
        MemoryStore(db_path)

        with sqlite3.connect(db_path) as conn:
            # Check memories table
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            assert cursor.fetchone() is not None

            # Check FTS table
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            )
            assert cursor.fetchone() is not None

    def test_init_creates_indexes(self, tmp_path: Path) -> None:
        """Test that initialization creates indexes."""
        db_path = tmp_path / "test.db"
        MemoryStore(db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_memories_category" in indexes
            assert "idx_memories_importance" in indexes


class TestMemoryStoreAdd:
    """Tests for MemoryStore.add()."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> MemoryStore:
        """Create a temporary MemoryStore."""
        return MemoryStore(tmp_path / "test.db")

    def test_add_memory(self, store: MemoryStore) -> None:
        """Test adding a memory."""
        memory_id = store.add("Test content", category="test", importance=0.8)
        assert memory_id is not None
        assert len(memory_id) == 16  # SHA256 hash prefix

    def test_add_generates_consistent_id(self, store: MemoryStore) -> None:
        """Test that same content generates same ID."""
        id1 = store.add("Same content")
        id2 = store.add("Same content")
        assert id1 == id2

    def test_add_different_content_different_id(self, store: MemoryStore) -> None:
        """Test that different content generates different IDs."""
        id1 = store.add("Content A")
        id2 = store.add("Content B")
        assert id1 != id2

    def test_add_reinforces_existing(self, store: MemoryStore) -> None:
        """Test that adding existing memory reinforces it."""
        store.add("Test content")
        store.add("Test content")

        # Check access_count increased
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_count FROM memories WHERE content = ?",
                ("Test content",)
            )
            row = cursor.fetchone()
            assert row[0] == 1  # Reinforced once

    def test_add_with_default_category(self, store: MemoryStore) -> None:
        """Test adding memory with default category."""
        store.add("Test content")

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute(
                "SELECT category FROM memories WHERE content = ?",
                ("Test content",)
            )
            row = cursor.fetchone()
            assert row[0] == "general"

    def test_add_with_custom_importance(self, store: MemoryStore) -> None:
        """Test adding memory with custom importance."""
        store.add("Test content", importance=0.9)

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute(
                "SELECT importance FROM memories WHERE content = ?",
                ("Test content",)
            )
            row = cursor.fetchone()
            assert row[0] == 0.9


class TestMemoryStoreSearch:
    """Tests for MemoryStore.search()."""

    @pytest.fixture
    def store_with_data(self, tmp_path: Path) -> MemoryStore:
        """Create a MemoryStore with test data."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Python programming language", category="code")
        store.add("Python snake reptile", category="nature")
        store.add("JavaScript web development", category="code")
        store.add("Machine learning algorithms", category="ai")
        return store

    def test_search_finds_matches(self, store_with_data: MemoryStore) -> None:
        """Test that search finds matching memories."""
        results = store_with_data.search("Python")
        assert len(results) == 2

    def test_search_returns_search_results(self, store_with_data: MemoryStore) -> None:
        """Test that search returns SearchResult objects."""
        results = store_with_data.search("Python")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_with_limit(self, store_with_data: MemoryStore) -> None:
        """Test that search respects limit."""
        results = store_with_data.search("Python", limit=1)
        assert len(results) == 1

    def test_search_with_category_filter(self, store_with_data: MemoryStore) -> None:
        """Test search with category filter."""
        results = store_with_data.search("Python", category="code")
        assert len(results) == 1
        assert results[0].item.category == "code"

    def test_search_no_results(self, store_with_data: MemoryStore) -> None:
        """Test search with no matches."""
        results = store_with_data.search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_search_fts_syntax_error_handled(self, store_with_data: MemoryStore) -> None:
        """Test that malformed FTS queries are handled gracefully."""
        # FTS5 special characters can cause syntax errors
        results = store_with_data.search("test AND OR NOT")
        # Should return empty list, not raise exception
        assert isinstance(results, list)


class TestMemoryStoreGetByCategory:
    """Tests for MemoryStore.get_by_category()."""

    @pytest.fixture
    def store_with_categories(self, tmp_path: Path) -> MemoryStore:
        """Create a MemoryStore with categorized data."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Code memory 1", category="code", importance=0.9)
        store.add("Code memory 2", category="code", importance=0.7)
        store.add("Other memory", category="other", importance=0.5)
        return store

    def test_get_by_category(self, store_with_categories: MemoryStore) -> None:
        """Test getting memories by category."""
        results = store_with_categories.get_by_category("code")
        assert len(results) == 2
        assert all(r.category == "code" for r in results)

    def test_get_by_category_sorted_by_importance(
        self, store_with_categories: MemoryStore
    ) -> None:
        """Test that results are sorted by importance."""
        results = store_with_categories.get_by_category("code")
        assert results[0].importance >= results[1].importance

    def test_get_by_category_with_limit(
        self, store_with_categories: MemoryStore
    ) -> None:
        """Test limit parameter."""
        results = store_with_categories.get_by_category("code", limit=1)
        assert len(results) == 1

    def test_get_by_category_empty(self, store_with_categories: MemoryStore) -> None:
        """Test getting non-existent category."""
        results = store_with_categories.get_by_category("nonexistent")
        assert len(results) == 0


class TestMemoryStoreGetStats:
    """Tests for MemoryStore.get_stats()."""

    def test_stats_empty_store(self, tmp_path: Path) -> None:
        """Test stats on empty store."""
        store = MemoryStore(tmp_path / "test.db")
        stats = store.get_stats()
        assert stats["total_memories"] == 0
        assert stats["average_importance"] == 0

    def test_stats_with_data(self, tmp_path: Path) -> None:
        """Test stats with data."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Memory 1", category="a", importance=0.8)
        store.add("Memory 2", category="b", importance=0.6)

        stats = store.get_stats()
        assert stats["total_memories"] == 2
        assert stats["average_importance"] == 0.7
        assert "a" in stats["categories"]
        assert "b" in stats["categories"]


class TestMemoryStoreGetRecent:
    """Tests for MemoryStore.get_recent()."""

    def test_get_recent_empty(self, tmp_path: Path) -> None:
        """Test getting recent from empty store."""
        store = MemoryStore(tmp_path / "test.db")
        results = store.get_recent()
        assert len(results) == 0

    def test_get_recent_with_data(self, tmp_path: Path) -> None:
        """Test getting recent memories."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Recent memory 1", category="test")
        store.add("Recent memory 2", category="test")

        results = store.get_recent(limit=10, hours=24)
        assert len(results) == 2

    def test_get_recent_respects_limit(self, tmp_path: Path) -> None:
        """Test get_recent respects limit."""
        store = MemoryStore(tmp_path / "test.db")
        for i in range(5):
            store.add(f"Memory {i}")

        results = store.get_recent(limit=2)
        assert len(results) == 2


class TestMemoryStoreDecay:
    """Tests for MemoryStore.decay_importance()."""

    def test_decay_reduces_importance(self, tmp_path: Path) -> None:
        """Test that decay reduces importance."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Test memory", importance=1.0)

        decayed = store.decay_importance(decay_rate=0.1)
        assert decayed > 0

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT importance FROM memories")
            row = cursor.fetchone()
            assert row[0] < 1.0

    def test_decay_respects_minimum(self, tmp_path: Path) -> None:
        """Test that decay respects minimum importance."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("Test memory", importance=0.15)

        # Decay multiple times
        for _ in range(10):
            store.decay_importance(decay_rate=0.1)

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT importance FROM memories")
            row = cursor.fetchone()
            assert row[0] >= 0.1  # Minimum importance


class TestMemoryStoreDelete:
    """Tests for MemoryStore.delete()."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        """Test deleting an existing memory."""
        store = MemoryStore(tmp_path / "test.db")
        memory_id = store.add("Test memory")

        result = store.delete(memory_id)
        assert result is True

        # Verify deleted
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM memories WHERE id = ?", (memory_id,)
            )
            assert cursor.fetchone() is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting non-existent memory."""
        store = MemoryStore(tmp_path / "test.db")
        result = store.delete("nonexistent_id")
        assert result is False


class TestMemoryStoreSearchMinImportance:
    """Tests for MemoryStore.search() with min_importance."""

    def test_search_with_min_importance(self, tmp_path: Path) -> None:
        """Test search filtering by minimum importance."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("High importance memory", importance=0.9)
        store.add("Low importance memory", importance=0.3)

        # Search with high min_importance
        results = store.search("memory", min_importance=0.5)
        assert len(results) == 1
        assert results[0].item.importance >= 0.5


class TestMemoryStoreCleanup:
    """Tests for MemoryStore.cleanup()."""

    def test_cleanup_removes_low_importance(self, tmp_path: Path) -> None:
        """Test that cleanup removes low importance memories."""
        store = MemoryStore(tmp_path / "test.db")
        store.add("High importance", importance=0.9)
        store.add("Low importance", importance=0.05)

        # Make the low-importance memory old enough for cleanup
        with sqlite3.connect(store.db_path) as conn:
            old_date = (datetime.now() - timedelta(days=100)).isoformat()
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE importance < 0.1",
                (old_date,)
            )
            conn.commit()

        removed = store.cleanup(min_importance=0.1, max_age_days=90)
        assert removed == 1

        stats = store.get_stats()
        assert stats["total_memories"] == 1
