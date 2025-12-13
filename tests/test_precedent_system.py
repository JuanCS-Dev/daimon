"""
Tests for memory/precedent_system.py and memory/precedent_models.py

Scientific tests covering:
- Precedent models (Precedent, PrecedentMeta, PrecedentMatch)
- Precedent recording and retrieval
- FTS5 search functionality
- Outcome filtering and updates
- Relevance feedback loop
- Application tracking
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from memory.precedent_models import (
    DEFAULT_DB_PATH,
    OutcomeType,
    Precedent,
    PrecedentMatch,
    PrecedentMeta,
)
from memory.precedent_system import PrecedentSystem


class TestPrecedentMeta:
    """Tests for PrecedentMeta dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating PrecedentMeta with default values."""
        meta = PrecedentMeta()
        assert meta.created_at == ""
        assert meta.updated_at == ""
        assert meta.relevance == 0.5
        assert meta.application_count == 0
        assert meta.success_rate == 0.0
        assert meta.tags == []

    def test_create_with_values(self) -> None:
        """Test creating PrecedentMeta with custom values."""
        meta = PrecedentMeta(
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-02T00:00:00",
            relevance=0.8,
            application_count=5,
            success_rate=0.9,
            tags=["code", "testing"],
        )
        assert meta.created_at == "2025-01-01T00:00:00"
        assert meta.updated_at == "2025-01-02T00:00:00"
        assert meta.relevance == 0.8
        assert meta.application_count == 5
        assert meta.success_rate == 0.9
        assert meta.tags == ["code", "testing"]


class TestPrecedent:
    """Tests for Precedent dataclass."""

    def test_create_precedent(self) -> None:
        """Test creating a Precedent."""
        meta = PrecedentMeta(
            created_at="2025-01-01T00:00:00",
            relevance=0.7,
            tags=["test"],
        )
        precedent = Precedent(
            id="test123",
            context="User requested optimization",
            decision="Used caching",
            outcome="success",
            lesson="Caching improves performance",
            meta=meta,
        )
        assert precedent.id == "test123"
        assert precedent.context == "User requested optimization"
        assert precedent.decision == "Used caching"
        assert precedent.outcome == "success"
        assert precedent.lesson == "Caching improves performance"
        assert precedent.relevance == 0.7
        assert precedent.tags == ["test"]

    def test_precedent_properties(self) -> None:
        """Test Precedent property accessors."""
        meta = PrecedentMeta(
            created_at="2025-01-01",
            updated_at="2025-01-02",
            relevance=0.9,
            application_count=3,
            success_rate=0.8,
            tags=["a", "b"],
        )
        precedent = Precedent(
            id="p1",
            context="ctx",
            decision="dec",
            outcome="partial",
            lesson="les",
            meta=meta,
        )
        assert precedent.created_at == "2025-01-01"
        assert precedent.updated_at == "2025-01-02"
        assert precedent.relevance == 0.9
        assert precedent.application_count == 3
        assert precedent.success_rate == 0.8
        assert precedent.tags == ["a", "b"]

    def test_from_row(self) -> None:
        """Test creating Precedent from database row."""
        row = (
            "id1",  # id
            "context text",  # context
            "decision text",  # decision
            "success",  # outcome
            "lesson text",  # lesson
            "2025-01-01T00:00:00",  # created_at
            "2025-01-02T00:00:00",  # updated_at
            0.75,  # relevance
            2,  # application_count
            0.8,  # success_rate
            '["tag1", "tag2"]',  # tags (JSON)
        )
        precedent = Precedent.from_row(row)
        assert precedent.id == "id1"
        assert precedent.context == "context text"
        assert precedent.decision == "decision text"
        assert precedent.outcome == "success"
        assert precedent.lesson == "lesson text"
        assert precedent.created_at == "2025-01-01T00:00:00"
        assert precedent.updated_at == "2025-01-02T00:00:00"
        assert precedent.relevance == 0.75
        assert precedent.application_count == 2
        assert precedent.success_rate == 0.8
        assert precedent.tags == ["tag1", "tag2"]

    def test_from_row_empty_tags(self) -> None:
        """Test creating Precedent from row with empty tags."""
        row = (
            "id1", "ctx", "dec", "unknown", "les",
            "2025-01-01", "2025-01-01", 0.5, 0, 0.0, "",
        )
        precedent = Precedent.from_row(row)
        assert precedent.tags == []

    def test_to_dict(self) -> None:
        """Test converting Precedent to dictionary."""
        meta = PrecedentMeta(
            created_at="2025-01-01",
            updated_at="2025-01-02",
            relevance=0.8,
            application_count=1,
            success_rate=1.0,
            tags=["test"],
        )
        precedent = Precedent(
            id="p1",
            context="ctx",
            decision="dec",
            outcome="success",
            lesson="les",
            meta=meta,
        )
        d = precedent.to_dict()
        assert d["id"] == "p1"
        assert d["context"] == "ctx"
        assert d["decision"] == "dec"
        assert d["outcome"] == "success"
        assert d["lesson"] == "les"
        assert d["created_at"] == "2025-01-01"
        assert d["updated_at"] == "2025-01-02"
        assert d["relevance"] == 0.8
        assert d["application_count"] == 1
        assert d["success_rate"] == 1.0
        assert d["tags"] == ["test"]


class TestPrecedentMatch:
    """Tests for PrecedentMatch dataclass."""

    def test_create_match(self) -> None:
        """Test creating a PrecedentMatch."""
        meta = PrecedentMeta()
        precedent = Precedent(
            id="p1",
            context="ctx",
            decision="dec",
            outcome="success",
            lesson="les",
            meta=meta,
        )
        match = PrecedentMatch(
            precedent=precedent,
            score=0.9,
            match_reason="Matched 'optimization'",
        )
        assert match.precedent == precedent
        assert match.score == 0.9
        assert match.match_reason == "Matched 'optimization'"


class TestDefaultDbPath:
    """Tests for default database path constant."""

    def test_default_path(self) -> None:
        """Test default database path is in home directory."""
        assert DEFAULT_DB_PATH.parent.name == "memory"
        assert DEFAULT_DB_PATH.name == "precedents.db"


class TestPrecedentSystemInit:
    """Tests for PrecedentSystem initialization."""

    def test_init_creates_db(self, tmp_path: Path) -> None:
        """Test that initialization creates the database."""
        db_path = tmp_path / "test.db"
        system = PrecedentSystem(db_path)
        assert db_path.exists()
        assert system.db_path == db_path

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that initialization creates parent directories."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        PrecedentSystem(db_path)
        assert db_path.exists()

    def test_init_creates_tables(self, tmp_path: Path) -> None:
        """Test that initialization creates required tables."""
        db_path = tmp_path / "test.db"
        PrecedentSystem(db_path)

        with sqlite3.connect(db_path) as conn:
            # Check precedents table
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='precedents'"
            )
            assert cursor.fetchone() is not None

            # Check FTS table
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='precedents_fts'"
            )
            assert cursor.fetchone() is not None

    def test_init_creates_indexes(self, tmp_path: Path) -> None:
        """Test that initialization creates indexes."""
        db_path = tmp_path / "test.db"
        PrecedentSystem(db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_precedents_outcome" in indexes
            assert "idx_precedents_relevance" in indexes


class TestPrecedentSystemRecord:
    """Tests for PrecedentSystem.record()."""

    @pytest.fixture
    def system(self, tmp_path: Path) -> PrecedentSystem:
        """Create a temporary PrecedentSystem."""
        return PrecedentSystem(tmp_path / "test.db")

    def test_record_precedent(self, system: PrecedentSystem) -> None:
        """Test recording a precedent."""
        pid = system.record(
            context="User asked about performance",
            decision="Suggested indexing",
            outcome="success",
        )
        assert pid is not None
        assert len(pid) == 16  # SHA256 hash prefix

    def test_record_generates_consistent_id(self, system: PrecedentSystem) -> None:
        """Test that same context+decision generates same ID."""
        id1 = system.record(context="ctx", decision="dec")
        id2 = system.record(context="ctx", decision="dec")
        assert id1 == id2

    def test_record_different_content_different_id(self, system: PrecedentSystem) -> None:
        """Test that different content generates different IDs."""
        id1 = system.record(context="ctx1", decision="dec")
        id2 = system.record(context="ctx2", decision="dec")
        assert id1 != id2

    def test_record_reinforces_existing(self, system: PrecedentSystem) -> None:
        """Test that recording same precedent reinforces it."""
        pid = system.record(context="ctx", decision="dec")
        system.record(context="ctx", decision="dec")

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT application_count, relevance FROM precedents WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            assert row[0] == 1  # Reinforced once
            assert row[1] >= 0.5  # Relevance increased

    def test_record_with_meta(self, system: PrecedentSystem) -> None:
        """Test recording with metadata."""
        meta = PrecedentMeta(tags=["test", "code"], relevance=0.8)
        pid = system.record(
            context="ctx",
            decision="dec",
            outcome="success",
            meta=meta,
        )

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT relevance, tags FROM precedents WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            assert row[0] == 0.8
            assert "test" in row[1]

    def test_record_with_default_outcome(self, system: PrecedentSystem) -> None:
        """Test recording with default outcome."""
        pid = system.record(context="ctx", decision="dec")

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT outcome FROM precedents WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            assert row[0] == "unknown"


class TestPrecedentSystemSearch:
    """Tests for PrecedentSystem.search()."""

    @pytest.fixture
    def system_with_data(self, tmp_path: Path) -> PrecedentSystem:
        """Create a PrecedentSystem with test data."""
        system = PrecedentSystem(tmp_path / "test.db")
        system.record(
            context="Database optimization needed",
            decision="Used index-only scans",
            outcome="success",
        )
        system.record(
            context="API performance issue",
            decision="Added caching layer",
            outcome="success",
        )
        system.record(
            context="Memory leak detected",
            decision="Fixed circular reference",
            outcome="failure",
        )
        return system

    def test_search_finds_matches(self, system_with_data: PrecedentSystem) -> None:
        """Test that search finds matching precedents."""
        results = system_with_data.search("optimization")
        assert len(results) >= 1

    def test_search_returns_precedent_matches(
        self, system_with_data: PrecedentSystem
    ) -> None:
        """Test that search returns PrecedentMatch objects."""
        results = system_with_data.search("database")
        assert all(isinstance(r, PrecedentMatch) for r in results)

    def test_search_with_limit(self, system_with_data: PrecedentSystem) -> None:
        """Test that search respects limit."""
        results = system_with_data.search("*", limit=1)
        # FTS5 wildcard search
        assert len(results) <= 1

    def test_search_with_outcome_filter(self, system_with_data: PrecedentSystem) -> None:
        """Test search with outcome filter."""
        results = system_with_data.search("*", outcome_filter="success")
        assert all(r.precedent.outcome == "success" for r in results)

    def test_search_no_results(self, system_with_data: PrecedentSystem) -> None:
        """Test search with no matches."""
        results = system_with_data.search("nonexistent_xyz_term")
        assert len(results) == 0

    def test_search_fts_syntax_error_handled(
        self, system_with_data: PrecedentSystem
    ) -> None:
        """Test that malformed FTS queries are handled gracefully."""
        results = system_with_data.search("test AND OR NOT")
        # Should return empty list, not raise exception
        assert isinstance(results, list)

    def test_search_with_min_relevance(self, system_with_data: PrecedentSystem) -> None:
        """Test search with minimum relevance filter."""
        results = system_with_data.search("*", min_relevance=0.9)
        assert all(r.precedent.relevance >= 0.9 for r in results)


class TestPrecedentSystemUpdateOutcome:
    """Tests for PrecedentSystem.update_outcome()."""

    @pytest.fixture
    def system(self, tmp_path: Path) -> PrecedentSystem:
        """Create a temporary PrecedentSystem."""
        return PrecedentSystem(tmp_path / "test.db")

    def test_update_outcome_success(self, system: PrecedentSystem) -> None:
        """Test updating outcome to success."""
        pid = system.record(context="ctx", decision="dec", outcome="unknown")
        result = system.update_outcome(pid, "success", "It worked!")
        assert result is True

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT outcome, lesson FROM precedents WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            assert row[0] == "success"
            assert "It worked!" in row[1]

    def test_update_outcome_failure(self, system: PrecedentSystem) -> None:
        """Test updating outcome to failure reduces relevance."""
        pid = system.record(context="ctx", decision="dec")

        # Get initial relevance
        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT relevance FROM precedents WHERE id = ?",
                (pid,)
            )
            initial_relevance = cursor.fetchone()[0]

        system.update_outcome(pid, "failure")

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT relevance FROM precedents WHERE id = ?",
                (pid,)
            )
            new_relevance = cursor.fetchone()[0]
            assert new_relevance < initial_relevance

    def test_update_nonexistent(self, system: PrecedentSystem) -> None:
        """Test updating non-existent precedent."""
        result = system.update_outcome("nonexistent", "success")
        assert result is False

    def test_update_appends_lesson(self, system: PrecedentSystem) -> None:
        """Test that updating appends to existing lesson."""
        pid = system.record(context="ctx", decision="dec")
        system.update_outcome(pid, "partial", "First lesson")
        system.update_outcome(pid, "success", "Second lesson")

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT lesson FROM precedents WHERE id = ?",
                (pid,)
            )
            lesson = cursor.fetchone()[0]
            assert "First lesson" in lesson
            assert "Second lesson" in lesson


class TestPrecedentSystemApplyPrecedent:
    """Tests for PrecedentSystem.apply_precedent()."""

    @pytest.fixture
    def system(self, tmp_path: Path) -> PrecedentSystem:
        """Create a temporary PrecedentSystem."""
        return PrecedentSystem(tmp_path / "test.db")

    def test_apply_successful(self, system: PrecedentSystem) -> None:
        """Test applying precedent successfully."""
        pid = system.record(context="ctx", decision="dec")
        result = system.apply_precedent(pid, was_successful=True)
        assert result is True

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT application_count FROM precedents WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            assert row[0] >= 1

    def test_apply_unsuccessful(self, system: PrecedentSystem) -> None:
        """Test applying precedent unsuccessfully reduces relevance."""
        pid = system.record(context="ctx", decision="dec")

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT relevance FROM precedents WHERE id = ?",
                (pid,)
            )
            initial_relevance = cursor.fetchone()[0]

        system.apply_precedent(pid, was_successful=False)

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT relevance FROM precedents WHERE id = ?",
                (pid,)
            )
            new_relevance = cursor.fetchone()[0]
            assert new_relevance < initial_relevance

    def test_apply_nonexistent(self, system: PrecedentSystem) -> None:
        """Test applying non-existent precedent."""
        result = system.apply_precedent("nonexistent", was_successful=True)
        assert result is False

    def test_apply_updates_success_rate(self, system: PrecedentSystem) -> None:
        """Test that applying updates success rate."""
        pid = system.record(context="ctx", decision="dec")

        # Apply successfully multiple times
        for _ in range(3):
            system.apply_precedent(pid, was_successful=True)

        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT success_rate FROM precedents WHERE id = ?",
                (pid,)
            )
            success_rate = cursor.fetchone()[0]
            assert success_rate > 0


class TestPrecedentSystemGetByOutcome:
    """Tests for PrecedentSystem.get_by_outcome()."""

    @pytest.fixture
    def system_with_outcomes(self, tmp_path: Path) -> PrecedentSystem:
        """Create a PrecedentSystem with varied outcomes."""
        system = PrecedentSystem(tmp_path / "test.db")
        system.record(context="ctx1", decision="dec1", outcome="success")
        system.record(context="ctx2", decision="dec2", outcome="success")
        system.record(context="ctx3", decision="dec3", outcome="failure")
        return system

    def test_get_by_success(self, system_with_outcomes: PrecedentSystem) -> None:
        """Test getting successful precedents."""
        results = system_with_outcomes.get_by_outcome("success")
        assert len(results) == 2
        assert all(r.outcome == "success" for r in results)

    def test_get_by_failure(self, system_with_outcomes: PrecedentSystem) -> None:
        """Test getting failed precedents."""
        results = system_with_outcomes.get_by_outcome("failure")
        assert len(results) == 1
        assert results[0].outcome == "failure"

    def test_get_by_outcome_with_limit(
        self, system_with_outcomes: PrecedentSystem
    ) -> None:
        """Test limit parameter."""
        results = system_with_outcomes.get_by_outcome("success", limit=1)
        assert len(results) == 1


class TestPrecedentSystemGetTopPrecedents:
    """Tests for PrecedentSystem.get_top_precedents()."""

    def test_get_top_empty(self, tmp_path: Path) -> None:
        """Test getting top from empty system."""
        system = PrecedentSystem(tmp_path / "test.db")
        results = system.get_top_precedents()
        assert len(results) == 0

    def test_get_top_with_data(self, tmp_path: Path) -> None:
        """Test getting top precedents."""
        system = PrecedentSystem(tmp_path / "test.db")
        system.record(
            context="ctx1",
            decision="dec1",
            meta=PrecedentMeta(relevance=0.9),
        )
        system.record(
            context="ctx2",
            decision="dec2",
            meta=PrecedentMeta(relevance=0.5),
        )

        results = system.get_top_precedents(limit=2)
        assert len(results) == 2
        # Should be sorted by relevance
        assert results[0].relevance >= results[1].relevance


class TestPrecedentSystemGetStats:
    """Tests for PrecedentSystem.get_stats()."""

    def test_stats_empty_system(self, tmp_path: Path) -> None:
        """Test stats on empty system."""
        system = PrecedentSystem(tmp_path / "test.db")
        stats = system.get_stats()
        assert stats["total_precedents"] == 0
        assert stats["average_relevance"] == 0.0
        assert stats["total_applications"] == 0

    def test_stats_with_data(self, tmp_path: Path) -> None:
        """Test stats with data."""
        system = PrecedentSystem(tmp_path / "test.db")
        system.record(context="ctx1", decision="dec1", outcome="success")
        system.record(context="ctx2", decision="dec2", outcome="failure")

        stats = system.get_stats()
        assert stats["total_precedents"] == 2
        assert stats["average_relevance"] > 0
        assert "success" in stats["by_outcome"]
        assert "failure" in stats["by_outcome"]
        assert "db_path" in stats


class TestPrecedentSystemDelete:
    """Tests for PrecedentSystem.delete()."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        """Test deleting an existing precedent."""
        system = PrecedentSystem(tmp_path / "test.db")
        pid = system.record(context="ctx", decision="dec")

        result = system.delete(pid)
        assert result is True

        # Verify deleted
        with sqlite3.connect(system.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM precedents WHERE id = ?", (pid,)
            )
            assert cursor.fetchone() is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting non-existent precedent."""
        system = PrecedentSystem(tmp_path / "test.db")
        result = system.delete("nonexistent_id")
        assert result is False
