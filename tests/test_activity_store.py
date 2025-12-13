"""Tests for ActivityStore class."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from memory.activity_store import (
    ActivityStore,
    ActivityRecord,
    ActivitySummary,
    get_activity_store,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_activity.db"
        yield db_path


@pytest.fixture
def store(temp_db):
    """Create ActivityStore with temporary database."""
    return ActivityStore(db_path=temp_db, retention_days=7)


class TestActivityRecord:
    """Tests for ActivityRecord dataclass."""

    def test_to_dict(self):
        """Convert to dict."""
        record = ActivityRecord(
            id=1,
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            watcher_type="window_watcher",
            data={"app": "code"},
        )
        d = record.to_dict()

        assert d["id"] == 1
        assert d["timestamp"] == "2025-01-15T10:30:00"
        assert d["watcher_type"] == "window_watcher"
        assert d["data"]["app"] == "code"

    def test_from_row(self):
        """Create from database row."""
        row = (42, "2025-01-15T10:30:00", "test", '{"key": "value"}')
        record = ActivityRecord.from_row(row)

        assert record.id == 42
        assert record.timestamp == datetime(2025, 1, 15, 10, 30, 0)
        assert record.watcher_type == "test"
        assert record.data["key"] == "value"

    def test_from_row_empty_data(self):
        """Create from row with empty data."""
        row = (1, "2025-01-15T10:00:00", "test", None)
        record = ActivityRecord.from_row(row)

        assert record.data == {}


class TestActivitySummary:
    """Tests for ActivitySummary dataclass."""

    def test_to_dict(self):
        """Convert to dict."""
        summary = ActivitySummary(
            watcher_type="window_watcher",
            record_count=100,
            first_activity=datetime(2025, 1, 15, 8, 0, 0),
            last_activity=datetime(2025, 1, 15, 17, 0, 0),
            aggregated_data={"total_focus": 3600},
        )
        d = summary.to_dict()

        assert d["watcher_type"] == "window_watcher"
        assert d["record_count"] == 100
        assert d["first_activity"] == "2025-01-15T08:00:00"
        assert d["last_activity"] == "2025-01-15T17:00:00"
        assert d["aggregated_data"]["total_focus"] == 3600

    def test_to_dict_none_times(self):
        """Convert to dict with None times."""
        summary = ActivitySummary(
            watcher_type="test",
            record_count=0,
            first_activity=None,
            last_activity=None,
            aggregated_data={},
        )
        d = summary.to_dict()

        assert d["first_activity"] is None
        assert d["last_activity"] is None


class TestActivityStoreInit:
    """Tests for ActivityStore initialization."""

    def test_creates_database(self, temp_db):
        """Creates database file."""
        store = ActivityStore(db_path=temp_db)
        assert temp_db.exists()

    def test_creates_tables(self, store):
        """Creates required tables."""
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]

        assert "activities" in table_names


class TestActivityStoreAdd:
    """Tests for add method."""

    def test_add_record(self, store):
        """Add single record."""
        record_id = store.add(
            watcher_type="window_watcher",
            timestamp=datetime.now(),
            data={"app": "code"},
        )

        assert record_id is not None
        assert record_id > 0

    def test_add_multiple_records(self, store):
        """Add multiple records."""
        id1 = store.add("window_watcher", datetime.now(), {"app": "code"})
        id2 = store.add("window_watcher", datetime.now(), {"app": "firefox"})

        assert id2 > id1


class TestActivityStoreAddBatch:
    """Tests for add_batch method."""

    def test_add_batch(self, store):
        """Add batch of records."""
        records = [
            ("window_watcher", datetime.now(), {"app": f"app{i}"})
            for i in range(10)
        ]

        count = store.add_batch(records)

        assert count == 10

    def test_add_empty_batch(self, store):
        """Add empty batch."""
        count = store.add_batch([])
        assert count == 0


class TestActivityStoreQuery:
    """Tests for query method."""

    def test_query_all(self, store):
        """Query all records."""
        for i in range(5):
            store.add("test", datetime.now(), {"num": i})

        results = store.query(limit=100)

        assert len(results) == 5

    def test_query_by_watcher_type(self, store):
        """Query by watcher type."""
        store.add("window_watcher", datetime.now(), {"app": "code"})
        store.add("afk_watcher", datetime.now(), {"event": "start"})
        store.add("window_watcher", datetime.now(), {"app": "firefox"})

        results = store.query(watcher_type="window_watcher")

        assert len(results) == 2
        assert all(r.watcher_type == "window_watcher" for r in results)

    def test_query_by_time_range(self, store):
        """Query by time range."""
        now = datetime.now()
        store.add("test", now - timedelta(hours=2), {"old": True})
        store.add("test", now, {"old": False})

        results = store.query(start_time=now - timedelta(hours=1))

        assert len(results) == 1
        assert results[0].data["old"] is False

    def test_query_with_limit_offset(self, store):
        """Query with limit and offset."""
        for i in range(10):
            store.add("test", datetime.now() + timedelta(seconds=i), {"num": i})

        results = store.query(limit=3, offset=2)

        assert len(results) == 3


class TestActivityStoreGetRecent:
    """Tests for get_recent method."""

    def test_get_recent(self, store):
        """Get recent records."""
        now = datetime.now()
        store.add("test", now - timedelta(hours=48), {"old": True})
        store.add("test", now - timedelta(hours=12), {"recent": True})
        store.add("test", now, {"now": True})

        results = store.get_recent(hours=24)

        assert len(results) == 2


class TestActivityStoreGetSummary:
    """Tests for get_summary method."""

    def test_get_summary(self, store):
        """Get summary by watcher type."""
        now = datetime.now()
        for i in range(5):
            store.add("window_watcher", now - timedelta(minutes=i), {"app": "code"})
        for i in range(3):
            store.add("afk_watcher", now - timedelta(minutes=i), {"event": "start"})

        summary = store.get_summary()

        assert "window_watcher" in summary
        assert summary["window_watcher"].record_count == 5
        assert "afk_watcher" in summary
        assert summary["afk_watcher"].record_count == 3

    def test_get_summary_with_time_range(self, store):
        """Get summary within time range."""
        now = datetime.now()
        store.add("test", now - timedelta(days=2), {"old": True})
        store.add("test", now, {"new": True})

        summary = store.get_summary(start_time=now - timedelta(days=1))

        assert "test" in summary
        assert summary["test"].record_count == 1


class TestAggregateWindowTime:
    """Tests for aggregate_window_time method."""

    def test_aggregate_window_time(self, store):
        """Aggregate time by app."""
        now = datetime.now()
        store.add("window_watcher", now, {"app_name": "code", "focus_duration": 600})
        store.add("window_watcher", now, {"app_name": "code", "focus_duration": 300})
        store.add("window_watcher", now, {"app_name": "firefox", "focus_duration": 120})

        result = store.aggregate_window_time()

        assert result["code"] == 900
        assert result["firefox"] == 120

    def test_aggregate_window_time_handles_missing_data(self, store):
        """Handles records without focus_duration."""
        now = datetime.now()
        store.add("window_watcher", now, {"app_name": "code"})

        result = store.aggregate_window_time()

        assert result.get("code", 0) == 0


class TestAggregateDomainTime:
    """Tests for aggregate_domain_time method."""

    def test_aggregate_domain_time(self, store):
        """Aggregate time by domain."""
        now = datetime.now()
        store.add(
            "browser_watcher",
            now,
            {
                "domains": [
                    {"domain": "github.com", "time_spent": 300},
                    {"domain": "stackoverflow.com", "time_spent": 200},
                ]
            },
        )
        store.add(
            "browser_watcher",
            now,
            {
                "domains": [
                    {"domain": "github.com", "time_spent": 100},
                ]
            },
        )

        result = store.aggregate_domain_time()

        assert result["github.com"] == 400
        assert result["stackoverflow.com"] == 200


class TestActivityStoreCleanup:
    """Tests for cleanup method."""

    def test_cleanup_old_records(self, store):
        """Remove old records."""
        now = datetime.now()
        # Add old record (10 days ago)
        store.add("test", now - timedelta(days=10), {"old": True})
        # Add recent record
        store.add("test", now, {"recent": True})

        deleted = store.cleanup(retention_days=7)

        # Old record should be deleted
        assert deleted >= 1

        results = store.query()
        assert len(results) == 1
        assert results[0].data["recent"] is True


class TestActivityStoreGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self, store):
        """Get store statistics."""
        for i in range(10):
            store.add("window_watcher", datetime.now(), {"num": i})
        for i in range(5):
            store.add("afk_watcher", datetime.now(), {"num": i})

        stats = store.get_stats()

        assert stats["total_records"] == 15
        assert stats["watchers"]["window_watcher"] == 10
        assert stats["watchers"]["afk_watcher"] == 5
        assert stats["db_size_bytes"] > 0
        assert "db_size_mb" in stats
        assert "retention_days" in stats


class TestGetActivityStoreSingleton:
    """Tests for get_activity_store singleton function."""

    def test_returns_same_instance(self):
        """Returns same instance."""
        # Note: This uses the actual global store
        store1 = get_activity_store()
        store2 = get_activity_store()
        assert store1 is store2
