"""Tests for collectors base module."""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict, Optional

from collectors.base import Heartbeat, BaseWatcher


class TestHeartbeat:
    """Tests for Heartbeat dataclass."""

    def test_creation(self):
        """Create heartbeat with required fields."""
        hb = Heartbeat(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            watcher_type="test",
        )
        assert hb.timestamp == datetime(2025, 1, 15, 10, 0, 0)
        assert hb.watcher_type == "test"
        assert hb.data == {}

    def test_with_data(self):
        """Create heartbeat with data."""
        hb = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="window",
            data={"app": "code", "title": "test.py"},
        )
        assert hb.data["app"] == "code"
        assert hb.data["title"] == "test.py"

    def test_to_dict(self):
        """Convert heartbeat to dict."""
        hb = Heartbeat(
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            watcher_type="test",
            data={"key": "value"},
        )
        d = hb.to_dict()

        assert d["timestamp"] == "2025-01-15T10:30:00"
        assert d["watcher_type"] == "test"
        assert d["data"] == {"key": "value"}


class ConcreteWatcher(BaseWatcher):
    """Concrete implementation for testing."""

    name = "test_watcher"
    version = "1.0.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collect_count = 0
        self.collect_returns: list = []

    async def collect(self) -> Optional[Heartbeat]:
        """Return configured heartbeats."""
        self.collect_count += 1
        if self.collect_returns:
            return self.collect_returns.pop(0)
        return Heartbeat(
            timestamp=datetime.now(),
            watcher_type=self.name,
            data={"count": self.collect_count},
        )

    def get_config(self) -> Dict[str, Any]:
        """Return test config."""
        return {"batch_interval": self.batch_interval}


class TestBaseWatcherInit:
    """Tests for BaseWatcher initialization."""

    def test_default_batch_interval(self):
        """Default batch interval is 30 seconds."""
        watcher = ConcreteWatcher()
        assert watcher.batch_interval == 30.0

    def test_custom_batch_interval(self):
        """Custom batch interval is respected."""
        watcher = ConcreteWatcher(batch_interval=60.0)
        assert watcher.batch_interval == 60.0

    def test_initial_state(self):
        """Initial state is not running."""
        watcher = ConcreteWatcher()
        assert not watcher.running
        assert len(watcher.heartbeats) == 0
        assert watcher._last_heartbeat is None


class TestAddHeartbeat:
    """Tests for _add_heartbeat method."""

    def test_adds_heartbeat(self):
        """Adds heartbeat to buffer."""
        watcher = ConcreteWatcher()
        hb = Heartbeat(timestamp=datetime.now(), watcher_type="test", data={"a": 1})

        watcher._add_heartbeat(hb)

        assert len(watcher.heartbeats) == 1
        assert watcher._last_heartbeat == hb

    def test_merges_identical_data(self):
        """Merges heartbeats with identical data."""
        watcher = ConcreteWatcher()
        hb1 = Heartbeat(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            watcher_type="test",
            data={"key": "value"},
        )
        hb2 = Heartbeat(
            timestamp=datetime(2025, 1, 15, 10, 1, 0),
            watcher_type="test",
            data={"key": "value"},
        )

        watcher._add_heartbeat(hb1)
        watcher._add_heartbeat(hb2)

        # Should only have one heartbeat with updated timestamp
        assert len(watcher.heartbeats) == 1
        assert watcher._last_heartbeat.timestamp == hb2.timestamp

    def test_does_not_merge_different_data(self):
        """Does not merge heartbeats with different data."""
        watcher = ConcreteWatcher()
        hb1 = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value1"},
        )
        hb2 = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value2"},
        )

        watcher._add_heartbeat(hb1)
        watcher._add_heartbeat(hb2)

        assert len(watcher.heartbeats) == 2

    def test_buffer_overflow_protection(self):
        """Trims buffer when exceeds 1000 heartbeats."""
        watcher = ConcreteWatcher()

        # Add exactly 1001 to trigger trim
        for i in range(1001):
            hb = Heartbeat(
                timestamp=datetime.now(),
                watcher_type="test",
                data={"count": i},  # Different data each time
            )
            watcher._add_heartbeat(hb)

        # Should have trimmed to 500
        assert len(watcher.heartbeats) == 500
        # Should keep most recent 500
        assert watcher.heartbeats[-1].data["count"] == 1000


class TestShouldMerge:
    """Tests for _should_merge method."""

    def test_no_previous_heartbeat(self):
        """Returns False when no previous heartbeat."""
        watcher = ConcreteWatcher()
        hb = Heartbeat(timestamp=datetime.now(), watcher_type="test")

        assert not watcher._should_merge(hb)

    def test_identical_data_merges(self):
        """Returns True for identical data."""
        watcher = ConcreteWatcher()
        watcher._last_heartbeat = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value"},
        )

        hb = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value"},
        )

        assert watcher._should_merge(hb)

    def test_different_data_no_merge(self):
        """Returns False for different data."""
        watcher = ConcreteWatcher()
        watcher._last_heartbeat = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value1"},
        )

        hb = Heartbeat(
            timestamp=datetime.now(),
            watcher_type="test",
            data={"key": "value2"},
        )

        assert not watcher._should_merge(hb)


class TestFlush:
    """Tests for flush method."""

    @pytest.mark.asyncio
    async def test_returns_heartbeats(self):
        """Returns and clears heartbeats."""
        watcher = ConcreteWatcher()
        hb1 = Heartbeat(timestamp=datetime.now(), watcher_type="test", data={"a": 1})
        hb2 = Heartbeat(timestamp=datetime.now(), watcher_type="test", data={"b": 2})
        watcher._add_heartbeat(hb1)
        watcher._add_heartbeat(hb2)

        flushed = await watcher.flush()

        assert len(flushed) == 2
        assert len(watcher.heartbeats) == 0

    @pytest.mark.asyncio
    async def test_empty_flush(self):
        """Returns empty list when no heartbeats."""
        watcher = ConcreteWatcher()
        flushed = await watcher.flush()

        assert flushed == []


class TestGetStatus:
    """Tests for get_status method."""

    def test_returns_status_dict(self):
        """Returns status dictionary."""
        watcher = ConcreteWatcher(batch_interval=45.0)
        hb = Heartbeat(timestamp=datetime.now(), watcher_type="test")
        watcher._add_heartbeat(hb)

        status = watcher.get_status()

        assert status["name"] == "test_watcher"
        assert status["version"] == "1.0.0"
        assert status["running"] is False
        assert status["pending_heartbeats"] == 1
        assert status["config"]["batch_interval"] == 45.0


class TestGetCollectionInterval:
    """Tests for get_collection_interval method."""

    def test_default_interval(self):
        """Default interval is 1 second."""
        watcher = ConcreteWatcher()
        assert watcher.get_collection_interval() == 1.0


class TestStartStop:
    """Tests for start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Start sets running flag and creates tasks."""
        watcher = ConcreteWatcher()

        # Start in background, stop quickly
        async def run_briefly():
            task = asyncio.create_task(watcher.start())
            await asyncio.sleep(0.1)
            await watcher.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run_briefly()

        # Should have collected at least once
        assert watcher.collect_count >= 1

    @pytest.mark.asyncio
    async def test_start_warns_if_running(self, caplog):
        """Start warns if already running."""
        watcher = ConcreteWatcher()
        watcher.running = True

        await watcher.start()

        # Should have logged warning
        assert "already running" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_stop_flushes(self):
        """Stop flushes remaining heartbeats."""
        watcher = ConcreteWatcher()
        hb = Heartbeat(timestamp=datetime.now(), watcher_type="test", data={"x": 1})
        watcher._add_heartbeat(hb)
        watcher.running = True

        await watcher.stop()

        assert not watcher.running
        # Heartbeats should be cleared after flush
        assert len(watcher.heartbeats) == 0


class TestCollectionLoop:
    """Tests for _collection_loop method."""

    @pytest.mark.asyncio
    async def test_collects_heartbeats(self):
        """Collection loop collects heartbeats."""
        watcher = ConcreteWatcher()
        watcher.running = True

        # Run collection loop briefly
        task = asyncio.create_task(watcher._collection_loop())
        await asyncio.sleep(0.05)
        watcher.running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert watcher.collect_count >= 1
        assert len(watcher.heartbeats) >= 1

    @pytest.mark.asyncio
    async def test_handles_none_return(self):
        """Collection loop handles None returns."""
        watcher = ConcreteWatcher()
        watcher.collect_returns = [None, None, None]
        watcher.running = True

        task = asyncio.create_task(watcher._collection_loop())
        await asyncio.sleep(0.05)
        watcher.running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have called collect but not added heartbeats
        assert watcher.collect_count >= 1
        assert len(watcher.heartbeats) == 0
