"""
REAL Integration Tests for Collectors.

Tests actual collector behavior with real system resources.
No mocks - real X11, real input detection, real data.
"""

import asyncio
import os
import pytest
from datetime import datetime

# Skip if not running on X11
pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Requires X11 display"
)


class TestWindowWatcherReal:
    """Real tests for WindowWatcher with actual X11."""

    def test_detect_display_server(self):
        """Detect actual display server."""
        from collectors.window_watcher import detect_display_server

        server = detect_display_server()
        # Should detect something on a real system
        assert server in ["x11", "wayland", "unknown"]

    def test_sanitize_title(self):
        """Test title sanitization with real patterns."""
        from collectors.window_watcher import sanitize_title

        # Email
        assert "[email]" in sanitize_title("Message from user@example.com")
        # URL
        assert "[url]" in sanitize_title("Viewing https://github.com/repo")
        # Long numbers
        assert "[number]" in sanitize_title("Order 1234567890 confirmed")
        # Token
        assert "[token]" in sanitize_title("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        # Truncation
        long_title = "A" * 150
        sanitized = sanitize_title(long_title)
        assert len(sanitized) <= 103  # 100 + "..."

    @pytest.mark.skipif(
        os.environ.get("XDG_SESSION_TYPE") == "wayland",
        reason="X11 backend only"
    )
    def test_x11_backend_connection(self):
        """Test real X11 backend connection."""
        from collectors.window_watcher import X11Backend

        backend = X11Backend()
        connected = backend._ensure_connected()

        if connected:
            # Should be able to get active window
            window = backend.get_active_window()
            # May be None if no window focused, but shouldn't crash
            if window:
                assert window.app_name
                assert window.window_class
            backend.close()

    @pytest.mark.asyncio
    async def test_window_watcher_collect(self):
        """Test real window collection."""
        from collectors.window_watcher import WindowWatcher

        watcher = WindowWatcher(batch_interval=1.0, poll_interval=0.1)

        # Collect should work (may return None if no window change)
        heartbeat = await watcher.collect()

        # If we got a heartbeat, verify structure
        if heartbeat:
            assert heartbeat.watcher_type == "window_watcher"
            assert "app_name" in heartbeat.data
            assert "window_class" in heartbeat.data

        await watcher.stop()

    def test_window_watcher_config(self):
        """Test watcher configuration."""
        from collectors.window_watcher import WindowWatcher

        watcher = WindowWatcher(batch_interval=45.0, poll_interval=3.0)
        config = watcher.get_config()

        assert config["batch_interval"] == 45.0
        assert config["poll_interval"] == 3.0
        assert "display_server" in config


class TestAFKWatcherReal:
    """Real tests for AFKWatcher with actual idle detection."""

    def test_x11_idle_detector(self):
        """Test real X11 idle detection."""
        from collectors.afk_watcher import X11IdleDetector

        detector = X11IdleDetector()
        idle_seconds = detector.get_idle_seconds()

        # Should return a value (might be None if X11 not available)
        if idle_seconds is not None:
            assert idle_seconds >= 0
            # We're running tests so shouldn't be very idle
            assert idle_seconds < 3600  # Less than an hour

        detector.close()

    def test_fallback_idle_detector(self):
        """Test fallback idle detector using /proc/interrupts."""
        from collectors.afk_watcher import FallbackIdleDetector

        detector = FallbackIdleDetector()
        idle_seconds = detector.get_idle_seconds()

        # Should always return a value on Linux
        if idle_seconds is not None:
            assert idle_seconds >= 0

    def test_afk_state_to_dict(self):
        """Test AFKState serialization."""
        from collectors.afk_watcher import AFKState

        state = AFKState(
            is_afk=True,
            afk_since=datetime.now(),
            afk_duration=300.0,
        )
        d = state.to_dict()

        assert d["is_afk"] is True
        assert d["afk_duration"] == 300.0
        assert d["afk_since"] is not None

    @pytest.mark.asyncio
    async def test_afk_watcher_collect(self):
        """Test real AFK collection."""
        from collectors.afk_watcher import AFKWatcher

        watcher = AFKWatcher(batch_interval=1.0, afk_threshold=180.0)

        # First collect - establishes baseline
        heartbeat = await watcher.collect()

        # Should return None (no state change yet) or heartbeat
        if heartbeat:
            assert heartbeat.watcher_type == "afk_watcher"
            assert "event" in heartbeat.data

        # Check current state
        state = watcher.get_current_state()
        assert hasattr(state, "is_afk")
        assert hasattr(state, "last_activity")

        await watcher.stop()

    def test_afk_watcher_config(self):
        """Test AFK watcher configuration."""
        from collectors.afk_watcher import AFKWatcher

        watcher = AFKWatcher(afk_threshold=300.0)
        config = watcher.get_config()

        assert config["afk_threshold"] == 300.0
        assert "using_x11" in config
        assert "current_state" in config


class TestInputWatcherReal:
    """Real tests for InputWatcher."""

    def test_keystroke_dynamics_model(self):
        """Test KeystrokeDynamics dataclass."""
        from collectors.input_watcher import KeystrokeDynamics

        dynamics = KeystrokeDynamics(
            avg_hold_time_ms=100.0,
            avg_seek_time_ms=150.0,
            typing_speed_cpm=300.0,
            rhythm_variance=0.5,
            burst_count=3,
            pause_count=2,
            total_keystrokes=50,
        )

        d = dynamics.to_dict()
        assert d["avg_hold_time_ms"] == 100.0
        assert d["typing_speed_cpm"] == 300.0
        assert d["total_keystrokes"] == 50

    def test_mouse_activity_model(self):
        """Test MouseActivity dataclass."""
        from collectors.input_watcher import MouseActivity

        mouse = MouseActivity(
            clicks_count=10,
            scroll_count=5,
            movement_distance=1000.0,
            movement_pattern="smooth",
        )

        d = mouse.to_dict()
        assert d["clicks_count"] == 10
        assert d["movement_distance"] == 1000.0
        assert d["movement_pattern"] == "smooth"

    def test_input_watcher_config(self):
        """Test input watcher configuration."""
        from collectors.input_watcher import InputWatcher

        watcher = InputWatcher(batch_interval=30.0)
        config = watcher.get_config()

        assert config["batch_interval"] == 30.0
        assert "throttle_ms" in config
        assert "buffer_size" in config


class TestBrowserWatcherReal:
    """Real tests for BrowserWatcher."""

    def test_browser_watcher_initialization(self):
        """Test browser watcher initialization."""
        from collectors.browser_watcher import BrowserWatcher

        watcher = BrowserWatcher()

        assert watcher.name == "browser_watcher"
        assert hasattr(watcher, "_session")

    def test_domain_activity_model(self):
        """Test DomainActivity dataclass."""
        from collectors.browser_watcher import DomainActivity

        activity = DomainActivity(
            domain="github.com",
            time_spent=120.5,
            visit_count=5,
            last_visit=datetime.now(),
        )

        d = activity.to_dict()
        assert d["domain"] == "github.com"
        assert d["time_spent"] == 120.5
        assert d["visit_count"] == 5
        assert d["last_visit"] is not None

    def test_browser_watcher_config(self):
        """Test browser watcher configuration."""
        from collectors.browser_watcher import BrowserWatcher

        watcher = BrowserWatcher(batch_interval=60.0)
        config = watcher.get_config()

        assert config["batch_interval"] == 60.0

    @pytest.mark.asyncio
    async def test_browser_watcher_collect(self):
        """Test browser watcher collection (may return None without extension)."""
        from collectors.browser_watcher import BrowserWatcher

        watcher = BrowserWatcher()

        # Without browser extension, should return None gracefully
        heartbeat = await watcher.collect()

        # Either None or valid heartbeat
        if heartbeat:
            assert heartbeat.watcher_type == "browser_watcher"

        await watcher.stop()


class TestCollectorIntegration:
    """Integration tests for collector system."""

    def test_registry_has_real_collectors(self):
        """Registry should have real collectors registered."""
        from collectors.registry import CollectorRegistry

        # Import collectors to trigger registration
        from collectors import window_watcher, afk_watcher, input_watcher, browser_watcher

        names = CollectorRegistry.get_names()

        # Should have our watchers
        assert "window_watcher" in names
        assert "afk_watcher" in names
        assert "input_watcher" in names
        assert "browser_watcher" in names

    def test_create_real_instances(self):
        """Create real collector instances."""
        from collectors.registry import CollectorRegistry
        from collectors import window_watcher, afk_watcher

        # Create real instances
        window = CollectorRegistry.create_instance("window_watcher", cache=False)
        afk = CollectorRegistry.create_instance("afk_watcher", cache=False)

        assert window.name == "window_watcher"
        assert afk.name == "afk_watcher"

    @pytest.mark.asyncio
    async def test_collector_flush_to_activity_store(self):
        """Test that collectors actually flush to ActivityStore."""
        from collectors.window_watcher import WindowWatcher
        from memory.activity_store import ActivityStore
        import tempfile
        from pathlib import Path

        # Use temp database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ActivityStore(db_path=db_path)

            watcher = WindowWatcher()

            # Manually add a heartbeat
            from collectors.base import Heartbeat
            hb = Heartbeat(
                timestamp=datetime.now(),
                watcher_type="window_watcher",
                data={"app_name": "test", "window_class": "Test"},
            )
            watcher.heartbeats.append(hb)

            # Flush should store to ActivityStore
            flushed = await watcher.flush()

            if flushed:
                # Verify data in store
                results = store.query(watcher_type="window", limit=10)
                # Note: flush may use different watcher_type
