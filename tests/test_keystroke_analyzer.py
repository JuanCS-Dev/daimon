"""Tests for KeystrokeAnalyzer class."""

import pytest
import time
from learners.keystroke_analyzer import (
    KeystrokeAnalyzer,
    get_keystroke_analyzer,
    reset_keystroke_analyzer,
)
from learners.keystroke_models import CognitiveState


class TestKeystrokeAnalyzerInit:
    """Tests for KeystrokeAnalyzer initialization."""

    def test_default_values(self):
        """Default initialization values."""
        analyzer = KeystrokeAnalyzer()
        assert analyzer.window_size == 300
        assert analyzer.min_events == 20

    def test_custom_values(self):
        """Custom initialization values."""
        analyzer = KeystrokeAnalyzer(window_size=100, min_events=10)
        assert analyzer.window_size == 100
        assert analyzer.min_events == 10

    def test_empty_state(self):
        """Analyzer starts empty."""
        analyzer = KeystrokeAnalyzer()
        stats = analyzer.get_stats()
        assert stats["event_count"] == 0
        assert stats["pending_presses"] == 0


class TestAddEvent:
    """Tests for add_event method."""

    def test_add_press_event(self):
        """Add press event."""
        analyzer = KeystrokeAnalyzer()
        analyzer.add_event(key="a", event_type="press", timestamp=1.0)

        stats = analyzer.get_stats()
        assert stats["event_count"] == 1
        assert stats["pending_presses"] == 1

    def test_add_release_event(self):
        """Add release event clears pending press."""
        analyzer = KeystrokeAnalyzer()
        analyzer.add_event(key="a", event_type="press", timestamp=1.0)
        analyzer.add_event(key="a", event_type="release", timestamp=1.1)

        stats = analyzer.get_stats()
        assert stats["event_count"] == 2
        assert stats["pending_presses"] == 0

    def test_auto_timestamp(self):
        """Auto-generates timestamp if not provided."""
        analyzer = KeystrokeAnalyzer()
        before = time.time()
        analyzer.add_event(key="a", event_type="press")
        after = time.time()

        events = analyzer.get_recent_events(1)
        assert len(events) == 1
        assert before <= events[0].timestamp <= after

    def test_modifiers(self):
        """Stores modifier keys."""
        analyzer = KeystrokeAnalyzer()
        analyzer.add_event(
            key="a", event_type="press", timestamp=1.0, modifiers=["shift", "ctrl"]
        )

        events = analyzer.get_recent_events(1)
        assert events[0].modifiers == ["shift", "ctrl"]

    def test_invalidates_cache(self):
        """Adding event invalidates biometrics cache."""
        analyzer = KeystrokeAnalyzer()
        # Add enough events for analysis
        for i in range(25):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) + 0.1)

        # Compute biometrics to cache them
        analyzer.compute_biometrics()
        assert analyzer._cache_valid

        # Add new event
        analyzer.add_event(key="b", event_type="press", timestamp=100.0)
        assert not analyzer._cache_valid

    def test_window_size_limit(self):
        """Events are limited by window size."""
        analyzer = KeystrokeAnalyzer(window_size=10)
        for i in range(20):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        stats = analyzer.get_stats()
        assert stats["event_count"] == 10


class TestComputeBiometrics:
    """Tests for compute_biometrics method."""

    def test_insufficient_data(self):
        """Returns default biometrics with insufficient data."""
        analyzer = KeystrokeAnalyzer(min_events=20)
        for i in range(5):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        biometrics = analyzer.compute_biometrics()
        assert biometrics.avg_hold_time == 0.0
        assert biometrics.avg_seek_time == 0.0

    def test_computes_hold_times(self):
        """Computes average hold time."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) + 0.1)

        biometrics = analyzer.compute_biometrics()
        assert abs(biometrics.avg_hold_time - 0.1) < 0.01

    def test_computes_seek_times(self):
        """Computes average seek time."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i) * 0.2)
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) * 0.2 + 0.05)

        biometrics = analyzer.compute_biometrics()
        # Seek time between presses: 0.2 seconds (but first is 0.15 due to hold time)
        assert biometrics.avg_seek_time > 0.0

    def test_caches_results(self):
        """Caches computed biometrics."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) + 0.1)

        biometrics1 = analyzer.compute_biometrics()
        biometrics2 = analyzer.compute_biometrics()

        # Should be the same cached instance
        assert biometrics1 is biometrics2
        assert analyzer._cache_valid

    def test_computes_typing_speed(self):
        """Computes typing speed in KPM."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        # 10 keys in 1 second = 600 KPM
        for i in range(12):
            analyzer.add_event(key="a", event_type="press", timestamp=i * 0.1)
            analyzer.add_event(key="a", event_type="release", timestamp=i * 0.1 + 0.05)

        biometrics = analyzer.compute_biometrics()
        assert biometrics.typing_speed > 0


class TestDetectCognitiveState:
    """Tests for detect_cognitive_state method."""

    def test_idle_with_insufficient_data(self):
        """Returns idle state with insufficient data."""
        analyzer = KeystrokeAnalyzer(min_events=20)
        for i in range(5):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        state = analyzer.detect_cognitive_state()
        assert state.state == "idle"
        assert state.confidence == 0.0

    def test_returns_cognitive_state(self):
        """Returns CognitiveState with biometrics."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i) * 0.1)
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) * 0.1 + 0.05)

        state = analyzer.detect_cognitive_state()
        assert isinstance(state, CognitiveState)
        assert state.biometrics is not None
        assert state.state in ["flow", "focused", "distracted", "fatigued", "stressed", "idle"]

    def test_to_dict(self):
        """CognitiveState converts to dict."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i) * 0.1)
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) * 0.1 + 0.05)

        state = analyzer.detect_cognitive_state()
        d = state.to_dict()

        assert "state" in d
        assert "confidence" in d
        assert "detected_at" in d
        assert "biometrics" in d


class TestGetRecentEvents:
    """Tests for get_recent_events method."""

    def test_returns_requested_count(self):
        """Returns requested number of events."""
        analyzer = KeystrokeAnalyzer()
        for i in range(20):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        events = analyzer.get_recent_events(5)
        assert len(events) == 5

    def test_returns_all_if_fewer(self):
        """Returns all events if fewer than requested."""
        analyzer = KeystrokeAnalyzer()
        for i in range(3):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        events = analyzer.get_recent_events(10)
        assert len(events) == 3

    def test_returns_most_recent(self):
        """Returns most recent events."""
        analyzer = KeystrokeAnalyzer()
        for i in range(10):
            analyzer.add_event(key=f"key{i}", event_type="press", timestamp=float(i))

        events = analyzer.get_recent_events(3)
        assert events[-1].key == "key9"
        assert events[0].key == "key7"


class TestClear:
    """Tests for clear method."""

    def test_clears_events(self):
        """Clears all events."""
        analyzer = KeystrokeAnalyzer()
        for i in range(10):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        analyzer.clear()

        stats = analyzer.get_stats()
        assert stats["event_count"] == 0

    def test_clears_pending_presses(self):
        """Clears pending presses."""
        analyzer = KeystrokeAnalyzer()
        analyzer.add_event(key="a", event_type="press", timestamp=1.0)
        analyzer.add_event(key="b", event_type="press", timestamp=1.1)

        analyzer.clear()

        stats = analyzer.get_stats()
        assert stats["pending_presses"] == 0

    def test_invalidates_cache(self):
        """Invalidates biometrics cache."""
        analyzer = KeystrokeAnalyzer(min_events=5)
        for i in range(10):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) + 0.1)

        analyzer.compute_biometrics()
        assert analyzer._cache_valid

        analyzer.clear()
        assert not analyzer._cache_valid


class TestGetStats:
    """Tests for get_stats method."""

    def test_returns_all_fields(self):
        """Returns all expected fields."""
        analyzer = KeystrokeAnalyzer(min_events=10)
        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))
            analyzer.add_event(key="a", event_type="release", timestamp=float(i) + 0.1)

        stats = analyzer.get_stats()

        assert "event_count" in stats
        assert "window_size" in stats
        assert "min_events" in stats
        assert "has_enough_data" in stats
        assert "pending_presses" in stats
        assert "biometrics" in stats

    def test_has_enough_data_flag(self):
        """has_enough_data flag is correct."""
        analyzer = KeystrokeAnalyzer(min_events=10)

        stats = analyzer.get_stats()
        assert not stats["has_enough_data"]

        for i in range(15):
            analyzer.add_event(key="a", event_type="press", timestamp=float(i))

        stats = analyzer.get_stats()
        assert stats["has_enough_data"]


class TestSingleton:
    """Tests for singleton functions."""

    def test_get_returns_same_instance(self):
        """get_keystroke_analyzer returns same instance."""
        reset_keystroke_analyzer()
        analyzer1 = get_keystroke_analyzer()
        analyzer2 = get_keystroke_analyzer()
        assert analyzer1 is analyzer2

    def test_reset_creates_new_instance(self):
        """reset_keystroke_analyzer creates new instance."""
        reset_keystroke_analyzer()
        analyzer1 = get_keystroke_analyzer()
        analyzer1.add_event(key="a", event_type="press", timestamp=1.0)

        reset_keystroke_analyzer()
        analyzer2 = get_keystroke_analyzer()

        assert analyzer1 is not analyzer2
        assert analyzer2.get_stats()["event_count"] == 0
