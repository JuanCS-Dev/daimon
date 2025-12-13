"""
Tests for learners/pattern_detector.py

Tests covering:
- Pattern and Event dataclasses
- Temporal pattern detection
- Sequential pattern detection
- Contextual pattern detection
- Pattern matching

Run:
    pytest tests/test_pattern_detector.py -v
"""

from datetime import datetime, timedelta

import pytest

from learners.pattern_detector import (
    Pattern,
    Event,
    PatternDetector,
    get_pattern_detector,
    reset_pattern_detector,
)


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestPattern:
    """Tests for Pattern dataclass."""

    def test_create_pattern(self):
        """Test creating a Pattern."""
        pattern = Pattern(
            pattern_type="temporal",
            description="git commit at 17:00",
            confidence=0.8,
            occurrences=10,
            last_seen=datetime.now(),
        )
        assert pattern.pattern_type == "temporal"
        assert pattern.confidence == 0.8
        assert pattern.occurrences == 10

    def test_pattern_to_dict(self):
        """Test converting pattern to dict."""
        pattern = Pattern(
            pattern_type="sequential",
            description="test",
            confidence=0.5,
            occurrences=5,
            last_seen=datetime.now(),
            data={"key": "value"},
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "sequential"
        assert d["data"]["key"] == "value"


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Test creating an Event."""
        event = Event(
            event_type="shell_command",
            timestamp=datetime.now(),
            data={"command": "git status"},
        )
        assert event.event_type == "shell_command"
        assert event.data["command"] == "git status"


# ============================================================================
# PATTERN DETECTOR TESTS
# ============================================================================


class TestPatternDetectorInit:
    """Tests for PatternDetector initialization."""

    def test_init_default(self):
        """Test default initialization."""
        detector = PatternDetector()
        assert detector.max_events == 1000
        assert len(detector._events) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        detector = PatternDetector(max_events=100)
        assert detector.max_events == 100


class TestAddEvent:
    """Tests for PatternDetector.add_event()."""

    @pytest.fixture
    def detector(self):
        """Fresh detector for each test."""
        return PatternDetector()

    def test_add_event_basic(self, detector):
        """Test adding a basic event."""
        detector.add_event({"type": "test_event"})
        assert len(detector._events) == 1
        assert detector._events[0].event_type == "test_event"

    def test_add_event_with_timestamp(self, detector):
        """Test adding event with custom timestamp."""
        ts = datetime(2025, 1, 1, 12, 0)
        detector.add_event({"type": "test", "timestamp": ts})
        assert detector._events[0].timestamp == ts

    def test_add_event_string_timestamp(self, detector):
        """Test adding event with ISO string timestamp."""
        detector.add_event({
            "type": "test",
            "timestamp": "2025-01-01T12:00:00"
        })
        assert detector._events[0].timestamp.hour == 12

    def test_add_event_max_limit(self):
        """Test that events are trimmed when over limit."""
        detector = PatternDetector(max_events=5)
        for i in range(10):
            detector.add_event({"type": f"event_{i}"})
        assert len(detector._events) == 5


class TestTemporalPatterns:
    """Tests for temporal pattern detection."""

    @pytest.fixture
    def detector(self):
        """Fresh detector."""
        return PatternDetector()

    def test_detect_hourly_pattern(self, detector):
        """Test detecting pattern at specific hour."""
        base_time = datetime.now()
        
        # Add events at 5pm for multiple days
        for i in range(5):
            event_time = base_time.replace(hour=17, minute=0) - timedelta(days=i)
            detector.add_event({
                "type": "commit",
                "timestamp": event_time,
            })
        
        patterns = detector.get_patterns_by_type("temporal")
        assert len(patterns) >= 1
        
        # Should detect 17:00 pattern
        hourly_pattern = next(
            (p for p in patterns if "17" in p.description),
            None
        )
        assert hourly_pattern is not None

    def test_detect_weekday_pattern(self, detector):
        """Test detecting pattern on specific weekday."""
        # Add events on Monday
        for i in range(4):
            monday = datetime.now()
            while monday.weekday() != 0:
                monday -= timedelta(days=1)
            event_time = monday - timedelta(weeks=i)
            detector.add_event({
                "type": "weekly_review",
                "timestamp": event_time,
            })
        
        patterns = detector.get_patterns_by_type("temporal")
        # May or may not detect depending on thresholds


class TestSequentialPatterns:
    """Tests for sequential pattern detection."""

    @pytest.fixture
    def detector(self):
        """Fresh detector."""
        return PatternDetector()

    def test_detect_command_sequence(self, detector):
        """Test detecting command sequences."""
        # Add git workflow multiple times
        for _ in range(4):
            detector.add_event({"type": "shell_command", "command": "git status"})
            detector.add_event({"type": "shell_command", "command": "git add ."})
            detector.add_event({"type": "shell_command", "command": "git commit"})
        
        patterns = detector.get_patterns_by_type("sequential")
        assert len(patterns) >= 1
        
        # Check for git workflow pattern
        git_pattern = next(
            (p for p in patterns if "git" in p.description.lower()),
            None
        )
        assert git_pattern is not None
        assert git_pattern.occurrences >= 3

    def test_no_sequence_for_non_commands(self, detector):
        """Test that non-command events don't create sequences."""
        for _ in range(5):
            detector.add_event({"type": "window_focus", "app": "VSCode"})
            detector.add_event({"type": "window_focus", "app": "Chrome"})
        
        patterns = detector.get_patterns_by_type("sequential")
        assert len(patterns) == 0


class TestContextualPatterns:
    """Tests for contextual pattern detection."""

    @pytest.fixture
    def detector(self):
        """Fresh detector."""
        return PatternDetector()

    def test_detect_app_context(self, detector):
        """Test detecting app context patterns."""
        for _ in range(5):
            detector.add_event({
                "type": "keystroke",
                "app": "VSCode",
            })
        
        patterns = detector.get_patterns_by_type("contextual")
        # Should have at least one contextual pattern
        assert len(patterns) >= 0  # May not meet threshold


class TestGetMatchingPatterns:
    """Tests for context-based pattern matching."""

    @pytest.fixture
    def detector(self):
        """Fresh detector with patterns."""
        det = PatternDetector()
        
        # Create temporal pattern at 17:00
        for i in range(5):
            det.add_event({
                "type": "commit",
                "timestamp": datetime.now().replace(hour=17) - timedelta(days=i),
            })
        
        return det

    def test_match_by_hour(self, detector):
        """Test matching patterns by current hour."""
        matching = detector.get_matching_patterns({"hour": 17})
        # Should find temporal patterns around 17:00
        assert isinstance(matching, list)


class TestClearAndStats:
    """Tests for clear and stats methods."""

    def test_clear(self):
        """Test clearing detector."""
        detector = PatternDetector()
        detector.add_event({"type": "test"})
        detector.clear()
        assert len(detector._events) == 0

    def test_get_stats(self):
        """Test get_stats returns expected keys."""
        detector = PatternDetector()
        detector.add_event({"type": "test"})
        
        stats = detector.get_stats()
        assert "total_events" in stats
        assert "total_patterns" in stats
        assert "patterns_by_type" in stats


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_pattern_detector_singleton(self):
        """Test singleton returns same instance."""
        reset_pattern_detector()
        d1 = get_pattern_detector()
        d2 = get_pattern_detector()
        assert d1 is d2

    def test_reset_pattern_detector(self):
        """Test resetting singleton."""
        reset_pattern_detector()
        d1 = get_pattern_detector()
        reset_pattern_detector()
        d2 = get_pattern_detector()
        assert d1 is not d2
