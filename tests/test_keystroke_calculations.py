"""Tests for keystroke biometric calculations."""

import pytest
from learners.keystroke_models import KeystrokeEvent, KeystrokeBiometrics
from learners.keystroke_calculations import (
    calculate_hold_times,
    calculate_seek_times,
    calculate_rhythm_consistency,
    calculate_fatigue_index,
    calculate_focus_score,
    calculate_cognitive_load,
    calculate_typing_speed,
    calculate_error_rate,
    infer_state,
)


class TestCalculateHoldTimes:
    """Tests for hold time calculation."""

    def test_matched_press_release_pairs(self):
        """Calculate hold time for matching press/release pairs."""
        presses = [
            KeystrokeEvent(key="a", event_type="press", timestamp=1.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=1.5),
        ]
        releases = [
            KeystrokeEvent(key="a", event_type="release", timestamp=1.1),
            KeystrokeEvent(key="b", event_type="release", timestamp=1.6),
        ]
        hold_times = calculate_hold_times(presses, releases)
        assert len(hold_times) == 2
        assert abs(hold_times[0] - 0.1) < 0.001
        assert abs(hold_times[1] - 0.1) < 0.001

    def test_filters_unreasonable_hold_times(self):
        """Filter hold times that are too short or too long."""
        presses = [
            KeystrokeEvent(key="a", event_type="press", timestamp=1.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=2.0),
        ]
        releases = [
            KeystrokeEvent(key="a", event_type="release", timestamp=1.005),  # Too short
            KeystrokeEvent(key="b", event_type="release", timestamp=5.0),  # Too long
        ]
        hold_times = calculate_hold_times(presses, releases)
        assert len(hold_times) == 0

    def test_empty_inputs(self):
        """Handle empty input lists."""
        assert calculate_hold_times([], []) == []

    def test_no_matching_releases(self):
        """Handle case with no matching releases."""
        presses = [KeystrokeEvent(key="a", event_type="press", timestamp=1.0)]
        releases = [KeystrokeEvent(key="b", event_type="release", timestamp=1.1)]
        hold_times = calculate_hold_times(presses, releases)
        assert len(hold_times) == 0


class TestCalculateSeekTimes:
    """Tests for seek time calculation."""

    def test_calculates_intervals_between_presses(self):
        """Calculate intervals between consecutive key presses."""
        events = [
            KeystrokeEvent(key="a", event_type="press", timestamp=1.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=1.2),
            KeystrokeEvent(key="c", event_type="press", timestamp=1.5),
        ]
        seek_times = calculate_seek_times(events)
        assert len(seek_times) == 2
        assert abs(seek_times[0] - 0.2) < 0.001
        assert abs(seek_times[1] - 0.3) < 0.001

    def test_ignores_release_events(self):
        """Only consider press events for seek times."""
        events = [
            KeystrokeEvent(key="a", event_type="press", timestamp=1.0),
            KeystrokeEvent(key="a", event_type="release", timestamp=1.1),
            KeystrokeEvent(key="b", event_type="press", timestamp=1.3),
        ]
        seek_times = calculate_seek_times(events)
        assert len(seek_times) == 1
        assert abs(seek_times[0] - 0.3) < 0.001

    def test_filters_unreasonable_intervals(self):
        """Filter intervals that are too short or too long."""
        events = [
            KeystrokeEvent(key="a", event_type="press", timestamp=1.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=1.005),  # Too short
            KeystrokeEvent(key="c", event_type="press", timestamp=10.0),  # Too long
        ]
        seek_times = calculate_seek_times(events)
        assert len(seek_times) == 0

    def test_empty_events(self):
        """Handle empty event list."""
        assert calculate_seek_times([]) == []


class TestCalculateRhythmConsistency:
    """Tests for rhythm consistency calculation."""

    def test_perfect_consistency(self):
        """Identical intervals should give high consistency."""
        seek_times = [0.1, 0.1, 0.1, 0.1, 0.1]
        consistency = calculate_rhythm_consistency(seek_times)
        assert consistency == 1.0

    def test_variable_intervals(self):
        """Variable intervals should give lower consistency."""
        seek_times = [0.1, 0.5, 0.2, 0.8, 0.15]
        consistency = calculate_rhythm_consistency(seek_times)
        assert 0.0 < consistency < 0.5

    def test_insufficient_data(self):
        """Less than 3 data points returns 0."""
        assert calculate_rhythm_consistency([0.1, 0.2]) == 0.0
        assert calculate_rhythm_consistency([0.1]) == 0.0
        assert calculate_rhythm_consistency([]) == 0.0


class TestCalculateFatigueIndex:
    """Tests for fatigue index calculation."""

    def test_no_fatigue_constant_speed(self):
        """Constant speed should show no fatigue."""
        seek_times = [0.1] * 100
        fatigue = calculate_fatigue_index(seek_times)
        assert fatigue == 0.0

    def test_slowing_down_shows_fatigue(self):
        """Slowing down should increase fatigue index."""
        older = [0.1] * 50
        newer = [0.2] * 50  # 2x slower
        seek_times = older + newer
        fatigue = calculate_fatigue_index(seek_times)
        assert fatigue > 0.5

    def test_insufficient_data(self):
        """Less than 60 data points returns 0."""
        seek_times = [0.1] * 50
        fatigue = calculate_fatigue_index(seek_times)
        assert fatigue == 0.0


class TestCalculateFocusScore:
    """Tests for focus score calculation."""

    def test_long_bursts_high_focus(self):
        """Long bursts of fast typing indicate focus."""
        seek_times = [0.1] * 20  # All fast, one long burst
        focus = calculate_focus_score(seek_times)
        assert focus > 0.5

    def test_scattered_typing_low_focus(self):
        """Scattered typing with long pauses indicates low focus."""
        seek_times = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5]  # Alternating fast/slow
        focus = calculate_focus_score(seek_times)
        assert focus < 0.5

    def test_insufficient_data(self):
        """Less than 5 data points returns 0."""
        assert calculate_focus_score([0.1, 0.1]) == 0.0


class TestCalculateCognitiveLoad:
    """Tests for cognitive load calculation."""

    def test_fast_consistent_typing_low_load(self):
        """Fast, consistent typing indicates low cognitive load."""
        hold_times = [0.05] * 10
        seek_times = [0.1] * 10
        load = calculate_cognitive_load(hold_times, seek_times)
        assert load < 0.5

    def test_slow_variable_typing_high_load(self):
        """Slow, variable typing indicates high cognitive load."""
        hold_times = [0.3] * 10  # Long holds (thinking while typing)
        seek_times = [0.1, 0.5, 0.2, 0.8, 0.15, 0.3, 0.6, 0.1, 0.4, 0.2]  # Variable
        load = calculate_cognitive_load(hold_times, seek_times)
        assert load > 0.5

    def test_empty_inputs(self):
        """Handle empty inputs."""
        assert calculate_cognitive_load([], []) == 0.0
        assert calculate_cognitive_load([0.1], []) == 0.0
        assert calculate_cognitive_load([], [0.1]) == 0.0


class TestCalculateTypingSpeed:
    """Tests for typing speed calculation."""

    def test_calculates_keys_per_minute(self):
        """Calculate typing speed in keys per minute."""
        presses = [
            KeystrokeEvent(key="a", event_type="press", timestamp=0.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=0.5),
            KeystrokeEvent(key="c", event_type="press", timestamp=1.0),
        ]
        speed = calculate_typing_speed(presses)
        # 3 keys in 1 second = 180 KPM
        assert abs(speed - 180) < 1

    def test_insufficient_data(self):
        """Less than 2 presses returns 0."""
        presses = [KeystrokeEvent(key="a", event_type="press", timestamp=0.0)]
        assert calculate_typing_speed(presses) == 0.0
        assert calculate_typing_speed([]) == 0.0


class TestCalculateErrorRate:
    """Tests for error rate calculation."""

    def test_no_errors(self):
        """No backspaces means 0 error rate."""
        presses = [
            KeystrokeEvent(key="a", event_type="press", timestamp=0.0),
            KeystrokeEvent(key="b", event_type="press", timestamp=0.1),
        ]
        assert calculate_error_rate(presses) == 0.0

    def test_with_backspaces(self):
        """Calculate error rate from backspace usage."""
        presses = [
            KeystrokeEvent(key="a", event_type="press", timestamp=0.0),
            KeystrokeEvent(key="backspace", event_type="press", timestamp=0.1),
            KeystrokeEvent(key="b", event_type="press", timestamp=0.2),
            KeystrokeEvent(key="BackSpace", event_type="press", timestamp=0.3),
        ]
        error_rate = calculate_error_rate(presses)
        assert abs(error_rate - 0.5) < 0.001

    def test_empty_presses(self):
        """Handle empty input."""
        assert calculate_error_rate([]) == 0.0


class TestInferState:
    """Tests for cognitive state inference."""

    def test_flow_state(self):
        """Detect flow state from consistent fast typing."""
        biometrics = KeystrokeBiometrics(
            rhythm_consistency=0.9,
            avg_seek_time=0.05,  # Very fast typing needed for high confidence
            error_rate=0.01,
        )
        state, confidence = infer_state(biometrics)
        assert state == "flow"
        assert confidence > 0.5

    def test_fatigued_state(self):
        """Detect fatigue from high fatigue index."""
        biometrics = KeystrokeBiometrics(
            fatigue_index=0.6,
        )
        state, confidence = infer_state(biometrics)
        assert state == "fatigued"
        assert confidence == 0.6

    def test_stressed_state(self):
        """Detect stress from fast erratic typing with errors."""
        biometrics = KeystrokeBiometrics(
            typing_speed=250,
            rhythm_consistency=0.3,
            error_rate=0.15,
        )
        state, confidence = infer_state(biometrics)
        assert state == "stressed"

    def test_distracted_state(self):
        """Detect distraction from low consistency and focus."""
        biometrics = KeystrokeBiometrics(
            rhythm_consistency=0.2,
            focus_score=0.1,
        )
        state, confidence = infer_state(biometrics)
        assert state == "distracted"

    def test_focused_state(self):
        """Detect focused state from decent consistency and focus."""
        biometrics = KeystrokeBiometrics(
            rhythm_consistency=0.7,
            focus_score=0.6,
        )
        state, confidence = infer_state(biometrics)
        assert state == "focused"

    def test_default_focused(self):
        """Default to focused state when no clear pattern."""
        # Use values that don't match any specific pattern
        # but still avoid the "distracted" condition
        biometrics = KeystrokeBiometrics(
            rhythm_consistency=0.45,  # > 0.4 (not distracted), <= 0.5 (not focused)
            focus_score=0.35,  # >= 0.3 (not distracted), <= 0.4 (not focused)
        )
        state, confidence = infer_state(biometrics)
        assert state == "focused"
        assert confidence == 0.5
