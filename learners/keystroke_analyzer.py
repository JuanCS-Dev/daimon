"""
DAIMON Keystroke Analyzer - Cognitive State Detection
======================================================

Advanced keystroke dynamics analysis for inferring cognitive state.

Research Base:
- CNN-based Keystroke Dynamics (96.97% accuracy for identification)
- Keystroke Dynamics for Early Detection (cognitive decline detection)
- Behavioral Biometrics 2025 (continuous authentication)

Features:
- Rhythm consistency analysis
- Fatigue detection (increasing delays over time)
- Flow state detection (consistent rhythm + high speed)
- Cognitive load inference

Usage:
    analyzer = KeystrokeAnalyzer()
    analyzer.add_event(key='a', event_type='press', timestamp=time.time())
    analyzer.add_event(key='a', event_type='release', timestamp=time.time() + 0.1)
    state = analyzer.detect_cognitive_state()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .keystroke_models import (
    KeystrokeEvent,
    KeystrokeBiometrics,
    CognitiveState,
)
from .keystroke_calculations import (
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

logger = logging.getLogger("daimon.keystroke")

# Configuration
WINDOW_SIZE = 300  # 5 minutes of keystrokes
MIN_EVENTS_FOR_ANALYSIS = 20


class KeystrokeAnalyzer:
    """
    Keystroke dynamics analyzer for cognitive state detection.

    Maintains a rolling window of keystroke events and computes
    biometric features to infer cognitive state.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        min_events: int = MIN_EVENTS_FOR_ANALYSIS,
    ):
        """
        Initialize analyzer.

        Args:
            window_size: Number of events to keep in rolling window.
            min_events: Minimum events required for analysis.
        """
        self.window_size = window_size
        self.min_events = min_events

        # Rolling window of events
        self._events: Deque[KeystrokeEvent] = deque(maxlen=window_size)

        # Track press times for hold duration calculation
        self._pending_presses: Dict[str, float] = {}

        # Cache computed biometrics
        self._cached_biometrics: Optional[KeystrokeBiometrics] = None
        self._cache_valid = False

    def add_event(
        self,
        key: str,
        event_type: str,
        timestamp: Optional[float] = None,
        modifiers: Optional[List[str]] = None,
    ) -> None:
        """
        Add keystroke event to analyzer.

        Args:
            key: Key that was pressed/released.
            event_type: 'press' or 'release'.
            timestamp: Event timestamp (uses time.time() if not provided).
            modifiers: Active modifier keys (shift, ctrl, etc.).
        """
        if timestamp is None:
            timestamp = time.time()

        event = KeystrokeEvent(
            key=key,
            event_type=event_type,
            timestamp=timestamp,
            modifiers=modifiers or [],
        )

        self._events.append(event)
        self._cache_valid = False

        # Track press times
        if event_type == "press":
            self._pending_presses[key] = timestamp
        elif event_type == "release" and key in self._pending_presses:
            del self._pending_presses[key]

    def compute_biometrics(self) -> KeystrokeBiometrics:
        """
        Compute biometric features from current event window.

        Returns:
            KeystrokeBiometrics with computed features.
        """
        if self._cache_valid and self._cached_biometrics:
            return self._cached_biometrics

        biometrics = KeystrokeBiometrics()

        if len(self._events) < self.min_events:
            return biometrics

        # Extract press and release events
        events = list(self._events)
        presses = [e for e in events if e.event_type == "press"]
        releases = [e for e in events if e.event_type == "release"]

        # Use extracted calculation functions
        import statistics

        hold_times = calculate_hold_times(presses, releases)
        if hold_times:
            biometrics.avg_hold_time = statistics.mean(hold_times)

        seek_times = calculate_seek_times(events)
        if seek_times:
            biometrics.avg_seek_time = statistics.mean(seek_times)

        biometrics.rhythm_consistency = calculate_rhythm_consistency(seek_times)
        biometrics.fatigue_index = calculate_fatigue_index(seek_times)
        biometrics.focus_score = calculate_focus_score(seek_times)
        biometrics.cognitive_load = calculate_cognitive_load(hold_times, seek_times)
        biometrics.typing_speed = calculate_typing_speed(presses)
        biometrics.error_rate = calculate_error_rate(presses)
        biometrics.computed_at = datetime.now()

        self._cached_biometrics = biometrics
        self._cache_valid = True

        return biometrics

    def detect_cognitive_state(self) -> CognitiveState:
        """
        Detect current cognitive state from keystroke patterns.

        Returns:
            CognitiveState with state name, confidence, and biometrics.
        """
        biometrics = self.compute_biometrics()

        # Not enough data
        if len(self._events) < self.min_events:
            return CognitiveState(
                state="idle",
                confidence=0.0,
                biometrics=biometrics,
            )

        # Detect state based on biometrics
        state, confidence = infer_state(biometrics)

        return CognitiveState(
            state=state,
            confidence=confidence,
            biometrics=biometrics,
        )

    def get_recent_events(self, count: int = 10) -> List[KeystrokeEvent]:
        """Get most recent keystroke events."""
        events = list(self._events)
        return events[-count:] if len(events) >= count else events

    def clear(self) -> None:
        """Clear all events and reset analyzer."""
        self._events.clear()
        self._pending_presses.clear()
        self._cached_biometrics = None
        self._cache_valid = False

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        biometrics = self.compute_biometrics()
        return {
            "event_count": len(self._events),
            "window_size": self.window_size,
            "min_events": self.min_events,
            "has_enough_data": len(self._events) >= self.min_events,
            "pending_presses": len(self._pending_presses),
            "biometrics": {
                "avg_hold_time_ms": biometrics.avg_hold_time * 1000,
                "avg_seek_time_ms": biometrics.avg_seek_time * 1000,
                "rhythm_consistency": biometrics.rhythm_consistency,
                "typing_speed_kpm": biometrics.typing_speed,
            },
        }


# Singleton instance
_analyzer: Optional[KeystrokeAnalyzer] = None


def get_keystroke_analyzer() -> KeystrokeAnalyzer:
    """Get global keystroke analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = KeystrokeAnalyzer()
    return _analyzer


def reset_keystroke_analyzer() -> None:
    """Reset global keystroke analyzer instance."""
    global _analyzer
    _analyzer = None
