"""
Keystroke Analyzer Data Models.

Contains dataclasses for keystroke events and biometric features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class KeystrokeEvent:
    """Single keystroke event."""

    key: str
    event_type: str  # 'press' or 'release'
    timestamp: float
    modifiers: List[str] = field(default_factory=list)


@dataclass
class KeystrokeBiometrics:
    """
    Computed biometric features from keystroke analysis.

    Attributes:
        avg_hold_time: Average key hold duration (press to release).
        avg_seek_time: Average time between releasing one key and pressing next.
        rhythm_consistency: Standard deviation normalized (0-1, higher = more consistent).
        fatigue_index: Measure of increasing delays (0-1, higher = more fatigued).
        focus_score: Burst pattern detection (0-1, higher = more focused bursts).
        cognitive_load: Inferred cognitive load (0-1, higher = more loaded).
        typing_speed: Keys per minute.
        error_rate: Estimated error rate (backspace ratio).
    """

    avg_hold_time: float = 0.0
    avg_seek_time: float = 0.0
    rhythm_consistency: float = 0.0
    fatigue_index: float = 0.0
    focus_score: float = 0.0
    cognitive_load: float = 0.0
    typing_speed: float = 0.0
    error_rate: float = 0.0

    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveState:
    """
    Detected cognitive state from keystroke analysis.

    States:
        flow: Deep concentration, consistent fast typing
        focused: Good concentration, steady rhythm
        distracted: Inconsistent patterns, frequent pauses
        fatigued: Slowing down over time
        stressed: Fast but erratic, many errors
        idle: Not enough data or no recent activity
    """

    state: str
    confidence: float
    biometrics: KeystrokeBiometrics
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "biometrics": {
                "avg_hold_time": self.biometrics.avg_hold_time,
                "avg_seek_time": self.biometrics.avg_seek_time,
                "rhythm_consistency": self.biometrics.rhythm_consistency,
                "fatigue_index": self.biometrics.fatigue_index,
                "focus_score": self.biometrics.focus_score,
                "cognitive_load": self.biometrics.cognitive_load,
                "typing_speed": self.biometrics.typing_speed,
                "error_rate": self.biometrics.error_rate,
            },
        }
