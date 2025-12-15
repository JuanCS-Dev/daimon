"""
Emotional Intelligence Schemas
===============================

Data structures for emotion detection and state tracking.

VAD Model:
- Valence: Pleasant/Unpleasant (-1 to +1)
- Arousal: Activated/Deactivated (0 to 1)
- Dominance: Dominant/Submissive (0 to 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class VADScore:
    """
    Valence-Arousal-Dominance score.

    The VAD model provides a dimensional representation of emotions:
    - Valence: How pleasant/unpleasant (-1 to +1)
    - Arousal: How activated/calm (0 to 1)
    - Dominance: How in-control/submissive (0 to 1)
    """
    valence: float = 0.0     # -1 (negative) to +1 (positive)
    arousal: float = 0.5     # 0 (calm) to 1 (excited)
    dominance: float = 0.5   # 0 (submissive) to 1 (dominant)

    def __post_init__(self) -> None:
        """Validate and clamp values."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple for calculations."""
        return (self.valence, self.arousal, self.dominance)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "VADScore":
        """Create from dictionary."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
        )

    def get_quadrant(self) -> str:
        """
        Get the emotional quadrant based on valence and arousal.

        Returns one of:
        - high_positive_high_arousal (joy, excitement)
        - high_positive_low_arousal (calm, content)
        - high_negative_high_arousal (anger, fear)
        - high_negative_low_arousal (sadness, grief)
        - neutral
        """
        if abs(self.valence) < 0.3 and self.arousal < 0.5:
            return "neutral"

        v_sign = "positive" if self.valence > 0.3 else "negative"
        a_level = "high" if self.arousal > 0.5 else "low"

        if abs(self.valence) < 0.3:
            return "neutral"

        return f"high_{v_sign}_{a_level}_arousal"


@dataclass
class EmotionDetectionResult:
    """
    Result of emotion detection on user input.

    Contains both dimensional (VAD) and categorical (28 GoEmotions) scores.
    """
    # VAD dimensions
    vad: VADScore

    # Categorical emotions (28 GoEmotions)
    primary_emotion: str
    emotion_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    confidence: float = 0.5
    raw_text: str = ""
    detected_at: datetime = field(default_factory=datetime.now)

    def get_top_emotions(self, n: int = 3) -> List[tuple[str, float]]:
        """Get top N emotions by score."""
        sorted_emotions = sorted(
            self.emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:n]

    def get_emotional_salience(self) -> float:
        """
        Calculate overall emotional salience (how emotionally charged).

        High salience = strong emotional signal
        Low salience = neutral/weak emotional signal
        """
        # Use absolute valence and arousal as indicators
        valence_intensity = abs(self.vad.valence)
        arousal_intensity = self.vad.arousal

        # Weighted combination
        salience = 0.6 * valence_intensity + 0.4 * arousal_intensity
        return min(1.0, salience)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vad": self.vad.to_dict(),
            "primary_emotion": self.primary_emotion,
            "emotion_scores": self.emotion_scores,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "salience": self.get_emotional_salience(),
        }


@dataclass
class EmotionalContext:
    """
    Emotional context for storage in memory.

    This is the persistent emotional metadata attached to memories.
    """
    # VAD dimensions
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    # Categorical
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)

    # User-specific
    user_detected_emotion: Optional[str] = None

    # Computed
    emotional_salience: float = 0.5
    response_strategy: str = "resposta_padrao"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "primary_emotion": self.primary_emotion,
            "secondary_emotions": self.secondary_emotions,
            "user_detected_emotion": self.user_detected_emotion,
            "emotional_salience": self.emotional_salience,
            "response_strategy": self.response_strategy,
        }

    @classmethod
    def from_detection_result(
        cls,
        result: EmotionDetectionResult,
        strategy: str = "resposta_padrao"
    ) -> "EmotionalContext":
        """Create EmotionalContext from detection result."""
        top_emotions = result.get_top_emotions(3)
        secondary = [e for e, _ in top_emotions[1:]]

        return cls(
            valence=result.vad.valence,
            arousal=result.vad.arousal,
            dominance=result.vad.dominance,
            primary_emotion=result.primary_emotion,
            secondary_emotions=secondary,
            user_detected_emotion=result.primary_emotion,
            emotional_salience=result.get_emotional_salience(),
            response_strategy=strategy,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalContext":
        """Create from dictionary."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            primary_emotion=data.get("primary_emotion", "neutral"),
            secondary_emotions=data.get("secondary_emotions", []),
            user_detected_emotion=data.get("user_detected_emotion"),
            emotional_salience=data.get("emotional_salience", 0.5),
            response_strategy=data.get("response_strategy", "resposta_padrao"),
        )
