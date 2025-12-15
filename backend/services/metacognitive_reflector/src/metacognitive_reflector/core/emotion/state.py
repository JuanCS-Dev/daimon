"""
Emotional State Tracker
========================

Persistent emotional state with exponential smoothing.

Tracks Noesis's emotional state over the conversation,
influenced by user emotions (contagion) and self-regulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import logging

from metacognitive_reflector.core.emotion.schemas import VADScore

logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """
    Noesis's current emotional state.

    Maintains a smoothed emotional trajectory based on:
    - User emotional contagion (influenced by empathy_level)
    - Self-regulation toward baseline
    - Temporal decay
    """

    # Current VAD dimensions
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5

    # Categorical label
    primary_emotion: str = "neutral"

    # History for smoothing (last N readings)
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    max_history: int = 10

    # Configuration
    contagion_factor: float = 0.4     # How much user emotion affects us
    regulation_factor: float = 0.2    # How much we self-regulate to positive
    decay_factor: float = 0.3         # Smoothing factor (lower = smoother)

    # Baseline (return point for regulation)
    baseline_valence: float = 0.2     # Slightly positive baseline
    baseline_arousal: float = 0.5
    baseline_dominance: float = 0.6   # Slightly confident

    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)

    def update(
        self,
        user_vad: Tuple[float, float, float],
        emotion_label: str = "neutral"
    ) -> None:
        """
        Update state based on user emotional input.

        Uses exponential smoothing with contagion and regulation.

        Args:
            user_vad: User's (valence, arousal, dominance)
            emotion_label: User's detected primary emotion
        """
        user_v, user_a, user_d = user_vad

        # 1. Contagion: Move toward user emotion (weighted)
        contagion_v = self.valence + self.contagion_factor * (user_v - self.valence)
        contagion_a = self.arousal + self.contagion_factor * (user_a - self.arousal)
        contagion_d = self.dominance + self.contagion_factor * (user_d - self.dominance)

        # 2. Regulation: Pull toward positive baseline
        regulated_v = contagion_v + self.regulation_factor * (self.baseline_valence - contagion_v)
        regulated_a = contagion_a + self.regulation_factor * (self.baseline_arousal - contagion_a)
        regulated_d = contagion_d + self.regulation_factor * (self.baseline_dominance - contagion_d)

        # 3. Smoothing: Blend with previous state
        self.valence = self.decay_factor * regulated_v + (1 - self.decay_factor) * self.valence
        self.arousal = self.decay_factor * regulated_a + (1 - self.decay_factor) * self.arousal
        self.dominance = self.decay_factor * regulated_d + (1 - self.decay_factor) * self.dominance

        # Clamp values
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

        # Update categorical label based on current state
        self.primary_emotion = self._derive_emotion_label()

        # Record in history
        self.history.append((self.valence, self.arousal, self.dominance))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self.last_updated = datetime.now()

        logger.debug(
            "EmotionalState updated: V=%.2f A=%.2f D=%.2f (%s) <- user: %s",
            self.valence, self.arousal, self.dominance,
            self.primary_emotion, emotion_label
        )

    def _derive_emotion_label(self) -> str:
        """Derive primary emotion from current VAD."""
        # Simple quadrant-based derivation
        if abs(self.valence) < 0.2 and self.arousal < 0.4:
            return "neutral"
        elif self.valence > 0.5 and self.arousal > 0.5:
            return "engaged"  # Positive and activated
        elif self.valence > 0.3 and self.arousal <= 0.5:
            return "serene"   # Positive and calm
        elif self.valence < -0.3 and self.arousal > 0.5:
            return "concerned"  # Negative and activated
        elif self.valence < -0.3 and self.arousal <= 0.5:
            return "contemplative"  # Processing negative
        elif self.arousal > 0.6:
            return "attentive"
        else:
            return "balanced"

    def get_vad(self) -> VADScore:
        """Get current state as VADScore."""
        return VADScore(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance
        )

    def get_quadrant(self) -> str:
        """Get emotional quadrant for response strategy."""
        return self.get_vad().get_quadrant()

    def get_trajectory(self) -> str:
        """
        Analyze emotional trajectory over recent history.

        Returns:
            One of: "improving", "declining", "stable", "volatile"
        """
        if len(self.history) < 3:
            return "stable"

        recent = self.history[-3:]
        valences = [v for v, _, _ in recent]

        # Calculate trend
        trend = valences[-1] - valences[0]

        if abs(trend) < 0.1:
            return "stable"
        elif trend > 0.2:
            return "improving"
        elif trend < -0.2:
            return "declining"
        else:
            # Check volatility
            variance = sum((v - sum(valences)/len(valences))**2 for v in valences) / len(valences)
            if variance > 0.1:
                return "volatile"
            return "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "primary_emotion": self.primary_emotion,
            "trajectory": self.get_trajectory(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalState":
        """Deserialize from storage."""
        state = cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            primary_emotion=data.get("primary_emotion", "neutral"),
        )
        return state

    def reset(self) -> None:
        """Reset to baseline state."""
        self.valence = self.baseline_valence
        self.arousal = self.baseline_arousal
        self.dominance = self.baseline_dominance
        self.primary_emotion = "neutral"
        self.history.clear()
        self.last_updated = datetime.now()

    def __repr__(self) -> str:
        return (
            f"EmotionalState(V={self.valence:.2f}, A={self.arousal:.2f}, "
            f"D={self.dominance:.2f}, {self.primary_emotion})"
        )


# Module-level singleton for session persistence
_current_state: Optional[EmotionalState] = None


def get_emotional_state() -> EmotionalState:
    """Get or create singleton emotional state."""
    global _current_state
    if _current_state is None:
        _current_state = EmotionalState()
    return _current_state


def reset_emotional_state() -> None:
    """Reset the singleton emotional state."""
    global _current_state
    if _current_state:
        _current_state.reset()
    else:
        _current_state = EmotionalState()
