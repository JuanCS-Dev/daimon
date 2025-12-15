"""
NOESIS Emotional Intelligence Module
=====================================

Provides emotion detection, state tracking, and response modulation.

Components:
- EmotionDetector: LLM-based emotion detection (VAD + 28 GoEmotions)
- EmotionalState: Persistent emotional state with smoothing
- ResponseModulator: Empathetic response adaptation

Architecture:
    User Input → EmotionDetector → EmotionalState → ResponseModulator → Prompt
                      ↓
              EmotionalContext → Memory (persistent)

Based on: GoEmotions (Google), VAD Model (Russell), Affective Computing research
"""

from metacognitive_reflector.core.emotion.constants import (
    GOEMOTION_LABELS,
    EMOTION_TO_VAD,
    RESPONSE_STRATEGIES,
)
from metacognitive_reflector.core.emotion.schemas import (
    VADScore,
    EmotionDetectionResult,
    EmotionalContext,
)
from metacognitive_reflector.core.emotion.state import (
    EmotionalState,
    get_emotional_state,
    reset_emotional_state,
)
from metacognitive_reflector.core.emotion.detector import (
    EmotionDetector,
    detect_emotion,
)
from metacognitive_reflector.core.emotion.modulator import (
    ResponseModulator,
    modulate_response,
)

__all__ = [
    # Constants
    "GOEMOTION_LABELS",
    "EMOTION_TO_VAD",
    "RESPONSE_STRATEGIES",
    # Schemas
    "VADScore",
    "EmotionDetectionResult",
    "EmotionalContext",
    # Core classes
    "EmotionalState",
    "get_emotional_state",
    "reset_emotional_state",
    "EmotionDetector",
    "detect_emotion",
    "ResponseModulator",
    "modulate_response",
]
