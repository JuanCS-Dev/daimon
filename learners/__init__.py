"""
DAIMON Learners - Adaptive preference detection.
================================================

Learns user preferences from:
- Claude Code sessions (PreferenceLearner)
- Activity patterns (StyleLearner)
- Reflection cycles (ReflectionEngine)
- Keystroke dynamics (KeystrokeAnalyzer)
"""

from .preference_learner import PreferenceLearner, PreferenceSignal
from .reflection_engine import ReflectionEngine, get_engine, reset_engine
from .style_learner import (
    StyleLearner,
    CommunicationStyle,
    get_style_learner,
    TypingPace,
    EditingFrequency,
    InteractionPattern,
    FocusLevel,
)
from .keystroke_analyzer import (
    KeystrokeAnalyzer,
    KeystrokeBiometrics,
    KeystrokeEvent,
    CognitiveState,
    get_keystroke_analyzer,
    reset_keystroke_analyzer,
)
from .metacognitive_engine import (
    MetacognitiveEngine,
    MetacognitiveAnalysis,
    InsightRecord,
    CategoryEffectiveness,
    get_metacognitive_engine,
    reset_metacognitive_engine,
)

__all__ = [
    # Preference learning
    "PreferenceLearner",
    "PreferenceSignal",
    # Reflection engine
    "ReflectionEngine",
    "get_engine",
    "reset_engine",
    # Style learning
    "StyleLearner",
    "CommunicationStyle",
    "get_style_learner",
    "TypingPace",
    "EditingFrequency",
    "InteractionPattern",
    "FocusLevel",
    # Keystroke analysis
    "KeystrokeAnalyzer",
    "KeystrokeBiometrics",
    "KeystrokeEvent",
    "CognitiveState",
    "get_keystroke_analyzer",
    "reset_keystroke_analyzer",
    # Metacognitive engine
    "MetacognitiveEngine",
    "MetacognitiveAnalysis",
    "InsightRecord",
    "CategoryEffectiveness",
    "get_metacognitive_engine",
    "reset_metacognitive_engine",
]
