"""
DAIMON Preference Learning Package
===================================

Semantic decomposition of preference learning into cohesive modules.

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
    │   Scanner   │ ──► │  Detector   │ ──► │  Categorizer │
    └─────────────┘     └─────────────┘     └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   Insights   │
                                            └──────────────┘

Modules:
    - models: Core data structures (PreferenceSignal, etc.)
    - scanner: Session discovery and loading
    - detector: Signal detection with LLM/heuristics
    - categorizer: Category inference and scoring
    - insights: Actionable insight generation

Usage:
    from learners.preference import (
        PreferenceSignal,
        SessionScanner,
        SignalDetector,
        PreferenceCategorizer,
        InsightGenerator,
    )

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from .models import (
    SignalType,
    PreferenceCategory,
    PreferenceSignal,
    CategoryStats,
    PreferenceInsight,
)

from .scanner import SessionScanner
from .detector import SignalDetector
from .categorizer import PreferenceCategorizer
from .insights import InsightGenerator

__all__ = [
    # Models
    "SignalType",
    "PreferenceCategory",
    "PreferenceSignal",
    "CategoryStats",
    "PreferenceInsight",
    
    # Components
    "SessionScanner",
    "SignalDetector",
    "PreferenceCategorizer",
    "InsightGenerator",
]
