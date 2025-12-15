"""
NOESIS MAIEUTICA - Internal Socratic Questioning Engine (G4)
=============================================================

Enables the system to question its own premises and assumptions
before generating or finalizing responses.

Named after Socrates' maieutic method (μαιευτική):
"the art of midwifery" - helping ideas be born through questioning.

G4 Integration Spec:
- Internal questioning of high-confidence claims
- Evidence source evaluation
- Alternative hypothesis generation
- Confidence adjustment based on questioning
"""

from .engine import (
    InternalMaieuticaEngine,
    MaieuticaResult,
    QuestionCategory,
    create_maieutica_engine,
)

__all__ = [
    "InternalMaieuticaEngine",
    "MaieuticaResult",
    "QuestionCategory",
    "create_maieutica_engine",
]
