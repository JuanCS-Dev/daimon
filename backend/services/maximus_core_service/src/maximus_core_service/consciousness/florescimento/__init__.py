"""
Módulo Florescimento: Auto-Percepção Consciente e Introspecção.
"""

from .unified_self import UnifiedSelfConcept, ComputationalState, EpisodicMemorySnapshot
from .consciousness_bridge import ConsciousnessBridge, IntrospectiveResponse
from .mirror_test import MirrorTestValidator, MirrorTestResult
from .introspection_api import router as florescimento_router, initialize_florescimento
from .phenomenal_constraint import (
    PhenomenalConstraint,
    NarrativeMode,
    constraint_from_esgt_event,
    get_narrative_mode,
)
from .epistemic_humility import (
    EpistemicHumilityGuard,
    EpistemicAssessment,
    KnowledgeState,
    ConfidenceLevel,
    create_epistemic_guard,
)

__all__ = [
    "UnifiedSelfConcept",
    "ComputationalState",
    "EpisodicMemorySnapshot",
    "ConsciousnessBridge",
    "IntrospectiveResponse",
    "MirrorTestValidator",
    "MirrorTestResult",
    "florescimento_router",
    "initialize_florescimento",
    # G1+G2: PhenomenalConstraint
    "PhenomenalConstraint",
    "NarrativeMode",
    "constraint_from_esgt_event",
    "get_narrative_mode",
    # G6: EpistemicHumilityGuard
    "EpistemicHumilityGuard",
    "EpistemicAssessment",
    "KnowledgeState",
    "ConfidenceLevel",
    "create_epistemic_guard",
]