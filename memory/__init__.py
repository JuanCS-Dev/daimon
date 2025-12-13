"""
DAIMON Memory - Optimized local storage.
=========================================

Includes:
- MemoryStore: Semantic memory with FTS5
- PrecedentSystem: Decision jurisprudence
- ActivityStore: Time-series activity data
"""

from .optimized_store import MemoryStore, MemoryItem, SearchResult, MemoryTimestamps
from .precedent_models import Precedent, PrecedentMatch, OutcomeType
from .precedent_system import PrecedentSystem
from .activity_store import ActivityStore, ActivityRecord, ActivitySummary, get_activity_store

__all__ = [
    # Semantic memory
    "MemoryStore",
    "MemoryItem",
    "MemoryTimestamps",
    "SearchResult",
    # Precedent system
    "PrecedentSystem",
    "Precedent",
    "PrecedentMatch",
    "OutcomeType",
    # Activity store
    "ActivityStore",
    "ActivityRecord",
    "ActivitySummary",
    "get_activity_store",
]
