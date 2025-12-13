"""
DAIMON Dashboard - Route modules.
"""

from .status import router as status_router
from .corpus import router as corpus_router
from .memory import router as memory_router
from .cognitive import router as cognitive_router

__all__ = [
    "status_router",
    "corpus_router",
    "memory_router",
    "cognitive_router",
]
