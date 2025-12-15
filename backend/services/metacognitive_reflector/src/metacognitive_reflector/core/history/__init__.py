"""
NOESIS Memory Fortress - History Subpackage
=============================================

Provides persistent storage for:
- Criminal history (convictions)
- Tribunal precedents (G3 integration)
"""

from .models import Conviction, CriminalHistory
from .provider import CriminalHistoryProvider
from .precedent_ledger import (
    Precedent,
    PrecedentLedgerProvider,
    create_precedent_from_verdict,
)

__all__ = [
    "Conviction",
    "CriminalHistory",
    "CriminalHistoryProvider",
    # G3: Precedent Ledger
    "Precedent",
    "PrecedentLedgerProvider",
    "create_precedent_from_verdict",
]
