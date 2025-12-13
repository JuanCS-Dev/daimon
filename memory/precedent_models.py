"""
DAIMON Precedent Models.

Data models for the precedent/jurisprudence system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

# Default database location
DEFAULT_DB_PATH = Path.home() / ".daimon" / "memory" / "precedents.db"

# Outcome types
OutcomeType = Literal["success", "failure", "partial", "unknown"]

# Database row type for type safety
PrecedentRow = Tuple[str, str, str, str, str, str, str, float, int, float, str]


@dataclass
class PrecedentMeta:
    """Metadata and statistics for a precedent."""

    created_at: str = ""
    updated_at: str = ""
    relevance: float = 0.5
    application_count: int = 0
    success_rate: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class Precedent:
    """Single precedent entry."""

    id: str
    context: str
    decision: str
    outcome: OutcomeType
    lesson: str
    meta: PrecedentMeta = field(default_factory=PrecedentMeta)

    @property
    def created_at(self) -> str:
        """When precedent was created."""
        return self.meta.created_at

    @property
    def updated_at(self) -> str:
        """When precedent was last updated."""
        return self.meta.updated_at

    @property
    def relevance(self) -> float:
        """Relevance score 0-1."""
        return self.meta.relevance

    @property
    def application_count(self) -> int:
        """Number of times precedent was applied."""
        return self.meta.application_count

    @property
    def success_rate(self) -> float:
        """Success rate 0-1."""
        return self.meta.success_rate

    @property
    def tags(self) -> List[str]:
        """Tags for categorization."""
        return self.meta.tags

    @classmethod
    def from_row(cls, row: PrecedentRow) -> Precedent:
        """Create Precedent from database row."""
        tags = json.loads(row[10]) if row[10] else []
        meta = PrecedentMeta(
            created_at=row[5],
            updated_at=row[6],
            relevance=row[7],
            application_count=row[8],
            success_rate=row[9],
            tags=tags,
        )
        return cls(
            id=row[0],
            context=row[1],
            decision=row[2],
            outcome=row[3],  # type: ignore[arg-type]
            lesson=row[4],
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "context": self.context,
            "decision": self.decision,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "created_at": self.meta.created_at,
            "updated_at": self.meta.updated_at,
            "relevance": self.meta.relevance,
            "application_count": self.meta.application_count,
            "success_rate": self.meta.success_rate,
            "tags": self.meta.tags,
        }


@dataclass
class PrecedentMatch:
    """Search result with match score."""

    precedent: Precedent
    score: float
    match_reason: str
