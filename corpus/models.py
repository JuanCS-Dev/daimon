"""
Corpus Data Models.

Contains dataclasses for wisdom texts and metadata.
"""

from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class TextMetadata:
    """Metadata for a wisdom text."""

    source: str = ""
    added_at: str = ""
    relevance_score: float = 0.5
    themes: List[str] = field(default_factory=list)


@dataclass
class WisdomText:
    """Single wisdom text entry."""

    id: str
    author: str
    title: str
    category: str
    content: str
    themes: List[str]
    metadata: TextMetadata = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = TextMetadata()

    @property
    def source(self) -> str:
        """Source reference."""
        return self.metadata.source

    @property
    def added_at(self) -> str:
        """Timestamp when added."""
        return self.metadata.added_at

    @property
    def relevance_score(self) -> float:
        """Relevance score 0-1."""
        return self.metadata.relevance_score

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Flatten metadata for backwards compatibility
        meta = data.pop("metadata", {})
        # Don't let metadata themes overwrite main themes
        meta.pop("themes", None)
        data.update(meta)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WisdomText":
        """Create from dictionary."""
        # Handle both old (flat) and new (nested) format
        metadata = TextMetadata(
            source=data.pop("source", ""),
            added_at=data.pop("added_at", ""),
            relevance_score=data.pop("relevance_score", 0.5),
        )
        if "metadata" in data:
            data.pop("metadata")
        return cls(**data, metadata=metadata)
