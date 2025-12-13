"""
DAIMON Base Loader - Abstract interface for document loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    Loaded document with text and metadata.

    Attributes:
        text: Extracted text content.
        source: Original file path or URL.
        title: Document title (if available).
        metadata: Additional metadata dict.
        loaded_at: When document was loaded.
    """
    text: str
    source: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Generate title from source if not provided."""
        if not self.title and self.source:
            self.title = Path(self.source).stem

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "metadata": self.metadata,
            "loaded_at": self.loaded_at.isoformat(),
        }


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    All loaders must implement the load() method.
    """

    @abstractmethod
    def load(self, path: str) -> Document:
        """
        Load document from path.

        Args:
            path: File path or URL.

        Returns:
            Document with extracted text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """

    def load_text(self, path: str) -> str:
        """
        Load just the text content.

        Args:
            path: File path or URL.

        Returns:
            Extracted text content.
        """
        doc = self.load(path)
        return doc.text

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text.

        - Remove excessive whitespace
        - Normalize line endings
        - Strip leading/trailing whitespace

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        import re

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()
