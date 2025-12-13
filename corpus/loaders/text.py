"""
DAIMON Text Loader - Plain text and Markdown files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .base import BaseLoader, Document


class TextLoader(BaseLoader):
    """Loader for plain text files (.txt, .rst)."""

    def load(self, path: str) -> Document:
        """
        Load plain text file.

        Args:
            path: Path to text file.

        Returns:
            Document with text content.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = file_path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)

        return Document(
            text=text,
            source=str(file_path.absolute()),
            title=file_path.stem,
            metadata={
                "format": "text",
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
            },
        )


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files (.md).

    Optionally strips markdown formatting for cleaner embeddings.
    """

    def __init__(self, strip_formatting: bool = True):
        """
        Initialize Markdown loader.

        Args:
            strip_formatting: If True, strip markdown syntax.
        """
        self.strip_formatting = strip_formatting

    def load(self, path: str) -> Document:
        """
        Load Markdown file.

        Args:
            path: Path to markdown file.

        Returns:
            Document with text content.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = file_path.read_text(encoding="utf-8", errors="replace")

        # Extract title from first heading
        title = self._extract_title(text) or file_path.stem

        if self.strip_formatting:
            text = self._strip_markdown(text)

        text = self._clean_text(text)

        return Document(
            text=text,
            source=str(file_path.absolute()),
            title=title,
            metadata={
                "format": "markdown",
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
            },
        )

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """Extract title from first H1 heading."""
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Strip markdown formatting for cleaner text.

        Removes:
        - Headers (#)
        - Bold/italic (**/*, __/_)
        - Links [text](url)
        - Images ![alt](url)
        - Code blocks
        - Inline code
        """
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Convert links to just text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

        return text
