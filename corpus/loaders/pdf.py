"""
DAIMON PDF Loader - Extract text from PDF files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import BaseLoader, Document

logger = logging.getLogger("daimon.corpus.pdf")


class PDFLoader(BaseLoader):
    """
    Loader for PDF files (.pdf).

    Requires pypdf library for text extraction.
    Falls back gracefully if not available.
    """

    def __init__(self, extract_metadata: bool = True):
        """
        Initialize PDF loader.

        Args:
            extract_metadata: If True, extract PDF metadata (author, title, etc.)
        """
        self.extract_metadata = extract_metadata
        self._pypdf_available: Optional[bool] = None

    def _check_pypdf(self) -> bool:
        """Check if pypdf is available."""
        if self._pypdf_available is None:
            try:
                import pypdf  # noqa: F401
                self._pypdf_available = True
            except ImportError:
                self._pypdf_available = False
                logger.warning(
                    "pypdf not available. Install with: pip install pypdf"
                )
        return self._pypdf_available

    def load(self, path: str) -> Document:
        """
        Load PDF file and extract text.

        Args:
            path: Path to PDF file.

        Returns:
            Document with extracted text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If pypdf not available or PDF is invalid.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self._check_pypdf():
            raise ValueError(
                "pypdf library required for PDF loading. "
                "Install with: pip install pypdf"
            )

        import pypdf  # pylint: disable=import-outside-toplevel

        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}") from e

        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            except Exception as e:
                logger.warning("Failed to extract page %d: %s", i, e)

        text = "\n\n".join(pages_text)
        text = self._clean_text(text)

        # Extract metadata
        metadata = {
            "format": "pdf",
            "extension": ".pdf",
            "size_bytes": file_path.stat().st_size,
            "page_count": len(reader.pages),
        }

        title = file_path.stem

        if self.extract_metadata and reader.metadata:
            pdf_meta = reader.metadata
            if pdf_meta.title:
                title = pdf_meta.title
                metadata["pdf_title"] = pdf_meta.title
            if pdf_meta.author:
                metadata["author"] = pdf_meta.author
            if pdf_meta.subject:
                metadata["subject"] = pdf_meta.subject
            if pdf_meta.creator:
                metadata["creator"] = pdf_meta.creator

        return Document(
            text=text,
            source=str(file_path.absolute()),
            title=title,
            metadata=metadata,
        )
