"""
DAIMON Corpus Loaders - Universal Document Loading
===================================================

Format-specific loaders for corpus ingestion.
All loaders convert documents to plain text for embedding.

Supported formats:
- Text: .txt, .md, .rst
- PDF: .pdf (requires pypdf)
- Code: .py, .js, .ts, .go, .rs, etc.
- Web: URLs (requires beautifulsoup4)
"""

from .base import BaseLoader, Document
from .text import TextLoader, MarkdownLoader
from .code import CodeLoader
from .pdf import PDFLoader
from .web import WebLoader

__all__ = [
    "BaseLoader",
    "Document",
    "TextLoader",
    "MarkdownLoader",
    "CodeLoader",
    "PDFLoader",
    "WebLoader",
    "get_loader_for_path",
    "load_document",
]

# Loader registry by file extension
LOADERS = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".rst": TextLoader,
    ".py": CodeLoader,
    ".js": CodeLoader,
    ".ts": CodeLoader,
    ".jsx": CodeLoader,
    ".tsx": CodeLoader,
    ".go": CodeLoader,
    ".rs": CodeLoader,
    ".java": CodeLoader,
    ".c": CodeLoader,
    ".cpp": CodeLoader,
    ".h": CodeLoader,
    ".hpp": CodeLoader,
    ".rb": CodeLoader,
    ".php": CodeLoader,
    ".sh": CodeLoader,
    ".bash": CodeLoader,
    ".zsh": CodeLoader,
    ".yaml": TextLoader,
    ".yml": TextLoader,
    ".json": TextLoader,
    ".toml": TextLoader,
    ".pdf": PDFLoader,
}


def get_loader_for_path(path: str) -> BaseLoader:
    """
    Get appropriate loader for file path.

    Args:
        path: File path or URL.

    Returns:
        Loader instance for the file type.
    """
    from pathlib import Path

    # Check if URL
    if path.startswith(("http://", "https://")):
        return WebLoader()

    # Get extension
    ext = Path(path).suffix.lower()
    loader_class = LOADERS.get(ext, TextLoader)
    return loader_class()


def load_document(path: str) -> Document:
    """
    Load document from path or URL.

    Args:
        path: File path or URL.

    Returns:
        Document with text content and metadata.
    """
    loader = get_loader_for_path(path)
    return loader.load(path)
