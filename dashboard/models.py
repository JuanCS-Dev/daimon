"""
DAIMON Dashboard - Pydantic models.
"""

from pydantic import BaseModel


class ClaudeMdUpdate(BaseModel):
    """Request body for CLAUDE.md update."""
    content: str


class CorpusTextCreate(BaseModel):
    """Request body for creating corpus text."""
    author: str
    title: str
    category: str
    content: str
    themes: list[str] = []
    source: str = ""
    relevance: float = 0.5
