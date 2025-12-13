"""
DAIMON Corpus Manager.

Curated collection of wisdom texts for ethical grounding and philosophical reference.

Structure:
    ~/.daimon/corpus/
    ├── filosofia/
    │   ├── gregos/
    │   └── estoicos/
    ├── teologia/
    ├── ciencia/
    ├── logica/
    └── _index/
        ├── by_theme.json
        └── by_author.json

Usage:
    manager = CorpusManager()
    manager.add_text("Epictetus", "Enchiridion", "estoicos", content)
    results = manager.search("virtue")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import TextMetadata, WisdomText
from .search import keyword_search, keyword_search_scored, hybrid_search

if TYPE_CHECKING:
    from .semantic_search import SemanticCorpus

logger = logging.getLogger("daimon.corpus")

# Default corpus location
DEFAULT_CORPUS_PATH = Path.home() / ".daimon" / "corpus"

# Categories of wisdom
CATEGORIES = {
    "filosofia/gregos": "Greek Philosophy",
    "filosofia/estoicos": "Stoic Philosophy",
    "filosofia/modernos": "Modern Philosophy",
    "teologia": "Theology and Religion",
    "ciencia": "Science and Method",
    "logica": "Logic and Reasoning",
    "etica": "Ethics and Morality",
    "literatura": "Literature and Poetry",
}


class CorpusManager:
    """
    Manager for curated wisdom texts.

    Features:
    - Category-based organization
    - Theme and author indexing
    - Full-text search
    - Relevance scoring
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        enable_semantic: bool = True,
    ):
        """
        Initialize corpus manager.

        Args:
            corpus_path: Path to corpus directory. Defaults to ~/.daimon/corpus/
            enable_semantic: Enable semantic search (requires sentence-transformers).
        """
        self.corpus_path = corpus_path or DEFAULT_CORPUS_PATH
        self._enable_semantic = enable_semantic
        self._semantic_corpus: Optional["SemanticCorpus"] = None
        self._init_structure()
        self._load_indices()

    def _init_structure(self) -> None:
        """Initialize corpus directory structure."""
        # Create category directories
        for category in CATEGORIES:
            (self.corpus_path / category).mkdir(parents=True, exist_ok=True)

        # Create index directory
        (self.corpus_path / "_index").mkdir(parents=True, exist_ok=True)

        logger.debug("Corpus structure initialized at %s", self.corpus_path)

    def _load_indices(self) -> None:
        """Load or create indices."""
        self.by_theme: Dict[str, List[str]] = {}
        self.by_author: Dict[str, List[str]] = {}
        self.texts: Dict[str, WisdomText] = {}

        # Load theme index
        theme_file = self.corpus_path / "_index" / "by_theme.json"
        if theme_file.exists():
            self.by_theme = json.loads(theme_file.read_text())

        # Load author index
        author_file = self.corpus_path / "_index" / "by_author.json"
        if author_file.exists():
            self.by_author = json.loads(author_file.read_text())

        # Load all texts
        for category in CATEGORIES:
            category_path = self.corpus_path / category
            for text_file in category_path.glob("*.json"):
                try:
                    data = json.loads(text_file.read_text())
                    text = WisdomText.from_dict(data)
                    self.texts[text.id] = text
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to load %s: %s", text_file, e)

        logger.debug("Loaded %d texts from corpus", len(self.texts))

    def _get_semantic_corpus(self) -> Optional["SemanticCorpus"]:
        """Get semantic corpus instance (lazy initialization)."""
        if not self._enable_semantic:
            return None

        if self._semantic_corpus is None:
            try:
                from .semantic_search import SemanticCorpus  # pylint: disable=import-outside-toplevel

                semantic_path = self.corpus_path / "_semantic"
                self._semantic_corpus = SemanticCorpus(index_path=semantic_path)
                logger.debug("Initialized semantic corpus")
            except ImportError:
                logger.debug("Semantic search not available (missing dependencies)")
                self._enable_semantic = False
                return None

        return self._semantic_corpus

    def _save_indices(self) -> None:
        """Save indices to disk."""
        theme_file = self.corpus_path / "_index" / "by_theme.json"
        theme_file.write_text(json.dumps(self.by_theme, indent=2))

        author_file = self.corpus_path / "_index" / "by_author.json"
        author_file.write_text(json.dumps(self.by_author, indent=2))

    def generate_text_id(self, author: str, title: str) -> str:
        """
        Generate text ID from author and title.

        Public method for generating consistent IDs.
        """
        def clean(s: str) -> str:
            return "".join(c for c in s.lower() if c.isalnum() or c == " ")
        return f"{clean(author)[:20]}_{clean(title)[:30]}".replace(" ", "_")

    def add_text(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        author: str,
        title: str,
        category: str,
        content: str,
        metadata: Optional[TextMetadata] = None,
    ) -> str:
        """
        Add wisdom text to corpus.

        Args:
            author: Author name
            title: Text title
            category: Category path (e.g., "filosofia/estoicos")
            content: Text content
            metadata: Optional metadata (source, themes, relevance_score)

        Returns:
            Text ID
        """
        if category not in CATEGORIES:
            logger.warning("Unknown category: %s, using 'filosofia/modernos'", category)
            category = "filosofia/modernos"

        text_id = self.generate_text_id(author, title)

        if metadata is None:
            metadata = TextMetadata(added_at=datetime.now().isoformat())
        elif not metadata.added_at:
            metadata = TextMetadata(
                source=metadata.source,
                added_at=datetime.now().isoformat(),
                relevance_score=metadata.relevance_score,
                themes=metadata.themes,
            )

        themes = metadata.themes

        text = WisdomText(
            id=text_id,
            author=author,
            title=title,
            category=category,
            content=content,
            themes=themes,
            metadata=metadata,
        )

        # Save to file
        text_path = self.corpus_path / category / f"{text_id}.json"
        text_path.write_text(json.dumps(text.to_dict(), indent=2))

        # Update indices
        self.texts[text_id] = text

        # Update theme index
        for theme in themes:
            if theme not in self.by_theme:
                self.by_theme[theme] = []
            if text_id not in self.by_theme[theme]:
                self.by_theme[theme].append(text_id)

        # Update author index
        if author not in self.by_author:
            self.by_author[author] = []
        if text_id not in self.by_author[author]:
            self.by_author[author].append(text_id)

        self._save_indices()

        # Index in semantic corpus
        semantic = self._get_semantic_corpus()
        if semantic:
            semantic.add_document(
                text=content,
                metadata={
                    "author": author,
                    "title": title,
                    "category": category,
                    "themes": themes,
                },
                doc_id=text_id,
            )
            semantic.save()

        logger.info("Added text: %s by %s", title, author)

        return text_id

    def get_text(self, text_id: str) -> Optional[WisdomText]:
        """Get text by ID."""
        return self.texts.get(text_id)

    def get_by_author(self, author: str) -> List[WisdomText]:
        """Get all texts by author."""
        text_ids = self.by_author.get(author, [])
        return [self.texts[tid] for tid in text_ids if tid in self.texts]

    def get_by_theme(self, theme: str) -> List[WisdomText]:
        """Get all texts by theme."""
        text_ids = self.by_theme.get(theme, [])
        return [self.texts[tid] for tid in text_ids if tid in self.texts]

    def get_by_category(self, category: str) -> List[WisdomText]:
        """Get all texts in category."""
        return [t for t in self.texts.values() if t.category == category]

    def search(self, query: str, limit: int = 10) -> List[WisdomText]:
        """
        Search texts by content.

        Simple substring search. For advanced search, use MemoryStore.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching texts
        """
        return keyword_search(self.texts, query, limit)

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> List[WisdomText]:
        """
        Semantic similarity search using embeddings.

        Args:
            query: Search query (natural language).
            limit: Maximum results.
            min_score: Minimum similarity score (0-1).

        Returns:
            List of matching WisdomText objects.
        """
        semantic = self._get_semantic_corpus()
        if not semantic:
            # Fallback to keyword search
            logger.debug("Semantic search unavailable, falling back to keyword search")
            return self.search(query, limit)

        results = semantic.search(query, k=limit, min_score=min_score)

        # Convert SearchResult to WisdomText
        texts = []
        for result in results:
            if result.doc_id in self.texts:
                texts.append(self.texts[result.doc_id])
            else:
                logger.warning("Semantic result %s not in texts index", result.doc_id)

        return texts

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float = 0.7,
    ) -> List[WisdomText]:
        """
        Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query.
            limit: Maximum results.
            semantic_weight: Weight for semantic results (0-1).

        Returns:
            Combined results list.
        """
        semantic = self._get_semantic_corpus()
        return hybrid_search(self.texts, semantic, query, limit, semantic_weight)

    def list_authors(self) -> List[str]:
        """List all authors in corpus."""
        return list(self.by_author.keys())

    def list_themes(self) -> List[str]:
        """List all themes in corpus."""
        return list(self.by_theme.keys())

    def list_categories(self) -> Dict[str, str]:
        """List available categories with descriptions."""
        return CATEGORIES.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics.

        Returns:
            Dict with counts and breakdown
        """
        category_counts: Dict[str, int] = {}
        for text in self.texts.values():
            cat = text.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        stats = {
            "total_texts": len(self.texts),
            "total_authors": len(self.by_author),
            "total_themes": len(self.by_theme),
            "by_category": category_counts,
            "corpus_path": str(self.corpus_path),
        }

        # Add semantic corpus stats if available
        semantic = self._get_semantic_corpus()
        if semantic:
            sem_stats = semantic.get_stats()
            stats["semantic_enabled"] = True
            stats["semantic_indexed"] = sem_stats.get("document_count", 0)
        else:
            stats["semantic_enabled"] = False

        return stats

    def delete_text(self, text_id: str) -> bool:
        """
        Delete text from corpus.

        Args:
            text_id: Text ID to delete

        Returns:
            True if deleted, False if not found
        """
        if text_id not in self.texts:
            return False

        text = self.texts[text_id]

        # Remove from indices
        for theme in text.themes:
            if theme in self.by_theme:
                self.by_theme[theme] = [t for t in self.by_theme[theme] if t != text_id]

        if text.author in self.by_author:
            self.by_author[text.author] = [
                t for t in self.by_author[text.author] if t != text_id
            ]

        # Remove file
        text_path = self.corpus_path / text.category / f"{text_id}.json"
        if text_path.exists():
            text_path.unlink()

        # Remove from memory
        del self.texts[text_id]

        # Remove from semantic index
        semantic = self._get_semantic_corpus()
        if semantic:
            semantic.delete_document(text_id)

        self._save_indices()
        logger.info("Deleted text: %s", text_id)

        return True

    def reindex_semantic(self) -> int:
        """
        Rebuild semantic index from all corpus texts.

        Useful after bulk operations or to sync indices.

        Returns:
            Number of documents indexed.
        """
        semantic = self._get_semantic_corpus()
        if not semantic:
            logger.warning("Semantic search not available")
            return 0

        semantic.clear()

        count = 0
        for text in self.texts.values():
            semantic.add_document(
                text=text.content,
                metadata={
                    "author": text.author,
                    "title": text.title,
                    "category": text.category,
                    "themes": text.themes,
                },
                doc_id=text.id,
            )
            count += 1

        semantic.save()
        logger.info("Reindexed %d documents in semantic corpus", count)
        return count
