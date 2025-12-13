"""
DAIMON Semantic Search - FAISS + Sentence Transformers
=======================================================

Local semantic search over corpus documents.
Privacy-preserving: all embeddings computed and stored locally.

Features:
- Sentence-BERT embeddings (all-MiniLM-L6-v2, ~90MB)
- FAISS index for fast similarity search
- Lazy model loading (only when needed)
- Hybrid search: semantic + keyword fallback

Usage:
    corpus = SemanticCorpus()
    corpus.add_document("Some wisdom text", {"author": "Socrates"})
    results = corpus.search("virtue and knowledge", k=5)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("daimon.semantic")

# Default storage location
DEFAULT_INDEX_PATH = Path.home() / ".daimon" / "corpus" / "semantic"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


@dataclass
class SearchResult:
    """Semantic search result with score and metadata."""

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


class SemanticCorpus:
    """
    Semantic search corpus using FAISS and Sentence Transformers.

    Performance targets:
    - Embedding: ~50ms per document
    - Search: <10ms for 10k documents
    - Model loading: ~2s (lazy, once per session)

    Attributes:
        index_path: Path to store FAISS index and metadata.
        model_name: Sentence Transformer model name.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Initialize semantic corpus.

        Args:
            index_path: Path for FAISS index storage.
            model_name: Sentence Transformer model to use.
        """
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.model_name = model_name

        # Lazy-loaded components
        self._model: Optional[Any] = None
        self._index: Optional[Any] = None
        self._documents: List[Dict[str, Any]] = []
        self._initialized = False

        # Ensure storage directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of model and index.

        Returns:
            True if initialized successfully, False otherwise.
        """
        if self._initialized:
            return True

        try:
            # Try to import dependencies
            from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel
            import faiss  # pylint: disable=import-outside-toplevel

            # Load model
            logger.info("Loading sentence transformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

            # Load or create index
            index_file = self.index_path / "vectors.faiss"
            meta_file = self.index_path / "documents.json"

            if index_file.exists() and meta_file.exists():
                self._index = faiss.read_index(str(index_file))
                with open(meta_file, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                logger.info("Loaded existing index with %d documents", len(self._documents))
            else:
                # Create new index (Inner Product for cosine similarity)
                self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._documents = []
                logger.info("Created new FAISS index")

            self._initialized = True
            return True

        except ImportError as e:
            logger.warning(
                "Semantic search dependencies not available: %s. "
                "Install with: pip install sentence-transformers faiss-cpu",
                e
            )
            return False
        except Exception as e:
            logger.error("Failed to initialize semantic search: %s", e)
            return False

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add document with semantic embedding.

        Args:
            text: Document text content.
            metadata: Optional metadata dict.
            doc_id: Optional document ID (auto-generated if not provided).

        Returns:
            Document ID if successful, None otherwise.
        """
        if not self._ensure_initialized():
            return None

        if not text.strip():
            return None

        # Generate ID if not provided
        if not doc_id:
            import hashlib
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Check for duplicates
        existing_ids = [d.get("id") for d in self._documents]
        if doc_id in existing_ids:
            logger.debug("Document %s already exists, skipping", doc_id)
            return doc_id

        try:
            # Generate embedding
            embedding = self._model.encode([text], normalize_embeddings=True)[0]

            # Add to index
            self._index.add(np.array([embedding], dtype=np.float32))

            # Store metadata
            self._documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat(),
            })

            logger.debug("Added document %s to semantic index", doc_id)
            return doc_id

        except Exception as e:
            logger.error("Failed to add document: %s", e)
            return None

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Semantic similarity search.

        Args:
            query: Search query text.
            k: Number of results to return.
            min_score: Minimum similarity score (0-1).

        Returns:
            List of SearchResult sorted by relevance.
        """
        if not self._ensure_initialized():
            return []

        if not query.strip():
            return []

        if len(self._documents) == 0:
            return []

        try:
            # Encode query
            query_vec = self._model.encode([query], normalize_embeddings=True)[0]

            # Search
            k_actual = min(k, len(self._documents))
            scores, indices = self._index.search(
                np.array([query_vec], dtype=np.float32),
                k_actual
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._documents):
                    continue

                score = float(scores[0][i])
                if score < min_score:
                    continue

                doc = self._documents[idx]
                results.append(SearchResult(
                    text=doc["text"],
                    score=score,
                    metadata=doc.get("metadata", {}),
                    doc_id=doc.get("id", ""),
                ))

            return results

        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def save(self) -> bool:
        """
        Save index and documents to disk.

        Returns:
            True if saved successfully.
        """
        if not self._initialized or self._index is None:
            return False

        try:
            import faiss  # pylint: disable=import-outside-toplevel

            index_file = self.index_path / "vectors.faiss"
            meta_file = self.index_path / "documents.json"

            faiss.write_index(self._index, str(index_file))
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self._documents, f, indent=2)

            logger.info("Saved semantic index (%d documents)", len(self._documents))
            return True

        except Exception as e:
            logger.error("Failed to save index: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics.

        Returns:
            Dict with document count, index size, etc.
        """
        initialized = self._ensure_initialized()
        return {
            "initialized": initialized,
            "document_count": len(self._documents) if initialized else 0,
            "model_name": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
            "index_path": str(self.index_path),
            "index_trained": self._index.is_trained if initialized and self._index else False,
        }

    def clear(self) -> None:
        """Clear all documents and reset index."""
        if not self._ensure_initialized():
            return

        try:
            import faiss  # pylint: disable=import-outside-toplevel

            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._documents = []
            self.save()
            logger.info("Cleared semantic index")
        except Exception as e:
            logger.error("Failed to clear index: %s", e)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID.

        Note: FAISS doesn't support direct deletion. This rebuilds the index.

        Args:
            doc_id: Document ID to delete.

        Returns:
            True if deleted successfully.
        """
        if not self._ensure_initialized():
            return False

        # Find document
        idx_to_delete = None
        for i, doc in enumerate(self._documents):
            if doc.get("id") == doc_id:
                idx_to_delete = i
                break

        if idx_to_delete is None:
            return False

        # Remove from documents list
        del self._documents[idx_to_delete]

        # Rebuild index (FAISS doesn't support deletion)
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from documents."""
        try:
            import faiss  # pylint: disable=import-outside-toplevel

            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)

            if self._documents:
                texts = [d["text"] for d in self._documents]
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                self._index.add(np.array(embeddings, dtype=np.float32))

            self.save()
            logger.info("Rebuilt index with %d documents", len(self._documents))

        except Exception as e:
            logger.error("Failed to rebuild index: %s", e)


# Singleton storage
_singleton: Dict[str, Optional[SemanticCorpus]] = {"corpus": None}


def get_semantic_corpus() -> SemanticCorpus:
    """Get global semantic corpus instance."""
    if _singleton["corpus"] is None:
        _singleton["corpus"] = SemanticCorpus()
    return _singleton["corpus"]


def reset_semantic_corpus() -> None:
    """Reset global semantic corpus instance."""
    _singleton["corpus"] = None


if __name__ == "__main__":
    import time

    print("DAIMON Semantic Search - Demo")
    print("=" * 50)

    corpus = SemanticCorpus()

    # Add sample documents
    print("\nAdding sample documents...")
    docs = [
        ("The only true wisdom is in knowing you know nothing.", {"author": "Socrates"}),
        ("Virtue is knowledge.", {"author": "Socrates"}),
        ("The unexamined life is not worth living.", {"author": "Socrates"}),
        ("We are what we repeatedly do.", {"author": "Aristotle"}),
        ("Happiness depends upon ourselves.", {"author": "Aristotle"}),
    ]

    for text, meta in docs:
        doc_id = corpus.add_document(text, meta)
        print(f"  Added: {doc_id[:8]}... - {text[:40]}...")

    corpus.save()

    # Search
    print("\nSearching for 'wisdom and knowledge'...")
    start = time.perf_counter()
    results = corpus.search("wisdom and knowledge", k=3)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Search completed in {elapsed:.2f}ms")
    for r in results:
        print(f"  [{r.score:.3f}] {r.metadata.get('author', '?')}: {r.text[:50]}...")

    # Stats
    print("\nCorpus stats:")
    stats = corpus.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
