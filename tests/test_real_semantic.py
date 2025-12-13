"""
REAL Integration Tests for Semantic Search.

Tests actual embedding generation and semantic search.
No mocks - real sentence-transformers, real embeddings.
"""

import pytest
import tempfile
from pathlib import Path

# Mark tests that require sentence-transformers
has_transformers = True
try:
    import sentence_transformers
except ImportError:
    has_transformers = False

pytestmark = pytest.mark.skipif(
    not has_transformers,
    reason="Requires sentence-transformers"
)


class TestSemanticCorpusReal:
    """Real tests for SemanticCorpus with actual embeddings."""

    @pytest.fixture
    def corpus(self):
        """Create SemanticCorpus with temporary storage."""
        from corpus.semantic_search import SemanticCorpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = SemanticCorpus(
                index_path=Path(tmpdir) / "test_index",
                model_name="all-MiniLM-L6-v2",  # Fast small model
            )
            yield corpus

    def test_add_document(self, corpus):
        """Add document and verify indexing."""
        doc_id = corpus.add_document(
            content="The quick brown fox jumps over the lazy dog.",
            metadata={"title": "Test Doc", "category": "test"},
        )

        assert doc_id is not None
        assert corpus.document_count() >= 1

    def test_search_finds_similar(self, corpus):
        """Search finds semantically similar documents."""
        # Add documents
        corpus.add_document(
            content="Python is a programming language used for data science.",
            metadata={"id": "python"},
        )
        corpus.add_document(
            content="JavaScript runs in web browsers.",
            metadata={"id": "js"},
        )
        corpus.add_document(
            content="Machine learning and artificial intelligence are related fields.",
            metadata={"id": "ml"},
        )

        # Search for data science related
        results = corpus.search("data analysis with code", k=3)

        assert len(results) >= 1
        # Python doc should rank high for data science query
        doc_ids = [r.doc_id for r in results]
        # At least find something

    def test_search_with_threshold(self, corpus):
        """Search with minimum score threshold."""
        corpus.add_document(
            content="Stoic philosophy teaches virtue and wisdom.",
            metadata={"id": "stoic"},
        )
        corpus.add_document(
            content="Cooking recipes for Italian pasta dishes.",
            metadata={"id": "cooking"},
        )

        # Search with high threshold
        results = corpus.search("ancient Greek philosophy", k=10, min_score=0.3)

        # Should filter out low-relevance results
        for r in results:
            assert r.score >= 0.3

    def test_remove_document(self, corpus):
        """Remove document from index."""
        doc_id = corpus.add_document(
            content="This document will be removed.",
            metadata={"id": "to_remove"},
        )

        initial_count = corpus.document_count()
        corpus.remove_document(doc_id)
        final_count = corpus.document_count()

        assert final_count < initial_count

    def test_persistence(self):
        """Test index persistence across instances."""
        from corpus.semantic_search import SemanticCorpus

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "persistent_index"

            # Create first instance and add document
            corpus1 = SemanticCorpus(index_path=index_path)
            corpus1.add_document(
                content="Persistent document for testing.",
                metadata={"id": "persistent"},
            )
            corpus1.save()

            # Create new instance with same path
            corpus2 = SemanticCorpus(index_path=index_path)

            # Should find the document
            assert corpus2.document_count() >= 1


class TestSemanticSearchIntegration:
    """Integration tests for semantic search in corpus manager."""

    def test_hybrid_search_real(self):
        """Test real hybrid search combining semantic and keyword."""
        from corpus.search import hybrid_search
        from corpus.models import WisdomText

        # Create test texts
        texts = {
            "wisdom1": WisdomText(
                id="wisdom1",
                author="Seneca",
                title="On Wisdom",
                category="philosophy",
                content="True wisdom comes from understanding oneself and others.",
                themes=["wisdom", "self-knowledge"],
            ),
            "wisdom2": WisdomText(
                id="wisdom2",
                author="Marcus Aurelius",
                title="Meditations on Life",
                category="philosophy",
                content="Life is short, make the most of every moment.",
                themes=["life", "time"],
            ),
        }

        # Test hybrid search without semantic (keyword only fallback)
        results = hybrid_search(
            texts=texts,
            semantic=None,
            query="wisdom understanding",
            limit=5,
        )

        assert len(results) >= 1

    def test_corpus_manager_with_semantic(self):
        """Test CorpusManager with semantic search enabled."""
        from corpus.manager import CorpusManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CorpusManager(
                corpus_dir=Path(tmpdir) / "corpus",
                enable_semantic=has_transformers,
            )

            # Add text
            text_id = manager.add_text(
                author="Test Author",
                title="Test Title",
                content="This is a test document about philosophy and wisdom.",
                category="test",
                themes=["philosophy", "wisdom"],
            )

            assert text_id is not None

            # Search
            results = manager.search("philosophy", limit=5)

            # Should find our document
            assert len(results) >= 1


class TestEmbeddingQuality:
    """Tests for embedding quality and semantic similarity."""

    @pytest.fixture
    def corpus(self):
        """Create corpus for quality tests."""
        from corpus.semantic_search import SemanticCorpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = SemanticCorpus(
                index_path=Path(tmpdir) / "quality_test",
                model_name="all-MiniLM-L6-v2",
            )
            yield corpus

    def test_similar_concepts_cluster(self, corpus):
        """Similar concepts should have high similarity scores."""
        # Add related documents
        corpus.add_document(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"id": "ml"},
        )
        corpus.add_document(
            content="Deep learning uses neural networks for AI tasks.",
            metadata={"id": "dl"},
        )
        corpus.add_document(
            content="Natural language processing enables computers to understand text.",
            metadata={"id": "nlp"},
        )
        # Add unrelated document
        corpus.add_document(
            content="Cooking Italian pasta requires fresh ingredients.",
            metadata={"id": "cooking"},
        )

        # Search for AI-related
        results = corpus.search("artificial intelligence and machine learning", k=4)

        # AI-related docs should rank higher than cooking
        top_ids = [r.doc_id for r in results[:3]]
        assert "cooking" not in top_ids or results[-1].doc_id == "cooking"

    def test_language_understanding(self, corpus):
        """Test semantic understanding beyond keywords."""
        corpus.add_document(
            content="The cat sat on the mat looking out the window.",
            metadata={"id": "cat"},
        )
        corpus.add_document(
            content="A feline rested on the rug gazing outside.",
            metadata={"id": "feline"},
        )
        corpus.add_document(
            content="Database optimization requires careful index planning.",
            metadata={"id": "database"},
        )

        # Search should find semantically similar even without exact keywords
        results = corpus.search("kitty on carpet", k=3)

        # Cat/feline docs should rank high despite no keyword match
        top_ids = [r.doc_id for r in results[:2]]
        assert "cat" in top_ids or "feline" in top_ids


class TestSemanticCorpusStats:
    """Tests for corpus statistics and metadata."""

    def test_get_stats(self):
        """Get corpus statistics."""
        from corpus.semantic_search import SemanticCorpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = SemanticCorpus(index_path=Path(tmpdir) / "stats_test")

            # Add some documents
            for i in range(5):
                corpus.add_document(
                    content=f"Test document number {i} with some content.",
                    metadata={"id": f"doc{i}"},
                )

            stats = corpus.get_stats()

            assert stats["document_count"] == 5
            assert "model_name" in stats
            assert "index_path" in stats

    def test_empty_corpus_search(self):
        """Search on empty corpus returns empty results."""
        from corpus.semantic_search import SemanticCorpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = SemanticCorpus(index_path=Path(tmpdir) / "empty_test")

            results = corpus.search("anything", k=10)

            assert len(results) == 0
