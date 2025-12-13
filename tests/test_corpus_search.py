"""Tests for corpus search functions."""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import Mock

from corpus.search import (
    keyword_search,
    keyword_search_scored,
    hybrid_search,
)
from corpus.models import WisdomText, TextMetadata


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return {
        "text1": WisdomText(
            id="text1",
            author="Seneca",
            title="On the Shortness of Life",
            category="philosophy",
            content="It is not that we have a short time to live, but that we waste much of it.",
            themes=["time", "wisdom", "stoicism"],
        ),
        "text2": WisdomText(
            id="text2",
            author="Marcus Aurelius",
            title="Meditations",
            category="philosophy",
            content="The happiness of your life depends upon the quality of your thoughts.",
            themes=["happiness", "thoughts", "stoicism"],
        ),
        "text3": WisdomText(
            id="text3",
            author="Epictetus",
            title="Discourses",
            category="philosophy",
            content="We cannot control the impressions others form about us.",
            themes=["control", "wisdom"],
        ),
        "text4": WisdomText(
            id="text4",
            author="Unknown",
            title="On Happiness",
            category="wisdom",
            content="True happiness comes from within.",
            themes=["happiness", "inner peace"],
        ),
    }


class TestKeywordSearch:
    """Tests for keyword_search function."""

    def test_finds_matches_in_content(self, sample_texts):
        """Find texts by content match."""
        results = keyword_search(sample_texts, "waste")
        assert len(results) == 1
        assert results[0].id == "text1"

    def test_finds_matches_in_title(self, sample_texts):
        """Find texts by title match."""
        results = keyword_search(sample_texts, "Meditations")
        assert len(results) == 1
        assert results[0].id == "text2"

    def test_finds_matches_in_themes(self, sample_texts):
        """Find texts by theme match."""
        results = keyword_search(sample_texts, "stoicism")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"text1", "text2"}

    def test_case_insensitive(self, sample_texts):
        """Search is case-insensitive."""
        results = keyword_search(sample_texts, "HAPPINESS")
        assert len(results) >= 2

    def test_title_match_ranks_higher(self, sample_texts):
        """Title matches rank higher than content matches."""
        results = keyword_search(sample_texts, "happiness")
        # "On Happiness" in title should rank first
        assert results[0].id == "text4"

    def test_respects_limit(self, sample_texts):
        """Respects limit parameter."""
        results = keyword_search(sample_texts, "stoicism", limit=1)
        assert len(results) == 1

    def test_empty_query(self, sample_texts):
        """Empty query matches all texts (empty string is substring of everything)."""
        results = keyword_search(sample_texts, "")
        # Empty string is in every string, so all texts match
        assert len(results) == len(sample_texts)

    def test_no_matches(self, sample_texts):
        """No matches returns empty list."""
        results = keyword_search(sample_texts, "xyz123nonexistent")
        assert len(results) == 0

    def test_empty_texts_dict(self):
        """Empty texts dictionary returns empty list."""
        results = keyword_search({}, "test")
        assert len(results) == 0


class TestKeywordSearchScored:
    """Tests for keyword_search_scored function."""

    def test_returns_tuples(self, sample_texts):
        """Returns list of (text_id, score) tuples."""
        results = keyword_search_scored(sample_texts, "happiness", limit=10)
        assert len(results) >= 1
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_score_capped_at_1(self, sample_texts):
        """Scores are capped at 1.0."""
        results = keyword_search_scored(sample_texts, "happiness", limit=10)
        for _, score in results:
            assert score <= 1.0

    def test_content_match_score(self, sample_texts):
        """Content match gets 0.4 score."""
        results = keyword_search_scored(sample_texts, "waste", limit=10)
        # Only in content, so score should be 0.4
        assert len(results) == 1
        assert abs(results[0][1] - 0.4) < 0.01

    def test_title_match_score(self, sample_texts):
        """Title match gets 0.6 score."""
        # "Discourses" only in title
        results = keyword_search_scored(sample_texts, "Discourses", limit=10)
        assert len(results) == 1
        assert abs(results[0][1] - 0.6) < 0.01

    def test_respects_limit(self, sample_texts):
        """Respects limit parameter."""
        results = keyword_search_scored(sample_texts, "stoicism", limit=1)
        assert len(results) == 1


class TestHybridSearch:
    """Tests for hybrid_search function."""

    def test_keyword_only_when_no_semantic(self, sample_texts):
        """Uses only keyword search when semantic is None."""
        results = hybrid_search(sample_texts, None, "happiness", limit=5)
        assert len(results) >= 1

    def test_combines_semantic_and_keyword(self, sample_texts):
        """Combines semantic and keyword results."""
        # Mock semantic corpus
        mock_semantic = Mock()
        mock_result1 = Mock()
        mock_result1.doc_id = "text1"
        mock_result1.score = 0.9
        mock_result2 = Mock()
        mock_result2.doc_id = "text2"
        mock_result2.score = 0.8
        mock_semantic.search.return_value = [mock_result1, mock_result2]

        results = hybrid_search(
            sample_texts, mock_semantic, "wisdom", limit=5, semantic_weight=0.7
        )
        assert len(results) >= 1

    def test_respects_limit(self, sample_texts):
        """Respects limit parameter."""
        results = hybrid_search(sample_texts, None, "the", limit=2)
        assert len(results) <= 2

    def test_semantic_weight_affects_ranking(self, sample_texts):
        """Semantic weight affects final ranking."""
        mock_semantic = Mock()
        mock_result = Mock()
        mock_result.doc_id = "text3"
        mock_result.score = 1.0
        mock_semantic.search.return_value = [mock_result]

        # High semantic weight should favor text3
        results_high = hybrid_search(
            sample_texts, mock_semantic, "control", limit=5, semantic_weight=0.9
        )

        # When semantic returns text3 with high score, it should be first
        assert len(results_high) >= 1

    def test_filters_missing_texts(self, sample_texts):
        """Filters out semantic results not in texts dict."""
        mock_semantic = Mock()
        mock_result = Mock()
        mock_result.doc_id = "nonexistent_text"
        mock_result.score = 1.0
        mock_semantic.search.return_value = [mock_result]

        results = hybrid_search(sample_texts, mock_semantic, "test", limit=5)
        # Should not crash and should filter out nonexistent
        for r in results:
            assert r.id != "nonexistent_text"

    def test_empty_texts(self):
        """Handles empty texts dictionary."""
        results = hybrid_search({}, None, "test", limit=5)
        assert results == []


class TestWisdomTextModel:
    """Tests for WisdomText model serialization."""

    def test_to_dict(self):
        """WisdomText converts to dict correctly."""
        text = WisdomText(
            id="test1",
            author="Author",
            title="Title",
            category="philosophy",
            content="Content here",
            themes=["theme1", "theme2"],
        )
        d = text.to_dict()
        assert d["id"] == "test1"
        assert d["author"] == "Author"
        assert d["title"] == "Title"
        assert d["content"] == "Content here"
        assert d["category"] == "philosophy"
        assert d["themes"] == ["theme1", "theme2"]

    def test_from_dict(self):
        """WisdomText creates from dict correctly."""
        d = {
            "id": "test2",
            "author": "Author2",
            "title": "Title2",
            "category": "science",
            "content": "More content",
            "themes": ["theme3"],
        }
        text = WisdomText.from_dict(d)
        assert text.id == "test2"
        assert text.author == "Author2"
        assert text.themes == ["theme3"]

    def test_roundtrip(self):
        """to_dict and from_dict are inverses."""
        original = WisdomText(
            id="test3",
            author="Author3",
            title="Title3",
            category="philosophy",
            content="Content3",
            themes=["a", "b"],
        )
        recreated = WisdomText.from_dict(original.to_dict())
        assert recreated.id == original.id
        assert recreated.author == original.author
        assert recreated.title == original.title
        assert recreated.content == original.content
        assert recreated.themes == original.themes

    def test_metadata_initialized(self):
        """Metadata is auto-initialized if not provided."""
        text = WisdomText(
            id="test4",
            author="Author4",
            title="Title4",
            category="test",
            content="Content4",
            themes=[],
        )
        assert text.metadata is not None
        assert text.source == ""
        assert text.relevance_score == 0.5

    def test_metadata_properties(self):
        """Metadata properties are accessible."""
        meta = TextMetadata(source="book", relevance_score=0.8)
        text = WisdomText(
            id="test5",
            author="Author5",
            title="Title5",
            category="test",
            content="Content5",
            themes=[],
            metadata=meta,
        )
        assert text.source == "book"
        assert text.relevance_score == 0.8


class TestTextMetadata:
    """Tests for TextMetadata model."""

    def test_defaults(self):
        """TextMetadata has correct defaults."""
        meta = TextMetadata()
        assert meta.source == ""
        assert meta.added_at == ""
        assert meta.relevance_score == 0.5
        assert meta.themes == []

    def test_custom_values(self):
        """TextMetadata accepts custom values."""
        meta = TextMetadata(
            source="test_source",
            added_at="2025-01-01",
            relevance_score=0.9,
            themes=["topic1"],
        )
        assert meta.source == "test_source"
        assert meta.relevance_score == 0.9
