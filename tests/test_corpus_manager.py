"""
Tests for corpus/manager.py and corpus/bootstrap_texts.py

Scientific tests covering:
- TextMetadata and WisdomText dataclasses
- Corpus initialization
- Text CRUD operations
- Search functionality
- Index management
- Bootstrap texts loading
"""

import json
import shutil
from pathlib import Path

import pytest

from corpus.manager import (
    CATEGORIES,
    CorpusManager,
    TextMetadata,
    WisdomText,
)
from corpus.bootstrap_texts import BOOTSTRAP_TEXTS, bootstrap_corpus


class TestTextMetadata:
    """Tests for TextMetadata dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating metadata with default values."""
        meta = TextMetadata()
        assert meta.source == ""
        assert meta.added_at == ""
        assert meta.relevance_score == 0.5
        assert meta.themes == []

    def test_create_with_values(self) -> None:
        """Test creating metadata with custom values."""
        meta = TextMetadata(
            source="Meditations, Book II",
            added_at="2025-01-01T00:00:00",
            relevance_score=0.9,
            themes=["virtue", "stoicism"],
        )
        assert meta.source == "Meditations, Book II"
        assert meta.added_at == "2025-01-01T00:00:00"
        assert meta.relevance_score == 0.9
        assert meta.themes == ["virtue", "stoicism"]


class TestWisdomText:
    """Tests for WisdomText dataclass."""

    def test_create_text(self) -> None:
        """Test creating a WisdomText."""
        meta = TextMetadata(source="Test Source", relevance_score=0.8, themes=["test"])
        text = WisdomText(
            id="test123",
            author="Test Author",
            title="Test Title",
            category="filosofia/estoicos",
            content="Test content here.",
            themes=["wisdom", "test"],
            metadata=meta,
        )
        assert text.id == "test123"
        assert text.author == "Test Author"
        assert text.title == "Test Title"
        assert text.category == "filosofia/estoicos"
        assert text.content == "Test content here."
        assert text.themes == ["wisdom", "test"]
        assert text.source == "Test Source"
        assert text.relevance_score == 0.8

    def test_create_text_default_metadata(self) -> None:
        """Test creating text with default metadata."""
        text = WisdomText(
            id="test",
            author="Author",
            title="Title",
            category="logica",
            content="Content",
            themes=[],
        )
        assert text.metadata is not None
        assert text.source == ""
        assert text.added_at == ""

    def test_to_dict(self) -> None:
        """Test converting text to dictionary."""
        meta = TextMetadata(source="src", relevance_score=0.7, themes=["t1"])
        text = WisdomText(
            id="id1",
            author="Author",
            title="Title",
            category="logica",
            content="Content",
            themes=["t1", "t2"],
            metadata=meta,
        )
        d = text.to_dict()
        assert d["id"] == "id1"
        assert d["author"] == "Author"
        assert d["source"] == "src"
        assert d["relevance_score"] == 0.7

    def test_from_dict(self) -> None:
        """Test creating text from dictionary."""
        data = {
            "id": "id1",
            "author": "Author",
            "title": "Title",
            "category": "ciencia",
            "content": "Content",
            "themes": ["a", "b"],
            "source": "source text",
            "added_at": "2025-01-01",
            "relevance_score": 0.6,
        }
        text = WisdomText.from_dict(data)
        assert text.id == "id1"
        assert text.source == "source text"
        assert text.relevance_score == 0.6


class TestCategories:
    """Tests for category constants."""

    def test_categories_exist(self) -> None:
        """Test that categories are defined."""
        assert len(CATEGORIES) > 0

    def test_expected_categories(self) -> None:
        """Test expected categories exist."""
        assert "filosofia/gregos" in CATEGORIES
        assert "filosofia/estoicos" in CATEGORIES
        assert "logica" in CATEGORIES
        assert "ciencia" in CATEGORIES
        assert "etica" in CATEGORIES


class TestCorpusManagerInit:
    """Tests for CorpusManager initialization."""

    def test_init_creates_structure(self, tmp_path: Path) -> None:
        """Test that init creates directory structure."""
        manager = CorpusManager(tmp_path / "corpus")

        # Check category directories created
        for category in CATEGORIES:
            assert (tmp_path / "corpus" / category).exists()

        # Check index directory created
        assert (tmp_path / "corpus" / "_index").exists()

    def test_init_loads_empty_indices(self, tmp_path: Path) -> None:
        """Test init with no existing data."""
        manager = CorpusManager(tmp_path / "corpus")
        assert manager.by_theme == {}
        assert manager.by_author == {}
        assert manager.texts == {}


class TestGenerateTextId:
    """Tests for CorpusManager.generate_text_id()."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CorpusManager:
        """Create a CorpusManager instance."""
        return CorpusManager(tmp_path / "corpus")

    def test_generate_id_basic(self, manager: CorpusManager) -> None:
        """Test basic ID generation."""
        text_id = manager.generate_text_id("Marcus Aurelius", "Meditations")
        assert text_id is not None
        assert len(text_id) > 0
        assert "_" in text_id

    def test_generate_id_consistent(self, manager: CorpusManager) -> None:
        """Test ID generation is consistent."""
        id1 = manager.generate_text_id("Author", "Title")
        id2 = manager.generate_text_id("Author", "Title")
        assert id1 == id2

    def test_generate_id_different(self, manager: CorpusManager) -> None:
        """Test different inputs generate different IDs."""
        id1 = manager.generate_text_id("Author1", "Title")
        id2 = manager.generate_text_id("Author2", "Title")
        assert id1 != id2


class TestAddText:
    """Tests for CorpusManager.add_text()."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CorpusManager:
        """Create a CorpusManager instance."""
        return CorpusManager(tmp_path / "corpus")

    def test_add_text_basic(self, manager: CorpusManager) -> None:
        """Test adding a basic text."""
        text_id = manager.add_text(
            author="Test Author",
            title="Test Title",
            category="filosofia/estoicos",
            content="Test content here.",
        )
        assert text_id is not None
        assert text_id in manager.texts

    def test_add_text_with_metadata(self, manager: CorpusManager) -> None:
        """Test adding text with metadata."""
        meta = TextMetadata(
            source="Test Source",
            relevance_score=0.9,
            themes=["virtue", "wisdom"],
        )
        text_id = manager.add_text(
            author="Test Author",
            title="Test Title",
            category="filosofia/gregos",
            content="Content",
            metadata=meta,
        )
        text = manager.texts[text_id]
        assert text.source == "Test Source"
        assert text.relevance_score == 0.9

    def test_add_text_unknown_category(self, manager: CorpusManager) -> None:
        """Test adding text with unknown category defaults to filosofia/modernos."""
        text_id = manager.add_text(
            author="Author",
            title="Title",
            category="unknown_category",
            content="Content",
        )
        text = manager.texts[text_id]
        assert text.category == "filosofia/modernos"

    def test_add_text_updates_indices(self, manager: CorpusManager) -> None:
        """Test that adding text updates indices."""
        meta = TextMetadata(themes=["virtue", "ethics"])
        text_id = manager.add_text(
            author="Test Author",
            title="Test Title",
            category="etica",
            content="Content",
            metadata=meta,
        )

        # Check author index
        assert "Test Author" in manager.by_author
        assert text_id in manager.by_author["Test Author"]

        # Check theme index
        assert "virtue" in manager.by_theme
        assert text_id in manager.by_theme["virtue"]

    def test_add_text_creates_file(self, manager: CorpusManager, tmp_path: Path) -> None:
        """Test that adding text creates a file."""
        text_id = manager.add_text(
            author="Test",
            title="Title",
            category="logica",
            content="Content",
        )
        text_file = tmp_path / "corpus" / "logica" / f"{text_id}.json"
        assert text_file.exists()


class TestGetText:
    """Tests for CorpusManager.get_text()."""

    @pytest.fixture
    def manager_with_text(self, tmp_path: Path) -> CorpusManager:
        """Create a CorpusManager with a test text."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Test Author",
            title="Test Title",
            category="logica",
            content="Test content",
        )
        return manager

    def test_get_existing(self, manager_with_text: CorpusManager) -> None:
        """Test getting existing text."""
        text_id = list(manager_with_text.texts.keys())[0]
        text = manager_with_text.get_text(text_id)
        assert text is not None
        assert text.author == "Test Author"

    def test_get_nonexistent(self, manager_with_text: CorpusManager) -> None:
        """Test getting non-existent text."""
        text = manager_with_text.get_text("nonexistent_id")
        assert text is None


class TestGetByAuthor:
    """Tests for CorpusManager.get_by_author()."""

    def test_get_by_author(self, tmp_path: Path) -> None:
        """Test getting texts by author."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Marcus Aurelius",
            title="Meditations I",
            category="filosofia/estoicos",
            content="Content 1",
        )
        manager.add_text(
            author="Marcus Aurelius",
            title="Meditations II",
            category="filosofia/estoicos",
            content="Content 2",
        )

        texts = manager.get_by_author("Marcus Aurelius")
        assert len(texts) == 2
        assert all(t.author == "Marcus Aurelius" for t in texts)

    def test_get_by_author_empty(self, tmp_path: Path) -> None:
        """Test getting texts by non-existent author."""
        manager = CorpusManager(tmp_path / "corpus")
        texts = manager.get_by_author("Unknown Author")
        assert texts == []


class TestGetByTheme:
    """Tests for CorpusManager.get_by_theme()."""

    def test_get_by_theme(self, tmp_path: Path) -> None:
        """Test getting texts by theme."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Author1",
            title="Title1",
            category="filosofia/gregos",
            content="Content",
            metadata=TextMetadata(themes=["virtue", "ethics"]),
        )
        manager.add_text(
            author="Author2",
            title="Title2",
            category="filosofia/gregos",
            content="Content",
            metadata=TextMetadata(themes=["virtue", "wisdom"]),
        )

        texts = manager.get_by_theme("virtue")
        assert len(texts) == 2

    def test_get_by_theme_empty(self, tmp_path: Path) -> None:
        """Test getting texts by non-existent theme."""
        manager = CorpusManager(tmp_path / "corpus")
        texts = manager.get_by_theme("nonexistent")
        assert texts == []


class TestGetByCategory:
    """Tests for CorpusManager.get_by_category()."""

    def test_get_by_category(self, tmp_path: Path) -> None:
        """Test getting texts by category."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Author",
            title="Title1",
            category="logica",
            content="Content1",
        )
        manager.add_text(
            author="Author",
            title="Title2",
            category="logica",
            content="Content2",
        )
        manager.add_text(
            author="Author",
            title="Title3",
            category="ciencia",
            content="Content3",
        )

        texts = manager.get_by_category("logica")
        assert len(texts) == 2
        assert all(t.category == "logica" for t in texts)


class TestSearch:
    """Tests for CorpusManager.search()."""

    @pytest.fixture
    def manager_with_texts(self, tmp_path: Path) -> CorpusManager:
        """Create a CorpusManager with test texts."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Marcus Aurelius",
            title="Meditations",
            category="filosofia/estoicos",
            content="Begin the morning by saying to thyself about virtue and wisdom.",
            metadata=TextMetadata(themes=["virtue", "morning"]),
        )
        manager.add_text(
            author="Aristotle",
            title="Ethics",
            category="filosofia/gregos",
            content="Every art aims at some good purpose and virtue.",
            metadata=TextMetadata(themes=["virtue", "ethics"]),
        )
        manager.add_text(
            author="Popper",
            title="Science",
            category="ciencia",
            content="Falsifiability is the criterion of science.",
            metadata=TextMetadata(themes=["science", "method"]),
        )
        return manager

    def test_search_by_content(self, manager_with_texts: CorpusManager) -> None:
        """Test searching by content."""
        results = manager_with_texts.search("virtue")
        assert len(results) >= 2

    def test_search_by_title(self, manager_with_texts: CorpusManager) -> None:
        """Test searching by title."""
        results = manager_with_texts.search("Meditations")
        assert len(results) >= 1
        assert any(r.title == "Meditations" for r in results)

    def test_search_by_theme(self, manager_with_texts: CorpusManager) -> None:
        """Test searching by theme."""
        results = manager_with_texts.search("science")
        assert len(results) >= 1

    def test_search_no_results(self, manager_with_texts: CorpusManager) -> None:
        """Test search with no results."""
        results = manager_with_texts.search("nonexistent_xyz_term")
        assert len(results) == 0

    def test_search_with_limit(self, manager_with_texts: CorpusManager) -> None:
        """Test search respects limit."""
        results = manager_with_texts.search("virtue", limit=1)
        assert len(results) <= 1


class TestListMethods:
    """Tests for list methods."""

    def test_list_authors(self, tmp_path: Path) -> None:
        """Test listing authors."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="Author1", title="T1", category="logica", content="C",
        )
        manager.add_text(
            author="Author2", title="T2", category="logica", content="C",
        )

        authors = manager.list_authors()
        assert "Author1" in authors
        assert "Author2" in authors

    def test_list_themes(self, tmp_path: Path) -> None:
        """Test listing themes."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="A", title="T", category="logica", content="C",
            metadata=TextMetadata(themes=["theme1", "theme2"]),
        )

        themes = manager.list_themes()
        assert "theme1" in themes
        assert "theme2" in themes

    def test_list_categories(self, tmp_path: Path) -> None:
        """Test listing categories."""
        manager = CorpusManager(tmp_path / "corpus")
        categories = manager.list_categories()
        assert categories == CATEGORIES


class TestGetStats:
    """Tests for CorpusManager.get_stats()."""

    def test_stats_empty(self, tmp_path: Path) -> None:
        """Test stats on empty corpus."""
        manager = CorpusManager(tmp_path / "corpus")
        stats = manager.get_stats()
        assert stats["total_texts"] == 0
        assert stats["total_authors"] == 0
        assert stats["total_themes"] == 0

    def test_stats_with_data(self, tmp_path: Path) -> None:
        """Test stats with data."""
        manager = CorpusManager(tmp_path / "corpus")
        manager.add_text(
            author="A1", title="T1", category="logica", content="C",
            metadata=TextMetadata(themes=["t1"]),
        )
        manager.add_text(
            author="A2", title="T2", category="ciencia", content="C",
            metadata=TextMetadata(themes=["t2"]),
        )

        stats = manager.get_stats()
        assert stats["total_texts"] == 2
        assert stats["total_authors"] == 2
        assert stats["total_themes"] == 2
        assert "logica" in stats["by_category"]
        assert "ciencia" in stats["by_category"]


class TestDeleteText:
    """Tests for CorpusManager.delete_text()."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        """Test deleting existing text."""
        manager = CorpusManager(tmp_path / "corpus")
        text_id = manager.add_text(
            author="Author",
            title="Title",
            category="logica",
            content="Content",
            metadata=TextMetadata(themes=["test"]),
        )

        result = manager.delete_text(text_id)
        assert result is True
        assert text_id not in manager.texts

        # Check file removed
        text_file = tmp_path / "corpus" / "logica" / f"{text_id}.json"
        assert not text_file.exists()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting non-existent text."""
        manager = CorpusManager(tmp_path / "corpus")
        result = manager.delete_text("nonexistent_id")
        assert result is False

    def test_delete_updates_indices(self, tmp_path: Path) -> None:
        """Test that delete updates indices."""
        manager = CorpusManager(tmp_path / "corpus")
        text_id = manager.add_text(
            author="Test Author",
            title="Title",
            category="logica",
            content="Content",
            metadata=TextMetadata(themes=["test_theme"]),
        )

        manager.delete_text(text_id)

        # Check removed from indices
        assert text_id not in manager.by_author.get("Test Author", [])
        assert text_id not in manager.by_theme.get("test_theme", [])


class TestBootstrapTexts:
    """Tests for bootstrap_texts.py."""

    def test_bootstrap_texts_defined(self) -> None:
        """Test bootstrap texts are defined."""
        assert len(BOOTSTRAP_TEXTS) > 0

    def test_bootstrap_texts_structure(self) -> None:
        """Test bootstrap texts have required fields."""
        for text in BOOTSTRAP_TEXTS:
            assert "author" in text
            assert "title" in text
            assert "category" in text
            assert "content" in text
            assert "themes" in text

    def test_bootstrap_corpus(self, tmp_path: Path) -> None:
        """Test bootstrapping corpus."""
        corpus_path = tmp_path / "corpus"
        result = bootstrap_corpus(corpus_path)

        assert result["added"] == len(BOOTSTRAP_TEXTS)
        assert result["skipped"] == 0
        assert result["total"] == len(BOOTSTRAP_TEXTS)
        assert result["corpus_stats"]["total_texts"] == len(BOOTSTRAP_TEXTS)

    def test_bootstrap_corpus_idempotent(self, tmp_path: Path) -> None:
        """Test bootstrapping is idempotent."""
        corpus_path = tmp_path / "corpus"

        # First bootstrap
        result1 = bootstrap_corpus(corpus_path)
        assert result1["added"] == len(BOOTSTRAP_TEXTS)

        # Second bootstrap should skip all
        result2 = bootstrap_corpus(corpus_path)
        assert result2["added"] == 0
        assert result2["skipped"] == len(BOOTSTRAP_TEXTS)


class TestWisdomTextPostInit:
    """Tests for WisdomText.__post_init__."""

    def test_post_init_sets_metadata(self) -> None:
        """Test that __post_init__ creates metadata when None."""
        text = WisdomText(
            id="t1",
            author="Author",
            title="Title",
            category="logica",
            content="Content",
            themes=[],
            metadata=None,
        )
        assert text.metadata is not None
        assert text.source == ""


class TestCorpusManagerIndexLoading:
    """Tests for index loading from disk."""

    def test_load_existing_indices(self, tmp_path: Path) -> None:
        """Test loading existing indices from disk."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir(parents=True)
        (corpus_path / "_index").mkdir()

        # Create index files
        by_theme = {"virtue": ["text1", "text2"]}
        by_author = {"Marcus": ["text1"]}
        (corpus_path / "_index" / "by_theme.json").write_text(
            json.dumps(by_theme)
        )
        (corpus_path / "_index" / "by_author.json").write_text(
            json.dumps(by_author)
        )

        manager = CorpusManager(corpus_path)
        assert manager.by_theme == by_theme
        assert manager.by_author == by_author


class TestCorpusPersistence:
    """Tests for corpus persistence across sessions."""

    def test_persistence(self, tmp_path: Path) -> None:
        """Test that corpus persists across manager instances."""
        corpus_path = tmp_path / "corpus"

        # First instance - add text
        manager1 = CorpusManager(corpus_path)
        text_id = manager1.add_text(
            author="Test",
            title="Persistence Test",
            category="logica",
            content="Should persist",
            metadata=TextMetadata(themes=["persistence"]),
        )

        # Second instance - should load existing text
        manager2 = CorpusManager(corpus_path)
        assert text_id in manager2.texts
        assert manager2.texts[text_id].author == "Test"
        assert "persistence" in manager2.by_theme
