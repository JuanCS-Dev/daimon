"""
Tests for memory/user_model.py

Comprehensive tests covering:
- Dataclass creation and serialization
- UserModelService CRUD operations
- Local storage persistence
- Pattern management
- CLAUDE.md sync
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from memory.user_model import (
    UserPreferences,
    CognitiveProfile,
    UserModel,
    UserModelService,
    get_user_model_service,
    reset_user_model_service,
    MAX_PATTERNS,
)


class TestUserPreferences:
    """Tests for UserPreferences dataclass."""

    def test_create_default(self) -> None:
        """Test creating default preferences."""
        prefs = UserPreferences()
        assert prefs.communication_style == "balanced"
        assert prefs.code_style == "documented"
        assert prefs.preferred_tools == []
        assert prefs.language == "pt-BR"

    def test_create_custom(self) -> None:
        """Test creating custom preferences."""
        prefs = UserPreferences(
            communication_style="concise",
            code_style="minimal",
            preferred_tools=["Edit", "Bash"],
            language="en-US",
        )
        assert prefs.communication_style == "concise"
        assert prefs.code_style == "minimal"
        assert prefs.preferred_tools == ["Edit", "Bash"]
        assert prefs.language == "en-US"

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        prefs = UserPreferences(communication_style="detailed")
        d = prefs.to_dict()
        assert d["communication_style"] == "detailed"
        assert "code_style" in d
        assert "preferred_tools" in d

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "communication_style": "concise",
            "code_style": "verbose",
            "preferred_tools": ["Bash"],
            "language": "en-US",
        }
        prefs = UserPreferences.from_dict(data)
        assert prefs.communication_style == "concise"
        assert prefs.code_style == "verbose"
        assert prefs.preferred_tools == ["Bash"]

    def test_from_dict_with_defaults(self) -> None:
        """Test from_dict with missing keys uses defaults."""
        prefs = UserPreferences.from_dict({})
        assert prefs.communication_style == "balanced"
        assert prefs.code_style == "documented"


class TestCognitiveProfile:
    """Tests for CognitiveProfile dataclass."""

    def test_create_default(self) -> None:
        """Test creating default profile."""
        profile = CognitiveProfile()
        assert profile.avg_flow_duration == 45.0
        assert profile.fatigue_threshold == 0.7
        assert profile.break_preference == "flow"
        assert 10 in profile.peak_hours

    def test_create_custom(self) -> None:
        """Test creating custom profile."""
        profile = CognitiveProfile(
            avg_flow_duration=90.0,
            peak_hours=[9, 10, 11],
            fatigue_threshold=0.5,
            break_preference="pomodoro",
        )
        assert profile.avg_flow_duration == 90.0
        assert profile.peak_hours == [9, 10, 11]
        assert profile.fatigue_threshold == 0.5
        assert profile.break_preference == "pomodoro"

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        profile = CognitiveProfile(avg_flow_duration=60.0)
        d = profile.to_dict()
        assert d["avg_flow_duration"] == 60.0
        assert "peak_hours" in d

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "avg_flow_duration": 30.0,
            "peak_hours": [14, 15, 16],
            "fatigue_threshold": 0.8,
            "break_preference": "scheduled",
        }
        profile = CognitiveProfile.from_dict(data)
        assert profile.avg_flow_duration == 30.0
        assert profile.break_preference == "scheduled"


class TestUserModel:
    """Tests for UserModel dataclass."""

    def test_create_default(self) -> None:
        """Test creating default model."""
        model = UserModel(user_id="test")
        assert model.user_id == "test"
        assert model.version == 1
        assert isinstance(model.preferences, UserPreferences)
        assert isinstance(model.cognitive, CognitiveProfile)
        assert model.patterns == []

    def test_create_with_components(self) -> None:
        """Test creating model with custom components."""
        prefs = UserPreferences(code_style="minimal")
        cognitive = CognitiveProfile(avg_flow_duration=60.0)
        
        model = UserModel(
            user_id="custom",
            preferences=prefs,
            cognitive=cognitive,
            patterns=[{"name": "test", "confidence": 0.9}],
            version=5,
        )
        assert model.preferences.code_style == "minimal"
        assert model.cognitive.avg_flow_duration == 60.0
        assert len(model.patterns) == 1
        assert model.version == 5

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        model = UserModel(user_id="test")
        d = model.to_dict()
        
        assert d["user_id"] == "test"
        assert "preferences" in d
        assert "cognitive" in d
        assert "patterns" in d
        assert "version" in d
        assert "last_updated" in d

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "user_id": "loaded",
            "preferences": {"communication_style": "detailed"},
            "cognitive": {"avg_flow_duration": 90.0},
            "patterns": [{"name": "p1", "confidence": 0.8}],
            "version": 3,
            "last_updated": "2025-01-01T12:00:00",
        }
        model = UserModel.from_dict(data)
        
        assert model.user_id == "loaded"
        assert model.preferences.communication_style == "detailed"
        assert model.cognitive.avg_flow_duration == 90.0
        assert len(model.patterns) == 1
        assert model.version == 3

    def test_from_dict_truncates_patterns(self) -> None:
        """Test that from_dict limits patterns to MAX_PATTERNS."""
        patterns = [{"name": f"p{i}", "confidence": 0.5} for i in range(150)]
        data = {"user_id": "test", "patterns": patterns}
        
        model = UserModel.from_dict(data)
        assert len(model.patterns) == MAX_PATTERNS


class TestUserModelService:
    """Tests for UserModelService."""

    @pytest.fixture
    def tmp_storage(self, tmp_path: Path) -> Path:
        """Create temporary storage path."""
        return tmp_path / "user_model.json"

    @pytest.fixture
    def service(self, tmp_storage: Path) -> UserModelService:
        """Create service with temporary storage."""
        return UserModelService(storage_path=tmp_storage)

    def test_load_creates_default(self, service: UserModelService) -> None:
        """Test loading creates default model if not found."""
        model = service.load("new_user")
        assert model.user_id == "new_user"
        assert model.version == 1

    def test_save_and_load(self, service: UserModelService) -> None:
        """Test save and load cycle."""
        model = UserModel(user_id="test_user")
        model.preferences.code_style = "verbose"
        
        service.save(model)
        
        # Clear cache and reload
        service.clear_cache()
        loaded = service.load("test_user")
        
        assert loaded.user_id == "test_user"
        assert loaded.preferences.code_style == "verbose"

    def test_save_increments_version(self, service: UserModelService) -> None:
        """Test that save increments version."""
        model = service.load("test")
        initial_version = model.version
        
        service.save(model)
        
        assert model.version == initial_version + 1

    def test_update_preferences(self, service: UserModelService) -> None:
        """Test updating preferences."""
        model = service.update_preferences(
            "test",
            {"communication_style": "concise", "code_style": "minimal"},
        )
        
        assert model.preferences.communication_style == "concise"
        assert model.preferences.code_style == "minimal"

    def test_update_cognitive(self, service: UserModelService) -> None:
        """Test updating cognitive profile."""
        model = service.update_cognitive(
            "test",
            {"avg_flow_duration": 120.0, "fatigue_threshold": 0.5},
        )
        
        assert model.cognitive.avg_flow_duration == 120.0
        assert model.cognitive.fatigue_threshold == 0.5

    def test_add_pattern(self, service: UserModelService) -> None:
        """Test adding a pattern."""
        model = service.add_pattern(
            "test",
            {"name": "commit_at_5pm", "confidence": 0.85},
        )
        
        assert len(model.patterns) == 1
        assert model.patterns[0]["name"] == "commit_at_5pm"

    def test_add_pattern_maintains_limit(self, service: UserModelService) -> None:
        """Test that add_pattern maintains MAX_PATTERNS limit."""
        # Add more than MAX_PATTERNS
        for i in range(MAX_PATTERNS + 10):
            service.add_pattern(
                "test",
                {"name": f"pattern_{i}", "confidence": i / 200},
            )
        
        model = service.load("test")
        assert len(model.patterns) == MAX_PATTERNS
        
        # Verify sorted by confidence (highest first)
        confidences = [p["confidence"] for p in model.patterns]
        assert confidences == sorted(confidences, reverse=True)

    def test_local_storage_persistence(
        self, service: UserModelService, tmp_storage: Path
    ) -> None:
        """Test that data persists to local file."""
        model = UserModel(user_id="persist_test")
        model.preferences.language = "en-US"
        service.save(model)
        
        # Verify file exists and contains data
        assert tmp_storage.exists()
        data = json.loads(tmp_storage.read_text())
        assert "models" in data
        assert "persist_test" in data["models"]
        assert data["models"]["persist_test"]["preferences"]["language"] == "en-US"

    def test_get_stats(self, service: UserModelService) -> None:
        """Test getting service stats."""
        service.load("user1")
        service.load("user2")
        
        stats = service.get_stats()
        assert stats["cached_users"] == 2
        assert "storage_path" in stats

    def test_clear_cache(self, service: UserModelService) -> None:
        """Test clearing cache."""
        service.load("user1")
        assert len(service._cache) == 1
        
        service.clear_cache()
        assert len(service._cache) == 0


class TestClaudeMdSync:
    """Tests for CLAUDE.md synchronization."""

    @pytest.fixture
    def claude_md(self, tmp_path: Path) -> Path:
        """Create temporary CLAUDE.md file."""
        claude_path = tmp_path / "CLAUDE.md"
        claude_path.write_text("# My Project\n\nSome content here.\n")
        return claude_path

    @pytest.fixture
    def service(self, tmp_path: Path) -> UserModelService:
        """Create service with temporary storage."""
        return UserModelService(storage_path=tmp_path / "user_model.json")

    def test_sync_appends_section(
        self, service: UserModelService, claude_md: Path
    ) -> None:
        """Test that sync appends DAIMON section."""
        model = UserModel(user_id="test")
        model.preferences.communication_style = "concise"
        
        service.sync_with_claude_md(model, claude_md)
        
        content = claude_md.read_text()
        assert "<!-- DAIMON:START -->" in content
        assert "<!-- DAIMON:END -->" in content
        assert "concise" in content
        assert "Learned Preferences" in content

    def test_sync_replaces_existing_section(
        self, service: UserModelService, claude_md: Path
    ) -> None:
        """Test that sync replaces existing section."""
        # First sync
        model = UserModel(user_id="test")
        model.preferences.code_style = "minimal"
        service.sync_with_claude_md(model, claude_md)
        
        # Second sync with different value
        model.preferences.code_style = "verbose"
        service.sync_with_claude_md(model, claude_md)
        
        content = claude_md.read_text()
        assert content.count("<!-- DAIMON:START -->") == 1
        assert "verbose" in content
        assert "minimal" not in content

    def test_sync_nonexistent_file_no_error(
        self, service: UserModelService, tmp_path: Path
    ) -> None:
        """Test that sync handles nonexistent file gracefully."""
        model = UserModel(user_id="test")
        nonexistent = tmp_path / "nonexistent.md"
        
        # Should not raise
        service.sync_with_claude_md(model, nonexistent)


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_user_model_service()

    def test_get_user_model_service_singleton(self) -> None:
        """Test singleton instance."""
        service1 = get_user_model_service()
        service2 = get_user_model_service()
        assert service1 is service2

    def test_reset_user_model_service(self) -> None:
        """Test reset clears singleton."""
        service1 = get_user_model_service()
        reset_user_model_service()
        service2 = get_user_model_service()
        assert service1 is not service2
