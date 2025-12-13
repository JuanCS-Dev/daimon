"""
Integration Tests: Persistence
==============================

E2E tests for user model persistence.

Flow:
    Save → Clear Cache → Load → Verify

Follows CODE_CONSTITUTION: test_<method>_<scenario>_<expected>.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from memory.user_model import (
    UserModelService,
    get_user_model_service,
    reset_user_model_service,
)
from memory.user_models import UserModel, UserPreferences, CognitiveProfile


@pytest.mark.integration
class TestPersistenceE2E:
    """End-to-end tests for persistence."""

    def test_full_persistence_cycle(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test complete save → clear → load cycle.
        
        Given: UserModel with custom preferences
        When: Saving, clearing cache, and loading
        Then: Data is correctly persisted and retrieved
        """
        # Given: Custom user model
        service = UserModelService(storage_path=tmp_storage["user_model"])
        
        model = UserModel(
            user_id="persistence_test",
            preferences=UserPreferences(
                communication_style="concise",
                code_style="minimal",
                language="en-US",
            ),
            cognitive=CognitiveProfile(
                avg_flow_duration=90.0,
                peak_hours=[10, 11, 14, 15],
                fatigue_threshold=0.6,
            ),
        )
        
        # When: Save
        service.save(model)
        
        # Then: File exists
        assert tmp_storage["user_model"].exists()
        
        # When: Clear cache and load
        service.clear_cache()
        loaded = service.load("persistence_test")
        
        # Then: Data matches
        assert loaded.user_id == "persistence_test"
        assert loaded.preferences.communication_style == "concise"
        assert loaded.preferences.code_style == "minimal"
        assert loaded.cognitive.avg_flow_duration == 90.0
        assert loaded.cognitive.fatigue_threshold == 0.6

    def test_pattern_persistence(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test that patterns are correctly persisted.
        
        Given: Model with learned patterns
        When: Saving and reloading
        Then: Patterns are preserved
        """
        # Given: Model with patterns
        service = UserModelService(storage_path=tmp_storage["user_model"])
        
        service.add_pattern("pattern_test", {
            "name": "commit_at_5pm",
            "type": "temporal",
            "confidence": 0.9,
        })
        service.add_pattern("pattern_test", {
            "name": "test_before_push",
            "type": "sequential",
            "confidence": 0.85,
        })
        
        # When: Clear and reload
        service.clear_cache()
        loaded = service.load("pattern_test")
        
        # Then: Patterns preserved
        assert len(loaded.patterns) == 2
        pattern_names = {p["name"] for p in loaded.patterns}
        assert "commit_at_5pm" in pattern_names
        assert "test_before_push" in pattern_names

    def test_version_increments_on_save(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test that version increments on each save.
        
        Given: User model at version N
        When: Saving multiple times
        Then: Version increments correctly
        """
        # Given: Initial model
        service = UserModelService(storage_path=tmp_storage["user_model"])
        model = service.load("version_test")
        initial_version = model.version
        
        # When: Multiple saves
        service.save(model)
        service.save(model)
        service.save(model)
        
        # Then: Version incremented
        service.clear_cache()
        loaded = service.load("version_test")
        assert loaded.version == initial_version + 3


@pytest.mark.integration
class TestClaudeMdSync:
    """Integration tests for CLAUDE.md synchronization."""

    def test_sync_creates_section(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test syncing creates DAIMON section in CLAUDE.md.
        
        Given: CLAUDE.md without DAIMON section
        When: Syncing user model
        Then: DAIMON section is created
        """
        # Given: Model with preferences
        service = UserModelService(storage_path=tmp_storage["user_model"])
        model = UserModel(
            user_id="sync_test",
            preferences=UserPreferences(
                communication_style="detailed",
                code_style="verbose",
            ),
        )
        
        # When: Sync
        service.sync_with_claude_md(model, tmp_storage["claude_md"])
        
        # Then: Section created
        content = tmp_storage["claude_md"].read_text()
        assert "<!-- DAIMON:START -->" in content
        assert "<!-- DAIMON:END -->" in content
        assert "detailed" in content
        assert "verbose" in content

    def test_sync_updates_existing_section(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test syncing updates existing DAIMON section.
        
        Given: CLAUDE.md with existing DAIMON section
        When: Syncing updated model
        Then: Section is updated, not duplicated
        """
        # Given: Initial sync
        service = UserModelService(storage_path=tmp_storage["user_model"])
        model = UserModel(
            user_id="update_test",
            preferences=UserPreferences(communication_style="concise"),
        )
        service.sync_with_claude_md(model, tmp_storage["claude_md"])
        
        # When: Update and sync again
        model.preferences.communication_style = "detailed"
        service.sync_with_claude_md(model, tmp_storage["claude_md"])
        
        # Then: Only one section, with updated value
        content = tmp_storage["claude_md"].read_text()
        assert content.count("<!-- DAIMON:START -->") == 1
        assert "detailed" in content
        assert "concise" not in content


@pytest.mark.integration
class TestSingletonPersistence:
    """Tests for singleton service persistence."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_user_model_service()

    def test_singleton_shares_cache(self) -> None:
        """
        Test that singleton service shares cache.
        
        Given: Two references to singleton service
        When: One loads a model
        Then: Other can access from cache
        """
        # Given: Singleton references
        service1 = get_user_model_service()
        service2 = get_user_model_service()
        
        # When: One loads
        model = service1.load("cache_test")
        model.preferences.code_style = "cached_value"
        
        # Then: Other sees same instance (cache)
        cached = service2.load("cache_test")
        assert cached.preferences.code_style == "cached_value"
        assert cached is model  # Same instance
