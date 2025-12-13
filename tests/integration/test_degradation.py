"""
Integration Tests: Graceful Degradation
=======================================

E2E tests for fallback behavior when services are unavailable.

Scenarios:
    - NOESIS offline → Local storage fallback
    - LLM unavailable → Heuristic fallback

Follows CODE_CONSTITUTION: test_<method>_<scenario>_<expected>.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from memory.user_model import UserModelService, reset_user_model_service
from memory.user_models import UserModel, UserPreferences
from learners.preference import SignalDetector, InsightGenerator, CategoryStats
from learners.preference_learner import PreferenceLearner


@pytest.mark.integration
class TestNoesisOffline:
    """Tests for behavior when NOESIS is unavailable."""

    def test_user_model_falls_back_to_local(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test UserModelService falls back to local storage.
        
        Given: NOESIS is offline
        When: Saving and loading user model
        Then: Data is persisted locally
        """
        # Given: Service with NOESIS URL but offline
        service = UserModelService(
            storage_path=tmp_storage["user_model"],
            noesis_url="http://nonexistent:9999",
        )
        
        model = UserModel(
            user_id="fallback_test",
            preferences=UserPreferences(code_style="fallback_value"),
        )
        
        # When: Save (NOESIS will fail, local should succeed)
        success = service.save(model)
        
        # Then: Local save succeeded
        assert success
        assert tmp_storage["user_model"].exists()
        
        # When: Clear and reload
        service.clear_cache()
        loaded = service.load("fallback_test")
        
        # Then: Data loaded from local
        assert loaded.preferences.code_style == "fallback_value"

    def test_multiple_users_in_local_storage(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test multiple users can be stored locally.
        
        Given: Multiple users saved locally
        When: Loading each user
        Then: Correct data is returned for each
        """
        # Given: Service with local-only storage
        service = UserModelService(storage_path=tmp_storage["user_model"])
        
        # Create multiple users
        for i in range(3):
            model = UserModel(
                user_id=f"user_{i}",
                preferences=UserPreferences(code_style=f"style_{i}"),
            )
            service.save(model)
        
        # When: Clear and reload each
        service.clear_cache()
        
        for i in range(3):
            loaded = service.load(f"user_{i}")
            
            # Then: Correct data
            assert loaded.preferences.code_style == f"style_{i}"


@pytest.mark.integration
class TestLLMFallback:
    """Tests for LLM fallback to heuristics."""

    def test_detector_uses_heuristics_when_llm_disabled(self) -> None:
        """
        Test detector falls back to heuristics.
        
        Given: LLM disabled
        When: Detecting signals
        Then: Heuristics are used correctly
        """
        # Given: Detector with LLM disabled
        detector = SignalDetector(enable_llm=False)
        
        # When: Detect signal types
        approval = detector._detect_heuristic("sim, ficou perfeito!")
        rejection = detector._detect_heuristic("nao, ta errado")
        neutral = detector._detect_heuristic("continue trabalhando")
        
        # Then: Heuristics work
        assert approval == "approval"
        assert rejection == "rejection"
        assert neutral is None

    def test_insight_generator_uses_templates_when_llm_disabled(self) -> None:
        """
        Test insight generator uses templates without LLM.
        
        Given: LLM disabled
        When: Generating insights
        Then: Template-based insights are produced
        """
        # Given: Generator with LLM disabled
        generator = InsightGenerator(enable_llm=False)
        
        stats = {
            "verbosity": CategoryStats(approvals=1, rejections=9),
            "testing": CategoryStats(approvals=9, rejections=1),
        }
        
        # When: Generate insights
        insights = generator.get_insights(stats, min_signals=3)
        
        # Then: Templates used
        assert len(insights) >= 2
        assert all(not i.from_llm for i in insights)
        
        # Verify suggestions are template-based
        verbosity_insight = next(i for i in insights if i.category == "verbosity")
        assert len(verbosity_insight.suggestion) > 0

    def test_preference_learner_works_without_llm(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test full learner workflow without LLM.
        
        Given: LLM disabled, session with feedback
        When: Running full workflow
        Then: System works with heuristics
        """
        # Given: Session file
        create_session_file("no-llm-test", sample_messages)
        
        # When: Process without LLM
        learner = PreferenceLearner(
            projects_dir=tmp_storage["projects"],
            enable_llm=False,
        )
        signals = learner.scan_sessions(since_hours=24)
        summary = learner.get_preference_summary()
        insights = learner.get_actionable_insights(min_signals=1)
        
        # Then: System works
        assert len(signals) >= 1
        stats = learner.get_stats()
        assert stats["total_signals"] >= 1


@pytest.mark.integration
class TestCombinedDegradation:
    """Tests for combined degradation scenarios."""

    def test_full_system_degraded_mode(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test full system works in degraded mode.
        
        Given: NOESIS offline, LLM disabled
        When: Running complete workflow
        Then: System operates with local fallbacks
        """
        reset_user_model_service()
        
        # Given: Session file
        create_session_file("degraded-test", sample_messages)
        
        # When: Process in degraded mode
        learner = PreferenceLearner(
            projects_dir=tmp_storage["projects"],
            enable_llm=False,
        )
        signals = learner.scan_sessions(since_hours=24)
        
        # Save to local storage (NOESIS unavailable)
        model_service = UserModelService(
            storage_path=tmp_storage["user_model"],
            noesis_url="http://nonexistent:9999",
        )
        
        # Update model with findings
        for signal in signals:
            model_service.add_pattern("degraded_user", {
                "category": signal.category,
                "type": signal.signal_type,
                "confidence": signal.strength,
            })
        
        # Then: Data persisted locally
        model_service.clear_cache()
        loaded = model_service.load("degraded_user")
        
        if signals:
            assert len(loaded.patterns) >= 1

    def test_stats_reflect_degraded_mode(
        self,
        tmp_storage: Dict[str, Path],
    ) -> None:
        """
        Test stats correctly reflect degraded mode.
        
        Given: Service in degraded mode
        When: Checking stats
        Then: Stats indicate local operation
        """
        # Given: Service with failed NOESIS
        service = UserModelService(
            storage_path=tmp_storage["user_model"],
            noesis_url="http://nonexistent:9999",
        )
        
        # When: Get stats
        stats = service.get_stats()
        
        # Then: Stats show local path
        assert "storage_path" in stats
        assert str(tmp_storage["user_model"]) in stats["storage_path"]
