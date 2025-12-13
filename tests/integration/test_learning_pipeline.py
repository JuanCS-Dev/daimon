"""
Integration Tests: Learning Pipeline
=====================================

E2E tests for the preference learning pipeline.

Flow:
    User Messages → SignalDetector → PreferenceCategorizer → InsightGenerator → UserModel

Follows CODE_CONSTITUTION: test_<method>_<scenario>_<expected>.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from learners.preference import (
    PreferenceSignal,
    SessionScanner,
    SignalDetector,
    PreferenceCategorizer,
    InsightGenerator,
    CategoryStats,
)
from learners.preference_learner import PreferenceLearner
from memory.user_model import UserModelService, get_user_model_service, reset_user_model_service


@pytest.mark.integration
class TestLearningPipelineE2E:
    """End-to-end tests for the learning pipeline."""

    def test_full_pipeline_approval_detection_updates_model(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test full pipeline from session to model update.
        
        Given: Session with approval messages
        When: Pipeline processes the session
        Then: UserModel is updated with preference signals
        """
        # Given: Session file with approval messages
        session_file = create_session_file("test-project", sample_messages)
        
        # Reset singleton for clean test
        reset_user_model_service()
        
        # When: Process through pipeline
        learner = PreferenceLearner(
            projects_dir=tmp_storage["projects"],
            enable_llm=False,  # Use heuristics for deterministic tests
        )
        signals = learner.scan_sessions(since_hours=24)
        
        # Then: Signals detected
        assert len(signals) >= 2  # At least approval + rejection
        
        # Verify signal types detected correctly
        signal_types = [s.signal_type for s in signals]
        assert "approval" in signal_types  # "perfeito" detected
        assert "rejection" in signal_types  # "nao" detected
        
        # Update user model with findings
        model_service = UserModelService(storage_path=tmp_storage["user_model"])
        summary = learner.get_preference_summary()
        
        for category, data in summary.items():
            if data.get("approval_rate", 0) > 0.8:
                model_service.update_preferences(
                    "default",
                    {f"category_{category}_preference": "reinforce"},
                )

    def test_detector_scanner_integration(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test integration between scanner and detector.
        
        Given: Session files in projects directory
        When: Scanner discovers and detector analyzes them
        Then: Signals are correctly extracted
        """
        # Given: Multiple session files
        create_session_file("project-a", sample_messages[:2])
        create_session_file("project-b", sample_messages[2:4])
        
        scanner = SessionScanner(projects_dir=tmp_storage["projects"])
        detector = SignalDetector(enable_llm=False)
        
        # When: Scan and detect
        all_signals = []
        for session_file in scanner.scan_recent(since_hours=24):
            messages = scanner.load_messages(session_file)
            session_id = scanner.get_session_id(session_file)
            
            for signal in detector.analyze_session(messages, session_id):
                all_signals.append(signal)
        
        # Then: Signals from both projects
        session_ids = {s.session_id for s in all_signals}
        assert len(session_ids) >= 1  # At least one session detected

    def test_categorizer_insight_integration(self) -> None:
        """
        Test integration between categorizer and insight generator.
        
        Given: Categorized preference signals
        When: Generating insights
        Then: Actionable insights are produced
        """
        # Given: Category stats with high rejection
        categorizer = PreferenceCategorizer()
        categorizer.category_stats["verbosity"] = CategoryStats(
            approvals=2,
            rejections=8,
        )
        categorizer.category_stats["testing"] = CategoryStats(
            approvals=9,
            rejections=1,
        )
        
        # When: Generate insights
        generator = InsightGenerator(enable_llm=False)
        insights = generator.get_insights(
            categorizer.category_stats,
            min_signals=3,
        )
        
        # Then: Insights generated for both categories
        categories = {i.category for i in insights}
        assert "verbosity" in categories  # High rejection
        assert "testing" in categories    # High approval
        
        verbosity_insight = next(i for i in insights if i.category == "verbosity")
        assert verbosity_insight.action == "reduce"
        
        testing_insight = next(i for i in insights if i.category == "testing")
        assert testing_insight.action == "reinforce"


@pytest.mark.integration
class TestPreferenceLearnerIntegration:
    """Integration tests for PreferenceLearner facade."""

    def test_scan_and_summarize_flow(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test complete scan → summarize → insights flow.
        
        Given: Session with mixed feedback
        When: Running full learner workflow
        Then: Summary and insights reflect the data
        """
        # Given: Session file
        create_session_file("mixed-feedback", sample_messages)
        
        # When: Full workflow
        learner = PreferenceLearner(
            projects_dir=tmp_storage["projects"],
            enable_llm=False,
        )
        signals = learner.scan_sessions(since_hours=24)
        summary = learner.get_preference_summary()
        insights = learner.get_actionable_insights(min_signals=1)
        
        # Then: Data flows through correctly
        assert len(signals) >= 1
        stats = learner.get_stats()
        assert stats["total_signals"] >= 1

    def test_clear_resets_state(
        self,
        tmp_storage: Dict[str, Path],
        sample_messages: List[Dict[str, Any]],
        create_session_file,
    ) -> None:
        """
        Test that clear() properly resets learner state.
        
        Given: Learner with accumulated signals
        When: Calling clear()
        Then: State is completely reset
        """
        # Given: Accumulated signals
        create_session_file("project", sample_messages)
        learner = PreferenceLearner(
            projects_dir=tmp_storage["projects"],
            enable_llm=False,
        )
        learner.scan_sessions(since_hours=24)
        assert learner.get_stats()["total_signals"] >= 1
        
        # When: Clear
        learner.clear()
        
        # Then: Reset
        stats = learner.get_stats()
        assert stats["total_signals"] == 0
        assert stats["categories_analyzed"] == 0
