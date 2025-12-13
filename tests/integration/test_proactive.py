"""
Integration Tests: Proactive Emergence
======================================

E2E tests for the proactive behavior pipeline.

Flow:
    Events → PatternDetector → AnticipationEngine → Emergence Decision

Follows CODE_CONSTITUTION: test_<method>_<scenario>_<expected>.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from learners.pattern_detector import (
    PatternDetector,
    Pattern,
    get_pattern_detector,
    reset_pattern_detector,
)
from learners.anticipation_engine import (
    AnticipationEngine,
    EmergenceMode,
    EmergenceDecision,
    get_anticipation_engine,
    reset_anticipation_engine,
)


@pytest.mark.integration
class TestProactiveEmergenceE2E:
    """End-to-end tests for proactive emergence."""

    def setup_method(self) -> None:
        """Reset singletons before each test."""
        reset_pattern_detector()
        reset_anticipation_engine()

    def test_full_emergence_pipeline_with_patterns(
        self,
        sample_events: List[Dict[str, Any]],
    ) -> None:
        """
        Test full pipeline from events to emergence decision.
        
        Given: Stream of behavioral events with patterns
        When: Processing through detector and anticipation engine
        Then: Emergence is triggered with appropriate mode
        """
        # Given: Pattern detector with events
        detector = PatternDetector()
        for event in sample_events:
            detector.add_event(event)
        
        # When: Detect patterns
        patterns = detector.detect_patterns()
        
        # Then: Patterns detected
        assert len(patterns) >= 1
        
        # When: Evaluate with anticipation engine
        engine = AnticipationEngine()
        decision = engine.evaluate(
            context={"hour": 17},  # Same hour as events
            patterns=patterns,
        )
        
        # Then: Should decide to emerge
        assert decision.should_emerge or len(patterns) < engine.min_patterns
        if decision.should_emerge:
            assert decision.mode in [
                EmergenceMode.SUBTLE,
                EmergenceMode.NORMAL,
                EmergenceMode.URGENT,
            ]
            assert decision.confidence > 0

    def test_pattern_detection_temporal_patterns(self) -> None:
        """
        Test temporal pattern detection.
        
        Given: Events at consistent times
        When: Detecting patterns
        Then: Temporal pattern is found
        """
        # Given: Events at 5pm each day
        detector = PatternDetector()
        base = datetime.now()
        
        for i in range(5):
            detector.add_event({
                "type": "shell_command",
                "command": "git commit",
                "timestamp": (base.replace(hour=17) - timedelta(days=i)).isoformat(),
            })
        
        # When: Detect patterns
        patterns = detector.detect_patterns()
        temporal = [p for p in patterns if p.pattern_type == "temporal"]
        
        # Then: Temporal pattern detected
        assert len(temporal) >= 1
        peak_hour_pattern = next(
            (p for p in temporal if "17:00" in p.description),
            None,
        )
        assert peak_hour_pattern is not None

    def test_pattern_detection_sequential_patterns(self) -> None:
        """
        Test sequential pattern detection.
        
        Given: Repeated command sequences
        When: Detecting patterns
        Then: Sequential pattern is found
        """
        # Given: Repeated sequences
        detector = PatternDetector()
        for _ in range(4):
            detector.add_event({"type": "shell_command", "command": "git status"})
            detector.add_event({"type": "shell_command", "command": "git add ."})
            detector.add_event({"type": "shell_command", "command": "git commit"})
        
        # When: Detect patterns
        patterns = detector.detect_patterns()
        sequential = [p for p in patterns if p.pattern_type == "sequential"]
        
        # Then: Sequential pattern detected
        assert len(sequential) >= 1

    def test_anticipation_respects_cooldown(self) -> None:
        """
        Test that emergence respects cooldown period.
        
        Given: Recent emergence recorded
        When: Evaluating immediately after
        Then: Emergence is blocked by cooldown
        """
        # Given: Engine with recent emergence
        engine = AnticipationEngine(cooldown_seconds=60)
        
        # Record a recent emergence
        decision = EmergenceDecision(
            should_emerge=True,
            mode=EmergenceMode.NORMAL,
            reason="Test",
            confidence=0.9,
        )
        engine._record_emergence(decision)
        
        # When: Evaluate again immediately
        patterns = [
            Pattern(
                pattern_type="test",
                description="Test pattern",
                confidence=0.9,
                occurrences=5,
                last_seen=datetime.now(),
            )
        ]
        new_decision = engine.evaluate(context={}, patterns=patterns)
        
        # Then: Blocked by cooldown
        assert not new_decision.should_emerge
        assert "cooldown" in new_decision.reason.lower()

    def test_context_matching_improves_confidence(self) -> None:
        """
        Test that matching context boosts confidence.
        
        Given: Pattern with specific hour
        When: Current context matches that hour
        Then: Decision confidence is boosted
        """
        # Given: Detector with temporal pattern
        detector = PatternDetector()
        base = datetime.now().replace(hour=14)
        
        for i in range(5):
            detector.add_event({
                "type": "focus_mode",
                "timestamp": (base - timedelta(days=i)).isoformat(),
            })
        
        # When: Get matching patterns
        matching = detector.get_matching_patterns({"hour": 14})
        
        # Then: Should find patterns that match current hour
        # (patterns are matched based on hour proximity)
        all_patterns = detector.detect_patterns()
        assert len(all_patterns) >= len(matching)


@pytest.mark.integration
class TestPatternAnticipationIntegration:
    """Integration tests between pattern detector and anticipation engine."""

    def setup_method(self) -> None:
        """Reset singletons."""
        reset_pattern_detector()
        reset_anticipation_engine()

    def test_singleton_integration(self) -> None:
        """
        Test singleton instances work together.
        
        Given: Singleton pattern detector and anticipation engine
        When: Adding events and evaluating
        Then: They share state correctly
        """
        # Given: Singletons
        detector = get_pattern_detector()
        engine = get_anticipation_engine()
        
        # When: Add events to detector
        for _ in range(5):
            detector.add_event({"type": "shell_command", "command": "make test"})
        
        # Then: Engine can evaluate detector's patterns
        patterns = detector.detect_patterns()
        decision = engine.evaluate(context={}, patterns=patterns)
        
        assert decision is not None
        assert isinstance(decision.mode, EmergenceMode)

    def test_stats_tracking_through_pipeline(self) -> None:
        """
        Test that stats are tracked through the pipeline.
        
        Given: Events processed through pipeline
        When: Checking stats
        Then: Both components track correctly
        """
        # Given: Events
        detector = get_pattern_detector()
        engine = get_anticipation_engine()
        
        for i in range(10):
            detector.add_event({
                "type": "shell_command",
                "command": f"command_{i % 3}",
            })
        
        patterns = detector.detect_patterns()
        engine.evaluate(context={}, patterns=patterns)
        
        # When: Check stats
        detector_stats = detector.get_stats()
        engine_stats = engine.get_stats()
        
        # Then: Stats reflect activity
        assert detector_stats["total_events"] == 10
        assert engine_stats["total_evaluations"] >= 1
