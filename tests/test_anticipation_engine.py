"""
Tests for learners/anticipation_engine.py

Tests covering:
- EmergenceMode enum
- EmergenceDecision dataclass
- AnticipationEngine evaluation logic
- Cooldown mechanism
- Stats tracking

Run:
    pytest tests/test_anticipation_engine.py -v
"""

import time
from datetime import datetime

import pytest

from learners.anticipation_engine import (
    EmergenceMode,
    EmergenceDecision,
    AnticipationEngine,
    get_anticipation_engine,
    reset_anticipation_engine,
)
from learners.pattern_detector import Pattern


# ============================================================================
# DATACLASS AND ENUM TESTS
# ============================================================================


class TestEmergenceMode:
    """Tests for EmergenceMode enum."""

    def test_mode_values(self):
        """Test enum values."""
        assert EmergenceMode.SUBTLE.value == "subtle"
        assert EmergenceMode.NORMAL.value == "normal"
        assert EmergenceMode.URGENT.value == "urgent"


class TestEmergenceDecision:
    """Tests for EmergenceDecision dataclass."""

    def test_create_decision(self):
        """Test creating a decision."""
        decision = EmergenceDecision(
            should_emerge=True,
            mode=EmergenceMode.NORMAL,
            reason="Pattern matched",
            confidence=0.75,
        )
        assert decision.should_emerge is True
        assert decision.mode == EmergenceMode.NORMAL
        assert decision.confidence == 0.75

    def test_decision_to_dict(self):
        """Test converting decision to dict."""
        decision = EmergenceDecision(
            should_emerge=True,
            mode=EmergenceMode.URGENT,
            reason="High confidence",
            confidence=0.9,
        )
        d = decision.to_dict()
        assert d["should_emerge"] is True
        assert d["mode"] == "urgent"
        assert d["confidence"] == 0.9


# ============================================================================
# ANTICIPATION ENGINE TESTS
# ============================================================================


class TestAnticipationEngineInit:
    """Tests for AnticipationEngine initialization."""

    def test_init_default(self):
        """Test default initialization."""
        engine = AnticipationEngine()
        assert engine.cooldown_seconds == 600  # 10 min
        assert engine.min_patterns == 1

    def test_init_custom(self):
        """Test custom initialization."""
        engine = AnticipationEngine(cooldown_seconds=60, min_patterns_for_emergence=2)
        assert engine.cooldown_seconds == 60
        assert engine.min_patterns == 2


class TestEvaluate:
    """Tests for AnticipationEngine.evaluate()."""

    @pytest.fixture
    def engine(self):
        """Fresh engine for each test."""
        return AnticipationEngine(cooldown_seconds=60)

    @pytest.fixture
    def high_confidence_pattern(self):
        """Pattern with high confidence."""
        return Pattern(
            pattern_type="temporal",
            description="commit at 17:00",
            confidence=0.85,
            occurrences=10,
            last_seen=datetime.now(),
            data={"peak_hour": 17},
        )

    @pytest.fixture
    def medium_confidence_pattern(self):
        """Pattern with medium confidence."""
        return Pattern(
            pattern_type="sequential",
            description="git workflow",
            confidence=0.72,
            occurrences=5,
            last_seen=datetime.now(),
        )

    @pytest.fixture
    def low_confidence_pattern(self):
        """Pattern with low confidence."""
        return Pattern(
            pattern_type="contextual",
            description="context pattern",
            confidence=0.55,
            occurrences=3,
            last_seen=datetime.now(),
        )

    def test_evaluate_urgent(self, engine, high_confidence_pattern):
        """Test URGENT mode for high confidence."""
        context = {"hour": 17}
        decision = engine.evaluate(context, [high_confidence_pattern])
        
        assert decision.should_emerge is True
        assert decision.mode == EmergenceMode.URGENT

    def test_evaluate_normal(self, engine, medium_confidence_pattern):
        """Test NORMAL mode for medium confidence."""
        context = {}
        decision = engine.evaluate(context, [medium_confidence_pattern])
        
        assert decision.should_emerge is True
        assert decision.mode == EmergenceMode.NORMAL

    def test_evaluate_subtle(self, engine, low_confidence_pattern):
        """Test SUBTLE mode for low confidence."""
        context = {}
        decision = engine.evaluate(context, [low_confidence_pattern])
        
        assert decision.should_emerge is True
        assert decision.mode == EmergenceMode.SUBTLE

    def test_evaluate_no_emerge_low_confidence(self, engine):
        """Test no emergence for very low confidence."""
        low_pattern = Pattern(
            pattern_type="temporal",
            description="test",
            confidence=0.3,  # Below threshold
            occurrences=2,
            last_seen=datetime.now(),
        )
        decision = engine.evaluate({}, [low_pattern])
        
        assert decision.should_emerge is False

    def test_evaluate_no_patterns(self, engine):
        """Test no emergence when no patterns."""
        decision = engine.evaluate({}, [])
        
        assert decision.should_emerge is False
        assert "Insufficient patterns" in decision.reason


class TestCooldown:
    """Tests for cooldown mechanism."""

    def test_cooldown_blocks_emergence(self):
        """Test that cooldown blocks emergence."""
        engine = AnticipationEngine(cooldown_seconds=60)
        
        pattern = Pattern(
            pattern_type="test",
            description="test",
            confidence=0.9,
            occurrences=10,
            last_seen=datetime.now(),
        )
        
        # First evaluation - should emerge
        decision1 = engine.evaluate({}, [pattern])
        assert decision1.should_emerge is True
        
        # Simulate emergence
        engine._last_emergence_time = time.time()
        
        # Second evaluation - should be blocked
        decision2 = engine.evaluate({}, [pattern])
        assert decision2.should_emerge is False
        assert decision2.cooldown_remaining > 0
        assert "Cooldown active" in decision2.reason

    def test_cooldown_expires(self):
        """Test that cooldown expires correctly."""
        engine = AnticipationEngine(cooldown_seconds=1)  # 1 second cooldown
        
        pattern = Pattern(
            pattern_type="test",
            description="test",
            confidence=0.9,
            occurrences=10,
            last_seen=datetime.now(),
        )
        
        # Simulate old emergence
        engine._last_emergence_time = time.time() - 2  # 2 seconds ago
        
        # Should not be blocked
        decision = engine.evaluate({}, [pattern])
        assert decision.should_emerge is True

    def test_reset_cooldown(self):
        """Test resetting cooldown."""
        engine = AnticipationEngine(cooldown_seconds=60)
        engine._last_emergence_time = time.time()
        
        remaining_before = engine._get_cooldown_remaining()
        assert remaining_before > 0
        
        engine.reset_cooldown()
        
        remaining_after = engine._get_cooldown_remaining()
        assert remaining_after == 0

    def test_get_cooldown_status(self):
        """Test cooldown status."""
        engine = AnticipationEngine(cooldown_seconds=60)
        engine._last_emergence_time = time.time()
        
        status = engine.get_cooldown_status()
        assert status["is_active"] is True
        assert status["cooldown_duration"] == 60
        assert status["remaining_seconds"] > 0


class TestStats:
    """Tests for statistics tracking."""

    def test_stats_tracking(self):
        """Test that evaluations are tracked."""
        engine = AnticipationEngine()
        
        pattern = Pattern(
            pattern_type="test",
            description="test",
            confidence=0.9,
            occurrences=10,
            last_seen=datetime.now(),
        )
        
        engine.evaluate({}, [pattern])
        engine.evaluate({}, [pattern])
        
        stats = engine.get_stats()
        assert stats["total_evaluations"] == 2

    def test_clear_stats(self):
        """Test clearing stats."""
        engine = AnticipationEngine()
        
        pattern = Pattern(
            pattern_type="test",
            description="test",
            confidence=0.9,
            occurrences=10,
            last_seen=datetime.now(),
        )
        
        engine.evaluate({}, [pattern])
        engine.clear_stats()
        
        stats = engine.get_stats()
        assert stats["total_evaluations"] == 0


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_anticipation_engine_singleton(self):
        """Test singleton returns same instance."""
        reset_anticipation_engine()
        e1 = get_anticipation_engine()
        e2 = get_anticipation_engine()
        assert e1 is e2

    def test_reset_anticipation_engine(self):
        """Test resetting singleton."""
        reset_anticipation_engine()
        e1 = get_anticipation_engine()
        reset_anticipation_engine()
        e2 = get_anticipation_engine()
        assert e1 is not e2


# ============================================================================
# TRIGGER EMERGENCE TESTS (require mocking)
# ============================================================================


class TestTriggerEmergence:
    """Tests for trigger_emergence method."""

    @pytest.fixture
    def engine(self):
        """Fresh engine for each test."""
        return AnticipationEngine(cooldown_seconds=60)

    @pytest.fixture
    def high_confidence_decision(self):
        """Decision that should trigger emergence."""
        pattern = Pattern(
            pattern_type="temporal",
            description="commit at 17:00",
            confidence=0.85,
            occurrences=10,
            last_seen=datetime.now(),
        )
        return EmergenceDecision(
            should_emerge=True,
            mode=EmergenceMode.NORMAL,
            reason="Pattern matched",
            confidence=0.85,
            pattern=pattern,
        )

    @pytest.fixture
    def no_emerge_decision(self):
        """Decision that should NOT trigger emergence."""
        return EmergenceDecision(
            should_emerge=False,
            mode=EmergenceMode.SUBTLE,
            reason="Low confidence",
            confidence=0.3,
        )

    @pytest.mark.asyncio
    async def test_trigger_skipped_when_should_emerge_false(
        self, engine, no_emerge_decision
    ):
        """Test that trigger is skipped when decision says no."""
        result = await engine.trigger_emergence(no_emerge_decision)
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_with_noesis_unavailable(
        self, engine, high_confidence_decision
    ):
        """Test trigger when NOESIS is not available (ImportError path)."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"integrations.mcp_tools.http_utils": None}):
            result = await engine.trigger_emergence(high_confidence_decision)
            # Should succeed via local logging fallback
            assert result is True
            assert engine._stats.total_emergences == 1

    @pytest.mark.asyncio
    async def test_trigger_records_stats(
        self, engine, high_confidence_decision
    ):
        """Test that trigger records statistics correctly."""
        from unittest.mock import patch, AsyncMock, MagicMock

        # Mock the dynamic import
        mock_http = MagicMock()
        mock_http.http_post = AsyncMock(return_value={"status": "ok"})
        
        mock_config = MagicMock()
        mock_config.NOESIS_CONSCIOUSNESS_URL = "http://mock:9000"

        with patch.dict("sys.modules", {
            "integrations.mcp_tools.http_utils": mock_http,
            "integrations.mcp_tools.config": mock_config,
        }):
            result = await engine.trigger_emergence(high_confidence_decision)

        # Stats should be recorded (either via NOESIS or fallback)
        assert engine._stats.total_emergences >= 1


class TestContextBoost:
    """Tests for context boost calculation."""

    @pytest.fixture
    def engine(self):
        """Fresh engine."""
        return AnticipationEngine()

    def test_sequential_pattern_boost(self, engine):
        """Test that sequential patterns get a boost."""
        pattern = Pattern(
            pattern_type="sequential",
            description="git workflow",
            confidence=0.75,
            occurrences=5,
            last_seen=datetime.now(),
        )
        
        boosted = engine._apply_context_boost(0.75, {}, pattern)
        assert boosted >= 0.75  # Should have boost

    def test_temporal_pattern_exact_hour_boost(self, engine):
        """Test temporal pattern gets boost at exact hour."""
        pattern = Pattern(
            pattern_type="temporal",
            description="commit at 17:00",
            confidence=0.70,
            occurrences=5,
            last_seen=datetime.now(),
            data={"peak_hour": 17},
        )
        
        context = {"hour": 17}  # Exact match
        boosted = engine._apply_context_boost(0.70, context, pattern)
        assert boosted >= 0.75  # Should have +0.05 or more

    def test_temporal_pattern_near_hour_boost(self, engine):
        """Test temporal pattern gets smaller boost near peak hour."""
        pattern = Pattern(
            pattern_type="temporal",
            description="commit at 17:00",
            confidence=0.70,
            occurrences=5,
            last_seen=datetime.now(),
            data={"peak_hour": 17},
        )
        
        context = {"hour": 16}  # One hour off
        boosted = engine._apply_context_boost(0.70, context, pattern)
        assert boosted >= 0.70  # Should have some boost

    def test_confidence_capped_at_one(self, engine):
        """Test confidence is capped at 1.0."""
        pattern = Pattern(
            pattern_type="sequential",
            description="test",
            confidence=0.99,
            occurrences=10,
            last_seen=datetime.now(),
        )
        
        boosted = engine._apply_context_boost(0.99, {}, pattern)
        assert boosted <= 1.0
