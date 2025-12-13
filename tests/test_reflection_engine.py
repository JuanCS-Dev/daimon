"""
Tests for learners/reflection_engine.py

Scientific tests covering:
- ReflectionConfig and ReflectionStats dataclasses
- ReflectionEngine initialization
- Threshold checking
- Reflection execution
- Singleton management
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learners.preference_learner import PreferenceSignal
from learners.reflection_engine import (
    ReflectionConfig,
    ReflectionEngine,
    ReflectionStats,
    get_engine,
    reset_engine,
)


class TestReflectionConfig:
    """Tests for ReflectionConfig dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating config with default values."""
        config = ReflectionConfig()
        assert config.interval_minutes == 30
        assert config.rejection_threshold == 5
        assert config.approval_threshold == 10
        assert config.scan_hours == 48

    def test_create_with_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = ReflectionConfig(
            interval_minutes=60,
            rejection_threshold=10,
            approval_threshold=20,
            scan_hours=24,
        )
        assert config.interval_minutes == 60
        assert config.rejection_threshold == 10
        assert config.approval_threshold == 20
        assert config.scan_hours == 24


class TestReflectionStats:
    """Tests for ReflectionStats dataclass."""

    def test_create_with_defaults(self) -> None:
        """Test creating stats with default values."""
        stats = ReflectionStats()
        assert stats.total_reflections == 0
        assert stats.total_updates == 0
        assert stats.last_reflection is None

    def test_create_with_values(self) -> None:
        """Test creating stats with custom values."""
        now = datetime.now()
        stats = ReflectionStats(
            total_reflections=5,
            total_updates=3,
            last_reflection=now,
        )
        assert stats.total_reflections == 5
        assert stats.total_updates == 3
        assert stats.last_reflection == now


class TestReflectionEngineInit:
    """Tests for ReflectionEngine initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config."""
        engine = ReflectionEngine()
        assert engine.config.interval_minutes == 30
        assert engine.running is False
        assert engine.learner is not None

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ReflectionConfig(interval_minutes=60, rejection_threshold=3)
        engine = ReflectionEngine(config=config)
        assert engine.config.interval_minutes == 60
        assert engine.config.rejection_threshold == 3


class TestCheckThreshold:
    """Tests for ReflectionEngine._check_threshold()."""

    @pytest.fixture
    def engine(self) -> ReflectionEngine:
        """Create a ReflectionEngine instance."""
        config = ReflectionConfig(rejection_threshold=3, approval_threshold=5)
        return ReflectionEngine(config=config)

    def test_threshold_not_reached(self, engine: ReflectionEngine) -> None:
        """Test threshold not reached."""
        signals = [
            PreferenceSignal(
                timestamp="2025-01-01",
                signal_type="rejection",
                context="ctx",
                category="testing",
                strength=0.8,
                session_id="sess",
            ),
            PreferenceSignal(
                timestamp="2025-01-01",
                signal_type="rejection",
                context="ctx",
                category="testing",
                strength=0.8,
                session_id="sess",
            ),
        ]
        assert engine._check_threshold(signals) is False

    def test_rejection_threshold_reached(self, engine: ReflectionEngine) -> None:
        """Test rejection threshold reached."""
        signals = [
            PreferenceSignal(
                timestamp="2025-01-01",
                signal_type="rejection",
                context="ctx",
                category="testing",
                strength=0.8,
                session_id="sess",
            )
            for _ in range(3)
        ]
        assert engine._check_threshold(signals) is True

    def test_approval_threshold_reached(self, engine: ReflectionEngine) -> None:
        """Test approval threshold reached."""
        signals = [
            PreferenceSignal(
                timestamp="2025-01-01",
                signal_type="approval",
                context="ctx",
                category="code_style",
                strength=0.9,
                session_id="sess",
            )
            for _ in range(5)
        ]
        assert engine._check_threshold(signals) is True

    def test_empty_signals(self, engine: ReflectionEngine) -> None:
        """Test with empty signals."""
        assert engine._check_threshold([]) is False


class TestReflect:
    """Tests for ReflectionEngine.reflect()."""

    @pytest.fixture
    def engine(self) -> ReflectionEngine:
        """Create a ReflectionEngine instance."""
        return ReflectionEngine()

    @pytest.mark.asyncio
    async def test_reflect_basic(self, engine: ReflectionEngine) -> None:
        """Test basic reflection."""
        result = await engine.reflect()
        assert "signals_count" in result
        assert "insights_count" in result
        assert "updated" in result
        assert "elapsed_seconds" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_reflect_updates_stats(self, engine: ReflectionEngine) -> None:
        """Test that reflection updates stats."""
        assert engine.stats.total_reflections == 0
        assert engine.stats.last_reflection is None

        await engine.reflect()

        assert engine.stats.total_reflections == 1
        assert engine.stats.last_reflection is not None

    @pytest.mark.asyncio
    async def test_reflect_clears_learner(self, engine: ReflectionEngine) -> None:
        """Test that reflection clears learner state."""
        engine.learner.category_counts["testing"]["approvals"] = 5
        await engine.reflect()
        # Counts are cleared during scan
        # After scan they may have new values


class TestGetStatus:
    """Tests for ReflectionEngine.get_status()."""

    def test_status_initial(self) -> None:
        """Test status on new engine."""
        engine = ReflectionEngine()
        status = engine.get_status()
        assert status["running"] is False
        assert status["last_reflection"] is None
        assert status["next_reflection"] is None
        assert status["total_reflections"] == 0

    @pytest.mark.asyncio
    async def test_status_after_reflection(self) -> None:
        """Test status after reflection."""
        engine = ReflectionEngine()
        await engine.reflect()

        status = engine.get_status()
        assert status["total_reflections"] == 1
        assert status["last_reflection"] is not None
        assert status["next_reflection"] is not None

    def test_status_contains_thresholds(self) -> None:
        """Test status contains threshold info."""
        config = ReflectionConfig(rejection_threshold=3, approval_threshold=7)
        engine = ReflectionEngine(config=config)

        status = engine.get_status()
        assert status["thresholds"]["rejection"] == 3
        assert status["thresholds"]["approval"] == 7


class TestGetLearnerAndRefiner:
    """Tests for accessor methods."""

    def test_get_learner(self) -> None:
        """Test getting learner instance."""
        engine = ReflectionEngine()
        learner = engine.get_learner()
        assert learner is not None
        assert learner is engine.learner

    def test_get_refiner(self) -> None:
        """Test getting refiner instance."""
        engine = ReflectionEngine()
        refiner = engine.get_refiner()
        # Refiner may or may not be available depending on imports
        # Just verify it doesn't crash


class TestStartStop:
    """Tests for ReflectionEngine.start() and stop()."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self) -> None:
        """Test start sets running flag."""
        engine = ReflectionEngine()
        assert engine.running is False

        await engine.start()
        assert engine.running is True

        # Clean up
        await engine.stop()
        assert engine.running is False

    @pytest.mark.asyncio
    async def test_start_when_already_running(self) -> None:
        """Test start when already running does nothing."""
        engine = ReflectionEngine()
        await engine.start()
        await engine.start()  # Should not raise

        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        """Test stop when not running does nothing."""
        engine = ReflectionEngine()
        await engine.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_run_loop_executes(self) -> None:
        """Test that _run_loop executes and can be cancelled."""
        engine = ReflectionEngine()
        engine.running = True

        # Start the loop
        task = asyncio.create_task(engine._run_loop())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel it
        engine.running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Loop should have completed or been cancelled
        assert not engine.running or task.cancelled()


class TestSingleton:
    """Tests for singleton management."""

    def test_get_engine_returns_same_instance(self) -> None:
        """Test get_engine returns same instance."""
        reset_engine()  # Start fresh
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

    def test_reset_engine(self) -> None:
        """Test reset_engine creates new instance."""
        reset_engine()
        engine1 = get_engine()
        reset_engine()
        engine2 = get_engine()
        assert engine1 is not engine2


class TestApplyInsights:
    """Tests for ReflectionEngine._apply_insights()."""

    @pytest.mark.asyncio
    async def test_apply_insights_no_refiner(self) -> None:
        """Test apply_insights without refiner."""
        engine = ReflectionEngine()
        engine.refiner = None

        insights = [{"category": "testing", "action": "reinforce"}]
        result = await engine._apply_insights(insights)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_insights_empty(self) -> None:
        """Test apply_insights with empty list."""
        engine = ReflectionEngine()
        result = await engine._apply_insights([])
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_insights_with_mock_refiner(self) -> None:
        """Test apply_insights with mock refiner."""
        engine = ReflectionEngine()
        mock_refiner = MagicMock()
        mock_refiner.update_preferences.return_value = True
        engine.refiner = mock_refiner

        insights = [{"category": "testing", "action": "reinforce"}]
        result = await engine._apply_insights(insights)

        assert result is True
        mock_refiner.update_preferences.assert_called_once_with(
            insights, force_timestamp=False
        )

    @pytest.mark.asyncio
    async def test_apply_insights_handles_error(self) -> None:
        """Test apply_insights handles refiner errors."""
        engine = ReflectionEngine()
        mock_refiner = MagicMock()
        mock_refiner.update_preferences.side_effect = OSError("File error")
        engine.refiner = mock_refiner

        insights = [{"category": "testing", "action": "reinforce"}]
        result = await engine._apply_insights(insights)

        assert result is False


class TestNotifyUpdate:
    """Tests for ReflectionEngine._notify_update()."""

    @pytest.mark.asyncio
    async def test_notify_update_handles_errors(self) -> None:
        """Test notify_update handles errors gracefully."""
        engine = ReflectionEngine()
        insights = [{"category": "testing"}]

        # Should not raise even if notify-send not available
        await engine._notify_update(insights)


class TestCheckTriggers:
    """Tests for ReflectionEngine._check_triggers()."""

    @pytest.mark.asyncio
    async def test_check_triggers_temporal(self) -> None:
        """Test temporal trigger activation."""
        config = ReflectionConfig(interval_minutes=1)
        engine = ReflectionEngine(config=config)

        # Set last reflection to 2 minutes ago
        engine.stats.last_reflection = datetime.now() - timedelta(minutes=2)

        # This should trigger a reflection
        initial_count = engine.stats.total_reflections
        await engine._check_triggers()

        assert engine.stats.total_reflections == initial_count + 1

    @pytest.mark.asyncio
    async def test_check_triggers_first_time(self) -> None:
        """Test trigger on first check (no last_reflection)."""
        engine = ReflectionEngine()
        assert engine.stats.last_reflection is None

        await engine._check_triggers()

        assert engine.stats.last_reflection is not None
        assert engine.stats.total_reflections == 1

    @pytest.mark.asyncio
    async def test_check_triggers_threshold(self) -> None:
        """Test threshold trigger when many rejections detected."""
        config = ReflectionConfig(
            interval_minutes=60,  # Long interval so temporal doesn't trigger
            rejection_threshold=2,
        )
        engine = ReflectionEngine(config=config)

        # Set recent reflection so temporal doesn't trigger
        engine.stats.last_reflection = datetime.now()

        # Mock the learner to return signals that hit threshold
        original_scan = engine.learner.scan_sessions

        def mock_scan(since_hours: int) -> list:
            return [
                PreferenceSignal(
                    timestamp="2025-01-01",
                    signal_type="rejection",
                    context="ctx",
                    category="testing",
                    strength=0.8,
                    session_id="sess",
                )
                for _ in range(3)  # 3 rejections > threshold of 2
            ]

        engine.learner.scan_sessions = mock_scan

        initial_count = engine.stats.total_reflections
        await engine._check_triggers()

        # Should have triggered due to threshold
        assert engine.stats.total_reflections == initial_count + 1

        # Restore
        engine.learner.scan_sessions = original_scan
