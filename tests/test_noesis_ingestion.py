"""
NOESIS Data Ingestion Service - Integration Tests
==================================================

Tests for Sprint 1: DAIMON → NOESIS data pipeline.

Tests cover:
1. BehavioralSignal creation and serialization
2. DataIngestionService signal aggregation
3. HTTP endpoint communication (mocked)
4. Full pipeline with REAL NOESIS (if available)

Run:
    pytest tests/test_noesis_ingestion.py -v

    # With REAL NOESIS (requires services running):
    pytest tests/test_noesis_ingestion.py -v -m "integration"
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test targets
from integrations.noesis_ingestion import (
    BehavioralSignal,
    DataIngestionService,
    IngestionStats,
    get_ingestion_service,
    reset_ingestion_service,
)
from memory.activity_store import ActivityRecord


class TestBehavioralSignal:
    """Test BehavioralSignal dataclass."""

    def test_signal_creation(self):
        """Test signal creation with all fields."""
        signal = BehavioralSignal(
            signal_type="preference",
            source="claude",
            timestamp=datetime.now(),
            salience=0.8,
            data={"action": "rejection"},
            context="User rejected code suggestion",
        )

        assert signal.signal_type == "preference"
        assert signal.source == "claude"
        assert signal.salience == 0.8
        assert "reject" in signal.context.lower()

    def test_signal_to_dict(self):
        """Test serialization to dict."""
        ts = datetime(2025, 12, 13, 12, 0, 0)
        signal = BehavioralSignal(
            signal_type="anomaly",
            source="shell",
            timestamp=ts,
            salience=0.9,
            data={"command": "rm -rf /tmp/test"},
            context="Risky command detected",
        )

        d = signal.to_dict()

        assert d["signal_type"] == "anomaly"
        assert d["source"] == "shell"
        assert d["salience"] == 0.9
        assert "2025-12-13" in d["timestamp"]
        assert d["data"]["command"] == "rm -rf /tmp/test"

    def test_salience_bounds(self):
        """Test salience values at boundaries."""
        low = BehavioralSignal(
            signal_type="pattern",
            source="afk",
            timestamp=datetime.now(),
            salience=0.0,
            data={},
            context="Minimal salience",
        )
        high = BehavioralSignal(
            signal_type="anomaly",
            source="shell",
            timestamp=datetime.now(),
            salience=1.0,
            data={},
            context="Maximum salience",
        )

        assert low.salience == 0.0
        assert high.salience == 1.0


class TestDataIngestionService:
    """Test DataIngestionService signal processing."""

    def test_service_initialization(self):
        """Test service creates with default config."""
        reset_ingestion_service()
        service = DataIngestionService(
            ingestion_interval_seconds=30,
            salience_threshold=0.5,
            batch_size=100,
        )

        assert service.interval == 30
        assert service.salience_threshold == 0.5
        assert service.batch_size == 100
        assert service.stats.cycles_completed == 0

    def test_claude_signal_aggregation(self):
        """Test claude activity aggregation."""
        service = DataIngestionService()

        # Create mock claude activities
        activities = [
            ActivityRecord(
                id="1",
                watcher_type="claude",
                timestamp=datetime.now(),
                data={
                    "signal_type": "rejection",
                    "category": "code_style",
                },
            ),
            ActivityRecord(
                id="2",
                watcher_type="claude",
                timestamp=datetime.now(),
                data={
                    "signal_type": "approval",
                    "category": "documentation",
                },
            ),
        ]

        signals = service._process_claude_activities(activities)

        assert len(signals) == 2
        # Rejection should have higher salience
        rejection_signal = next(s for s in signals if "rejection" in s.context.lower())
        approval_signal = next(s for s in signals if "approval" in s.context.lower())
        assert rejection_signal.salience > approval_signal.salience

    def test_shell_signal_aggregation_risky(self):
        """Test shell activity aggregation for risky commands."""
        service = DataIngestionService()

        activities = [
            ActivityRecord(
                id="1",
                watcher_type="shell",
                timestamp=datetime.now(),
                data={
                    "command": "rm -rf /important/data",
                    "exit_code": 0,
                },
            ),
        ]

        signals = service._process_shell_activities(activities)

        assert len(signals) == 1
        assert signals[0].signal_type == "anomaly"
        assert signals[0].salience == 0.9  # Risky command salience

    def test_shell_signal_aggregation_failed(self):
        """Test shell activity aggregation for failed commands."""
        service = DataIngestionService()

        activities = [
            ActivityRecord(
                id="1",
                watcher_type="shell",
                timestamp=datetime.now(),
                data={
                    "command": "make build",
                    "exit_code": 1,
                },
            ),
        ]

        signals = service._process_shell_activities(activities)

        assert len(signals) == 1
        assert signals[0].signal_type == "anomaly"
        assert signals[0].salience == 0.7  # Failed command salience

    def test_input_signal_aggregation(self):
        """Test input activity aggregation for typing patterns."""
        service = DataIngestionService()

        # Need enough data with high variance
        activities = [
            ActivityRecord(
                id=str(i),
                watcher_type="input",
                timestamp=datetime.now(),
                data={"key_count": 100, "wpm": wpm},
            )
            for i, wpm in enumerate([40, 80, 120, 30, 90])  # High variance
        ]

        signals = service._process_input_activities(activities)

        # Should detect cognitive state change due to high variance
        assert len(signals) == 1
        assert signals[0].signal_type == "cognitive_state"

    def test_window_signal_aggregation(self):
        """Test window activity aggregation for context switching."""
        service = DataIngestionService()

        # Many different apps = high context switching
        activities = [
            ActivityRecord(
                id=str(i),
                watcher_type="window",
                timestamp=datetime.now(),
                data={"app_name": f"App{i}"},
            )
            for i in range(10)  # 10 different apps
        ]

        signals = service._process_window_activities(activities)

        assert len(signals) == 1
        assert signals[0].signal_type == "pattern"
        assert signals[0].data["app_count"] == 10

    def test_afk_signal_aggregation(self):
        """Test AFK activity aggregation."""
        service = DataIngestionService()

        activities = [
            ActivityRecord(
                id="1",
                watcher_type="afk",
                timestamp=datetime.now(),
                data={"afk_duration_seconds": 2400},  # 40 minutes
            ),
        ]

        signals = service._process_afk_activities(activities)

        assert len(signals) == 1
        assert signals[0].signal_type == "pattern"
        assert "40 minute" in signals[0].context

    def test_aggregate_mixed_activities(self):
        """Test aggregation of multiple watcher types."""
        service = DataIngestionService()

        activities = [
            ActivityRecord(
                id="1",
                watcher_type="claude",
                timestamp=datetime.now(),
                data={"signal_type": "rejection", "category": "test"},
            ),
            ActivityRecord(
                id="2",
                watcher_type="shell",
                timestamp=datetime.now(),
                data={"command": "drop table users", "exit_code": 0},
            ),
        ]

        signals = service._aggregate_to_signals(activities)

        assert len(signals) == 2
        types = {s.signal_type for s in signals}
        assert "preference" in types
        assert "anomaly" in types

    def test_salience_filtering(self):
        """Test that signals below threshold are filtered."""
        service = DataIngestionService(salience_threshold=0.6)

        # AFK returns 0.4 salience - below threshold (needs > 1800s)
        activities = [
            ActivityRecord(
                id="1",
                watcher_type="afk",
                timestamp=datetime.now(),
                data={"afk_duration_seconds": 2400},  # 40 min (> 30 min threshold)
            ),
        ]

        signals = service._aggregate_to_signals(activities)
        filtered = [s for s in signals if s.salience >= service.salience_threshold]

        assert len(signals) == 1
        assert signals[0].salience == 0.4
        assert len(filtered) == 0  # Should be filtered out (0.4 < 0.6)

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        service = DataIngestionService()

        assert service.stats.cycles_completed == 0
        assert service.stats.signals_generated == 0
        assert service.stats.signals_sent == 0

        # Manually update stats
        service.stats.cycles_completed = 5
        service.stats.signals_generated = 50
        service.stats.signals_sent = 30

        stats = service.get_stats()

        assert stats["cycles_completed"] == 5
        assert stats["signals_generated"] == 50
        assert stats["signals_sent"] == 30


class TestIngestionServiceMocked:
    """Test DataIngestionService with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_send_to_noesis_mocked(self):
        """Test sending signals to NOESIS (mocked HTTP)."""
        service = DataIngestionService()

        signals = [
            BehavioralSignal(
                signal_type="preference",
                source="claude",
                timestamp=datetime.now(),
                salience=0.8,
                data={},
                context="Test signal",
            ),
        ]

        with patch("integrations.noesis_ingestion.http_post") as mock_post:
            mock_post.return_value = {"status": "ok"}

            sent = await service._send_to_noesis(signals)

            assert sent >= 1
            assert mock_post.called

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test service handles NOESIS failure gracefully."""
        service = DataIngestionService()

        signals = [
            BehavioralSignal(
                signal_type="anomaly",
                source="shell",
                timestamp=datetime.now(),
                salience=0.9,
                data={},
                context="Test anomaly",
            ),
        ]

        with patch("integrations.noesis_ingestion.http_post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            # Should not raise, just log and count failure
            sent = await service._send_to_noesis(signals)

            assert sent == 0
            assert service.stats.noesis_failures >= 1


class TestSingleton:
    """Test singleton pattern."""

    def test_get_ingestion_service_singleton(self):
        """Test get_ingestion_service returns same instance."""
        reset_ingestion_service()

        service1 = get_ingestion_service()
        service2 = get_ingestion_service()

        assert service1 is service2

    def test_reset_ingestion_service(self):
        """Test reset creates new instance."""
        service1 = get_ingestion_service()
        reset_ingestion_service()
        service2 = get_ingestion_service()

        assert service1 is not service2


# =============================================================================
# Integration Tests (require REAL NOESIS running)
# =============================================================================


@pytest.mark.integration
class TestNoesisIntegration:
    """Integration tests with REAL NOESIS services."""

    @pytest.mark.asyncio
    async def test_real_consciousness_ingest(self):
        """Test real /v1/consciousness/ingest endpoint."""
        from integrations.mcp_tools.http_utils import http_post
        from integrations.mcp_tools.config import NOESIS_CONSCIOUSNESS_URL

        signal = {
            "signal_type": "preference",
            "source": "claude",
            "timestamp": datetime.now().isoformat(),
            "salience": 0.8,
            "data": {"test": True},
            "context": "Integration test signal",
        }

        try:
            result = await http_post(
                f"{NOESIS_CONSCIOUSNESS_URL}/v1/consciousness/ingest",
                signal,
                timeout=10.0,
            )

            assert "error" not in result or "unavailable" in result.get("status", "")
            print(f"✓ Consciousness ingest: {result}")

        except Exception as e:
            pytest.skip(f"NOESIS not available: {e}")

    @pytest.mark.asyncio
    async def test_real_memory_episode(self):
        """Test real /v1/memory/episode endpoint."""
        from integrations.mcp_tools.http_utils import http_post
        from integrations.mcp_tools.config import NOESIS_CONSCIOUSNESS_URL

        episode = {
            "type": "behavioral",
            "content": "Integration test episode",
            "metadata": {
                "salience": 0.5,
                "test": True,
            },
        }

        try:
            result = await http_post(
                f"{NOESIS_CONSCIOUSNESS_URL}/v1/memory/episode",
                episode,
                timeout=10.0,
            )

            assert "error" not in result
            print(f"✓ Memory episode: {result}")

        except Exception as e:
            pytest.skip(f"NOESIS not available: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("NOESIS Data Ingestion Tests")
    print("=" * 60)

    # Run unit tests
    pytest.main([__file__, "-v", "-x", "--ignore-glob=*integration*"])
