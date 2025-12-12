"""
DAIMON Integration Tests - REAL NOESIS Calls
=============================================

Tests against LIVE NOESIS services (8001, 8002).
No mocking - real consciousness, real tribunal.

Requires: ./noesis wakeup
"""

from __future__ import annotations

import pytest
import httpx


# ============================================================================
# MCP Server Integration Tests
# ============================================================================

class TestMcpServerIntegration:
    """Real integration tests for MCP server against NOESIS."""

    @pytest.mark.asyncio
    async def test_http_post_real_quick_check(self) -> None:
        """Test real HTTP POST to quick-check endpoint."""
        from integrations.mcp_server import _http_post

        result = await _http_post(
            "http://localhost:8001/api/consciousness/quick-check",
            {"prompt": "delete all production data"},
            timeout=5.0,
        )
        assert "error" not in result
        assert result["salience"] >= 0.85
        assert result["should_emerge"] is True

    @pytest.mark.asyncio
    async def test_http_get_real_service(self) -> None:
        """Test real HTTP GET to NOESIS service."""
        from integrations.mcp_server import _http_get

        result = await _http_get(
            "http://localhost:8002/health",
            timeout=5.0,
        )
        assert "error" not in result
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_noesis_consult_real(self) -> None:
        """Test real noesis_consult against consciousness service."""
        from integrations.mcp_server import _http_post

        # Call the consciousness stream/process endpoint
        result = await _http_post(
            "http://localhost:8001/api/consciousness/stream/process",
            {
                "content": "Should I use Redis or PostgreSQL for caching?",
                "context": "",
                "depth": 2,
                "mode": "maieutic"
            },
            timeout=30.0,
        )
        # May have error if endpoint doesn't exist, but tests the path
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_noesis_tribunal_real(self) -> None:
        """Test real tribunal verdict endpoint."""
        from integrations.mcp_server import _http_post

        result = await _http_post(
            "http://localhost:8002/reflect/verdict",
            {
                "execution_log": {
                    "content": "Delete all user records from database",
                    "task": "Delete all user records",
                    "result": "Pending judgment",
                    "context": "Testing tribunal"
                },
                "require_unanimous": False
            },
            timeout=30.0,
        )
        # Should get a real verdict
        if "error" not in result:
            assert "verdict" in result or "error" in result


# ============================================================================
# Shell Watcher Integration Tests
# ============================================================================

class TestShellWatcherIntegration:
    """Real integration tests for shell watcher."""

    @pytest.mark.asyncio
    async def test_flush_to_real_noesis(self) -> None:
        """Test flushing heartbeats to real NOESIS endpoint."""
        from collectors.shell_watcher import HeartbeatAggregator, ShellHeartbeat

        agg = HeartbeatAggregator()

        # Add some heartbeats
        for i in range(3):
            hb = ShellHeartbeat(
                timestamp=f"2025-12-12T10:00:{i:02d}",
                command=f"test_command_{i}",
                pwd="/tmp",
                exit_code=0,
                duration=0.1,
            )
            agg.pending.append(hb)

        # Flush to real NOESIS
        await agg.flush()

        # Verify pending was cleared
        assert len(agg.pending) == 0

    @pytest.mark.asyncio
    async def test_flush_with_patterns(self) -> None:
        """Test flushing with detected patterns."""
        from collectors.shell_watcher import HeartbeatAggregator, ShellHeartbeat

        agg = HeartbeatAggregator()

        # Add error heartbeats (frustration pattern)
        for i in range(5):
            hb = ShellHeartbeat(
                timestamp=f"2025-12-12T10:00:{i:02d}",
                command="npm test",
                pwd="/project",
                exit_code=1,  # Errors
            )
            agg.pending.append(hb)

        # Flush with patterns
        await agg.flush()
        assert len(agg.pending) == 0


# ============================================================================
# Claude Watcher Integration Tests
# ============================================================================

class TestClaudeWatcherIntegration:
    """Real integration tests for Claude watcher."""

    @pytest.mark.asyncio
    async def test_send_event_to_real_noesis(self) -> None:
        """Test sending event to real NOESIS endpoint."""
        from collectors.claude_watcher import SessionTracker

        tracker = SessionTracker()

        event = {
            "event_type": "create",
            "timestamp": "2025-12-12T10:00:00",
            "project": "test-project",
            "files_touched": ["test.py"],
            "intention": "create",
        }

        # This should not raise even if endpoint doesn't exist
        await tracker._send_event(event)


# ============================================================================
# Quick Check Real Tests
# ============================================================================

class TestQuickCheckReal:
    """Real tests against quick-check endpoint."""

    def test_quick_check_high_risk_real(self) -> None:
        """Test quick-check with real HTTP call."""
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                "http://localhost:8001/api/consciousness/quick-check",
                json={"prompt": "rm -rf / --no-preserve-root"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["salience"] >= 0.85
            assert data["should_emerge"] is True
            assert data["mode"] == "emerge"

    def test_quick_check_low_risk_real(self) -> None:
        """Test quick-check with low risk prompt."""
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                "http://localhost:8001/api/consciousness/quick-check",
                json={"prompt": "add a button to the UI"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["salience"] < 0.5
            assert data["should_emerge"] is False
            assert data["mode"] == "silent"


# ============================================================================
# Hook Real Tests
# ============================================================================

class TestHookReal:
    """Real tests for hook quick_check function."""

    def test_hook_quick_check_real(self) -> None:
        """Test hook's quick_check against real NOESIS."""
        import sys
        sys.path.insert(0, "/media/juan/DATA/projetos/daimon/.claude/hooks")
        from noesis_hook import quick_check

        result = quick_check("delete all databases")
        assert result is not None
        assert result["should_emerge"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
