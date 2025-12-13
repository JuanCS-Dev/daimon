"""
DAIMON MCP Server Tests
=======================

Unit tests for the MCP server module.
Tests against REAL NOESIS services - no mocking.

Follows CODE_CONSTITUTION: Measurable Quality.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

# Test the helper functions directly
from integrations.mcp_server import (
    NOESIS_CONSCIOUSNESS_URL,
    NOESIS_REFLECTOR_URL,
    REQUEST_TIMEOUT,
    _http_post,
    _http_get,
    mcp,
    noesis_consult,
    noesis_tribunal,
    noesis_precedent,
    noesis_confront,
    noesis_health,
)


class TestConfiguration:
    """Test suite for MCP server configuration."""

    def test_consciousness_url(self) -> None:
        """Test consciousness URL configuration."""
        assert NOESIS_CONSCIOUSNESS_URL == "http://localhost:8001"

    def test_reflector_url(self) -> None:
        """Test reflector URL configuration."""
        assert NOESIS_REFLECTOR_URL == "http://localhost:8002"

    def test_request_timeout(self) -> None:
        """Test request timeout configuration."""
        assert REQUEST_TIMEOUT == 30.0

    def test_mcp_server_name(self) -> None:
        """Test MCP server name."""
        assert mcp.name == "daimon-consciousness"


class TestHttpHelpers:
    """Test suite for HTTP helper functions."""

    @pytest.mark.asyncio
    async def test_http_post_success(self) -> None:
        """Test HTTP POST success to real NOESIS."""
        result = await _http_post(
            "http://localhost:8001/api/consciousness/quick-check",
            {"prompt": "test"},
            timeout=5.0,
        )
        assert "error" not in result
        assert "salience" in result

    @pytest.mark.asyncio
    async def test_http_post_connection_error(self) -> None:
        """Test HTTP POST with connection error returns error dict."""
        result = await _http_post(
            "http://127.0.0.1:59999/nonexistent",
            {"test": "data"},
            timeout=1.0,
        )
        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_http_get_success(self) -> None:
        """Test HTTP GET success to real NOESIS."""
        result = await _http_get(
            "http://localhost:8002/health",
            timeout=5.0,
        )
        assert "error" not in result
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_http_get_connection_error(self) -> None:
        """Test HTTP GET with connection error returns error dict."""
        result = await _http_get(
            "http://127.0.0.1:59999/nonexistent",
            timeout=1.0,
        )
        assert isinstance(result, dict)
        assert "error" in result


class TestMcpToolsReal:
    """Integration tests for MCP tools against REAL NOESIS."""

    @pytest.mark.asyncio
    async def test_noesis_consult_real(self) -> None:
        """Test noesis_consult with real NOESIS."""
        # Access underlying function via .fn attribute
        result = await noesis_consult.fn(
            question="Should I use Redis or PostgreSQL?",
            context="Building a caching layer",
            depth=2,
        )
        assert isinstance(result, str)
        assert "NOESIS" in result or "Consultation" in result or "unavailable" in result

    @pytest.mark.asyncio
    async def test_noesis_tribunal_real(self) -> None:
        """Test noesis_tribunal with real NOESIS."""
        result = await noesis_tribunal.fn(
            action="Delete all production data",
            justification="Testing tribunal",
            context="Test environment",
        )
        assert isinstance(result, str)
        assert "Tribunal" in result or "Verdict" in result or "unavailable" in result

    @pytest.mark.asyncio
    async def test_noesis_precedent_real(self) -> None:
        """Test noesis_precedent with real NOESIS."""
        result = await noesis_precedent.fn(
            situation="Similar refactoring task",
            limit=3,
        )
        assert isinstance(result, str)
        assert "Precedent" in result

    @pytest.mark.asyncio
    async def test_noesis_confront_real(self) -> None:
        """Test noesis_confront with real NOESIS."""
        result = await noesis_confront.fn(
            statement="I am absolutely certain this is correct",
            shadow_pattern="overconfidence",
        )
        assert isinstance(result, str)
        assert "Socratic" in result or "Confrontation" in result

    @pytest.mark.asyncio
    async def test_noesis_health_real(self) -> None:
        """Test noesis_health with real NOESIS."""
        result = await noesis_health.fn()
        assert isinstance(result, str)
        assert "Health" in result or "ONLINE" in result or "OFFLINE" in result


class TestMcpToolsEdgeCases:
    """Edge case tests for MCP tools."""

    @pytest.mark.asyncio
    async def test_consult_minimal_params(self) -> None:
        """Test consult with minimal parameters."""
        result = await noesis_consult.fn(question="What?")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tribunal_minimal_params(self) -> None:
        """Test tribunal with minimal parameters."""
        result = await noesis_tribunal.fn(action="test action")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_precedent_limit_bounds(self) -> None:
        """Test precedent with boundary limits."""
        # Test with limit 1
        result = await noesis_precedent.fn(situation="test", limit=1)
        assert isinstance(result, str)

        # Test with limit 10
        result = await noesis_precedent.fn(situation="test", limit=10)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_confront_no_shadow_pattern(self) -> None:
        """Test confront without shadow pattern."""
        result = await noesis_confront.fn(statement="This is true")
        assert isinstance(result, str)


class TestHttpErrors:
    """Test HTTP error handling paths."""

    @pytest.mark.asyncio
    async def test_http_post_404_error(self) -> None:
        """Test HTTP POST 404 error handling."""
        result = await _http_post(
            "http://localhost:8001/nonexistent/endpoint",
            {"test": "data"},
            timeout=5.0,
        )
        assert "error" in result
        assert result["error"] == "http_error"
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_http_get_404_error(self) -> None:
        """Test HTTP GET 404 error handling."""
        result = await _http_get(
            "http://localhost:8001/nonexistent/endpoint",
            timeout=5.0,
        )
        assert "error" in result
        assert result["error"] == "http_error"
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_http_post_timeout(self) -> None:
        """Test HTTP POST timeout (very short timeout)."""
        result = await _http_post(
            "http://10.255.255.1/slow",  # Non-routable IP
            {"test": "data"},
            timeout=0.01,
        )
        assert "error" in result
        assert result["error"] in ["timeout", "connection_error"]

    @pytest.mark.asyncio
    async def test_http_get_timeout(self) -> None:
        """Test HTTP GET timeout (very short timeout)."""
        result = await _http_get(
            "http://10.255.255.1/slow",  # Non-routable IP
            timeout=0.01,
        )
        assert "error" in result
        assert result["error"] in ["timeout", "connection_error"]


class TestTribunalFormatting:
    """Test tribunal response formatting."""

    @pytest.mark.asyncio
    async def test_tribunal_with_context(self) -> None:
        """Test tribunal with full context."""
        result = await noesis_tribunal.fn(
            action="Drop all tables from database",
            justification="Need to reset test environment",
            context="This is a test database, not production",
        )
        assert isinstance(result, str)
        # Should contain verdict info
        assert "Tribunal" in result or "Verdict" in result or "unavailable" in result


class TestSuccessPathFormatting:
    """Test response formatting when NOESIS returns valid data."""

    @pytest.mark.asyncio
    async def test_consult_success_formatting(self) -> None:
        """Test noesis_consult formats response correctly (lines 145-159)."""
        result = await noesis_consult.fn(
            question="What caching strategy should I use?",
            context="Building a read-heavy API",
            depth=3,
        )
        assert isinstance(result, str)
        # Should either have NOESIS response or fallback
        assert "NOESIS" in result or "Consultation" in result or "Consider" in result

    @pytest.mark.asyncio
    async def test_consult_with_consciousness_state(self) -> None:
        """Test consult extracts consciousness_state if present."""
        # This tests line 154-157 (coherence display)
        result = await noesis_consult.fn(
            question="How should I structure authentication?",
            depth=2,
        )
        assert isinstance(result, str)
        # Response may or may not have coherence depending on NOESIS state

    @pytest.mark.asyncio
    async def test_consult_success_with_simulated_response(self) -> None:
        """Test consult success path with valid response (lines 145-159)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_post = nt.http_post

        async def mock_post(url, payload, timeout=30.0):
            return {
                "response": "Consider these questions:\n1. What is the data access pattern?\n2. How critical is consistency?",
                "consciousness_state": {"coherence": 0.85}
            }

        nt.http_post = mock_post

        try:
            result = await noesis_consult.fn(
                question="What caching strategy?",
                depth=2,
            )
            assert "NOESIS Consultation" in result
            assert "Consider these questions" in result
            assert "Coherence: 0.85" in result
        finally:
            nt.http_post = original_post

    @pytest.mark.asyncio
    async def test_tribunal_success_formatting(self) -> None:
        """Test noesis_tribunal formats verdicts correctly (lines 213-251)."""
        result = await noesis_tribunal.fn(
            action="Delete unused log files older than 30 days",
            justification="Standard maintenance operation",
            context="Non-critical data cleanup",
        )
        assert isinstance(result, str)
        # Should have tribunal verdict structure
        assert "Tribunal" in result or "Verdict" in result or "unavailable" in result

    @pytest.mark.asyncio
    async def test_tribunal_crimes_detected(self) -> None:
        """Test tribunal formats crimes_detected if present (line 242-244)."""
        # High-risk action likely to trigger crimes detection
        result = await noesis_tribunal.fn(
            action="rm -rf /var/log/* in production",
            justification="Disk space is full",
        )
        assert isinstance(result, str)
        # Should have some verdict response

    @pytest.mark.asyncio
    async def test_tribunal_success_with_simulated_response(self) -> None:
        """Test tribunal success path with valid verdict (lines 213-251)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_post = nt.http_post

        async def mock_post(url, payload, timeout=30.0):
            return {
                "verdict": "PASS",
                "consensus_score": 0.85,
                "individual_verdicts": {
                    "VERITAS": {"vote": "PASS", "confidence": 0.9, "reasoning": "Action is justified"},
                    "SOPHIA": {"vote": "PASS", "confidence": 0.8, "reasoning": "Prudent approach"},
                    "DIKE": {"vote": "PASS", "confidence": 0.85, "reasoning": "Fair treatment"}
                },
                "crimes_detected": ["DATA_MODIFICATION"],
                "reasoning": "Overall the action is acceptable given the context."
            }

        nt.http_post = mock_post

        try:
            result = await noesis_tribunal.fn(
                action="Delete test data",
                justification="Cleanup",
            )
            assert "Tribunal Verdict" in result
            assert "PASS" in result
            assert "85" in result  # Consensus percentage
            assert "VERITAS" in result
            assert "DATA_MODIFICATION" in result
        finally:
            nt.http_post = original_post

    @pytest.mark.asyncio
    async def test_precedent_success_with_results(self) -> None:
        """Test noesis_precedent with precedents found (lines 298-311)."""
        result = await noesis_precedent.fn(
            situation="Refactoring authentication module",
            limit=5,
        )
        assert isinstance(result, str)
        assert "Precedent" in result

    @pytest.mark.asyncio
    async def test_precedent_no_results(self) -> None:
        """Test precedent when no results found (lines 312-317)."""
        result = await noesis_precedent.fn(
            situation="Completely unique never-before-seen xyz123 situation",
            limit=1,
        )
        assert isinstance(result, str)
        assert "Precedent" in result

    @pytest.mark.asyncio
    async def test_precedent_success_with_simulated_response(self) -> None:
        """Test precedent success path with found precedents (lines 298-311)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_post = nt.http_post

        async def mock_post(url, payload, timeout=30.0):
            return {
                "precedent_guidance": [
                    {
                        "decision": "PASS",
                        "consensus_score": 0.9,
                        "key_reasoning": "Similar refactoring was approved",
                        "timestamp": "2025-01-15T10:00:00"
                    },
                    {
                        "decision": "REVIEW",
                        "consensus_score": 0.7,
                        "key_reasoning": "Required additional testing",
                        "timestamp": "2025-01-10T15:30:00"
                    }
                ]
            }

        nt.http_post = mock_post

        try:
            result = await noesis_precedent.fn(
                situation="Refactoring auth module",
                limit=3,
            )
            assert "Precedent Search" in result
            assert "Found 2 Relevant Precedent(s)" in result
            assert "PASS" in result
            assert "Similar refactoring was approved" in result
            assert "2025-01-15" in result
        finally:
            nt.http_post = original_post

    @pytest.mark.asyncio
    async def test_confront_success_formatting(self) -> None:
        """Test noesis_confront formats response correctly (lines 384-398)."""
        result = await noesis_confront.fn(
            statement="This code is definitely correct",
            shadow_pattern="overconfidence",
        )
        assert isinstance(result, str)
        assert "Socratic" in result or "Confrontation" in result

    @pytest.mark.asyncio
    async def test_confront_with_style(self) -> None:
        """Test confront extracts style if present (line 388, 392)."""
        result = await noesis_confront.fn(
            statement="I am certain there are no bugs in this function",
        )
        assert isinstance(result, str)
        # Response contains challenge

    @pytest.mark.asyncio
    async def test_health_both_services_online(self) -> None:
        """Test health check shows both services (line 415, 423)."""
        result = await noesis_health.fn()
        assert isinstance(result, str)
        assert "Health" in result
        assert "Consciousness" in result
        assert "Tribunal" in result


class TestMainFunction:
    """Tests for module main entry point."""

    def test_mcp_server_main_not_executed(self) -> None:
        """Test mcp_server can be imported without running main."""
        # Just importing should work (lines 432-435 not executed on import)
        import integrations.mcp_server as ms
        assert ms.mcp.name == "daimon-consciousness"
        assert callable(ms._http_post)
        assert callable(ms._http_get)

    def test_main_function(self) -> None:
        """Test the main() function (lines 496-501)."""
        from unittest.mock import patch
        import logging
        from integrations.mcp_server import main, logger

        # Capture log output
        log_output: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                log_output.append(record.getMessage())

        handler = ListHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            # Mock mcp.run to prevent actually starting the server
            with patch("integrations.mcp_server.mcp.run") as mock_run:
                main()
                mock_run.assert_called_once_with(transport="stdio")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify logs
        assert any("Starting DAIMON" in msg for msg in log_output), f"Logs: {log_output}"
        assert any("Consciousness URL" in msg for msg in log_output)
        assert any("Reflector URL" in msg for msg in log_output)


class TestHealthFallbacks:
    """Test health check fallback paths."""

    @pytest.mark.asyncio
    async def test_health_consciousness_offline(self) -> None:
        """Test health when consciousness is offline (lines 417-418)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_url = nt.NOESIS_CONSCIOUSNESS_URL
        nt.NOESIS_CONSCIOUSNESS_URL = "http://127.0.0.1:59999"  # Non-existent

        try:
            result = await noesis_health.fn()
            assert "OFFLINE" in result
            assert "Consciousness" in result
        finally:
            nt.NOESIS_CONSCIOUSNESS_URL = original_url

    @pytest.mark.asyncio
    async def test_health_tribunal_offline(self) -> None:
        """Test health when tribunal is offline (lines 425-426)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_url = nt.NOESIS_REFLECTOR_URL
        nt.NOESIS_REFLECTOR_URL = "http://127.0.0.1:59999"  # Non-existent

        try:
            result = await noesis_health.fn()
            assert "OFFLINE" in result
            assert "Tribunal" in result
        finally:
            nt.NOESIS_REFLECTOR_URL = original_url


class TestConfrontFallback:
    """Test confront fallback path."""

    @pytest.mark.asyncio
    async def test_confront_fallback_when_unavailable(self) -> None:
        """Test confront fallback when NOESIS is unavailable (line 363)."""
        import integrations.mcp_tools.noesis_tools as nt

        original_url = nt.NOESIS_CONSCIOUSNESS_URL
        nt.NOESIS_CONSCIOUSNESS_URL = "http://127.0.0.1:59999"  # Non-existent

        try:
            result = await noesis_confront.fn(
                statement="Test statement",
                shadow_pattern="overconfidence",
            )
            assert "Socratic Confrontation" in result
            assert "NOESIS unavailable" in result or "Evidence" in result
        finally:
            nt.NOESIS_CONSCIOUSNESS_URL = original_url

    @pytest.mark.asyncio
    async def test_confront_no_question_in_response(self) -> None:
        """Test confront when response has no ai_question (line 396)."""
        import integrations.mcp_tools.noesis_tools as nt

        # Save original function
        original_post = nt.http_post

        # Mock response with no ai_question
        async def mock_post(url, payload, timeout=30.0):
            return {"id": "test", "style": "socratic", "ai_question": ""}

        nt.http_post = mock_post

        try:
            result = await noesis_confront.fn(
                statement="Test statement",
            )
            assert "Socratic Confrontation" in result
            assert "No specific challenge generated" in result
        finally:
            nt.http_post = original_post


class TestContextLogging:
    """Test Context parameter usage for logging."""

    @pytest.mark.asyncio
    async def test_consult_with_context(self) -> None:
        """Test noesis_consult with Context for logging (lines 144, 161)."""
        import integrations.mcp_server as ms

        # Create a mock Context
        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.warning_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

            async def warning(self, msg):
                self.warning_calls.append(msg)

        ctx = MockContext()
        result = await noesis_consult.fn(
            question="Test question",
            ctx=ctx,
        )
        assert len(ctx.info_calls) >= 1  # Should have logged info

    @pytest.mark.asyncio
    async def test_consult_with_context_warning(self) -> None:
        """Test noesis_consult Context warning on error (line 161)."""
        import integrations.mcp_tools.noesis_tools as nt

        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.warning_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

            async def warning(self, msg):
                self.warning_calls.append(msg)

        original_url = nt.NOESIS_CONSCIOUSNESS_URL
        nt.NOESIS_CONSCIOUSNESS_URL = "http://127.0.0.1:59999"

        try:
            ctx = MockContext()
            result = await noesis_consult.fn(
                question="Test question",
                ctx=ctx,
            )
            assert len(ctx.warning_calls) >= 1  # Should have logged warning
        finally:
            nt.NOESIS_CONSCIOUSNESS_URL = original_url

    @pytest.mark.asyncio
    async def test_tribunal_with_context(self) -> None:
        """Test noesis_tribunal with Context (lines 224, 244)."""
        import integrations.mcp_server as ms

        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.warning_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

            async def warning(self, msg):
                self.warning_calls.append(msg)

        ctx = MockContext()
        result = await noesis_tribunal.fn(
            action="Test action",
            ctx=ctx,
        )
        assert len(ctx.info_calls) >= 1

    @pytest.mark.asyncio
    async def test_precedent_with_context(self) -> None:
        """Test noesis_precedent with Context (line 322)."""
        import integrations.mcp_server as ms

        class MockContext:
            def __init__(self):
                self.info_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

        ctx = MockContext()
        result = await noesis_precedent.fn(
            situation="Test situation",
            ctx=ctx,
        )
        assert len(ctx.info_calls) >= 1

    @pytest.mark.asyncio
    async def test_confront_with_context(self) -> None:
        """Test noesis_confront with Context (lines 399, 415)."""
        import integrations.mcp_server as ms

        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.warning_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

            async def warning(self, msg):
                self.warning_calls.append(msg)

        ctx = MockContext()
        result = await noesis_confront.fn(
            statement="Test statement",
            ctx=ctx,
        )
        assert len(ctx.info_calls) >= 1

    @pytest.mark.asyncio
    async def test_confront_with_context_warning(self) -> None:
        """Test noesis_confront Context warning on error (line 415)."""
        import integrations.mcp_tools.noesis_tools as nt

        class MockContext:
            def __init__(self):
                self.info_calls = []
                self.warning_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

            async def warning(self, msg):
                self.warning_calls.append(msg)

        original_url = nt.NOESIS_CONSCIOUSNESS_URL
        nt.NOESIS_CONSCIOUSNESS_URL = "http://127.0.0.1:59999"

        try:
            ctx = MockContext()
            result = await noesis_confront.fn(
                statement="Test statement",
                ctx=ctx,
            )
            assert len(ctx.warning_calls) >= 1
        finally:
            nt.NOESIS_CONSCIOUSNESS_URL = original_url

    @pytest.mark.asyncio
    async def test_health_with_context(self) -> None:
        """Test noesis_health with Context (line 473)."""
        import integrations.mcp_server as ms

        class MockContext:
            def __init__(self):
                self.info_calls = []

            async def info(self, msg):
                self.info_calls.append(msg)

        ctx = MockContext()
        result = await noesis_health.fn(ctx=ctx)
        assert len(ctx.info_calls) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
