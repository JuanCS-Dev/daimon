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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
