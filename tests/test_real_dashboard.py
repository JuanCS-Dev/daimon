"""
REAL Integration Tests for Dashboard.

Tests actual FastAPI endpoints with real responses.
No mocks - real HTTP, real data processing.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for dashboard app."""
    from dashboard.app import app
    return TestClient(app)


class TestDashboardRoot:
    """Tests for main dashboard route."""

    def test_root_returns_html(self, client):
        """Root returns HTML template."""
        response = client.get("/", follow_redirects=True)
        # Should return HTML (either 200 or redirect to dashboard)
        assert response.status_code in [200, 307, 302]


class TestStatusRoutes:
    """Tests for /api/status endpoint."""

    def test_system_status(self, client):
        """Get system status."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()

        # Should have service statuses
        assert isinstance(data, dict)
        # Check for expected keys
        assert "dashboard" in data

    def test_preferences(self, client):
        """Get preferences status."""
        response = client.get("/api/preferences")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_claude_md_get(self, client):
        """Get CLAUDE.md content."""
        response = client.get("/api/claude-md")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        # Should have path info
        assert "path" in data

    def test_collectors_status(self, client):
        """Get collectors status."""
        response = client.get("/api/collectors")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        # Should have known collectors
        assert "shell_watcher" in data
        assert "claude_watcher" in data

    def test_backups_list(self, client):
        """Get backups list."""
        response = client.get("/api/backups")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "backups" in data

    def test_browser_status(self, client):
        """Get browser watcher status."""
        response = client.get("/api/browser/status")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_reflect_trigger(self, client):
        """Trigger manual reflection."""
        response = client.post("/api/reflect")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)


class TestCognitiveRoutes:
    """Tests for /api/cognitive and /api/style endpoints."""

    def test_cognitive_state(self, client):
        """Get cognitive state."""
        response = client.get("/api/cognitive")
        assert response.status_code == 200
        data = response.json()

        # Should have state info or error
        assert isinstance(data, dict)

    def test_style_profile(self, client):
        """Get communication style profile."""
        response = client.get("/api/style")
        assert response.status_code == 200
        data = response.json()

        # Should have style information
        assert isinstance(data, dict)

    def test_metacognitive_analysis(self, client):
        """Get metacognitive analysis."""
        response = client.get("/api/metacognitive")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_metacognitive_insights(self, client):
        """Get insight history."""
        response = client.get("/api/metacognitive/insights")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_cognitive_event_post(self, client):
        """Post keystroke event."""
        import time
        response = client.post(
            "/api/cognitive/event",
            json={"key": "a", "type": "press", "timestamp": time.time()},
        )
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)


class TestCorpusRoutes:
    """Tests for /api/corpus/* endpoints."""

    def test_corpus_stats(self, client):
        """Get corpus statistics."""
        response = client.get("/api/corpus/stats")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_corpus_tree(self, client):
        """Get corpus tree structure."""
        response = client.get("/api/corpus/tree")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_corpus_search(self, client):
        """Search corpus."""
        response = client.get("/api/corpus/search", params={"q": "wisdom"})
        assert response.status_code == 200
        data = response.json()

        # Should return search results
        assert isinstance(data, dict)

    def test_corpus_texts_list(self, client):
        """List all corpus texts."""
        response = client.get("/api/corpus/texts")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "texts" in data or "error" in data

    def test_corpus_texts_by_category(self, client):
        """List corpus texts filtered by category."""
        response = client.get("/api/corpus/texts", params={"category": "philosophy"})
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_corpus_text_get(self, client):
        """Get specific corpus text."""
        response = client.get("/api/corpus/text/nonexistent")
        assert response.status_code == 200
        data = response.json()

        # Should return error or text
        assert isinstance(data, dict)

    def test_corpus_text_add(self, client):
        """Add new corpus text."""
        response = client.post(
            "/api/corpus/text",
            json={
                "author": "Test Author",
                "title": "Test Title",
                "category": "philosophy",
                "content": "Test content for new text.",
                "themes": ["test"],
                "source": "test",
                "relevance": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_corpus_text_delete(self, client):
        """Delete corpus text."""
        response = client.delete("/api/corpus/text/nonexistent")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_corpus_text_add_invalid_category(self, client):
        """Add corpus text with invalid category."""
        response = client.post(
            "/api/corpus/text",
            json={
                "author": "Test",
                "title": "Test",
                "category": "invalid_category",
                "content": "Test",
                "themes": [],
                "source": "",
                "relevance": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Should return error about invalid category
        assert "error" in data


class TestMemoryRoutes:
    """Tests for /api/precedents/* and /api/memory/* endpoints."""

    def test_precedents_stats(self, client):
        """Get precedent system stats."""
        response = client.get("/api/precedents/stats")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_precedents_search(self, client):
        """Search precedents."""
        response = client.get("/api/precedents/search", params={"q": "test"})
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_memory_stats(self, client):
        """Get memory store stats."""
        response = client.get("/api/memory/stats")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_memory_search(self, client):
        """Search memory store."""
        response = client.get("/api/memory/search", params={"q": "test"})
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_memory_search_empty_query(self, client):
        """Search memory with empty query."""
        response = client.get("/api/memory/search", params={"q": ""})
        assert response.status_code == 200
        data = response.json()

        assert data["results"] == []


class TestActivityRoutes:
    """Tests for /api/activity/* endpoints."""

    def test_activity_stats(self, client):
        """Get activity stats."""
        response = client.get("/api/activity/stats")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_activity_recent(self, client):
        """Get recent activity."""
        response = client.get("/api/activity/recent")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "records" in data

    def test_activity_recent_filtered(self, client):
        """Get recent activity filtered by watcher."""
        response = client.get("/api/activity/recent", params={"watcher": "claude", "hours": 1})
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "watcher_filter" in data

    def test_activity_summary(self, client):
        """Get activity summary."""
        response = client.get("/api/activity/summary")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)


class TestDashboardHelpers:
    """Tests for dashboard helper functions."""

    def test_check_service_function_exists(self):
        """Check that check_service helper exists."""
        from dashboard.helpers import check_service

        assert callable(check_service)

    def test_check_socket_function(self):
        """Test check_socket helper."""
        from dashboard.helpers import check_socket

        result = check_socket()
        # Should return bool
        assert isinstance(result, bool)

    def test_check_process_function(self):
        """Test check_process helper."""
        from dashboard.helpers import check_process

        result = check_process("test_component")
        # Should return bool
        assert isinstance(result, bool)

    def test_service_urls_defined(self):
        """Test that service URLs are defined."""
        from dashboard.helpers import NOESIS_URL, REFLECTOR_URL

        assert isinstance(NOESIS_URL, str)
        assert isinstance(REFLECTOR_URL, str)
        assert NOESIS_URL.startswith("http")
        assert REFLECTOR_URL.startswith("http")


class TestDashboardModels:
    """Tests for dashboard Pydantic models."""

    def test_claude_md_update_model(self):
        """Test ClaudeMdUpdate model."""
        from dashboard.models import ClaudeMdUpdate

        update = ClaudeMdUpdate(content="# Test\nSome content")

        assert update.content == "# Test\nSome content"

    def test_corpus_text_create_model(self):
        """Test CorpusTextCreate model."""
        from dashboard.models import CorpusTextCreate

        text = CorpusTextCreate(
            author="Test Author",
            title="Test Title",
            category="philosophy",
            content="Test content here.",
            themes=["test", "example"],
            source="manual",
            relevance=0.8,
        )

        assert text.author == "Test Author"
        assert text.title == "Test Title"
        assert text.category == "philosophy"
        assert len(text.themes) == 2


class TestBrowserRoutes:
    """Tests for browser watcher endpoints."""

    def test_browser_status(self, client):
        """Get browser watcher status."""
        response = client.get("/api/browser/status")
        assert response.status_code == 200
        data = response.json()

        assert data.get("status") == "connected"


class TestRealDataFlow:
    """Integration tests for real data flow through dashboard."""

    def test_status_endpoint_real(self, client):
        """Test that status endpoint returns real data."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()

        # Should have dashboard as always healthy
        assert data.get("dashboard") == "healthy"

    def test_cognitive_state_real(self, client):
        """Test cognitive state with real keystroke analyzer."""
        from learners.keystroke_analyzer import get_keystroke_analyzer, reset_keystroke_analyzer

        # Reset to clean state
        reset_keystroke_analyzer()
        analyzer = get_keystroke_analyzer()

        # Add some real events
        import time
        for i in range(25):
            analyzer.add_event(key="a", event_type="press", timestamp=time.time() + i * 0.1)
            analyzer.add_event(key="a", event_type="release", timestamp=time.time() + i * 0.1 + 0.05)

        # Now query dashboard
        response = client.get("/api/cognitive")
        assert response.status_code == 200

        # Clean up
        reset_keystroke_analyzer()

    def test_style_learner_to_dashboard(self, client):
        """Test style learner data in dashboard."""
        from learners import get_style_learner

        learner = get_style_learner()

        # Add real samples
        for _ in range(10):
            learner.add_keystroke_sample({"typing_speed_cpm": 250})
            learner.add_window_sample({"focus_duration": 120})

        # Query style endpoint
        response = client.get("/api/style")
        assert response.status_code == 200

    def test_corpus_integration(self, client):
        """Test corpus endpoints return valid data."""
        # Get stats
        response = client.get("/api/corpus/stats")
        assert response.status_code == 200
        stats = response.json()

        # Should have count info
        assert isinstance(stats, dict)

    def test_precedents_integration(self, client):
        """Test precedent endpoints work."""
        response = client.get("/api/precedents/stats")
        assert response.status_code == 200
        stats = response.json()

        assert isinstance(stats, dict)
