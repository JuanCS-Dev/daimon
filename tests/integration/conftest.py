"""
Shared Fixtures for Integration Tests
======================================

Provides common fixtures used across all integration test files.

Fixtures:
- tmp_storage: Temporary storage paths
- mock_noesis: Mock NOESIS endpoints
- sample_messages: Sample Claude session messages
- sample_events: Sample behavioral events

Follows CODE_CONSTITUTION: Clarity Over Cleverness.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Dict[str, Path]:
    """
    Create temporary storage paths for tests.
    
    Returns:
        Dictionary with paths for:
        - user_model: UserModel JSON storage
        - projects: Mock Claude projects directory
        - claude_md: Mock CLAUDE.md file
    """
    user_model = tmp_path / "user_model.json"
    projects = tmp_path / "projects"
    projects.mkdir()
    
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project\n\nSome content.\n")
    
    return {
        "user_model": user_model,
        "projects": projects,
        "claude_md": claude_md,
        "root": tmp_path,
    }


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """
    Sample Claude session messages for testing.
    
    Returns:
        List of message dictionaries simulating a conversation
    """
    return [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Here is the refactored code:"},
                    {"type": "tool_use", "name": "Edit"},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "perfeito, ficou muito bom!"},
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": "Let me add the tests now."
            },
        },
        {
            "type": "user",
            "message": {
                "content": "sim, pode adicionar"
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": "Here are the unit tests."
            },
        },
        {
            "type": "user",
            "message": {
                "content": "nao, muito longo. Reduz."
            },
        },
    ]


@pytest.fixture
def sample_events() -> List[Dict[str, Any]]:
    """
    Sample behavioral events for pattern detection testing.
    
    Returns:
        List of event dictionaries simulating user behavior
    """
    base_time = datetime.now()
    
    events = []
    
    # Temporal pattern: git commits at 5pm
    for i in range(5):
        events.append({
            "type": "shell_command",
            "command": "git commit -m 'work'",
            "timestamp": (base_time.replace(hour=17) - timedelta(days=i)).isoformat(),
        })
    
    # Sequential pattern: git status → add → commit
    for _ in range(4):
        events.append({"type": "shell_command", "command": "git status"})
        events.append({"type": "shell_command", "command": "git add ."})
        events.append({"type": "shell_command", "command": "git commit"})
    
    return events


@pytest.fixture
def create_session_file(tmp_storage: Dict[str, Path]):
    """
    Factory fixture to create session files.
    
    Returns:
        Function that creates JSONL session files
    """
    def _create(
        project_name: str,
        messages: List[Dict[str, Any]],
    ) -> Path:
        project_dir = tmp_storage["projects"] / project_name
        project_dir.mkdir(exist_ok=True)
        
        session_file = project_dir / "session.jsonl"
        with open(session_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        
        return session_file
    
    return _create


@pytest.fixture
def mock_noesis_offline():
    """
    Mock NOESIS as offline/unavailable.
    
    Returns:
        Context manager that patches NOESIS calls to fail
    """
    with patch("integrations.mcp_tools.http_utils.http_get") as mock_get, \
         patch("integrations.mcp_tools.http_utils.http_post") as mock_post:
        
        mock_get.side_effect = ConnectionError("NOESIS offline")
        mock_post.side_effect = ConnectionError("NOESIS offline")
        
        yield {"get": mock_get, "post": mock_post}


@pytest.fixture
def mock_noesis_online():
    """
    Mock NOESIS as online and responding.
    
    Returns:
        Context manager with mock NOESIS responses
    """
    async def mock_get(url: str) -> Dict[str, Any]:
        return {"status": "ok", "data": {}}
    
    async def mock_post(url: str, data: Dict) -> Dict[str, Any]:
        return {"status": "ok", "data": data}
    
    with patch("integrations.mcp_tools.http_utils.http_get", new=mock_get), \
         patch("integrations.mcp_tools.http_utils.http_post", new=mock_post):
        yield


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service for testing without actual LLM calls.
    
    Returns:
        Mock LearnerLLMService
    """
    from learners.llm_models import ClassificationResult, InsightResult
    
    mock_service = MagicMock()
    
    # Mock classify
    async def mock_classify(content, options, context=""):
        if any(word in content.lower() for word in ["sim", "ok", "perfeito", "bom"]):
            return ClassificationResult(
                category="approval",
                confidence=0.9,
                reasoning="Positive sentiment detected",
            )
        elif any(word in content.lower() for word in ["nao", "errado", "ruim"]):
            return ClassificationResult(
                category="rejection",
                confidence=0.85,
                reasoning="Negative sentiment detected",
            )
        return ClassificationResult(
            category="neutral",
            confidence=0.6,
            reasoning="No clear sentiment",
        )
    
    mock_service.classify = AsyncMock(side_effect=mock_classify)
    
    # Mock extract_insights
    async def mock_extract(data: Dict) -> InsightResult:
        return InsightResult(
            insights=["User prefers concise responses"],
            suggestions=["Reduce verbosity in explanations"],
            confidence=0.8,
        )
    
    mock_service.extract_insights = AsyncMock(side_effect=mock_extract)
    
    return mock_service
