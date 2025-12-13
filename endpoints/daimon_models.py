"""
DAIMON API Models.

Pydantic models for DAIMON API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Shell Batch Models
class ShellHeartbeat(BaseModel):
    """Single shell command heartbeat."""

    timestamp: str = Field(..., description="ISO timestamp")
    command: str = Field(..., description="Shell command executed")
    pwd: str = Field(..., description="Working directory")
    exit_code: int = Field(..., description="Command exit code")
    duration: float = Field(0.0, description="Execution duration in seconds")
    git_branch: str = Field("", description="Current git branch if in repo")


class ShellBatchRequest(BaseModel):
    """Batch of shell heartbeats with detected patterns."""

    heartbeats: List[ShellHeartbeat] = Field(..., description="List of heartbeats")
    patterns: Dict[str, Any] = Field(default_factory=dict, description="Detected patterns")


class ShellBatchResponse(BaseModel):
    """Response for shell batch endpoint."""

    status: str = Field(..., description="Processing status")
    stored: int = Field(..., description="Number of heartbeats stored")
    insights: List[str] = Field(default_factory=list, description="Generated insights")


# Claude Event Models
class ClaudeEvent(BaseModel):
    """Event from Claude Code session."""

    event_type: str = Field(..., description="Event type: create, fix, refactor, understand, delete")
    timestamp: str = Field(..., description="ISO timestamp")
    project: str = Field("", description="Project identifier")
    files_touched: List[str] = Field(default_factory=list, description="Files involved")
    intention: str = Field("", description="Detected intention")


class ClaudeEventResponse(BaseModel):
    """Response for Claude event endpoint."""

    status: str = Field(..., description="Processing status")
    stored: bool = Field(..., description="Whether event was stored")


# Session End Models
class SessionEndRequest(BaseModel):
    """Request to record session end as precedent."""

    session_id: str = Field(..., description="Session identifier")
    summary: str = Field(..., description="Session summary")
    outcome: str = Field("success", description="Session outcome: success, failure, partial")
    duration_minutes: float = Field(0.0, description="Session duration")
    files_changed: int = Field(0, description="Number of files modified")


class SessionEndResponse(BaseModel):
    """Response for session end endpoint."""

    status: str = Field(..., description="Processing status")
    precedent_id: Optional[str] = Field(None, description="Created precedent ID if significant")


# Preferences Models
class PreferencesResponse(BaseModel):
    """Response for learned preferences endpoint."""

    preferences: Dict[str, Any] = Field(default_factory=dict, description="Learned preferences by category")
    total_signals: int = Field(0, description="Total signals analyzed")
    last_reflection: Optional[str] = Field(None, description="Last reflection timestamp")
    approval_rate: float = Field(0.0, description="Overall approval rate")


# Reflect Models
class ReflectResponse(BaseModel):
    """Response for reflection trigger."""

    status: str = Field(..., description="Reflection status")
    signals_count: int = Field(0, description="Signals analyzed")
    insights_count: int = Field(0, description="Insights generated")
    updated: bool = Field(False, description="Whether CLAUDE.md was updated")
    elapsed_seconds: float = Field(0.0, description="Time taken")


# Memory Models
class MemoryItem(BaseModel):
    """Single memory item for dashboard display."""

    id: str = Field(..., description="Memory identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    type: str = Field(..., description="Memory type: command, event, preference")
    content: str = Field(..., description="Memory content summary")
    importance: float = Field(0.5, description="Importance score 0-1")


class RecentMemoriesResponse(BaseModel):
    """Response for recent memories endpoint."""

    memories: List[MemoryItem] = Field(default_factory=list, description="Recent memories")
    total_count: int = Field(0, description="Total memories in store")
