"""
DAIMON Preference Models
========================

Core data structures for the preference learning system.

This module defines the fundamental types used throughout the preference
learning pipeline: signals, categories, and related enums.

Architecture:
    PreferenceSignal is the primary data structure that flows through:
    Scanner → Detector → Categorizer → InsightGenerator

Usage:
    from learners.preference.models import PreferenceSignal, SignalType

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SignalType(str, Enum):
    """Types of preference signals detected in conversations."""
    
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    NEUTRAL = "neutral"


class PreferenceCategory(str, Enum):
    """Categories of user preferences."""
    
    CODE_STYLE = "code_style"
    VERBOSITY = "verbosity"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    WORKFLOW = "workflow"
    SECURITY = "security"
    PERFORMANCE = "performance"
    GENERAL = "general"


@dataclass
class PreferenceSignal:
    """
    A detected preference signal from user-assistant interaction.
    
    Represents a single instance of user feedback detected during
    conversation analysis. Used by the preference learning pipeline
    to build user preference profiles.
    
    Attributes:
        timestamp: ISO format timestamp of detection
        signal_type: Type of signal (approval/rejection/modification)
        context: Description of what triggered the signal
        category: Preference category (code_style, testing, etc.)
        strength: Signal confidence from 0.0 to 1.0
        session_id: Identifier of source session
        tool_involved: Name of tool that triggered feedback (if any)
    """
    
    timestamp: str
    signal_type: str
    context: str
    category: str
    strength: float
    session_id: str
    tool_involved: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceSignal":
        """Create PreferenceSignal from dictionary."""
        return cls(
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            signal_type=data.get("signal_type", SignalType.NEUTRAL.value),
            context=data.get("context", ""),
            category=data.get("category", PreferenceCategory.GENERAL.value),
            strength=float(data.get("strength", 0.5)),
            session_id=data.get("session_id", "unknown"),
            tool_involved=data.get("tool_involved"),
        )


@dataclass
class CategoryStats:
    """Statistics for a preference category."""
    
    approvals: int = 0
    rejections: int = 0
    
    @property
    def total(self) -> int:
        """Total signals in this category."""
        return self.approvals + self.rejections
    
    @property
    def approval_rate(self) -> float:
        """Approval rate as fraction (0.0-1.0)."""
        if self.total == 0:
            return 0.0
        return self.approvals / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approvals": self.approvals,
            "rejections": self.rejections,
            "total": self.total,
            "approval_rate": round(self.approval_rate, 2),
        }


@dataclass
class PreferenceInsight:
    """
    An actionable insight derived from preference analysis.
    
    Used to generate recommendations for updating CLAUDE.md
    based on observed user preferences.
    """
    
    category: str
    action: str  # "reduce" | "reinforce"
    confidence: float
    approval_rate: float
    total_signals: int
    suggestion: str
    from_llm: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
