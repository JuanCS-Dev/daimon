"""
DAIMON User Model Data Structures
=================================

Core data models for user personalization.

Contains:
- UserPreferences: Communication and code style preferences
- CognitiveProfile: Work patterns and cognitive characteristics
- UserModel: Complete user profile with learned patterns

Extracted from user_model.py for CODE_CONSTITUTION compliance.

Usage:
    from memory.user_models import UserModel, UserPreferences, CognitiveProfile

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List

MAX_PATTERNS = 100


@dataclass
class UserPreferences:
    """
    User interaction preferences.
    
    Attributes:
        communication_style: How verbose responses should be
        code_style: Code documentation level preference
        preferred_tools: List of preferred assistant tools
        language: User's preferred language
    """
    
    communication_style: str = "balanced"  # "concise" | "detailed" | "balanced"
    code_style: str = "documented"         # "minimal" | "documented" | "verbose"
    preferred_tools: List[str] = field(default_factory=list)
    language: str = "pt-BR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary."""
        return cls(
            communication_style=data.get("communication_style", "balanced"),
            code_style=data.get("code_style", "documented"),
            preferred_tools=data.get("preferred_tools", []),
            language=data.get("language", "pt-BR"),
        )


@dataclass
class CognitiveProfile:
    """
    User cognitive patterns and work characteristics.
    
    Attributes:
        avg_flow_duration: Average flow state duration in minutes
        peak_hours: Hours of day with highest productivity (0-23)
        fatigue_threshold: Fatigue level that triggers break suggestions
        break_preference: Preferred break pattern style
    """
    
    avg_flow_duration: float = 45.0  # minutes
    peak_hours: List[int] = field(default_factory=lambda: [10, 11, 14, 15, 16])
    fatigue_threshold: float = 0.7
    break_preference: str = "flow"  # "pomodoro" | "flow" | "scheduled"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveProfile":
        """Create from dictionary."""
        return cls(
            avg_flow_duration=float(data.get("avg_flow_duration", 45.0)),
            peak_hours=data.get("peak_hours", [10, 11, 14, 15, 16]),
            fatigue_threshold=float(data.get("fatigue_threshold", 0.7)),
            break_preference=data.get("break_preference", "flow"),
        )


@dataclass
class UserModel:
    """
    Complete user model for personalization.
    
    Aggregates preferences, cognitive profile, and learned patterns
    to enable personalized AI assistance.
    
    Attributes:
        user_id: Unique user identifier
        preferences: User interaction preferences
        cognitive: Cognitive work patterns
        patterns: Learned behavioral patterns (max 100)
        version: Model version for sync
        last_updated: Timestamp of last update
    """
    
    user_id: str
    preferences: UserPreferences = field(default_factory=UserPreferences)
    cognitive: CognitiveProfile = field(default_factory=CognitiveProfile)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences.to_dict(),
            "cognitive": self.cognitive.to_dict(),
            "patterns": self.patterns,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
        """Create UserModel from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except ValueError:
                last_updated = datetime.now()
        elif not isinstance(last_updated, datetime):
            last_updated = datetime.now()
        
        return cls(
            user_id=data.get("user_id", "default"),
            preferences=UserPreferences.from_dict(data.get("preferences", {})),
            cognitive=CognitiveProfile.from_dict(data.get("cognitive", {})),
            patterns=data.get("patterns", [])[:MAX_PATTERNS],
            version=int(data.get("version", 1)),
            last_updated=last_updated,
        )
