"""
Style Learner Data Models.

Contains enums and dataclasses for communication style inference.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TypingPace(str, Enum):
    """Typing speed classification."""
    FAST = "fast"           # >300 CPM
    MODERATE = "moderate"   # 150-300 CPM
    DELIBERATE = "deliberate"  # <150 CPM


class EditingFrequency(str, Enum):
    """How often user edits/corrects."""
    MINIMAL = "minimal"     # <10% correction rate
    MODERATE = "moderate"   # 10-25% correction rate
    HEAVY = "heavy"         # >25% correction rate


class InteractionPattern(str, Enum):
    """Work session pattern."""
    BURST = "burst"         # Short intense sessions
    STEADY = "steady"       # Consistent pacing
    SPORADIC = "sporadic"   # Irregular with long breaks


class FocusLevel(str, Enum):
    """Attention focus pattern."""
    DEEP = "deep"           # Long focus on single app
    SWITCHING = "switching" # Frequent app switches
    MULTITASK = "multitask" # Multiple apps simultaneously


@dataclass
class CommunicationStyle:  # pylint: disable=too-many-instance-attributes
    """
    Inferred communication style preferences.

    Derived from observed activity patterns, not direct observation of content.

    Attributes:
        typing_pace: Speed classification.
        editing_frequency: How often corrections are made.
        interaction_pattern: Work session pattern.
        focus_level: Attention distribution.
        work_hours: Preferred working hours (list of hour numbers).
        peak_productivity: Most productive time of day.
        afk_frequency: How often user takes breaks.
        confidence: Confidence in inference (0.0 - 1.0).
        last_updated: When style was last computed.
    """
    typing_pace: TypingPace = TypingPace.MODERATE
    editing_frequency: EditingFrequency = EditingFrequency.MODERATE
    interaction_pattern: InteractionPattern = InteractionPattern.STEADY
    focus_level: FocusLevel = FocusLevel.SWITCHING
    work_hours: List[int] = field(default_factory=lambda: list(range(9, 18)))
    peak_productivity: Optional[int] = None  # Hour of day (0-23)
    afk_frequency: str = "normal"  # "low", "normal", "high"
    confidence: float = 0.0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "typing_pace": self.typing_pace.value,
            "editing_frequency": self.editing_frequency.value,
            "interaction_pattern": self.interaction_pattern.value,
            "focus_level": self.focus_level.value,
            "work_hours": self.work_hours,
            "peak_productivity": self.peak_productivity,
            "afk_frequency": self.afk_frequency,
            "confidence": round(self.confidence, 2),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    def to_claude_suggestions(self) -> List[str]:
        """
        Generate suggestions for CLAUDE.md based on style.

        Returns:
            List of markdown-formatted suggestions.
        """
        suggestions = []

        # Typing pace suggests response verbosity preference
        if self.typing_pace == TypingPace.FAST:
            suggestions.append(
                "User types quickly - may prefer concise responses that don't slow them down."
            )
        elif self.typing_pace == TypingPace.DELIBERATE:
            suggestions.append(
                "User types deliberately - may appreciate thorough explanations."
            )

        # Editing frequency suggests iteration preference
        if self.editing_frequency == EditingFrequency.MINIMAL:
            suggestions.append(
                "User rarely edits - prefers getting it right first time. "
                "Provide polished solutions."
            )
        elif self.editing_frequency == EditingFrequency.HEAVY:
            suggestions.append(
                "User frequently iterates - comfortable with incremental refinement. "
                "Quick drafts welcome."
            )

        # Focus level suggests detail preference
        if self.focus_level == FocusLevel.DEEP:
            suggestions.append(
                "User works in deep focus mode - avoid unnecessary context switches. "
                "Group related information together."
            )
        elif self.focus_level == FocusLevel.MULTITASK:
            suggestions.append(
                "User frequently switches context - provide clear section headers "
                "and summaries for easy scanning."
            )

        # Work pattern suggests session structure
        if self.interaction_pattern == InteractionPattern.BURST:
            suggestions.append(
                "User works in intense bursts - prioritize actionable information first."
            )
        elif self.interaction_pattern == InteractionPattern.SPORADIC:
            suggestions.append(
                "User has irregular schedule - include brief context reminders "
                "when resuming topics."
            )

        return suggestions
