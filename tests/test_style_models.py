"""Tests for style learner data models."""

import pytest
from datetime import datetime

from learners.style_models import (
    TypingPace,
    EditingFrequency,
    InteractionPattern,
    FocusLevel,
    CommunicationStyle,
)


class TestTypingPaceEnum:
    """Tests for TypingPace enum."""

    def test_values(self):
        """All pace values are correct."""
        assert TypingPace.FAST.value == "fast"
        assert TypingPace.MODERATE.value == "moderate"
        assert TypingPace.DELIBERATE.value == "deliberate"

    def test_string_comparison(self):
        """Enum compares as string."""
        assert TypingPace.FAST == "fast"


class TestEditingFrequencyEnum:
    """Tests for EditingFrequency enum."""

    def test_values(self):
        """All frequency values are correct."""
        assert EditingFrequency.MINIMAL.value == "minimal"
        assert EditingFrequency.MODERATE.value == "moderate"
        assert EditingFrequency.HEAVY.value == "heavy"


class TestInteractionPatternEnum:
    """Tests for InteractionPattern enum."""

    def test_values(self):
        """All pattern values are correct."""
        assert InteractionPattern.BURST.value == "burst"
        assert InteractionPattern.STEADY.value == "steady"
        assert InteractionPattern.SPORADIC.value == "sporadic"


class TestFocusLevelEnum:
    """Tests for FocusLevel enum."""

    def test_values(self):
        """All focus values are correct."""
        assert FocusLevel.DEEP.value == "deep"
        assert FocusLevel.SWITCHING.value == "switching"
        assert FocusLevel.MULTITASK.value == "multitask"


class TestCommunicationStyleDefaults:
    """Tests for CommunicationStyle defaults."""

    def test_default_values(self):
        """Default values are set correctly."""
        style = CommunicationStyle()
        assert style.typing_pace == TypingPace.MODERATE
        assert style.editing_frequency == EditingFrequency.MODERATE
        assert style.interaction_pattern == InteractionPattern.STEADY
        assert style.focus_level == FocusLevel.SWITCHING
        assert style.work_hours == list(range(9, 18))
        assert style.peak_productivity is None
        assert style.afk_frequency == "normal"
        assert style.confidence == 0.0
        assert style.last_updated is None


class TestCommunicationStyleToDict:
    """Tests for CommunicationStyle.to_dict()."""

    def test_basic_serialization(self):
        """Basic serialization works."""
        style = CommunicationStyle()
        d = style.to_dict()

        assert d["typing_pace"] == "moderate"
        assert d["editing_frequency"] == "moderate"
        assert d["interaction_pattern"] == "steady"
        assert d["focus_level"] == "switching"
        assert d["work_hours"] == list(range(9, 18))
        assert d["peak_productivity"] is None
        assert d["afk_frequency"] == "normal"
        assert d["confidence"] == 0.0
        assert d["last_updated"] is None

    def test_with_last_updated(self):
        """Serializes datetime correctly."""
        now = datetime(2025, 1, 15, 10, 30, 0)
        style = CommunicationStyle(last_updated=now)
        d = style.to_dict()

        assert d["last_updated"] == "2025-01-15T10:30:00"

    def test_confidence_rounded(self):
        """Confidence is rounded to 2 decimals."""
        style = CommunicationStyle(confidence=0.666666)
        d = style.to_dict()

        assert d["confidence"] == 0.67

    def test_custom_values(self):
        """Custom values serialize correctly."""
        style = CommunicationStyle(
            typing_pace=TypingPace.FAST,
            editing_frequency=EditingFrequency.HEAVY,
            interaction_pattern=InteractionPattern.BURST,
            focus_level=FocusLevel.DEEP,
            work_hours=[22, 23, 0, 1, 2],  # Night owl
            peak_productivity=23,
            afk_frequency="low",
            confidence=0.85,
        )
        d = style.to_dict()

        assert d["typing_pace"] == "fast"
        assert d["editing_frequency"] == "heavy"
        assert d["interaction_pattern"] == "burst"
        assert d["focus_level"] == "deep"
        assert d["work_hours"] == [22, 23, 0, 1, 2]
        assert d["peak_productivity"] == 23
        assert d["afk_frequency"] == "low"
        assert d["confidence"] == 0.85


class TestCommunicationStyleSuggestions:
    """Tests for to_claude_suggestions()."""

    def test_fast_typing_suggestion(self):
        """Fast typing generates concise suggestion."""
        style = CommunicationStyle(typing_pace=TypingPace.FAST)
        suggestions = style.to_claude_suggestions()

        assert any("concise" in s for s in suggestions)

    def test_deliberate_typing_suggestion(self):
        """Deliberate typing generates thorough explanation suggestion."""
        style = CommunicationStyle(typing_pace=TypingPace.DELIBERATE)
        suggestions = style.to_claude_suggestions()

        assert any("thorough" in s for s in suggestions)

    def test_minimal_editing_suggestion(self):
        """Minimal editing generates polished solution suggestion."""
        style = CommunicationStyle(editing_frequency=EditingFrequency.MINIMAL)
        suggestions = style.to_claude_suggestions()

        assert any("polished" in s.lower() for s in suggestions)

    def test_heavy_editing_suggestion(self):
        """Heavy editing generates incremental refinement suggestion."""
        style = CommunicationStyle(editing_frequency=EditingFrequency.HEAVY)
        suggestions = style.to_claude_suggestions()

        assert any("incremental" in s.lower() or "drafts" in s.lower() for s in suggestions)

    def test_deep_focus_suggestion(self):
        """Deep focus generates grouped information suggestion."""
        style = CommunicationStyle(focus_level=FocusLevel.DEEP)
        suggestions = style.to_claude_suggestions()

        assert any("focus" in s.lower() or "group" in s.lower() for s in suggestions)

    def test_multitask_suggestion(self):
        """Multitask focus generates clear headers suggestion."""
        style = CommunicationStyle(focus_level=FocusLevel.MULTITASK)
        suggestions = style.to_claude_suggestions()

        assert any("header" in s.lower() or "scanning" in s.lower() for s in suggestions)

    def test_burst_pattern_suggestion(self):
        """Burst pattern generates actionable info suggestion."""
        style = CommunicationStyle(interaction_pattern=InteractionPattern.BURST)
        suggestions = style.to_claude_suggestions()

        assert any("actionable" in s.lower() for s in suggestions)

    def test_sporadic_pattern_suggestion(self):
        """Sporadic pattern generates context reminder suggestion."""
        style = CommunicationStyle(interaction_pattern=InteractionPattern.SPORADIC)
        suggestions = style.to_claude_suggestions()

        assert any("context" in s.lower() or "reminder" in s.lower() for s in suggestions)

    def test_moderate_values_no_suggestions(self):
        """Moderate values don't generate specific suggestions."""
        style = CommunicationStyle(
            typing_pace=TypingPace.MODERATE,
            editing_frequency=EditingFrequency.MODERATE,
            focus_level=FocusLevel.SWITCHING,
            interaction_pattern=InteractionPattern.STEADY,
        )
        suggestions = style.to_claude_suggestions()

        # Default moderate values shouldn't trigger special suggestions
        assert len(suggestions) == 0

    def test_combined_suggestions(self):
        """Multiple non-default values generate multiple suggestions."""
        style = CommunicationStyle(
            typing_pace=TypingPace.FAST,
            editing_frequency=EditingFrequency.HEAVY,
            focus_level=FocusLevel.DEEP,
            interaction_pattern=InteractionPattern.BURST,
        )
        suggestions = style.to_claude_suggestions()

        # Should have suggestions for all non-default values
        assert len(suggestions) == 4
