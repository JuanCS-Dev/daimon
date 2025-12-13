"""Tests for StyleLearner class."""

import pytest
from datetime import datetime
from learners.style_learner import StyleLearner, get_style_learner
from learners.style_models import (
    TypingPace,
    EditingFrequency,
    InteractionPattern,
    FocusLevel,
    CommunicationStyle,
)


class TestStyleLearnerInit:
    """Tests for StyleLearner initialization."""

    def test_starts_empty(self):
        """Starts with empty sample lists."""
        learner = StyleLearner()
        style = learner.get_current_style()

        assert style.confidence == 0.0
        assert len(learner._keystroke_samples) == 0
        assert len(learner._window_samples) == 0


class TestAddKeystrokeSample:
    """Tests for add_keystroke_sample method."""

    def test_adds_sample(self):
        """Adds keystroke sample with timestamp."""
        learner = StyleLearner()
        learner.add_keystroke_sample({"typing_speed_cpm": 250})

        assert len(learner._keystroke_samples) == 1
        assert "timestamp" in learner._keystroke_samples[0]
        assert learner._keystroke_samples[0]["typing_speed_cpm"] == 250

    def test_limits_samples(self):
        """Limits samples to 100."""
        learner = StyleLearner()
        for i in range(150):
            learner.add_keystroke_sample({"typing_speed_cpm": i})

        assert len(learner._keystroke_samples) == 100
        # Should keep most recent
        assert learner._keystroke_samples[-1]["typing_speed_cpm"] == 149


class TestAddWindowSample:
    """Tests for add_window_sample method."""

    def test_adds_sample(self):
        """Adds window sample with timestamp."""
        learner = StyleLearner()
        learner.add_window_sample({"app_name": "code", "focus_duration": 120})

        assert len(learner._window_samples) == 1
        assert learner._window_samples[0]["app_name"] == "code"

    def test_records_session_hour(self):
        """Records session hour for work pattern analysis."""
        learner = StyleLearner()
        learner.add_window_sample({"app_name": "code"})

        assert len(learner._session_hours) == 1
        assert 0 <= learner._session_hours[0] <= 23

    def test_limits_samples(self):
        """Limits samples to 500."""
        learner = StyleLearner()
        for i in range(600):
            learner.add_window_sample({"focus_duration": i})

        assert len(learner._window_samples) == 500


class TestAddAfkSample:
    """Tests for add_afk_sample method."""

    def test_adds_sample(self):
        """Adds AFK sample."""
        learner = StyleLearner()
        learner.add_afk_sample({"event": "afk_start", "idle_seconds": 300})

        assert len(learner._afk_samples) == 1
        assert learner._afk_samples[0]["event"] == "afk_start"

    def test_limits_samples(self):
        """Limits samples to 100."""
        learner = StyleLearner()
        for i in range(150):
            learner.add_afk_sample({"event": "afk_start"})

        assert len(learner._afk_samples) == 100


class TestAddShellSample:
    """Tests for add_shell_sample method."""

    def test_adds_sample(self):
        """Adds shell command sample."""
        learner = StyleLearner()
        learner.add_shell_sample({"command": "git status", "exit_code": 0})

        assert len(learner._shell_samples) == 1
        assert learner._shell_samples[0]["command"] == "git status"

    def test_records_session_hour(self):
        """Records session hour."""
        learner = StyleLearner()
        learner.add_shell_sample({"command": "ls"})

        assert len(learner._session_hours) >= 1

    def test_limits_samples(self):
        """Limits samples to 200."""
        learner = StyleLearner()
        for i in range(250):
            learner.add_shell_sample({"command": f"cmd{i}"})

        assert len(learner._shell_samples) == 200


class TestAddClaudeSample:
    """Tests for add_claude_sample method."""

    def test_adds_sample(self):
        """Adds Claude session sample."""
        learner = StyleLearner()
        learner.add_claude_sample({"intention": "code_generation", "preference_signal": "approval"})

        assert len(learner._claude_samples) == 1
        assert learner._claude_samples[0]["preference_signal"] == "approval"

    def test_limits_samples(self):
        """Limits samples to 200."""
        learner = StyleLearner()
        for i in range(250):
            learner.add_claude_sample({"intention": f"task{i}"})

        assert len(learner._claude_samples) == 200


class TestAnalyzeTyping:
    """Tests for _analyze_typing method."""

    def test_fast_typing(self):
        """Detects fast typing pace."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 350, "total_keystrokes": 100, "pause_count": 5})

        style = learner.compute_style()
        assert style.typing_pace == TypingPace.FAST

    def test_deliberate_typing(self):
        """Detects deliberate typing pace."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 100, "total_keystrokes": 100, "pause_count": 5})

        style = learner.compute_style()
        assert style.typing_pace == TypingPace.DELIBERATE

    def test_moderate_typing(self):
        """Detects moderate typing pace."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 200, "total_keystrokes": 100, "pause_count": 15})

        style = learner.compute_style()
        assert style.typing_pace == TypingPace.MODERATE

    def test_heavy_editing(self):
        """Detects heavy editing frequency."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 200, "total_keystrokes": 100, "pause_count": 30})

        style = learner.compute_style()
        assert style.editing_frequency == EditingFrequency.HEAVY

    def test_minimal_editing(self):
        """Detects minimal editing frequency."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 200, "total_keystrokes": 100, "pause_count": 5})

        style = learner.compute_style()
        assert style.editing_frequency == EditingFrequency.MINIMAL


class TestAnalyzeFocus:
    """Tests for _analyze_focus method."""

    def test_deep_focus(self):
        """Detects deep focus from long durations."""
        learner = StyleLearner()
        for _ in range(60):
            learner.add_window_sample({"app_name": "code", "focus_duration": 400})

        style = learner.compute_style()
        assert style.focus_level == FocusLevel.DEEP

    def test_multitask(self):
        """Detects multitask from short durations."""
        learner = StyleLearner()
        for _ in range(60):
            learner.add_window_sample({"app_name": "code", "focus_duration": 30})

        style = learner.compute_style()
        assert style.focus_level == FocusLevel.MULTITASK

    def test_switching(self):
        """Detects switching focus."""
        learner = StyleLearner()
        for _ in range(60):
            learner.add_window_sample({"app_name": "code", "focus_duration": 120})

        style = learner.compute_style()
        assert style.focus_level == FocusLevel.SWITCHING


class TestAnalyzeWorkHours:
    """Tests for _analyze_work_hours method."""

    def test_identifies_work_hours(self):
        """Identifies active work hours."""
        learner = StyleLearner()
        # Simulate activity during hours 9-17
        learner._session_hours = [10, 10, 11, 11, 14, 14, 15, 15] * 20

        style = learner.compute_style()
        # Should have detected work hours
        assert len(style.work_hours) > 0

    def test_identifies_peak_productivity(self):
        """Identifies peak productivity hour."""
        learner = StyleLearner()
        # Most activity at hour 14
        learner._session_hours = [9, 10, 14, 14, 14, 14, 14, 16, 17] * 15

        style = learner.compute_style()
        assert style.peak_productivity == 14


class TestAnalyzeAfkPattern:
    """Tests for _analyze_afk_pattern method."""

    def test_burst_pattern(self):
        """Detects burst work pattern with few breaks."""
        learner = StyleLearner()
        # Very few AFK events relative to window
        # afk_per_hour < 1 means burst pattern
        # With 30 samples, sample_hours = 3, so need < 3 afk_starts
        for i in range(2):
            learner.add_afk_sample({"event": "afk_start", "idle_seconds": 300})
        for i in range(28):
            learner.add_afk_sample({"event": "afk_end"})

        style = learner.compute_style()
        assert style.interaction_pattern == InteractionPattern.BURST
        assert style.afk_frequency == "low"

    def test_sporadic_pattern(self):
        """Detects sporadic work pattern with many breaks."""
        learner = StyleLearner()
        # Many AFK events
        for i in range(50):
            learner.add_afk_sample({"event": "afk_start", "idle_seconds": 300})

        style = learner.compute_style()
        assert style.interaction_pattern == InteractionPattern.SPORADIC
        assert style.afk_frequency == "high"


class TestAnalyzeShellIntensity:
    """Tests for _analyze_shell_intensity method."""

    def test_high_error_rate(self):
        """High error rate suggests heavy editing."""
        learner = StyleLearner()
        # Many failed commands
        for i in range(50):
            learner.add_shell_sample({"command": f"cmd{i}", "exit_code": 1 if i % 2 == 0 else 0})

        style = learner.compute_style()
        assert style.editing_frequency == EditingFrequency.HEAVY

    def test_low_error_rate(self):
        """Low error rate suggests careful work."""
        learner = StyleLearner()
        for i in range(50):
            learner.add_shell_sample({"command": f"cmd{i}", "exit_code": 0})
        # Add some keystroke data to establish baseline
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 200, "total_keystrokes": 100, "pause_count": 15})

        style = learner.compute_style()
        assert style.editing_frequency == EditingFrequency.MINIMAL


class TestAnalyzeClaudePatterns:
    """Tests for _analyze_claude_patterns method."""

    def test_high_rejection_rate(self):
        """High rejection rate affects editing frequency."""
        learner = StyleLearner()
        for i in range(30):
            signal = "rejection" if i < 20 else "approval"
            learner.add_claude_sample({"intention": "test", "preference_signal": signal})

        style = learner.compute_style()
        assert style.editing_frequency == EditingFrequency.HEAVY


class TestComputeStyle:
    """Tests for compute_style method."""

    def test_returns_communication_style(self):
        """Returns CommunicationStyle instance."""
        learner = StyleLearner()
        style = learner.compute_style()
        assert isinstance(style, CommunicationStyle)

    def test_sets_last_updated(self):
        """Sets last_updated timestamp."""
        learner = StyleLearner()
        style = learner.compute_style()
        assert style.last_updated is not None

    def test_calculates_confidence(self):
        """Calculates confidence from data volume."""
        learner = StyleLearner()
        # Add enough data
        for _ in range(50):
            learner.add_keystroke_sample({"typing_speed_cpm": 250})
            learner.add_window_sample({"focus_duration": 120})
            learner.add_afk_sample({"event": "afk_start"})

        style = learner.compute_style()
        assert style.confidence > 0.0

    def test_updates_stored_style(self):
        """Updates internal style reference."""
        learner = StyleLearner()
        for _ in range(25):
            learner.add_keystroke_sample({"typing_speed_cpm": 350})

        learner.compute_style()
        current = learner.get_current_style()
        assert current.typing_pace == TypingPace.FAST


class TestGetClaudeMdSection:
    """Tests for get_claude_md_section method."""

    def test_empty_with_low_confidence(self):
        """Returns empty string with low confidence."""
        learner = StyleLearner()
        section = learner.get_claude_md_section()
        assert section == ""

    def test_generates_markdown(self):
        """Generates markdown with enough data."""
        learner = StyleLearner()
        # Add data for high confidence
        for _ in range(100):
            learner.add_keystroke_sample({"typing_speed_cpm": 350})
            learner.add_window_sample({"focus_duration": 120})
            learner.add_afk_sample({"event": "afk_start"})

        learner.compute_style()
        section = learner.get_claude_md_section()

        assert "## Communication Style" in section
        assert "Confidence:" in section

    def test_includes_peak_productivity(self):
        """Includes peak productivity if available."""
        learner = StyleLearner()
        learner._session_hours = [14] * 200
        for _ in range(100):
            learner.add_keystroke_sample({"typing_speed_cpm": 350})
            learner.add_window_sample({"focus_duration": 120})
            learner.add_afk_sample({"event": "afk_start"})

        learner.compute_style()
        section = learner.get_claude_md_section()

        assert "14:00" in section


class TestReset:
    """Tests for reset method."""

    def test_clears_all_samples(self):
        """Clears all sample lists."""
        learner = StyleLearner()
        learner.add_keystroke_sample({"typing_speed_cpm": 250})
        learner.add_window_sample({"focus_duration": 120})
        learner.add_afk_sample({"event": "afk_start"})
        learner.add_shell_sample({"command": "ls"})
        learner.add_claude_sample({"intention": "test"})

        learner.reset()

        assert len(learner._keystroke_samples) == 0
        assert len(learner._window_samples) == 0
        assert len(learner._afk_samples) == 0
        assert len(learner._shell_samples) == 0
        assert len(learner._claude_samples) == 0
        assert len(learner._session_hours) == 0

    def test_resets_style(self):
        """Resets computed style."""
        learner = StyleLearner()
        for _ in range(50):
            learner.add_keystroke_sample({"typing_speed_cpm": 350})

        learner.compute_style()
        learner.reset()

        style = learner.get_current_style()
        assert style.confidence == 0.0


class TestGetStyleLearner:
    """Tests for get_style_learner singleton function."""

    def test_returns_same_instance(self):
        """Returns same instance on multiple calls."""
        learner1 = get_style_learner()
        learner2 = get_style_learner()
        assert learner1 is learner2
