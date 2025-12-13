"""
Tests for learners/preference_learner.py

Scientific tests covering:
- PreferenceSignal dataclass
- Pattern detection (approval/rejection)
- Category inference
- Session scanning
- Summary and insights generation
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from learners.preference_learner import (
    APPROVAL_PATTERNS,
    CATEGORY_KEYWORDS,
    REJECTION_PATTERNS,
    PreferenceLearner,
    PreferenceSignal,
)


class TestPreferenceSignal:
    """Tests for PreferenceSignal dataclass."""

    def test_create_signal(self) -> None:
        """Test creating a PreferenceSignal."""
        signal = PreferenceSignal(
            timestamp="2025-01-01T00:00:00",
            signal_type="approval",
            context="User asked for refactoring",
            category="code_style",
            strength=0.9,
            session_id="abc123",
            tool_involved="Edit",
        )
        assert signal.timestamp == "2025-01-01T00:00:00"
        assert signal.signal_type == "approval"
        assert signal.context == "User asked for refactoring"
        assert signal.category == "code_style"
        assert signal.strength == 0.9
        assert signal.session_id == "abc123"
        assert signal.tool_involved == "Edit"

    def test_signal_with_no_tool(self) -> None:
        """Test signal without tool involved."""
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="rejection",
            context="ctx",
            category="general",
            strength=0.5,
            session_id="sess",
        )
        assert signal.tool_involved is None

    def test_to_dict(self) -> None:
        """Test converting signal to dictionary."""
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="approval",
            context="ctx",
            category="testing",
            strength=0.8,
            session_id="sess",
            tool_involved="Bash",
        )
        d = signal.to_dict()
        assert d["timestamp"] == "2025-01-01"
        assert d["signal_type"] == "approval"
        assert d["category"] == "testing"
        assert d["tool_involved"] == "Bash"


class TestPatterns:
    """Tests for approval/rejection pattern constants."""

    def test_approval_patterns_exist(self) -> None:
        """Test approval patterns are defined."""
        assert len(APPROVAL_PATTERNS) > 0
        assert any("sim" in p for p in APPROVAL_PATTERNS)
        assert any("ok" in p for p in APPROVAL_PATTERNS)

    def test_rejection_patterns_exist(self) -> None:
        """Test rejection patterns are defined."""
        assert len(REJECTION_PATTERNS) > 0
        assert any("nao" in p or "não" in p for p in REJECTION_PATTERNS)
        assert any("errado" in p for p in REJECTION_PATTERNS)

    def test_category_keywords_coverage(self) -> None:
        """Test category keywords cover important areas."""
        assert "code_style" in CATEGORY_KEYWORDS
        assert "verbosity" in CATEGORY_KEYWORDS
        assert "testing" in CATEGORY_KEYWORDS
        assert "architecture" in CATEGORY_KEYWORDS
        assert "documentation" in CATEGORY_KEYWORDS


class TestPreferenceLearnerInit:
    """Tests for PreferenceLearner initialization."""

    def test_init_with_default_path(self) -> None:
        """Test initialization with default path."""
        learner = PreferenceLearner()
        assert learner.projects_dir.name == "projects"
        assert learner.signals == []
        assert len(learner.category_counts) == 0

    def test_init_with_custom_path(self, tmp_path: Path) -> None:
        """Test initialization with custom path."""
        learner = PreferenceLearner(projects_dir=tmp_path)
        assert learner.projects_dir == tmp_path


class TestDetectSignalType:
    """Tests for PreferenceLearner._detect_signal_type()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_detect_approval_sim(self, learner: PreferenceLearner) -> None:
        """Test detecting 'sim' as approval."""
        assert learner._detect_signal_type("sim") == "approval"
        assert learner._detect_signal_type("Sim!") == "approval"

    def test_detect_approval_ok(self, learner: PreferenceLearner) -> None:
        """Test detecting 'ok' as approval."""
        assert learner._detect_signal_type("ok") == "approval"
        assert learner._detect_signal_type("OK") == "approval"

    def test_detect_approval_perfeito(self, learner: PreferenceLearner) -> None:
        """Test detecting 'perfeito' as approval."""
        assert learner._detect_signal_type("perfeito") == "approval"

    def test_detect_approval_yes(self, learner: PreferenceLearner) -> None:
        """Test detecting English approval."""
        assert learner._detect_signal_type("yes") == "approval"
        assert learner._detect_signal_type("great!") == "approval"

    def test_detect_rejection_nao(self, learner: PreferenceLearner) -> None:
        """Test detecting 'nao' as rejection."""
        assert learner._detect_signal_type("nao") == "rejection"
        assert learner._detect_signal_type("Nao!") == "rejection"

    def test_detect_rejection_errado(self, learner: PreferenceLearner) -> None:
        """Test detecting 'errado' as rejection."""
        assert learner._detect_signal_type("errado") == "rejection"

    def test_detect_rejection_english(self, learner: PreferenceLearner) -> None:
        """Test detecting English rejection."""
        assert learner._detect_signal_type("no") == "rejection"
        assert learner._detect_signal_type("wrong") == "rejection"

    def test_detect_neutral(self, learner: PreferenceLearner) -> None:
        """Test neutral content returns None."""
        assert learner._detect_signal_type("please continue") is None
        assert learner._detect_signal_type("show me the code") is None


class TestInferCategory:
    """Tests for PreferenceLearner._infer_category()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_infer_code_style(self, learner: PreferenceLearner) -> None:
        """Test inferring code_style category."""
        assert learner._infer_category("fix the formatting") == "code_style"
        assert learner._infer_category("naming convention") == "code_style"

    def test_infer_testing(self, learner: PreferenceLearner) -> None:
        """Test inferring testing category."""
        assert learner._infer_category("add unit tests") == "testing"
        assert learner._infer_category("increase coverage") == "testing"

    def test_infer_verbosity(self, learner: PreferenceLearner) -> None:
        """Test inferring verbosity category."""
        assert learner._infer_category("too verbose") == "verbosity"
        assert learner._infer_category("mais resumo") == "verbosity"

    def test_infer_architecture(self, learner: PreferenceLearner) -> None:
        """Test inferring architecture category."""
        assert learner._infer_category("refactor this") == "architecture"
        assert learner._infer_category("design pattern") == "architecture"

    def test_infer_documentation(self, learner: PreferenceLearner) -> None:
        """Test inferring documentation category."""
        assert learner._infer_category("add docstring") == "documentation"
        assert learner._infer_category("update readme") == "documentation"

    def test_infer_workflow(self, learner: PreferenceLearner) -> None:
        """Test inferring workflow category."""
        assert learner._infer_category("commit this") == "workflow"
        assert learner._infer_category("git push") == "workflow"

    def test_infer_general(self, learner: PreferenceLearner) -> None:
        """Test falling back to general category."""
        assert learner._infer_category("random text") == "general"
        assert learner._infer_category("hello world") == "general"


class TestCalculateStrength:
    """Tests for PreferenceLearner._calculate_strength()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_short_response_high_strength(self, learner: PreferenceLearner) -> None:
        """Test short responses have high strength."""
        assert learner._calculate_strength("ok") == 0.9
        assert learner._calculate_strength("sim") == 0.9
        assert learner._calculate_strength("yes please") == 0.9

    def test_medium_response(self, learner: PreferenceLearner) -> None:
        """Test medium responses have medium strength."""
        assert learner._calculate_strength("yes that looks good to me") == 0.7

    def test_long_response(self, learner: PreferenceLearner) -> None:
        """Test long responses have lower strength."""
        long_text = " ".join(["word"] * 50)
        assert learner._calculate_strength(long_text) == 0.3


class TestExtractUserContent:
    """Tests for PreferenceLearner._extract_user_content()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_extract_string_content(self, learner: PreferenceLearner) -> None:
        """Test extracting string content."""
        msg = {"message": {"content": "simple text"}}
        assert learner._extract_user_content(msg) == "simple text"

    def test_extract_list_content_text(self, learner: PreferenceLearner) -> None:
        """Test extracting from list with text item."""
        msg = {"message": {"content": [{"type": "text", "text": "hello"}]}}
        assert learner._extract_user_content(msg) == "hello"

    def test_extract_list_content_multiple(self, learner: PreferenceLearner) -> None:
        """Test extracting from list with multiple items."""
        msg = {
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ]
            }
        }
        assert learner._extract_user_content(msg) == "hello world"

    def test_extract_empty(self, learner: PreferenceLearner) -> None:
        """Test extracting from empty content."""
        msg = {"message": {"content": ""}}
        assert learner._extract_user_content(msg) == ""

    def test_extract_missing_message(self, learner: PreferenceLearner) -> None:
        """Test extracting from missing message."""
        msg = {}
        assert learner._extract_user_content(msg) == ""


class TestUpdateCounts:
    """Tests for PreferenceLearner._update_counts()."""

    def test_update_approval_counts(self) -> None:
        """Test updating approval counts."""
        learner = PreferenceLearner()
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="approval",
            context="ctx",
            category="testing",
            strength=0.8,
            session_id="sess",
        )
        learner._update_counts(signal)
        assert learner.category_counts["testing"]["approvals"] == 1
        assert learner.category_counts["testing"]["rejections"] == 0

    def test_update_rejection_counts(self) -> None:
        """Test updating rejection counts."""
        learner = PreferenceLearner()
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="rejection",
            context="ctx",
            category="verbosity",
            strength=0.8,
            session_id="sess",
        )
        learner._update_counts(signal)
        assert learner.category_counts["verbosity"]["rejections"] == 1
        assert learner.category_counts["verbosity"]["approvals"] == 0


class TestGetPreferenceSummary:
    """Tests for PreferenceLearner.get_preference_summary()."""

    def test_summary_empty(self) -> None:
        """Test summary with no data."""
        learner = PreferenceLearner()
        summary = learner.get_preference_summary()
        assert summary == {}

    def test_summary_with_data(self) -> None:
        """Test summary with data."""
        learner = PreferenceLearner()
        learner.category_counts["testing"]["approvals"] = 8
        learner.category_counts["testing"]["rejections"] = 2

        summary = learner.get_preference_summary()
        assert "testing" in summary
        assert summary["testing"]["approval_rate"] == 0.8
        assert summary["testing"]["total_signals"] == 10
        assert summary["testing"]["trend"] == "positive"

    def test_summary_negative_trend(self) -> None:
        """Test summary with negative trend."""
        learner = PreferenceLearner()
        learner.category_counts["verbosity"]["approvals"] = 2
        learner.category_counts["verbosity"]["rejections"] = 8

        summary = learner.get_preference_summary()
        assert summary["verbosity"]["approval_rate"] == 0.2
        assert summary["verbosity"]["trend"] == "negative"


class TestGetActionableInsights:
    """Tests for PreferenceLearner.get_actionable_insights()."""

    def test_insights_empty(self) -> None:
        """Test insights with no data."""
        learner = PreferenceLearner()
        insights = learner.get_actionable_insights()
        assert insights == []

    def test_insights_too_few_signals(self) -> None:
        """Test insights with too few signals."""
        learner = PreferenceLearner()
        learner.category_counts["testing"]["approvals"] = 1
        learner.category_counts["testing"]["rejections"] = 1

        insights = learner.get_actionable_insights(min_signals=3)
        assert insights == []

    def test_insights_high_rejection(self) -> None:
        """Test insights with high rejection rate."""
        learner = PreferenceLearner()
        learner.category_counts["verbosity"]["approvals"] = 1
        learner.category_counts["verbosity"]["rejections"] = 9

        insights = learner.get_actionable_insights(min_signals=3)
        assert len(insights) == 1
        assert insights[0]["category"] == "verbosity"
        assert insights[0]["action"] == "reduce"
        assert "suggestion" in insights[0]

    def test_insights_high_approval(self) -> None:
        """Test insights with high approval rate."""
        learner = PreferenceLearner()
        learner.category_counts["testing"]["approvals"] = 9
        learner.category_counts["testing"]["rejections"] = 1

        insights = learner.get_actionable_insights(min_signals=3)
        assert len(insights) == 1
        assert insights[0]["category"] == "testing"
        assert insights[0]["action"] == "reinforce"


class TestGenerateSuggestion:
    """Tests for PreferenceLearner._generate_suggestion()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_suggestion_verbosity_reduce(self, learner: PreferenceLearner) -> None:
        """Test suggestion for reducing verbosity."""
        suggestion = learner._generate_suggestion("verbosity", "reduce")
        assert "concis" in suggestion.lower()

    def test_suggestion_testing_reinforce(self, learner: PreferenceLearner) -> None:
        """Test suggestion for reinforcing testing."""
        suggestion = learner._generate_suggestion("testing", "reinforce")
        assert "test" in suggestion.lower()

    def test_suggestion_unknown(self, learner: PreferenceLearner) -> None:
        """Test suggestion for unknown category."""
        suggestion = learner._generate_suggestion("unknown_category", "reduce")
        assert "unknown_category" in suggestion


class TestClearAndStats:
    """Tests for PreferenceLearner.clear() and get_stats()."""

    def test_clear(self) -> None:
        """Test clearing learner state."""
        learner = PreferenceLearner()
        learner.category_counts["testing"]["approvals"] = 5
        learner.signals.append(
            PreferenceSignal(
                timestamp="2025-01-01",
                signal_type="approval",
                context="ctx",
                category="testing",
                strength=0.8,
                session_id="sess",
            )
        )

        learner.clear()
        assert len(learner.signals) == 0
        assert len(learner.category_counts) == 0

    def test_get_stats_empty(self) -> None:
        """Test stats on empty learner."""
        learner = PreferenceLearner()
        stats = learner.get_stats()
        assert stats["total_signals"] == 0
        assert stats["overall_approval_rate"] == 0
        assert stats["categories_analyzed"] == 0

    def test_get_stats_with_data(self) -> None:
        """Test stats with data."""
        learner = PreferenceLearner()
        learner.category_counts["testing"]["approvals"] = 3
        learner.category_counts["testing"]["rejections"] = 1
        learner.category_counts["code_style"]["approvals"] = 2
        learner.category_counts["code_style"]["rejections"] = 0

        stats = learner.get_stats()
        assert stats["total_signals"] == 6
        assert stats["total_approvals"] == 5
        assert stats["total_rejections"] == 1
        assert stats["categories_analyzed"] == 2


class TestScanSessions:
    """Tests for PreferenceLearner.scan_sessions()."""

    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test scanning non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        learner = PreferenceLearner(projects_dir=nonexistent)
        signals = learner.scan_sessions(since_hours=24)
        assert signals == []

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning empty directory."""
        learner = PreferenceLearner(projects_dir=tmp_path)
        signals = learner.scan_sessions(since_hours=24)
        assert signals == []

    def test_scan_with_session_file(self, tmp_path: Path) -> None:
        """Test scanning directory with session file."""
        # Create project directory
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create session file with approval
        session_file = project_dir / "session123.jsonl"
        messages = [
            {"type": "assistant", "message": {"content": "Here is the code"}},
            {"type": "user", "message": {"content": "ok perfeito"}},
        ]
        with open(session_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        learner = PreferenceLearner(projects_dir=tmp_path)
        signals = learner.scan_sessions(since_hours=24)
        assert len(signals) >= 1
        assert signals[0].signal_type == "approval"

    def test_scan_skips_agent_files(self, tmp_path: Path) -> None:
        """Test scanning skips agent-* files."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create agent file (should be skipped)
        agent_file = project_dir / "agent-abc123.jsonl"
        messages = [
            {"type": "user", "message": {"content": "sim"}},
        ]
        with open(agent_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        learner = PreferenceLearner(projects_dir=tmp_path)
        signals = learner.scan_sessions(since_hours=24)
        assert len(signals) == 0  # Agent files skipped


class TestExtractAssistantContext:
    """Tests for PreferenceLearner._extract_assistant_context()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_extract_string_context(self, learner: PreferenceLearner) -> None:
        """Test extracting string context."""
        msg = {"message": {"content": "Here is the refactored code"}}
        context = learner._extract_assistant_context(msg)
        assert context == "Here is the refactored code"

    def test_extract_list_context(self, learner: PreferenceLearner) -> None:
        """Test extracting from list context."""
        msg = {"message": {"content": [{"type": "text", "text": "Result"}]}}
        context = learner._extract_assistant_context(msg)
        assert context == "Result"


class TestExtractToolName:
    """Tests for PreferenceLearner._extract_tool_name()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_extract_tool_name(self, learner: PreferenceLearner) -> None:
        """Test extracting tool name."""
        msg = {
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Edit"},
                    {"type": "text", "text": "Updated the file"},
                ]
            }
        }
        tool_name = learner._extract_tool_name(msg)
        assert tool_name == "Edit"

    def test_no_tool_name(self, learner: PreferenceLearner) -> None:
        """Test when no tool used."""
        msg = {"message": {"content": [{"type": "text", "text": "Just text"}]}}
        tool_name = learner._extract_tool_name(msg)
        assert tool_name is None


class TestLoadSessionMessages:
    """Tests for PreferenceLearner._load_session_messages()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_load_valid_jsonl(self, learner: PreferenceLearner, tmp_path: Path) -> None:
        """Test loading valid JSONL file."""
        session_file = tmp_path / "session.jsonl"
        messages = [
            {"type": "user", "message": {"content": "hello"}},
            {"type": "assistant", "message": {"content": "hi"}},
        ]
        with open(session_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        loaded = learner._load_session_messages(session_file)
        assert len(loaded) == 2

    def test_load_handles_invalid_json_lines(
        self, learner: PreferenceLearner, tmp_path: Path
    ) -> None:
        """Test that invalid JSON lines are skipped."""
        session_file = tmp_path / "session.jsonl"
        with open(session_file, "w") as f:
            f.write('{"valid": true}\n')
            f.write("invalid json line\n")
            f.write('{"also": "valid"}\n')

        loaded = learner._load_session_messages(session_file)
        assert len(loaded) == 2

    def test_load_nonexistent_file(self, learner: PreferenceLearner) -> None:
        """Test loading nonexistent file returns empty list."""
        result = learner._load_session_messages(Path("/nonexistent/file.jsonl"))
        assert result == []


class TestGetSignalFromToolResult:
    """Tests for PreferenceLearner._get_signal_from_tool_result()."""

    @pytest.fixture
    def learner(self) -> PreferenceLearner:
        """Create a PreferenceLearner instance."""
        return PreferenceLearner()

    def test_failed_tool_result(self, learner: PreferenceLearner) -> None:
        """Test detecting failed tool result."""
        msg = {"toolUseResult": {"status": "failed"}}
        signal = learner._get_signal_from_tool_result(msg)
        assert signal == "rejection"

    def test_interrupted_tool_result(self, learner: PreferenceLearner) -> None:
        """Test detecting interrupted tool result."""
        msg = {"toolUseResult": {"interrupted": True}}
        signal = learner._get_signal_from_tool_result(msg)
        assert signal == "rejection"

    def test_success_tool_result(self, learner: PreferenceLearner) -> None:
        """Test success tool result returns None."""
        msg = {"toolUseResult": {"status": "success"}}
        signal = learner._get_signal_from_tool_result(msg)
        assert signal is None
