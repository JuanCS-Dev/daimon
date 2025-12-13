"""
Tests for learners/preference_learner.py

Scientific tests covering:
- PreferenceSignal dataclass
- Pattern detection (approval/rejection)
- Category inference
- Session scanning
- Summary and insights generation

Updated for modular architecture (preference/ subpackage).
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import from new modular architecture
from learners.preference import (
    PreferenceSignal,
    SignalType,
    PreferenceCategory,
    CategoryStats,
    SessionScanner,
    SignalDetector,
    PreferenceCategorizer,
    InsightGenerator,
)
from learners.preference.detector import APPROVAL_PATTERNS, REJECTION_PATTERNS
from learners.preference.categorizer import CATEGORY_KEYWORDS
from learners.preference_learner import PreferenceLearner


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
        assert learner.scanner.projects_dir.name == "projects"
        assert learner.signals == []
        assert len(learner.categorizer.category_stats) == 0

    def test_init_with_custom_path(self, tmp_path: Path) -> None:
        """Test initialization with custom path."""
        learner = PreferenceLearner(projects_dir=tmp_path)
        assert learner.scanner.projects_dir == tmp_path


class TestSignalDetector:
    """Tests for SignalDetector._detect_heuristic()."""

    @pytest.fixture
    def detector(self) -> SignalDetector:
        """Create a SignalDetector instance."""
        return SignalDetector(enable_llm=False)

    def test_detect_approval_sim(self, detector: SignalDetector) -> None:
        """Test detecting 'sim' as approval."""
        assert detector._detect_heuristic("sim") == "approval"
        assert detector._detect_heuristic("Sim!") == "approval"

    def test_detect_approval_ok(self, detector: SignalDetector) -> None:
        """Test detecting 'ok' as approval."""
        assert detector._detect_heuristic("ok") == "approval"
        assert detector._detect_heuristic("OK") == "approval"

    def test_detect_approval_perfeito(self, detector: SignalDetector) -> None:
        """Test detecting 'perfeito' as approval."""
        assert detector._detect_heuristic("perfeito") == "approval"

    def test_detect_approval_yes(self, detector: SignalDetector) -> None:
        """Test detecting English approval."""
        assert detector._detect_heuristic("yes") == "approval"
        assert detector._detect_heuristic("great!") == "approval"

    def test_detect_rejection_nao(self, detector: SignalDetector) -> None:
        """Test detecting 'nao' as rejection."""
        assert detector._detect_heuristic("nao") == "rejection"
        assert detector._detect_heuristic("Nao!") == "rejection"

    def test_detect_rejection_errado(self, detector: SignalDetector) -> None:
        """Test detecting 'errado' as rejection."""
        assert detector._detect_heuristic("errado") == "rejection"

    def test_detect_rejection_english(self, detector: SignalDetector) -> None:
        """Test detecting English rejection."""
        assert detector._detect_heuristic("no") == "rejection"
        assert detector._detect_heuristic("wrong") == "rejection"

    def test_detect_neutral(self, detector: SignalDetector) -> None:
        """Test neutral content returns None."""
        assert detector._detect_heuristic("please continue") is None
        assert detector._detect_heuristic("show me the code") is None


class TestPreferenceCategorizer:
    """Tests for PreferenceCategorizer.infer_category()."""

    @pytest.fixture
    def categorizer(self) -> PreferenceCategorizer:
        """Create a PreferenceCategorizer instance."""
        return PreferenceCategorizer()

    def test_infer_code_style(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring code_style category."""
        assert categorizer.infer_category("fix the formatting") == "code_style"
        assert categorizer.infer_category("naming convention") == "code_style"

    def test_infer_testing(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring testing category."""
        assert categorizer.infer_category("add unit tests") == "testing"
        assert categorizer.infer_category("increase coverage") == "testing"

    def test_infer_verbosity(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring verbosity category."""
        assert categorizer.infer_category("too verbose") == "verbosity"
        assert categorizer.infer_category("mais resumo") == "verbosity"

    def test_infer_architecture(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring architecture category."""
        assert categorizer.infer_category("refactor this") == "architecture"
        assert categorizer.infer_category("design pattern") == "architecture"

    def test_infer_documentation(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring documentation category."""
        assert categorizer.infer_category("add docstring") == "documentation"
        assert categorizer.infer_category("update readme") == "documentation"

    def test_infer_workflow(self, categorizer: PreferenceCategorizer) -> None:
        """Test inferring workflow category."""
        assert categorizer.infer_category("commit this") == "workflow"
        assert categorizer.infer_category("git push") == "workflow"

    def test_infer_general(self, categorizer: PreferenceCategorizer) -> None:
        """Test falling back to general category."""
        assert categorizer.infer_category("random text") == "general"
        assert categorizer.infer_category("hello world") == "general"


class TestCalculateStrength:
    """Tests for PreferenceCategorizer.calculate_strength()."""

    @pytest.fixture
    def categorizer(self) -> PreferenceCategorizer:
        """Create a PreferenceCategorizer instance."""
        return PreferenceCategorizer()

    def test_short_response_high_strength(self, categorizer: PreferenceCategorizer) -> None:
        """Test short responses have high strength."""
        assert categorizer.calculate_strength("ok") == 0.9
        assert categorizer.calculate_strength("sim") == 0.9
        assert categorizer.calculate_strength("yes please") == 0.9

    def test_medium_response(self, categorizer: PreferenceCategorizer) -> None:
        """Test medium responses have medium strength."""
        assert categorizer.calculate_strength("yes that looks good to me") == 0.7

    def test_long_response(self, categorizer: PreferenceCategorizer) -> None:
        """Test long responses have lower strength."""
        long_text = " ".join(["word"] * 50)
        strength = categorizer.calculate_strength(long_text)
        assert strength < 0.5


class TestExtractUserContent:
    """Tests for SignalDetector.extract_user_content()."""

    @pytest.fixture
    def detector(self) -> SignalDetector:
        """Create a SignalDetector instance."""
        return SignalDetector(enable_llm=False)

    def test_extract_string_content(self, detector: SignalDetector) -> None:
        """Test extracting string content."""
        msg = {"message": {"content": "simple text"}}
        assert detector.extract_user_content(msg) == "simple text"

    def test_extract_list_content_text(self, detector: SignalDetector) -> None:
        """Test extracting from list with text item."""
        msg = {"message": {"content": [{"type": "text", "text": "hello"}]}}
        assert detector.extract_user_content(msg) == "hello"

    def test_extract_list_content_multiple(self, detector: SignalDetector) -> None:
        """Test extracting from list with multiple items."""
        msg = {
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ]
            }
        }
        assert detector.extract_user_content(msg) == "hello world"

    def test_extract_empty(self, detector: SignalDetector) -> None:
        """Test extracting from empty content."""
        msg = {"message": {"content": ""}}
        assert detector.extract_user_content(msg) == ""

    def test_extract_missing_message(self, detector: SignalDetector) -> None:
        """Test extracting from missing message."""
        msg = {}
        assert detector.extract_user_content(msg) == ""


class TestCategoryStats:
    """Tests for CategoryStats update and stats."""

    def test_update_approval_counts(self) -> None:
        """Test updating approval counts."""
        categorizer = PreferenceCategorizer()
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="approval",
            context="ctx",
            category="testing",
            strength=0.8,
            session_id="sess",
        )
        categorizer.update_stats(signal)
        assert categorizer.category_stats["testing"].approvals == 1
        assert categorizer.category_stats["testing"].rejections == 0

    def test_update_rejection_counts(self) -> None:
        """Test updating rejection counts."""
        categorizer = PreferenceCategorizer()
        signal = PreferenceSignal(
            timestamp="2025-01-01",
            signal_type="rejection",
            context="ctx",
            category="verbosity",
            strength=0.8,
            session_id="sess",
        )
        categorizer.update_stats(signal)
        assert categorizer.category_stats["verbosity"].rejections == 1
        assert categorizer.category_stats["verbosity"].approvals == 0


class TestGetPreferenceSummary:
    """Tests for InsightGenerator.get_summary()."""

    def test_summary_empty(self) -> None:
        """Test summary with no data."""
        generator = InsightGenerator(enable_llm=False)
        summary = generator.get_summary({})
        assert summary == {}

    def test_summary_with_data(self) -> None:
        """Test summary with data."""
        generator = InsightGenerator(enable_llm=False)
        stats = {"testing": CategoryStats(approvals=8, rejections=2)}
        
        summary = generator.get_summary(stats)
        assert "testing" in summary
        assert summary["testing"]["approval_rate"] == 0.8
        assert summary["testing"]["total_signals"] == 10
        assert summary["testing"]["trend"] == "positive"

    def test_summary_negative_trend(self) -> None:
        """Test summary with negative trend."""
        generator = InsightGenerator(enable_llm=False)
        stats = {"verbosity": CategoryStats(approvals=2, rejections=8)}
        
        summary = generator.get_summary(stats)
        assert summary["verbosity"]["approval_rate"] == 0.2
        assert summary["verbosity"]["trend"] == "negative"


class TestGetActionableInsights:
    """Tests for InsightGenerator.get_insights()."""

    def test_insights_empty(self) -> None:
        """Test insights with no data."""
        generator = InsightGenerator(enable_llm=False)
        insights = generator.get_insights({})
        assert insights == []

    def test_insights_too_few_signals(self) -> None:
        """Test insights with too few signals."""
        generator = InsightGenerator(enable_llm=False)
        stats = {"testing": CategoryStats(approvals=1, rejections=1)}
        
        insights = generator.get_insights(stats, min_signals=3)
        assert insights == []

    def test_insights_high_rejection(self) -> None:
        """Test insights with high rejection rate."""
        generator = InsightGenerator(enable_llm=False)
        stats = {"verbosity": CategoryStats(approvals=1, rejections=9)}
        
        insights = generator.get_insights(stats, min_signals=3)
        assert len(insights) == 1
        assert insights[0].category == "verbosity"
        assert insights[0].action == "reduce"
        assert insights[0].suggestion

    def test_insights_high_approval(self) -> None:
        """Test insights with high approval rate."""
        generator = InsightGenerator(enable_llm=False)
        stats = {"testing": CategoryStats(approvals=9, rejections=1)}
        
        insights = generator.get_insights(stats, min_signals=3)
        assert len(insights) == 1
        assert insights[0].category == "testing"
        assert insights[0].action == "reinforce"


class TestSuggestions:
    """Tests for InsightGenerator._get_suggestion()."""

    @pytest.fixture
    def generator(self) -> InsightGenerator:
        """Create an InsightGenerator instance."""
        return InsightGenerator(enable_llm=False)

    def test_suggestion_verbosity_reduce(self, generator: InsightGenerator) -> None:
        """Test suggestion for reducing verbosity."""
        suggestion = generator._get_suggestion("verbosity", "reduce")
        assert "concis" in suggestion.lower()

    def test_suggestion_testing_reinforce(self, generator: InsightGenerator) -> None:
        """Test suggestion for reinforcing testing."""
        suggestion = generator._get_suggestion("testing", "reinforce")
        assert "test" in suggestion.lower()

    def test_suggestion_unknown(self, generator: InsightGenerator) -> None:
        """Test suggestion for unknown category."""
        suggestion = generator._get_suggestion("unknown_category", "reduce")
        assert "unknown_category" in suggestion


class TestClearAndStats:
    """Tests for PreferenceLearner.clear() and get_stats()."""

    def test_clear(self) -> None:
        """Test clearing learner state."""
        learner = PreferenceLearner()
        learner.categorizer.category_stats["testing"] = CategoryStats(approvals=5)
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
        assert len(learner.categorizer.category_stats) == 0

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
        learner.categorizer.category_stats["testing"] = CategoryStats(approvals=3, rejections=1)
        learner.categorizer.category_stats["code_style"] = CategoryStats(approvals=2, rejections=0)

        stats = learner.get_stats()
        assert stats["total_signals"] == 6
        assert stats["total_approvals"] == 5
        assert stats["total_rejections"] == 1
        assert stats["categories_analyzed"] == 2


class TestSessionScanner:
    """Tests for SessionScanner."""

    def test_scan_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test scanning non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        scanner = SessionScanner(projects_dir=nonexistent)
        sessions = list(scanner.scan_recent(since_hours=24))
        assert sessions == []

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning empty directory."""
        scanner = SessionScanner(projects_dir=tmp_path)
        sessions = list(scanner.scan_recent(since_hours=24))
        assert sessions == []

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        """Test loading valid JSONL file."""
        session_file = tmp_path / "session.jsonl"
        messages = [
            {"type": "user", "message": {"content": "hello"}},
            {"type": "assistant", "message": {"content": "hi"}},
        ]
        with open(session_file, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        scanner = SessionScanner()
        loaded = scanner.load_messages(session_file)
        assert len(loaded) == 2

    def test_load_handles_invalid_json_lines(self, tmp_path: Path) -> None:
        """Test that invalid JSON lines are skipped."""
        session_file = tmp_path / "session.jsonl"
        with open(session_file, "w") as f:
            f.write('{"valid": true}\n')
            f.write("invalid json line\n")
            f.write('{"also": "valid"}\n')

        scanner = SessionScanner()
        loaded = scanner.load_messages(session_file)
        assert len(loaded) == 2

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file returns empty list."""
        scanner = SessionScanner()
        result = scanner.load_messages(Path("/nonexistent/file.jsonl"))
        assert result == []

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

        scanner = SessionScanner(projects_dir=tmp_path)
        sessions = list(scanner.scan_recent(since_hours=24))
        assert len(sessions) == 0


class TestSignalDetectorExtraction:
    """Tests for SignalDetector context extraction."""

    @pytest.fixture
    def detector(self) -> SignalDetector:
        """Create a SignalDetector instance."""
        return SignalDetector(enable_llm=False)

    def test_extract_assistant_context(self, detector: SignalDetector) -> None:
        """Test extracting assistant context."""
        msg = {"message": {"content": "Here is the refactored code"}}
        context = detector._extract_assistant_context(msg)
        assert context == "Here is the refactored code"

    def test_extract_list_context(self, detector: SignalDetector) -> None:
        """Test extracting from list context."""
        msg = {"message": {"content": [{"type": "text", "text": "Result"}]}}
        context = detector._extract_assistant_context(msg)
        assert context == "Result"

    def test_extract_tool_name(self, detector: SignalDetector) -> None:
        """Test extracting tool name."""
        msg = {
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Edit"},
                    {"type": "text", "text": "Updated the file"},
                ]
            }
        }
        tool_name = detector._extract_tool_name(msg)
        assert tool_name == "Edit"

    def test_no_tool_name(self, detector: SignalDetector) -> None:
        """Test when no tool used."""
        msg = {"message": {"content": [{"type": "text", "text": "Just text"}]}}
        tool_name = detector._extract_tool_name(msg)
        assert tool_name is None


class TestToolResultDetection:
    """Tests for SignalDetector._check_tool_result()."""

    @pytest.fixture
    def detector(self) -> SignalDetector:
        """Create a SignalDetector instance."""
        return SignalDetector(enable_llm=False)

    def test_failed_tool_result(self, detector: SignalDetector) -> None:
        """Test detecting failed tool result."""
        msg = {"toolUseResult": {"status": "failed"}}
        signal = detector._check_tool_result(msg)
        assert signal == "rejection"

    def test_interrupted_tool_result(self, detector: SignalDetector) -> None:
        """Test detecting interrupted tool result."""
        msg = {"toolUseResult": {"interrupted": True}}
        signal = detector._check_tool_result(msg)
        assert signal == "rejection"

    def test_success_tool_result(self, detector: SignalDetector) -> None:
        """Test success tool result returns None."""
        msg = {"toolUseResult": {"status": "success"}}
        signal = detector._check_tool_result(msg)
        assert signal is None


class TestPreferenceLearnerIntegration:
    """Integration tests for PreferenceLearner."""

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
