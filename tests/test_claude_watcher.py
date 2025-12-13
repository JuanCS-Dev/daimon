"""
DAIMON Claude Watcher Tests
===========================

Unit tests for the Claude Code watcher module.

Follows CODE_CONSTITUTION: Measurable Quality.
"""

from __future__ import annotations

import pytest

from collectors.claude_watcher import (
    INTENT_PATTERNS,
    detect_intention,
    extract_files_touched,
    SessionTracker,
)


class TestDetectIntention:
    """Test suite for intention detection."""

    def test_detect_create_intention(self) -> None:
        """Test detecting create intention."""
        assert detect_intention("Create a new component") == "create"
        assert detect_intention("Add authentication") == "create"
        assert detect_intention("implement the feature") == "create"

    def test_detect_fix_intention(self) -> None:
        """Test detecting fix intention."""
        assert detect_intention("Fix the bug") == "fix"
        assert detect_intention("There's an error here") == "fix"
        assert detect_intention("This is broken") == "fix"

    def test_detect_refactor_intention(self) -> None:
        """Test detecting refactor intention."""
        assert detect_intention("Refactor this module") == "refactor"
        assert detect_intention("Clean up the code") == "refactor"
        assert detect_intention("Reorganize the structure") == "refactor"

    def test_detect_understand_intention(self) -> None:
        """Test detecting understand intention."""
        assert detect_intention("Explain this function") == "understand"
        assert detect_intention("What does this do?") == "understand"
        assert detect_intention("How does this work?") == "understand"
        assert detect_intention("Help me with this") == "understand"

    def test_detect_delete_intention(self) -> None:
        """Test detecting delete intention."""
        assert detect_intention("Delete the old files") == "delete"
        assert detect_intention("Remove this function") == "delete"
        assert detect_intention("Drop the table") == "delete"

    def test_detect_test_intention(self) -> None:
        """Test detecting test intention."""
        assert detect_intention("Test this endpoint") == "test"
        assert detect_intention("Verify the output") == "test"
        assert detect_intention("Check the coverage") == "test"

    def test_detect_deploy_intention(self) -> None:
        """Test detecting deploy intention."""
        assert detect_intention("Deploy to production") == "deploy"
        assert detect_intention("Release the new version") == "deploy"
        assert detect_intention("Ship this feature") == "deploy"

    def test_detect_unknown_intention(self) -> None:
        """Test unknown intention fallback."""
        assert detect_intention("Let's do something") == "unknown"
        assert detect_intention("Random text here") == "unknown"

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert detect_intention("CREATE A NEW FILE") == "create"
        assert detect_intention("FIX THE BUG") == "fix"

    def test_intent_patterns_exist(self) -> None:
        """Test that all expected intent patterns exist."""
        expected_intents = ["create", "fix", "refactor", "understand", "delete", "test", "deploy"]
        for intent in expected_intents:
            assert intent in INTENT_PATTERNS


class TestExtractFilesTouched:
    """Test suite for file path extraction."""

    def test_extract_quoted_paths(self) -> None:
        """Test extracting quoted file paths."""
        files = extract_files_touched('Edit the file "src/main.py"')
        assert "src/main.py" in files

    def test_extract_unquoted_paths(self) -> None:
        """Test extracting unquoted file paths."""
        files = extract_files_touched("Check components/App.tsx for errors")
        assert "components/App.tsx" in files

    def test_extract_multiple_paths(self) -> None:
        """Test extracting multiple file paths."""
        files = extract_files_touched("Compare config.json and settings.yaml")
        assert any("config.json" in f for f in files)
        assert any("settings.yaml" in f for f in files)

    def test_limit_extracted_files(self) -> None:
        """Test that extracted files are limited to 10."""
        # Create message with many file references
        many_files = " ".join([f"file{i}.py" for i in range(20)])
        files = extract_files_touched(many_files)
        assert len(files) <= 10

    def test_no_files_found(self) -> None:
        """Test when no files are found."""
        files = extract_files_touched("Just regular text without files")
        assert len(files) == 0

    def test_deduplicate_files(self) -> None:
        """Test that duplicate files are removed."""
        files = extract_files_touched("Edit main.py and then main.py again")
        # Count occurrences of main.py
        main_count = sum(1 for f in files if "main.py" in f)
        assert main_count == 1


class TestSessionTracker:
    """Test suite for SessionTracker class."""

    def test_init(self) -> None:
        """Test SessionTracker initialization."""
        tracker = SessionTracker()
        assert tracker.positions == {}
        assert tracker.session_events == []

    @pytest.mark.asyncio
    async def test_scan_projects_no_directory(self) -> None:
        """Test scanning when claude directory doesn't exist."""
        tracker = SessionTracker()
        # Should not raise, just return
        await tracker.scan_projects()
        # No events should be captured
        assert tracker.session_events == []

    @pytest.mark.asyncio
    async def test_send_event_real(self) -> None:
        """Test sending event to real NOESIS."""
        tracker = SessionTracker()
        event = {
            "event_type": "test",
            "timestamp": "2025-12-12T12:00:00",
            "project": "test",
            "files_touched": [],
            "intention": "test",
        }
        # Should not raise even if endpoint exists or not
        await tracker._send_event(event)

    @pytest.mark.asyncio
    async def test_process_entry_non_user(self) -> None:
        """Test processing non-user entry is skipped."""
        tracker = SessionTracker()
        entry = {"role": "assistant", "content": "Hello"}
        await tracker._process_entry(entry, "test-project")
        assert len(tracker.session_events) == 0

    @pytest.mark.asyncio
    async def test_process_entry_user_message(self) -> None:
        """Test processing user message extracts intention."""
        tracker = SessionTracker()
        entry = {
            "role": "user",
            "content": "Create a new component for authentication"
        }
        await tracker._process_entry(entry, "test-project")
        assert len(tracker.session_events) == 1
        assert tracker.session_events[0]["intention"] == "create"

    @pytest.mark.asyncio
    async def test_process_entry_empty_content(self) -> None:
        """Test processing empty content is skipped."""
        tracker = SessionTracker()
        entry = {"role": "user", "content": ""}
        await tracker._process_entry(entry, "test-project")
        assert len(tracker.session_events) == 0


class TestScanProjects:
    """Tests for project scanning."""

    @pytest.mark.asyncio
    async def test_scan_real_claude_directory(self) -> None:
        """Test scanning real .claude directory if it exists."""
        from pathlib import Path
        tracker = SessionTracker()
        claude_dir = Path.home() / ".claude" / "projects"

        if claude_dir.exists():
            await tracker.scan_projects()
            # Should not raise, events may or may not be captured
            assert isinstance(tracker.session_events, list)

    @pytest.mark.asyncio
    async def test_scan_nonexistent_directory(self) -> None:
        """Test scanning when CLAUDE_DIR doesn't exist (lines 120-121)."""
        import tempfile
        from pathlib import Path
        import collectors.claude_watcher as cw

        original_dir = cw.CLAUDE_DIR
        cw.CLAUDE_DIR = Path("/nonexistent/path/that/doesnt/exist")

        try:
            tracker = SessionTracker()
            await tracker.scan_projects()
            # Should return early, no events
            assert len(tracker.session_events) == 0
        finally:
            cw.CLAUDE_DIR = original_dir

    @pytest.mark.asyncio
    async def test_scan_with_file_instead_of_directory(self) -> None:
        """Test scanning when project is a file, not directory (line 125)."""
        import tempfile
        from pathlib import Path
        import collectors.claude_watcher as cw

        original_dir = cw.CLAUDE_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir)
            # Create a FILE instead of directory
            (projects_dir / "not-a-directory").write_text("I'm a file")

            cw.CLAUDE_DIR = projects_dir

            try:
                tracker = SessionTracker()
                await tracker.scan_projects()
                # Should skip the file, no events
                assert len(tracker.session_events) == 0
            finally:
                cw.CLAUDE_DIR = original_dir

    @pytest.mark.asyncio
    async def test_scan_project_without_sessions_dir(self) -> None:
        """Test scanning project without sessions directory (line 129)."""
        import tempfile
        from pathlib import Path
        import collectors.claude_watcher as cw

        original_dir = cw.CLAUDE_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir)
            # Create project dir but NO sessions subdir
            (projects_dir / "my-project").mkdir()

            cw.CLAUDE_DIR = projects_dir

            try:
                tracker = SessionTracker()
                await tracker.scan_projects()
                assert len(tracker.session_events) == 0
            finally:
                cw.CLAUDE_DIR = original_dir

    @pytest.mark.asyncio
    async def test_scan_with_temp_structure(self) -> None:
        """Test scanning with temporary Claude directory structure."""
        import tempfile
        import json
        from pathlib import Path
        import collectors.claude_watcher as cw

        # Save original CLAUDE_DIR
        original_dir = cw.CLAUDE_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake Claude directory structure
            projects_dir = Path(tmpdir)
            project_dir = projects_dir / "test-project"
            sessions_dir = project_dir / "sessions"
            sessions_dir.mkdir(parents=True)

            # Create a session file
            session_file = sessions_dir / "session.jsonl"
            with open(session_file, 'w') as f:
                f.write(json.dumps({"role": "user", "content": "Refactor the code"}) + "\n")

            # Temporarily override CLAUDE_DIR
            cw.CLAUDE_DIR = projects_dir

            try:
                tracker = SessionTracker()
                await tracker.scan_projects()
                # Should have captured the event
                assert len(tracker.session_events) == 1
                assert tracker.session_events[0]["intention"] == "refactor"
            finally:
                cw.CLAUDE_DIR = original_dir


class TestProcessFile:
    """Tests for file processing."""

    @pytest.mark.asyncio
    async def test_process_file_not_found(self) -> None:
        """Test processing non-existent file (line 166-167 IOError)."""
        from pathlib import Path
        tracker = SessionTracker()
        # Should not raise, triggers IOError handler
        await tracker._process_file(Path("/nonexistent/file.jsonl"), "test")
        assert len(tracker.session_events) == 0

    @pytest.mark.asyncio
    async def test_process_file_already_read(self) -> None:
        """Test processing file that was already read (line 148)."""
        import tempfile
        import json
        from pathlib import Path

        tracker = SessionTracker()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"role": "user", "content": "Test"}) + "\n")
            temp_path = Path(f.name)

        try:
            # First read
            await tracker._process_file(temp_path, "test")
            assert len(tracker.session_events) == 1

            # Second read - file unchanged, should return early (line 148)
            await tracker._process_file(temp_path, "test")
            # Still only 1 event (not re-processed)
            assert len(tracker.session_events) == 1
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_file_with_empty_lines(self) -> None:
        """Test processing file with empty lines (line 158)."""
        import tempfile
        import json
        from pathlib import Path

        tracker = SessionTracker()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"role": "user", "content": "Test"}) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write(json.dumps({"role": "user", "content": "Test2"}) + "\n")
            temp_path = Path(f.name)

        try:
            await tracker._process_file(temp_path, "test")
            # Should have processed 2 events, skipping empty lines
            assert len(tracker.session_events) == 2
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_file_with_data(self) -> None:
        """Test processing file with real JSONL data."""
        import tempfile
        import json
        from pathlib import Path

        tracker = SessionTracker()

        # Create temp file with JSONL data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write some test entries
            f.write(json.dumps({"role": "user", "content": "Create a new test"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "Sure"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "Fix the bug"}) + "\n")
            temp_path = Path(f.name)

        try:
            await tracker._process_file(temp_path, "test-project")
            # Should have processed 2 user messages
            assert len(tracker.session_events) == 2
            assert tracker.session_events[0]["intention"] == "create"
            assert tracker.session_events[1]["intention"] == "fix"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_file_invalid_json(self) -> None:
        """Test processing file with invalid JSON lines (line 163-164)."""
        import tempfile
        from pathlib import Path

        tracker = SessionTracker()

        # Create temp file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("not valid json\n")
            f.write("{invalid\n")
            temp_path = Path(f.name)

        try:
            await tracker._process_file(temp_path, "test-project")
            # Should handle gracefully, no events
            assert len(tracker.session_events) == 0
        finally:
            temp_path.unlink()


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_no_args_shows_help(self) -> None:
        """Test main with no args shows help."""
        import sys
        from io import StringIO
        from collectors.claude_watcher import main

        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = StringIO()
        sys.stdout = StringIO()
        old_argv = sys.argv
        sys.argv = ["claude_watcher.py"]

        try:
            main()
            # Should print help to stderr or stdout
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
            assert "claude" in output.lower() or "usage" in output.lower() or "daemon" in output.lower()
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            sys.argv = old_argv


class TestSendEventExceptions:
    """Test _send_event exception handling paths."""

    @pytest.mark.asyncio
    async def test_send_event_connection_refused(self) -> None:
        """Test _send_event handles connection errors (lines 225-226)."""
        import collectors.claude_watcher as cw

        original_url = cw.NOESIS_URL
        cw.NOESIS_URL = "http://127.0.0.1:59999"  # Non-existent port

        try:
            tracker = SessionTracker()
            event = {
                "event_type": "test",
                "timestamp": "2025-12-12T12:00:00",
                "project": "test",
                "files_touched": [],
                "intention": "test",
            }
            # Should not raise - exception is logged
            await tracker._send_event(event)
        finally:
            cw.NOESIS_URL = original_url

    @pytest.mark.asyncio
    async def test_send_event_timeout(self) -> None:
        """Test _send_event handles timeout (lines 225-226)."""
        tracker = SessionTracker()
        event = {
            "event_type": "timeout_test",
            "timestamp": "2025-12-12T12:00:00",
            "project": "test",
            "files_touched": [],
            "intention": "test",
        }
        # Should not raise even with potentially slow endpoint
        await tracker._send_event(event)


class TestRunDaemon:
    """Test run_daemon function."""

    @pytest.mark.asyncio
    async def test_run_daemon_cancellation(self) -> None:
        """Test run_daemon handles CancelledError (lines 239-244)."""
        import asyncio
        from collectors.claude_watcher import run_daemon

        # Create and cancel the daemon task
        task = asyncio.create_task(run_daemon())

        # Let it start
        await asyncio.sleep(0.1)

        # Cancel it
        task.cancel()

        # Should handle cancellation gracefully
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected


class TestProcessEntryAdvanced:
    """Advanced tests for _process_entry."""

    @pytest.mark.asyncio
    async def test_process_entry_with_all_intents(self) -> None:
        """Test processing entries with various intentions."""
        tracker = SessionTracker()

        test_cases = [
            ("Create a new API endpoint", "create"),
            ("Fix the authentication bug", "fix"),
            ("Refactor the database module", "refactor"),
            ("Explain how this works", "understand"),
            ("Delete the old files", "delete"),
            ("Test the payment system", "test"),
            ("Deploy to production", "deploy"),
        ]

        for content, expected_intent in test_cases:
            entry = {"role": "user", "content": content}
            await tracker._process_entry(entry, "test-project")
            assert tracker.session_events[-1]["intention"] == expected_intent

    @pytest.mark.asyncio
    async def test_process_entry_with_file_paths(self) -> None:
        """Test extracting file paths from content."""
        tracker = SessionTracker()

        entry = {
            "role": "user",
            "content": "Edit the file 'src/components/App.tsx' and also check config.json"
        }
        await tracker._process_entry(entry, "test-project")

        event = tracker.session_events[-1]
        assert len(event["files_touched"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
