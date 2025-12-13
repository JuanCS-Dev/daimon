"""
Tests for actuators/config_refiner.py

Scientific tests covering:
- CLAUDE.md reading/writing
- Section generation
- Content merging
- Backup creation and management
- Update logging
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from actuators.config_refiner import (
    DAIMON_SECTION_END,
    DAIMON_SECTION_START,
    ConfigRefiner,
)


class TestConfigRefinerInit:
    """Tests for ConfigRefiner initialization."""

    def test_init_with_defaults(self, tmp_path: Path) -> None:
        """Test initialization with custom paths (avoiding real ~/.claude)."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        assert refiner.claude_md == tmp_path / "CLAUDE.md"
        assert refiner.backup_dir == tmp_path / "backups"
        assert refiner.backup_dir.exists()

    def test_init_creates_backup_dir(self, tmp_path: Path) -> None:
        """Test that init creates backup directory."""
        backup_dir = tmp_path / "nested" / "backups"
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=backup_dir,
        )
        assert backup_dir.exists()


class TestCreateBackup:
    """Tests for ConfigRefiner._create_backup()."""

    def test_backup_nonexistent_file(self, tmp_path: Path) -> None:
        """Test backup when CLAUDE.md doesn't exist."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        result = refiner._create_backup()
        assert result is None

    def test_backup_existing_file(self, tmp_path: Path) -> None:
        """Test backup of existing CLAUDE.md."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("original content")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        backup_path = refiner._create_backup()

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text() == "original content"

    def test_backup_keeps_only_10(self, tmp_path: Path) -> None:
        """Test that only 10 backups are kept."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("content")
        backup_dir = tmp_path / "backups"

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=backup_dir,
        )

        # Create 15 backups
        for i in range(15):
            # Manually create old backups
            old_backup = backup_dir / f"CLAUDE.md.2025010100000{i:02d}"
            old_backup.write_text(f"backup {i}")

        # Create new backup (should trigger cleanup)
        refiner._create_backup()

        # Count backups
        backups = list(backup_dir.glob("CLAUDE.md.*"))
        assert len(backups) <= 11  # 10 old + 1 new


class TestReadCurrent:
    """Tests for ConfigRefiner._read_current()."""

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        """Test reading non-existent file."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        content = refiner._read_current()
        assert content == ""

    def test_read_existing(self, tmp_path: Path) -> None:
        """Test reading existing file."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("test content")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        content = refiner._read_current()
        assert content == "test content"


class TestGenerateSection:
    """Tests for ConfigRefiner._generate_section()."""

    @pytest.fixture
    def refiner(self, tmp_path: Path) -> ConfigRefiner:
        """Create a ConfigRefiner instance."""
        return ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )

    def test_generate_section_basic(self, refiner: ConfigRefiner) -> None:
        """Test generating section with basic insights."""
        insights = [
            {
                "category": "testing",
                "action": "reinforce",
                "confidence": 0.9,
                "suggestion": "Continue generating tests",
            }
        ]
        section = refiner._generate_section(insights)

        assert DAIMON_SECTION_START in section
        assert DAIMON_SECTION_END in section
        assert "## Testing" in section
        assert "[Alta]" in section  # High confidence
        assert "Continue generating tests" in section

    def test_generate_section_multiple_categories(self, refiner: ConfigRefiner) -> None:
        """Test generating section with multiple categories."""
        insights = [
            {
                "category": "testing",
                "confidence": 0.8,
                "suggestion": "Testing suggestion",
            },
            {
                "category": "verbosity",
                "confidence": 0.5,
                "suggestion": "Verbosity suggestion",
            },
        ]
        section = refiner._generate_section(insights)

        assert "## Testing" in section
        assert "## Verbosity" in section
        assert "[Alta]" in section  # 0.8 confidence
        assert "[Media]" in section  # 0.5 confidence

    def test_generate_section_low_confidence(self, refiner: ConfigRefiner) -> None:
        """Test generating section with low confidence."""
        insights = [
            {
                "category": "code_style",
                "confidence": 0.2,
                "suggestion": "Low confidence suggestion",
            }
        ]
        section = refiner._generate_section(insights)
        assert "[Baixa]" in section

    def test_generate_section_includes_timestamp(self, refiner: ConfigRefiner) -> None:
        """Test that section includes timestamp."""
        insights = [{"category": "general", "confidence": 0.5, "suggestion": "test"}]
        section = refiner._generate_section(insights)
        assert "Ultima atualizacao:" in section


class TestMergeContent:
    """Tests for ConfigRefiner._merge_content()."""

    @pytest.fixture
    def refiner(self, tmp_path: Path) -> ConfigRefiner:
        """Create a ConfigRefiner instance."""
        return ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )

    def test_merge_empty_current(self, refiner: ConfigRefiner) -> None:
        """Test merging into empty file."""
        new_section = f"{DAIMON_SECTION_START}\nnew content\n{DAIMON_SECTION_END}"
        result = refiner._merge_content("", new_section)
        assert result == new_section

    def test_merge_with_manual_content(self, refiner: ConfigRefiner) -> None:
        """Test merging with existing manual content."""
        current = "# My Manual Instructions\n\nDo this and that."
        new_section = f"{DAIMON_SECTION_START}\nnew\n{DAIMON_SECTION_END}"
        result = refiner._merge_content(current, new_section)

        assert new_section in result
        assert "My Manual Instructions" in result
        assert "---" in result  # Separator

    def test_merge_replace_existing_section(self, refiner: ConfigRefiner) -> None:
        """Test replacing existing DAIMON section."""
        current = f"Manual\n\n{DAIMON_SECTION_START}\nold\n{DAIMON_SECTION_END}\n\nMore manual"
        new_section = f"{DAIMON_SECTION_START}\nnew content\n{DAIMON_SECTION_END}"
        result = refiner._merge_content(current, new_section)

        assert "old" not in result
        assert "new content" in result
        assert "Manual" in result
        assert result.count(DAIMON_SECTION_START) == 1


class TestSectionsEqual:
    """Tests for ConfigRefiner._sections_equal()."""

    @pytest.fixture
    def refiner(self, tmp_path: Path) -> ConfigRefiner:
        """Create a ConfigRefiner instance."""
        return ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )

    def test_equal_sections(self, refiner: ConfigRefiner) -> None:
        """Test equal sections."""
        s1 = "## Testing\n- [Alta] Continue testing"
        s2 = "## Testing\n- [Alta] Continue testing"
        assert refiner._sections_equal(s1, s2) is True

    def test_sections_different_only_timestamp(self, refiner: ConfigRefiner) -> None:
        """Test sections that differ only in timestamp."""
        s1 = "*Ultima atualizacao: 2025-01-01 10:00*\n## Testing"
        s2 = "*Ultima atualizacao: 2025-01-02 15:00*\n## Testing"
        assert refiner._sections_equal(s1, s2) is True

    def test_sections_actually_different(self, refiner: ConfigRefiner) -> None:
        """Test actually different sections."""
        s1 = "## Testing\n- [Alta] Old suggestion"
        s2 = "## Testing\n- [Alta] New suggestion"
        assert refiner._sections_equal(s1, s2) is False


class TestUpdatePreferences:
    """Tests for ConfigRefiner.update_preferences()."""

    def test_update_empty_insights(self, tmp_path: Path) -> None:
        """Test update with empty insights."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        result = refiner.update_preferences([])
        assert result is False

    def test_update_creates_file(self, tmp_path: Path) -> None:
        """Test update creates CLAUDE.md if not exists."""
        claude_md = tmp_path / "CLAUDE.md"
        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )

        insights = [
            {
                "category": "testing",
                "confidence": 0.8,
                "suggestion": "Test suggestion",
            }
        ]
        result = refiner.update_preferences(insights)

        assert result is True
        assert claude_md.exists()
        content = claude_md.read_text()
        assert DAIMON_SECTION_START in content

    def test_update_preserves_manual(self, tmp_path: Path) -> None:
        """Test update preserves manual content."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# My Manual Config\n\nImportant stuff.")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )

        insights = [
            {"category": "testing", "confidence": 0.8, "suggestion": "Test"}
        ]
        refiner.update_preferences(insights)

        content = claude_md.read_text()
        assert "My Manual Config" in content
        assert "Important stuff" in content

    def test_update_creates_backup(self, tmp_path: Path) -> None:
        """Test update creates backup."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("original")
        backup_dir = tmp_path / "backups"

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=backup_dir,
        )

        insights = [{"category": "testing", "confidence": 0.8, "suggestion": "Test"}]
        refiner.update_preferences(insights)

        backups = list(backup_dir.glob("CLAUDE.md.*"))
        assert len(backups) >= 1


class TestLogUpdate:
    """Tests for ConfigRefiner._log_update()."""

    def test_log_update_creates_file(self, tmp_path: Path) -> None:
        """Test logging creates log file."""
        backup_dir = tmp_path / "backups"
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=backup_dir,
        )

        insights = [
            {"category": "testing", "action": "reinforce"},
            {"category": "verbosity", "action": "reduce"},
        ]
        refiner._log_update(insights)

        log_path = backup_dir / "update_log.jsonl"
        assert log_path.exists()

        with open(log_path) as f:
            entry = json.loads(f.readline())
            assert entry["insights_count"] == 2
            assert "testing" in entry["categories"]
            assert "verbosity" in entry["categories"]


class TestGetCurrentPreferences:
    """Tests for ConfigRefiner.get_current_preferences()."""

    def test_get_no_file(self, tmp_path: Path) -> None:
        """Test getting preferences when file doesn't exist."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_current_preferences()
        assert result is None

    def test_get_no_section(self, tmp_path: Path) -> None:
        """Test getting preferences when no DAIMON section."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Just manual content")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_current_preferences()
        assert result is None

    def test_get_with_section(self, tmp_path: Path) -> None:
        """Test getting preferences with DAIMON section."""
        claude_md = tmp_path / "CLAUDE.md"
        section = f"{DAIMON_SECTION_START}\n## Testing\n{DAIMON_SECTION_END}"
        claude_md.write_text(f"Manual\n\n{section}\n\nMore manual")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_current_preferences()

        assert result is not None
        assert DAIMON_SECTION_START in result
        assert "## Testing" in result


class TestGetManualContent:
    """Tests for ConfigRefiner.get_manual_content()."""

    def test_get_manual_no_file(self, tmp_path: Path) -> None:
        """Test getting manual content when no file."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_manual_content()
        assert result == ""

    def test_get_manual_no_daimon_section(self, tmp_path: Path) -> None:
        """Test getting manual content when no DAIMON section."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("All manual content here")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_manual_content()
        assert result == "All manual content here"

    def test_get_manual_with_daimon_section(self, tmp_path: Path) -> None:
        """Test getting manual content excludes DAIMON section."""
        claude_md = tmp_path / "CLAUDE.md"
        content = f"Manual 1\n\n{DAIMON_SECTION_START}\nAuto\n{DAIMON_SECTION_END}\n\nManual 2"
        claude_md.write_text(content)

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )
        result = refiner.get_manual_content()

        assert "Manual 1" in result
        assert "Manual 2" in result
        assert "Auto" not in result
        assert DAIMON_SECTION_START not in result


class TestGetBackupList:
    """Tests for ConfigRefiner.get_backup_list()."""

    def test_backup_list_empty(self, tmp_path: Path) -> None:
        """Test backup list when no backups."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        backups = refiner.get_backup_list()
        assert backups == []

    def test_backup_list_with_backups(self, tmp_path: Path) -> None:
        """Test backup list with backups."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create some backup files
        (backup_dir / "CLAUDE.md.20250101_100000").write_text("backup1")
        (backup_dir / "CLAUDE.md.20250102_100000").write_text("backup2")

        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=backup_dir,
        )
        backups = refiner.get_backup_list()

        assert len(backups) == 2
        assert all("path" in b for b in backups)
        assert all("filename" in b for b in backups)
        assert all("size_bytes" in b for b in backups)


class TestUpdatePreferencesNoChange:
    """Tests for update_preferences when content hasn't changed."""

    def test_update_no_change_when_same_insights(self, tmp_path: Path) -> None:
        """Test that update returns False when insights produce same content."""
        claude_md = tmp_path / "CLAUDE.md"
        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=tmp_path / "backups",
        )

        insights = [
            {"category": "testing", "confidence": 0.8, "suggestion": "Test"}
        ]

        # First update
        result1 = refiner.update_preferences(insights)
        assert result1 is True

        # Second update with same insights should return False
        result2 = refiner.update_preferences(insights)
        assert result2 is False


class TestRestoreBackup:
    """Tests for ConfigRefiner.restore_backup()."""

    def test_restore_nonexistent(self, tmp_path: Path) -> None:
        """Test restoring non-existent backup."""
        refiner = ConfigRefiner(
            claude_md_path=tmp_path / "CLAUDE.md",
            backup_dir=tmp_path / "backups",
        )
        result = refiner.restore_backup("/nonexistent/backup")
        assert result is False

    def test_restore_existing(self, tmp_path: Path) -> None:
        """Test restoring existing backup."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("current content")
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        backup = backup_dir / "CLAUDE.md.20250101_100000"
        backup.write_text("backup content")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=backup_dir,
        )
        result = refiner.restore_backup(str(backup))

        assert result is True
        assert claude_md.read_text() == "backup content"

    def test_restore_creates_backup_first(self, tmp_path: Path) -> None:
        """Test that restore creates backup of current state first."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("current content")
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        old_backup = backup_dir / "CLAUDE.md.20250101_100000"
        old_backup.write_text("old backup")

        refiner = ConfigRefiner(
            claude_md_path=claude_md,
            backup_dir=backup_dir,
        )
        refiner.restore_backup(str(old_backup))

        # Should have created a new backup before restoring
        backups = list(backup_dir.glob("CLAUDE.md.*"))
        assert len(backups) >= 2
