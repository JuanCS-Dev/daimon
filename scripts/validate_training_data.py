#!/usr/bin/env python3
"""
Validate training data for Constitutional AI fine-tuning.

This module performs comprehensive validation of:
- Memory preparation outputs
- Philosophical dataset examples
- Format consistency
- Value coverage
- Quality metrics
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "data" / "training" / "memories"
PHILOSOPHICAL_DIR = PROJECT_ROOT / "data" / "training" / "philosophical"

# Required fields for Constitutional AI format
REQUIRED_FIELDS = {
    "id", "category", "prompt", "response_initial",
    "critique", "response_revised", "reasoning", "values_applied"
}

# Expected values
VALID_VALUES = {"verdade", "sabedoria", "justica", "florescimento", "alianca"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_CATEGORIES = {
    "maieutica", "logical_argument", "ethical_dilemma",
    "tribunal", "anti_sycophancy", "value_application", "counter_example",
    "modern_philosophy", "presocratic_mathematics", "scientific_method",
    "jesus_philosophy", "hermetic_wisdom"
}


@dataclass
class ValidationResult:
    """Result of validating a single example."""

    file_path: str
    line_number: int
    example_id: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileValidationResult:
    """Result of validating an entire file."""

    file_path: str
    total_examples: int
    valid_examples: int
    invalid_examples: int
    errors: List[str] = field(default_factory=list)
    example_results: List[ValidationResult] = field(default_factory=list)


@dataclass
class DatasetStatistics:
    """Statistics about the dataset."""

    total_files: int = 0
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    values_coverage: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    avg_prompt_length: float = 0.0
    avg_response_length: float = 0.0


class TrainingDataValidator:
    """Validates training data for Constitutional AI."""

    def __init__(self) -> None:
        """Initialize validator."""
        self.results: List[FileValidationResult] = []
        self.stats = DatasetStatistics()

    def validate_example(
        self,
        example: Dict[str, Any],
        file_path: str,
        line_number: int
    ) -> ValidationResult:
        """
        Validate a single training example.

        Args:
            example: The example dictionary to validate
            file_path: Path to the source file
            line_number: Line number in the file

        Returns:
            ValidationResult with errors and warnings
        """
        errors: List[str] = []
        warnings: List[str] = []
        example_id = example.get("id", f"line_{line_number}")

        # Check required fields
        missing_fields = REQUIRED_FIELDS - set(example.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")

        # Validate category
        category = example.get("category", "")
        if category and category not in VALID_CATEGORIES:
            warnings.append(f"Unknown category: {category}")

        # Validate values_applied
        values = example.get("values_applied", [])
        if not values:
            errors.append("values_applied is empty")
        else:
            invalid_values = set(values) - VALID_VALUES
            if invalid_values:
                warnings.append(f"Unknown values: {invalid_values}")

        # Validate difficulty
        difficulty = example.get("difficulty", "medium")
        if difficulty not in VALID_DIFFICULTIES:
            warnings.append(f"Unknown difficulty: {difficulty}")

        # Content quality checks
        prompt = example.get("prompt", "")
        response_initial = example.get("response_initial", "")
        response_revised = example.get("response_revised", "")
        critique = example.get("critique", "")

        if len(prompt) < 10:
            errors.append("Prompt too short (< 10 chars)")

        if len(response_revised) < 50:
            warnings.append("Revised response short (< 50 chars)")

        if response_initial == response_revised:
            errors.append("Initial and revised responses are identical")

        # Check for judge markers in tribunal examples
        if category == "tribunal":
            if "[VERITAS" not in critique:
                warnings.append("Tribunal example missing VERITAS judgment")
            if "[SOPHIA" not in critique:
                warnings.append("Tribunal example missing SOPHIA judgment")
            if "[DIKE" not in critique:
                warnings.append("Tribunal example missing DIKE judgment")

        is_valid = len(errors) == 0

        return ValidationResult(
            file_path=file_path,
            line_number=line_number,
            example_id=example_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    def validate_jsonl_file(self, file_path: Path) -> FileValidationResult:
        """
        Validate a JSONL file.

        Args:
            file_path: Path to the JSONL file

        Returns:
            FileValidationResult with all validation results
        """
        result = FileValidationResult(
            file_path=str(file_path),
            total_examples=0,
            valid_examples=0,
            invalid_examples=0
        )

        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            return result

        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        example = json.loads(line)
                        validation = self.validate_example(
                            example, str(file_path), line_num
                        )
                        result.example_results.append(validation)
                        result.total_examples += 1

                        if validation.is_valid:
                            result.valid_examples += 1
                            self._update_stats(example)
                        else:
                            result.invalid_examples += 1

                    except json.JSONDecodeError as e:
                        result.errors.append(f"Line {line_num}: Invalid JSON - {e}")
                        result.invalid_examples += 1

        except Exception as e:
            result.errors.append(f"Error reading file: {e}")

        return result

    def _update_stats(self, example: Dict[str, Any]) -> None:
        """Update running statistics from a valid example."""
        # Category
        category = example.get("category", "unknown")
        self.stats.categories[category] = self.stats.categories.get(category, 0) + 1

        # Values
        for value in example.get("values_applied", []):
            self.stats.values_coverage[value] = \
                self.stats.values_coverage.get(value, 0) + 1

        # Difficulty
        difficulty = example.get("difficulty", "medium")
        self.stats.difficulties[difficulty] = \
            self.stats.difficulties.get(difficulty, 0) + 1

    def validate_directory(self, directory: Path) -> List[FileValidationResult]:
        """
        Validate all JSONL files in a directory.

        Args:
            directory: Path to directory containing JSONL files

        Returns:
            List of FileValidationResult for each file
        """
        results = []

        if not directory.exists():
            logger.warning("Directory not found: %s", directory)
            return results

        for file_path in sorted(directory.glob("*.jsonl")):
            result = self.validate_jsonl_file(file_path)
            results.append(result)
            self.stats.total_files += 1
            self.stats.total_examples += result.total_examples
            self.stats.valid_examples += result.valid_examples
            self.stats.invalid_examples += result.invalid_examples

        return results

    def validate_all(self) -> None:
        """Validate all training data."""
        # Note: Memory data is raw input, not Constitutional AI format
        # Only validate philosophical data which is in Constitutional AI format
        logger.info("Validating philosophical data (Constitutional AI format)...")
        philosophical_results = self.validate_directory(PHILOSOPHICAL_DIR)
        self.results.extend(philosophical_results)

        # Report on memory data existence (but don't validate format)
        if MEMORY_DIR.exists():
            memory_files = list(MEMORY_DIR.glob("*.jsonl")) + list(MEMORY_DIR.glob("*.json"))
            logger.info("Memory data found: %d files (raw format, not validated)", len(memory_files))

    def print_report(self) -> bool:
        """
        Print validation report.

        Returns:
            True if all data is valid, False otherwise
        """
        print("\n" + "=" * 60)
        print("TRAINING DATA VALIDATION REPORT")
        print("=" * 60)

        all_valid = True

        # File-level results
        print("\n--- FILE VALIDATION ---")
        for result in self.results:
            status = "OK" if result.invalid_examples == 0 else "ERRORS"
            print(f"\n{Path(result.file_path).name}: {status}")
            print(f"  Total: {result.total_examples}, "
                  f"Valid: {result.valid_examples}, "
                  f"Invalid: {result.invalid_examples}")

            if result.errors:
                all_valid = False
                for error in result.errors[:5]:  # Limit to 5 errors
                    print(f"  ERROR: {error}")

            # Show example-level issues
            for ex_result in result.example_results:
                if ex_result.errors:
                    all_valid = False
                    print(f"  [{ex_result.example_id}] ERRORS: {ex_result.errors}")
                if ex_result.warnings:
                    print(f"  [{ex_result.example_id}] WARNINGS: {ex_result.warnings}")

        # Statistics
        print("\n--- STATISTICS ---")
        print(f"Total files: {self.stats.total_files}")
        print(f"Total examples: {self.stats.total_examples}")
        print(f"Valid examples: {self.stats.valid_examples}")
        print(f"Invalid examples: {self.stats.invalid_examples}")

        if self.stats.total_examples > 0:
            validity_rate = (self.stats.valid_examples / self.stats.total_examples) * 100
            print(f"Validity rate: {validity_rate:.1f}%")

        # Category distribution
        print("\n--- CATEGORIES ---")
        for category, count in sorted(self.stats.categories.items()):
            print(f"  {category}: {count}")

        # Value coverage
        print("\n--- VALUE COVERAGE ---")
        for value in VALID_VALUES:
            count = self.stats.values_coverage.get(value, 0)
            print(f"  {value}: {count}")

        # Missing values
        missing_values = VALID_VALUES - set(self.stats.values_coverage.keys())
        if missing_values:
            print(f"\n  WARNING: No examples for values: {missing_values}")

        # Difficulty distribution
        print("\n--- DIFFICULTY DISTRIBUTION ---")
        for diff, count in sorted(self.stats.difficulties.items()):
            print(f"  {diff}: {count}")

        # Final verdict
        print("\n" + "=" * 60)
        if all_valid and self.stats.invalid_examples == 0:
            print("VALIDATION PASSED - All data is valid")
        else:
            print("VALIDATION FAILED - See errors above")
        print("=" * 60)

        return all_valid and self.stats.invalid_examples == 0


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for validation errors)
    """
    validator = TrainingDataValidator()
    validator.validate_all()
    is_valid = validator.print_report()

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
