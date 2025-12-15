"""
Input Sanitizer - Early Validation for Noise Reduction
=======================================================

Centralized validation layer to filter malformed data before it enters
the consciousness pipeline. Part of the Noesis Entropy Audit (2025-12-15).

Benefits:
- Reduces entropic noise in the system
- Prevents cascading errors from bad input
- Normalizes data formats across modules
- Provides clear error messages for debugging

Usage:
    from shared.validators import InputSanitizer, SanitizedInput

    sanitizer = InputSanitizer()
    result = sanitizer.validate_user_input("  Hello, Noesis!  ")

    if result.is_valid:
        clean_text = result.cleaned
    else:
        log_error(result.error_message)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    """Types of validation errors."""
    EMPTY_INPUT = "empty_input"
    TOO_LONG = "too_long"
    INVALID_ENCODING = "invalid_encoding"
    POTENTIALLY_MALICIOUS = "potentially_malicious"
    INVALID_FORMAT = "invalid_format"
    INVALID_TYPE = "invalid_type"


@dataclass
class SanitizedInput:
    """Result of input sanitization."""
    is_valid: bool
    cleaned: str
    original: str
    error_type: Optional[ValidationErrorType] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class SanitizedSignal:
    """Result of signal/event validation."""
    is_valid: bool
    cleaned: Dict[str, Any]
    error_type: Optional[ValidationErrorType] = None
    error_message: Optional[str] = None


class InputSanitizer:
    """
    Central validation layer for all input entering Noesis.

    Performs:
    - Whitespace normalization
    - Length validation
    - Encoding cleanup
    - Basic security filtering
    """

    DEFAULT_MAX_LENGTH = 50000  # 50k chars
    MIN_LENGTH = 1

    # Patterns to detect potentially malicious content
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',
        r'data:text/html',
        r'\x00',  # Null bytes
    ]

    def __init__(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        strip_html: bool = True,
        normalize_whitespace: bool = True
    ):
        self.max_length = max_length
        self.strip_html = strip_html
        self.normalize_whitespace = normalize_whitespace
        self._dangerous_regex = re.compile(
            '|'.join(self.DANGEROUS_PATTERNS),
            re.IGNORECASE | re.DOTALL
        )

    def validate_user_input(self, raw: str) -> SanitizedInput:
        """
        Validate and sanitize user input.

        Args:
            raw: Raw input string from user

        Returns:
            SanitizedInput with cleaned text or error details
        """
        original = raw
        warnings = []

        # Type check
        if not isinstance(raw, str):
            return SanitizedInput(
                is_valid=False,
                cleaned="",
                original=str(raw),
                error_type=ValidationErrorType.INVALID_TYPE,
                error_message=f"Expected string, got {type(raw).__name__}"
            )

        # Encoding cleanup - remove invalid unicode
        try:
            cleaned = raw.encode('utf-8', errors='replace').decode('utf-8')
            if cleaned != raw:
                warnings.append("Fixed invalid UTF-8 characters")
        # pylint: disable=broad-exception-caught
        except Exception as e:
            return SanitizedInput(
                is_valid=False,
                cleaned="",
                original=original,
                error_type=ValidationErrorType.INVALID_ENCODING,
                error_message=f"Encoding error: {e}"
            )

        # Whitespace normalization
        if self.normalize_whitespace:
            cleaned = cleaned.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces

        # Empty check
        if not cleaned:
            return SanitizedInput(
                is_valid=False,
                cleaned="",
                original=original,
                error_type=ValidationErrorType.EMPTY_INPUT,
                error_message="Input is empty after sanitization"
            )

        # Length check
        if len(cleaned) > self.max_length:
            return SanitizedInput(
                is_valid=False,
                cleaned=cleaned[:self.max_length],
                original=original,
                error_type=ValidationErrorType.TOO_LONG,
                error_message=f"Input exceeds maximum length of {self.max_length}"
            )

        # Security check
        if self._dangerous_regex.search(cleaned):
            # Don't reject, but sanitize
            cleaned = self._dangerous_regex.sub('[FILTERED]', cleaned)
            warnings.append("Potentially dangerous content filtered")
            logger.warning("[Sanitizer] Filtered dangerous content from input")

        # Strip HTML tags if configured
        if self.strip_html:
            html_pattern = re.compile(r'<[^>]+>')
            if html_pattern.search(cleaned):
                cleaned = html_pattern.sub('', cleaned)
                warnings.append("HTML tags removed")

        return SanitizedInput(
            is_valid=True,
            cleaned=cleaned,
            original=original,
            warnings=warnings
        )

    def validate_signal(self, signal: Dict[str, Any]) -> SanitizedSignal:
        """
        Validate an internal signal/event dictionary.

        Args:
            signal: Signal dictionary with topic, payload, etc.

        Returns:
            SanitizedSignal with validation result
        """
        if not isinstance(signal, dict):
            return SanitizedSignal(
                is_valid=False,
                cleaned={},
                error_type=ValidationErrorType.INVALID_TYPE,
                error_message=f"Signal must be dict, got {type(signal).__name__}"
            )

        # Check required fields
        required_fields = ["topic", "payload"]
        for field_name in required_fields:
            if field_name not in signal:
                return SanitizedSignal(
                    is_valid=False,
                    cleaned=signal,
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    error_message=f"Missing required field: {field_name}"
                )

        # Validate topic format (should be dot-separated, lowercase)
        topic = signal.get("topic", "")
        if not re.match(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$', topic):
            return SanitizedSignal(
                is_valid=False,
                cleaned=signal,
                error_type=ValidationErrorType.INVALID_FORMAT,
                error_message=f"Invalid topic format: {topic}. Use lowercase.dotted.format"
            )

        # Cleaned signal
        cleaned = {
            "topic": topic.lower(),
            "payload": signal.get("payload", {}),
            "source": signal.get("source", "unknown"),
            "priority": signal.get("priority", "NORMAL"),
        }

        return SanitizedSignal(is_valid=True, cleaned=cleaned)

    def validate_coherence(self, value: float) -> bool:
        """Validate a coherence value is in valid range [0, 1]."""
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0

    def validate_depth(self, value: int) -> bool:
        """Validate a depth value is in valid range [1, 5]."""
        return isinstance(value, int) and 1 <= value <= 5


# Singleton instance
_global_sanitizer: Optional[InputSanitizer] = None


def get_input_sanitizer() -> InputSanitizer:
    """Get or create the global input sanitizer instance."""
    # pylint: disable=global-statement
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = InputSanitizer()
    return _global_sanitizer
