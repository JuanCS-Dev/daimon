
import pytest
from shared.validators import InputSanitizer, ValidationErrorType

def test_sanitizer_valid_input():
    sanitizer = InputSanitizer()
    res = sanitizer.validate_user_input("  Test Input  ")
    
    assert res.is_valid
    assert res.cleaned == "Test Input"
    assert res.warnings == []

def test_sanitizer_html_stripping():
    sanitizer = InputSanitizer(strip_html=True)
    res = sanitizer.validate_user_input("<b>Bold</b> text")
    
    assert res.is_valid
    assert res.cleaned == "Bold text"
    assert "HTML tags removed" in res.warnings

def test_sanitizer_dangerous_content():
    sanitizer = InputSanitizer()
    # Simulation of XSS
    res = sanitizer.validate_user_input("<script>alert(1)</script>")
    
    assert res.is_valid # It sanitizes, doesn't necessarily reject valid strings unless empty
    assert "[FILTERED]" in res.cleaned
    assert "Potentially dangerous content filtered" in res.warnings

def test_sanitizer_max_length():
    sanitizer = InputSanitizer(max_length=5)
    res = sanitizer.validate_user_input("Too long input")
    
    assert not res.is_valid
    assert res.error_type == ValidationErrorType.TOO_LONG

def test_signal_validation():
    sanitizer = InputSanitizer()
    valid_signal = {
        "topic": "test.event",
        "payload": {"foo": "bar"}
    }
    res = sanitizer.validate_signal(valid_signal)
    assert res.is_valid
    assert res.cleaned["topic"] == "test.event"

def test_signal_validation_invalid_format():
    sanitizer = InputSanitizer()
    invalid_signal = {
        "topic": "INVALID TOPIC", # Caps/spaces not allowed
        "payload": {}
    }
    res = sanitizer.validate_signal(invalid_signal)
    assert not res.is_valid
    assert res.error_type == ValidationErrorType.INVALID_FORMAT
