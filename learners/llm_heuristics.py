"""
DAIMON LLM Service Heuristics
=============================

Heuristic patterns and rule-based fallbacks for the LLM service.

Contains:
- Classification patterns (approval/rejection)
- Cognitive state definitions
- Rule-based analysis functions

Extracted from llm_service.py for CODE_CONSTITUTION compliance (< 500 lines).

Usage:
    from learners.llm_heuristics import (
        classify_with_heuristics,
        analyze_cognitive_with_rules,
    )

Follows CODE_CONSTITUTION: Clarity Over Cleverness.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .llm_models import ClassificationResult, CognitiveAnalysis, InsightResult


# =============================================================================
# CLASSIFICATION PATTERNS
# =============================================================================

APPROVAL_PATTERNS = [
    r"\b(sim|yes|ok|perfeito|otimo|excelente|isso|gostei)\b",
    r"\b(aceito|aprovo|pode|manda|vai|bora|certo|correto)\b",
    r"^(s|y|ok|sim)$",
    r"(thumbs.?up|great|good|nice|awesome)",
]

REJECTION_PATTERNS = [
    r"\b(nao|no|nope|errado|ruim|feio|pare|espera)\b",
    r"\b(rejeito|recuso|para|cancela|volta|desfaz)\b",
    r"\b(menos|mais simples|muito|demais|longo)\b",
    r"(thumbs.?down|bad|wrong|incorrect)",
]


# =============================================================================
# COGNITIVE STATE DEFINITIONS
# =============================================================================

COGNITIVE_STATES = {
    "flow": {
        "description": "Deep focus, high productivity, optimal rhythm",
        "triggers": ["high_rhythm", "fast_typing", "low_errors"],
    },
    "focus": {
        "description": "Concentrated work, steady performance",
        "triggers": ["moderate_rhythm", "consistent_speed"],
    },
    "fatigue": {
        "description": "Declining performance, needs break",
        "triggers": ["increasing_delays", "rising_errors"],
    },
    "distracted": {
        "description": "Fragmented attention, inconsistent rhythm",
        "triggers": ["variable_rhythm", "long_pauses"],
    },
    "idle": {
        "description": "Insufficient activity for analysis",
        "triggers": ["low_activity"],
    },
}


# =============================================================================
# HEURISTIC FUNCTIONS
# =============================================================================


def classify_with_heuristics(
    content: str,
    options: List[str],
) -> ClassificationResult:
    """
    Classify content using regex patterns (fallback).
    
    Args:
        content: Text to classify
        options: Valid classification options
        
    Returns:
        ClassificationResult with matched category
    """
    content_lower = content.lower()

    # Check approval
    if "approval" in options:
        for pattern in APPROVAL_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return ClassificationResult(
                    category="approval",
                    confidence=0.6,
                    reasoning="Matched approval pattern",
                    from_llm=False,
                )

    # Check rejection
    if "rejection" in options:
        for pattern in REJECTION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return ClassificationResult(
                    category="rejection",
                    confidence=0.6,
                    reasoning="Matched rejection pattern",
                    from_llm=False,
                )

    # Default to first option or neutral
    default = "neutral" if "neutral" in options else options[0]
    return ClassificationResult(
        category=default,
        confidence=0.3,
        reasoning="No pattern matched, using default",
        from_llm=False,
    )


def extract_insights_with_templates(
    data: Dict[str, Any],
) -> InsightResult:
    """
    Extract insights using templates (fallback).
    
    Args:
        data: Preference data with approvals/rejections per category
        
    Returns:
        InsightResult with template-based insights
    """
    insights = []
    suggestions = []

    for category, counts in data.items():
        if not isinstance(counts, dict):
            continue

        total = counts.get("approvals", 0) + counts.get("rejections", 0)
        if total < 3:
            continue

        rate = counts.get("approvals", 0) / total if total > 0 else 0

        if rate < 0.3:
            insights.append(f"High rejection rate ({100-rate*100:.0f}%) in {category}")
            suggestions.append(f"Reduce automatic {category} behavior")
        elif rate > 0.8:
            insights.append(f"High approval rate ({rate*100:.0f}%) in {category}")
            suggestions.append(f"Continue current {category} approach")

    return InsightResult(
        insights=insights,
        suggestions=suggestions,
        confidence=0.5,
        from_llm=False,
    )


def analyze_cognitive_with_rules(
    biometrics: Dict[str, Any],
) -> CognitiveAnalysis:
    """
    Analyze cognitive state using rules (fallback).
    
    Args:
        biometrics: Keystroke biometrics data
        
    Returns:
        CognitiveAnalysis based on rule thresholds
    """
    rhythm = biometrics.get("rhythm_consistency", 0.5)
    fatigue = biometrics.get("fatigue_index", 0.0)
    focus = biometrics.get("focus_score", 0.5)
    wpm = biometrics.get("typing_speed", 0)

    # Detect flow state
    if rhythm > 0.8 and focus > 0.8 and wpm > 60:
        return CognitiveAnalysis(
            state="flow",
            confidence=0.8,
            description=COGNITIVE_STATES["flow"]["description"],
            recommendations=["Avoid interruptions", "Continue current task"],
            from_llm=False,
        )

    # Detect fatigue
    if fatigue > 0.6:
        return CognitiveAnalysis(
            state="fatigue",
            confidence=0.7,
            description=COGNITIVE_STATES["fatigue"]["description"],
            recommendations=["Take a short break", "Stretch or walk"],
            from_llm=False,
        )

    # Detect distraction
    if rhythm < 0.3:
        return CognitiveAnalysis(
            state="distracted",
            confidence=0.6,
            description=COGNITIVE_STATES["distracted"]["description"],
            recommendations=["Close distracting apps", "Try pomodoro technique"],
            from_llm=False,
        )

    # Default to focus
    return CognitiveAnalysis(
        state="focus",
        confidence=0.5,
        description=COGNITIVE_STATES["focus"]["description"],
        recommendations=[],
        from_llm=False,
    )
