"""
DAIMON Preference Categorizer
=============================

Categorizes preference signals and calculates signal strength.

Responsible for:
- Inferring preference category from context keywords
- Calculating signal strength based on content characteristics
- Tracking category-level statistics

Architecture:
    PreferenceCategorizer processes PreferenceSignals from the detector
    and prepares data for the InsightGenerator.

Usage:
    from learners.preference.categorizer import PreferenceCategorizer
    
    categorizer = PreferenceCategorizer()
    category = categorizer.infer_category(text)
    strength = categorizer.calculate_strength(content)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .models import CategoryStats, PreferenceCategory, PreferenceSignal

logger = logging.getLogger("daimon.preference.categorizer")

# Keywords for category inference
CATEGORY_KEYWORDS: Dict[str, list] = {
    PreferenceCategory.CODE_STYLE.value: [
        "formatacao", "estilo", "naming", "indent", "lint", "format", "style"
    ],
    PreferenceCategory.VERBOSITY.value: [
        "verboso", "longo", "curto", "resumo", "detalhado", "verbose", "brief"
    ],
    PreferenceCategory.TESTING.value: [
        "teste", "test", "coverage", "mock", "assert", "spec", "unit"
    ],
    PreferenceCategory.ARCHITECTURE.value: [
        "arquitetura", "estrutura", "pattern", "design", "refactor"
    ],
    PreferenceCategory.DOCUMENTATION.value: [
        "doc", "comment", "readme", "docstring", "jsdoc"
    ],
    PreferenceCategory.WORKFLOW.value: [
        "commit", "branch", "git", "deploy", "ci", "cd", "push"
    ],
    PreferenceCategory.SECURITY.value: [
        "security", "auth", "password", "token", "secret", "vulnerability"
    ],
    PreferenceCategory.PERFORMANCE.value: [
        "performance", "speed", "fast", "slow", "optimize", "cache"
    ],
}


class PreferenceCategorizer:
    """
    Categorizes and scores preference signals.
    
    Uses keyword matching to infer categories and content
    analysis to calculate signal strength.
    """
    
    def __init__(self):
        """Initialize categorizer."""
        self.category_stats: Dict[str, CategoryStats] = {}
    
    def infer_category(self, text: str) -> str:
        """
        Infer preference category based on keywords in text.
        
        Scans the text for category-specific keywords and returns
        the best matching category.
        
        Args:
            text: Combined user content and context
            
        Returns:
            Category string (e.g., "code_style", "testing")
        """
        text_lower = text.lower()
        
        best_category = PreferenceCategory.GENERAL.value
        best_score = 0
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def calculate_strength(self, content: str) -> float:
        """
        Calculate signal strength from 0.0 to 1.0.
        
        Strength is based on:
        - Content length (shorter = more decisive)
        - Exclamation marks (enthusiasm)
        - Caps (emphasis)
        
        Args:
            content: User message content
            
        Returns:
            Strength value between 0.0 and 1.0
        """
        if not content:
            return 0.3
        
        # Base strength (shorter = stronger signal)
        length = len(content)
        if length < 20:
            base = 0.9  # Very short, decisive
        elif length < 50:
            base = 0.7  # Short
        elif length < 100:
            base = 0.5  # Medium
        else:
            base = 0.4  # Long, less decisive
        
        # Enthusiasm bonus
        exclamation_bonus = min(content.count("!") * 0.05, 0.1)
        
        # Emphasis bonus (caps)
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        caps_bonus = min(caps_ratio * 0.2, 0.1)
        
        return min(base + exclamation_bonus + caps_bonus, 1.0)
    
    def update_stats(self, signal: PreferenceSignal) -> None:
        """
        Update category statistics with a new signal.
        
        Args:
            signal: PreferenceSignal to incorporate
        """
        category = signal.category
        
        if category not in self.category_stats:
            self.category_stats[category] = CategoryStats()
        
        stats = self.category_stats[category]
        
        if signal.signal_type == "approval":
            stats.approvals += 1
        elif signal.signal_type == "rejection":
            stats.rejections += 1
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all categories.
        
        Returns:
            Dictionary mapping category to stats
        """
        return {
            cat: stats.to_dict()
            for cat, stats in self.category_stats.items()
        }
    
    def clear(self) -> None:
        """Clear all statistics."""
        self.category_stats.clear()
