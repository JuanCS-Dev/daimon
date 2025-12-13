"""
DAIMON Preference Learner - Facade
==================================

Main entry point for the preference learning system.

This facade orchestrates the modular components:
- Scanner: Discovers and loads sessions
- Detector: Identifies preference signals
- Categorizer: Classifies and scores signals
- InsightGenerator: Creates actionable recommendations

Architecture:
    ┌─────────────────────────────────────────────────┐
    │            PreferenceLearner (Facade)           │
    │                                                 │
    │   ┌─────────┐   ┌──────────┐   ┌────────────┐  │
    │   │ Scanner │ → │ Detector │ → │ Categorizer│  │
    │   └─────────┘   └──────────┘   └────────────┘  │
    │                                       ↓         │
    │                              ┌────────────────┐ │
    │                              │ InsightGen     │ │
    │                              └────────────────┘ │
    └─────────────────────────────────────────────────┘

Usage:
    from learners.preference_learner import PreferenceLearner
    
    learner = PreferenceLearner()
    signals = learner.scan_sessions(since_hours=24)
    summary = learner.get_preference_summary()
    insights = learner.get_actionable_insights()

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import modular components
from .preference import (
    PreferenceSignal,
    CategoryStats,
    SessionScanner,
    SignalDetector,
    PreferenceCategorizer,
    InsightGenerator,
)

logger = logging.getLogger("daimon.preference_learner")


class PreferenceLearner:
    """
    Main facade for the preference learning system.
    
    Orchestrates the scanning, detection, categorization, and insight
    generation pipeline to learn user preferences from Claude Code sessions.
    
    Attributes:
        scanner: SessionScanner for discovering sessions
        detector: SignalDetector for finding preference signals
        categorizer: PreferenceCategorizer for classification
        insight_gen: InsightGenerator for actionable recommendations
        signals: Accumulated preference signals
    """
    
    def __init__(
        self,
        projects_dir: Optional[Path] = None,
        enable_llm: bool = True,
    ):
        """
        Initialize the preference learner.
        
        Args:
            projects_dir: Custom Claude projects directory
            enable_llm: Whether to use LLM for detection/insights
        """
        self.scanner = SessionScanner(projects_dir)
        self.detector = SignalDetector(enable_llm=enable_llm)
        self.categorizer = PreferenceCategorizer()
        self.insight_gen = InsightGenerator(enable_llm=enable_llm)
        
        self.signals: List[PreferenceSignal] = []
    
    def scan_sessions(self, since_hours: int = 24) -> List[PreferenceSignal]:
        """
        Scan recent sessions for preference signals.
        
        Uses ActivityStore first (if available), then falls back
        to direct file scanning.
        
        Args:
            since_hours: Hours to look back
            
        Returns:
            List of detected PreferenceSignals
        """
        # Try ActivityStore first (pre-computed signals)
        activity_signals = self.scanner.scan_from_activity_store(since_hours)
        if activity_signals:
            for signal in activity_signals:
                self.signals.append(signal)
                self.categorizer.update_stats(signal)
            return self.signals
        
        # Fallback: Direct session scanning
        for session_file in self.scanner.scan_recent(since_hours):
            messages = self.scanner.load_messages(session_file)
            session_id = self.scanner.get_session_id(session_file)
            
            for signal in self.detector.analyze_session(messages, session_id):
                self.signals.append(signal)
                self.categorizer.update_stats(signal)
        
        return self.signals
    
    def get_preference_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of preferences by category.
        
        Returns:
            Dictionary mapping category to stats (approval_rate, total, trend)
        """
        return self.insight_gen.get_summary(self.categorizer.category_stats)
    
    def get_actionable_insights(
        self,
        min_signals: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable insights for CLAUDE.md updates.
        
        Uses LLM when available for semantic insights.
        
        Args:
            min_signals: Minimum signals to generate insight
            
        Returns:
            List of insight dictionaries with suggestions
        """
        insights = self.insight_gen.get_insights(
            self.categorizer.category_stats,
            min_signals=min_signals,
        )
        return [insight.to_dict() for insight in insights]
    
    def clear(self) -> None:
        """Clear all signals and statistics."""
        self.signals.clear()
        self.categorizer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall learner statistics.
        
        Returns:
            Dictionary with totals, rates, and category breakdown
        """
        stats = self.categorizer.get_stats()
        
        total_approvals = sum(
            s.get("approvals", 0) for s in stats.values()
        )
        total_rejections = sum(
            s.get("rejections", 0) for s in stats.values()
        )
        total = total_approvals + total_rejections
        
        return {
            "total_signals": total,
            "total_approvals": total_approvals,
            "total_rejections": total_rejections,
            "overall_approval_rate": (
                round(total_approvals / total, 2) if total > 0 else 0
            ),
            "categories_analyzed": len(stats),
            "signals_by_category": stats,
        }


# Re-export PreferenceSignal for backward compatibility
__all__ = ["PreferenceLearner", "PreferenceSignal"]
