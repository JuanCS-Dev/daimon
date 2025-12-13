#!/usr/bin/env python3
"""
DAIMON Style Learner - Communication Style Inference
====================================================

Analyzes activity patterns from collectors to infer communication style.
Uses keystroke dynamics, work patterns, and activity data to understand
how the user prefers to work and communicate.

This feeds into CLAUDE.md to personalize Claude's responses.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

from .style_models import (
    TypingPace,
    EditingFrequency,
    InteractionPattern,
    FocusLevel,
    CommunicationStyle,
)

logger = logging.getLogger("daimon.style")


class StyleLearner:
    """
    Learns communication style from activity data.

    Analyzes:
    - Keystroke dynamics (typing patterns)
    - Window focus patterns
    - AFK periods (break patterns)
    - Session timing (work hours)

    Does NOT analyze content - only patterns and timing.
    """

    def __init__(self):
        self._style = CommunicationStyle()
        self._keystroke_samples: List[Dict[str, Any]] = []
        self._window_samples: List[Dict[str, Any]] = []
        self._afk_samples: List[Dict[str, Any]] = []
        self._session_hours: List[int] = []
        # NEW: Shell and Claude data for full data coverage
        self._shell_samples: List[Dict[str, Any]] = []
        self._claude_samples: List[Dict[str, Any]] = []

    def add_keystroke_sample(self, dynamics: Dict[str, Any]) -> None:
        """
        Add keystroke dynamics sample.

        Args:
            dynamics: Dict with typing_speed_cpm, pause_count, burst_count, etc.
        """
        self._keystroke_samples.append({
            "timestamp": datetime.now(),
            **dynamics,
        })

        # Keep only last 100 samples
        if len(self._keystroke_samples) > 100:
            self._keystroke_samples = self._keystroke_samples[-100:]

    def add_window_sample(self, window_data: Dict[str, Any]) -> None:
        """
        Add window focus sample.

        Args:
            window_data: Dict with app_name, focus_duration, etc.
        """
        self._window_samples.append({
            "timestamp": datetime.now(),
            **window_data,
        })

        # Record session hour
        hour = datetime.now().hour
        self._session_hours.append(hour)

        # Keep only last 500 samples
        if len(self._window_samples) > 500:
            self._window_samples = self._window_samples[-500:]
        if len(self._session_hours) > 1000:
            self._session_hours = self._session_hours[-1000:]

    def add_afk_sample(self, afk_data: Dict[str, Any]) -> None:
        """
        Add AFK period sample.

        Args:
            afk_data: Dict with event (afk_start/afk_end), idle_seconds, etc.
        """
        self._afk_samples.append({
            "timestamp": datetime.now(),
            **afk_data,
        })

        # Keep only last 100 samples
        if len(self._afk_samples) > 100:
            self._afk_samples = self._afk_samples[-100:]

    def add_shell_sample(self, shell_data: Dict[str, Any]) -> None:
        """
        Add shell command sample for work intensity analysis.

        Args:
            shell_data: Dict with command, exit_code, duration, etc.
        """
        self._shell_samples.append({
            "timestamp": datetime.now(),
            **shell_data,
        })

        # Record session hour
        hour = datetime.now().hour
        self._session_hours.append(hour)

        # Keep only last 200 samples
        if len(self._shell_samples) > 200:
            self._shell_samples = self._shell_samples[-200:]

    def add_claude_sample(self, claude_data: Dict[str, Any]) -> None:
        """
        Add Claude session event sample.

        Args:
            claude_data: Dict with intention, preference_signal, etc.
        """
        self._claude_samples.append({
            "timestamp": datetime.now(),
            **claude_data,
        })

        # Record session hour
        hour = datetime.now().hour
        self._session_hours.append(hour)

        # Keep only last 200 samples
        if len(self._claude_samples) > 200:
            self._claude_samples = self._claude_samples[-200:]

    def compute_style(self) -> CommunicationStyle:
        """
        Compute communication style from collected samples.

        Returns:
            CommunicationStyle with inferred preferences.
        """
        style = CommunicationStyle(last_updated=datetime.now())
        confidence_factors: List[float] = []

        # Analyze each dimension
        self._analyze_typing(style, confidence_factors)
        self._analyze_focus(style, confidence_factors)
        self._analyze_work_hours(style, confidence_factors)
        self._analyze_afk_pattern(style, confidence_factors)
        self._analyze_shell_intensity(style, confidence_factors)
        self._analyze_claude_patterns(style, confidence_factors)

        # Calculate overall confidence
        style.confidence = statistics.mean(confidence_factors) if confidence_factors else 0.0
        self._style = style
        return style

    def _analyze_typing(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """Analyze typing pace and editing frequency from keystroke samples."""
        if not self._keystroke_samples:
            return

        # Typing pace
        cpms = [s.get("typing_speed_cpm", 0) for s in self._keystroke_samples]
        valid_cpms = [c for c in cpms if c > 0]
        if valid_cpms:
            avg_cpm = statistics.mean(valid_cpms)
            if avg_cpm > 300:
                style.typing_pace = TypingPace.FAST
            elif avg_cpm > 150:
                style.typing_pace = TypingPace.MODERATE
            else:
                style.typing_pace = TypingPace.DELIBERATE
            confidence.append(min(1.0, len(valid_cpms) / 20))

        # Editing frequency
        pause_ratios = []
        for sample in self._keystroke_samples:
            total = sample.get("total_keystrokes", 1)
            pauses = sample.get("pause_count", 0)
            if total > 0:
                pause_ratios.append(pauses / total)

        if pause_ratios:
            avg_ratio = statistics.mean(pause_ratios)
            if avg_ratio < 0.1:
                style.editing_frequency = EditingFrequency.MINIMAL
            elif avg_ratio < 0.25:
                style.editing_frequency = EditingFrequency.MODERATE
            else:
                style.editing_frequency = EditingFrequency.HEAVY

    def _analyze_focus(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """Analyze focus level from window samples."""
        if not self._window_samples:
            return

        durations = [s.get("focus_duration", 0) for s in self._window_samples]
        valid_durations = [d for d in durations if d > 0]

        if valid_durations:
            avg = statistics.mean(valid_durations)
            if avg > 300:  # >5 min average focus
                style.focus_level = FocusLevel.DEEP
            elif avg > 60:  # 1-5 min
                style.focus_level = FocusLevel.SWITCHING
            else:
                style.focus_level = FocusLevel.MULTITASK
            confidence.append(min(1.0, len(valid_durations) / 50))

    def _analyze_work_hours(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """Analyze work hours from session data."""
        if not self._session_hours:
            return

        hour_counts: Dict[int, int] = {}
        for h in self._session_hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1

        # Find active hours (>5% of activity)
        threshold = len(self._session_hours) * 0.05
        style.work_hours = sorted([h for h, c in hour_counts.items() if c > threshold])

        # Find peak hour
        if hour_counts:
            style.peak_productivity = max(hour_counts, key=hour_counts.get)

        confidence.append(min(1.0, len(self._session_hours) / 100))

    def _analyze_afk_pattern(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """Analyze interaction pattern from AFK data."""
        if not self._afk_samples:
            return

        afk_starts = sum(1 for s in self._afk_samples if s.get("event") == "afk_start")
        sample_hours = max(1, len(self._afk_samples) / 10)
        afk_per_hour = afk_starts / sample_hours

        if afk_per_hour < 1:
            style.interaction_pattern = InteractionPattern.BURST
            style.afk_frequency = "low"
        elif afk_per_hour < 3:
            style.interaction_pattern = InteractionPattern.STEADY
            style.afk_frequency = "normal"
        else:
            style.interaction_pattern = InteractionPattern.SPORADIC
            style.afk_frequency = "high"

        confidence.append(min(1.0, len(self._afk_samples) / 20))

    def _analyze_shell_intensity(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """
        Analyze work intensity from shell command patterns.

        High error rates may indicate frustration or experimental work.
        High command frequency indicates active development.
        """
        if not self._shell_samples:
            return

        # Calculate error rate
        total_commands = len(self._shell_samples)
        error_commands = sum(
            1 for s in self._shell_samples
            if s.get("exit_code", 0) != 0
        )
        error_rate = error_commands / total_commands if total_commands > 0 else 0

        # Error rate affects editing frequency inference
        if error_rate > 0.3:  # >30% errors suggests experimental/iterative work
            style.editing_frequency = EditingFrequency.HEAVY
        elif error_rate < 0.1:  # <10% errors suggests careful work
            if style.editing_frequency == EditingFrequency.MODERATE:
                style.editing_frequency = EditingFrequency.MINIMAL

        confidence.append(min(1.0, total_commands / 30))

    def _analyze_claude_patterns(
        self,
        style: CommunicationStyle,
        confidence: List[float],
    ) -> None:
        """
        Analyze interaction patterns from Claude session events.

        Approval/rejection signals indicate communication preferences.
        """
        if not self._claude_samples:
            return

        # Count preference signals
        approvals = sum(
            1 for s in self._claude_samples
            if s.get("preference_signal") == "approval"
        )
        rejections = sum(
            1 for s in self._claude_samples
            if s.get("preference_signal") == "rejection"
        )
        total_signals = approvals + rejections

        if total_signals > 0:
            approval_rate = approvals / total_signals

            # High rejection rate suggests user prefers different approach
            if approval_rate < 0.4:  # >60% rejections
                # User is discerning - may want concise responses
                if style.typing_pace == TypingPace.FAST:
                    pass  # Already expects concise responses
                else:
                    style.editing_frequency = EditingFrequency.HEAVY

            confidence.append(min(1.0, total_signals / 15))

    def get_current_style(self) -> CommunicationStyle:
        """Get current computed style."""
        return self._style

    def get_claude_md_section(self) -> str:
        """
        Generate markdown section for CLAUDE.md.

        Returns:
            Formatted markdown with style-based suggestions.
        """
        style = self._style

        if style.confidence < 0.3:
            return ""  # Not enough data yet

        # Format timestamp
        timestamp_str = "N/A"
        if style.last_updated:
            timestamp_str = style.last_updated.strftime("%Y-%m-%d %H:%M")

        lines = [
            "## Communication Style (Inferred)",
            f"*Confidence: {style.confidence:.0%} | Last updated: {timestamp_str}*",
            "",
        ]

        suggestions = style.to_claude_suggestions()
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")

        if style.peak_productivity is not None:
            lines.append(f"- User is most active around {style.peak_productivity}:00.")

        lines.append("")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all collected samples."""
        self._keystroke_samples.clear()
        self._window_samples.clear()
        self._afk_samples.clear()
        self._session_hours.clear()
        self._shell_samples.clear()
        self._claude_samples.clear()
        self._style = CommunicationStyle()


# Singleton storage (avoids global statement)
_singleton: Dict[str, Optional[StyleLearner]] = {"learner": None}


def get_style_learner() -> StyleLearner:
    """Get global style learner instance."""
    if _singleton["learner"] is None:
        _singleton["learner"] = StyleLearner()
    return _singleton["learner"]
