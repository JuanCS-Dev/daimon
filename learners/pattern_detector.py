"""
DAIMON Pattern Detector - Behavioral Pattern Detection
=======================================================

Detects patterns in user behavior for proactive emergence.

Pattern Types:
- Temporal: "commit at 5pm daily"
- Sequential: "git status → add → commit"
- Contextual: "dark theme at night"

Research Base:
- Behavioral Pattern Mining (2025)
- Human Activity Recognition via Keystroke Dynamics
- Temporal Pattern Discovery in Time Series

Usage:
    detector = PatternDetector()
    detector.add_event({"type": "shell_command", "command": "git status", ...})
    patterns = detector.detect_patterns()

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("daimon.pattern_detector")


@dataclass
class Pattern:
    """A detected behavioral pattern."""
    
    pattern_type: str  # "temporal", "sequential", "contextual"
    description: str
    confidence: float  # 0.0 - 1.0
    occurrences: int
    last_seen: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen.isoformat(),
            "data": self.data,
        }


@dataclass
class Event:
    """A behavioral event for pattern analysis."""
    
    event_type: str  # "shell_command", "window_focus", "keystroke", etc.
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


class PatternDetector:
    """
    Detects patterns in user behavior.
    
    Analyzes:
    - Temporal patterns: recurring actions at specific times
    - Sequential patterns: command sequences (n-grams)
    - Contextual patterns: correlations between context and behavior
    
    Features:
    - Rolling window analysis (last 1000 events)
    - Confidence-based pattern scoring
    - Pattern aging (older patterns decay)
    """
    
    # Minimum occurrences to consider a pattern
    MIN_OCCURRENCES = 3
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.6
    LOW_CONFIDENCE = 0.4
    
    # Time windows
    HOUR_WINDOW = 2  # Hours around pattern time
    DAY_WINDOW = 1   # Days of week
    
    def __init__(
        self,
        max_events: int = 1000,
        pattern_decay_hours: float = 24.0,
    ):
        """
        Initialize pattern detector.
        
        Args:
            max_events: Maximum events to keep in memory
            pattern_decay_hours: Hours before patterns start decaying
        """
        self.max_events = max_events
        self.pattern_decay_hours = pattern_decay_hours
        
        # Event storage
        self._events: List[Event] = []
        
        # Pattern caches
        self._temporal_patterns: Dict[str, Dict] = defaultdict(
            lambda: {"occurrences": 0, "hours": [], "weekdays": []}
        )
        self._sequential_patterns: Dict[str, Dict] = defaultdict(
            lambda: {"occurrences": 0, "last_seen": None}
        )
        self._contextual_patterns: Dict[str, Dict] = defaultdict(
            lambda: {"occurrences": 0, "contexts": []}
        )
        
        # Recent command sequence (for n-gram detection)
        self._recent_commands: List[str] = []
        self._max_sequence_length = 5
        
    def add_event(self, event_dict: Dict[str, Any]) -> None:
        """
        Add a behavioral event for analysis.
        
        Args:
            event_dict: Event data with at least 'type' key
        """
        event_type = event_dict.get("type", event_dict.get("event_type", "unknown"))
        timestamp = event_dict.get("timestamp")
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        event = Event(
            event_type=event_type,
            timestamp=timestamp,
            data=event_dict,
        )
        
        self._events.append(event)
        
        # Trim if over limit
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
        
        # Update pattern caches
        self._update_temporal_cache(event)
        self._update_sequential_cache(event)
        self._update_contextual_cache(event)
        
        logger.debug("Added event: %s at %s", event_type, timestamp)
    
    def _update_temporal_cache(self, event: Event) -> None:
        """Update temporal pattern cache."""
        key = event.event_type
        hour = event.timestamp.hour
        weekday = event.timestamp.weekday()
        
        self._temporal_patterns[key]["occurrences"] += 1
        self._temporal_patterns[key]["hours"].append(hour)
        self._temporal_patterns[key]["weekdays"].append(weekday)
        
        # Keep only last 100 entries
        if len(self._temporal_patterns[key]["hours"]) > 100:
            self._temporal_patterns[key]["hours"] = \
                self._temporal_patterns[key]["hours"][-100:]
            self._temporal_patterns[key]["weekdays"] = \
                self._temporal_patterns[key]["weekdays"][-100:]
    
    def _update_sequential_cache(self, event: Event) -> None:
        """Update sequential pattern cache."""
        # Only track shell commands for sequences
        if event.event_type not in ("shell_command", "command"):
            return
        
        command = event.data.get("command", "")
        if not command:
            return
        
        # Normalize command (first word only)
        command = command.split()[0] if command else ""
        
        self._recent_commands.append(command)
        if len(self._recent_commands) > self._max_sequence_length:
            self._recent_commands.pop(0)
        
        # Generate n-grams (2 and 3 length)
        for n in (2, 3):
            if len(self._recent_commands) >= n:
                ngram = " → ".join(self._recent_commands[-n:])
                self._sequential_patterns[ngram]["occurrences"] += 1
                self._sequential_patterns[ngram]["last_seen"] = event.timestamp
    
    def _update_contextual_cache(self, event: Event) -> None:
        """Update contextual pattern cache."""
        # Build context key
        context_parts = []
        
        # App context
        app = event.data.get("app", event.data.get("window_title", ""))
        if app:
            context_parts.append(f"app:{app[:20]}")
        
        # Time of day context
        hour = event.timestamp.hour
        if hour < 6:
            time_context = "night"
        elif hour < 12:
            time_context = "morning"
        elif hour < 18:
            time_context = "afternoon"
        else:
            time_context = "evening"
        context_parts.append(f"time:{time_context}")
        
        if context_parts:
            context_key = f"{event.event_type}|{'+'.join(context_parts)}"
            self._contextual_patterns[context_key]["occurrences"] += 1
            self._contextual_patterns[context_key]["contexts"].append({
                "timestamp": event.timestamp,
                "data": event.data,
            })
            
            # Keep only last 50 contexts
            if len(self._contextual_patterns[context_key]["contexts"]) > 50:
                self._contextual_patterns[context_key]["contexts"] = \
                    self._contextual_patterns[context_key]["contexts"][-50:]
    
    def detect_patterns(self) -> List[Pattern]:
        """
        Detect all patterns from accumulated events.
        
        Returns:
            List of detected patterns sorted by confidence
        """
        patterns = []
        
        # Detect temporal patterns
        patterns.extend(self._detect_temporal_patterns())
        
        # Detect sequential patterns
        patterns.extend(self._detect_sequential_patterns())
        
        # Detect contextual patterns
        patterns.extend(self._detect_contextual_patterns())
        
        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return patterns
    
    def _detect_temporal_patterns(self) -> List[Pattern]:
        """Detect temporal patterns (recurring times)."""
        patterns = []
        
        for event_type, data in self._temporal_patterns.items():
            if data["occurrences"] < self.MIN_OCCURRENCES:
                continue
            
            hours = data["hours"]
            weekdays = data["weekdays"]
            
            # Find peak hour (most common)
            if hours:
                hour_counts = defaultdict(int)
                for h in hours:
                    hour_counts[h] += 1
                
                peak_hour, peak_count = max(hour_counts.items(), key=lambda x: x[1])
                hour_ratio = peak_count / len(hours)
                
                if hour_ratio >= 0.3 and peak_count >= self.MIN_OCCURRENCES:
                    confidence = min(hour_ratio * 1.5, 1.0)
                    patterns.append(Pattern(
                        pattern_type="temporal",
                        description=f"{event_type} typically occurs around {peak_hour}:00",
                        confidence=confidence,
                        occurrences=peak_count,
                        last_seen=datetime.now(),
                        data={
                            "event_type": event_type,
                            "peak_hour": peak_hour,
                            "hour_distribution": dict(hour_counts),
                        },
                    ))
            
            # Find peak weekday
            if weekdays:
                day_counts = defaultdict(int)
                for d in weekdays:
                    day_counts[d] += 1
                
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                peak_day, peak_count = max(day_counts.items(), key=lambda x: x[1])
                day_ratio = peak_count / len(weekdays)
                
                if day_ratio >= 0.25 and peak_count >= self.MIN_OCCURRENCES:
                    confidence = min(day_ratio * 1.4, 1.0)
                    patterns.append(Pattern(
                        pattern_type="temporal",
                        description=f"{event_type} peaks on {day_names[peak_day]}s",
                        confidence=confidence,
                        occurrences=peak_count,
                        last_seen=datetime.now(),
                        data={
                            "event_type": event_type,
                            "peak_weekday": peak_day,
                            "weekday_name": day_names[peak_day],
                        },
                    ))
        
        return patterns
    
    def _detect_sequential_patterns(self) -> List[Pattern]:
        """Detect sequential patterns (command sequences)."""
        patterns = []
        
        for sequence, data in self._sequential_patterns.items():
            occurrences = data["occurrences"]
            
            if occurrences < self.MIN_OCCURRENCES:
                continue
            
            # Confidence based on frequency
            confidence = min(occurrences / 10, 1.0)
            
            patterns.append(Pattern(
                pattern_type="sequential",
                description=f"Sequence: {sequence}",
                confidence=confidence,
                occurrences=occurrences,
                last_seen=data["last_seen"] or datetime.now(),
                data={
                    "sequence": sequence,
                    "commands": sequence.split(" → "),
                },
            ))
        
        return patterns
    
    def _detect_contextual_patterns(self) -> List[Pattern]:
        """Detect contextual patterns (environment correlations)."""
        patterns = []
        
        for context_key, data in self._contextual_patterns.items():
            occurrences = data["occurrences"]
            
            if occurrences < self.MIN_OCCURRENCES:
                continue
            
            # Parse context key
            parts = context_key.split("|")
            event_type = parts[0]
            context = parts[1] if len(parts) > 1 else ""
            
            # Confidence based on frequency
            confidence = min(occurrences / 15, 1.0)
            
            patterns.append(Pattern(
                pattern_type="contextual",
                description=f"{event_type} in context: {context}",
                confidence=confidence,
                occurrences=occurrences,
                last_seen=datetime.now(),
                data={
                    "event_type": event_type,
                    "context": context,
                },
            ))
        
        return patterns
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """
        Get patterns of a specific type.
        
        Args:
            pattern_type: "temporal", "sequential", or "contextual"
            
        Returns:
            List of patterns of that type
        """
        all_patterns = self.detect_patterns()
        return [p for p in all_patterns if p.pattern_type == pattern_type]
    
    def get_matching_patterns(
        self,
        current_context: Dict[str, Any],
    ) -> List[Pattern]:
        """
        Get patterns that match current context.
        
        Args:
            current_context: Current user context (hour, app, etc.)
            
        Returns:
            Patterns relevant to current context
        """
        all_patterns = self.detect_patterns()
        matching = []
        
        current_hour = current_context.get("hour", datetime.now().hour)
        current_weekday = current_context.get("weekday", datetime.now().weekday())
        
        for pattern in all_patterns:
            if pattern.pattern_type == "temporal":
                # Check if current time matches pattern
                peak_hour = pattern.data.get("peak_hour")
                if peak_hour is not None:
                    if abs(current_hour - peak_hour) <= self.HOUR_WINDOW:
                        matching.append(pattern)
                        continue
                
                peak_weekday = pattern.data.get("peak_weekday")
                if peak_weekday is not None:
                    if current_weekday == peak_weekday:
                        matching.append(pattern)
                        continue
            
            elif pattern.pattern_type == "contextual":
                # Check if context matches
                pattern_context = pattern.data.get("context", "")
                
                current_app = current_context.get("app", "")
                if current_app and current_app.lower() in pattern_context.lower():
                    matching.append(pattern)
                    continue
                
                # Time of day context
                if current_hour < 6:
                    time_ctx = "night"
                elif current_hour < 12:
                    time_ctx = "morning"
                elif current_hour < 18:
                    time_ctx = "afternoon"
                else:
                    time_ctx = "evening"
                
                if f"time:{time_ctx}" in pattern_context:
                    matching.append(pattern)
        
        return matching
    
    def clear(self) -> None:
        """Clear all events and patterns."""
        self._events.clear()
        self._temporal_patterns.clear()
        self._sequential_patterns.clear()
        self._contextual_patterns.clear()
        self._recent_commands.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        patterns = self.detect_patterns()
        return {
            "total_events": len(self._events),
            "max_events": self.max_events,
            "total_patterns": len(patterns),
            "patterns_by_type": {
                "temporal": len([p for p in patterns if p.pattern_type == "temporal"]),
                "sequential": len([p for p in patterns if p.pattern_type == "sequential"]),
                "contextual": len([p for p in patterns if p.pattern_type == "contextual"]),
            },
            "recent_commands": self._recent_commands[-5:],
        }


# Singleton instance
_detector: Optional[PatternDetector] = None


def get_pattern_detector() -> PatternDetector:
    """Get singleton pattern detector instance."""
    global _detector
    if _detector is None:
        _detector = PatternDetector()
    return _detector


def reset_pattern_detector() -> None:
    """Reset singleton pattern detector."""
    global _detector
    if _detector:
        _detector.clear()
    _detector = None

