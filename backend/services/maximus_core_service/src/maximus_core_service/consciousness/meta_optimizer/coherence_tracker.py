"""
Coherence Tracker - Internal Quality Metrics
=============================================

Tracks the coherence and quality of Noesis's outputs over time.
Implements Meta-Reflector validation as internal feedback loop.

Key Metrics:
- Response coherence (Kuramoto-derived)
- Processing latency
- Error rate
- Self-assessment scores

Usage:
    tracker = CoherenceTracker()
    
    # Record after each response
    tracker.record(
        coherence=0.85,
        latency_ms=1200,
        was_successful=True
    )
    
    # Get trend analysis
    trends = tracker.get_trends(window_size=100)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoherenceSnapshot:
    """A single coherence measurement."""
    timestamp: float
    coherence: float  # 0.0 - 1.0
    latency_ms: float
    was_successful: bool
    source: str = "unknown"
    depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality_score(self) -> float:
        """Composite quality score combining coherence and latency."""
        # Penalize high latency (target: <5000ms)
        latency_penalty = max(0, 1.0 - (self.latency_ms / 10000))
        # Success multiplier
        success_mult = 1.0 if self.was_successful else 0.5
        return self.coherence * latency_penalty * success_mult


@dataclass
class TrendAnalysis:
    """Analysis of coherence trends over a window."""
    window_size: int
    avg_coherence: float
    avg_latency_ms: float
    success_rate: float
    avg_quality: float
    trend_direction: str  # "improving", "stable", "degrading"
    improvement_pct: float  # vs previous window


class CoherenceTracker:
    """
    Tracks and analyzes coherence metrics over time.
    
    Implements the Meta-Reflector validation pattern for
    continuous self-assessment.
    """
    
    DEFAULT_WINDOW_SIZE = 100
    MAX_HISTORY_SIZE = 10000
    
    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        low_coherence_threshold: float = 0.55,
        target_latency_ms: float = 5000
    ):
        self.window_size = window_size
        self.low_coherence_threshold = low_coherence_threshold
        self.target_latency_ms = target_latency_ms
        
        self._history: Deque[CoherenceSnapshot] = deque(maxlen=self.MAX_HISTORY_SIZE)
        self._low_coherence_count: int = 0
        self._total_count: int = 0
        
        # Prometheus-compatible counters
        self._metrics: Dict[str, float] = {
            "total_recordings": 0,
            "low_coherence_events": 0,
            "high_latency_events": 0,
            "failures": 0,
        }
    
    def record(
        self,
        coherence: float,
        latency_ms: float,
        was_successful: bool = True,
        source: str = "unknown",
        depth: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CoherenceSnapshot:
        """
        Record a coherence measurement.
        
        Args:
            coherence: Kuramoto coherence value (0-1)
            latency_ms: Processing time in milliseconds
            was_successful: Whether the operation succeeded
            source: Source module identifier
            depth: Processing depth (1-5)
            metadata: Additional context
            
        Returns:
            The recorded CoherenceSnapshot
        """
        snapshot = CoherenceSnapshot(
            timestamp=time.time(),
            coherence=coherence,
            latency_ms=latency_ms,
            was_successful=was_successful,
            source=source,
            depth=depth,
            metadata=metadata or {}
        )
        
        self._history.append(snapshot)
        self._total_count += 1
        self._metrics["total_recordings"] += 1
        
        # Track anomalies
        if coherence < self.low_coherence_threshold:
            self._low_coherence_count += 1
            self._metrics["low_coherence_events"] += 1
            logger.warning(
                "[Meta] Low coherence detected: %.3f (source: %s)",
                coherence, source
            )
        
        if latency_ms > self.target_latency_ms:
            self._metrics["high_latency_events"] += 1
        
        if not was_successful:
            self._metrics["failures"] += 1
        
        return snapshot
    
    def get_trends(self, window_size: Optional[int] = None) -> TrendAnalysis:
        """
        Analyze coherence trends over a window.
        
        Args:
            window_size: Number of recent snapshots to analyze
            
        Returns:
            TrendAnalysis with aggregated metrics
        """
        ws = window_size or self.window_size
        
        if len(self._history) == 0:
            return TrendAnalysis(
                window_size=ws,
                avg_coherence=0.0,
                avg_latency_ms=0.0,
                success_rate=0.0,
                avg_quality=0.0,
                trend_direction="stable",
                improvement_pct=0.0
            )
        
        # Get recent window
        recent = list(self._history)[-ws:]
        
        avg_coherence = sum(s.coherence for s in recent) / len(recent)
        avg_latency = sum(s.latency_ms for s in recent) / len(recent)
        success_rate = sum(1 for s in recent if s.was_successful) / len(recent)
        avg_quality = sum(s.quality_score for s in recent) / len(recent)
        
        # Compare with previous window
        if len(self._history) >= ws * 2:
            previous = list(self._history)[-(ws * 2):-ws]
            prev_quality = sum(s.quality_score for s in previous) / len(previous)
            improvement_pct = ((avg_quality - prev_quality) / prev_quality) * 100 if prev_quality > 0 else 0
            
            if improvement_pct > 2:
                direction = "improving"
            elif improvement_pct < -2:
                direction = "degrading"
            else:
                direction = "stable"
        else:
            improvement_pct = 0.0
            direction = "stable"
        
        return TrendAnalysis(
            window_size=ws,
            avg_coherence=avg_coherence,
            avg_latency_ms=avg_latency,
            success_rate=success_rate,
            avg_quality=avg_quality,
            trend_direction=direction,
            improvement_pct=improvement_pct
        )
    
    def get_recent_snapshots(self, count: int = 10) -> List[CoherenceSnapshot]:
        """Get the N most recent snapshots."""
        return list(self._history)[-count:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics for Prometheus."""
        trends = self.get_trends()
        return {
            **self._metrics,
            "history_size": len(self._history),
            "avg_coherence": trends.avg_coherence,
            "avg_latency_ms": trends.avg_latency_ms,
            "success_rate": trends.success_rate,
            "avg_quality": trends.avg_quality,
            "trend_direction": trends.trend_direction,
        }
    
    def should_trigger_optimization(self) -> bool:
        """
        Determine if coherence patterns suggest optimization is needed.
        
        Returns True if:
        - Success rate drops below 80%
        - Average coherence drops below threshold
        - Trend is degrading for multiple windows
        """
        if len(self._history) < 20:
            return False  # Not enough data
        
        trends = self.get_trends()
        
        return (
            trends.success_rate < 0.80 or
            trends.avg_coherence < self.low_coherence_threshold or
            (trends.trend_direction == "degrading" and trends.improvement_pct < -5)
        )


# Global instance
_global_tracker: Optional[CoherenceTracker] = None


def get_coherence_tracker() -> CoherenceTracker:
    """Get or create the global coherence tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CoherenceTracker()
    return _global_tracker
