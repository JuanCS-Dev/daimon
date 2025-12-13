"""
DAIMON Metacognitive Engine - Reflection on Reflection
=======================================================

Second-order reflection system that analyzes the effectiveness
of past insights and adjusts learning strategies.

Research Base:
- SOFAI Architecture - Self-improving AI through reflection
- Metacognitive AI 2025 - Agents that reason about their reasoning
- AI Companions ACM 2025 - Personal knowledge assistants

Features:
- Track insight effectiveness over time
- Measure behavior change after insights
- Adjust confidence thresholds dynamically
- Identify which insight categories are most useful

Usage:
    metacog = MetacognitiveEngine()
    metacog.log_insight(insight, timestamp)
    effectiveness = metacog.analyze_effectiveness()
    recommendations = metacog.get_learning_recommendations()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metacognitive_models import (
    InsightRecord,
    CategoryEffectiveness,
    MetacognitiveAnalysis,
)
from .metacognitive_analysis import generate_recommendations, generate_adjustments

logger = logging.getLogger("daimon.metacognitive")

# Configuration
DEFAULT_LEDGER_PATH = Path.home() / ".daimon" / "metacognitive"
INSIGHT_HISTORY_DAYS = 90
EFFECTIVENESS_WINDOW_HOURS = 24
MIN_INSIGHTS_FOR_ANALYSIS = 5


class MetacognitiveEngine:
    """
    Second-order reflection engine.

    Tracks insight history and analyzes which types of insights
    actually lead to behavior improvement.
    """

    def __init__(
        self,
        ledger_path: Optional[Path] = None,
        history_days: int = INSIGHT_HISTORY_DAYS,
    ):
        """
        Initialize metacognitive engine.

        Args:
            ledger_path: Path to store insight history.
            history_days: Number of days to keep insight history.
        """
        self.ledger_path = ledger_path or DEFAULT_LEDGER_PATH
        self.history_days = history_days

        # In-memory insight history
        self._insights: List[InsightRecord] = []

        # Ensure storage directory exists
        self.ledger_path.mkdir(parents=True, exist_ok=True)

        # Load existing history
        self._load_history()

    def _load_history(self) -> None:
        """Load insight history from disk."""
        history_file = self.ledger_path / "insight_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cutoff = datetime.now() - timedelta(days=self.history_days)

            for record_data in data:
                record = InsightRecord.from_dict(record_data)
                if record.timestamp > cutoff:
                    self._insights.append(record)

            logger.debug("Loaded %d insight records", len(self._insights))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load insight history: %s", e)

    def _save_history(self) -> None:
        """Save insight history to disk."""
        history_file = self.ledger_path / "insight_history.json"

        # Prune old records
        cutoff = datetime.now() - timedelta(days=self.history_days)
        self._insights = [i for i in self._insights if i.timestamp > cutoff]

        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump([i.to_dict() for i in self._insights], f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to save insight history: %s", e)

    def log_insight(
        self,
        insight: Dict[str, Any],
        was_applied: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an insight from the reflection engine.

        Args:
            insight: Insight dictionary from ReflectionEngine.
            was_applied: Whether the insight was applied to CLAUDE.md.
            context: Additional context about when the insight was generated.

        Returns:
            Insight ID for tracking.
        """
        import hashlib

        timestamp = datetime.now()
        insight_str = f"{insight.get('category', '')}{insight.get('suggestion', '')}{timestamp.isoformat()}"
        insight_id = hashlib.sha256(insight_str.encode()).hexdigest()[:12]

        record = InsightRecord(
            insight_id=insight_id,
            category=insight.get("category", "unknown"),
            action=insight.get("action", "add"),
            confidence=insight.get("confidence", 0.5),
            suggestion=insight.get("suggestion", ""),
            timestamp=timestamp,
            context=context or {},
            was_applied=was_applied,
        )

        # Capture current metrics snapshot for later effectiveness measurement
        metrics_snapshot = self._capture_metrics_snapshot()
        record.context["metrics_at_log"] = metrics_snapshot

        self._insights.append(record)
        self._save_history()

        # Auto-measure old insights
        self._auto_measure_old_insights()

        logger.debug("Logged insight %s: %s", insight_id, record.category)
        return insight_id

    def _capture_metrics_snapshot(self) -> Dict[str, Any]:
        """Capture current preference signal metrics for effectiveness tracking."""
        try:
            from memory.activity_store import get_activity_store
            store = get_activity_store()

            # Get recent claude events for signal counting
            recent = store.get_recent(watcher_type="claude", hours=24)
            signals_count = len(recent)

            # Calculate approval rate from events
            approvals = sum(1 for r in recent if r.data.get("preference_signal") == "approval")
            approval_rate = approvals / max(signals_count, 1)

            return {
                "signals_count": signals_count,
                "approval_rate": approval_rate,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception:
            return {"signals_count": 0, "approval_rate": 0.5, "timestamp": datetime.now().isoformat()}

    def _auto_measure_old_insights(self) -> None:
        """Auto-measure effectiveness of insights older than 24 hours."""
        cutoff = datetime.now() - timedelta(hours=24)
        current_metrics = self._capture_metrics_snapshot()

        unmeasured = [
            i for i in self._insights
            if i.timestamp < cutoff and i.effectiveness_score is None and i.was_applied
        ]

        for record in unmeasured[:3]:  # Limit to 3 per cycle to avoid overload
            try:
                baseline = record.context.get("metrics_at_log", {})
                if not baseline:
                    continue

                # Measure effectiveness
                record.effectiveness_score = self._calculate_effectiveness(
                    signals_before=baseline.get("signals_count", 0),
                    signals_after=current_metrics["signals_count"],
                    approval_before=baseline.get("approval_rate", 0.5),
                    approval_after=current_metrics["approval_rate"],
                )
                logger.debug("Auto-measured insight %s: %.2f", record.insight_id, record.effectiveness_score)
            except Exception as e:
                logger.debug("Failed to measure insight %s: %s", record.insight_id, e)

        if unmeasured:
            self._save_history()

    def _calculate_effectiveness(
        self,
        signals_before: int,
        signals_after: int,
        approval_before: float,
        approval_after: float,
    ) -> float:
        """Calculate effectiveness score from before/after metrics."""
        # Signal improvement (more signals = more data)
        signal_factor = 0.0
        if signals_before > 0:
            signal_change = (signals_after - signals_before) / signals_before
            signal_factor = min(1.0, max(0.0, 0.5 + signal_change * 0.5))
        elif signals_after > 0:
            signal_factor = 0.7  # New signals is good

        # Approval rate improvement
        approval_change = approval_after - approval_before
        approval_factor = min(1.0, max(0.0, 0.5 + approval_change))

        # Combined score
        return (signal_factor * 0.3 + approval_factor * 0.7)

    def measure_effectiveness(
        self,
        insight_id: str,
        signals_before: int,
        signals_after: int,
        approval_rate_before: float,
        approval_rate_after: float,
    ) -> float:
        """
        Measure effectiveness of an insight based on before/after metrics.

        Args:
            insight_id: ID of the insight to measure.
            signals_before: Signal count before insight was applied.
            signals_after: Signal count after insight was applied.
            approval_rate_before: Approval rate before.
            approval_rate_after: Approval rate after.

        Returns:
            Effectiveness score (0-1, higher = more effective).
        """
        # Find the insight record
        record = None
        for r in self._insights:
            if r.insight_id == insight_id:
                record = r
                break

        if record is None:
            logger.warning("Insight %s not found", insight_id)
            return 0.0

        # Calculate effectiveness
        # Improvement in approval rate is the primary metric
        approval_improvement = approval_rate_after - approval_rate_before

        # Normalize to 0-1 scale
        # +0.2 improvement = 1.0 effectiveness
        # No change = 0.5 effectiveness
        # -0.2 decline = 0.0 effectiveness
        effectiveness = 0.5 + (approval_improvement * 2.5)
        effectiveness = max(0.0, min(1.0, effectiveness))

        # Update record
        record.effectiveness_score = effectiveness
        record.measured_at = datetime.now()
        self._save_history()

        logger.debug(
            "Measured effectiveness for %s: %.2f (approval: %.2f -> %.2f)",
            insight_id,
            effectiveness,
            approval_rate_before,
            approval_rate_after,
        )

        return effectiveness

    def analyze_effectiveness(self) -> MetacognitiveAnalysis:
        """
        Analyze overall effectiveness of insights.

        Returns:
            MetacognitiveAnalysis with breakdown by category and recommendations.
        """
        # Group insights by category
        by_category: Dict[str, List[InsightRecord]] = defaultdict(list)
        for insight in self._insights:
            by_category[insight.category].append(insight)

        # Calculate effectiveness for each category
        category_breakdown: Dict[str, CategoryEffectiveness] = {}

        for category, insights in by_category.items():
            eff = CategoryEffectiveness(category=category)
            eff.total_insights = len(insights)
            eff.applied_insights = sum(1 for i in insights if i.was_applied)

            # Calculate average effectiveness for measured insights
            measured = [
                i for i in insights if i.effectiveness_score is not None
            ]
            if measured:
                eff.effective_insights = sum(
                    1 for i in measured if i.effectiveness_score > 0.6
                )
                eff.avg_effectiveness = sum(
                    i.effectiveness_score for i in measured
                ) / len(measured)

                # Calculate trend (compare older vs newer)
                if len(measured) >= 4:
                    half = len(measured) // 2
                    older_avg = sum(
                        i.effectiveness_score for i in measured[:half]
                    ) / half
                    newer_avg = sum(
                        i.effectiveness_score for i in measured[half:]
                    ) / (len(measured) - half)

                    if newer_avg > older_avg + 0.1:
                        eff.trend = "improving"
                    elif newer_avg < older_avg - 0.1:
                        eff.trend = "declining"

            category_breakdown[category] = eff

        # Calculate overall effectiveness
        all_measured = [
            i for i in self._insights if i.effectiveness_score is not None
        ]
        overall_effectiveness = 0.5
        if all_measured:
            overall_effectiveness = sum(
                i.effectiveness_score for i in all_measured
            ) / len(all_measured)

        # Generate recommendations
        recommendations = generate_recommendations(category_breakdown)

        # Generate adjustment suggestions
        adjustment_suggestions = generate_adjustments(category_breakdown)

        return MetacognitiveAnalysis(
            analyzed_at=datetime.now(),
            insights_analyzed=len(self._insights),
            categories_analyzed=len(category_breakdown),
            overall_effectiveness=overall_effectiveness,
            category_breakdown=category_breakdown,
            recommendations=recommendations,
            adjustment_suggestions=adjustment_suggestions,
        )

    def get_learning_recommendations(self) -> List[str]:
        """Get current learning recommendations."""
        analysis = self.analyze_effectiveness()
        return analysis.recommendations

    def reflect_on_reflection(self) -> Dict[str, Any]:
        """
        Meta-analysis of reflection effectiveness.

        This is the second-order reflection - reflecting on how well
        our reflection process is working.

        Returns:
            Dict with meta-analysis results.
        """
        analysis = self.analyze_effectiveness()

        # Calculate meta-metrics
        measured_ratio = 0.0
        if self._insights:
            measured_count = sum(
                1 for i in self._insights if i.effectiveness_score is not None
            )
            measured_ratio = measured_count / len(self._insights)

        # Overall assessment
        if analysis.overall_effectiveness > 0.7:
            assessment = "excellent"
            message = "Reflection process is highly effective"
        elif analysis.overall_effectiveness > 0.5:
            assessment = "good"
            message = "Reflection process is working well with room for improvement"
        elif analysis.overall_effectiveness > 0.3:
            assessment = "moderate"
            message = "Reflection process needs optimization"
        else:
            assessment = "poor"
            message = "Reflection process may need significant changes"

        return {
            "assessment": assessment,
            "message": message,
            "overall_effectiveness": analysis.overall_effectiveness,
            "insights_analyzed": analysis.insights_analyzed,
            "measurement_ratio": measured_ratio,
            "effective_categories": [
                cat
                for cat, eff in analysis.category_breakdown.items()
                if eff.avg_effectiveness > 0.6
            ],
            "recommendations": analysis.recommendations,
            "adjustment_suggestions": analysis.adjustment_suggestions,
            "analyzed_at": analysis.analyzed_at.isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get metacognitive engine statistics."""
        measured_count = sum(
            1 for i in self._insights if i.effectiveness_score is not None
        )

        return {
            "total_insights_tracked": len(self._insights),
            "measured_insights": measured_count,
            "history_days": self.history_days,
            "ledger_path": str(self.ledger_path),
            "categories_tracked": list(
                set(i.category for i in self._insights)
            ),
        }

    def clear_history(self) -> None:
        """Clear all insight history."""
        self._insights.clear()
        self._save_history()
        logger.info("Cleared metacognitive history")


# Singleton instance
_metacog: Optional[MetacognitiveEngine] = None


def get_metacognitive_engine() -> MetacognitiveEngine:
    """Get global metacognitive engine instance."""
    global _metacog
    if _metacog is None:
        _metacog = MetacognitiveEngine()
    return _metacog


def reset_metacognitive_engine() -> None:
    """Reset global metacognitive engine instance."""
    global _metacog
    _metacog = None
