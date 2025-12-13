"""
Metacognitive Engine Data Models.

Contains dataclasses for insight tracking and analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class InsightRecord:
    """Record of a generated insight."""

    insight_id: str
    category: str
    action: str  # 'add', 'reinforce', 'remove'
    confidence: float
    suggestion: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    # Effectiveness tracking (updated later)
    was_applied: bool = False
    effectiveness_score: Optional[float] = None
    measured_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "insight_id": self.insight_id,
            "category": self.category,
            "action": self.action,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "was_applied": self.was_applied,
            "effectiveness_score": self.effectiveness_score,
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightRecord":
        """Create from dictionary."""
        return cls(
            insight_id=data["insight_id"],
            category=data["category"],
            action=data["action"],
            confidence=data["confidence"],
            suggestion=data["suggestion"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            was_applied=data.get("was_applied", False),
            effectiveness_score=data.get("effectiveness_score"),
            measured_at=datetime.fromisoformat(data["measured_at"])
            if data.get("measured_at")
            else None,
        )


@dataclass
class CategoryEffectiveness:
    """Effectiveness metrics for an insight category."""

    category: str
    total_insights: int = 0
    applied_insights: int = 0
    effective_insights: int = 0
    avg_effectiveness: float = 0.0
    trend: str = "stable"  # 'improving', 'declining', 'stable'

    @property
    def application_rate(self) -> float:
        """Percentage of insights that were applied."""
        if self.total_insights == 0:
            return 0.0
        return self.applied_insights / self.total_insights

    @property
    def success_rate(self) -> float:
        """Percentage of applied insights that were effective."""
        if self.applied_insights == 0:
            return 0.0
        return self.effective_insights / self.applied_insights


@dataclass
class MetacognitiveAnalysis:
    """Results of metacognitive analysis."""

    analyzed_at: datetime
    insights_analyzed: int
    categories_analyzed: int
    overall_effectiveness: float
    category_breakdown: Dict[str, CategoryEffectiveness]
    recommendations: List[str]
    adjustment_suggestions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyzed_at": self.analyzed_at.isoformat(),
            "insights_analyzed": self.insights_analyzed,
            "categories_analyzed": self.categories_analyzed,
            "overall_effectiveness": self.overall_effectiveness,
            "category_breakdown": {
                cat: {
                    "total": eff.total_insights,
                    "applied": eff.applied_insights,
                    "effective": eff.effective_insights,
                    "avg_effectiveness": eff.avg_effectiveness,
                    "application_rate": eff.application_rate,
                    "success_rate": eff.success_rate,
                    "trend": eff.trend,
                }
                for cat, eff in self.category_breakdown.items()
            },
            "recommendations": self.recommendations,
            "adjustment_suggestions": self.adjustment_suggestions,
        }
