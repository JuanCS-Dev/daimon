"""
Metacognitive Analysis Functions.

Helper functions for generating recommendations and threshold adjustments.
"""

from typing import Any, Dict, List

from .metacognitive_models import CategoryEffectiveness


def generate_recommendations(
    category_breakdown: Dict[str, CategoryEffectiveness],
) -> List[str]:
    """Generate recommendations based on effectiveness analysis."""
    recommendations = []

    # Find most effective categories
    effective_cats = [
        cat
        for cat, eff in category_breakdown.items()
        if eff.avg_effectiveness > 0.7
    ]
    if effective_cats:
        recommendations.append(
            f"Focus on: {', '.join(effective_cats)} - these categories show high effectiveness"
        )

    # Find declining categories
    declining_cats = [
        cat
        for cat, eff in category_breakdown.items()
        if eff.trend == "declining"
    ]
    if declining_cats:
        recommendations.append(
            f"Review: {', '.join(declining_cats)} - effectiveness declining, may need strategy change"
        )

    # Find low application rate categories
    low_application = [
        cat
        for cat, eff in category_breakdown.items()
        if eff.application_rate < 0.3 and eff.total_insights >= 3
    ]
    if low_application:
        recommendations.append(
            f"Investigate: {', '.join(low_application)} - low application rate suggests insights may not be actionable"
        )

    # Find high volume but low effectiveness
    high_volume_low_eff = [
        cat
        for cat, eff in category_breakdown.items()
        if eff.total_insights >= 10 and eff.avg_effectiveness < 0.4
    ]
    if high_volume_low_eff:
        recommendations.append(
            f"Reduce focus on: {', '.join(high_volume_low_eff)} - high volume but low effectiveness"
        )

    if not recommendations:
        recommendations.append(
            "Insufficient data for recommendations - continue gathering insight effectiveness metrics"
        )

    return recommendations


def generate_adjustments(
    category_breakdown: Dict[str, CategoryEffectiveness],
) -> Dict[str, Any]:
    """Generate suggested threshold/config adjustments."""
    adjustments: Dict[str, Any] = {}

    # Suggest confidence threshold adjustments
    for cat, eff in category_breakdown.items():
        if eff.avg_effectiveness < 0.4 and eff.total_insights >= 5:
            # Low effectiveness - raise confidence threshold
            adjustments[f"{cat}_confidence_threshold"] = {
                "current": 0.5,  # Default
                "suggested": 0.7,
                "reason": "Low average effectiveness suggests low-confidence insights aren't useful",
            }
        elif eff.avg_effectiveness > 0.8 and eff.total_insights >= 5:
            # High effectiveness - can lower threshold to capture more
            adjustments[f"{cat}_confidence_threshold"] = {
                "current": 0.5,
                "suggested": 0.3,
                "reason": "High effectiveness suggests we can capture more insights",
            }

    # Suggest scan frequency adjustments
    total_insights = sum(e.total_insights for e in category_breakdown.values())
    if total_insights > 50:
        adjustments["scan_frequency"] = {
            "current": "30min",
            "suggested": "1hour",
            "reason": "High insight volume - reduce frequency to avoid noise",
        }
    elif total_insights < 10:
        adjustments["scan_frequency"] = {
            "current": "30min",
            "suggested": "15min",
            "reason": "Low insight volume - increase frequency to capture more patterns",
        }

    return adjustments
