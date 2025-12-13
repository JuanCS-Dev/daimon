"""Tests for metacognitive analysis functions."""

import pytest
from learners.metacognitive_models import CategoryEffectiveness
from learners.metacognitive_analysis import (
    generate_recommendations,
    generate_adjustments,
)


class TestGenerateRecommendations:
    """Tests for recommendation generation."""

    def test_recommends_focus_on_effective_categories(self):
        """Recommend focusing on highly effective categories."""
        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=10,
                avg_effectiveness=0.85,
            ),
            "verbosity": CategoryEffectiveness(
                category="verbosity",
                total_insights=8,
                avg_effectiveness=0.45,
            ),
        }
        recommendations = generate_recommendations(breakdown)
        assert any("Focus on" in r and "code_style" in r for r in recommendations)

    def test_recommends_review_declining_categories(self):
        """Recommend reviewing declining categories."""
        breakdown = {
            "testing": CategoryEffectiveness(
                category="testing",
                total_insights=10,
                trend="declining",
            ),
        }
        recommendations = generate_recommendations(breakdown)
        assert any("Review" in r and "testing" in r for r in recommendations)

    def test_recommends_investigating_low_application(self):
        """Recommend investigating categories with low application rate."""
        eff = CategoryEffectiveness(
            category="documentation",
            total_insights=10,
            applied_insights=2,  # 20% application rate
        )
        breakdown = {"documentation": eff}
        recommendations = generate_recommendations(breakdown)
        assert any("Investigate" in r and "documentation" in r for r in recommendations)

    def test_recommends_reducing_high_volume_low_effectiveness(self):
        """Recommend reducing focus on high volume but low effectiveness."""
        breakdown = {
            "architecture": CategoryEffectiveness(
                category="architecture",
                total_insights=15,
                avg_effectiveness=0.3,
            ),
        }
        recommendations = generate_recommendations(breakdown)
        assert any("Reduce focus" in r and "architecture" in r for r in recommendations)

    def test_returns_insufficient_data_message(self):
        """Return message when no patterns detected."""
        breakdown = {}
        recommendations = generate_recommendations(breakdown)
        assert len(recommendations) == 1
        assert "Insufficient data" in recommendations[0]


class TestGenerateAdjustments:
    """Tests for adjustment generation."""

    def test_suggests_higher_threshold_for_low_effectiveness(self):
        """Suggest raising confidence threshold for low effectiveness categories."""
        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=10,
                avg_effectiveness=0.3,
            ),
        }
        adjustments = generate_adjustments(breakdown)
        key = "code_style_confidence_threshold"
        assert key in adjustments
        assert adjustments[key]["suggested"] == 0.7
        assert "low-confidence" in adjustments[key]["reason"].lower()

    def test_suggests_lower_threshold_for_high_effectiveness(self):
        """Suggest lowering confidence threshold for high effectiveness categories."""
        breakdown = {
            "testing": CategoryEffectiveness(
                category="testing",
                total_insights=10,
                avg_effectiveness=0.9,
            ),
        }
        adjustments = generate_adjustments(breakdown)
        key = "testing_confidence_threshold"
        assert key in adjustments
        assert adjustments[key]["suggested"] == 0.3
        assert "capture more" in adjustments[key]["reason"].lower()

    def test_suggests_slower_scan_for_high_volume(self):
        """Suggest slower scan frequency for high insight volume."""
        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=30,
            ),
            "testing": CategoryEffectiveness(
                category="testing",
                total_insights=25,
            ),
        }
        adjustments = generate_adjustments(breakdown)
        assert "scan_frequency" in adjustments
        assert adjustments["scan_frequency"]["suggested"] == "1hour"

    def test_suggests_faster_scan_for_low_volume(self):
        """Suggest faster scan frequency for low insight volume."""
        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=3,
            ),
        }
        adjustments = generate_adjustments(breakdown)
        assert "scan_frequency" in adjustments
        assert adjustments["scan_frequency"]["suggested"] == "15min"

    def test_returns_empty_for_insufficient_data(self):
        """Return empty adjustments when insufficient data."""
        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=2,  # Below threshold of 5
                avg_effectiveness=0.3,
            ),
        }
        adjustments = generate_adjustments(breakdown)
        # Should not have confidence threshold adjustment (not enough insights)
        assert "code_style_confidence_threshold" not in adjustments


class TestCategoryEffectivenessProperties:
    """Tests for CategoryEffectiveness computed properties."""

    def test_application_rate(self):
        """Calculate application rate correctly."""
        eff = CategoryEffectiveness(
            category="test",
            total_insights=10,
            applied_insights=3,
        )
        assert abs(eff.application_rate - 0.3) < 0.001

    def test_application_rate_zero_insights(self):
        """Handle zero total insights."""
        eff = CategoryEffectiveness(
            category="test",
            total_insights=0,
        )
        assert eff.application_rate == 0.0

    def test_success_rate(self):
        """Calculate success rate correctly."""
        eff = CategoryEffectiveness(
            category="test",
            applied_insights=10,
            effective_insights=7,
        )
        assert abs(eff.success_rate - 0.7) < 0.001

    def test_success_rate_zero_applied(self):
        """Handle zero applied insights."""
        eff = CategoryEffectiveness(
            category="test",
            applied_insights=0,
        )
        assert eff.success_rate == 0.0
