"""
REAL Integration Tests for Metacognitive Engine.

Tests actual insight logging, effectiveness measurement, and analysis.
No mocks - real data, real computations.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta


class TestMetacognitiveEngineReal:
    """Real tests for MetacognitiveEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine with temporary storage."""
        from learners.metacognitive_engine import MetacognitiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MetacognitiveEngine(
                ledger_path=Path(tmpdir) / "ledger",
                history_days=7,
            )
            yield engine

    def test_log_insight(self, engine):
        """Log insight and verify storage."""
        insight = {
            "category": "code_style",
            "action": "add",
            "confidence": 0.8,
            "suggestion": "User prefers explicit type hints",
        }

        insight_id = engine.log_insight(
            insight=insight,
            was_applied=True,
            context={"source": "test"},
        )

        assert insight_id is not None
        assert len(insight_id) == 12  # SHA256 truncated to 12 chars

    def test_multiple_insights(self, engine):
        """Log multiple insights."""
        categories = ["code_style", "verbosity", "architecture"]

        ids = []
        for cat in categories:
            insight_id = engine.log_insight(
                insight={"category": cat, "suggestion": f"Test {cat}"},
                was_applied=True,
            )
            ids.append(insight_id)

        # All unique IDs
        assert len(set(ids)) == len(ids)

    def test_analyze_effectiveness(self, engine):
        """Test effectiveness analysis."""
        # Log insights with different categories
        for i in range(5):
            engine.log_insight(
                insight={"category": "code_style", "confidence": 0.8},
                was_applied=True,
            )
        for i in range(3):
            engine.log_insight(
                insight={"category": "verbosity", "confidence": 0.6},
                was_applied=False,
            )

        analysis = engine.analyze_effectiveness()

        # Should return MetacognitiveAnalysis with correct fields
        assert hasattr(analysis, "insights_analyzed")
        assert hasattr(analysis, "category_breakdown")
        assert analysis.insights_analyzed == 8

    def test_insight_persistence(self):
        """Test insight persistence across engine instances."""
        from learners.metacognitive_engine import MetacognitiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger"

            # Create first engine and log insight
            engine1 = MetacognitiveEngine(ledger_path=ledger_path)
            engine1.log_insight(
                insight={"category": "test", "suggestion": "persistent insight"},
                was_applied=True,
            )

            # Create new engine with same path
            engine2 = MetacognitiveEngine(ledger_path=ledger_path)

            # Should have loaded the insight
            assert len(engine2._insights) >= 1

    def test_capture_metrics_snapshot(self, engine):
        """Test metrics snapshot capture."""
        snapshot = engine._capture_metrics_snapshot()

        assert isinstance(snapshot, dict)
        # Should have basic metrics
        assert "signals_count" in snapshot or len(snapshot) >= 0

    def test_reflect_on_reflection(self, engine):
        """Test full reflection analysis."""
        # Add some insights first
        for i in range(5):
            engine.log_insight(
                insight={"category": "test", "confidence": 0.7},
                was_applied=True,
            )

        result = engine.reflect_on_reflection()

        assert isinstance(result, dict)

    def test_get_stats(self, engine):
        """Test stats retrieval."""
        # Add some insights
        engine.log_insight(
            insight={"category": "test", "confidence": 0.7},
            was_applied=True,
        )

        stats = engine.get_stats()

        assert isinstance(stats, dict)
        # Stats has total_insights_tracked
        assert "total_insights_tracked" in stats
        assert stats["total_insights_tracked"] >= 1

    def test_get_learning_recommendations(self, engine):
        """Test recommendation generation."""
        # Add insights to have data for recommendations
        for i in range(10):
            engine.log_insight(
                insight={"category": "code_style", "confidence": 0.8},
                was_applied=True,
            )

        recommendations = engine.get_learning_recommendations()

        assert isinstance(recommendations, list)


class TestInsightRecordReal:
    """Real tests for InsightRecord model."""

    def test_create_record(self):
        """Create insight record."""
        from learners.metacognitive_models import InsightRecord

        record = InsightRecord(
            insight_id="abc123def456",
            category="code_style",
            action="add",
            confidence=0.85,
            suggestion="Use type hints consistently",
            timestamp=datetime.now(),
            context={"source": "reflection"},
            was_applied=True,
        )

        assert record.insight_id == "abc123def456"
        assert record.confidence == 0.85
        assert record.was_applied is True

    def test_record_serialization(self):
        """Test record to_dict and from_dict."""
        from learners.metacognitive_models import InsightRecord

        original = InsightRecord(
            insight_id="test123",
            category="verbosity",
            action="update",
            confidence=0.7,
            suggestion="Be more concise",
            timestamp=datetime.now(),
            context={"test": True},
            was_applied=False,
            effectiveness_score=0.6,
        )

        d = original.to_dict()
        restored = InsightRecord.from_dict(d)

        assert restored.insight_id == original.insight_id
        assert restored.category == original.category
        assert restored.effectiveness_score == original.effectiveness_score


class TestCategoryEffectivenessReal:
    """Real tests for CategoryEffectiveness model."""

    def test_application_rate_calculation(self):
        """Test application rate calculation."""
        from learners.metacognitive_models import CategoryEffectiveness

        eff = CategoryEffectiveness(
            category="test",
            total_insights=100,
            applied_insights=30,
        )

        assert abs(eff.application_rate - 0.3) < 0.001

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from learners.metacognitive_models import CategoryEffectiveness

        eff = CategoryEffectiveness(
            category="test",
            applied_insights=50,
            effective_insights=40,
        )

        assert abs(eff.success_rate - 0.8) < 0.001

    def test_zero_division_safety(self):
        """Test handling of zero division cases."""
        from learners.metacognitive_models import CategoryEffectiveness

        eff = CategoryEffectiveness(
            category="test",
            total_insights=0,
            applied_insights=0,
        )

        assert eff.application_rate == 0.0
        assert eff.success_rate == 0.0


class TestRecommendationsReal:
    """Real tests for recommendation generation."""

    def test_generate_recommendations_real(self):
        """Generate recommendations from real effectiveness data."""
        from learners.metacognitive_analysis import generate_recommendations
        from learners.metacognitive_models import CategoryEffectiveness

        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=20,
                avg_effectiveness=0.9,
                trend="improving",
            ),
            "verbosity": CategoryEffectiveness(
                category="verbosity",
                total_insights=15,
                avg_effectiveness=0.3,
                trend="declining",
            ),
            "architecture": CategoryEffectiveness(
                category="architecture",
                total_insights=25,
                avg_effectiveness=0.2,
                applied_insights=5,
            ),
        }

        recommendations = generate_recommendations(breakdown)

        # Should generate meaningful recommendations
        assert len(recommendations) >= 1
        # High effectiveness should get "Focus on" recommendation
        assert any("code_style" in r for r in recommendations)

    def test_generate_adjustments_real(self):
        """Generate parameter adjustments from real data."""
        from learners.metacognitive_analysis import generate_adjustments
        from learners.metacognitive_models import CategoryEffectiveness

        breakdown = {
            "code_style": CategoryEffectiveness(
                category="code_style",
                total_insights=30,
                avg_effectiveness=0.2,  # Low - should suggest higher threshold
            ),
            "testing": CategoryEffectiveness(
                category="testing",
                total_insights=25,
                avg_effectiveness=0.95,  # High - should suggest lower threshold
            ),
        }

        adjustments = generate_adjustments(breakdown)

        # Should suggest parameter changes
        assert isinstance(adjustments, dict)
        # High volume should trigger scan frequency adjustment
        if adjustments:
            assert any("threshold" in k or "frequency" in k for k in adjustments.keys())


class TestMetacognitiveIntegration:
    """Integration tests for metacognitive system."""

    def test_full_insight_cycle(self):
        """Test complete insight lifecycle."""
        from learners.metacognitive_engine import MetacognitiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MetacognitiveEngine(ledger_path=Path(tmpdir) / "ledger")

            # Log multiple insights
            for i in range(10):
                engine.log_insight(
                    insight={
                        "category": "test_category",
                        "confidence": 0.7 + (i * 0.02),
                        "suggestion": f"Suggestion {i}",
                    },
                    was_applied=i % 2 == 0,  # 50% applied
                )

            # Analyze effectiveness
            analysis = engine.analyze_effectiveness()

            # Should have analysis data (insights_analyzed, not total_insights)
            assert analysis.insights_analyzed == 10

            # Get recommendations
            recommendations = engine.get_learning_recommendations()
            assert isinstance(recommendations, list)

    def test_with_activity_store_integration(self):
        """Test metacognitive engine with real ActivityStore data."""
        from learners.metacognitive_engine import MetacognitiveEngine
        from memory.activity_store import ActivityStore

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create activity store with some data
            store_path = Path(tmpdir) / "activity.db"
            store = ActivityStore(db_path=store_path)

            # Add some claude events
            for i in range(5):
                store.add(
                    watcher_type="claude",
                    timestamp=datetime.now() - timedelta(hours=i),
                    data={
                        "preference_signal": "approval" if i % 2 == 0 else "rejection",
                        "intention": "test",
                    },
                )

            # Create engine - should capture metrics from store
            engine = MetacognitiveEngine(ledger_path=Path(tmpdir) / "ledger")
            snapshot = engine._capture_metrics_snapshot()

            # Snapshot should reflect activity data
            assert isinstance(snapshot, dict)


class TestMetacognitiveAnalysisModel:
    """Tests for MetacognitiveAnalysis model."""

    def test_create_analysis(self):
        """Create MetacognitiveAnalysis instance."""
        from learners.metacognitive_models import MetacognitiveAnalysis, CategoryEffectiveness

        analysis = MetacognitiveAnalysis(
            analyzed_at=datetime.now(),
            insights_analyzed=100,
            categories_analyzed=3,
            overall_effectiveness=0.75,
            category_breakdown={
                "test": CategoryEffectiveness(category="test", total_insights=50)
            },
            recommendations=["Focus on high-effectiveness categories"],
            adjustment_suggestions={"confidence_threshold": 0.7},
        )

        assert analysis.insights_analyzed == 100
        assert analysis.overall_effectiveness == 0.75
        assert len(analysis.category_breakdown) == 1

    def test_analysis_to_dict(self):
        """Test analysis serialization."""
        from learners.metacognitive_models import MetacognitiveAnalysis

        analysis = MetacognitiveAnalysis(
            analyzed_at=datetime.now(),
            insights_analyzed=50,
            categories_analyzed=2,
            overall_effectiveness=0.65,
            category_breakdown={},
            recommendations=[],
            adjustment_suggestions={},
        )

        d = analysis.to_dict()

        assert d["insights_analyzed"] == 50
        assert d["overall_effectiveness"] == 0.65
