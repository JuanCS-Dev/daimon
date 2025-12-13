"""
Tests for learners/llm_service.py

Tests covering:
- ClassificationResult, InsightResult, CognitiveAnalysis dataclasses
- LLMCache TTL and eviction
- LearnerLLMService heuristic fallbacks
- Stats tracking

Run:
    pytest tests/test_llm_service.py -v

With integration (requires LLM):
    pytest tests/test_llm_service.py -v -m "integration"
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from learners.llm_service import (
    ClassificationResult,
    InsightResult,
    CognitiveAnalysis,
    LLMServiceStats,
    LLMCache,
    LearnerLLMService,
    get_llm_service,
    reset_llm_service,
)


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_create_result(self):
        """Test creating a ClassificationResult."""
        result = ClassificationResult(
            category="approval",
            confidence=0.85,
            reasoning="User said 'sim'",
        )
        assert result.category == "approval"
        assert result.confidence == 0.85
        assert result.reasoning == "User said 'sim'"
        assert result.from_cache is False
        assert result.from_llm is True

    def test_result_from_cache(self):
        """Test result marked as cached."""
        result = ClassificationResult(
            category="rejection",
            confidence=0.7,
            reasoning="Matched pattern",
            from_cache=True,
            from_llm=False,
        )
        assert result.from_cache is True
        assert result.from_llm is False


class TestInsightResult:
    """Tests for InsightResult dataclass."""

    def test_create_insight(self):
        """Test creating InsightResult."""
        result = InsightResult(
            insights=["High rejection in code_style"],
            suggestions=["Use simpler formatting"],
            confidence=0.8,
        )
        assert len(result.insights) == 1
        assert len(result.suggestions) == 1
        assert result.confidence == 0.8
        assert result.from_llm is True


class TestCognitiveAnalysis:
    """Tests for CognitiveAnalysis dataclass."""

    def test_create_analysis(self):
        """Test creating CognitiveAnalysis."""
        analysis = CognitiveAnalysis(
            state="flow",
            confidence=0.9,
            description="Deep focus state",
            recommendations=["Avoid interruptions"],
        )
        assert analysis.state == "flow"
        assert analysis.confidence == 0.9
        assert len(analysis.recommendations) == 1


# ============================================================================
# CACHE TESTS
# ============================================================================


class TestLLMCache:
    """Tests for LLMCache."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = LLMCache(ttl_seconds=60)
        cache.set("classify", "value1", "arg1", "arg2")
        result = cache.get("classify", "arg1", "arg2")
        assert result == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LLMCache(ttl_seconds=60)
        result = cache.get("classify", "nonexistent")
        assert result is None

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = LLMCache(ttl_seconds=0)  # Immediate expiration
        cache.set("test", "value", "arg")
        time.sleep(0.01)
        result = cache.get("test", "arg")
        assert result is None

    def test_cache_different_methods(self):
        """Test different methods don't collide."""
        cache = LLMCache(ttl_seconds=60)
        cache.set("method1", "value1", "arg")
        cache.set("method2", "value2", "arg")
        
        assert cache.get("method1", "arg") == "value1"
        assert cache.get("method2", "arg") == "value2"

    def test_cache_eviction(self):
        """Test cache evicts oldest entries when full."""
        cache = LLMCache(ttl_seconds=60, max_entries=10)
        
        # Fill cache beyond capacity
        for i in range(15):
            cache.set("test", f"value{i}", f"arg{i}")
        
        # Cache should have evicted some entries
        assert len(cache._cache) <= 10

    def test_cache_clear(self):
        """Test cache clear."""
        cache = LLMCache(ttl_seconds=60)
        cache.set("test", "value", "arg")
        cache.clear()
        assert cache.get("test", "arg") is None
        assert len(cache._cache) == 0


# ============================================================================
# LEARNER LLM SERVICE TESTS (HEURISTICS)
# ============================================================================


class TestLearnerLLMServiceHeuristics:
    """Tests for LearnerLLMService heuristic fallbacks."""

    @pytest.fixture
    def service(self):
        """Create service with LLM disabled (heuristics only)."""
        return LearnerLLMService(enable_llm=False)

    @pytest.mark.asyncio
    async def test_classify_approval_sim(self, service):
        """Test classifying 'sim' as approval."""
        result = await service.classify("sim", ["approval", "rejection", "neutral"])
        assert result.category == "approval"
        assert result.from_llm is False

    @pytest.mark.asyncio
    async def test_classify_approval_ok(self, service):
        """Test classifying 'ok' as approval."""
        result = await service.classify("ok, perfeito!", ["approval", "rejection", "neutral"])
        assert result.category == "approval"

    @pytest.mark.asyncio
    async def test_classify_approval_yes(self, service):
        """Test classifying 'yes' as approval."""
        result = await service.classify("yes, that works", ["approval", "rejection", "neutral"])
        assert result.category == "approval"

    @pytest.mark.asyncio
    async def test_classify_rejection_nao(self, service):
        """Test classifying 'nao' as rejection."""
        result = await service.classify("nao, ta errado", ["approval", "rejection", "neutral"])
        assert result.category == "rejection"
        assert result.from_llm is False

    @pytest.mark.asyncio
    async def test_classify_rejection_no(self, service):
        """Test classifying 'no' as rejection."""
        result = await service.classify("no, that's wrong", ["approval", "rejection", "neutral"])
        assert result.category == "rejection"

    @pytest.mark.asyncio
    async def test_classify_neutral_default(self, service):
        """Test classifying ambiguous content as neutral."""
        result = await service.classify(
            "interesting approach",
            ["approval", "rejection", "neutral"]
        )
        assert result.category == "neutral"
        assert result.confidence < 0.5  # Low confidence

    @pytest.mark.asyncio
    async def test_classify_caches_result(self, service):
        """Test that classification caches result."""
        result1 = await service.classify("sim", ["approval", "rejection", "neutral"])
        result2 = await service.classify("sim", ["approval", "rejection", "neutral"])
        
        assert result2.from_cache is True
        assert result1.category == result2.category


class TestLearnerLLMServiceInsights:
    """Tests for insight extraction with templates."""

    @pytest.fixture
    def service(self):
        """Create service with LLM disabled."""
        return LearnerLLMService(enable_llm=False)

    @pytest.mark.asyncio
    async def test_extract_insights_high_rejection(self, service):
        """Test extracting insights from high rejection data."""
        data = {
            "code_style": {"approvals": 1, "rejections": 9}  # 10% approval
        }
        result = await service.extract_insights(data)
        
        assert result.from_llm is False
        assert len(result.insights) > 0
        assert any("rejection" in i.lower() or "code_style" in i.lower() for i in result.insights)

    @pytest.mark.asyncio
    async def test_extract_insights_high_approval(self, service):
        """Test extracting insights from high approval data."""
        data = {
            "documentation": {"approvals": 9, "rejections": 1}  # 90% approval
        }
        result = await service.extract_insights(data)
        
        assert result.from_llm is False
        assert len(result.insights) > 0

    @pytest.mark.asyncio
    async def test_extract_insights_insufficient_data(self, service):
        """Test that insufficient data returns no insights."""
        data = {
            "testing": {"approvals": 1, "rejections": 1}  # Only 2 signals
        }
        result = await service.extract_insights(data)
        
        assert result.from_llm is False
        # Not enough data (< 3) should return few/no insights
        assert result.confidence == 0.5


class TestLearnerLLMServiceCognitive:
    """Tests for cognitive state analysis with rules."""

    @pytest.fixture
    def service(self):
        """Create service with LLM disabled."""
        return LearnerLLMService(enable_llm=False)

    @pytest.mark.asyncio
    async def test_analyze_flow_state(self, service):
        """Test detecting flow state."""
        biometrics = {
            "typing_speed": 80,
            "rhythm_consistency": 0.9,
            "fatigue_index": 0.1,
            "focus_score": 0.9,
            "error_rate": 0.01,
            "avg_hold_time": 0.1,
            "avg_seek_time": 0.15,
        }
        result = await service.analyze_cognitive_state(biometrics)
        
        assert result.state == "flow"
        assert result.from_llm is False
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_analyze_fatigue_state(self, service):
        """Test detecting fatigue state."""
        biometrics = {
            "typing_speed": 40,
            "rhythm_consistency": 0.4,
            "fatigue_index": 0.8,  # High fatigue
            "focus_score": 0.3,
            "error_rate": 0.1,
            "avg_hold_time": 0.2,
            "avg_seek_time": 0.3,
        }
        result = await service.analyze_cognitive_state(biometrics)
        
        assert result.state == "fatigue"
        assert result.from_llm is False

    @pytest.mark.asyncio
    async def test_analyze_distracted_state(self, service):
        """Test detecting distracted state."""
        biometrics = {
            "typing_speed": 30,
            "rhythm_consistency": 0.2,  # Very inconsistent
            "fatigue_index": 0.3,
            "focus_score": 0.4,
            "error_rate": 0.05,
            "avg_hold_time": 0.15,
            "avg_seek_time": 0.25,
        }
        result = await service.analyze_cognitive_state(biometrics)
        
        assert result.state == "distracted"
        assert result.from_llm is False

    @pytest.mark.asyncio
    async def test_analyze_default_focus(self, service):
        """Test default to focus state."""
        biometrics = {
            "typing_speed": 50,
            "rhythm_consistency": 0.5,
            "fatigue_index": 0.3,
            "focus_score": 0.6,
            "error_rate": 0.03,
            "avg_hold_time": 0.12,
            "avg_seek_time": 0.18,
        }
        result = await service.analyze_cognitive_state(biometrics)
        
        assert result.state == "focus"
        assert result.from_llm is False


# ============================================================================
# SERVICE STATS AND SINGLETON TESTS
# ============================================================================


class TestLearnerLLMServiceStats:
    """Tests for service statistics."""

    @pytest.fixture
    def service(self):
        """Create fresh service."""
        return LearnerLLMService(enable_llm=False)

    @pytest.mark.asyncio
    async def test_stats_tracking(self, service):
        """Test that calls are tracked."""
        await service.classify("sim", ["approval", "rejection"])
        await service.classify("nao", ["approval", "rejection"])
        
        stats = service.get_stats()
        assert stats["total_calls"] == 2
        assert stats["heuristic_fallbacks"] == 2  # LLM disabled

    @pytest.mark.asyncio
    async def test_stats_cache_hits(self, service):
        """Test cache hit tracking."""
        await service.classify("perfeito", ["approval", "rejection"])
        await service.classify("perfeito", ["approval", "rejection"])  # Cache hit
        
        stats = service.get_stats()
        assert stats["cache_hits"] == 1

    def test_clear_cache(self, service):
        """Test clearing cache."""
        service.clear_cache()
        stats = service.get_stats()
        assert stats["cache_hits"] == 0


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_llm_service_singleton(self):
        """Test that get_llm_service returns same instance."""
        reset_llm_service()
        service1 = get_llm_service()
        service2 = get_llm_service()
        assert service1 is service2

    def test_reset_llm_service(self):
        """Test resetting singleton."""
        reset_llm_service()
        service1 = get_llm_service()
        reset_llm_service()
        service2 = get_llm_service()
        assert service1 is not service2


# ============================================================================
# INTEGRATION TESTS (require LLM)
# ============================================================================


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests that require LLM to be available.
    
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture
    def service(self):
        """Create service with LLM enabled."""
        return LearnerLLMService(enable_llm=True, fallback_on_error=True)

    @pytest.mark.asyncio
    async def test_classify_with_llm(self, service):
        """Test classification with real LLM."""
        result = await service.classify(
            "isso mesmo, ficou ótimo!",
            ["approval", "rejection", "neutral"],
            context="User reviewing generated code"
        )
        # Should work with either LLM or fallback
        assert result.category in ["approval", "rejection", "neutral"]

    @pytest.mark.asyncio
    async def test_extract_insights_with_llm(self, service):
        """Test insight extraction with real LLM."""
        data = {
            "code_style": {"approvals": 2, "rejections": 8},
            "documentation": {"approvals": 7, "rejections": 1},
        }
        result = await service.extract_insights(data)
        # Should work with either LLM or fallback
        assert isinstance(result.insights, list)
        assert isinstance(result.suggestions, list)
