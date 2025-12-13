"""
DAIMON LLM Service - Semantic Analysis for Learners
====================================================

Provides LLM-powered semantic analysis to enhance DAIMON learners.
Replaces regex/heuristics with semantic understanding while maintaining
fallback behavior for graceful degradation.

Features:
- classify(): Semantic classification of user responses
- extract_insights(): Generate actionable insights from patterns
- analyze_cognitive_state(): Interpret biometric data semantically

Architecture:
    classify("nao gostei")
           │
           ▼
    ┌──────────────────┐
    │  Check Cache     │◄── 5 min TTL
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  LLM Call        │◄── Claude Haiku 3.5 / Nebius / Gemini
    └──────────────────┘
           │ (on failure)
           ▼
    ┌──────────────────┐
    │  Heuristic       │◄── Regex fallback
    └──────────────────┘

Usage:
    from learners.llm_service import get_llm_service

    service = get_llm_service()
    result = await service.classify("sim, perfeito!", ["approval", "rejection", "neutral"])

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import models from extracted module
from .llm_models import (
    ClassificationResult,
    InsightResult,
    CognitiveAnalysis,
    LLMServiceStats,
    LLMCache,
)

# Import heuristics from extracted module
from .llm_heuristics import (
    COGNITIVE_STATES,
    classify_with_heuristics,
    extract_insights_with_templates,
    analyze_cognitive_with_rules,
)

logger = logging.getLogger("daimon.llm_service")


class LearnerLLMService:
    """
    LLM service for DAIMON learners.

    Provides semantic analysis capabilities with:
    - Response caching (5 min TTL)
    - Heuristic fallbacks
    - Multiple provider support (via metacognitive_reflector)
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        enable_llm: bool = True,
        fallback_on_error: bool = True,
    ):
        """
        Initialize LLM service.

        Args:
            cache_ttl: Cache time-to-live in seconds
            enable_llm: Whether to use LLM (False = heuristics only)
            fallback_on_error: Whether to fallback to heuristics on LLM error
        """
        self.cache = LLMCache(ttl_seconds=cache_ttl)
        self.enable_llm = enable_llm
        self.fallback_on_error = fallback_on_error
        self.stats = LLMServiceStats()
        self._llm_client: Any = None
        self._llm_available: Optional[bool] = None

    async def _get_llm_client(self) -> Any:
        """Lazy load LLM client from metacognitive_reflector."""
        if self._llm_client is not None:
            return self._llm_client

        if self._llm_available is False:
            return None

        try:
            from metacognitive_reflector.llm import get_llm_client
            self._llm_client = get_llm_client()
            self._llm_available = True
            logger.info("LLM client initialized from metacognitive_reflector")
            return self._llm_client
        except ImportError:
            logger.debug("metacognitive_reflector not available")
            self._llm_available = False
            return None
        except Exception as e:
            logger.warning("Failed to initialize LLM client: %s", e)
            self._llm_available = False
            return None

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    async def classify(
        self,
        content: str,
        options: List[str],
        context: str = "",
    ) -> ClassificationResult:
        """
        Semantically classify content into one of the given options.

        Args:
            content: Text to classify
            options: List of classification options
            context: Optional context about the classification task

        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        self.stats.total_calls += 1
        self.stats.last_call = datetime.now()

        # Check cache
        cached = self.cache.get("classify", content, options, context)
        if cached:
            self.stats.cache_hits += 1
            cached.from_cache = True
            return cached

        start = time.time()

        # Try LLM if enabled
        if self.enable_llm:
            result = await self._classify_with_llm(content, options, context)
            if result:
                self._record_llm_success(start)
                self.cache.set("classify", result, content, options, context)
                return result

        # Fallback to heuristics
        self.stats.heuristic_fallbacks += 1
        result = classify_with_heuristics(content, options)
        self.cache.set("classify", result, content, options, context)
        return result

    async def _classify_with_llm(
        self,
        content: str,
        options: List[str],
        context: str,
    ) -> Optional[ClassificationResult]:
        """Classify using LLM."""
        client = await self._get_llm_client()
        if not client:
            return None

        try:
            prompt = self._build_classification_prompt(content, options, context)
            response = await client.chat([
                {"role": "system", "content": (
                    "You are a precise classification assistant. "
                    "Output JSON only: {\"category\": \"...\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
                )},
                {"role": "user", "content": prompt},
            ], max_tokens=150, temperature=0.3)

            data = self._parse_json_response(response.text)
            category = data.get("category", options[0])
            if category not in options:
                category = options[0]

            return ClassificationResult(
                category=category,
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "LLM classification"),
                from_llm=True,
            )

        except Exception as e:
            logger.debug("LLM classification failed: %s", e)
            self.stats.llm_failures += 1
            return None

    def _build_classification_prompt(
        self, content: str, options: List[str], context: str
    ) -> str:
        """Build classification prompt for LLM."""
        options_str = ", ".join(f'"{o}"' for o in options)
        prompt = f'Classify this response into: {options_str}\n\nUser response: "{content}"'
        if context:
            prompt += f"\nContext: {context}"
        prompt += "\n\nRespond with JSON only."
        return prompt

    # =========================================================================
    # INSIGHTS
    # =========================================================================

    async def extract_insights(
        self,
        data: Dict[str, Any],
        context: str = "",
    ) -> InsightResult:
        """
        Extract actionable insights from preference data.

        Args:
            data: Preference data (approvals, rejections, categories)
            context: Optional context about the user/project

        Returns:
            InsightResult with insights, suggestions, and confidence
        """
        self.stats.total_calls += 1
        self.stats.last_call = datetime.now()

        # Check cache
        cached = self.cache.get("extract_insights", data, context)
        if cached:
            self.stats.cache_hits += 1
            return cached

        # Try LLM if enabled
        if self.enable_llm:
            result = await self._extract_insights_with_llm(data, context)
            if result:
                self.stats.llm_successes += 1
                self.cache.set("extract_insights", result, data, context)
                return result

        # Fallback to templates
        self.stats.heuristic_fallbacks += 1
        result = extract_insights_with_templates(data)
        self.cache.set("extract_insights", result, data, context)
        return result

    async def _extract_insights_with_llm(
        self, data: Dict[str, Any], context: str
    ) -> Optional[InsightResult]:
        """Extract insights using LLM."""
        client = await self._get_llm_client()
        if not client:
            return None

        try:
            data_str = json.dumps(data, indent=2, default=str)
            prompt = f"Analyze this preference data:\n{data_str}"
            if context:
                prompt += f"\nContext: {context}"
            prompt += "\n\nGenerate insights and suggestions. JSON only."

            response = await client.chat([
                {"role": "system", "content": (
                    "You analyze user behavior and extract actionable insights. "
                    "Output JSON: {\"insights\": [...], \"suggestions\": [...], \"confidence\": 0.0-1.0}"
                )},
                {"role": "user", "content": prompt},
            ], max_tokens=300, temperature=0.5)

            result = self._parse_json_response(response.text)
            return InsightResult(
                insights=result.get("insights", []),
                suggestions=result.get("suggestions", []),
                confidence=float(result.get("confidence", 0.7)),
                from_llm=True,
            )

        except Exception as e:
            logger.debug("LLM insight extraction failed: %s", e)
            self.stats.llm_failures += 1
            return None

    # =========================================================================
    # COGNITIVE ANALYSIS
    # =========================================================================

    async def analyze_cognitive_state(
        self,
        biometrics: Dict[str, Any],
    ) -> CognitiveAnalysis:
        """
        Analyze cognitive state from keystroke biometrics.

        Args:
            biometrics: Dictionary with typing metrics

        Returns:
            CognitiveAnalysis with state, confidence, description
        """
        self.stats.total_calls += 1
        self.stats.last_call = datetime.now()

        # Check cache
        cached = self.cache.get("analyze_cognitive", biometrics)
        if cached:
            self.stats.cache_hits += 1
            return cached

        # Try LLM if enabled
        if self.enable_llm:
            result = await self._analyze_cognitive_with_llm(biometrics)
            if result:
                self.stats.llm_successes += 1
                self.cache.set("analyze_cognitive", result, biometrics)
                return result

        # Fallback to rules
        self.stats.heuristic_fallbacks += 1
        result = analyze_cognitive_with_rules(biometrics)
        self.cache.set("analyze_cognitive", result, biometrics)
        return result

    async def _analyze_cognitive_with_llm(
        self, biometrics: Dict[str, Any]
    ) -> Optional[CognitiveAnalysis]:
        """Analyze cognitive state using LLM."""
        client = await self._get_llm_client()
        if not client:
            return None

        try:
            prompt = self._build_cognitive_prompt(biometrics)
            response = await client.chat([
                {"role": "system", "content": (
                    "You analyze keystroke biometrics to infer cognitive state. "
                    "States: flow, focus, fatigue, distracted, idle. "
                    "Output JSON: {\"state\": \"...\", \"confidence\": 0.0-1.0, "
                    "\"description\": \"...\", \"recommendations\": [...]}"
                )},
                {"role": "user", "content": prompt},
            ], max_tokens=200, temperature=0.3)

            result = self._parse_json_response(response.text)
            state = result.get("state", "focus")
            if state not in COGNITIVE_STATES:
                state = "focus"

            return CognitiveAnalysis(
                state=state,
                confidence=float(result.get("confidence", 0.7)),
                description=result.get("description", COGNITIVE_STATES[state]["description"]),
                recommendations=result.get("recommendations", []),
                from_llm=True,
            )

        except Exception as e:
            logger.debug("LLM cognitive analysis failed: %s", e)
            self.stats.llm_failures += 1
            return None

    def _build_cognitive_prompt(self, biometrics: Dict[str, Any]) -> str:
        """Build cognitive analysis prompt for LLM."""
        return (
            f"Analyze keystroke biometrics:\n"
            f"- Typing speed: {biometrics.get('typing_speed', 'N/A')} keys/min\n"
            f"- Rhythm: {biometrics.get('rhythm_consistency', 0):.2f}\n"
            f"- Fatigue: {biometrics.get('fatigue_index', 0):.2f}\n"
            f"- Focus: {biometrics.get('focus_score', 0):.2f}\n"
            f"Determine cognitive state."
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)

    def _record_llm_success(self, start_time: float) -> None:
        """Record successful LLM call stats."""
        self.stats.llm_successes += 1
        latency_ms = (time.time() - start_time) * 1000
        self.stats.avg_latency_ms = (
            (self.stats.avg_latency_ms * (self.stats.llm_successes - 1) + latency_ms)
            / self.stats.llm_successes
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total = self.stats.total_calls
        llm_total = self.stats.llm_successes + self.stats.llm_failures
        return {
            "total_calls": total,
            "cache_hits": self.stats.cache_hits,
            "cache_hit_rate": self.stats.cache_hits / total if total > 0 else 0,
            "llm_successes": self.stats.llm_successes,
            "llm_failures": self.stats.llm_failures,
            "llm_success_rate": self.stats.llm_successes / llm_total if llm_total > 0 else 0,
            "heuristic_fallbacks": self.stats.heuristic_fallbacks,
            "avg_latency_ms": round(self.stats.avg_latency_ms, 1),
            "last_call": self.stats.last_call.isoformat() if self.stats.last_call else None,
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        self.cache.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_llm_service: Optional[LearnerLLMService] = None


def get_llm_service() -> LearnerLLMService:
    """Get singleton LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LearnerLLMService()
    return _llm_service


def reset_llm_service() -> None:
    """Reset singleton LLM service instance."""
    global _llm_service
    if _llm_service:
        _llm_service.clear_cache()
    _llm_service = None


# Re-export models for backward compatibility
__all__ = [
    "ClassificationResult",
    "InsightResult", 
    "CognitiveAnalysis",
    "LLMServiceStats",
    "LLMCache",
    "LearnerLLMService",
    "get_llm_service",
    "reset_llm_service",
]
