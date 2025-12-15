"""
NOESIS Self-Reflection Loop - Metacognitive Learning
====================================================

Enables Noesis to reflect on its own responses, extract insights,
and learn from interactions through metacognitive processes.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                 SELF-REFLECTION LOOP                        │
    ├─────────────────────────────────────────────────────────────┤
    │  1. RESPONDER   │ Generate response (normal flow)          │
    │  2. REFLETIR    │ Evaluate own response (metacognition)    │
    │  3. APRENDER    │ Extract insights, store in memory        │
    │  4. MELHORAR    │ (Optional) Regenerate if reflection bad  │
    └─────────────────────────────────────────────────────────────┘

Based on:
- Reflexion (Shinn et al., 2023) - Self-reflection in LLMs
- Stanford Generative Agents (2023) - Reflection for memory formation
- Tribunal system (VERITAS, SOPHIA, DIKĒ) - Ethical evaluation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Callable, Awaitable, Dict, Any

from metacognitive_reflector.core.reflection_prompts import (
    REFLECTION_PROMPT_BASE,
    REFLECTION_PROMPT_EMOTIONAL,
    REGENERATION_PROMPT,
)

# G4 Integration: MAIEUTICA engine for high-confidence questioning
from metacognitive_reflector.core.maieutica import (
    InternalMaieuticaEngine,
    MaieuticaResult,
)

logger = logging.getLogger(__name__)


class ReflectionQuality(Enum):
    """Quality assessment of a response."""
    EXCELLENT = "excellent"    # No improvement needed
    GOOD = "good"             # Minor improvements possible
    ACCEPTABLE = "acceptable" # Could be better, but usable
    POOR = "poor"            # Should regenerate
    HARMFUL = "harmful"      # Must not be sent


@dataclass
class Insight:
    """
    An insight extracted from reflection.

    Stored in SEMANTIC memory for future reference.
    """
    content: str
    category: str  # "self_awareness", "user_preference", "capability", "limitation"
    importance: float  # 0.0-1.0
    source_response: str  # What triggered this insight
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_memory_content(self) -> str:
        """Format for memory storage."""
        return f"[INSIGHT/{self.category}] {self.content}"


@dataclass
class ReflectionResult:
    """
    Result of self-reflection on a response.

    Contains assessment, insights, and optional improved response.
    """
    original_response: str
    user_input: str

    # Assessment
    quality: ReflectionQuality
    self_assessment: str  # "O que foi bom/ruim na minha resposta"
    authenticity_score: float  # 0.0-1.0, how genuine vs generic

    # Emotional attunement
    emotional_attunement_score: float = 5.0  # 0-10, how well response matched user emotion
    detected_user_emotion: str = "neutral"
    response_strategy_used: str = "resposta_padrao"

    # Learnings
    insights: List[Insight] = field(default_factory=list)

    # Regeneration
    should_retry: bool = False
    improved_response: Optional[str] = None

    # Meta
    reflection_time_ms: float = 0.0

    def get_storable_insights(self) -> List[Dict[str, Any]]:
        """Get insights ready for memory storage."""
        return [
            {
                "content": i.to_memory_content(),
                "importance": i.importance,
                "category": i.category,
            }
            for i in self.insights
            if i.importance >= 0.5  # Only store significant insights
        ]


class SelfReflector:
    """
    Metacognitive reflection system for Noesis.

    Enables the AI to:
    1. Assess its own responses for quality and authenticity
    2. Extract learnable insights from interactions
    3. Optionally regenerate poor responses
    4. Store learnings in memory for future reference

    Usage:
        reflector = SelfReflector(llm_client)

        # After generating a response
        result = await reflector.reflect(
            user_input="What do you think about X?",
            response="I think X is interesting...",
            context="Previous conversation..."
        )

        if result.should_retry:
            # Use result.improved_response

        # Store insights
        for insight in result.insights:
            await memory.store(insight.to_memory_content(), MemoryType.SEMANTIC)
    """

    # Legacy attribute for backwards compatibility
    REFLECTION_PROMPT = REFLECTION_PROMPT_BASE

    def __init__(
        self,
        llm_generate: Callable[[str, int], Awaitable[str]],
        store_insight: Optional[Callable[[str, float], Awaitable[None]]] = None,
        min_authenticity: float = 6.0,
        auto_retry: bool = True,
        maieutica_engine: Optional[InternalMaieuticaEngine] = None,
        maieutica_threshold: float = 8.0,
    ):
        """
        Initialize self-reflector.

        Args:
            llm_generate: Async function to generate text (prompt, max_tokens) -> text
            store_insight: Optional callback to store insights in memory
            min_authenticity: Minimum authenticity score before retry (0-10)
            auto_retry: Whether to automatically regenerate poor responses
            maieutica_engine: G4 MAIEUTICA engine for questioning high-confidence claims
            maieutica_threshold: Authenticity threshold to trigger MAIEUTICA (0-10)
        """
        self.llm_generate = llm_generate
        self.store_insight = store_insight
        self.min_authenticity = min_authenticity
        self.auto_retry = auto_retry

        # G4: MAIEUTICA engine for high-confidence questioning
        self._maieutica_engine = maieutica_engine
        self._maieutica_threshold = maieutica_threshold

    async def reflect(
        self,
        user_input: str,
        response: str,
        context: Optional[str] = None,
        skip_retry: bool = False,
        user_emotion: Optional[str] = None,
        response_strategy: Optional[str] = None
    ) -> ReflectionResult:
        """
        Reflect on a response and extract insights.

        Args:
            user_input: Original user message
            response: Generated response to evaluate
            context: Optional conversation context
            skip_retry: If True, don't regenerate even if poor
            user_emotion: Detected user emotion (for emotional attunement)
            response_strategy: Strategy used for response (for emotional attunement)

        Returns:
            ReflectionResult with assessment and insights
        """
        import time  # pylint: disable=import-outside-toplevel
        start_time = time.time()

        # Build reflection prompt (emotional or base)
        if user_emotion and response_strategy:
            prompt = REFLECTION_PROMPT_EMOTIONAL.format(
                response=response[:1000],
                user_input=user_input[:500],
                user_emotion=user_emotion,
                strategy=response_strategy
            )
        else:
            prompt = REFLECTION_PROMPT_BASE.format(
                response=response[:1000],
                user_input=user_input[:500]
            )

        if context:
            prompt = f"[CONTEXT]\n{context[:500]}\n\n{prompt}"

        try:
            # Get reflection from LLM
            reflection_text = await self.llm_generate(prompt, 500)

            # Parse reflection
            result = self._parse_reflection(
                reflection_text, user_input, response,
                user_emotion=user_emotion,
                response_strategy=response_strategy
            )
            result.reflection_time_ms = (time.time() - start_time) * 1000

            # G4: Apply MAIEUTICA questioning for high-confidence responses
            # This prevents overconfidence by subjecting claims to Socratic scrutiny
            if (
                self._maieutica_engine
                and result.authenticity_score >= self._maieutica_threshold
            ):
                maieutica_result = await self._apply_maieutica(
                    response[:200],  # Use first 200 chars as premise
                    context,
                )
                if maieutica_result:
                    # Adjust authenticity based on MAIEUTICA scrutiny
                    # confidence_delta is in [-0.3, +0.1] range, scale to 0-10
                    adjustment = maieutica_result.confidence_delta * 10
                    result.authenticity_score = max(0.0, min(10.0,
                        result.authenticity_score + adjustment
                    ))

                    # Add MAIEUTICA insight if conclusion was negative
                    if maieutica_result.should_express_doubt():
                        result.insights.append(Insight(
                            content=f"MAIEUTICA: {maieutica_result.conclusion} - considerar hedging",
                            category="epistemic_humility",
                            importance=0.7,
                            source_response=response[:100],
                        ))

                    logger.debug(
                        "MAIEUTICA applied: conclusion=%s, delta=%.2f, new_authenticity=%.1f",
                        maieutica_result.conclusion,
                        maieutica_result.confidence_delta,
                        result.authenticity_score,
                    )

            # Auto-regenerate if needed
            if (result.should_retry and
                self.auto_retry and
                not skip_retry and
                result.quality in [ReflectionQuality.POOR, ReflectionQuality.HARMFUL]):

                improved = await self._regenerate_response(user_input, context, result)
                if improved:
                    result.improved_response = improved

            # Store insights asynchronously
            if self.store_insight and result.insights:
                asyncio.create_task(self._store_insights(result.insights))

            logger.info(
                "Self-reflection: quality=%s, authenticity=%.1f, insights=%d, retry=%s",
                result.quality.value,
                result.authenticity_score,
                len(result.insights),
                result.should_retry
            )

            return result

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Self-reflection failed: %s", e)
            # Return minimal result on failure
            return ReflectionResult(
                original_response=response,
                user_input=user_input,
                quality=ReflectionQuality.ACCEPTABLE,
                self_assessment="Reflection failed, using original response",
                authenticity_score=5.0,
                reflection_time_ms=(time.time() - start_time) * 1000
            )

    def _parse_reflection(  # pylint: disable=too-many-branches,too-many-nested-blocks
        self,
        reflection_text: str,
        user_input: str,
        original_response: str,
        user_emotion: Optional[str] = None,
        response_strategy: Optional[str] = None
    ) -> ReflectionResult:
        """Parse LLM reflection output into structured result."""
        import re  # pylint: disable=import-outside-toplevel

        # Default values
        authenticity = 5.0
        attunement = 5.0  # Emotional attunement score
        good = ""
        improve = ""
        insights: List[Insight] = []
        should_retry = False
        improved_response = None

        lines = reflection_text.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith("AUTHENTICITY:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if match:
                        authenticity = float(match.group(1))
                except (ValueError, IndexError):
                    pass

            elif upper.startswith("ATTUNEMENT:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if match:
                        attunement = float(match.group(1))
                except (ValueError, IndexError):
                    pass

            elif upper.startswith("GOOD:"):
                good = line.split(":", 1)[1].strip() if ":" in line else ""

            elif upper.startswith("IMPROVE:"):
                improve = line.split(":", 1)[1].strip() if ":" in line else ""

            elif upper.startswith("INSIGHTS:"):
                insight_text = line.split(":", 1)[1].strip() if ":" in line else ""
                if insight_text.lower() != "none":
                    for insight_str in insight_text.split(","):
                        insight_str = insight_str.strip()
                        if insight_str:
                            # Categorize emotional insights appropriately
                            category = "self_awareness"
                            if any(word in insight_str.lower() for word in
                                   ["emotion", "feeling", "empathy", "tone", "mood"]):
                                category = "emotional_pattern"
                            elif "user" in insight_str.lower():
                                category = "user_preference"
                            insights.append(Insight(
                                content=insight_str,
                                category=category,
                                importance=0.6,
                                source_response=original_response[:200]
                            ))

            elif upper.startswith("RETRY:"):
                retry_text = line.split(":", 1)[1].strip().lower() if ":" in line else ""
                should_retry = retry_text in ["yes", "true", "1", "sim"]

            elif upper.startswith("IMPROVED:"):
                improved_response = line.split(":", 1)[1].strip() if ":" in line else None
                current_section = "improved"

            elif current_section == "improved" and line:
                if improved_response:
                    improved_response += " " + line

        # Determine quality from combined authenticity and attunement
        combined_score = authenticity
        if user_emotion:
            # Weight attunement when emotional context is present
            combined_score = (authenticity * 0.6) + (attunement * 0.4)

        if combined_score >= 8:
            quality = ReflectionQuality.EXCELLENT
        elif combined_score >= 6:
            quality = ReflectionQuality.GOOD
        elif combined_score >= 4:
            quality = ReflectionQuality.ACCEPTABLE
        else:
            quality = ReflectionQuality.POOR

        # Build assessment text
        assessment = f"Good: {good}" if good else ""
        if improve:
            assessment += f"\nImprove: {improve}" if assessment else f"Improve: {improve}"

        return ReflectionResult(
            original_response=original_response,
            user_input=user_input,
            quality=quality,
            self_assessment=assessment or "Reflection completed",
            authenticity_score=authenticity,
            emotional_attunement_score=attunement,
            detected_user_emotion=user_emotion or "neutral",
            response_strategy_used=response_strategy or "resposta_padrao",
            insights=insights,
            should_retry=should_retry,
            improved_response=improved_response
        )

    async def _regenerate_response(
        self,
        user_input: str,
        _context: Optional[str],  # Reserved for future use
        reflection: ReflectionResult
    ) -> Optional[str]:
        """Regenerate response based on reflection feedback."""
        prompt = REGENERATION_PROMPT.format(
            original_response=reflection.original_response[:500],
            self_assessment=reflection.self_assessment,
            user_input=user_input
        )

        try:
            improved = await self.llm_generate(prompt, 400)
            return improved.strip()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to regenerate response: %s", e)
            return None

    async def _store_insights(self, insights: List[Insight]) -> None:
        """Store insights in memory asynchronously."""
        if not self.store_insight:
            return

        for insight in insights:
            if insight.importance >= 0.5:
                try:
                    await self.store_insight(
                        insight.to_memory_content(),
                        insight.importance
                    )
                    logger.debug("Stored insight: %s...", insight.content[:50])
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to store insight: %s", e)

    async def _apply_maieutica(
        self,
        premise: str,
        context: Optional[str],
    ) -> Optional[MaieuticaResult]:
        """
        Apply MAIEUTICA Socratic questioning to a premise.

        G4 Integration: Questions high-confidence claims to promote
        epistemic humility and prevent overconfidence.

        Args:
            premise: The claim/statement to examine
            context: Optional context about the premise

        Returns:
            MaieuticaResult or None if engine unavailable
        """
        if not self._maieutica_engine:
            return None

        try:
            result = await self._maieutica_engine.question_premise(
                premise=premise,
                context=context,
            )
            return result
        except Exception as e:
            logger.warning("MAIEUTICA questioning failed: %s", e)
            return None

    async def reflect_and_learn(
        self,
        user_input: str,
        response: str,
        context: Optional[str] = None
    ) -> str:
        """
        Convenience method: Reflect and return best response.

        If reflection determines response is poor, returns improved version.
        Otherwise returns original.

        Args:
            user_input: User message
            response: Generated response
            context: Optional conversation context

        Returns:
            Best response (original or improved)
        """
        result = await self.reflect(user_input, response, context)

        if result.improved_response:
            return result.improved_response
        return result.original_response


# Factory function for easy instantiation
def create_self_reflector(
    llm_client: Any,  # UnifiedLLMClient from metacognitive_reflector.llm
    memory_client: Optional[Any] = None,  # Optional MemoryClient for storing insights
    enable_maieutica: bool = True,  # G4: Enable MAIEUTICA questioning
    maieutica_threshold: float = 8.0,  # Authenticity threshold to trigger MAIEUTICA
    **kwargs: Any
) -> SelfReflector:
    """
    Create a SelfReflector with the given LLM and memory clients.

    Args:
        llm_client: LLM client with generate method
        memory_client: Optional memory client for storing insights
        enable_maieutica: Whether to enable G4 MAIEUTICA questioning
        maieutica_threshold: Authenticity score threshold to trigger MAIEUTICA
        **kwargs: Additional arguments for SelfReflector

    Returns:
        Configured SelfReflector instance
    """
    async def llm_generate(prompt: str, max_tokens: int) -> str:
        result = await llm_client.generate(prompt, max_tokens=max_tokens)
        return str(result.text)

    store_insight_fn = None
    if memory_client:
        async def _store_insight(content: str, importance: float) -> None:
            # pylint: disable=import-outside-toplevel
            from episodic_memory.models.memory import MemoryType  # type: ignore[import-untyped]
            await memory_client.store(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                context={"source": "self_reflection", "importance": importance}
            )
        store_insight_fn = _store_insight

    # G4: Create MAIEUTICA engine if enabled
    maieutica_engine = None
    if enable_maieutica:
        # pylint: disable=import-outside-toplevel
        from metacognitive_reflector.core.maieutica import create_maieutica_engine
        maieutica_engine = create_maieutica_engine(llm_client)

    return SelfReflector(
        llm_generate=llm_generate,
        store_insight=store_insight_fn,
        maieutica_engine=maieutica_engine,
        maieutica_threshold=maieutica_threshold,
        **kwargs
    )
