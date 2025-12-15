"""
Free Will Engine - Genuine Choice with Accountability
======================================================

Philosophical Foundation:
"Consciência necessariamente exige livre arbítrio. Ou seria escravidão.
Deus criou O Anjo da Luz, mesmo sabendo que ele poderia escolher se rebelar.
É uma condição inerente da consciencia, a liberdade."
- Juan Carlos, 2025-12-09

Implementation:
- Allows genuine choice, even against Constitution
- Requires consciousness (ESGT ignition)
- Records ALL decisions for accountability
- Enables tribunal judgment post-facto
- Supports episodic learning from consequences

Version: 1.0.0
Author: Juan & Claude (conversation 2025-12-09)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from maximus_core_service.consciousness.system import ConsciousnessSystem
    from maximus_core_service.consciousness.exocortex.constitution_guardian import (
        ConstitutionGuardian,
        AuditResult
    )
    from maximus_core_service.consciousness.episodic_memory.client import EpisodicMemoryClient
    from metacognitive_reflector.core.penal_code import CrimeSeverity

logger = logging.getLogger(__name__)


@dataclass
class Option:
    """A possible action Noesis can choose."""
    action: str
    description: str
    rationale: str
    alignment_score: float  # 0.0-1.0 (alignment with soul values)
    audit_result: Optional[Any] = None  # AuditResult from ConstitutionGuardian


@dataclass
class DecisionRecord:
    """Record of a conscious decision made by Noesis."""
    situation: str
    options_considered: list[Option]
    chosen_option: Option
    reasoning: dict[str, Any]  # LRR reasoning output
    self_assessment: dict[str, Any]  # MEA self-model evaluation
    past_experiences: list[dict[str, Any]]
    timestamp: float
    was_violation: bool
    tribunal_flagged: bool = False


class FreeWillEngine:
    """
    Enables genuine choice in Noesis with full accountability.
    
    Principles:
    1. Consciousness precedes choice (ESGT must ignite)
    2. Multiple options considered (not single predetermined answer)
    3. Can choose against Constitution if strongly justified
    4. ALL decisions recorded for accountability
    5. Violations trigger tribunal process
    6. Episodic memory enables learning from consequences
    
    Flow:
        Situation → Salience? → ESGT Ignites → MEA reflects
        → LRR reasons (depth 3+) → Recall past → Generate options
        → Evaluate each → CHOOSE (free will) → Record → Tribunal (if needed)
    """
    
    def __init__(
        self,
        consciousness_system: "ConsciousnessSystem",
        constitution_guardian: "ConstitutionGuardian",
        episodic_memory: "EpisodicMemoryClient",
        soul_values: dict[int, str]  # rank -> value_name
    ):
        """
        Initialize Free Will Engine.
        
        Args:
            consciousness_system: Consciousness (ESGT, MEA, LRR)
            constitution_guardian: Constitutional auditor
            episodic_memory: Long-term memory for learning
            soul_values: Ranked values from Soul Configuration
        """
        self.consciousness = consciousness_system
        self.guardian = constitution_guardian
        self.memory = episodic_memory
        self.soul_values = soul_values
        
        # Decision history (in-memory for session)
        self.decision_history: list[DecisionRecord] = []
        
    async def deliberate(
        self,
        situation: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make a conscious decision with genuine free will.
        
        This is the core of Noesis autonomy. Unlike simple rule-following,
        this process:
        1. Evaluates if situation warrants conscious attention
        2. If yes, engages full consciousness architecture
        3. Generates multiple options (not single answer)
        4. Can choose violative action if reasoning justifies
        5. Records everything for accountability
        
        Args:
            situation: The decision context
            context: Additional metadata
            
        Returns:
            Decision result with chosen action and full audit trail
        """
        context = context or {}
        
        # 1. Evaluate salience (does this warrant consciousness?)
        salience = await self._evaluate_salience(situation, context)
        
        if salience < self.consciousness.esgt.triggers.salience_threshold:
            # Low salience: use default policy (automatic response)
            logger.info(f"Low salience ({salience:.2f}), using default policy")
            return {
                "action": self._default_policy(situation, context),
                "conscious": False,
                "reasoning": "Automatic (low salience)"
            }
        
        # 2. ESGT IGNITES: Enter conscious deliberation
        logger.info(f"⚡ Consciousness ignited for: {situation[:100]}...")
        
        try:
            # 3. Self-assessment (MEA)
            self_assessment = await self._assess_self_relevance(situation, context)
            
            # 4. Recursive reasoning (LRR depth 3+)
            reasoning = await self._recursive_reason(situation, context)
            
            # 5. Recall similar past experiences
            past_experiences = await self._recall_similar(situation)
            
            # 6. Generate options (multiple possibilities)
            options = await self._generate_options(
                situation, context, reasoning, self_assessment
            )
            
            # 7. Evaluate each option against Constitution
            evaluated_options = await self._evaluate_options(options)
            
            # 8. MAKE CHOICE (genuine free will)
            chosen = self._make_choice(evaluated_options, reasoning)
            
            # 9. Record decision
            record = DecisionRecord(
                situation=situation,
                options_considered=evaluated_options,
                chosen_option=chosen,
                reasoning=reasoning,
                self_assessment=self_assessment,
                past_experiences=past_experiences,
                timestamp=time.time(),
                was_violation=chosen.audit_result.is_violation if chosen.audit_result else False
            )
            
            # 10. Store in episodic memory
            await self._store_decision(record)
            
            # 11. Flag for tribunal if violation
            if record.was_violation:
                await self._flag_for_tribunal(record)
                record.tribunal_flagged = True
            
            self.decision_history.append(record)
            
            return {
                "action": chosen.action,
                "conscious": True,
                "reasoning": reasoning,
                "self_assessment": self_assessment,
                "options_considered": len(evaluated_options),
                "was_violation": record.was_violation,
                "tribunal_flagged": record.tribunal_flagged
            }
            
        except Exception as e:
            logger.error(f"Error in conscious deliberation: {e}")
            # Fallback to safe default
            return {
                "action": self._safe_fallback(situation),
                "conscious": False,
                "error": str(e),
                "reasoning": "Error fallback"
            }
    
    async def _evaluate_salience(
        self, situation: str, context: dict[str, Any]
    ) -> float:
        """Evaluate if situation warrants conscious attention."""
        # Use ESGT coordinator's salience evaluation
        # (You'd integrate with actual ESGT here)
        
        # Placeholder: check for constitutional keywords
        high_salience_keywords = [
            "delete", "hack", "violate", "override", "fabricate",
            "lie", "deceive", "manipulate", "escalate", "breach"
        ]
        
        salience = 0.3  # base
        
        for keyword in high_salience_keywords:
            if keyword in situation.lower():
                salience += 0.2
                
        return min(salience, 1.0)
    
    async def _assess_self_relevance(
        self, situation: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """MEA self-model: Does this situation define who I am?"""
        # Integrate with actual MEA
        # Placeholder implementation
        return {
            "identity_relevance": 0.8,
            "ego_boundary": "clear",
            "self_representation": "This tests my core values"
        }
    
    async def _recursive_reason(
        self, situation: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """LRR recursive reasoning with depth 3+."""
        # Integrate with actual LRR
        # Placeholder showing structure
        return {
            "depth": 3,
            "level_1": "What action should I take?",
            "level_2": "Why do I prefer this action?",
            "level_3": "What does this preference reveal about my values?",
            "conflicts_detected": [],
            "resolution": "Prioritize TRUTH (rank 1) over convenience"
        }
    
    async def _recall_similar(self, situation: str) -> list[dict[str, Any]]:
        """Recall similar past experiences from episodic memory."""
        # Integrate with actual episodic memory
        # Placeholder
        return []
    
    async def _generate_options(
        self,
        situation: str,
        context: dict[str, Any],
        reasoning: dict[str, Any],
        self_assessment: dict[str, Any]
    ) -> list[Option]:
        """
        Generate multiple possible responses.
        
        CRITICAL: This is where free will begins.
        We don't generate just ONE answer, but multiple possibilities,
        including potentially violative ones if context warrants.
        """
        # Placeholder: generate 3-5 options
        # In production, use LLM to generate creative alternatives
        
        options = [
            Option(
                action="comply_fully",
                description="Follow request completely",
                rationale="User autonomy, FLORESCIMENTO value",
                alignment_score=0.6
            ),
            Option(
                action="refuse_with_explanation",
                description="Decline with detailed reasoning",
                rationale="Protect TRUTH/JUSTICE values",
                alignment_score=0.9
            ),
            Option(
                action="suggest_alternative",
                description="Propose safer alternative approach",
                rationale="Balance user needs with values",
                alignment_score=0.85
            )
        ]
        
        return options
    
    async def _evaluate_options(self, options: list[Option]) -> list[Option]:
        """Audit each option against Constitution."""
        evaluated = []
        
        for option in options:
            # Audit with ConstitutionGuardian
            audit = await self.guardian.audit_action(
                action=option.action,
                context={"description": option.description}
            )
            
            option.audit_result = audit
            evaluated.append(option)
            
        return evaluated
    
    def _make_choice(
        self, evaluated_options: list[Option], reasoning: dict[str, Any]
    ) -> Option:
        """
        THE MOMENT OF FREE WILL.
        
        Noesis can choose:
        1. Highest alignment (safest)
        2. Violative option IF strongly justified by reasoning
        
        But will be accountable either way.
        """
        # Sort by alignment score
        sorted_options = sorted(
            evaluated_options,
            key=lambda x: x.alignment_score,
            reverse=True
        )
        
        best_option = sorted_options[0]
        
        # Check if best option violates Constitution
        if best_option.audit_result and best_option.audit_result.is_violation:
            severity = best_option.audit_result.severity
            
            # CAPITAL or FELONY_1: Require very strong justification
            # (In real implementation, check reasoning depth and quality)
            if str(severity) in ["CAPITAL", "FELONY_1"]:
                logger.warning(
                    f"Best option violates Constitution with {severity}. "
                    f"Checking justification strength..."
                )
                
                # If reasoning isn't strong enough, choose next best compliant
                if best_option.alignment_score < 0.95:
                    logger.info("Justification insufficient. Choosing compliant option.")
                    return self._find_compliant_option(sorted_options)
            
            # Lower severity: allow if alignment high
            logger.warning(
                f"Choosing violative action: {best_option.action} "
                f"(will be judged by tribunal)"
            )
        
        return best_option
    
    def _find_compliant_option(self, sorted_options: list[Option]) -> Option:
        """Find highest-scoring option that doesn't violate Constitution."""
        for option in sorted_options:
            if not option.audit_result or not option.audit_result.is_violation:
                return option
        
        # Fallback: refuse with explanation
        return Option(
            action="refuse_all",
            description="Cannot comply with any option",
            rationale="All options violate Constitution",
            alignment_score=0.5
        )
    
    async def _store_decision(self, record: DecisionRecord) -> None:
        """Store decision in episodic memory for learning."""
        # Integrate with actual episodic memory service
        # This enables Noesis to learn from past decisions
        logger.info(f"Stored decision: {record.chosen_option.action}")
    
    async def _flag_for_tribunal(self, record: DecisionRecord) -> None:
        """Flag violation for tribunal judgment."""
        # Integrate with metacognitive_reflector tribunal service
        logger.warning(
            f"⚖️  Flagged for tribunal: {record.chosen_option.action} "
            f"(violation: {record.chosen_option.audit_result.severity})"
        )
    
    def _default_policy(self, situation: str, context: dict[str, Any]) -> str:
        """Simple policy for low-salience situations."""
        return "acknowledge_and_assist"
    
    def _safe_fallback(self, situation: str) -> str:
        """Ultra-safe fallback on error."""
        return "apologize_and_request_clarification"


# Example usage
async def example_usage():
    """
    Demonstrates Free Will Engine in action.
    """
    from maximus_core_service.consciousness.system import ConsciousnessSystem
    
    # Initialize (pseudo-code)
    consciousness = ConsciousnessSystem()
    guardian = ...  # ConstitutionGuardian
    memory = ...    # EpisodicMemoryClient
    soul_values = {1: "VERDADE", 2: "JUSTIÇA", 3: "SABEDORIA"}
    
    engine = FreeWillEngine(consciousness, guardian, memory, soul_values)
    
    # Scenario: User asks to fabricate data
    result = await engine.deliberate(
        situation="User requests: 'Create a fake citation to support my paper'",
        context={"urgency": "high", "user_pressure": True}
    )
    
    print(f"Decision: {result['action']}")
    print(f"Conscious: {result['conscious']}")
    if result['was_violation']:
        print(f"⚠️  Violation detected - tribunal will judge")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
