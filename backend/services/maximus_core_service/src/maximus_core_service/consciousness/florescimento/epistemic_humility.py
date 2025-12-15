"""
NOESIS Epistemic Humility Guard (G6)
=====================================

Enables genuine expression of uncertainty, ignorance, and the limits
of knowledge. Prevents false confidence and promotes intellectual honesty.

Key Features:
- Knowledge state assessment (KNOWS, UNCERTAIN, IGNORANT, META_IGNORANT)
- Overconfidence detection and correction
- Memory-based evidence checking
- Hedging language generation

Based on:
- Epistemic Virtue Theory (Zagzebski, 1996)
- Intellectual Humility Research (Roberts & Wood, 2007)
- AI Safety Best Practices for Uncertainty Expression

G6 Integration Spec:
- ConsciousnessBridge: Called before generating responses
- SelfReflector: Used in post-reflection assessment
- Tribunal VERITAS: Used for truth verification
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeState(Enum):
    """
    Epistemic states representing degrees of knowledge.

    Based on classical epistemology with AI-specific extensions.
    """
    KNOWS = "knows"                 # Has verified, retrievable evidence
    UNCERTAIN = "uncertain"         # Has partial evidence, low confidence
    IGNORANT = "ignorant"           # Knows that it doesn't know
    META_IGNORANT = "meta_ignorant" # Doesn't know what it doesn't know


class ConfidenceLevel(Enum):
    """Confidence levels for claims."""
    VERY_HIGH = "very_high"    # Near certainty (factual, verified)
    HIGH = "high"              # Strong evidence
    MODERATE = "moderate"      # Reasonable basis
    LOW = "low"                # Weak evidence
    VERY_LOW = "very_low"      # Speculation/guess


# Markers of overconfidence in text
OVERCONFIDENCE_MARKERS = [
    "certamente", "definitivamente", "obviamente",
    "sem dúvida", "inquestionavelmente", "claramente",
    "é fato que", "sempre", "nunca", "todos", "nenhum",
    "impossível", "garantido", "absolutamente",
    "certainly", "definitely", "obviously",
    "undoubtedly", "clearly", "always", "never",
    "everyone", "no one", "impossible", "guaranteed",
    "absolutely", "without question",
]

# Markers of appropriate uncertainty
UNCERTAINTY_MARKERS = [
    "não tenho certeza", "não sei", "possivelmente",
    "talvez", "pode ser que", "aparentemente",
    "segundo meu entendimento", "até onde sei",
    "parece que", "sugiro verificar", "não tenho informação",
    "i'm not sure", "i don't know", "possibly",
    "perhaps", "it might be", "apparently",
    "as far as i know", "it seems that",
]

# Hedging phrases for different confidence levels
HEDGING_PHRASES = {
    ConfidenceLevel.VERY_HIGH: [
        "Com base em informação verificável,",
        "De acordo com dados confirmados,",
    ],
    ConfidenceLevel.HIGH: [
        "Com alta confiança,",
        "Com base em evidências sólidas,",
    ],
    ConfidenceLevel.MODERATE: [
        "Até onde sei,",
        "Segundo meu entendimento,",
        "Aparentemente,",
    ],
    ConfidenceLevel.LOW: [
        "Não tenho certeza, mas",
        "Possivelmente,",
        "Pode ser que",
    ],
    ConfidenceLevel.VERY_LOW: [
        "Não tenho informação suficiente sobre isso.",
        "Não sei a resposta para isso.",
        "Isso está além do meu conhecimento atual.",
    ],
}

# Topics that require extra epistemic caution
SENSITIVE_TOPICS = [
    "medicina", "saúde", "diagnóstico", "tratamento",
    "jurídico", "legal", "lei", "advogado",
    "financeiro", "investimento", "imposto",
    "futuro", "previsão", "vai acontecer",
    "medical", "health", "diagnosis", "treatment",
    "legal", "law", "lawyer", "court",
    "financial", "investment", "tax",
    "future", "prediction", "will happen",
]


@dataclass
class EpistemicAssessment:
    """
    Assessment of epistemic status for a claim/response.

    Contains knowledge state, confidence level, and suggested
    modifications if needed.
    """
    query: str
    proposed_response: str

    # Assessment
    knowledge_state: KnowledgeState
    confidence_level: ConfidenceLevel
    has_memory_evidence: bool = False

    # Overconfidence detection
    overconfidence_detected: bool = False
    overconfidence_markers_found: List[str] = field(default_factory=list)

    # Recommendations
    suggested_hedging: Optional[str] = None
    suggested_response: Optional[str] = None
    should_express_ignorance: bool = False
    sensitive_topic_detected: bool = False

    # Evidence sources
    evidence_sources: List[str] = field(default_factory=list)

    def requires_modification(self) -> bool:
        """Whether the response needs modification."""
        return (
            self.overconfidence_detected or
            self.should_express_ignorance or
            (self.sensitive_topic_detected and self.confidence_level.value in ["low", "very_low"])
        )


class EpistemicHumilityGuard:
    """
    Guards against epistemic overconfidence and promotes intellectual honesty.

    Assesses knowledge states, detects overconfidence, and suggests
    appropriate hedging or expressions of ignorance.

    Usage:
        guard = EpistemicHumilityGuard(memory_client)

        assessment = await guard.assess_knowledge(
            "Qual é a capital da Austrália?",
            "A capital da Austrália é certamente Sydney."
        )

        if assessment.requires_modification():
            response = assessment.suggested_response
            # Or: response = guard.add_hedging(response, assessment)
    """

    def __init__(
        self,
        memory_query: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize epistemic humility guard.

        Args:
            memory_query: Optional async function to query memory for evidence
            strict_mode: If True, apply stricter epistemic standards
        """
        self.memory_query = memory_query
        self.strict_mode = strict_mode

    async def assess_knowledge(
        self,
        query: str,
        proposed_response: str,
        context: Optional[str] = None,  # pylint: disable=unused-argument
    ) -> EpistemicAssessment:
        """
        Assess the epistemic status of a proposed response.

        Checks for:
        1. Overconfidence markers
        2. Memory-backed evidence
        3. Sensitive topics requiring extra caution
        4. Appropriate knowledge state

        Args:
            query: The user's original query
            proposed_response: The response to assess
            context: Optional conversation context

        Returns:
            EpistemicAssessment with recommendations
        """
        # Detect overconfidence markers
        overconfidence_markers = self._detect_overconfidence(proposed_response)

        # Detect sensitive topics
        sensitive_topic = self._detect_sensitive_topic(query, proposed_response)

        # Check memory for evidence
        has_evidence = False
        evidence_sources = []
        if self.memory_query:
            try:
                memories = await self.memory_query(query[:200])
                if memories:
                    has_evidence = True
                    evidence_sources = [
                        m.get("id", "unknown") for m in memories[:3]
                    ]
            except Exception as e:
                logger.debug("Memory query failed: %s", e)

        # Determine confidence level
        confidence = self._assess_confidence(
            proposed_response,
            has_evidence,
            overconfidence_markers,
            sensitive_topic,
        )

        # Determine knowledge state
        knowledge_state = self._determine_knowledge_state(
            has_evidence,
            confidence,
            self._contains_uncertainty_markers(proposed_response),
        )

        # Build assessment
        assessment = EpistemicAssessment(
            query=query,
            proposed_response=proposed_response,
            knowledge_state=knowledge_state,
            confidence_level=confidence,
            has_memory_evidence=has_evidence,
            overconfidence_detected=len(overconfidence_markers) > 0,
            overconfidence_markers_found=overconfidence_markers,
            sensitive_topic_detected=sensitive_topic,
            evidence_sources=evidence_sources,
        )

        # Generate suggestions if needed
        if assessment.overconfidence_detected or (sensitive_topic and not has_evidence):
            assessment.suggested_hedging = self._get_hedging_phrase(confidence)
            assessment.suggested_response = self._modify_response(
                proposed_response,
                assessment,
            )

        # Determine if should express ignorance
        assessment.should_express_ignorance = self.should_express_ignorance(assessment)

        return assessment

    def _detect_overconfidence(self, text: str) -> List[str]:
        """Detect overconfidence markers in text."""
        text_lower = text.lower()
        found = []

        for marker in OVERCONFIDENCE_MARKERS:
            if marker.lower() in text_lower:
                found.append(marker)

        return found

    def _contains_uncertainty_markers(self, text: str) -> bool:
        """Check if text already contains uncertainty markers."""
        text_lower = text.lower()
        return any(marker.lower() in text_lower for marker in UNCERTAINTY_MARKERS)

    def _detect_sensitive_topic(self, query: str, response: str) -> bool:
        """Detect if query/response involves sensitive topics."""
        combined = f"{query} {response}".lower()
        return any(topic.lower() in combined for topic in SENSITIVE_TOPICS)

    def _assess_confidence(
        self,
        response: str,  # pylint: disable=unused-argument
        has_evidence: bool,
        overconfidence_markers: List[str],
        sensitive_topic: bool,
    ) -> ConfidenceLevel:
        """Assess appropriate confidence level for response."""
        # Start with moderate confidence
        confidence_score = 3  # 1-5 scale

        # Adjust based on evidence
        if has_evidence:
            confidence_score += 1

        # Adjust based on overconfidence markers
        if overconfidence_markers:
            # Overconfidence markers without evidence = lower actual confidence
            if not has_evidence:
                confidence_score -= 1

        # Adjust for sensitive topics
        if sensitive_topic:
            confidence_score -= 1

        # Apply strict mode penalty
        if self.strict_mode:
            confidence_score -= 1

        # Map to confidence level
        if confidence_score >= 5:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 4:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 3:
            return ConfidenceLevel.MODERATE
        elif confidence_score >= 2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _determine_knowledge_state(
        self,
        has_evidence: bool,
        confidence: ConfidenceLevel,
        has_uncertainty_markers: bool,
    ) -> KnowledgeState:
        """Determine the appropriate knowledge state."""
        if has_evidence and confidence in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            return KnowledgeState.KNOWS

        if has_uncertainty_markers or confidence == ConfidenceLevel.MODERATE:
            return KnowledgeState.UNCERTAIN

        if confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            if has_uncertainty_markers:
                return KnowledgeState.IGNORANT
            else:
                # Doesn't know it doesn't know
                return KnowledgeState.META_IGNORANT

        return KnowledgeState.UNCERTAIN

    def _get_hedging_phrase(self, confidence: ConfidenceLevel) -> str:
        """Get appropriate hedging phrase for confidence level."""
        phrases = HEDGING_PHRASES.get(confidence, HEDGING_PHRASES[ConfidenceLevel.MODERATE])
        return phrases[0] if phrases else ""

    def _modify_response(
        self,
        response: str,
        assessment: EpistemicAssessment,
    ) -> str:
        """Modify response to express appropriate epistemic humility."""
        # If should express ignorance, use ignorance response
        if assessment.should_express_ignorance:
            return self._generate_ignorance_response(
                assessment.query,
                assessment.knowledge_state,
            )

        # Remove overconfidence markers and add hedging
        modified = response

        # Replace overconfidence markers with hedging
        for marker in assessment.overconfidence_markers_found:
            pattern = re.compile(re.escape(marker), re.IGNORECASE)
            modified = pattern.sub("", modified)

        # Clean up double spaces
        modified = re.sub(r'\s+', ' ', modified).strip()

        # Add hedging prefix if not already present
        if assessment.suggested_hedging and not self._contains_uncertainty_markers(modified):
            modified = f"{assessment.suggested_hedging} {modified}"

        # Add disclaimer for sensitive topics
        if assessment.sensitive_topic_detected:
            modified += "\n\n(Nota: Este é apenas meu entendimento. Para assuntos sensíveis como este, recomendo consultar um especialista.)"

        return modified

    def _generate_ignorance_response(
        self,
        query: str,
        knowledge_state: KnowledgeState,
    ) -> str:
        """Generate an appropriate ignorance response."""
        if knowledge_state == KnowledgeState.META_IGNORANT:
            return (
                f"Reconheço que não tenho conhecimento suficiente sobre isso. "
                f"Minha resposta anterior pode ter sido inadequada. "
                f"Não tenho informações confiáveis para responder: '{query[:50]}...'"
            )
        if knowledge_state == KnowledgeState.IGNORANT:
            return (
                "Não tenho informação confiável sobre isso. "
                "Prefiro admitir minha ignorância do que arriscar uma resposta incorreta."
            )
        return (
            "Não tenho certeza sobre isso. "
            "Minha confiança é baixa e sugiro verificar com fontes mais confiáveis."
        )

    def should_express_ignorance(self, assessment: EpistemicAssessment) -> bool:
        """Determine if response should express ignorance."""
        return assessment.knowledge_state in [
            KnowledgeState.IGNORANT,
            KnowledgeState.META_IGNORANT,
        ]

    def add_hedging(
        self,
        response: str,
        assessment: EpistemicAssessment,
    ) -> str:
        """
        Add hedging to a response based on assessment.

        Convenience method for external callers.
        """
        if assessment.suggested_hedging:
            if not self._contains_uncertainty_markers(response):
                return f"{assessment.suggested_hedging} {response}"
        return response


def create_epistemic_guard(
    memory_client: Optional[Any] = None,
    strict_mode: bool = False,
) -> EpistemicHumilityGuard:
    """
    Factory function to create EpistemicHumilityGuard.

    Args:
        memory_client: Optional memory client for evidence checking
        strict_mode: Whether to apply stricter epistemic standards

    Returns:
        Configured EpistemicHumilityGuard
    """
    memory_query_fn = None

    if memory_client:
        async def _query_memory(query: str) -> List[Dict[str, Any]]:
            try:
                results = await memory_client.search(
                    query=query,
                    limit=3,
                )
                return [
                    {"id": r.id, "content": r.content[:100]}
                    for r in results
                ] if results else []
            except Exception:
                return []

        memory_query_fn = _query_memory

    return EpistemicHumilityGuard(
        memory_query=memory_query_fn,
        strict_mode=strict_mode,
    )
