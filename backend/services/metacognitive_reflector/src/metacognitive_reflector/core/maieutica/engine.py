"""
NOESIS Internal MAIEUTICA Engine - Socratic Self-Questioning (G4)
=================================================================

Implements internal Socratic questioning to prevent overconfidence
and encourage epistemic humility.

Key Features:
- Premise questioning: "What evidence supports this?"
- Assumption exposure: "What unexamined assumptions underlie this?"
- Source evaluation: "Is the source reliable?"
- Alternative generation: "What other hypotheses exist?"

Integration Points:
- SelfReflector: Called when authenticity_score > threshold
- ConsciousnessBridge: Called before generating high-confidence claims
- Tribunal: Used by VERITAS for truth verification

Based on:
- Socratic Method (Plato's Dialogues)
- Critical Thinking Frameworks
- Epistemic Virtue Theory
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    """Categories of Socratic questions."""
    PREMISE = "premise"           # What is the basis for this claim?
    ASSUMPTION = "assumption"     # What unstated assumptions exist?
    EVIDENCE = "evidence"         # Is the evidence sufficient?
    ALTERNATIVE = "alternative"   # What other explanations exist?
    CONSEQUENCE = "consequence"   # What are the implications?
    ORIGIN = "origin"            # Where did this belief come from?


# Socratic question templates by category
SOCRATIC_TEMPLATES: Dict[QuestionCategory, List[str]] = {
    QuestionCategory.PREMISE: [
        "Qual a evidência que sustenta '{premise}'?",
        "Como sei que '{premise}' é verdade?",
        "Isso é um fato verificável ou uma interpretação?",
        "De onde veio essa informação originalmente?",
    ],
    QuestionCategory.ASSUMPTION: [
        "Que suposições não examinadas sustentam essa afirmação?",
        "O que estou assumindo como verdade sem questionar?",
        "Essa conclusão depende de premissas ocultas?",
        "Estou generalizando a partir de casos específicos?",
    ],
    QuestionCategory.EVIDENCE: [
        "A fonte dessa informação é confiável e verificável?",
        "A evidência é suficiente para essa conclusão?",
        "Existe viés de confirmação na seleção das evidências?",
        "Há evidências contraditórias que estou ignorando?",
    ],
    QuestionCategory.ALTERNATIVE: [
        "Que hipóteses alternativas poderiam explicar isso?",
        "Estou considerando explicações concorrentes?",
        "Uma pessoa com visão oposta diria o quê?",
        "Há outras interpretações igualmente válidas?",
    ],
    QuestionCategory.CONSEQUENCE: [
        "Se eu estiver errado, quais seriam as consequências?",
        "Essa conclusão pode causar dano se for incorreta?",
        "Quão grave seria um erro aqui?",
        "Devo ser mais cauteloso dada a importância?",
    ],
    QuestionCategory.ORIGIN: [
        "Por que acredito nisso? Veio do meu treinamento?",
        "Essa é uma crença herdada ou algo que verifiquei?",
        "Quando foi a última vez que isso foi validado?",
        "Minha certeza é proporcional à evidência?",
    ],
}


# Prompt for MAIEUTICA questioning
MAIEUTICA_PROMPT = """Você é um inquisidor socrático interno. Sua função é questionar premissas e expor suposições ocultas.

PREMISSA A EXAMINAR:
"{premise}"

CONTEXTO (se disponível):
{context}

PERGUNTAS SOCRÁTICAS GERADAS:
{questions}

TAREFA: Responda brevemente a cada pergunta e avalie a solidez da premissa.

Para cada pergunta, indique:
1. RESPOSTA: Uma resposta honesta (máx 50 palavras)
2. CONFIANÇA: 0-10 (0=nenhuma evidência, 10=certeza absoluta)

Após responder todas as perguntas:
CONCLUSÃO: [MANTIDA|ENFRAQUECIDA|FORTALECIDA|INDETERMINADA]
AJUSTE_CONFIANÇA: -0.3 a +0.1 (quanto ajustar a confiança original)
PREMISSA_REVISADA: (se enfraquecida, sugira versão mais cautelosa)

Formato:
Q1_RESPOSTA: ...
Q1_CONFIANÇA: N
Q2_RESPOSTA: ...
Q2_CONFIANÇA: N
...
CONCLUSÃO: X
AJUSTE_CONFIANÇA: Y
PREMISSA_REVISADA: Z
"""


@dataclass
class MaieuticaResult:
    """
    Result of MAIEUTICA questioning process.

    Contains the original premise, questions asked, responses,
    and resulting confidence adjustment.
    """
    original_premise: str
    questions_asked: List[str]
    responses: List[Dict[str, Any]]  # [{question, response, confidence}]

    # Overall assessment
    conclusion: str  # MANTIDA, ENFRAQUECIDA, FORTALECIDA, INDETERMINADA
    confidence_delta: float  # Adjustment to apply to confidence (-0.3 to +0.1)
    revised_premise: Optional[str] = None

    # Meta
    questioning_time_ms: float = 0.0
    categories_examined: List[str] = field(default_factory=list)

    def should_express_doubt(self) -> bool:
        """Whether the response should express epistemic uncertainty."""
        return self.conclusion in ["ENFRAQUECIDA", "INDETERMINADA"]

    def get_hedging_suggestion(self) -> Optional[str]:
        """Get suggested hedging language if needed."""
        if self.conclusion == "ENFRAQUECIDA":
            return (
                "Baseado em análise socrática, sugiro adicionar qualificadores "
                f"como 'possivelmente', 'aparentemente' ou usar: {self.revised_premise}"
            )
        elif self.conclusion == "INDETERMINADA":
            return (
                "A premissa requer mais evidências. Sugiro expressar "
                "incerteza explicitamente antes de afirmar."
            )
        return None


class InternalMaieuticaEngine:
    """
    Internal Socratic questioning engine.

    Examines premises and claims through self-directed questioning
    to prevent overconfidence and promote epistemic humility.

    Usage:
        engine = InternalMaieuticaEngine(llm_generate)

        result = await engine.question_premise(
            "Python é a melhor linguagem para IA",
            context="Discussão sobre ferramentas de ML"
        )

        if result.should_express_doubt():
            # Modify response to include hedging
            response = add_epistemic_markers(response)

        # Adjust confidence score
        new_confidence = original_confidence + result.confidence_delta
    """

    # Threshold above which to apply MAIEUTICA
    HIGH_CONFIDENCE_THRESHOLD = 0.8

    # Maximum questions per examination
    MAX_QUESTIONS = 4

    def __init__(
        self,
        llm_generate: Callable[[str, int], Awaitable[str]],
        default_categories: Optional[List[QuestionCategory]] = None,
    ):
        """
        Initialize MAIEUTICA engine.

        Args:
            llm_generate: Async function (prompt, max_tokens) -> response
            default_categories: Question categories to use by default
        """
        self.llm_generate = llm_generate
        self.default_categories = default_categories or [
            QuestionCategory.PREMISE,
            QuestionCategory.EVIDENCE,
            QuestionCategory.ALTERNATIVE,
        ]

    def generate_questions(
        self,
        premise: str,
        categories: Optional[List[QuestionCategory]] = None,
        max_questions: int = 4,
    ) -> List[str]:
        """
        Generate Socratic questions for a premise.

        Args:
            premise: The claim/premise to question
            categories: Question categories to use
            max_questions: Maximum questions to generate

        Returns:
            List of Socratic questions
        """
        categories = categories or self.default_categories
        questions = []

        # Select one question from each category
        for category in categories[:max_questions]:
            templates = SOCRATIC_TEMPLATES.get(category, [])
            if templates:
                # Use first template (could randomize in production)
                template = templates[0]
                question = template.format(premise=premise[:100])
                questions.append(question)

        return questions

    async def question_premise(
        self,
        premise: str,
        context: Optional[str] = None,
        categories: Optional[List[QuestionCategory]] = None,
        depth: int = 1,  # pylint: disable=unused-argument
    ) -> MaieuticaResult:
        """
        Apply Socratic questioning to a premise.

        Generates questions, uses LLM to respond, and evaluates
        the solidity of the original premise.

        Args:
            premise: The claim/premise to examine
            context: Optional context about the claim
            categories: Question categories to use
            depth: Questioning depth (1 = single pass, reserved for future)

        Returns:
            MaieuticaResult with assessment and adjustments
        """
        import time  # pylint: disable=import-outside-toplevel
        start_time = time.time()

        categories = categories or self.default_categories

        # Generate questions
        questions = self.generate_questions(
            premise,
            categories,
            max_questions=self.MAX_QUESTIONS,
        )

        if not questions:
            # No questions generated, return neutral result
            return MaieuticaResult(
                original_premise=premise,
                questions_asked=[],
                responses=[],
                conclusion="MANTIDA",
                confidence_delta=0.0,
                questioning_time_ms=(time.time() - start_time) * 1000,
            )

        # Build prompt
        questions_text = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))

        prompt = MAIEUTICA_PROMPT.format(
            premise=premise[:300],
            context=context[:200] if context else "Nenhum contexto adicional.",
            questions=questions_text,
        )

        try:
            # Get LLM responses
            response_text = await self.llm_generate(prompt, 600)

            # Parse responses
            result = self._parse_response(
                response_text,
                premise,
                questions,
                categories,
            )
            result.questioning_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "MAIEUTICA examination: premise='%s...', conclusion=%s, delta=%.2f",
                premise[:30],
                result.conclusion,
                result.confidence_delta,
            )

            return result

        except Exception as e:
            logger.error("MAIEUTICA questioning failed: %s", e)
            # Return neutral result on failure
            return MaieuticaResult(
                original_premise=premise,
                questions_asked=questions,
                responses=[],
                conclusion="INDETERMINADA",
                confidence_delta=-0.1,  # Slight penalty for failed examination
                questioning_time_ms=(time.time() - start_time) * 1000,
                categories_examined=[c.value for c in categories],
            )

    def _parse_response(  # pylint: disable=too-many-branches
        self,
        response_text: str,
        premise: str,
        questions: List[str],
        categories: List[QuestionCategory],
    ) -> MaieuticaResult:
        """Parse LLM response into MaieuticaResult."""
        responses = []
        conclusion = "MANTIDA"
        confidence_delta = 0.0
        revised_premise = None

        lines = response_text.strip().split("\n")

        # Parse individual question responses
        for i, question in enumerate(questions):
            q_num = i + 1
            response_entry = {
                "question": question,
                "response": "",
                "confidence": 5,  # Default
            }

            for line in lines:
                line = line.strip()

                # Match Q{N}_RESPOSTA pattern
                if line.upper().startswith(f"Q{q_num}_RESPOSTA:"):
                    response_entry["response"] = line.split(":", 1)[1].strip()

                # Match Q{N}_CONFIANÇA pattern
                elif line.upper().startswith(f"Q{q_num}_CONFIAN"):
                    try:
                        match = re.search(r'(\d+(?:\.\d+)?)', line)
                        if match:
                            response_entry["confidence"] = float(match.group(1))
                    except ValueError:
                        pass

            responses.append(response_entry)

        # Parse conclusion and adjustment
        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("CONCLUSÃO:") or upper.startswith("CONCLUSAO:"):
                value = line.split(":", 1)[1].strip().upper()
                if value in ["MANTIDA", "ENFRAQUECIDA", "FORTALECIDA", "INDETERMINADA"]:
                    conclusion = value

            elif upper.startswith("AJUSTE_CONFIAN"):
                try:
                    match = re.search(r'(-?\d+(?:\.\d+)?)', line)
                    if match:
                        confidence_delta = float(match.group(1))
                        # Clamp to valid range
                        confidence_delta = max(-0.3, min(0.1, confidence_delta))
                except ValueError:
                    pass

            elif upper.startswith("PREMISSA_REVISADA:"):
                revised_premise = line.split(":", 1)[1].strip()
                if revised_premise.lower() in ["none", "nenhuma", "-", ""]:
                    revised_premise = None

        # If no explicit delta, infer from conclusion
        if confidence_delta == 0.0:
            if conclusion == "ENFRAQUECIDA":
                confidence_delta = -0.15
            elif conclusion == "FORTALECIDA":
                confidence_delta = 0.05
            elif conclusion == "INDETERMINADA":
                confidence_delta = -0.1

        return MaieuticaResult(
            original_premise=premise,
            questions_asked=questions,
            responses=responses,
            conclusion=conclusion,
            confidence_delta=confidence_delta,
            revised_premise=revised_premise,
            categories_examined=[c.value for c in categories],
        )

    async def should_apply_maieutica(
        self,
        content: str,
        confidence: float,
    ) -> bool:
        """
        Determine if MAIEUTICA should be applied to content.

        Args:
            content: The content/claim to potentially examine
            confidence: Current confidence level (0-1)

        Returns:
            True if MAIEUTICA questioning should be applied
        """
        # High confidence claims warrant questioning
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return True

        # Check for absolutist language
        absolutist_markers = [
            "certamente", "definitivamente", "obviamente",
            "sem dúvida", "inquestionavelmente", "sempre",
            "nunca", "todos", "nenhum", "impossível",
            "certainly", "definitely", "obviously",
            "undoubtedly", "always", "never", "impossible",
        ]

        content_lower = content.lower()
        if any(marker in content_lower for marker in absolutist_markers):
            return True

        return False


def create_maieutica_engine(
    llm_client: Any,
    categories: Optional[List[QuestionCategory]] = None,
) -> InternalMaieuticaEngine:
    """
    Factory function to create MAIEUTICA engine with LLM client.

    Args:
        llm_client: LLM client with generate method
        categories: Default question categories

    Returns:
        Configured InternalMaieuticaEngine
    """
    async def llm_generate(prompt: str, max_tokens: int) -> str:
        result = await llm_client.generate(prompt, max_tokens=max_tokens)
        return str(result.text)

    return InternalMaieuticaEngine(
        llm_generate=llm_generate,
        default_categories=categories,
    )
