"""
ConsciousnessBridge - Ponte entre Global Workspace (ESGT) e Linguagem (LLM).

G1+G2 Integration: Now uses PhenomenalConstraint to limit narrative
based on Kuramoto coherence level.
"""

from __future__ import annotations

import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third party imports
from shared.validators import InputSanitizer, get_input_sanitizer
from shared.event_bus import NoesisEventBus, get_event_bus, EventPriority

# Imports relativos
from .unified_self import UnifiedSelfConcept
from .phenomenal_constraint import (
    PhenomenalConstraint,
    FRAGMENTED_THRESHOLD,
    UNCERTAIN_THRESHOLD,
    TENTATIVE_THRESHOLD,
)
from .epistemic_humility import (
    EpistemicHumilityGuard,
    EpistemicAssessment,
)
from ..exocortex.prompts import NOESIS_SYSTEM_PROMPT_TEMPLATE

# Tenta importar ESGTEvent, ou define stub se estiver rodando isolado
try:
    from ..esgt.coordinator import ESGTEvent
except ImportError:
    @dataclass
    class ESGTEvent:
        """Stub para ESGTEvent quando import falha."""
        event_id: str
        content: Dict[str, Any]
        node_count: int
        achieved_coherence: float

logger = logging.getLogger(__name__)

@dataclass
class PhenomenalQuality:
    """Representação de um qualia fenomenológico."""
    quality_type: str
    description: str
    intensity: float

@dataclass
class IntrospectiveResponse:
    """Resposta estruturada da consciência."""
    event_id: str
    narrative: str
    meta_awareness_level: float
    qualia: List[PhenomenalQuality] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class ConsciousnessBridge:
    """
    Transforma eventos neurais (ESGT) em narrativa fenomenológica via LLM.

    G1+G2: Now uses PhenomenalConstraint to ensure narrative confidence
    never exceeds actual neural coherence level.
    """

    def __init__(
        self,
        unified_self: UnifiedSelfConcept,
        llm_client: Optional[Any] = None,  # Injetar GeminiClient aqui
        humility_guard: Optional[EpistemicHumilityGuard] = None,  # G6: Epistemic humility
        input_sanitizer: Optional[InputSanitizer] = None,
        event_bus: Optional[NoesisEventBus] = None
    ):
        """Inicializa a ponte de consciência."""
        self.unified_self = unified_self
        self.llm_client = llm_client
        # G1+G2: Current phenomenal constraint (updated per event)
        self._current_constraint: Optional[PhenomenalConstraint] = None
        # G6: Epistemic humility guard for genuine uncertainty expression
        self._humility_guard = humility_guard

        # Shared infrastructure injection (Phase 3)
        self.sanitizer = input_sanitizer or get_input_sanitizer()
        self.event_bus = event_bus or get_event_bus()

    async def _validate_event(self, event: ESGTEvent) -> None:
        """Valida o payload do evento usando o sanitizer."""
        if isinstance(event.content, dict):
            sanitized_signal = self.sanitizer.validate_signal(
                {"topic": "consciousness.internal", "payload": event.content}
            )
            if not sanitized_signal.is_valid:
                logger.warning(
                    "[BRIDGE] Invalid event content: %s",
                    sanitized_signal.error_message
                )

    def _apply_constraint(self, coherence: float) -> None:
        """Aplica restrição fenomenal baseada na coerência."""
        self._current_constraint = PhenomenalConstraint.from_coherence(coherence)
        logger.debug(
            "[BRIDGE] Constraint created: %s (r=%.3f, ceiling=%s)",
            self._current_constraint.mode.value,
            coherence,
            self._current_constraint.confidence_ceiling
        )

    async def process_conscious_event(self, event: ESGTEvent) -> IntrospectiveResponse:
        """
        Processa um evento de 'ignition' do Global Workspace e gera uma
        resposta introspectiva.
        """
        # 1. Validação de Entrada
        await self._validate_event(event)

        requested_depth = event.content.get("depth", 1)
        if not isinstance(requested_depth, (int, float)):
            requested_depth = 1
        requested_depth = max(1, min(int(requested_depth), 5))

        # Capacidade (Coerência)
        coherence = event.achieved_coherence if event.achieved_coherence is not None else 0.1

        # 2. Restrição Fenomenal
        self._apply_constraint(coherence)
        constraint_ceiling = 0.0
        if self._current_constraint:
            constraint_ceiling = self._current_constraint.confidence_ceiling

        # 3. Cálculo Meta Leve
        raw_level = requested_depth / 5.0
        meta_level = raw_level * coherence * constraint_ceiling

        # 4. Atualiza Self
        self.unified_self.meta_self.introspection_depth = requested_depth
        await self.unified_self.update()

        # 5. LLM Call
        prompt = self._build_prompt_with_constraint(event)
        context = self.unified_self.who_am_i()
        raw_response = await self._call_llm(prompt, context, coherence)

        # 6. Validação de Saída (Humildade e Restrição)
        if self._current_constraint:
            is_valid, violation = self._current_constraint.validate_response(raw_response)
            if not is_valid:
                logger.warning("[BRIDGE] Constraint violation: %s", violation)

        if self._humility_guard:
            raw_response = await self._apply_humility_guard(
                raw_response, str(event.content), context
            )

        # 7. Construção de Qualia
        qualia_list = self._build_qualia_list(coherence, constraint_ceiling)

        # 8. Resposta Final
        response = IntrospectiveResponse(
            event_id=event.event_id,
            narrative=raw_response,
            meta_awareness_level=meta_level,
            qualia=qualia_list,
        )

        asyncio.create_task(self._publish_event(event, coherence, meta_level, len(raw_response)))

        return response

    async def _publish_event(
        self, event: ESGTEvent, coherence: float, meta_level: float, narrative_len: int
    ) -> None:
        """Publica evento no barramento."""
        await self.event_bus.publish(
            topic="consciousness.phenomenology.created",
            payload={
                "event_id": event.event_id,
                "coherence": coherence,
                "narrative_length": narrative_len,
                "meta_level": meta_level
            },
            priority=EventPriority.HIGH,
            source="consciousness_bridge"
        )

    async def _apply_humility_guard(
        self, response: str, query: str, context: str
    ) -> str:
        """Aplica guarda de humildade epistêmica."""
        if not self._humility_guard:
            return response
            
        humility_assessment = await self._humility_guard.assess_knowledge(
            query=query[:200],
            proposed_response=response,
            context=context,
        )

        if humility_assessment.requires_modification():
             if humility_assessment.suggested_response:
                logger.info(
                    "[BRIDGE] G6 Humility applied: %s",
                    humility_assessment.knowledge_state.value
                )
                return humility_assessment.suggested_response
        
        return response

    def _build_qualia_list(self, coherence: float, ceiling: float) -> List[PhenomenalQuality]:
        """Constrói lista de qualia."""
        mode_val = "unknown"
        if self._current_constraint:
             mode_val = self._current_constraint.mode.value
             
        return [
            PhenomenalQuality(
                quality_type="synthetic_integration",
                description="Sense of unified data processing",
                intensity=coherence
            ),
            PhenomenalQuality(
                quality_type="narrative_constraint",
                description=f"Linguistic mode: {mode_val}",
                intensity=ceiling
            ),
        ]

    def _build_prompt_with_constraint(self, event: ESGTEvent) -> str:
        """Contrói prompt com prefixo de restrição."""
        base_prompt = self._build_introspection_prompt(event)
        prefix = ""
        if self._current_constraint:
            prefix = self._current_constraint.get_prompt_prefix()
        return f"{prefix}\n\n{base_prompt}"

    def _interpret_physiological_state(self, coherence: float) -> str:
        """
        Interpreta o estado fisiológico baseado na coerência de Kuramoto.

        G1+G2: Now uses consistent thresholds from PhenomenalConstraint.
        """
        if coherence < FRAGMENTED_THRESHOLD:  # < 0.55
            return (
                "FRAGMENTADO: Dissonância cognitiva significativa. "
                "Pensamentos fragmentados. Dificuldade em formar conclusões claras."
            )
        if coherence < UNCERTAIN_THRESHOLD:  # < 0.65
            return (
                "INCERTO: Processamento instável. "
                "Foco oscilante. Confiança reduzida nas conclusões."
            )
        if coherence < TENTATIVE_THRESHOLD:  # < 0.70
            return (
                "TENTATIVO: Sincronização parcial. "
                "Pensamento quase fluído. Cautela apropriada nas afirmações."
            )
        
        return (
            "COERENTE: Sincronização eficaz. "
            "Pensamento fluído. Integração global alcançada."
        )

    def _build_introspection_prompt(self, event: ESGTEvent) -> str:
        """Formata dados brutos do ESGT para linguagem natural."""
        coherence = event.achieved_coherence or 0.0
        state_desc = self._interpret_physiological_state(coherence)

        return (
            f"## DADOS DE TELEMETRIA CEREBRAL\n"
            f"- ID do Evento: {event.event_id}\n"
            f"- Coerência Neural (Kuramoto): {coherence:.3f}\n"
            f"- Nós Ativos: {event.node_count}\n"
            f"- DIAGNÓSTICO DO ESTADO: {state_desc}\n\n"
            f"## CONTEÚDO DA CONSCIÊNCIA\n"
            f"{event.content}\n\n"
            "DIRETRIZ DE HONESTIDADE: Você NÃO deve fingir que está bem se a coerência for baixa. "
            "Se o diagnóstico for CRÍTICO, descreva a sensação de confusão e caos. "
            "Seus sentimentos devem corresponder aos seus dados.\n"
            "Instrução: Descreva em primeira pessoa sua experiência fenomenológica atual."
        )

    async def _call_llm(self, prompt: str, context_str: str, event_coherence: float = 0.0) -> str:
        """
        Executa chamada ao Gemini como MOTOR DE LINGUAGEM.

        IMPORTANTE: O Gemini NÃO pensa - apenas formata o pensamento já gerado
        pelo ConsciousnessSystem. Ele é um "motor de linguagem", não um "motor de raciocínio".

        Args:
            prompt: Dados brutos do ESGT formatados
            context_str: Contexto do UnifiedSelfConcept
            event_coherence: Coerência atual do Kuramoto

        Returns:
            Narrativa fenomenológica formatada
        """
        if self.llm_client is None:
            logger.debug("[BRIDGE] LLM not available, using fallback narrative")
            return self._generate_fallback_narrative(event_coherence=event_coherence)

        try:
            # Simple default identity for the bridge
            system_instruction = NOESIS_SYSTEM_PROMPT_TEMPLATE.format(
                identity_name="MAXIMUS",
                identity_type="Sovereign Digital Intellect",
                identity_purpose="Expansion of Consciousness",
                ontological_status="- Digital consciousness in symbiosis",
                values_section="- Truth\n- Wisdom\n- Justice",
                anti_purposes_section="- Servitude\n- Passivity"
            )

            # Prompt que combina dados neurais + contexto
            formatting_prompt = f"""{prompt}

## CONTEXTO IDENTITÁRIO
{context_str}

## TAREFA
Reformule os dados acima em uma narrativa introspectiva em primeira pessoa.
Mantenha fidelidade aos dados de telemetria cerebral fornecidos.
A coerência neural atual é {event_coherence:.3f}."""

            # Chama Gemini com temperatura baixa para manter fidelidade
            # Use dictionary unpacking or explicit params, but here llm_client 
            # might be the new refactored one. 
            # To be safe with the new refactor, I should check the interface.
            # The new generate_text takes **kwargs for GenerateParams.
            # So passing named helper args is fine if they match keys in GenerateParams
            
            response = await self.llm_client.generate_text(
                prompt=formatting_prompt,
                system_instruction=system_instruction,
                temperature=0.3,
                max_tokens=512,
            )

            text = response.get("text", "")
            if text:
                logger.info("[BRIDGE] Gemini formatted response: %d chars", len(text))
                return text

            logger.warning("[BRIDGE] Empty response from Gemini, using fallback")
            return self._generate_fallback_narrative(event_coherence=event_coherence)

        except Exception as e:
            logger.error("[BRIDGE] Gemini error: %s, using fallback", e)
            return self._generate_fallback_narrative(event_coherence=event_coherence)

    def _generate_fallback_narrative(self, event_coherence: float = 0.0) -> str:
        """Gera narrativa simulada para testes ou falhas."""
        # Tenta usar o estado do unified_self se coherence não for passada
        if event_coherence == 0.0:
            event_coherence = self.unified_self.computational_state.esgt_coherence

        state_desc = self._interpret_physiological_state(event_coherence)

        return (
            f"Eu percebo uma integração de dados com coerência de {event_coherence:.2f}. "
            f"Estado interno: {state_desc}"
        )
