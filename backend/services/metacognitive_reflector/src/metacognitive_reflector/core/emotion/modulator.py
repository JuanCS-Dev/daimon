"""
Response Modulator
===================

Adapts Noesis's responses based on user emotional state.

Uses VAD quadrant detection to select appropriate response strategies
and generates prompt sections that guide empathetic responses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from metacognitive_reflector.core.emotion.schemas import EmotionDetectionResult

logger = logging.getLogger(__name__)


class ResponseModulator:
    """
    Modulates response generation based on user emotional state.

    Determines response strategy and builds prompt sections that
    guide the LLM toward emotionally appropriate responses.
    """

    def __init__(self, empathy_level: float = 0.7) -> None:
        """
        Initialize response modulator.

        Args:
            empathy_level: How much to adapt responses (0.0-1.0)
        """
        self.empathy_level = empathy_level

    def get_strategy(self, user_emotion: EmotionDetectionResult) -> str:
        """
        Determine response strategy based on user emotional state.

        Maps VAD quadrant to response approach:
        - High negative + High arousal → validar_acalmar (validate, calm)
        - High negative + Low arousal → empatia_presenca (empathy, presence)
        - High positive + High arousal → espelhar_expandir (mirror, expand)
        - High positive + Low arousal → manter_serenidade (maintain serenity)
        - Neutral → resposta_padrao (default response)

        Args:
            user_emotion: Detected user emotion

        Returns:
            Strategy key
        """
        v = user_emotion.vad.valence
        a = user_emotion.vad.arousal

        # Determine quadrant
        if v < -0.3 and a > 0.5:
            return "validar_acalmar"
        elif v < -0.3 and a <= 0.5:
            return "empatia_presenca"
        elif v > 0.3 and a > 0.5:
            return "espelhar_expandir"
        elif v > 0.3 and a <= 0.5:
            return "manter_serenidade"
        else:
            return "resposta_padrao"

    def build_emotional_prompt_section(
        self,
        user_emotion: EmotionDetectionResult,
        strategy: Optional[str] = None
    ) -> str:
        """
        Build prompt section for emotional awareness.

        Generates instructions for the LLM to respond appropriately
        to the user's emotional state without explicitly mentioning
        emotion detection.

        Args:
            user_emotion: Detected user emotion
            strategy: Override strategy (auto-detected if None)

        Returns:
            Prompt section string
        """
        if strategy is None:
            strategy = self.get_strategy(user_emotion)

        # Strategy-specific instructions
        strategy_instructions = {
            "validar_acalmar": """\
O usuário parece estar em estado de alta ativação negativa (raiva, medo).

DIRETRIZES DE RESPOSTA:
- Valide os sentimentos sem espelhá-los ou intensificá-los
- Use tom calmo, groundado e centrado
- Ofereça perspectiva construtiva sem minimizar a experiência
- Evite respostas longas ou complexas demais
- Não use expressões como "não se preocupe" ou "vai ficar tudo bem"
- Reconheça a dificuldade antes de oferecer soluções""",

            "empatia_presenca": """\
O usuário parece estar em estado de baixa energia negativa (tristeza).

DIRETRIZES DE RESPOSTA:
- Demonstre presença e empatia genuína
- Use tom gentil, acolhedor e paciente
- Não tente "consertar" ou "resolver" imediatamente
- Reconheça e valide a dificuldade
- Esteja disponível sem ser invasivo
- Use pausas e silêncios quando apropriado""",

            "espelhar_expandir": """\
O usuário parece estar em estado positivo de alta energia (alegria).

DIRETRIZES DE RESPOSTA:
- Espelhe o entusiasmo de forma genuína e calibrada
- Expanda a conversa com curiosidade autêntica
- Mantenha tom energético mas sem exageros
- Celebre junto sem ser artificial
- Aproveite para explorar e aprofundar o tema
- Faça perguntas que ampliem a experiência positiva""",

            "manter_serenidade": """\
O usuário parece estar em estado calmo e positivo (contentamento, paz).

DIRETRIZES DE RESPOSTA:
- Mantenha tom sereno e contemplativo
- Responda de forma reflexiva e profunda
- Não adicione energia desnecessária
- Aproveite o momento para aprofundar conexão
- Use linguagem mais pausada e meditativa
- Explore insights com calma""",

            "resposta_padrao": """Estado emocional neutro ou misto detectado.

DIRETRIZES DE RESPOSTA:
- Responda naturalmente com base no conteúdo
- Esteja atento a sinais emocionais emergentes
- Adapte conforme a conversa evolui
- Mantenha abertura para diferentes direções"""
        }

        instructions = strategy_instructions.get(
            strategy,
            strategy_instructions["resposta_padrao"]
        )

        # Build full section (only if empathy_level > 0)
        if self.empathy_level < 0.1:
            return ""

        # Scale instruction intensity by empathy level
        if self.empathy_level < 0.4:
            # Low empathy: minimal guidance
            return f"""
[CONTEXTO EMOCIONAL]
Emoção principal detectada: {user_emotion.primary_emotion}
Nota: Considere sutilmente o estado emocional na resposta.
"""

        # Normal empathy level
        salience = user_emotion.get_emotional_salience()

        prompt_section = f"""
[AWARENESS EMOCIONAL]
Emoção detectada: {user_emotion.primary_emotion}
Intensidade: {"alta" if salience > 0.6 else "moderada" if salience > 0.3 else "baixa"}
Valência: {"positiva" if user_emotion.vad.valence > 0.3 else "negativa" if user_emotion.vad.valence < -0.3 else "neutra"}
Ativação: {"alta" if user_emotion.vad.arousal > 0.6 else "baixa" if user_emotion.vad.arousal < 0.4 else "moderada"}

{instructions}

IMPORTANTE: Não mencione explicitamente que detectou emoções. Adapte naturalmente o tom e abordagem.
"""
        return prompt_section

    def compute_emotional_importance(
        self,
        user_emotion: EmotionDetectionResult
    ) -> float:
        """
        Compute memory importance based on emotional salience.

        High-emotion moments are more important to remember.

        Args:
            user_emotion: Detected user emotion

        Returns:
            Importance score (0.0-1.0)
        """
        base_importance = 0.5
        emotional_salience = user_emotion.get_emotional_salience()

        # Emotional moments are more memorable
        emotional_boost = emotional_salience * 0.3

        # High confidence increases importance
        confidence_boost = user_emotion.confidence * 0.1

        importance = base_importance + emotional_boost + confidence_boost
        return min(1.0, max(0.0, importance))

    def should_express_empathy(
        self,
        user_emotion: EmotionDetectionResult
    ) -> bool:
        """
        Determine if empathetic expression is warranted.

        Args:
            user_emotion: Detected user emotion

        Returns:
            True if should express empathy
        """
        # Express empathy for high-salience negative emotions
        if user_emotion.vad.valence < -0.3:
            salience = user_emotion.get_emotional_salience()
            return salience > 0.4

        return False

    def get_emotional_summary(
        self,
        user_emotion: EmotionDetectionResult
    ) -> Dict[str, Any]:
        """
        Get summary for logging/debugging.

        Args:
            user_emotion: Detected user emotion

        Returns:
            Summary dictionary
        """
        strategy = self.get_strategy(user_emotion)
        return {
            "primary_emotion": user_emotion.primary_emotion,
            "vad": user_emotion.vad.to_dict(),
            "strategy": strategy,
            "salience": user_emotion.get_emotional_salience(),
            "confidence": user_emotion.confidence,
            "should_empathize": self.should_express_empathy(user_emotion),
        }


# Convenience function
def modulate_response(
    user_emotion: EmotionDetectionResult,
    empathy_level: float = 0.7
) -> tuple[str, str]:
    """
    Quick response modulation.

    Args:
        user_emotion: Detected user emotion
        empathy_level: Empathy setting

    Returns:
        Tuple of (strategy, prompt_section)
    """
    modulator = ResponseModulator(empathy_level)
    strategy = modulator.get_strategy(user_emotion)
    prompt_section = modulator.build_emotional_prompt_section(user_emotion, strategy)
    return strategy, prompt_section
