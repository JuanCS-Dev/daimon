"""
Speech Decider - Decides IF and WHEN to speak proactively
==========================================================

Evaluates candidate thoughts and applies rate limiting, context checks,
and coherence requirements before allowing proactive speech.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, List

from .models import (
    ProactiveConfig,
    ProactiveState,
    ProactiveThought,
    SpeechDecision,
    ProactiveUrgency,
)

if TYPE_CHECKING:
    from maximus_core_service.consciousness.florescimento.consciousness_bridge import ConsciousnessBridge

logger = logging.getLogger(__name__)


class SpeechDecider:
    """Decides whether to convert a thought into speech."""
    
    def __init__(
        self,
        config: ProactiveConfig,
        state: ProactiveState,
    ):
        self.config = config
        self.state = state
        
    def evaluate(self, thought: ProactiveThought) -> SpeechDecision:
        """
        Evaluate whether this thought should become speech.
        
        Returns SpeechDecision with should_speak=True/False and reason.
        """
        # Check if proactive is enabled
        if not self.config.enabled:
            return SpeechDecision(
                should_speak=False,
                thought=thought,
                reason="Proactive disabled",
                suppression_reason="proactive_enabled=False",
                speaks_this_hour=self.state.speaks_this_hour,
                max_speaks_per_hour=self.config.max_speaks_per_hour,
            )
        
        # Check minimum silence
        if thought.time_since_last_interaction < self.config.min_silence_before_speech:
            return SpeechDecision(
                should_speak=False,
                thought=thought,
                reason="Too soon after last interaction",
                suppression_reason=f"silence={thought.time_since_last_interaction:.0f}s < min={self.config.min_silence_before_speech}s",
                speaks_this_hour=self.state.speaks_this_hour,
                max_speaks_per_hour=self.config.max_speaks_per_hour,
            )
        
        # Check coherence threshold
        if thought.current_coherence < self.config.min_coherence_to_speak:
            return SpeechDecision(
                should_speak=False,
                thought=thought,
                reason="Coherence too low",
                suppression_reason=f"coherence={thought.current_coherence:.2f} < min={self.config.min_coherence_to_speak}",
                speaks_this_hour=self.state.speaks_this_hour,
                max_speaks_per_hour=self.config.max_speaks_per_hour,
            )
        
        # Check rate limiting
        if not self.state.can_speak(self.config):
            return SpeechDecision(
                should_speak=False,
                thought=thought,
                reason="Rate limited",
                suppression_reason=f"speaks_this_hour={self.state.speaks_this_hour}/{self.config.max_speaks_per_hour}",
                speaks_this_hour=self.state.speaks_this_hour,
                max_speaks_per_hour=self.config.max_speaks_per_hour,
            )
        
        # All checks passed - allow speech!
        logger.info(
            "🗣️ [DECIDER] Approved speech: type=%s, urgency=%s, reason=%s",
            thought.event_type.value,
            thought.urgency.value,
            thought.trigger_reason
        )
        
        return SpeechDecision(
            should_speak=True,
            thought=thought,
            reason=f"Thought approved: {thought.trigger_reason}",
            speaks_this_hour=self.state.speaks_this_hour,
            max_speaks_per_hour=self.config.max_speaks_per_hour,
        )
    
    def select_best_thought(self, thoughts: List[ProactiveThought]) -> Optional[ProactiveThought]:
        """Select the best thought from candidates based on urgency."""
        if not thoughts:
            return None
        
        # Sort by urgency (CRITICAL > HIGH > MEDIUM > LOW)
        urgency_order = {
            ProactiveUrgency.CRITICAL: 4,
            ProactiveUrgency.HIGH: 3,
            ProactiveUrgency.MEDIUM: 2,
            ProactiveUrgency.LOW: 1,
        }
        
        sorted_thoughts = sorted(
            thoughts,
            key=lambda t: urgency_order.get(t.urgency, 0),
            reverse=True
        )
        
        return sorted_thoughts[0]
