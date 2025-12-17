"""
Proactive Push - Sends spontaneous speech to clients
=====================================================

Uses the existing WebSocket/SSE infrastructure to push
proactive speech events to connected clients.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .models import (
    ProactiveSpeechEvent,
    ProactiveThought,
)

if TYPE_CHECKING:
    from maximus_core_service.consciousness.api.helpers import APIState

logger = logging.getLogger(__name__)


class ProactivePush:
    """Pushes proactive speech to connected clients."""
    
    def __init__(self, api_state: Optional["APIState"] = None):
        self.api_state = api_state
        self._speech_history: List[ProactiveSpeechEvent] = []
        
    async def push_speech(
        self,
        thought: ProactiveThought,
        narrative: str,
    ) -> bool:
        """
        Push spontaneous speech to all connected clients.
        
        Uses the existing api_state.broadcast_to_consumers() mechanism.
        """
        event = ProactiveSpeechEvent(
            type="spontaneous_speech",
            thought_id=thought.thought_id,
            event_type=thought.event_type,
            narrative=narrative,
            coherence_at_generation=thought.current_coherence,
            timestamp=time.time(),
            trigger_reason=thought.trigger_reason,
            time_since_last_interaction=thought.time_since_last_interaction,
        )
        
        # Store in history
        self._speech_history.append(event)
        if len(self._speech_history) > 50:
            self._speech_history = self._speech_history[-50:]
        
        # Push to clients
        if self.api_state:
            try:
                message = event.model_dump()
                await self.api_state.broadcast_to_consumers(message)
                logger.info(
                    "📢 [PUSH] Spontaneous speech sent: type=%s, len=%d",
                    thought.event_type.value,
                    len(narrative)
                )
                return True
            except Exception as e:
                logger.error("[PUSH] Failed to broadcast: %s", e)
                return False
        else:
            # No api_state - just log
            logger.warning("[PUSH] No api_state available, logging only")
            logger.info("🗣️ SPONTANEOUS: %s", narrative[:100])
            return False
    
    def get_recent_speeches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent proactive speeches for debugging."""
        return [s.model_dump() for s in self._speech_history[-limit:]]
