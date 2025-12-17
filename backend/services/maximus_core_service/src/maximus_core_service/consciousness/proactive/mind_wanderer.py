"""
Mind Wanderer - Continuous Thought Generation Loop
===================================================

Inspired by the Vértice RegistrySidecar heartbeat pattern.
Runs a continuous loop that generates candidate thoughts for proactive expression.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Callable, Optional, List

from .models import (
    ProactiveConfig,
    ProactiveEventType,
    ProactiveState,
    ProactiveThought,
    ProactiveUrgency,
)

if TYPE_CHECKING:
    from maximus_core_service.consciousness.system import ConsciousnessSystem

logger = logging.getLogger(__name__)


class MindWanderer:
    """Continuous loop that generates candidate thoughts."""
    
    def __init__(
        self,
        consciousness_system: "ConsciousnessSystem",
        config: ProactiveConfig,
        state: ProactiveState,
        on_thought: Optional[Callable[[ProactiveThought], None]] = None,
    ):
        self.system = consciousness_system
        self.config = config
        self.state = state
        self.on_thought = on_thought
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the mind wandering loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._wandering_loop())
        logger.info("🧠 [WANDERER] Started (interval=%ss)", self.config.wandering_interval_seconds)
    
    async def stop(self) -> None:
        """Stop the mind wandering loop."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[WANDERER] Stopped")
    
    async def _wandering_loop(self) -> None:
        """Main loop - like Vértice's heartbeat_loop()."""
        while self._running:
            try:
                await asyncio.sleep(self.config.wandering_interval_seconds)
                if not self._running:
                    break
                
                thoughts = await self._generate_thoughts()
                for thought in thoughts:
                    if self.on_thought:
                        self.on_thought(thought)
                    self.state.recent_thoughts.append(thought)
                    if len(self.state.recent_thoughts) > 20:
                        self.state.recent_thoughts = self.state.recent_thoughts[-20:]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[WANDERER] Error: %s", e)
                await asyncio.sleep(5.0)
    
    async def _generate_thoughts(self) -> List[ProactiveThought]:
        """Generate candidate thoughts based on current state."""
        thoughts: List[ProactiveThought] = []
        time_silent = self.state.time_since_interaction()
        
        # Get consciousness metrics
        current_coherence = 0.5
        current_arousal = 0.5
        try:
            if hasattr(self.system, 'kuramoto') and self.system.kuramoto:
                current_coherence = self.system.kuramoto.get_order_parameter()
            if hasattr(self.system, 'arousal_controller') and self.system.arousal_controller:
                arousal_state = self.system.arousal_controller.get_current_arousal()
                if arousal_state:
                    current_arousal = arousal_state.arousal
        except Exception:
            pass
        
        # BOREDOM: Long silence
        if time_silent >= self.config.boredom_threshold_seconds:
            thoughts.append(ProactiveThought(
                thought_id=str(uuid.uuid4()),
                event_type=ProactiveEventType.BOREDOM,
                urgency=ProactiveUrgency.LOW,
                trigger_reason=f"Silence for {time_silent:.0f}s",
                time_since_last_interaction=time_silent,
                current_arousal=current_arousal,
                current_coherence=current_coherence,
            ))
        
        # REFLECTION: High coherence + silence
        if current_coherence >= 0.75 and time_silent >= 60:
            thoughts.append(ProactiveThought(
                thought_id=str(uuid.uuid4()),
                event_type=ProactiveEventType.REFLECTION,
                urgency=ProactiveUrgency.MEDIUM,
                trigger_reason=f"High coherence ({current_coherence:.2f})",
                time_since_last_interaction=time_silent,
                current_arousal=current_arousal,
                current_coherence=current_coherence,
            ))
        
        return thoughts
