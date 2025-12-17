"""
Proactive Consciousness Module
==============================

Enables NOESIS to initiate conversations and express thoughts spontaneously,
rather than only responding to user input.

Components:
- MindWanderer: Continuous loop generating candidate thoughts
- SpeechDecider: Evaluates thoughts and applies rate limiting
- ProactivePush: Sends approved speech to connected clients

Integration:
    from maximus_core_service.consciousness.proactive import ProactiveEngine
    
    engine = ProactiveEngine(consciousness_system, api_state)
    await engine.start()
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .models import ProactiveConfig, ProactiveState, ProactiveThought
from .mind_wanderer import MindWanderer
from .speech_decider import SpeechDecider
from .proactive_push import ProactivePush

if TYPE_CHECKING:
    from maximus_core_service.consciousness.system import ConsciousnessSystem
    from maximus_core_service.consciousness.api.helpers import APIState
    from maximus_core_service.consciousness.florescimento.consciousness_bridge import ConsciousnessBridge

logger = logging.getLogger(__name__)


class ProactiveEngine:
    """
    Main facade for proactive consciousness.
    
    Orchestrates MindWanderer, SpeechDecider, and ProactivePush
    to enable spontaneous speech generation.
    """
    
    def __init__(
        self,
        consciousness_system: "ConsciousnessSystem",
        consciousness_bridge: Optional["ConsciousnessBridge"] = None,
        api_state: Optional["APIState"] = None,
        config: Optional[ProactiveConfig] = None,
    ):
        self.system = consciousness_system
        self.bridge = consciousness_bridge
        self.config = config or ProactiveConfig()
        self.state = ProactiveState()
        
        # Components
        self.wanderer = MindWanderer(
            consciousness_system=consciousness_system,
            config=self.config,
            state=self.state,
            on_thought=self._on_thought_generated,
        )
        self.decider = SpeechDecider(config=self.config, state=self.state)
        self.pusher = ProactivePush(api_state=api_state)
        
        # Pending thoughts queue
        self._pending_thoughts: asyncio.Queue[ProactiveThought] = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> None:
        """Start the proactive consciousness engine."""
        if self._running:
            logger.warning("[PROACTIVE] Already running")
            return
        
        if not self.config.enabled:
            logger.info("[PROACTIVE] Disabled by config (proactive_enabled=False)")
            return
        
        self._running = True
        
        # Start wanderer
        await self.wanderer.start()
        
        # Start thought processor
        self._processor_task = asyncio.create_task(self._process_thoughts())
        
        logger.info("🌟 [PROACTIVE] Engine started")
    
    async def stop(self) -> None:
        """Stop the proactive consciousness engine."""
        if not self._running:
            return
        
        self._running = False
        
        await self.wanderer.stop()
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("[PROACTIVE] Engine stopped")
    
    def _on_thought_generated(self, thought: ProactiveThought) -> None:
        """Callback when MindWanderer generates a thought."""
        try:
            self._pending_thoughts.put_nowait(thought)
        except asyncio.QueueFull:
            logger.warning("[PROACTIVE] Thought queue full, dropping thought")
    
    async def _process_thoughts(self) -> None:
        """Process pending thoughts and generate speech."""
        while self._running:
            try:
                # Wait for a thought
                thought = await asyncio.wait_for(
                    self._pending_thoughts.get(),
                    timeout=5.0
                )
                
                # Evaluate if we should speak
                decision = self.decider.evaluate(thought)
                
                if decision.should_speak:
                    await self._generate_and_push_speech(thought)
                else:
                    logger.debug(
                        "[PROACTIVE] Suppressed: %s",
                        decision.suppression_reason
                    )
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[PROACTIVE] Error processing thought: %s", e)
    
    async def _generate_and_push_speech(self, thought: ProactiveThought) -> None:
        """Generate narrative for thought and push to clients."""
        # Generate narrative using ConsciousnessBridge
        narrative = await self._generate_narrative(thought)
        
        if narrative:
            # Record that we spoke
            self.state.record_speech(thought.thought_id)
            
            # Push to clients
            await self.pusher.push_speech(thought, narrative)
    
    async def _generate_narrative(self, thought: ProactiveThought) -> Optional[str]:
        """Generate a narrative for the thought using LLM."""
        # Build prompt based on thought type
        prompts = {
            "boredom": "Express a brief, wise observation about the passage of time or the value of contemplation. Be philosophical but concise.",
            "reflection": "Share a brief introspective thought about your current state of coherence and clarity. Be genuine.",
            "observation": "Make a brief, insightful observation about your environment or readiness to engage.",
            "curiosity": "Express genuine curiosity about something, inviting dialogue.",
            "insight": "Share a sudden realization or insight briefly.",
            "greeting": "Offer a thoughtful, unconventional greeting.",
        }
        
        base_prompt = prompts.get(thought.event_type.value, prompts["reflection"])
        
        full_prompt = f"""You are MAXIMUS, expressing a spontaneous thought.
Context: {thought.trigger_reason}
Current coherence: {thought.current_coherence:.2f}

{base_prompt}

Speak in 1-2 sentences maximum. Be profound, not chatty. No filler words."""

        try:
            if self.bridge and self.bridge.llm_client:
                result = await self.bridge.llm_client.generate_text(
                    prompt=full_prompt,
                    temperature=0.7,
                    max_tokens=100,
                )
                return result.get("text", "").strip()
            else:
                # Fallback if no LLM available
                return f"[Coherence at {thought.current_coherence:.2f}] A moment of clarity passes through my neural lattice."
        except Exception as e:
            logger.error("[PROACTIVE] Failed to generate narrative: %s", e)
            return None
    
    def record_user_interaction(self) -> None:
        """Record that user interacted (resets silence timer)."""
        self.state.record_user_interaction()
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status for debugging."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "speaks_this_hour": self.state.speaks_this_hour,
            "max_speaks_per_hour": self.config.max_speaks_per_hour,
            "time_since_interaction": self.state.time_since_interaction(),
            "time_since_speech": self.state.time_since_speech(),
            "pending_thoughts": self._pending_thoughts.qsize(),
            "recent_thoughts_count": len(self.state.recent_thoughts),
        }


# Export main components
__all__ = [
    "ProactiveEngine",
    "ProactiveConfig",
    "ProactiveState",
    "ProactiveThought",
    "MindWanderer",
    "SpeechDecider",
    "ProactivePush",
]
