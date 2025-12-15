"""
Chat Streaming Endpoint - Full Consciousness Pipeline
======================================================

Integrates with the unified ConsciousnessSystem (ESGT/AKOrN/Meta-Optimizer).
Replaces manual orchestration with the architecturally audited pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from maximus_core_service.consciousness.system import ConsciousnessSystem, ConsciousnessConfig
from maximus_core_service.consciousness.florescimento.consciousness_bridge import IntrospectiveResponse
from .daimon_endpoints import ShortTermSensoryMemory
from maximus_core_service.consciousness.persistence.chat_store import ChatStore

logger = logging.getLogger(__name__)

# Global Persistent Store
chat_store = ChatStore()

class PersistentSession:
    """Session wrapper around SQLite ChatStore."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Ensure session exists
        chat_store.create_session_with_id(session_id)

    def add_turn(self, role: str, content: str):
        chat_store.add_message(self.session_id, role, content)

    def get_context(self, last_n: int = 10) -> str:
        s_data = chat_store.get_session(self.session_id)
        if not s_data:
            return ""
        recent = s_data["messages"][-last_n:]
        context_lines = [f"{t['role'].upper()}: {t['content']}" for t in recent]
        return "\n".join(context_lines)

def get_or_create_session(session_id: str) -> PersistentSession:
    return PersistentSession(session_id)

# Singleton System Instance
_system_instance: Optional[ConsciousnessSystem] = None

async def get_system() -> ConsciousnessSystem:
    """Get or create the global ConsciousnessSystem instance."""
    global _system_instance
    if _system_instance is None:
        config = ConsciousnessConfig()
        _system_instance = ConsciousnessSystem(config)
        await _system_instance.start()
    return _system_instance


def register_chat_streaming_endpoints(router: APIRouter) -> None:
    """Register the full consciousness chat endpoint."""
    
    @router.get("/chat/stream")
    async def chat_stream(
        request: Request,
        content: str = Query(..., description="User message"),
        session_id: str = Query("default", description="Session ID for memory"),
        depth: int = Query(3, ge=1, le=5, description="Analysis depth"),
    ) -> StreamingResponse:
        """
        Full Consciousness Chat Stream via Unified Pipeline.
        """
        
        async def _generate_events() -> AsyncGenerator[bytes, None]:
            try:
                # 1. START
                yield _sse_event({
                    "type": "start",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                })
                
                # Get System & Session
                system = await get_system()
                session = get_or_create_session(session_id)
                session.add_turn("user", content)
                
                # Load Context
                session_context = session.get_context(last_n=10)
                sensory_block = ShortTermSensoryMemory.get_context_block()
                
                # 2. PROCESSING PHASE
                yield _sse_event({"type": "phase", "phase": "processing"})
                
                # Prepare input data with FULL CONTEXT
                input_data = {
                    "user_input": content,
                    "depth": depth,
                    "session_id": session_id,
                    "source": "dashboard_chat",
                    "context": session_context,
                    "sensory_data": sensory_block
                }
                
                # Execute Pipeline
                response: IntrospectiveResponse = await system.process_input(input_data)
                
                # 3. REPORT METRICS
                coherence = response.metadata.get("coherence", 0.0)
                yield _sse_event({"type": "coherence", "value": round(coherence, 3)})
                
                emotion = response.metadata.get("emotion", "neutral")
                yield _sse_event({
                     "type": "emotion",
                     "emotion": emotion,
                     "valence": response.metadata.get("valence", 0.0),
                     "arousal": response.metadata.get("arousal", 0.0)
                })

                # 4. GENERATION PHASE
                yield _sse_event({"type": "phase", "phase": "generate"})
                
                full_text = response.narrative
                
                # Save to DB
                session.add_turn("assistant", full_text)
                
                # Stream
                words = full_text.split(' ')
                for i, word in enumerate(words):
                    token = word + (' ' if i < len(words) - 1 else '')
                    yield _sse_event({"type": "token", "token": token})
                    await asyncio.sleep(0.01) 
                    
                    if await request.is_disconnected():
                        break
                        
                # 5. COMPLETE
                yield _sse_event({
                    "type": "complete",
                    "coherence": round(coherence, 3),
                    "timestamp": datetime.now().isoformat(),
                    "meta_awareness": response.meta_awareness
                })
                
            except Exception as e:
                logger.error(f"[CHAT] Stream error: {e}", exc_info=True)
                yield _sse_event({"type": "error", "message": str(e)})
        
        return StreamingResponse(
            _generate_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def _sse_event(data: Dict[str, Any]) -> bytes:
    """Format data as SSE event."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")
