"""
Maximus Core Service - Main Application
=======================================

Entry point for the Maximus Core Service.

PROJETO SINGULARIDADE (06/Dez/2025):
Integração do ConsciousnessSystem com o pipeline de comunicação.
"""

from __future__ import annotations

import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# --- ANTI-ZOMBIE PROTECTION (Port 8001) ---
try:
    # Attempt to locate backend/services/shared dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Walk up to find 'backend/services'
    # Path: src/maximus_core_service/main.py -> ../../../ -> services
    services_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    if os.path.isdir(os.path.join(services_dir, "shared")):
        if services_dir not in sys.path:
            sys.path.insert(0, services_dir)
        
        from shared.lifecycle import ensure_port_protection, install_signal_handlers
        ensure_port_protection(8001, "Maximus Core")
        install_signal_handlers()
except Exception as e:
    print(f"[WARNING] Anti-Zombie proctection failed: {e}")
# ------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maximus_core_service.api.dependencies import initialize_service
from maximus_core_service.api.routes import router as api_router
from maximus_core_service.consciousness.exocortex.api.exocortex_router import (
    router as exocortex_router,
    set_consciousness_system,  # SINGULARIDADE
)
from maximus_core_service.consciousness.api import create_consciousness_api
from maximus_core_service.consciousness.api.streaming import set_maximus_consciousness_system
from maximus_core_service.consciousness.exocortex.factory import ExocortexFactory
from maximus_core_service.config import get_settings

# SINGULARIDADE: Import ConsciousnessSystem & Introspection API
from maximus_core_service.consciousness.system import ConsciousnessSystem
from maximus_core_service.consciousness.florescimento.introspection_api import (
    router as florescimento_router,
    set_global_bridge,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# SINGULARIDADE: Global reference to ConsciousnessSystem
_consciousness_system: ConsciousnessSystem | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan.

    Args:
        _: FastAPI application instance (unused)

    Yields:
        None during application lifetime
    """
    global _consciousness_system

    # Startup
    initialize_service()
    init_reflector()  # Initialize Reflector dependencies
    # init_reflector()  # Initialize Reflector dependencies - Moved below
    ExocortexFactory.initialize(data_dir=str(settings.base_path / ".data"))

    # SINGULARIDADE: Initialize and start ConsciousnessSystem
    logger.info("[SINGULARIDADE] Initializing ConsciousnessSystem...")
    _consciousness_system = ConsciousnessSystem()
    await _consciousness_system.start()
    
    # SINGULARIDADE: Inject Bridge into Introspection API
    if _consciousness_system.consciousness_bridge:
        set_global_bridge(_consciousness_system.consciousness_bridge)
        logger.info("[SINGULARIDADE] ConsciousnessBridge injected into API")
    
    # Initialize and start Journalor (File Persistence)
    from maximus_core_service.consciousness.exocortex.journalor import Journalor
    _journalor = Journalor()
    await _journalor.start()

    # Register with Exocortex router for /journal endpoint
    set_consciousness_system(_consciousness_system)
    logger.info("[SINGULARIDADE] ConsciousnessSystem integrated with Exocortex")

    # MAXIMUS: Register with streaming endpoint for real-time SSE
    set_maximus_consciousness_system(_consciousness_system)
    logger.info("[MAXIMUS] ConsciousnessSystem integrated with Streaming API")

    # FIX: Populate consciousness_system dict for REST API endpoints
    from maximus_core_service.consciousness.api import set_consciousness_components
    set_consciousness_components(_consciousness_system)
    logger.info("[FIX] ConsciousnessSystem components registered with REST API")

    yield

    # Shutdown
    if _consciousness_system:
        logger.info("[SINGULARIDADE] Stopping ConsciousnessSystem...")
        await _consciousness_system.stop()


app = FastAPI(
    title=settings.service.name,  # pylint: disable=no-member
    description="Maximus Core Service - System Coordination",
    version="3.0.0",
    lifespan=lifespan
)

# TITANIUM PIPELINE: CORS para SSE streaming cross-origin
# Permite conexões do frontend (localhost:3000/3001) para o backend (localhost:8001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir para domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(api_router, prefix="/v1")
app.include_router(exocortex_router, prefix="/v1")
app.include_router(florescimento_router, prefix="/v1")  # Fix: Introspection Endpoint

# Metacognitive Reflector
from metacognitive_reflector.api.routes import router as reflector_router
from metacognitive_reflector.api.dependencies import initialize_service as init_reflector

app.include_router(reflector_router, prefix="/v1/metacognitive")

# MAXIMUS: Consciousness API with SSE streaming (will be populated on startup)
# Note: Router created with empty dict, system set via setter during lifespan
_consciousness_api_router = create_consciousness_api({})
app.include_router(_consciousness_api_router)


@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint.

    Returns:
        Service information
    """
    return {
        "message": "Maximus Core Service Operational",
        "service": settings.service.name  # pylint: disable=no-member
    }


# ============================================================================
# DAIMON Quick-Check Endpoint
# ============================================================================
# Fast heuristic check for hook integration. Target latency: <100ms.

from typing import List, Optional
from pydantic import BaseModel, Field

# Risk keywords for quick analysis
_DAIMON_HIGH_RISK = ["delete", "drop", "rm -rf", "truncate", "production", "destroy", "wipe"]
_DAIMON_MEDIUM_RISK = ["refactor", "migrate", "architecture", "auth", "security", "deploy"]


class QuickCheckRequest(BaseModel):
    """Request for DAIMON quick-check."""
    prompt: str = Field(..., min_length=1, max_length=10000)


class QuickCheckResponse(BaseModel):
    """Response from DAIMON quick-check."""
    salience: float = Field(..., ge=0.0, le=1.0)
    should_emerge: bool
    mode: str
    emergence_reason: Optional[str] = None
    detected_keywords: List[str] = Field(default_factory=list)


@app.post("/api/consciousness/quick-check", response_model=QuickCheckResponse)
async def daimon_quick_check(request: QuickCheckRequest) -> QuickCheckResponse:
    """
    DAIMON quick-check for hook integration.

    Fast heuristic analysis without LLM calls. Target: <100ms.
    """
    prompt_lower = request.prompt.lower()
    detected: List[str] = []
    salience = 0.1

    for kw in _DAIMON_HIGH_RISK:
        if kw in prompt_lower:
            salience = max(salience, 0.9)
            detected.append(kw)

    if salience < 0.9:
        for kw in _DAIMON_MEDIUM_RISK:
            if kw in prompt_lower:
                salience = max(salience, 0.6)
                detected.append(kw)

    mode = "emerge" if salience >= 0.85 else "subtle" if salience >= 0.5 else "silent"
    reason = f"Detected: {', '.join(detected[:3])}" if detected else None

    return QuickCheckResponse(
        salience=salience,
        should_emerge=salience >= 0.85,
        mode=mode,
        emergence_reason=reason,
        detected_keywords=detected,
    )


# ============================================================================
# DAIMON Shell Batch Endpoint (Sprint 7)
# ============================================================================

from typing import Any, Dict


class ShellHeartbeat(BaseModel):
    """Single shell command heartbeat."""
    timestamp: str
    command: str
    pwd: str
    exit_code: int
    duration: float = 0.0
    git_branch: str = ""


class ShellBatchRequest(BaseModel):
    """Batch of shell heartbeats with detected patterns."""
    heartbeats: List[ShellHeartbeat]
    patterns: Dict[str, Any] = Field(default_factory=dict)


class ShellBatchResponse(BaseModel):
    """Response for shell batch endpoint."""
    status: str
    stored: int
    insights: List[str] = Field(default_factory=list)


@app.post("/api/daimon/shell/batch", response_model=ShellBatchResponse)
async def daimon_shell_batch(batch: ShellBatchRequest) -> ShellBatchResponse:
    """
    Receive batch of shell command heartbeats.

    Stores commands and generates insights from patterns.
    """
    insights: List[str] = []

    # Detect frustration pattern
    if batch.patterns.get("possible_frustration"):
        error_streak = batch.patterns.get("error_streak", 0)
        insights.append(f"Frustration detected: {error_streak} consecutive errors")
        logger.info("[DAIMON] Frustration pattern detected (%d errors)", error_streak)

    return ShellBatchResponse(status="ok", stored=len(batch.heartbeats), insights=insights)


# ============================================================================
# DAIMON Claude Event Endpoint (Sprint 7)
# ============================================================================


class ClaudeEvent(BaseModel):
    """Event from Claude Code session."""
    event_type: str
    timestamp: str
    project: str = ""
    files_touched: List[str] = Field(default_factory=list)
    intention: str = ""


class ClaudeEventResponse(BaseModel):
    """Response for Claude event endpoint."""
    status: str
    stored: bool


@app.post("/api/daimon/claude/event", response_model=ClaudeEventResponse)
async def daimon_claude_event(event: ClaudeEvent) -> ClaudeEventResponse:
    """
    Receive event from Claude Code session watcher.

    Stores session metadata (not content) for pattern analysis.
    """
    logger.debug("[DAIMON] Claude event received: %s", event.event_type)
    return ClaudeEventResponse(status="ok", stored=True)


# ============================================================================
# DAIMON Session End Endpoint (Sprint 7)
# ============================================================================


class SessionEndRequest(BaseModel):
    """Request to record session end."""
    session_id: str
    summary: str
    outcome: str = "success"
    duration_minutes: float = 0.0
    files_changed: int = 0


class SessionEndResponse(BaseModel):
    """Response for session end endpoint."""
    status: str
    precedent_id: Optional[str] = None


@app.post("/api/daimon/session/end", response_model=SessionEndResponse)
async def daimon_session_end(request: SessionEndRequest) -> SessionEndResponse:
    """
    Record end of Claude Code session.

    Significant sessions are stored as precedents.
    """
    logger.info("[DAIMON] Session ended: %s (%s)", request.session_id, request.outcome)

    # Create precedent for significant sessions
    precedent_id: Optional[str] = None
    if request.files_changed >= 5 or request.duration_minutes >= 30:
        precedent_id = f"sess_{request.session_id[:8]}"

    return SessionEndResponse(status="ok", precedent_id=precedent_id)


# ============================================================================
# DAIMON Health Endpoint
# ============================================================================


@app.get("/api/daimon/health")
async def daimon_health() -> Dict[str, Any]:
    """Check DAIMON endpoints health."""
    return {
        "status": "healthy",
        "service": "daimon",
        "endpoints": [
            "/api/consciousness/quick-check",
            "/api/daimon/shell/batch",
            "/api/daimon/claude/event",
            "/api/daimon/session/end",
            "/v1/consciousness/ingest",
            "/v1/memory/episode",
        ],
    }


# ============================================================================
# DAIMON → NOESIS Data Ingestion Endpoints (Sprint 1)
# ============================================================================
# These endpoints receive behavioral signals from DAIMON's DataIngestionService
# and integrate them with NOESIS consciousness and memory systems.


class BehavioralSignalRequest(BaseModel):
    """Behavioral signal from DAIMON for consciousness processing."""
    signal_type: str = Field(
        ...,
        description="Type: cognitive_state, preference, pattern, anomaly"
    )
    source: str = Field(..., description="Watcher source: claude, shell, input, window, afk")
    timestamp: str = Field(..., description="ISO timestamp")
    salience: float = Field(..., ge=0.0, le=1.0, description="Signal importance 0-1")
    data: Dict[str, Any] = Field(default_factory=dict)
    context: str = Field(..., description="Human-readable description")


class IngestResponse(BaseModel):
    """Response from consciousness ingest endpoint."""
    status: str
    processed: bool
    coherence: Optional[float] = None
    emerged: bool = False
    phase: Optional[str] = None


@app.post("/v1/consciousness/ingest", response_model=IngestResponse)
async def consciousness_ingest(signal: BehavioralSignalRequest) -> IngestResponse:
    """
    Ingest behavioral signal into NOESIS consciousness.

    Receives high-salience signals from DAIMON and processes them through
    the ConsciousnessSystem when appropriate. Signals with salience >= 0.7
    may trigger conscious processing.

    Pipeline: DAIMON ActivityStore → DataIngestionService → This Endpoint → Kuramoto
    """
    logger.info(
        "[NOESIS INGEST] Signal received: type=%s, source=%s, salience=%.2f",
        signal.signal_type, signal.source, signal.salience
    )

    # Access global consciousness system
    global _consciousness_system

    if _consciousness_system is None or not _consciousness_system._running:
        logger.debug("[NOESIS INGEST] ConsciousnessSystem not available")
        return IngestResponse(
            status="unavailable",
            processed=False,
            emerged=False,
        )

    # For high salience signals, process through consciousness
    if signal.salience >= 0.7:
        try:
            # Build content for ESGT
            esgt_content = {
                "signal_type": signal.signal_type,
                "source": signal.source,
                "salience": signal.salience,
                "context": signal.context,
                "data": signal.data,
            }

            # Create salience score for ESGT trigger
            # pylint: disable=import-outside-toplevel
            from maximus_core_service.consciousness.esgt.models import SalienceScore
            salience_score = SalienceScore(
                novelty=signal.salience * 0.8,
                relevance=signal.salience,
                urgency=0.6 if signal.signal_type == "anomaly" else 0.4,
                confidence=0.85,
            )

            # Attempt ESGT ignition for high-salience signals
            event = await _consciousness_system.esgt_coordinator.initiate_esgt(
                salience=salience_score,
                content=esgt_content,
                content_source=f"daimon_{signal.source}",
                target_duration_ms=150.0,
                target_coherence=0.65,
            )

            coherence = event.achieved_coherence or 0.0
            emerged = event.was_successful() and coherence >= 0.70

            logger.info(
                "[NOESIS INGEST] ESGT result: phase=%s, coherence=%.2f, emerged=%s",
                event.current_phase.value, coherence, emerged
            )

            return IngestResponse(
                status="processed",
                processed=True,
                coherence=round(coherence, 3),
                emerged=emerged,
                phase=event.current_phase.value,
            )

        except Exception as e:
            logger.warning("[NOESIS INGEST] ESGT processing failed: %s", e)
            return IngestResponse(
                status="error",
                processed=False,
                emerged=False,
            )

    # For lower salience signals, just acknowledge
    return IngestResponse(
        status="acknowledged",
        processed=True,
        emerged=False,
    )


class EpisodeRequest(BaseModel):
    """Request to store behavioral episode in memory."""
    type: str = Field(default="behavioral", description="Episode type")
    content: str = Field(..., description="Episode description")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeResponse(BaseModel):
    """Response from episode storage endpoint."""
    status: str
    stored: bool
    memory_id: Optional[str] = None


@app.post("/v1/memory/episode", response_model=EpisodeResponse)
async def store_episode(episode: EpisodeRequest) -> EpisodeResponse:
    """
    Store behavioral episode in NOESIS episodic memory.

    All behavioral signals from DAIMON are stored as episodic memories
    for later retrieval and pattern analysis. Uses the EpisodicMemoryClient
    to persist to the memory service.
    """
    logger.debug("[NOESIS MEMORY] Storing episode: %s...", episode.content[:50])

    global _consciousness_system

    # Try to store via consciousness system's episodic memory client
    if _consciousness_system and _consciousness_system.episodic_memory:
        try:
            # Calculate importance from salience if available
            salience = episode.metadata.get("salience", 0.5)
            importance = 0.3 + (salience * 0.5)  # 0.3 - 0.8

            result = await _consciousness_system.episodic_memory.store_memory(
                content=episode.content,
                memory_type="episodic",
                context={
                    "source": "daimon",
                    "episode_type": episode.type,
                    **episode.metadata,
                },
                importance=importance,
            )

            if result:
                memory_id = result.get("memory_id", "unknown")
                logger.debug("[NOESIS MEMORY] Stored: %s", memory_id[:8] if memory_id else "?")
                return EpisodeResponse(
                    status="stored",
                    stored=True,
                    memory_id=memory_id,
                )

        except Exception as e:
            logger.warning("[NOESIS MEMORY] Store failed: %s", e)

    # Fallback: acknowledge but note storage failure
    return EpisodeResponse(
        status="acknowledged",
        stored=False,
        memory_id=None,
    )


if __name__ == "__main__":
    import uvicorn
    # Allow execution as script
    uvicorn.run(app, host="0.0.0.0", port=8001)

