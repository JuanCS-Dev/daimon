"""
DAIMON Routes - FastAPI Router for NOESIS Integration
======================================================

API routes for DAIMON exocortex functionality.
To be integrated into maximus_core_service.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Files <300 lines.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from .quick_check import QuickCheckRequest, QuickCheckResponse, analyze_prompt
from .daimon_models import (
    ShellHeartbeat,
    ShellBatchRequest,
    ShellBatchResponse,
    ClaudeEvent,
    ClaudeEventResponse,
    SessionEndRequest,
    SessionEndResponse,
    PreferencesResponse,
    ReflectResponse,
    MemoryItem,
    RecentMemoriesResponse,
)

logger = logging.getLogger(__name__)

# Router with DAIMON prefix
router = APIRouter(prefix="/api/daimon", tags=["DAIMON"])


@router.post("/quick-check", response_model=QuickCheckResponse)
async def quick_check(request: QuickCheckRequest) -> QuickCheckResponse:
    """
    Fast heuristic check for NOESIS emergence.

    Analyzes prompt for risk keywords and returns salience score.
    Target latency: <100ms (no LLM calls).
    """
    return analyze_prompt(request.prompt)


@router.post("/shell/batch", response_model=ShellBatchResponse)
async def receive_shell_batch(batch: ShellBatchRequest) -> ShellBatchResponse:
    """
    Receive batch of shell command heartbeats.

    Stores commands in episodic memory and generates insights from patterns.

    Args:
        batch: ShellBatchRequest with heartbeats and patterns.

    Returns:
        ShellBatchResponse with processing status.
    """
    insights: List[str] = []
    stored_count = 0

    # Actually store in ActivityStore
    try:
        from datetime import datetime
        from memory.activity_store import get_activity_store
        store = get_activity_store()

        for hb in batch.heartbeats:
            try:
                ts = datetime.fromisoformat(hb.timestamp)
            except ValueError:
                ts = datetime.now()

            store.add(
                watcher_type="shell",
                timestamp=ts,
                data={
                    "command": hb.command,
                    "pwd": hb.pwd,
                    "exit_code": hb.exit_code,
                    "duration": hb.duration,
                    "git_branch": hb.git_branch,
                },
            )
            stored_count += 1
    except ImportError:
        logger.debug("ActivityStore not available")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to store heartbeats: %s", e)

    # Detect frustration pattern
    if batch.patterns.get("possible_frustration"):
        error_streak = batch.patterns.get("error_streak", 0)
        insights.append(f"Frustration detected: {error_streak} consecutive errors")
        logger.info("DAIMON: Frustration pattern detected (%d errors)", error_streak)

    # Log significant commands
    for hb in batch.heartbeats:
        if hb.exit_code != 0:
            logger.debug("DAIMON: Command failed: %s (exit %d)", hb.command[:50], hb.exit_code)

    return ShellBatchResponse(
        status="ok",
        stored=stored_count,  # Now honest!
        insights=insights,
    )


@router.post("/claude/event", response_model=ClaudeEventResponse)
async def receive_claude_event(event: ClaudeEvent) -> ClaudeEventResponse:
    """
    Receive event from Claude Code session watcher.

    Stores session metadata (not content) for pattern analysis.

    Args:
        event: ClaudeEvent with session metadata.

    Returns:
        ClaudeEventResponse with processing status.
    """
    logger.debug("DAIMON: Claude event received: %s", event.event_type)
    stored = False

    # Actually store in ActivityStore
    try:
        from datetime import datetime
        from memory.activity_store import get_activity_store
        store = get_activity_store()

        try:
            ts = datetime.fromisoformat(event.timestamp)
        except ValueError:
            ts = datetime.now()

        store.add(
            watcher_type="claude",
            timestamp=ts,
            data={
                "intention": event.intention,
                "event_type": event.event_type,
                "project": event.project,
                "files_touched": event.files_touched,
            },
        )
        stored = True
    except ImportError:
        logger.debug("ActivityStore not available")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to store claude event: %s", e)

    return ClaudeEventResponse(
        status="ok",
        stored=stored,  # Now honest!
    )


@router.post("/session/end", response_model=SessionEndResponse)
async def record_session_end(request: SessionEndRequest) -> SessionEndResponse:
    """
    Record end of Claude Code session.

    Significant sessions are stored as precedents via NOESIS Tribunal.

    Args:
        request: SessionEndRequest with session summary.

    Returns:
        SessionEndResponse with precedent ID if created.
    """
    logger.info(
        "DAIMON: Session ended - %s (%s, %d files)",
        request.session_id,
        request.outcome,
        request.files_changed,
    )

    # Store session in ActivityStore
    try:
        from datetime import datetime
        from memory.activity_store import get_activity_store
        store = get_activity_store()
        store.add(
            watcher_type="session",
            timestamp=datetime.now(),
            data={
                "session_id": request.session_id,
                "summary": request.summary,
                "outcome": request.outcome,
                "duration_minutes": request.duration_minutes,
                "files_changed": request.files_changed,
            },
        )
    except ImportError:
        pass
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to store session: %s", e)

    # Only create precedent for significant sessions via real NOESIS integration
    precedent_id: Optional[str] = None
    if request.files_changed >= 5 or request.duration_minutes >= 30:
        precedent_id = await _create_real_precedent(request)

    return SessionEndResponse(
        status="ok",
        precedent_id=precedent_id,
    )


def _feed_verdict_to_learner(verdict: Dict[str, Any], request: SessionEndRequest) -> None:
    """
    AIR GAP #3 FIX: Feed Tribunal verdict back to PreferenceLearner.

    Creates a preference signal based on the verdict to close the feedback loop.

    Args:
        verdict: Verdict dict from NOESIS Tribunal
        request: Original session end request
    """
    try:
        from learners.preference_learner import PreferenceSignal
        from learners import get_engine
        from datetime import datetime

        engine = get_engine()
        learner = engine.learner

        # Extract verdict details
        verdict_result = verdict.get("verdict", "unknown")
        confidence = verdict.get("confidence", 0.5)
        reasoning = verdict.get("reasoning", "")

        # Map verdict to signal type
        if verdict_result in ("approved", "accept", "success"):
            signal_type = "approval"
            category = "session_quality"
        elif verdict_result in ("rejected", "deny", "failure"):
            signal_type = "rejection"
            category = "session_quality"
        else:
            # Skip neutral verdicts
            return

        # Create signal with correct PreferenceSignal fields
        signal = PreferenceSignal(
            timestamp=datetime.now().isoformat(),
            signal_type=signal_type,
            context=f"Tribunal verdict: {reasoning[:200]}",
            category=category,
            strength=min(confidence, 1.0),
            session_id=request.session_id,
            tool_involved=None,
        )

        # Add signal to learner and update counts
        learner.signals.append(signal)
        learner._update_counts(signal)  # pylint: disable=protected-access

        logger.debug("Fed Tribunal verdict to PreferenceLearner: %s", signal_type)

    except ImportError:
        logger.debug("PreferenceLearner not available for verdict feedback")
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Failed to feed verdict to learner: %s", e)


async def _create_real_precedent(request: SessionEndRequest) -> Optional[str]:
    """
    Create real precedent via NOESIS Tribunal integration.

    Falls back to local PrecedentSystem if NOESIS unavailable.

    Args:
        request: Session end request data.

    Returns:
        Precedent ID if created successfully.
    """
    import os
    import uuid

    noesis_url = os.getenv("NOESIS_REFLECTOR_URL", "http://localhost:8002")

    # Try NOESIS first
    try:
        import httpx  # pylint: disable=import-outside-toplevel

        payload = {
            "trace_id": str(uuid.uuid4()),
            "agent_id": "daimon-session",
            "task": f"Session: {request.summary[:100]}",
            "action": request.summary,
            "outcome": request.outcome,
            "reasoning_trace": f"Files: {request.files_changed}, Duration: {request.duration_minutes}min",
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{noesis_url}/reflect/verdict",
                json=payload,
            )
            if response.status_code == 200:
                result = response.json()
                precedent_id = result.get("precedent_id") or f"sess_{request.session_id[:8]}"
                logger.info("DAIMON: Created real precedent %s via NOESIS", precedent_id)

                # AIR GAP #3 FIX: Feed Tribunal verdict back to PreferenceLearner
                _feed_verdict_to_learner(result, request)

                return precedent_id
    except ImportError:
        logger.debug("httpx not available for NOESIS integration")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("NOESIS precedent creation failed: %s", e)

    # AIR GAP #2 FIX: Store locally in PrecedentSystem when NOESIS unavailable
    try:
        from memory.precedent_system import PrecedentSystem
        from memory.precedent_models import PrecedentMeta

        system = PrecedentSystem()
        outcome_map = {"success": "success", "failure": "failure", "partial": "partial"}
        outcome = outcome_map.get(request.outcome, "unknown")

        meta = PrecedentMeta(
            tags=["session", "daimon"],
            relevance=0.6 if request.files_changed >= 10 else 0.5,
        )

        precedent_id = system.record(
            context=f"Session {request.session_id[:8]}: {request.summary[:200]}",
            decision=f"Changed {request.files_changed} files in {request.duration_minutes}min",
            outcome=outcome,
            meta=meta,
        )
        logger.info("DAIMON: Created local precedent %s", precedent_id)
        return precedent_id
    except ImportError:
        logger.debug("PrecedentSystem not available")
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Local precedent creation failed: %s", e)

    return None


@router.get("/preferences/learned", response_model=PreferencesResponse)
async def get_learned_preferences() -> PreferencesResponse:
    """
    Get learned preferences from DAIMON reflection engine.

    Returns preferences detected from Claude Code session patterns.

    Returns:
        PreferencesResponse with categorized preferences.
    """
    try:
        from learners import get_engine
        engine = get_engine()
        status = engine.get_status()

        return PreferencesResponse(
            preferences=status.get("current_preferences", {}),
            total_signals=status.get("signals_in_memory", 0),
            last_reflection=status.get("last_reflection"),
            approval_rate=_calculate_approval_rate(status.get("current_preferences", {})),
        )
    except ImportError:
        logger.warning("DAIMON learners not available")
        return PreferencesResponse()


def _calculate_approval_rate(preferences: Dict[str, Any]) -> float:
    """Calculate overall approval rate from preferences."""
    if not preferences:
        return 0.0
    rates = [p.get("approval_rate", 0.0) for p in preferences.values()]
    return sum(rates) / len(rates) if rates else 0.0


@router.post("/reflect", response_model=ReflectResponse)
async def trigger_reflection() -> ReflectResponse:
    """
    Trigger manual reflection.

    Scans recent Claude Code sessions and updates preferences.

    Returns:
        ReflectResponse with reflection results.
    """
    try:
        from learners import get_engine
        engine = get_engine()
        result = await engine.reflect()

        return ReflectResponse(
            status="completed",
            signals_count=result.get("signals_count", 0),
            insights_count=result.get("insights_count", 0),
            updated=result.get("updated", False),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
        )
    except ImportError:
        logger.warning("DAIMON learners not available")
        return ReflectResponse(status="unavailable")
    except Exception as e:
        logger.error("Reflection failed: %s", e)
        return ReflectResponse(status=f"error: {e}")


@router.get("/memories/recent", response_model=RecentMemoriesResponse)
async def get_recent_memories(limit: int = 20) -> RecentMemoriesResponse:
    """
    Get recent memories for dashboard display.

    Returns most recent shell commands and Claude events.

    Args:
        limit: Maximum number of memories to return.

    Returns:
        RecentMemoriesResponse with memory items.
    """
    memories: List[MemoryItem] = []

    # Try to get signals from preference learner
    try:
        from learners import get_engine
        engine = get_engine()

        for i, signal in enumerate(engine.learner.signals[-limit:]):
            memories.append(
                MemoryItem(
                    id=f"sig_{i}",
                    timestamp=signal.timestamp.isoformat(),
                    type="preference",
                    content=f"[{signal.signal_type}] {signal.category}: {signal.content[:100]}",
                    importance=0.6 if signal.signal_type == "rejection" else 0.4,
                )
            )
    except ImportError:
        pass

    return RecentMemoriesResponse(
        memories=sorted(memories, key=lambda m: m.timestamp, reverse=True)[:limit],
        total_count=len(memories),
    )


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def daimon_health() -> Dict[str, Any]:
    """
    Check DAIMON endpoints health.

    Returns:
        Health status dict.
    """
    return {
        "status": "healthy",
        "service": "daimon",
        "endpoints": [
            "/api/daimon/quick-check",
            "/api/daimon/shell/batch",
            "/api/daimon/claude/event",
            "/api/daimon/session/end",
            "/api/daimon/preferences/learned",
            "/api/daimon/reflect",
            "/api/daimon/memories/recent",
        ],
    }
