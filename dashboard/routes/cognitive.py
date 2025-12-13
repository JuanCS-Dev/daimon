"""
DAIMON Dashboard - Cognitive and metacognitive endpoints.

Includes: style learner, cognitive state (keystroke), metacognitive analysis.
"""

import time

from fastapi import APIRouter, Request


router = APIRouter(tags=["cognitive"])


# === Style Learner Endpoints ===

@router.get("/api/style")
async def get_communication_style():
    """Estilo de comunicacao inferido."""
    try:
        from learners import get_style_learner
        learner = get_style_learner()
        style = learner.compute_style()
        return {
            "style": style.to_dict(),
            "suggestions": style.to_claude_suggestions(),
            "claude_md_section": learner.get_claude_md_section(),
        }
    except ImportError:
        return {"error": "Style learner not available"}


# === Cognitive State Endpoints ===

@router.get("/api/cognitive")
async def get_cognitive_state():
    """Estado cognitivo inferido via keystroke dynamics."""
    try:
        from learners import get_keystroke_analyzer
        analyzer = get_keystroke_analyzer()
        state = analyzer.detect_cognitive_state()
        return {
            "state": state.state,
            "confidence": state.confidence,
            "biometrics": {
                "avg_hold_time": state.biometrics.avg_hold_time,
                "avg_seek_time": state.biometrics.avg_seek_time,
                "rhythm_consistency": state.biometrics.rhythm_consistency,
                "fatigue_index": state.biometrics.fatigue_index,
                "focus_score": state.biometrics.focus_score,
                "cognitive_load": state.biometrics.cognitive_load,
                "typing_speed": state.biometrics.typing_speed,
                "error_rate": state.biometrics.error_rate,
            },
            "detected_at": state.detected_at.isoformat(),
        }
    except ImportError:
        return {"error": "Keystroke analyzer not available"}
    except (AttributeError, ValueError) as e:
        return {"error": str(e), "state": "unknown"}


@router.post("/api/cognitive/event")
async def log_keystroke_event(request: Request):
    """Log keystroke event for cognitive state analysis."""
    try:
        from learners import get_keystroke_analyzer
        from datetime import datetime

        data = await request.json()
        analyzer = get_keystroke_analyzer()

        # Use time.time() for timestamp since that's what the analyzer expects
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp).timestamp()
        elif timestamp is None:
            timestamp = time.time()

        analyzer.add_event(
            key=data.get("key", ""),
            event_type=data.get("type", "press"),
            timestamp=timestamp,
            modifiers=data.get("modifiers", []),
        )

        return {"status": "logged", "event_count": len(analyzer._events)}
    except ImportError:
        return {"error": "Keystroke analyzer not available"}
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


# === Metacognitive Endpoints ===

@router.get("/api/metacognitive")
async def get_metacognitive_analysis():
    """Analise metacognitiva da efetividade das reflexoes."""
    try:
        from learners import get_metacognitive_engine
        engine = get_metacognitive_engine()
        # reflect_on_reflection() already returns a dict
        return engine.reflect_on_reflection()
    except ImportError:
        return {"error": "Metacognitive engine not available"}
    except (AttributeError, ValueError) as e:
        return {"error": str(e)}


@router.get("/api/metacognitive/insights")
async def get_insight_history():
    """Historico de insights gerados."""
    try:
        from learners import get_metacognitive_engine
        engine = get_metacognitive_engine()
        # _insights is the internal list
        return {
            "insights": [i.to_dict() for i in engine._insights[-50:]],
            "total": len(engine._insights),
        }
    except ImportError:
        return {"error": "Metacognitive engine not available"}
