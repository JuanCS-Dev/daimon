"""
NOESIS Human-In-The-Loop (HITL) Module
=======================================

Provides continuous human oversight and intervention capabilities
for the consciousness system.

G5 Integration Spec:
- Continuous overlay (not checkpoint-based)
- Multiple priority levels (OBSERVE, SUGGEST, OVERRIDE, EMERGENCY)
- Component-targeted interventions
- Time-limited overlays with automatic expiration
"""

from .human_overlay import (
    HumanCortexBridge,
    HumanOverlay,
    OverlayPriority,
    OverlayTarget,
    create_human_cortex_bridge,
)

__all__ = [
    "HumanCortexBridge",
    "HumanOverlay",
    "OverlayPriority",
    "OverlayTarget",
    "create_human_cortex_bridge",
]
