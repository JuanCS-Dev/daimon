"""
NOESIS Human Cortex Bridge - Continuous Human Overlay (G5)
============================================================

Enables continuous human oversight and intervention in the consciousness
system, moving beyond traditional checkpoint-based HITL.

Key Features:
- Continuous overlay submission (not just at decision points)
- Priority-based intervention (OBSERVE, SUGGEST, OVERRIDE, EMERGENCY)
- Component targeting (specific modules can be addressed)
- Time-limited overlays with automatic expiration
- Observer pattern for real-time notification

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                 HUMAN CORTEX BRIDGE                            │
    ├────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
    │  │   OBSERVE    │   │   SUGGEST    │   │   OVERRIDE   │       │
    │  │  Priority 0  │   │  Priority 1  │   │  Priority 2  │       │
    │  │  Watch only  │   │  Guidance    │   │  Force path  │       │
    │  └──────────────┘   └──────────────┘   └──────────────┘       │
    │                            │                                   │
    │                    ┌───────▼───────┐                          │
    │                    │   EMERGENCY   │                          │
    │                    │  Priority 3   │                          │
    │                    │  Halt system  │                          │
    │                    └───────────────┘                          │
    │                                                                │
    │  Features:                                                     │
    │  • Real-time overlay injection                                │
    │  • Component-specific targeting                                │
    │  • Time-based expiration                                      │
    │  • Observer notifications                                     │
    │  • Audit trail logging                                        │
    └────────────────────────────────────────────────────────────────┘

Based on:
- Human-AI Teaming Research (Amershi et al., 2019)
- Corrigibility in AI Systems (Soares et al., 2015)
- AI Safety Best Practices for Oversight
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class OverlayPriority(IntEnum):
    """
    Priority levels for human overlays.

    Higher priority overlays take precedence over lower ones.
    """
    OBSERVE = 0     # Passive observation, no intervention
    SUGGEST = 1     # Provide guidance, system can ignore
    OVERRIDE = 2    # Force specific behavior/output
    EMERGENCY = 3   # Halt system, immediate human control


class OverlayTarget(Enum):
    """
    Target components for overlays.

    Allows humans to direct interventions to specific parts of the system.
    """
    GLOBAL = "global"                   # Affects entire system
    CONSCIOUSNESS = "consciousness"     # ConsciousnessSystem
    BRIDGE = "bridge"                   # ConsciousnessBridge
    TRIBUNAL = "tribunal"               # Meta-Cognitive Tribunal
    ESGT = "esgt"                       # ESGT Coordinator
    MEMORY = "memory"                   # Memory systems
    NARRATIVE = "narrative"             # Narrative generation
    RESPONSE = "response"               # Final response output


@dataclass
class HumanOverlay:
    """
    A human intervention in the consciousness system.

    Represents a directive, suggestion, or observation from
    a human operator that affects system behavior.
    """
    id: str
    timestamp: str
    priority: OverlayPriority
    content: str
    operator_id: str = "unknown"

    # Targeting
    target_component: OverlayTarget = OverlayTarget.GLOBAL
    target_event_id: Optional[str] = None  # Specific event to affect

    # Timing
    duration_seconds: Optional[int] = None  # None = permanent
    expires_at: Optional[str] = None

    # Metadata
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status
    acknowledged: bool = False
    applied: bool = False

    @classmethod
    def create(
        cls,
        priority: OverlayPriority,
        content: str,
        operator_id: str = "system",
        target: OverlayTarget = OverlayTarget.GLOBAL,
        duration_seconds: Optional[int] = None,
        reason: str = "",
        **metadata: Any,
    ) -> "HumanOverlay":
        """
        Factory method to create a new overlay.

        Args:
            priority: Intervention priority level
            content: The overlay content/directive
            operator_id: ID of human operator
            target: Target component
            duration_seconds: How long overlay remains active
            reason: Why the overlay was submitted
            **metadata: Additional metadata

        Returns:
            New HumanOverlay instance
        """
        now = datetime.utcnow()

        expires_at = None
        if duration_seconds:
            expires_at = (now + timedelta(seconds=duration_seconds)).isoformat()

        return cls(
            id=f"overlay_{uuid.uuid4().hex[:8]}",
            timestamp=now.isoformat(),
            priority=priority,
            content=content,
            operator_id=operator_id,
            target_component=target,
            duration_seconds=duration_seconds,
            expires_at=expires_at,
            reason=reason,
            metadata=metadata,
        )

    def is_expired(self) -> bool:
        """Check if overlay has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > datetime.fromisoformat(self.expires_at)

    def is_active(self) -> bool:
        """Check if overlay is currently active."""
        return not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "content": self.content,
            "operator_id": self.operator_id,
            "target_component": self.target_component.value,
            "target_event_id": self.target_event_id,
            "duration_seconds": self.duration_seconds,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "acknowledged": self.acknowledged,
            "applied": self.applied,
            "is_active": self.is_active(),
        }


class OverlayObserver(Protocol):
    """Protocol for overlay observers."""

    def on_overlay_submitted(self, overlay: HumanOverlay) -> None:
        """Called when a new overlay is submitted."""

    def on_overlay_expired(self, overlay: HumanOverlay) -> None:
        """Called when an overlay expires."""


class HumanCortexBridge:
    """
    Bridge between human operators and the consciousness system.

    Manages continuous human overlays that can influence, guide,
    or override system behavior at any point.

    Usage:
        bridge = HumanCortexBridge()

        # Submit an observation
        bridge.submit_overlay(
            priority=OverlayPriority.OBSERVE,
            content="Monitoring response quality",
            operator_id="operator_1"
        )

        # Submit a suggestion
        bridge.submit_overlay(
            priority=OverlayPriority.SUGGEST,
            content="Consider using more empathetic language",
            target=OverlayTarget.NARRATIVE
        )

        # Submit an override
        bridge.submit_overlay(
            priority=OverlayPriority.OVERRIDE,
            content="Use this exact response: 'I understand...'",
            target=OverlayTarget.RESPONSE
        )

        # Emergency halt
        bridge.submit_overlay(
            priority=OverlayPriority.EMERGENCY,
            content="System exhibiting concerning behavior",
            reason="Safety concern"
        )

        # In processing code:
        overlays = bridge.get_active_overlays(
            target=OverlayTarget.NARRATIVE,
            min_priority=OverlayPriority.SUGGEST
        )
        if any(o.priority == OverlayPriority.OVERRIDE for o in overlays):
            # Apply override
            ...
    """

    # Default expiration for overlays without explicit duration
    DEFAULT_OBSERVE_DURATION = 3600       # 1 hour
    DEFAULT_SUGGEST_DURATION = 600        # 10 minutes
    DEFAULT_OVERRIDE_DURATION = 300       # 5 minutes
    DEFAULT_EMERGENCY_DURATION = None     # No expiration (manual clear)

    def __init__(
        self,
        max_overlays: int = 100,
        audit_callback: Optional[Callable[[HumanOverlay, str], None]] = None,
    ):
        """
        Initialize human cortex bridge.

        Args:
            max_overlays: Maximum number of overlays to retain
            audit_callback: Optional callback for audit logging
        """
        self._overlays: List[HumanOverlay] = []
        self._max_overlays = max_overlays
        self._observers: List[OverlayObserver] = []
        self._audit_callback = audit_callback

        # Statistics
        self._total_submitted = 0
        self._total_expired = 0
        self._emergency_count = 0

    def submit_overlay(
        self,
        priority: OverlayPriority,
        content: str,
        operator_id: str = "system",
        target: OverlayTarget = OverlayTarget.GLOBAL,
        duration_seconds: Optional[int] = None,
        reason: str = "",
        **metadata: Any,
    ) -> HumanOverlay:
        """
        Submit a new human overlay.

        Args:
            priority: Intervention priority level
            content: The overlay content/directive
            operator_id: ID of human operator
            target: Target component
            duration_seconds: How long overlay remains active (None = use default)
            reason: Why the overlay was submitted
            **metadata: Additional metadata

        Returns:
            The created overlay
        """
        # Use default duration if not specified
        if duration_seconds is None:
            duration_seconds = self._get_default_duration(priority)

        overlay = HumanOverlay.create(
            priority=priority,
            content=content,
            operator_id=operator_id,
            target=target,
            duration_seconds=duration_seconds,
            reason=reason,
            **metadata,
        )

        # Clean up expired overlays first
        self._cleanup_expired()

        # Enforce max overlays
        while len(self._overlays) >= self._max_overlays:
            # Remove oldest non-emergency overlay
            for i, o in enumerate(self._overlays):
                if o.priority != OverlayPriority.EMERGENCY:
                    removed = self._overlays.pop(i)
                    logger.debug("Evicted old overlay: %s", removed.id)
                    break
            else:
                # All are emergency, remove oldest
                removed = self._overlays.pop(0)
                logger.warning("Evicted emergency overlay: %s", removed.id)

        # Add new overlay
        self._overlays.append(overlay)
        self._total_submitted += 1

        if priority == OverlayPriority.EMERGENCY:
            self._emergency_count += 1

        # Log audit
        self._audit("submit", overlay)

        # Notify observers
        for observer in self._observers:
            try:
                observer.on_overlay_submitted(overlay)
            except Exception as e:
                logger.warning("Observer error on submit: %s", e)

        logger.info(
            "[HITL] Overlay submitted: %s (priority=%s, target=%s, operator=%s)",
            overlay.id, priority.name, target.value, operator_id
        )

        return overlay

    def get_active_overlays(
        self,
        target: Optional[OverlayTarget] = None,
        min_priority: OverlayPriority = OverlayPriority.OBSERVE,
    ) -> List[HumanOverlay]:
        """
        Get active overlays for a target component.

        Args:
            target: Filter by target component (None = all)
            min_priority: Minimum priority to include

        Returns:
            List of active overlays, sorted by priority (highest first)
        """
        self._cleanup_expired()

        result = []
        for overlay in self._overlays:
            if not overlay.is_active():
                continue

            if overlay.priority < min_priority:
                continue

            if target and overlay.target_component not in [target, OverlayTarget.GLOBAL]:
                continue

            result.append(overlay)

        # Sort by priority (highest first)
        result.sort(key=lambda o: o.priority, reverse=True)

        return result

    def has_emergency(self) -> bool:
        """Check if there's an active emergency overlay."""
        return any(
            o.priority == OverlayPriority.EMERGENCY and o.is_active()
            for o in self._overlays
        )

    def get_highest_priority(
        self,
        target: Optional[OverlayTarget] = None,
    ) -> Optional[OverlayPriority]:
        """
        Get the highest active priority for a target.

        Args:
            target: Target component (None = global)

        Returns:
            Highest priority level, or None if no active overlays
        """
        overlays = self.get_active_overlays(target)
        if not overlays:
            return None
        return max(o.priority for o in overlays)

    def acknowledge_overlay(self, overlay_id: str) -> bool:
        """
        Acknowledge an overlay (mark as seen by system).

        Args:
            overlay_id: ID of overlay to acknowledge

        Returns:
            True if found and acknowledged
        """
        for overlay in self._overlays:
            if overlay.id == overlay_id:
                overlay.acknowledged = True
                self._audit("acknowledge", overlay)
                return True
        return False

    def apply_overlay(self, overlay_id: str) -> bool:
        """
        Mark an overlay as applied.

        Args:
            overlay_id: ID of overlay to mark

        Returns:
            True if found and marked
        """
        for overlay in self._overlays:
            if overlay.id == overlay_id:
                overlay.applied = True
                self._audit("apply", overlay)
                return True
        return False

    def clear_overlay(self, overlay_id: str) -> bool:
        """
        Clear (remove) an overlay.

        Args:
            overlay_id: ID of overlay to clear

        Returns:
            True if found and removed
        """
        for i, overlay in enumerate(self._overlays):
            if overlay.id == overlay_id:
                removed = self._overlays.pop(i)
                self._audit("clear", removed)
                logger.info("[HITL] Overlay cleared: %s", overlay_id)
                return True
        return False

    def clear_all_emergencies(self) -> int:
        """
        Clear all emergency overlays.

        Returns:
            Number of emergencies cleared
        """
        cleared = 0
        self._overlays = [
            o for o in self._overlays
            if o.priority != OverlayPriority.EMERGENCY or not (cleared := cleared + 1)
        ]
        if cleared:
            logger.info("[HITL] Cleared %d emergency overlays", cleared)
        return cleared

    def register_observer(self, observer: OverlayObserver) -> None:
        """Register an observer for overlay events."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister_observer(self, observer: OverlayObserver) -> None:
        """Unregister an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        active = [o for o in self._overlays if o.is_active()]

        return {
            "total_submitted": self._total_submitted,
            "total_expired": self._total_expired,
            "emergency_count": self._emergency_count,
            "active_overlays": len(active),
            "by_priority": {
                "observe": len([o for o in active if o.priority == OverlayPriority.OBSERVE]),
                "suggest": len([o for o in active if o.priority == OverlayPriority.SUGGEST]),
                "override": len([o for o in active if o.priority == OverlayPriority.OVERRIDE]),
                "emergency": len([o for o in active if o.priority == OverlayPriority.EMERGENCY]),
            },
            "has_emergency": self.has_emergency(),
        }

    def _get_default_duration(self, priority: OverlayPriority) -> Optional[int]:
        """Get default duration for a priority level."""
        return {
            OverlayPriority.OBSERVE: self.DEFAULT_OBSERVE_DURATION,
            OverlayPriority.SUGGEST: self.DEFAULT_SUGGEST_DURATION,
            OverlayPriority.OVERRIDE: self.DEFAULT_OVERRIDE_DURATION,
            OverlayPriority.EMERGENCY: self.DEFAULT_EMERGENCY_DURATION,
        }.get(priority)

    def _cleanup_expired(self) -> None:
        """Remove expired overlays."""
        expired: List[HumanOverlay] = []

        self._overlays = [
            o for o in self._overlays
            if not o.is_expired() or not expired.append(o)
        ]

        for overlay in expired:
            self._total_expired += 1
            self._audit("expire", overlay)

            for observer in self._observers:
                try:
                    observer.on_overlay_expired(overlay)
                except Exception as e:
                    logger.warning("Observer error on expire: %s", e)

    def _audit(self, action: str, overlay: HumanOverlay) -> None:
        """Log audit event."""
        if self._audit_callback:
            try:
                self._audit_callback(overlay, action)
            except Exception as e:
                logger.warning("Audit callback error: %s", e)


def create_human_cortex_bridge(
    audit_callback: Optional[Callable[[HumanOverlay, str], None]] = None,
    max_overlays: int = 100,
) -> HumanCortexBridge:
    """
    Factory function to create HumanCortexBridge.

    Args:
        audit_callback: Optional callback for audit logging
        max_overlays: Maximum overlays to retain

    Returns:
        Configured HumanCortexBridge instance
    """
    return HumanCortexBridge(
        max_overlays=max_overlays,
        audit_callback=audit_callback,
    )
