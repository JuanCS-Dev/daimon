"""
DAIMON Anticipation Engine - Proactive Emergence Decision
==========================================================

Decides when and how NOESIS should proactively emerge based on patterns.

Emergence Modes:
- SUBTLE: Background notification only
- NORMAL: Chat message to user
- URGENT: Immediate action required

Features:
- Pattern-based emergence decisions
- Cooldown to prevent spam (default 10 min)
- Context-aware mode selection
- NOESIS integration for triggering emergence

Usage:
    from learners.anticipation_engine import get_anticipation_engine
    from learners.pattern_detector import get_pattern_detector
    
    engine = get_anticipation_engine()
    patterns = get_pattern_detector().get_matching_patterns(context)
    decision = engine.evaluate(context, patterns)
    
    if decision.should_emerge:
        await engine.trigger_emergence(decision)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("daimon.anticipation_engine")


class EmergenceMode(Enum):
    """Mode of proactive emergence."""
    SUBTLE = "subtle"       # Background notification only
    NORMAL = "normal"       # Chat message to user
    URGENT = "urgent"       # Immediate action required


@dataclass
class EmergenceDecision:
    """Decision about whether to emerge proactively."""
    
    should_emerge: bool
    mode: EmergenceMode
    reason: str
    confidence: float
    pattern: Optional[Any] = None  # Pattern that triggered decision
    cooldown_remaining: float = 0.0  # Seconds until can emerge again
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_emerge": self.should_emerge,
            "mode": self.mode.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "pattern": self.pattern.to_dict() if self.pattern else None,
            "cooldown_remaining": self.cooldown_remaining,
        }


@dataclass
class EmergenceStats:
    """Statistics for monitoring emergence behavior."""
    
    total_evaluations: int = 0
    total_emergences: int = 0
    emergences_by_mode: Dict[str, int] = field(default_factory=lambda: {
        "subtle": 0,
        "normal": 0,
        "urgent": 0,
    })
    blocked_by_cooldown: int = 0
    last_emergence: Optional[datetime] = None


class AnticipationEngine:
    """
    Decides when NOESIS should proactively emerge.
    
    Logic:
    - Evaluates current context against detected patterns
    - Decides if emergence is warranted (confidence >= threshold)
    - Selects emergence mode based on confidence level
    - Enforces cooldown to prevent spam
    
    Thresholds:
    - URGENT: confidence >= 0.8
    - NORMAL: confidence >= 0.7
    - SUBTLE: confidence >= 0.5
    - No emergence: confidence < 0.5
    """
    
    # Confidence thresholds for emergence modes
    URGENT_THRESHOLD = 0.8
    NORMAL_THRESHOLD = 0.7
    SUBTLE_THRESHOLD = 0.5
    
    # Default cooldown: 10 minutes
    DEFAULT_COOLDOWN_SECONDS = 600
    
    def __init__(
        self,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        min_patterns_for_emergence: int = 1,
    ):
        """
        Initialize anticipation engine.
        
        Args:
            cooldown_seconds: Seconds between emergences (default 10 min)
            min_patterns_for_emergence: Minimum patterns needed to emerge
        """
        self.cooldown_seconds = cooldown_seconds
        self.min_patterns = min_patterns_for_emergence
        
        self._last_emergence_time: float = 0.0
        self._stats = EmergenceStats()
        
    def evaluate(
        self,
        context: Dict[str, Any],
        patterns: List[Any],
    ) -> EmergenceDecision:
        """
        Evaluate whether to emerge based on context and patterns.
        
        Args:
            context: Current user context (app, time, etc.)
            patterns: Patterns matching current context
            
        Returns:
            EmergenceDecision with recommendation
        """
        self._stats.total_evaluations += 1
        
        # Check cooldown
        cooldown_remaining = self._get_cooldown_remaining()
        if cooldown_remaining > 0:
            self._stats.blocked_by_cooldown += 1
            return EmergenceDecision(
                should_emerge=False,
                mode=EmergenceMode.SUBTLE,
                reason=f"Cooldown active ({cooldown_remaining:.0f}s remaining)",
                confidence=0.0,
                cooldown_remaining=cooldown_remaining,
            )
        
        # Check if enough patterns
        if len(patterns) < self.min_patterns:
            return EmergenceDecision(
                should_emerge=False,
                mode=EmergenceMode.SUBTLE,
                reason=f"Insufficient patterns ({len(patterns)} < {self.min_patterns})",
                confidence=0.0,
            )
        
        # Find best pattern
        best_pattern = max(patterns, key=lambda p: p.confidence)
        confidence = best_pattern.confidence
        
        # Apply context boost
        confidence = self._apply_context_boost(confidence, context, best_pattern)
        
        # Determine mode based on confidence
        if confidence >= self.URGENT_THRESHOLD:
            mode = EmergenceMode.URGENT
            should_emerge = True
        elif confidence >= self.NORMAL_THRESHOLD:
            mode = EmergenceMode.NORMAL
            should_emerge = True
        elif confidence >= self.SUBTLE_THRESHOLD:
            mode = EmergenceMode.SUBTLE
            should_emerge = True
        else:
            mode = EmergenceMode.SUBTLE
            should_emerge = False
        
        # Build reason
        if should_emerge:
            reason = f"Pattern matched: {best_pattern.description}"
        else:
            reason = f"Confidence too low ({confidence:.2f} < {self.SUBTLE_THRESHOLD})"
        
        return EmergenceDecision(
            should_emerge=should_emerge,
            mode=mode,
            reason=reason,
            confidence=confidence,
            pattern=best_pattern if should_emerge else None,
        )
    
    def _apply_context_boost(
        self,
        base_confidence: float,
        context: Dict[str, Any],
        pattern: Any,
    ) -> float:
        """Apply context-based confidence boost."""
        boost = 0.0
        
        # Boost if multiple patterns match
        # (handled externally, but we can boost based on pattern type)
        pattern_type = getattr(pattern, "pattern_type", "")
        
        if pattern_type == "sequential":
            # Sequential patterns are more reliable
            boost += 0.05
        
        if pattern_type == "temporal":
            # Check if we're close to the peak time
            peak_hour = pattern.data.get("peak_hour") if hasattr(pattern, "data") else None
            if peak_hour is not None:
                current_hour = context.get("hour", datetime.now().hour)
                if current_hour == peak_hour:
                    boost += 0.1  # Exact time match
                elif abs(current_hour - peak_hour) == 1:
                    boost += 0.05  # Within 1 hour
        
        return min(base_confidence + boost, 1.0)
    
    def _get_cooldown_remaining(self) -> float:
        """Get seconds remaining in cooldown."""
        if self._last_emergence_time == 0:
            return 0.0
        
        elapsed = time.time() - self._last_emergence_time
        remaining = self.cooldown_seconds - elapsed
        
        return max(0.0, remaining)
    
    async def trigger_emergence(
        self,
        decision: EmergenceDecision,
    ) -> bool:
        """
        Trigger proactive emergence via NOESIS.
        
        Args:
            decision: The emergence decision
            
        Returns:
            True if emergence was triggered successfully
        """
        if not decision.should_emerge:
            logger.debug("Skipping emergence: decision is not to emerge")
            return False
        
        try:
            # Try to send emergence signal to NOESIS
            from integrations.mcp_tools.http_utils import http_post
            from integrations.mcp_tools.config import NOESIS_CONSCIOUSNESS_URL
            
            payload = {
                "mode": decision.mode.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "pattern": decision.pattern.to_dict() if decision.pattern else None,
                "timestamp": datetime.now().isoformat(),
            }
            
            # POST to NOESIS consciousness endpoint
            url = f"{NOESIS_CONSCIOUSNESS_URL}/v1/consciousness/emerge"
            response = await http_post(url, payload)
            
            if response.get("status") == "ok":
                self._record_emergence(decision)
                logger.info(
                    "Emergence triggered: mode=%s, reason=%s",
                    decision.mode.value,
                    decision.reason,
                )
                return True
            else:
                logger.warning("NOESIS emergence failed: %s", response)
                return False
                
        except ImportError:
            # NOESIS not available, log locally
            logger.info(
                "NOESIS unavailable - logging emergence: mode=%s, reason=%s",
                decision.mode.value,
                decision.reason,
            )
            self._record_emergence(decision)
            return True
            
        except Exception as e:
            logger.error("Failed to trigger emergence: %s", e)
            return False
    
    def _record_emergence(self, decision: EmergenceDecision) -> None:
        """Record that an emergence occurred."""
        self._last_emergence_time = time.time()
        self._stats.total_emergences += 1
        self._stats.emergences_by_mode[decision.mode.value] += 1
        self._stats.last_emergence = datetime.now()
    
    def get_cooldown_status(self) -> Dict[str, Any]:
        """Get current cooldown status."""
        remaining = self._get_cooldown_remaining()
        return {
            "is_active": remaining > 0,
            "remaining_seconds": remaining,
            "cooldown_duration": self.cooldown_seconds,
            "last_emergence": (
                self._stats.last_emergence.isoformat()
                if self._stats.last_emergence else None
            ),
        }
    
    def reset_cooldown(self) -> None:
        """Reset cooldown (for testing or manual override)."""
        self._last_emergence_time = 0.0
        logger.info("Cooldown reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_evaluations": self._stats.total_evaluations,
            "total_emergences": self._stats.total_emergences,
            "emergences_by_mode": self._stats.emergences_by_mode,
            "blocked_by_cooldown": self._stats.blocked_by_cooldown,
            "cooldown_status": self.get_cooldown_status(),
        }
    
    def clear_stats(self) -> None:
        """Clear statistics."""
        self._stats = EmergenceStats()


# Singleton instance
_engine: Optional[AnticipationEngine] = None


def get_anticipation_engine() -> AnticipationEngine:
    """Get singleton anticipation engine instance."""
    global _engine
    if _engine is None:
        _engine = AnticipationEngine()
    return _engine


def reset_anticipation_engine() -> None:
    """Reset singleton anticipation engine."""
    global _engine
    if _engine:
        _engine.clear_stats()
    _engine = None


if __name__ == "__main__":
    print("=" * 60)
    print("DAIMON Anticipation Engine - Manual Test")
    print("=" * 60)
    
    from learners.pattern_detector import Pattern
    
    engine = AnticipationEngine(cooldown_seconds=5)  # 5s for testing
    
    # Create mock patterns
    patterns = [
        Pattern(
            pattern_type="temporal",
            description="git commit typically occurs around 17:00",
            confidence=0.75,
            occurrences=10,
            last_seen=datetime.now(),
            data={"peak_hour": 17},
        ),
    ]
    
    context = {"hour": 17, "app": "terminal"}
    
    # Test evaluation
    print("\n1. Testing evaluation...")
    decision = engine.evaluate(context, patterns)
    print(f"   Should emerge: {decision.should_emerge}")
    print(f"   Mode: {decision.mode.value}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Reason: {decision.reason}")
    
    # Test cooldown
    print("\n2. Testing cooldown...")
    engine._last_emergence_time = time.time()  # Simulate recent emergence
    decision2 = engine.evaluate(context, patterns)
    print(f"   Blocked by cooldown: {not decision2.should_emerge}")
    print(f"   Cooldown remaining: {decision2.cooldown_remaining:.0f}s")
    
    # Reset and test again
    print("\n3. After cooldown reset...")
    engine.reset_cooldown()
    decision3 = engine.evaluate(context, patterns)
    print(f"   Should emerge: {decision3.should_emerge}")
    
    print("\n4. Stats:")
    for k, v in engine.get_stats().items():
        print(f"   {k}: {v}")
    
    print("\nTest completed!")
