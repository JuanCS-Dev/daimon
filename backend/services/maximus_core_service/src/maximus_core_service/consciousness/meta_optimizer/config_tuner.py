"""
Config Tuner - Automatic Hyperparameter Optimization
=====================================================

Automatically adjusts system parameters based on coherence feedback.
Implements the recursive self-improvement pattern from Dec 2025 research.

Tunable Parameters:
- Kuramoto coupling strength
- ESGT trigger thresholds
- LLM temperature
- Processing timeouts

Safety:
- All adjustments are bounded within safe ranges
- Changes are logged for HITL review
- Rollback capability for failed optimizations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TuningStrategy(Enum):
    """Strategy for parameter adjustment."""
    CONSERVATIVE = "conservative"  # Small 1-2% changes
    MODERATE = "moderate"          # 3-5% changes
    AGGRESSIVE = "aggressive"      # 5-10% changes


@dataclass
class TuningResult:
    """Result of a tuning operation."""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    was_applied: bool = True
    
    @property
    def change_pct(self) -> float:
        if self.old_value == 0:
            return 0.0
        return ((self.new_value - self.old_value) / self.old_value) * 100


@dataclass
class ParameterBounds:
    """Safe bounds for a tunable parameter."""
    min_value: float
    max_value: float
    default_value: float
    description: str


class ConfigTuner:
    """
    Automatic configuration tuner for Noesis parameters.
    
    Uses coherence feedback to adjust parameters within safe bounds.
    All changes are logged for HITL review.
    """
    
    # Define tunable parameters with safe bounds
    PARAMETER_BOUNDS: Dict[str, ParameterBounds] = {
        "kuramoto_coupling": ParameterBounds(
            min_value=0.1, max_value=2.0, default_value=0.5,
            description="Kuramoto oscillator coupling strength"
        ),
        "esgt_min_salience": ParameterBounds(
            min_value=0.3, max_value=0.9, default_value=0.65,
            description="Minimum salience for ESGT trigger"
        ),
        "llm_temperature": ParameterBounds(
            min_value=0.0, max_value=1.0, default_value=0.7,
            description="LLM generation temperature"
        ),
        "processing_timeout_ms": ParameterBounds(
            min_value=1000, max_value=30000, default_value=10000,
            description="Maximum processing time before timeout"
        ),
        "coherence_target": ParameterBounds(
            min_value=0.5, max_value=0.95, default_value=0.70,
            description="Target coherence for ESGT ignition"
        ),
    }
    
    def __init__(
        self,
        strategy: TuningStrategy = TuningStrategy.CONSERVATIVE,
        min_samples_before_tuning: int = 50
    ):
        self.strategy = strategy
        self.min_samples = min_samples_before_tuning
        
        # Current parameter values
        self._current_values: Dict[str, float] = {
            name: bounds.default_value
            for name, bounds in self.PARAMETER_BOUNDS.items()
        }
        
        # Tuning history for rollback
        self._history: List[TuningResult] = []
        self._max_history = 100
    
    def get_current_value(self, parameter: str) -> Optional[float]:
        """Get current value of a parameter."""
        return self._current_values.get(parameter)
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current parameter values."""
        return self._current_values.copy()
    
    def suggest_adjustment(
        self,
        parameter: str,
        current_coherence: float,
        target_coherence: float = 0.70,
        latency_ms: Optional[float] = None
    ) -> Optional[TuningResult]:
        """
        Suggest a parameter adjustment based on coherence feedback.
        
        Args:
            parameter: Parameter name to adjust
            current_coherence: Current system coherence
            target_coherence: Desired coherence level
            latency_ms: Optional latency data for timeout tuning
            
        Returns:
            TuningResult if adjustment suggested, None otherwise
        """
        if parameter not in self.PARAMETER_BOUNDS:
            logger.warning("[Tuner] Unknown parameter: %s", parameter)
            return None
        
        bounds = self.PARAMETER_BOUNDS[parameter]
        current_value = self._current_values[parameter]
        
        # Calculate adjustment magnitude based on strategy
        if self.strategy == TuningStrategy.CONSERVATIVE:
            max_change_pct = 0.02  # 2%
        elif self.strategy == TuningStrategy.MODERATE:
            max_change_pct = 0.05  # 5%
        else:
            max_change_pct = 0.10  # 10%
        
        # Determine direction based on coherence gap
        coherence_gap = target_coherence - current_coherence
        
        if abs(coherence_gap) < 0.02:
            # Close enough to target, no adjustment needed
            return None
        
        # Adjust based on parameter type
        if parameter == "kuramoto_coupling":
            # Low coherence -> increase coupling
            if coherence_gap > 0:
                change = current_value * max_change_pct
            else:
                change = -current_value * max_change_pct
            new_value = current_value + change
            reason = f"Coherence gap: {coherence_gap:.3f}"
            
        elif parameter == "esgt_min_salience":
            # Low coherence might mean threshold too high
            if coherence_gap > 0:
                change = -current_value * max_change_pct  # Lower threshold
            else:
                change = current_value * max_change_pct  # Raise threshold
            new_value = current_value + change
            reason = f"Adjusting trigger sensitivity"
            
        elif parameter == "processing_timeout_ms":
            # Use latency data if available
            if latency_ms and latency_ms > current_value * 0.8:
                new_value = min(current_value * 1.1, bounds.max_value)
                reason = f"Latency {latency_ms:.0f}ms approaching timeout"
            else:
                return None
                
        else:
            # Generic adjustment
            change = current_value * max_change_pct * (1 if coherence_gap > 0 else -1)
            new_value = current_value + change
            reason = f"Auto-tune based on coherence"
        
        # Clamp to bounds
        new_value = max(bounds.min_value, min(bounds.max_value, new_value))
        
        if new_value == current_value:
            return None
        
        return TuningResult(
            parameter=parameter,
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            was_applied=False  # Not yet applied
        )
    
    def apply_adjustment(self, result: TuningResult) -> bool:
        """
        Apply a suggested adjustment.
        
        Args:
            result: TuningResult from suggest_adjustment
            
        Returns:
            True if applied successfully
        """
        if result.parameter not in self.PARAMETER_BOUNDS:
            return False
        
        bounds = self.PARAMETER_BOUNDS[result.parameter]
        
        # Validate within bounds
        if not (bounds.min_value <= result.new_value <= bounds.max_value):
            logger.error(
                "[Tuner] Value %.3f out of bounds [%.3f, %.3f]",
                result.new_value, bounds.min_value, bounds.max_value
            )
            return False
        
        # Apply change
        old_value = self._current_values[result.parameter]
        self._current_values[result.parameter] = result.new_value
        result.was_applied = True
        
        # Log for HITL review
        logger.info(
            "[Tuner] Applied: %s %.4f -> %.4f (reason: %s)",
            result.parameter, old_value, result.new_value, result.reason
        )
        
        # Track in history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return True
    
    def rollback_last(self) -> Optional[TuningResult]:
        """
        Rollback the most recent adjustment.
        
        Returns:
            The rolled-back TuningResult, or None if no history
        """
        if not self._history:
            return None
        
        last = self._history.pop()
        self._current_values[last.parameter] = last.old_value
        
        logger.warning(
            "[Tuner] Rolled back: %s %.4f -> %.4f",
            last.parameter, last.new_value, last.old_value
        )
        
        return last
    
    def get_tuning_history(self, count: int = 10) -> List[TuningResult]:
        """Get recent tuning history."""
        return self._history[-count:]
    
    def reset_to_defaults(self) -> None:
        """Reset all parameters to default values."""
        for name, bounds in self.PARAMETER_BOUNDS.items():
            self._current_values[name] = bounds.default_value
        
        logger.info("[Tuner] Reset all parameters to defaults")


# Global instance
_global_tuner: Optional[ConfigTuner] = None


def get_config_tuner() -> ConfigTuner:
    """Get or create the global config tuner."""
    global _global_tuner
    if _global_tuner is None:
        _global_tuner = ConfigTuner()
    return _global_tuner
