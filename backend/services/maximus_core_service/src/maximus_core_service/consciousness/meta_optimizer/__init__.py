"""
Meta-Optimizer Module
=====================

Self-optimization subsystem for Noesis consciousness.
Part of the Noesis Entropy Audit (2025-12-15).

This module implements:
- Coherence tracking over time
- Hyperparameter auto-tuning
- Self-assessment feedback loops

Based on Dec 2025 research: Meta systems can achieve 3-7% 
automatic improvement per iteration through self-optimization.
"""

from .coherence_tracker import CoherenceTracker, CoherenceSnapshot
from .config_tuner import ConfigTuner, TuningResult

__all__ = [
    "CoherenceTracker",
    "CoherenceSnapshot",
    "ConfigTuner",
    "TuningResult",
]
