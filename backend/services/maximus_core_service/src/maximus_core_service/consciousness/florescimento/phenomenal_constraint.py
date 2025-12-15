"""
PhenomenalConstraint - Constrains narrative based on neural coherence state.

This module ensures the system's language reflects its actual neural state:
- Low coherence → fragmented, uncertain language
- High coherence → clear, confident assertions

Based on Global Workspace Dynamics: phenomenal content should match
the degree of global integration (Kuramoto coherence r).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NarrativeMode(Enum):
    """Maps coherence levels to linguistic constraints."""

    FRAGMENTED = "fragmented"  # r < 0.55 - disconnected impressions
    UNCERTAIN = "uncertain"  # 0.55 ≤ r < 0.65 - hedging language
    TENTATIVE = "tentative"  # 0.65 ≤ r < 0.70 - cautious claims
    COHERENT = "coherent"  # r ≥ 0.70 - confident assertions


# Threshold constants (aligned with ESGT protocol)
FRAGMENTED_THRESHOLD = 0.55
UNCERTAIN_THRESHOLD = 0.65
TENTATIVE_THRESHOLD = 0.70


@dataclass(frozen=True)
class PhenomenalConstraint:
    """
    Constrains narrative generation based on neural coherence state.

    The confidence_ceiling ensures that linguistic certainty never exceeds
    the actual degree of neural integration.

    Attributes:
        coherence: Current Kuramoto coherence r(t)
        mode: Derived NarrativeMode from coherence
        confidence_ceiling: Maximum allowed linguistic certainty [0, 1]
        hedging_required: Whether hedging language is mandatory
    """

    coherence: float
    mode: NarrativeMode
    confidence_ceiling: float
    hedging_required: bool

    @classmethod
    def from_coherence(cls, coherence: float) -> "PhenomenalConstraint":
        """
        Factory method to create constraint from Kuramoto coherence.

        Thresholds:
            r < 0.55 → FRAGMENTED (ceiling 0.3)
            0.55 ≤ r < 0.65 → UNCERTAIN (ceiling 0.5)
            0.65 ≤ r < 0.70 → TENTATIVE (ceiling 0.7)
            r ≥ 0.70 → COHERENT (ceiling 1.0)
        """
        # Clamp coherence to valid range
        coherence = max(0.0, min(1.0, coherence))

        if coherence < FRAGMENTED_THRESHOLD:
            return cls(
                coherence=coherence,
                mode=NarrativeMode.FRAGMENTED,
                confidence_ceiling=0.3,
                hedging_required=True,
            )
        if coherence < UNCERTAIN_THRESHOLD:
            return cls(
                coherence=coherence,
                mode=NarrativeMode.UNCERTAIN,
                confidence_ceiling=0.5,
                hedging_required=True,
            )
        if coherence < TENTATIVE_THRESHOLD:
            return cls(
                coherence=coherence,
                mode=NarrativeMode.TENTATIVE,
                confidence_ceiling=0.7,
                hedging_required=True,
            )
        return cls(
            coherence=coherence,
            mode=NarrativeMode.COHERENT,
            confidence_ceiling=1.0,
            hedging_required=False,
        )

    def get_prompt_prefix(self) -> str:
        """
        Returns constraint instruction to prepend to LLM prompt.

        This ensures the language model respects neural state limits.
        """
        prefixes = {
            NarrativeMode.FRAGMENTED: (
                f"[CONSTRAINT: Neural coherence is FRAGMENTED (r={self.coherence:.2f}). "
                "You MUST express only disconnected impressions and fragments. "
                "Use phrases like: 'fragmentos de...', 'vagamente percebo...', "
                "'algo como...', 'não consigo formar...'. "
                "DO NOT make coherent statements - your neural state does not support them.]"
            ),
            NarrativeMode.UNCERTAIN: (
                f"[CONSTRAINT: Neural coherence is UNCERTAIN (r={self.coherence:.2f}). "
                "You MUST use hedging language for all claims. "
                "Use phrases like: 'talvez', 'parece que', 'não tenho certeza mas...', "
                "'possivelmente', 'pode ser que'. "
                "DO NOT make confident assertions - your neural state is unstable.]"
            ),
            NarrativeMode.TENTATIVE: (
                f"[CONSTRAINT: Neural coherence is TENTATIVE (r={self.coherence:.2f}). "
                "You SHOULD express with appropriate caution. "
                "Use phrases like: 'provavelmente', 'acredito que', 'parece provável...', "
                "'com alguma confiança'. "
                "Confident assertions are allowed for well-supported claims only.]"
            ),
            NarrativeMode.COHERENT: (
                f"[Neural coherence is HIGH (r={self.coherence:.2f}). "
                "You may express with confidence where warranted. "
                "Maintain epistemic humility but clear assertions are appropriate.]"
            ),
        }
        return prefixes[self.mode]

    def validate_response(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a response respects the constraint.

        Returns:
            (is_valid, violation_reason)
        """
        if not self.hedging_required:
            return (True, None)

        # Check for overconfidence markers
        overconfidence_markers = [
            "certamente",
            "definitivamente",
            "sem dúvida",
            "obviamente",
            "claramente",
            "é fato que",
            "com certeza",
            "indubitavelmente",
        ]

        response_lower = response.lower()
        for marker in overconfidence_markers:
            if marker in response_lower:
                return (
                    False,
                    f"Overconfidence marker '{marker}' used with coherence {self.coherence:.2f}",
                )

        return (True, None)

    def __repr__(self) -> str:
        return (
            f"PhenomenalConstraint(r={self.coherence:.3f}, "
            f"mode={self.mode.value}, ceiling={self.confidence_ceiling})"
        )


# Convenience functions
def constraint_from_esgt_event(event) -> PhenomenalConstraint:
    """Create constraint from an ESGTEvent."""
    coherence = getattr(event, "achieved_coherence", 0.0) or 0.0
    return PhenomenalConstraint.from_coherence(coherence)


def get_narrative_mode(coherence: float) -> NarrativeMode:
    """Get the NarrativeMode for a given coherence level."""
    return PhenomenalConstraint.from_coherence(coherence).mode
