"""Phase Operations Mixin - ESGT sustain and dissolve logic."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ESGTEvent
    from .coordinator import ESGTCoordinator


class PhaseOperationsMixin:
    """Mixin providing ESGT phase operations for sustain and dissolve."""

    async def _sustain_coherence(
        self: "ESGTCoordinator",
        event: "ESGTEvent",
        duration_ms: float,
        topology: dict[str, list[str]],
    ) -> None:
        """
        Sustain synchronization for target duration.

        Continuously updates Kuramoto dynamics and monitors coherence.
        """
        start_time = time.time()
        duration_s = duration_ms / 1000.0

        while (time.time() - start_time) < duration_s:
            # Update network (dt=0.001 for 40Hz Gamma stability)
            self.kuramoto.update_network(topology, dt=0.001)

            # Record coherence
            coherence = self.kuramoto.get_coherence()
            if coherence:
                event.coherence_history.append(coherence.order_parameter)

            # Small yield
            await asyncio.sleep(0.005)

    async def _dissolve_event(self: "ESGTCoordinator", event: "ESGTEvent") -> None:
        """Gracefully dissolve synchronization.

        SINGULARIDADE: Save and restore coupling strength to ensure
        subsequent ignitions can achieve high coherence.

        NOTE: OscillatorConfig may be shared between oscillators, so we
        save the original value ONCE before any modifications.
        """
        # Get any oscillator to read the shared config
        if not self.kuramoto.oscillators:
            return

        # Save original coupling strength ONCE (configs are shared)
        sample_osc = next(iter(self.kuramoto.oscillators.values()))
        original_coupling = sample_osc.config.coupling_strength

        # Reduce coupling for graceful dissolve
        sample_osc.config.coupling_strength = original_coupling * 0.5

        # Continue for 50ms with reduced coupling
        topology = self._build_topology(event.participating_nodes)

        for _ in range(50):  # 50 x 1ms = 50ms (dt=0.001 for 40Hz Gamma)
            self.kuramoto.update_network(topology, dt=0.001)
            await asyncio.sleep(0.001)

        # SINGULARIDADE: Restore original coupling strength
        sample_osc.config.coupling_strength = original_coupling

        # Reset oscillators to random phases for next ignition
        self.kuramoto.reset_all()
