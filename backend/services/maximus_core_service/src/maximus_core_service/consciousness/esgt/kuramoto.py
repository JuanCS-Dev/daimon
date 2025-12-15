"""
Kuramoto Model - Phase Synchronization for ESGT Coherence.

Implements coupled oscillators (Kuramoto, 1975) for ESGT phase coherence.
Each TIG node has an oscillator that couples with neighbors during ignition.

Order Parameter (Coherence): r(t) = (1/N) |Σⱼ exp(iθⱼ)|
- r = 0: complete incoherence (random phases)
- r ≥ 0.70: conscious-level coherence (required for ESGT)
- r = 1: perfect synchronization
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from maximus_core_service.consciousness.esgt.kuramoto_models import (
    OscillatorConfig,
    OscillatorState,
    PhaseCoherence,
    SynchronizationDynamics,
)


class KuramotoOscillator:
    """
    Single Kuramoto oscillator for ESGT phase synchronization.

    Analogous to a cortical neural population with gamma-band oscillations (~40 Hz).
    During conscious states, populations phase-lock through synaptic coupling.
    """

    def __init__(self, node_id: str, config: OscillatorConfig | None = None) -> None:
        self.node_id = node_id
        self.config = config or OscillatorConfig()

        self.phase: float = np.random.uniform(0, 2 * np.pi)
        self.frequency: float = self.config.natural_frequency
        self.state: OscillatorState = OscillatorState.IDLE

        self.phase_history: list[float] = [self.phase]
        self.frequency_history: list[float] = [self.frequency]

    def _compute_phase_velocity(
        self,
        current_phase: float,
        neighbor_phases: dict[str, float],
        coupling_weights: dict[str, float],
        N: int,
    ) -> float:
        """Compute phase velocity: dθᵢ/dt = ωᵢ + (K/N)Σⱼ wⱼ sin(θⱼ - θᵢ)"""
        phase_velocity = 2 * np.pi * self.frequency

        if neighbor_phases:
            coupling_sum = 0.0
            for neighbor_id, neighbor_phase in neighbor_phases.items():
                weight = coupling_weights.get(neighbor_id, 1.0)
                phase_diff = neighbor_phase - current_phase
                coupling_sum += weight * np.sin(phase_diff)

            coupling_term = self.config.coupling_strength * (coupling_sum / N)
            phase_velocity += coupling_term

        return phase_velocity

    def update(
        self,
        neighbor_phases: dict[str, float],
        coupling_weights: dict[str, float],
        dt: float = 0.001,  # SINGULARIDADE: 0.001 for 40Hz Gamma (was 0.005)
        N: int | None = None,
    ) -> float:
        """Update oscillator phase based on Kuramoto dynamics."""
        self.state = OscillatorState.COUPLING

        if N is None:
            N = len(neighbor_phases) if neighbor_phases else 1

        noise = np.random.normal(0, self.config.phase_noise)

        if self.config.integration_method == "rk4":
            k1 = dt * self._compute_phase_velocity(self.phase, neighbor_phases, coupling_weights, N)
            k2 = dt * self._compute_phase_velocity(
                self.phase + 0.5 * k1, neighbor_phases, coupling_weights, N
            )
            k3 = dt * self._compute_phase_velocity(
                self.phase + 0.5 * k2, neighbor_phases, coupling_weights, N
            )
            k4 = dt * self._compute_phase_velocity(
                self.phase + k3, neighbor_phases, coupling_weights, N
            )
            self.phase += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0 + noise * dt
        else:
            phase_velocity = self._compute_phase_velocity(
                self.phase, neighbor_phases, coupling_weights, N
            )
            self.phase += (phase_velocity + noise) * dt

        self.phase = self.phase % (2 * np.pi)

        self.phase_history.append(self.phase)
        current_velocity = self._compute_phase_velocity(
            self.phase, neighbor_phases, coupling_weights, N
        )
        self.frequency_history.append(current_velocity / (2 * np.pi))

        if len(self.phase_history) > 1000:
            self.phase_history.pop(0)
            self.frequency_history.pop(0)

        return self.phase

    def get_phase(self) -> float:
        """Get current phase (radians)."""
        return self.phase

    def set_phase(self, phase: float) -> None:
        """Set phase explicitly."""
        self.phase = phase % (2 * np.pi)

    def reset(self) -> None:
        """Reset to random phase."""
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.state = OscillatorState.IDLE
        self.phase_history = [self.phase]

    def __repr__(self) -> str:
        return f"KuramotoOscillator(node={self.node_id}, phase={self.phase:.3f}, freq={self.frequency:.1f}Hz)"


class KuramotoNetwork:
    """
    Network of coupled Kuramoto oscillators for ESGT.

    Coordinates phase synchronization across TIG nodes during ignition events,
    computing global coherence in real-time.
    """

    def __init__(self, config: OscillatorConfig | None = None) -> None:
        self.default_config = config or OscillatorConfig()
        self.oscillators: dict[str, KuramotoOscillator] = {}
        self.dynamics = SynchronizationDynamics()
        self._coherence_cache: PhaseCoherence | None = None
        self._last_coherence_time: float = 0.0

    def add_oscillator(self, node_id: str, config: OscillatorConfig | None = None) -> None:
        """Add oscillator for a TIG node."""
        osc_config = config or self.default_config
        self.oscillators[node_id] = KuramotoOscillator(node_id, osc_config)

    def remove_oscillator(self, node_id: str) -> None:
        """Remove oscillator."""
        if node_id in self.oscillators:
            del self.oscillators[node_id]

    def reset_all(self) -> None:
        """Reset all oscillators to random phases."""
        for osc in self.oscillators.values():
            osc.reset()
        self.dynamics = SynchronizationDynamics()
        self._coherence_cache = None

    def _compute_network_derivatives(
        self,
        phases: dict[str, float],
        topology: dict[str, list[str]],
        coupling_weights: dict[tuple[str, str], float] | None,
    ) -> dict[str, float]:
        """Compute phase velocities for all oscillators."""
        N = len(self.oscillators)
        velocities = {}

        for node_id, osc in self.oscillators.items():
            neighbors = topology.get(node_id, [])
            neighbor_phases = {n: phases[n] for n in neighbors if n in phases}

            weights = {}
            if coupling_weights:
                for n in neighbors:
                    key = (node_id, n)
                    weights[n] = coupling_weights.get(key, 1.0)
            else:
                weights = {n: 1.0 for n in neighbors}

            velocities[node_id] = osc._compute_phase_velocity(
                phases[node_id], neighbor_phases, weights, N
            )

        return velocities

    def update_network(
        self,
        topology: dict[str, list[str]],
        coupling_weights: dict[tuple[str, str], float] | None = None,
        dt: float = 0.001,  # SINGULARIDADE: 0.001 for 40Hz Gamma (was 0.005)
    ) -> None:
        """Update all oscillators in parallel based on network topology."""
        current_phases = {node_id: osc.get_phase() for node_id, osc in self.oscillators.items()}
        N = len(self.oscillators)

        integration_method = next(iter(self.oscillators.values())).config.integration_method

        if integration_method == "rk4":
            velocities_k1 = self._compute_network_derivatives(
                current_phases, topology, coupling_weights
            )
            k1 = {node_id: dt * vel for node_id, vel in velocities_k1.items()}

            phases_k2 = {
                node_id: current_phases[node_id] + 0.5 * k1[node_id] for node_id in current_phases
            }
            velocities_k2 = self._compute_network_derivatives(phases_k2, topology, coupling_weights)
            k2 = {node_id: dt * vel for node_id, vel in velocities_k2.items()}

            phases_k3 = {
                node_id: current_phases[node_id] + 0.5 * k2[node_id] for node_id in current_phases
            }
            velocities_k3 = self._compute_network_derivatives(phases_k3, topology, coupling_weights)
            k3 = {node_id: dt * vel for node_id, vel in velocities_k3.items()}

            phases_k4 = {
                node_id: current_phases[node_id] + k3[node_id] for node_id in current_phases
            }
            velocities_k4 = self._compute_network_derivatives(phases_k4, topology, coupling_weights)
            k4 = {node_id: dt * vel for node_id, vel in velocities_k4.items()}

            for node_id, osc in self.oscillators.items():
                noise = np.random.normal(0, osc.config.phase_noise)
                new_phase = (
                    current_phases[node_id]
                    + (k1[node_id] + 2 * k2[node_id] + 2 * k3[node_id] + k4[node_id]) / 6.0
                    + noise * dt
                )
                osc.phase = new_phase % (2 * np.pi)
                osc.phase_history.append(osc.phase)
                osc.frequency_history.append(velocities_k1[node_id] / (2 * np.pi))
        else:
            for node_id, osc in self.oscillators.items():
                neighbors = topology.get(node_id, [])
                neighbor_phases = {n: current_phases[n] for n in neighbors if n in current_phases}

                weights = {}
                if coupling_weights:
                    for n in neighbors:
                        key = (node_id, n)
                        weights[n] = coupling_weights.get(key, 1.0)
                else:
                    weights = {n: 1.0 for n in neighbors}

                osc.update(neighbor_phases, weights, dt, N)

        self._update_coherence(time.time())

    def _update_coherence(self, timestamp: float) -> None:
        """Compute current phase coherence (order parameter)."""
        if not self.oscillators:
            return

        phases = [osc.get_phase() for osc in self.oscillators.values()]
        complex_sum = np.sum([np.exp(1j * phase) for phase in phases])
        r = np.abs(complex_sum) / len(phases)

        mean_phase = np.angle(complex_sum)
        phase_variance = np.var(phases)

        if r < 0.30:
            quality = "unconscious"
        elif r < 0.70:
            quality = "preconscious"
        elif r < 0.90:
            quality = "conscious"
        else:
            quality = "deep"

        coherence = PhaseCoherence(
            order_parameter=r,
            mean_phase=mean_phase,
            phase_variance=phase_variance,
            coherence_quality=quality,
            timestamp=timestamp,
        )

        self._coherence_cache = coherence
        self._last_coherence_time = timestamp
        self.dynamics.add_coherence_sample(r, timestamp)

    def get_coherence(self) -> PhaseCoherence | None:
        """Get latest phase coherence measurement."""
        if self._coherence_cache is None and self.oscillators:
            self._update_coherence(time.time())
        return self._coherence_cache

    def get_order_parameter(self) -> float:
        """Quick access to order parameter r."""
        if self._coherence_cache:
            return self._coherence_cache.order_parameter
        return 0.0

    async def synchronize(
        self,
        topology: dict[str, list[str]],
        duration_ms: float = 200.0,
        target_coherence: float = 0.70,
        dt: float = 0.001,  # SINGULARIDADE: 0.001 for 40Hz Gamma (was 0.005)
    ) -> SynchronizationDynamics:
        """Run synchronization protocol for specified duration."""
        start_time = time.time()
        duration_s = duration_ms / 1000.0
        steps = int(duration_s / dt)

        for step in range(steps):
            self.update_network(topology, dt=dt)

            if self._coherence_cache and self._coherence_cache.order_parameter >= target_coherence:
                if self.dynamics.time_to_sync is None:
                    elapsed = time.time() - start_time
                    self.dynamics.time_to_sync = elapsed
                self.dynamics.sustained_duration += dt

            if step % 10 == 0:
                await asyncio.sleep(0)

        return self.dynamics

    def get_phase_distribution(self) -> np.ndarray:
        """Get current phase distribution for visualization."""
        return np.array([osc.get_phase() for osc in self.oscillators.values()])

    # =========================================================================
    # NOESIS ENTROPY AUDIT (2025-12-15): Adaptive Coupling Enhancement
    # Based on Dec 2025 AKOrN (Artificial Kuramoto Oscillatory Neurons) research
    # =========================================================================

    def compute_adaptive_coupling(
        self,
        base_coupling: float,
        current_coherence: float,
        target_coherence: float = 0.70,
        learning_rate: float = 0.1,
        min_coupling: float = 0.1,
        max_coupling: float = 2.0,
    ) -> float:
        """
        Compute adaptive coupling strength based on coherence feedback.
        
        Per Dec 2025 AKOrN research: coupling should increase when coherence
        is below target (to promote synchronization) and decrease when above
        (to prevent over-synchronization and maintain information diversity).
        
        Args:
            base_coupling: Current coupling strength
            current_coherence: Current order parameter r
            target_coherence: Desired coherence level
            learning_rate: How fast to adapt (0.0-1.0)
            min_coupling: Minimum allowed coupling
            max_coupling: Maximum allowed coupling
            
        Returns:
            Adjusted coupling strength
        """
        # Compute coherence error
        coherence_delta = target_coherence - current_coherence
        
        # Adaptive adjustment: positive delta -> increase coupling
        adjustment = base_coupling * coherence_delta * learning_rate
        
        # Apply adjustment with bounds
        new_coupling = base_coupling + adjustment
        new_coupling = max(min_coupling, min(max_coupling, new_coupling))
        
        return new_coupling

    async def synchronize_adaptive(
        self,
        topology: dict[str, list[str]],
        duration_ms: float = 200.0,
        target_coherence: float = 0.70,
        dt: float = 0.001,
        learning_rate: float = 0.05,
    ) -> SynchronizationDynamics:
        """
        Run synchronization with adaptive coupling adjustment.
        
        This enhanced version adjusts coupling strength dynamically based on
        real-time coherence feedback, implementing the AKOrN pattern from
        Dec 2025 research.
        
        Args:
            topology: Network connectivity
            duration_ms: Maximum synchronization duration
            target_coherence: Desired coherence level
            dt: Time step
            learning_rate: Adaptation speed (lower = more stable)
            
        Returns:
            SynchronizationDynamics with timing and success info
        """
        start_time = time.time()
        duration_s = duration_ms / 1000.0
        steps = int(duration_s / dt)
        
        # Track coupling adaptation
        initial_coupling = self.default_config.coupling_strength
        current_coupling = initial_coupling
        coupling_history = [current_coupling]
        
        for step in range(steps):
            # Update network with current coupling
            # Note: coupling_weights could be used for heterogeneous adaptation
            self.update_network(topology, coupling_weights=None, dt=dt)
            
            # Get current coherence
            coherence = self._coherence_cache.order_parameter if self._coherence_cache else 0.0
            
            # Adapt coupling every 10 steps to avoid oscillation
            if step % 10 == 0 and step > 0:
                current_coupling = self.compute_adaptive_coupling(
                    base_coupling=current_coupling,
                    current_coherence=coherence,
                    target_coherence=target_coherence,
                    learning_rate=learning_rate,
                )
                # Update default config for subsequent iterations
                self.default_config.coupling_strength = current_coupling
                coupling_history.append(current_coupling)
            
            # Track synchronization dynamics
            if coherence >= target_coherence:
                if self.dynamics.time_to_sync is None:
                    elapsed = time.time() - start_time
                    self.dynamics.time_to_sync = elapsed
                self.dynamics.sustained_duration += dt
            
            # Yield control periodically
            if step % 10 == 0:
                await asyncio.sleep(0)
        
        # Restore original coupling (let ConfigTuner decide if change is permanent)
        self.default_config.coupling_strength = initial_coupling
        
        # Store adaptation history in dynamics for analysis
        self.dynamics.coupling_history = coupling_history  # type: ignore
        
        return self.dynamics

    def __repr__(self) -> str:
        coherence = self.get_order_parameter()
        return f"KuramotoNetwork(oscillators={len(self.oscillators)}, coherence={coherence:.3f})"

