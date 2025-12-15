"""Trigger Validation Mixin - ESGT trigger checks and node recruitment."""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SalienceScore
    from .coordinator import ESGTCoordinator


class TriggerValidationMixin:
    """Mixin providing trigger validation and node recruitment for ESGT."""

    async def _check_triggers(
        self: "ESGTCoordinator", salience: "SalienceScore"
    ) -> tuple[bool, str]:
        """Check if all trigger conditions are met. Returns (success, failure_reason)."""
        # Salience check
        if not self.triggers.check_salience(salience):
            return (
                False,
                f"Salience too low "
                f"({salience.compute_total():.2f} < {self.triggers.min_salience:.2f})",
            )

        # Resource check
        tig_metrics = self.tig.get_metrics()
        tig_latency = tig_metrics.avg_latency_us / 1000.0  # Convert to ms
        available_nodes = sum(
            1
            for node in self.tig.nodes.values()
            if node.node_state.value in ["active", "esgt_mode"]
        )
        cpu_capacity = 0.60  # Simulated - would query actual metrics

        if not self.triggers.check_resources(
            tig_latency_ms=tig_latency,
            available_nodes=available_nodes,
            cpu_capacity=cpu_capacity,
        ):
            return (
                False,
                f"Insufficient resources "
                f"(nodes={available_nodes}, latency={tig_latency:.1f}ms)",
            )

        # Temporal gating
        time_since_last = (
            time.time() - self.last_esgt_time if self.last_esgt_time > 0 else float("inf")
        )
        recent_count = sum(
            1 for e in self.event_history[-10:] if time.time() - e.timestamp_start < 1.0
        )

        if not self.triggers.check_temporal_gating(time_since_last, recent_count):
            return (
                False,
                f"Refractory period violation "
                f"(time_since_last={time_since_last * 1000:.1f}ms < "
                f"{self.triggers.refractory_period_ms:.1f}ms)",
            )

        # Arousal check (simulated - would query MCEA)
        arousal = 0.70  # Simulated
        if not self.triggers.check_arousal(arousal):
            return (
                False,
                f"Arousal too low ({arousal:.2f} < {self.triggers.min_arousal_level:.2f})",
            )

        return True, ""

    async def _recruit_nodes(
        self: "ESGTCoordinator", content: dict[str, Any]
    ) -> set[str]:
        """
        Recruit participating nodes for ESGT.

        Selection based on:
        - Relevance to content
        - Current load
        - Connectivity quality

        SINGULARIDADE: Use Kuramoto oscillators as source of truth.
        TIG nodes may be marked "dead" by health monitoring, but
        oscillators are the actual computational substrate for ESGT.
        """
        # SINGULARIDADE: Recruit all nodes that have initialized oscillators
        # This bypasses TIG health status which may falsely mark nodes as dead
        if self.kuramoto.oscillators:
            return set(self.kuramoto.oscillators.keys())

        # Fallback: use TIG nodes if Kuramoto not initialized
        recruited: set[str] = set()
        for node_id, node in self.tig.nodes.items():
            if node.node_state.value in ["active", "esgt_mode", "idle"]:
                recruited.add(node_id)

        return recruited

    def _build_topology(
        self: "ESGTCoordinator", node_ids: set[str]
    ) -> dict[str, list[str]]:
        """Build connectivity topology for Kuramoto network.

        SINGULARIDADE: If TIG topology is unavailable or sparse,
        falls back to fully-connected topology for guaranteed sync.
        """
        import logging
        logger = logging.getLogger(__name__)

        topology: dict[str, list[str]] = {}
        total_connections = 0

        for node_id in node_ids:
            node = self.tig.nodes.get(node_id) if self.tig else None
            if node:
                # Get neighbors that are also participating
                neighbors = [
                    conn.remote_node_id
                    for conn in node.connections.values()
                    if conn.active and conn.remote_node_id in node_ids
                ]
                topology[node_id] = neighbors
                total_connections += len(neighbors)

        # SINGULARIDADE: Debug logging for topology issues
        logger.debug(
            f"[ESGT TOPOLOGY] node_ids={len(node_ids)}, "
            f"tig_nodes={len(self.tig.nodes) if self.tig else 0}, "
            f"topology_size={len(topology)}, "
            f"total_connections={total_connections}"
        )

        # SINGULARIDADE: Fallback to well-connected topology if sparse
        # This ensures Kuramoto sync can achieve high coherence even if
        # TIG connections are not established yet
        avg_degree = total_connections / len(node_ids) if node_ids else 0
        if avg_degree < 3.0:  # Less than 3 neighbors on average = sparse
            logger.warning(
                f"[ESGT TOPOLOGY] Sparse topology detected (avg_degree={avg_degree:.1f}). "
                "Using small-world fallback for guaranteed sync."
            )
            # Build small-world topology: ring with extra connections
            # Each node connects to K nearest neighbors + random long-range links
            node_list = list(node_ids)
            n = len(node_list)

            if n <= 20:
                # Small network: use fully-connected
                for node_id in node_list:
                    topology[node_id] = [nid for nid in node_list if nid != node_id]
            else:
                # Larger network: ring lattice with K=10 neighbors each side
                # Plus random shortcuts for small-world property
                import random
                k_neighbors = min(10, n // 4)  # Connect to 10 nearest on each side
                for i, node_id in enumerate(node_list):
                    neighbors = []
                    # Ring neighbors
                    for j in range(1, k_neighbors + 1):
                        neighbors.append(node_list[(i + j) % n])
                        neighbors.append(node_list[(i - j) % n])
                    # Random long-range connections (for small-world property)
                    for _ in range(3):
                        rand_idx = random.randint(0, n - 1)
                        if node_list[rand_idx] != node_id:
                            neighbors.append(node_list[rand_idx])
                    topology[node_id] = list(set(neighbors))  # Remove duplicates

            total_edges = sum(len(v) for v in topology.values())
            logger.info(
                f"[ESGT TOPOLOGY] Small-world fallback: {n} nodes, "
                f"~{total_edges // n} avg neighbors, {total_edges} total connections"
            )

        return topology
