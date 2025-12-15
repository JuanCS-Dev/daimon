#!/usr/bin/env python3
"""
Kuramoto Synchronization Stability Test - 10 Minutes
=====================================================

Tests that the ESGT/Kuramoto system maintains stable high coherence
over extended periods with multiple ignition events.

Run with:
    PYTHONPATH="backend/services/maximus_core_service/src" python3 \
        backend/services/maximus_core_service/tests/stability/test_kuramoto_stability_10min.py

Or with pytest:
    PYTHONPATH="backend/services/maximus_core_service/src" python3 -m pytest \
        backend/services/maximus_core_service/tests/stability/test_kuramoto_stability_10min.py -v -s

Author: Claude Code
Date: 2024-12-08
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from maximus_core_service.consciousness.esgt.coordinator import (
    ESGTCoordinator,
    TriggerConditions,
)
from maximus_core_service.consciousness.esgt.models import SalienceScore
from maximus_core_service.consciousness.esgt.kuramoto_models import OscillatorConfig
from maximus_core_service.consciousness.tig.fabric import TIGFabric, TopologyConfig


@dataclass
class StabilityMetrics:
    """Tracks stability metrics over the test duration."""

    total_ignitions: int = 0
    successful_ignitions: int = 0
    failed_ignitions: int = 0

    coherence_values: list[float] = field(default_factory=list)
    sync_times_ms: list[float] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    start_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.total_ignitions == 0:
            return 0.0
        return self.successful_ignitions / self.total_ignitions

    @property
    def avg_coherence(self) -> float:
        if not self.coherence_values:
            return 0.0
        return statistics.mean(self.coherence_values)

    @property
    def min_coherence(self) -> float:
        if not self.coherence_values:
            return 0.0
        return min(self.coherence_values)

    @property
    def max_coherence(self) -> float:
        if not self.coherence_values:
            return 0.0
        return max(self.coherence_values)

    @property
    def coherence_std(self) -> float:
        if len(self.coherence_values) < 2:
            return 0.0
        return statistics.stdev(self.coherence_values)

    @property
    def avg_sync_time_ms(self) -> float:
        if not self.sync_times_ms:
            return 0.0
        return statistics.mean(self.sync_times_ms)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def report(self) -> str:
        """Generate stability report."""
        elapsed_min = self.elapsed_seconds / 60

        lines = [
            "",
            "=" * 60,
            "KURAMOTO STABILITY TEST REPORT",
            "=" * 60,
            f"Duration: {elapsed_min:.1f} minutes",
            f"Total Ignitions: {self.total_ignitions}",
            f"Successful: {self.successful_ignitions} ({self.success_rate * 100:.1f}%)",
            f"Failed: {self.failed_ignitions}",
            "",
            "COHERENCE METRICS:",
            f"  Average: {self.avg_coherence:.3f}",
            f"  Min: {self.min_coherence:.3f}",
            f"  Max: {self.max_coherence:.3f}",
            f"  Std Dev: {self.coherence_std:.4f}",
            "",
            "SYNC TIME:",
            f"  Average: {self.avg_sync_time_ms:.1f} ms",
            "",
        ]

        if self.failure_reasons:
            lines.append("FAILURES:")
            # Count unique failure reasons
            from collections import Counter
            reason_counts = Counter(self.failure_reasons)
            for reason, count in reason_counts.most_common(5):
                lines.append(f"  - {reason}: {count}x")

        lines.append("=" * 60)

        # Stability verdict
        if self.success_rate >= 0.95 and self.min_coherence >= 0.70:
            lines.append("VERDICT: STABLE - System meets production requirements")
        elif self.success_rate >= 0.80 and self.avg_coherence >= 0.70:
            lines.append("VERDICT: ACCEPTABLE - Minor instabilities detected")
        else:
            lines.append("VERDICT: UNSTABLE - Requires investigation")

        lines.append("=" * 60)

        return "\n".join(lines)


async def run_stability_test(
    duration_minutes: float = 10.0,
    ignition_interval_seconds: float = 3.0,
    tig_node_count: int = 50,
    target_coherence: float = 0.85,
) -> StabilityMetrics:
    """
    Run stability test for specified duration.

    Args:
        duration_minutes: How long to run the test
        ignition_interval_seconds: Time between ignition attempts
        tig_node_count: Number of TIG nodes (affects sync time)
        target_coherence: Target coherence for ignitions

    Returns:
        StabilityMetrics with test results
    """
    metrics = StabilityMetrics()
    duration_seconds = duration_minutes * 60

    logger.info(f"Starting Kuramoto Stability Test")
    logger.info(f"  Duration: {duration_minutes} minutes")
    logger.info(f"  Ignition interval: {ignition_interval_seconds}s")
    logger.info(f"  TIG nodes: {tig_node_count}")
    logger.info(f"  Target coherence: {target_coherence}")

    # Initialize TIG Fabric
    logger.info("Initializing TIG Fabric...")
    tig_config = TopologyConfig(
        node_count=tig_node_count,
        target_density=0.25
    )
    tig = TIGFabric(tig_config)
    await tig.initialize()
    logger.info(f"TIG ready: {len(tig.nodes)} nodes")

    # Initialize ESGT Coordinator
    triggers = TriggerConditions(
        min_salience=0.5,
        min_available_nodes=10,
        refractory_period_ms=500.0,  # Allow frequent ignitions for testing
    )

    kuramoto_config = OscillatorConfig(
        coupling_strength=60.0,  # High coupling for fast sync
        phase_noise=0.0005,
    )

    esgt = ESGTCoordinator(
        tig_fabric=tig,
        triggers=triggers,
        kuramoto_config=kuramoto_config,
    )
    await esgt.start()
    logger.info(f"ESGT ready: {len(esgt.kuramoto.oscillators)} oscillators")

    # Run test loop
    logger.info("")
    logger.info("=" * 40)
    logger.info("STARTING STABILITY TEST")
    logger.info("=" * 40)

    try:
        while metrics.elapsed_seconds < duration_seconds:
            # Create varying salience scores to test different conditions
            import random
            salience = SalienceScore(
                novelty=0.6 + random.random() * 0.4,
                relevance=0.7 + random.random() * 0.3,
                urgency=0.5 + random.random() * 0.5,
                confidence=0.8 + random.random() * 0.2,
            )

            # Attempt ignition
            event = await esgt.initiate_esgt(
                salience=salience,
                content={"test_iteration": metrics.total_ignitions},
                content_source="stability_test",
                target_coherence=target_coherence,
            )

            metrics.total_ignitions += 1

            if event.was_successful():
                metrics.successful_ignitions += 1
                metrics.coherence_values.append(event.achieved_coherence)
                if event.time_to_sync_ms:
                    metrics.sync_times_ms.append(event.time_to_sync_ms)

                # Progress indicator (every 10 ignitions)
                if metrics.total_ignitions % 10 == 0:
                    elapsed_min = metrics.elapsed_seconds / 60
                    logger.info(
                        f"[{elapsed_min:.1f}m] Ignitions: {metrics.total_ignitions}, "
                        f"Success: {metrics.success_rate * 100:.0f}%, "
                        f"Avg Coherence: {metrics.avg_coherence:.3f}"
                    )
            else:
                metrics.failed_ignitions += 1
                reason = event.failure_reason or "unknown"
                metrics.failure_reasons.append(reason)
                logger.warning(f"Ignition {metrics.total_ignitions} failed: {reason}")

            # Wait before next ignition
            await asyncio.sleep(ignition_interval_seconds)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await esgt.stop()
        await tig.stop()

    return metrics


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("KURAMOTO SYNCHRONIZATION STABILITY TEST")
    print("Testing consciousness coherence stability over 10 minutes")
    print("=" * 60 + "\n")

    # Run test
    metrics = await run_stability_test(
        duration_minutes=10.0,
        ignition_interval_seconds=3.0,
        tig_node_count=50,
        target_coherence=0.85,
    )

    # Print report
    print(metrics.report())

    # Return exit code based on stability
    if metrics.success_rate >= 0.95 and metrics.min_coherence >= 0.70:
        return 0  # Success
    else:
        return 1  # Failure


# Pytest-compatible test function
async def test_kuramoto_stability_10min():
    """
    Pytest test: Verify Kuramoto synchronization is stable for 10 minutes.

    Success criteria:
    - Success rate >= 95%
    - Minimum coherence >= 0.70 (conscious level)
    - Average coherence >= 0.85
    """
    metrics = await run_stability_test(
        duration_minutes=10.0,
        ignition_interval_seconds=3.0,
        tig_node_count=50,
        target_coherence=0.85,
    )

    print(metrics.report())

    assert metrics.success_rate >= 0.95, \
        f"Success rate {metrics.success_rate:.1%} below 95% threshold"

    assert metrics.min_coherence >= 0.70, \
        f"Min coherence {metrics.min_coherence:.3f} below 0.70 (conscious level)"

    assert metrics.avg_coherence >= 0.85, \
        f"Avg coherence {metrics.avg_coherence:.3f} below 0.85 target"


# Quick test (1 minute) for CI/CD
async def test_kuramoto_stability_1min():
    """
    Quick stability test (1 minute) for CI/CD pipelines.
    """
    metrics = await run_stability_test(
        duration_minutes=1.0,
        ignition_interval_seconds=2.0,
        tig_node_count=30,
        target_coherence=0.80,
    )

    print(metrics.report())

    assert metrics.success_rate >= 0.90, \
        f"Success rate {metrics.success_rate:.1%} below 90% threshold"

    assert metrics.min_coherence >= 0.70, \
        f"Min coherence {metrics.min_coherence:.3f} below 0.70"


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
