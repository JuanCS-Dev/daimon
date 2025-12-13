"""
DAIMON → NOESIS Data Ingestion Service
======================================

Bridges DAIMON telemetry with NOESIS consciousness.

Converts raw behavioral data from ActivityStore into structured signals
that NOESIS can process for learning and consciousness emergence.

Pipeline:
    ActivityStore → aggregate_to_signals() → filter(salience≥0.5) → NOESIS

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from memory.activity_store import ActivityRecord, get_activity_store
from integrations.mcp_tools.http_utils import http_post
from integrations.mcp_tools.config import NOESIS_CONSCIOUSNESS_URL

logger = logging.getLogger("daimon.ingestion")


@dataclass
class BehavioralSignal:
    """
    A signal derived from user behavior for NOESIS processing.

    Attributes:
        signal_type: Category of signal (cognitive_state, preference, pattern, anomaly)
        source: Which watcher generated this signal
        timestamp: When the signal was detected
        salience: Importance 0.0-1.0 (higher = more significant)
        data: Signal payload
        context: Human-readable description
    """
    signal_type: str
    source: str
    timestamp: datetime
    salience: float
    data: Dict[str, Any]
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "signal_type": self.signal_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "salience": self.salience,
            "data": self.data,
            "context": self.context,
        }


@dataclass
class IngestionStats:
    """Statistics for monitoring ingestion service."""
    cycles_completed: int = 0
    signals_generated: int = 0
    signals_sent: int = 0
    noesis_failures: int = 0
    last_cycle: Optional[datetime] = None


class DataIngestionService:
    """
    Service that periodically processes DAIMON data and sends to NOESIS.

    Responsibilities:
    1. Aggregate raw heartbeats into meaningful signals
    2. Calculate salience (importance) of each signal
    3. Send high-salience signals to NOESIS consciousness
    4. Store episodes in NOESIS episodic memory

    Features:
    - Graceful degradation when NOESIS offline
    - Configurable salience threshold
    - Batching for efficiency
    """

    def __init__(
        self,
        ingestion_interval_seconds: int = 60,
        salience_threshold: float = 0.5,
        batch_size: int = 50,
    ):
        """
        Initialize the ingestion service.

        Args:
            ingestion_interval_seconds: Seconds between processing cycles
            salience_threshold: Minimum salience to send to NOESIS (0.0-1.0)
            batch_size: Max activities to process per cycle
        """
        self.interval = ingestion_interval_seconds
        self.salience_threshold = salience_threshold
        self.batch_size = batch_size
        self._running = False
        self._last_processed: Optional[datetime] = None
        self.stats = IngestionStats()

    async def start(self) -> None:
        """Start the ingestion loop."""
        self._running = True
        logger.info(
            "DataIngestionService started (interval=%ds, threshold=%.1f)",
            self.interval, self.salience_threshold
        )

        while self._running:
            try:
                await self._process_cycle()
            except Exception as e:
                logger.error("Ingestion cycle failed: %s", e)

            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        """Stop the ingestion loop."""
        self._running = False
        logger.info("DataIngestionService stopped")

    async def _process_cycle(self) -> None:
        """Single processing cycle: query → aggregate → filter → send."""
        store = get_activity_store()

        # Get activities since last processed
        since = self._last_processed or (datetime.now() - timedelta(minutes=5))
        activities = store.query(
            start_time=since,
            limit=self.batch_size * 2,
        )

        if not activities:
            logger.debug("No new activities to process")
            return

        # Aggregate into signals
        signals = self._aggregate_to_signals(activities)
        self.stats.signals_generated += len(signals)

        # Filter by salience
        salient_signals = [s for s in signals if s.salience >= self.salience_threshold]

        if salient_signals:
            sent = await self._send_to_noesis(salient_signals)
            self.stats.signals_sent += sent
            logger.info(
                "Processed %d activities → %d signals → %d sent to NOESIS",
                len(activities), len(signals), sent
            )

        self._last_processed = datetime.now()
        self.stats.cycles_completed += 1
        self.stats.last_cycle = datetime.now()

    def _aggregate_to_signals(
        self, activities: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Convert raw activities into behavioral signals."""
        signals: List[BehavioralSignal] = []

        # Group by watcher type
        by_type: Dict[str, List[ActivityRecord]] = {}
        for act in activities:
            by_type.setdefault(act.watcher_type, []).append(act)

        # Process each type
        for watcher_type, acts in by_type.items():
            if watcher_type in ("claude", "claude_watcher"):
                signals.extend(self._process_claude_activities(acts))
            elif watcher_type in ("shell", "shell_watcher"):
                signals.extend(self._process_shell_activities(acts))
            elif watcher_type in ("input", "input_watcher"):
                signals.extend(self._process_input_activities(acts))
            elif watcher_type in ("window", "window_watcher"):
                signals.extend(self._process_window_activities(acts))
            elif watcher_type in ("afk", "afk_watcher"):
                signals.extend(self._process_afk_activities(acts))

        return signals

    def _process_claude_activities(
        self, acts: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Process Claude Code session activities."""
        signals: List[BehavioralSignal] = []

        for act in acts:
            data = act.data

            # High salience for approvals/rejections
            signal_type = data.get("signal_type")
            if signal_type in ("approval", "rejection"):
                salience = 0.8 if signal_type == "rejection" else 0.6
                signals.append(BehavioralSignal(
                    signal_type="preference",
                    source="claude",
                    timestamp=act.timestamp,
                    salience=salience,
                    data=data,
                    context=f"User {signal_type}: {data.get('category', 'unknown')}",
                ))

            # Session duration patterns
            duration = data.get("session_duration_seconds", 0)
            if duration > 3600:  # 1+ hour session
                signals.append(BehavioralSignal(
                    signal_type="pattern",
                    source="claude",
                    timestamp=act.timestamp,
                    salience=0.5,
                    data={"duration_seconds": duration},
                    context=f"Long coding session: {duration // 60} minutes",
                ))

        return signals

    def _process_shell_activities(
        self, acts: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Process shell command activities."""
        signals: List[BehavioralSignal] = []

        for act in acts:
            data = act.data
            cmd = data.get("command", "")
            exit_code = data.get("exit_code", 0)

            # High salience for failed commands
            if exit_code != 0:
                signals.append(BehavioralSignal(
                    signal_type="anomaly",
                    source="shell",
                    timestamp=act.timestamp,
                    salience=0.7,
                    data=data,
                    context=f"Command failed (exit {exit_code}): {cmd[:50]}",
                ))

            # Very high salience for risky commands
            cmd_lower = cmd.lower()
            risky_patterns = ["rm -rf", "drop ", "delete ", "truncate ", "format "]
            if any(p in cmd_lower for p in risky_patterns):
                signals.append(BehavioralSignal(
                    signal_type="anomaly",
                    source="shell",
                    timestamp=act.timestamp,
                    salience=0.9,
                    data=data,
                    context=f"Risky command detected: {cmd[:50]}",
                ))

        return signals

    def _process_input_activities(
        self, acts: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Process keyboard input activities."""
        signals: List[BehavioralSignal] = []

        if not acts:
            return signals

        # Aggregate typing dynamics
        total_keys = sum(a.data.get("key_count", 0) for a in acts)

        if total_keys < 50:  # Not enough data
            return signals

        # Calculate WPM variance
        wpms = [a.data.get("wpm", 0) for a in acts if a.data.get("wpm", 0) > 0]
        if len(wpms) >= 3:
            avg_wpm = sum(wpms) / len(wpms)
            variance = sum((w - avg_wpm) ** 2 for w in wpms) / len(wpms)

            # High variance indicates cognitive state changes
            if variance > 100:
                signals.append(BehavioralSignal(
                    signal_type="cognitive_state",
                    source="input",
                    timestamp=acts[-1].timestamp,
                    salience=0.6,
                    data={"avg_wpm": round(avg_wpm, 1), "variance": round(variance, 1)},
                    context=f"Typing rhythm variance detected (WPM var: {variance:.0f})",
                ))

        return signals

    def _process_window_activities(
        self, acts: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Process window focus activities."""
        signals: List[BehavioralSignal] = []

        if not acts:
            return signals

        # Detect context switching
        unique_apps = set(
            a.data.get("app_name", "")
            for a in acts
            if a.data.get("app_name")
        )

        if len(unique_apps) > 5:  # High context switching
            signals.append(BehavioralSignal(
                signal_type="pattern",
                source="window",
                timestamp=acts[-1].timestamp,
                salience=0.5,
                data={
                    "app_count": len(unique_apps),
                    "apps": list(unique_apps)[:10],
                },
                context=f"High context switching: {len(unique_apps)} apps",
            ))

        return signals

    def _process_afk_activities(
        self, acts: List[ActivityRecord]
    ) -> List[BehavioralSignal]:
        """Process AFK (away from keyboard) activities."""
        signals: List[BehavioralSignal] = []

        for act in acts:
            data = act.data
            duration = data.get("afk_duration_seconds", 0)

            # Return from long AFK
            if duration > 1800:  # 30+ minutes
                signals.append(BehavioralSignal(
                    signal_type="pattern",
                    source="afk",
                    timestamp=act.timestamp,
                    salience=0.4,
                    data=data,
                    context=f"Returned after {duration // 60} minute break",
                ))

        return signals

    async def _send_to_noesis(
        self, signals: List[BehavioralSignal]
    ) -> int:
        """
        Send behavioral signals to NOESIS consciousness.

        Returns number of signals successfully sent.
        """
        sent_count = 0

        for signal in signals:
            payload = signal.to_dict()

            try:
                # Send to consciousness for potential emergence
                if signal.salience >= 0.7:
                    result = await http_post(
                        f"{NOESIS_CONSCIOUSNESS_URL}/v1/consciousness/ingest",
                        payload,
                        timeout=5.0,
                    )
                    if "error" not in result:
                        sent_count += 1
                        logger.debug("Signal sent to consciousness: %s", signal.context)
                    else:
                        self.stats.noesis_failures += 1

                # Store in episodic memory (all signals)
                await http_post(
                    f"{NOESIS_CONSCIOUSNESS_URL}/v1/memory/episode",
                    {
                        "type": "behavioral",
                        "content": signal.context,
                        "metadata": payload,
                    },
                    timeout=5.0,
                )
                sent_count += 1

            except Exception as e:
                self.stats.noesis_failures += 1
                logger.debug("Failed to send signal to NOESIS: %s", e)

        return sent_count

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "running": self._running,
            "interval_seconds": self.interval,
            "salience_threshold": self.salience_threshold,
            "cycles_completed": self.stats.cycles_completed,
            "signals_generated": self.stats.signals_generated,
            "signals_sent": self.stats.signals_sent,
            "noesis_failures": self.stats.noesis_failures,
            "last_cycle": (
                self.stats.last_cycle.isoformat()
                if self.stats.last_cycle else None
            ),
        }


# Singleton
_ingestion_service: Optional[DataIngestionService] = None


def get_ingestion_service() -> DataIngestionService:
    """Get singleton ingestion service."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DataIngestionService()
    return _ingestion_service


def reset_ingestion_service() -> None:
    """Reset singleton (for testing)."""
    global _ingestion_service
    if _ingestion_service:
        _ingestion_service.stop()
    _ingestion_service = None


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DAIMON Data Ingestion Service - Manual Test")
    print("=" * 60)

    # Check ActivityStore
    store = get_activity_store()
    stats = store.get_stats()
    print(f"\nActivityStore: {stats['total_records']} records")
    print(f"Watchers: {stats['watchers']}")

    if stats['total_records'] == 0:
        print("\nNo activity data to process. Run collectors first.")
        sys.exit(0)

    # Create service
    service = DataIngestionService(
        ingestion_interval_seconds=10,
        salience_threshold=0.3,  # Lower for testing
    )

    # Process once
    async def test_cycle():
        await service._process_cycle()
        print(f"\nStats: {service.get_stats()}")

    asyncio.run(test_cycle())
    print("\nTest completed!")
