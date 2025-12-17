"""
Noesis Event Bus - Inline Copy (To resolve import hell)
==================================================

An in-process async event bus for efficient inter-module communication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels for processing order."""
    EMERGENCY = 0   # Safety, HITL override
    HIGH = 1        # Consciousness events
    NORMAL = 2      # Standard events
    LOW = 3         # Background/logging


@dataclass
class Event:
    """A signal propagating through Noesis."""
    topic: str
    payload: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: f"evt_{time.time_ns()}")

    def __repr__(self) -> str:
        return f"Event({self.topic}, priority={self.priority.name}, source={self.source})"


EventHandler = Callable[[Event], Any]


@dataclass
class BusMetrics:
    """Metrics container for the event bus."""
    published: int = 0
    delivered: int = 0
    total_latency_ms: float = 0.0


class NoesisEventBus:
    """
    Centralized async event bus for signal propagation.
    """

    def __init__(self, max_queue_size: int = 10000):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._running: bool = False
        self._dead_letters: List[Event] = []
        self._max_dead_letters: int = 100

        # Metrics
        self.metrics = BusMetrics()

    async def start(self) -> None:
        """Start the event bus processor."""
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._process_events())
        logger.info("[EventBus] Started signal propagation processor")

    async def stop(self) -> None:
        """Stop the event bus processor."""
        self._running = False
        logger.info("[EventBus] Stopped")

    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe a handler to a topic."""
        self._handlers[topic].append(handler)
        logger.debug("[EventBus] Subscribed handler to topic: %s", topic)

    async def unsubscribe(self, topic: str, handler: EventHandler) -> bool:
        """Unsubscribe a handler from a topic."""
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)
            return True
        return False

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "unknown"
    ) -> Event:
        """Publish an event to the bus."""
        event = Event(
            topic=topic,
            payload=payload,
            priority=priority,
            source=source
        )

        await self._queue.put((priority.value, event.timestamp, event))
        self.metrics.published += 1

        return event

    async def _process_events(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                try:
                    _, _, event = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                start_time = time.time()
                handlers = self._get_matching_handlers(event.topic)

                tasks = [
                    self._safe_execute_handler(handler, event)
                    for handler in handlers
                ]
                if tasks:
                    await asyncio.gather(*tasks)

                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_latency_ms += latency_ms
                self.metrics.delivered += 1

                if latency_ms > 100:
                    logger.warning(
                        "[EventBus] Slow event: %s took %.2fms",
                        event.topic, latency_ms
                    )

            except Exception as e:
                logger.error("[EventBus] Processing error: %s", e)

    def _get_matching_handlers(self, topic: str) -> List[EventHandler]:
        """Find handlers matching the topic (including wildcards)."""
        handlers = []

        if topic in self._handlers:
            handlers.extend(self._handlers[topic])

        for pattern, pattern_handlers in self._handlers.items():
            if "*" in pattern and self._matches_pattern(pattern, topic):
                handlers.extend(pattern_handlers)

        return handlers

    @staticmethod
    def _matches_pattern(pattern: str, topic: str) -> bool:
        """Check if topic matches a wildcard pattern."""
        if "*" not in pattern:
            return pattern == topic

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")

        return False

    async def _safe_execute_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute a handler with error handling."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(
                "[EventBus] Handler failed for %s: %s",
                event.topic, e
            )
            self._add_to_dead_letters(event)

    def _add_to_dead_letters(self, event: Event) -> None:
        """Add failed event to dead letter queue."""
        self._dead_letters.append(event)
        if len(self._dead_letters) > self._max_dead_letters:
            self._dead_letters.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        avg_latency = (
            self.metrics.total_latency_ms / self.metrics.delivered
            if self.metrics.delivered > 0 else 0
        )
        return {
            "events_published": self.metrics.published,
            "events_delivered": self.metrics.delivered,
            "avg_latency_ms": avg_latency,
            "dead_letters": len(self._dead_letters),
            "queue_size": self._queue.qsize(),
            "topics_subscribed": len(self._handlers),
        }


# Singleton instance for global access
_global_bus: Optional[NoesisEventBus] = None


def get_event_bus() -> NoesisEventBus:
    """Get or create the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = NoesisEventBus()
    return _global_bus
