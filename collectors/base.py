#!/usr/bin/env python3
"""
DAIMON Collector Base - Abstract Interface for Watchers
========================================================

Defines the base interface that all DAIMON collectors must implement.
Uses heartbeat pattern (ActivityWatch style) for efficient data collection.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("daimon.collectors")


@dataclass
class Heartbeat:
    """
    Base heartbeat structure for all collectors.

    Heartbeats represent state snapshots, not isolated events.
    They merge when similar, reducing data volume.

    Attributes:
        timestamp: When the heartbeat was captured.
        watcher_type: Identifier of the collector that generated it.
        data: Watcher-specific payload.
    """
    timestamp: datetime
    watcher_type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "watcher_type": self.watcher_type,
            "data": self.data,
        }


class BaseWatcher(ABC):
    """
    Abstract base class for all DAIMON watchers.

    Implements heartbeat pattern with configurable batch intervals.
    Subclasses must implement collect() and get_config().

    Attributes:
        name: Unique identifier for this watcher type.
        version: Semantic version of the watcher.
        batch_interval: Seconds between automatic flushes.
    """

    name: str = "base"
    version: str = "1.0.0"

    def __init__(self, batch_interval: float = 30.0):
        """
        Initialize watcher with configuration.

        Args:
            batch_interval: Seconds between batch flushes (default 30s).
        """
        self.batch_interval = batch_interval
        self.running = False
        self.heartbeats: List[Heartbeat] = []
        self._task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._last_heartbeat: Optional[Heartbeat] = None

    @abstractmethod
    async def collect(self) -> Optional[Heartbeat]:
        """
        Collect a single heartbeat.

        Returns None if no significant change since last collection.
        Subclasses implement watcher-specific collection logic.

        Returns:
            Heartbeat with current state, or None if unchanged.
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return watcher configuration for introspection.

        Returns:
            Dict with configuration values.
        """

    async def start(self) -> None:
        """
        Start the watcher collection loop.

        Spawns background tasks for collection and periodic flushing.
        """
        if self.running:
            logger.warning("%s already running", self.name)
            return

        self.running = True
        logger.info("Starting %s v%s", self.name, self.version)

        # Start collection and flush loops
        self._task = asyncio.create_task(self._collection_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Wait for both tasks
        try:
            await asyncio.gather(self._task, self._flush_task)
        except asyncio.CancelledError:
            logger.info("%s stopped", self.name)

    async def stop(self) -> None:
        """Stop the watcher and flush remaining heartbeats."""
        logger.info("Stopping %s", self.name)
        self.running = False

        # Cancel tasks
        for task in [self._task, self._flush_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Final flush
        await self.flush()

    async def _collection_loop(self) -> None:
        """
        Main collection loop.

        Calls collect() at regular intervals and stores heartbeats.
        Override get_collection_interval() to customize timing.
        """
        while self.running:
            try:
                heartbeat = await self.collect()
                if heartbeat:
                    self._add_heartbeat(heartbeat)
            except (OSError, ValueError, RuntimeError, AttributeError) as e:
                logger.error("%s collection error: %s", self.name, e)

            await asyncio.sleep(self.get_collection_interval())

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self.running:
            await asyncio.sleep(self.batch_interval)
            await self.flush()

    def _add_heartbeat(self, heartbeat: Heartbeat) -> None:
        """
        Add heartbeat to buffer, merging if similar to last.

        Args:
            heartbeat: New heartbeat to add.
        """
        # Check if should merge with last heartbeat
        if self._should_merge(heartbeat):
            # Update timestamp of last heartbeat instead of adding new
            if self._last_heartbeat:
                self._last_heartbeat.timestamp = heartbeat.timestamp
            return

        self.heartbeats.append(heartbeat)
        self._last_heartbeat = heartbeat

        # Prevent unbounded growth
        if len(self.heartbeats) > 1000:
            self.heartbeats = self.heartbeats[-500:]
            logger.warning("%s buffer overflow, trimmed to 500", self.name)

    def _should_merge(self, heartbeat: Heartbeat) -> bool:
        """
        Check if heartbeat should merge with previous.

        Override in subclasses for custom merge logic.
        Default: merge if data is identical.

        Args:
            heartbeat: New heartbeat to check.

        Returns:
            True if should merge, False to add as new.
        """
        if not self._last_heartbeat:
            return False
        return self._last_heartbeat.data == heartbeat.data

    async def flush(self) -> List[Heartbeat]:
        """
        Flush pending heartbeats.

        Override in subclasses to send to specific endpoints.
        Default implementation just clears buffer and returns heartbeats.

        Returns:
            List of flushed heartbeats.
        """
        if not self.heartbeats:
            return []

        flushed = self.heartbeats.copy()
        self.heartbeats.clear()

        logger.debug("%s flushed %d heartbeats", self.name, len(flushed))
        return flushed

    def get_collection_interval(self) -> float:
        """
        Get interval between collections in seconds.

        Override for custom intervals. Default is 1 second.

        Returns:
            Collection interval in seconds.
        """
        return 1.0

    def get_status(self) -> Dict[str, Any]:
        """
        Get current watcher status.

        Returns:
            Dict with status information.
        """
        return {
            "name": self.name,
            "version": self.version,
            "running": self.running,
            "pending_heartbeats": len(self.heartbeats),
            "config": self.get_config(),
        }
