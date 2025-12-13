#!/usr/bin/env python3
"""
DAIMON AFK Watcher - Inactivity Detection
==========================================

Detects when user is away from keyboard (AFK).
Uses input events from InputWatcher or X11 screensaver extension.

Privacy-preserving: only tracks presence/absence, not activity content.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .base import BaseWatcher, Heartbeat
from .registry import register_collector
from memory.activity_store import get_activity_store
from learners import get_style_learner

logger = logging.getLogger("daimon.afk")

# Configuration
DEFAULT_AFK_THRESHOLD = 180.0  # 3 minutes of inactivity = AFK
POLL_INTERVAL = 5.0  # Check every 5 seconds


@dataclass
class AFKState:
    """
    AFK state information.

    Attributes:
        is_afk: True if user is currently AFK.
        afk_since: When AFK period started (None if not AFK).
        last_activity: Timestamp of last detected activity.
        afk_duration: Total seconds AFK (if currently AFK).
    """
    is_afk: bool = False
    afk_since: Optional[datetime] = None
    last_activity: datetime = None
    afk_duration: float = 0.0

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "is_afk": self.is_afk,
            "afk_since": self.afk_since.isoformat() if self.afk_since else None,
            "last_activity": self.last_activity.isoformat(),
            "afk_duration": round(self.afk_duration, 1),
        }


class X11IdleDetector:
    """
    Detect idle time using X11 screensaver extension.

    Uses XScreenSaver extension to get idle time in milliseconds.
    This is the same method ActivityWatch uses.
    """

    def __init__(self):
        self._display = None
        self._root = None
        self._available = False

    def _ensure_connected(self) -> bool:
        """Initialize X11 connection if needed."""
        if self._display is not None:
            return self._available

        try:
            from Xlib import display  # pylint: disable=import-outside-toplevel

            self._display = display.Display()
            self._root = self._display.screen().root

            # Check if screensaver extension is available
            if not self._display.has_extension("MIT-SCREEN-SAVER"):
                logger.warning("X11 screensaver extension not available")
                self._available = False
                return False

            self._available = True
            logger.debug("X11 idle detector initialized")
            return True

        except ImportError:
            logger.error("python-xlib not installed. Run: pip install python-xlib")
            self._available = False
            return False
        except (OSError, IOError, ValueError) as e:
            logger.error("X11 idle detector init failed: %s", e)
            self._available = False
            return False

    def get_idle_seconds(self) -> Optional[float]:
        """
        Get idle time in seconds.

        Returns:
            Seconds since last user input, or None if unavailable.
        """
        if not self._ensure_connected():
            return None

        try:
            # pylint: disable=import-outside-toplevel
            from Xlib.ext import screensaver

            info = screensaver.query_info(self._root)
            idle_ms = info.idle
            return idle_ms / 1000.0

        except (OSError, IOError, AttributeError) as e:
            logger.debug("Error getting idle time: %s", e)
            return None

    def close(self) -> None:
        """Close X11 connection."""
        if self._display:
            try:
                self._display.close()
            except (OSError, AttributeError):
                pass
            self._display = None
            self._root = None


class FallbackIdleDetector:
    """
    Fallback idle detector using /proc/interrupts.

    Less accurate but works without X11 extensions.
    Monitors keyboard/mouse interrupts.
    """

    def __init__(self) -> None:
        self._last_interrupts = 0
        self._last_check = time.time()
        self._idle_start = time.time()

    def get_idle_seconds(self) -> Optional[float]:
        """
        Estimate idle time from interrupt counts.

        Returns:
            Estimated seconds since last input, or None if unavailable.
        """
        try:
            # Read interrupt counts for keyboard and mouse
            with open("/proc/interrupts", "r", encoding="utf-8") as f:
                content = f.read()

            # Sum up interrupts (rough approximation)
            total = self._parse_interrupts(content)
            now = time.time()

            # If interrupts changed, user is active
            if total != self._last_interrupts:
                self._last_interrupts = total
                self._idle_start = now

            return now - self._idle_start

        except (OSError, IOError) as e:
            logger.debug("Fallback idle detection failed: %s", e)
            return None

    def _parse_interrupts(self, content: str) -> int:
        """Parse interrupt counts from /proc/interrupts content."""
        total = 0
        for line in content.split("\n"):
            # Look for i8042 (keyboard/mouse controller) or USB HID
            if "i8042" in line or "ehci" in line or "xhci" in line:
                parts = line.split()
                for part in parts[1:]:
                    try:
                        total += int(part)
                    except ValueError:
                        break
        return total

    def reset(self) -> None:
        """Reset idle tracking state."""
        self._last_interrupts = 0
        self._idle_start = time.time()


@register_collector
class AFKWatcher(BaseWatcher):
    """
    Watcher for AFK (Away From Keyboard) detection.

    Detects when user becomes inactive and tracks AFK periods.
    Uses X11 screensaver extension for accurate idle detection.

    Attributes:
        name: "afk_watcher"
        version: "1.0.0"
    """

    name = "afk_watcher"
    version = "1.0.0"

    def __init__(
        self,
        batch_interval: float = 60.0,
        afk_threshold: float = DEFAULT_AFK_THRESHOLD,
    ):
        """
        Initialize AFK watcher.

        Args:
            batch_interval: Seconds between batch flushes.
            afk_threshold: Seconds of inactivity before marking as AFK.
        """
        super().__init__(batch_interval=batch_interval)
        self._afk_threshold = afk_threshold
        self._poll_interval = POLL_INTERVAL

        # State
        self._state = AFKState()
        self._last_state_is_afk = False  # Track state changes

        # Detectors (try X11 first, fallback to /proc)
        self._x11_detector: Optional[X11IdleDetector] = None
        self._fallback_detector: Optional[FallbackIdleDetector] = None
        self._using_x11 = False

    def _ensure_detector(self) -> bool:
        """Initialize idle detector."""
        if self._x11_detector is not None or self._fallback_detector is not None:
            return True

        # Try X11 first
        self._x11_detector = X11IdleDetector()
        if self._x11_detector.get_idle_seconds() is not None:
            self._using_x11 = True
            logger.info("Using X11 screensaver extension for idle detection")
            return True

        # Fallback to /proc/interrupts
        logger.info("X11 unavailable, using /proc/interrupts fallback")
        self._fallback_detector = FallbackIdleDetector()
        self._using_x11 = False
        return True

    def _get_idle_seconds(self) -> float:
        """Get current idle time from active detector."""
        if self._using_x11 and self._x11_detector:
            idle = self._x11_detector.get_idle_seconds()
            if idle is not None:
                return idle

        if self._fallback_detector:
            idle = self._fallback_detector.get_idle_seconds()
            if idle is not None:
                return idle

        return 0.0

    async def collect(self) -> Optional[Heartbeat]:
        """
        Collect AFK state.

        Returns:
            Heartbeat on state changes, or None if unchanged.
        """
        if not self._ensure_detector():
            return None

        now = datetime.now()
        idle_seconds = self._get_idle_seconds()

        # Determine if currently AFK
        is_afk = idle_seconds >= self._afk_threshold

        # Update state
        if is_afk:
            if not self._state.is_afk:
                # Transition to AFK
                self._state.is_afk = True
                self._state.afk_since = now - timedelta(seconds=idle_seconds)
            self._state.afk_duration = idle_seconds
        else:
            if self._state.is_afk:
                # Transition from AFK to active
                self._state.is_afk = False
                self._state.afk_since = None
                self._state.afk_duration = 0.0
            self._state.last_activity = now

        # Only emit on state changes
        if is_afk != self._last_state_is_afk:
            self._last_state_is_afk = is_afk

            return Heartbeat(
                timestamp=now,
                watcher_type=self.name,
                data={
                    "event": "afk_start" if is_afk else "afk_end",
                    "state": self._state.to_dict(),
                    "idle_seconds": round(idle_seconds, 1),
                },
            )

        return None

    def _should_merge(self, heartbeat: Heartbeat) -> bool:
        """Never merge AFK heartbeats - each state change is significant."""
        return False

    def get_collection_interval(self) -> float:
        """Return poll interval."""
        return self._poll_interval

    def get_config(self) -> Dict[str, Any]:
        """Return watcher configuration."""
        return {
            "batch_interval": self.batch_interval,
            "afk_threshold": self._afk_threshold,
            "poll_interval": self._poll_interval,
            "using_x11": self._using_x11,
            "current_state": self._state.to_dict(),
        }

    def get_current_state(self) -> AFKState:
        """Get current AFK state."""
        return self._state

    async def flush(self) -> list:
        """Flush heartbeats to ActivityStore and StyleLearner."""
        flushed = await super().flush()
        if not flushed:
            return flushed

        # Store in ActivityStore
        try:
            store = get_activity_store()
            for hb in flushed:
                store.add(watcher_type="afk", timestamp=hb.timestamp, data=hb.data)
        except Exception as e:
            logger.warning("Failed to store afk data: %s", e)

        # Feed StyleLearner
        try:
            learner = get_style_learner()
            for hb in flushed:
                learner.add_afk_sample(hb.data)
        except Exception as e:
            logger.debug("StyleLearner feed failed: %s", e)

        return flushed

    async def stop(self) -> None:
        """Stop watcher and clean up detectors."""
        await super().stop()

        if self._x11_detector:
            self._x11_detector.close()
            self._x11_detector = None

        self._fallback_detector = None
