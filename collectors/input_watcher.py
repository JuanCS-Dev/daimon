#!/usr/bin/env python3
"""
DAIMON Input Watcher - Keystroke Dynamics & Mouse Activity
==========================================================

Captures typing patterns (timing) and mouse activity metrics.
Privacy-preserving: captures TIMING, not CONTENT. Never logs which keys.

Keystroke dynamics can reveal:
- Typing speed and rhythm
- Work patterns (bursts vs steady)
- Potential fatigue or stress indicators

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Optional

from .base import BaseWatcher, Heartbeat
from .registry import register_collector
from memory.activity_store import get_activity_store
from learners import get_style_learner

logger = logging.getLogger("daimon.input")

# Configuration
THROTTLE_MS = 50  # Minimum ms between key events
BUFFER_SIZE = 200  # Max events in timing buffers
AGGREGATION_INTERVAL = 60.0  # Seconds between metric aggregation


@dataclass
class KeystrokeDynamics:
    """
    Typing pattern metrics - NO key content.

    These metrics characterize HOW someone types, not WHAT they type.
    """
    avg_hold_time_ms: float = 0.0  # Average time key held down
    avg_seek_time_ms: float = 0.0  # Average time between keystrokes
    typing_speed_cpm: float = 0.0  # Characters per minute
    rhythm_variance: float = 0.0  # Variance in typing rhythm
    burst_count: int = 0  # Number of fast typing bursts
    pause_count: int = 0  # Number of long pauses (>2s)
    total_keystrokes: int = 0  # Total keys pressed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "avg_hold_time_ms": round(self.avg_hold_time_ms, 2),
            "avg_seek_time_ms": round(self.avg_seek_time_ms, 2),
            "typing_speed_cpm": round(self.typing_speed_cpm, 1),
            "rhythm_variance": round(self.rhythm_variance, 2),
            "burst_count": self.burst_count,
            "pause_count": self.pause_count,
            "total_keystrokes": self.total_keystrokes,
        }


@dataclass
class MouseActivity:
    """
    Mouse activity metrics.
    """
    clicks_count: int = 0
    scroll_count: int = 0
    movement_distance: float = 0.0  # Total pixels moved
    movement_pattern: str = "minimal"  # minimal, smooth, erratic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "clicks_count": self.clicks_count,
            "scroll_count": self.scroll_count,
            "movement_distance": round(self.movement_distance, 1),
            "movement_pattern": self.movement_pattern,
        }


class InputCollector:  # pylint: disable=too-many-instance-attributes
    """
    Low-level input event collector using pynput.

    Runs in a separate thread to avoid blocking asyncio.
    Captures timing data only - never stores which keys were pressed.
    """

    def __init__(self):
        self._running = False
        self._lock = threading.Lock()

        # Keystroke timing buffers (circular)
        self._key_press_times: Deque[float] = deque(maxlen=BUFFER_SIZE)
        self._key_release_times: Deque[float] = deque(maxlen=BUFFER_SIZE)
        self._seek_times: Deque[float] = deque(maxlen=BUFFER_SIZE)
        self._hold_times: Deque[float] = deque(maxlen=BUFFER_SIZE)

        # Mouse buffers
        self._clicks = 0
        self._scrolls = 0
        self._last_mouse_pos: Optional[tuple] = None
        self._mouse_distance = 0.0
        self._mouse_movements: Deque[float] = deque(maxlen=50)

        # Throttling state
        self._last_key_event_ms = 0

        # Keystroke tracking
        self._total_keystrokes = 0
        self._key_down_time: Dict[Any, float] = {}  # Track press time per key

        # Listeners
        self._keyboard_listener = None
        self._mouse_listener = None

    def start(self) -> bool:
        """
        Start input listeners.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            return True

        try:
            # pylint: disable=import-outside-toplevel
            from pynput import keyboard, mouse

            # Keyboard listener
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.start()

            # Mouse listener
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
            )
            self._mouse_listener.start()

            self._running = True
            logger.info("Input collector started")
            return True

        except ImportError:
            logger.error("pynput not installed. Run: pip install pynput")
            return False
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to start input collector: %s", e)
            return False

    @property
    def is_running(self) -> bool:
        """Check if collector is running."""
        return self._running

    def stop(self) -> None:
        """Stop input listeners."""
        self._running = False

        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

        logger.info("Input collector stopped")

    def _on_key_press(self, key) -> None:
        """
        Handle key press - capture timing only.

        IMPORTANT: We never store which key was pressed, only WHEN.
        """
        now_ms = time.time() * 1000

        # Throttle to reduce CPU usage
        if now_ms - self._last_key_event_ms < THROTTLE_MS:
            return

        self._last_key_event_ms = now_ms

        with self._lock:
            # Calculate seek time (time since last key release)
            if self._key_release_times:
                seek_time = now_ms - self._key_release_times[-1]
                self._seek_times.append(seek_time)

            self._key_press_times.append(now_ms)
            self._total_keystrokes += 1

            # Track press time for hold duration (use id of key, not the key itself)
            key_id = id(key)  # Use object id, not key value
            self._key_down_time[key_id] = now_ms

        # Feed KeystrokeAnalyzer (outside lock to avoid deadlock)
        try:
            from learners.keystroke_analyzer import get_keystroke_analyzer
            analyzer = get_keystroke_analyzer()
            analyzer.add_event(key=str(id(key)), event_type="press", timestamp=now_ms / 1000)
        except Exception as e:
            logger.debug("KeystrokeAnalyzer feed failed: %s", e)

    def _on_key_release(self, key) -> None:
        """Handle key release - calculate hold time."""
        now_ms = time.time() * 1000

        with self._lock:
            self._key_release_times.append(now_ms)

            # Calculate hold time
            key_id = id(key)
            if key_id in self._key_down_time:
                hold_time = now_ms - self._key_down_time[key_id]
                self._hold_times.append(hold_time)
                del self._key_down_time[key_id]

        # Feed KeystrokeAnalyzer (outside lock to avoid deadlock)
        try:
            from learners.keystroke_analyzer import get_keystroke_analyzer
            analyzer = get_keystroke_analyzer()
            analyzer.add_event(key=str(id(key)), event_type="release", timestamp=now_ms / 1000)
        except Exception as e:
            logger.debug("KeystrokeAnalyzer feed failed: %s", e)

    def _on_mouse_move(self, x: int, y: int) -> None:
        """Track mouse movement distance."""
        if self._last_mouse_pos:
            dx = x - self._last_mouse_pos[0]
            dy = y - self._last_mouse_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5

            with self._lock:
                self._mouse_distance += distance
                self._mouse_movements.append(distance)

        self._last_mouse_pos = (x, y)

    def _on_mouse_click(
        self, _x: int, _y: int, _button: Any, pressed: bool
    ) -> None:
        """Count mouse clicks."""
        if pressed:
            with self._lock:
                self._clicks += 1

    def _on_mouse_scroll(
        self, _x: int, _y: int, _dx: int, _dy: int
    ) -> None:
        """Count scroll events."""
        with self._lock:
            self._scrolls += 1

    def get_and_reset_metrics(self) -> tuple[KeystrokeDynamics, MouseActivity]:
        """
        Get aggregated metrics and reset buffers.

        Returns:
            Tuple of (KeystrokeDynamics, MouseActivity).
        """
        with self._lock:
            # Calculate keystroke dynamics
            dynamics = self._calculate_keystroke_dynamics()

            # Calculate mouse activity
            mouse = self._calculate_mouse_activity()

            # Reset buffers
            self._reset_buffers()

            return dynamics, mouse

    def _calculate_keystroke_dynamics(self) -> KeystrokeDynamics:
        """Calculate keystroke dynamics from buffers."""
        if not self._key_press_times:
            return KeystrokeDynamics(total_keystrokes=self._total_keystrokes)

        # Average hold time
        avg_hold = (
            statistics.mean(self._hold_times)
            if self._hold_times else 0.0
        )

        # Average seek time
        avg_seek = (
            statistics.mean(self._seek_times)
            if self._seek_times else 0.0
        )

        # Rhythm variance
        variance = 0.0
        if len(self._seek_times) > 1:
            try:
                variance = statistics.variance(self._seek_times)
            except statistics.StatisticsError:
                pass

        # Typing speed (characters per minute)
        duration_ms = 0
        if len(self._key_press_times) >= 2:
            duration_ms = self._key_press_times[-1] - self._key_press_times[0]

        cpm = 0.0
        if duration_ms > 0:
            cpm = (len(self._key_press_times) / duration_ms) * 60000

        # Count bursts (sequences with <100ms between keys)
        bursts = sum(1 for t in self._seek_times if t < 100)

        # Count pauses (>2000ms between keys)
        pauses = sum(1 for t in self._seek_times if t > 2000)

        return KeystrokeDynamics(
            avg_hold_time_ms=avg_hold,
            avg_seek_time_ms=avg_seek,
            typing_speed_cpm=cpm,
            rhythm_variance=variance,
            burst_count=bursts,
            pause_count=pauses,
            total_keystrokes=self._total_keystrokes,
        )

    def _calculate_mouse_activity(self) -> MouseActivity:
        """Calculate mouse activity from buffers."""
        # Classify movement pattern
        pattern = "minimal"
        if self._mouse_movements:
            avg_movement = statistics.mean(self._mouse_movements)
            if len(self._mouse_movements) > 10:
                try:
                    variance = statistics.variance(self._mouse_movements)
                    if variance > 1000:
                        pattern = "erratic"
                    elif avg_movement > 50:
                        pattern = "smooth"
                except statistics.StatisticsError:
                    pass

        return MouseActivity(
            clicks_count=self._clicks,
            scroll_count=self._scrolls,
            movement_distance=self._mouse_distance,
            movement_pattern=pattern,
        )

    def _reset_buffers(self) -> None:
        """Reset all metric buffers."""
        self._key_press_times.clear()
        self._key_release_times.clear()
        self._seek_times.clear()
        self._hold_times.clear()
        self._clicks = 0
        self._scrolls = 0
        self._mouse_distance = 0.0
        self._mouse_movements.clear()
        self._total_keystrokes = 0
        self._key_down_time.clear()


@register_collector
class InputWatcher(BaseWatcher):
    """
    Watcher for keyboard and mouse activity patterns.

    Captures keystroke dynamics (timing patterns) and mouse metrics.
    NEVER logs which keys are pressed - only timing information.

    Attributes:
        name: "input_watcher"
        version: "1.0.0"
    """

    name = "input_watcher"
    version = "1.0.0"

    def __init__(self, batch_interval: float = 60.0):
        """
        Initialize input watcher.

        Args:
            batch_interval: Seconds between metric aggregation.
        """
        super().__init__(batch_interval=batch_interval)
        self._collector: Optional[InputCollector] = None
        self._collection_interval = batch_interval  # Collect when we aggregate

    async def start(self) -> None:
        """Start watcher with input collector."""
        self._collector = InputCollector()
        if not self._collector.start():
            logger.error("Failed to start input collector")
            return

        await super().start()

    async def stop(self) -> None:
        """Stop watcher and input collector."""
        await super().stop()
        if self._collector:
            self._collector.stop()
            self._collector = None

    async def collect(self) -> Optional[Heartbeat]:
        """
        Collect aggregated input metrics.

        Returns:
            Heartbeat with keystroke dynamics and mouse activity.
        """
        if not self._collector:
            return None

        dynamics, mouse = self._collector.get_and_reset_metrics()

        # Only emit if there was activity
        if dynamics.total_keystrokes == 0 and mouse.clicks_count == 0:
            return None

        return Heartbeat(
            timestamp=datetime.now(),
            watcher_type=self.name,
            data={
                "keystroke_dynamics": dynamics.to_dict(),
                "mouse_activity": mouse.to_dict(),
            },
        )

    def _should_merge(self, heartbeat: Heartbeat) -> bool:
        """Never merge input heartbeats - each aggregation is unique."""
        return False

    def get_collection_interval(self) -> float:
        """Return aggregation interval."""
        return self._collection_interval

    def get_config(self) -> Dict[str, Any]:
        """Return watcher configuration."""
        return {
            "batch_interval": self.batch_interval,
            "throttle_ms": THROTTLE_MS,
            "buffer_size": BUFFER_SIZE,
            "collector_running": self._collector is not None and self._collector.is_running,
        }

    async def flush(self) -> list:
        """Flush heartbeats to ActivityStore and StyleLearner."""
        flushed = await super().flush()
        if not flushed:
            return flushed

        # Store in ActivityStore
        try:
            store = get_activity_store()
            for hb in flushed:
                store.add(watcher_type="input", timestamp=hb.timestamp, data=hb.data)
        except Exception as e:
            logger.warning("Failed to store input data: %s", e)

        # Feed StyleLearner with keystroke dynamics
        try:
            learner = get_style_learner()
            for hb in flushed:
                dynamics = hb.data.get("keystroke_dynamics", {})
                learner.add_keystroke_sample(dynamics)
        except Exception as e:
            logger.debug("StyleLearner feed failed: %s", e)

        return flushed
