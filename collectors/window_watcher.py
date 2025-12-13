#!/usr/bin/env python3
"""
DAIMON Window Watcher - Active Window Tracking
==============================================

Tracks the currently focused window (app name, title).
Uses X11 backend via python-xlib. Privacy-preserving: titles are sanitized.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .base import BaseWatcher, Heartbeat
from .registry import register_collector
from memory.activity_store import get_activity_store
from learners import get_style_learner

logger = logging.getLogger("daimon.window")

# Patterns for sanitizing sensitive data from window titles
SANITIZE_PATTERNS = [
    (re.compile(r"\S+@\S+\.\S+"), "[email]"),  # Email addresses
    (re.compile(r"https?://[^\s]+"), "[url]"),  # URLs
    (re.compile(r"\b\d{6,}\b"), "[number]"),  # Long numbers (IDs, etc)
    (re.compile(r"Bearer\s+\S+"), "[token]"),  # Auth tokens
    (re.compile(r"password[=:]\S+", re.I), "[password]"),  # Passwords
]

# Maximum title length after sanitization
MAX_TITLE_LENGTH = 100


@dataclass
class WindowInfo:
    """Information about a window."""
    app_name: str
    window_class: str
    window_title: str
    pid: int = 0


def detect_display_server() -> str:
    """
    Detect current display server.

    Returns:
        "x11", "wayland", or "unknown"
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    if session_type == "wayland":
        return "wayland"
    if session_type == "x11":
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"

    return "unknown"


def sanitize_title(title: str) -> str:
    """
    Remove sensitive data from window title.

    Args:
        title: Raw window title.

    Returns:
        Sanitized title, truncated to MAX_TITLE_LENGTH.
    """
    if not title:
        return ""

    result = title
    for pattern, replacement in SANITIZE_PATTERNS:
        result = pattern.sub(replacement, result)

    # Truncate
    if len(result) > MAX_TITLE_LENGTH:
        result = result[:MAX_TITLE_LENGTH] + "..."

    return result


class X11Backend:
    """
    X11 backend for window tracking using python-xlib.

    Provides active window information via EWMH (_NET_ACTIVE_WINDOW).
    """

    def __init__(self):
        self._display = None
        self._root = None
        self._atoms: Dict[str, Any] = {}

    def _ensure_connected(self) -> bool:
        """
        Ensure X11 connection is established.

        Returns:
            True if connected, False otherwise.
        """
        if self._display is not None:
            return True

        try:
            from Xlib import display  # pylint: disable=import-outside-toplevel

            self._display = display.Display()
            self._root = self._display.screen().root

            # Cache commonly used atoms
            self._atoms = {
                "NET_ACTIVE_WINDOW": self._display.intern_atom("_NET_ACTIVE_WINDOW"),
                "NET_WM_NAME": self._display.intern_atom("_NET_WM_NAME"),
                "NET_WM_PID": self._display.intern_atom("_NET_WM_PID"),
                "WM_NAME": self._display.intern_atom("WM_NAME"),
                "WM_CLASS": self._display.intern_atom("WM_CLASS"),
                "UTF8_STRING": self._display.intern_atom("UTF8_STRING"),
            }

            logger.debug("X11 connection established")
            return True

        except ImportError:
            logger.error("python-xlib not installed. Run: pip install python-xlib")
            return False
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("X11 connection failed: %s", e)
            return False

    def get_active_window(self) -> Optional[WindowInfo]:
        """
        Get information about the currently active window.

        Returns:
            WindowInfo or None if unavailable.
        """
        if not self._ensure_connected():
            return None

        try:
            # Import inside method to avoid loading Xlib at module level
            from Xlib import X  # pylint: disable=import-outside-toplevel

            # Get active window ID
            prop = self._root.get_full_property(
                self._atoms["NET_ACTIVE_WINDOW"],
                X.AnyPropertyType,
            )

            if not prop or not prop.value:
                return None

            window_id = prop.value[0]
            if not window_id:
                return None

            window = self._display.create_resource_object("window", window_id)

            # Get window properties
            app_name, window_class = self._get_wm_class(window)
            title = self._get_window_title(window)
            pid = self._get_window_pid(window)

            return WindowInfo(
                app_name=app_name,
                window_class=window_class,
                window_title=title,
                pid=pid,
            )

        except (OSError, AttributeError, KeyError, TypeError) as e:
            logger.debug("Error getting active window: %s", e)
            return None

    def _get_wm_class(self, window: Any) -> tuple[str, str]:
        """Get WM_CLASS property (instance name, class name)."""
        try:
            prop = window.get_full_property(self._atoms["WM_CLASS"], 0)
            if prop and prop.value:
                # WM_CLASS is null-separated: "instance\0class\0"
                parts = prop.value.decode("utf-8", errors="replace").split("\0")
                if len(parts) >= 2:
                    return parts[0], parts[1]
                if parts:
                    return parts[0], parts[0]
        except (OSError, AttributeError, UnicodeDecodeError):
            pass
        return "unknown", "unknown"

    def _get_window_title(self, window: Any) -> str:
        """Get window title (_NET_WM_NAME or WM_NAME)."""
        try:
            # Try _NET_WM_NAME first (UTF-8)
            prop = window.get_full_property(
                self._atoms["NET_WM_NAME"],
                self._atoms["UTF8_STRING"],
            )
            if prop and prop.value:
                return prop.value.decode("utf-8", errors="replace")

            # Fallback to WM_NAME
            prop = window.get_full_property(self._atoms["WM_NAME"], 0)
            if prop and prop.value:
                return prop.value.decode("latin-1", errors="replace")
        except (OSError, AttributeError, UnicodeDecodeError):
            pass
        return ""

    def _get_window_pid(self, window: Any) -> int:
        """Get window PID (_NET_WM_PID)."""
        try:
            from Xlib import X  # pylint: disable=import-outside-toplevel
            prop = window.get_full_property(
                self._atoms["NET_WM_PID"],
                X.AnyPropertyType,
            )
            if prop and prop.value:
                return int(prop.value[0])
        except (OSError, AttributeError, IndexError, ImportError):
            pass
        return 0

    def close(self) -> None:
        """Close X11 connection."""
        if self._display:
            try:
                self._display.close()
            except (OSError, AttributeError):
                pass
            self._display = None
            self._root = None


@register_collector
class WindowWatcher(BaseWatcher):
    """
    Watcher for active window tracking.

    Captures which application has focus and for how long.
    Sanitizes window titles to remove sensitive information.

    Attributes:
        name: "window_watcher"
        version: "1.0.0"
    """

    name = "window_watcher"
    version = "1.0.0"

    def __init__(self, batch_interval: float = 30.0, poll_interval: float = 2.0):
        """
        Initialize window watcher.

        Args:
            batch_interval: Seconds between batch flushes.
            poll_interval: Seconds between window checks.
        """
        super().__init__(batch_interval=batch_interval)
        self._poll_interval = poll_interval
        self._backend: Optional[X11Backend] = None
        self._last_window: Optional[WindowInfo] = None
        self._focus_start: Optional[datetime] = None
        self._display_server = detect_display_server()

    def _ensure_backend(self) -> bool:
        """Initialize backend if needed."""
        if self._backend is not None:
            return True

        if self._display_server == "x11":
            self._backend = X11Backend()
            return True

        if self._display_server == "wayland":
            logger.warning(
                "Wayland detected. Window tracking limited. "
                "Install GNOME extension 'Focused Window D-Bus' for full support."
            )
            # Wayland backend not yet implemented - requires D-Bus integration
            return False

        logger.error("Unknown display server, window tracking disabled")
        return False

    async def collect(self) -> Optional[Heartbeat]:
        """
        Collect current window information.

        Returns:
            Heartbeat with window data, or None if unchanged.
        """
        if not self._ensure_backend():
            return None

        window = self._backend.get_active_window()
        if not window:
            return None

        now = datetime.now()

        # Check if window changed
        if self._last_window and self._is_same_window(window, self._last_window):
            return None

        # Calculate focus duration for previous window
        focus_duration = 0.0
        if self._focus_start:
            focus_duration = (now - self._focus_start).total_seconds()

        # Update tracking
        self._focus_start = now
        self._last_window = window

        # Sanitize title
        safe_title = sanitize_title(window.window_title)

        return Heartbeat(
            timestamp=now,
            watcher_type=self.name,
            data={
                "app_name": window.app_name,
                "window_class": window.window_class,
                "window_title": safe_title,
                "pid": window.pid,
                "focus_duration": focus_duration,
            },
        )

    def _is_same_window(self, w1: WindowInfo, w2: WindowInfo) -> bool:
        """Check if two windows are the same (by app and class)."""
        return w1.app_name == w2.app_name and w1.window_class == w2.window_class

    def _should_merge(self, heartbeat: Heartbeat) -> bool:
        """
        Check if heartbeat should merge with previous.

        Merges if same app_name and window_class (ignores title changes).
        """
        if not self._last_heartbeat:
            return False

        last_data = self._last_heartbeat.data
        new_data = heartbeat.data

        return (
            last_data.get("app_name") == new_data.get("app_name") and
            last_data.get("window_class") == new_data.get("window_class")
        )

    def get_collection_interval(self) -> float:
        """Return poll interval for window checks."""
        return self._poll_interval

    def get_config(self) -> Dict[str, Any]:
        """Return watcher configuration."""
        return {
            "batch_interval": self.batch_interval,
            "poll_interval": self._poll_interval,
            "display_server": self._display_server,
            "backend": "x11" if self._backend else "none",
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
                store.add(watcher_type="window", timestamp=hb.timestamp, data=hb.data)
        except Exception as e:
            logger.warning("Failed to store window data: %s", e)

        # Feed StyleLearner
        try:
            learner = get_style_learner()
            for hb in flushed:
                learner.add_window_sample(hb.data)
        except Exception as e:
            logger.debug("StyleLearner feed failed: %s", e)

        return flushed

    async def stop(self) -> None:
        """Stop watcher and clean up backend."""
        await super().stop()
        if self._backend:
            self._backend.close()
            self._backend = None
