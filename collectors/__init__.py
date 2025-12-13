"""
DAIMON Collectors - Auto-surveillance components.
=================================================

Plugin-based system for OS observability.
All collectors register automatically via @register_collector decorator.

Usage:
    from collectors import CollectorRegistry

    # Get all available collectors
    for name in CollectorRegistry.get_names():
        watcher = CollectorRegistry.create_instance(name)
        await watcher.start()
"""

from .base import BaseWatcher, Heartbeat
from .registry import CollectorRegistry, register_collector

# Import collectors to trigger registration
from . import shell_watcher
from . import claude_watcher
from . import window_watcher
from . import input_watcher
from . import afk_watcher
from . import browser_watcher

__all__ = [
    # Core
    "BaseWatcher",
    "Heartbeat",
    "CollectorRegistry",
    "register_collector",
    # Collectors
    "shell_watcher",
    "claude_watcher",
    "window_watcher",
    "input_watcher",
    "afk_watcher",
    "browser_watcher",
]
