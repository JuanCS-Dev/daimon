#!/usr/bin/env python3
"""
DAIMON Collector Registry - Plugin System for Watchers
=======================================================

Central registry for automatic discovery and management of collectors.
Supports decorator-based registration and lazy instantiation.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BaseWatcher

logger = logging.getLogger("daimon.registry")


class CollectorRegistry:
    """
    Central registry for all DAIMON collectors.

    Provides decorator-based registration and instance management.
    Collectors are instantiated lazily on first request.

    Usage:
        @CollectorRegistry.register
        class MyWatcher(BaseWatcher):
            name = "my_watcher"
            ...

        # Get all registered collectors
        for name, cls in CollectorRegistry.get_all().items():
            watcher = CollectorRegistry.create_instance(name)
    """

    _collectors: Dict[str, Type[BaseWatcher]] = {}
    _instances: Dict[str, BaseWatcher] = {}

    @classmethod
    def register(cls, watcher_class: Type[BaseWatcher]) -> Type[BaseWatcher]:
        """
        Decorator to register a collector class.

        Args:
            watcher_class: The watcher class to register.

        Returns:
            The same class (allows stacking decorators).

        Raises:
            ValueError: If watcher has no name or name already registered.
        """
        name = getattr(watcher_class, "name", None)

        if not name or name == "base":
            raise ValueError(
                f"Watcher {watcher_class.__name__} must define a unique 'name' attribute"
            )

        if name in cls._collectors:
            logger.warning(
                "Overwriting collector '%s' (%s -> %s)",
                name,
                cls._collectors[name].__name__,
                watcher_class.__name__,
            )

        cls._collectors[name] = watcher_class
        logger.debug("Registered collector: %s (%s)", name, watcher_class.__name__)

        return watcher_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a collector from the registry.

        Args:
            name: Name of the collector to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in cls._collectors:
            del cls._collectors[name]
            if name in cls._instances:
                del cls._instances[name]
            logger.debug("Unregistered collector: %s", name)
            return True
        return False

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseWatcher]]:
        """
        Get a collector class by name.

        Args:
            name: Name of the collector.

        Returns:
            Watcher class or None if not found.
        """
        return cls._collectors.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseWatcher]]:
        """
        Get all registered collectors.

        Returns:
            Dict mapping names to watcher classes.
        """
        return cls._collectors.copy()

    @classmethod
    def get_names(cls) -> List[str]:
        """
        Get names of all registered collectors.

        Returns:
            List of collector names.
        """
        return list(cls._collectors.keys())

    @classmethod
    def create_instance(
        cls,
        name: str,
        cache: bool = True,
        **kwargs: Any,
    ) -> BaseWatcher:
        """
        Create or retrieve a watcher instance.

        Args:
            name: Name of the collector.
            cache: If True, cache and reuse instances (default True).
            **kwargs: Arguments passed to watcher constructor.

        Returns:
            Watcher instance.

        Raises:
            ValueError: If collector not found.
        """
        if cache and name in cls._instances:
            return cls._instances[name]

        if name not in cls._collectors:
            raise ValueError(
                f"Unknown collector: {name}. "
                f"Available: {', '.join(cls._collectors.keys())}"
            )

        watcher_class = cls._collectors[name]
        instance = watcher_class(**kwargs)

        if cache:
            cls._instances[name] = instance

        logger.debug("Created instance: %s", name)
        return instance

    @classmethod
    def get_instance(cls, name: str) -> Optional[BaseWatcher]:
        """
        Get cached instance if exists.

        Args:
            name: Name of the collector.

        Returns:
            Cached instance or None.
        """
        return cls._instances.get(name)

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances."""
        cls._instances.clear()
        logger.debug("Cleared all cached instances")

    @classmethod
    def clear_all(cls) -> None:
        """Clear both registrations and instances."""
        cls._collectors.clear()
        cls._instances.clear()
        logger.debug("Cleared registry completely")

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """
        Get registry status summary.

        Returns:
            Dict with registry information.
        """
        return {
            "registered_collectors": list(cls._collectors.keys()),
            "active_instances": list(cls._instances.keys()),
            "total_registered": len(cls._collectors),
            "total_active": len(cls._instances),
        }


# Convenience function for module-level registration
def register_collector(watcher_class: Type[BaseWatcher]) -> Type[BaseWatcher]:
    """
    Convenience decorator for registering collectors.

    Same as CollectorRegistry.register but shorter to type.

    Usage:
        @register_collector
        class MyWatcher(BaseWatcher):
            name = "my_watcher"
    """
    return CollectorRegistry.register(watcher_class)
