"""Tests for CollectorRegistry class."""

import pytest
from typing import Any, Dict, Optional

from collectors.registry import CollectorRegistry, register_collector
from collectors.base import BaseWatcher, Heartbeat


class DummyWatcher(BaseWatcher):
    """Dummy watcher for testing."""

    name = "dummy_watcher"
    version = "1.0.0"

    async def collect(self) -> Optional[Heartbeat]:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {"test": True}


class AnotherWatcher(BaseWatcher):
    """Another dummy watcher."""

    name = "another_watcher"
    version = "1.0.0"

    def __init__(self, value: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    async def collect(self) -> Optional[Heartbeat]:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {"value": self.value}


class NoNameWatcher(BaseWatcher):
    """Watcher without name."""

    async def collect(self) -> Optional[Heartbeat]:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test."""
    # Store original state
    original_collectors = CollectorRegistry._collectors.copy()
    original_instances = CollectorRegistry._instances.copy()

    # Clear for test
    CollectorRegistry.clear_all()

    yield

    # Restore original state
    CollectorRegistry._collectors = original_collectors
    CollectorRegistry._instances = original_instances


class TestRegister:
    """Tests for register method."""

    def test_register_watcher(self):
        """Register a watcher class."""
        CollectorRegistry.register(DummyWatcher)

        assert "dummy_watcher" in CollectorRegistry.get_names()

    def test_register_returns_class(self):
        """Register returns the same class."""
        result = CollectorRegistry.register(DummyWatcher)

        assert result is DummyWatcher

    def test_register_no_name_raises(self):
        """Register raises for watcher without name."""
        with pytest.raises(ValueError, match="must define.*name"):
            CollectorRegistry.register(NoNameWatcher)

    def test_register_base_name_raises(self):
        """Register raises for watcher with name='base'."""

        class BaseNameWatcher(BaseWatcher):
            name = "base"

            async def collect(self):
                return None

            def get_config(self):
                return {}

        with pytest.raises(ValueError, match="must define.*name"):
            CollectorRegistry.register(BaseNameWatcher)

    def test_register_overwrite_warns(self, caplog):
        """Overwriting existing registration warns."""
        CollectorRegistry.register(DummyWatcher)

        class DuplicateWatcher(BaseWatcher):
            name = "dummy_watcher"  # Same name

            async def collect(self):
                return None

            def get_config(self):
                return {}

        CollectorRegistry.register(DuplicateWatcher)

        assert "overwriting" in caplog.text.lower()


class TestUnregister:
    """Tests for unregister method."""

    def test_unregister_existing(self):
        """Unregister existing watcher."""
        CollectorRegistry.register(DummyWatcher)
        result = CollectorRegistry.unregister("dummy_watcher")

        assert result is True
        assert "dummy_watcher" not in CollectorRegistry.get_names()

    def test_unregister_nonexistent(self):
        """Unregister nonexistent returns False."""
        result = CollectorRegistry.unregister("nonexistent")

        assert result is False

    def test_unregister_clears_instance(self):
        """Unregister also clears cached instance."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.create_instance("dummy_watcher")

        CollectorRegistry.unregister("dummy_watcher")

        assert CollectorRegistry.get_instance("dummy_watcher") is None


class TestGet:
    """Tests for get method."""

    def test_get_existing(self):
        """Get existing watcher class."""
        CollectorRegistry.register(DummyWatcher)
        cls = CollectorRegistry.get("dummy_watcher")

        assert cls is DummyWatcher

    def test_get_nonexistent(self):
        """Get nonexistent returns None."""
        cls = CollectorRegistry.get("nonexistent")

        assert cls is None


class TestGetAll:
    """Tests for get_all method."""

    def test_get_all(self):
        """Get all registered collectors."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.register(AnotherWatcher)

        all_collectors = CollectorRegistry.get_all()

        assert "dummy_watcher" in all_collectors
        assert "another_watcher" in all_collectors

    def test_get_all_returns_copy(self):
        """Get all returns a copy, not the original."""
        CollectorRegistry.register(DummyWatcher)
        all_collectors = CollectorRegistry.get_all()

        # Modifying returned dict shouldn't affect registry
        all_collectors["test"] = None

        assert "test" not in CollectorRegistry._collectors


class TestGetNames:
    """Tests for get_names method."""

    def test_get_names(self):
        """Get names of registered collectors."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.register(AnotherWatcher)

        names = CollectorRegistry.get_names()

        assert "dummy_watcher" in names
        assert "another_watcher" in names

    def test_get_names_empty(self):
        """Get names when empty."""
        names = CollectorRegistry.get_names()

        assert names == []


class TestCreateInstance:
    """Tests for create_instance method."""

    def test_create_instance(self):
        """Create watcher instance."""
        CollectorRegistry.register(DummyWatcher)
        instance = CollectorRegistry.create_instance("dummy_watcher")

        assert isinstance(instance, DummyWatcher)

    def test_create_with_kwargs(self):
        """Create instance with constructor arguments."""
        CollectorRegistry.register(AnotherWatcher)
        instance = CollectorRegistry.create_instance("another_watcher", value=100)

        assert instance.value == 100

    def test_caches_instance(self):
        """Instances are cached by default."""
        CollectorRegistry.register(DummyWatcher)
        instance1 = CollectorRegistry.create_instance("dummy_watcher")
        instance2 = CollectorRegistry.create_instance("dummy_watcher")

        assert instance1 is instance2

    def test_no_cache(self):
        """Create without caching."""
        CollectorRegistry.register(DummyWatcher)
        instance1 = CollectorRegistry.create_instance("dummy_watcher", cache=False)
        instance2 = CollectorRegistry.create_instance("dummy_watcher", cache=False)

        assert instance1 is not instance2

    def test_unknown_raises(self):
        """Create unknown collector raises."""
        with pytest.raises(ValueError, match="Unknown collector"):
            CollectorRegistry.create_instance("unknown")


class TestGetInstance:
    """Tests for get_instance method."""

    def test_get_cached(self):
        """Get cached instance."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.create_instance("dummy_watcher")

        instance = CollectorRegistry.get_instance("dummy_watcher")

        assert instance is not None
        assert isinstance(instance, DummyWatcher)

    def test_get_uncached(self):
        """Get uncached returns None."""
        CollectorRegistry.register(DummyWatcher)

        instance = CollectorRegistry.get_instance("dummy_watcher")

        assert instance is None


class TestClearInstances:
    """Tests for clear_instances method."""

    def test_clear_instances(self):
        """Clear all cached instances."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.register(AnotherWatcher)
        CollectorRegistry.create_instance("dummy_watcher")
        CollectorRegistry.create_instance("another_watcher")

        CollectorRegistry.clear_instances()

        assert CollectorRegistry.get_instance("dummy_watcher") is None
        assert CollectorRegistry.get_instance("another_watcher") is None

    def test_clear_keeps_registrations(self):
        """Clear instances keeps registrations."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.create_instance("dummy_watcher")

        CollectorRegistry.clear_instances()

        assert "dummy_watcher" in CollectorRegistry.get_names()


class TestClearAll:
    """Tests for clear_all method."""

    def test_clear_all(self):
        """Clear everything."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.create_instance("dummy_watcher")

        CollectorRegistry.clear_all()

        assert CollectorRegistry.get_names() == []
        assert CollectorRegistry.get_instance("dummy_watcher") is None


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status(self):
        """Get registry status."""
        CollectorRegistry.register(DummyWatcher)
        CollectorRegistry.register(AnotherWatcher)
        CollectorRegistry.create_instance("dummy_watcher")

        status = CollectorRegistry.get_status()

        assert "dummy_watcher" in status["registered_collectors"]
        assert "another_watcher" in status["registered_collectors"]
        assert "dummy_watcher" in status["active_instances"]
        assert status["total_registered"] == 2
        assert status["total_active"] == 1


class TestRegisterCollectorDecorator:
    """Tests for register_collector convenience decorator."""

    def test_decorator(self):
        """Register via decorator."""

        @register_collector
        class DecoratedWatcher(BaseWatcher):
            name = "decorated_watcher"

            async def collect(self):
                return None

            def get_config(self):
                return {}

        assert "decorated_watcher" in CollectorRegistry.get_names()
