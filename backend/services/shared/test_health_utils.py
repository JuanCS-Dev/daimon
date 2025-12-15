"""
Test Health Utils - Validates the health check utilities work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from health_utils import (
    DependencyHealth,
    check_redis,
    check_http,
    check_qdrant,
    aggregate_health,
)


def test_dependency_health_dataclass():
    """Test DependencyHealth dataclass."""
    health = DependencyHealth(name="test", healthy=True, latency_ms=5.5)
    assert health.name == "test"
    assert health.healthy is True
    assert health.latency_ms == 5.5
    assert health.error is None
    print("  [PASS] DependencyHealth dataclass works")


def test_aggregate_health_all_healthy():
    """Test aggregation with all healthy deps."""
    deps = [
        DependencyHealth("redis", True, 2.5),
        DependencyHealth("qdrant", True, 5.0),
    ]
    result = aggregate_health(deps)
    assert result["status"] == "healthy"
    assert result["dependencies"]["redis"]["healthy"] is True
    assert result["dependencies"]["qdrant"]["healthy"] is True
    print("  [PASS] aggregate_health (all healthy)")


def test_aggregate_health_some_unhealthy():
    """Test aggregation with some unhealthy deps."""
    deps = [
        DependencyHealth("redis", True, 2.5),
        DependencyHealth("http", False, error="connection refused"),
    ]
    result = aggregate_health(deps)
    # Should be "healthy" because at least one dep works (require_all=False by default)
    assert result["status"] in ["healthy", "degraded"]
    assert result["dependencies"]["http"]["error"] == "connection refused"
    print("  [PASS] aggregate_health (partial)")


def test_aggregate_health_empty():
    """Test aggregation with no deps."""
    result = aggregate_health([])
    # Empty deps = degraded (no failures but nothing verified)
    assert result["status"] in ["degraded", "unhealthy"]
    assert result["dependencies"] == {}
    print("  [PASS] aggregate_health (empty)")


async def test_check_redis_offline():
    """Test Redis check when offline."""
    # Use invalid port to ensure failure
    result = await check_redis("redis://localhost:65535", timeout=1.0)
    assert result.name == "redis"
    assert result.healthy is False
    assert result.error is not None
    print("  [PASS] check_redis (offline detection)")


async def test_check_http_offline():
    """Test HTTP check when offline."""
    result = await check_http("http://localhost:65535", timeout=1.0)
    assert result.healthy is False
    assert result.error is not None
    print("  [PASS] check_http (offline detection)")


async def test_check_qdrant_offline():
    """Test Qdrant check when offline."""
    result = await check_qdrant("http://localhost:65535", timeout=1.0)
    assert result.name == "qdrant"
    assert result.healthy is False
    print("  [PASS] check_qdrant (offline detection)")


def run_sync_tests():
    """Run synchronous tests."""
    print("\n=== Sync Tests ===")
    test_dependency_health_dataclass()
    test_aggregate_health_all_healthy()
    test_aggregate_health_some_unhealthy()
    test_aggregate_health_empty()


async def run_async_tests():
    """Run async tests."""
    print("\n=== Async Tests ===")
    await test_check_redis_offline()
    await test_check_http_offline()
    await test_check_qdrant_offline()


def main():
    """Run all tests."""
    print("=" * 50)
    print("HEALTH UTILS TESTS")
    print("=" * 50)

    try:
        run_sync_tests()
        asyncio.run(run_async_tests())
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
