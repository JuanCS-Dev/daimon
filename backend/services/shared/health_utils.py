"""
Health Check Utilities - Honest Dependency Verification
========================================================

Provides reusable health check functions that actually verify dependencies
instead of always returning "healthy".

Usage:
    from shared.health_utils import check_redis, check_http, aggregate_health

    deps = await asyncio.gather(
        check_redis(redis_url),
        check_http(service_url),
    )
    result = aggregate_health(deps)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DependencyHealth:
    """Health status of a dependency."""

    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


async def check_redis(redis_url: str, timeout: float = 3.0) -> DependencyHealth:
    """
    Check Redis connectivity with actual ping.

    Args:
        redis_url: Redis connection URL
        timeout: Timeout in seconds

    Returns:
        DependencyHealth with actual connectivity status
    """
    try:
        import redis.asyncio as aioredis

        start = time.monotonic()
        client = await aioredis.from_url(
            redis_url, socket_timeout=timeout, socket_connect_timeout=timeout
        )
        await asyncio.wait_for(client.ping(), timeout=timeout)
        await client.close()
        latency = (time.monotonic() - start) * 1000

        return DependencyHealth("redis", True, latency)
    except ImportError:
        return DependencyHealth("redis", False, error="redis package not installed")
    except asyncio.TimeoutError:
        return DependencyHealth("redis", False, error="connection timeout")
    except Exception as e:
        return DependencyHealth("redis", False, error=str(e)[:100])


async def check_http(
    url: str, path: str = "/health", timeout: float = 5.0
) -> DependencyHealth:
    """
    Check HTTP service connectivity.

    Args:
        url: Base URL of the service
        path: Health check endpoint path
        timeout: Timeout in seconds

    Returns:
        DependencyHealth with actual connectivity status

    Raises:
        TypeError: If url is None
    """
    if url is None:
        raise TypeError("url cannot be None")

    if not isinstance(url, str) or not url.strip():
        return DependencyHealth(str(url), False, error="invalid url")

    try:
        import httpx

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url.rstrip('/')}{path}")
            latency = (time.monotonic() - start) * 1000

            if response.status_code == 200:
                return DependencyHealth(url, True, latency)
            else:
                return DependencyHealth(
                    url, False, latency, error=f"HTTP {response.status_code}"
                )
    except ImportError:
        return DependencyHealth(url, False, error="httpx package not installed")
    except Exception as e:
        return DependencyHealth(url, False, error=str(e)[:100])


async def check_qdrant(url: str, timeout: float = 5.0) -> DependencyHealth:
    """
    Check Qdrant vector database connectivity.

    Args:
        url: Qdrant server URL
        timeout: Timeout in seconds

    Returns:
        DependencyHealth with actual connectivity status
    """
    try:
        import httpx

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Qdrant uses /healthz endpoint
            response = await client.get(f"{url.rstrip('/')}/healthz")
            latency = (time.monotonic() - start) * 1000

            if response.status_code == 200:
                return DependencyHealth("qdrant", True, latency)
            else:
                return DependencyHealth(
                    "qdrant", False, latency, error=f"HTTP {response.status_code}"
                )
    except Exception as e:
        return DependencyHealth("qdrant", False, error=str(e)[:100])


async def check_database(db_url: str, timeout: float = 5.0) -> DependencyHealth:
    """
    Check PostgreSQL database connectivity.

    Args:
        db_url: Database connection URL
        timeout: Timeout in seconds

    Returns:
        DependencyHealth with actual connectivity status
    """
    try:
        import asyncpg

        start = time.monotonic()
        conn = await asyncpg.connect(db_url, timeout=timeout)
        await conn.fetchval("SELECT 1")
        await conn.close()
        latency = (time.monotonic() - start) * 1000

        return DependencyHealth("database", True, latency)
    except ImportError:
        return DependencyHealth("database", False, error="asyncpg not installed")
    except Exception as e:
        return DependencyHealth("database", False, error=str(e)[:100])


def aggregate_health(
    deps: List[DependencyHealth], require_all: bool = False
) -> Dict[str, Any]:
    """
    Aggregate dependency health into response.

    Args:
        deps: List of dependency health checks
        require_all: If True, all deps must be healthy. If False, at least one.

    Returns:
        Dict with aggregated health status
    """
    valid_deps = [d for d in deps if isinstance(d, DependencyHealth)]

    if require_all:
        all_healthy = all(d.healthy for d in valid_deps) if valid_deps else False
    else:
        # Service is degraded but not dead if at least one dep works
        all_healthy = any(d.healthy for d in valid_deps) if valid_deps else False

    critical_healthy = all(
        d.healthy
        for d in valid_deps
        if d.name in ["redis", "database", "qdrant"]
    )

    def safe_round(val):
        """Safely round a value, handling non-numeric types."""
        if val is None:
            return None
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return None

    return {
        "status": "healthy" if all_healthy else ("degraded" if critical_healthy else "unhealthy"),
        "dependencies": {
            d.name: {
                "healthy": bool(d.healthy),  # Ensure boolean
                "latency_ms": safe_round(d.latency_ms),
                "error": str(d.error)[:500] if d.error else None,  # Truncate long errors
            }
            for d in valid_deps
        },
    }


async def full_health_check(
    redis_url: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    db_url: Optional[str] = None,
    http_services: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run full health check on all configured dependencies.

    Args:
        redis_url: Redis URL (optional)
        qdrant_url: Qdrant URL (optional)
        db_url: Database URL (optional)
        http_services: Dict of service_name -> url to check

    Returns:
        Complete health status
    """
    tasks = []

    if redis_url:
        tasks.append(check_redis(redis_url))
    if qdrant_url:
        tasks.append(check_qdrant(qdrant_url))
    if db_url:
        tasks.append(check_database(db_url))
    if http_services:
        for name, url in http_services.items():
            tasks.append(check_http(url))

    if not tasks:
        return {"status": "healthy", "dependencies": {}, "note": "no dependencies configured"}

    results = await asyncio.gather(*tasks, return_exceptions=True)

    deps = []
    for r in results:
        if isinstance(r, DependencyHealth):
            deps.append(r)
        elif isinstance(r, Exception):
            deps.append(DependencyHealth("unknown", False, error=str(r)[:100]))

    return aggregate_health(deps)
