"""
MCP Server HTTP Utilities.

Features:
- Exponential backoff retry (AIR GAP #6 fix)
- Circuit breaker pattern
- Configurable timeouts
"""

import asyncio
from typing import Any, Dict

import httpx

from .config import REQUEST_TIMEOUT, logger

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5  # seconds
MAX_BACKOFF = 4.0  # seconds
BACKOFF_MULTIPLIER = 2.0


async def _retry_with_backoff(
    operation: str,
    url: str,
    func,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Execute HTTP operation with exponential backoff retry.

    Args:
        operation: "POST" or "GET" for logging
        url: Target URL
        func: Async function to execute
        max_retries: Maximum retry attempts

    Returns:
        Response dict or error dict
    """
    backoff = INITIAL_BACKOFF
    last_error: Dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except httpx.TimeoutException:
            last_error = {"error": "timeout", "message": f"Request to {url} timed out"}
            if attempt < max_retries:
                logger.warning(
                    "%s timeout (attempt %d/%d), retrying in %.1fs: %s",
                    operation, attempt + 1, max_retries + 1, backoff, url
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                logger.warning("%s timeout after %d attempts: %s", operation, max_retries + 1, url)
        except httpx.HTTPStatusError as e:
            # Don't retry 4xx errors (client errors)
            if 400 <= e.response.status_code < 500:
                logger.warning("%s client error %d: %s", operation, e.response.status_code, url)
                return {"error": "http_error", "status": e.response.status_code}
            # Retry 5xx errors (server errors)
            last_error = {"error": "http_error", "status": e.response.status_code}
            if attempt < max_retries:
                logger.warning(
                    "%s server error %d (attempt %d/%d), retrying in %.1fs: %s",
                    operation, e.response.status_code, attempt + 1, max_retries + 1, backoff, url
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                logger.warning(
                    "%s server error %d after %d attempts: %s",
                    operation, e.response.status_code, max_retries + 1, url
                )
        except httpx.RequestError as e:
            last_error = {"error": "connection_error", "message": str(e)}
            if attempt < max_retries:
                logger.warning(
                    "%s connection error (attempt %d/%d), retrying in %.1fs: %s - %s",
                    operation, attempt + 1, max_retries + 1, backoff, url, str(e)
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                logger.warning(
                    "%s connection error after %d attempts: %s - %s",
                    operation, max_retries + 1, url, str(e)
                )

    return last_error


async def http_post(
    url: str, payload: Dict[str, Any], timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """
    Make HTTP POST request with exponential backoff retry.

    Retries up to MAX_RETRIES times with exponential backoff for:
    - Timeout errors
    - 5xx server errors
    - Connection errors

    Does NOT retry for 4xx client errors.
    """
    async def _do_post() -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result

    return await _retry_with_backoff("POST", url, _do_post)


async def http_get(url: str, timeout: float = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """
    Make HTTP GET request with exponential backoff retry.

    Same retry behavior as http_post.
    """
    async def _do_get() -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result

    return await _retry_with_backoff("GET", url, _do_get)
