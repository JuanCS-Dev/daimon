"""
MCP Server HTTP Utilities.
"""

from typing import Any, Dict

import httpx

from .config import REQUEST_TIMEOUT, logger


async def http_post(
    url: str, payload: Dict[str, Any], timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """Make HTTP POST request with error handling."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result
    except httpx.TimeoutException:
        logger.warning("Request timeout: %s", url)
        return {"error": "timeout", "message": f"Request to {url} timed out"}
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error %d: %s", e.response.status_code, url)
        return {"error": "http_error", "status": e.response.status_code}
    except httpx.RequestError as e:
        logger.warning("Request error: %s - %s", url, str(e))
        return {"error": "connection_error", "message": str(e)}


async def http_get(url: str, timeout: float = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """Make HTTP GET request with error handling."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result
    except httpx.TimeoutException:
        logger.warning("GET timeout: %s", url)
        return {"error": "timeout", "message": f"Request to {url} timed out"}
    except httpx.HTTPStatusError as exc:
        logger.warning("GET HTTP error %d: %s", exc.response.status_code, url)
        return {"error": "http_error", "status": exc.response.status_code}
    except httpx.RequestError as exc:
        logger.warning("GET error: %s - %s", url, str(exc))
        return {"error": "request_error", "message": str(exc)}
