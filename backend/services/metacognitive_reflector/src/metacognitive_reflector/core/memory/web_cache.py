"""
NOESIS Web Cache - Search Results Persistence
==============================================

Stores web search results in RESOURCE memory for future reference.
Enables Noesis to remember and reference previous searches.

Architecture:
    ┌─────────────────┐    async     ┌─────────────────┐
    │   Web Search    │──────────────│ Episodic Memory │
    │    Results      │  WebCache    │  (RESOURCE)     │
    └─────────────────┘              └─────────────────┘

Features:
- Stores search queries and results as RESOURCE memories
- 30-day TTL (per MIRIX architecture)
- Deduplication via query hash
- Graceful degradation

Based on: MIRIX RESOURCE memory type
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WebCache:
    """
    Caches web search results in RESOURCE memory.

    RESOURCE memories have 30-day TTL and LOW priority per MIRIX architecture.

    Usage:
        cache = WebCache()

        # Store search results
        await cache.store_search(
            query="MIRIX memory architecture",
            results=[
                {"title": "...", "url": "...", "snippet": "..."},
                ...
            ],
            session_id="abc123"
        )

        # Check if query was recently searched
        cached = await cache.get_cached_search("MIRIX memory architecture")
    """

    def __init__(self, bridge: Optional[Any] = None) -> None:
        """
        Initialize web cache.

        Args:
            bridge: MemoryBridge instance (or None for lazy init)
        """
        self._bridge = bridge
        self._local_cache: Dict[str, Dict[str, Any]] = {}  # Query hash -> results

    def _get_bridge(self) -> Optional[Any]:
        """Get bridge instance (lazy init)."""
        if self._bridge is None:
            try:
                from metacognitive_reflector.core.memory.memory_bridge import get_memory_bridge
                self._bridge = get_memory_bridge()
            except ImportError:
                logger.warning("Memory bridge not available for web cache")
                return None
        return self._bridge

    def _hash_query(self, query: str) -> str:
        """Generate hash for query (for deduplication)."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    async def store_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        source: str = "web_search"
    ) -> Optional[str]:
        """
        Store web search results as RESOURCE memory.

        Args:
            query: Search query
            results: List of search results (title, url, snippet)
            session_id: Optional session context
            source: Source identifier (web_search, api_call, etc.)

        Returns:
            Memory ID if stored successfully, None otherwise
        """
        bridge = self._get_bridge()
        if not bridge:
            # Fallback to local cache only
            query_hash = self._hash_query(query)
            self._local_cache[query_hash] = {
                "query": query,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            logger.debug("Stored search in local cache: %s", query_hash)
            return None

        try:
            # Format results for storage
            result_texts = []
            for i, r in enumerate(results[:10], 1):  # Max 10 results
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")[:200]
                result_texts.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")

            content = f"[WEB_SEARCH] Query: {query}\n\n" + "\n\n".join(result_texts)

            # Store via bridge (using internal method)
            if await bridge._check_service():
                client = await bridge._get_client()
                if client:
                    payload = {
                        "content": content,
                        "type": "resource",  # RESOURCE type per MIRIX
                        "context": {
                            "query": query,
                            "query_hash": self._hash_query(query),
                            "result_count": len(results),
                            "source": source,
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    }

                    response = await client.post(
                        f"{bridge.base_url}/v1/memories",
                        json=payload
                    )

                    if response.status_code in (200, 201):
                        data = response.json()
                        memory_id = data.get("memory_id")
                        logger.debug("Stored web search: %s -> %s", query[:30], memory_id)

                        # Also cache locally
                        query_hash = self._hash_query(query)
                        self._local_cache[query_hash] = {
                            "query": query,
                            "results": results,
                            "memory_id": memory_id,
                            "timestamp": datetime.now().isoformat()
                        }

                        return memory_id

        except Exception as e:
            logger.error("Error storing web search: %s", e)

        return None

    async def get_cached_search(
        self,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached search results for a query.

        Args:
            query: Search query

        Returns:
            Cached results or None if not found
        """
        query_hash = self._hash_query(query)

        # Check local cache first
        if query_hash in self._local_cache:
            return self._local_cache[query_hash]

        # Check episodic memory
        bridge = self._get_bridge()
        if bridge:
            try:
                memories = await bridge.search_memories(
                    f"[WEB_SEARCH] Query: {query}",
                    limit=1,
                    memory_type="resource"
                )
                if memories:
                    return {
                        "query": query,
                        "results": [],  # Would need to parse from content
                        "memory_id": memories[0].get("memory_id"),
                        "timestamp": memories[0].get("timestamp"),
                        "from_ltm": True
                    }
            except Exception as e:
                logger.error("Error searching cached results: %s", e)

        return None

    async def search_past_queries(
        self,
        topic: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search past web searches by topic.

        Args:
            topic: Topic to search for
            limit: Maximum results

        Returns:
            List of past search entries
        """
        bridge = self._get_bridge()
        if not bridge:
            return []

        try:
            memories = await bridge.search_memories(
                f"[WEB_SEARCH] {topic}",
                limit=limit,
                memory_type="resource"
            )
            return memories
        except Exception as e:
            logger.error("Error searching past queries: %s", e)
            return []

    def clear_local_cache(self) -> None:
        """Clear local cache."""
        self._local_cache.clear()


# Singleton instance
_web_cache_instance: Optional[WebCache] = None


def get_web_cache() -> WebCache:
    """Get or create singleton web cache instance."""
    global _web_cache_instance
    if _web_cache_instance is None:
        _web_cache_instance = WebCache()
    return _web_cache_instance
