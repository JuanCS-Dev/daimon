"""
NOESIS Unified Memory Client - Single Point of Access
======================================================

Provides a unified interface for all memory operations, coordinating:
- SessionMemory (short-term, conversation context)
- MemoryBridge (persistence to episodic memory service)
- WebCache (web search results)
- EntityIndex (associative retrieval)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  UNIFIED MEMORY CLIENT                      │
    ├─────────────────────────────────────────────────────────────┤
    │  add_turn()      │ Session + Episodic (dual storage)       │
    │  get_context()   │ Session context for prompts             │
    │  store_insight() │ Semantic memory via bridge              │
    │  store_search()  │ Resource memory via web cache           │
    │  search()        │ Search across all memory types          │
    │  close()         │ Cleanup all resources                   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    memory = get_unified_memory()
    await memory.add_turn("user", "Hello!")
    context = memory.get_context()
    await memory.close()

Based on: Mem0 unified access + MIRIX 6-type architecture
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedMemoryClient:
    """
    Unified memory client for Noesis.

    Coordinates SessionMemory (short-term) with Episodic Memory Service
    (long-term) to provide hermetic memory persistence.

    All operations that write to long-term memory are async and non-blocking.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        auto_start_service: bool = True
    ) -> None:
        """
        Initialize unified memory client.

        Args:
            session_id: Optional session ID (creates new if None)
            auto_start_service: Attempt to start episodic memory service if offline
        """
        self._session: Optional[Any] = None
        self._bridge: Optional[Any] = None
        self._web_cache: Optional[Any] = None
        self._session_id = session_id
        self._auto_start = auto_start_service
        self._pending_tasks: List[asyncio.Task] = []
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return

        # Initialize session memory
        from metacognitive_reflector.core.memory.session import (
            create_session,
            get_or_create_session
        )

        if self._session_id:
            self._session = get_or_create_session(self._session_id)
        else:
            self._session = create_session()

        # Initialize memory bridge
        from metacognitive_reflector.core.memory.memory_bridge import MemoryBridge
        self._bridge = MemoryBridge(auto_start=self._auto_start)

        # Initialize web cache
        from metacognitive_reflector.core.memory.web_cache import WebCache
        self._web_cache = WebCache(bridge=self._bridge)

        self._initialized = True
        logger.info("UnifiedMemoryClient initialized: session=%s", self.session_id)

    @property
    def session(self) -> Any:
        """Get session memory instance."""
        self._ensure_initialized()
        return self._session

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        self._ensure_initialized()
        assert self._session is not None  # Guaranteed by _ensure_initialized
        return self._session.session_id

    @property
    def bridge(self) -> Any:
        """Get memory bridge instance."""
        self._ensure_initialized()
        return self._bridge

    @property
    def web_cache(self) -> Any:
        """Get web cache instance."""
        self._ensure_initialized()
        return self._web_cache

    async def add_turn(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        emotional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to session AND episodic memory.

        This is the primary method for recording conversation history.
        The turn is immediately added to session memory (sync) and
        asynchronously persisted to episodic memory (non-blocking).

        Args:
            role: "user" or "assistant"
            content: Message content
            importance: Memory importance (0.0-1.0)
            emotional_context: Emotional metadata (VAD, primary_emotion, strategy)
        """
        self._ensure_initialized()
        assert self._session is not None  # Guaranteed by _ensure_initialized
        assert self._bridge is not None   # Guaranteed by _ensure_initialized

        # Immediate: Add to session memory (sync)
        self._session.add_turn(role, content)

        # Background: Persist to episodic memory (async, non-blocking)
        task = asyncio.create_task(
            self._bridge.store_turn(
                session_id=self.session_id,
                role=role,
                content=content,
                importance=importance,
                emotional_context=emotional_context
            )
        )
        self._pending_tasks.append(task)
        self._cleanup_completed_tasks()

    def get_context(self, last_n: Optional[int] = None) -> str:
        """
        Get formatted conversation context for prompts.

        Args:
            last_n: Only include last N turns (None = all)

        Returns:
            Formatted context string
        """
        self._ensure_initialized()
        assert self._session is not None
        return self._session.get_context(last_n=last_n)

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        self._ensure_initialized()
        assert self._session is not None
        return self._session.get_last_user_message()

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message."""
        self._ensure_initialized()
        assert self._session is not None
        return self._session.get_last_assistant_message()

    async def store_insight(
        self,
        content: str,
        importance: float,
        category: str = "self_awareness"
    ) -> Optional[str]:
        """
        Store an insight in SEMANTIC memory.

        Args:
            content: Insight content
            importance: Memory importance (0.0-1.0)
            category: Insight category

        Returns:
            Memory ID if stored successfully
        """
        self._ensure_initialized()
        assert self._bridge is not None
        return await self._bridge.store_insight(content, importance, category)

    async def store_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str = "web_search"
    ) -> Optional[str]:
        """
        Store web search results in RESOURCE memory.

        Args:
            query: Search query
            results: List of search results
            source: Source identifier

        Returns:
            Memory ID if stored successfully
        """
        self._ensure_initialized()
        assert self._web_cache is not None
        return await self._web_cache.store_search(
            query=query,
            results=results,
            session_id=self.session_id,
            source=source
        )

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memories.

        Args:
            query: Search query
            limit: Maximum results
            memory_type: Optional filter (episodic, semantic, resource, etc.)

        Returns:
            List of matching memories
        """
        self._ensure_initialized()
        assert self._bridge is not None
        return await self._bridge.search_memories(query, limit, memory_type)

    def clear_session(self) -> None:
        """Clear session memory (keeps long-term memories intact)."""
        self._ensure_initialized()
        assert self._session is not None
        self._session.clear()
        logger.info("Session cleared: %s", self.session_id)

    def save_session(self) -> str:
        """
        Save session to disk.

        Returns:
            Path to saved file
        """
        self._ensure_initialized()
        assert self._session is not None
        return self._session.save_to_disk()

    def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from pending list."""
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

    async def wait_for_pending(self, timeout: float = 5.0) -> None:
        """
        Wait for all pending background tasks to complete.

        Args:
            timeout: Maximum wait time in seconds
        """
        self._cleanup_completed_tasks()
        if self._pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pending_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for pending tasks")
            self._pending_tasks.clear()

    async def close(self) -> None:
        """
        Close unified memory client and cleanup resources.

        Waits for pending tasks, saves session, and closes bridge.
        """
        if not self._initialized:
            return

        # Wait for pending background tasks
        await self.wait_for_pending(timeout=3.0)

        # Save session to disk
        if self._session is not None:
            self._session.save_to_disk()

        # Close bridge
        if self._bridge is not None:
            await self._bridge.close()

        logger.info("UnifiedMemoryClient closed: session=%s",
                    self._session.session_id if self._session else "unknown")

    def __repr__(self) -> str:
        if not self._initialized or self._session is None:
            return "UnifiedMemoryClient(not initialized)"
        return (
            f"UnifiedMemoryClient(session={self._session.session_id}, "
            f"turns={len(self._session)}, "
            f"pending_tasks={len(self._pending_tasks)})"
        )


# Singleton instance
_unified_memory_instance: Optional[UnifiedMemoryClient] = None


def get_unified_memory(
    session_id: Optional[str] = None,
    auto_start_service: bool = True
) -> UnifiedMemoryClient:
    """
    Get or create singleton unified memory client.

    Args:
        session_id: Optional session ID
        auto_start_service: Attempt to start episodic memory service if offline

    Returns:
        UnifiedMemoryClient instance
    """
    global _unified_memory_instance
    if _unified_memory_instance is None:
        _unified_memory_instance = UnifiedMemoryClient(
            session_id=session_id,
            auto_start_service=auto_start_service
        )
    return _unified_memory_instance


async def cleanup_unified_memory() -> None:
    """Cleanup unified memory client resources."""
    global _unified_memory_instance
    if _unified_memory_instance:
        await _unified_memory_instance.close()
        _unified_memory_instance = None
