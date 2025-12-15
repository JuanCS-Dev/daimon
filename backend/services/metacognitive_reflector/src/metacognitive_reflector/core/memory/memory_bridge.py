"""
NOESIS Memory Bridge - Session to Episodic Memory Sync
======================================================

Bridges SessionMemory (short-term) to Episodic Memory Service (long-term).
Ensures all conversations and insights are persisted hermetically.

Architecture:
    ┌─────────────────┐    async     ┌─────────────────┐
    │  noesis_chat.py │──────────────│ Episodic Memory │
    │  SessionMemory  │   Bridge     │  Service (L3)   │
    └─────────────────┘              └─────────────────┘

Features:
- Graceful degradation (no crash if service offline)
- Auto-start service if not running
- Async, non-blocking writes
- Connection pooling with httpx

Based on: Mem0 client patterns + MIRIX persistence
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EPISODIC_MEMORY_URL = "http://localhost:8102"
DEFAULT_TIMEOUT = 5.0


class MemoryBridge:
    """
    Bridges session memory to episodic memory service.

    Provides non-blocking, fault-tolerant persistence of conversation
    turns and insights to long-term storage.

    Usage:
        bridge = MemoryBridge()

        # Store a conversation turn
        memory_id = await bridge.store_turn(
            session_id="abc123",
            role="user",
            content="Hello, how are you?",
            importance=0.5
        )

        # Store an insight from self-reflection
        await bridge.store_insight(
            content="User prefers concise responses",
            importance=0.8,
            category="user_preference"
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        auto_start: bool = True
    ) -> None:
        """
        Initialize memory bridge.

        Args:
            base_url: Episodic memory service URL
            timeout: HTTP request timeout in seconds
            auto_start: Attempt to start service if not running
        """
        self.base_url = base_url or os.environ.get(
            "NOESIS_EPISODIC_MEMORY_URL",
            DEFAULT_EPISODIC_MEMORY_URL
        )
        self.timeout = timeout
        self.auto_start = auto_start
        self._client: Optional[Any] = None
        self._service_checked = False
        self._service_available = False

    async def _get_client(self) -> Any:
        """Get or create HTTP client (lazy initialization)."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    timeout=self.timeout,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                )
            except ImportError:
                logger.warning("httpx not installed, memory bridge disabled")
                return None
        return self._client

    async def _check_service(self) -> bool:
        """Check if episodic memory service is available."""
        if self._service_checked:
            return self._service_available

        client = await self._get_client()
        if not client:
            self._service_checked = True
            self._service_available = False
            return False

        try:
            response = await client.get(f"{self.base_url}/health")
            self._service_available = response.status_code == 200

            if self._service_available:
                logger.debug("Episodic memory service available at %s", self.base_url)
            else:
                logger.warning("Episodic memory service returned %d", response.status_code)

        except Exception as e:
            logger.debug("Episodic memory service not available: %s", e)
            self._service_available = False

            # Attempt auto-start if configured
            if self.auto_start:
                self._service_available = await self._try_auto_start()

        self._service_checked = True
        return self._service_available

    async def _try_auto_start(self) -> bool:
        """Attempt to auto-start the episodic memory service."""
        logger.info("Attempting to auto-start episodic memory service...")

        try:
            # Find project directory
            project_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
            service_dir = project_dir / "backend" / "services" / "episodic_memory"

            if not service_dir.exists():
                logger.warning("Episodic memory service directory not found: %s", service_dir)
                return False

            # Start service in background
            env = os.environ.copy()
            env["PYTHONPATH"] = str(service_dir / "src")

            subprocess.Popen(
                [
                    "python", "-m", "uvicorn",
                    "episodic_memory.api.routes:app",
                    "--host", "0.0.0.0",
                    "--port", "8102"
                ],
                cwd=str(service_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            # Wait for service to start
            await asyncio.sleep(3)

            # Verify service started
            client = await self._get_client()
            if client:
                try:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        logger.info("Episodic memory service auto-started successfully")
                        return True
                except Exception as e:
                    logger.debug("Health check after auto-start failed: %s", e)

            logger.warning("Episodic memory service failed to auto-start")
            return False

        except Exception as e:
            logger.error("Auto-start failed: %s", e)
            return False

    async def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        importance: float = 0.5,
        emotional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store a conversation turn as episodic memory.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            importance: Memory importance (0.0-1.0)
            emotional_context: Emotional metadata (VAD, primary_emotion, strategy)

        Returns:
            Memory ID if stored successfully, None otherwise
        """
        if not await self._check_service():
            logger.debug("Service unavailable, skipping turn storage")
            return None

        client = await self._get_client()
        if not client:
            return None

        try:
            # Format content with role prefix for searchability
            formatted_content = f"[{role.upper()}] {content}"

            # Build context with optional emotional metadata
            context: Dict[str, Any] = {
                "session_id": session_id,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "importance": importance,
                "source": "session_memory"
            }

            # Add emotional context if provided
            if emotional_context:
                context["emotional_context"] = emotional_context

            payload = {
                "content": formatted_content,
                "type": "episodic",
                "context": context
            }

            response = await client.post(
                f"{self.base_url}/v1/memories",
                json=payload
            )

            if response.status_code in (200, 201):
                data = response.json()
                memory_id = data.get("memory_id")
                logger.debug("Stored turn as memory: %s", memory_id)
                return memory_id
            else:
                logger.warning("Failed to store turn: %d", response.status_code)
                return None

        except Exception as e:
            logger.error("Error storing turn: %s", e)
            return None

    async def store_insight(
        self,
        content: str,
        importance: float,
        category: str = "self_awareness"
    ) -> Optional[str]:
        """
        Store a self-reflection insight as semantic memory.

        Args:
            content: Insight content
            importance: Memory importance (0.0-1.0)
            category: Insight category (self_awareness, user_preference, etc.)

        Returns:
            Memory ID if stored successfully, None otherwise
        """
        if not await self._check_service():
            logger.debug("Service unavailable, skipping insight storage")
            return None

        client = await self._get_client()
        if not client:
            return None

        try:
            # Format insight with category for searchability
            formatted_content = f"[INSIGHT/{category.upper()}] {content}"

            payload = {
                "content": formatted_content,
                "type": "semantic",
                "context": {
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                    "importance": importance,
                    "source": "self_reflection"
                }
            }

            response = await client.post(
                f"{self.base_url}/v1/memories",
                json=payload
            )

            if response.status_code in (200, 201):
                data = response.json()
                memory_id = data.get("memory_id")
                logger.debug("Stored insight as memory: %s", memory_id)
                return memory_id
            else:
                logger.warning("Failed to store insight: %d", response.status_code)
                return None

        except Exception as e:
            logger.error("Error storing insight: %s", e)
            return None

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> list:
        """
        Search memories by content.

        Args:
            query: Search query
            limit: Maximum results
            memory_type: Optional filter by type (episodic, semantic, etc.)

        Returns:
            List of matching memories
        """
        if not await self._check_service():
            return []

        client = await self._get_client()
        if not client:
            return []

        try:
            payload = {
                "query_text": query,
                "limit": limit
            }

            if memory_type:
                payload["type"] = memory_type

            response = await client.post(
                f"{self.base_url}/v1/memories/search",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("memories", [])
            else:
                logger.warning("Search failed: %d", response.status_code)
                return []

        except Exception as e:
            logger.error("Error searching memories: %s", e)
            return []

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def reset_service_check(self) -> None:
        """Reset service availability check (for retry after failure)."""
        self._service_checked = False
        self._service_available = False


# Singleton instance
_bridge_instance: Optional[MemoryBridge] = None


def get_memory_bridge() -> MemoryBridge:
    """Get or create singleton memory bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MemoryBridge()
    return _bridge_instance


async def cleanup_bridge() -> None:
    """Cleanup memory bridge resources."""
    global _bridge_instance
    if _bridge_instance:
        await _bridge_instance.close()
        _bridge_instance = None
