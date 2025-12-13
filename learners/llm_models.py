"""
DAIMON LLM Service Models
=========================

Data models and cache for the LLM service.

Contains:
- ClassificationResult: Result of semantic classification
- InsightResult: Result of insight extraction
- CognitiveAnalysis: Result of cognitive state analysis
- LLMServiceStats: Statistics for monitoring
- LLMCache: TTL cache for LLM responses

Extracted from llm_service.py for CODE_CONSTITUTION compliance (< 500 lines).

Usage:
    from learners.llm_models import ClassificationResult, LLMCache

Follows CODE_CONSTITUTION: Clarity Over Cleverness.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ClassificationResult:
    """Result of semantic classification."""
    category: str
    confidence: float
    reasoning: str
    from_cache: bool = False
    from_llm: bool = True


@dataclass
class InsightResult:
    """Result of insight extraction."""
    insights: List[str]
    suggestions: List[str]
    confidence: float
    from_llm: bool = True


@dataclass
class CognitiveAnalysis:
    """Result of cognitive state analysis."""
    state: str
    confidence: float
    description: str
    recommendations: List[str]
    from_llm: bool = True


@dataclass
class LLMServiceStats:
    """Statistics for monitoring LLM service."""
    total_calls: int = 0
    cache_hits: int = 0
    llm_successes: int = 0
    llm_failures: int = 0
    heuristic_fallbacks: int = 0
    avg_latency_ms: float = 0.0
    last_call: Optional[datetime] = None


class LLMCache:
    """
    Simple TTL cache for LLM responses.

    Uses SHA256 hash of (method + args) as key.
    Default TTL: 5 minutes.
    Max entries: 1000 (with LRU eviction).
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds
            max_entries: Maximum cache entries
        """
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def _make_key(self, method: str, *args: Any) -> str:
        """Create cache key from method and arguments."""
        content = json.dumps([method, args], sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, method: str, *args: Any) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        key = self._make_key(method, *args)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self._cache[key]
        return None

    def set(self, method: str, value: Any, *args: Any) -> None:
        """Store value in cache."""
        if len(self._cache) >= self.max_entries:
            # Evict oldest entries (25%)
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])
            for k, _ in oldest[:len(self._cache) // 4]:
                del self._cache[k]

        key = self._make_key(method, *args)
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl,
        }
