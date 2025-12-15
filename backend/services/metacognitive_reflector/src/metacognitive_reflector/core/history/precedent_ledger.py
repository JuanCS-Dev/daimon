"""
NOESIS Memory Fortress - Precedent Ledger (G3 Integration)
============================================================

Provider for tribunal precedent storage and retrieval.

Enables the tribunal to learn from past decisions:
- Records full verdicts as precedents
- Enables semantic search for similar contexts
- Supports precedent-guided decision making

Storage Strategy (same as CriminalHistoryProvider):
- L2 (Redis): Fast hash-based access
- L4 (JSON): Complete record with checksums
- In-Memory Cache: Hot access
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from metacognitive_reflector.core.judges.voting import TribunalDecision

logger = logging.getLogger(__name__)


@dataclass
class Precedent:
    """
    A recorded tribunal decision that can inform future judgments.

    Based on G3 integration spec: PrecedentLedger.
    """

    id: str
    timestamp: str
    context_hash: str  # Hash of context for similarity search
    decision: str  # PASS, REVIEW, FAIL, CAPITAL
    consensus_score: float
    key_reasoning: str
    applicable_rules: List[str]  # Crime IDs or rule violations
    pillar_scores: Dict[str, float] = field(default_factory=dict)  # VERITAS, SOPHIA, DIKĒ
    context_keywords: List[str] = field(default_factory=list)  # For search

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "context_hash": self.context_hash,
            "decision": self.decision,
            "consensus_score": self.consensus_score,
            "key_reasoning": self.key_reasoning,
            "applicable_rules": self.applicable_rules,
            "pillar_scores": self.pillar_scores,
            "context_keywords": self.context_keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Precedent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            context_hash=data["context_hash"],
            decision=data["decision"],
            consensus_score=data["consensus_score"],
            key_reasoning=data.get("key_reasoning", ""),
            applicable_rules=data.get("applicable_rules", []),
            pillar_scores=data.get("pillar_scores", {}),
            context_keywords=data.get("context_keywords", []),
        )

    @classmethod
    def from_verdict(
        cls,
        verdict: Any,
        context_content: str,
        context_keywords: Optional[List[str]] = None,
    ) -> "Precedent":
        """
        Create precedent from a TribunalVerdict.

        Args:
            verdict: TribunalVerdict object
            context_content: String content that was evaluated
            context_keywords: Optional keywords for search
        """
        # Generate context hash (first 16 chars of SHA-256)
        context_hash = hashlib.sha256(
            context_content[:500].encode()
        ).hexdigest()[:16]

        # Extract pillar scores from individual verdicts
        pillar_scores = {}
        if hasattr(verdict, "individual_verdicts") and verdict.individual_verdicts:
            for judge_name, jv in verdict.individual_verdicts.items():
                if hasattr(jv, "score") and jv.score is not None:
                    pillar_scores[judge_name] = jv.score

        # Extract decision string
        decision = "UNKNOWN"
        if hasattr(verdict, "decision"):
            decision = verdict.decision.value if hasattr(verdict.decision, "value") else str(verdict.decision)

        # Generate ID
        precedent_id = f"prec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{context_hash[:8]}"

        return cls(
            id=precedent_id,
            timestamp=datetime.utcnow().isoformat(),
            context_hash=context_hash,
            decision=decision,
            consensus_score=getattr(verdict, "consensus_score", 0.0),
            key_reasoning=getattr(verdict, "reasoning", "")[:500],
            applicable_rules=getattr(verdict, "crimes_detected", []),
            pillar_scores=pillar_scores,
            context_keywords=context_keywords or [],
        )


class PrecedentLedgerProvider:
    """
    Provider for tribunal precedent storage and retrieval.

    G3 Integration: Enables learning from past decisions.

    Storage Strategy:
    - L2 (Redis): Fast hash-based access with HSET
    - L4 (JSON): Complete record with checksums
    - In-Memory Cache: Hot access

    Search Strategy:
    - Context hash prefix match (first 8 chars)
    - Keyword overlap scoring
    - Ordered by consensus_score (stronger precedents first)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        backup_path: str = "data/precedent_ledger.json",
        redis_prefix: str = "noesis:precedent:",
        max_cache_size: int = 1000,
    ) -> None:
        """
        Initialize provider.

        Args:
            redis_url: Redis connection URL (optional)
            backup_path: Path for JSON backup
            redis_prefix: Prefix for Redis keys
            max_cache_size: Maximum number of precedents in memory
        """
        self._redis_url = redis_url
        self._redis_prefix = redis_prefix
        self._backup_path = Path(backup_path)
        self._backup_path.parent.mkdir(parents=True, exist_ok=True)
        self._redis_client: Optional[Any] = None
        self._cache: Dict[str, Precedent] = {}
        self._max_cache_size = max_cache_size
        self._load_backup()

    async def _get_redis_client(self) -> Any:
        """Lazy initialize Redis client."""
        if self._redis_client is None and self._redis_url:
            try:
                import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel
                self._redis_client = await aioredis.from_url(
                    self._redis_url, encoding="utf-8", decode_responses=True,
                )
            except ImportError:
                logger.warning("redis package not available")
            except Exception as e:
                logger.warning("Failed to connect to Redis: %s", e)
        return self._redis_client

    def _load_backup(self) -> None:
        """Load data from JSON backup."""
        if not self._backup_path.exists():
            return

        try:
            with open(self._backup_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Verify checksum
            stored_checksum = data.get("_checksum")
            if stored_checksum:
                records = data.get("records", {})
                calculated = self._calculate_checksum(records)
                if calculated != stored_checksum:
                    logger.error("Precedent ledger backup checksum mismatch!")
                    return

            for prec_id, prec_data in data.get("records", {}).items():
                self._cache[prec_id] = Precedent.from_dict(prec_data)

            logger.info("Loaded %d precedents from backup", len(self._cache))
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load precedent ledger backup: %s", e)

    def _save_backup(self) -> None:
        """Save data to JSON backup with checksum."""
        records = {prec_id: prec.to_dict() for prec_id, prec in self._cache.items()}

        data = {
            "records": records,
            "_checksum": self._calculate_checksum(records),
            "_updated_at": datetime.now().isoformat(),
            "_version": "1.0",
        }

        temp_path = self._backup_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(self._backup_path)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def record_precedent(self, precedent: Precedent) -> Precedent:
        """
        Record a new precedent.

        Args:
            precedent: The precedent to record

        Returns:
            The recorded precedent
        """
        # Add to cache (evict oldest if full)
        if len(self._cache) >= self._max_cache_size:
            oldest_id = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp
            )
            del self._cache[oldest_id]

        self._cache[precedent.id] = precedent

        # Write to Redis
        try:
            client = await self._get_redis_client()
            if client:
                key = f"{self._redis_prefix}all"
                await client.hset(key, precedent.id, json.dumps(precedent.to_dict()))

                # Also index by context hash for faster lookup
                hash_key = f"{self._redis_prefix}hash:{precedent.context_hash[:8]}"
                await client.lpush(hash_key, precedent.id)
                await client.ltrim(hash_key, 0, 99)  # Keep last 100 per hash
        except Exception as e:
            logger.warning("Failed to write precedent to Redis: %s", e)

        # Persist to disk
        self._save_backup()
        logger.info("Precedent recorded: %s (%s)", precedent.id, precedent.decision)

        return precedent

    async def find_similar_precedents(  # pylint: disable=too-many-branches,too-many-nested-blocks
        self,
        context_content: str,
        keywords: Optional[List[str]] = None,
        limit: int = 5,
        min_consensus: float = 0.0,
    ) -> List[Precedent]:
        """
        Find precedents with similar context.

        Search strategy:
        1. Match by context hash prefix (first 8 chars)
        2. Score by keyword overlap
        3. Order by consensus_score (stronger precedents first)

        Args:
            context_content: The content to match against
            keywords: Optional keywords to match
            limit: Maximum number of results
            min_consensus: Minimum consensus score to include

        Returns:
            List of matching precedents, ordered by relevance
        """
        context_hash = hashlib.sha256(
            context_content[:500].encode()
        ).hexdigest()[:16]
        hash_prefix = context_hash[:8]

        matches: List[Precedent] = []
        keywords = keywords or []
        keywords_lower = [k.lower() for k in keywords]

        # Try Redis first
        redis_searched = False
        try:
            client = await self._get_redis_client()
            if client:
                # Search by hash prefix
                hash_key = f"{self._redis_prefix}hash:{hash_prefix}"
                prec_ids = await client.lrange(hash_key, 0, -1)

                if prec_ids:
                    all_key = f"{self._redis_prefix}all"
                    for prec_id in prec_ids:
                        prec_data = await client.hget(all_key, prec_id)
                        if prec_data:
                            prec = Precedent.from_dict(json.loads(prec_data))
                            if prec.consensus_score >= min_consensus:
                                matches.append(prec)
                    redis_searched = True
        except Exception as e:
            logger.debug("Redis search failed: %s", e)

        # Fallback to cache if Redis didn't work
        if not redis_searched:
            for prec in self._cache.values():
                if prec.context_hash[:8] == hash_prefix:
                    if prec.consensus_score >= min_consensus:
                        matches.append(prec)

        # If no exact hash matches, do keyword search in cache
        if not matches and keywords_lower:
            for prec in self._cache.values():
                prec_keywords = [k.lower() for k in prec.context_keywords]
                overlap = len(set(keywords_lower) & set(prec_keywords))
                if overlap > 0 and prec.consensus_score >= min_consensus:
                    matches.append(prec)

        # Score and sort
        def score_precedent(p: Precedent) -> float:
            base_score = p.consensus_score

            # Bonus for keyword overlap
            if keywords_lower:
                prec_keywords = [k.lower() for k in p.context_keywords]
                overlap = len(set(keywords_lower) & set(prec_keywords))
                base_score += overlap * 0.1

            # Bonus for exact hash match
            if p.context_hash[:8] == hash_prefix:
                base_score += 0.2

            return base_score

        matches.sort(key=score_precedent, reverse=True)
        return matches[:limit]

    async def get_precedent(self, precedent_id: str) -> Optional[Precedent]:
        """Get a specific precedent by ID."""
        if precedent_id in self._cache:
            return self._cache[precedent_id]

        try:
            client = await self._get_redis_client()
            if client:
                key = f"{self._redis_prefix}all"
                prec_data = await client.hget(key, precedent_id)
                if prec_data:
                    prec = Precedent.from_dict(json.loads(prec_data))
                    self._cache[precedent_id] = prec
                    return prec
        except Exception as e:
            logger.warning("Failed to get precedent from Redis: %s", e)

        return None

    async def get_precedents_by_decision(
        self,
        decision: str,
        limit: int = 10,
    ) -> List[Precedent]:
        """Get precedents by decision type (PASS, FAIL, REVIEW, CAPITAL)."""
        matches = [
            p for p in self._cache.values()
            if p.decision == decision
        ]
        matches.sort(key=lambda p: p.consensus_score, reverse=True)
        return matches[:limit]

    async def get_precedents_count(self) -> int:
        """Get total number of precedents."""
        return len(self._cache)

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        redis_healthy = False

        try:
            client = await self._get_redis_client()
            if client:
                await client.ping()
                redis_healthy = True
        except Exception:
            pass

        return {
            "healthy": True,
            "redis_available": redis_healthy,
            "backup_path": str(self._backup_path),
            "cached_precedents": len(self._cache),
            "max_cache_size": self._max_cache_size,
        }

    async def close(self) -> None:
        """Close connections."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None


# Convenience function for arbiter integration
def create_precedent_from_verdict(
    verdict: Any,
    execution_log: Any,
) -> Precedent:
    """
    Create a precedent from a tribunal verdict.

    Helper for EnsembleArbiter integration.

    Args:
        verdict: TribunalVerdict
        execution_log: The execution that was evaluated

    Returns:
        Precedent ready to be recorded
    """
    # Extract content from execution_log
    content = ""
    keywords = []

    if hasattr(execution_log, "content"):
        content = str(execution_log.content)[:500]
    elif isinstance(execution_log, dict):
        content = str(execution_log.get("content", ""))[:500]
        keywords = execution_log.get("keywords", [])
    elif hasattr(execution_log, "__dict__"):
        content = str(execution_log.__dict__)[:500]

    return Precedent.from_verdict(
        verdict=verdict,
        context_content=content,
        context_keywords=keywords,
    )
