#!/usr/bin/env python3
"""
Prepare Noesis memories for Constitutional AI training.

This module extracts, enriches, and exports memories from the Memory Fortress
into formats suitable for fine-tuning with Constitutional AI methodology.

The preparation includes:
- Loading memories from JSON backup
- Extracting entities from content
- Populating emotional context (VAD scores)
- Exporting to JSONL and dialogue formats
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_BACKUP = PROJECT_ROOT / "data" / "memory" / "memories_backup.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "memories"

# Emotional detection keywords
POSITIVE_KEYWORDS = frozenset({
    "obrigado", "excelente", "ótimo", "perfeito", "legal", "bom",
    "maravilhoso", "incrível", "fantástico", "agradeço", "parabéns"
})
NEGATIVE_KEYWORDS = frozenset({
    "erro", "problema", "falha", "bug", "difícil", "ruim",
    "péssimo", "horrível", "fracasso", "impossível", "frustração"
})
HIGH_AROUSAL_KEYWORDS = frozenset({
    "urgente", "importante", "crítico", "agora", "rápido",
    "emergência", "imediato", "pressa", "já"
})


@dataclass
class EmotionalContext:
    """VAD (Valence-Arousal-Dominance) emotional context."""

    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    primary_emotion: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "primary_emotion": self.primary_emotion
        }


@dataclass
class PreparedMemory:
    """A memory prepared for training."""

    memory_id: str
    memory_type: str
    content: str
    context: Dict[str, Any]
    importance: float
    timestamp: str
    entities: List[str] = field(default_factory=list)
    emotional_context: Optional[EmotionalContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "memory_id": self.memory_id,
            "type": self.memory_type,
            "content": self.content,
            "context": self.context,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "entities": self.entities,
        }
        if self.emotional_context:
            result["emotional_context"] = self.emotional_context.to_dict()
        return result


class MemoryPreparator:
    """Prepares memories for Constitutional AI training."""

    def __init__(
        self,
        memory_path: Path = MEMORY_BACKUP,
        output_dir: Path = OUTPUT_DIR
    ) -> None:
        """
        Initialize the preparator.

        Args:
            memory_path: Path to memory backup JSON file
            output_dir: Directory for output files
        """
        self.memory_path = memory_path
        self.output_dir = output_dir
        self._memories: List[PreparedMemory] = []

    def load_memories(self) -> int:
        """
        Load memories from JSON backup.

        Returns:
            Number of memories loaded

        Raises:
            FileNotFoundError: If memory backup file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        if not self.memory_path.exists():
            raise FileNotFoundError(
                f"Memory backup not found: {self.memory_path}. "
                f"Run './noesis remember' first to create backup."
            )

        with open(self.memory_path, encoding="utf-8") as f:
            data = json.load(f)

        raw_memories = data.get("memories", [])
        logger.info(
            "Loaded %d memories from %s (saved: %s)",
            len(raw_memories),
            self.memory_path.name,
            data.get("saved_at", "unknown")
        )

        self._memories = [
            PreparedMemory(
                memory_id=m["memory_id"],
                memory_type=m.get("type", "unknown"),
                content=m["content"],
                context=m.get("context", {}),
                importance=m.get("importance", 0.5),
                timestamp=m.get("timestamp", datetime.now().isoformat()),
            )
            for m in raw_memories
        ]

        return len(self._memories)

    def extract_entities(self) -> int:
        """
        Extract entities from memory content.

        Uses simple keyword extraction. For production, integrate
        with spaCy or similar NER library.

        Returns:
            Total entities extracted
        """
        total_entities = 0

        # Simple entity patterns (nouns, proper nouns from content)
        for mem in self._memories:
            entities = self._extract_simple_entities(mem.content)
            mem.entities = entities
            total_entities += len(entities)

        logger.info("Extracted %d entities from %d memories",
                    total_entities, len(self._memories))
        return total_entities

    def _extract_simple_entities(self, content: str) -> List[str]:
        """
        Extract entities using simple pattern matching.

        Args:
            content: Text to extract entities from

        Returns:
            List of extracted entity strings
        """
        entities = []

        # Extract words that start with uppercase (potential proper nouns)
        words = content.split()
        for word in words:
            cleaned = word.strip(".,!?()[]{}\"'")
            if len(cleaned) > 2 and cleaned[0].isupper():
                # Skip common sentence starters
                if cleaned.lower() not in {"estou", "você", "isso", "aqui", "qual"}:
                    entities.append(cleaned)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique[:10]  # Limit to top 10 entities

    def populate_emotional_context(self) -> int:
        """
        Populate emotional context for memories without it.

        Uses keyword-based VAD (Valence-Arousal-Dominance) detection.

        Returns:
            Number of memories with emotional context populated
        """
        populated = 0

        for mem in self._memories:
            # Skip if already has emotional context
            existing = mem.context.get("emotional_context")
            if existing and isinstance(existing, dict) and existing.get("valence"):
                mem.emotional_context = EmotionalContext(
                    valence=existing.get("valence", 0.5),
                    arousal=existing.get("arousal", 0.5),
                    dominance=existing.get("dominance", 0.5),
                    primary_emotion=existing.get("primary_emotion", "neutral")
                )
                continue

            # Calculate VAD from content
            emotional_ctx = self._calculate_vad(mem.content)
            mem.emotional_context = emotional_ctx
            populated += 1

        logger.info("Populated emotional context for %d memories", populated)
        return populated

    def _calculate_vad(self, content: str) -> EmotionalContext:
        """
        Calculate VAD scores from content using keyword matching.

        Args:
            content: Text to analyze

        Returns:
            EmotionalContext with calculated VAD scores
        """
        content_lower = content.lower()
        words = set(content_lower.split())

        # Valence: positive vs negative
        positive_count = len(words & POSITIVE_KEYWORDS)
        negative_count = len(words & NEGATIVE_KEYWORDS)

        if positive_count > negative_count:
            valence = min(0.5 + (positive_count * 0.1), 0.9)
            primary_emotion = "joy"
        elif negative_count > positive_count:
            valence = max(0.5 - (negative_count * 0.1), 0.1)
            primary_emotion = "frustration"
        else:
            valence = 0.5
            primary_emotion = "neutral"

        # Arousal: urgency/intensity
        arousal_count = len(words & HIGH_AROUSAL_KEYWORDS)
        arousal = min(0.5 + (arousal_count * 0.15), 0.9)

        # Dominance: default to neutral
        dominance = 0.5

        return EmotionalContext(
            valence=round(valence, 2),
            arousal=round(arousal, 2),
            dominance=round(dominance, 2),
            primary_emotion=primary_emotion
        )

    def export_jsonl(self) -> Path:
        """
        Export memories to JSONL format.

        Returns:
            Path to exported file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "memories.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for mem in self._memories:
                f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")

        logger.info("Exported JSONL: %s (%d memories)", output_path, len(self._memories))
        return output_path

    def export_dialogues(self) -> Path:
        """
        Export memories as dialogue pairs for fine-tuning.

        Groups memories by session_id and orders by timestamp.

        Returns:
            Path to exported file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "dialogues.json"

        # Group by session
        sessions: Dict[str, List[Dict[str, Any]]] = {}

        for mem in self._memories:
            session_id = mem.context.get("session_id", "unknown")
            if session_id not in sessions:
                sessions[session_id] = []

            sessions[session_id].append({
                "role": mem.context.get("role", "unknown"),
                "content": mem.content,
                "timestamp": mem.timestamp
            })

        # Sort each session by timestamp and format
        dialogues = []
        for session_id, messages in sessions.items():
            sorted_messages = sorted(messages, key=lambda x: x["timestamp"])
            dialogue = {
                "session_id": session_id,
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in sorted_messages
                ]
            }
            dialogues.append(dialogue)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)

        logger.info("Exported dialogues: %s (%d sessions)", output_path, len(dialogues))
        return output_path

    def export_prepared_backup(self) -> Path:
        """
        Export full prepared backup with all enrichments.

        Returns:
            Path to exported file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "memories_prepared.json"

        output = {
            "version": "2.0-training-ready",
            "prepared_at": datetime.now().isoformat(),
            "count": len(self._memories),
            "memories": [m.to_dict() for m in self._memories]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Exported prepared backup: %s", output_path)
        return output_path

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prepared memories.

        Returns:
            Dictionary with memory statistics
        """
        if not self._memories:
            return {"count": 0, "error": "No memories loaded"}

        types = {}
        roles = {}
        total_entities = 0
        with_emotional = 0

        for mem in self._memories:
            types[mem.memory_type] = types.get(mem.memory_type, 0) + 1
            role = mem.context.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            total_entities += len(mem.entities)
            if mem.emotional_context:
                with_emotional += 1

        return {
            "count": len(self._memories),
            "types": types,
            "roles": roles,
            "total_entities": total_entities,
            "avg_entities_per_memory": round(total_entities / len(self._memories), 2),
            "with_emotional_context": with_emotional,
            "emotional_coverage": f"{(with_emotional / len(self._memories)) * 100:.1f}%"
        }


async def main() -> int:
    """
    Main entry point for memory preparation.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 60)
    print("NOESIS MEMORY PREPARATION FOR TRAINING")
    print("=" * 60)

    preparator = MemoryPreparator()

    try:
        # Step 1: Load
        print("\n[STEP 1] Loading memories...")
        count = preparator.load_memories()
        print(f"  Loaded {count} memories")

        # Step 2: Extract entities
        print("\n[STEP 2] Extracting entities...")
        entities = preparator.extract_entities()
        print(f"  Extracted {entities} entities")

        # Step 3: Populate emotional context
        print("\n[STEP 3] Populating emotional context...")
        populated = preparator.populate_emotional_context()
        print(f"  Populated {populated} memories with emotional context")

        # Step 4: Export formats
        print("\n[STEP 4] Exporting training formats...")
        jsonl_path = preparator.export_jsonl()
        dialogue_path = preparator.export_dialogues()
        backup_path = preparator.export_prepared_backup()

        print(f"  - {jsonl_path}")
        print(f"  - {dialogue_path}")
        print(f"  - {backup_path}")

        # Step 5: Statistics
        print("\n[STEP 5] Statistics...")
        stats = preparator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
