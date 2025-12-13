"""
DAIMON Precedent System.

Jurisprudence-style decision tracking with relevance feedback.

Features:
- Record decisions with context and outcomes
- Search by similarity and outcome
- Relevance increases with successful application
- Learn from both successes and failures

Usage:
    system = PrecedentSystem()
    pid = system.record(
        context="User asked for database optimization",
        decision="Used index-only scans",
        outcome="success",
        lesson="Index-only scans are 10x faster for read-heavy workloads"
    )
    results = system.search("database performance", outcome_filter="success")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .precedent_models import (
    DEFAULT_DB_PATH,
    OutcomeType,
    Precedent,
    PrecedentMatch,
    PrecedentMeta,
)

logger = logging.getLogger("daimon.precedent")


class PrecedentSystem:
    """
    Jurisprudence-style precedent tracking.

    Features:
    - Record decisions and outcomes
    - FTS5 search with outcome filtering
    - Relevance feedback loop
    - Learn from historical patterns
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize precedent system.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.daimon/memory/precedents.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precedents (
                    id TEXT PRIMARY KEY,
                    context TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    outcome TEXT DEFAULT 'unknown',
                    lesson TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    relevance REAL DEFAULT 0.5,
                    application_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    tags TEXT DEFAULT '[]'
                )
            """)

            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS precedents_fts
                USING fts5(context, decision, lesson, content=precedents, content_rowid=rowid)
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS precedents_ai AFTER INSERT ON precedents BEGIN
                    INSERT INTO precedents_fts(rowid, context, decision, lesson)
                    VALUES (new.rowid, new.context, new.decision, new.lesson);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS precedents_ad AFTER DELETE ON precedents BEGIN
                    INSERT INTO precedents_fts(precedents_fts, rowid, context, decision, lesson)
                    VALUES ('delete', old.rowid, old.context, old.decision, old.lesson);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS precedents_au AFTER UPDATE ON precedents BEGIN
                    INSERT INTO precedents_fts(precedents_fts, rowid, context, decision, lesson)
                    VALUES ('delete', old.rowid, old.context, old.decision, old.lesson);
                    INSERT INTO precedents_fts(rowid, context, decision, lesson)
                    VALUES (new.rowid, new.context, new.decision, new.lesson);
                END
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_precedents_outcome ON precedents(outcome)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_precedents_relevance ON precedents(relevance DESC)
            """)
            conn.commit()
            logger.debug("Precedent database initialized at %s", self.db_path)

    def _generate_id(self, context: str, decision: str) -> str:
        """Generate unique ID from context and decision hash."""
        combined = f"{context}:{decision}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def record(
        self,
        context: str,
        decision: str,
        outcome: OutcomeType = "unknown",
        meta: Optional[PrecedentMeta] = None,
    ) -> str:
        """
        Record a new precedent.

        Args:
            context: The situation/context for this decision
            decision: The decision that was made
            outcome: Result of the decision
            meta: Optional metadata (lesson, tags, relevance)

        Returns:
            Precedent ID
        """
        precedent_id = self._generate_id(context, decision)
        tags_list = meta.tags if meta else []
        relevance = meta.relevance if meta else 0.5
        lesson = ""  # Lesson is now in meta or via update_outcome
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM precedents WHERE id = ?", (precedent_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE precedents
                    SET updated_at = ?, application_count = application_count + 1,
                        relevance = MIN(1.0, relevance + 0.1),
                        outcome = CASE WHEN ? != 'unknown' THEN ? ELSE outcome END,
                        lesson = CASE WHEN ? != '' THEN ? ELSE lesson END
                    WHERE id = ?
                    """,
                    (now, outcome, outcome, lesson, lesson, precedent_id),
                )
                logger.debug("Precedent reinforced: %s", precedent_id)
            else:
                values = (
                    precedent_id, context, decision, outcome, lesson,
                    now, now, relevance, json.dumps(tags_list)
                )
                conn.execute(
                    """
                    INSERT INTO precedents
                    (id, context, decision, outcome, lesson,
                     created_at, updated_at, relevance, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                logger.debug("Precedent recorded: %s", precedent_id)
            conn.commit()

        return precedent_id

    def search(
        self,
        query: str,
        outcome_filter: Optional[OutcomeType] = None,
        min_relevance: float = 0.0,
        limit: int = 10,
    ) -> List[PrecedentMatch]:
        """
        Search precedents by content. Target latency: <20ms.

        Args:
            query: Search query
            outcome_filter: Filter by outcome type
            min_relevance: Minimum relevance score
            limit: Maximum results

        Returns:
            List of PrecedentMatch sorted by relevance
        """
        with sqlite3.connect(self.db_path) as conn:
            if outcome_filter:
                sql = """
                    SELECT p.*, bm25(precedents_fts) as score FROM precedents_fts f
                    JOIN precedents p ON f.rowid = p.rowid
                    WHERE precedents_fts MATCH ? AND p.outcome = ? AND p.relevance >= ?
                    ORDER BY p.relevance DESC, score LIMIT ?
                """
                params: Tuple[Any, ...] = (query, outcome_filter, min_relevance, limit)
            else:
                sql = """
                    SELECT p.*, bm25(precedents_fts) as score FROM precedents_fts f
                    JOIN precedents p ON f.rowid = p.rowid
                    WHERE precedents_fts MATCH ? AND p.relevance >= ?
                    ORDER BY p.relevance DESC, score LIMIT ?
                """
                params = (query, min_relevance, limit)

            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("FTS query failed: %s", e)
                return []

        results: List[PrecedentMatch] = []
        for row in rows:
            precedent = Precedent.from_row(row[:11])
            score = abs(row[11]) if len(row) > 11 else 0.5
            match = PrecedentMatch(
                precedent=precedent,
                score=score,
                match_reason=f"Matched '{query}'",
            )
            results.append(match)
        return results

    def update_outcome(self, precedent_id: str, outcome: OutcomeType, lesson: str = "") -> bool:
        """
        Update outcome of a precedent. Adjusts relevance based on success/failure.

        Args:
            precedent_id: Precedent to update
            outcome: New outcome
            lesson: Additional lesson learned

        Returns:
            True if updated, False if not found
        """
        now = datetime.now().isoformat()
        deltas = {"success": 0.15, "partial": 0.05, "failure": -0.1, "unknown": 0.0}
        relevance_delta = deltas.get(outcome, 0.0)

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                UPDATE precedents SET outcome = ?,
                    lesson = CASE WHEN ? != '' THEN lesson || ' | ' || ? ELSE lesson END,
                    updated_at = ?, relevance = MAX(0.1, MIN(1.0, relevance + ?)),
                    application_count = application_count + 1
                WHERE id = ?
                """,
                (outcome, lesson, lesson, now, relevance_delta, precedent_id),
            )
            conn.commit()
            if result.rowcount > 0:
                logger.info("Precedent %s updated to %s", precedent_id, outcome)
                return True
            return False

    def apply_precedent(self, precedent_id: str, was_successful: bool) -> bool:
        """
        Record application of a precedent. Updates success rate and relevance.

        Args:
            precedent_id: Precedent that was applied
            was_successful: Whether application was successful

        Returns:
            True if updated, False if not found
        """
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT application_count, success_rate FROM precedents WHERE id = ?",
                (precedent_id,),
            ).fetchone()

            if not row:
                return False

            app_count = row[0] + 1
            new_rate = row[1] * 0.9 + (1.0 if was_successful else 0.0) * 0.1
            relevance_delta = 0.1 if was_successful else -0.05

            conn.execute(
                """
                UPDATE precedents SET application_count = ?, success_rate = ?,
                    relevance = MAX(0.1, MIN(1.0, relevance + ?)), updated_at = ?
                WHERE id = ?
                """,
                (app_count, new_rate, relevance_delta, now, precedent_id),
            )
            conn.commit()
            logger.debug("Precedent %s applied (success=%s)", precedent_id, was_successful)
            return True

    def get_by_outcome(self, outcome: OutcomeType, limit: int = 20) -> List[Precedent]:
        """Get precedents by outcome type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM precedents WHERE outcome = ? ORDER BY relevance DESC LIMIT ?",
                (outcome, limit),
            ).fetchall()
        return [Precedent.from_row(row) for row in rows]

    def get_top_precedents(self, limit: int = 10) -> List[Precedent]:
        """Get highest relevance precedents."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM precedents ORDER BY relevance DESC, application_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Precedent.from_row(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get precedent system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM precedents"
            ).fetchone()[0]
            outcomes = conn.execute(
                "SELECT outcome, COUNT(*), AVG(relevance) "
                "FROM precedents GROUP BY outcome"
            ).fetchall()
            avg_relevance = conn.execute(
                "SELECT AVG(relevance) FROM precedents"
            ).fetchone()[0] or 0.0
            total_apps = conn.execute(
                "SELECT SUM(application_count) FROM precedents"
            ).fetchone()[0] or 0

        by_outcome = {
            out: {"count": cnt, "avg_relevance": round(avg or 0, 3)}
            for out, cnt, avg in outcomes
        }
        return {
            "total_precedents": total,
            "average_relevance": round(avg_relevance, 3),
            "total_applications": total_apps,
            "by_outcome": by_outcome,
            "db_path": str(self.db_path),
        }

    def delete(self, precedent_id: str) -> bool:
        """Delete a precedent."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM precedents WHERE id = ?", (precedent_id,))
            conn.commit()
            return result.rowcount > 0
