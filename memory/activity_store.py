#!/usr/bin/env python3
"""
DAIMON Activity Store - Time-Series Activity Data
=================================================

Stores heartbeat/activity data from collectors in time-series format.
Optimized for:
- Fast inserts (batch)
- Time-range queries
- Aggregation queries

Uses SQLite with partitioning by date for efficient cleanup.

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("daimon.activity")

# Default storage location
DEFAULT_DB_PATH = Path.home() / ".daimon" / "activity" / "activities.db"

# Retention settings
DEFAULT_RETENTION_DAYS = 7


@dataclass
class ActivityRecord:
    """
    Single activity record from a collector.

    Attributes:
        id: Unique record ID.
        timestamp: When activity occurred.
        watcher_type: Which collector generated this.
        data: Collector-specific payload.
    """
    id: Optional[int]
    timestamp: datetime
    watcher_type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "watcher_type": self.watcher_type,
            "data": self.data,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "ActivityRecord":
        """Create from database row."""
        return cls(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            watcher_type=row[2],
            data=json.loads(row[3]) if row[3] else {},
        )


@dataclass
class ActivitySummary:
    """
    Summary of activities over a time period.

    Attributes:
        watcher_type: Collector type.
        record_count: Number of records.
        first_activity: Earliest timestamp.
        last_activity: Latest timestamp.
        aggregated_data: Summarized metrics.
    """
    watcher_type: str
    record_count: int
    first_activity: Optional[datetime]
    last_activity: Optional[datetime]
    aggregated_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "watcher_type": self.watcher_type,
            "record_count": self.record_count,
            "first_activity": self.first_activity.isoformat() if self.first_activity else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "aggregated_data": self.aggregated_data,
        }


class ActivityStore:
    """
    Time-series store for activity data.

    Optimized for high-frequency writes and time-range queries.
    Uses SQLite with Write-Ahead Logging (WAL) for performance.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ):
        """
        Initialize activity store.

        Args:
            db_path: Path to SQLite database.
            retention_days: Days to keep data (default 7).
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.retention_days = retention_days
        self._ensure_db_initialized()

    def _ensure_db_initialized(self) -> None:
        """Initialize database and schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better write performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Create activities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    watcher_type TEXT NOT NULL,
                    data TEXT,
                    date_partition TEXT NOT NULL
                )
            """)

            # Indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_timestamp
                ON activities(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_watcher
                ON activities(watcher_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_partition
                ON activities(date_partition)
            """)

            conn.commit()

        logger.debug("Activity store initialized at %s", self.db_path)

    def add(self, watcher_type: str, timestamp: datetime, data: Dict[str, Any]) -> int:
        """
        Add single activity record.

        Args:
            watcher_type: Collector name.
            timestamp: When activity occurred.
            data: Activity payload.

        Returns:
            Record ID.
        """
        date_partition = timestamp.strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO activities (timestamp, watcher_type, data, date_partition)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp.isoformat(), watcher_type, json.dumps(data), date_partition),
            )
            conn.commit()
            return cursor.lastrowid

    def add_batch(
        self,
        batch_records: List[tuple[str, datetime, Dict[str, Any]]],
    ) -> int:
        """
        Add multiple activity records in a batch.

        Args:
            batch_records: List of (watcher_type, timestamp, data) tuples.

        Returns:
            Number of records inserted.
        """
        if not batch_records:
            return 0

        rows = [
            (ts.isoformat(), watcher, json.dumps(data), ts.strftime("%Y-%m-%d"))
            for watcher, ts, data in batch_records
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO activities (timestamp, watcher_type, data, date_partition)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        logger.debug("Added %d activity records", len(rows))
        return len(rows)

    def query(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        watcher_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ActivityRecord]:
        """Query activity records with optional filters."""
        conditions = []
        params = []

        if watcher_type:
            conditions.append("watcher_type = ?")
            params.append(watcher_type)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id, timestamp, watcher_type, data
                FROM activities
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        return [ActivityRecord.from_row(row) for row in rows]

    def get_recent(
        self,
        watcher_type: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[ActivityRecord]:
        """Get recent activity records within the last N hours."""
        start_time = datetime.now() - timedelta(hours=hours)
        return self.query(
            watcher_type=watcher_type,
            start_time=start_time,
            limit=limit,
        )

    def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, ActivitySummary]:
        """Get summary of activities by watcher type."""
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    watcher_type,
                    COUNT(*) as count,
                    MIN(timestamp) as first,
                    MAX(timestamp) as last
                FROM activities
                WHERE {where_clause}
                GROUP BY watcher_type
                """,
                params,
            ).fetchall()

        result = {}
        for watcher, count, first, last in rows:
            result[watcher] = ActivitySummary(
                watcher_type=watcher,
                record_count=count,
                first_activity=datetime.fromisoformat(first) if first else None,
                last_activity=datetime.fromisoformat(last) if last else None,
                aggregated_data={},
            )

        return result

    def aggregate_window_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Aggregate time spent per app from window_watcher data."""
        activity_records = self.query(
            watcher_type="window_watcher",
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        app_time: Dict[str, float] = {}
        for record in activity_records:
            app = record.data.get("app_name", "unknown")
            duration = record.data.get("focus_duration", 0.0)
            app_time[app] = app_time.get(app, 0.0) + duration

        return dict(sorted(app_time.items(), key=lambda x: x[1], reverse=True))

    def aggregate_domain_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Aggregate time spent per domain from browser_watcher data.

        Args:
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            Dict mapping domain to total seconds.
        """
        records = self.query(
            watcher_type="browser_watcher",
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        domain_time: Dict[str, float] = {}
        for record in records:
            domains = record.data.get("domains", [])
            for domain_data in domains:
                domain = domain_data.get("domain", "unknown")
                time_spent = domain_data.get("time_spent", 0.0)
                domain_time[domain] = domain_time.get(domain, 0.0) + time_spent

        return dict(sorted(domain_time.items(), key=lambda x: x[1], reverse=True))

    def cleanup(self, retention_days: Optional[int] = None) -> int:
        """
        Remove old activity records.

        Args:
            retention_days: Days to keep (uses default if None).

        Returns:
            Number of records deleted.
        """
        days = retention_days or self.retention_days
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM activities WHERE date_partition < ?",
                (cutoff,),
            )
            conn.commit()
            count = result.rowcount

        # VACUUM must run outside transaction
        with sqlite3.connect(self.db_path, isolation_level=None) as conn:
            conn.execute("VACUUM")

        if count > 0:
            logger.info("Cleaned up %d old activity records", count)

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get activity store statistics.

        Returns:
            Dict with counts and storage info.
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

            watchers = conn.execute(
                """
                SELECT watcher_type, COUNT(*)
                FROM activities
                GROUP BY watcher_type
                """
            ).fetchall()

            partitions = conn.execute(
                """
                SELECT date_partition, COUNT(*)
                FROM activities
                GROUP BY date_partition
                ORDER BY date_partition DESC
                LIMIT 7
                """
            ).fetchall()

        # Get database size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_records": total,
            "watchers": dict(watchers),
            "recent_partitions": dict(partitions),
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "retention_days": self.retention_days,
            "db_path": str(self.db_path),
        }


# Singleton storage (avoids global statement)
_singleton: Dict[str, Optional[ActivityStore]] = {"store": None}


def get_activity_store() -> ActivityStore:
    """Get global activity store instance."""
    if _singleton["store"] is None:
        _singleton["store"] = ActivityStore()
    return _singleton["store"]


if __name__ == "__main__":
    import time
    print("DAIMON Activity Store - Performance Test")
    test_db = Path("/tmp/daimon_activity_test.db")
    if test_db.exists():
        test_db.unlink()
    store = ActivityStore(test_db)
    # Batch insert test
    test_records = [
        ("window_watcher", datetime.now() - timedelta(minutes=i), {"app_name": f"app_{i % 10}"})
        for i in range(1000)
    ]
    start = time.perf_counter()
    store.add_batch(test_records)
    print(f"Insert 1000: {(time.perf_counter() - start) * 1000:.1f}ms")
    # Query test
    results = store.query(watcher_type="window_watcher", limit=100)
    print(f"Query results: {len(results)}")
    # Stats
    stats = store.get_stats()
    print(f"Total: {stats['total_records']}, Size: {stats['db_size_mb']}MB")
    test_db.unlink()
    print("Tests passed!")
