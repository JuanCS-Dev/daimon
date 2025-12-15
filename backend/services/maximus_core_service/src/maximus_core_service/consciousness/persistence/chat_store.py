import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import uuid

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".noesis" / "chat_history.db"

class ChatStore:
    """SQLite-backed chat history storage with search capabilities."""

    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        try:
            # Sessions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            # Messages Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)
            # Full Text Search (FTS5) for messages
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
                USING fts5(content, session_id UNINDEXED)
            """)
            conn.commit()
        finally:
            conn.close()

    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        if not title:
            title = f"Chat {timestamp[:16]}" # Default title

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, timestamp, timestamp)
            )
            conn.commit()
            return session_id
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata and recent messages."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            # Get Session
            cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session = cur.fetchone()
            if not session:
                return None
            
            # Get Messages
            cur = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC", 
                (session_id,)
            )
            messages = [dict(row) for row in cur.fetchall()]
            
            s_dict = dict(session)
            s_dict["messages"] = messages
            return s_dict
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to a session."""
        timestamp = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            # Check if session exists, if not create (auto-vivify for legacy/lazy support)
            cur = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
            if not cur.fetchone():
                self.create_session_with_id(session_id)

            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp)
            )
            # Update Session Timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id)
            )
            # Update FTS Index
            conn.execute(
                "INSERT INTO messages_fts (session_id, content) VALUES (?, ?)",
                (session_id, content)
            )
            conn.commit()
        finally:
            conn.close()

    def create_session_with_id(self, session_id: str):
        """Helper to create session with specific ID (if client generated)."""
        timestamp = datetime.now().isoformat()
        title = f"Session {session_id[:8]}"
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, timestamp, timestamp)
            )
            conn.commit()
        finally:
            conn.close()

    def list_sessions(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List sessions ordered by update time."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Full text search across all chats."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            # Query FTS
            cur = conn.execute("""
                SELECT m.session_id, m.role, m.content, m.timestamp, s.title
                FROM messages_fts f
                JOIN messages m ON m.content = f.content AND m.session_id = f.session_id
                JOIN sessions s ON s.id = m.session_id
                WHERE f.content MATCH ?
                ORDER BY m.timestamp DESC
                LIMIT 20
            """, (query,))
            return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return []
        finally:
            conn.close()

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            # FTS cleanup needed? Technically yes but FTS5 is virtual.
            # Optimized deletion for FTS is complex, let it be for MVP.
            conn.commit()
        finally:
            conn.close()
