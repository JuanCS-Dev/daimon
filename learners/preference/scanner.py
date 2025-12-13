"""
DAIMON Preference Scanner
=========================

Session discovery and message loading for preference analysis.

Responsible for:
- Discovering recent Claude Code sessions
- Loading messages from JSONL session files
- Integrating with ActivityStore for pre-computed signals

Architecture:
    SessionScanner is the first stage of the preference pipeline.
    It provides raw session data to the SignalDetector.

Usage:
    from learners.preference.scanner import SessionScanner
    
    scanner = SessionScanner()
    for session in scanner.scan_recent(since_hours=24):
        messages = scanner.load_messages(session)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .models import PreferenceSignal

logger = logging.getLogger("daimon.preference.scanner")

# Default Claude projects directory
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


class SessionScanner:
    """
    Scans and loads Claude Code session data.
    
    Provides two scanning strategies:
    1. ActivityStore integration (preferred, uses pre-computed signals)
    2. Direct file scanning (fallback, scans raw JSONL files)
    
    Attributes:
        projects_dir: Path to Claude projects directory
    """
    
    def __init__(self, projects_dir: Optional[Path] = None):
        """
        Initialize scanner.
        
        Args:
            projects_dir: Custom projects directory (default: ~/.claude/projects)
        """
        self.projects_dir = projects_dir or CLAUDE_PROJECTS
    
    def scan_recent(self, since_hours: int = 24) -> Generator[Path, None, None]:
        """
        Discover sessions modified within the given time window.
        
        Args:
            since_hours: How many hours back to scan
            
        Yields:
            Paths to session JSONL files
        """
        if not self.projects_dir.exists():
            logger.debug("Projects directory not found: %s", self.projects_dir)
            return
        
        cutoff = datetime.now().timestamp() - (since_hours * 3600)
        
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            # Skip hidden and special directories
            if project_dir.name.startswith("."):
                continue
            
            for jsonl in project_dir.glob("*.jsonl"):
                # Skip agent sidechains (focus on main sessions)
                if jsonl.name.startswith("agent-"):
                    continue
                
                try:
                    if jsonl.stat().st_mtime > cutoff:
                        yield jsonl
                except OSError:
                    continue
    
    def load_messages(self, session_file: Path) -> List[Dict[str, Any]]:
        """
        Load all messages from a session JSONL file.
        
        Args:
            session_file: Path to the JSONL session file
            
        Returns:
            List of message dictionaries
        """
        messages: List[Dict[str, Any]] = []
        
        try:
            with open(session_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except (OSError, IOError) as e:
            logger.warning("Failed to load session %s: %s", session_file, e)
        
        return messages
    
    def scan_from_activity_store(
        self,
        since_hours: int = 24,
    ) -> List[PreferenceSignal]:
        """
        Scan preference signals from ActivityStore.
        
        Uses pre-computed signals from claude_watcher if available.
        This is the preferred scanning method as it's more efficient.
        
        Args:
            since_hours: Hours to look back
            
        Returns:
            List of PreferenceSignals or empty list if unavailable
        """
        try:
            from memory.activity_store import get_activity_store
            
            store = get_activity_store()
            records = store.get_recent(
                watcher_type="claude",
                hours=since_hours,
                limit=1000,
            )
            
            signals = []
            for record in records:
                data = record.data
                preference_signal = data.get("preference_signal")
                
                if not preference_signal:
                    continue
                
                signals.append(PreferenceSignal(
                    timestamp=record.timestamp.isoformat(),
                    signal_type=preference_signal,
                    context=f"Project: {data.get('project', 'unknown')}",
                    category=data.get("preference_category", "general"),
                    strength=float(data.get("signal_strength", 0.5)),
                    session_id=data.get("project", "unknown"),
                    tool_involved=None,
                ))
            
            return signals
            
        except ImportError:
            logger.debug("ActivityStore not available")
            return []
        except Exception as e:
            logger.debug("ActivityStore scan failed: %s", e)
            return []
    
    def get_session_id(self, session_file: Path) -> str:
        """
        Extract session identifier from file path.
        
        Args:
            session_file: Path to session file
            
        Returns:
            Session ID (project name + file stem)
        """
        return f"{session_file.parent.name}/{session_file.stem}"
