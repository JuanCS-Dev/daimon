#!/usr/bin/env python3
"""
DAIMON Claude Code Watcher - Session Intent Capture
====================================================

Monitors Claude Code sessions and extracts intention metadata.
Captures INTENT, not content - respects privacy.

Usage:
    python claude_watcher.py --daemon

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration
NOESIS_URL = os.getenv("NOESIS_URL", "http://localhost:8001")
CLAUDE_DIR = Path.home() / ".claude" / "projects"
POLL_INTERVAL_SECONDS = 5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("daimon-claude")

# Intent detection patterns (match user messages, not content)
INTENT_PATTERNS: Dict[str, List[str]] = {
    "create": ["create", "add", "implement", "build", "make", "generate", "write"],
    "fix": ["fix", "bug", "error", "broken", "issue", "problem", "crash"],
    "refactor": ["refactor", "clean", "reorganize", "restructure", "improve"],
    "understand": ["explain", "what", "how", "why", "understand", "help me"],
    "delete": ["delete", "remove", "drop", "destroy", "clean up"],
    "test": ["test", "verify", "check", "validate", "coverage"],
    "deploy": ["deploy", "release", "production", "publish", "ship"],
}

# Preference signal patterns (for PreferenceLearner integration)
APPROVAL_PATTERNS: List[str] = [
    r"\b(sim|yes|ok|perfeito|otimo|excelente|isso|gostei)\b",
    r"\b(aceito|aprovo|pode|manda|vai|bora|certo|correto)\b",
    r"^(s|y|ok|sim)$",
    r"(thumbs.?up|great|good|nice|awesome)",
]

REJECTION_PATTERNS: List[str] = [
    r"\b(nao|no|nope|errado|ruim|feio|pare|espera)\b",
    r"\b(rejeito|recuso|para|cancela|volta|desfaz)\b",
    r"\b(menos|mais simples|muito|demais|longo)\b",
    r"(thumbs.?down|bad|wrong|incorrect)",
]

# Category keywords for classification
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "code_style": ["formatacao", "estilo", "naming", "indent", "lint", "format", "style"],
    "verbosity": ["verboso", "longo", "curto", "resumo", "detalhado", "verbose", "brief"],
    "testing": ["teste", "test", "coverage", "mock", "assert", "spec", "unit"],
    "architecture": ["arquitetura", "estrutura", "pattern", "design", "refactor"],
    "documentation": ["doc", "comment", "readme", "docstring", "jsdoc"],
    "workflow": ["commit", "branch", "git", "deploy", "ci", "cd", "push"],
    "security": ["security", "auth", "password", "token", "secret", "vulnerability"],
    "performance": ["performance", "speed", "fast", "slow", "optimize", "cache"],
}


def detect_intention(message: str) -> str:
    """
    Detect intention category from message.

    Args:
        message: User message text.

    Returns:
        Intent category or "unknown".
    """
    message_lower = message.lower()

    for intent, keywords in INTENT_PATTERNS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return intent

    return "unknown"


def detect_preference_signal(message: str) -> Optional[str]:
    """
    Detect preference signal (approval/rejection) from message.

    Args:
        message: User message text.

    Returns:
        "approval", "rejection", or None.
    """
    message_lower = message.lower()

    # Check approval patterns
    for pattern in APPROVAL_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return "approval"

    # Check rejection patterns
    for pattern in REJECTION_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return "rejection"

    return None


def detect_category(message: str) -> str:
    """
    Detect category from message content.

    Args:
        message: User message text.

    Returns:
        Category string or "general".
    """
    message_lower = message.lower()
    scores: Dict[str, int] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in message_lower)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=lambda k: scores[k])

    return "general"


def calculate_signal_strength(message: str) -> float:
    """
    Calculate signal strength based on message length.

    Short, direct responses are stronger signals.

    Args:
        message: User message text.

    Returns:
        Strength 0.0-1.0.
    """
    word_count = len(message.split())

    if word_count <= 3:
        return 0.9  # "sim", "ok", "perfeito"
    if word_count <= 10:
        return 0.7  # Short response
    if word_count <= 30:
        return 0.5  # Medium response
    return 0.3  # Long response (diluted feedback)


def extract_files_touched(message: str) -> List[str]:
    """
    Extract file paths mentioned in message.

    Only extracts paths, not content.

    Args:
        message: Message text.

    Returns:
        List of file paths found.
    """
    # Match common file path patterns
    patterns = [
        r'[\'"`]([^\'"`]+\.[a-zA-Z]{2,4})[\'"`]',  # Quoted paths with extension
        r'(\S+\.[a-zA-Z]{2,4})\b',  # Unquoted paths with extension
    ]

    files: List[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, message)
        files.extend(matches)

    # Deduplicate and filter
    return list(set(f for f in files if "/" in f or "." in f))[:10]


class SessionTracker:  # pylint: disable=too-few-public-methods
    """
    Tracks Claude Code session state.

    Monitors JSONL files for new messages and extracts metadata.

    Note: Single public method (scan_projects) is the intended API.
    Internal methods handle the actual processing.
    """

    def __init__(self) -> None:
        """Initialize tracker with empty state."""
        self.positions: Dict[Path, int] = {}
        self.session_events: List[Dict[str, Any]] = []

    async def scan_projects(self) -> None:
        """
        Scan all project directories for session files.

        Monitors ~/.claude/projects/*/sessions/*.jsonl.
        """
        if not CLAUDE_DIR.exists():
            logger.debug("Claude directory not found: %s", CLAUDE_DIR)
            return

        for project_dir in CLAUDE_DIR.iterdir():
            if not project_dir.is_dir():
                continue

            sessions_dir = project_dir / "sessions"
            if not sessions_dir.exists():
                continue

            for jsonl_file in sessions_dir.glob("*.jsonl"):
                await self._process_file(jsonl_file, project_dir.name)

    async def _process_file(self, file_path: Path, project: str) -> None:
        """
        Process JSONL file for new messages.

        Args:
            file_path: Path to JSONL file.
            project: Project identifier.
        """
        # Get current position
        current_pos = self.positions.get(file_path, 0)

        try:
            file_size = file_path.stat().st_size
            if file_size <= current_pos:
                return

            with open(file_path, "r", encoding="utf-8") as f:
                f.seek(current_pos)
                new_lines = f.readlines()
                self.positions[file_path] = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    await self._process_entry(entry, project)
                except json.JSONDecodeError:
                    continue

        except (IOError, OSError) as exc:
            logger.warning("Error reading %s: %s", file_path, exc)

    async def _process_entry(self, entry: Dict[str, Any], project: str) -> None:
        """
        Process single JSONL entry.

        Extracts metadata without capturing content.
        Also detects preference signals for PreferenceLearner integration.

        Args:
            entry: Parsed JSON entry.
            project: Project identifier.
        """
        # Only process user messages
        role = entry.get("role", "")
        if role != "user":
            return

        # Extract content for detection (then discard content)
        content = entry.get("content", "")
        if not content:
            return

        # Detect intention
        intention = detect_intention(content)
        files_touched = extract_files_touched(content)

        # Detect preference signal (for PreferenceLearner)
        preference_signal = detect_preference_signal(content)
        category = detect_category(content) if preference_signal else "general"
        strength = calculate_signal_strength(content) if preference_signal else 0.0

        # Create event (no content stored)
        event = {
            "event_type": intention,
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "files_touched": files_touched,
            "intention": intention,
            # Preference signal data (for PreferenceLearner)
            "preference_signal": preference_signal,
            "preference_category": category,
            "signal_strength": strength,
        }

        self.session_events.append(event)
        logger.debug("Session event: %s (%s), signal: %s", intention, project, preference_signal)

        # Send to NOESIS and store locally
        await self._send_event(event)

    async def _send_event(self, event: Dict[str, Any]) -> None:
        """
        Send event to NOESIS and store locally.

        Args:
            event: Event metadata to send.
        """
        # Store locally in ActivityStore for StyleLearner/PreferenceLearner
        self._store_locally(event)

        # Send to NOESIS
        try:
            import httpx  # pylint: disable=import-outside-toplevel

            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    f"{NOESIS_URL}/api/daimon/claude/event",
                    json=event,
                )
        except ImportError:
            logger.debug("httpx not available")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to send event: %s", exc)

    def _store_locally(self, event: Dict[str, Any]) -> None:
        """
        Store event in local ActivityStore and feed StyleLearner.

        Includes preference signal data for PreferenceLearner integration.

        Args:
            event: Event metadata to store.
        """
        try:
            from memory.activity_store import get_activity_store  # pylint: disable=import-outside-toplevel

            store = get_activity_store()

            # Parse timestamp or use now
            ts_str = event.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                ts = datetime.now()

            data = {
                "intention": event.get("intention", "unknown"),
                "event_type": event.get("event_type", "unknown"),
                "project": event.get("project", ""),
                "files_touched": event.get("files_touched", []),
                # Preference signal data for PreferenceLearner
                "preference_signal": event.get("preference_signal"),
                "preference_category": event.get("preference_category", "general"),
                "signal_strength": event.get("signal_strength", 0.0),
            }

            store.add(
                watcher_type="claude",
                timestamp=ts,
                data=data,
            )

            # Also feed StyleLearner for communication pattern analysis
            self._feed_style_learner(data)

            logger.debug("Stored claude event locally: %s, signal: %s",
                        event.get("intention"), event.get("preference_signal"))
        except ImportError:
            logger.debug("ActivityStore not available, skipping local storage")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to store locally: %s", exc)

    def _feed_style_learner(self, data: Dict[str, Any]) -> None:
        """
        Feed claude data to StyleLearner for communication pattern analysis.

        Args:
            data: Claude event data dict.
        """
        try:
            from learners import get_style_learner  # pylint: disable=import-outside-toplevel
            learner = get_style_learner()
            learner.add_claude_sample(data)
        except ImportError:
            pass  # StyleLearner not available
        except Exception:  # pylint: disable=broad-except
            pass  # Don't fail if StyleLearner has issues

    async def run(self) -> None:
        """
        Run the tracker loop.

        Polls Claude session files at regular intervals.
        Used by daimon_daemon.py for unified startup.
        """
        logger.info("DAIMON Claude Watcher started")
        logger.info("Monitoring: %s", CLAUDE_DIR)

        try:
            while True:
                await self.scan_projects()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("DAIMON Claude Watcher stopped")


async def run_daemon() -> None:
    """
    Run the watcher daemon.

    Polls Claude session files at regular intervals.
    """
    tracker = SessionTracker()
    logger.info("DAIMON Claude Watcher started")
    logger.info("Monitoring: %s", CLAUDE_DIR)

    try:
        while True:
            await tracker.scan_projects()
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        logger.info("DAIMON Claude Watcher stopped")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DAIMON Claude Code Watcher")
    parser.add_argument("--daemon", action="store_true", help="Start daemon mode")
    args = parser.parse_args()

    if args.daemon:
        try:
            asyncio.run(run_daemon())
        except KeyboardInterrupt:
            logger.info("DAIMON Claude Watcher stopped")
        return

    # Default: show usage
    parser.print_help()


if __name__ == "__main__":
    main()
