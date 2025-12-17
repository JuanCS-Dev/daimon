#!/usr/bin/env python3
"""
NOESIS Remember - Query Past Memories
======================================

CLI tool to search and retrieve memories from Noesis's long-term memory.

Usage:
    noesis remember "what do I know about X?"
    noesis remember --type episodic "conversations about Y"
    noesis remember --limit 20 "Z"

Returns memories formatted for human readability.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
SERVICES_DIR = PROJECT_DIR / "backend" / "services"
sys.path.insert(0, str(SERVICES_DIR / "metacognitive_reflector" / "src"))
sys.path.insert(0, str(SERVICES_DIR / "episodic_memory" / "src"))


class Colors:
    """Terminal colors."""
    C = '\033[0;36m'   # Cyan
    G = '\033[0;32m'   # Green
    R = '\033[0;31m'   # Red
    Y = '\033[1;33m'   # Yellow
    M = '\033[0;35m'   # Magenta
    W = '\033[1;37m'   # White
    D = '\033[0;90m'   # Dim
    E = '\033[0m'      # End


EPISODIC_MEMORY_URL = os.environ.get(
    "NOESIS_EPISODIC_MEMORY_URL",
    "http://localhost:8102"
)


async def search_memories(
    query: str,
    limit: int = 10,
    memory_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search memories via Episodic Memory Service API.

    Args:
        query: Search query
        limit: Maximum results
        memory_type: Optional type filter

    Returns:
        List of memory dictionaries
    """
    try:
        import httpx
    except ImportError:
        print(f"{Colors.R}Error: httpx not installed{Colors.E}")
        print(f"  Run: pip install httpx")
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "query_text": query,
                "limit": limit
            }

            if memory_type:
                payload["type"] = memory_type

            response = await client.post(
                f"{EPISODIC_MEMORY_URL}/v1/memories/search",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("memories", [])
            else:
                print(f"{Colors.Y}Warning: Search returned {response.status_code}{Colors.E}")
                return []

    except httpx.ConnectError:
        print(f"{Colors.Y}Note: Episodic memory service not running{Colors.E}")
        print(f"  Start with: noesis wake")
        return []
    except Exception as e:
        print(f"{Colors.R}Error searching memories: {e}{Colors.E}")
        return []


def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return timestamp[:16] if timestamp else "unknown"


def format_memory(memory: Dict[str, Any], index: int) -> str:
    """Format a memory for display."""
    content = memory.get("content", "")
    mem_type = memory.get("type", "unknown")
    timestamp = format_timestamp(memory.get("timestamp", ""))
    importance = memory.get("importance", 0.0)
    memory_id = memory.get("memory_id", "")[:8]

    # Type color
    type_colors = {
        "episodic": Colors.C,
        "semantic": Colors.G,
        "resource": Colors.Y,
        "core": Colors.M,
        "vault": Colors.W,
        "procedural": Colors.C,
    }
    type_color = type_colors.get(mem_type, Colors.D)

    # Truncate content for display
    max_content_len = 300
    if len(content) > max_content_len:
        content = content[:max_content_len] + "..."

    lines = [
        f"{Colors.D}[{index}]{Colors.E} {type_color}{mem_type.upper()}{Colors.E} "
        f"│ {timestamp} │ importance: {importance:.2f} │ id: {memory_id}",
        f"    {content}"
    ]

    return "\n".join(lines)


async def list_sessions() -> List[Dict[str, Any]]:
    """List recent sessions from session memory files."""
    from metacognitive_reflector.core.memory.session import SESSION_DIR, SessionMemory

    sessions: List[Dict[str, Any]] = []
    session_dir = Path(SESSION_DIR)

    if not session_dir.exists():
        return sessions

    for filepath in sorted(session_dir.glob("session_*.json"), reverse=True)[:10]:
        try:
            session = SessionMemory.load_from_disk(
                filepath.stem.replace("session_", ""),
                str(session_dir)
            )
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "turns": len(session.turns),
                    "created_at": session.created_at,
                    "has_summary": bool(session.summary)
                })
        except Exception:
            pass

    return sessions


async def get_stats() -> Dict[str, Any]:
    """Get memory store statistics."""
    try:
        import httpx
    except ImportError:
        return {}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{EPISODIC_MEMORY_URL}/v1/memories/stats")
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass

    return {}


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Search Noesis's long-term memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  noesis remember "conversations about philosophy"
  noesis remember --type semantic "insights about authenticity"
  noesis remember --type episodic --limit 20 "Juan"
  noesis remember --sessions
  noesis remember --stats
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["episodic", "semantic", "resource", "core", "vault", "procedural"],
        help="Filter by memory type"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum results (default: 10)"
    )
    parser.add_argument(
        "--sessions", "-s",
        action="store_true",
        help="List recent sessions"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show memory statistics"
    )

    args = parser.parse_args()

    # Print header
    print(f"\n{Colors.M}NOESIS MEMORY{Colors.E}")
    print(f"{Colors.D}{'─' * 50}{Colors.E}\n")

    # Handle --sessions
    if args.sessions:
        sessions = await list_sessions()
        if sessions:
            print(f"{Colors.C}Recent Sessions:{Colors.E}\n")
            for s in sessions:
                created = format_timestamp(s.get("created_at", ""))
                print(f"  {Colors.G}{s['session_id']}{Colors.E} │ "
                      f"{s['turns']} turns │ {created} │ "
                      f"summary: {'yes' if s['has_summary'] else 'no'}")
            print()
        else:
            print(f"{Colors.D}No sessions found{Colors.E}\n")
        return 0

    # Handle --stats
    if args.stats:
        stats = await get_stats()
        if stats:
            print(f"{Colors.C}Memory Statistics:{Colors.E}\n")
            print(f"  Total memories: {Colors.W}{stats.get('total_memories', 0)}{Colors.E}")
            by_type = stats.get("by_type", {})
            if by_type:
                print(f"\n  By type:")
                for mtype, count in by_type.items():
                    print(f"    {mtype}: {count}")
            avg_importance = stats.get("avg_importance", 0)
            print(f"\n  Average importance: {avg_importance:.3f}")
            print()
        else:
            print(f"{Colors.Y}Could not retrieve stats (service offline?){Colors.E}\n")
        return 0

    # Handle search query
    if not args.query:
        parser.print_help()
        return 1

    print(f"Searching: {Colors.C}{args.query}{Colors.E}")
    if args.type:
        print(f"Type filter: {Colors.Y}{args.type}{Colors.E}")
    print()

    memories = await search_memories(
        query=args.query,
        limit=args.limit,
        memory_type=args.type
    )

    if not memories:
        print(f"{Colors.D}No memories found matching your query.{Colors.E}\n")
        return 0

    print(f"Found {Colors.G}{len(memories)}{Colors.E} memories:\n")

    for i, memory in enumerate(memories, 1):
        print(format_memory(memory, i))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
