"""
DAIMON MCP Server - Personal Exocortex Interface
=================================================

Exposes NOESIS consciousness to Claude Code via 8 MCP tools.

NOESIS Tools:
- noesis_consult: Maieutic questioning (returns questions, not answers)
- noesis_tribunal: Ethical judgment (3-judge verdict)
- noesis_precedent: Search past decisions for guidance
- noesis_confront: Socratic confrontation of premises
- noesis_health: Check NOESIS services status

Corpus Tools:
- corpus_search: Semantic search over wisdom texts
- corpus_add: Add new knowledge entries
- corpus_stats: View corpus statistics

Usage:
    claude mcp add daimon-consciousness -- python /path/to/mcp_server.py

Follows FastMCP 2.0 best practices (December 2025):
- MCP Annotations for behavioral hints
- ToolError for user-facing errors
- Context for logging and progress
- Field constraints for validation
- Annotated types for parameter descriptions
"""

# Re-export everything for backwards compatibility
from .mcp_tools import mcp, logger, NOESIS_CONSCIOUSNESS_URL, NOESIS_REFLECTOR_URL
from .mcp_tools.config import REQUEST_TIMEOUT
from .mcp_tools.http_utils import http_post as _http_post, http_get as _http_get
from .mcp_tools.noesis_tools import (
    noesis_consult,
    noesis_tribunal,
    noesis_precedent,
    noesis_confront,
    noesis_health,
)
from .mcp_tools.corpus_tools import (
    corpus_search,
    corpus_add,
    corpus_stats,
)

__all__ = [
    "mcp",
    "logger",
    "NOESIS_CONSCIOUSNESS_URL",
    "NOESIS_REFLECTOR_URL",
    "REQUEST_TIMEOUT",
    "_http_post",
    "_http_get",
    "noesis_consult",
    "noesis_tribunal",
    "noesis_precedent",
    "noesis_confront",
    "noesis_health",
    "corpus_search",
    "corpus_add",
    "corpus_stats",
]


def main() -> None:
    """Main entry point for the DAIMON MCP Server."""
    logger.info("Starting DAIMON MCP Server...")
    logger.info("Consciousness URL: %s", NOESIS_CONSCIOUSNESS_URL)
    logger.info("Reflector URL: %s", NOESIS_REFLECTOR_URL)
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
