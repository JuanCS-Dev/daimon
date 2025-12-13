"""
DAIMON MCP Module - Modular MCP Server Components.

Exposes NOESIS consciousness to Claude Code via MCP tools.
"""

from .server import mcp
from .config import logger, NOESIS_CONSCIOUSNESS_URL, NOESIS_REFLECTOR_URL

# Import tools to register them with the mcp instance
from . import noesis_tools  # noqa: F401
from . import corpus_tools  # noqa: F401

__all__ = [
    "mcp",
    "logger",
    "NOESIS_CONSCIOUSNESS_URL",
    "NOESIS_REFLECTOR_URL",
]
