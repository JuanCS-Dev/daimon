"""
MCP Server Configuration.
"""

import logging

# Service URLs
NOESIS_CONSCIOUSNESS_URL = "http://localhost:8001"  # maximus_core_service
NOESIS_REFLECTOR_URL = "http://localhost:8002"  # metacognitive_reflector
REQUEST_TIMEOUT = 30.0

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("daimon-mcp")
