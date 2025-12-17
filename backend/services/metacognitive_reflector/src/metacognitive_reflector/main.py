"""
Metacognitive Reflector: Service Entry Point
============================================

FastAPI application for tribunal evaluation and metacognitive analysis.
"""

from __future__ import annotations

import logging
import sys
import os
from contextlib import asynccontextmanager

# --- ANTI-ZOMBIE PROTECTION (Port 8101) ---
try:
    # Attempt to locate backend/services/shared dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path: src/metacognitive_reflector/main.py -> ../../../ -> services
    services_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    if os.path.isdir(os.path.join(services_dir, "shared")):
        if services_dir not in sys.path:
            sys.path.insert(0, services_dir)
        
        from shared.lifecycle import ensure_port_protection, install_signal_handlers
        ensure_port_protection(8101, "Metacognitive Reflector")
        install_signal_handlers()
except Exception as e:
    print(f"[WARNING] Anti-Zombie proctection failed: {e}")
# ------------------------------------------

from fastapi import FastAPI

from metacognitive_reflector.api.dependencies import initialize_service
from metacognitive_reflector.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize service components on startup."""
    initialize_service()
    yield


app = FastAPI(
    title="Metacognitive Reflector",
    description="MAXIMUS Tribunal evaluation and metacognitive analysis",
    version="3.0.0",
    lifespan=lifespan,
)

# Include the API router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
