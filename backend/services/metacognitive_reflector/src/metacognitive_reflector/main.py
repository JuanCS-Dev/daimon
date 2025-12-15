"""
Metacognitive Reflector: Service Entry Point
============================================

FastAPI application for tribunal evaluation and metacognitive analysis.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

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
