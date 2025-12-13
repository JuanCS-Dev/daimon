"""
DAIMON Dashboard v2 - Interface de controle local.

Porta: 8003
Stack: FastAPI + HTML + Tailwind + Alpine.js (sem build)

Endpoints organizados em modulos:
- routes/status.py: status, preferences, collectors, backups, browser
- routes/corpus.py: corpus CRUD e search
- routes/memory.py: memory, precedents, activity
- routes/cognitive.py: style, cognitive state, metacognitive
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import status_router, corpus_router, memory_router, cognitive_router


# Setup paths
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# FastAPI app
app = FastAPI(
    title="DAIMON Dashboard",
    description="Interface de controle do exocortex pessoal",
    version="2.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
app.include_router(status_router)
app.include_router(corpus_router)
app.include_router(memory_router)
app.include_router(cognitive_router)


# === Main Route ===

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Central de Operacoes DAIMON."""
    return templates.TemplateResponse("ops.html", {
        "request": request,
        "title": "DAIMON OPS",
    })


# === Run ===

def run_dashboard(host: str = "127.0.0.1", port: int = 8003):
    """Inicia o dashboard."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
