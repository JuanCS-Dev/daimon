"""
DAIMON Dashboard - Status and system endpoints.

Includes: status, preferences, reflect, claude-md, collectors, backups.
"""

import subprocess
from pathlib import Path

from fastapi import APIRouter

from ..helpers import NOESIS_URL, REFLECTOR_URL, check_service, check_socket, check_process
from ..models import ClaudeMdUpdate


router = APIRouter(tags=["status"])


@router.get("/api/status")
async def get_status():
    """Status de todos os servicos."""
    status = {
        "noesis": await check_service(f"{NOESIS_URL}/v1/health"),
        "reflector": await check_service(f"{REFLECTOR_URL}/health"),
        "shell_watcher": "active" if check_socket() else "inactive",
        "dashboard": "healthy",
    }
    return status


@router.get("/api/preferences")
async def get_preferences():
    """Preferencias aprendidas e status do engine."""
    try:
        from learners.reflection_engine import get_engine
        engine = get_engine()
        return engine.get_status()
    except ImportError:
        return {"error": "Reflection engine not available"}


@router.post("/api/reflect")
async def trigger_reflection():
    """Dispara reflexao manual."""
    try:
        from learners.reflection_engine import get_engine
        engine = get_engine()
        result = await engine.reflect()
        return {"status": "completed", **result}
    except ImportError:
        return {"error": "Reflection engine not available"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/claude-md")
async def get_claude_md():
    """Retorna conteudo do CLAUDE.md."""
    path = Path.home() / ".claude" / "CLAUDE.md"

    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
            return {"content": content, "exists": True, "path": str(path)}
        except Exception as e:
            return {"content": "", "exists": True, "error": str(e)}

    return {"content": "", "exists": False, "path": str(path)}


@router.put("/api/claude-md")
async def update_claude_md(data: ClaudeMdUpdate):
    """Atualiza CLAUDE.md (cuidado: sobrescreve tudo)."""
    path = Path.home() / ".claude" / "CLAUDE.md"

    try:
        # Backup via ConfigRefiner se disponivel
        try:
            from actuators.config_refiner import ConfigRefiner
            refiner = ConfigRefiner()
            refiner._create_backup()
        except ImportError:
            pass

        # Escrever conteudo
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.content, encoding="utf-8")

        return {"status": "updated", "path": str(path)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/collectors")
async def get_collectors():
    """Status dos collectors."""
    return {
        "shell_watcher": {
            "socket_exists": check_socket(),
            "process_running": check_process("shell_watcher"),
        },
        "claude_watcher": {
            "process_running": check_process("claude_watcher"),
        },
    }


@router.post("/api/collectors/{name}/start")
async def start_collector(name: str):
    """Inicia um collector."""
    valid_collectors = ["shell_watcher", "claude_watcher"]

    if name not in valid_collectors:
        return {"error": f"Unknown collector: {name}"}

    try:
        # Encontrar script do collector
        daimon_dir = Path(__file__).parent.parent.parent
        script = daimon_dir / "collectors" / f"{name}.py"

        if not script.exists():
            return {"error": f"Collector script not found: {script}"}

        # Iniciar em background
        subprocess.Popen(
            ["python3", str(script)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {"status": "started", "collector": name}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/collectors/{name}/stop")
async def stop_collector(name: str):
    """Para um collector."""
    valid_collectors = ["shell_watcher", "claude_watcher"]

    if name not in valid_collectors:
        return {"error": f"Unknown collector: {name}"}

    try:
        subprocess.run(
            ["pkill", "-f", f"collectors/{name}"],
            capture_output=True,
        )
        return {"status": "stopped", "collector": name}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/backups")
async def get_backups():
    """Lista backups do CLAUDE.md."""
    try:
        from actuators.config_refiner import ConfigRefiner
        refiner = ConfigRefiner()
        return {"backups": refiner.get_backup_list()}
    except ImportError:
        return {"backups": []}


@router.post("/api/backups/restore")
async def restore_backup(data: dict):
    """Restaura um backup."""
    backup_path = data.get("path", "")

    if not backup_path:
        return {"error": "No backup path provided"}

    try:
        from actuators.config_refiner import ConfigRefiner
        refiner = ConfigRefiner()

        if refiner.restore_backup(backup_path):
            return {"status": "restored", "path": backup_path}
        else:
            return {"error": "Failed to restore backup"}
    except ImportError:
        return {"error": "ConfigRefiner not available"}


# Browser watcher endpoints

@router.get("/api/browser/status")
async def browser_status():
    """Status endpoint for browser extension."""
    return {"status": "connected", "watcher": "browser_watcher"}


@router.post("/api/browser/event")
async def browser_event(request):
    """Receive events from browser extension."""
    from fastapi import Request as FastAPIRequest
    try:
        event = await request.json()

        # Forward to browser watcher
        try:
            from collectors import CollectorRegistry
            watcher = CollectorRegistry.get_instance("browser_watcher")
            if watcher:
                await watcher.receive_event(event)
                return {"status": "received", "type": event.get("type")}
            else:
                # Store directly if watcher not instantiated
                from memory.activity_store import get_activity_store
                from datetime import datetime
                store = get_activity_store()
                store.add(
                    watcher_type="browser_watcher",
                    timestamp=datetime.now(),
                    data=event,
                )
                return {"status": "stored", "type": event.get("type")}
        except ImportError:
            return {"status": "received", "type": event.get("type")}

    except Exception as e:
        return {"error": str(e)}
