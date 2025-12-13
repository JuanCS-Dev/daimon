"""
DAIMON Dashboard - Helper functions.
"""

from pathlib import Path

import httpx


# Service URLs
NOESIS_URL = "http://localhost:8001"
REFLECTOR_URL = "http://localhost:8002"


async def check_service(url: str) -> str:
    """Verifica se servico esta healthy."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("status", "healthy")
            return "unhealthy"
    except Exception:
        return "offline"


def check_socket() -> bool:
    """Verifica se socket do shell watcher existe."""
    socket_path = Path.home() / ".daimon" / "daimon.sock"
    return socket_path.exists()


def check_process(name: str) -> bool:
    """Verifica se componente esta rodando no daemon."""
    state_file = Path.home() / ".daimon" / "state.json"
    try:
        if state_file.exists():
            import json
            state = json.loads(state_file.read_text())
            if state.get("running") and name in state.get("components", []):
                return True
        return False
    except Exception:
        return False
