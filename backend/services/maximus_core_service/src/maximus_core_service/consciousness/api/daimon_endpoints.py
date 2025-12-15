"""
Daimon Sensory Endpoints - The Neural Link
==========================================

Receives raw sensory input from DAIMON collectors (Shell, Browser, etc.)
and stores it in ShortTermSensoryMemory for immediate conscious access.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body

import json
from pathlib import Path

# ... (imports anteriores mantidos)

SENSORY_FILE = Path.home() / ".noesis" / "sensory_memory.json"

COMMAND_BUFFER_SIZE = 15
BROWSER_BUFFER_SIZE = 5

class ShortTermSensoryMemory:
    """Buffer for recent sensory inputs with persistence."""
    
    _shell_buffer: deque = deque(maxlen=COMMAND_BUFFER_SIZE)
    _browser_buffer: deque = deque(maxlen=BROWSER_BUFFER_SIZE)
    _last_browser_state: Dict[str, Any] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_init(cls):
        if not cls._initialized:
            cls._load_state()
            cls._initialized = True

    @classmethod
    def _load_state(cls):
        """Hydrate memory from disk."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            if SENSORY_FILE.exists():
                with open(SENSORY_FILE, "r") as f:
                    data = json.load(f)
                    # Load Shell
                    for item in data.get("shell", []):
                        cls._shell_buffer.append(item)
                    # Load Browser
                    for item in data.get("browser", []):
                        cls._browser_buffer.append(item)
                    cls._last_browser_state = data.get("last_visual", {})
                logger.info(f"[SENSE] Memory hydrated from disk: {len(cls._shell_buffer)} shell, {len(cls._browser_buffer)} browser events.")
        except Exception as e:
            logger.warning(f"[SENSE] Failed to load sensory memory: {e}")

    _last_save_time: datetime = datetime.min
    SAVE_INTERVAL_SECONDS = 5

    @classmethod
    def _save_state(cls):
        """Persist memory to disk (throttled)."""
        now = datetime.now()
        if (now - cls._last_save_time).total_seconds() < cls.SAVE_INTERVAL_SECONDS:
            return

        try:
            data = {
                "shell": list(cls._shell_buffer),
                "browser": list(cls._browser_buffer),
                "last_visual": cls._last_browser_state
            }
            # Ensure dir exists
            SENSORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SENSORY_FILE, "w") as f:
                json.dump(data, f, default=str) # default=str handles datetime
            cls._last_save_time = now
        except Exception as e:
            logger.warning(f"[SENSE] Failed to save sensory memory: {e}")

    @classmethod
    def add_shell_event(cls, command: str, exit_code: int, pwd: str):
        cls._ensure_init()
        event = {
            "type": "shell",
            "command": command,
            "exit_code": exit_code,
            "pwd": pwd,
            "timestamp": datetime.now().isoformat()
        }
        cls._shell_buffer.append(event)
        cls._save_state()
        # Log less verbosely
        # logger.debug(f"[SENSE] Shell: {command}")

    @classmethod
    def add_browser_event(cls, active_domain: str):
        cls._ensure_init()
        if not active_domain:
            return
            
        event = {
            "type": "browser",
            "domain": active_domain,
            "timestamp": datetime.now().isoformat()
        }
        # Only add if different from last to avoid spam
        if not cls._last_browser_state or cls._last_browser_state.get("domain") != active_domain:
            cls._browser_buffer.append(event)
            cls._last_browser_state = event
            cls._save_state()
            logger.debug(f"[SENSE] Visual focus switched to: {active_domain}")

    @classmethod
    def get_context_block(cls) -> str:
        cls._ensure_init()
        """Generate a text block of recent sensory inputs for the LLM."""
        lines = []
        
        # Browser Context (Visual)
        if cls._last_browser_state:
            dom = cls._last_browser_state.get("domain")
            # Handle string timestamp from JSON or datetime object
            ts = cls._last_browser_state["timestamp"]
            if isinstance(ts, str):
                ts_obj = datetime.fromisoformat(ts)
            else:
                ts_obj = ts
                
            since = (datetime.now() - ts_obj).total_seconds()
            lines.append(f"👁️ VISUAL CONTEXT: User is looking at '{dom}' (for {int(since)}s)")
        
        # Shell Context (Auditory/Action)
        if cls._shell_buffer:
            lines.append("⌨️ RECENT TERMINAL ACTIVITY:")
            # Show last 5
            recent = list(cls._shell_buffer)[-5:]
            for event in recent:
                status = "✅" if event["exit_code"] == 0 else "❌"
                ts = event["timestamp"]
                if isinstance(ts, str):
                    ts_obj = datetime.fromisoformat(ts)
                else:
                    ts_obj = ts
                ago = (datetime.now() - ts_obj).total_seconds()
                lines.append(f"  - [{int(ago)}s ago] {status} `{event['command']}` (in {event['pwd'].split('/')[-1]})")
        
        if not lines:
            return ""
            
        return "\n".join(lines)

# ==============================================================================
# Endpoints registration
# ==============================================================================

def register_daimon_endpoints(router: APIRouter):
    """Register sensory endpoints."""
    
    @router.post("/sense/shell")
    async def sense_shell(
        heartbeats: List[Dict[str, Any]] = Body(..., embed=True),
        patterns: Dict[str, Any] = Body(None, embed=True)
    ):
        """Receive shell impulses."""
        count = 0
        for hb in heartbeats:
            ShortTermSensoryMemory.add_shell_event(
                command=hb.get("command", ""),
                exit_code=hb.get("exit_code", 0),
                pwd=hb.get("pwd", "")
            )
            count += 1
        return {"ack": True, "processed": count}

    @router.post("/sense/browser")
    async def sense_browser(
        heartbeats: List[Dict[str, Any]] = Body(..., embed=True)
    ):
        """Receive visual/browser impulses."""
        # We expect a list of heartbeats. We care about the latest active domain.
        if heartbeats:
            latest = heartbeats[-1]
            data = latest.get("data", {})
            # We will modify BrowserWatcher to send 'active_domain' in data
            active_domain = data.get("active_domain")
            
            # Fallback: check domains list
            if not active_domain and "domains" in data:
                 # Logic to guess active domain from stats is unreliable here
                 # We will strictly rely on the watcher sending the active one.
                 pass

            if active_domain:
                ShortTermSensoryMemory.add_browser_event(active_domain)
                
        return {"ack": True}



