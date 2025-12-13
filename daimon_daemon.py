#!/usr/bin/env python3
"""
DAIMON Daemon - Unified Process Manager

Starts all DAIMON components in a single process:
- Shell Watcher (Unix socket server)
- Claude Watcher (Session polling)
- Dashboard (FastAPI on port 8003)
- Reflection Engine (Background learning loop)

Usage:
    python daimon_daemon.py              # Run in foreground
    python daimon_daemon.py --daemon     # Run as daemon (background)
    python daimon_daemon.py --status     # Check if running
    python daimon_daemon.py --stop       # Stop daemon

Systemd integration:
    systemctl --user start daimon
    systemctl --user enable daimon
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging first
LOG_DIR = Path.home() / ".daimon" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daimon.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daimon.daemon")

# Paths
DAIMON_DIR = Path(__file__).parent.absolute()
PID_FILE = Path.home() / ".daimon" / "daimon.pid"
STATE_FILE = Path.home() / ".daimon" / "state.json"


class DaimonDaemon:
    """
    Unified daemon that manages all DAIMON components.

    Components:
    - shell_watcher: Unix socket server for shell command capture
    - claude_watcher: Polling monitor for Claude Code sessions
    - dashboard: FastAPI web interface on port 8003
    - reflection_engine: Background learning loop
    """

    def __init__(self) -> None:
        self.running = False
        self.tasks: dict[str, asyncio.Task] = {}
        self.start_time: Optional[datetime] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start all DAIMON components."""
        logger.info("Starting DAIMON daemon...")
        self.running = True
        self.start_time = datetime.now()

        # Write PID file
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        # Start components
        try:
            await self._start_components()
            logger.info("DAIMON daemon started successfully")
            self._save_state()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error("Error starting daemon: %s", e)
            raise
        finally:
            await self._cleanup()

    async def _start_components(self) -> None:
        """Start all daemon components as asyncio tasks."""

        # 1. Shell Watcher (special - runs as Unix socket server)
        try:
            from collectors.shell_watcher import start_server as start_shell_watcher
            self.tasks["shell_watcher"] = asyncio.create_task(
                start_shell_watcher(),
                name="shell_watcher"
            )
            logger.info("Shell Watcher started")
        except ImportError as e:
            logger.warning("Shell Watcher not available: %s", e)

        # 2. Claude Watcher (special - uses SessionTracker)
        try:
            from collectors.claude_watcher import SessionTracker
            tracker = SessionTracker()
            self.tasks["claude_watcher"] = asyncio.create_task(
                tracker.run(),
                name="claude_watcher"
            )
            logger.info("Claude Watcher started")
        except ImportError as e:
            logger.warning("Claude Watcher not available: %s", e)
        except AttributeError:
            logger.warning("Claude Watcher run() not available")

        # 3. New Watchers via CollectorRegistry
        await self._start_registry_watchers()

        # 4. Dashboard
        try:
            import uvicorn
            from dashboard.app import app

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=8003,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            self.tasks["dashboard"] = asyncio.create_task(
                server.serve(),
                name="dashboard"
            )
            logger.info("Dashboard started on http://localhost:8003")
        except ImportError as e:
            logger.warning("Dashboard not available: %s", e)

        # 5. Reflection Engine
        try:
            from learners import get_engine
            engine = get_engine()
            self.tasks["reflection_engine"] = asyncio.create_task(
                engine.start(),
                name="reflection_engine"
            )
            logger.info("Reflection Engine started")
        except ImportError as e:
            logger.warning("Reflection Engine not available: %s", e)

        # 6. Activity Store Flush Loop
        await self._start_activity_store()

    async def _start_registry_watchers(self) -> None:
        """Start watchers registered via CollectorRegistry."""
        try:
            from collectors import CollectorRegistry
            from memory.activity_store import get_activity_store
            from learners import get_style_learner

            activity_store = get_activity_store()
            style_learner = get_style_learner()

            # Watchers to start from registry
            registry_watchers = [
                "window_watcher",
                "input_watcher",
                "afk_watcher",
                "browser_watcher",  # Now enabled - HTTP endpoint in dashboard/app.py
            ]

            for watcher_name in registry_watchers:
                try:
                    watcher = CollectorRegistry.create_instance(watcher_name)

                    # Create wrapper task that stores heartbeats
                    async def watcher_loop(w, name):
                        """Run watcher and store heartbeats."""
                        logger.info("Starting %s via registry", name)
                        try:
                            while self.running:
                                heartbeat = await w.collect()
                                if heartbeat:
                                    # Store in activity store
                                    activity_store.add(
                                        watcher_type=heartbeat.watcher_type,
                                        timestamp=heartbeat.timestamp,
                                        data=heartbeat.data,
                                    )

                                    # Feed to style learner (all sources)
                                    if name == "input_watcher":
                                        style_learner.add_keystroke_sample(
                                            heartbeat.data.get("keystroke_dynamics", {})
                                        )
                                    elif name == "window_watcher":
                                        style_learner.add_window_sample(heartbeat.data)
                                    elif name == "afk_watcher":
                                        style_learner.add_afk_sample(heartbeat.data)
                                    elif name == "browser_watcher":
                                        # Browser data also useful for focus patterns
                                        style_learner.add_window_sample(heartbeat.data)

                                await asyncio.sleep(w.get_collection_interval())
                        except asyncio.CancelledError:
                            logger.info("%s stopped", name)
                        except Exception as e:
                            logger.error("%s error: %s", name, e)

                    self.tasks[watcher_name] = asyncio.create_task(
                        watcher_loop(watcher, watcher_name),
                        name=watcher_name
                    )
                    logger.info("%s registered and started", watcher_name)

                except Exception as e:
                    logger.warning("Could not start %s: %s", watcher_name, e)

        except ImportError as e:
            logger.warning("CollectorRegistry not available: %s", e)

    async def _start_activity_store(self) -> None:
        """Start activity store maintenance loop."""
        try:
            from memory.activity_store import get_activity_store

            store = get_activity_store()

            async def cleanup_loop():
                """Periodic cleanup of old activity data."""
                while self.running:
                    try:
                        # Cleanup every hour
                        await asyncio.sleep(3600)
                        deleted = store.cleanup()
                        if deleted > 0:
                            logger.info("Activity store cleanup: %d records removed", deleted)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error("Activity store cleanup error: %s", e)

            self.tasks["activity_cleanup"] = asyncio.create_task(
                cleanup_loop(),
                name="activity_cleanup"
            )
            logger.info("Activity store initialized")

        except ImportError as e:
            logger.warning("Activity store not available: %s", e)

    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    async def _cleanup(self) -> None:
        """Stop all components and cleanup."""
        logger.info("Stopping DAIMON daemon...")
        self.running = False

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.info("Stopping %s...", name)
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Stop reflection engine properly
        try:
            from learners import get_engine
            await get_engine().stop()
        except Exception as e:
            logger.debug("Reflection engine stop failed: %s", e)

        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

        self._save_state()
        logger.info("DAIMON daemon stopped")

    def _save_state(self) -> None:
        """Save daemon state to file."""
        state = {
            "running": self.running,
            "pid": os.getpid() if self.running else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "components": list(self.tasks.keys()),
            "updated_at": datetime.now().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))


def get_status() -> dict:
    """Get daemon status."""
    status = {
        "running": False,
        "pid": None,
        "start_time": None,
        "components": [],
    }

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            status["running"] = True
            status["pid"] = pid
        except (ValueError, ProcessLookupError, PermissionError):
            # Process not running, clean up PID file
            PID_FILE.unlink(missing_ok=True)

    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            status["start_time"] = state.get("start_time")
            status["components"] = state.get("components", [])
        except json.JSONDecodeError:
            pass

    return status


def stop_daemon() -> bool:
    """Stop running daemon."""
    if not PID_FILE.exists():
        print("DAIMON daemon is not running")
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
        return True
    except (ValueError, ProcessLookupError) as e:
        print(f"Could not stop daemon: {e}")
        PID_FILE.unlink(missing_ok=True)
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DAIMON Daemon")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as daemon (background process)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show daemon status"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running daemon"
    )
    args = parser.parse_args()

    # Change to DAIMON directory for imports
    os.chdir(DAIMON_DIR)
    sys.path.insert(0, str(DAIMON_DIR))

    if args.status:
        status = get_status()
        if status["running"]:
            print(f"DAIMON daemon is running (PID: {status['pid']})")
            print(f"  Started: {status['start_time']}")
            print(f"  Components: {', '.join(status['components'])}")
        else:
            print("DAIMON daemon is not running")
        return

    if args.stop:
        stop_daemon()
        return

    # Check if already running
    status = get_status()
    if status["running"]:
        print(f"DAIMON daemon is already running (PID: {status['pid']})")
        print("Use --stop to stop it first")
        sys.exit(1)

    if args.daemon:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            print(f"DAIMON daemon started (PID: {pid})")
            sys.exit(0)

        # Detach from terminal
        os.setsid()
        os.umask(0)

        # Close standard file descriptors
        sys.stdin.close()

        # Redirect stdout/stderr to log
        log_file = open(LOG_DIR / "daimon.log", "a")
        sys.stdout = log_file
        sys.stderr = log_file

    # Run daemon
    daemon = DaimonDaemon()
    asyncio.run(daemon.start())


if __name__ == "__main__":
    main()
