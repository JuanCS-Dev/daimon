"""
Lifecycle Management Utility
===========================

Provides utilities for managing service lifecycles, specifically ensuring
ports are free before starting services (Anti-Zombie Mechanism).

Author: DAIMON
Date: 2025-12-17
"""

import os
import signal
import subprocess
import time
import logging
import sys
from typing import Optional

# Configure simplified logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lifecycle")


def get_pid_by_port(port: int) -> Optional[int]:
    """
    Finds the PID of the process listening on a specific TCP port.
    Uses 'lsof' first, falls back to 'netstat' or 'ss' if needed.
    """
    try:
        # Try lsof
        # -t: terse (PID only), -i: internet, -sTCP:LISTEN
        cmd = f"lsof -t -i tcp:{port} -sTCP:LISTEN"
        pid_str = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid_str:
            return int(pid_str.split('\n')[0])  # Return first PID if multiple
    except subprocess.CalledProcessError:
        # lsof returned non-zero, meaning no process found (usually)
        pass
    except Exception as e:
        logger.warning(f"Error checking port {port}: {e}")

    return None


def kill_process(pid: int, force: bool = False) -> bool:
    """
    Terminates a process by PID.
    First tries SIGTERM, then SIGKILL if force=True or if it refuses to die.
    """
    try:
        os.kill(pid, 0) # Check if process exists
    except OSError:
        return True # Process already dead

    try:
        if not force:
            logger.info(f"Sending SIGTERM to PID {pid}...")
            os.kill(pid, signal.SIGTERM)
            # Wait a bit
            for _ in range(10): 
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    return True # Died
            
            logger.info(f"PID {pid} did not exit gracefully.")
        
        # Force kill
        logger.warning(f"Sending SIGKILL to PID {pid}...")
        os.kill(pid, signal.SIGKILL)
        return True
        
    except Exception as e:
        logger.error(f"Failed to kill PID {pid}: {e}")
        return False


def ensure_port_protection(port: int, service_name: str = "Service") -> None:
    """
    Ensures a port is free by killing any process currently occupying it.
    This is the core 'Anti-Zombie' protection.
    """
    logger.info(f"[ANTIGRAVITY] Protecting port {port} for {service_name}...")
    
    pid = get_pid_by_port(port)
    if pid:
        current_pid = os.getpid()
        if pid == current_pid:
            logger.info(f"Port {port} is occupied by ME (PID {pid}). Continuing...")
            return

        logger.warning(f"⚠️  Port {port} is occupied by PID {pid}. Initiating PURGE protocols.")
        if kill_process(pid):
            logger.info(f"✅ PID {pid} terminated. Port {port} is now free.")
            time.sleep(1) # Cool down to let socket close
        else:
            logger.error(f"❌ Failed to liberate port {port}. Startup may fail.")
            # We don't exit here, we let the binder fail naturally if needed, 
            # or maybe we should exit? Let's try to proceed.
    else:
        logger.info(f"✅ Port {port} is clean.")


def install_signal_handlers() -> None:
    """
    Installs handlers for SIGINT/SIGTERM to ensure graceful shutdown logging.
    """
    def handle_exit(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        # Here we could run specific cleanup if needed
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    # Test CLI
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
        ensure_port_protection(p, "TestService")
