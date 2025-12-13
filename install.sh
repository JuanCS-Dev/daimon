#!/bin/bash
#
# DAIMON Installation Script
#
# This script:
# 1. Installs Python dependencies
# 2. Sets up systemd user service
# 3. Configures shell hooks (optional)
# 4. Enables autostart on boot
#
# Usage:
#   ./install.sh          # Full installation
#   ./install.sh --deps   # Only install dependencies
#   ./install.sh --systemd # Only setup systemd
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAIMON_DIR="$SCRIPT_DIR"
SERVICE_DIR="$HOME/.config/systemd/user"
DAIMON_DATA="$HOME/.daimon"

echo "================================"
echo "DAIMON Installation"
echo "================================"
echo ""
echo "Installation directory: $DAIMON_DIR"
echo ""

# Parse arguments
INSTALL_DEPS=true
INSTALL_SYSTEMD=true
INSTALL_SHELL=false

for arg in "$@"; do
    case $arg in
        --deps)
            INSTALL_DEPS=true
            INSTALL_SYSTEMD=false
            ;;
        --systemd)
            INSTALL_DEPS=false
            INSTALL_SYSTEMD=true
            ;;
        --shell)
            INSTALL_SHELL=true
            ;;
        --all)
            INSTALL_DEPS=true
            INSTALL_SYSTEMD=true
            INSTALL_SHELL=true
            ;;
        --help)
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --deps      Only install Python dependencies"
            echo "  --systemd   Only setup systemd service"
            echo "  --shell     Also install shell hooks (zsh/bash)"
            echo "  --all       Full installation with shell hooks"
            echo "  --help      Show this help"
            exit 0
            ;;
    esac
done

# Step 1: Install dependencies
if [ "$INSTALL_DEPS" = true ]; then
    echo "[1/4] Installing Python dependencies..."
    pip install --quiet fastmcp httpx pydantic watchdog pytest pytest-asyncio uvicorn jinja2
    echo "      Done."
fi

# Step 2: Create data directories
echo "[2/4] Creating data directories..."
mkdir -p "$DAIMON_DATA/memory"
mkdir -p "$DAIMON_DATA/corpus"
mkdir -p "$DAIMON_DATA/logs"
mkdir -p "$HOME/.claude/backups"
mkdir -p "$HOME/.claude/agents"
echo "      Done."

# Step 3: Copy Claude Code integration files
echo "[3/4] Setting up Claude Code integration..."
if [ -f "$DAIMON_DIR/.claude/agents/noesis-sage.md" ]; then
    cp "$DAIMON_DIR/.claude/agents/noesis-sage.md" "$HOME/.claude/agents/"
    echo "      Copied noesis-sage.md to ~/.claude/agents/"
fi

if [ -d "$DAIMON_DIR/.claude/hooks" ]; then
    mkdir -p "$HOME/.claude/hooks"
    cp "$DAIMON_DIR/.claude/hooks/noesis_hook.py" "$HOME/.claude/hooks/"
    echo "      Copied noesis_hook.py to ~/.claude/hooks/"
fi

if [ -f "$DAIMON_DIR/.claude/settings.json" ]; then
    # Merge or copy settings
    if [ -f "$HOME/.claude/settings.json" ]; then
        echo "      WARNING: ~/.claude/settings.json already exists"
        echo "      Please manually merge hooks from $DAIMON_DIR/.claude/settings.json"
    else
        cp "$DAIMON_DIR/.claude/settings.json" "$HOME/.claude/"
        echo "      Copied settings.json to ~/.claude/"
    fi
fi
echo "      Done."

# Step 4: Setup systemd
if [ "$INSTALL_SYSTEMD" = true ]; then
    echo "[4/4] Setting up systemd user service..."

    # Create service directory
    mkdir -p "$SERVICE_DIR"

    # Generate service file with correct paths
    cat > "$SERVICE_DIR/daimon.service" << EOF
[Unit]
Description=DAIMON - Personal Exocortex Daemon
Documentation=file://$DAIMON_DIR/README.md
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $DAIMON_DIR/daimon_daemon.py
ExecStop=/usr/bin/python3 $DAIMON_DIR/daimon_daemon.py --stop
WorkingDirectory=$DAIMON_DIR
Restart=on-failure
RestartSec=5

# Environment
Environment=PYTHONPATH=$DAIMON_DIR
Environment=HOME=$HOME

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=daimon

[Install]
WantedBy=default.target
EOF

    # Reload systemd
    systemctl --user daemon-reload

    echo "      Service file created at $SERVICE_DIR/daimon.service"
    echo ""
    echo "      To enable autostart on boot:"
    echo "        systemctl --user enable daimon"
    echo ""
    echo "      To start now:"
    echo "        systemctl --user start daimon"
    echo ""
    echo "      To check status:"
    echo "        systemctl --user status daimon"
    echo ""
    echo "      To view logs:"
    echo "        journalctl --user -u daimon -f"
fi

# Optional: Shell hooks
if [ "$INSTALL_SHELL" = true ]; then
    echo ""
    echo "[OPTIONAL] Setting up shell hooks..."

    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_RC" ]; then
        # Check if hooks already installed
        if grep -q "daimon.sock" "$SHELL_RC"; then
            echo "      Shell hooks already installed in $SHELL_RC"
        else
            # Generate and append hooks
            python3 "$DAIMON_DIR/collectors/shell_watcher.py" --zshrc >> "$SHELL_RC"
            echo "      Added shell hooks to $SHELL_RC"
            echo "      Run 'source $SHELL_RC' to activate"
        fi
    else
        echo "      Could not detect shell rc file"
    fi
fi

echo ""
echo "================================"
echo "Installation complete!"
echo "================================"
echo ""
echo "Quick start:"
echo "  systemctl --user enable --now daimon"
echo ""
echo "Dashboard:"
echo "  http://localhost:8003"
echo ""
echo "Logs:"
echo "  journalctl --user -u daimon -f"
echo "  cat ~/.daimon/logs/daimon.log"
echo ""
