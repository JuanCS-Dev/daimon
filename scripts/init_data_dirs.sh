#!/bin/bash
# =============================================================================
# NOESIS Data Directory Initialization
# =============================================================================
# Creates permanent data directories for hermetic memory persistence.
# Run this once after cloning the project or when setting up a new environment.
#
# Directory Structure:
#   data/
#   ├── sessions/       # Session memory files (JSON)
#   ├── memory/         # Memory backup files
#   ├── vault/          # L4 vault backups (long-term)
#   └── wal/            # Write-ahead log for crash recovery
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "Initializing Noesis data directories..."
echo "Project: $PROJECT_DIR"
echo "Data:    $DATA_DIR"
echo ""

# Create primary data directories
mkdir -p "$DATA_DIR/sessions"
mkdir -p "$DATA_DIR/memory"
mkdir -p "$DATA_DIR/vault"
mkdir -p "$DATA_DIR/wal"

# Set permissions (owner read/write/execute)
chmod 700 "$DATA_DIR"
chmod 700 "$DATA_DIR/sessions"
chmod 700 "$DATA_DIR/memory"
chmod 700 "$DATA_DIR/vault"
chmod 700 "$DATA_DIR/wal"

# Create .gitkeep files to track empty directories
touch "$DATA_DIR/sessions/.gitkeep"
touch "$DATA_DIR/memory/.gitkeep"
touch "$DATA_DIR/vault/.gitkeep"
touch "$DATA_DIR/wal/.gitkeep"

echo "Created directories:"
echo "  $DATA_DIR/sessions/  - Session memory (conversation history)"
echo "  $DATA_DIR/memory/    - Memory backups"
echo "  $DATA_DIR/vault/     - Long-term vault storage"
echo "  $DATA_DIR/wal/       - Write-ahead log"
echo ""
echo "Noesis data directories initialized successfully."
