#!/bin/bash
# =============================================================================
# NOESIS Memory Backup - Redundancy Script
# =============================================================================
# Syncs Noesis memory data to a secondary backup location for redundancy.
#
# Strategy:
#   PRIMARY:   data/               (in project directory)
#   SECONDARY: /media/juan/DATA/noesis_backup/ (external storage)
#
# Usage:
#   ./scripts/backup_memory.sh           # Manual backup
#   0 * * * * /path/to/backup_memory.sh  # Hourly cron job
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source and destination
SOURCE="$PROJECT_DIR/data"
BACKUP_BASE="/media/juan/DATA/noesis_backup"

# Timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Check if source exists
if [ ! -d "$SOURCE" ]; then
    echo "[$TIMESTAMP] ERROR: Source directory not found: $SOURCE"
    exit 1
fi

# Check if backup location is accessible (external drive mounted)
if [ ! -d "/media/juan/DATA" ]; then
    echo "[$TIMESTAMP] WARNING: External storage not mounted, skipping backup"
    exit 0
fi

# Create backup directory if needed
mkdir -p "$BACKUP_BASE"

# Perform sync with rsync
# Options:
#   -a  archive mode (preserves permissions, timestamps, etc.)
#   -v  verbose
#   --delete  remove files in destination that don't exist in source
#   --exclude  skip temporary files and logs
echo "[$TIMESTAMP] Starting Noesis memory backup..."
echo "  Source:      $SOURCE"
echo "  Destination: $BACKUP_BASE"

rsync -av --delete \
    --exclude '*.tmp' \
    --exclude '*.log' \
    --exclude '__pycache__' \
    "$SOURCE/" "$BACKUP_BASE/"

# Log completion
echo "[$TIMESTAMP] Backup completed successfully."
echo ""

# Show backup stats
echo "Backup contents:"
du -sh "$BACKUP_BASE"/* 2>/dev/null || echo "  (empty)"

# Create backup manifest
MANIFEST="$BACKUP_BASE/.backup_manifest"
echo "Last backup: $TIMESTAMP" > "$MANIFEST"
echo "Source: $SOURCE" >> "$MANIFEST"
echo "Files:" >> "$MANIFEST"
find "$BACKUP_BASE" -type f -name "*.json" | wc -l | xargs echo "  JSON files:" >> "$MANIFEST"
