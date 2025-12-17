#!/bin/bash
# Startup script for Maximus Core (Noesis Backend)
# Run from Noesis/Daimon root

# Add shared services and source dirs to python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/services:$(pwd)/backend/services/maximus_core_service/src:$(pwd)/backend/services/metacognitive_reflector/src

echo "Starting Maximus Core Service..."
echo "PYTHONPATH=$PYTHONPATH"

# Run with buffering disabled
python3 -u backend/services/maximus_core_service/src/maximus_core_service/main.py
