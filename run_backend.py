
import sys
import os
import runpy

# Hardcode paths relative to project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = [
    os.path.join(ROOT_DIR, 'backend/services'),
    os.path.join(ROOT_DIR, 'backend/services/maximus_core_service/src'),
    os.path.join(ROOT_DIR, 'backend/services/metacognitive_reflector/src'),
]

for p in PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"BOOTSTRAP: Added paths: {PATHS}")

# Target script
TARGET = "backend/services/maximus_core_service/src/maximus_core_service/main.py"

print(f"BOOTSTRAP: Running {TARGET}")
runpy.run_path(TARGET, run_name="__main__")
