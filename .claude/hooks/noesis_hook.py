#!/usr/bin/env python3
"""
Backward compatibility shim.

Redirects to daimon_hook.py for sessions using old config.
New sessions use daimon_hook.py directly via absolute path in ~/.claude/settings.json.
"""
import os
import sys

# Get the actual hook path
hook_dir = os.path.dirname(os.path.abspath(__file__))
daimon_hook = os.path.join(hook_dir, "daimon_hook.py")

# Execute daimon_hook.py with same stdin/stdout
if os.path.exists(daimon_hook):
    with open(daimon_hook) as f:
        code = f.read()
    exec(compile(code, daimon_hook, 'exec'))
else:
    # Fallback: exit silently if hook not found
    sys.exit(0)
