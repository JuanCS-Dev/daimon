
import sys
import os

print(f"CWD: {os.getcwd()}")
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path:", sys.path)

print("-" * 20)

try:
    import shared
    print(f"SUCCESS: import shared (file: {shared.__file__})")
except ImportError as e:
    print(f"FAIL: import shared: {e}")

try:
    import shared.validators
    print("SUCCESS: import shared.validators")
except ImportError as e:
    print(f"FAIL: import shared.validators: {e}")

try:
    import dashboard
    print(f"SUCCESS: import dashboard (file: {dashboard.__file__})")
except ImportError as e:
    print(f"FAIL: import dashboard: {e}")

try:
    import dashboard.main
    print("SUCCESS: import dashboard.main")
except ImportError as e:
    print(f"FAIL: import dashboard.main: {e}")
