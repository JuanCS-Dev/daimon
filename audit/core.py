"""
Audit Core - Logging and reporting utilities.
"""

from typing import Any

# Results storage
RESULTS: dict[str, dict[str, Any]] = {}


def log(section: str, test: str, status: str, details: str = "") -> None:
    """Log a test result."""
    if section not in RESULTS:
        RESULTS[section] = {"passed": 0, "failed": 0, "tests": []}

    passed = status == "PASS"
    RESULTS[section]["passed" if passed else "failed"] += 1
    RESULTS[section]["tests"].append({
        "test": test,
        "status": status,
        "details": details
    })

    icon = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{icon}{reset} {test}: {status}")
    if details and not passed:
        print(f"    └─ {details}")


def generate_report() -> int:
    """Generate final audit report. Returns failure count."""
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    for section, data in RESULTS.items():
        passed = data["passed"]
        failed = data["failed"]
        total = passed + failed
        total_passed += passed
        total_failed += failed

        pct = (passed / total * 100) if total > 0 else 0
        status = "✓" if failed == 0 else "✗"
        color = "\033[92m" if failed == 0 else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status}{reset} {section}: {passed}/{total} ({pct:.0f}%)")

    print("\n" + "-" * 60)
    total = total_passed + total_failed
    pct = (total_passed / total * 100) if total > 0 else 0

    print(f"\nTOTAL: {total_passed}/{total} tests passed ({pct:.1f}%)")
    print(f"PASSED: {total_passed}")
    print(f"FAILED: {total_failed}")

    # List all failures
    if total_failed > 0:
        print("\n" + "=" * 60)
        print("FAILURES (AIRGAPS)")
        print("=" * 60)
        for section, data in RESULTS.items():
            for test in data["tests"]:
                if test["status"] == "FAIL":
                    print(f"\n[{section}] {test['test']}")
                    if test["details"]:
                        print(f"  Reason: {test['details']}")

    return total_failed
