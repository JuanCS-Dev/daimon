#!/usr/bin/env python3
"""
DAIMON System Audit - Entry point.

Usage:
    python -m audit
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .core import RESULTS, generate_report
from .api_tests import test_dashboard_api, test_noesis_api, test_reflector_api
from .component_tests import (
    test_collectors,
    test_memory_systems,
    test_learners,
    test_actuators,
    test_corpus,
)
from .integration_tests import (
    test_hooks,
    test_mcp_server,
    test_files_structure,
    test_data_directories,
    test_integration,
)
from .quality_tests import test_performance, test_edge_cases


async def main() -> int:
    """Run all tests. Returns failure count."""
    print("\n" + "#" * 60)
    print("# DAIMON SYSTEM AUDIT")
    print(f"# Date: {datetime.now().isoformat()}")
    print("#" * 60)

    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    # Run all tests
    await test_dashboard_api()
    await test_noesis_api()
    await test_reflector_api()
    test_collectors()
    test_memory_systems()
    test_learners()
    test_actuators()
    test_corpus()
    test_hooks()
    test_mcp_server()
    test_files_structure()
    test_data_directories()
    test_integration()
    test_performance()
    test_edge_cases()

    # Generate report
    failures = generate_report()

    # Save results
    report_path = project_root / "docs" / "AUDIT_REPORT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "results": RESULTS,
        "total_failures": failures
    }, indent=2))
    print(f"\nReport saved to: {report_path}")

    return failures


if __name__ == "__main__":
    failures = asyncio.run(main())
    sys.exit(1 if failures > 0 else 0)
