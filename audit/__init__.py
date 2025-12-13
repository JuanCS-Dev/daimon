"""
DAIMON Audit System - Modular validation framework.

Usage:
    python -m audit
"""

from .core import log, generate_report, RESULTS

__all__ = ["log", "generate_report", "RESULTS"]
