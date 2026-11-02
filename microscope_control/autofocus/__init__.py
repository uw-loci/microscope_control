"""
Autofocus package - Autofocus algorithms and utilities.

This package contains all autofocus-related functionality including
focus metrics, search algorithms, and benchmarking tools.

Modules:
    core: Main autofocus utilities (AutofocusUtils)
    tissue_detection: Empty region detection (EmptyRegionDetector)
    benchmark: Autofocus benchmarking tools
    test: Interactive autofocus testing utilities
"""

from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.autofocus.tissue_detection import EmptyRegionDetector

__all__ = ["AutofocusUtils", "EmptyRegionDetector"]
