"""
Microscope Control - Hardware Abstraction and Control Library
==============================================================

A hardware control library for microscopes via Pycromanager (Micro-Manager).
Provides:

- Abstract hardware interface (MicroscopeHardware)
- Pycromanager implementation for real hardware
- Autofocus algorithms and metrics
- Configuration management for microscope settings
- Stage positioning and movement control

This library provides the low-level hardware control needed for automated
microscopy, including stage movement, camera control, and autofocus.

Example Usage:
-------------
from microscope_control.hardware.pycromanager import PycromanagerHardware, init_pycromanager
from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.config import ConfigManager

# Initialize hardware
core, studio = init_pycromanager()
config_mgr = ConfigManager()
settings = config_mgr.get_config('config_PPM')

hardware = PycromanagerHardware(core, studio, settings)

# Perform autofocus
autofocus = AutofocusUtils(hardware, settings)
best_z = autofocus.run_autofocus(current_position)
"""

__version__ = "1.0.0"
__author__ = "Mike Nelson, Bin Li, Jenu Chacko"

# Make key classes easily accessible
from microscope_control.hardware.base import MicroscopeHardware, Position, is_mm_running, is_coordinate_in_range
from microscope_control.hardware.pycromanager import PycromanagerHardware, init_pycromanager
from microscope_control.autofocus.core import AutofocusUtils
# Note: AutofocusMetrics and EmptyRegionDetector are not imported here to avoid
# requiring OpenCV at import time. Import them directly when needed:
#   from microscope_control.autofocus.metrics import AutofocusMetrics
#   from microscope_control.autofocus.tissue_detection import EmptyRegionDetector
from microscope_control.config.manager import ConfigManager

__all__ = [
    "MicroscopeHardware",
    "Position",
    "PycromanagerHardware",
    "init_pycromanager",
    "is_mm_running",
    "is_coordinate_in_range",
    "AutofocusUtils",
    "ConfigManager",
]
