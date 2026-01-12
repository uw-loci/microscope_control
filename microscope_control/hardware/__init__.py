"""
Hardware package - Microscope hardware abstraction.

This package contains the hardware abstraction layer that interfaces
with microscope control systems (currently Pycromanager/Micro-Manager).

Modules:
    base: Abstract hardware interface, Position class, and utility functions
    pycromanager: Pycromanager-based hardware implementation
"""

from microscope_control.hardware.base import (
    Position,
    MicroscopeHardware,
    is_mm_running,
    is_coordinate_in_range,
)

__all__ = [
    "Position",
    "MicroscopeHardware",
    "is_mm_running",
    "is_coordinate_in_range",
]

# Optional: Export PycromanagerHardware if pycromanager is installed
try:
    from microscope_control.hardware.pycromanager import PycromanagerHardware, init_pycromanager
    __all__.extend(["PycromanagerHardware", "init_pycromanager"])
except ImportError:
    # pycromanager not installed, skip these exports
    pass
