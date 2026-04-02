"""
Camera abstraction package.

Provides a Camera ABC and concrete implementations for different
microscope camera types connected via Micro-Manager.
"""

from microscope_control.hardware.camera.base import Camera

__all__ = [
    "Camera",
]

# Optional: Export concrete implementations if pycromanager is installed
try:
    from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
    from microscope_control.hardware.camera.jai_camera import JAICamera
    from microscope_control.hardware.camera.laser_scanning_camera import LaserScanningCamera
    __all__.extend(["PycromanagerCamera", "JAICamera", "LaserScanningCamera"])
except ImportError:
    # pycromanager not installed, skip concrete exports
    pass
