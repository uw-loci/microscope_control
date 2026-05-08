"""
Hardware package - Microscope hardware abstraction.

This package contains the hardware abstraction layer that interfaces
with microscope control systems (currently Pycromanager/Micro-Manager).

A microscope is composed of swappable components:

    Camera (required)       -- image capture, exposure, white balance
    Stage (required)        -- XYZ positioning, optional turret + condenser
    RotationStage (optional)-- polarization angle control
    Illumination (optional) -- light sources (LED, laser/Pockels cell)
    Detector (optional)     -- external detectors (PMT, APD)

Modules:
    base: MicroscopeHardware ABC, Position class, utility functions
    camera/: Camera ABC + implementations (generic, JAI 3-CCD, laser scanning)
    stage: Stage ABC + PycromanagerStage (with turret, condenser support)
    rotation: RotationStage ABC + PIZ, Thor, Dummy implementations
    illumination: Illumination ABC + LED, PockelsCell implementations
    detector: Detector ABC + PMTDetector implementation
    pycromanager: PycromanagerHardware (composes all of the above)
"""

from microscope_control.hardware.base import (
    Position,
    MicroscopeHardware,
    is_mm_running,
    is_coordinate_in_range,
)

from microscope_control.hardware.camera.base import Camera
from microscope_control.hardware.stage import Stage
from microscope_control.hardware.rotation import (
    RotationStage,
    DummyRotationStage,
)
from microscope_control.hardware.illumination import (
    Illumination,
    AnalogIllumination,
    LEDIllumination,
    PockelsCell,
)
from microscope_control.hardware.detector import (
    Detector,
    PMTDetector,
)

__all__ = [
    # Core
    "Position",
    "MicroscopeHardware",
    "is_mm_running",
    "is_coordinate_in_range",
    # Component ABCs
    "Camera",
    "Stage",
    "RotationStage",
    "Illumination",
    "Detector",
    # Concrete components (no pycromanager dependency)
    "DummyRotationStage",
    "AnalogIllumination",
    "LEDIllumination",
    "PockelsCell",
    "PMTDetector",
]

# Optional: Export concrete implementations if pycromanager is installed
try:
    from microscope_control.hardware.pycromanager import (
        PycromanagerHardware,
        init_pycromanager,
        MicroManagerConnectionError,
    )
    from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
    from microscope_control.hardware.camera.jai_camera import JAICamera
    from microscope_control.hardware.camera.laser_scanning_camera import LaserScanningCamera
    from microscope_control.hardware.stage import PycromanagerStage
    from microscope_control.hardware.rotation import (
        PIZRotationStage,
        ThorRotationStage,
    )

    __all__.extend(
        [
            "PycromanagerHardware",
            "init_pycromanager",
            "MicroManagerConnectionError",
            "PycromanagerCamera",
            "JAICamera",
            "LaserScanningCamera",
            "PycromanagerStage",
            "PIZRotationStage",
            "ThorRotationStage",
        ]
    )
except ImportError:
    # pycromanager not installed, skip concrete exports
    pass
