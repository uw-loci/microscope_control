"""
JAI Camera Control and Calibration
===================================

This subpackage provides JAI camera-specific functionality for the
JAI AP-3200T-USB 3-CCD prism camera.

The JAI prism camera has unique capabilities not found in typical cameras:
- Independent exposure control per R/G/B channel
- Independent analog and digital gain per channel
- Per-channel black level adjustment

These features require specialized handling for white balance calibration
that differs from software-based corrections used with Bayer cameras.

Modules:
    properties: JAICameraProperties - Type-safe property access via Micro-Manager
    calibration: JAIWhiteBalanceCalibrator - Hardware white balance calibration

Example Usage:
--------------
from microscope_control.jai import JAICameraProperties, JAIWhiteBalanceCalibrator

# Direct property control
props = JAICameraProperties(core)
props.enable_individual_exposure()
props.set_channel_exposures(red=50.0, green=60.0, blue=70.0)

# Use camera's built-in auto white balance
props.run_auto_white_balance()  # One-shot calibration
# Or use presets: props.set_white_balance_preset(5000)  # 5000K preset

# Automated calibration with saved settings (default tolerance=2 for 2-level precision)
calibrator = JAIWhiteBalanceCalibrator(hardware)
result = calibrator.calibrate(target_value=180)  # tolerance=2 by default

Note:
    Requires Micro-Manager with JAI camera device adapter supporting
    individual channel control (PR #781 or later).
"""

from microscope_control.jai.properties import JAICameraProperties, PropertyLimits
from microscope_control.jai.calibration import (
    JAIWhiteBalanceCalibrator,
    WhiteBalanceResult,
    CalibrationConfig,
    db_to_linear,
    linear_to_db,
)

__all__ = [
    "JAICameraProperties",
    "PropertyLimits",
    "JAIWhiteBalanceCalibrator",
    "WhiteBalanceResult",
    "CalibrationConfig",
    "db_to_linear",
    "linear_to_db",
]
