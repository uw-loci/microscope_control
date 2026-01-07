"""Hardware abstraction layer for microscope control."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import warnings
import logging

logger = logging.getLogger(__name__)


class Position:
    """Simple position class to replace sp_position dataclass."""

    def __init__(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        self.x = x
        self.y = y
        self.z = z

    def populate_missing(self, current_position: "Position") -> None:
        """Populate missing coordinates with values from current_position."""
        if self.x is None:
            self.x = current_position.x
        if self.y is None:
            self.y = current_position.y
        if self.z is None:
            self.z = current_position.z

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"


class MicroscopeHardware(ABC):
    """Abstract base class for microscope hardware control."""

    @abstractmethod
    def move_to_position(self, position: Position) -> None:
        """Move stage to specified position."""
        pass

    @abstractmethod
    def get_current_position(self) -> Position:
        """Get current stage position."""
        pass


def is_mm_running() -> bool:
    """Check if Micro-Manager is running as a Windows executable."""
    import platform
    import psutil

    if platform.system() != "Windows":
        return False

    for proc in psutil.process_iter(["name"]):
        try:
            if proc.exe().find("Micro-Manager") > 0:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def is_coordinate_in_range(settings: Dict[str, Any], position: Position) -> bool:
    """
    Check if position is within stage limits defined in settings.

    Args:
        settings: Dictionary containing microscope configuration
        position: Position object to check

    Returns:
        True if position is within limits, False otherwise
    """
    _within_y_limit = _within_x_limit = False

    # Check if stage limits exist in settings
    stage_limits = settings.get('stage', {}).get('limits', {})

    # Check X limits
    x_limits = stage_limits.get('x_um', {})
    if x_limits and position.x is not None:
        x_low = x_limits.get('low')
        x_high = x_limits.get('high')

        if x_low is not None and x_high is not None:
            if x_low < position.x < x_high:
                _within_x_limit = True
            else:
                logger.warning(f"X position {position.x} out of range [{x_low}, {x_high}]")
                warnings.warn(f"X position {position.x} out of range [{x_low}, {x_high}]")
        else:
            logger.warning(f"X limit values are not properly defined: {x_limits}")
            warnings.warn(f"X limit values are not properly defined: {x_limits}")
    else:
        logger.warning("X limits not found in configuration")
        warnings.warn("X limits not found in configuration")

    # Check Y limits
    y_limits = stage_limits.get('y_um', {})
    if y_limits and position.y is not None:
        y_low = y_limits.get('low')
        y_high = y_limits.get('high')

        if y_low is not None and y_high is not None:
            if y_low < position.y < y_high:
                _within_y_limit = True
            else:
                logger.warning(f"Y position {position.y} out of range [{y_low}, {y_high}]")
                warnings.warn(f"Y position {position.y} out of range [{y_low}, {y_high}]")
        else:
            logger.warning(f"Y limit values are not properly defined: {y_limits}")
            warnings.warn(f"Y limit values are not properly defined: {y_limits}")
    else:
        logger.warning("Y limits not found in configuration")
        warnings.warn("Y limits not found in configuration")

    # If no Z position specified, just check X and Y
    if position.z is None:
        return _within_x_limit and _within_y_limit

    # Check Z limits
    z_limits = stage_limits.get('z_um', {})
    if z_limits and position.z is not None:
        z_low = z_limits.get('low')
        z_high = z_limits.get('high')

        if z_low is not None and z_high is not None:
            if z_low < position.z < z_high:
                _within_z_limit = True
            else:
                logger.warning(f"Z position {position.z} out of range [{z_low}, {z_high}]")
                warnings.warn(f"Z position {position.z} out of range [{z_low}, {z_high}]")
                return False
        else:
            logger.warning(f"Z limit values are not properly defined: {z_limits}")
            warnings.warn(f"Z limit values are not properly defined: {z_limits}")
            return False
    else:
        logger.warning("Z limits not found in configuration")
        warnings.warn("Z limits not found in configuration")
        return False

    return _within_x_limit and _within_y_limit and _within_z_limit
