"""Hardware abstraction layer for microscope control."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, Tuple, TYPE_CHECKING
import warnings
import logging

if TYPE_CHECKING:
    from microscope_control.hardware.camera.base import Camera
    from microscope_control.hardware.stage import Stage
    from microscope_control.hardware.rotation import RotationStage
    from microscope_control.hardware.illumination import Illumination
    from microscope_control.hardware.detector import Detector

logger = logging.getLogger(__name__)


class Position:
    """Simple position class to replace sp_position dataclass."""

    def __init__(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        self.x = x
        self.y = y
        self.z = z

    def get_specified_axes(self) -> Set[str]:
        """Return the set of axes that have non-None values.

        Useful for recording which axes were originally specified before
        calling populate_missing(), so that validation can be limited to
        only the axes the caller intended to move.
        """
        axes = set()
        if self.x is not None:
            axes.add("x")
        if self.y is not None:
            axes.add("y")
        if self.z is not None:
            axes.add("z")
        return axes

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
    """Abstract base class for microscope hardware control.

    A microscope is composed of three hardware components:
    - **Camera** (required): image acquisition, exposure, white balance
    - **Stage** (required): XY and Z positioning
    - **RotationStage** (optional): polarization angle control

    Subclasses must provide ``camera`` and ``stage`` properties.
    Convenience methods delegate to the appropriate component so that
    callers can write ``hardware.snap_image()`` or
    ``hardware.move_to_position()`` without reaching into components.
    """

    # --- Component composition (subclasses must implement) ---

    @property
    @abstractmethod
    def camera(self) -> "Camera":
        """The camera attached to this microscope."""
        ...

    @property
    @abstractmethod
    def stage(self) -> "Stage":
        """The XYZ stage attached to this microscope."""
        ...

    @property
    def rotation_stage(self) -> Optional["RotationStage"]:
        """The rotation stage, or None if not configured."""
        return None

    @property
    def illumination(self) -> Optional["Illumination"]:
        """The primary illumination source, or None if not configured."""
        return None

    @property
    def detector(self) -> Optional["Detector"]:
        """An external detector (e.g. PMT), or None if not present.

        For cameras with built-in detectors (CCD/CMOS), this is None.
        For scanning microscopes with separate PMTs, this provides
        gain control, enable/disable, and overload protection.
        """
        return None

    # --- Stage convenience delegations ---

    def move_to_position(self, position: Position) -> None:
        """Move stage to the specified position (delegates to stage).

        Moves only the axes that are non-None in the Position object.
        For example, Position(x=100, y=200) moves XY without touching Z.
        """
        specified = position.get_specified_axes()
        has_xy = "x" in specified and "y" in specified
        has_z = "z" in specified

        if has_z:
            self.stage.move_z_no_wait(position.z)
        if has_xy:
            self.stage.move_xy_no_wait(position.x, position.y)
        if has_xy:
            self.stage.wait_xy()
        if has_z:
            self.stage.wait_z()

    def get_current_position(self) -> Position:
        """Get current XYZ stage position (delegates to stage)."""
        x, y, z = self.stage.get_xyz()
        return Position(x, y, z)

    def move_xy_no_wait(self, x: float, y: float) -> None:
        """Issue XY move without waiting (delegates to stage)."""
        self.stage.move_xy_no_wait(x, y)

    def wait_for_xy(self) -> None:
        """Block until XY stage arrives (delegates to stage)."""
        self.stage.wait_xy()

    def set_z_no_wait(self, z: float) -> None:
        """Issue Z move without waiting (delegates to stage)."""
        self.stage.move_z_no_wait(z)

    def get_z_position(self) -> float:
        """Get current Z position only (delegates to stage)."""
        return self.stage.get_z()

    # --- Camera convenience delegations ---

    def snap_image(self, **kwargs):
        """Capture a single image (delegates to camera)."""
        return self.camera.snap_image(**kwargs)

    def get_camera_name(self) -> str:
        """Get camera device name (delegates to camera)."""
        return self.camera.get_name()

    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure in milliseconds (delegates to camera)."""
        self.camera.set_exposure(exposure_ms)

    def get_exposure(self) -> float:
        """Get exposure in milliseconds (delegates to camera)."""
        return self.camera.get_exposure()

    def get_pixel_size_um(self) -> float:
        """Get pixel size in micrometers (delegates to camera)."""
        return self.camera.get_pixel_size_um()

    def get_fov(self) -> Tuple[float, float]:
        """Get field of view in micrometers as (fov_x, fov_y) (delegates to camera)."""
        return self.camera.get_fov_um()

    def start_continuous_acquisition(self) -> None:
        """Start continuous frame acquisition (delegates to camera)."""
        self.camera.start_continuous_acquisition()

    def stop_continuous_acquisition(self) -> None:
        """Stop continuous frame acquisition (delegates to camera)."""
        self.camera.stop_continuous_acquisition()

    def get_live_frame(self):
        """Get the latest frame from the circular buffer (delegates to camera)."""
        return self.camera.get_live_frame()

    # --- Rotation stage convenience delegations ---

    def set_psg_ticks(self, theta: float) -> None:
        """Set rotation angle in degrees (delegates to rotation_stage)."""
        rs = self.rotation_stage
        if rs is None:
            raise RuntimeError("No rotation stage configured")
        rs.set_angle(theta)

    def set_psg_ticks_no_wait(self, theta: float) -> None:
        """Set rotation angle without waiting (delegates to rotation_stage)."""
        rs = self.rotation_stage
        if rs is None:
            raise RuntimeError("No rotation stage configured")
        rs.set_angle_no_wait(theta)

    def get_psg_ticks(self) -> float:
        """Get current rotation angle in degrees (delegates to rotation_stage)."""
        rs = self.rotation_stage
        if rs is None:
            raise RuntimeError("No rotation stage configured")
        return rs.get_angle()

    def home_psg(self) -> None:
        """Home the rotation stage (delegates to rotation_stage)."""
        rs = self.rotation_stage
        if rs is None:
            raise RuntimeError("No rotation stage configured")
        rs.home()

    def wait_for_rotation(self) -> None:
        """Wait for rotation stage to reach target (delegates to rotation_stage)."""
        rs = self.rotation_stage
        if rs is None:
            return  # No-op if no rotation stage
        rs.wait()


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


def is_coordinate_in_range(
    settings: Dict[str, Any],
    position: Position,
    axes: Optional[Set[str]] = None,
) -> bool:
    """
    Check if position is within stage limits defined in settings.

    Args:
        settings: Dictionary containing microscope configuration
        position: Position object to check
        axes: Optional set of axis names ("x", "y", "z") to validate.
              When provided, only the listed axes are checked -- unlisted
              axes are treated as in-range regardless of their value.
              When None (default), every axis whose value is not None is
              validated (backward-compatible behavior).

    Returns:
        True if position is within limits, False otherwise
    """
    # Determine which axes to validate.  If the caller did not specify,
    # fall back to checking every axis that has a non-None value.
    if axes is None:
        axes = position.get_specified_axes()

    _within_x_limit = _within_y_limit = _within_z_limit = True

    # Check if stage limits exist in settings
    stage_limits = settings.get('stage', {}).get('limits', {})

    # --- X axis ---
    if "x" not in axes or position.x is None:
        # Not requested for validation or not specified -- skip
        pass
    else:
        _within_x_limit = False
        x_limits = stage_limits.get('x_um', {})
        if x_limits:
            x_low = x_limits.get('low')
            x_high = x_limits.get('high')

            if x_low is not None and x_high is not None:
                if x_low <= position.x <= x_high:
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

    # --- Y axis ---
    if "y" not in axes or position.y is None:
        pass
    else:
        _within_y_limit = False
        y_limits = stage_limits.get('y_um', {})
        if y_limits:
            y_low = y_limits.get('low')
            y_high = y_limits.get('high')

            if y_low is not None and y_high is not None:
                if y_low <= position.y <= y_high:
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

    # --- Z axis ---
    if "z" not in axes or position.z is None:
        pass
    else:
        _within_z_limit = False
        z_limits = stage_limits.get('z_um', {})
        if z_limits:
            z_low = z_limits.get('low')
            z_high = z_limits.get('high')

            if z_low is not None and z_high is not None:
                if z_low <= position.z <= z_high:
                    _within_z_limit = True
                else:
                    logger.warning(f"Z position {position.z} out of range [{z_low}, {z_high}]")
                    warnings.warn(f"Z position {position.z} out of range [{z_low}, {z_high}]")
            else:
                logger.warning(f"Z limit values are not properly defined: {z_limits}")
                warnings.warn(f"Z limit values are not properly defined: {z_limits}")
        else:
            logger.warning("Z limits not found in configuration")
            warnings.warn("Z limits not found in configuration")

    return _within_x_limit and _within_y_limit and _within_z_limit
