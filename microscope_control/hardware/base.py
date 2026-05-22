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

    def __init__(
        self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None
    ):
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


def _check_axis_in_range(axis_name, value, limits_key, stage_limits):
    """Check if a single axis value is within configured limits."""
    if value is None:
        return True
    axis_limits = stage_limits.get(limits_key, {})
    if not axis_limits:
        logger.warning("%s limits not found in configuration", axis_name)
        warnings.warn(f"{axis_name} limits not found in configuration")
        return False
    low = axis_limits.get("low")
    high = axis_limits.get("high")
    if low is None or high is None:
        logger.warning("%s limit values are not properly defined: %s", axis_name, axis_limits)
        warnings.warn(f"{axis_name} limit values are not properly defined: {axis_limits}")
        return False
    # 'low'/'high' are min/max labels in either order. An inverted-axis scope
    # naturally yields a descending pair when limits are copied from the stage
    # readout; normalize so the check never rejects every position.
    lo = min(low, high)
    hi = max(low, high)
    if lo <= value <= hi:
        return True
    logger.warning("%s position %s out of range [%s, %s]", axis_name, value, lo, hi)
    warnings.warn(f"{axis_name} position {value} out of range [{lo}, {hi}]")
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

    stage_limits = settings.get("stage", {}).get("limits", {})

    x_ok = "x" not in axes or _check_axis_in_range("X", position.x, "x_um", stage_limits)
    y_ok = "y" not in axes or _check_axis_in_range("Y", position.y, "y_um", stage_limits)
    z_ok = "z" not in axes or _check_axis_in_range("Z", position.z, "z_um", stage_limits)

    return x_ok and y_ok and z_ok
