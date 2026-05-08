"""Rotation stage abstraction.

Provides a RotationStage ABC and concrete implementations for
different rotation stage hardware used in polarized light microscopy.
Each implementation handles the angle-to-device-position conversion
for its specific hardware.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RotationStage(ABC):
    """Abstract base class for polarization rotation stages.

    All angles are in degrees (birefringence/PSG angle space).
    Implementations handle conversion to device-specific units.
    """

    @abstractmethod
    def set_angle(self, theta: float) -> None:
        """Set rotation angle (blocking -- waits for stage to arrive).

        Args:
            theta: Target angle in degrees (birefringence angle space)
        """
        ...

    @abstractmethod
    def set_angle_no_wait(self, theta: float) -> None:
        """Set rotation angle without waiting for completion.

        Call wait() before any operation that depends on the rotation
        being complete (e.g. image acquisition).

        Args:
            theta: Target angle in degrees
        """
        ...

    @abstractmethod
    def get_angle(self) -> float:
        """Get current rotation angle in degrees.

        Returns:
            Current angle in birefringence angle space
        """
        ...

    @abstractmethod
    def home(self) -> None:
        """Home the rotation stage (blocking)."""
        ...

    @abstractmethod
    def wait(self) -> None:
        """Block until the rotation stage reaches its target position."""
        ...

    # --- Raw device-level access (for calibration) ---

    @property
    def device_name(self) -> str:
        """Return the Micro-Manager device name (e.g. 'PIZStage').

        Used by calibration code that needs to identify the device type.
        Returns empty string for virtual stages.
        """
        return ""

    @property
    def hw_per_deg(self) -> float:
        """Encoder counts per optical degree for this stage.

        Used by calibration code that sweeps in hardware units.
        Returns 1.0 by default (virtual stage).
        """
        return 1.0

    def set_raw_position(self, hw_pos: float) -> None:
        """Set raw device position in encoder counts (blocking).

        This bypasses angle conversion -- use ONLY for calibration.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Raw positioning not supported on this stage")

    def get_raw_position(self) -> float:
        """Get current raw device position in encoder counts.

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Raw positioning not supported on this stage")

    def wait_raw(self) -> None:
        """Wait for device after raw positioning. Delegates to wait()."""
        self.wait()


class PIZRotationStage(RotationStage):
    """PI nano-positioning stage for polarization rotation.

    Conversion: device_position = (theta * hw_per_deg) + offset
    where offset is a calibration constant from polarizer calibration
    and hw_per_deg is the hardware units (encoder counts) per degree
    for this specific PIZ stage model.
    """

    def __init__(self, core, device_name: str, offset: float, units_per_deg: float):
        """
        Args:
            core: Pycromanager Core object
            device_name: MM device name (e.g. "PIZStage")
            offset: Calibration offset for angle-to-position conversion
            units_per_deg: Hardware units (encoder counts) per degree.
                Must be specified in config -- different PIZ stage models
                use different scales.
        """
        self._core = core
        self._device = device_name
        self._offset = offset
        self._units_per_deg = float(units_per_deg)
        logger.info(
            "Initialized PIZRotationStage (device=%s, offset=%.1f, " "units_per_deg=%.1f)",
            device_name,
            offset,
            self._units_per_deg,
        )

    def set_angle(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        self._core.wait_for_device(self._device)
        logger.debug("Set rotation angle to %.1f deg (PIZ position: %.1f)", theta, pos)

    def set_angle_no_wait(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        logger.debug("Set rotation angle (no wait) to %.1f deg", theta)

    def get_angle(self) -> float:
        pos = self._core.get_position(self._device)
        return self._device_to_angle(pos)

    def home(self) -> None:
        self._core.home(self._device)
        self._core.wait_for_device(self._device)
        logger.debug("Homed PIZ rotation stage")

    def wait(self) -> None:
        self._core.wait_for_device(self._device)

    # --- Raw device-level access (for calibration) ---

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def hw_per_deg(self) -> float:
        return self._units_per_deg

    def set_raw_position(self, hw_pos: float) -> None:
        self._core.set_position(self._device, hw_pos)
        self._core.wait_for_device(self._device)

    def get_raw_position(self) -> float:
        return self._core.get_position(self._device)

    # --- Internal conversion ---

    def _angle_to_device(self, theta: float) -> float:
        """Convert birefringence angle (degrees) to PIZ stage position."""
        return (theta * self._units_per_deg) + self._offset

    def _device_to_angle(self, position: float) -> float:
        """Convert PIZ stage position to birefringence angle (degrees)."""
        return (position - self._offset) / self._units_per_deg


class ThorRotationStage(RotationStage):
    """Thorlabs KBD101 rotation mount for polarization rotation.

    Conversion: device_position = -units_per_deg * theta + offset
    where units_per_deg and offset are hardware-specific constants
    read from config.
    """

    def __init__(self, core, device_name: str, units_per_deg: float, offset: float):
        """
        Args:
            core: Pycromanager Core object
            device_name: MM device name (e.g. "KBD101_Thor_Rotation")
            units_per_deg: Hardware units per degree for this stage model.
                Must be specified in config.
            offset: Calibration offset for angle-to-position conversion.
                Must be specified in config.
        """
        self._core = core
        self._device = device_name
        self._units_per_deg = float(units_per_deg)
        self._offset = float(offset)
        logger.info(
            "Initialized ThorRotationStage (device=%s, " "units_per_deg=%.1f, offset=%.1f)",
            device_name,
            self._units_per_deg,
            self._offset,
        )

    def set_angle(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        self._core.wait_for_device(self._device)
        logger.debug("Set rotation angle to %.1f deg (Thor position: %.1f)", theta, pos)

    def set_angle_no_wait(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        logger.debug("Set rotation angle (no wait) to %.1f deg", theta)

    def get_angle(self) -> float:
        pos = self._core.get_position(self._device)
        return self._device_to_angle(pos)

    def home(self) -> None:
        self._core.home(self._device)
        self._core.wait_for_device(self._device)
        logger.debug("Homed Thor rotation stage")

    def wait(self) -> None:
        self._core.wait_for_device(self._device)

    # --- Raw device-level access (for calibration) ---

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def hw_per_deg(self) -> float:
        return self._units_per_deg

    def set_raw_position(self, hw_pos: float) -> None:
        self._core.set_position(self._device, hw_pos)
        self._core.wait_for_device(self._device)

    def get_raw_position(self) -> float:
        return self._core.get_position(self._device)

    # --- Internal conversion ---

    def _angle_to_device(self, theta: float) -> float:
        """Convert birefringence angle (degrees) to Thor stage position."""
        return -self._units_per_deg * theta + self._offset

    def _device_to_angle(self, position: float) -> float:
        """Convert Thor stage position to birefringence angle (degrees)."""
        return (self._offset - position) / self._units_per_deg


class DummyRotationStage(RotationStage):
    """In-memory rotation stage for when optics are disabled (ppm_optics=NA).

    Stores the angle in memory without touching hardware. Used when
    the physical rotation stage exists but polarization optics are
    not installed.
    """

    def __init__(self):
        self._angle = 0.0
        logger.info("Initialized DummyRotationStage (no hardware)")

    def set_angle(self, theta: float) -> None:
        self._angle = theta

    def set_angle_no_wait(self, theta: float) -> None:
        self._angle = theta

    def get_angle(self) -> float:
        return self._angle

    def home(self) -> None:
        self._angle = 0.0

    def wait(self) -> None:
        pass  # No hardware to wait for
