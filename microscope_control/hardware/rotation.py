"""Rotation stage abstraction.

Provides a RotationStage ABC and concrete implementations for
different rotation stage hardware used in polarized light microscopy.
Each implementation handles the angle-to-device-position conversion
for its specific hardware.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

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


class PIZRotationStage(RotationStage):
    """PI nano-positioning stage for polarization rotation.

    Conversion: device_position = (theta * 1000) + offset
    where offset is a calibration constant (default 50280.0).
    """

    def __init__(self, core, device_name: str, offset: float = 50280.0):
        """
        Args:
            core: Pycromanager Core object
            device_name: MM device name (e.g. "PIZStage")
            offset: Calibration offset for angle-to-position conversion
        """
        self._core = core
        self._device = device_name
        self._offset = offset
        logger.info("Initialized PIZRotationStage (device=%s, offset=%.1f)",
                    device_name, offset)

    def set_angle(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        self._core.wait_for_device(self._device)
        logger.debug("Set rotation angle to %.1f deg (PIZ position: %.1f)",
                     theta, pos)

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

    def _angle_to_device(self, theta: float) -> float:
        """Convert birefringence angle (degrees) to PIZ stage position."""
        return (theta * 1000) + self._offset

    def _device_to_angle(self, position: float) -> float:
        """Convert PIZ stage position to birefringence angle (degrees)."""
        return (position - self._offset) / 1000.0


class ThorRotationStage(RotationStage):
    """Thorlabs KBD101 rotation mount for polarization rotation.

    Conversion: device_position = -2 * theta + 276
    """

    def __init__(self, core, device_name: str):
        """
        Args:
            core: Pycromanager Core object
            device_name: MM device name (e.g. "KBD101_Thor_Rotation")
        """
        self._core = core
        self._device = device_name
        logger.info("Initialized ThorRotationStage (device=%s)", device_name)

    def set_angle(self, theta: float) -> None:
        pos = self._angle_to_device(theta)
        self._core.set_position(self._device, pos)
        self._core.wait_for_device(self._device)
        logger.debug("Set rotation angle to %.1f deg (Thor position: %.1f)",
                     theta, pos)

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

    @staticmethod
    def _angle_to_device(theta: float) -> float:
        """Convert birefringence angle (degrees) to Thor stage position."""
        return -2 * theta + 276

    @staticmethod
    def _device_to_angle(position: float) -> float:
        """Convert Thor stage position to birefringence angle (degrees)."""
        return (276 - position) / 2


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
