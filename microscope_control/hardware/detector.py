"""Detector abstraction for photon-counting and analog detectors.

Separate from Camera because some microscope systems have detectors
(PMTs, APDs, hybrid detectors) that are independent devices from the
scan engine / camera. The detector controls gain, enable/disable, and
overload protection.

Reference hardware:
- Becker & Hickl DCC-100 PMT controller (controls gain, overload, on/off)
- Future: Hamamatsu H7422 PMT module, APD detectors
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Detector(ABC):
    """Abstract base class for photon detectors (PMTs, APDs, etc.).

    Detectors are separate from Cameras -- a Camera (or scan engine)
    triggers the acquisition, while the Detector amplifies and converts
    photons to signal. On systems like the CAMM, the OSc-LSM scan
    engine and the DCC-100 PMT are independent devices.
    """

    @abstractmethod
    def enable(self) -> None:
        """Power on / enable the detector."""
        ...

    @abstractmethod
    def disable(self) -> None:
        """Power off / disable the detector."""
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the detector is currently active."""
        ...

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """Set detector gain.

        Args:
            gain: Gain value (units are detector-specific: percentage,
                voltage, or arbitrary units)
        """
        ...

    @abstractmethod
    def get_gain(self) -> float:
        """Get current detector gain."""
        ...

    @abstractmethod
    def get_gain_range(self) -> Tuple[float, float]:
        """Return (min_gain, max_gain) for this detector."""
        ...


class PMTDetector(Detector):
    """Becker & Hickl DCC-100 PMT controller.

    Controls a photomultiplier tube via the DCC-100 module in
    Micro-Manager. Provides on/off, gain (high-voltage percentage),
    and overload protection.

    MM device properties used:
    - DCC100 status: 'On' / 'Off'
    - ConnectorNGainHV_Percent: 0-100 (gain as percentage of max HV)
    - ClearOverload: 'Clear' (write-only, resets overload latch)

    SAFETY: The PMT can be damaged by excess light. Always disable
    the PMT before switching illumination modes or opening shutters
    to bright light.

    Args:
        core: Pycromanager Core object
        device_name: MM device name (default 'DCC100')
        connector: Which PMT connector to control (1-4, default 1)
        max_gain_percent: Maximum safe gain percentage (default 100)
    """

    def __init__(self, core, device_name: str = "DCC100",
                 connector: int = 1,
                 max_gain_percent: float = 100.0):
        self._core = core
        self._device = device_name
        self._connector = connector
        self._max_gain = max_gain_percent
        self._gain_property = f"Connector{connector}GainHV_Percent"
        logger.info(
            "Initialized PMTDetector: %s connector %d (max gain %.0f%%)",
            device_name, connector, max_gain_percent,
        )

    def enable(self) -> None:
        """Power on the DCC-100 and clear any overload state.

        The DCC-100 requires a specific startup sequence:
        1. Set status to On
        2. Clear overload (resets protection latch)
        3. Wait for system
        4. Confirm On state (sometimes needs a second set)
        """
        self._core.set_property(self._device, "DCC100 status", "On")
        self._core.wait_for_system()
        self.clear_overload()
        # Confirm on (DCC-100 sometimes needs a second assertion)
        self._core.set_property(self._device, "DCC100 status", "On")
        self._core.wait_for_system()
        logger.info("PMT enabled (%s)", self._device)

    def disable(self) -> None:
        """Power off the DCC-100. Always do this before bright light."""
        self._core.set_property(self._device, "DCC100 status", "Off")
        self._core.wait_for_system()
        logger.info("PMT disabled (%s)", self._device)

    def is_enabled(self) -> bool:
        try:
            status = self._core.get_property(self._device, "DCC100 status")
            return status == "On"
        except Exception:
            return False

    def set_gain(self, gain: float) -> None:
        """Set PMT high-voltage gain as a percentage.

        Args:
            gain: Gain as fraction 0.0-1.0 (multiplied by 100 internally).
                For example, 0.40 sets 40% of maximum HV.
        """
        percent = max(0.0, min(gain * 100, self._max_gain))
        self._core.set_property(self._device, self._gain_property, percent)
        logger.debug("PMT gain set to %.1f%% (connector %d)",
                     percent, self._connector)

    def get_gain(self) -> float:
        """Get current gain as a fraction (0.0 - 1.0)."""
        try:
            percent = float(self._core.get_property(
                self._device, self._gain_property))
            return percent / 100.0
        except Exception:
            return 0.0

    def get_gain_range(self) -> Tuple[float, float]:
        """Return gain range as fractions."""
        return (0.0, self._max_gain / 100.0)

    def clear_overload(self) -> None:
        """Clear the overload protection latch.

        The DCC-100 latches into overload state when excess photons
        are detected. This resets it. Call after enable() or after
        recovering from an overload event during acquisition.
        """
        self._core.set_property(self._device, "ClearOverload", "Clear")
        logger.debug("PMT overload cleared")

    def reset(self) -> None:
        """Full reset sequence: off -> on -> clear overload.

        Used by acquisition hooks when overload is detected mid-scan.
        """
        self.disable()
        self._core.wait_for_system()
        self.enable()
        logger.info("PMT reset complete")
