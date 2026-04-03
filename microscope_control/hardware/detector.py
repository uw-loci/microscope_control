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
    """Single-module PMT controller (e.g. Becker & Hickl DCC-100).

    Controls a photomultiplier tube via a Micro-Manager device adapter.
    Provides on/off, gain (high-voltage percentage), and overload
    protection. All MM property names are configurable so this class
    works with any PMT controller that follows the on/off + gain pattern.

    Defaults match the BH DCC-100 adapter:
    - status_property: 'DCC100 status' ('On' / 'Off')
    - gain_property_fmt: 'Connector{connector}GainHV_Percent' (0-100)
    - overload_property: 'ClearOverload' (write 'Clear' to reset latch)

    SAFETY: The PMT can be damaged by excess light. Always disable
    the PMT before switching illumination modes or opening shutters
    to bright light.

    Args:
        core: Pycromanager Core object
        device_name: MM device name
        connector: Which PMT connector to control (1-4, default 1)
        max_gain_percent: Maximum safe gain percentage (default 100)
        status_property: MM property for on/off control
        gain_property_fmt: Format string for gain property (use {connector})
        overload_property: MM property to clear overload (None to skip)
        overload_value: Value to write to clear overload
    """

    def __init__(self, core, device_name: str = None,
                 connector: int = 1,
                 max_gain_percent: float = 100.0,
                 status_property: str = "DCC100 status",
                 gain_property_fmt: str = "Connector{connector}GainHV_Percent",
                 overload_property: Optional[str] = "ClearOverload",
                 overload_value: str = "Clear"):
        if not device_name:
            raise ValueError("device_name is required for PMTDetector")
        self._core = core
        self._device = device_name
        self._connector = connector
        self._max_gain = max_gain_percent
        self._status_property = status_property
        self._gain_property = gain_property_fmt.format(connector=connector)
        self._overload_property = overload_property
        self._overload_value = overload_value
        logger.info(
            "Initialized PMTDetector: %s connector %d (max gain %.0f%%)",
            device_name, connector, max_gain_percent,
        )

    def enable(self) -> None:
        """Power on the PMT and clear any overload state.

        The startup sequence: set status On, clear overload, confirm On.
        """
        self._core.set_property(self._device, self._status_property, "On")
        self._core.wait_for_system()
        self.clear_overload()
        # Confirm on (some controllers need a second assertion)
        self._core.set_property(self._device, self._status_property, "On")
        self._core.wait_for_system()
        logger.info("PMT enabled (%s)", self._device)

    def disable(self) -> None:
        """Power off the PMT. Always do this before bright light."""
        self._core.set_property(self._device, self._status_property, "Off")
        self._core.wait_for_system()
        logger.info("PMT disabled (%s)", self._device)

    def is_enabled(self) -> bool:
        try:
            status = self._core.get_property(self._device, self._status_property)
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

        Resets the overload state when excess photons are detected.
        Call after enable() or after recovering from an overload event.
        No-op if overload_property was set to None.
        """
        if self._overload_property:
            self._core.set_property(
                self._device, self._overload_property, self._overload_value)
            logger.debug("PMT overload cleared")

    def reset(self) -> None:
        """Full reset sequence: off -> on -> clear overload.

        Used by acquisition hooks when overload is detected mid-scan.
        """
        self.disable()
        self._core.wait_for_system()
        self.enable()
        logger.info("PMT reset complete")


class DCUDetector(Detector):
    """Multi-channel PMT controller (e.g. Becker & Hickl DCU).

    Controls photomultiplier tubes via a multi-channel PMT module in
    Micro-Manager. Each channel has independent control of HV enable,
    gain, 12V supply, and cooling. All property names are configurable
    so this class works with any multi-channel PMT controller.

    Defaults match the BH DCU adapter (property pattern: C{N}_{suffix}):
    - C{N}_EnableOutputs: 'On' / 'Off'
    - C{N}_GainHV: 0.0 - 100.0
    - C{N}_Plus12V: 'On' / 'Off'
    - C{N}_Cooling: 'On' / 'Off'
    - C{N}_CoolerVoltage / C{N}_CoolerCurrentLimit

    SAFETY: Always disable outputs before switching light paths or
    exposing the PMT to bright light.

    Args:
        core: Pycromanager Core object
        device_name: MM device name
        channel: Which PMT channel to control (1-4, default 1)
        max_gain_percent: Maximum safe gain percentage (default 100)
        channel_prefix_fmt: Format string for channel prefix (default 'C{channel}')
        enable_suffix: Property suffix for HV enable
        gain_suffix: Property suffix for gain percentage
        power_suffix: Property suffix for 12V photocathode supply
        cooling_suffix: Property suffix for Peltier cooling
        cooler_voltage_suffix: Property suffix for cooler voltage
        cooler_current_suffix: Property suffix for cooler current limit
        num_channels: Total number of channels (for disable_all_channels)
    """

    def __init__(self, core, device_name: str = None,
                 channel: int = 1,
                 max_gain_percent: float = 100.0,
                 channel_prefix_fmt: str = "C{channel}",
                 enable_suffix: str = "_EnableOutputs",
                 gain_suffix: str = "_GainHV",
                 power_suffix: str = "_Plus12V",
                 cooling_suffix: str = "_Cooling",
                 cooler_voltage_suffix: str = "_CoolerVoltage",
                 cooler_current_suffix: str = "_CoolerCurrentLimit",
                 num_channels: int = 4):
        if not device_name:
            raise ValueError("device_name is required for DCUDetector")
        self._core = core
        self._device = device_name
        self._channel = channel
        self._max_gain = max_gain_percent
        self._num_channels = num_channels
        self._prefix = channel_prefix_fmt.format(channel=channel)
        # Build property names from prefix + suffix
        self._enable_prop = f"{self._prefix}{enable_suffix}"
        self._gain_prop = f"{self._prefix}{gain_suffix}"
        self._power_prop = f"{self._prefix}{power_suffix}"
        self._cooling_prop = f"{self._prefix}{cooling_suffix}"
        self._cooler_v_prop = f"{self._prefix}{cooler_voltage_suffix}"
        self._cooler_i_prop = f"{self._prefix}{cooler_current_suffix}"
        # Store fmt + suffixes for disable_all_channels
        self._prefix_fmt = channel_prefix_fmt
        self._enable_suffix = enable_suffix
        self._power_suffix = power_suffix
        logger.info(
            "Initialized DCUDetector: %s channel %d (max gain %.0f%%)",
            device_name, channel, max_gain_percent,
        )

    def enable(self) -> None:
        """Enable PMT channel: turn on cooling, 12V supply, then HV outputs."""
        self._core.set_property(self._device, self._cooling_prop, "On")
        self._core.set_property(self._device, self._power_prop, "On")
        self._core.wait_for_system()
        self._core.set_property(self._device, self._enable_prop, "On")
        self._core.wait_for_system()
        logger.info("DCU channel %d enabled (%s)", self._channel, self._device)

    def disable(self) -> None:
        """Disable PMT channel: turn off HV outputs, then 12V."""
        self._core.set_property(self._device, self._enable_prop, "Off")
        self._core.set_property(self._device, self._power_prop, "Off")
        self._core.wait_for_system()
        logger.info("DCU channel %d disabled (%s)",
                     self._channel, self._device)

    def is_enabled(self) -> bool:
        try:
            return self._core.get_property(
                self._device, self._enable_prop) == "On"
        except Exception:
            return False

    def set_gain(self, gain: float) -> None:
        """Set PMT high-voltage gain as a fraction (0.0-1.0)."""
        percent = max(0.0, min(gain * 100, self._max_gain))
        self._core.set_property(self._device, self._gain_prop, percent)
        logger.debug("DCU channel %d gain set to %.1f%%",
                     self._channel, percent)

    def get_gain(self) -> float:
        """Get current gain as a fraction (0.0 - 1.0)."""
        try:
            return float(self._core.get_property(
                self._device, self._gain_prop)) / 100.0
        except Exception:
            return 0.0

    def get_gain_range(self) -> Tuple[float, float]:
        return (0.0, self._max_gain / 100.0)

    def set_cooling(self, enabled: bool,
                    voltage: float = 5.0,
                    current_limit: float = 2.0) -> None:
        """Configure Peltier cooling for this channel."""
        self._core.set_property(self._device, self._cooler_v_prop, voltage)
        self._core.set_property(self._device, self._cooler_i_prop, current_limit)
        state = "On" if enabled else "Off"
        self._core.set_property(self._device, self._cooling_prop, state)
        logger.debug("DCU channel %d cooling %s (%.1fV, %.1fA limit)",
                     self._channel, state, voltage, current_limit)

    def disable_all_channels(self) -> None:
        """Safety: disable outputs on ALL channels.

        Call before switching to brightfield or any bright light source.
        """
        for ch in range(1, self._num_channels + 1):
            prefix = self._prefix_fmt.format(channel=ch)
            try:
                self._core.set_property(
                    self._device, f"{prefix}{self._enable_suffix}", "Off")
                self._core.set_property(
                    self._device, f"{prefix}{self._power_suffix}", "Off")
            except Exception as e:
                logger.warning("Could not disable channel %d: %s", ch, e)
        self._core.wait_for_system()
        logger.info("All %d channels disabled (%s)",
                     self._num_channels, self._device)
