"""Illumination source abstraction.

Provides control for microscope light sources: LEDs, lasers (via Pockels
cells or direct power control), arc lamps, etc. Each source has on/off
control and a power/intensity level.

Reference hardware:
- LED via NI DAQ analog output (0-5V voltage control)
- Pockels cell via NI DAQ analog output (0-1V, controls laser transmission)
- Direct laser power control (percentage or mW)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class Illumination(ABC):
    """Abstract base class for microscope illumination sources.

    All illumination sources support on/off and a power level. The
    meaning of "power" is source-specific (voltage, percentage, mW).
    """

    @abstractmethod
    def on(self) -> None:
        """Turn the illumination source on."""
        ...

    @abstractmethod
    def off(self) -> None:
        """Turn the illumination source off (zero output)."""
        ...

    @abstractmethod
    def set_power(self, power: float) -> None:
        """Set illumination power/intensity.

        Args:
            power: Power level (units are source-specific)
        """
        ...

    @abstractmethod
    def get_power(self) -> float:
        """Get current illumination power level."""
        ...

    @abstractmethod
    def get_power_range(self) -> tuple:
        """Return (min_power, max_power) for this source."""
        ...

    def is_on(self) -> bool:
        """Whether the source is currently emitting."""
        return self.get_power() > 0


class AnalogIllumination(Illumination):
    """Illumination controlled by an analog voltage output.

    Wraps a Micro-Manager device that accepts a Voltage property
    (e.g. NI DAQ analog output channels). Power = voltage.

    Args:
        core: Pycromanager Core object
        device_name: MM device name (e.g. 'LED-Dev1ao0')
        property_name: MM property for voltage control (default 'Voltage')
        min_voltage: Minimum voltage (default 0.0)
        max_voltage: Maximum voltage (default 5.0)
    """

    def __init__(self, core, device_name: str,
                 property_name: str = "Voltage",
                 min_voltage: float = 0.0,
                 max_voltage: float = 5.0,
                 label: str = ""):
        self._core = core
        self._device = device_name
        self._property = property_name
        self._min_v = min_voltage
        self._max_v = max_voltage
        self._label = label or device_name
        logger.info("Initialized AnalogIllumination: %s (%s, %.1f-%.1fV)",
                    self._label, device_name, min_voltage, max_voltage)

    def on(self) -> None:
        """Turn on at maximum voltage."""
        self.set_power(self._max_v)

    def off(self) -> None:
        """Set voltage to zero."""
        self.set_power(0.0)

    def set_power(self, power: float) -> None:
        """Set output voltage.

        Args:
            power: Voltage level (clamped to configured range)
        """
        voltage = max(self._min_v, min(power, self._max_v))
        self._core.set_property(self._device, self._property, voltage)
        logger.debug("%s voltage set to %.3f", self._label, voltage)

    def get_power(self) -> float:
        try:
            return float(self._core.get_property(self._device, self._property))
        except Exception:
            return 0.0

    def get_power_range(self) -> tuple:
        return (self._min_v, self._max_v)


class LEDIllumination(AnalogIllumination):
    """LED controlled by analog voltage (e.g. NI DAQ output).

    Typical setup: LED-Dev1ao0 on NI DAQ, 0-5V range.
    """

    def __init__(self, core, device_name: str = "LED-Dev1ao0",
                 max_voltage: float = 5.0, label: str = "LED"):
        super().__init__(
            core, device_name,
            property_name="Voltage",
            min_voltage=0.0,
            max_voltage=max_voltage,
            label=label,
        )


class PockelsCell(AnalogIllumination):
    """Pockels cell for laser power modulation.

    Controls laser transmission via electro-optic modulation.
    Voltage range is typically 0-1V (0% to ~100% transmission).
    The laser itself stays on; the Pockels cell gates the beam.

    Typical setup: PockelsCell-Dev1ao1 on NI DAQ, 0-1V range.
    """

    def __init__(self, core, device_name: str = "PockelsCell-Dev1ao1",
                 max_voltage: float = 1.0, label: str = "Pockels Cell"):
        super().__init__(
            core, device_name,
            property_name="Voltage",
            min_voltage=0.0,
            max_voltage=max_voltage,
            label=label,
        )

    def set_transmission(self, fraction: float) -> None:
        """Set laser transmission as a fraction (0.0 - 1.0).

        Convenience method -- maps directly to voltage for a 0-1V cell.

        Args:
            fraction: Transmission fraction (0.0 = blocked, 1.0 = max)
        """
        voltage = max(0.0, min(fraction, 1.0)) * self._max_v
        self.set_power(voltage)
