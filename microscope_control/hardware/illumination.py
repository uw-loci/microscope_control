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

    def __init__(
        self,
        core,
        device_name: str,
        property_name: str = "Voltage",
        min_voltage: float = 0.0,
        max_voltage: float = 5.0,
        label: str = "",
    ):
        self._core = core
        self._device = device_name
        self._property = property_name
        self._min_v = min_voltage
        self._max_v = max_voltage
        self._label = label or device_name
        logger.info(
            "Initialized AnalogIllumination: %s (%s, %.1f-%.1fV)",
            self._label,
            device_name,
            min_voltage,
            max_voltage,
        )

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


class DevicePropertyIllumination(Illumination):
    """Illumination controlled by MM device State + Intensity properties.

    For microscope-integrated light sources (Nikon DiaLamp, Lumencor,
    CoolLED, etc.) that expose on/off via a State property and brightness
    via an Intensity property -- as opposed to raw analog voltage control.

    Args:
        core: Pycromanager Core object
        device_name: MM device name (e.g. 'DiaLamp')
        state_property: MM property for on/off (default 'State')
        intensity_property: MM property for brightness (default 'Intensity')
        max_intensity: Maximum allowed intensity value (default 2100)
        label: Human-readable label for logging
    """

    def __init__(
        self,
        core,
        device_name: str,
        state_property: str = "State",
        intensity_property: str = "Intensity",
        max_intensity: float = 2100.0,
        label: str = "",
    ):
        self._core = core
        self._device = device_name
        self._state_prop = state_property
        self._intensity_prop = intensity_property
        self._max_intensity = max_intensity
        self._label = label or device_name
        logger.info(
            "Initialized DevicePropertyIllumination: %s (%s, max=%s)",
            self._label,
            device_name,
            max_intensity,
        )

    def _is_binary(self) -> bool:
        """True when this source has no separate intensity property.

        Some MM device adapters expose a single enumerated "State" property
        with discrete values like "0"/"1" and no continuous intensity
        knob. Configs declare these by pointing both ``state_property``
        and ``intensity_property`` at the same MM property name (e.g.
        Epi LED on OWS3: both = ``State``). For these sources, the
        intensity write must be a string-encoded enumerated value, never
        a float -- pycromanager's set_property serializes Python floats
        as ``"1.000000"`` which MM rejects against the enumerated allowed
        set.
        """
        return self._state_prop == self._intensity_prop

    def on(self) -> None:
        """Turn on the light source (set State=1).

        State is passed as the string "1" because pycromanager
        serializes numeric arguments to float strings like
        "1.000000", which MM rejects for discrete-valued properties
        (DiaLamp, Nikon LightPath, etc.).
        """
        self._core.set_property(self._device, self._state_prop, "1")
        logger.debug("%s turned on", self._label)

    def off(self) -> None:
        """Turn off the light source (set State=0 and Intensity=0)."""
        if not self._is_binary():
            self._core.set_property(self._device, self._intensity_prop, 0)
        self._core.set_property(self._device, self._state_prop, "0")
        logger.debug("%s turned off", self._label)

    def set_power(self, power: float) -> None:
        """Set illumination intensity.

        Automatically turns the source on if power > 0, off if power == 0.
        For binary sources (state_property == intensity_property) the
        intensity write is skipped -- the State write IS the intensity --
        and the value is encoded as the string ``"1"`` so MM doesn't
        reject ``"1.000000"`` against its enumerated allowed set.

        Args:
            power: Intensity level (clamped to 0 .. max_intensity)
        """
        intensity = max(0.0, min(power, self._max_intensity))
        if self._is_binary():
            # Binary source: only the State property exists. Treat any
            # non-zero power as "on" (State="1"), zero as "off"
            # (State="0"). Skip the redundant intensity write entirely.
            self._core.set_property(
                self._device,
                self._state_prop,
                "1" if intensity > 0 else "0",
            )
            logger.debug(
                "%s (binary) state set to %s",
                self._label,
                "ON" if intensity > 0 else "OFF",
            )
            return

        if intensity > 0:
            # String "1" avoids pycromanager float-string coercion that
            # MM rejects on discrete State properties.
            self._core.set_property(self._device, self._state_prop, "1")
        self._core.set_property(self._device, self._intensity_prop, intensity)
        if intensity == 0:
            self._core.set_property(self._device, self._state_prop, "0")
        logger.debug("%s intensity set to %.0f", self._label, intensity)

    def get_power(self) -> float:
        try:
            return float(self._core.get_property(self._device, self._intensity_prop))
        except Exception:
            return 0.0

    def get_power_range(self) -> tuple:
        return (0.0, self._max_intensity)

    def is_on(self) -> bool:
        """Check State property rather than relying on intensity > 0."""
        try:
            return str(self._core.get_property(self._device, self._state_prop)) == "1"
        except Exception:
            return False


class LEDIllumination(AnalogIllumination):
    """LED controlled by analog voltage (e.g. NI DAQ output).

    Typical setup: LED-Dev1ao0 on NI DAQ, 0-5V range.
    """

    def __init__(self, core, device_name: str = None, max_voltage: float = 5.0, label: str = "LED"):
        if not device_name:
            raise ValueError("device_name is required for LEDIllumination")
        super().__init__(
            core,
            device_name,
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

    def __init__(
        self, core, device_name: str = None, max_voltage: float = 1.0, label: str = "Pockels Cell"
    ):
        if not device_name:
            raise ValueError("device_name is required for PockelsCell")
        super().__init__(
            core,
            device_name,
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
