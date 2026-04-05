"""JAI AP-3200T-USB 3-CCD prism camera implementation.

The JAI camera uses a 3-sensor prism design (no Bayer filter), meaning:
- No debayering needed (each sensor captures R, G, or B directly)
- Green channel for autofocus: simple mean across RGB channels
- White balance via per-channel exposure and analog gain (hardware-level)
- Frame rate must be adjusted before setting exposure
- WhiteBalance property must be set to Off before snap to prevent
  active color adjustment during acquisition
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
from pycromanager import Core, Studio

from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera

logger = logging.getLogger(__name__)


class JAICamera(PycromanagerCamera):
    """JAI AP-3200T-USB 3-CCD prism camera.

    Overrides the generic PycromanagerCamera behavior for:
    - No debayering (prism design)
    - Green channel extraction via mean across RGB
    - Exposure requires frame rate adjustment
    - Per-channel exposure and gain control
    - Hardware white balance
    """

    # Frame rate limits (Hz) for the JAI AP-3200T-USB
    FRAME_RATE_MIN = 0.125
    FRAME_RATE_MAX = 38.0

    def __init__(self, core: Core, studio: Optional[Studio],
                 detector_config: Optional[Dict[str, Any]] = None):
        super().__init__(core, studio, detector_config)

        # Lazily create JAICameraProperties when first needed
        self._properties = None
        logger.info("Initialized JAICamera (3-CCD prism, no debayering)")

    @property
    def properties(self):
        """Lazily-initialized JAICameraProperties instance.

        Returns the JAI-specific property manager for direct access
        to per-channel exposure, gain, black level, etc.
        """
        if self._properties is None:
            from microscope_control.jai import JAICameraProperties
            self._properties = JAICameraProperties(self._core)
        return self._properties

    # --- Overridden abstract methods ---

    def extract_green_channel(self, img: np.ndarray) -> np.ndarray:
        """Extract grayscale from RGB via mean across channels.

        The JAI 3-CCD prism design captures true RGB (not Bayer).
        A simple mean across channels gives a good grayscale for
        autofocus scoring.
        """
        if img.ndim == 3:
            return np.mean(img, axis=2).astype(np.float32)
        return img.astype(np.float32)

    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure with frame rate adjustment.

        The JAI camera has hardware-coupled exposure and frame rate:
        the maximum exposure depends on the frame rate. We compute
        the minimum frame rate that allows the requested exposure
        and set both.
        """
        margin = 1.01
        exposure_s = exposure_ms / 1000.0
        required_frame_rate = round(1.0 / (exposure_s * margin), 3)
        frame_rate = min(max(required_frame_rate, self.FRAME_RATE_MIN),
                         self.FRAME_RATE_MAX)

        self._core.set_property("JAICamera", "FrameRateHz", frame_rate)
        self._core.set_property("JAICamera", "Exposure", exposure_ms)
        self._core.wait_for_device(self._name)

    def get_fov_pixels(self) -> Tuple[int, int]:
        """Get FOV from detector config (JAI dimensions not in device properties)."""
        if "width_px" in self._detector_config and "height_px" in self._detector_config:
            return (self._detector_config["width_px"],
                    self._detector_config["height_px"])
        # Fallback to image dimensions from core
        return self._core.get_image_width(), self._core.get_image_height()

    # --- Capability flags ---

    def supports_per_channel_exposure(self) -> bool:
        return True

    def supports_hardware_white_balance(self) -> bool:
        return True

    # --- Protected hooks ---

    def _pre_snap_setup(self) -> None:
        """Set WhiteBalance to Off before snap.

        The JAI camera's active white balance (Continuous/Once modes)
        adjusts colors in real-time. We must disable it before
        acquisition to get consistent, calibratable images.
        """
        try:
            try:
                temp = self._core.get_property("JAICamera", "Temperature")
                logger.debug("    JAI sensor temperature: %s", temp)
            except Exception:
                pass
            self._core.set_property("JAICamera", "WhiteBalance", "Off")
        except Exception as e:
            logger.debug("    WhiteBalance property not writable (non-fatal): %s", e)

    def _should_debayer(self) -> bool:
        """JAI 3-CCD prism camera never needs debayering."""
        return False

    # --- JAI-specific methods (not on base Camera) ---

    def set_channel_exposures(self, red: float, green: float, blue: float,
                              auto_enable: bool = True) -> None:
        """Set per-channel exposure times in milliseconds.

        Args:
            red: Red channel exposure (ms)
            green: Green channel exposure (ms)
            blue: Blue channel exposure (ms)
            auto_enable: Auto-enable individual exposure mode if needed
        """
        self.properties.set_channel_exposures(red, green, blue,
                                              auto_enable=auto_enable)

    def get_channel_exposures(self) -> Dict[str, float]:
        """Get per-channel exposure times.

        Returns:
            Dict with 'red', 'green', 'blue' keys in milliseconds
        """
        return self.properties.get_channel_exposures()

    def set_unified_gain(self, gain: float) -> None:
        """Set unified gain applied to all channels (1.0 - 8.0x).

        Delegates to JAICameraProperties which handles clamping,
        conditional GainIsIndividual toggle, and read-back verification.
        """
        self.properties.set_unified_gain(gain)

    def get_unified_gain(self) -> float:
        """Get unified gain value."""
        return self.properties.get_unified_gain()

    def set_rb_analog_gains(self, analog_red: float, analog_blue: float) -> None:
        """Set per-channel analog gains for red and blue.

        Args:
            analog_red: Red analog gain (0.47 - 4.0x)
            analog_blue: Blue analog gain (0.47 - 4.0x)
        """
        self.properties.set_rb_analog_gains(red=analog_red, blue=analog_blue)

    def get_rb_analog_gains(self) -> Dict[str, float]:
        """Get current red/blue analog gain values.

        Returns:
            Dict with 'analog_red', 'analog_blue' keys (Camera ABC convention).
            Delegates to JAICameraProperties which returns 'red'/'blue' keys,
            then remaps to the ABC key names.
        """
        props = self.properties.get_rb_analog_gains()
        return {
            "analog_red": props.get("red", 1.0),
            "analog_blue": props.get("blue", 1.0),
        }

    def clear_awb_corrections(self) -> None:
        """Clear accumulated auto-white-balance corrections.

        Delegates to JAICameraProperties which sets WhiteBalance=Off
        (with wait_for_device to clear AWB state) and resets analog
        R/B gains to 1.0 for a clean starting state.
        """
        self.properties.clear_awb_corrections()

    def enable_individual_exposure(self) -> None:
        """Enable per-channel exposure mode."""
        self.properties.enable_individual_exposure()

    def disable_individual_exposure(self) -> None:
        """Disable per-channel exposure mode (use unified exposure)."""
        self.properties.disable_individual_exposure()

    def enable_individual_gain(self) -> None:
        """Enable per-channel gain mode."""
        self.properties.enable_individual_gain()

    def disable_individual_gain(self) -> None:
        """Disable per-channel gain mode (use unified gain)."""
        self.properties.disable_individual_gain()
