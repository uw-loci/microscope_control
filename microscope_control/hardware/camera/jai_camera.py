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
        # Cache last-applied settings to skip redundant hardware I/O.
        # During acquisition, the same angle gets identical settings for every tile.
        self._last_applied = None
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
        """Set per-channel exposure times in milliseconds."""
        self._last_applied = None
        self.properties.set_channel_exposures(red, green, blue,
                                              auto_enable=auto_enable)

    def get_channel_exposures(self) -> Dict[str, float]:
        """Get per-channel exposure times.

        Returns:
            Dict with 'red', 'green', 'blue' keys in milliseconds
        """
        return self.properties.get_channel_exposures()

    def set_unified_gain(self, gain: float) -> None:
        """Set unified gain applied to all channels (1.0 - 8.0x)."""
        self._last_applied = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_unified_gain(%.2f) -- before: Gain=%s, GainIsIndividual=%s",
                gain,
                self._safe_get("Gain"),
                self._safe_get("GainIsIndividual"),
            )
        self.properties.set_unified_gain(gain)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_unified_gain -- after: Gain=%s, GainIsIndividual=%s",
                self._safe_get("Gain"),
                self._safe_get("GainIsIndividual"),
            )

    def get_unified_gain(self) -> float:
        """Get unified gain value."""
        return self.properties.get_unified_gain()

    def set_rb_analog_gains(self, analog_red: float, analog_blue: float) -> None:
        """Set per-channel analog gains for red and blue."""
        self._last_applied = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_rb_analog_gains(R=%.3f, B=%.3f) -- before: aR=%s, aB=%s",
                analog_red, analog_blue,
                self._safe_get("Gain_AnalogRed"),
                self._safe_get("Gain_AnalogBlue"),
            )
        self.properties.set_rb_analog_gains(red=analog_red, blue=analog_blue)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_rb_analog_gains -- after: aR=%s, aB=%s",
                self._safe_get("Gain_AnalogRed"),
                self._safe_get("Gain_AnalogBlue"),
            )

    def _safe_get(self, prop: str) -> str:
        """Read a camera property, returning '?' on failure."""
        try:
            return self._core.get_property("JAICamera", prop)
        except Exception:
            return "?"

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

    def apply_settings(self, exposures, unified_gain=1.0, analog_red=1.0,
                        analog_blue=1.0, individual_exposure=True,
                        force=False):
        """Apply all camera settings atomically, stopping streaming once.

        JAI camera properties cannot be changed while streaming. This method
        stops streaming once, applies all settings, then returns. The caller
        restarts streaming if needed.

        Uses a cache to skip redundant hardware I/O when the same settings
        are applied repeatedly (e.g., same angle across tiles in acquisition).

        Args:
            force: If True, bypass cache and always apply settings.
        """
        # Build cache key from all settings
        exp_key = tuple(sorted(exposures.items()))
        settings_key = (individual_exposure, exp_key, unified_gain,
                        round(analog_red, 4), round(analog_blue, 4))

        if not force and self._last_applied == settings_key:
            logger.debug("JAI apply_settings: skipped (same as last applied)")
            return

        self.stop_if_streaming()

        if individual_exposure:
            self.properties.enable_individual_exposure()
            self.properties.set_channel_exposures(
                red=exposures.get("r", exposures.get("all", 1.0)),
                green=exposures.get("g", exposures.get("all", 1.0)),
                blue=exposures.get("b", exposures.get("all", 1.0)),
            )
        else:
            self.properties.disable_individual_exposure()
            exp = exposures.get("all", exposures.get("g", 1.0))
            self._core.set_exposure(exp)

        self.set_unified_gain(unified_gain)
        self.set_rb_analog_gains(analog_red=analog_red, analog_blue=analog_blue)
        self.disable_individual_gain()

        self._last_applied = settings_key

        logger.info(
            "JAI apply_settings: mode=%s, exp=%s, gain=%.2f, aR=%.3f, aB=%.3f",
            "individual" if individual_exposure else "unified",
            exposures, unified_gain, analog_red, analog_blue,
        )

    def clear_awb_corrections(self) -> None:
        """Clear accumulated auto-white-balance corrections."""
        self._last_applied = None
        self.properties.clear_awb_corrections()

    def enable_individual_exposure(self) -> None:
        """Enable per-channel exposure mode."""
        self._last_applied = None
        self.properties.enable_individual_exposure()

    def disable_individual_exposure(self) -> None:
        """Disable per-channel exposure mode (use unified exposure)."""
        self._last_applied = None
        self.properties.disable_individual_exposure()

    def enable_individual_gain(self) -> None:
        """Enable per-channel gain mode."""
        self.properties.enable_individual_gain()

    def disable_individual_gain(self) -> None:
        """Disable per-channel gain mode (use unified gain)."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.disable_individual_gain -- before: GainIsIndividual=%s, Gain=%s",
                self._safe_get("GainIsIndividual"),
                self._safe_get("Gain"),
            )
        self.properties.disable_individual_gain()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.disable_individual_gain -- after: GainIsIndividual=%s, Gain=%s",
                self._safe_get("GainIsIndividual"),
                self._safe_get("Gain"),
            )
