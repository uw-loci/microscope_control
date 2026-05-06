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

    # Frame rate limits (Hz) for the JAI AP-3200T-USB.
    #
    # FRAME_RATE_MAX 2026-05-06: lowered from 38.0 to 25.0 Hz. User
    # confirmed that the live-mode contamination bar (TODO_LIST.md
    # "Stale frame data populating a small region of the live frame")
    # is frame-rate triggered: short-exposure calibrations (uncrossed
    # Simple WB ~3 ms) auto-couple to FRAME_RATE_MAX (38 Hz) via
    # set_exposure, which is when the bar appears. Long-exposure
    # calibrations (negative -7 deg ~107 ms) auto-couple to ~9.3 Hz
    # and stay clean. The camera firmware appears to truncate trailing
    # rows in the buffer at high continuous-mode rates -- snap mode
    # is unaffected because it doesn't sustain the timing pressure.
    #
    # 25 Hz is below the empirical danger zone but still smooth for
    # live preview. If users find 25 too slow for short-exposure
    # workflows, the right answer is to bisect down (33, 30, 25, ...)
    # and pin the highest known-clean rate, or push for a JAI firmware
    # fix and bump back to 38.
    #
    # Override per-microscope via stage / camera config if a particular
    # JAI body has a different known-safe ceiling.
    FRAME_RATE_MIN = 0.125
    FRAME_RATE_MAX = 25.0

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
        """Set per-channel exposure times in milliseconds."""
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
        if self._state_matches("unified_gain", gain):
            logger.debug("set_unified_gain(%.2f): skipped (no change)", gain)
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_unified_gain(%.2f) -- before: Gain=%s, GainIsIndividual=%s",
                gain,
                self._safe_get("Gain"),
                self._safe_get("GainIsIndividual"),
            )
        self.properties.set_unified_gain(gain)
        self._update_tracked_state("unified_gain", gain)
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
        if (self._state_matches("analog_red", analog_red)
                and self._state_matches("analog_blue", analog_blue)):
            logger.debug(
                "set_rb_analog_gains(R=%.3f, B=%.3f): skipped (no change)",
                analog_red, analog_blue,
            )
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.set_rb_analog_gains(R=%.3f, B=%.3f) -- before: aR=%s, aB=%s",
                analog_red, analog_blue,
                self._safe_get("Gain_AnalogRed"),
                self._safe_get("Gain_AnalogBlue"),
            )
        self.properties.set_rb_analog_gains(red=analog_red, blue=analog_blue)
        self._update_tracked_state("analog_red", analog_red)
        self._update_tracked_state("analog_blue", analog_blue)
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

        Per-property state tracking in each setter skips redundant hardware
        I/O automatically. Only properties whose values have actually changed
        are written to hardware.

        Args:
            force: If True, invalidate tracked state and apply everything.
        """
        if force:
            self.invalidate_settings_state()

        self.stop_if_streaming()

        # 2026-05-06: when caller asks for individual mode but the three
        # channel exposures actually match, drop to unified mode. The
        # multi-CCD assembly path inside the camera firmware appears to
        # have a buffer-reuse bug that produces a fixed-row stale-content
        # band on continuous-mode frames (TODO_LIST.md "Stale frame data
        # populating a small region of the live frame"). Falling through
        # to unified mode when individual isn't needed is semantically
        # identical -- same exposure on each CCD either way -- but
        # avoids triggering the suspect firmware path. Tolerance is
        # tight (1 us) since the JAI exposure precision is in
        # microseconds; anything closer is just float noise.
        if individual_exposure:
            r_exp = exposures.get("r", exposures.get("all", 1.0))
            g_exp = exposures.get("g", exposures.get("all", 1.0))
            b_exp = exposures.get("b", exposures.get("all", 1.0))
            if (abs(r_exp - g_exp) < 0.001
                    and abs(g_exp - b_exp) < 0.001
                    and abs(r_exp - b_exp) < 0.001):
                logger.debug(
                    "JAI apply_settings: caller asked for individual mode but "
                    "R/G/B exposures match (%.4f/%.4f/%.4f); falling through "
                    "to unified to avoid the multi-CCD assembly bug",
                    r_exp, g_exp, b_exp,
                )
                individual_exposure = False
                exposures = dict(exposures)
                exposures["all"] = g_exp

        # 2026-05-06: avoid the JAI continuous-mode trigger window for
        # blue exposure. When individual mode is active with unequal
        # channels and the BLUE channel exposure lands in approximately
        # 4.0 < B < 5.6 ms, the camera produces frames with a
        # fixed-row stale-content band at the bottom in continuous
        # acquisition. The trigger boundary was characterised
        # empirically (2026-05-06) on the JAI AP-3200T-USB at LOCI:
        # B = 4.0 ms is clean, B = 4.32 ms shows the bar,
        # B = 5.5 ms shows the bar, B = 5.6 ms is clean,
        # B >= ~100 ms is also clean (auto-coupled to a low frame
        # rate that escapes the trigger). The bug reproduces in
        # eBUS Player and so is below the host software (camera
        # firmware / silicon level). See:
        #   claude-reports/2026-05-06_jai-contamination-bar-internal.md
        #   claude-reports/2026-05-06_jai-contamination-bar-vendor.md
        #
        # Mitigation strategy: when the requested blue exposure lands
        # inside the trigger window, snap it to the nearer safe edge
        # (4.0 ms or 5.6 ms) and rescale R and G by the same factor to
        # PRESERVE the white-balance ratio (R/B and G/B unchanged).
        # The white-balance behaviour the calibration was designed for
        # is preserved exactly; only the absolute light level changes
        # by a few percent, which is recovered downstream by image
        # normalisation. Snap-mode acquisitions are unaffected by the
        # bug at any value, so this only matters for live preview and
        # any continuous-mode sampling (streaming AF, rapid scan).
        TRIGGER_BLUE_LOW_MS = 4.0
        TRIGGER_BLUE_HIGH_MS = 5.6
        if individual_exposure:
            r_exp = exposures.get("r", exposures.get("all", 1.0))
            g_exp = exposures.get("g", exposures.get("all", 1.0))
            b_exp = exposures.get("b", exposures.get("all", 1.0))
            if TRIGGER_BLUE_LOW_MS < b_exp < TRIGGER_BLUE_HIGH_MS:
                # Pick the nearer safe boundary
                if (b_exp - TRIGGER_BLUE_LOW_MS) <= (TRIGGER_BLUE_HIGH_MS - b_exp):
                    b_safe = TRIGGER_BLUE_LOW_MS
                else:
                    b_safe = TRIGGER_BLUE_HIGH_MS
                # Preserve WB ratio: scale R and G by the same factor.
                # Because R_new/B_new = (R_old * s)/(B_old * s) = R_old/B_old.
                scale = b_safe / b_exp
                r_safe = r_exp * scale
                g_safe = g_exp * scale
                # Hard clamp to camera minimum so we never under-shoot
                # the hardware's exposure floor while rescaling. EXPOSURE_MIN
                # in JAICameraProperties is around 0.04 ms; pick a safer
                # 0.05 ms here as a buffer.
                MIN_AFTER_RESCALE_MS = 0.05
                r_safe = max(r_safe, MIN_AFTER_RESCALE_MS)
                g_safe = max(g_safe, MIN_AFTER_RESCALE_MS)
                logger.warning(
                    "JAI apply_settings: blue exposure %.4f ms in continuous-"
                    "mode contamination trigger window (%.1f, %.1f) ms; "
                    "snapping B to %.4f ms (nearer safe edge) and "
                    "rescaling R %.4f -> %.4f and G %.4f -> %.4f by factor "
                    "%.4f to preserve the WB ratio. See JAI ticket / "
                    "claude-reports/2026-05-06_jai-contamination-bar-*.md.",
                    b_exp, TRIGGER_BLUE_LOW_MS, TRIGGER_BLUE_HIGH_MS,
                    b_safe, r_exp, r_safe, g_exp, g_safe, scale,
                )
                exposures = dict(exposures)
                exposures["r"] = r_safe
                exposures["g"] = g_safe
                exposures["b"] = b_safe
                # If after rescaling all three are equal (within 1 us)
                # we can drop to unified mode below; recheck.
                r_exp, g_exp, b_exp = r_safe, g_safe, b_safe
                if (abs(r_exp - g_exp) < 0.001
                        and abs(g_exp - b_exp) < 0.001
                        and abs(r_exp - b_exp) < 0.001):
                    individual_exposure = False
                    exposures["all"] = g_exp

        if individual_exposure:
            self.enable_individual_exposure()
            self.set_channel_exposures(
                red=exposures.get("r", exposures.get("all", 1.0)),
                green=exposures.get("g", exposures.get("all", 1.0)),
                blue=exposures.get("b", exposures.get("all", 1.0)),
                auto_enable=False,
            )
        else:
            self.disable_individual_exposure()
            exp = exposures.get("all", exposures.get("g", 1.0))
            self._core.set_exposure(exp)

        self.set_unified_gain(unified_gain)
        self.set_rb_analog_gains(analog_red=analog_red, analog_blue=analog_blue)
        self.disable_individual_gain()

        logger.info(
            "JAI apply_settings: mode=%s, exp=%s, gain=%.2f, aR=%.3f, aB=%.3f",
            "individual" if individual_exposure else "unified",
            exposures, unified_gain, analog_red, analog_blue,
        )

    def clear_awb_corrections(self) -> None:
        """Clear accumulated auto-white-balance corrections."""
        self.invalidate_settings_state()
        self.properties.clear_awb_corrections()

    def enable_individual_exposure(self) -> None:
        """Enable per-channel exposure mode."""
        if self._state_matches("individual_exposure", True):
            logger.debug("enable_individual_exposure: skipped (no change)")
            return
        self.properties.enable_individual_exposure()
        self._update_tracked_state("individual_exposure", True)

    def disable_individual_exposure(self) -> None:
        """Disable per-channel exposure mode (use unified exposure)."""
        if self._state_matches("individual_exposure", False):
            logger.debug("disable_individual_exposure: skipped (no change)")
            return
        self.properties.disable_individual_exposure()
        self._update_tracked_state("individual_exposure", False)

    def enable_individual_gain(self) -> None:
        """Enable per-channel gain mode."""
        self.properties.enable_individual_gain()

    def disable_individual_gain(self) -> None:
        """Disable per-channel gain mode (use unified gain)."""
        if self._state_matches("individual_gain", False):
            logger.debug("disable_individual_gain: skipped (no change)")
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.disable_individual_gain -- before: GainIsIndividual=%s, Gain=%s",
                self._safe_get("GainIsIndividual"),
                self._safe_get("Gain"),
            )
        self.properties.disable_individual_gain()
        self._update_tracked_state("individual_gain", False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "JAICamera.disable_individual_gain -- after: GainIsIndividual=%s, Gain=%s",
                self._safe_get("GainIsIndividual"),
                self._safe_get("Gain"),
            )
