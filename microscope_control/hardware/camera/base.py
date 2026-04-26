"""Abstract base class for microscope cameras.

Defines the interface that all camera implementations must provide.
Generic cameras do software white balance and may require debayering.
Specialized cameras (e.g. JAI 3-CCD prism) override methods as needed.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Camera(ABC):
    """Abstract base class for microscope cameras.

    Subclasses must implement all abstract methods. Concrete methods
    provide sensible defaults that subclasses may override.
    """

    # --- Abstract methods (must be implemented by all cameras) ---

    @abstractmethod
    def snap_image(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Capture a single image.

        Returns:
            Tuple of (image_array, metadata_tags).
            image_array is (H, W) for grayscale or (H, W, 3) for RGB.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the camera device name (e.g. 'JAICamera', 'MicroPublisher6')."""
        ...

    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """Set camera exposure time in milliseconds."""
        ...

    @abstractmethod
    def get_exposure(self) -> float:
        """Get current camera exposure time in milliseconds."""
        ...

    @abstractmethod
    def get_fov_pixels(self) -> Tuple[int, int]:
        """Return field of view as (width_px, height_px)."""
        ...

    @abstractmethod
    def get_pixel_size_um(self) -> float:
        """Return pixel size in micrometers (from MM calibration)."""
        ...

    @abstractmethod
    def extract_green_channel(self, img: np.ndarray) -> np.ndarray:
        """Extract a grayscale/green channel suitable for autofocus scoring.

        Different camera types extract the green channel differently:
        - Bayer-filter cameras: extract green pixels from the Bayer pattern
        - 3-CCD prism cameras (JAI): take mean across RGB channels
        - Monochrome cameras: return image as-is

        Args:
            img: Raw image from snap_image(), shape (H, W) or (H, W, 3)

        Returns:
            2D grayscale array suitable for focus metric computation
        """
        ...

    # --- Concrete methods with defaults ---

    def get_fov_um(self) -> Tuple[float, float]:
        """Return field of view in micrometers as (fov_x_um, fov_y_um)."""
        w, h = self.get_fov_pixels()
        px = self.get_pixel_size_um()
        return w * px, h * px

    # --- Optical flip (per-detector) ---

    @property
    def flip_x(self) -> bool:
        """Whether this detector's image is optically flipped on the X axis.

        Optical flip is a property of the light path between the sample
        and this specific detector. Different detectors on the same
        microscope may have different flip states (e.g. a brightfield
        camera may be flipped relative to a laser scanning detector
        because they use different optical paths).

        This is NOT stage axis inversion -- see the CLAUDE.md coordinate
        system terminology section.
        """
        return False

    @property
    def flip_y(self) -> bool:
        """Whether this detector's image is optically flipped on the Y axis."""
        return False

    def supports_per_channel_exposure(self) -> bool:
        """Whether the camera supports independent per-channel exposure control."""
        return False

    def supports_hardware_white_balance(self) -> bool:
        """Whether the camera supports hardware-level white balance."""
        return False

    # --- Binning (optional capability) ---
    # MM cameras typically expose a "Binning" property whose allowed values
    # describe the supported binning factors. Cameras whose MM device does
    # NOT expose that property (or for which binning is meaningless) keep
    # the safe default of [1] from the base class. PycromanagerCamera and
    # subclasses override these to read/write the actual MM property.

    def get_available_binnings(self) -> "list[int]":
        """Return supported binning factors as ascending ints.

        Default: ``[1]`` (no binning). Subclasses with MM "Binning" support
        override to query ``core.get_allowed_property_values``.
        """
        return [1]

    def get_binning(self) -> int:
        """Return the current binning factor.

        Default: 1. Subclasses override to read MM "Binning" property.
        """
        return 1

    def set_binning(self, value: int) -> None:
        """Set the binning factor. No-op when only ``[1]`` is supported.

        Subclasses override to write MM "Binning" property. Callers should
        treat this as best-effort and re-query ``get_binning()`` if the
        post-write value matters (some cameras snap to nearest supported).
        """
        if value != 1:
            logger.warning(
                "set_binning(%d) ignored: %s reports no binning support",
                value, self.get_name(),
            )

    # --- Exposure / gain ranges (Camera Control v2 phase 2) ---
    # Used by GETCAP to populate the dialog spinners with hardware-honest
    # bounds instead of hardcoded constants in the Java UI. PycromanagerCamera
    # overrides these to read the MM property limits; cameras that don't expose
    # limits return safe wide defaults.

    def get_min_exposure_ms(self) -> float:
        """Return the camera's minimum supported exposure in ms. Default: 0.01."""
        return 0.01

    def get_max_exposure_ms(self) -> float:
        """Return the camera's maximum supported exposure in ms. Default: 10000."""
        return 10000.0

    def get_gain_range(self) -> "tuple[float, float] | None":
        """Return (min, max) gain, or None when the camera doesn't expose gain.

        Default: None. Subclasses with MM "Gain" property override.
        """
        return None

    def start_continuous_acquisition(self) -> None:
        """Start continuous frame acquisition into a circular buffer.

        Default: no-op. Override if the camera supports live streaming.
        """
        logger.warning("start_continuous_acquisition not implemented for %s", self.get_name())

    def stop_continuous_acquisition(self) -> None:
        """Stop continuous acquisition. Safe to call if not running.

        Default: no-op.
        """
        pass

    def get_live_frame(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Get the latest frame from the circular buffer.

        Returns:
            Tuple of (image_array, metadata) or (None, None) if unavailable.
        """
        return None, None

    def is_streaming(self) -> bool:
        """Whether the camera is currently in continuous acquisition mode."""
        return False

    def stop_if_streaming(self) -> None:
        """Stop continuous acquisition if currently running. Safe no-op otherwise."""
        if self.is_streaming():
            self.stop_continuous_acquisition()

    def white_balance(self, img, background_image=None, gain=1.0,
                      white_balance_profile=None, settings=None):
        """Apply software white balance correction to an image.

        Default implementation applies RGB multiplier scaling.
        Cameras with hardware white balance may not need this.

        Args:
            img: Input image array (H, W, 3), uint8
            background_image: Optional background for per-pixel WB
            gain: Luminance gain multiplier
            white_balance_profile: [R_mult, G_mult, B_mult] scaling factors
            settings: Optional settings dict for default WB lookup

        Returns:
            White-balanced image as uint8
        """
        if white_balance_profile is None:
            if settings is not None:
                wb_settings = settings.get("white_balance", {})
                default_wb = wb_settings.get("default", {}).get("default", [1.0, 1.0, 1.0])
                white_balance_profile = default_wb
            else:
                white_balance_profile = [1.0, 1.0, 1.0]

        if img is None:
            raise ValueError("Input image 'img' must not be None for white balancing.")

        # Guard: monochrome images (2D) cannot be white-balanced with RGB
        # multipliers. Return the image with only luminance gain applied.
        if img.ndim == 2:
            if gain != 1.0:
                img_wb = img.astype(np.float64) * gain
                return np.clip(img_wb, 0, 255).astype(np.uint8)
            return img

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)

    # --- Per-channel exposure/gain control (optional capabilities) ---
    # Default implementations are no-ops or return neutral values.
    # Cameras with per-channel control (e.g. JAI 3-CCD) override these.
    # Handler code checks supports_per_channel_exposure() before calling.

    def get_channel_exposures(self) -> Dict[str, float]:
        """Get per-channel exposure times in ms.

        Returns:
            Dict with 'red', 'green', 'blue' keys. Default: all equal
            to the unified exposure.
        """
        exp = self.get_exposure()
        return {"red": exp, "green": exp, "blue": exp}

    def set_channel_exposures(self, red: float, green: float, blue: float,
                              auto_enable: bool = True) -> None:
        """Set independent per-channel exposure times.

        Default: sets unified exposure to the green channel value.
        Cameras with per-channel support override this.
        """
        self.set_exposure(green)

    def is_individual_exposure_enabled(self) -> bool:
        """Whether the camera is currently in per-channel exposure mode."""
        return False

    def enable_individual_exposure(self) -> None:
        """Switch to per-channel exposure mode. No-op if not supported."""
        pass

    def disable_individual_exposure(self) -> None:
        """Switch to unified exposure mode. No-op if not supported."""
        pass

    def get_unified_gain(self) -> float:
        """Get the unified (all-channel) gain. Default: 1.0."""
        return 1.0

    def set_unified_gain(self, gain: float) -> None:
        """Set unified gain for all channels. No-op if not supported."""
        pass

    def get_rb_analog_gains(self) -> Dict[str, float]:
        """Get per-channel analog gains for R/B color correction.

        Returns:
            Dict with 'analog_red', 'analog_blue' keys. Default: 1.0 each.
        """
        return {"analog_red": 1.0, "analog_blue": 1.0}

    def set_rb_analog_gains(self, analog_red: float, analog_blue: float) -> None:
        """Set per-channel R/B analog gains. No-op if not supported."""
        pass

    def enable_individual_gain(self) -> None:
        """Switch to per-channel gain mode. No-op if not supported."""
        pass

    def disable_individual_gain(self) -> None:
        """Switch to unified gain mode. No-op if not supported."""
        pass

    def apply_settings(
        self,
        exposures: Dict[str, float],
        unified_gain: float = 1.0,
        analog_red: float = 1.0,
        analog_blue: float = 1.0,
        individual_exposure: bool = True,
    ) -> None:
        """Apply camera mode, exposures, and gains atomically.

        Consolidates enable/disable_individual_exposure + set_exposure /
        set_channel_exposures + set_unified_gain + set_rb_analog_gains into
        one call. Subclasses that need to stop streaming before changing
        properties do so once instead of per-setting.

        Default implementation calls the individual methods sequentially.
        Override for cameras that benefit from batching (e.g. JAI).

        Args:
            exposures: {'r': ms, 'g': ms, 'b': ms} for per-channel,
                       or {'all': ms} for unified.
            unified_gain: Unified gain value (1.0 = no gain)
            analog_red: R analog gain (1.0 = no correction)
            analog_blue: B analog gain (1.0 = no correction)
            individual_exposure: True for per-channel, False for unified
        """
        if individual_exposure and self.supports_per_channel_exposure():
            self.enable_individual_exposure()
            self.set_channel_exposures(
                red=exposures.get("r", exposures.get("all", 50.0)),
                green=exposures.get("g", exposures.get("all", 50.0)),
                blue=exposures.get("b", exposures.get("all", 50.0)),
                auto_enable=False,
            )
        else:
            self.disable_individual_exposure()
            # Use green or 'all' as the unified exposure
            exp = exposures.get("all", exposures.get("g", 50.0))
            self._core.set_exposure(exp)

        self.set_unified_gain(unified_gain)
        self.set_rb_analog_gains(analog_red=analog_red, analog_blue=analog_blue)
        self.disable_individual_gain()

    # --- Per-property state tracking ---
    # Tracks the last value sent to hardware for each property, allowing
    # setters to skip redundant I/O when the value hasn't changed.
    _tracked_state = None  # Initialized lazily on first _update_tracked_state

    def invalidate_settings_state(self):
        """Reset tracked state so next setter calls go through to hardware."""
        self._tracked_state = None

    def _state_matches(self, key, value):
        """Check if tracked state matches incoming value.

        Uses tolerance of 0.005 for float comparisons.
        """
        if self._tracked_state is None:
            return False
        tracked = self._tracked_state.get(key)
        if tracked is None:
            return False
        if isinstance(value, float) and isinstance(tracked, float):
            return abs(tracked - value) < 0.005
        return tracked == value

    def _update_tracked_state(self, key, value):
        """Update tracked state for a property after hardware write."""
        if self._tracked_state is None:
            self._tracked_state = {}
        self._tracked_state[key] = value

    def clear_awb_corrections(self) -> None:
        """Clear any automatic white balance corrections. No-op if not supported."""
        pass

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Return (width, height, num_channels) of captured images.

        Default: derived from get_fov_pixels with 3 channels assumed.
        Override for cameras with different channel counts.
        """
        w, h = self.get_fov_pixels()
        return w, h, 3
