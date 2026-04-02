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

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """Return (width, height, num_channels) of captured images.

        Default: derived from get_fov_pixels with 3 channels assumed.
        Override for cameras with different channel counts.
        """
        w, h = self.get_fov_pixels()
        return w, h, 3
