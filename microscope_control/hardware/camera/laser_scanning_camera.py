"""Laser scanning microscope camera implementation.

For galvo-based scanning microscopes where the "camera" is actually a
scanning detector. Key differences from area cameras:

- Resolution is configurable (e.g. 256, 512, 1024, 2048) rather than fixed
- "Exposure" is controlled by pixel dwell time (1/PixelRateHz)
- Images are always square and monochrome (grayscale 16-bit from PMT)
- No Bayer filter, no debayering, no white balance
- Pixel size depends on scan resolution and zoom factor

All MM property names, valid resolutions, and valid pixel rates are read
from the detector config dict so this class works with any scanning engine
(OpenScan, PrairieView, ScanImage, etc.) that exposes resolution and
pixel rate as MM device properties.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from pycromanager import Core, Studio

from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera

logger = logging.getLogger(__name__)


class LaserScanningCamera(PycromanagerCamera):
    """Laser scanning microscope (galvo-based scanning detector).

    Controls a scanning detector via Micro-Manager. The "image" is
    assembled from point samples as the galvo sweeps across the field.

    All device property names are configurable via detector_config:
    - resolution_property: MM property for scan grid size (default 'LSM-Resolution')
    - pixel_rate_property: MM property for pixel clock Hz (default 'LSM-PixelRateHz')
    - valid_resolutions: List of accepted resolution values (default [256, 512, 1024, 2048])
    - base_pixel_size_um: Pixel size at base resolution -- REQUIRED, no default

    The actual detector (PMT) is controlled separately via the Detector
    abstraction. This class handles only the scan engine and image readout.
    """

    def __init__(self, core: Core, studio: Optional[Studio],
                 detector_config: Optional[Dict[str, Any]] = None):
        super().__init__(core, studio, detector_config)

        cfg = self._detector_config or {}

        # MM property names (configurable per scan engine)
        self._resolution_property = cfg.get("resolution_property", "LSM-Resolution")
        self._pixel_rate_property = cfg.get("pixel_rate_property", "LSM-PixelRateHz")

        # Valid values (configurable per hardware)
        self._valid_resolutions: List[int] = cfg.get(
            "valid_resolutions", [256, 512, 1024, 2048]
        )

        # Base pixel size at minimum resolution -- must come from config
        self._base_pixel_size_um = cfg.get("base_pixel_size_um")
        if self._base_pixel_size_um is None:
            logger.warning(
                "base_pixel_size_um not set in detector config for %s. "
                "Pixel size calculations will be incorrect.", self._name,
            )
            self._base_pixel_size_um = 1.0  # safe fallback, obviously wrong

        # Base resolution for pixel size scaling (smallest valid resolution)
        self._base_resolution = min(self._valid_resolutions) if self._valid_resolutions else 256

        # Current resolution (read from device)
        self._resolution = self._read_resolution()
        logger.info(
            "Initialized LaserScanningCamera: %s (resolution=%d, "
            "res_prop=%s, rate_prop=%s)",
            self._name, self._resolution,
            self._resolution_property, self._pixel_rate_property,
        )

    # --- Resolution control ---

    def get_resolution(self) -> int:
        """Get current scan resolution (pixels per line, square image)."""
        return self._resolution

    def set_resolution(self, resolution: int) -> None:
        """Set scan resolution.

        Args:
            resolution: Pixels per line (must be in valid_resolutions)
        """
        if resolution not in self._valid_resolutions:
            raise ValueError(
                f"Invalid resolution {resolution}. "
                f"Valid values: {self._valid_resolutions}"
            )
        self._core.set_property(self._name, self._resolution_property, resolution)
        self._core.wait_for_device(self._name)
        self._resolution = resolution
        logger.info("LSM resolution set to %d", resolution)

    def _read_resolution(self) -> int:
        """Read current resolution from device."""
        try:
            return int(self._core.get_property(self._name, self._resolution_property))
        except Exception:
            return self._base_resolution

    # --- Pixel rate (dwell time) control ---

    def get_pixel_rate_hz(self) -> float:
        """Get current pixel clock rate in Hz."""
        try:
            return float(self._core.get_property(self._name, self._pixel_rate_property))
        except Exception:
            return 250000.0

    def set_pixel_rate_hz(self, rate_hz: float) -> None:
        """Set pixel clock rate.

        The dwell time per pixel = 1 / rate_hz. Lower rates give
        longer integration time and better SNR but slower scanning.

        Args:
            rate_hz: Pixel clock rate in Hz
        """
        rate_str = f"{rate_hz:.4f}"
        self._core.set_property(self._name, self._pixel_rate_property, rate_str)
        self._core.wait_for_device(self._name)
        logger.info("LSM pixel rate set to %s Hz", rate_str)

    def get_dwell_time_us(self) -> float:
        """Get pixel dwell time in microseconds (derived from pixel rate)."""
        rate = self.get_pixel_rate_hz()
        if rate > 0:
            return 1e6 / rate
        return 0.0

    # --- Camera ABC overrides ---

    def snap_image(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Capture a scanned image.

        The OSc-LSM returns a grayscale 16-bit image. No debayering
        or channel reordering needed.
        """
        self._stop_streaming_before_snap()
        self._core.snap_image()
        tagged_image = self._core.get_tagged_image()
        from collections import OrderedDict
        tags = OrderedDict(sorted(tagged_image.tags.items()))

        pixels = tagged_image.pix
        height = self._resolution
        width = self._resolution
        pixels = pixels.reshape(height, width)

        return pixels, tags

    def set_exposure(self, exposure_ms: float) -> None:
        """Set 'exposure' by adjusting pixel rate.

        For a scanning microscope, exposure = dwell_time * num_pixels.
        This method adjusts the pixel rate to approximate the requested
        per-pixel dwell time. Note: total frame time also depends on
        resolution.

        Args:
            exposure_ms: Desired per-pixel dwell time in milliseconds.
                Converted to pixel rate internally.
        """
        if exposure_ms <= 0:
            return
        # Convert ms dwell time to Hz pixel rate
        rate_hz = 1e3 / exposure_ms
        self.set_pixel_rate_hz(rate_hz)

    def get_exposure(self) -> float:
        """Get per-pixel dwell time in milliseconds."""
        return self.get_dwell_time_us() / 1000.0

    def get_fov_pixels(self) -> Tuple[int, int]:
        """Return (width, height) -- always square for scanning."""
        res = self._resolution
        return res, res

    def get_pixel_size_um(self) -> float:
        """Pixel size scales inversely with resolution.

        At base resolution, pixel size = base_pixel_size_um.
        At higher resolutions, pixel size = base * base_resolution / resolution.
        """
        return self._base_pixel_size_um * self._base_resolution / self._resolution

    def extract_green_channel(self, img: np.ndarray) -> np.ndarray:
        """LSM images are already grayscale -- return as-is."""
        if img.ndim == 2:
            return img.astype(np.float32)
        # Shouldn't happen, but handle gracefully
        return np.mean(img, axis=2).astype(np.float32)

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        """LSM produces square monochrome images."""
        return self._resolution, self._resolution, 1

    # --- Capability flags ---

    def supports_per_channel_exposure(self) -> bool:
        return False

    def supports_hardware_white_balance(self) -> bool:
        return False

    # --- LSM does not need debayering ---

    def _should_debayer(self) -> bool:
        return False

    def _pre_snap_setup(self) -> None:
        """No special pre-snap setup for LSM (PMT/shutter handled by Detector)."""
        pass
