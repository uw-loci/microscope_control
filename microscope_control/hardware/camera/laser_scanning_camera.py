"""Laser scanning microscope camera implementation.

For galvo-based scanning microscopes (e.g. OpenScan OSc-LSM) where the
"camera" is actually a scanning detector. Key differences from area cameras:

- Resolution is configurable (256, 512, 1024, 2048) rather than fixed
- "Exposure" is controlled by pixel dwell time (1/PixelRateHz)
- Images are always square and monochrome (grayscale 16-bit from PMT)
- No Bayer filter, no debayering, no white balance
- Pixel size depends on scan resolution and zoom factor

Reference: uw-loci/smart-wsi-scanner SPAcquisition class.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
from pycromanager import Core, Studio

from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera

logger = logging.getLogger(__name__)

# Valid scan resolutions for OpenScan LSM devices
VALID_RESOLUTIONS = [256, 512, 1024, 2048]

# Valid pixel rates (Hz) as strings (MM property values)
VALID_PIXEL_RATES = [
    "50000.0000", "100000.0000", "125000.0000",
    "200000.0000", "250000.0000", "400000.0000",
    "500000.0000", "625000.0000", "1000000.0000",
    "1250000.0000",
]


class LaserScanningCamera(PycromanagerCamera):
    """OpenScan OSc-LSM laser scanning microscope.

    Controls a galvo-based scanning detector via Micro-Manager. The "image"
    is assembled from point samples as the galvo sweeps across the field.

    Key MM device properties:
    - LSM-Resolution: scan grid size (256-2048, square)
    - LSM-PixelRateHz: pixel clock frequency (controls dwell time)

    The actual detector (PMT) is controlled separately via the Detector
    abstraction. This class handles only the scan engine and image readout.
    """

    def __init__(self, core: Core, studio: Optional[Studio],
                 detector_config: Optional[Dict[str, Any]] = None):
        super().__init__(core, studio, detector_config)

        # Base pixel size at 256 resolution (um/px) -- from config
        self._base_pixel_size_um = self._detector_config.get(
            "base_pixel_size_um", 0.509
        )
        # Current resolution (read from device)
        self._resolution = self._read_resolution()
        logger.info(
            "Initialized LaserScanningCamera: %s (resolution=%d)",
            self._name, self._resolution,
        )

    # --- Resolution control ---

    def get_resolution(self) -> int:
        """Get current scan resolution (pixels per line, square image)."""
        return self._resolution

    def set_resolution(self, resolution: int) -> None:
        """Set scan resolution.

        Args:
            resolution: Pixels per line (256, 512, 1024, or 2048)
        """
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution {resolution}. "
                f"Valid values: {VALID_RESOLUTIONS}"
            )
        self._core.set_property(self._name, "LSM-Resolution", resolution)
        self._core.wait_for_device(self._name)
        self._resolution = resolution
        logger.info("LSM resolution set to %d", resolution)

    def _read_resolution(self) -> int:
        """Read current resolution from device."""
        try:
            return int(self._core.get_property(self._name, "LSM-Resolution"))
        except Exception:
            return 256  # safe default

    # --- Pixel rate (dwell time) control ---

    def get_pixel_rate_hz(self) -> float:
        """Get current pixel clock rate in Hz."""
        try:
            return float(self._core.get_property(self._name, "LSM-PixelRateHz"))
        except Exception:
            return 250000.0

    def set_pixel_rate_hz(self, rate_hz: float) -> None:
        """Set pixel clock rate.

        The dwell time per pixel = 1 / rate_hz. Lower rates give
        longer integration time and better SNR but slower scanning.

        Args:
            rate_hz: Pixel clock rate in Hz (50000 - 1250000)
        """
        # MM expects rate as a specific string format
        rate_str = f"{rate_hz:.4f}"
        if rate_str not in VALID_PIXEL_RATES:
            # Find the closest valid rate
            closest = min(VALID_PIXEL_RATES,
                          key=lambda r: abs(float(r) - rate_hz))
            logger.warning(
                "Rate %.0f Hz not in valid set, using closest: %s",
                rate_hz, closest,
            )
            rate_str = closest
        self._core.set_property(self._name, "LSM-PixelRateHz", rate_str)
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

        At base resolution (256), pixel size = base_pixel_size_um.
        At higher resolutions, pixel size = base * 256 / resolution.
        """
        return self._base_pixel_size_um * 256 / self._resolution

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
