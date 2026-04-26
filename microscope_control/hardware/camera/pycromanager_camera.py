"""Generic Micro-Manager camera implementation.

Handles any standard camera connected through Micro-Manager, including
Bayer-filter cameras (e.g. MicroPublisher6) and monochrome cameras.
Camera-specific subclasses (e.g. JAICamera) override methods as needed.
"""

import time
import logging
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

import numpy as np
from pycromanager import Core, Studio

from microscope_control.hardware.camera.base import Camera

logger = logging.getLogger(__name__)


class PycromanagerCamera(Camera):
    """Generic Micro-Manager camera.

    Supports any camera accessible through the MM Core API. Handles
    Bayer-filter debayering (configurable), channel reordering from
    BGRA to RGB, and alpha channel removal.

    For cameras that need specialized behavior (JAI 3-CCD, laser
    scanning, etc.), subclass this and override the relevant methods.
    """

    def __init__(self, core: Core, studio: Optional[Studio],
                 detector_config: Optional[Dict[str, Any]] = None):
        """Initialize with MM Core connection and optional detector config.

        Args:
            core: Pycromanager Core object
            studio: Pycromanager Studio object (may be None)
            detector_config: Optional dict from YAML config with detector
                settings (width_px, height_px, bayer_pattern, bit_depth, etc.)
        """
        self._core = core
        self._studio = studio
        self._detector_config = detector_config or {}
        # Use device name from config if available; fall back to current MM active camera
        self._name = self._detector_config.get("device") or core.get_property("Core", "Camera")

        # Detect camera capabilities from config or defaults
        self._bayer_pattern = self._detector_config.get("bayer_pattern", None)
        self._bit_depth = self._detector_config.get("bit_depth", None)

        # Per-detector optical flip (read from config)
        self._flip_x = bool(self._detector_config.get("flip_x", False))
        self._flip_y = bool(self._detector_config.get("flip_y", False))

        logger.info("Initialized PycromanagerCamera: %s (flip_x=%s, flip_y=%s)",
                    self._name, self._flip_x, self._flip_y)

    # --- Core access (for subclasses and internal use) ---

    @property
    def core(self) -> Core:
        """Access to the underlying MM Core (for subclasses)."""
        return self._core

    @property
    def studio(self) -> Optional[Studio]:
        """Access to the underlying MM Studio (for subclasses)."""
        return self._studio

    # --- Optical flip (per-detector, from config) ---

    @property
    def flip_x(self) -> bool:
        """Whether this detector's image is flipped on X (from detector config)."""
        return self._flip_x

    @property
    def flip_y(self) -> bool:
        """Whether this detector's image is flipped on Y (from detector config)."""
        return self._flip_y

    # --- Abstract method implementations ---

    def get_name(self) -> str:
        return self._name

    def snap_image(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Capture a single image via MM Core.

        Stops any running sequence acquisition before snapping.
        Applies debayering for Bayer-filter cameras if configured.

        Kwargs:
            remove_alpha: Remove alpha channel from BGRA images (default True)
            debayering: Debayering mode - "auto" (default), True, or False.
                "auto" uses camera type detection; False suppresses debayering
                even for Bayer cameras (needed for raw data workflows).

        Returns:
            Tuple of (image_array, metadata_tags)
        """
        remove_alpha = kwargs.get("remove_alpha", True)
        debayering = kwargs.get("debayering", "auto")
        t_snap_start = time.perf_counter()

        # Stop any running sequence before snapping
        self._stop_streaming_before_snap()
        t_live_check = time.perf_counter()
        logger.debug("    [TIMING] Check/stop live mode: %.1fms",
                     (t_live_check - t_snap_start) * 1000)

        # Pre-snap camera setup (subclasses override for camera-specific prep)
        self._pre_snap_setup()

        # Handle debayering: "auto" uses camera type detection, True forces it,
        # False suppresses it (needed for raw Bayer data in PPM workflows)
        if debayering == "auto":
            needs_debayer = self._should_debayer()
        elif debayering:
            needs_debayer = self._should_debayer()
        else:
            needs_debayer = False
        if needs_debayer:
            self._set_debayer_mode_for_snap(color_off=True)

        # Capture
        t_cam_start = time.perf_counter()
        self._core.snap_image()
        t_cam_snap = time.perf_counter()
        logger.debug("    [TIMING] Camera snap: %.1fms",
                     (t_cam_snap - t_cam_start) * 1000)

        tagged_image = self._core.get_tagged_image()
        t_cam_transfer = time.perf_counter()
        logger.debug("    [TIMING] Buffer transfer: %.1fms",
                     (t_cam_transfer - t_cam_snap) * 1000)

        # Parse metadata
        tags = OrderedDict(sorted(tagged_image.tags.items()))

        # Reshape pixels
        pixels = tagged_image.pix
        total_pixels = pixels.shape[0]
        height, width = tags["Height"], tags["Width"]
        assert (total_pixels % (height * width)) == 0
        nchannels = total_pixels // (height * width)

        if nchannels > 1:
            pixels = pixels.reshape(height, width, nchannels)
        else:
            pixels = pixels.reshape(height, width)

        # Apply debayering if needed
        if needs_debayer:
            pixels = self._debayer_image(pixels)
            self._set_debayer_mode_for_snap(color_off=False)
            return pixels, tags

        # Channel reordering for multi-channel cameras
        pixels = self._reorder_channels(pixels, nchannels, remove_alpha)

        return pixels, tags

    def set_exposure(self, exposure_ms: float) -> None:
        self._core.set_exposure(exposure_ms)
        self._core.wait_for_device(self._name)

    def get_exposure(self) -> float:
        return self._core.get_exposure()

    def get_fov_pixels(self) -> Tuple[int, int]:
        """Get FOV in pixels from detector config or device properties."""
        # Prefer detector config if available
        if "width_px" in self._detector_config and "height_px" in self._detector_config:
            return (self._detector_config["width_px"],
                    self._detector_config["height_px"])

        # Fall back to device properties
        try:
            width = int(self._core.get_property(self._name, "X-dimension"))
            height = int(self._core.get_property(self._name, "Y-dimension"))
            return width, height
        except Exception:
            # Last resort: use core image dimensions
            return self._core.get_image_width(), self._core.get_image_height()

    def get_pixel_size_um(self) -> float:
        return self._core.get_pixel_size_um()

    # --- Binning ---
    # Reads / writes / queries MM's "Binning" device property. Cameras that
    # don't expose that property (or for which the allowed-values query
    # fails) fall back gracefully to [1] / no-op via the base class
    # behaviour -- the helpers here catch and log so a missing property
    # doesn't break GETCAP responses.

    def get_available_binnings(self) -> "list[int]":
        try:
            values = self._core.get_allowed_property_values(self._name, "Binning")
        except Exception as e:
            logger.debug("Camera %s exposes no Binning allowed-values: %s",
                         self._name, e)
            return [1]
        out: set = set()
        for v in values:
            try:
                # MM sometimes reports "1x1" / "2x2" formats; sometimes just
                # the integer factor. Accept both, normalize to int.
                s = str(v).strip()
                if "x" in s.lower():
                    s = s.lower().split("x", 1)[0]
                out.add(int(s))
            except (ValueError, TypeError):
                continue
        return sorted(out) if out else [1]

    def get_binning(self) -> int:
        try:
            raw = str(self._core.get_property(self._name, "Binning")).strip()
            if "x" in raw.lower():
                raw = raw.lower().split("x", 1)[0]
            return int(raw)
        except Exception as e:
            logger.debug("Camera %s could not read Binning: %s", self._name, e)
            return 1

    def set_binning(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"Binning value must be >= 1, got {value}")
        # Probe allowed values once so we honour whatever the camera
        # accepts (some MM drivers want "2x2", some want "2"). Default
        # encoding is the bare integer.
        try:
            allowed = self._core.get_allowed_property_values(self._name, "Binning")
        except Exception:
            allowed = []
        target = str(value)
        for candidate in allowed:
            cs = str(candidate).strip()
            try:
                if cs == target:
                    target = cs
                    break
                if "x" in cs.lower() and cs.lower().split("x", 1)[0] == target:
                    target = cs
                    break
            except Exception:
                continue
        self._core.set_property(self._name, "Binning", target)
        logger.info("Camera %s binning set to %s", self._name, target)

    def extract_green_channel(self, img: np.ndarray) -> np.ndarray:
        """Extract green channel from Bayer-pattern image for autofocus.

        For raw Bayer images, extracts the two green sub-pixels and averages.
        For already-debayered RGB images, takes the green channel.
        For monochrome images, returns as-is.
        """
        if img.ndim == 2:
            # Monochrome or raw Bayer -- if Bayer, extract green pixels
            if self._bayer_pattern:
                green1 = img[0::2, 0::2]
                green2 = img[1::2, 1::2]
                return ((green1 + green2) / 2.0).astype(np.float32)
            return img.astype(np.float32)

        # RGB image -- take green channel
        if img.ndim == 3 and img.shape[2] >= 2:
            return img[:, :, 1].astype(np.float32)

        return img.astype(np.float32)

    # --- Streaming / live mode ---

    def start_continuous_acquisition(self) -> None:
        try:
            if self._core.is_sequence_running():
                logger.debug("Sequence already running, not starting another")
                return
            self._core.clear_circular_buffer()
            self._core.start_continuous_sequence_acquisition(0)
            logger.info("Continuous sequence acquisition started (core-level)")
        except Exception as e:
            logger.error("Failed to start continuous acquisition: %s", e)
            raise

    def stop_continuous_acquisition(self) -> None:
        try:
            if not self._core.is_sequence_running():
                logger.debug("No sequence running, nothing to stop")
                return
            self._core.stop_sequence_acquisition()
            time.sleep(0.05)
            # Verify the stop actually took effect
            if self._core.is_sequence_running():
                logger.warning("Sequence still running after stop -- retrying")
                self._core.stop_sequence_acquisition()
                time.sleep(0.1)
                if self._core.is_sequence_running():
                    logger.warning(
                        "Sequence STILL running after retry -- "
                        "attempting studio live mode off"
                    )
                    if self._studio is not None:
                        try:
                            self._studio.live().set_live_mode(False)
                            time.sleep(0.1)
                        except Exception as e2:
                            logger.error(
                                "Failed to stop via studio fallback: %s", e2
                            )
            logger.info("Continuous sequence acquisition stopped")
        except Exception as e:
            logger.error("Failed to stop continuous acquisition: %s", e)
            raise

    def get_live_frame(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        try:
            remaining = self._core.get_remaining_image_count()
            if remaining == 0:
                return None, None

            pixels = self._core.get_last_image()
            if pixels is None:
                return None, None

            width = self._core.get_image_width()
            height = self._core.get_image_height()
            nchannels = self._core.get_number_of_components()

            # Validate pixel count matches current dimensions before reshape.
            # After an ROI change the circular buffer may still hold a frame
            # captured at the OLD dimensions while get_image_width/height
            # already reflect the NEW ROI.  Reshaping would crash; skip the
            # stale frame instead and let the next call pick up a good one.
            expected = int(height) * int(width) * max(int(nchannels), 1)
            actual = pixels.size
            if actual != expected:
                logger.debug(
                    "Skipping stale live frame: pixel count %d does not match "
                    "current dimensions %dx%dx%d (expected %d) -- likely ROI "
                    "transition race",
                    actual, width, height, nchannels, expected,
                )
                return None, None

            processed = self._process_raw_image(pixels, width, height, nchannels)

            meta = {
                "width": processed.shape[1],
                "height": processed.shape[0],
                "channels": 1 if processed.ndim == 2 else processed.shape[2],
                "bytesPerPixel": processed.dtype.itemsize,
            }

            return processed, meta

        except Exception as e:
            msg = str(e)
            if "Circular buffer" in msg or "buffer is empty" in msg:
                logger.debug("Live frame buffer empty (race condition, harmless)")
            else:
                logger.error("get_live_frame failed: %s", e)
            return None, None

    def is_streaming(self) -> bool:
        try:
            return self._core.is_sequence_running()
        except Exception:
            return False

    # --- Protected methods for subclass customization ---

    def _pre_snap_setup(self) -> None:
        """Hook for subclasses to do camera-specific setup before snap.

        Called after stopping any running sequence but before the actual
        snap_image() call. Default: no-op.
        """
        pass

    def _should_debayer(self) -> bool:
        """Whether this camera needs Bayer-pattern debayering.

        Default: True if camera is MicroPublisher6 (the only known
        Bayer camera in the current hardware fleet). Override in
        subclasses that do NOT need debayering.
        """
        return self._name == "MicroPublisher6"

    def _debayer_image(self, pixels: np.ndarray) -> np.ndarray:
        """Apply Bayer-pattern debayering to raw sensor data.

        Args:
            pixels: Raw sensor data, shape (H, W) or (H, W, nch)

        Returns:
            Debayered RGB image, uint16
        """
        from microscope_imageprocessing.debayering import CPUDebayer

        pattern = self._bayer_pattern or "GRBG"
        bit_depth = self._bit_depth or 14

        debayerx = CPUDebayer(
            pattern=pattern,
            image_bit_clipmax=(2 ** bit_depth) - 1,
            image_dtype=np.uint16,
            convolution_mode="wrap",
        )

        pixels = debayerx.debayer(pixels)
        logger.debug("Before bit scaling: mean %s", pixels.mean((0, 1)))
        # Scale to 16-bit range
        pixels = ((pixels.astype(np.float32) / ((2 ** bit_depth) - 1)) * 65535).astype(np.uint16)
        pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
        logger.debug("After bit scaling: mean %s", pixels.mean((0, 1)))
        return pixels

    def _set_debayer_mode_for_snap(self, color_off: bool) -> None:
        """Toggle camera Color property for Bayer debayering.

        MicroPublisher6 needs Color=OFF before snap so we get raw
        Bayer data, then Color=ON after debayering.
        """
        if self._name == "MicroPublisher6":
            self._core.set_property("MicroPublisher6", "Color",
                                    "OFF" if color_off else "ON")

    def _reorder_channels(self, pixels: np.ndarray, nchannels: int,
                          remove_alpha: bool = True) -> np.ndarray:
        """Reorder channels from BGRA to RGB and optionally remove alpha.

        Args:
            pixels: Image array, possibly (H, W, 4) BGRA
            nchannels: Number of channels detected
            remove_alpha: Whether to strip the alpha channel

        Returns:
            Reordered image array
        """
        if nchannels > 1 and pixels.ndim == 3:
            pixels = pixels[:, :, ::-1]  # BGRA to ARGB
            if remove_alpha and pixels.shape[2] == 4 and self._name != "QCamera":
                pixels = pixels[:, :, 1:]  # Remove alpha -> RGB
        return pixels

    def _stop_streaming_before_snap(self) -> None:
        """Stop any running sequence or live mode before a snap."""
        if self._core.is_sequence_running():
            try:
                self._core.stop_sequence_acquisition()
                time.sleep(0.1)
                logger.debug("Stopped sequence acquisition via core before snap")
            except Exception as e:
                logger.warning("Failed to stop sequence via core: %s", e)
                if self._studio is not None:
                    try:
                        self._studio.live().set_live_mode(False)
                        time.sleep(0.1)
                    except Exception as e2:
                        logger.warning("Failed to stop live mode via studio: %s", e2)

    def _process_raw_image(self, pixels, width, height, nchannels,
                           debayering="auto"):
        """Shared post-processing for raw pixel data from snap or live buffer.

        Handles debayering, channel reordering (BGRA->RGB), and alpha removal.

        Args:
            pixels: Raw pixel array from MM
            width: Image width in pixels
            height: Image height in pixels
            nchannels: Number of channels
            debayering: Debayering mode ("auto", True, False)

        Returns:
            Processed numpy array
        """
        # Reshape if flat
        if pixels.ndim == 1:
            if nchannels > 1:
                pixels = pixels.reshape(height, width, nchannels)
            else:
                pixels = pixels.reshape(height, width)

        # Determine debayering
        needs_debayer = False
        if debayering == "auto":
            needs_debayer = self._should_debayer()
        elif debayering:
            needs_debayer = self._should_debayer()

        if needs_debayer:
            return self._debayer_image(pixels)

        # Channel reordering
        return self._reorder_channels(pixels, nchannels)

    # --- Device properties ---

    def get_device_properties(self, scope: str = "used") -> Dict[str, Any]:
        """Get all device properties for this camera from MM.

        Args:
            scope: 'used' for current values, 'allowed' for possible values

        Returns:
            Dict of property_name -> value
        """
        props = {}
        try:
            prop_names = self._core.get_device_property_names(self._name)
            for i in range(prop_names.size()):
                name = prop_names.get(i)
                if scope == "allowed":
                    values = self._core.get_allowed_property_values(self._name, name)
                    props[name] = [values.get(j) for j in range(values.size())]
                else:
                    props[name] = self._core.get_property(self._name, name)
        except Exception as e:
            logger.warning("Failed to read camera properties: %s", e)
        return props
