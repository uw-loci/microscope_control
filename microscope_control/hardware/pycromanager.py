"""Pycromanager hardware implementation for microscope control."""

import warnings
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from pycromanager import Core, Studio
from microscope_control.hardware.base import MicroscopeHardware, is_mm_running, is_coordinate_in_range, Position
from microscope_control.autofocus.core import AutofocusUtils

import numpy as np
import skimage.color
import skimage.filters
import scipy.interpolate
import matplotlib.pyplot as plt
from ppm_library.debayering import CPUDebayer

logger = logging.getLogger(__name__)


def obj_2_list(name):
    """Convert Java object to Python list."""
    return [name.get(i) for i in range(name.size())]


class MicroManagerConnectionError(Exception):
    """Raised when connection to Micro-Manager fails."""
    pass


def init_pycromanager(timeout_seconds: float = 30.0):
    """
    Initialize Pycromanager connection to Micro-Manager.

    Args:
        timeout_seconds: Maximum time to wait for connection (default 30s)

    Returns:
        Tuple of (core, studio) if successful, (None, None) if failed

    Raises:
        MicroManagerConnectionError: If connection fails with details about the failure
    """
    # Check if MM process is running
    if not is_mm_running():
        error_msg = (
            "Micro-Manager is not running.\n"
            "Please start Micro-Manager before starting the server."
        )
        logger.error(error_msg)
        raise MicroManagerConnectionError(error_msg)

    # Try to connect to Micro-Manager
    logger.info("Connecting to Micro-Manager...")
    try:
        # Core() can hang if MM is locked - pycromanager doesn't have a timeout option
        # so we just try and catch any exceptions
        core = Core()
        studio = Studio()
        core.set_timeout_ms(20000)

        # Verify connection is working by getting a simple property
        try:
            _ = core.get_version_info()
            logger.info("Successfully connected to Micro-Manager")
        except Exception as e:
            logger.warning(f"Connected but version check failed: {e}")

        return core, studio

    except Exception as e:
        error_type = type(e).__name__
        error_msg = (
            f"Failed to connect to Micro-Manager.\n"
            f"Error type: {error_type}\n"
            f"Error: {e}\n"
            f"\n"
            f"Possible causes:\n"
            f"  1. Micro-Manager is frozen/locked - restart Micro-Manager\n"
            f"  2. Another application is using Micro-Manager\n"
            f"  3. Micro-Manager crashed - check for error dialogs\n"
            f"  4. ZMQ port conflict - restart Micro-Manager"
        )
        logger.error(error_msg)
        raise MicroManagerConnectionError(error_msg) from e


def ppm_psgticks_to_thor(bi_angle: float) -> float:
    """Convert PPM angle (in degrees) to Thor rotation stage position."""
    return -2 * bi_angle + 276


def ppm_thor_to_psgticks(kinesis_pos: float) -> float:
    """Convert Thor rotation stage position to PPM angle (in degrees)."""
    return (276 - kinesis_pos) / 2


class PycromanagerHardware(MicroscopeHardware):
    """Implementation for Pycromanager-based microscopes."""

    def __init__(self, core: Core, studio: Studio, settings: Dict[str, Any]):
        """
        Initialize PycromanagerHardware with dictionary-based settings.

        Args:
            core: Pycromanager Core object
            studio: Pycromanager Studio object
            settings: Dictionary containing microscope configuration
        """
        self.core = core
        self.studio = studio
        self.settings = settings

        ## PPM Specific attributes

        ## PPM Specific attributes
        self.psg_angle = None
        self.rotation_device = None

        # Cache camera name to avoid expensive get_device_properties() calls
        # This was taking ~800ms per snap_image() call!
        self._camera_name = None

        # Log microscope info
        microscope_info = settings.get("microscope", {})
        logger.info(
            f"Initializing hardware for microscope: {microscope_info.get('name', 'Unknown')}"
        )

        # Set up microscope-specific methods based on name
        self._initialize_microscope_methods()

        # Cache the camera name immediately after initialization
        self._camera_name = self.get_device_properties()["Core"]["Camera"]
        logger.info(f"Cached camera name: {self._camera_name}")

    def _initialize_microscope_methods(self):
        """Initialize microscope-specific methods based on settings.

        This can be called both during __init__ and when settings are updated.
        """
        microscope_info = self.settings.get("microscope", {})
        microscope_name = microscope_info.get("name", "")

        if microscope_name == "PPM":
            ppm_optics_value = self.settings.get("ppm_optics", "ZCutQuartz")
            logger.info(
                f"DEBUG: ppm_optics value = {ppm_optics_value!r} (type: {type(ppm_optics_value).__name__})"
            )
            if ppm_optics_value != "NA":

                self.set_psg_ticks = self._ppm_set_psgticks
                self.get_psg_ticks = self._ppm_get_psgticks
                self.home_psg = self._ppm_home

                ppm_config = self.settings.get("modalities", {}).get("ppm", {})
                r_device_name = ppm_config.get("rotation_stage", {}).get("device")
                self.rotation_device = (
                    self.settings.get("id_stage", {}).get(r_device_name, {}).get("device")
                )
                if not self.rotation_device:
                    raise ValueError(
                        f"No rotation stage device found in configuration. "
                        f"Expected device '{r_device_name}' in id_stage section."
                    )
                try:
                    _ = self._ppm_get_psgticks()  # initialize psg_angle
                    logger.info("PPM-specific methods initialized")
                except Exception as e:
                    # logger.error("Failed to initialize PPM rotation stage", e)
                    print("Rot-stage Exception: ", e)
                    logger.info("Continuing without PPM rotation stage functionality")
            else:
                logger.info("PPM optics not installed, skipping PPM-specific methods")
                self.psg_angle = 0.0

                # self.get_psg_ticks = lambda theta: self.psg_angle
                # self.set_psg_ticks = lambda theta: (print(f"Setting psg_angle to: {theta}"), setattr(self, 'psg_angle', theta))[1]
                def dummy_get_psg_ticks():
                    logger.info("PPM optics not installed, skipping PPM-specific methods")
                    return self.psg_angle

                def dummy_set_psg_ticks(theta):
                    logger.info("PPM optics not installed, skipping PPM-specific methods")
                    self.psg_angle = theta

                def dummy_home_psg():
                    logger.info("PPM optics not installed, skipping PPM-specific methods")
                    self.psg_angle = 0.0

                self.home_psg = dummy_home_psg
                self.set_psg_ticks = dummy_set_psg_ticks
                self.get_psg_ticks = dummy_get_psg_ticks

                # TODO: change the ppm properties to mutable container
                ## MUTATBLE CONTAINER SOLUTION
                # state = {'theta': 0}
                # get_psg_ticks = lambda : state['theta']
                # set_psg_ticks = lambda theta: state.update({'theta':theta}) or theta

        if microscope_name == "CAMM":
            self.swap_objective_lens = self._camm_swap_objective_lens
            logger.info("CAMM-specific methods initialized")

    def move_to_position(self, position: Position) -> None:
        """Move stage to specified position."""
        # Get current position and populate any missing coordinates
        current_position = self.get_current_position()
        position.populate_missing(current_position)

        # Validate position is within range
        if not is_coordinate_in_range(self.settings, position):
            logger.info("Current stage limits:", self.settings["stage"])
            logger.info(f"Requested position: {position}")
            raise ValueError(f"Position out of range: {position}")

        # Get focus device from settings if available
        stage_config = self.settings.get("stage", {})
        z_stage_device = stage_config.get("z_stage", None)

        if z_stage_device and self.core.get_focus_device() != z_stage_device:
            self.core.set_focus_device(z_stage_device)

        # Move to position
        self.core.set_position(position.z)
        self.core.set_xy_position(position.x, position.y)
        self.core.wait_for_device(self.core.get_xy_stage_device())
        self.core.wait_for_device(self.core.get_focus_device())

        logger.debug(f"Moved to position: {position}")

    def get_current_position(self) -> Position:
        """Get current stage position."""
        return Position(
            self.core.get_x_position(), self.core.get_y_position(), self.core.get_position()
        )

    def snap_image(self, background_correction=False, remove_alpha=True, debayering="auto"):
        """
        Snap an image using MM Core and return img, tags.

        Args:
            background_correction: Apply background correction (if implemented)
            remove_alpha: Remove alpha channel from BGRA images
            debayering: Debayering mode:
                - "auto" (default): Automatically debayer based on camera type
                  (MicroPublisher6 requires debayering, JAI prism camera does not)
                - True: Force debayering (only applies to MicroPublisher6)
                - False: Disable debayering

        Returns:
            Tuple of (image_array, metadata_tags)
        """
        import time
        t_snap_start = time.perf_counter()

        if self.core.is_sequence_running() and self.studio is not None:
            self.studio.live().set_live_mode(False)
        t_live_check = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Check/stop live mode: {(t_live_check - t_snap_start)*1000:.1f}ms")

        # Use cached camera name instead of expensive get_device_properties() call
        # This optimization saves ~800ms per snap!
        camera = self._camera_name
        t_get_props = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Get camera name (cached): {(t_get_props - t_live_check)*1000:.1f}ms")

        # Determine if debayering should be applied
        # MicroPublisher6 has a Bayer filter and needs debayering
        # JAI prism camera does NOT need debayering (3-sensor prism design)
        needs_debayer = False
        if debayering == "auto":
            # Auto-detect based on camera type
            needs_debayer = (camera == "MicroPublisher6")
        elif debayering:
            # Explicitly requested - only works for MicroPublisher6
            needs_debayer = (camera == "MicroPublisher6")
        # If debayering=False, needs_debayer stays False

        logger.debug(f"    Camera: {camera}, Debayering: {needs_debayer} (requested: {debayering})")

        # Handle debayering for MicroPublisher6
        if needs_debayer:
            self.core.set_property("MicroPublisher6", "Color", "OFF")

        # Handle white balance for JAI
        if camera == "JAICamera":
            t_wb_start = time.perf_counter()
            self.core.set_property("JAICamera", "WhiteBalance", "Off")
            t_wb_end = time.perf_counter()
            logger.debug(f"    [TIMING-INTERNAL] Set WhiteBalance property: {(t_wb_end - t_wb_start)*1000:.1f}ms")

        # Capture image
        t_cam_start = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Pre-snap overhead (setup to snap): {(t_cam_start - t_snap_start)*1000:.1f}ms")
        self.core.snap_image()
        t_cam_snap = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Camera snap (exposure+readout): {(t_cam_snap - t_cam_start)*1000:.1f}ms")

        tagged_image = self.core.get_tagged_image()
        t_cam_transfer = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Get tagged image (buffer transfer): {(t_cam_transfer - t_cam_snap)*1000:.1f}ms")

        # Sort tags for consistency
        tags = OrderedDict(sorted(tagged_image.tags.items()))

        # Process pixels
        t_proc_start = time.perf_counter()
        pixels = tagged_image.pix
        total_pixels = pixels.shape[0]
        height, width = tags["Height"], tags["Width"]
        assert (total_pixels % (height * width)) == 0
        nchannels = total_pixels // (height * width)

        if nchannels > 1:
            pixels = pixels.reshape(height, width, nchannels)
        else:
            pixels = pixels.reshape(height, width)
        t_reshape = time.perf_counter()
        logger.debug(f"    [TIMING-INTERNAL] Pixel reshape: {(t_reshape - t_proc_start)*1000:.1f}ms")

        # Apply debayering if needed
        if needs_debayer:
            debayerx = CPUDebayer(
                pattern="GRBG",
                image_bit_clipmax=(2**14) - 1,
                image_dtype=np.uint16,
                convolution_mode="wrap",
            )

            pixels = debayerx.debayer(pixels)
            logger.debug(f"Before uint14->uint16 scaling: mean {pixels.mean((0, 1))}")
            # Scale 14-bit sensor data to 16-bit range, preserving precision
            # Old code converted to 8-bit which caused quantization artifacts
            pixels = ((pixels.astype(np.float32) / ((2**14) - 1)) * 65535).astype(np.uint16)
            pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
            logger.debug(f"After uint14->uint16 scaling: mean {pixels.mean((0, 1))}")
            self.core.set_property("MicroPublisher6", "Color", "ON")

            return pixels, tags

        # Handle different camera types
        if camera in ["QCamera", "MicroPublisher6", "JAICamera"]:
            if nchannels > 1:
                t_color_start = time.perf_counter()
                pixels = pixels[:, :, ::-1]  # BGRA to ARGB
                if (camera != "QCamera") and remove_alpha:
                    pixels = pixels[:, :, 1:]  # Remove alpha channel
                t_color_end = time.perf_counter()
                logger.debug(f"    [TIMING-INTERNAL] Color channel processing: {(t_color_end - t_color_start)*1000:.1f}ms")

        elif camera == "OSc-LSM":
            pass
        else:
            logger.error(
                f"Capture Failed: Unrecognized camera: {tags.get('Core-Camera', 'Unknown')}"
            )
            return None, None

        return pixels, tags

    def get_fov(self) -> Tuple[float, float]:
        """
        Get field of view in micrometers.

        Returns:
            Tuple of (fov_x, fov_y) in micrometers
        """
        camera = self.core.get_property("Core", "Camera")

        if camera == "OSc-LSM":
            height = int(self.core.get_property(camera, "LSM-Resolution"))
            width = height
        elif camera == "JAICamera":
            height = self.settings["id_detector"]["LOCI_DETECTOR_JAI_001"]["height_px"]
            width = self.settings["id_detector"]["LOCI_DETECTOR_JAI_001"]["width_px"]

        elif camera in ["QCamera", "MicroPublisher6"]:
            height = int(self.core.get_property(camera, "Y-dimension"))
            width = int(self.core.get_property(camera, "X-dimension"))

        else:
            raise ValueError(f"Unknown camera type: {camera}")

        pixel_size_um = self.core.get_pixel_size_um()
        fov_y = height * pixel_size_um
        fov_x = width * pixel_size_um

        return fov_x, fov_y

    def set_exposure(self, exposure_ms: float) -> None:
        """Set camera exposure time in milliseconds."""
        camera = self.core.get_property("Core", "Camera")
        if camera == "JAICamera":
            frame_rate_min = 0.125
            frame_rate_max = 38.0
            margin = 1.01
            exposure_s = exposure_ms / 1000.0
            required_frame_rate = round(1.0 / (exposure_s * margin), 3)
            frame_rate = min(max(required_frame_rate, frame_rate_min), frame_rate_max)
            self.core.set_property("JAICamera", "FrameRateHz", frame_rate)
            self.core.set_property("JAICamera", "Exposure", exposure_ms)
        else:
            self.core.set_exposure(exposure_ms)
        self.core.wait_for_device(camera)

    def autofocus(
        self,
        n_steps=4000000,
        search_range=20,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=skimage.filters.sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
        raise_on_invalid_peak=True,
        diagnostic_output_path=None,
        position_index=None,
    ):
        """
        Perform autofocus using specified score metric.

        Args:
            n_steps: Number of Z positions to sample
            search_range: Total Z range to search in micrometers
            interp_strength: Interpolation density factor
            interp_kind: Type of interpolation ('linear', 'quadratic', 'cubic')
            score_metric: Function to score image focus
            pop_a_plot: Whether to show a focus score plot
            move_stage_to_estimate: Whether to move to best focus position
            raise_on_invalid_peak: If True (default), raises RuntimeError when peak validation fails,
                                  stopping acquisitions. If False, logs warning and continues
                                  (use for diagnostic tests where plots must always be generated).
            diagnostic_output_path: Optional path to save autofocus diagnostic CSV (for standard autofocus during acquisition)
            position_index: Optional position index to include in CSV filename

        Returns:
            float: Best focus Z position (in micrometers) on success
            dict: Failure information dict with keys:
                - 'success': False
                - 'message': Error message
                - 'quality_score': Focus quality score
                - 'peak_prominence': Peak prominence value
                - 'attempted_z': Z position that was attempted
                - 'original_z': Original Z position before autofocus
                - 'validation': Full validation dict
        """
        steps = np.linspace(0, search_range, n_steps) - (search_range / 2)

        current_pos = self.get_current_position()
        z_steps = current_pos.z + steps
        # print(z_steps)
        try:
            scores = []
            for step_number in range(n_steps):
                new_pos = Position(current_pos.x, current_pos.y, current_pos.z + steps[step_number])
                self.move_to_position(new_pos)

                img, tags = self.snap_image()

                # Extract green channel for focus calculation
                if self.core.get_property("Core", "Camera") == "JAICamera":
                    img_gray = np.mean(img, 2)
                else:
                    # TODO: debayer to go to gray ?
                    # TODO support other cameras!
                    green1 = img[0::2, 0::2]
                    green2 = img[1::2, 1::2]
                    img_gray = ((green1 + green2) / 2.0).astype(np.float32)

                score = score_metric(img_gray)
                if hasattr(score, "ndim") and score.ndim == 2:
                    score = np.mean(score)
                scores.append(score)

            # VALIDATE FOCUS PEAK QUALITY
            scores_array = np.array(scores)
            from microscope_control.autofocus.core import AutofocusUtils
            validation = AutofocusUtils.validate_focus_peak(z_steps, scores_array)

            # Interpolate to find best focus (do this before validation check)
            interp_x = np.linspace(z_steps[0], z_steps[-1], n_steps * interp_strength)
            interp_y = scipy.interpolate.interp1d(z_steps, scores, kind=interp_kind)(interp_x)
            new_z = interp_x[np.argmax(interp_y)]

            # Save diagnostic CSV BEFORE validation check (so it saves even on failure)
            if diagnostic_output_path is not None:
                self._save_autofocus_diagnostic_csv(
                    z_steps, scores_array, validation, new_z,
                    diagnostic_output_path, position_index, current_pos
                )

            if not validation['is_valid']:
                logger.warning("*** AUTOFOCUS PEAK QUALITY WARNING ***")
                logger.warning(f"  {validation['message']}")
                for warning in validation['warnings']:
                    logger.warning(f"    - {warning}")
                logger.warning(f"  Quality metrics: prominence={validation['peak_prominence']:.2f}, "
                             f"quality={validation['quality_score']:.2f}")

                if raise_on_invalid_peak:
                    # Return failure dict for manual focus fallback loop
                    # Move stage to computed best position before returning, so user
                    # can manually adjust from a reasonable starting point
                    logger.warning("  Autofocus failed - moving to computed best Z and returning failure dict")
                    best_pos = Position(current_pos.x, current_pos.y, new_z)
                    self.move_to_position(best_pos)
                    return {
                        'success': False,
                        'message': validation['message'],
                        'quality_score': validation['quality_score'],
                        'peak_prominence': validation['peak_prominence'],
                        'attempted_z': new_z,
                        'original_z': current_pos.z,
                        'validation': validation
                    }
                else:
                    # Just log warning, continue for diagnostic purposes (test mode)
                    logger.warning("  Proceeding with autofocus result for diagnostic analysis")
            else:
                logger.info(f"Autofocus peak validation: {validation['message']}")
                logger.debug(f"  Quality score: {validation['quality_score']:.2f}, "
                           f"prominence: {validation['peak_prominence']:.2f}")

            if pop_a_plot:
                plt.figure()
                plt.bar(z_steps, scores)
                plt.plot(interp_x, interp_y, "k")
                plt.plot(interp_x[np.argmax(interp_y)], interp_y.max(), "or")
                plt.xlabel("Z-axis (um)")
                plt.title(f"Autofocus at X={current_pos.x:.1f}, Y={current_pos.y:.1f}")
                plt.show()

            if move_stage_to_estimate:
                new_pos = Position(current_pos.x, current_pos.y, new_z)
                self.move_to_position(new_pos)

            return new_z

        except Exception as e:
            logger.error(f"Autofocus failed: {e}")
            self.move_to_position(current_pos)
            raise e

    def _save_autofocus_diagnostic_csv(self, z_positions, scores, validation, result_z,
                                       output_path, position_index, current_pos):
        """
        Save autofocus diagnostic data to CSV file in the acquisition folder.

        Args:
            z_positions: Array of Z positions sampled
            scores: Array of focus scores
            validation: Validation results dict
            result_z: Resulting Z position from autofocus
            output_path: Path to save the CSV file
            position_index: Position index for filename
            current_pos: Starting position for autofocus
        """
        try:
            import csv
            from datetime import datetime
            from pathlib import Path

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pos_str = f"_pos{position_index}" if position_index is not None else ""
            csv_filename = f"autofocus_diagnostic{pos_str}_{timestamp}.csv"
            csv_path = output_path / csv_filename

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header with metadata
                writer.writerow(['# Autofocus Diagnostic Data (Standard Autofocus)'])
                writer.writerow(['# Timestamp', timestamp])
                if position_index is not None:
                    writer.writerow(['# Position Index', position_index])
                writer.writerow(['# Starting Position', f'X={current_pos.x:.2f}, Y={current_pos.y:.2f}, Z={current_pos.z:.2f}'])
                writer.writerow(['# Autofocus Result Z', f'{result_z:.2f}'])
                writer.writerow(['# Z Shift', f'{result_z - current_pos.z:.2f}'])
                writer.writerow(['#'])
                writer.writerow(['# VALIDATION RESULTS'])
                writer.writerow(['# Status', 'VALID' if validation['is_valid'] else 'INVALID'])
                writer.writerow(['# Quality Score', f"{validation['quality_score']:.3f}"])
                writer.writerow(['# Peak Prominence', f"{validation['peak_prominence']:.3f}"])
                writer.writerow(['# Has Ascending', validation['has_ascending']])
                writer.writerow(['# Has Descending', validation['has_descending']])
                writer.writerow(['# Symmetry Score', f"{validation['symmetry_score']:.3f}"])
                writer.writerow(['# Message', validation['message']])
                if validation['warnings']:
                    for warning in validation['warnings']:
                        writer.writerow(['# Warning', warning])
                writer.writerow(['#'])

                # Write data header
                writer.writerow(['Z_Position_um', 'Focus_Score'])

                # Write data
                for z, score in zip(z_positions, scores):
                    writer.writerow([f'{z:.2f}', f'{score:.4f}'])

            logger.info(f"Autofocus diagnostic CSV saved: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save autofocus diagnostic CSV: {e}")

    def autofocus_adaptive_search(
        self,
        initial_step_size=10,
        min_step_size=2,
        focus_threshold=0.95,
        max_total_steps=25,
        score_metric=None,
        pop_a_plot=False,
        move_stage_to_estimate=True,
    ) -> float:
        """
        Drift-check autofocus: samples symmetrically around current position
        to detect focus drift and correct it with minimal acquisitions.

        This simplified approach:
        - Samples 5 positions around current Z (e.g., -4, -2, 0, +2, +4 um)
        - Finds peak using symmetric interpolation (NO directional bias)
        - Only moves if improvement > 1um (avoids chasing noise)

        Parameters are kept for compatibility but reinterpreted:
        - initial_step_size: Used to set sampling range (range = 2*step_size)
        - min_step_size: Minimum movement threshold (won't move if peak < this)
        - Other parameters: Ignored in this simplified version
        """
        if score_metric is None:
            score_metric = AutofocusUtils.autofocus_profile_laplacian_variance

        current_pos = self.get_current_position()
        initial_z = current_pos.z

        # Get Z limits from settings
        stage_limits = self.settings.get("stage", {}).get("limits", {})
        z_limits = stage_limits.get("z_um", {})
        z_min = z_limits.get("low", -1000)
        z_max = z_limits.get("high", 1000)

        # Sampling parameters
        sample_range = initial_step_size * 0.8  # e.g., 10um -> 8um range (+/-4um)
        n_samples = 5
        move_threshold = min_step_size / 2.0  # e.g., 2um -> 1um threshold

        # Create symmetric sample positions around current Z
        half_range = sample_range / 2
        z_positions = np.linspace(initial_z - half_range,
                                  initial_z + half_range,
                                  n_samples)

        # Clamp to stage limits
        z_positions = np.clip(z_positions, z_min + 5, z_max - 5)

        # Helper function to acquire and score at a position
        def measure_at_z(z):
            self.move_to_position(Position(current_pos.x, current_pos.y, z))
            img, tags = self.snap_image()

            if img is None:
                logger.error(f"Failed to acquire image at Z={z}")
                return -np.inf

            # Process image
            if len(img.shape) == 2:  # Bayer pattern
                green1 = img[0::2, 0::2]
                green2 = img[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            elif len(img.shape) == 3:  # RGB image
                img_gray = skimage.color.rgb2gray(img)
            else:
                img_gray = img.astype(np.float32)

            score = score_metric(img_gray)
            if hasattr(score, "ndim") and score.ndim == 2:
                score = np.mean(score)

            return float(score)

        # Score all positions
        scores = []
        for z in z_positions:
            score = measure_at_z(z)
            if score == -np.inf:
                logger.error(f"Failed to measure at Z={z:.2f}")
                scores.append(0)
            else:
                scores.append(score)
            logger.debug(f"  Z={z:.2f}um: score={score:.1f}")

        scores = np.array(scores)

        # Find discrete best position
        best_idx = np.argmax(scores)
        best_z_discrete = z_positions[best_idx]

        # Refine with symmetric interpolation
        # CRITICAL FIX: Symmetric window (+/-2 positions, not +3)
        start_idx = max(0, best_idx - 2)
        end_idx = min(len(z_positions), best_idx + 2)  # SYMMETRIC: +2 not +3

        # Ensure minimum 3 points for quadratic fit
        if end_idx - start_idx < 3:
            if start_idx == 0:
                end_idx = min(3, len(z_positions))
            else:
                start_idx = max(0, len(z_positions) - 3)

        # Quadratic interpolation to find refined peak
        z_subset = z_positions[start_idx:end_idx]
        scores_subset = scores[start_idx:end_idx]

        try:
            # Fit parabola and find peak
            coeffs = np.polyfit(z_subset, scores_subset, 2)
            if coeffs[0] < 0:  # Parabola opens downward (valid peak)
                refined_z = -coeffs[1] / (2 * coeffs[0])
                # Clamp to sampled range
                refined_z = np.clip(refined_z, z_positions[0], z_positions[-1])
            else:
                # Parabola opens upward (shouldn't happen), use discrete best
                refined_z = best_z_discrete
                logger.warning("Invalid parabola fit, using discrete best")
        except:
            refined_z = best_z_discrete
            logger.warning("Interpolation failed, using discrete best")

        # Decide whether to move
        delta = refined_z - initial_z

        if abs(delta) < move_threshold:
            # Peak is at current position (within noise tolerance)
            logger.info(f"Drift check: focus peak at current position "
                       f"(delta={delta:+.2f}um < {move_threshold:.1f}um threshold)")
            best_z = initial_z
        else:
            # Significant drift detected
            logger.info(f"Drift check: focus drift {delta:+.2f}um detected, "
                       f"moving from {initial_z:.2f} to {refined_z:.2f}um")
            best_z = refined_z

        # Optional plot
        if pop_a_plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(z_positions, scores, s=100, c='blue', label='Measured', zorder=3)
            plt.axvline(initial_z, color='gray', linestyle='--', label='Initial Z')
            plt.axvline(best_z, color='red', linestyle='-', linewidth=2, label='Final Z')
            plt.xlabel("Z position (um)")
            plt.ylabel("Focus score")
            plt.title(f"Drift-check autofocus: {n_samples} samples")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        logger.info(f"Drift-check complete: Z={best_z:.2f}um "
                   f"(sampled {n_samples} positions)")

        if move_stage_to_estimate:
            self.move_to_position(Position(current_pos.x, current_pos.y, best_z))

        return best_z

    def white_balance(self, img=None, background_image=None, gain=1.0, white_balance_profile=None):
        """Apply white balance correction to image."""
        if white_balance_profile is None:
            # Try to get default from settings
            wb_settings = self.settings.get("white_balance", {})
            default_wb = wb_settings.get("default", {}).get("default", [1.0, 1.0, 1.0])
            white_balance_profile = default_wb

        if img is None:
            raise ValueError("Input image 'img' must not be None for white balancing.")

        if background_image is not None:
            r, g, b = background_image.mean((0, 1))
            r1, g1, b1 = (r, g, b) / max(r, g, b)
        else:
            r1, g1, b1 = white_balance_profile

        img_wb = img.astype(np.float64) * gain / [r1, g1, b1]
        return np.clip(img_wb, 0, 255).astype(np.uint8)

    def get_device_properties(self, scope: str = "used") -> Dict[str, Dict[str, Any]]:
        """
        Get device properties from MM device manager.

        Args:
            scope: 'used' for current values, 'allowed' for possible values

        Returns:
            Dictionary of device properties
        """
        device_dict = {}
        for device_name in obj_2_list(self.core.get_loaded_devices()):
            device_property_names = self.core.get_device_property_names(device_name)
            property_names = obj_2_list(device_property_names)
            prop_dict = {}

            for prop in property_names:
                if scope == "allowed":
                    values = self.core.get_allowed_property_values(device_name, prop)
                    prop_dict[prop] = obj_2_list(values)
                elif scope == "used":
                    values = self.core.get_property(device_name, prop)
                    prop_dict[prop] = values
                else:
                    warnings.warn(f"Unknown metadata scope {scope}")

            device_dict[device_name] = prop_dict

        return device_dict

    def _ppm_set_psgticks(self, theta: float) -> None:
        """Set the PPM rotation stage to a specific angle."""
        # Try to get rotation stage device from settings
        rotation_device = self.rotation_device

        if rotation_device == "PIZStage":
            theta_pistage = self.ppm_rlpangle_to_PIStage(theta)
            self.core.set_position(rotation_device, theta_pistage)
            self.core.wait_for_device(rotation_device)
            logger.debug(f"Set rotation angle to {theta} deg (Thor position: {theta_pistage})")
        elif rotation_device == "KBD101_Thor_Rotation":
            theta_thor = ppm_psgticks_to_thor(theta)
            self.core.set_position(rotation_device, theta_thor)
            self.core.wait_for_device(rotation_device)
            logger.debug(f"Set rotation angle to {theta} deg (Thor position: {theta_thor})")
        else:
            logger.error(f"Unknown rotation device: {rotation_device} in config")
            raise ValueError(
                f"Unknown rotation device: {rotation_device} \
                             supports only [KBD101_Thor_Rotation, PIZStage]"
            )

    def _ppm_get_psgticks(self) -> float:
        """Get the current PPM rotation angle."""
        rotation_device = self.rotation_device

        if rotation_device == "PIZStage":
            pistage_pos = self.core.get_position(rotation_device)
            return self.ppm_PIStage_to_rlpangle(pistage_pos)
        elif rotation_device == "KBD101_Thor_Rotation":
            thor_pos = self.core.get_position(rotation_device)
            return ppm_thor_to_psgticks(thor_pos)
        else:
            raise ValueError(
                f"Unknown rotation device: {rotation_device} \
                             supports only [KBD101_Thor_Rotation, PIZStage]"
            )

    def _ppm_home(self) -> None:
        """Set the PPM rotation stage to a specific angle."""
        # Try to get rotation stage device from settings
        rotation_device = self.rotation_device
        self.core.home(rotation_device)
        self.core.wait_for_device(rotation_device)
        logger.debug("Homed rotation stage")

    ## since offset is going to be optimized, it need to stay with the config
    def ppm_rlpangle_to_PIStage(self, theta: float) -> float:
        """Convert PPM angle (in degrees) to PI stage position."""
        offset = self.settings.get("ppm_pizstage_offset", 50280.0)
        return (theta * 1000) + offset

    def ppm_PIStage_to_rlpangle(self, pi_pos: float) -> float:
        """Convert PI stage position to PPM angle (in degrees)."""
        offset = self.settings.get("ppm_pizstage_offset", 50280.0)
        return (pi_pos - offset) / 1000.0

    def _camm_swap_objective_lens(self, desired_imaging_mode: Dict[str, Any]):
        """
        Swap objective lens for CAMM microscope.

        Args:
            desired_imaging_mode: Dictionary containing imaging mode configuration
        """
        # Get objective slider device from settings
        obj_slider = self.settings.get("obj_slider")
        if not obj_slider:
            raise ValueError("No objective slider configuration found")

        current_slider_position = self.core.get_property(*obj_slider)
        desired_position = desired_imaging_mode.get("objective_position_label")

        if not desired_position:
            raise ValueError("No objective position label in imaging mode")

        if desired_position != current_slider_position:
            mode_name = desired_imaging_mode.get("name", "")
            stage_config = self.settings.get("stage", {})
            z_stage = stage_config.get("z_stage")
            f_stage = stage_config.get("f_stage")

            if not z_stage or not f_stage:
                raise ValueError("Stage devices not properly configured")

            # Handle different objectives differently
            if mode_name.startswith("4X"):
                self.core.set_focus_device(z_stage)
                self.core.set_position(desired_imaging_mode.get("z", 0))
                self.core.wait_for_device(z_stage)
                self.core.set_property(*obj_slider, desired_position)
                self.core.set_focus_device(f_stage)
                self.core.wait_for_system()

            elif mode_name.startswith("20X"):
                self.core.set_property(*obj_slider, desired_position)
                self.core.wait_for_device(obj_slider[0])
                self.core.set_focus_device(z_stage)
                self.core.set_position(desired_imaging_mode.get("z", 0))
                self.core.set_focus_device(f_stage)
                self.core.set_position(desired_imaging_mode.get("f", 0))
                self.core.wait_for_system()

            self.core.set_focus_device(z_stage)

            # Update current imaging mode in settings
            self.settings["imaging_mode"] = desired_imaging_mode
            logger.info(f"Swapped to objective: {desired_position}")
