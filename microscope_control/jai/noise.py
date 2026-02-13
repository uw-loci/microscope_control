"""
JAI Camera Noise Measurement Utilities.

This module provides tools for measuring per-channel noise characteristics
of the JAI AP-3200T-USB 3-CCD prism camera. Noise measurements are used
to evaluate white balance calibration quality and to characterize the
tradeoffs between gain, exposure, and image noise.

Two measurement modes are supported:

1. Multi-frame temporal noise: Captures multiple frames and computes the
   temporal standard deviation per pixel, then averages across the image.
   This is the most accurate noise measurement but requires multiple captures.

2. Single-frame spatial noise: Estimates noise from the spatial standard
   deviation within a single frame. Less accurate but suitable for real-time
   display during live viewing.

Usage:
    from microscope_control.jai import JAINoiseMeasurement, NoiseStats

    noise_meter = JAINoiseMeasurement(hardware)
    stats = noise_meter.measure_noise(num_frames=10, settle_frames=2)
    print(f"Red SNR: {stats.channel_snr['red']:.1f}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseStats:
    """Results from noise measurement.

    Contains per-channel mean, standard deviation, and signal-to-noise ratio
    computed either from multi-frame temporal analysis or single-frame spatial
    analysis.
    """

    # Per-channel mean intensities
    channel_means: Dict[str, float]

    # Per-channel standard deviations (noise)
    channel_stddevs: Dict[str, float]

    # Per-channel signal-to-noise ratios (mean / stddev)
    channel_snr: Dict[str, float]

    # Number of frames used for measurement (1 for single-frame)
    num_frames: int

    # Current per-channel exposure times in ms
    exposure_ms: Dict[str, float]

    # Current unified gain value
    unified_gain: float

    # Current R/B analog gain values
    analog_gains: Dict[str, float]

    # Optional noise verification result (populated when verify_noise=True)
    # Contains: passes, channel_stddevs, channel_snr, thresholds, per_channel
    verification_result: Optional[Dict] = None


class JAINoiseMeasurement:
    """Noise measurement tool for JAI prism cameras.

    Provides multi-frame temporal noise analysis and single-frame spatial
    noise estimation for the JAI AP-3200T-USB 3-CCD camera.
    """

    def __init__(
        self,
        hardware: Any,
        jai_props: Optional[Any] = None,
    ):
        """
        Initialize the noise measurement tool.

        Args:
            hardware: PycromanagerHardware instance with JAI camera configured
            jai_props: Optional JAICameraProperties instance (created if not provided)
        """
        self.hardware = hardware
        self.jai_props = jai_props

        # Lazily create jai_props if not provided
        if self.jai_props is None:
            try:
                from microscope_control.jai.properties import JAICameraProperties
                self.jai_props = JAICameraProperties(hardware.core)
            except Exception as e:
                logger.warning(f"Could not create JAICameraProperties: {e}")

    def _get_current_settings(self) -> tuple:
        """Read current camera settings for inclusion in NoiseStats.

        Returns:
            Tuple of (exposure_ms dict, unified_gain float, analog_gains dict)
        """
        exposure_ms = {"red": 0.0, "green": 0.0, "blue": 0.0}
        unified_gain = 1.0
        analog_gains = {"red": 1.0, "blue": 1.0}

        if self.jai_props is not None:
            try:
                exposure_ms = self.jai_props.get_channel_exposures()
            except Exception:
                try:
                    exp = float(self.hardware.core.get_exposure())
                    exposure_ms = {"red": exp, "green": exp, "blue": exp}
                except Exception:
                    pass

            try:
                unified_gain = self.jai_props.get_unified_gain()
            except Exception:
                pass

            try:
                all_gains = self.jai_props.get_analog_gains()
                analog_gains = {
                    "red": all_gains.get("red", 1.0),
                    "blue": all_gains.get("blue", 1.0),
                }
            except Exception:
                pass

        return exposure_ms, unified_gain, analog_gains

    def measure_noise(
        self,
        num_frames: int = 10,
        settle_frames: int = 2,
    ) -> NoiseStats:
        """
        Measure per-channel noise using multi-frame temporal analysis.

        Captures num_frames images, discards the first settle_frames to allow
        the camera to stabilize, then computes the temporal standard deviation
        at each pixel position across the remaining frames. The per-pixel
        temporal stddev is averaged across the image to produce a single noise
        value per channel.

        Args:
            num_frames: Total number of frames to capture (including settle frames)
            settle_frames: Number of initial frames to discard for camera settling

        Returns:
            NoiseStats with per-channel noise measurements

        Raises:
            RuntimeError: If unable to capture sufficient frames
        """
        total_frames = num_frames + settle_frames
        logger.info(
            f"Measuring noise: capturing {total_frames} frames "
            f"({settle_frames} settle + {num_frames} analysis)"
        )

        frames = []
        for i in range(total_frames):
            img, _ = self.hardware.snap_image()
            if img is None:
                logger.warning(f"Frame {i} capture failed, skipping")
                continue
            frames.append(img)
            time.sleep(0.05)  # Small delay between captures

        if len(frames) <= settle_frames:
            raise RuntimeError(
                f"Only captured {len(frames)} frames, need at least "
                f"{settle_frames + 1} (settle_frames + 1)"
            )

        # Discard settle frames
        analysis_frames = frames[settle_frames:]
        actual_count = len(analysis_frames)
        logger.debug(f"Analyzing {actual_count} frames after settling")

        # Stack frames: shape (N, H, W, C)
        stack = np.stack(analysis_frames, axis=0).astype(np.float32)

        if stack.ndim != 4 or stack.shape[3] < 3:
            raise RuntimeError(
                f"Expected RGB image stack, got shape: {stack.shape}"
            )

        # Compute temporal mean and stddev per pixel, then average spatially
        temporal_mean = stack.mean(axis=0)  # (H, W, C)
        temporal_std = stack.std(axis=0)    # (H, W, C)

        channel_means = {
            "red": float(temporal_mean[:, :, 0].mean()),
            "green": float(temporal_mean[:, :, 1].mean()),
            "blue": float(temporal_mean[:, :, 2].mean()),
        }

        channel_stddevs = {
            "red": float(temporal_std[:, :, 0].mean()),
            "green": float(temporal_std[:, :, 1].mean()),
            "blue": float(temporal_std[:, :, 2].mean()),
        }

        channel_snr = {}
        for ch in ["red", "green", "blue"]:
            if channel_stddevs[ch] > 0:
                channel_snr[ch] = channel_means[ch] / channel_stddevs[ch]
            else:
                channel_snr[ch] = float('inf')

        exposure_ms, unified_gain, analog_gains = self._get_current_settings()

        stats = NoiseStats(
            channel_means=channel_means,
            channel_stddevs=channel_stddevs,
            channel_snr=channel_snr,
            num_frames=actual_count,
            exposure_ms=exposure_ms,
            unified_gain=unified_gain,
            analog_gains=analog_gains,
        )

        logger.info(
            f"Noise measurement complete ({actual_count} frames): "
            f"R: mean={channel_means['red']:.1f}, std={channel_stddevs['red']:.2f}, SNR={channel_snr['red']:.1f} | "
            f"G: mean={channel_means['green']:.1f}, std={channel_stddevs['green']:.2f}, SNR={channel_snr['green']:.1f} | "
            f"B: mean={channel_means['blue']:.1f}, std={channel_stddevs['blue']:.2f}, SNR={channel_snr['blue']:.1f}"
        )

        return stats

    def measure_single_frame_noise(self, image: np.ndarray) -> Dict[str, float]:
        """
        Estimate per-channel noise from a single frame using spatial statistics.

        Computes the standard deviation of pixel values within each channel.
        This is a rough noise estimate suitable for real-time display but less
        accurate than multi-frame temporal analysis (it conflates signal
        variation with noise).

        Args:
            image: RGB image array with shape (H, W, 3) or (H, W, C) where C >= 3

        Returns:
            Dictionary with per-channel spatial standard deviations:
            {'red': float, 'green': float, 'blue': float}
        """
        if image is None:
            return {"red": 0.0, "green": 0.0, "blue": 0.0}

        if image.ndim != 3 or image.shape[2] < 3:
            logger.warning(
                f"Expected RGB image, got shape: {image.shape}. "
                f"Returning zero noise."
            )
            return {"red": 0.0, "green": 0.0, "blue": 0.0}

        return {
            "red": float(np.std(image[:, :, 0])),
            "green": float(np.std(image[:, :, 1])),
            "blue": float(np.std(image[:, :, 2])),
        }
