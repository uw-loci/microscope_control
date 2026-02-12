"""
JAI Camera Calibration Utilities.

This module provides calibration tools specific to JAI prism cameras,
including white balance calibration via per-channel exposure adjustment.

The JAI camera uses a 3-sensor prism design (no Bayer filter), which means
white balance must be achieved through per-channel exposure/gain adjustment
rather than software demosaicing corrections.

Calibration Algorithm Overview
------------------------------
1. Optionally calibrate black level using dark frames
2. Enable individual exposure mode
3. Capture image with current settings
4. Analyze per-channel histogram (R, G, B)
5. Iteratively adjust per-channel exposure to balance channels
6. If exposure ratio exceeds threshold, compensate with per-channel gain
7. Save calibration results for use during acquisition

Usage
-----
    from microscope_control.jai import JAIWhiteBalanceCalibrator

    calibrator = JAIWhiteBalanceCalibrator(hardware)
    results = calibrator.calibrate(target_value=180, tolerance=5)

    # Results contain per-channel exposure and gain settings
    print(results.exposures_ms)  # {'red': 10.5, 'green': 8.2, 'blue': 12.1}
    print(results.gains)         # {'red': 1.0, 'green': 1.0, 'blue': 1.0}

Note
----
This module is JAI camera-specific and requires the JAI camera to be
configured in Micro-Manager with individual channel control support (PR #781).
"""

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from microscope_control.jai.properties import JAICameraProperties

logger = logging.getLogger(__name__)


# Quality presets for noise-aware calibration
# Each preset defines tradeoffs between speed, noise, and signal quality
QUALITY_PRESETS = {
    'fast': {
        'max_exposure_ms': 50.0,
        'max_noise_stddev': 8.0,
        'min_snr': 15.0,
        'unified_gain_mode': 'fixed',
        'base_gain': 5.0,
    },
    'balanced': {
        'max_exposure_ms': 200.0,
        'max_noise_stddev': 5.0,
        'min_snr': 25.0,
        'unified_gain_mode': 'auto',
        'base_gain': 3.0,
    },
    'quality': {
        'max_exposure_ms': 500.0,
        'max_noise_stddev': 3.0,
        'min_snr': 40.0,
        'unified_gain_mode': 'auto',
        'base_gain': 1.0,
    },
}


def db_to_linear(db: float) -> float:
    """
    Convert decibels to linear gain multiplier.

    Args:
        db: Gain in decibels

    Returns:
        Linear gain multiplier (e.g., 3 dB -> 1.41, 6 dB -> 2.0)
    """
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """
    Convert linear gain multiplier to decibels.

    Args:
        linear: Linear gain multiplier

    Returns:
        Gain in decibels (e.g., 1.41 -> 3 dB, 2.0 -> 6 dB)
    """
    if linear <= 0:
        return float('-inf')
    return 20.0 * np.log10(linear)


@dataclass
class WhiteBalanceResult:
    """Results from white balance calibration.

    The new gain model uses:
    - unified_gain: Applied to all channels equally (1.0-8.0)
    - analog_red/analog_blue: R/B fine-tuning via analog gains in unified mode (0.47-4.0)
    - Green is the reference channel (no per-channel gain adjustment)
    """

    # Per-channel exposure times in milliseconds
    exposures_ms: Dict[str, float]

    # Black level offsets per channel (for dark frame subtraction)
    black_levels: Dict[str, float]

    # Final channel means after calibration
    final_means: Dict[str, float]

    # Target value that was used
    target_value: float

    # Unified gain value (1.0-8.0, applied to all channels)
    unified_gain: float = 1.0

    # R/B analog gain correction (0.47-4.0, applied in unified gain mode)
    analog_red: float = 1.0
    analog_blue: float = 1.0

    # Noise statistics at final settings (if measured)
    noise_stats: Optional[Any] = None

    # White balance method that produced these settings.
    # Values: "manual_simple", "manual_ppm", "continuous", "once", "unknown"
    wb_method: str = "unknown"

    # Whether calibration converged successfully
    converged: bool = False

    # Number of iterations to converge
    iterations: int = 0


@dataclass
class CalibrationConfig:
    """Configuration for white balance calibration.

    The calibration uses a 2-phase approach:
    - Phase 1 (Coarse): Adjust per-channel exposures to reach target intensity
    - Phase 2 (Fine): Lock exposures, adjust R/B analog gains for color balance
    """

    # Target mean value for all channels (0-255 for 8-bit, scaled for 16-bit)
    target_value: float = 180.0

    # Acceptable deviation from target (channels within tolerance are considered balanced)
    # Default of 2.0 achieves within 2 intensity levels precision
    tolerance: float = 2.0

    # Phase 1 coarse tolerance - Phase 1 converges at this wider threshold,
    # then Phase 2 fine-tunes with R/B analog gains to reach final tolerance.
    coarse_tolerance: float = 10.0

    # Maximum iterations before giving up
    max_iterations: int = 30

    # Minimum exposure time in milliseconds
    min_exposure_ms: float = 0.1

    # Maximum exposure time in milliseconds (soft limit, can be auto-extended)
    # Per-channel exposure limit depends on frame rate (lower frame rate = longer exposure)
    # At min frame rate (0.125 Hz), theoretical max is ~7900ms
    # This limit may be automatically extended up to hardware_max_exposure_ms when
    # channels are stuck at ceiling.
    max_exposure_ms: float = 200.0

    # Absolute hardware ceiling for exposure time in milliseconds
    # This is the maximum that max_exposure_ms can be extended to when the algorithm
    # detects channels stuck at ceiling. Based on min frame rate 0.125 Hz with margin.
    hardware_max_exposure_ms: float = 7900.0

    # Exposure ratio threshold before applying gain compensation
    # If brightest_channel_exposure / darkest_channel_exposure > this, use gain
    gain_threshold_ratio: float = 2.0

    # Base unified gain to start calibration with (reduces initial exposure requirements)
    # Applied as unified gain (1.0-8.0 range). Default 5.0 provides strong initial boost.
    base_gain: float = 5.0

    # Whether to avoid digital gain (digital gain adds more noise than analog)
    # Digital gain range is very narrow anyway: 0.9-1.1 linear (-0.915 to 0.828 dB)
    avoid_digital_gain: bool = True

    # Damping factor for exposure adjustments (prevents oscillation)
    # Use lower damping for fine-tuning phase when close to target
    damping_factor: float = 0.7

    # Fine-tuning damping factor (used when within 2x tolerance of target)
    fine_damping_factor: float = 0.5

    # Whether to perform black level calibration
    calibrate_black_level: bool = True

    # Number of dark frames to average for black level
    dark_frame_count: int = 10

    # Number of frames for noise measurement at end of calibration
    noise_frames: int = 5

    # Defocus offset in micrometers (for PPM calibration on blank slide area)
    defocus_offset_um: Optional[float] = None

    # Bit depth of camera (for target value scaling)
    bit_depth: int = 8

    # Noise constraint parameters (for noise-aware calibration)
    # Maximum acceptable per-channel noise standard deviation
    max_noise_stddev: float = 5.0

    # Minimum acceptable signal-to-noise ratio per channel
    min_snr: float = 20.0

    # Unified gain selection mode:
    # - 'auto': Automatically select lowest gain that meets exposure constraints
    # - 'fixed': Use base_gain value directly
    # - 'minimize_noise': Prioritize lowest gain even if exposure is longer
    # - 'minimize_exposure': Prioritize higher gain to reduce exposure time
    unified_gain_mode: str = 'auto'

    # Quality mode preset: 'fast', 'balanced', or 'quality'
    # When set to a valid preset name, overrides individual noise parameters
    # Set to None or empty string to use individual parameter values
    quality_mode: str = ''

    # Whether to run noise verification at end of calibration
    verify_noise: bool = True


@dataclass
class ConvergenceLog:
    """Log of calibration iterations for diagnostics."""

    iterations: List[Dict[str, Any]] = field(default_factory=list)

    def add_iteration(
        self,
        iteration: int,
        means: Dict[str, float],
        exposures: Dict[str, float],
        gains: Dict[str, float],
        converged: Dict[str, bool],
        notes: str = "",
        unified_gain: float = 1.0,
    ) -> None:
        """Add an iteration to the log."""
        self.iterations.append(
            {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "red_intensity": means.get("red", 0),
                "green_intensity": means.get("green", 0),
                "blue_intensity": means.get("blue", 0),
                "red_exposure_ms": exposures.get("red", 0),
                "green_exposure_ms": exposures.get("green", 0),
                "blue_exposure_ms": exposures.get("blue", 0),
                "unified_gain": unified_gain,
                "red_gain": gains.get("red", 1.0),
                "green_gain": gains.get("green", 1.0),
                "blue_gain": gains.get("blue", 1.0),
                "red_converged": converged.get("red", False),
                "green_converged": converged.get("green", False),
                "blue_converged": converged.get("blue", False),
                "notes": notes,
            }
        )

    def save_csv(self, output_path: Path) -> None:
        """Save convergence log to CSV file."""
        if not self.iterations:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(self.iterations[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.iterations)

        logger.info(f"Saved convergence log to {output_path}")


class JAIWhiteBalanceCalibrator:
    """
    White balance calibrator for JAI prism cameras.

    This calibrator adjusts per-channel exposure times (and optionally gains)
    to achieve balanced color response from the JAI 3-sensor prism camera.

    The JAI camera exposes each color channel independently, so white balance
    is achieved by finding the exposure time for each channel that produces
    equal mean values when imaging a neutral gray or white target.
    """

    def __init__(
        self,
        hardware: Any,
        jai_props: Optional[JAICameraProperties] = None,
    ):
        """
        Initialize the calibrator.

        Args:
            hardware: PycromanagerHardware instance with JAI camera configured
            jai_props: Optional JAICameraProperties instance (created if not provided)
        """
        self.hardware = hardware
        self.jai_props = jai_props or JAICameraProperties(hardware.core)
        self._convergence_log = ConvergenceLog()
        self._black_levels: Dict[str, float] = {"red": 0, "green": 0, "blue": 0}

    def _validate_camera(self) -> None:
        """Verify that JAI camera is available."""
        if not self.jai_props.validate_camera():
            raise RuntimeError(
                "JAI camera not available or not active. "
                "Ensure JAI camera is connected and set as active camera in Micro-Manager."
            )

    def calibrate(
        self,
        config: Optional[CalibrationConfig] = None,
        output_path: Optional[Path] = None,
        ppm_rotation_callback: Optional[Callable[[float], None]] = None,
        defocus_callback: Optional[Callable[[float], Tuple[float, Callable]]] = None,
    ) -> WhiteBalanceResult:
        """
        Run 2-phase white balance calibration.

        Phase 1 (Coarse - exposure balancing):
            Set unified gain, then iteratively adjust per-channel exposures
            until all channels are within coarse_tolerance of target.

        Phase 2 (Fine - R/B analog gain tuning):
            Lock exposures from Phase 1. Compute residual R/B imbalance
            relative to green (reference channel). Apply R/B analog gains
            to fine-tune color balance. These work in unified gain mode
            (GainIsIndividual=Off).

        Args:
            config: Calibration configuration. Uses defaults if None.
            output_path: Optional path to save diagnostic output (histograms, log)
            ppm_rotation_callback: Optional callback to set PPM rotation angle.
                                  Called with angle=90 for max intensity during calibration.
            defocus_callback: Optional callback to defocus the stage.
                             Takes offset_um, returns (original_z, restore_callback).

        Returns:
            WhiteBalanceResult with calibrated settings
        """
        if config is None:
            config = CalibrationConfig()

        # Apply quality preset if specified (modifies config in place)
        config = self._apply_quality_preset(config)

        self._validate_camera()
        self._convergence_log = ConvergenceLog()

        logger.info("Starting JAI white balance calibration (2-phase)")
        logger.info(f"Target value: {config.target_value}, tolerance: {config.tolerance}")
        if config.quality_mode:
            logger.info(f"Quality mode: {config.quality_mode}")
        logger.info(f"Gain mode: {config.unified_gain_mode}, base_gain: {config.base_gain}")

        # Scale target value for bit depth
        if config.bit_depth == 16:
            target = config.target_value * (65535 / 255)
        else:
            target = config.target_value

        restore_z_callback = None
        original_ppm_angle = None

        try:
            # Step 1: Black level calibration (if enabled)
            if config.calibrate_black_level:
                logger.info("Black level calibration would be performed here")
                logger.info("(Requires user to cover lens - skipping in automated mode)")

            # Step 2: Setup for calibration
            # Set PPM to 90 degrees (max intensity) if callback provided
            if ppm_rotation_callback is not None:
                try:
                    original_ppm_angle = self.hardware.get_psg_ticks()
                    ppm_rotation_callback(90.0)
                    logger.info("Set PPM rotation to 90 degrees for calibration")
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Failed to set PPM rotation: {e}")

            # Apply defocus if callback provided and offset configured
            if defocus_callback is not None and config.defocus_offset_um is not None:
                try:
                    original_z, restore_z_callback = defocus_callback(config.defocus_offset_um)
                    logger.info(f"Defocused by {config.defocus_offset_um}um for calibration")
                except Exception as e:
                    logger.warning(f"Failed to apply defocus: {e}")

            # Step 3: Set unified gain and R/B analog gains
            # Use noise-aware gain selection when mode is 'auto' or other dynamic modes
            analog_red = 1.0
            analog_blue = 1.0

            if config.unified_gain_mode in ('auto', 'minimize_noise', 'minimize_exposure'):
                # Noise-aware gain selection - test light levels first
                unified_gain = self._select_optimal_unified_gain(config, target)
            else:
                # Fixed mode - use base_gain directly
                unified_gain = max(
                    JAICameraProperties.GAIN_UNIFIED_RANGE[0],
                    min(config.base_gain, JAICameraProperties.GAIN_UNIFIED_RANGE[1])
                )

            logger.info(
                f"Setting unified gain to {unified_gain:.1f}x "
                f"({linear_to_db(unified_gain):.1f} dB)"
            )
            try:
                self.jai_props.set_unified_gain(unified_gain)
            except Exception as e:
                logger.warning(f"Failed to set unified gain: {e}. Starting with gain=1.0")
                unified_gain = 1.0

            # Reset R/B analog gains to 1.0
            try:
                self.jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
            except Exception as e:
                logger.warning(f"Failed to reset R/B analog gains: {e}")

            # Step 4: Enable individual exposure mode
            self.jai_props.enable_individual_exposure()

            # Initial capture
            means = self._capture_and_analyze()
            logger.info(
                f"Initial channel means: R={means['red']:.1f}, "
                f"G={means['green']:.1f}, B={means['blue']:.1f}"
            )

            # Get current exposures or use defaults
            try:
                exposures = self.jai_props.get_channel_exposures()
            except Exception:
                exposures = {"red": 50.0, "green": 50.0, "blue": 50.0}
                self.jai_props.set_channel_exposures(**exposures)

            # Initial exposure estimation
            for channel in ["red", "green", "blue"]:
                if means[channel] > 0:
                    estimated = exposures[channel] * (target / means[channel])
                    exposures[channel] = self._clamp_exposure(estimated, config)

            # Log gains as a compatible dict for ConvergenceLog
            gains_log = {"red": 1.0, "green": 1.0, "blue": 1.0}

            # Log initial state
            converged_flags = self._check_convergence(means, target, config.coarse_tolerance)
            self._convergence_log.add_iteration(
                0, means, exposures, gains_log, converged_flags,
                "Initial capture (Phase 1)",
                unified_gain=unified_gain
            )

            # ==================== PHASE 1: Coarse exposure balancing ====================
            logger.info(
                f"Phase 1: Coarse exposure balancing "
                f"(target={target:.0f}, coarse_tolerance={config.coarse_tolerance})"
            )

            fine_tune_threshold = config.coarse_tolerance * 2
            converged_means = None
            hw_limit_checks = 0
            iteration = 0

            for iteration in range(1, config.max_iterations + 1):
                self.jai_props.set_channel_exposures(**exposures, auto_enable=False)
                time.sleep(0.1)

                means = self._capture_and_analyze()

                # Phase 1 uses coarse_tolerance
                converged_flags = self._check_convergence(
                    means, target, config.coarse_tolerance
                )
                all_converged = all(converged_flags.values())

                deviations = {ch: abs(means[ch] - target) for ch in ["red", "green", "blue"]}
                max_deviation = max(deviations.values())

                notes = "P1:Converged" if all_converged else f"P1:max_dev={max_deviation:.1f}"
                self._convergence_log.add_iteration(
                    iteration, means, exposures, gains_log, converged_flags, notes,
                    unified_gain=unified_gain
                )

                logger.debug(
                    f"P1 Iter {iteration}: R={means['red']:.1f} (exp={exposures['red']:.2f}ms), "
                    f"G={means['green']:.1f} (exp={exposures['green']:.2f}ms), "
                    f"B={means['blue']:.1f} (exp={exposures['blue']:.2f}ms) "
                    f"| max_dev={max_deviation:.1f} (coarse +/-{config.coarse_tolerance})"
                )

                if all_converged:
                    logger.info(
                        f"Phase 1 converged after {iteration} iterations "
                        f"(all channels within {config.coarse_tolerance} of target {target:.0f})"
                    )
                    converged_means = means.copy()
                    break

                # Adjust exposures with adaptive damping
                for channel in ["red", "green", "blue"]:
                    if not converged_flags[channel] and means[channel] > 0:
                        deviation = abs(means[channel] - target)
                        ratio = target / means[channel]

                        if deviation <= fine_tune_threshold:
                            damping = config.fine_damping_factor
                        else:
                            damping = config.damping_factor

                        damped_ratio = 1.0 + (ratio - 1.0) * damping
                        new_exposure = exposures[channel] * damped_ratio
                        exposures[channel] = self._clamp_exposure(new_exposure, config)

                # Periodically check if stuck at ceiling
                if iteration % 5 == 0 or iteration == 1:
                    exposures, unified_gain, boosted = self._boost_unified_gain(
                        means, exposures, unified_gain, config, target
                    )

                    exposures, _dummy_gains, unified_gain, ceiling_extended = (
                        self._handle_stuck_at_ceiling(
                            means, exposures, {"red": 1.0, "green": 1.0, "blue": 1.0},
                            config, target, unified_gain
                        )
                    )

                    # Early termination: detect truly stuck at hardware limits
                    ug_max = JAICameraProperties.GAIN_UNIFIED_RANGE[1]
                    stuck_channels = []
                    for ch in ["red", "green", "blue"]:
                        at_hw_exp = abs(exposures[ch] - config.hardware_max_exposure_ms) < 0.01
                        at_ug_max = abs(unified_gain - ug_max) < 0.01
                        below_target = means[ch] < (target - config.coarse_tolerance)
                        if at_hw_exp and at_ug_max and below_target:
                            stuck_channels.append(ch)

                    if stuck_channels:
                        hw_limit_checks += 1
                        if hw_limit_checks >= 3:
                            logger.warning(
                                f"Phase 1 stopping: hardware limits reached. "
                                f"Channels {stuck_channels} at max gain and max exposure."
                            )
                            break
                    else:
                        hw_limit_checks = 0

            phase1_iterations = iteration

            # ==================== PHASE 2: Fine R/B analog gain tuning ====================
            logger.info("Phase 2: Fine R/B analog gain tuning")

            # Lock exposures from Phase 1
            self.jai_props.set_channel_exposures(**exposures, auto_enable=False)
            time.sleep(0.1)

            # Capture with locked exposures
            means = self._capture_and_analyze()
            logger.info(
                f"Phase 2 start: R={means['red']:.1f}, G={means['green']:.1f}, B={means['blue']:.1f}"
            )

            # Compute R/B corrections relative to green (reference channel)
            # If green is at target but red is low, we need analog_red > 1.0
            max_phase2_iters = 10
            phase2_damping = 0.6

            for p2_iter in range(1, max_phase2_iters + 1):
                # Check if already within final tolerance
                converged_flags = self._check_convergence(means, target, config.tolerance)
                all_converged = all(converged_flags.values())

                deviations = {ch: abs(means[ch] - target) for ch in ["red", "green", "blue"]}
                max_deviation = max(deviations.values())

                gains_log = {"red": analog_red, "green": 1.0, "blue": analog_blue}
                notes = f"P2:{'Converged' if all_converged else f'max_dev={max_deviation:.1f}'}"
                self._convergence_log.add_iteration(
                    phase1_iterations + p2_iter, means, exposures, gains_log,
                    converged_flags, notes, unified_gain=unified_gain
                )

                if all_converged:
                    logger.info(
                        f"Phase 2 converged after {p2_iter} iterations "
                        f"(all channels within {config.tolerance} of target)"
                    )
                    converged_means = means.copy()
                    break

                # Compute corrections - use green as reference
                # If red is below target, increase analog_red
                for ch_name, ch_attr in [("red", "analog_red"), ("blue", "analog_blue")]:
                    if not converged_flags[ch_name] and means[ch_name] > 0:
                        correction = target / means[ch_name]
                        damped_correction = 1.0 + (correction - 1.0) * phase2_damping

                        current_gain = analog_red if ch_name == "red" else analog_blue
                        new_gain = current_gain * damped_correction

                        # Clamp to hardware range
                        rb_min, rb_max = JAICameraProperties.GAIN_ANALOG_RED_RANGE
                        new_gain = max(rb_min, min(rb_max, new_gain))

                        if ch_name == "red":
                            analog_red = new_gain
                        else:
                            analog_blue = new_gain

                # Apply R/B analog gains
                try:
                    self.jai_props.set_rb_analog_gains(red=analog_red, blue=analog_blue)
                except Exception as e:
                    logger.warning(f"Failed to set R/B analog gains: {e}")
                    break

                time.sleep(0.1)
                means = self._capture_and_analyze()

                logger.debug(
                    f"P2 Iter {p2_iter}: R={means['red']:.1f} (aR={analog_red:.3f}), "
                    f"G={means['green']:.1f}, B={means['blue']:.1f} (aB={analog_blue:.3f}) "
                    f"| max_dev={max_deviation:.1f} (final +/-{config.tolerance})"
                )

            total_iterations = phase1_iterations + p2_iter if 'p2_iter' in dir() else phase1_iterations

            # Final validation
            if converged_means is not None:
                final_means = converged_means
                all_converged = True
            else:
                final_means = self._capture_and_analyze()
                final_converged = self._check_convergence(final_means, target, config.tolerance)
                all_converged = all(final_converged.values())

            final_deviations = {ch: abs(final_means[ch] - target) for ch in ["red", "green", "blue"]}
            max_final_deviation = max(final_deviations.values())

            if all_converged:
                logger.info(
                    f"White balance achieved within {config.tolerance} intensity levels. "
                    f"Final: R={final_means['red']:.1f}, G={final_means['green']:.1f}, "
                    f"B={final_means['blue']:.1f} (target: {target:.0f}, max_dev: {max_final_deviation:.1f})"
                )
            else:
                logger.warning(
                    f"Calibration did not achieve target precision of +/-{config.tolerance}. "
                    f"Final means: R={final_means['red']:.1f} (dev={final_deviations['red']:.1f}), "
                    f"G={final_means['green']:.1f} (dev={final_deviations['green']:.1f}), "
                    f"B={final_means['blue']:.1f} (dev={final_deviations['blue']:.1f}) "
                    f"(target: {target:.0f}, max_dev: {max_final_deviation:.1f})"
                )

            # Measure noise at final settings if configured
            noise_stats = None
            noise_verification_result = None
            if config.noise_frames > 0:
                try:
                    from microscope_control.jai.noise import JAINoiseMeasurement
                    noise_meter = JAINoiseMeasurement(self.hardware, self.jai_props)
                    noise_stats = noise_meter.measure_noise(
                        num_frames=config.noise_frames, settle_frames=1
                    )
                except Exception as e:
                    logger.warning(f"Failed to measure noise: {e}")

            # Run noise constraint verification if enabled
            if config.verify_noise and config.noise_frames > 0:
                noise_passes, noise_verification_result = self._verify_noise_constraints(config)
                if not noise_passes:
                    # Log warning but don't fail calibration - user can decide what to do
                    logger.warning(
                        f"Noise constraints not met (max_stddev={config.max_noise_stddev}, "
                        f"min_snr={config.min_snr}). Consider using a lower gain or longer exposure."
                    )

            # Build result
            result = WhiteBalanceResult(
                exposures_ms=exposures,
                unified_gain=unified_gain,
                analog_red=analog_red,
                analog_blue=analog_blue,
                noise_stats=noise_stats,
                black_levels=self._black_levels.copy(),
                converged=all_converged,
                iterations=total_iterations,
                final_means=final_means,
                target_value=config.target_value,
            )

            # Attach noise verification result to noise_stats if available
            if noise_verification_result is not None and result.noise_stats is not None:
                # Store verification result in noise_stats for later access
                result.noise_stats.verification_result = noise_verification_result

            # Save diagnostics if output path provided
            if output_path is not None:
                self._save_diagnostics(result, config, output_path)

            return result

        finally:
            # Restore PPM rotation if we changed it
            if original_ppm_angle is not None:
                try:
                    ppm_rotation_callback(original_ppm_angle)
                    logger.info(f"Restored PPM rotation to {original_ppm_angle} degrees")
                except Exception as e:
                    logger.warning(f"Failed to restore PPM rotation: {e}")

            # Restore Z position if we defocused
            if restore_z_callback is not None:
                try:
                    restore_z_callback()
                    logger.info("Restored original Z position")
                except Exception as e:
                    logger.warning(f"Failed to restore Z position: {e}")

    def calibrate_black_level(
        self,
        num_frames: int = 10,
        user_prompt_callback: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate black level using dark frames.

        Captures images with the light path blocked to measure sensor dark current
        and fixed pattern noise. The resulting black levels can be subtracted
        from acquired images.

        Args:
            num_frames: Number of dark frames to average
            user_prompt_callback: Callback to prompt user (returns True when ready).
                                 If None, assumes lens is already covered.

        Returns:
            Dictionary of per-channel black level offsets
        """
        logger.info(f"Starting black level calibration with {num_frames} dark frames")

        # Prompt user if callback provided
        if user_prompt_callback is not None:
            ready = user_prompt_callback(
                "Please cover the lens or close the shutter for dark frame capture. "
                "Press OK when ready."
            )
            if not ready:
                logger.warning("Black level calibration cancelled by user")
                return {"red": 0, "green": 0, "blue": 0}

        dark_frames = []
        for i in range(num_frames):
            img, _ = self.hardware.snap_image()
            if img is not None:
                dark_frames.append(img)
            time.sleep(0.1)  # Small delay between captures

        if not dark_frames:
            logger.error("Failed to capture any dark frames")
            return {"red": 0, "green": 0, "blue": 0}

        # Stack and average
        dark_stack = np.stack(dark_frames, axis=0)
        dark_mean = dark_stack.mean(axis=0)

        # Calculate per-channel statistics
        black_levels = {
            "red": float(dark_mean[:, :, 0].mean()),
            "green": float(dark_mean[:, :, 1].mean()),
            "blue": float(dark_mean[:, :, 2].mean()),
        }

        # Also calculate standard deviation for diagnostics
        black_std = {
            "red": float(dark_mean[:, :, 0].std()),
            "green": float(dark_mean[:, :, 1].std()),
            "blue": float(dark_mean[:, :, 2].std()),
        }

        logger.info(
            f"Black levels: R={black_levels['red']:.1f} (+/-{black_std['red']:.1f}), "
            f"G={black_levels['green']:.1f} (+/-{black_std['green']:.1f}), "
            f"B={black_levels['blue']:.1f} (+/-{black_std['blue']:.1f})"
        )

        # Prompt user to uncover lens
        if user_prompt_callback is not None:
            user_prompt_callback("Please uncover the lens. Press OK when ready.")

        self._black_levels = black_levels
        return black_levels

    def _capture_and_analyze(self) -> Dict[str, float]:
        """
        Capture an image and return per-channel mean values.

        Returns:
            Dictionary with 'red', 'green', 'blue' keys and mean values
        """
        img, tags = self.hardware.snap_image()

        if img is None:
            raise RuntimeError("Failed to capture image")

        # JAI camera returns RGB image (no Bayer)
        if len(img.shape) != 3 or img.shape[2] < 3:
            raise ValueError(f"Expected RGB image, got shape: {img.shape}")

        # Subtract black levels if calibrated
        means = {
            "red": float(np.mean(img[:, :, 0])) - self._black_levels.get("red", 0),
            "green": float(np.mean(img[:, :, 1])) - self._black_levels.get("green", 0),
            "blue": float(np.mean(img[:, :, 2])) - self._black_levels.get("blue", 0),
        }

        # Ensure non-negative
        for channel in means:
            means[channel] = max(0.0, means[channel])

        return means

    def _clamp_exposure(self, exposure: float, config: CalibrationConfig) -> float:
        """Clamp exposure to valid range."""
        return max(config.min_exposure_ms, min(config.max_exposure_ms, exposure))

    def _check_convergence(
        self,
        means: Dict[str, float],
        target: float,
        tolerance: float,
    ) -> Dict[str, bool]:
        """Check if each channel is within tolerance of target."""
        return {
            channel: abs(means[channel] - target) <= tolerance
            for channel in ["red", "green", "blue"]
        }

    def _select_optimal_unified_gain(
        self,
        config: CalibrationConfig,
        target: float,
    ) -> float:
        """
        Select lowest unified gain that meets exposure constraints.

        Based on noise test data:
        - Unified gain 1.0: R=2.4, G=1.8, B=1.0 stddev at 25ms - best for bright scenes
        - Unified gain 2.0: R=4.5, G=3.4, B=1.9 stddev - optimal balance
        - Unified gain 3.0+: Increasing noise - only for very dark scenes

        Critical insight: Analog R/B gains (0.48-4.0) have minimal noise impact -
        they're post-amplification scalars, not true hardware gain. Exposure is
        the primary control for signal quality.

        Args:
            config: Calibration configuration with gain mode settings
            target: Target intensity value

        Returns:
            Selected unified gain value (1.0-8.0)
        """
        if config.unified_gain_mode == 'fixed':
            logger.debug(f"Using fixed unified gain: {config.base_gain}")
            return config.base_gain

        # Save current camera state to restore later
        try:
            saved_exposures = self.jai_props.get_channel_exposures()
        except Exception:
            saved_exposures = {"red": 50.0, "green": 50.0, "blue": 50.0}

        # Test at minimum gain to assess light level
        test_gain = 1.0
        test_exposure = 50.0  # ms

        try:
            self.jai_props.set_unified_gain(test_gain)
            self.jai_props.set_channel_exposures(
                red=test_exposure, green=test_exposure, blue=test_exposure,
                auto_enable=True
            )
            time.sleep(0.2)  # Allow settings to take effect

            means = self._capture_and_analyze()
        except Exception as e:
            logger.warning(f"Failed to test at minimum gain: {e}. Using base_gain={config.base_gain}")
            return min(config.base_gain, 5.0)

        min_mean = min(means.values())
        if min_mean <= 0:
            logger.warning("Zero signal at minimum gain - using default base_gain")
            return min(config.base_gain, 5.0)

        # Calculate how much signal boost we need
        required_factor = target / min_mean
        logger.debug(
            f"Light assessment at gain={test_gain}, exp={test_exposure}ms: "
            f"min_mean={min_mean:.1f}, target={target:.0f}, required_factor={required_factor:.1f}"
        )

        # Select gain based on required signal boost
        # These thresholds are derived from noise characterization data
        if config.unified_gain_mode == 'minimize_noise':
            # Prioritize lowest gain - accept longer exposures
            if required_factor <= 2.0:
                selected_gain = 1.0
            elif required_factor <= 4.0:
                selected_gain = 2.0
            elif required_factor <= 8.0:
                selected_gain = 3.0
            else:
                # Very dark - use higher gain but still conservative
                selected_gain = min(required_factor / 3, 6.0)

        elif config.unified_gain_mode == 'minimize_exposure':
            # Prioritize shorter exposures - accept more noise
            if required_factor <= 1.2:
                selected_gain = 1.0
            elif required_factor <= 2.0:
                selected_gain = 2.0
            elif required_factor <= 3.0:
                selected_gain = 3.0
            elif required_factor <= 5.0:
                selected_gain = 4.0
            else:
                selected_gain = min(required_factor / 1.5, 8.0)

        else:  # 'auto' - balanced approach
            # Empirical gain selection based on noise test data
            if required_factor <= 1.5:
                selected_gain = 1.0  # Bright - minimal gain, best quality
            elif required_factor <= 3.0:
                selected_gain = 2.0  # Moderate - optimal balance
            elif required_factor <= 5.0:
                selected_gain = 3.0  # Dim - acceptable tradeoff
            else:
                # Very dark - use higher gain but cap at reasonable level
                selected_gain = min(required_factor / 2, 8.0)

        # Clamp to hardware limits
        ug_min, ug_max = JAICameraProperties.GAIN_UNIFIED_RANGE
        selected_gain = max(ug_min, min(ug_max, selected_gain))

        logger.info(
            f"Noise-aware gain selection ({config.unified_gain_mode}): "
            f"required_factor={required_factor:.1f} -> unified_gain={selected_gain:.1f}"
        )

        return selected_gain

    def _verify_noise_constraints(
        self,
        config: CalibrationConfig,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that final calibration settings meet noise constraints.

        Captures frames and measures temporal noise using JAINoiseMeasurement,
        then checks if per-channel noise (stddev) and SNR meet the configured
        thresholds.

        Args:
            config: Calibration configuration with noise constraints

        Returns:
            Tuple of (passes, details_dict) where:
            - passes: True if all constraints are met
            - details_dict: Contains stddevs, snr, and per-channel pass/fail info
        """
        try:
            from microscope_control.jai.noise import JAINoiseMeasurement

            noise_meter = JAINoiseMeasurement(self.hardware, self.jai_props)
            stats = noise_meter.measure_noise(
                num_frames=config.noise_frames,
                settle_frames=1
            )
        except Exception as e:
            logger.warning(f"Noise verification failed to measure noise: {e}")
            return False, {'error': str(e)}

        passes = True
        channel_results = {}

        for channel in ['red', 'green', 'blue']:
            stddev = stats.channel_stddevs.get(channel, 0.0)
            snr = stats.channel_snr.get(channel, 0.0)

            stddev_ok = stddev <= config.max_noise_stddev
            snr_ok = snr >= config.min_snr

            channel_results[channel] = {
                'stddev': stddev,
                'snr': snr,
                'stddev_ok': stddev_ok,
                'snr_ok': snr_ok,
                'passes': stddev_ok and snr_ok,
            }

            if not (stddev_ok and snr_ok):
                passes = False

        result = {
            'passes': passes,
            'channel_stddevs': stats.channel_stddevs,
            'channel_snr': stats.channel_snr,
            'channel_means': stats.channel_means,
            'thresholds': {
                'max_noise_stddev': config.max_noise_stddev,
                'min_snr': config.min_snr,
            },
            'per_channel': channel_results,
        }

        if passes:
            logger.info(
                f"Noise verification PASSED: "
                f"R(std={stats.channel_stddevs['red']:.2f}, SNR={stats.channel_snr['red']:.1f}) "
                f"G(std={stats.channel_stddevs['green']:.2f}, SNR={stats.channel_snr['green']:.1f}) "
                f"B(std={stats.channel_stddevs['blue']:.2f}, SNR={stats.channel_snr['blue']:.1f})"
            )
        else:
            failures = []
            for ch, info in channel_results.items():
                if not info['passes']:
                    issues = []
                    if not info['stddev_ok']:
                        issues.append(f"stddev={info['stddev']:.2f}>{config.max_noise_stddev}")
                    if not info['snr_ok']:
                        issues.append(f"SNR={info['snr']:.1f}<{config.min_snr}")
                    failures.append(f"{ch}({', '.join(issues)})")
            logger.warning(
                f"Noise verification FAILED: {', '.join(failures)}"
            )

        return passes, result

    def _apply_quality_preset(self, config: CalibrationConfig) -> CalibrationConfig:
        """
        Apply quality preset values to config if a valid preset is specified.

        Args:
            config: Original calibration configuration

        Returns:
            Configuration with preset values applied (or original if no preset)
        """
        if not config.quality_mode or config.quality_mode not in QUALITY_PRESETS:
            return config

        preset = QUALITY_PRESETS[config.quality_mode]
        logger.info(f"Applying quality preset: {config.quality_mode}")

        # Apply preset values - these override individual settings
        config.max_exposure_ms = preset.get('max_exposure_ms', config.max_exposure_ms)
        config.max_noise_stddev = preset.get('max_noise_stddev', config.max_noise_stddev)
        config.min_snr = preset.get('min_snr', config.min_snr)
        config.unified_gain_mode = preset.get('unified_gain_mode', config.unified_gain_mode)
        config.base_gain = preset.get('base_gain', config.base_gain)

        logger.debug(
            f"Preset values: max_exp={config.max_exposure_ms}ms, "
            f"max_noise={config.max_noise_stddev}, min_snr={config.min_snr}, "
            f"gain_mode={config.unified_gain_mode}, base_gain={config.base_gain}"
        )

        return config

    def _boost_unified_gain(
        self,
        means: Dict[str, float],
        exposures: Dict[str, float],
        unified_gain: float,
        config: CalibrationConfig,
        target: float,
    ) -> Tuple[Dict[str, float], float, bool]:
        """
        Check if unified gain should be boosted when channels are stuck at max exposure.

        When any channel is at max_exposure AND below target, boost the unified gain
        and proportionally reduce ALL channel exposures.

        Args:
            means: Current per-channel mean intensities
            exposures: Current per-channel exposures in ms
            unified_gain: Current unified gain value
            config: Calibration configuration
            target: Target intensity value

        Returns:
            Tuple of (updated_exposures, updated_unified_gain, was_boosted)
        """
        ug_max = JAICameraProperties.GAIN_UNIFIED_RANGE[1]

        # Already at max unified gain
        if unified_gain >= ug_max - 0.01:
            return exposures, unified_gain, False

        # Find channels stuck at ceiling and below target
        stuck_channels = []
        for ch in ["red", "green", "blue"]:
            at_ceiling = abs(exposures[ch] - config.max_exposure_ms) < 0.01
            below_target = means[ch] < (target - config.tolerance)
            if at_ceiling and below_target and means[ch] > 0:
                stuck_channels.append(ch)

        if not stuck_channels:
            return exposures, unified_gain, False

        # Calculate gain factor needed based on the dimmest stuck channel
        stuck_means = [means[ch] for ch in stuck_channels]
        min_stuck_mean = min(stuck_means)
        gain_factor = target / min_stuck_mean

        # Compute new unified gain
        old_gain = unified_gain
        new_gain = min(unified_gain * gain_factor, ug_max)
        actual_factor = new_gain / old_gain

        # Proportionally reduce ALL channel exposures
        new_exposures = {}
        for ch in ["red", "green", "blue"]:
            new_exposures[ch] = self._clamp_exposure(
                exposures[ch] / actual_factor, config
            )

        # Apply to hardware
        try:
            self.jai_props.set_unified_gain(new_gain)
        except Exception as e:
            logger.warning(f"Failed to boost unified gain: {e}")
            return exposures, unified_gain, False

        logger.info(
            f"Boosting unified gain: {old_gain:.1f}x -> {new_gain:.1f}x, "
            f"reducing all exposures by {actual_factor:.1f}x "
            f"(stuck channels: {stuck_channels})"
        )

        return new_exposures, new_gain, True

    def _handle_stuck_at_ceiling(
        self,
        means: Dict[str, float],
        exposures: Dict[str, float],
        gains: Dict[str, float],
        config: CalibrationConfig,
        target: float,
        unified_gain: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, float], float, bool]:
        """
        Detect channels stuck at max exposure with insufficient signal and
        try extending the exposure ceiling.

        This is called after _boost_unified_gain has already attempted to
        increase the unified gain. If unified gain is maxed out and channels
        are still stuck, this extends the max_exposure_ms up to the hardware
        limit.

        Args:
            means: Current per-channel mean intensities
            exposures: Current per-channel exposures in ms
            gains: Current per-channel gain values (unused, kept for API compat)
            config: Calibration configuration (may be modified if exposure extended)
            target: Target intensity value
            unified_gain: Current unified gain value

        Returns:
            Tuple of (updated_exposures, gains, unified_gain, ceiling_extended)
            where ceiling_extended indicates if config.max_exposure_ms was raised
        """
        ceiling_extended = False
        new_exposures = exposures.copy()

        ug_max = JAICameraProperties.GAIN_UNIFIED_RANGE[1]

        # Check if any channel is stuck
        any_stuck = False
        for channel in ["red", "green", "blue"]:
            at_ceiling = abs(exposures[channel] - config.max_exposure_ms) < 0.01
            below_target = means[channel] < (target - config.coarse_tolerance)
            if at_ceiling and below_target and means[channel] > 0:
                any_stuck = True
                break

        if not any_stuck:
            return new_exposures, gains, unified_gain, False

        if unified_gain < ug_max - 0.01:
            # _boost_unified_gain handles this case
            return new_exposures, gains, unified_gain, False

        # Unified gain maxed out - try extending exposure ceiling
        for channel in ["red", "green", "blue"]:
            at_ceiling = abs(exposures[channel] - config.max_exposure_ms) < 0.01
            below_target = means[channel] < (target - config.coarse_tolerance)

            if not (at_ceiling and below_target):
                continue
            if means[channel] <= 0:
                continue

            needed_exp = exposures[channel] * (target / means[channel])
            clamped_exp = min(needed_exp, config.hardware_max_exposure_ms)

            if clamped_exp > config.max_exposure_ms:
                old_max = config.max_exposure_ms
                config.max_exposure_ms = clamped_exp
                new_exposures[channel] = clamped_exp
                ceiling_extended = True

                logger.warning(
                    f"Channel {channel} at max unified gain ({unified_gain:.1f}x) "
                    f"and max exposure ({old_max:.0f}ms) but only reaching "
                    f"{means[channel]:.0f}/{target:.0f}. "
                    f"Extending max exposure to {clamped_exp:.0f}ms."
                )

                if needed_exp > config.hardware_max_exposure_ms:
                    estimate = means[channel] * (
                        config.hardware_max_exposure_ms / exposures[channel]
                    )
                    logger.warning(
                        f"Channel {channel} cannot reach target {target:.0f} even at "
                        f"hardware limits (unified gain={unified_gain:.1f}x, "
                        f"exposure={config.hardware_max_exposure_ms:.0f}ms). "
                        f"Best achievable: ~{estimate:.0f}"
                    )

        return new_exposures, gains, unified_gain, ceiling_extended

    def _save_diagnostics(
        self,
        result: WhiteBalanceResult,
        config: CalibrationConfig,
        output_path: Path,
    ) -> None:
        """Save diagnostic output files to output folder.

        Note: QuPath already includes 'white_balance_calibration' in the output path,
        so we use the output_path directly without adding another subfolder.
        """
        output_path = Path(output_path)

        # Use output_path directly - QuPath already includes the subfolder structure
        diagnostics_folder = output_path
        diagnostics_folder.mkdir(parents=True, exist_ok=True)

        # Save convergence log
        self._convergence_log.save_csv(diagnostics_folder / "convergence_log.csv")

        # Save settings YAML (in diagnostics folder for reference)
        self.save_calibration(result, diagnostics_folder / "white_balance_settings.yml")

        # Save verification image with calibrated white balance settings
        try:
            img, _ = self.hardware.snap_image()
            if img is not None:
                verification_path = diagnostics_folder / "white_balance_verification.tif"
                try:
                    import tifffile
                    tifffile.imwrite(str(verification_path), img)
                except ImportError:
                    # Fallback to numpy save if tifffile not available
                    verification_path = diagnostics_folder / "white_balance_verification.npy"
                    np.save(str(verification_path), img)
                logger.info(f"Saved verification image to {verification_path}")

                # Log per-channel means of verification image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    r_mean = float(np.mean(img[:, :, 0]))
                    g_mean = float(np.mean(img[:, :, 1]))
                    b_mean = float(np.mean(img[:, :, 2]))
                    logger.info(
                        f"Verification image channel means: "
                        f"R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}"
                    )
        except Exception as e:
            logger.warning(f"Failed to save verification image: {e}")

        # Generate histogram plot if matplotlib is available
        try:
            self._save_histogram_plot(result, config, diagnostics_folder / "intensity_histograms.png")
        except Exception as e:
            logger.warning(f"Failed to generate histogram plot: {e}")

        # Save black level calibration if performed
        if any(v > 0 for v in self._black_levels.values()):
            black_level_data = {
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "method": "dark_frame_average",
                    "frames_averaged": config.dark_frame_count,
                },
                "per_channel_black_levels": {
                    "red": {"mean": self._black_levels["red"]},
                    "green": {"mean": self._black_levels["green"]},
                    "blue": {"mean": self._black_levels["blue"]},
                },
            }
            with open(diagnostics_folder / "black_level_calibration.yml", "w") as f:
                yaml.dump(black_level_data, f, default_flow_style=False)

    def _save_histogram_plot(
        self,
        result: WhiteBalanceResult,
        config: CalibrationConfig,
        output_path: Path,
    ) -> None:
        """Generate histogram visualization."""
        # Use non-interactive backend to avoid Tkinter threading issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Top row: Capture final image and show histograms
        img, _ = self.hardware.snap_image()
        if img is not None:
            colors = ["red", "green", "blue"]
            for i, (ax, color) in enumerate(zip(axes[0], colors)):
                channel_data = img[:, :, i].flatten()
                ax.hist(channel_data, bins=50, color=color, alpha=0.7)
                ax.axvline(result.final_means[color], color="black", linestyle="--", linewidth=2)
                ax.axvline(config.target_value, color="gray", linestyle=":", linewidth=2)
                ax.set_title(f"{color.capitalize()} Channel")
                ax.set_xlabel("Intensity")
                ax.set_ylabel("Count")

        # Bottom row: Convergence plots
        if self._convergence_log.iterations:
            iterations = [d["iteration"] for d in self._convergence_log.iterations]
            for i, (ax, color) in enumerate(zip(axes[1], ["red", "green", "blue"])):
                intensities = [d[f"{color}_intensity"] for d in self._convergence_log.iterations]
                ax.plot(iterations, intensities, f"{color[0]}-o", linewidth=2)
                ax.axhline(config.target_value, color="gray", linestyle="--", linewidth=1)
                ax.fill_between(
                    iterations,
                    config.target_value - config.tolerance,
                    config.target_value + config.tolerance,
                    alpha=0.2,
                    color="gray",
                )
                ax.set_title(f"{color.capitalize()} Convergence")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Mean Intensity")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved histogram plot to {output_path}")

    def save_calibration(
        self,
        result: WhiteBalanceResult,
        output_path: Path,
    ) -> None:
        """
        Save calibration results to YAML file.

        Args:
            result: Calibration results to save
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build noise stats section if available
        noise_section = {}
        if result.noise_stats is not None:
            noise_section = {
                "channel_means": result.noise_stats.channel_means,
                "channel_stddevs": result.noise_stats.channel_stddevs,
                "channel_snr": result.noise_stats.channel_snr,
                "num_frames": result.noise_stats.num_frames,
            }

        data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "version": "2.0",
                "description": "QPSC White Balance Settings - Per-channel exposure with unified gain + R/B analog correction",
                "wb_method": result.wb_method,
                "reproducible": result.wb_method.startswith("manual_"),
            },
            "hardware": {
                "camera": "JAICamera",
                "detector": "JAI",
            },
            "calibration_parameters": {
                "target_intensity_8bit": result.target_value,
                "achieved_intensities": {
                    "red": round(result.final_means["red"], 1),
                    "green": round(result.final_means["green"], 1),
                    "blue": round(result.final_means["blue"], 1),
                },
                "converged": result.converged,
                "iterations_used": result.iterations,
            },
            "per_channel_exposures_ms": {
                "red": round(result.exposures_ms["red"], 2),
                "green": round(result.exposures_ms["green"], 2),
                "blue": round(result.exposures_ms["blue"], 2),
                "exposure_mode": "individual",
            },
            "gains": {
                "unified_gain": round(result.unified_gain, 3),
                "analog_red": round(result.analog_red, 3),
                "analog_blue": round(result.analog_blue, 3),
                "gain_mode": "unified_with_rb_correction",
            },
            "black_levels": result.black_levels,
            "combined_settings": {
                "ExposureIsIndividual": "On",
                "Exposure_Red": round(result.exposures_ms["red"], 2),
                "Exposure_Green": round(result.exposures_ms["green"], 2),
                "Exposure_Blue": round(result.exposures_ms["blue"], 2),
                "GainIsIndividual": "Off",
                "Gain": round(result.unified_gain, 3),
                "Gain_AnalogRed": round(result.analog_red, 3),
                "Gain_AnalogBlue": round(result.analog_blue, 3),
            },
            "notes": [
                "Apply these settings before acquisition for consistent color balance",
                "If lighting conditions change significantly, recalibrate",
                "Settings are specific to this detector/modality/objective combination",
                "Green is the reference channel - only R/B analog gains are adjusted",
            ],
        }

        if noise_section:
            data["noise_stats"] = noise_section

        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved white balance calibration to: {output_path}")

    def update_imageprocessing_config(
        self,
        config_path: Path,
        result: WhiteBalanceResult,
        calibration_type: str = "simple",
        angle_name: Optional[str] = None,
        modality: Optional[str] = None,
        objective: Optional[str] = None,
        detector: Optional[str] = None,
    ) -> bool:
        """
        Update the imageprocessing YAML file with calibration results.

        The imageprocessing file is derived from the config path:
        config_PPM.yml -> imageprocessing_PPM.yml

        Saves calibration results to:
        imaging_profiles.{modality}.{objective}.{detector} (exposures_ms and gains)

        Args:
            config_path: Path to the main config file (e.g., config_PPM.yml)
            result: Calibration results to save
            calibration_type: 'simple' for single calibration, 'ppm' for per-angle
            angle_name: For PPM calibration, the angle name (e.g., 'negative', 'crossed')
            modality: Modality name (e.g., 'ppm') for imaging_profiles update
            objective: Objective ID for imaging_profiles update
            detector: Detector ID for imaging_profiles update

        Returns:
            True if successfully updated, False otherwise
        """
        config_path = Path(config_path)

        # Derive imageprocessing file path from config path
        # config_PPM.yml -> imageprocessing_PPM.yml
        config_name = config_path.stem  # e.g., "config_PPM"
        if config_name.startswith("config_"):
            microscope_name = config_name[7:]  # e.g., "PPM"
            imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
        else:
            imageprocessing_name = f"imageprocessing_{config_name}.yml"

        imageprocessing_path = config_path.parent / imageprocessing_name

        try:
            # Load existing imageprocessing file or create empty dict
            if imageprocessing_path.exists():
                with open(imageprocessing_path, "r") as f:
                    ip_data = yaml.safe_load(f) or {}
            else:
                ip_data = {}

            # Remove legacy audit trail section if present (no longer used)
            ip_data.pop("white_balance_calibration", None)

            # Update imaging_profiles section with calibration results
            if modality and objective and detector and angle_name:
                if "imaging_profiles" not in ip_data:
                    ip_data["imaging_profiles"] = {}
                if modality not in ip_data["imaging_profiles"]:
                    ip_data["imaging_profiles"][modality] = {}
                if objective not in ip_data["imaging_profiles"][modality]:
                    ip_data["imaging_profiles"][modality][objective] = {}
                if detector not in ip_data["imaging_profiles"][modality][objective]:
                    ip_data["imaging_profiles"][modality][objective][detector] = {}

                profile = ip_data["imaging_profiles"][modality][objective][detector]

                # Ensure exposures_ms section exists
                if "exposures_ms" not in profile:
                    profile["exposures_ms"] = {}

                # Update the per-channel exposures for this angle
                profile["exposures_ms"][angle_name] = {
                    "all": round(result.exposures_ms["green"], 2),  # Use green as 'all' reference
                    "r": round(result.exposures_ms["red"], 2),
                    "g": round(result.exposures_ms["green"], 2),
                    "b": round(result.exposures_ms["blue"], 2),
                }

                # Update gains with new model (unified + R/B analog)
                if "gains" not in profile:
                    profile["gains"] = {}
                profile["gains"][angle_name] = {
                    "unified_gain": round(result.unified_gain, 3),
                    "analog_red": round(result.analog_red, 3),
                    "analog_blue": round(result.analog_blue, 3),
                    "wb_method": result.wb_method,
                }

                logger.info(
                    f"Updated imaging_profiles.{modality}.{objective}.{detector}.exposures_ms.{angle_name}"
                )

            # Save updated imageprocessing file
            with open(imageprocessing_path, "w") as f:
                yaml.dump(ip_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Updated imageprocessing config: {imageprocessing_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to update imageprocessing config: {e}")
            return False

    def load_calibration(self, input_path: Path) -> WhiteBalanceResult:
        """
        Load calibration results from YAML file.

        Args:
            input_path: Path to YAML file

        Returns:
            WhiteBalanceResult loaded from file
        """
        input_path = Path(input_path)

        with open(input_path, "r") as f:
            data = yaml.safe_load(f)

        exposures = data.get("per_channel_exposures_ms", {})
        cal_params = data.get("calibration_parameters", {})
        metadata = data.get("metadata", {})

        # New format (v2.0): gains section with unified_gain, analog_red, analog_blue
        gain_data = data.get("gains", {})
        if gain_data and "analog_red" in gain_data:
            unified_gain = gain_data.get("unified_gain", 1.0)
            analog_red = gain_data.get("analog_red", 1.0)
            analog_blue = gain_data.get("analog_blue", 1.0)
        else:
            # Old format (v1.0): per_channel_gains with analog {red, green, blue}
            old_gain_data = data.get("per_channel_gains", {})
            unified_gain = old_gain_data.get("unified_gain", 1.0)
            old_analog = old_gain_data.get("analog", {})
            analog_red = old_analog.get("red", 1.0)
            analog_blue = old_analog.get("blue", 1.0)
            if old_analog:
                logger.info(
                    "Loaded calibration from old format (v1.0) - "
                    "mapped per-channel gains to analog_red/analog_blue"
                )

        return WhiteBalanceResult(
            exposures_ms={
                "red": exposures.get("red", 50.0),
                "green": exposures.get("green", 50.0),
                "blue": exposures.get("blue", 50.0),
            },
            unified_gain=unified_gain,
            analog_red=analog_red,
            analog_blue=analog_blue,
            wb_method=metadata.get("wb_method", "unknown"),
            black_levels=data.get("black_levels", {"red": 0, "green": 0, "blue": 0}),
            converged=cal_params.get("converged", True),
            iterations=cal_params.get("iterations_used", 0),
            final_means=cal_params.get("achieved_intensities", {}),
            target_value=cal_params.get("target_intensity_8bit", 180.0),
        )

    def apply_calibration(self, result: WhiteBalanceResult) -> None:
        """
        Apply calibration results to the camera.

        Uses unified gain mode with R/B analog corrections.

        Args:
            result: WhiteBalanceResult to apply
        """
        self.jai_props.set_channel_exposures(
            red=result.exposures_ms["red"],
            green=result.exposures_ms["green"],
            blue=result.exposures_ms["blue"],
        )

        if result.unified_gain > 1.01:
            self.jai_props.set_unified_gain(result.unified_gain)
            logger.info(f"Applied unified gain: {result.unified_gain:.2f}x")

        if abs(result.analog_red - 1.0) > 0.01 or abs(result.analog_blue - 1.0) > 0.01:
            self.jai_props.set_rb_analog_gains(
                red=result.analog_red,
                blue=result.analog_blue,
            )
            logger.info(
                f"Applied R/B analog gains: R={result.analog_red:.3f}, "
                f"B={result.analog_blue:.3f}"
            )

        logger.info("Applied white balance calibration to camera")

    def calibrate_simple(
        self,
        initial_exposure_ms: float,
        target: float = 180.0,
        tolerance: float = 5.0,
        output_path: Optional[Path] = None,
        gain_threshold_ratio: Optional[float] = None,
        max_iterations: Optional[int] = None,
        calibrate_black_level: Optional[bool] = None,
        base_gain: Optional[float] = None,
        # Legacy params accepted but ignored for backward compatibility
        max_gain_db: Optional[float] = None,
        exposure_soft_cap_ms: Optional[float] = None,
        boosted_max_gain_db: Optional[float] = None,
    ) -> WhiteBalanceResult:
        """
        Run simple white balance calibration at a single exposure.

        This is a convenience method for basic white balance calibration
        at the current PPM angle (no rotation). Use this for non-PPM imaging
        or when you want to calibrate at the current angle only.

        Args:
            initial_exposure_ms: Starting exposure time for all channels (ms)
            target: Target intensity (0-255 for 8-bit)
            tolerance: Acceptable deviation from target
            output_path: Optional path to save calibration results
            gain_threshold_ratio: Optional exposure ratio threshold (default 2.0)
            max_iterations: Optional max calibration iterations (default 30)
            calibrate_black_level: Optional whether to calibrate black level (default True)
            base_gain: Optional unified gain starting value (default 5.0)

        Returns:
            WhiteBalanceResult with calibrated per-channel exposures
        """
        logger.info(f"Starting simple white balance calibration (initial exp={initial_exposure_ms}ms)")

        # Stop live mode if running - camera properties cannot be changed during live streaming
        if self.hardware.core.is_sequence_running():
            if self.hardware.studio is not None:
                self.hardware.studio.live().set_live_mode(False)
                logger.info("Stopped live mode before calibration")

        # Set initial exposure on all channels
        self.jai_props.set_channel_exposures(
            red=initial_exposure_ms,
            green=initial_exposure_ms,
            blue=initial_exposure_ms,
        )

        # Build config
        config_kwargs = {
            "target_value": target,
            "tolerance": tolerance,
        }
        if gain_threshold_ratio is not None:
            config_kwargs["gain_threshold_ratio"] = gain_threshold_ratio
        if max_iterations is not None:
            config_kwargs["max_iterations"] = max_iterations
        if calibrate_black_level is not None:
            config_kwargs["calibrate_black_level"] = calibrate_black_level
        if base_gain is not None:
            config_kwargs["base_gain"] = base_gain

        config = CalibrationConfig(**config_kwargs)

        logger.info(
            f"Simple WB config: target={target}, tolerance={tolerance}, "
            f"gain_threshold={config.gain_threshold_ratio}, "
            f"max_iter={config.max_iterations}, calibrate_bl={config.calibrate_black_level}, "
            f"base_gain={config.base_gain}"
        )

        # Run 2-phase calibration without PPM rotation
        result = self.calibrate(
            config=config,
            output_path=output_path,
            ppm_rotation_callback=None,
            defocus_callback=None,
        )
        result.wb_method = "manual_simple"
        return result

    def calibrate_ppm(
        self,
        angle_exposures: Dict[str, Tuple[float, float]],
        target: float = 180.0,
        tolerance: float = 5.0,
        output_path: Optional[Path] = None,
        ppm_rotation_callback: Optional[Callable[[float], None]] = None,
        per_angle_targets: Optional[Dict[str, float]] = None,
        gain_threshold_ratio: Optional[float] = None,
        max_iterations: Optional[int] = None,
        calibrate_black_level: Optional[bool] = None,
        base_gain: Optional[float] = None,
        # Legacy params accepted but ignored for backward compatibility
        max_gain_db: Optional[float] = None,
        exposure_soft_cap_ms: Optional[float] = None,
        boosted_max_gain_db: Optional[float] = None,
    ) -> Dict[str, WhiteBalanceResult]:
        """
        Run white balance calibration for each PPM angle.

        Calibrates the camera at each of the specified PPM angles, using
        different initial exposures for each angle. This is useful for
        PPM imaging where different polarization angles require different
        exposure settings.

        Args:
            angle_exposures: Dictionary mapping angle names to (angle, exposure_ms) tuples.
                           Standard names are 'positive', 'negative', 'crossed', 'uncrossed'.
                           Example: {'positive': (7.0, 50.0), 'uncrossed': (90.0, 1.2)}
            target: Default target intensity (0-255 for 8-bit), used when per_angle_targets
                   doesn't specify a value for the angle.
            tolerance: Acceptable deviation from target
            output_path: Optional base path to save calibration results.
                        Results are saved to {output_path}/{angle_name}/ subdirectories.
            ppm_rotation_callback: Callback function to rotate PPM stage.
                                  Takes angle in degrees as argument.
            per_angle_targets: Optional dictionary mapping angle names to target intensities.
                              Example: {'crossed': 125.0, 'positive': 160.0, 'negative': 160.0, 'uncrossed': 245.0}
                              If not provided or angle not found, uses the 'target' parameter.
            max_gain_db: Optional max analog gain in dB (default 3.0)
            gain_threshold_ratio: Optional exposure ratio before using gain (default 2.0)
            max_iterations: Optional max calibration iterations (default 30)
            calibrate_black_level: Optional whether to calibrate black level (default True)

        Returns:
            Dictionary mapping angle names to WhiteBalanceResult objects
        """
        logger.info(f"Starting PPM white balance calibration for {len(angle_exposures)} angles")
        if per_angle_targets:
            logger.info(f"Using per-angle targets: {per_angle_targets}")
        logger.info(
            f"PPM WB settings: gain_threshold={gain_threshold_ratio}, "
            f"max_iter={max_iterations}, calibrate_bl={calibrate_black_level}, "
            f"base_gain={base_gain}"
        )

        # Stop live mode if running - camera properties cannot be changed during live streaming
        if self.hardware.core.is_sequence_running():
            if self.hardware.studio is not None:
                self.hardware.studio.live().set_live_mode(False)
                logger.info("Stopped live mode before calibration")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        for name, (angle, exposure) in angle_exposures.items():
            # CRITICAL: Reset gain state before each angle to ensure clean state.
            # Without this, gain settings from a previous angle persist and
            # corrupt calibration of subsequent angles.
            try:
                self.jai_props.set_unified_gain(1.0)
                self.jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
                logger.debug(f"Reset gains before calibrating '{name}'")
            except Exception as e:
                logger.warning(f"Could not reset gain mode before '{name}': {e}")

            # Determine target for this angle
            angle_target = target  # Default
            if per_angle_targets and name in per_angle_targets:
                angle_target = per_angle_targets[name]

            # Crossed polarizers need much longer exposures since most light is blocked
            # Hardware supports up to ~7900ms at min frame rate (0.125 Hz)
            # Use 2000ms limit for crossed (dim), 500ms for others (reasonable calibration speed)
            if name == "crossed":
                max_exp = 2000.0
            else:
                max_exp = 500.0

            logger.info(
                f"PPM WB: Calibrating '{name}' at {angle} deg, "
                f"initial exp={exposure}ms, target={angle_target}, max_exp={max_exp}ms"
            )

            # Create config for this angle with its specific target and max exposure
            config_kwargs = {
                "target_value": angle_target,
                "tolerance": tolerance,
                "max_exposure_ms": max_exp,
            }
            if gain_threshold_ratio is not None:
                config_kwargs["gain_threshold_ratio"] = gain_threshold_ratio
            if max_iterations is not None:
                config_kwargs["max_iterations"] = max_iterations
            if calibrate_black_level is not None:
                config_kwargs["calibrate_black_level"] = calibrate_black_level
            if base_gain is not None:
                config_kwargs["base_gain"] = base_gain

            config = CalibrationConfig(**config_kwargs)

            # Rotate to target angle
            if ppm_rotation_callback is not None:
                try:
                    ppm_rotation_callback(angle)
                    time.sleep(0.5)  # Allow rotation to stabilize
                except Exception as e:
                    logger.warning(f"Failed to rotate PPM to {angle} deg: {e}")

            # Set initial exposure
            self.jai_props.set_channel_exposures(
                red=exposure, green=exposure, blue=exposure
            )

            # Determine output path for this angle
            angle_output = None
            if output_path is not None:
                angle_output = output_path / name

            # Run calibration at this angle (no further rotation)
            result = self.calibrate(
                config=config,
                output_path=angle_output,
                ppm_rotation_callback=None,  # Already at target angle
                defocus_callback=None,
            )
            result.wb_method = "manual_ppm"
            results[name] = result

            logger.info(
                f"PPM WB '{name}' complete (target={angle_target}): "
                f"R={result.exposures_ms['red']:.2f}ms, "
                f"G={result.exposures_ms['green']:.2f}ms, "
                f"B={result.exposures_ms['blue']:.2f}ms, "
                f"converged={result.converged}"
            )

        # Summary
        all_converged = all(r.converged for r in results.values())
        logger.info(f"PPM white balance complete: all_converged={all_converged}")

        return results
