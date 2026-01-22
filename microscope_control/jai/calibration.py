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
    """Results from white balance calibration."""

    # Per-channel exposure times in milliseconds
    exposures_ms: Dict[str, float]

    # Per-channel gain multipliers (1.0 = no gain adjustment)
    gains: Dict[str, float]

    # Black level offsets per channel (for dark frame subtraction)
    black_levels: Dict[str, float]

    # Whether calibration converged successfully
    converged: bool

    # Number of iterations to converge
    iterations: int

    # Final channel means after calibration
    final_means: Dict[str, float]

    # Target value that was used
    target_value: float


@dataclass
class CalibrationConfig:
    """Configuration for white balance calibration."""

    # Target mean value for all channels (0-255 for 8-bit, scaled for 16-bit)
    target_value: float = 180.0

    # Acceptable deviation from target (channels within tolerance are considered balanced)
    # Default of 2.0 achieves within 2 intensity levels precision
    tolerance: float = 2.0

    # Maximum iterations before giving up
    max_iterations: int = 30

    # Minimum exposure time in milliseconds
    min_exposure_ms: float = 0.1

    # Maximum exposure time in milliseconds
    # Per-channel exposure limit depends on frame rate (lower frame rate = longer exposure)
    # At min frame rate (0.125 Hz), theoretical max is ~7900ms
    # Keep reasonable for calibration speed; gain compensation handles bright scenes
    max_exposure_ms: float = 200.0

    # Exposure ratio threshold before applying gain compensation
    # If brightest_channel_exposure / darkest_channel_exposure > this, use gain
    gain_threshold_ratio: float = 2.0

    # Maximum analog gain in dB (JAI supports 0-36.13 dB, but keep low to minimize noise)
    # 3 dB = 1.41x linear gain (sqrt(2))
    max_analog_gain_db: float = 3.0

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

    # Defocus offset in micrometers (for PPM calibration on blank slide area)
    defocus_offset_um: Optional[float] = None

    # Bit depth of camera (for target value scaling)
    bit_depth: int = 8


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
        Run white balance calibration.

        This should be run with a neutral gray or white target in the field of view.
        The calibrator will iteratively adjust per-channel exposures until all
        channels produce similar mean values.

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

        self._validate_camera()
        self._convergence_log = ConvergenceLog()

        logger.info("Starting JAI white balance calibration")
        logger.info(f"Target value: {config.target_value}, tolerance: {config.tolerance}")

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
                # In a full implementation, this would prompt the user
                # self._black_levels = self.calibrate_black_level(config.dark_frame_count)

            # Step 2: Setup for calibration
            # Set PPM to 90 degrees (max intensity) if callback provided
            if ppm_rotation_callback is not None:
                try:
                    original_ppm_angle = self.hardware.get_psg_ticks()
                    ppm_rotation_callback(90.0)
                    logger.info("Set PPM rotation to 90 degrees for calibration")
                    time.sleep(0.5)  # Allow rotation to stabilize
                except Exception as e:
                    logger.warning(f"Failed to set PPM rotation: {e}")

            # Apply defocus if callback provided and offset configured
            if defocus_callback is not None and config.defocus_offset_um is not None:
                try:
                    original_z, restore_z_callback = defocus_callback(config.defocus_offset_um)
                    logger.info(f"Defocused by {config.defocus_offset_um}um for calibration")
                except Exception as e:
                    logger.warning(f"Failed to apply defocus: {e}")

            # Step 3: Enable individual exposure mode
            self.jai_props.enable_individual_exposure()

            # Step 4: Initial capture and exposure estimation
            means = self._capture_and_analyze()
            logger.info(f"Initial channel means: R={means['red']:.1f}, G={means['green']:.1f}, B={means['blue']:.1f}")

            # Get current exposures or use defaults
            try:
                exposures = self.jai_props.get_channel_exposures()
            except Exception:
                # Default to 50ms if can't read current
                exposures = {"red": 50.0, "green": 50.0, "blue": 50.0}
                self.jai_props.set_channel_exposures(**exposures)

            gains = {"red": 1.0, "green": 1.0, "blue": 1.0}

            # Initial exposure estimation
            for channel in ["red", "green", "blue"]:
                if means[channel] > 0:
                    estimated = exposures[channel] * (target / means[channel])
                    exposures[channel] = self._clamp_exposure(estimated, config)

            # Log initial state
            converged_flags = self._check_convergence(means, target, config.tolerance)
            self._convergence_log.add_iteration(
                0, means, exposures, gains, converged_flags, "Initial capture"
            )

            # Step 5: Iterative refinement
            # Define fine-tuning threshold (switch to finer adjustments when within 2x tolerance)
            fine_tune_threshold = config.tolerance * 2

            # Track if we converged in the loop (to avoid re-capture noise in final validation)
            converged_means = None

            for iteration in range(1, config.max_iterations + 1):
                # Apply current exposures
                self.jai_props.set_channel_exposures(**exposures, auto_enable=False)
                time.sleep(0.1)  # Allow settings to take effect

                # Capture and analyze
                means = self._capture_and_analyze()

                # Check convergence (strict: within tolerance of target)
                converged_flags = self._check_convergence(means, target, config.tolerance)
                all_converged = all(converged_flags.values())

                # Calculate max deviation for logging
                deviations = {ch: abs(means[ch] - target) for ch in ["red", "green", "blue"]}
                max_deviation = max(deviations.values())

                # Log this iteration
                notes = "Converged" if all_converged else f"max_dev={max_deviation:.1f}"
                self._convergence_log.add_iteration(
                    iteration, means, exposures, gains, converged_flags, notes
                )

                # Log with precision info
                logger.debug(
                    f"Iter {iteration}: R={means['red']:.1f} (exp={exposures['red']:.2f}ms), "
                    f"G={means['green']:.1f} (exp={exposures['green']:.2f}ms), "
                    f"B={means['blue']:.1f} (exp={exposures['blue']:.2f}ms) "
                    f"| max_dev={max_deviation:.1f} (target +/-{config.tolerance})"
                )

                if all_converged:
                    logger.info(
                        f"Calibration converged after {iteration} iterations "
                        f"(all channels within {config.tolerance} of target {target:.0f})"
                    )
                    # Store the converged means - don't re-capture for final validation
                    # as noise could push values slightly out of tolerance
                    converged_means = means.copy()
                    break

                # Calculate adjustments with adaptive damping
                # Use finer damping when close to target for better precision
                for channel in ["red", "green", "blue"]:
                    if not converged_flags[channel] and means[channel] > 0:
                        deviation = abs(means[channel] - target)
                        ratio = target / means[channel]

                        # Use fine damping when close to target
                        if deviation <= fine_tune_threshold:
                            damping = config.fine_damping_factor
                        else:
                            damping = config.damping_factor

                        # Apply damping to prevent oscillation
                        damped_ratio = 1.0 + (ratio - 1.0) * damping
                        new_exposure = exposures[channel] * damped_ratio
                        exposures[channel] = self._clamp_exposure(new_exposure, config)

                # Check if we need gain compensation (only check periodically, not every iteration)
                if iteration % 5 == 0 or iteration == 1:
                    exposures, gains = self._check_gain_compensation(
                        exposures, gains, config
                    )

            # Step 6: Final gain compensation check
            # This ensures gain compensation is applied even if we converged on a
            # non-check iteration (2, 3, 4, 6, 7, 8, 9, etc.)
            # IMPORTANT: Only apply if no gains were already applied during the loop.
            # If gains were applied mid-loop, the subsequent iterations converged WITH
            # those gains active, so re-compressing would double-compress the exposures.
            if all(abs(g - 1.0) < 0.001 for g in gains.values()):
                exposures, gains = self._check_gain_compensation(exposures, gains, config)

            # Step 7: Final validation
            # If we converged in the loop, use those means (avoid re-capture noise)
            # If we didn't converge, capture again to see final state
            if converged_means is not None:
                final_means = converged_means
                all_converged = True
            else:
                final_means = self._capture_and_analyze()
                final_converged = self._check_convergence(final_means, target, config.tolerance)
                all_converged = all(final_converged.values())

            # Calculate final deviations
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

            # Build result
            result = WhiteBalanceResult(
                exposures_ms=exposures,
                gains=gains,
                black_levels=self._black_levels.copy(),
                converged=all_converged,
                iterations=iteration if all_converged else config.max_iterations,
                final_means=final_means,
                target_value=config.target_value,
            )

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

    def _check_gain_compensation(
        self,
        exposures: Dict[str, float],
        gains: Dict[str, float],
        config: CalibrationConfig,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Check if gain compensation is needed and apply if necessary.

        Strategy:
        1. Only use gain when exposure ratio between channels exceeds threshold (2x)
        2. Limit analog gain to max_analog_gain_db (default 3 dB = 1.41x linear)
        3. Avoid digital gain unless absolutely necessary (adds more noise)
        4. If gain cannot fully compensate, allow some exposure imbalance

        Returns:
            Updated (exposures, gains) tuple
        """
        exp_values = list(exposures.values())
        max_exp = max(exp_values)
        min_exp = min(exp_values)

        if min_exp <= 0:
            return exposures, gains

        ratio = max_exp / min_exp
        if ratio < config.gain_threshold_ratio:
            logger.debug(
                f"Exposure ratio {ratio:.2f} below threshold {config.gain_threshold_ratio}, "
                f"no gain compensation needed (exp: R={exposures['red']:.2f}, "
                f"G={exposures['green']:.2f}, B={exposures['blue']:.2f})"
            )
            return exposures, gains

        # Convert max gain from dB to linear multiplier
        max_linear_gain = db_to_linear(config.max_analog_gain_db)

        logger.info(
            f"Exposure ratio {ratio:.2f} exceeds threshold {config.gain_threshold_ratio}. "
            f"Applying gain compensation (max {config.max_analog_gain_db} dB = {max_linear_gain:.3f}x linear)."
        )

        # Enable individual gain mode
        try:
            self.jai_props.enable_individual_gain()
        except Exception as e:
            logger.warning(f"Failed to enable individual gain: {e}")
            return exposures, gains

        # Strategy: reduce the longer exposures and compensate with gain
        # Target exposure is min_exp * threshold, but we're limited by max gain
        target_max_exp = min_exp * config.gain_threshold_ratio
        new_exposures = exposures.copy()
        new_gains = gains.copy()

        for channel, exp in exposures.items():
            if exp > target_max_exp:
                # Calculate required gain
                required_gain = exp / target_max_exp

                if required_gain <= max_linear_gain:
                    # Full compensation possible within gain limit
                    new_gains[channel] = required_gain
                    new_exposures[channel] = target_max_exp
                    logger.info(
                        f"  {channel}: exposure {exp:.2f}ms -> {target_max_exp:.2f}ms, "
                        f"gain 1.0 -> {required_gain:.3f}x ({linear_to_db(required_gain):.1f} dB)"
                    )
                else:
                    # Gain-limited: apply max gain and keep some exposure imbalance
                    new_gains[channel] = max_linear_gain
                    # New exposure = exp / max_linear_gain (partial compensation)
                    new_exposure = exp / max_linear_gain
                    new_exposures[channel] = new_exposure
                    logger.warning(
                        f"  {channel}: gain-limited! exposure {exp:.2f}ms -> {new_exposure:.2f}ms, "
                        f"gain capped at {max_linear_gain:.3f}x ({config.max_analog_gain_db} dB)"
                    )

        # Apply analog gains
        try:
            self.jai_props.set_analog_gains(
                red=new_gains["red"],
                green=new_gains["green"],
                blue=new_gains["blue"],
                auto_enable=False,
            )
            logger.info(
                f"Applied analog gains: R={new_gains['red']:.3f}x, "
                f"G={new_gains['green']:.3f}x, B={new_gains['blue']:.3f}x"
            )
        except Exception as e:
            logger.warning(f"Failed to set analog gains: {e}")
            return exposures, gains

        # Note: We intentionally avoid digital gain per config.avoid_digital_gain
        # Digital gain range is very narrow (0.9-1.1x) and adds more noise than analog
        if not config.avoid_digital_gain:
            logger.debug("Digital gain is enabled but not currently used in this implementation")

        return new_exposures, new_gains

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

        data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "version": "1.0",
                "description": "QPSC White Balance Settings - Per-channel exposure and gain for color balance",
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
            "per_channel_gains": {
                "analog": {
                    "red": round(result.gains["red"], 3),
                    "green": round(result.gains["green"], 3),
                    "blue": round(result.gains["blue"], 3),
                },
                "gain_mode": "individual" if any(g != 1.0 for g in result.gains.values()) else "unified",
            },
            "black_levels": result.black_levels,
            "combined_settings": {
                "ExposureIsIndividual": "On",
                "Exposure_Red": round(result.exposures_ms["red"], 2),
                "Exposure_Green": round(result.exposures_ms["green"], 2),
                "Exposure_Blue": round(result.exposures_ms["blue"], 2),
            },
            "notes": [
                "Apply these settings before acquisition for consistent color balance",
                "If lighting conditions change significantly, recalibrate",
                "Settings are specific to this detector/modality/objective combination",
            ],
        }

        # Add gain settings to combined if gains were adjusted
        if any(g != 1.0 for g in result.gains.values()):
            data["combined_settings"]["GainIsIndividual"] = "On"
            data["combined_settings"]["Gain_AnalogRed"] = round(result.gains["red"], 3)
            data["combined_settings"]["Gain_AnalogGreen"] = round(result.gains["green"], 3)
            data["combined_settings"]["Gain_AnalogBlue"] = round(result.gains["blue"], 3)

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

        Saves to two locations:
        1. white_balance_calibration section (for reference/audit trail)
        2. imaging_profiles.{modality}.{objective}.{detector}.exposures_ms (for actual use)

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

            # Ensure white_balance_calibration section exists
            if "white_balance_calibration" not in ip_data:
                ip_data["white_balance_calibration"] = {}

            wb_cal = ip_data["white_balance_calibration"]

            # Structure for JAI camera calibration
            if calibration_type == "simple":
                # Simple WB - same settings for all angles
                wb_cal["jai_simple"] = {
                    "calibrated": datetime.now().isoformat(),
                    "converged": result.converged,
                    "target": result.target_value,
                    "exposures_ms": {
                        "r": round(result.exposures_ms["red"], 2),
                        "g": round(result.exposures_ms["green"], 2),
                        "b": round(result.exposures_ms["blue"], 2),
                    },
                    "gains": {
                        "r": round(result.gains["red"], 3),
                        "g": round(result.gains["green"], 3),
                        "b": round(result.gains["blue"], 3),
                    },
                    "final_means": {
                        "r": round(result.final_means["red"], 1),
                        "g": round(result.final_means["green"], 1),
                        "b": round(result.final_means["blue"], 1),
                    },
                }
            elif calibration_type == "ppm" and angle_name:
                # PPM WB - per-angle settings
                if "jai_ppm" not in wb_cal:
                    wb_cal["jai_ppm"] = {
                        "calibrated": datetime.now().isoformat(),
                        "angles": {},
                    }
                wb_cal["jai_ppm"]["calibrated"] = datetime.now().isoformat()
                wb_cal["jai_ppm"]["angles"][angle_name] = {
                    "converged": result.converged,
                    "exposures_ms": {
                        "r": round(result.exposures_ms["red"], 2),
                        "g": round(result.exposures_ms["green"], 2),
                        "b": round(result.exposures_ms["blue"], 2),
                    },
                    "gains": {
                        "r": round(result.gains["red"], 3),
                        "g": round(result.gains["green"], 3),
                        "b": round(result.gains["blue"], 3),
                    },
                }

            # Also update imaging_profiles section if modality/objective/detector provided
            # This is where BackgroundCollectionController reads the values from
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

                # Also update gains if present
                if "gains" not in profile:
                    profile["gains"] = {}
                profile["gains"][angle_name] = {
                    "r": round(result.gains["red"], 3),
                    "g": round(result.gains["green"], 3),
                    "b": round(result.gains["blue"], 3),
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
        gains = data.get("per_channel_gains", {}).get("analog", {})
        cal_params = data.get("calibration_parameters", {})

        return WhiteBalanceResult(
            exposures_ms={
                "red": exposures.get("red", 50.0),
                "green": exposures.get("green", 50.0),
                "blue": exposures.get("blue", 50.0),
            },
            gains={
                "red": gains.get("red", 1.0),
                "green": gains.get("green", 1.0),
                "blue": gains.get("blue", 1.0),
            },
            black_levels=data.get("black_levels", {"red": 0, "green": 0, "blue": 0}),
            converged=cal_params.get("converged", True),
            iterations=cal_params.get("iterations_used", 0),
            final_means=cal_params.get("achieved_intensities", {}),
            target_value=cal_params.get("target_intensity_8bit", 180.0),
        )

    def apply_calibration(self, result: WhiteBalanceResult) -> None:
        """
        Apply calibration results to the camera.

        Args:
            result: WhiteBalanceResult to apply
        """
        self.jai_props.set_channel_exposures(
            red=result.exposures_ms["red"],
            green=result.exposures_ms["green"],
            blue=result.exposures_ms["blue"],
        )

        if any(g != 1.0 for g in result.gains.values()):
            self.jai_props.set_analog_gains(
                red=result.gains["red"],
                green=result.gains["green"],
                blue=result.gains["blue"],
            )

        logger.info("Applied white balance calibration to camera")

    def calibrate_simple(
        self,
        initial_exposure_ms: float,
        target: float = 180.0,
        tolerance: float = 5.0,
        output_path: Optional[Path] = None,
        max_gain_db: Optional[float] = None,
        gain_threshold_ratio: Optional[float] = None,
        max_iterations: Optional[int] = None,
        calibrate_black_level: Optional[bool] = None,
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
            max_gain_db: Optional max analog gain in dB (default 3.0)
            gain_threshold_ratio: Optional exposure ratio before using gain (default 2.0)
            max_iterations: Optional max calibration iterations (default 30)
            calibrate_black_level: Optional whether to calibrate black level (default True)

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

        # Build config with optional advanced settings
        config_kwargs = {
            "target_value": target,
            "tolerance": tolerance,
        }
        if max_gain_db is not None:
            config_kwargs["max_analog_gain_db"] = max_gain_db
        if gain_threshold_ratio is not None:
            config_kwargs["gain_threshold_ratio"] = gain_threshold_ratio
        if max_iterations is not None:
            config_kwargs["max_iterations"] = max_iterations
        if calibrate_black_level is not None:
            config_kwargs["calibrate_black_level"] = calibrate_black_level

        config = CalibrationConfig(**config_kwargs)

        logger.info(
            f"Simple WB config: target={target}, tolerance={tolerance}, "
            f"max_gain_db={config.max_analog_gain_db}, gain_threshold={config.gain_threshold_ratio}, "
            f"max_iter={config.max_iterations}, calibrate_bl={config.calibrate_black_level}"
        )

        # Run calibration without PPM rotation
        return self.calibrate(
            config=config,
            output_path=output_path,
            ppm_rotation_callback=None,
            defocus_callback=None,
        )

    def calibrate_ppm(
        self,
        angle_exposures: Dict[str, Tuple[float, float]],
        target: float = 180.0,
        tolerance: float = 5.0,
        output_path: Optional[Path] = None,
        ppm_rotation_callback: Optional[Callable[[float], None]] = None,
        per_angle_targets: Optional[Dict[str, float]] = None,
        max_gain_db: Optional[float] = None,
        gain_threshold_ratio: Optional[float] = None,
        max_iterations: Optional[int] = None,
        calibrate_black_level: Optional[bool] = None,
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
            f"PPM WB advanced settings: max_gain_db={max_gain_db}, gain_threshold={gain_threshold_ratio}, "
            f"max_iter={max_iterations}, calibrate_bl={calibrate_black_level}"
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
            if max_gain_db is not None:
                config_kwargs["max_analog_gain_db"] = max_gain_db
            if gain_threshold_ratio is not None:
                config_kwargs["gain_threshold_ratio"] = gain_threshold_ratio
            if max_iterations is not None:
                config_kwargs["max_iterations"] = max_iterations
            if calibrate_black_level is not None:
                config_kwargs["calibrate_black_level"] = calibrate_black_level

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
