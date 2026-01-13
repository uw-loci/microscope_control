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
    tolerance: float = 5.0

    # Maximum iterations before giving up
    max_iterations: int = 20

    # Minimum exposure time in milliseconds
    min_exposure_ms: float = 0.1

    # Maximum exposure time in milliseconds
    # Note: Actual limit depends on frame rate. At min frame rate (0.125 Hz),
    # max exposure is ~7900ms. Keep this reasonable for calibration speed.
    max_exposure_ms: float = 200.0

    # Exposure ratio threshold before applying gain compensation
    # If brightest_channel_exposure / darkest_channel_exposure > this, use gain
    gain_threshold_ratio: float = 2.0

    # Damping factor for exposure adjustments (prevents oscillation)
    damping_factor: float = 0.7

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
            for iteration in range(1, config.max_iterations + 1):
                # Apply current exposures
                self.jai_props.set_channel_exposures(**exposures, auto_enable=False)
                time.sleep(0.1)  # Allow settings to take effect

                # Capture and analyze
                means = self._capture_and_analyze()

                # Check convergence
                converged_flags = self._check_convergence(means, target, config.tolerance)
                all_converged = all(converged_flags.values())

                # Log this iteration
                self._convergence_log.add_iteration(
                    iteration, means, exposures, gains, converged_flags,
                    "Converged" if all_converged else "Adjusting"
                )

                logger.debug(
                    f"Iteration {iteration}: R={means['red']:.1f} ({exposures['red']:.1f}ms), "
                    f"G={means['green']:.1f} ({exposures['green']:.1f}ms), "
                    f"B={means['blue']:.1f} ({exposures['blue']:.1f}ms)"
                )

                if all_converged:
                    logger.info(f"Calibration converged after {iteration} iterations")
                    break

                # Calculate adjustments with damping
                for channel in ["red", "green", "blue"]:
                    if not converged_flags[channel] and means[channel] > 0:
                        ratio = target / means[channel]
                        # Apply damping to prevent oscillation
                        damped_ratio = 1.0 + (ratio - 1.0) * config.damping_factor
                        new_exposure = exposures[channel] * damped_ratio
                        exposures[channel] = self._clamp_exposure(new_exposure, config)

                # Check if we need gain compensation
                exposures, gains = self._check_gain_compensation(
                    exposures, gains, config
                )

            # Step 6: Final validation
            final_means = self._capture_and_analyze()
            final_converged = self._check_convergence(final_means, target, config.tolerance)
            all_converged = all(final_converged.values())

            if not all_converged:
                logger.warning(
                    "Calibration did not fully converge. "
                    f"Final means: R={final_means['red']:.1f}, G={final_means['green']:.1f}, "
                    f"B={final_means['blue']:.1f} (target: {target:.1f})"
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

        If the ratio between max and min exposure exceeds the threshold,
        use gain to compensate for the longer exposures.

        Returns:
            Updated (exposures, gains) tuple
        """
        exp_values = list(exposures.values())
        max_exp = max(exp_values)
        min_exp = min(exp_values)

        if min_exp <= 0:
            return exposures, gains

        ratio = max_exp / min_exp
        if ratio <= config.gain_threshold_ratio:
            return exposures, gains

        logger.info(
            f"Exposure ratio {ratio:.2f} exceeds threshold {config.gain_threshold_ratio}. "
            "Applying gain compensation."
        )

        # Enable individual gain mode
        try:
            self.jai_props.enable_individual_gain()
        except Exception as e:
            logger.warning(f"Failed to enable individual gain: {e}")
            return exposures, gains

        # Target: bring all exposures within threshold of each other
        target_max_exp = min_exp * config.gain_threshold_ratio
        new_exposures = exposures.copy()
        new_gains = gains.copy()

        for channel, exp in exposures.items():
            if exp > target_max_exp:
                gain_factor = exp / target_max_exp
                new_gains[channel] = gain_factor
                new_exposures[channel] = target_max_exp
                logger.info(
                    f"  {channel}: exposure {exp:.1f}ms -> {target_max_exp:.1f}ms, "
                    f"gain 1.0 -> {gain_factor:.2f}"
                )

        # Apply gains
        try:
            self.jai_props.set_analog_gains(
                red=new_gains["red"],
                green=new_gains["green"],
                blue=new_gains["blue"],
                auto_enable=False,
            )
        except Exception as e:
            logger.warning(f"Failed to set analog gains: {e}")
            return exposures, gains

        return new_exposures, new_gains

    def _save_diagnostics(
        self,
        result: WhiteBalanceResult,
        config: CalibrationConfig,
        output_path: Path,
    ) -> None:
        """Save diagnostic output files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save convergence log
        self._convergence_log.save_csv(output_path / "convergence_log.csv")

        # Save settings YAML
        self.save_calibration(result, output_path / "white_balance_settings.yml")

        # Generate histogram plot if matplotlib is available
        try:
            self._save_histogram_plot(result, config, output_path / "intensity_histograms.png")
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
            with open(output_path / "black_level_calibration.yml", "w") as f:
                yaml.dump(black_level_data, f, default_flow_style=False)

    def _save_histogram_plot(
        self,
        result: WhiteBalanceResult,
        config: CalibrationConfig,
        output_path: Path,
    ) -> None:
        """Generate histogram visualization."""
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
