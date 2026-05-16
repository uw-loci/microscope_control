"""
Autofocus Testing and Diagnostic Module
=======================================

Provides comprehensive autofocus testing with:
- Diagnostic plots showing focus curves and interpolation
- Detailed logging of focus scores at each Z position
- Comparison of different focus metrics
- Analysis of raw vs interpolated peak positions
- Parameter sensitivity testing

This module is designed to be called from the socket server to test
autofocus settings interactively during microscope setup.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.interpolate
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.hardware.base import Position

# Use validate_focus_peak from AutofocusUtils
validate_focus_peak = AutofocusUtils.validate_focus_peak


def test_standard_autofocus_at_current_position(
    hardware,
    config_manager,
    yaml_file_path: str,
    output_folder_path: str,
    objective: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Test STANDARD autofocus at current microscope position.
    Calls hardware.autofocus() with settings from config file.

    This performs a symmetric sweep around current position using fixed n_steps.

    Args:
        hardware: PycromanagerHardware instance
        config_manager: ConfigManager instance
        yaml_file_path: Path to microscope config YAML
        output_folder_path: Where to save diagnostic plots and data
        objective: Objective identifier
        logger: Optional logger instance

    Returns:
        Dict containing test results and plot path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=== STANDARD AUTOFOCUS TEST STARTED ===")
    logger.info(f"  Objective: {objective}")
    logger.info(f"  Config file: {yaml_file_path}")

    # Create output directory
    output_path = Path(output_folder_path)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "initial_z": None,
        "final_z": None,
        "z_shift": None,
        "raw_best_z": None,
        "interp_best_z": None,
        "raw_scores": [],
        "plot_path": None,
        "message": "",
        "test_type": "standard",
        "peak_validation": None,  # Will contain validation results
    }

    try:
        # Get current position
        initial_pos = hardware.get_current_position()
        result["initial_z"] = initial_pos.z
        logger.info(
            f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}"
        )

        # Load autofocus settings
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        logger.info("  Autofocus settings:")
        logger.info(f"    n_steps: {af_settings['n_steps']}")
        logger.info(f"    search_range: {af_settings['search_range']} um (centered on current Z)")
        logger.info(f"    interp_strength: {af_settings['interp_strength']}")
        logger.info(f"    interp_kind: {af_settings['interp_kind']}")
        logger.info(f"    score_metric: {af_settings['score_metric_name']}")

        # Call the ACTUAL hardware.autofocus() method
        logger.info("  Calling hardware.autofocus() with config settings...")

        final_z = hardware.autofocus(
            n_steps=af_settings["n_steps"],
            search_range=af_settings["search_range"],
            interp_strength=af_settings["interp_strength"],
            interp_kind=af_settings["interp_kind"],
            score_metric=af_settings["score_metric"],
            pop_a_plot=False,
            move_stage_to_estimate=True,
            raise_on_invalid_peak=False,  # Always generate diagnostics for test
        )

        result["final_z"] = final_z
        result["z_shift"] = final_z - initial_pos.z

        logger.info("  Standard autofocus completed:")
        logger.info(f"    Final Z: {final_z:.2f} um")
        logger.info(f"    Z shift: {result['z_shift']:.2f} um")

        # Generate diagnostic plot by doing a post-hoc scan
        # This shows what the focus curve looks like with current settings
        logger.info("  Generating diagnostic plot...")
        try:
            plot_path, validation = _generate_diagnostic_scan_plot(
                hardware, final_z, af_settings, output_path, logger
            )
            result["plot_path"] = str(plot_path)
            result["peak_validation"] = validation
            logger.info(f"  Diagnostic plot saved: {plot_path}")

            # Check if autofocus found a valid peak
            if not validation["is_valid"]:
                logger.error("  *** AUTOFOCUS FAILED: Invalid focus peak detected ***")
                logger.error(f"  {validation['message']}")
                logger.error(
                    "  RECOMMENDATION: Check focus manually or increase autofocus search range"
                )
                result["success"] = False
                result["message"] = (
                    f"Autofocus failed: {validation['message']}. Check focus manually or increase search range."
                )
            else:
                result["message"] = (
                    f"Standard autofocus completed. Z shift: {result['z_shift']:.2f} um."
                )
                result["success"] = True

        except Exception as e:
            logger.warning(f"Failed to generate diagnostic plot: {e}")
            result["plot_path"] = "None"
            # Still mark as success if autofocus itself worked, just plotting failed
            result["message"] = (
                f"Standard autofocus completed. Z shift: {result['z_shift']:.2f} um."
            )
            result["success"] = True

        logger.info("=== STANDARD AUTOFOCUS TEST COMPLETED ===")

    except Exception as e:
        logger.error(f"Standard autofocus test failed: {e}", exc_info=True)
        result["message"] = f"Standard autofocus test failed: {str(e)}"

        # Try to return to initial position
        if result["initial_z"] is not None:
            try:
                hardware.move_to_position(
                    Position(initial_pos.x, initial_pos.y, result["initial_z"])
                )
                logger.info("Returned to initial Z position after error")
            except Exception as e:
                logger.warning("Failed to return to initial position: %s", e)

    return result


def test_adaptive_autofocus_at_current_position(
    hardware,
    config_manager,
    yaml_file_path: str,
    output_folder_path: str,
    objective: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Test sweep drift check at current microscope position.
    Calls hardware.autofocus_sweep_drift_check() with settings from config file.

    This performs a quick Z sweep around current position to detect and correct focus drift.

    Note: Function name kept as test_adaptive_autofocus_at_current_position for
    socket protocol compatibility (TESTADAF command).

    Args:
        hardware: PycromanagerHardware instance
        config_manager: ConfigManager instance
        yaml_file_path: Path to microscope config YAML
        output_folder_path: Where to save diagnostic plots and data
        objective: Objective identifier
        logger: Optional logger instance

    Returns:
        Dict containing test results and plot path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=== SWEEP DRIFT CHECK TEST STARTED ===")
    logger.info(f"  Objective: {objective}")
    logger.info(f"  Config file: {yaml_file_path}")

    # Create output directory
    output_path = Path(output_folder_path)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "initial_z": None,
        "final_z": None,
        "z_shift": None,
        "plot_path": None,
        "message": "",
        "test_type": "sweep",
    }

    try:
        # Get current position
        initial_pos = hardware.get_current_position()
        result["initial_z"] = initial_pos.z
        logger.info(
            f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}"
        )

        # Load autofocus settings
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        # Get sweep parameters (with legacy fallback)
        sweep_range = af_settings.get("sweep_range_um", 10.0)
        sweep_n_steps = af_settings.get("sweep_n_steps", 6)
        score_metric = af_settings.get("score_metric_name", "normalized_variance")

        # Legacy support: old adaptive_initial_step_um -> sweep_range_um
        if "sweep_range_um" not in af_settings and "adaptive_initial_step_um" in af_settings:
            sweep_range = af_settings["adaptive_initial_step_um"] * 2

        logger.info("  Sweep drift check settings:")
        logger.info(f"    sweep_range_um: {sweep_range} um")
        logger.info(f"    sweep_n_steps: {sweep_n_steps}")
        logger.info(f"    score_metric: {score_metric}")

        # Call the sweep drift check. Pass a samples_out list so we can
        # write the (z, metric) curve to disk for empirical analysis --
        # production callers pass None and pay no allocation cost.
        logger.info("  Calling hardware.autofocus_sweep_drift_check()...")

        sweep_samples: list = []
        final_z = hardware.autofocus_sweep_drift_check(
            range_um=sweep_range,
            n_steps=sweep_n_steps,
            score_metric=score_metric,
            samples_out=sweep_samples,
        )

        result["final_z"] = final_z
        result["z_shift"] = final_z - initial_pos.z

        logger.info("  Sweep drift check completed:")
        logger.info(f"    Final Z: {final_z:.2f} um")
        logger.info(f"    Z shift: {result['z_shift']:.2f} um")
        logger.info(f"    Samples captured: {len(sweep_samples)}")

        # Write diagnostic CSV + PNG. Mirrors the format used by the
        # Standard test in _write_test_results so analysis scripts can
        # ingest both with the same parser.
        if sweep_samples:
            try:
                _write_sweep_test_results(
                    output_path=output_path,
                    samples=sweep_samples,
                    initial_z=initial_pos.z,
                    final_z=final_z,
                    sweep_range=sweep_range,
                    sweep_n_steps=sweep_n_steps,
                    score_metric=score_metric,
                    logger=logger,
                )
                # Surface the plot path in the result so the editor
                # status line can show it (matches Standard test
                # behavior).
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result["plot_path"] = str(output_path / f"autofocus_test_sweep_{timestamp}.png")
            except Exception as e:
                logger.warning(f"Failed to write sweep test diagnostic files: {e}")
                result["plot_path"] = None
        else:
            result["plot_path"] = None
            logger.warning("Sweep produced no samples; no diagnostic files written")

        result["message"] = f"Sweep drift check completed. Z shift: {result['z_shift']:.2f} um."
        result["success"] = True

        logger.info("=== ADAPTIVE AUTOFOCUS TEST COMPLETED ===")

    except Exception as e:
        logger.error(f"Adaptive autofocus test failed: {e}", exc_info=True)
        result["message"] = f"Adaptive autofocus test failed: {str(e)}"

        # Try to return to initial position
        if result["initial_z"] is not None:
            try:
                hardware.move_to_position(
                    Position(initial_pos.x, initial_pos.y, result["initial_z"])
                )
                logger.info("Returned to initial Z position after error")
            except Exception as e:
                logger.warning("Failed to return to initial position: %s", e)

    return result


def _write_sweep_test_results(
    output_path: Path,
    samples: list,
    initial_z: float,
    final_z: float,
    sweep_range: float,
    sweep_n_steps: int,
    score_metric: str,
    logger: logging.Logger,
) -> None:
    """Write a CSV + PNG for one sweep drift check test run.

    Mirrors the Standard test's filename pattern
    (``autofocus_test_<type>_<timestamp>.{csv,png}``) so analysis
    scripts can ingest both with the same parser. The CSV has commented
    header rows with run parameters followed by per-step data:
    ``Window, Z_Position_um, Focus_Score``.

    Args:
        output_path: Directory to write into (must exist).
        samples: List of ``(window_idx, z, score)`` tuples produced by
            ``autofocus_sweep_drift_check`` when ``samples_out`` is set.
            window_idx is 0 for the initial sweep, 1..N for edge
            retries.
        initial_z, final_z: Stage Z before and after the sweep.
        sweep_range, sweep_n_steps, score_metric: Sweep parameters.
        logger: Logger.
    """
    import csv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"autofocus_test_sweep_{timestamp}.csv"
    png_path = output_path / f"autofocus_test_sweep_{timestamp}.png"

    z_shift = final_z - initial_z
    z_arr = np.array([s[1] for s in samples])
    score_arr = np.array([s[2] for s in samples])
    window_arr = np.array([s[0] for s in samples])
    peak_idx = int(np.argmax(score_arr))
    peak_z = float(z_arr[peak_idx])

    try:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["# Sweep Drift Check Diagnostic Data"])
            writer.writerow(["# Timestamp", timestamp])
            writer.writerow(["# Test Type", "sweep"])
            writer.writerow(["# Metric", score_metric])
            writer.writerow(["# Sweep Range (um)", f"{sweep_range:.2f}"])
            writer.writerow(["# Sweep N Steps", sweep_n_steps])
            writer.writerow(["# Initial Z", f"{initial_z:.3f}"])
            writer.writerow(["# Final Z (committed)", f"{final_z:.3f}"])
            writer.writerow(["# Z Shift", f"{z_shift:+.3f}"])
            writer.writerow(["# Scan Peak Z", f"{peak_z:.3f}"])
            writer.writerow(["# Total Samples", len(samples)])
            writer.writerow(["# Windows", int(window_arr.max()) + 1])
            writer.writerow(["#"])
            writer.writerow(["Window", "Z_Position_um", "Focus_Score"])
            for w, z, sc in samples:
                writer.writerow([w, f"{z:.3f}", f"{sc:.4f}"])
        logger.info(f"  CSV data saved: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save sweep CSV: {e}")
        return

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Color per window so retries are visually distinct from the
        # initial sweep. Plot in order so connecting line shows scan
        # sequence.
        n_windows = int(window_arr.max()) + 1
        colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
        for w in range(n_windows):
            mask = window_arr == w
            label = "Initial sweep" if w == 0 else f"Edge retry {w}"
            ax.plot(
                z_arr[mask],
                score_arr[mask],
                "o-",
                markersize=6,
                linewidth=1.8,
                color=colors[w % len(colors)],
                label=label,
            )
        ax.axvline(
            initial_z,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"Initial Z ({initial_z:.2f} um)",
        )
        ax.axvline(
            final_z,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label=f"Committed Z ({final_z:.2f} um)",
        )

        ax.set_xlabel("Z Position (um)", fontsize=11)
        ax.set_ylabel("Focus Score", fontsize=11)
        ax.set_title(
            f"Sweep Drift Check Test\n"
            f"Metric: {score_metric} | Range: {sweep_range:.1f} um, "
            f"{sweep_n_steps} steps | Z shift: {z_shift:+.2f} um",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        textstr = (
            f"Initial Z: {initial_z:.3f} um\n"
            f"Final Z:   {final_z:.3f} um\n"
            f"Z shift:   {z_shift:+.3f} um\n"
            f"Scan peak: {peak_z:.3f} um\n"
            f"Samples:   {len(samples)} ({n_windows} window"
            f"{'s' if n_windows != 1 else ''})"
        )
        props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.7)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Plot saved: {png_path}")
    except Exception as e:
        logger.warning(f"Failed to generate sweep plot: {e}")


def test_autofocus_validation(
    hardware,
    config_manager,
    yaml_file_path: str,
    objective: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Two-phase autofocus validation test.

    Verifies that autofocus settings work on the user's specific tissue
    before committing to a multi-hour acquisition.

    Phase 1: From current (user-focused) position, run sweep drift check.
              This confirms the sweep can find focus when already focused.
    Phase 2: Move Z to 80% of search_range away from ground truth, then
              run full standard autofocus. This confirms AF can recover
              from significant defocus (as happens between acquisition tiles).

    The stage is always returned to ground truth Z at the end, regardless
    of success or failure.

    Args:
        hardware: PycromanagerHardware instance
        config_manager: ConfigManager instance
        yaml_file_path: Path to microscope config YAML
        objective: Objective identifier (e.g. "20x")
        logger: Optional logger instance

    Returns:
        Dict with validation results for both phases, including:
            success, ground_truth_z, sweep_z, sweep_delta_um,
            defocus_distance_um, recovery_z, recovery_delta_um,
            message, test_type
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=== AUTOFOCUS VALIDATION TEST STARTED ===")
    logger.info(f"  Objective: {objective}")
    logger.info(f"  Config file: {yaml_file_path}")

    # Record ground truth Z (user's manual focus position)
    ground_truth_z = hardware.get_current_position().z
    logger.info(f"  Ground truth Z (manual focus): {ground_truth_z:.2f}")

    # Load autofocus settings
    af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

    result = {
        "success": False,
        "ground_truth_z": f"{ground_truth_z:.2f}",
        "sweep_z": "N/A",
        "sweep_delta_um": "N/A",
        "defocus_distance_um": "N/A",
        "recovery_z": "N/A",
        "recovery_delta_um": "N/A",
        "message": "",
        "test_type": "validation",
    }

    try:
        # === PHASE 1: Sweep drift check from focus ===
        logger.info("--- Phase 1: Sweep drift check from focus ---")

        sweep_range = af_settings.get("sweep_range_um", 10.0)
        sweep_n_steps = af_settings.get("sweep_n_steps", 6)
        score_metric_name = af_settings.get("score_metric_name", "normalized_variance")

        logger.info(f"  sweep_range_um: {sweep_range}")
        logger.info(f"  sweep_n_steps: {sweep_n_steps}")
        logger.info(f"  score_metric: {score_metric_name}")

        sweep_z = hardware.autofocus_sweep_drift_check(
            range_um=sweep_range,
            n_steps=sweep_n_steps,
            score_metric=score_metric_name,
        )
        sweep_delta = abs(sweep_z - ground_truth_z)

        result["sweep_z"] = f"{sweep_z:.2f}"
        result["sweep_delta_um"] = f"{sweep_delta:.2f}"

        logger.info(f"  Sweep result: Z={sweep_z:.2f}, delta={sweep_delta:.2f} um")

        # === PHASE 2: Recovery from defocus ===
        logger.info("--- Phase 2: Full autofocus recovery from defocus ---")

        search_range = af_settings.get("search_range", 50.0)
        defocus_distance = search_range * 0.8

        result["defocus_distance_um"] = f"{defocus_distance:.1f}"

        # Move to defocused position (positive direction from ground truth)
        defocus_z = ground_truth_z + defocus_distance
        hardware.core.set_position(defocus_z)
        hardware.core.wait_for_device(hardware.core.get_focus_device())
        logger.info(
            f"  Moved to defocus position: {defocus_z:.2f} (+{defocus_distance:.1f} um from ground truth)"
        )

        # Run standard autofocus from defocused position
        recovery_result = hardware.autofocus(
            n_steps=af_settings["n_steps"],
            search_range=search_range,
            interp_strength=af_settings.get("interp_strength", 100),
            interp_kind=af_settings.get("interp_kind", "quadratic"),
            score_metric=af_settings["score_metric"],
            pop_a_plot=False,
            move_stage_to_estimate=True,
            raise_on_invalid_peak=False,  # Don't raise -- we want to report the result
        )

        if isinstance(recovery_result, dict):
            # autofocus returns dict on failure (invalid peak)
            recovery_delta_str = "FAILED"
            recovery_z_value = defocus_z
            logger.warning(f"  Recovery FAILED: {recovery_result.get('message', 'unknown')}")
            result["recovery_z"] = f"{recovery_z_value:.2f}"
            result["recovery_delta_um"] = recovery_delta_str
        else:
            recovery_z_value = float(recovery_result)
            recovery_delta = abs(recovery_z_value - ground_truth_z)
            result["recovery_z"] = f"{recovery_z_value:.2f}"
            result["recovery_delta_um"] = f"{recovery_delta:.2f}"
            logger.info(
                f"  Recovery result: Z={recovery_z_value:.2f}, delta={recovery_delta:.2f} um"
            )

        # Mark overall success (both phases completed without exception)
        result["success"] = True
        result["message"] = "Autofocus validation complete"

        logger.info("=== AUTOFOCUS VALIDATION TEST COMPLETED ===")
        logger.info(f"  Phase 1 (sweep drift): delta = {result['sweep_delta_um']} um")
        logger.info(f"  Phase 2 (recovery):    delta = {result['recovery_delta_um']} um")

    except Exception as e:
        logger.error(f"Autofocus validation test failed: {e}", exc_info=True)
        result["message"] = f"Autofocus validation test failed: {str(e)}"

    finally:
        # ALWAYS return to ground truth Z
        try:
            hardware.core.set_position(ground_truth_z)
            hardware.core.wait_for_device(hardware.core.get_focus_device())
            logger.info(f"  Returned to ground truth Z: {ground_truth_z:.2f}")
        except Exception as e2:
            logger.error(f"  Failed to return to ground truth Z: {e2}")

    return result


def _generate_diagnostic_scan_plot(
    hardware, center_z, af_settings, output_path, logger, test_type="standard"
):
    """
    Generate a diagnostic plot by scanning around the center_z position.

    Args:
        hardware: Hardware instance
        center_z: Z position to center the scan around
        af_settings: Autofocus settings dict
        output_path: Path object for output directory
        logger: Logger instance
        test_type: "standard" or "adaptive" for plot labeling

    Returns:
        Tuple of (plot_path, validation_dict):
            - plot_path: Path to generated plot file
            - validation_dict: Peak quality validation results
    """
    from datetime import datetime

    # Do a diagnostic scan centered on the final Z
    scan_range = af_settings["search_range"]
    n_steps = af_settings["n_steps"]
    score_metric = af_settings["score_metric"]

    # Generate Z positions centered on final result
    z_positions = np.linspace(center_z - scan_range / 2, center_z + scan_range / 2, n_steps)

    logger.info(
        f"  Scanning {n_steps} positions from {z_positions[0]:.2f} to {z_positions[-1]:.2f} um"
    )

    scores = []
    for i, z in enumerate(z_positions):
        # Move to position
        new_pos = Position(hardware.get_current_position().x, hardware.get_current_position().y, z)
        hardware.move_to_position(new_pos)

        # Acquire and score
        img, tags = hardware.snap_image()

        # Extract grayscale
        if hardware.core.get_property("Core", "Camera") == "JAICamera":
            img_gray = np.mean(img, 2)
        else:
            green1 = img[0::2, 0::2]
            green2 = img[1::2, 1::2]
            img_gray = ((green1 + green2) / 2.0).astype(np.float32)

        score = score_metric(img_gray)
        if hasattr(score, "ndim") and score.ndim == 2:
            score = np.mean(score)
        scores.append(float(score))

    scores = np.array(scores)

    # VALIDATE FOCUS PEAK QUALITY
    validation = validate_focus_peak(z_positions, scores)
    logger.info(f"  Focus peak validation: {validation['message']}")
    logger.info(f"    Quality score: {validation['quality_score']:.2f}")
    logger.info(f"    Peak prominence: {validation['peak_prominence']:.2f}")
    logger.info(f"    Has ascending trend: {validation['has_ascending']}")
    logger.info(f"    Has descending trend: {validation['has_descending']}")
    logger.info(f"    Symmetry score: {validation['symmetry_score']:.2f}")

    if not validation["is_valid"]:
        logger.warning("  *** AUTOFOCUS PEAK QUALITY CHECK FAILED ***")
        for warning in validation["warnings"]:
            logger.warning(f"    - {warning}")

    # Find peak in the diagnostic scan
    peak_idx = np.argmax(scores)
    peak_z = z_positions[peak_idx]

    # Restore stage to the AUTOFOCUS RESULT (center_z), not the diagnostic
    # scan's argmax. The Test button reports what Standard AF actually
    # does -- silently moving to a different Z based on a follow-up
    # diagnostic scan makes the CSV's "Autofocus Result Z" disagree with
    # where the stage actually ends, which masks real AF problems
    # (2026-05-15: empirical analysis caught run std_223650 reporting
    # Result Z=1830.08 while the diagnostic scan peak was at Z=1823.20
    # and the stage had been moved there silently). Disagreement is
    # logged for visibility; it indicates the AF's interpolation or
    # search range needs tuning.
    disagreement_um = abs(peak_z - center_z)
    if disagreement_um > 1.0:
        logger.warning(
            "  Diagnostic scan peak at Z=%.2f um disagrees with AF result "
            "Z=%.2f um by %.2f um -- AF interpolation may be off, or scan "
            "drift between AF and diagnostic. Stage left at AF result; "
            "CSV reports both values.",
            peak_z,
            center_z,
            disagreement_um,
        )
    else:
        logger.info(
            "  Diagnostic scan peak Z=%.2f matches AF result Z=%.2f " "within %.2f um",
            peak_z,
            center_z,
            disagreement_um,
        )
    # Stage is currently at z_positions[-1] (last diagnostic step).
    # Move back to AF's reported result so the post-test state matches
    # what the CSV reports.
    focused_pos = Position(
        hardware.get_current_position().x, hardware.get_current_position().y, center_z
    )
    hardware.move_to_position(focused_pos)

    # Generate plot and CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"autofocus_test_{test_type}_{timestamp}.png"
    csv_filename = f"autofocus_test_{test_type}_{timestamp}.csv"
    plot_path = output_path / plot_filename
    csv_path = output_path / csv_filename

    # Save CSV with all diagnostic data
    try:
        import csv

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header with validation results
            writer.writerow(["# Autofocus Diagnostic Data"])
            writer.writerow(["# Timestamp", timestamp])
            writer.writerow(["# Test Type", test_type])
            writer.writerow(["# Metric", af_settings["score_metric_name"]])
            writer.writerow(["# Autofocus Result Z", f"{center_z:.2f}"])
            writer.writerow(["# Scan Peak Z", f"{peak_z:.2f}"])
            writer.writerow(["#"])
            writer.writerow(["# VALIDATION RESULTS"])
            writer.writerow(["# Status", "VALID" if validation["is_valid"] else "INVALID"])
            writer.writerow(["# Quality Score", f"{validation['quality_score']:.3f}"])
            writer.writerow(["# Peak Prominence", f"{validation['peak_prominence']:.3f}"])
            writer.writerow(["# Has Ascending", validation["has_ascending"]])
            writer.writerow(["# Has Descending", validation["has_descending"]])
            writer.writerow(["# Symmetry Score", f"{validation['symmetry_score']:.3f}"])
            writer.writerow(["# Message", validation["message"]])
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    writer.writerow(["# Warning", warning])
            writer.writerow(["#"])

            # Write data header
            writer.writerow(["Z_Position_um", "Focus_Score"])

            # Write data
            for z, score in zip(z_positions, scores):
                writer.writerow([f"{z:.2f}", f"{score:.4f}"])

        logger.info(f"  CSV data saved: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save CSV: {e}")

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot focus curve
        ax.plot(
            z_positions,
            scores,
            "o-",
            markersize=6,
            linewidth=2,
            color="steelblue",
            label="Focus scores",
        )
        ax.axvline(
            center_z,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Autofocus result ({center_z:.2f} um)",
        )

        # Plot the peak we found and moved to (calculated earlier)
        ax.axvline(
            peak_z,
            color="green",
            linestyle=":",
            linewidth=1.5,
            label=f"Scan peak ({peak_z:.2f} um)",
        )

        ax.set_xlabel("Z Position (um)", fontsize=11)
        ax.set_ylabel("Focus Score", fontsize=11)

        # Add validation status to title
        validation_status = "VALID" if validation["is_valid"] else "INVALID"
        title_color = "green" if validation["is_valid"] else "red"
        ax.set_title(
            f"{test_type.capitalize()} Autofocus Test - Diagnostic Scan\n"
            + f'Metric: {af_settings["score_metric_name"]} | Peak: {validation_status}',
            fontsize=12,
            fontweight="bold",
            color=title_color,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add text summary with validation info
        textstr = f"Autofocus result: {center_z:.2f} um\n"
        textstr += f"Scan peak: {peak_z:.2f} um\n"
        textstr += f"Difference: {abs(center_z - peak_z):.2f} um\n"
        textstr += f"Score at result: {scores[np.argmin(np.abs(z_positions - center_z))]:.2f}\n"
        textstr += f"Score at peak: {scores[peak_idx]:.2f}\n\n"
        textstr += "PEAK VALIDATION:\n"
        textstr += f"  Status: {validation_status}\n"
        textstr += f'  Quality: {validation["quality_score"]:.2f}\n'
        textstr += f'  Prominence: {validation["peak_prominence"]:.2f}\n'
        textstr += f'  Ascending: {validation["has_ascending"]}\n'
        textstr += f'  Descending: {validation["has_descending"]}\n'
        textstr += f'  Symmetry: {validation["symmetry_score"]:.2f}'

        box_color = "lightgreen" if validation["is_valid"] else "lightcoral"
        props = dict(boxstyle="round", facecolor=box_color, alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  Plot saved: {plot_path}")

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)
        raise

    return plot_path, validation


def test_autofocus_at_current_position(
    hardware,
    config_manager,
    yaml_file_path: str,
    output_folder_path: str,
    objective: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive autofocus testing at current microscope position.

    Args:
        hardware: PycromanagerHardware instance
        config_manager: ConfigManager instance
        yaml_file_path: Path to microscope config YAML
        output_folder_path: Where to save diagnostic plots and data
        objective: Objective identifier (e.g., "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001")
        logger: Optional logger instance

    Returns:
        Dict containing:
            - success: bool
            - initial_z: Starting Z position
            - final_z: Z position after autofocus
            - z_shift: Difference between initial and final Z
            - raw_best_z: Best Z from raw scores (no interpolation)
            - interp_best_z: Best Z from interpolated curve
            - raw_scores: List of (z, score) tuples
            - plot_path: Path to diagnostic plot
            - message: Status message
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=== AUTOFOCUS TEST STARTED ===")
    logger.info(f"  Objective: {objective}")
    logger.info(f"  Config file: {yaml_file_path}")

    # Create output directory
    output_path = Path(output_folder_path)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "initial_z": None,
        "final_z": None,
        "z_shift": None,
        "raw_best_z": None,
        "interp_best_z": None,
        "raw_scores": [],
        "plot_path": None,
        "message": "",
    }

    try:
        # Get current position
        initial_pos = hardware.get_current_position()
        result["initial_z"] = initial_pos.z
        logger.info(
            f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}"
        )

        # Load autofocus settings for this objective
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        logger.info("  Autofocus settings:")
        logger.info(f"    n_steps: {af_settings['n_steps']}")
        logger.info(f"    search_range: {af_settings['search_range']} um")
        logger.info(f"    interp_strength: {af_settings['interp_strength']}")
        logger.info(f"    interp_kind: {af_settings['interp_kind']}")
        logger.info(f"    score_metric: {af_settings['score_metric_name']}")

        # Perform autofocus with detailed logging
        z_positions, scores, metrics_data = _detailed_autofocus_scan(
            hardware, initial_pos, af_settings, logger
        )

        # Find best focus from raw scores
        best_raw_idx = np.argmax(scores)
        raw_best_z = z_positions[best_raw_idx]
        raw_best_score = scores[best_raw_idx]

        result["raw_best_z"] = raw_best_z
        result["raw_scores"] = list(zip(z_positions, scores))

        logger.info("  Raw scores analysis:")
        logger.info(
            f"    Best Z (no interpolation): {raw_best_z:.2f} um (score={raw_best_score:.2f})"
        )
        logger.info(f"    Mean score: {np.mean(scores):.2f}")
        logger.info(f"    Std score: {np.std(scores):.2f}")
        logger.info(f"    Score range: {np.min(scores):.2f} - {np.max(scores):.2f}")

        # Perform interpolation
        interp_x = np.linspace(
            z_positions[0], z_positions[-1], af_settings["n_steps"] * af_settings["interp_strength"]
        )

        try:
            interp_y = scipy.interpolate.interp1d(
                z_positions, scores, kind=af_settings["interp_kind"]
            )(interp_x)

            interp_best_idx = np.argmax(interp_y)
            interp_best_z = interp_x[interp_best_idx]
            interp_best_score = interp_y[interp_best_idx]

            result["interp_best_z"] = interp_best_z

            logger.info("  Interpolated scores analysis:")
            logger.info(
                f"    Best Z (interpolated): {interp_best_z:.2f} um (score={interp_best_score:.2f})"
            )
            logger.info(f"    Difference from raw: {abs(interp_best_z - raw_best_z):.2f} um")

            # Check if interpolation is creating artifacts
            if abs(interp_best_z - raw_best_z) > (
                af_settings["search_range"] / af_settings["n_steps"]
            ):
                logger.warning("  WARNING: Interpolated peak differs significantly from raw peak!")
                logger.warning("  This suggests interpolation may be creating artifacts.")
                logger.warning("  Consider reducing interp_strength or using linear interpolation.")

        except Exception as e:
            logger.error(f"  Interpolation failed: {e}")
            interp_x = z_positions
            interp_y = scores
            interp_best_z = raw_best_z

        # Move to best focus position
        final_pos = Position(initial_pos.x, initial_pos.y, interp_best_z)
        hardware.move_to_position(final_pos)

        result["final_z"] = interp_best_z
        result["z_shift"] = interp_best_z - initial_pos.z

        logger.info(f"  Moved to best focus: Z={interp_best_z:.2f} um")
        logger.info(f"  Z shift from starting position: {result['z_shift']:.2f} um")

        # Generate diagnostic plot
        plot_path = _generate_diagnostic_plot(
            z_positions,
            scores,
            interp_x,
            interp_y,
            initial_pos.z,
            raw_best_z,
            interp_best_z,
            af_settings,
            metrics_data,
            output_path,
            logger,
        )

        result["plot_path"] = str(plot_path)
        result["success"] = True
        result["message"] = f"Autofocus test completed. Z shift: {result['z_shift']:.2f} um"

        logger.info("=== AUTOFOCUS TEST COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        logger.error(f"Autofocus test failed: {e}", exc_info=True)
        result["message"] = f"Autofocus test failed: {str(e)}"

        # Try to return to initial position
        if result["initial_z"] is not None:
            try:
                hardware.move_to_position(
                    Position(initial_pos.x, initial_pos.y, result["initial_z"])
                )
                logger.info("Returned to initial Z position after error")
            except Exception as e:
                logger.warning("Failed to return to initial position: %s", e)

    return result


def _load_autofocus_settings(yaml_file_path: str, objective: str, logger) -> Dict[str, Any]:
    """Load autofocus settings for specified objective."""
    import yaml

    # Derive autofocus config path from main config path
    config_path = Path(yaml_file_path)
    config_name = config_path.stem
    microscope_name = config_name.replace("config_", "")
    autofocus_file = config_path.parent / f"autofocus_{microscope_name}.yml"

    # Defaults
    settings = {
        "n_steps": 11,
        "search_range": 15.0,
        "interp_strength": 100,
        "interp_kind": "quadratic",
        "score_metric_name": "normalized_variance",
        "sweep_range_um": 10.0,
        "sweep_n_steps": 6,
    }

    if autofocus_file.exists():
        try:
            with open(autofocus_file, "r") as f:
                autofocus_config = yaml.safe_load(f)

            af_settings_list = autofocus_config.get("autofocus_settings", [])
            for af_setting in af_settings_list:
                if af_setting.get("objective") == objective:
                    settings["n_steps"] = af_setting.get("n_steps", settings["n_steps"])
                    settings["search_range"] = af_setting.get(
                        "search_range_um", settings["search_range"]
                    )
                    settings["interp_strength"] = af_setting.get(
                        "interp_strength", settings["interp_strength"]
                    )
                    settings["interp_kind"] = af_setting.get("interp_kind", settings["interp_kind"])
                    settings["score_metric_name"] = af_setting.get(
                        "score_metric", settings["score_metric_name"]
                    )
                    settings["sweep_range_um"] = af_setting.get(
                        "sweep_range_um", settings["sweep_range_um"]
                    )
                    settings["sweep_n_steps"] = af_setting.get(
                        "sweep_n_steps", settings["sweep_n_steps"]
                    )
                    # Legacy support: old adaptive_initial_step_um -> sweep_range_um
                    if (
                        "sweep_range_um" not in af_setting
                        and "adaptive_initial_step_um" in af_setting
                    ):
                        settings["sweep_range_um"] = af_setting["adaptive_initial_step_um"] * 2
                    break
        except Exception as e:
            logger.warning(f"Error loading autofocus settings, using defaults: {e}")
    else:
        logger.warning(f"Autofocus config file not found: {autofocus_file}, using defaults")

    # Map score metric name to function (used by standard autofocus test)
    score_metric_map = {
        "normalized_variance": AutofocusUtils.autofocus_profile_laplacian_variance,  # fallback for standard AF
        "laplacian_variance": AutofocusUtils.autofocus_profile_laplacian_variance,
        "sobel": AutofocusUtils.autofocus_profile_sobel,
        "brenner_gradient": AutofocusUtils.autofocus_profile_brenner_gradient,
        "robust_sharpness": AutofocusUtils.autofocus_profile_robust_sharpness_metric,
        "hybrid_sharpness": AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
        "p98_p2": AutofocusUtils.autofocus_profile_laplacian_variance,  # fallback for standard AF
    }

    settings["score_metric"] = score_metric_map.get(
        settings["score_metric_name"], AutofocusUtils.autofocus_profile_laplacian_variance
    )

    return settings


def _detailed_autofocus_scan(
    hardware, initial_pos, af_settings, logger
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform autofocus scan with detailed logging.

    Returns:
        z_positions: Array of Z positions sampled
        scores: Array of focus scores at each position
        metrics_data: Dict with additional metrics for plotting
    """
    n_steps = af_settings["n_steps"]
    search_range = af_settings["search_range"]
    score_metric = af_settings["score_metric"]

    # Calculate Z steps
    steps = np.linspace(0, search_range, n_steps) - (search_range / 2)
    z_positions = initial_pos.z + steps

    logger.info("  Starting autofocus scan:")
    logger.info(f"    Z range: {z_positions[0]:.2f} to {z_positions[-1]:.2f} um")
    logger.info(f"    Step size: {search_range / (n_steps - 1):.2f} um")

    scores = []
    image_means = []

    for i, z in enumerate(z_positions):
        # Move to position
        new_pos = Position(initial_pos.x, initial_pos.y, z)
        hardware.move_to_position(new_pos)

        # Acquire image
        img, tags = hardware.snap_image()

        # Extract grayscale for focus calculation
        if hardware.core.get_property("Core", "Camera") == "JAICamera":
            img_gray = np.mean(img, 2)
        else:
            # Bayer pattern - extract green channels
            green1 = img[0::2, 0::2]
            green2 = img[1::2, 1::2]
            img_gray = ((green1 + green2) / 2.0).astype(np.float32)

        # Calculate focus score
        score = score_metric(img_gray)
        if hasattr(score, "ndim") and score.ndim == 2:
            score = np.mean(score)

        scores.append(float(score))
        image_means.append(np.mean(img_gray))

        logger.info(
            f"    Step {i+1}/{n_steps}: Z={z:.2f} um, score={score:.2f}, mean_intensity={image_means[-1]:.1f}"
        )

    metrics_data = {
        "image_means": np.array(image_means),
        "score_metric_name": af_settings["score_metric_name"],
    }

    return np.array(z_positions), np.array(scores), metrics_data


def _generate_diagnostic_plot(
    z_positions,
    scores,
    interp_x,
    interp_y,
    initial_z,
    raw_best_z,
    interp_best_z,
    af_settings,
    metrics_data,
    output_path,
    logger,
):
    """Generate comprehensive diagnostic plot for autofocus analysis."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"autofocus_test_{timestamp}.png"
    plot_path = output_path / plot_filename

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Autofocus Test Results - {timestamp}", fontsize=14, fontweight="bold")

        # Plot 1: Focus curve with raw and interpolated scores
        ax1 = axes[0, 0]
        ax1.bar(
            z_positions,
            scores,
            width=(z_positions[1] - z_positions[0]) * 0.8,
            alpha=0.6,
            color="steelblue",
            label="Raw scores",
        )
        ax1.plot(interp_x, interp_y, "k-", linewidth=2, label="Interpolated curve")
        ax1.axvline(
            initial_z,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Initial Z ({initial_z:.2f})",
        )
        ax1.axvline(
            raw_best_z,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Raw best ({raw_best_z:.2f})",
        )
        ax1.axvline(
            interp_best_z,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Interp best ({interp_best_z:.2f})",
        )
        ax1.set_xlabel("Z Position (um)", fontsize=11)
        ax1.set_ylabel("Focus Score", fontsize=11)
        ax1.set_title("Focus Curve Analysis", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Focus scores normalized to show relative differences
        ax2 = axes[0, 1]
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        interp_y_norm = (interp_y - np.min(interp_y)) / (
            np.max(interp_y) - np.min(interp_y) + 1e-10
        )
        ax2.plot(
            z_positions,
            scores_norm,
            "o-",
            markersize=8,
            linewidth=2,
            color="steelblue",
            label="Raw (normalized)",
        )
        ax2.plot(
            interp_x,
            interp_y_norm,
            "-",
            linewidth=1,
            color="orange",
            alpha=0.7,
            label="Interpolated (normalized)",
        )
        ax2.axvline(interp_best_z, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax2.set_xlabel("Z Position (um)", fontsize=11)
        ax2.set_ylabel("Normalized Focus Score (0-1)", fontsize=11)
        ax2.set_title("Normalized Focus Scores", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Mean intensity at each Z position (to check for saturation/issues)
        ax3 = axes[1, 0]
        ax3.plot(
            z_positions,
            metrics_data["image_means"],
            "o-",
            markersize=6,
            linewidth=2,
            color="purple",
        )
        ax3.axhline(
            255, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Saturation (255)"
        )
        ax3.axvline(
            interp_best_z, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Best focus Z"
        )
        ax3.set_xlabel("Z Position (um)", fontsize=11)
        ax3.set_ylabel("Mean Image Intensity", fontsize=11)
        ax3.set_title("Image Brightness vs Z Position", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics text
        ax4 = axes[1, 1]
        ax4.axis("off")

        summary_text = f"""AUTOFOCUS TEST SUMMARY

Initial Z Position:     {initial_z:.2f} um

Raw Scores:
  Best Z:               {raw_best_z:.2f} um
  Best Score:           {np.max(scores):.2f}
  Mean Score:           {np.mean(scores):.2f}
  Score Range:          {np.min(scores):.2f} - {np.max(scores):.2f}
  Score StdDev:         {np.std(scores):.2f}

Interpolated:
  Best Z:               {interp_best_z:.2f} um
  Diff from Raw:        {abs(interp_best_z - raw_best_z):.3f} um

Z Shift:               {interp_best_z - initial_z:+.2f} um

Settings:
  n_steps:              {af_settings['n_steps']}
  search_range:         {af_settings['search_range']:.1f} um
  step_size:            {af_settings['search_range']/(af_settings['n_steps']-1):.2f} um
  interp_strength:      {af_settings['interp_strength']}
  interp_kind:          {af_settings['interp_kind']}
  score_metric:         {metrics_data['score_metric_name']}

Status:
  {"OK - Focus found" if abs(interp_best_z - initial_z) < 10 else "WARNING - Large Z shift"}
  {"WARNING - Interp differs from raw" if abs(interp_best_z - raw_best_z) > af_settings['search_range']/af_settings['n_steps'] else "Interp consistent with raw"}
"""

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  Diagnostic plot saved: {plot_path}")

    except Exception as e:
        logger.error(f"Failed to generate diagnostic plot: {e}", exc_info=True)

    return plot_path
