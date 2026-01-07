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
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.interpolate
import skimage.filters
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
        logger.info(f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}")

        # Load autofocus settings
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        logger.info(f"  Autofocus settings:")
        logger.info(f"    n_steps: {af_settings['n_steps']}")
        logger.info(f"    search_range: {af_settings['search_range']} um (centered on current Z)")
        logger.info(f"    interp_strength: {af_settings['interp_strength']}")
        logger.info(f"    interp_kind: {af_settings['interp_kind']}")
        logger.info(f"    score_metric: {af_settings['score_metric_name']}")

        # Call the ACTUAL hardware.autofocus() method
        logger.info("  Calling hardware.autofocus() with config settings...")

        final_z = hardware.autofocus(
            n_steps=af_settings['n_steps'],
            search_range=af_settings['search_range'],
            interp_strength=af_settings['interp_strength'],
            interp_kind=af_settings['interp_kind'],
            score_metric=af_settings['score_metric'],
            pop_a_plot=False,
            move_stage_to_estimate=True,
            raise_on_invalid_peak=False,  # Always generate diagnostics for test
        )

        result["final_z"] = final_z
        result["z_shift"] = final_z - initial_pos.z

        logger.info(f"  Standard autofocus completed:")
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
                logger.error("  RECOMMENDATION: Check focus manually or increase autofocus search range")
                result["success"] = False
                result["message"] = f"Autofocus failed: {validation['message']}. Check focus manually or increase search range."
            else:
                result["message"] = f"Standard autofocus completed. Z shift: {result['z_shift']:.2f} um."
                result["success"] = True

        except Exception as e:
            logger.warning(f"Failed to generate diagnostic plot: {e}")
            result["plot_path"] = "None"
            # Still mark as success if autofocus itself worked, just plotting failed
            result["message"] = f"Standard autofocus completed. Z shift: {result['z_shift']:.2f} um."
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
            except:
                pass

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
    Test ADAPTIVE autofocus at current microscope position.
    Calls hardware.autofocus_adaptive_search() with settings from config file.

    This starts at current position and searches intelligently, stopping when "good enough".

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

    logger.info("=== ADAPTIVE AUTOFOCUS TEST STARTED ===")
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
        "test_type": "adaptive",
    }

    try:
        # Get current position
        initial_pos = hardware.get_current_position()
        result["initial_z"] = initial_pos.z
        logger.info(f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}")

        # Load autofocus settings
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        logger.info(f"  Adaptive autofocus settings:")
        logger.info(f"    initial_step_size: {af_settings['adaptive_initial_step_um']} um")
        logger.info(f"    min_step_size: {af_settings['adaptive_min_step_um']} um")
        logger.info(f"    max_total_steps: {af_settings['adaptive_max_steps']}")
        logger.info(f"    focus_threshold: {af_settings['adaptive_focus_threshold']}")
        logger.info(f"    score_metric: {af_settings['score_metric_name']}")

        # Call the ACTUAL hardware.autofocus_adaptive_search() method
        logger.info("  Calling hardware.autofocus_adaptive_search()...")

        final_z = hardware.autofocus_adaptive_search(
            initial_step_size=af_settings['adaptive_initial_step_um'],
            min_step_size=af_settings['adaptive_min_step_um'],
            focus_threshold=af_settings['adaptive_focus_threshold'],
            max_total_steps=af_settings['adaptive_max_steps'],
            score_metric=af_settings['score_metric'],
            pop_a_plot=False,
            move_stage_to_estimate=True,
        )

        result["final_z"] = final_z
        result["z_shift"] = final_z - initial_pos.z

        logger.info(f"  Adaptive autofocus completed:")
        logger.info(f"    Final Z: {final_z:.2f} um")
        logger.info(f"    Z shift: {result['z_shift']:.2f} um")

        # Adaptive autofocus does not generate a diagnostic plot
        # It is designed to be fast and efficient - running an additional scan defeats that purpose
        # The hardware.autofocus_adaptive_search() method already completed and found focus
        result["plot_path"] = None
        logger.info("  No diagnostic plot for adaptive autofocus (designed for speed)")

        result["message"] = f"Adaptive autofocus completed. Z shift: {result['z_shift']:.2f} um."
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
            except:
                pass

    return result


def _generate_diagnostic_scan_plot(hardware, center_z, af_settings, output_path, logger, test_type="standard"):
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
    scan_range = af_settings['search_range']
    n_steps = af_settings['n_steps']
    score_metric = af_settings['score_metric']

    # Generate Z positions centered on final result
    z_positions = np.linspace(center_z - scan_range/2, center_z + scan_range/2, n_steps)

    logger.info(f"  Scanning {n_steps} positions from {z_positions[0]:.2f} to {z_positions[-1]:.2f} um")

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
        if hasattr(score, 'ndim') and score.ndim == 2:
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

    if not validation['is_valid']:
        logger.warning("  *** AUTOFOCUS PEAK QUALITY CHECK FAILED ***")
        for warning in validation['warnings']:
            logger.warning(f"    - {warning}")

    # Find peak in the diagnostic scan
    peak_idx = np.argmax(scores)
    peak_z = z_positions[peak_idx]

    # CRITICAL: Move to the peak position found in diagnostic scan
    # The scan leaves us at the last Z position, need to move to actual focus peak
    logger.info(f"  Diagnostic scan peak found at Z={peak_z:.2f} um (autofocus result was {center_z:.2f} um)")
    focused_pos = Position(hardware.get_current_position().x, hardware.get_current_position().y, peak_z)
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
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header with validation results
            writer.writerow(['# Autofocus Diagnostic Data'])
            writer.writerow(['# Timestamp', timestamp])
            writer.writerow(['# Test Type', test_type])
            writer.writerow(['# Metric', af_settings['score_metric_name']])
            writer.writerow(['# Autofocus Result Z', f'{center_z:.2f}'])
            writer.writerow(['# Scan Peak Z', f'{peak_z:.2f}'])
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

        logger.info(f"  CSV data saved: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save CSV: {e}")

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot focus curve
        ax.plot(z_positions, scores, 'o-', markersize=6, linewidth=2, color='steelblue', label='Focus scores')
        ax.axvline(center_z, color='red', linestyle='--', linewidth=2, label=f'Autofocus result ({center_z:.2f} um)')

        # Plot the peak we found and moved to (calculated earlier)
        ax.axvline(peak_z, color='green', linestyle=':', linewidth=1.5, label=f'Scan peak ({peak_z:.2f} um)')

        ax.set_xlabel('Z Position (um)', fontsize=11)
        ax.set_ylabel('Focus Score', fontsize=11)

        # Add validation status to title
        validation_status = "VALID" if validation['is_valid'] else "INVALID"
        title_color = 'green' if validation['is_valid'] else 'red'
        ax.set_title(f'{test_type.capitalize()} Autofocus Test - Diagnostic Scan\n' +
                    f'Metric: {af_settings["score_metric_name"]} | Peak: {validation_status}',
                    fontsize=12, fontweight='bold', color=title_color)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add text summary with validation info
        textstr = f'Autofocus result: {center_z:.2f} um\n'
        textstr += f'Scan peak: {peak_z:.2f} um\n'
        textstr += f'Difference: {abs(center_z - peak_z):.2f} um\n'
        textstr += f'Score at result: {scores[np.argmin(np.abs(z_positions - center_z))]:.2f}\n'
        textstr += f'Score at peak: {scores[peak_idx]:.2f}\n\n'
        textstr += f'PEAK VALIDATION:\n'
        textstr += f'  Status: {validation_status}\n'
        textstr += f'  Quality: {validation["quality_score"]:.2f}\n'
        textstr += f'  Prominence: {validation["peak_prominence"]:.2f}\n'
        textstr += f'  Ascending: {validation["has_ascending"]}\n'
        textstr += f'  Descending: {validation["has_descending"]}\n'
        textstr += f'  Symmetry: {validation["symmetry_score"]:.2f}'

        box_color = 'lightgreen' if validation['is_valid'] else 'lightcoral'
        props = dict(boxstyle='round', facecolor=box_color, alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
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
        logger.info(f"  Initial position: X={initial_pos.x:.2f}, Y={initial_pos.y:.2f}, Z={initial_pos.z:.2f}")

        # Load autofocus settings for this objective
        af_settings = _load_autofocus_settings(yaml_file_path, objective, logger)

        logger.info(f"  Autofocus settings:")
        logger.info(f"    n_steps: {af_settings['n_steps']}")
        logger.info(f"    search_range: {af_settings['search_range']} um")
        logger.info(f"    interp_strength: {af_settings['interp_strength']}")
        logger.info(f"    interp_kind: {af_settings['interp_kind']}")
        logger.info(f"    score_metric: {af_settings['score_metric_name']}")

        # Perform autofocus with detailed logging
        z_positions, scores, metrics_data = _detailed_autofocus_scan(
            hardware,
            initial_pos,
            af_settings,
            logger
        )

        # Find best focus from raw scores
        best_raw_idx = np.argmax(scores)
        raw_best_z = z_positions[best_raw_idx]
        raw_best_score = scores[best_raw_idx]

        result["raw_best_z"] = raw_best_z
        result["raw_scores"] = list(zip(z_positions, scores))

        logger.info(f"  Raw scores analysis:")
        logger.info(f"    Best Z (no interpolation): {raw_best_z:.2f} um (score={raw_best_score:.2f})")
        logger.info(f"    Mean score: {np.mean(scores):.2f}")
        logger.info(f"    Std score: {np.std(scores):.2f}")
        logger.info(f"    Score range: {np.min(scores):.2f} - {np.max(scores):.2f}")

        # Perform interpolation
        interp_x = np.linspace(z_positions[0], z_positions[-1],
                              af_settings['n_steps'] * af_settings['interp_strength'])

        try:
            interp_y = scipy.interpolate.interp1d(
                z_positions, scores, kind=af_settings['interp_kind']
            )(interp_x)

            interp_best_idx = np.argmax(interp_y)
            interp_best_z = interp_x[interp_best_idx]
            interp_best_score = interp_y[interp_best_idx]

            result["interp_best_z"] = interp_best_z

            logger.info(f"  Interpolated scores analysis:")
            logger.info(f"    Best Z (interpolated): {interp_best_z:.2f} um (score={interp_best_score:.2f})")
            logger.info(f"    Difference from raw: {abs(interp_best_z - raw_best_z):.2f} um")

            # Check if interpolation is creating artifacts
            if abs(interp_best_z - raw_best_z) > (af_settings['search_range'] / af_settings['n_steps']):
                logger.warning(f"  WARNING: Interpolated peak differs significantly from raw peak!")
                logger.warning(f"  This suggests interpolation may be creating artifacts.")
                logger.warning(f"  Consider reducing interp_strength or using linear interpolation.")

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
            z_positions, scores,
            interp_x, interp_y,
            initial_pos.z, raw_best_z, interp_best_z,
            af_settings,
            metrics_data,
            output_path,
            logger
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
            except:
                pass

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
        'n_steps': 11,
        'search_range': 15.0,
        'interp_strength': 100,
        'interp_kind': 'quadratic',
        'score_metric_name': 'laplacian_variance',
        'adaptive_initial_step_um': 10.0,
        'adaptive_min_step_um': 2.0,
        'adaptive_max_steps': 25,
        'adaptive_focus_threshold': 0.95,
    }

    if autofocus_file.exists():
        try:
            with open(autofocus_file, 'r') as f:
                autofocus_config = yaml.safe_load(f)

            af_settings_list = autofocus_config.get('autofocus_settings', [])
            for af_setting in af_settings_list:
                if af_setting.get('objective') == objective:
                    settings['n_steps'] = af_setting.get('n_steps', settings['n_steps'])
                    settings['search_range'] = af_setting.get('search_range_um', settings['search_range'])
                    settings['interp_strength'] = af_setting.get('interp_strength', settings['interp_strength'])
                    settings['interp_kind'] = af_setting.get('interp_kind', settings['interp_kind'])
                    settings['score_metric_name'] = af_setting.get('score_metric', settings['score_metric_name'])
                    settings['adaptive_initial_step_um'] = af_setting.get('adaptive_initial_step_um', settings['adaptive_initial_step_um'])
                    settings['adaptive_min_step_um'] = af_setting.get('adaptive_min_step_um', settings['adaptive_min_step_um'])
                    settings['adaptive_max_steps'] = af_setting.get('adaptive_max_steps', settings['adaptive_max_steps'])
                    settings['adaptive_focus_threshold'] = af_setting.get('adaptive_focus_threshold', settings['adaptive_focus_threshold'])
                    break
        except Exception as e:
            logger.warning(f"Error loading autofocus settings, using defaults: {e}")
    else:
        logger.warning(f"Autofocus config file not found: {autofocus_file}, using defaults")

    # Map score metric name to function
    score_metric_map = {
        'laplacian_variance': AutofocusUtils.autofocus_profile_laplacian_variance,
        'sobel': AutofocusUtils.autofocus_profile_sobel,
        'brenner_gradient': AutofocusUtils.autofocus_profile_brenner_gradient,
        'robust_sharpness': AutofocusUtils.autofocus_profile_robust_sharpness_metric,
        'hybrid_sharpness': AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
    }

    settings['score_metric'] = score_metric_map.get(
        settings['score_metric_name'],
        AutofocusUtils.autofocus_profile_laplacian_variance
    )

    return settings


def _detailed_autofocus_scan(hardware, initial_pos, af_settings, logger) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform autofocus scan with detailed logging.

    Returns:
        z_positions: Array of Z positions sampled
        scores: Array of focus scores at each position
        metrics_data: Dict with additional metrics for plotting
    """
    n_steps = af_settings['n_steps']
    search_range = af_settings['search_range']
    score_metric = af_settings['score_metric']

    # Calculate Z steps
    steps = np.linspace(0, search_range, n_steps) - (search_range / 2)
    z_positions = initial_pos.z + steps

    logger.info(f"  Starting autofocus scan:")
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
        if hasattr(score, 'ndim') and score.ndim == 2:
            score = np.mean(score)

        scores.append(float(score))
        image_means.append(np.mean(img_gray))

        logger.info(f"    Step {i+1}/{n_steps}: Z={z:.2f} um, score={score:.2f}, mean_intensity={image_means[-1]:.1f}")

    metrics_data = {
        'image_means': np.array(image_means),
        'score_metric_name': af_settings['score_metric_name'],
    }

    return np.array(z_positions), np.array(scores), metrics_data


def _generate_diagnostic_plot(z_positions, scores, interp_x, interp_y,
                              initial_z, raw_best_z, interp_best_z,
                              af_settings, metrics_data, output_path, logger):
    """Generate comprehensive diagnostic plot for autofocus analysis."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"autofocus_test_{timestamp}.png"
    plot_path = output_path / plot_filename

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Autofocus Test Results - {timestamp}', fontsize=14, fontweight='bold')

        # Plot 1: Focus curve with raw and interpolated scores
        ax1 = axes[0, 0]
        ax1.bar(z_positions, scores, width=(z_positions[1]-z_positions[0])*0.8,
                alpha=0.6, color='steelblue', label='Raw scores')
        ax1.plot(interp_x, interp_y, 'k-', linewidth=2, label='Interpolated curve')
        ax1.axvline(initial_z, color='gray', linestyle='--', linewidth=1.5, label=f'Initial Z ({initial_z:.2f})')
        ax1.axvline(raw_best_z, color='green', linestyle='--', linewidth=1.5, label=f'Raw best ({raw_best_z:.2f})')
        ax1.axvline(interp_best_z, color='red', linestyle='--', linewidth=2, label=f'Interp best ({interp_best_z:.2f})')
        ax1.set_xlabel('Z Position (um)', fontsize=11)
        ax1.set_ylabel('Focus Score', fontsize=11)
        ax1.set_title('Focus Curve Analysis', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Focus scores normalized to show relative differences
        ax2 = axes[0, 1]
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        interp_y_norm = (interp_y - np.min(interp_y)) / (np.max(interp_y) - np.min(interp_y) + 1e-10)
        ax2.plot(z_positions, scores_norm, 'o-', markersize=8, linewidth=2, color='steelblue', label='Raw (normalized)')
        ax2.plot(interp_x, interp_y_norm, '-', linewidth=1, color='orange', alpha=0.7, label='Interpolated (normalized)')
        ax2.axvline(interp_best_z, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Z Position (um)', fontsize=11)
        ax2.set_ylabel('Normalized Focus Score (0-1)', fontsize=11)
        ax2.set_title('Normalized Focus Scores', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Mean intensity at each Z position (to check for saturation/issues)
        ax3 = axes[1, 0]
        ax3.plot(z_positions, metrics_data['image_means'], 'o-', markersize=6,
                linewidth=2, color='purple')
        ax3.axhline(255, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Saturation (255)')
        ax3.axvline(interp_best_z, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best focus Z')
        ax3.set_xlabel('Z Position (um)', fontsize=11)
        ax3.set_ylabel('Mean Image Intensity', fontsize=11)
        ax3.set_title('Image Brightness vs Z Position', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics text
        ax4 = axes[1, 1]
        ax4.axis('off')

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

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Diagnostic plot saved: {plot_path}")

    except Exception as e:
        logger.error(f"Failed to generate diagnostic plot: {e}", exc_info=True)

    return plot_path
