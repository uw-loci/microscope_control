"""
Autofocus Parameter Benchmarking Module
========================================

Systematic testing of autofocus parameters to find optimal settings for
different focus distances and objectives. Performs grid search across
parameter combinations and measures time-to-focus performance.

Key capabilities:
- Grid search across n_steps, search_range, interp_kind, and score_metric
- Distance-based testing (tests from multiple Z offsets from true focus)
- Timing measurements for each autofocus attempt
- Focus accuracy validation (how close to true focus)
- Comprehensive CSV/JSON reports for analysis
- SAFETY: Hardcoded Z limits to prevent objective-sample collision

SAFETY NOTE (Upright Microscope):
    On this upright microscope, MORE NEGATIVE Z = stage raised = closer to objective.
    The safety system prevents moving past a configurable limit to protect the objective.

Usage:
    benchmark = AutofocusBenchmark(hardware, config_manager, logger)
    results = benchmark.run_benchmark(
        reference_z=100.0,  # Known good focus position
        test_distances=[5, 10, 20, 30, 50],  # um from focus
        output_path="/path/to/results"
    )
"""

import numpy as np
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
import logging

from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.hardware.base import Position


# =============================================================================
# SAFETY CONFIGURATION - CRITICAL FOR OBJECTIVE PROTECTION
# =============================================================================
# On this UPRIGHT microscope:
#   - More NEGATIVE Z = stage RAISED = CLOSER to objective (DANGER!)
#   - Less negative Z = stage lowered = further from objective (safe)
#
# The Z_SAFETY_LIMIT_UM is the most negative (highest stage position) allowed.
# This should be set with a safety margin from the actual collision point.
#
# Per-objective limits account for different working distances:
#   - 10x: ~10mm working distance -> can get closer
#   - 20x: ~3mm working distance -> moderate limit
#   - 40x: ~0.5mm working distance -> strictest limit
# =============================================================================

# Default absolute safety limit (um) - NEVER move more negative than this
# This is a hardcoded failsafe that applies regardless of objective
Z_ABSOLUTE_SAFETY_LIMIT_UM: float = -5550.0  # 50um buffer from config limit of -5601

# Per-objective safety limits (um) - more specific limits based on working distance
# Key: objective identifier substring, Value: most negative Z allowed
OBJECTIVE_SAFETY_LIMITS_UM: Dict[str, float] = {
    "10X": -5550.0,   # 10x has longest working distance
    "20X": -5500.0,   # 20x has moderate working distance
    "40X": -5400.0,   # 40x has shortest working distance - most conservative
}

# Additional safety margin (um) to add beyond the calculated test positions
# This accounts for autofocus overshoot during search
AUTOFOCUS_OVERSHOOT_MARGIN_UM: float = 30.0

# Maximum acceptable Z error (um) for a trial to be considered successful
# If autofocus returns a position further than this from the reference, it's a failure
MAX_ACCEPTABLE_Z_ERROR_UM: float = 5.0


class ZSafetyError(Exception):
    """Raised when a Z movement would violate safety limits."""
    pass


@dataclass
class BenchmarkResult:
    """Results from a single autofocus benchmark trial."""
    # Test configuration
    start_z: float
    reference_z: float
    distance_from_focus: float
    direction: str  # 'above' or 'below'

    # Parameters tested
    n_steps: int
    search_range_um: float
    interp_kind: str
    score_metric_name: str
    autofocus_method: str  # 'standard' or 'adaptive'

    # Results
    success: bool
    final_z: float
    z_error: float  # Difference from reference_z
    duration_ms: float

    # Validation metrics
    peak_valid: bool
    quality_score: float
    peak_prominence: float
    symmetry_score: float

    # Optional diagnostics
    message: str = ""
    raw_scores: List[float] = field(default_factory=list)
    z_positions: List[float] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark grid search."""
    # Standard autofocus parameters to test
    n_steps_values: List[int] = field(default_factory=lambda: [11, 15, 21, 25, 35])
    search_range_values: List[float] = field(default_factory=lambda: [15.0, 25.0, 35.0, 50.0])
    interp_kind_values: List[str] = field(default_factory=lambda: ['linear', 'quadratic', 'cubic'])
    score_metric_names: List[str] = field(default_factory=lambda: [
        'laplacian_variance', 'sobel', 'brenner_gradient'
    ])

    # Adaptive autofocus parameters
    adaptive_initial_step_values: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])
    adaptive_min_step_values: List[float] = field(default_factory=lambda: [1.0, 2.0])

    # Distance offsets to test (um from true focus)
    test_distances: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0, 50.0])

    # Test both above and below focus
    test_both_directions: bool = True

    # Number of repetitions per configuration
    repetitions: int = 1

    # Interpolation strength (usually fixed)
    interp_strength: int = 100

    # Methods to test
    test_standard: bool = True
    test_adaptive: bool = True


class AutofocusBenchmark:
    """
    Systematic benchmarking of autofocus parameters.

    Performs grid search across parameter combinations to find optimal
    settings for different focus distances and objectives.

    SAFETY: All Z movements are validated against hardcoded safety limits
    to prevent objective-sample collision on this upright microscope.
    """

    # Score metric mapping
    SCORE_METRICS = {
        'laplacian_variance': AutofocusUtils.autofocus_profile_laplacian_variance,
        'sobel': AutofocusUtils.autofocus_profile_sobel,
        'brenner_gradient': AutofocusUtils.autofocus_profile_brenner_gradient,
        'robust_sharpness': AutofocusUtils.autofocus_profile_robust_sharpness_metric,
        'hybrid_sharpness': AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
    }

    def _get_safety_limit_for_objective(self, objective: Optional[str] = None) -> float:
        """
        Get the Z safety limit for the specified objective.

        Args:
            objective: Objective identifier (e.g., "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001")

        Returns:
            Most negative Z value allowed (um). More negative = closer to objective.
        """
        if objective:
            # Check for objective-specific limit
            objective_upper = objective.upper()
            for key, limit in OBJECTIVE_SAFETY_LIMITS_UM.items():
                if key in objective_upper:
                    return limit

        # Fall back to absolute limit
        return Z_ABSOLUTE_SAFETY_LIMIT_UM

    def _validate_z_safe(
        self,
        z_position: float,
        objective: Optional[str] = None,
        context: str = ""
    ) -> bool:
        """
        Check if a Z position is safe (won't crash objective into sample).

        Args:
            z_position: Target Z position in um
            objective: Objective identifier for objective-specific limits
            context: Description of the operation for logging

        Returns:
            True if position is safe, False otherwise

        Raises:
            ZSafetyError: If position would violate safety limits
        """
        safety_limit = self._get_safety_limit_for_objective(objective)

        # On upright microscope: more negative Z = closer to objective = DANGER
        if z_position < safety_limit:
            msg = (
                f"SAFETY VIOLATION: Z={z_position:.2f}um exceeds safety limit "
                f"of {safety_limit:.2f}um for objective '{objective or 'default'}'. "
                f"Context: {context}"
            )
            self.logger.error("!" * 60)
            self.logger.error(msg)
            self.logger.error("!" * 60)
            raise ZSafetyError(msg)

        return True

    def _validate_benchmark_range_safe(
        self,
        reference_z: float,
        test_distances: List[float],
        max_search_range: float,
        objective: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Validate that all planned benchmark positions are within safety limits.

        This pre-flight check ensures the entire benchmark can run safely before
        starting any movements.

        Args:
            reference_z: The reference (in-focus) Z position
            test_distances: List of distances to test from reference
            max_search_range: Maximum autofocus search range that will be used
            objective: Objective identifier

        Returns:
            Tuple of (is_safe, message)
        """
        safety_limit = self._get_safety_limit_for_objective(objective)

        # Calculate the most extreme Z position that could be reached:
        # - Start from reference_z
        # - Move by largest test distance (in the dangerous direction)
        # - Add half the max search range (autofocus searches both directions)
        # - Add overshoot margin for safety

        max_distance = max(test_distances) if test_distances else 50.0

        # On upright: more negative = closer to objective
        # When testing "above" focus, we move MORE negative
        most_extreme_z = reference_z - max_distance - (max_search_range / 2) - AUTOFOCUS_OVERSHOOT_MARGIN_UM

        self.logger.info(f"Safety pre-flight check:")
        self.logger.info(f"  Reference Z: {reference_z:.2f} um")
        self.logger.info(f"  Max test distance: {max_distance:.2f} um")
        self.logger.info(f"  Max search range: {max_search_range:.2f} um")
        self.logger.info(f"  Overshoot margin: {AUTOFOCUS_OVERSHOOT_MARGIN_UM:.2f} um")
        self.logger.info(f"  Most extreme Z possible: {most_extreme_z:.2f} um")
        self.logger.info(f"  Safety limit for {objective or 'default'}: {safety_limit:.2f} um")

        if most_extreme_z < safety_limit:
            margin_needed = safety_limit - most_extreme_z
            msg = (
                f"UNSAFE: Benchmark would reach Z={most_extreme_z:.2f}um, "
                f"which exceeds safety limit of {safety_limit:.2f}um by {margin_needed:.2f}um. "
                f"Reduce test distances or search range."
            )
            return False, msg

        safety_margin = most_extreme_z - safety_limit
        msg = f"SAFE: {safety_margin:.2f}um margin to safety limit"
        self.logger.info(f"  {msg}")
        return True, msg

    def _safe_move_to_z(
        self,
        xy_pos: Tuple[float, float],
        z_position: float,
        objective: Optional[str] = None,
        context: str = ""
    ):
        """
        Move to a Z position with safety validation.

        Args:
            xy_pos: Tuple of (x, y) position
            z_position: Target Z position
            objective: Objective identifier for safety limits
            context: Description of the operation for logging

        Raises:
            ZSafetyError: If the position would violate safety limits
        """
        self._validate_z_safe(z_position, objective, context)
        self.hardware.move_to_position(Position(xy_pos[0], xy_pos[1], z_position))

    def __init__(
        self,
        hardware,
        config_manager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize benchmark runner.

        Args:
            hardware: PycromanagerHardware instance
            config_manager: ConfigManager instance
            logger: Optional logger instance
        """
        self.hardware = hardware
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)

        # Results storage
        self.results: List[BenchmarkResult] = []
        self.benchmark_start_time: Optional[datetime] = None

        # Current objective (set during run_benchmark)
        self._current_objective: Optional[str] = None

    def run_benchmark(
        self,
        reference_z: float,
        config: Optional[BenchmarkConfig] = None,
        output_path: Optional[str] = None,
        objective: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete autofocus benchmark.

        Args:
            reference_z: Known good focus Z position (user must verify this is in focus)
            config: BenchmarkConfig with parameters to test (uses defaults if None)
            output_path: Directory to save results (creates timestamped subdir)
            objective: Objective identifier for logging and safety limits
            progress_callback: Optional callback function(current_trial, total_trials, status_message)
                              Called after each trial to report progress. Useful for socket
                              communication to keep connections alive during long benchmarks.

        Returns:
            Dict with summary statistics and path to detailed results

        Raises:
            ZSafetyError: If benchmark would exceed safety limits
        """
        if config is None:
            config = BenchmarkConfig()

        # Store objective for safety checks throughout benchmark
        self._current_objective = objective

        self.benchmark_start_time = datetime.now()
        self.results = []

        # Setup output directory
        if output_path:
            output_dir = Path(output_path)
            timestamp = self.benchmark_start_time.strftime("%Y%m%d_%H%M%S")
            results_dir = output_dir / f"autofocus_benchmark_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir = None

        # Get current XY position (keep constant during benchmark)
        current_pos = self.hardware.get_current_position()
        xy_pos = (current_pos.x, current_pos.y)

        self.logger.info("=" * 60)
        self.logger.info("AUTOFOCUS BENCHMARK STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Reference Z (true focus): {reference_z:.2f} um")
        self.logger.info(f"XY Position: ({xy_pos[0]:.2f}, {xy_pos[1]:.2f})")
        if objective:
            self.logger.info(f"Objective: {objective}")
            self.logger.info(f"Safety limit: {self._get_safety_limit_for_objective(objective):.2f} um")
        self.logger.info(f"Test distances: {config.test_distances} um")
        self.logger.info(f"Directions: {'both' if config.test_both_directions else 'below only'}")

        # =====================================================================
        # SAFETY PRE-FLIGHT CHECK
        # =====================================================================
        # Validate that all planned movements are within safety limits BEFORE
        # starting any stage movements
        max_search_range = max(config.search_range_values) if config.test_standard else 20.0
        is_safe, safety_msg = self._validate_benchmark_range_safe(
            reference_z,
            config.test_distances,
            max_search_range,
            objective
        )

        if not is_safe:
            self.logger.error("!" * 60)
            self.logger.error("BENCHMARK ABORTED - SAFETY CHECK FAILED")
            self.logger.error(safety_msg)
            self.logger.error("!" * 60)
            raise ZSafetyError(safety_msg)

        self.logger.info("-" * 60)

        # Calculate total trials
        total_trials = self._calculate_total_trials(config)
        self.logger.info(f"Total trials to run: {total_trials}")
        self.logger.info("-" * 60)

        trial_count = 0

        # Build test positions
        test_positions = self._build_test_positions(reference_z, config)

        for start_z, distance, direction in test_positions:
            self.logger.info(f"\n--- Testing from Z={start_z:.2f} um ({distance:.1f} um {direction} focus) ---")

            # Test standard autofocus if enabled
            if config.test_standard:
                trial_count = self._run_standard_af_grid(
                    reference_z, start_z, distance, direction,
                    xy_pos, config, trial_count, total_trials,
                    progress_callback
                )

            # Test adaptive autofocus if enabled
            if config.test_adaptive:
                trial_count = self._run_adaptive_af_grid(
                    reference_z, start_z, distance, direction,
                    xy_pos, config, trial_count, total_trials,
                    progress_callback
                )

        # Return to reference Z (with safety check)
        self.logger.info(f"\nReturning to reference Z: {reference_z:.2f} um")
        self._safe_move_to_z(xy_pos, reference_z, objective, "Return to reference Z")

        # Generate summary
        summary = self._generate_summary(config, objective)

        # Save results
        if results_dir:
            self._save_results(results_dir, config, objective, summary)
            summary['results_directory'] = str(results_dir)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("AUTOFOCUS BENCHMARK COMPLETED")
        self.logger.info("=" * 60)
        self._log_summary(summary)

        return summary

    def run_quick_benchmark(
        self,
        reference_z: float,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a quick benchmark with reduced parameter space.

        Good for initial testing or when time is limited.
        Tests only key parameters at a few distances.
        """
        quick_config = BenchmarkConfig(
            n_steps_values=[15, 25],
            search_range_values=[25.0, 50.0],
            interp_kind_values=['quadratic'],
            score_metric_names=['laplacian_variance', 'brenner_gradient'],
            adaptive_initial_step_values=[10.0],
            adaptive_min_step_values=[2.0],
            test_distances=[10.0, 30.0],
            test_both_directions=False,
            repetitions=1,
        )

        return self.run_benchmark(reference_z, quick_config, output_path)

    def run_distance_sweep(
        self,
        reference_z: float,
        distances: List[float],
        output_path: Optional[str] = None,
        n_steps: int = 21,
        search_range: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Test autofocus at many distances with fixed parameters.

        Useful for understanding how autofocus performance degrades
        with distance from focus.

        Args:
            reference_z: Known good focus position
            distances: List of distances to test (um)
            output_path: Where to save results
            n_steps: Fixed n_steps value
            search_range: Fixed search range (um)
        """
        distance_config = BenchmarkConfig(
            n_steps_values=[n_steps],
            search_range_values=[search_range],
            interp_kind_values=['quadratic'],
            score_metric_names=['laplacian_variance'],
            adaptive_initial_step_values=[10.0],
            adaptive_min_step_values=[2.0],
            test_distances=distances,
            test_both_directions=True,
            repetitions=1,
        )

        return self.run_benchmark(reference_z, distance_config, output_path)

    def _calculate_total_trials(self, config: BenchmarkConfig) -> int:
        """Calculate total number of trials for the benchmark.

        Accounts for skipped impossible combinations where search_range < distance.
        """
        directions = 2 if config.test_both_directions else 1

        standard_trials = 0
        if config.test_standard:
            # Count trials per distance, only including valid range/distance combinations
            for distance in config.test_distances:
                valid_ranges = [r for r in config.search_range_values if r >= distance]
                trials_for_distance = (
                    len(config.n_steps_values) *
                    len(valid_ranges) *
                    len(config.interp_kind_values) *
                    len(config.score_metric_names) *
                    directions *
                    config.repetitions
                )
                standard_trials += trials_for_distance

        adaptive_trials = 0
        if config.test_adaptive:
            # Adaptive doesn't have the same range/distance constraint
            n_positions = len(config.test_distances) * directions
            adaptive_trials = (
                len(config.adaptive_initial_step_values) *
                len(config.adaptive_min_step_values) *
                len(config.score_metric_names) *
                n_positions *
                config.repetitions
            )

        return standard_trials + adaptive_trials

    def _build_test_positions(
        self,
        reference_z: float,
        config: BenchmarkConfig
    ) -> List[Tuple[float, float, str]]:
        """Build list of (start_z, distance, direction) tuples to test."""
        positions = []

        for distance in config.test_distances:
            # Test below focus (positive Z offset typically)
            positions.append((reference_z + distance, distance, 'below'))

            if config.test_both_directions:
                # Test above focus
                positions.append((reference_z - distance, distance, 'above'))

        return positions

    def _run_standard_af_grid(
        self,
        reference_z: float,
        start_z: float,
        distance: float,
        direction: str,
        xy_pos: Tuple[float, float],
        config: BenchmarkConfig,
        trial_count: int,
        total_trials: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Run grid search for standard autofocus parameters.

        Note: Skips combinations where search_range < distance, as these
        cannot possibly find the true focus position.
        """

        for n_steps in config.n_steps_values:
            for search_range in config.search_range_values:
                # Skip impossible combinations: if search range is smaller than
                # distance from focus, autofocus cannot reach the true focus
                if search_range < distance:
                    self.logger.debug(
                        f"Skipping: range={search_range}um < distance={distance}um (impossible)"
                    )
                    continue

                for interp_kind in config.interp_kind_values:
                    for metric_name in config.score_metric_names:
                        for rep in range(config.repetitions):
                            trial_count += 1

                            self.logger.info(
                                f"Trial {trial_count}/{total_trials}: "
                                f"standard AF, n={n_steps}, range={search_range}um, "
                                f"interp={interp_kind}, metric={metric_name}"
                            )

                            result = self._run_single_standard_trial(
                                reference_z, start_z, distance, direction,
                                xy_pos, n_steps, search_range, interp_kind,
                                metric_name, config.interp_strength
                            )

                            self.results.append(result)
                            self._log_trial_result(result)

                            # Send progress update to keep connection alive
                            if progress_callback:
                                status = "OK" if result.success else "FAIL"
                                msg = f"Trial {trial_count}/{total_trials} [{status}] error={result.z_error:.1f}um"
                                try:
                                    progress_callback(trial_count, total_trials, msg)
                                except Exception as e:
                                    self.logger.warning(f"Progress callback failed: {e}")

        return trial_count

    def _run_adaptive_af_grid(
        self,
        reference_z: float,
        start_z: float,
        distance: float,
        direction: str,
        xy_pos: Tuple[float, float],
        config: BenchmarkConfig,
        trial_count: int,
        total_trials: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Run grid search for adaptive autofocus parameters."""

        for initial_step in config.adaptive_initial_step_values:
            for min_step in config.adaptive_min_step_values:
                for metric_name in config.score_metric_names:
                    for rep in range(config.repetitions):
                        trial_count += 1

                        self.logger.info(
                            f"Trial {trial_count}/{total_trials}: "
                            f"adaptive AF, init={initial_step}um, min={min_step}um, "
                            f"metric={metric_name}"
                        )

                        result = self._run_single_adaptive_trial(
                            reference_z, start_z, distance, direction,
                            xy_pos, initial_step, min_step, metric_name
                        )

                        self.results.append(result)
                        self._log_trial_result(result)

                        # Send progress update to keep connection alive
                        if progress_callback:
                            status = "OK" if result.success else "FAIL"
                            msg = f"Trial {trial_count}/{total_trials} [{status}] error={result.z_error:.1f}um"
                            try:
                                progress_callback(trial_count, total_trials, msg)
                            except Exception as e:
                                self.logger.warning(f"Progress callback failed: {e}")

        return trial_count

    def _run_single_standard_trial(
        self,
        reference_z: float,
        start_z: float,
        distance: float,
        direction: str,
        xy_pos: Tuple[float, float],
        n_steps: int,
        search_range: float,
        interp_kind: str,
        metric_name: str,
        interp_strength: int,
    ) -> BenchmarkResult:
        """Run a single standard autofocus trial and measure results."""

        # Move to start position WITH SAFETY CHECK
        try:
            self._safe_move_to_z(
                xy_pos, start_z, self._current_objective,
                f"Standard AF trial: move to start Z={start_z:.2f}um"
            )
        except ZSafetyError as e:
            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=n_steps,
                search_range_um=search_range,
                interp_kind=interp_kind,
                score_metric_name=metric_name,
                autofocus_method='standard',
                success=False,
                final_z=start_z,
                z_error=distance,
                duration_ms=0,
                peak_valid=False,
                quality_score=0.0,
                peak_prominence=0.0,
                symmetry_score=0.0,
                message=f"SAFETY: {str(e)}",
            )
        time.sleep(0.1)  # Brief settle time

        # Get score metric function
        score_metric = self.SCORE_METRICS.get(
            metric_name,
            AutofocusUtils.autofocus_profile_laplacian_variance
        )

        # Run autofocus with timing
        start_time = time.perf_counter()

        try:
            result_z = self.hardware.autofocus(
                n_steps=n_steps,
                search_range=search_range,
                interp_strength=interp_strength,
                interp_kind=interp_kind,
                score_metric=score_metric,
                pop_a_plot=False,
                move_stage_to_estimate=True,
                raise_on_invalid_peak=False,  # Don't raise, we want to measure failures
            )

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Check if autofocus returned failure dict
            if isinstance(result_z, dict) and not result_z.get('success', True):
                return BenchmarkResult(
                    start_z=start_z,
                    reference_z=reference_z,
                    distance_from_focus=distance,
                    direction=direction,
                    n_steps=n_steps,
                    search_range_um=search_range,
                    interp_kind=interp_kind,
                    score_metric_name=metric_name,
                    autofocus_method='standard',
                    success=False,
                    final_z=result_z.get('attempted_z', start_z),
                    z_error=abs(result_z.get('attempted_z', start_z) - reference_z),
                    duration_ms=duration_ms,
                    peak_valid=False,
                    quality_score=result_z.get('quality_score', 0.0),
                    peak_prominence=result_z.get('peak_prominence', 0.0),
                    symmetry_score=result_z.get('validation', {}).get('symmetry_score', 0.0),
                    message=result_z.get('message', 'Autofocus failed'),
                )

            # Success case - but validate that we actually found the true focus
            final_z = float(result_z)
            z_error = abs(final_z - reference_z)

            # Get current position to verify
            actual_pos = self.hardware.get_current_position()

            # Determine if this is a TRUE success (close to reference)
            # vs autofocus found a local peak but not the true focus
            is_accurate = z_error <= MAX_ACCEPTABLE_Z_ERROR_UM

            # Check if the search range was even sufficient to reach the reference
            search_half = search_range / 2
            target_reachable = abs(start_z - reference_z) <= search_half

            if not target_reachable:
                message = (f"WARNING: Reference Z not reachable! Start={start_z:.2f}, "
                          f"Reference={reference_z:.2f}, Search range=+/-{search_half:.1f}um. "
                          f"Found local peak at Z={final_z:.2f}um, error={z_error:.2f}um")
            elif is_accurate:
                message = f"Focus found at Z={final_z:.2f}um, error={z_error:.2f}um"
            else:
                message = (f"Peak found but inaccurate: Z={final_z:.2f}um, "
                          f"error={z_error:.2f}um > {MAX_ACCEPTABLE_Z_ERROR_UM}um threshold")

            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=n_steps,
                search_range_um=search_range,
                interp_kind=interp_kind,
                score_metric_name=metric_name,
                autofocus_method='standard',
                success=is_accurate,  # Only true if within error threshold
                final_z=final_z,
                z_error=z_error,
                duration_ms=duration_ms,
                peak_valid=True,
                quality_score=1.0 if is_accurate else 0.5,
                peak_prominence=1.0,
                symmetry_score=1.0,
                message=message,
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=n_steps,
                search_range_um=search_range,
                interp_kind=interp_kind,
                score_metric_name=metric_name,
                autofocus_method='standard',
                success=False,
                final_z=start_z,
                z_error=distance,
                duration_ms=duration_ms,
                peak_valid=False,
                quality_score=0.0,
                peak_prominence=0.0,
                symmetry_score=0.0,
                message=f"Exception: {str(e)}",
            )

    def _run_single_adaptive_trial(
        self,
        reference_z: float,
        start_z: float,
        distance: float,
        direction: str,
        xy_pos: Tuple[float, float],
        initial_step: float,
        min_step: float,
        metric_name: str,
    ) -> BenchmarkResult:
        """Run a single adaptive autofocus trial and measure results."""

        # Move to start position WITH SAFETY CHECK
        try:
            self._safe_move_to_z(
                xy_pos, start_z, self._current_objective,
                f"Adaptive AF trial: move to start Z={start_z:.2f}um"
            )
        except ZSafetyError as e:
            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=0,
                search_range_um=initial_step * 2,
                interp_kind='quadratic',
                score_metric_name=metric_name,
                autofocus_method='adaptive',
                success=False,
                final_z=start_z,
                z_error=distance,
                duration_ms=0,
                peak_valid=False,
                quality_score=0.0,
                peak_prominence=0.0,
                symmetry_score=0.0,
                message=f"SAFETY: {str(e)}",
            )
        time.sleep(0.1)  # Brief settle time

        # Get score metric function
        score_metric = self.SCORE_METRICS.get(
            metric_name,
            AutofocusUtils.autofocus_profile_laplacian_variance
        )

        # Run adaptive autofocus with timing
        start_time = time.perf_counter()

        try:
            result_z = self.hardware.autofocus_adaptive_search(
                initial_step_size=initial_step,
                min_step_size=min_step,
                focus_threshold=0.95,
                max_total_steps=25,
                score_metric=score_metric,
                pop_a_plot=False,
                move_stage_to_estimate=True,
            )

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            final_z = float(result_z)
            z_error = abs(final_z - reference_z)

            # Determine if this is a TRUE success (close to reference)
            is_accurate = z_error <= MAX_ACCEPTABLE_Z_ERROR_UM

            if is_accurate:
                message = f"Adaptive focus at Z={final_z:.2f}um, error={z_error:.2f}um"
            else:
                message = (f"Adaptive found peak but inaccurate: Z={final_z:.2f}um, "
                          f"error={z_error:.2f}um > {MAX_ACCEPTABLE_Z_ERROR_UM}um threshold")

            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=0,  # Adaptive doesn't use fixed n_steps
                search_range_um=initial_step * 2,  # Approximate
                interp_kind='quadratic',  # Adaptive uses quadratic
                score_metric_name=metric_name,
                autofocus_method='adaptive',
                success=is_accurate,  # Only true if within error threshold
                final_z=final_z,
                z_error=z_error,
                duration_ms=duration_ms,
                peak_valid=True,
                quality_score=1.0 if is_accurate else 0.5,
                peak_prominence=1.0,
                symmetry_score=1.0,
                message=message,
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            return BenchmarkResult(
                start_z=start_z,
                reference_z=reference_z,
                distance_from_focus=distance,
                direction=direction,
                n_steps=0,
                search_range_um=initial_step * 2,
                interp_kind='quadratic',
                score_metric_name=metric_name,
                autofocus_method='adaptive',
                success=False,
                final_z=start_z,
                z_error=distance,
                duration_ms=duration_ms,
                peak_valid=False,
                quality_score=0.0,
                peak_prominence=0.0,
                symmetry_score=0.0,
                message=f"Exception: {str(e)}",
            )

    def _log_trial_result(self, result: BenchmarkResult):
        """Log a single trial result."""
        status = "OK" if result.success else "FAIL"
        self.logger.info(
            f"  [{status}] {result.duration_ms:.0f}ms, "
            f"error={result.z_error:.2f}um, "
            f"final_z={result.final_z:.2f}um"
        )

    def _generate_summary(
        self,
        config: BenchmarkConfig,
        objective: Optional[str]
    ) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""

        if not self.results:
            return {"error": "No results to summarize"}

        # Overall statistics
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        summary = {
            "benchmark_time": self.benchmark_start_time.isoformat() if self.benchmark_start_time else None,
            "objective": objective,
            "total_trials": len(self.results),
            "successful_trials": len(successful),
            "failed_trials": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
        }

        if successful:
            durations = [r.duration_ms for r in successful]
            errors = [r.z_error for r in successful]

            summary["timing_stats"] = {
                "mean_duration_ms": np.mean(durations),
                "median_duration_ms": np.median(durations),
                "min_duration_ms": np.min(durations),
                "max_duration_ms": np.max(durations),
                "std_duration_ms": np.std(durations),
            }

            summary["accuracy_stats"] = {
                "mean_z_error_um": np.mean(errors),
                "median_z_error_um": np.median(errors),
                "min_z_error_um": np.min(errors),
                "max_z_error_um": np.max(errors),
                "std_z_error_um": np.std(errors),
            }

        # Best configurations by speed (for successful trials)
        if successful:
            # Best for standard AF
            standard_results = [r for r in successful if r.autofocus_method == 'standard']
            if standard_results:
                fastest_standard = min(standard_results, key=lambda r: r.duration_ms)
                summary["fastest_standard"] = {
                    "n_steps": fastest_standard.n_steps,
                    "search_range_um": fastest_standard.search_range_um,
                    "interp_kind": fastest_standard.interp_kind,
                    "score_metric": fastest_standard.score_metric_name,
                    "duration_ms": fastest_standard.duration_ms,
                    "z_error_um": fastest_standard.z_error,
                    "distance_tested": fastest_standard.distance_from_focus,
                }

                # Most accurate standard
                most_accurate_standard = min(standard_results, key=lambda r: r.z_error)
                summary["most_accurate_standard"] = {
                    "n_steps": most_accurate_standard.n_steps,
                    "search_range_um": most_accurate_standard.search_range_um,
                    "interp_kind": most_accurate_standard.interp_kind,
                    "score_metric": most_accurate_standard.score_metric_name,
                    "duration_ms": most_accurate_standard.duration_ms,
                    "z_error_um": most_accurate_standard.z_error,
                    "distance_tested": most_accurate_standard.distance_from_focus,
                }

            # Best for adaptive AF
            adaptive_results = [r for r in successful if r.autofocus_method == 'adaptive']
            if adaptive_results:
                fastest_adaptive = min(adaptive_results, key=lambda r: r.duration_ms)
                summary["fastest_adaptive"] = {
                    "initial_step_um": fastest_adaptive.search_range_um / 2,
                    "score_metric": fastest_adaptive.score_metric_name,
                    "duration_ms": fastest_adaptive.duration_ms,
                    "z_error_um": fastest_adaptive.z_error,
                    "distance_tested": fastest_adaptive.distance_from_focus,
                }

        # Performance by distance
        summary["by_distance"] = {}
        for distance in set(r.distance_from_focus for r in self.results):
            dist_results = [r for r in successful if r.distance_from_focus == distance]
            if dist_results:
                summary["by_distance"][f"{distance:.1f}um"] = {
                    "success_rate": len(dist_results) / len([r for r in self.results if r.distance_from_focus == distance]),
                    "mean_duration_ms": np.mean([r.duration_ms for r in dist_results]),
                    "mean_z_error_um": np.mean([r.z_error for r in dist_results]),
                }

        # Performance by metric
        summary["by_metric"] = {}
        for metric in set(r.score_metric_name for r in self.results):
            metric_results = [r for r in successful if r.score_metric_name == metric]
            if metric_results:
                summary["by_metric"][metric] = {
                    "success_rate": len(metric_results) / len([r for r in self.results if r.score_metric_name == metric]),
                    "mean_duration_ms": np.mean([r.duration_ms for r in metric_results]),
                    "mean_z_error_um": np.mean([r.z_error for r in metric_results]),
                }

        # =====================================================================
        # COMPARATIVE ANALYSIS - Compare parameter values across all other settings
        # =====================================================================
        # These sections isolate each parameter's effect by averaging across
        # all other parameter combinations.

        summary["comparative_analysis"] = {}

        # Standard autofocus comparisons
        standard_results = [r for r in successful if r.autofocus_method == 'standard']
        if standard_results:
            # Compare n_steps values
            summary["comparative_analysis"]["by_n_steps"] = {}
            for n_steps in set(r.n_steps for r in standard_results):
                n_step_results = [r for r in standard_results if r.n_steps == n_steps]
                all_n_step = [r for r in self.results if r.autofocus_method == 'standard' and r.n_steps == n_steps]
                if n_step_results:
                    summary["comparative_analysis"]["by_n_steps"][str(n_steps)] = {
                        "trials": len(all_n_step),
                        "success_rate": len(n_step_results) / len(all_n_step) if all_n_step else 0,
                        "mean_duration_ms": float(np.mean([r.duration_ms for r in n_step_results])),
                        "std_duration_ms": float(np.std([r.duration_ms for r in n_step_results])),
                        "mean_z_error_um": float(np.mean([r.z_error for r in n_step_results])),
                        "std_z_error_um": float(np.std([r.z_error for r in n_step_results])),
                    }

            # Compare search_range values
            summary["comparative_analysis"]["by_search_range"] = {}
            for search_range in set(r.search_range_um for r in standard_results):
                range_results = [r for r in standard_results if r.search_range_um == search_range]
                all_range = [r for r in self.results if r.autofocus_method == 'standard' and r.search_range_um == search_range]
                if range_results:
                    summary["comparative_analysis"]["by_search_range"][f"{search_range:.1f}um"] = {
                        "trials": len(all_range),
                        "success_rate": len(range_results) / len(all_range) if all_range else 0,
                        "mean_duration_ms": float(np.mean([r.duration_ms for r in range_results])),
                        "std_duration_ms": float(np.std([r.duration_ms for r in range_results])),
                        "mean_z_error_um": float(np.mean([r.z_error for r in range_results])),
                        "std_z_error_um": float(np.std([r.z_error for r in range_results])),
                    }

            # Compare interp_kind values
            summary["comparative_analysis"]["by_interp_kind"] = {}
            for interp_kind in set(r.interp_kind for r in standard_results):
                interp_results = [r for r in standard_results if r.interp_kind == interp_kind]
                all_interp = [r for r in self.results if r.autofocus_method == 'standard' and r.interp_kind == interp_kind]
                if interp_results:
                    summary["comparative_analysis"]["by_interp_kind"][interp_kind] = {
                        "trials": len(all_interp),
                        "success_rate": len(interp_results) / len(all_interp) if all_interp else 0,
                        "mean_duration_ms": float(np.mean([r.duration_ms for r in interp_results])),
                        "std_duration_ms": float(np.std([r.duration_ms for r in interp_results])),
                        "mean_z_error_um": float(np.mean([r.z_error for r in interp_results])),
                        "std_z_error_um": float(np.std([r.z_error for r in interp_results])),
                    }

        # Adaptive autofocus comparisons
        adaptive_results = [r for r in successful if r.autofocus_method == 'adaptive']
        if adaptive_results:
            # Compare initial_step values (stored as search_range_um / 2)
            summary["comparative_analysis"]["by_initial_step"] = {}
            initial_steps = set(r.search_range_um / 2 for r in adaptive_results)
            for initial_step in initial_steps:
                step_results = [r for r in adaptive_results if abs(r.search_range_um / 2 - initial_step) < 0.1]
                all_step = [r for r in self.results if r.autofocus_method == 'adaptive' and abs(r.search_range_um / 2 - initial_step) < 0.1]
                if step_results:
                    summary["comparative_analysis"]["by_initial_step"][f"{initial_step:.1f}um"] = {
                        "trials": len(all_step),
                        "success_rate": len(step_results) / len(all_step) if all_step else 0,
                        "mean_duration_ms": float(np.mean([r.duration_ms for r in step_results])),
                        "std_duration_ms": float(np.std([r.duration_ms for r in step_results])),
                        "mean_z_error_um": float(np.mean([r.z_error for r in step_results])),
                        "std_z_error_um": float(np.std([r.z_error for r in step_results])),
                    }

        # Generate rankings for quick reference
        if standard_results:
            summary["rankings"] = {
                "fastest_metric": self._rank_by_field(summary["by_metric"], "mean_duration_ms", ascending=True),
                "most_accurate_metric": self._rank_by_field(summary["by_metric"], "mean_z_error_um", ascending=True),
                "fastest_n_steps": self._rank_by_field(
                    summary["comparative_analysis"].get("by_n_steps", {}), "mean_duration_ms", ascending=True
                ),
                "most_accurate_n_steps": self._rank_by_field(
                    summary["comparative_analysis"].get("by_n_steps", {}), "mean_z_error_um", ascending=True
                ),
                "fastest_interp": self._rank_by_field(
                    summary["comparative_analysis"].get("by_interp_kind", {}), "mean_duration_ms", ascending=True
                ),
                "most_accurate_interp": self._rank_by_field(
                    summary["comparative_analysis"].get("by_interp_kind", {}), "mean_z_error_um", ascending=True
                ),
            }

        return summary

    def _rank_by_field(
        self,
        data: Dict[str, Dict[str, Any]],
        field: str,
        ascending: bool = True
    ) -> List[str]:
        """Rank keys in data dict by a specific field value."""
        if not data:
            return []

        items = [(key, vals.get(field, float('inf'))) for key, vals in data.items()]
        items.sort(key=lambda x: x[1], reverse=not ascending)
        return [item[0] for item in items]

    def _save_results(
        self,
        results_dir: Path,
        config: BenchmarkConfig,
        objective: Optional[str],
        summary: Dict[str, Any]
    ):
        """Save benchmark results to files."""

        # Save detailed CSV with all trials
        csv_path = results_dir / "benchmark_results.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                # Remove large fields from CSV
                fieldnames = [f for f in fieldnames if f not in ['raw_scores', 'z_positions']]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = asdict(result)
                    row.pop('raw_scores', None)
                    row.pop('z_positions', None)
                    writer.writerow(row)

        self.logger.info(f"Detailed results saved: {csv_path}")

        # Save summary JSON
        json_path = results_dir / "benchmark_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Summary saved: {json_path}")

        # Save config
        config_path = results_dir / "benchmark_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        self.logger.info(f"Config saved: {config_path}")

    def _log_summary(self, summary: Dict[str, Any]):
        """Log summary statistics."""

        self.logger.info(f"Total trials: {summary.get('total_trials', 0)}")
        self.logger.info(f"Success rate: {summary.get('success_rate', 0):.1%}")

        if "timing_stats" in summary:
            ts = summary["timing_stats"]
            self.logger.info(f"Duration: {ts['mean_duration_ms']:.0f}ms mean, "
                           f"{ts['min_duration_ms']:.0f}-{ts['max_duration_ms']:.0f}ms range")

        if "accuracy_stats" in summary:
            acc = summary["accuracy_stats"]
            self.logger.info(f"Z error: {acc['mean_z_error_um']:.2f}um mean, "
                           f"{acc['min_z_error_um']:.2f}-{acc['max_z_error_um']:.2f}um range")

        if "fastest_standard" in summary:
            fs = summary["fastest_standard"]
            self.logger.info(f"\nFastest standard config:")
            self.logger.info(f"  n_steps={fs['n_steps']}, range={fs['search_range_um']}um, "
                           f"metric={fs['score_metric']}")
            self.logger.info(f"  Duration: {fs['duration_ms']:.0f}ms, Error: {fs['z_error_um']:.2f}um")

        if "fastest_adaptive" in summary:
            fa = summary["fastest_adaptive"]
            self.logger.info(f"\nFastest adaptive config:")
            self.logger.info(f"  initial_step={fa['initial_step_um']}um, metric={fa['score_metric']}")
            self.logger.info(f"  Duration: {fa['duration_ms']:.0f}ms, Error: {fa['z_error_um']:.2f}um")


def run_autofocus_benchmark_from_server(
    hardware,
    config_manager,
    reference_z: float,
    output_folder: str,
    test_distances: Optional[List[float]] = None,
    quick_mode: bool = False,
    objective: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Entry point for running benchmark from socket server.

    Args:
        hardware: PycromanagerHardware instance
        config_manager: ConfigManager instance
        reference_z: Known good focus Z position
        output_folder: Where to save results
        test_distances: List of distances to test (um from focus)
        quick_mode: If True, run reduced parameter space
        objective: Objective identifier for safety limits (e.g., "20X", "40X")
        logger: Optional logger
        progress_callback: Optional callback function(current_trial, total_trials, status_message)
                          Called after each trial. Used for socket progress updates.

    Returns:
        Dict with benchmark summary. On safety error, returns dict with
        'error' key containing the safety violation message.
    """
    benchmark = AutofocusBenchmark(hardware, config_manager, logger)

    try:
        if quick_mode:
            # Quick benchmark also supports progress callback
            quick_config = BenchmarkConfig(
                n_steps_values=[15, 25],
                search_range_values=[25.0, 50.0],
                interp_kind_values=['quadratic'],
                score_metric_names=['laplacian_variance', 'brenner_gradient'],
                adaptive_initial_step_values=[10.0],
                adaptive_min_step_values=[2.0],
                test_distances=[10.0, 30.0],
                test_both_directions=False,
                repetitions=1,
            )
            return benchmark.run_benchmark(
                reference_z, quick_config, output_folder,
                progress_callback=progress_callback
            )

        if test_distances:
            config = BenchmarkConfig(test_distances=test_distances)
            return benchmark.run_benchmark(
                reference_z, config, output_folder, objective,
                progress_callback=progress_callback
            )

        return benchmark.run_benchmark(
            reference_z, output_path=output_folder, objective=objective,
            progress_callback=progress_callback
        )

    except ZSafetyError as e:
        return {
            "error": str(e),
            "safety_violation": True,
            "total_trials": 0,
            "success_rate": 0,
        }
