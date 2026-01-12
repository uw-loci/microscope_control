"""
Core autofocus utilities for position calculation and focus metrics.

This module contains the main AutofocusUtils class with functions for:
- Calculating autofocus positions across tile grids
- Focus quality metrics (Laplacian, Sobel, Brenner, etc.)
- Tissue detection for autofocus decision-making
- Focus peak validation
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage.morphology
import skimage.filters
import logging

logger = logging.getLogger(__name__)


class AutofocusUtils:
    """Utilities for autofocus position calculation and focus metrics."""

    def __init__(self):
        pass

    @staticmethod
    def get_distance_sorted_xy_dict(positions):
        """Sort positions by radial distance from origin."""
        left_bottom = np.argmin(np.array([x[0] ** 2 + x[1] ** 2 for x in positions]))
        xa = positions[left_bottom]
        distances = np.round(cdist([xa], positions).ravel(), 2)
        positions_d = {ix: (positions[ix], distances[ix]) for ix in range(len(distances))}
        positions_d = dict(sorted(positions_d.items(), key=lambda item: item[1][1]))
        return positions_d

    @staticmethod
    def get_autofocus_positions(
        fov: Tuple[float, float], positions: List[Tuple[float, float]], n_tiles: float
    ) -> Tuple[List[int], float]:
        """
        Determine which tile positions require autofocus.

        For grids with >= 9 tiles, the first autofocus position is moved
        1 diagonal FOV inward from the starting corner to avoid focusing
        on areas outside the tissue (e.g., buffer regions).

        Args:
            fov: Field of view (x, y) in micrometers
            positions: List of (x, y) tile positions
            n_tiles: Number of tiles between autofocus positions

        Returns:
            Tuple of (autofocus position indices, minimum distance)
        """
        fov_x, fov_y = fov

        # Compute the minimum required distance between autofocus positions
        # Use average FOV dimension (not diagonal) for consistent spacing
        af_min_distance = ((fov_x + fov_y) / 2) * n_tiles

        if not positions:
            return [], af_min_distance

        # Determine the first autofocus position index
        # For grids with >= 9 tiles, move 1 diagonal FOV inward to avoid edge issues
        first_af_index = 0

        if len(positions) >= 9:
            # Calculate direction from start corner toward grid center
            start_pos = np.array(positions[0])
            center_pos = np.mean(positions, axis=0)
            direction = center_pos - start_pos

            # Normalize direction and scale by 1 FOV diagonal
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                # Move 1 diagonal FOV inward
                diagonal_fov = np.sqrt(fov_x**2 + fov_y**2)
                target_pos = start_pos + direction * diagonal_fov

                # Find the tile closest to the target position
                distances = cdist([target_pos], positions)[0]
                first_af_index = int(np.argmin(distances))

                logger.info(
                    f"Grid has {len(positions)} tiles (>= 9) - "
                    f"moving first AF from tile 0 to tile {first_af_index} "
                    f"(1 diagonal FOV inward)"
                )

        if first_af_index == 0 and len(positions) < 9:
            logger.debug(f"Grid has {len(positions)} tiles (< 9) - keeping first AF at tile 0")

        # Build autofocus position list starting with the computed first position
        af_positions = [first_af_index]
        af_xy_pos = positions[first_af_index]

        for ix, pos in enumerate(positions):
            if ix == first_af_index:
                continue  # Already added as first AF position

            # Calculate distance from last AF position
            dist_to_last_af_xy_pos = cdist([af_xy_pos], [pos])[0][0]

            # If we've moved more than the AF minimum distance, add new AF point
            if dist_to_last_af_xy_pos > af_min_distance:
                af_positions.append(ix)
                af_xy_pos = pos  # Update last autofocus position

        return af_positions, af_min_distance

    @staticmethod
    def visualize_autofocus_locations(
        fov: Tuple[float, float], positions: List[Tuple[float, float]], ntiles: float = 1.35
    ):
        """Visualize autofocus positions on a plot."""
        af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
            fov, positions, ntiles
        )
        ax = plt.subplot(111)
        for ix, pos in enumerate(positions):
            if ix in af_positions:
                crc = matplotlib.patches.Circle(
                    (pos[0], pos[1]),
                    af_min_distance,
                    fill=False,
                )
                ax.add_artist(crc)
                ax.plot(pos[0], pos[1], "s", label="Autofocus" if ix == af_positions[0] else "")
            else:
                ax.plot(pos[0], pos[1], "o", markeredgecolor="k", label="Tile" if ix == 1 else "")

        # Set axis limits
        xstd = 5
        lims = np.array(
            [
                [np.mean(positions, 0) - (np.std(positions, 0) * xstd)],
                [np.mean(positions, 0) + np.std(positions, 0) * xstd],
            ]
        ).T.ravel()
        ax.axis(tuple(lims))
        ax.set_aspect("equal")

        ax.set_title(f"Autofocus positions with {ntiles} tiles distance")
        ax.set_xlabel("X position (um)")
        ax.set_ylabel("Y position (um)")
        ax.legend()
        plt.show()
        return af_positions, af_min_distance

    @staticmethod
    def autofocus_profile_laplacian_variance(image: np.ndarray) -> float:
        """Fast general sharpness metric - ~5ms for 2500x1900."""
        laplacian = skimage.filters.laplace(image)
        return float(laplacian.var())

    @staticmethod
    def autofocus_profile_sobel(image: np.ndarray) -> float:
        """Fast general sharpness metric - ~5ms for 2500x1900."""
        sobel = skimage.filters.sobel(image)
        return float(sobel.var())

    @staticmethod
    def autofocus_profile_brenner_gradient(image: np.ndarray) -> float:
        """Fastest option - ~3ms for 2500x1900."""
        gy, gx = np.gradient(image.astype(np.float32))
        return float(np.mean(gx**2 + gy**2))

    @staticmethod
    def autofocus_profile_robust_sharpness_metric(image: np.ndarray) -> float:
        """Particle-resistant but slower - ~20ms for 2500x1900."""
        # Median filter to remove particles (this is the slow part)
        filtered = skimage.filters.median(image, skimage.morphology.disk(3))

        # Calculate sharpness on filtered image
        laplacian = skimage.filters.laplace(filtered)

        # Exclude very dark regions from calculation
        threshold = skimage.filters.threshold_otsu(image)
        mask = image > (threshold * 0.5)

        return float(laplacian[mask].var()) if mask.any() else float(laplacian.var())

    @staticmethod
    def autofocus_profile_hybrid_sharpness_metric(image: np.ndarray) -> float:
        """Compromise: Fast with some particle resistance - ~8ms."""
        # Gaussian blur to reduce particle influence (faster than median)
        smoothed = skimage.filters.gaussian(image, sigma=1.5)

        # Use Brenner gradient on smoothed image
        gy, gx = np.gradient(smoothed.astype(np.float32))
        gradient_magnitude = gx**2 + gy**2

        # Soft masking: reduce weight of very dark/bright regions
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        weight_mask = 1 - np.abs(normalized - 0.5) * 2  # Peak at mid-gray

        return float(np.mean(gradient_magnitude * weight_mask))

    @staticmethod
    def has_sufficient_tissue(
        image: np.ndarray,
        texture_threshold: float = 0.02,
        tissue_area_threshold: float = 0.15,
        modality: Optional[str] = None,
        logger=None,
        return_stats: bool = False,
        rgb_brightness_threshold: float = 225.0,
    ):
        """
        Determine if image has sufficient tissue texture for reliable autofocus.

        Args:
            image: Input image (grayscale or RGB)
            texture_threshold: Minimum texture variance (normalized)
            tissue_area_threshold: Minimum fraction of image that must contain tissue
            modality: Imaging modality for modality-specific adjustments
            logger: Optional logger instance
            return_stats: If True, return (bool, dict) with detection statistics
            rgb_brightness_threshold: Maximum average RGB brightness for tissue (default 225).
                Images brighter than this are considered blank/background. Set to None to disable.

        Returns:
            If return_stats=False: True if sufficient tissue is present for autofocus
            If return_stats=True: (bool, dict) where dict contains detection statistics
        """
        # EARLY REJECTION: Check RGB brightness to filter out blank tiles
        # Blank glass/background is very bright (RGB ~230-240)
        # Tissue is darker (RGB ~190-224)
        rgb_mean = None
        brightness_rejected = False

        if rgb_brightness_threshold is not None and len(image.shape) == 3:
            # Calculate mean RGB across entire image
            rgb_mean = np.mean(image, axis=(0, 1))
            avg_brightness = np.mean(rgb_mean)

            if avg_brightness > rgb_brightness_threshold:
                brightness_rejected = True
                if logger:
                    logger.info(
                        f"Blank tile detected: avg RGB brightness {avg_brightness:.1f} > {rgb_brightness_threshold:.1f} "
                        f"(RGB: [{rgb_mean[0]:.1f}, {rgb_mean[1]:.1f}, {rgb_mean[2]:.1f}])"
                    )

                if return_stats:
                    stats = {
                        "texture": 0.0,
                        "texture_threshold": texture_threshold,
                        "area": 0.0,
                        "area_threshold": tissue_area_threshold,
                        "sufficient_texture": False,
                        "sufficient_area": False,
                        "rgb_mean": rgb_mean.tolist() if rgb_mean is not None else None,
                        "avg_brightness": float(avg_brightness),
                        "brightness_threshold": rgb_brightness_threshold,
                        "brightness_rejected": True,
                    }
                    return False, stats
                else:
                    return False
        # Modality-specific parameter adjustments
        if modality:
            modality_lower = modality.lower()

            # Polarized light microscopy adjustments
            if "ppm" in modality_lower or "polarized" in modality_lower:
                # Polarized images can have wider intensity ranges and different tissue appearance
                # More inclusive tissue mask to capture birefringent structures
                tissue_mask_range = (0.05, 0.95)  # Wider range
                if texture_threshold == 0.02:  # Only adjust if using default
                    texture_threshold = 0.015  # Slightly more sensitive

            # Brightfield microscopy
            elif "bf" in modality_lower or "brightfield" in modality_lower:
                # Standard tissue detection works well for brightfield
                tissue_mask_range = (0.15, 0.85)  # Focus on typical tissue intensity

            # Multi-photon or SHG
            elif "shg" in modality_lower or "multiphoton" in modality_lower:
                # High contrast features, different background characteristics
                tissue_mask_range = (0.1, 0.9)
                if texture_threshold == 0.02:
                    texture_threshold = 0.025  # Slightly less sensitive due to sparse features

            else:
                # Default mask range
                tissue_mask_range = (0.1, 0.9)
        else:
            tissue_mask_range = (0.1, 0.9)
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = np.mean(image, axis=2).astype(np.float32)
        elif len(image.shape) == 2:
            # Handle Bayer pattern
            if image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0:
                green1 = image[0::2, 0::2]
                green2 = image[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            else:
                img_gray = image.astype(np.float32)
        else:
            img_gray = image.astype(np.float32)

        # Normalize image to [0, 1] range
        img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-10)
        norm_p5 = np.percentile(img_norm, 5)
        norm_p95 = np.percentile(img_norm, 95)

        if logger:
            logger.debug(f"Normalized percentiles - 5th: {norm_p5:.3f}, 95th: {norm_p95:.3f}")

        # Adaptive tissue mask based on actual data distribution
        if norm_p95 - norm_p5 < 0.5:  # Very narrow distribution
            # Use percentile-based mask for low contrast images
            margin = 0.02
            tissue_mask = (img_norm > norm_p5 + margin) & (img_norm < norm_p95 - margin)
            if logger:
                logger.debug(
                    f"Using adaptive mask for narrow range: ({norm_p5 + margin:.3f}, {norm_p95 - margin:.3f})"
                )
        else:
            # Original modality-specific masks
            if modality and "ppm" in modality.lower():
                tissue_mask = (img_norm > 0.05) & (img_norm < 0.95)
            else:
                tissue_mask = (img_norm > 0.1) & (img_norm < 0.9)
        # Calculate local texture using gradient magnitude
        gy, gx = np.gradient(img_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Calculate overall texture strength
        texture_strength = np.std(gradient_magnitude)

        # Identify potential tissue regions using modality-specific intensity ranges
        tissue_mask = (img_norm > tissue_mask_range[0]) & (img_norm < tissue_mask_range[1])

        # Calculate texture in tissue regions only
        if np.any(tissue_mask):
            tissue_texture = np.std(gradient_magnitude[tissue_mask])
            tissue_area_fraction = np.sum(tissue_mask) / tissue_mask.size
        else:
            tissue_texture = 0.0
            tissue_area_fraction = 0.0

        # Decision criteria
        sufficient_texture = tissue_texture > texture_threshold
        sufficient_area = tissue_area_fraction > tissue_area_threshold

        # Overall decision
        has_tissue = sufficient_texture and sufficient_area

        if logger:
            logger.debug(
                f"Tissue detection: texture={tissue_texture:.4f} (>{texture_threshold}), "
                f"area={tissue_area_fraction:.3f} (>{tissue_area_threshold}), "
                f"sufficient={has_tissue}"
            )

        if return_stats:
            stats = {
                "texture": tissue_texture,
                "texture_threshold": texture_threshold,
                "area": tissue_area_fraction,
                "area_threshold": tissue_area_threshold,
                "sufficient_texture": sufficient_texture,
                "sufficient_area": sufficient_area,
                "rgb_mean": rgb_mean.tolist() if rgb_mean is not None else None,
                "avg_brightness": float(np.mean(rgb_mean)) if rgb_mean is not None else None,
                "brightness_threshold": rgb_brightness_threshold,
                "brightness_rejected": brightness_rejected,
            }
            return has_tissue, stats
        else:
            return has_tissue

    @staticmethod
    def defer_autofocus_to_next_tile(
        current_pos_idx: int,
        original_af_positions: List[int],
        total_positions: int,
        af_min_distance: float,
        positions: List[Tuple[float, float]],
        logger=None,
    ) -> Optional[int]:
        """
        Find the next suitable tile position for autofocus when current tile lacks tissue.

        Args:
            current_pos_idx: Current tile index that was supposed to get autofocus
            original_af_positions: Original list of autofocus positions
            total_positions: Total number of tile positions
            af_min_distance: Minimum distance required between autofocus positions
            positions: List of (x, y) positions for all tiles
            logger: Optional logger instance

        Returns:
            Index of next tile to perform autofocus on, or None if no suitable tile found
        """
        if not positions or current_pos_idx >= len(positions):
            return None

        current_xy = positions[current_pos_idx]

        # Look ahead for the next tile that's far enough away and within bounds
        for candidate_idx in range(current_pos_idx + 1, total_positions):
            candidate_xy = positions[candidate_idx]

            # Check distance from current position
            distance = cdist([current_xy], [candidate_xy])[0][0]

            if distance >= af_min_distance * 0.7:  # Slightly relax distance requirement
                if logger:
                    logger.info(
                        f"Deferring autofocus from tile {current_pos_idx} to tile {candidate_idx} "
                        f"(distance: {distance:.1f} >= {af_min_distance * 0.7:.1f})"
                    )
                return candidate_idx

        # If no suitable position found nearby, try to find any position beyond minimum distance
        for candidate_idx in range(current_pos_idx + 1, min(current_pos_idx + 10, total_positions)):
            if logger:
                logger.warning(
                    f"No ideal autofocus position found, using tile {candidate_idx} as backup"
                )
            return candidate_idx

        if logger:
            logger.warning(
                f"Could not find suitable autofocus deferral position after tile {current_pos_idx}"
            )

        return None

    @staticmethod
    def test_tissue_detection(
        image: np.ndarray,
        modality: str = "unknown",
        texture_thresholds: List[float] = [0.01, 0.02, 0.03, 0.05],
        area_thresholds: List[float] = [0.10, 0.15, 0.20, 0.25],
        show_analysis: bool = True,
        logger=None,
    ) -> Dict[str, Any]:
        """
        Test tissue detection function with different threshold combinations.

        Args:
            image: Input image to analyze
            modality: Imaging modality name for reporting
            texture_thresholds: List of texture thresholds to test
            area_thresholds: List of area thresholds to test
            show_analysis: Whether to show detailed analysis
            logger: Optional logger instance

        Returns:
            Dictionary with analysis results and recommendations
        """
        import matplotlib.pyplot as plt

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            img_gray = np.mean(image, axis=2).astype(np.float32)
        elif len(image.shape) == 2:
            if image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0:
                green1 = image[0::2, 0::2]
                green2 = image[1::2, 1::2]
                img_gray = ((green1 + green2) / 2.0).astype(np.float32)
            else:
                img_gray = image.astype(np.float32)
        else:
            img_gray = image.astype(np.float32)

        # Normalize image
        img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-10)

        # Calculate gradient and tissue metrics
        gy, gx = np.gradient(img_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Intensity analysis for modality-specific insights
        intensity_stats = {
            "min": float(img_norm.min()),
            "max": float(img_norm.max()),
            "mean": float(img_norm.mean()),
            "std": float(img_norm.std()),
            "median": float(np.median(img_norm)),
        }

        # Gradient analysis
        gradient_stats = {
            "mean": float(gradient_magnitude.mean()),
            "std": float(gradient_magnitude.std()),
            "max": float(gradient_magnitude.max()),
            "p95": float(np.percentile(gradient_magnitude, 95)),
        }

        # Test different tissue masks for modality analysis
        tissue_masks = {
            "conservative": (img_norm > 0.1) & (img_norm < 0.9),  # Original
            "brightfield_like": (img_norm > 0.2) & (img_norm < 0.8),  # Typical tissue range
            "polarized_inclusive": (img_norm > 0.05)
            & (img_norm < 0.95),  # Wider range for polarized
            "high_contrast": (img_norm > 0.15) & (img_norm < 0.85),  # Focus on mid-range
        }

        mask_analysis = {}
        for mask_name, mask in tissue_masks.items():
            if np.any(mask):
                mask_texture = np.std(gradient_magnitude[mask])
                mask_area = np.sum(mask) / mask.size
            else:
                mask_texture = 0.0
                mask_area = 0.0

            mask_analysis[mask_name] = {"texture": mask_texture, "area_fraction": mask_area}

        # Test threshold combinations
        results_matrix = []
        for tex_thresh in texture_thresholds:
            for area_thresh in area_thresholds:
                result = AutofocusUtils.has_sufficient_tissue(
                    image, tex_thresh, area_thresh, logger=None
                )
                results_matrix.append(
                    {
                        "texture_threshold": tex_thresh,
                        "area_threshold": area_thresh,
                        "has_tissue": result,
                    }
                )

        # Analysis summary
        analysis_summary = {
            "modality": modality,
            "image_shape": image.shape,
            "intensity_stats": intensity_stats,
            "gradient_stats": gradient_stats,
            "mask_analysis": mask_analysis,
            "threshold_results": results_matrix,
            "recommendations": {},
        }

        # Generate recommendations based on analysis
        best_mask = max(mask_analysis.keys(), key=lambda k: mask_analysis[k]["texture"])
        analysis_summary["recommendations"] = {
            "best_tissue_mask": best_mask,
            "suggested_texture_threshold": max(0.01, gradient_stats["std"] * 0.5),
            "suggested_area_threshold": max(0.1, mask_analysis[best_mask]["area_fraction"] * 0.5),
            "intensity_range": f"{intensity_stats['min']:.3f} - {intensity_stats['max']:.3f}",
            "has_good_contrast": intensity_stats["std"] > 0.15,
        }

        if show_analysis and logger:
            logger.info(f"=== TISSUE DETECTION TEST: {modality.upper()} ===")
            logger.info(f"Image shape: {image.shape}")
            logger.info(
                f"Intensity range: {intensity_stats['min']:.3f} - {intensity_stats['max']:.3f} (std: {intensity_stats['std']:.3f})"
            )
            logger.info(
                f"Gradient stats: mean={gradient_stats['mean']:.4f}, std={gradient_stats['std']:.4f}"
            )

            logger.info("Tissue mask analysis:")
            for mask_name, stats in mask_analysis.items():
                logger.info(
                    f"  {mask_name}: texture={stats['texture']:.4f}, area={stats['area_fraction']:.3f}"
                )

            logger.info("Threshold test results:")
            for result in results_matrix:
                status = "PASS" if result["has_tissue"] else "FAIL"
                logger.info(
                    f"  tex={result['texture_threshold']:.3f}, area={result['area_threshold']:.3f} -> {status}"
                )

            logger.info(f"Recommendations:")
            logger.info(f"  Best mask: {analysis_summary['recommendations']['best_tissue_mask']}")
            logger.info(
                f"  Suggested texture threshold: {analysis_summary['recommendations']['suggested_texture_threshold']:.4f}"
            )
            logger.info(
                f"  Suggested area threshold: {analysis_summary['recommendations']['suggested_area_threshold']:.3f}"
            )

        return analysis_summary

    @staticmethod
    def validate_focus_peak(z_positions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """
        Validate that the focus curve has a proper peak suitable for autofocus.

        A good focus peak should have:
        1. A clear maximum that stands out from neighboring values
        2. Gradual increase leading up to the peak
        3. Gradual decrease after the peak
        4. Reasonable symmetry around the peak

        Args:
            z_positions: Array of Z positions sampled
            scores: Array of focus scores at each position

        Returns:
            Dict containing:
                - is_valid: bool - Whether peak passes quality checks
                - peak_prominence: float - How much peak stands out (0-1 normalized)
                - has_ascending: bool - Has increasing trend before peak
                - has_descending: bool - Has decreasing trend after peak
                - symmetry_score: float - Measure of left/right symmetry (0-1, 1=perfect)
                - quality_score: float - Overall quality score (0-1)
                - warnings: List[str] - List of quality issues found
                - message: str - Human-readable summary
        """
        result = {
            "is_valid": False,
            "peak_prominence": 0.0,
            "has_ascending": False,
            "has_descending": False,
            "symmetry_score": 0.0,
            "quality_score": 0.0,
            "warnings": [],
            "message": ""
        }

        if len(scores) < 5:
            result["warnings"].append("Too few samples for reliable peak detection")
            result["message"] = "Insufficient data points for peak validation"
            return result

        # Find peak position
        peak_idx = np.argmax(scores)
        peak_score = scores[peak_idx]
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        score_range = max_score - min_score
        score_std = np.std(scores)

        # 1. Check absolute score variation (detect flat/noisy curves)
        # A proper focus curve should have significant variation
        relative_range = score_range / mean_score if mean_score > 0 else 0

        # CRITICAL: Check for minimum absolute variation
        # If score range is too small, it's just noise with no real focus gradient
        # Note: These are conservative thresholds - adjust based on your microscope/metric
        MIN_ABSOLUTE_RANGE = 0.5   # Minimum score range (was 2.0, too strict)
        MIN_RELATIVE_RANGE = 0.005 # Minimum 0.5% variation (was 5%, too strict)

        if score_range < MIN_ABSOLUTE_RANGE:
            result["warnings"].append(
                f"Insufficient absolute score variation ({score_range:.2f} < {MIN_ABSOLUTE_RANGE})")
            result["message"] = f"No focus gradient detected - score range too small ({score_range:.2f})"
            return result

        if relative_range < MIN_RELATIVE_RANGE:
            result["warnings"].append(
                f"Insufficient relative score variation ({relative_range:.2%} < {MIN_RELATIVE_RANGE:.0%})")
            result["message"] = f"No focus gradient detected - scores too flat ({relative_range:.2%} variation)"
            return result

        # 2. Check peak prominence (how much it stands out within the range)
        result["peak_prominence"] = (peak_score - mean_score) / score_range

        if result["peak_prominence"] < 0.2:
            result["warnings"].append(f"Peak prominence too low ({result['peak_prominence']:.2f})")

        # 3. Check for ascending trend before peak
        if peak_idx >= 2:
            # Count how many points before peak show increasing trend
            ascending_count = 0
            for i in range(peak_idx):
                if i == 0 or scores[i] >= scores[i-1]:
                    ascending_count += 1
            result["has_ascending"] = (ascending_count / peak_idx) >= 0.5
        else:
            result["warnings"].append("Peak too close to start - cannot verify ascending trend")
            result["has_ascending"] = False

        # 4. Check for descending trend after peak
        if peak_idx < len(scores) - 2:
            # Count how many points after peak show decreasing trend
            descending_count = 0
            for i in range(peak_idx + 1, len(scores)):
                if scores[i] <= scores[i-1]:
                    descending_count += 1
            points_after = len(scores) - peak_idx - 1
            result["has_descending"] = (descending_count / points_after) >= 0.5
        else:
            result["warnings"].append("Peak too close to end - cannot verify descending trend")
            result["has_descending"] = False

        # 5. Check symmetry around peak
        # Compare left and right side score ranges
        left_scores = scores[:peak_idx] if peak_idx > 0 else np.array([])
        right_scores = scores[peak_idx+1:] if peak_idx < len(scores)-1 else np.array([])

        if len(left_scores) > 0 and len(right_scores) > 0:
            left_range = np.max(left_scores) - np.min(left_scores) if len(left_scores) > 1 else 0
            right_range = np.max(right_scores) - np.min(right_scores) if len(right_scores) > 1 else 0

            if left_range + right_range > 0:
                result["symmetry_score"] = 1.0 - abs(left_range - right_range) / (left_range + right_range)
            else:
                result["symmetry_score"] = 1.0  # Both sides flat = perfect symmetry
        else:
            result["warnings"].append("Peak at edge - cannot assess symmetry")
            result["symmetry_score"] = 0.0

        # 6. Calculate overall quality score
        weights = {
            "prominence": 0.4,
            "ascending": 0.2,
            "descending": 0.2,
            "symmetry": 0.2
        }

        result["quality_score"] = (
            weights["prominence"] * result["peak_prominence"] +
            weights["ascending"] * (1.0 if result["has_ascending"] else 0.0) +
            weights["descending"] * (1.0 if result["has_descending"] else 0.0) +
            weights["symmetry"] * result["symmetry_score"]
        )

        # 7. Determine if peak is valid (passes minimum quality threshold)
        MIN_QUALITY_THRESHOLD = 0.5
        MIN_PROMINENCE = 0.15

        result["is_valid"] = (
            result["quality_score"] >= MIN_QUALITY_THRESHOLD and
            result["peak_prominence"] >= MIN_PROMINENCE and
            (result["has_ascending"] or result["has_descending"])  # At least one side must show trend
        )

        # 8. Generate human-readable message
        if result["is_valid"]:
            result["message"] = f"Valid focus peak detected (quality: {result['quality_score']:.2f})"
        else:
            issues = []
            if result["quality_score"] < MIN_QUALITY_THRESHOLD:
                issues.append(f"low quality score ({result['quality_score']:.2f})")
            if result["peak_prominence"] < MIN_PROMINENCE:
                issues.append(f"weak peak ({result['peak_prominence']:.2f})")
            if not result["has_ascending"] and not result["has_descending"]:
                issues.append("no clear focus trend")
            result["message"] = "Invalid focus peak: " + ", ".join(issues)

        return result
