"""
Unit tests for autofocus metric calculations.

Tests all 13 autofocus metrics from microscope_control.autofocus.metrics module
using synthetic focused and blurred images.
"""

import numpy as np
import pytest
from microscope_control.autofocus.metrics import AutofocusMetrics


class TestAutofocusMetricsBasic:
    """Test basic properties of autofocus metrics."""

    def test_variance_focused_vs_blurred(self, synthetic_focused_image, synthetic_blurred_image):
        """Variance should be higher for focused images than blurred images."""
        focused_score = AutofocusMetrics.variance(synthetic_focused_image)
        blurred_score = AutofocusMetrics.variance(synthetic_blurred_image)

        assert focused_score > blurred_score, (
            f"Focused variance ({focused_score:.2f}) should be > blurred ({blurred_score:.2f})"
        )

    def test_normalized_variance_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Normalized variance should be higher for focused images."""
        focused_score = AutofocusMetrics.normalized_variance(synthetic_focused_image)
        blurred_score = AutofocusMetrics.normalized_variance(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_laplacian_variance_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Laplacian variance (edge energy) should be higher for focused images."""
        focused_score = AutofocusMetrics.laplacian_variance(synthetic_focused_image)
        blurred_score = AutofocusMetrics.laplacian_variance(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_tenenbaum_gradient_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Tenenbaum gradient (recommended metric) should be higher for focused images."""
        focused_score = AutofocusMetrics.tenenbaum_gradient(synthetic_focused_image)
        blurred_score = AutofocusMetrics.tenenbaum_gradient(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_brenner_gradient_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Brenner gradient should be higher for focused images."""
        focused_score = AutofocusMetrics.brenner_gradient(synthetic_focused_image)
        blurred_score = AutofocusMetrics.brenner_gradient(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_modified_laplacian_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Modified Laplacian should be higher for focused images."""
        focused_score = AutofocusMetrics.modified_laplacian(synthetic_focused_image)
        blurred_score = AutofocusMetrics.modified_laplacian(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_energy_laplace_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Energy of Laplacian should be higher for focused images."""
        focused_score = AutofocusMetrics.energy_laplace(synthetic_focused_image)
        blurred_score = AutofocusMetrics.energy_laplace(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_vollath_f4_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Vollath F4 autocorrelation should be higher for focused images."""
        focused_score = AutofocusMetrics.vollath_f4(synthetic_focused_image)
        blurred_score = AutofocusMetrics.vollath_f4(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_vollath_f5_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Vollath F5 autocorrelation should be higher for focused images."""
        focused_score = AutofocusMetrics.vollath_f5(synthetic_focused_image)
        blurred_score = AutofocusMetrics.vollath_f5(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_entropy_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Entropy should be higher for focused images (more detail)."""
        focused_score = AutofocusMetrics.entropy(synthetic_focused_image)
        blurred_score = AutofocusMetrics.entropy(synthetic_blurred_image)

        # Note: Entropy can sometimes be tricky - it measures information content
        # Focused images typically have more detail/information
        assert focused_score > blurred_score

    def test_dct_energy_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """DCT high-frequency energy should be higher for focused images."""
        focused_score = AutofocusMetrics.dct_energy(synthetic_focused_image)
        blurred_score = AutofocusMetrics.dct_energy(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_wavelet_energy_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Wavelet high-frequency energy should be higher for focused images."""
        focused_score = AutofocusMetrics.wavelet_energy(synthetic_focused_image)
        blurred_score = AutofocusMetrics.wavelet_energy(synthetic_blurred_image)

        assert focused_score > blurred_score

    def test_histogram_range_focused_vs_blurred(
        self, synthetic_focused_image, synthetic_blurred_image
    ):
        """Histogram range should be larger for focused images (more contrast)."""
        focused_score = AutofocusMetrics.histogram_range(synthetic_focused_image)
        blurred_score = AutofocusMetrics.histogram_range(synthetic_blurred_image)

        assert focused_score > blurred_score


class TestAutofocusMetricsEdgeCases:
    """Test edge cases and error handling for autofocus metrics."""

    def test_variance_uniform_image(self):
        """Variance of uniform image should be zero."""
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        score = AutofocusMetrics.variance(uniform_image)

        assert score == 0.0

    def test_normalized_variance_uniform_image(self):
        """Normalized variance of uniform image should be zero."""
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        score = AutofocusMetrics.normalized_variance(uniform_image)

        assert score == 0.0

    def test_variance_with_roi(self, synthetic_focused_image):
        """Variance calculation should work with ROI."""
        full_score = AutofocusMetrics.variance(synthetic_focused_image)

        # ROI in center quarter of image
        roi = (128, 128, 256, 256)
        roi_score = AutofocusMetrics.variance(synthetic_focused_image, roi=roi)

        # ROI score should be different from full image
        assert roi_score != full_score
        # Both should be valid positive numbers
        assert roi_score > 0
        assert full_score > 0

    def test_metrics_with_uint16_image(self):
        """Metrics should work with uint16 images."""
        # Create uint16 focused image
        image_uint16 = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)

        # Add some structure
        image_uint16[::10, :] = 50000
        image_uint16[:, ::10] = 50000

        score = AutofocusMetrics.variance(image_uint16)
        assert score > 0

        score = AutofocusMetrics.tenenbaum_gradient(image_uint16)
        assert score > 0

    def test_metrics_with_single_pixel(self):
        """Metrics should not crash with very small images."""
        tiny_image = np.array([[128]], dtype=np.uint8)

        # Some metrics may return 0 or small values, but shouldn't crash
        try:
            AutofocusMetrics.variance(tiny_image)
            AutofocusMetrics.laplacian_variance(tiny_image)
        except Exception as e:
            pytest.fail(f"Metric calculation failed on single pixel image: {e}")

    def test_normalized_variance_zero_mean(self):
        """Normalized variance should handle zero mean gracefully."""
        # Image with mean very close to zero
        zero_mean_image = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

        # Should not crash or return inf/nan
        score = AutofocusMetrics.normalized_variance(zero_mean_image)
        assert np.isfinite(score)

    def test_empty_image_handling(self):
        """Metrics should handle all-zero images gracefully."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)

        score = AutofocusMetrics.variance(empty_image)
        assert score == 0.0

        score = AutofocusMetrics.normalized_variance(empty_image)
        assert np.isfinite(score)


class TestAutofocusMetricsColorImages:
    """Test metrics with color (BGR/RGB) images."""

    def test_variance_bgr_image(self):
        """Variance should work with BGR color images."""
        # Create synthetic BGR image
        bgr_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Add structure
        bgr_image[::10, :, :] = [255, 0, 0]
        bgr_image[:, ::10, :] = [0, 255, 0]

        score = AutofocusMetrics.variance(bgr_image)
        assert score > 0

    def test_tenenbaum_gradient_bgr_image(self):
        """Tenenbaum gradient should work with BGR images."""
        bgr_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bgr_image[::10, :, :] = [255, 255, 255]

        score = AutofocusMetrics.tenenbaum_gradient(bgr_image)
        assert score > 0


class TestAutofocusMetricsNumericalStability:
    """Test numerical stability and consistency of metrics."""

    def test_variance_consistency(self, synthetic_focused_image):
        """Multiple calls should return same variance value."""
        score1 = AutofocusMetrics.variance(synthetic_focused_image)
        score2 = AutofocusMetrics.variance(synthetic_focused_image)

        assert score1 == score2

    def test_metric_returns_finite_values(self, synthetic_focused_image):
        """All metrics should return finite values (no inf/nan)."""
        metrics_to_test = [
            AutofocusMetrics.variance,
            AutofocusMetrics.normalized_variance,
            AutofocusMetrics.laplacian_variance,
            AutofocusMetrics.tenenbaum_gradient,
            AutofocusMetrics.brenner_gradient,
            AutofocusMetrics.modified_laplacian,
            AutofocusMetrics.energy_laplace,
            AutofocusMetrics.vollath_f4,
            AutofocusMetrics.vollath_f5,
            AutofocusMetrics.entropy,
            AutofocusMetrics.dct_energy,
            AutofocusMetrics.wavelet_energy,
            AutofocusMetrics.histogram_range,
        ]

        for metric_func in metrics_to_test:
            score = metric_func(synthetic_focused_image)
            assert np.isfinite(score), f"{metric_func.__name__} returned non-finite value: {score}"

    def test_metric_returns_non_negative(self, synthetic_focused_image):
        """Most metrics should return non-negative values."""
        # Note: Some metrics like Vollath F4/F5 can be negative
        non_negative_metrics = [
            AutofocusMetrics.variance,
            AutofocusMetrics.normalized_variance,
            AutofocusMetrics.laplacian_variance,
            AutofocusMetrics.tenenbaum_gradient,
            AutofocusMetrics.brenner_gradient,
            AutofocusMetrics.modified_laplacian,
            AutofocusMetrics.energy_laplace,
            AutofocusMetrics.entropy,
            AutofocusMetrics.dct_energy,
            AutofocusMetrics.wavelet_energy,
            AutofocusMetrics.histogram_range,
        ]

        for metric_func in non_negative_metrics:
            score = metric_func(synthetic_focused_image)
            assert score >= 0, f"{metric_func.__name__} returned negative value: {score}"


class TestMultiMetricAnalysis:
    """Test the multi-metric analysis function if it exists."""

    def test_multi_metric_with_focused_image(self, synthetic_focused_image):
        """Multi-metric analysis should work with focused image."""
        try:
            result = AutofocusMetrics.multi_metric_analysis(synthetic_focused_image)
            assert isinstance(result, dict)
            assert 'consensus_score' in result or 'metrics' in result
        except AttributeError:
            # multi_metric_analysis may not exist in all versions
            pytest.skip("multi_metric_analysis not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
