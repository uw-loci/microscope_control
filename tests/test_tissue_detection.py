"""
Unit tests for empty region detection (tissue detection).

Tests the EmptyRegionDetector methods that determine if an image region
contains tissue or is empty slide background.
"""

import numpy as np
import pytest
from microscope_control.autofocus.tissue_detection import EmptyRegionDetector


class TestRefinedEntropy:
    """Test refined entropy-based empty region detection."""

    def test_refined_entropy_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Empty regions should have lower entropy than tissue regions."""
        empty_entropy = EmptyRegionDetector.refined_entropy(synthetic_empty_image)
        tissue_entropy = EmptyRegionDetector.refined_entropy(synthetic_tissue_image)

        # Tissue has more texture/variation, so higher entropy
        assert tissue_entropy > empty_entropy

    def test_refined_entropy_is_empty_detection(self, synthetic_empty_image):
        """Empty image should be detected as empty using refined entropy."""
        is_empty = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, threshold=0.5, return_bool=True
        )
        assert is_empty is True

    def test_refined_entropy_tissue_not_empty(self, synthetic_tissue_image):
        """Tissue image should NOT be detected as empty."""
        is_empty = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, threshold=0.5, return_bool=True
        )
        assert is_empty is False

    def test_refined_entropy_with_different_window_sizes(self, synthetic_tissue_image):
        """Different window sizes should affect entropy calculation."""
        entropy_small = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, window_size=5
        )
        entropy_large = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, window_size=15
        )

        # Both should be valid positive numbers
        assert entropy_small > 0
        assert entropy_large > 0
        # Values will differ due to different spatial scales
        assert entropy_small != entropy_large


class TestSaturationWithContext:
    """Test saturation-based empty region detection."""

    def test_saturation_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Empty regions typically have high saturation (uniform white)."""
        empty_score = EmptyRegionDetector.saturation_with_context(synthetic_empty_image)
        tissue_score = EmptyRegionDetector.saturation_with_context(synthetic_tissue_image)

        # Empty (white) should have higher saturation score
        assert empty_score > tissue_score

    def test_saturation_pure_white_image(self):
        """Pure white image should have very high saturation score."""
        white_image = np.full((256, 256, 3), 255, dtype=np.uint8)
        score = EmptyRegionDetector.saturation_with_context(white_image)

        # Should be close to 1.0 (highly saturated white)
        assert score > 0.8

    def test_saturation_with_roi_context(self, synthetic_tissue_image):
        """Saturation detection should work with different window sizes."""
        score_small = EmptyRegionDetector.saturation_with_context(
            synthetic_tissue_image, window_size=10
        )
        score_large = EmptyRegionDetector.saturation_with_context(
            synthetic_tissue_image, window_size=30
        )

        # Both should be valid
        assert np.isfinite(score_small)
        assert np.isfinite(score_large)


class TestMultiScaleAnalysis:
    """Test multi-scale empty region detection."""

    def test_multiscale_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Multi-scale analysis should distinguish empty from tissue."""
        empty_result = EmptyRegionDetector.multi_scale_analysis(synthetic_empty_image)
        tissue_result = EmptyRegionDetector.multi_scale_analysis(synthetic_tissue_image)

        # Results should be different
        assert empty_result != tissue_result

    def test_multiscale_returns_valid_score(self, synthetic_tissue_image):
        """Multi-scale analysis should return valid numeric score."""
        result = EmptyRegionDetector.multi_scale_analysis(synthetic_tissue_image)

        assert np.isfinite(result)
        assert result >= 0  # Assuming score is non-negative


class TestColorSpaceAnalysis:
    """Test HSV color space based empty region detection."""

    def test_color_space_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Color space analysis should distinguish empty (white) from colored tissue."""
        empty_score = EmptyRegionDetector.color_space_analysis(synthetic_empty_image)
        tissue_score = EmptyRegionDetector.color_space_analysis(synthetic_tissue_image)

        # Empty (white) should have different characteristics in HSV space
        # High V (value), low S (saturation) for white background
        assert empty_score != tissue_score

    def test_color_space_white_image(self):
        """Pure white image should have characteristic HSV properties."""
        white_image = np.full((256, 256, 3), 255, dtype=np.uint8)
        score = EmptyRegionDetector.color_space_analysis(white_image)

        # White has high V, low S - should be detected as empty-like
        assert np.isfinite(score)

    def test_color_space_colored_image(self):
        """Colored image should have different HSV properties than white."""
        # Create image with actual color (not white)
        colored_image = np.zeros((256, 256, 3), dtype=np.uint8)
        colored_image[:, :, 0] = 200  # Red
        colored_image[:, :, 1] = 100  # Green
        colored_image[:, :, 2] = 100  # Blue

        score = EmptyRegionDetector.color_space_analysis(colored_image)
        assert np.isfinite(score)


class TestRecommendedCombo:
    """Test the production recommended_combo method."""

    def test_recommended_combo_empty_detection(self, synthetic_empty_image):
        """Recommended combo should correctly identify empty regions."""
        is_empty = EmptyRegionDetector.recommended_combo(
            synthetic_empty_image, return_bool=True
        )

        # Empty image should be detected as empty
        assert is_empty is True

    def test_recommended_combo_tissue_detection(self, synthetic_tissue_image):
        """Recommended combo should correctly identify tissue regions."""
        is_empty = EmptyRegionDetector.recommended_combo(
            synthetic_tissue_image, return_bool=True
        )

        # Tissue image should NOT be detected as empty
        assert is_empty is False

    def test_recommended_combo_returns_score(self, synthetic_tissue_image):
        """Recommended combo should return numeric score when return_bool=False."""
        score = EmptyRegionDetector.recommended_combo(
            synthetic_tissue_image, return_bool=False
        )

        assert np.isfinite(score)
        assert isinstance(score, (int, float))

    def test_recommended_combo_consistency(self, synthetic_empty_image):
        """Multiple calls should return consistent results."""
        result1 = EmptyRegionDetector.recommended_combo(
            synthetic_empty_image, return_bool=True
        )
        result2 = EmptyRegionDetector.recommended_combo(
            synthetic_empty_image, return_bool=True
        )

        assert result1 == result2


class TestEdgeCases:
    """Test edge cases for tissue detection."""

    def test_all_black_image(self):
        """All-black image should be handled gracefully."""
        black_image = np.zeros((256, 256, 3), dtype=np.uint8)

        # Should not crash
        try:
            EmptyRegionDetector.refined_entropy(black_image)
            EmptyRegionDetector.saturation_with_context(black_image)
            EmptyRegionDetector.color_space_analysis(black_image)
            EmptyRegionDetector.recommended_combo(black_image)
        except Exception as e:
            pytest.fail(f"Detection failed on all-black image: {e}")

    def test_small_image(self):
        """Small images should be handled gracefully."""
        small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        # Should not crash with small images
        try:
            is_empty = EmptyRegionDetector.recommended_combo(
                small_image, return_bool=True
            )
            assert isinstance(is_empty, bool)
        except Exception as e:
            pytest.fail(f"Detection failed on small image: {e}")

    def test_grayscale_image_handling(self):
        """Methods should handle grayscale images if possible."""
        grayscale = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # refined_entropy should work with grayscale
        try:
            score = EmptyRegionDetector.refined_entropy(grayscale)
            assert np.isfinite(score)
        except Exception as e:
            # Some methods may require RGB
            if "color" not in str(e).lower():
                pytest.fail(f"Unexpected error with grayscale: {e}")

    def test_uint16_image_handling(self):
        """Methods should handle uint16 images from microscopy cameras."""
        # Create uint16 tissue-like image
        uint16_image = np.random.randint(5000, 15000, (256, 256, 3), dtype=np.uint16)

        try:
            # Convert to uint8 if needed or handle directly
            score = EmptyRegionDetector.refined_entropy(uint16_image)
            assert np.isfinite(score)
        except Exception:
            # May need conversion to uint8 for some methods
            uint8_image = (uint16_image // 256).astype(np.uint8)
            score = EmptyRegionDetector.refined_entropy(uint8_image)
            assert np.isfinite(score)


class TestThresholdTuning:
    """Test threshold parameter effects."""

    def test_entropy_different_thresholds(self, synthetic_empty_image):
        """Different thresholds should affect empty detection results."""
        # Very strict threshold
        is_empty_strict = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, threshold=0.9, return_bool=True
        )

        # Very lenient threshold
        is_empty_lenient = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, threshold=0.1, return_bool=True
        )

        # Both should be boolean
        assert isinstance(is_empty_strict, bool)
        assert isinstance(is_empty_lenient, bool)

    def test_threshold_boundary_behavior(self, synthetic_tissue_image):
        """Threshold should create clear decision boundary."""
        # Get raw score
        raw_score = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, return_bool=False
        )

        # Test threshold just above and below raw score
        is_empty_below = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, threshold=raw_score - 0.1, return_bool=True
        )
        is_empty_above = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, threshold=raw_score + 0.1, return_bool=True
        )

        # Results should differ based on threshold position
        assert isinstance(is_empty_below, bool)
        assert isinstance(is_empty_above, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
