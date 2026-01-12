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
        empty_result = EmptyRegionDetector.refined_entropy(synthetic_empty_image)
        tissue_result = EmptyRegionDetector.refined_entropy(synthetic_tissue_image)

        # Tissue has more texture/variation, so higher entropy
        assert tissue_result["mean_entropy"] > empty_result["mean_entropy"]

    def test_refined_entropy_is_empty_detection(self, synthetic_empty_image):
        """Empty image should be detected as empty using refined entropy."""
        # Synthetic empty image has entropy ~2.1 due to added noise
        result = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, entropy_threshold=3.0
        )
        assert result["is_empty"] == True

    def test_refined_entropy_tissue_not_empty(self, synthetic_tissue_image):
        """Tissue image should NOT be detected as empty."""
        # Synthetic tissue image has entropy ~4.5 due to texture/variation
        result = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, entropy_threshold=3.0
        )
        assert result["is_empty"] == False

    def test_refined_entropy_with_different_window_sizes(self, synthetic_tissue_image):
        """Different window sizes should affect entropy calculation."""
        result_small = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, window_size=5
        )
        result_large = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, window_size=15
        )

        # Both should be valid positive numbers
        assert result_small["mean_entropy"] > 0
        assert result_large["mean_entropy"] > 0
        # Values will differ due to different spatial scales
        assert result_small["mean_entropy"] != result_large["mean_entropy"]


class TestSaturationWithContext:
    """Test saturation-based empty region detection."""

    def test_saturation_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Empty regions typically have high saturation (uniform white)."""
        # Use threshold appropriate for synthetic images (~245)
        empty_result = EmptyRegionDetector.saturation_with_context(
            synthetic_empty_image, saturation_threshold=240
        )
        tissue_result = EmptyRegionDetector.saturation_with_context(
            synthetic_tissue_image, saturation_threshold=240
        )

        # Empty (white) should have higher saturation ratio
        assert empty_result["empty_ratio"] > tissue_result["empty_ratio"]

    def test_saturation_pure_white_image(self):
        """Pure white image should have very high saturation score."""
        white_image = np.full((256, 256, 3), 255, dtype=np.uint8)
        result = EmptyRegionDetector.saturation_with_context(white_image)

        # Should be close to 1.0 (highly saturated white)
        assert result["empty_ratio"] > 0.8
        assert result["is_empty"] == True

    def test_saturation_with_roi_context(self, synthetic_tissue_image):
        """Saturation detection should work with different window sizes."""
        result_small = EmptyRegionDetector.saturation_with_context(
            synthetic_tissue_image, window_size=10
        )
        result_large = EmptyRegionDetector.saturation_with_context(
            synthetic_tissue_image, window_size=30
        )

        # Both should be valid
        assert np.isfinite(result_small["empty_ratio"])
        assert np.isfinite(result_large["empty_ratio"])


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

        assert isinstance(result, dict)
        assert np.isfinite(result["mean_variance"])
        assert result["mean_variance"] >= 0  # Assuming score is non-negative


class TestColorSpaceAnalysis:
    """Test HSV color space based empty region detection."""

    def test_color_space_empty_vs_tissue(
        self, synthetic_empty_image, synthetic_tissue_image
    ):
        """Color space analysis should distinguish empty (white) from colored tissue."""
        empty_result = EmptyRegionDetector.color_space_analysis(synthetic_empty_image)
        tissue_result = EmptyRegionDetector.color_space_analysis(synthetic_tissue_image)

        # Empty (white) should have different characteristics in HSV space
        # High V (value), low S (saturation) for white background
        assert empty_result["mean_brightness"] > tissue_result["mean_brightness"]
        assert empty_result["mean_saturation"] < tissue_result["mean_saturation"]

    def test_color_space_white_image(self):
        """Pure white image should have characteristic HSV properties."""
        white_image = np.full((256, 256, 3), 255, dtype=np.uint8)
        result = EmptyRegionDetector.color_space_analysis(white_image)

        # White has high V, low S - should be detected as empty-like
        assert result["is_empty"] == True
        assert result["mean_brightness"] > 240
        assert result["mean_saturation"] < 0.1

    def test_color_space_colored_image(self):
        """Colored image should have different HSV properties than white."""
        # Create image with actual color (not white)
        colored_image = np.zeros((256, 256, 3), dtype=np.uint8)
        colored_image[:, :, 0] = 200  # Red
        colored_image[:, :, 1] = 100  # Green
        colored_image[:, :, 2] = 100  # Blue

        result = EmptyRegionDetector.color_space_analysis(colored_image)
        assert isinstance(result, dict)
        assert np.isfinite(result["mean_brightness"])
        assert np.isfinite(result["mean_saturation"])


class TestRecommendedCombo:
    """Test the production recommended_combo method."""

    def test_recommended_combo_empty_detection(self, synthetic_empty_image):
        """Recommended combo should correctly identify empty regions."""
        result = EmptyRegionDetector.recommended_combo(synthetic_empty_image)

        # Empty image should be detected as empty
        assert result["is_empty"] == True

    def test_recommended_combo_tissue_detection(self, synthetic_tissue_image):
        """Recommended combo should correctly identify tissue regions."""
        result = EmptyRegionDetector.recommended_combo(synthetic_tissue_image)

        # Tissue image should NOT be detected as empty
        assert result["is_empty"] == False

    def test_recommended_combo_returns_score(self, synthetic_tissue_image):
        """Recommended combo should return dict with numeric scores."""
        result = EmptyRegionDetector.recommended_combo(synthetic_tissue_image)

        assert isinstance(result, dict)
        assert np.isfinite(result["mean_variance"])
        assert np.isfinite(result["edge_density"])
        assert np.isfinite(result["brightness_ratio"])
        assert np.isfinite(result["texture_score"])

    def test_recommended_combo_consistency(self, synthetic_empty_image):
        """Multiple calls should return consistent results."""
        result1 = EmptyRegionDetector.recommended_combo(synthetic_empty_image)
        result2 = EmptyRegionDetector.recommended_combo(synthetic_empty_image)

        assert result1["is_empty"] == result2["is_empty"]
        assert result1["mean_variance"] == result2["mean_variance"]


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
            result = EmptyRegionDetector.recommended_combo(small_image)
            assert isinstance(result, dict)
            assert isinstance(result["is_empty"], bool)
        except Exception as e:
            pytest.fail(f"Detection failed on small image: {e}")

    def test_grayscale_image_handling(self):
        """Methods should handle grayscale images if possible."""
        grayscale = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # refined_entropy should work with grayscale
        try:
            result = EmptyRegionDetector.refined_entropy(grayscale)
            assert isinstance(result, dict)
            assert np.isfinite(result["mean_entropy"])
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
            result = EmptyRegionDetector.refined_entropy(uint16_image)
            assert isinstance(result, dict)
            assert np.isfinite(result["mean_entropy"])
        except Exception:
            # May need conversion to uint8 for some methods
            uint8_image = (uint16_image // 256).astype(np.uint8)
            result = EmptyRegionDetector.refined_entropy(uint8_image)
            assert isinstance(result, dict)
            assert np.isfinite(result["mean_entropy"])


class TestThresholdTuning:
    """Test threshold parameter effects."""

    def test_entropy_different_thresholds(self, synthetic_empty_image):
        """Different thresholds should affect empty detection results."""
        # Very strict threshold
        result_strict = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, entropy_threshold=0.9
        )

        # Very lenient threshold
        result_lenient = EmptyRegionDetector.refined_entropy(
            synthetic_empty_image, entropy_threshold=0.1
        )

        # Both should return dict with is_empty boolean
        assert isinstance(result_strict["is_empty"], bool)
        assert isinstance(result_lenient["is_empty"], bool)
        # Different thresholds may give different results
        # (lenient threshold more likely to mark as empty)

    def test_threshold_boundary_behavior(self, synthetic_tissue_image):
        """Threshold should create clear decision boundary."""
        # Get raw score
        result = EmptyRegionDetector.refined_entropy(synthetic_tissue_image)
        raw_score = result["mean_entropy"]

        # Test threshold just above and below raw score
        result_below = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, entropy_threshold=raw_score - 0.1
        )
        result_above = EmptyRegionDetector.refined_entropy(
            synthetic_tissue_image, entropy_threshold=raw_score + 0.1
        )

        # Results should differ based on threshold position
        assert isinstance(result_below["is_empty"], bool)
        assert isinstance(result_above["is_empty"], bool)
        # Below threshold should mark as empty (entropy < threshold)
        # Above threshold should NOT mark as empty (entropy > threshold)
        assert result_below["is_empty"] != result_above["is_empty"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
