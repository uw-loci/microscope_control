import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Dict, Any
import logging


class EmptyRegionDetector:
    """
    Detector for identifying empty/white regions in microscope images.
    Used for white balance calibration and avoiding empty acquisition areas.
    """

    def __init__(self, logger=None):
        """Initialize the detector with optional logger."""
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def refined_entropy(
        image: np.ndarray, window_size: int = 15, entropy_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate refined entropy using sliding windows.

        Args:
            image: Input image (grayscale or BGR)
            window_size: Size of sliding window for local entropy
            entropy_threshold: Threshold below which region is considered empty

        Returns:
            Dict with 'is_empty' bool, 'mean_entropy' float, and 'entropy_map' array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate local entropy
        def entropy_filter(window):
            hist, _ = np.histogram(window.flatten(), bins=256, range=(0, 256))
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) == 0:
                return 0
            prob = hist / hist.sum()
            return -np.sum(prob * np.log2(prob + 1e-10))

        # Apply entropy filter
        entropy_map = ndimage.generic_filter(gray, entropy_filter, size=window_size)
        mean_entropy = np.mean(entropy_map)

        return {
            "is_empty": mean_entropy < entropy_threshold,
            "mean_entropy": mean_entropy,
            "entropy_map": entropy_map,
        }

    @staticmethod
    def saturation_with_context(
        image: np.ndarray,
        saturation_threshold: int = 250,
        min_saturation_ratio: float = 0.95,
        window_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Detect overexposure vs empty regions using saturation analysis.

        Args:
            image: Input image (BGR or grayscale)
            saturation_threshold: Pixel value considered saturated
            min_saturation_ratio: Minimum ratio of saturated pixels for empty region
            window_size: Size of context window

        Returns:
            Dict with detection results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find saturated pixels
        saturated = gray >= saturation_threshold

        # Analyze in windows
        h, w = gray.shape
        results = []

        for y in range(0, h - window_size, window_size // 2):
            for x in range(0, w - window_size, window_size // 2):
                window = saturated[y : y + window_size, x : x + window_size]
                saturation_ratio = np.mean(window)

                # Check for continuous saturation
                labeled, num_features = ndimage.label(window)
                if num_features == 1 and saturation_ratio > min_saturation_ratio:
                    results.append(True)
                else:
                    results.append(False)

        # Overall assessment
        empty_ratio = sum(results) / len(results) if results else 0

        return {
            "is_empty": empty_ratio > 0.8,
            "empty_ratio": empty_ratio,
            "saturation_map": saturated.astype(float),
        }

    @staticmethod
    def multi_scale_analysis(
        image: np.ndarray, scales: list = [1.0, 0.5, 0.25], variance_threshold: float = 5.0
    ) -> Dict[str, Any]:
        """
        Analyze consistency across multiple scales.

        Args:
            image: Input image
            scales: List of scaling factors to test
            variance_threshold: Maximum variance for empty region

        Returns:
            Dict with multi-scale analysis results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        variances = []

        for scale in scales:
            if scale != 1.0:
                scaled = cv2.resize(gray, None, fx=scale, fy=scale)
            else:
                scaled = gray.copy()

            # Calculate local variance
            local_var = ndimage.generic_filter(scaled, np.var, size=5)
            variances.append(np.mean(local_var))

        # Check consistency across scales
        variance_range = max(variances) - min(variances)
        mean_variance = np.mean(variances)

        return {
            "is_empty": mean_variance < variance_threshold and variance_range < 2.0,
            "mean_variance": mean_variance,
            "scale_consistency": variance_range,
            "variances_per_scale": dict(zip(scales, variances)),
        }

    @staticmethod
    def color_space_analysis(
        image: np.ndarray, min_brightness: int = 240, max_saturation: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze in HSV color space for better empty region detection.

        Args:
            image: Input BGR image
            min_brightness: Minimum V channel value for empty
            max_saturation: Maximum S channel value for empty

        Returns:
            Dict with color space analysis results
        """
        if len(image.shape) != 3:
            # Convert grayscale to BGR for HSV conversion
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Analyze channels
        mean_saturation = np.mean(s) / 255.0  # Normalize to 0-1
        mean_brightness = np.mean(v)

        # Check RGB similarity for true white
        b, g, r = cv2.split(image)
        channel_diff = np.max([np.std(b), np.std(g), np.std(r)])

        return {
            "is_empty": (
                mean_brightness > min_brightness
                and mean_saturation < max_saturation
                and channel_diff < 10
            ),
            "mean_brightness": mean_brightness,
            "mean_saturation": mean_saturation,
            "channel_variance": channel_diff,
        }

    @staticmethod
    def recommended_combo(
        image: np.ndarray,
        edge_threshold: float = 0.01,
        variance_threshold: float = 5.0,
        saturation_ratio_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Recommended combination: local variance + edge density + saturation ratio.

        Args:
            image: Input image
            edge_threshold: Maximum edge pixel ratio for empty region
            variance_threshold: Maximum local variance for empty region
            saturation_ratio_threshold: Minimum bright pixel ratio

        Returns:
            Dict with comprehensive analysis results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Local variance analysis
        local_var = ndimage.generic_filter(gray, np.var, size=15)
        mean_variance = np.mean(local_var)

        # 2. Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 3. Saturation/brightness analysis
        bright_pixels = gray > 240
        brightness_ratio = np.mean(bright_pixels)

        # 4. Texture analysis using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.std(laplacian)

        # Combined decision
        is_empty = (
            mean_variance < variance_threshold
            and edge_density < edge_threshold
            and brightness_ratio > saturation_ratio_threshold
            and texture_score < 10.0
        )

        return {
            "is_empty": is_empty,
            "mean_variance": mean_variance,
            "edge_density": edge_density,
            "brightness_ratio": brightness_ratio,
            "texture_score": texture_score,
            "edge_map": edges,
            "variance_map": local_var,
        }

    def detect_empty_region(
        self, image: np.ndarray, method: str = "recommended_combo", **kwargs
    ) -> Dict[str, Any]:
        """
        Main detection method that routes to specific algorithms.

        Args:
            image: Input image
            method: Detection method to use
            **kwargs: Additional parameters for the selected method

        Returns:
            Dict with detection results including 'is_empty' boolean
        """
        methods = {
            "refined_entropy": self.refined_entropy,
            "saturation_with_context": self.saturation_with_context,
            "multi_scale": self.multi_scale_analysis,
            "color_space": self.color_space_analysis,
            "recommended_combo": self.recommended_combo,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        self.logger.info(f"Detecting empty regions using method: {method}")
        result = methods[method](image, **kwargs)

        # Log result
        self.logger.debug(f"Detection result: is_empty={result['is_empty']}")

        return result


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create detector
    detector = EmptyRegionDetector()

    # Load test image (placeholder - replace with actual image)
    # image = cv2.imread('test_slide.png')

    # Example with synthetic empty region
    empty_image = np.ones((500, 500, 3), dtype=np.uint8) * 250

    # Test all methods
    methods = [
        "refined_entropy",
        "saturation_with_context",
        "multi_scale",
        "color_space",
        "recommended_combo",
    ]

    print("Testing empty region detection methods:")
    print("-" * 50)

    for method in methods:
        result = detector.detect_empty_region(empty_image, method=method)
        print(f"{method}: is_empty = {result['is_empty']}")
        print(f"  Additional metrics: {[k for k in result.keys() if k != 'is_empty']}")
        print()
