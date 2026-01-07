import numpy as np
import cv2
from scipy import ndimage
from typing import Dict, Any, Optional, Tuple
import logging


class AutofocusMetrics:
    """
    Library of autofocus metric calculations for microscopy applications.
    Higher metric values typically indicate better focus.
    """

    def __init__(self, logger=None):
        """Initialize the autofocus metrics calculator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def variance(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Calculate variance of pixel intensities.
        Simple but effective for many samples.

        Args:
            image: Input image (grayscale or BGR)
            roi: Optional region of interest (x, y, width, height)

        Returns:
            Variance value (higher = better focus)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        return np.var(gray)

    @staticmethod
    def normalized_variance(
        image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Variance normalized by mean to reduce sensitivity to illumination changes.

        Args:
            image: Input image
            roi: Optional region of interest

        Returns:
            Normalized variance value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        mean_val = np.mean(gray)
        if mean_val == 0:
            return 0

        return np.var(gray) / mean_val

    @staticmethod
    def laplacian_variance(image: np.ndarray, ksize: int = 3) -> float:
        """
        Variance of Laplacian - sensitive to sharp edges.
        Good for samples with fine details.

        Args:
            image: Input image
            ksize: Kernel size for Laplacian

        Returns:
            Variance of Laplacian
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        return np.var(laplacian)

    @staticmethod
    def modified_laplacian(image: np.ndarray) -> float:
        """
        Modified Laplacian - sum of absolute values of second derivatives.
        More robust than standard Laplacian.

        Args:
            image: Input image

        Returns:
            Modified Laplacian score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate second derivatives
        kernel_x = np.array([[-1, 2, -1]])
        kernel_y = kernel_x.T

        lap_x = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel_x))
        lap_y = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel_y))

        return np.sum(lap_x + lap_y)

    @staticmethod
    def brenner_gradient(image: np.ndarray, threshold: float = 0) -> float:
        """
        Brenner gradient - sum of squared differences.
        Fast and effective for many applications.

        Args:
            image: Input image
            threshold: Minimum difference to consider

        Returns:
            Brenner gradient score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # Calculate differences with offset of 2 pixels
        diff_x = gray[:, 2:] - gray[:, :-2]
        diff_y = gray[2:, :] - gray[:-2, :]

        # Apply threshold
        diff_x[np.abs(diff_x) < threshold] = 0
        diff_y[np.abs(diff_y) < threshold] = 0

        return np.sum(diff_x**2) + np.sum(diff_y**2)

    @staticmethod
    def tenenbaum_gradient(image: np.ndarray, ksize: int = 3) -> float:
        """
        Tenenbaum gradient using Sobel operators.
        Good balance between noise resistance and sensitivity.

        Args:
            image: Input image
            ksize: Sobel kernel size

        Returns:
            Tenenbaum gradient score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        return np.sum(sobel_x**2 + sobel_y**2)

    @staticmethod
    def energy_laplace(image: np.ndarray) -> float:
        """
        Energy of Laplacian - sum of squared Laplacian values.

        Args:
            image: Input image

        Returns:
            Energy of Laplacian score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.sum(laplacian**2)

    @staticmethod
    def vollath_f4(image: np.ndarray) -> float:
        """
        Vollath's F4 correlation metric.
        Based on autocorrelation with different pixel shifts.

        Args:
            image: Input image

        Returns:
            Vollath F4 score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        mean = np.mean(gray)

        # Calculate autocorrelations
        auto_1 = np.sum(gray[:-1, :] * gray[1:, :])
        auto_2 = np.sum(gray[:-2, :] * gray[2:, :])
        N = gray.size

        return auto_1 - auto_2 - N * mean**2

    @staticmethod
    def vollath_f5(image: np.ndarray) -> float:
        """
        Vollath's F5 metric - normalized autocorrelation.
        Less sensitive to image content than F4.

        Args:
            image: Input image

        Returns:
            Vollath F5 score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        mean = np.mean(gray)

        # Calculate autocorrelation
        auto_1 = np.sum(gray[:-1, :] * gray[1:, :])
        N = gray.size

        denominator = N * mean**2
        if denominator == 0:
            return 0

        return auto_1 / denominator - 1

    @staticmethod
    def entropy(image: np.ndarray, bins: int = 256) -> float:
        """
        Shannon entropy of the image histogram.
        Higher entropy often indicates better focus.

        Args:
            image: Input image
            bins: Number of histogram bins

        Returns:
            Entropy value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) == 0:
            return 0

        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))

    @staticmethod
    def dct_energy(image: np.ndarray, cutoff_ratio: float = 0.1) -> float:
        """
        Energy in high-frequency DCT coefficients.
        Good for detecting fine details.

        Args:
            image: Input image
            cutoff_ratio: Ratio of frequencies to consider as "high"

        Returns:
            High-frequency energy
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Apply DCT
        dct = cv2.dct(gray)

        # Calculate energy in high frequencies
        h, w = dct.shape
        cutoff_h = int(h * cutoff_ratio)
        cutoff_w = int(w * cutoff_ratio)

        # High frequencies are in the bottom-right
        high_freq = dct[cutoff_h:, cutoff_w:]

        return np.sum(high_freq**2)

    @staticmethod
    def wavelet_energy(image: np.ndarray, level: int = 3) -> float:
        """
        Energy in wavelet detail coefficients.
        Requires pywt library for full implementation.

        Args:
            image: Input image
            level: Wavelet decomposition level

        Returns:
            Detail coefficient energy
        """
        # Simplified version using edge detection as proxy
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Use Gaussian pyramid as simplified wavelet decomposition
        energy = 0
        current = gray.astype(np.float64)

        for i in range(level):
            next_level = cv2.pyrDown(current)
            upsampled = cv2.pyrUp(next_level, dstsize=(current.shape[1], current.shape[0]))

            # Detail = difference between levels
            detail = current - upsampled
            energy += np.sum(detail**2)

            current = next_level

        return energy

    @staticmethod
    def histogram_range(image: np.ndarray) -> float:
        """
        Dynamic range of the histogram.
        Well-focused images typically have wider range.

        Args:
            image: Input image

        Returns:
            Histogram range (95th - 5th percentile)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)

        return p95 - p5

    def calculate_metric(
        self, image: np.ndarray, metric: str = "tenenbaum_gradient", **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate specified autofocus metric with additional diagnostics.

        Args:
            image: Input image
            metric: Metric name to calculate
            **kwargs: Additional parameters for the metric

        Returns:
            Dict with metric value and additional information
        """
        metrics = {
            "variance": self.variance,
            "normalized_variance": self.normalized_variance,
            "laplacian_variance": self.laplacian_variance,
            "modified_laplacian": self.modified_laplacian,
            "brenner_gradient": self.brenner_gradient,
            "tenenbaum_gradient": self.tenenbaum_gradient,
            "energy_laplace": self.energy_laplace,
            "vollath_f4": self.vollath_f4,
            "vollath_f5": self.vollath_f5,
            "entropy": self.entropy,
            "dct_energy": self.dct_energy,
            "wavelet_energy": self.wavelet_energy,
            "histogram_range": self.histogram_range,
        }

        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(metrics.keys())}")

        self.logger.info(f"Calculating autofocus metric: {metric}")

        # Calculate metric
        value = metrics[metric](image, **kwargs)

        # Add diagnostic information
        result = {
            "metric": metric,
            "value": value,
            "normalized_value": None,  # Can be filled by calibration
            "timestamp": None,  # Can be filled by caller
        }

        # Log result
        self.logger.debug(f"Autofocus {metric}: {value:.4f}")

        return result

    def multi_metric_analysis(
        self,
        image: np.ndarray,
        metrics: list = ["tenenbaum_gradient", "laplacian_variance", "brenner_gradient"],
    ) -> Dict[str, Any]:
        """
        Calculate multiple metrics for comparison or voting.

        Args:
            image: Input image
            metrics: List of metrics to calculate

        Returns:
            Dict with all metric values and statistics
        """
        results = {}
        values = []

        for metric in metrics:
            result = self.calculate_metric(image, metric)
            results[metric] = result["value"]
            values.append(result["value"])

        # Normalize values to 0-1 range for comparison
        values = np.array(values)
        if np.std(values) > 0:
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            normalized = values

        return {
            "metrics": results,
            "normalized_metrics": dict(zip(metrics, normalized)),
            "mean_normalized": np.mean(normalized),
            "consensus_score": np.mean(normalized) * np.min(normalized),  # High only if all agree
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create calculator
    autofocus = AutofocusMetrics()

    # Test with synthetic images
    # Well-focused image (high contrast, sharp edges)
    focused = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    focused_filtered = cv2.GaussianBlur(focused, (3, 3), 0.5)

    # Out-of-focus image (blurred)
    blurred = cv2.GaussianBlur(focused, (21, 21), 10)

    print("Autofocus Metric Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} {'Focused':>15} {'Blurred':>15} {'Ratio':>10}")
    print("-" * 60)

    metrics = [
        "variance",
        "normalized_variance",
        "laplacian_variance",
        "tenenbaum_gradient",
        "brenner_gradient",
        "entropy",
    ]

    for metric in metrics:
        focused_val = autofocus.calculate_metric(focused_filtered, metric)["value"]
        blurred_val = autofocus.calculate_metric(blurred, metric)["value"]
        ratio = focused_val / (blurred_val + 1e-10)

        print(f"{metric:<25} {focused_val:>15.2f} {blurred_val:>15.2f} {ratio:>10.2f}")

    # Multi-metric analysis
    print("\nMulti-metric consensus:")
    consensus = autofocus.multi_metric_analysis(focused_filtered)
    print(f"Consensus score: {consensus['consensus_score']:.4f}")
