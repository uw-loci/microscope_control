#!/usr/bin/env python3
"""
JAI Camera Noise Characterization Tool.

This diagnostic tool systematically measures noise characteristics of the JAI
AP-3200T-USB 3-CCD prism camera across a grid of gain and exposure settings.
Use this tool to:

1. Characterize noise behavior when hardware changes (new camera, different cable, etc.)
2. Validate that current noise models are accurate
3. Generate noise lookup tables for noise-aware calibration
4. Produce reports and visualizations for documentation

The tool measures:
- Per-channel noise (standard deviation)
- Per-channel signal level (mean intensity)
- Signal-to-noise ratio (SNR)
- Saturation percentage

Usage:
    # Full characterization with default grid
    python noise_characterization.py config.yml --output ./noise_results

    # Quick test with smaller grid
    python noise_characterization.py config.yml --quick

    # Custom parameter grid
    python noise_characterization.py config.yml --gains 1.0,2.0,3.0 --exposures 10,25,50,100

Key Findings from Initial Characterization:
------------------------------------------
| Unified Gain | Noise (StdDev at 25ms) | Signal Level | Recommendation |
|--------------|------------------------|--------------|----------------|
| 1.0 | R=2.4, G=1.8, B=1.0 | Low (95/62/23) | Best for bright scenes |
| 2.0 | R=4.5, G=3.4, B=1.9 | Good (177/115/38) | Optimal balance |
| 3.0+ | Increasing | Saturating | Only for very dark scenes |

Critical insight: Analog R/B gains (0.48-4.0) have minimal noise impact -
they're post-amplification scalars, not true hardware gain. Exposure is
the primary control for signal quality.
"""

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseTestResult:
    """Result from a single noise test configuration."""

    unified_gain: float
    exposure_ms: float
    analog_red: float
    analog_blue: float

    # Per-channel means
    red_mean: float
    green_mean: float
    blue_mean: float

    # Per-channel stddevs (noise)
    red_stddev: float
    green_stddev: float
    blue_stddev: float

    # Per-channel SNR
    red_snr: float
    green_snr: float
    blue_snr: float

    # Saturation info
    saturation_pct: float = 0.0
    num_frames: int = 10


@dataclass
class NoiseCharacterizationResults:
    """Complete results from noise characterization run."""

    timestamp: str
    camera_name: str = "JAI AP-3200T-USB"
    results: List[NoiseTestResult] = field(default_factory=list)
    notes: str = ""

    def to_csv(self, output_path: Path) -> None:
        """Save results to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'unified_gain', 'exposure_ms', 'analog_red', 'analog_blue',
            'red_mean', 'green_mean', 'blue_mean',
            'red_stddev', 'green_stddev', 'blue_stddev',
            'red_snr', 'green_snr', 'blue_snr',
            'saturation_pct', 'num_frames'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'unified_gain': result.unified_gain,
                    'exposure_ms': result.exposure_ms,
                    'analog_red': result.analog_red,
                    'analog_blue': result.analog_blue,
                    'red_mean': round(result.red_mean, 2),
                    'green_mean': round(result.green_mean, 2),
                    'blue_mean': round(result.blue_mean, 2),
                    'red_stddev': round(result.red_stddev, 3),
                    'green_stddev': round(result.green_stddev, 3),
                    'blue_stddev': round(result.blue_stddev, 3),
                    'red_snr': round(result.red_snr, 1),
                    'green_snr': round(result.green_snr, 1),
                    'blue_snr': round(result.blue_snr, 1),
                    'saturation_pct': round(result.saturation_pct, 2),
                    'num_frames': result.num_frames,
                })

        logger.info(f"Saved {len(self.results)} results to {output_path}")


class JAINoiseCharacterization:
    """
    Systematic noise characterization tool for JAI prism cameras.

    Tests noise behavior across a configurable grid of:
    - Unified gain values (1.0 - 8.0)
    - Exposure times (0.5ms - 500ms)
    - Optionally: R/B analog gain variations
    """

    # Default test grids
    DEFAULT_GAINS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    DEFAULT_EXPOSURES = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    DEFAULT_RB_GAINS = [1.0]  # Test at neutral by default

    # Quick test grids
    QUICK_GAINS = [1.0, 2.0, 3.0, 5.0]
    QUICK_EXPOSURES = [10.0, 25.0, 50.0, 100.0]

    def __init__(
        self,
        hardware: Any,
        jai_props: Optional[Any] = None,
        num_frames: int = 10,
        settle_frames: int = 2,
    ):
        """
        Initialize the noise characterization tool.

        Args:
            hardware: PycromanagerHardware instance
            jai_props: Optional JAICameraProperties instance
            num_frames: Number of frames to average for each measurement
            settle_frames: Frames to discard for camera settling
        """
        self.hardware = hardware
        self.num_frames = num_frames
        self.settle_frames = settle_frames

        # Create or use provided jai_props
        if jai_props is not None:
            self.jai_props = jai_props
        else:
            try:
                from microscope_control.jai.properties import JAICameraProperties
                self.jai_props = JAICameraProperties(hardware.core)
            except Exception as e:
                logger.warning(f"Could not create JAICameraProperties: {e}")
                self.jai_props = None

    def _capture_frames(self, count: int) -> List[np.ndarray]:
        """Capture multiple frames from the camera."""
        frames = []
        for i in range(count):
            img, _ = self.hardware.snap_image()
            if img is not None:
                frames.append(img)
            time.sleep(0.05)  # Small delay between captures
        return frames

    def _analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze captured frames for noise statistics.

        Args:
            frames: List of RGB image arrays

        Returns:
            Dictionary with per-channel mean, stddev, SNR, and saturation info
        """
        if not frames:
            return {
                'red_mean': 0, 'green_mean': 0, 'blue_mean': 0,
                'red_stddev': 0, 'green_stddev': 0, 'blue_stddev': 0,
                'red_snr': 0, 'green_snr': 0, 'blue_snr': 0,
                'saturation_pct': 0,
            }

        # Stack frames: shape (N, H, W, C)
        stack = np.stack(frames, axis=0).astype(np.float32)

        if stack.ndim != 4 or stack.shape[3] < 3:
            logger.warning(f"Unexpected image shape: {stack.shape}")
            return {
                'red_mean': 0, 'green_mean': 0, 'blue_mean': 0,
                'red_stddev': 0, 'green_stddev': 0, 'blue_stddev': 0,
                'red_snr': 0, 'green_snr': 0, 'blue_snr': 0,
                'saturation_pct': 0,
            }

        # Compute temporal mean and stddev per pixel, then average spatially
        temporal_mean = stack.mean(axis=0)  # (H, W, C)
        temporal_std = stack.std(axis=0)    # (H, W, C)

        channel_means = {
            'red': float(temporal_mean[:, :, 0].mean()),
            'green': float(temporal_mean[:, :, 1].mean()),
            'blue': float(temporal_mean[:, :, 2].mean()),
        }

        channel_stddevs = {
            'red': float(temporal_std[:, :, 0].mean()),
            'green': float(temporal_std[:, :, 1].mean()),
            'blue': float(temporal_std[:, :, 2].mean()),
        }

        channel_snr = {}
        for ch in ['red', 'green', 'blue']:
            if channel_stddevs[ch] > 0:
                channel_snr[ch] = channel_means[ch] / channel_stddevs[ch]
            else:
                channel_snr[ch] = float('inf') if channel_means[ch] > 0 else 0

        # Check for saturation (assuming 8-bit, threshold at 250)
        max_val = temporal_mean.max()
        saturation_threshold = 250 if max_val <= 255 else 64000
        saturation_pct = 100.0 * np.sum(temporal_mean >= saturation_threshold) / temporal_mean.size

        return {
            'red_mean': channel_means['red'],
            'green_mean': channel_means['green'],
            'blue_mean': channel_means['blue'],
            'red_stddev': channel_stddevs['red'],
            'green_stddev': channel_stddevs['green'],
            'blue_stddev': channel_stddevs['blue'],
            'red_snr': channel_snr['red'],
            'green_snr': channel_snr['green'],
            'blue_snr': channel_snr['blue'],
            'saturation_pct': saturation_pct,
        }

    def measure_at_settings(
        self,
        unified_gain: float,
        exposure_ms: float,
        analog_red: float = 1.0,
        analog_blue: float = 1.0,
    ) -> NoiseTestResult:
        """
        Measure noise at specific camera settings.

        Args:
            unified_gain: Unified gain value (1.0-8.0)
            exposure_ms: Exposure time in milliseconds
            analog_red: R channel analog gain (0.47-4.0)
            analog_blue: B channel analog gain (0.47-4.0)

        Returns:
            NoiseTestResult with all measured statistics
        """
        # Apply settings
        try:
            self.jai_props.set_unified_gain(unified_gain)
        except Exception as e:
            logger.warning(f"Failed to set unified gain: {e}")

        try:
            self.jai_props.set_channel_exposures(
                red=exposure_ms, green=exposure_ms, blue=exposure_ms,
                auto_enable=True
            )
        except Exception as e:
            logger.warning(f"Failed to set exposure: {e}")

        try:
            self.jai_props.set_rb_analog_gains(red=analog_red, blue=analog_blue)
        except Exception as e:
            logger.warning(f"Failed to set R/B gains: {e}")

        # Allow settings to stabilize
        time.sleep(0.2)

        # Capture frames (including settle frames)
        total_frames = self.num_frames + self.settle_frames
        all_frames = self._capture_frames(total_frames)

        # Discard settle frames
        analysis_frames = all_frames[self.settle_frames:] if len(all_frames) > self.settle_frames else all_frames

        # Analyze
        stats = self._analyze_frames(analysis_frames)

        return NoiseTestResult(
            unified_gain=unified_gain,
            exposure_ms=exposure_ms,
            analog_red=analog_red,
            analog_blue=analog_blue,
            red_mean=stats['red_mean'],
            green_mean=stats['green_mean'],
            blue_mean=stats['blue_mean'],
            red_stddev=stats['red_stddev'],
            green_stddev=stats['green_stddev'],
            blue_stddev=stats['blue_stddev'],
            red_snr=stats['red_snr'],
            green_snr=stats['green_snr'],
            blue_snr=stats['blue_snr'],
            saturation_pct=stats['saturation_pct'],
            num_frames=len(analysis_frames),
        )

    def run_characterization(
        self,
        gains: Optional[List[float]] = None,
        exposures: Optional[List[float]] = None,
        rb_gains: Optional[List[float]] = None,
        quick: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> NoiseCharacterizationResults:
        """
        Run full noise characterization across parameter grid.

        Args:
            gains: List of unified gain values to test
            exposures: List of exposure times to test (ms)
            rb_gains: List of R/B analog gain values to test
            quick: Use quick (smaller) test grid
            progress_callback: Optional callback(current, total, message)

        Returns:
            NoiseCharacterizationResults with all measurements
        """
        if quick:
            gains = gains or self.QUICK_GAINS
            exposures = exposures or self.QUICK_EXPOSURES
        else:
            gains = gains or self.DEFAULT_GAINS
            exposures = exposures or self.DEFAULT_EXPOSURES

        rb_gains = rb_gains or self.DEFAULT_RB_GAINS

        total_tests = len(gains) * len(exposures) * len(rb_gains)
        logger.info(f"Starting noise characterization: {total_tests} test configurations")
        logger.info(f"  Gains: {gains}")
        logger.info(f"  Exposures: {exposures}")
        logger.info(f"  R/B gains: {rb_gains}")

        results = NoiseCharacterizationResults(
            timestamp=datetime.now().isoformat(),
        )

        current_test = 0
        for gain in gains:
            for exposure in exposures:
                for rb_gain in rb_gains:
                    current_test += 1

                    if progress_callback:
                        progress_callback(
                            current_test, total_tests,
                            f"Testing gain={gain}, exp={exposure}ms, rb={rb_gain}"
                        )

                    logger.debug(
                        f"[{current_test}/{total_tests}] "
                        f"gain={gain}, exp={exposure}ms, rb_gain={rb_gain}"
                    )

                    try:
                        result = self.measure_at_settings(
                            unified_gain=gain,
                            exposure_ms=exposure,
                            analog_red=rb_gain,
                            analog_blue=rb_gain,
                        )
                        results.results.append(result)

                        # Log key metrics
                        logger.info(
                            f"  gain={gain:.1f} exp={exposure:.0f}ms: "
                            f"R(m={result.red_mean:.0f},s={result.red_stddev:.2f}) "
                            f"G(m={result.green_mean:.0f},s={result.green_stddev:.2f}) "
                            f"B(m={result.blue_mean:.0f},s={result.blue_stddev:.2f}) "
                            f"sat={result.saturation_pct:.1f}%"
                        )

                    except Exception as e:
                        logger.error(f"Test failed at gain={gain}, exp={exposure}: {e}")

        logger.info(f"Characterization complete: {len(results.results)} successful tests")
        return results

    def generate_report(
        self,
        results: NoiseCharacterizationResults,
        output_path: Path,
    ) -> None:
        """
        Generate human-readable report and visualizations.

        Args:
            results: Characterization results
            output_path: Directory to save report files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save CSV
        results.to_csv(output_path / "noise_characterization.csv")

        # Generate summary report
        report_path = output_path / "noise_characterization_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("JAI CAMERA NOISE CHARACTERIZATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {results.timestamp}\n")
            f.write(f"Camera: {results.camera_name}\n")
            f.write(f"Total configurations tested: {len(results.results)}\n\n")

            # Group by gain for analysis
            gain_groups = {}
            for r in results.results:
                if r.unified_gain not in gain_groups:
                    gain_groups[r.unified_gain] = []
                gain_groups[r.unified_gain].append(r)

            f.write("=" * 70 + "\n")
            f.write("NOISE BY UNIFIED GAIN (at 25ms exposure)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Gain':>6} | {'R_stddev':>8} | {'G_stddev':>8} | {'B_stddev':>8} | {'R_mean':>8} | {'G_mean':>8} | {'B_mean':>8}\n")
            f.write("-" * 70 + "\n")

            for gain in sorted(gain_groups.keys()):
                # Find 25ms exposure result (or closest)
                group = gain_groups[gain]
                exp_25 = [r for r in group if abs(r.exposure_ms - 25.0) < 1.0]
                if exp_25:
                    r = exp_25[0]
                    f.write(
                        f"{gain:>6.1f} | {r.red_stddev:>8.2f} | {r.green_stddev:>8.2f} | {r.blue_stddev:>8.2f} | "
                        f"{r.red_mean:>8.1f} | {r.green_mean:>8.1f} | {r.blue_mean:>8.1f}\n"
                    )

            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 70 + "\n\n")

            # Find optimal settings
            # Best SNR at reasonable signal level
            good_results = [r for r in results.results
                          if r.red_mean > 50 and r.saturation_pct < 1.0]
            if good_results:
                best_snr = max(good_results, key=lambda r: min(r.red_snr, r.green_snr, r.blue_snr))
                f.write(f"Best balanced SNR:\n")
                f.write(f"  Unified gain: {best_snr.unified_gain}\n")
                f.write(f"  Exposure: {best_snr.exposure_ms}ms\n")
                f.write(f"  SNR: R={best_snr.red_snr:.1f}, G={best_snr.green_snr:.1f}, B={best_snr.blue_snr:.1f}\n\n")

            # Lowest noise at each gain
            f.write("Lowest noise settings per gain:\n")
            for gain in sorted(gain_groups.keys()):
                group = [r for r in gain_groups[gain] if r.saturation_pct < 5.0]
                if group:
                    lowest = min(group, key=lambda r: max(r.red_stddev, r.green_stddev, r.blue_stddev))
                    f.write(
                        f"  Gain {gain:.1f}: exp={lowest.exposure_ms:.0f}ms, "
                        f"max_stddev={max(lowest.red_stddev, lowest.green_stddev, lowest.blue_stddev):.2f}\n"
                    )

            f.write("\n")
            if results.notes:
                f.write(f"Notes: {results.notes}\n")

        logger.info(f"Saved report to {report_path}")

        # Generate visualization if matplotlib available
        try:
            self._generate_plots(results, output_path)
        except ImportError:
            logger.warning("matplotlib not available - skipping visualization")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

    def _generate_plots(
        self,
        results: NoiseCharacterizationResults,
        output_path: Path,
    ) -> None:
        """Generate noise characterization plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_path = Path(output_path)

        # Plot 1: Noise vs Gain at fixed exposure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Group by exposure
        exp_groups = {}
        for r in results.results:
            if r.exposure_ms not in exp_groups:
                exp_groups[r.exposure_ms] = []
            exp_groups[r.exposure_ms].append(r)

        colors = ['red', 'green', 'blue']
        for exp_ms in sorted(exp_groups.keys())[:4]:  # Limit to 4 exposures
            group = sorted(exp_groups[exp_ms], key=lambda r: r.unified_gain)
            gains = [r.unified_gain for r in group]

            for i, (ax, ch) in enumerate(zip(axes, colors)):
                stddevs = [getattr(r, f'{ch}_stddev') for r in group]
                ax.plot(gains, stddevs, 'o-', label=f'{exp_ms}ms')
                ax.set_xlabel('Unified Gain')
                ax.set_ylabel('Noise (StdDev)')
                ax.set_title(f'{ch.capitalize()} Channel Noise vs Gain')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'noise_vs_gain.png', dpi=150)
        plt.close()

        # Plot 2: SNR heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        gains = sorted(set(r.unified_gain for r in results.results))
        exposures = sorted(set(r.exposure_ms for r in results.results))

        for i, ch in enumerate(colors):
            snr_matrix = np.zeros((len(gains), len(exposures)))
            for r in results.results:
                gi = gains.index(r.unified_gain)
                ei = exposures.index(r.exposure_ms)
                snr_matrix[gi, ei] = getattr(r, f'{ch}_snr')

            im = axes[i].imshow(snr_matrix, aspect='auto', origin='lower',
                               extent=[0, len(exposures), 0, len(gains)])
            axes[i].set_xlabel('Exposure Index')
            axes[i].set_ylabel('Gain Index')
            axes[i].set_title(f'{ch.capitalize()} SNR Heatmap')
            plt.colorbar(im, ax=axes[i], label='SNR')

        plt.tight_layout()
        plt.savefig(output_path / 'snr_heatmap.png', dpi=150)
        plt.close()

        logger.info(f"Saved plots to {output_path}")


def main():
    """CLI entry point for noise characterization."""
    parser = argparse.ArgumentParser(
        description='JAI Camera Noise Characterization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full characterization
  python noise_characterization.py config.yml --output ./noise_results

  # Quick test
  python noise_characterization.py config.yml --quick

  # Custom grid
  python noise_characterization.py config.yml --gains 1.0,2.0,4.0 --exposures 10,25,50
"""
    )

    parser.add_argument('config_yaml',
                       help='Path to microscope configuration YAML file')
    parser.add_argument('--output', '-o', default='./noise_characterization',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick (smaller) test grid')
    parser.add_argument('--gains', type=str, default=None,
                       help='Comma-separated list of gains to test')
    parser.add_argument('--exposures', type=str, default=None,
                       help='Comma-separated list of exposures (ms) to test')
    parser.add_argument('--rb-gains', type=str, default=None,
                       help='Comma-separated list of R/B analog gains to test')
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to average per measurement')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Microscope server host')
    parser.add_argument('--port', type=int, default=5000,
                       help='Microscope server port')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Parse custom grids
    gains = None
    if args.gains:
        gains = [float(g) for g in args.gains.split(',')]

    exposures = None
    if args.exposures:
        exposures = [float(e) for e in args.exposures.split(',')]

    rb_gains = None
    if args.rb_gains:
        rb_gains = [float(g) for g in args.rb_gains.split(',')]

    # Connect to server
    try:
        from microscope_command_server.client import QuPathTestClient
        client = QuPathTestClient(host=args.host, port=args.port)
        client.connect()
        logger.info(f"Connected to server at {args.host}:{args.port}")
    except Exception as e:
        logger.error(f"Failed to connect to microscope server: {e}")
        logger.error("Make sure the microscope server is running.")
        sys.exit(1)

    # Note: This script requires direct hardware access which isn't available
    # through the socket protocol. For socket-based operation, the characterization
    # would need to be implemented as a server command.
    logger.warning(
        "Note: Full noise characterization requires direct hardware access. "
        "This CLI is a template - actual usage requires running on the same machine "
        "as the microscope server with direct hardware access."
    )

    # For now, print what would be tested
    logger.info("Would test the following configurations:")
    test_gains = gains or (JAINoiseCharacterization.QUICK_GAINS if args.quick
                          else JAINoiseCharacterization.DEFAULT_GAINS)
    test_exposures = exposures or (JAINoiseCharacterization.QUICK_EXPOSURES if args.quick
                                   else JAINoiseCharacterization.DEFAULT_EXPOSURES)
    test_rb = rb_gains or JAINoiseCharacterization.DEFAULT_RB_GAINS

    total = len(test_gains) * len(test_exposures) * len(test_rb)
    logger.info(f"  Gains: {test_gains}")
    logger.info(f"  Exposures: {test_exposures}")
    logger.info(f"  R/B gains: {test_rb}")
    logger.info(f"  Total configurations: {total}")
    logger.info(f"  Output directory: {args.output}")


if __name__ == '__main__':
    main()
