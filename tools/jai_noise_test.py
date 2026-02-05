"""
JAI Camera Noise Characterization Tool
=======================================

Standalone script that communicates with the microscope command server via
direct socket protocol to characterize noise behavior across different
gain and exposure settings for the JAI AP-3200T-USB 3-CCD prism camera.

Uses the 8-byte binary protocol (see protocol.py in microscope_command_server).
Does NOT import any project packages - fully standalone.

Usage:
    python jai_noise_test.py [--host HOST] [--port PORT] [--output-dir DIR]

Output:
    - CSV file with all measurements
    - PNG plots of noise vs gain, noise vs exposure, SNR vs gain, etc.
"""

import argparse
import csv
import os
import socket
import struct
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# ---- Protocol constants (mirror protocol.py) ----
SETGAIN = b"setgain_"
SETEXP = b"setexp__"
GETNOISE = b"getnoise"
SETMODE = b"setmode_"


class MicroscopeConnection:
    """Simple socket client for the microscope command server."""

    def __init__(self, host="127.0.0.1", port=5000, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _send_command(self, cmd_bytes, payload=None):
        """Send 8-byte command + optional payload."""
        self.sock.sendall(cmd_bytes)
        if payload:
            self.sock.sendall(payload)

    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def set_camera_mode(self, exp_individual=True, gain_individual=False):
        """Set camera mode: exposure individual, gain always unified."""
        payload = bytes([1 if exp_individual else 0, 1 if gain_individual else 0])
        self._send_command(SETMODE, payload)
        resp = self._recv_exact(8)
        resp_str = resp.decode("utf-8", errors="replace").strip().replace("\0", "")
        if not resp_str.startswith("ACK"):
            raise RuntimeError(f"SETMODE failed: {resp_str}")

    def set_gains(self, unified_gain, analog_red=1.0, analog_blue=1.0):
        """Set gains: count=3 -> [unified_gain, analog_red, analog_blue]."""
        payload = struct.pack(">Bfff", 3, unified_gain, analog_red, analog_blue)
        self._send_command(SETGAIN, payload)
        resp = self._recv_exact(8)
        resp_str = resp.decode("utf-8", errors="replace").strip().replace("\0", "")
        if not resp_str.startswith("ACK"):
            raise RuntimeError(f"SETGAIN failed: {resp_str}")

    def set_exposure_unified(self, exposure_ms):
        """Set unified exposure (count=1)."""
        payload = struct.pack(">Bf", 1, exposure_ms)
        self._send_command(SETEXP, payload)
        resp = self._recv_exact(8)
        resp_str = resp.decode("utf-8", errors="replace").strip().replace("\0", "")
        if not resp_str.startswith("ACK"):
            raise RuntimeError(f"SETEXP failed: {resp_str}")

    def get_noise(self, num_frames=10):
        """Get noise stats. Returns dict with per-channel mean, std, snr."""
        payload = bytes([num_frames])
        self._send_command(GETNOISE, payload)
        # Response: 9 big-endian floats (36 bytes)
        resp = self._recv_exact(36)
        values = struct.unpack(">fffffffff", resp)
        return {
            "red_mean": values[0],
            "green_mean": values[1],
            "blue_mean": values[2],
            "red_std": values[3],
            "green_std": values[4],
            "blue_std": values[5],
            "red_snr": values[6],
            "green_snr": values[7],
            "blue_snr": values[8],
        }


def run_test_grid(conn, output_dir, num_noise_frames=10):
    """Run the full test grid and collect measurements."""

    # Test parameters
    unified_gains = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    analog_rb_gains = [0.48, 1.0, 2.0, 3.0, 4.0]
    exposures_ms = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    results = []

    # Ensure camera is in unified gain + individual exposure mode
    print("Setting camera mode: exposure=individual, gain=unified")
    conn.set_camera_mode(exp_individual=False, gain_individual=False)
    time.sleep(0.5)

    total_tests = (
        len(unified_gains) * len(exposures_ms)
        + len(analog_rb_gains) * 2  # R and B sweeps at fixed gain/exposure
    )
    test_num = 0

    # ---- Test 1: Unified gain sweep at fixed exposure ----
    print("\n=== Test 1: Noise vs Unified Gain (fixed exposure) ===")
    for exp_ms in [25.0, 100.0]:  # Two reference exposures
        for gain in unified_gains:
            test_num += 1
            print(f"  [{test_num}] gain={gain:.1f}, exp={exp_ms:.0f}ms ... ", end="", flush=True)
            try:
                conn.set_gains(gain, 1.0, 1.0)
                conn.set_exposure_unified(exp_ms)
                time.sleep(0.3)  # Settle
                noise = conn.get_noise(num_noise_frames)
                noise["unified_gain"] = gain
                noise["analog_red"] = 1.0
                noise["analog_blue"] = 1.0
                noise["exposure_ms"] = exp_ms
                noise["test_type"] = "unified_gain_sweep"
                results.append(noise)
                print(f"R={noise['red_std']:.2f} G={noise['green_std']:.2f} B={noise['blue_std']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")

    # ---- Test 2: Exposure sweep at fixed gain ----
    print("\n=== Test 2: Noise vs Exposure (fixed gain) ===")
    for gain in [1.0, 5.0]:  # Two reference gains
        for exp_ms in exposures_ms:
            test_num += 1
            print(f"  [{test_num}] gain={gain:.1f}, exp={exp_ms:.0f}ms ... ", end="", flush=True)
            try:
                conn.set_gains(gain, 1.0, 1.0)
                conn.set_exposure_unified(exp_ms)
                time.sleep(0.3)
                noise = conn.get_noise(num_noise_frames)
                noise["unified_gain"] = gain
                noise["analog_red"] = 1.0
                noise["analog_blue"] = 1.0
                noise["exposure_ms"] = exp_ms
                noise["test_type"] = "exposure_sweep"
                results.append(noise)
                print(f"R={noise['red_std']:.2f} G={noise['green_std']:.2f} B={noise['blue_std']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")

    # ---- Test 3: R/B analog gain sweep at fixed unified gain ----
    print("\n=== Test 3: R/B Analog Gain Effect on Noise ===")
    fixed_gain = 5.0
    fixed_exp = 50.0
    for rb_gain in analog_rb_gains:
        test_num += 1
        # Red sweep
        print(f"  [{test_num}] unified={fixed_gain:.1f}, analog_red={rb_gain:.2f}, exp={fixed_exp:.0f}ms ... ",
              end="", flush=True)
        try:
            conn.set_gains(fixed_gain, rb_gain, 1.0)
            conn.set_exposure_unified(fixed_exp)
            time.sleep(0.3)
            noise = conn.get_noise(num_noise_frames)
            noise["unified_gain"] = fixed_gain
            noise["analog_red"] = rb_gain
            noise["analog_blue"] = 1.0
            noise["exposure_ms"] = fixed_exp
            noise["test_type"] = "analog_red_sweep"
            results.append(noise)
            print(f"R_std={noise['red_std']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

        test_num += 1
        # Blue sweep
        print(f"  [{test_num}] unified={fixed_gain:.1f}, analog_blue={rb_gain:.2f}, exp={fixed_exp:.0f}ms ... ",
              end="", flush=True)
        try:
            conn.set_gains(fixed_gain, 1.0, rb_gain)
            conn.set_exposure_unified(fixed_exp)
            time.sleep(0.3)
            noise = conn.get_noise(num_noise_frames)
            noise["unified_gain"] = fixed_gain
            noise["analog_red"] = 1.0
            noise["analog_blue"] = rb_gain
            noise["exposure_ms"] = fixed_exp
            noise["test_type"] = "analog_blue_sweep"
            results.append(noise)
            print(f"B_std={noise['blue_std']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Reset to safe defaults
    print("\nResetting gains/exposure to defaults...")
    try:
        conn.set_gains(1.0, 1.0, 1.0)
        conn.set_exposure_unified(50.0)
    except Exception as e:
        print(f"Warning: Reset failed: {e}")

    return results


def save_csv(results, output_dir):
    """Save results to CSV."""
    csv_path = os.path.join(output_dir, "noise_measurements.csv")
    if not results:
        print("No results to save")
        return csv_path

    fieldnames = [
        "test_type", "unified_gain", "analog_red", "analog_blue", "exposure_ms",
        "red_mean", "green_mean", "blue_mean",
        "red_std", "green_std", "blue_std",
        "red_snr", "green_snr", "blue_snr",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} measurements to {csv_path}")
    return csv_path


def generate_plots(results, output_dir):
    """Generate noise characterization plots."""
    if not results:
        print("No results to plot")
        return

    # Convert to numpy arrays for easier filtering
    arr = {k: np.array([r[k] for r in results]) for k in results[0].keys() if k != "test_type"}
    test_types = [r["test_type"] for r in results]

    # ---- Plot 1: Noise vs Unified Gain ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp_ms in [25.0, 100.0]:
        mask = np.array([
            t == "unified_gain_sweep" and r["exposure_ms"] == exp_ms
            for t, r in zip(test_types, results)
        ])
        if mask.any():
            gains = arr["unified_gain"][mask]
            ax.plot(gains, arr["red_std"][mask], "r-o", label=f"Red (exp={exp_ms:.0f}ms)", alpha=0.8)
            ax.plot(gains, arr["green_std"][mask], "g-s", label=f"Green (exp={exp_ms:.0f}ms)", alpha=0.8)
            ax.plot(gains, arr["blue_std"][mask], "b-^", label=f"Blue (exp={exp_ms:.0f}ms)", alpha=0.8)
    ax.set_xlabel("Unified Gain")
    ax.set_ylabel("Noise (StdDev)")
    ax.set_title("Noise vs Unified Gain (per channel)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_vs_unified_gain.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 2: Noise vs Exposure ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for gain in [1.0, 5.0]:
        mask = np.array([
            t == "exposure_sweep" and r["unified_gain"] == gain
            for t, r in zip(test_types, results)
        ])
        if mask.any():
            exps = arr["exposure_ms"][mask]
            ax.plot(exps, arr["red_std"][mask], "r-o", label=f"Red (gain={gain:.0f})", alpha=0.8)
            ax.plot(exps, arr["green_std"][mask], "g-s", label=f"Green (gain={gain:.0f})", alpha=0.8)
            ax.plot(exps, arr["blue_std"][mask], "b-^", label=f"Blue (gain={gain:.0f})", alpha=0.8)
    ax.set_xlabel("Exposure (ms)")
    ax.set_ylabel("Noise (StdDev)")
    ax.set_title("Noise vs Exposure Time (per channel)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_vs_exposure.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 3: SNR vs Unified Gain ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp_ms in [25.0, 100.0]:
        mask = np.array([
            t == "unified_gain_sweep" and r["exposure_ms"] == exp_ms
            for t, r in zip(test_types, results)
        ])
        if mask.any():
            gains = arr["unified_gain"][mask]
            ax.plot(gains, arr["red_snr"][mask], "r-o", label=f"Red (exp={exp_ms:.0f}ms)", alpha=0.8)
            ax.plot(gains, arr["green_snr"][mask], "g-s", label=f"Green (exp={exp_ms:.0f}ms)", alpha=0.8)
            ax.plot(gains, arr["blue_snr"][mask], "b-^", label=f"Blue (exp={exp_ms:.0f}ms)", alpha=0.8)
    ax.set_xlabel("Unified Gain")
    ax.set_ylabel("SNR (Mean / StdDev)")
    ax.set_title("SNR vs Unified Gain (per channel)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "snr_vs_unified_gain.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 4: R/B Analog Gain Effect ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Red analog gain effect
    mask_r = np.array([t == "analog_red_sweep" for t in test_types])
    if mask_r.any():
        rb = arr["analog_red"][mask_r]
        ax1.plot(rb, arr["red_std"][mask_r], "r-o", label="Red StdDev")
        ax1.plot(rb, arr["red_mean"][mask_r], "r--s", label="Red Mean", alpha=0.5)
        ax1.set_xlabel("Analog Red Gain")
        ax1.set_ylabel("Value")
        ax1.set_title("Red Analog Gain Effect")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    # Blue analog gain effect
    mask_b = np.array([t == "analog_blue_sweep" for t in test_types])
    if mask_b.any():
        bb = arr["analog_blue"][mask_b]
        ax2.plot(bb, arr["blue_std"][mask_b], "b-o", label="Blue StdDev")
        ax2.plot(bb, arr["blue_mean"][mask_b], "b--s", label="Blue Mean", alpha=0.5)
        ax2.set_xlabel("Analog Blue Gain")
        ax2.set_ylabel("Value")
        ax2.set_title("Blue Analog Gain Effect")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "analog_gain_effect.png"), dpi=150)
    plt.close(fig)

    print(f"Saved plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="JAI Camera Noise Characterization Tool")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ./noise_results_YYYYMMDD_HHMMSS)")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames for noise measurement (default: 10)")
    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"noise_results_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    conn = MicroscopeConnection(args.host, args.port)
    try:
        conn.connect()
        results = run_test_grid(conn, args.output_dir, args.frames)
        save_csv(results, args.output_dir)
        generate_plots(results, args.output_dir)
        print(f"\nDone! Results in: {args.output_dir}")
    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to {args.host}:{args.port}")
        print("Make sure the microscope command server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
