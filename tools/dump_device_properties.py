#!/usr/bin/env python3
"""Dump all Micro-Manager device properties to a text file.

This script connects to a running Micro-Manager instance via Pycromanager
and writes every loaded device's properties (name, current value, read/write
status) to a structured text file.

Usage:
    1. Start Micro-Manager with your hardware configuration loaded
    2. Run this script from the microscope_control venv:

        python tools/dump_device_properties.py --config PPM
        python tools/dump_device_properties.py --config OWS3_widefield
        python tools/dump_device_properties.py --config OWS3_multiphoton

    3. The output file is written to the current directory:
        mm_device_properties_PPM_20260422.txt

    The --config label identifies which MM .cfg was loaded, since
    available devices and properties change between configurations
    (e.g., OWS3 widefield has camera + XY stage, multiphoton adds
    the laser scanning PMTs and galvo mirrors).

When to use:
    - After initial hardware setup to document all available properties
    - When debugging property access issues (e.g., stage speed control,
      camera settings, filter wheel positions)
    - Before and after Micro-Manager or device adapter updates to detect
      property changes
    - To share hardware configuration with developers who don't have
      physical access to the microscope

The output file should be committed to microscope_configurations/ for
reference (e.g., mm_device_properties_PPM.txt). This gives developers
a complete picture of available device properties without needing to
run the Device Property Browser interactively.
"""

import argparse
import datetime
import platform
import sys


def dump_properties(output_path=None, config_label=None):
    """Connect to Micro-Manager and dump all device properties.

    Args:
        output_path: Output file path. If None, auto-generated from
                     config_label (or hostname) + date.
        config_label: Label identifying which MM config is loaded
                      (e.g., "PPM", "OWS3_widefield", "OWS3_multiphoton").
                      Included in the default filename and file header.
    """
    try:
        from pycromanager import Core
    except ImportError:
        print("ERROR: pycromanager not installed. Run: pip install pycromanager")
        sys.exit(1)

    try:
        core = Core()
    except Exception as e:
        print(f"ERROR: Could not connect to Micro-Manager: {e}")
        print("Make sure Micro-Manager is running with a configuration loaded.")
        sys.exit(1)

    # Default output filename
    if output_path is None:
        label = config_label or platform.node().replace(" ", "_")
        date = datetime.date.today().strftime("%Y%m%d")
        output_path = f"mm_device_properties_{label}_{date}.txt"

    devices = list(core.get_loaded_devices())
    print(f"Found {len(devices)} loaded devices")

    lines = []
    lines.append(f"Micro-Manager Device Properties Dump")
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    if config_label:
        lines.append(f"MM Config: {config_label}")
    lines.append(f"Host: {platform.node()}")
    lines.append(f"OS: {platform.system()} {platform.release()}")
    lines.append(f"Devices: {len(devices)}")
    lines.append("")

    # Summary table
    lines.append("=" * 70)
    lines.append("DEVICE SUMMARY")
    lines.append("=" * 70)
    for dev in devices:
        try:
            dev_type = core.get_device_type(dev)
            dev_lib = core.get_device_library(dev)
            lines.append(f"  {dev:30s}  type={dev_type}  lib={dev_lib}")
        except Exception:
            lines.append(f"  {dev:30s}  (could not read type/library)")
    lines.append("")

    # Per-device properties
    for dev in devices:
        try:
            props = list(core.get_device_property_names(dev))
        except Exception:
            props = []

        lines.append("=" * 70)
        lines.append(f"DEVICE: {dev}  ({len(props)} properties)")
        lines.append("=" * 70)

        for prop in props:
            try:
                value = core.get_property(dev, prop)
            except Exception:
                value = "<error reading>"

            try:
                read_only = core.is_property_read_only(dev, prop)
                rw = "RO" if read_only else "RW"
            except Exception:
                rw = "??"

            # Check if property has allowed values
            allowed = ""
            try:
                vals = list(core.get_allowed_property_values(dev, prop))
                if vals:
                    allowed = f"  allowed: {vals}"
            except Exception:
                pass

            lines.append(f"  {prop:35s} = {str(value):20s} [{rw}]{allowed}")

        lines.append("")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Written {len(devices)} devices to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Dump all Micro-Manager device properties to a text file.",
        epilog="Examples:\n"
               "  python dump_device_properties.py --config PPM\n"
               "  python dump_device_properties.py --config OWS3_widefield\n"
               "  python dump_device_properties.py --config OWS3_multiphoton\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        help="Label for the MM config currently loaded (e.g., PPM, "
             "OWS3_widefield, OWS3_multiphoton). Used in the filename "
             "and file header.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: mm_device_properties_<config>_<date>.txt)",
    )
    args = parser.parse_args()
    dump_properties(args.output, args.config)


if __name__ == "__main__":
    main()
