# Hardware Device Properties Guide

## Overview

Micro-Manager exposes hardware capabilities through **device properties** -- named key-value pairs on each loaded device. Understanding what properties exist and their allowed values is essential for:

- Setting stage speed, camera exposure, gain, ROI
- Querying hardware state (position, busy, temperature)
- Debugging when a property name doesn't match expectations
- Developing new features that interact with hardware

## Dumping Device Properties

Use the included tool to export all device properties from a running Micro-Manager instance:

```bash
# From the microscope_control venv, with Micro-Manager running:
python tools/dump_device_properties.py

# Or specify an output path:
python tools/dump_device_properties.py --output mm_device_properties_PPM.txt
```

The output includes:
- Device summary (name, type, library)
- Every property on every device with current value, read/write status, and allowed values

### When to run this

- **After initial hardware setup** -- commit the output to `microscope_configurations/` as a reference
- **After MM or adapter updates** -- compare before/after to detect property changes
- **When debugging property access** -- find the exact property name for stage speed, camera mode, etc.
- **For remote developers** -- share the full hardware picture without physical microscope access

### Where to save the output

Commit dump files to `microscope_configurations/` with descriptive names:
```
microscope_configurations/
  mm_device_properties_PPM.txt      # PPM upright scope
  mm_device_properties_OWS3.txt     # OWS3 inverted scope
```

## Common Property Patterns

### Stage Speed (Prior ProScan)

The Prior ProScan controller exposes speed as a percentage (1-100) of maximum velocity:

| Device | Property | Range | Notes |
|--------|----------|-------|-------|
| ZStage | MaxSpeed | 1-100 | Z axis speed. "1" = slowest (~11.5 um/s) |
| XYStage | MaxSpeed | 1-100 | XY speed. "100" = full speed (~20 mm/s) |

**Important:** The property name and device name vary by MM adapter version. Always check the dump file for your specific configuration rather than hardcoding names.

### Camera Properties

| Device | Property | Example Values | Notes |
|--------|----------|---------------|-------|
| JAICamera | Exposure | 0.1-1000 ms | Per-channel on 3-CCD |
| JAICamera | WhiteBalance | Off, Once, Continuous | Must be Off during calibrated acquisition |
| JAICamera | FrameRateHz | 1-38 | Affects streaming capture rate |
| JAICamera | PixelFormat | BGR8, BGR12, etc. | Selects color depth; used for PPM high-bit mode |
| QCamera | Exposure | 0.01-10000 ms | Teledyne MicroPublisher 6 |

### JAI High-Bit-Depth PixelFormat Configuration (PPM)

The JAI 3-CCD camera delivers 8-bit RGB by default. For PPM (polarized light microscopy), the birefringence is computed from a normalized difference of angle images. Computing from higher-bit inputs (e.g., 12-bit) sharply reduces quantization noise in dark regions (crossed-polarizer angles).

To enable high-bit-depth capture, add a `high_bit_depth` block to the JAI detector entry in your microscope YAML configuration:

```yaml
detectors:
  LOCI_DETECTOR_JAI_001:
    # ... existing camera configuration ...
    high_bit_depth:
      property: PixelFormat           # MM device property name (discover via dump tool)
      high_value: BGR12               # Value to select high-bit format
      low_value: BGR8                 # Optional; if omitted, captured from HW when mode entered
      device: JAICamera               # Optional; defaults to the camera device name
      bit_depth: 12                   # Optional; real sensor bits, for logging/reference
```

**How to discover your camera's PixelFormat property values:**

1. Run the device properties dump tool:
   ```bash
   python tools/dump_device_properties.py --output mm_device_properties.txt
   ```

2. Search the output for `JAICamera` and look for any property containing "Pixel" or "Format"
   - Common examples: `PixelFormat`, `BitDepth`, `Color Depth`
   - Note the exact property name and all allowed values

3. Identify the 8-bit and higher-bit values (e.g., `BGR8` for 8-bit, `BGR12` for 12-bit)

4. Add them to the YAML block as shown above

**Behavior:**
- When high-bit mode is enabled, the camera's PixelFormat property is switched to the high-value
- Subsequent image captures return uint16 frames instead of uint8
- When high-bit mode is disabled, the previous format is restored (or falls back to `low_value` if the original format could not be captured)
- If the configuration is absent or incomplete, high-bit mode is a no-op and the camera stays at its current format
- The feature is safe to configure but inactive until explicitly triggered by acquisition workflows

### How Properties Are Used in QPSC

| Feature | Device | Property | Code Location |
|---------|--------|----------|---------------|
| Streaming AF Z speed | ZStage | MaxSpeed | `streaming_focus.py` |
| Rapid scan XY speed | XYStage | MaxSpeed | `rapid_scan.py` |
| Live viewer exposure | Camera | Exposure | `handlers/camera.py` |
| Camera ROI (AF crop) | Camera | various | `streaming_focus.py` |

## Troubleshooting

### "No speed property found"

The property search iterates candidates: MaxSpeed, Velocity, Speed, MaxVelocity. If none match:
1. Run the dump tool to see the actual property names
2. Check if the property is on a different device (e.g., controller hub vs. stage)
3. The Pycromanager ZMQ bridge returns Java StrVector objects -- `list()` may not convert them to Python strings for comparison

### Property exists but set_property fails

- Check if the property is read-only (marked `[RO]` in the dump)
- Check if the value is within allowed values (listed in the dump)
- Some properties require the device to be in a specific state (e.g., not during acquisition)
