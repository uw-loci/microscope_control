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
| QCamera | Exposure | 0.01-10000 ms | Teledyne MicroPublisher 6 |

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
