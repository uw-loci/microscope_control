# White Balance Developer Guide

## 1. Overview

White balance (WB) ensures that a neutral-colored subject (e.g., a blank glass slide) produces equal intensity across all color channels (R, G, B) in the captured image. Without white balance, color cameras introduce color casts that corrupt quantitative measurements -- particularly problematic in polarized light microscopy (PPM), where subtle color differences encode birefringence information.

This guide covers:
- What white balance knobs exist on camera hardware and in software
- How the current JAI implementation works
- How to implement white balance for a new camera

## 2. The WB Problem: What Knobs Exist

White balance can be achieved through **hardware controls** (before digitization) or **software processing** (after capture). Each approach has different tradeoffs.

### Hardware Controls

| Control | What It Does | Where It Lives |
|---------|-------------|----------------|
| **Per-channel exposure** | Sets independent exposure times for R, G, B channels | Camera firmware (JAI trilinear) |
| **Per-channel gain** | Applies analog amplification per channel before ADC | Camera firmware |
| **Unified gain** | Single gain value applied equally to all channels | Camera firmware |
| **Hardware auto-WB** | Camera automatically adjusts internal WB parameters | Camera firmware (Off/Continuous/Once modes) |

### Software Controls

| Control | What It Does | Where It Lives |
|---------|-------------|----------------|
| **Post-processing gain** | Multiplies pixel values per channel after capture | `ppm_library/pipeline.py` |
| **Background correction** | Divides image by a background reference image | `ppm_library/pipeline.py` |

### Hardware vs Software WB Tradeoffs

| Factor | Hardware WB | Software WB |
|--------|------------|-------------|
| **Dynamic range** | Preserved -- adjustment happens before digitization | Reduced -- boosting dark channels amplifies noise |
| **Noise** | Minimal -- analog gain is cleaner than digital | Higher -- digital multiplication amplifies quantization noise |
| **Speed** | No processing overhead | Requires per-pixel computation |
| **Reproducibility** | Excellent if using manual calibration (WBSIMPLE/WBPPM) | Excellent (pure math) |
| **Camera dependency** | Requires per-channel exposure/gain support | Works with any camera |
| **Complexity** | Higher -- needs camera-specific property access | Lower -- generic image math |

**Bottom line:** Use hardware WB when the camera supports it (better image quality). Fall back to software WB for cameras without per-channel control.

## 3. Current JAI Implementation

### Architecture Overview

```
JAI White Balance System
========================

Calibration time (WBSIMPLE/WBPPM commands):
  QuPath (Java) --> Socket --> qp_server.py --> JAIWhiteBalanceCalibrator
                                                  |
                                          Uses JAICameraProperties to:
                                          - Set per-channel exposures
                                          - Set per-channel gains
                                          - Measure resulting intensity
                                          - Iterate until target reached
                                                  |
                                          Writes results to:
                                          imageprocessing_{microscope}.yml

Acquisition time (during tile capture):
  workflow.py::load_jai_calibration_from_imageprocessing()
      --> Reads calibrated values from imageprocessing YAML
  workflow.py::apply_jai_calibration_for_angle()
      --> Applies per-channel exposure/gain via JAICameraProperties
      --> Captures image with white-balanced settings
      --> Caller resets to unified mode after capture
```

### Key Files

| File | Role |
|------|------|
| `microscope_control/microscope_control/jai/calibration.py` | `JAIWhiteBalanceCalibrator` -- iterative calibration algorithm |
| `microscope_control/microscope_control/jai/properties.py` | `JAICameraProperties` -- low-level camera property access |
| `microscope_command_server/.../server/protocol.py` | Socket command definitions (WBSIMPLE, WBPPM, etc.) |
| `microscope_command_server/.../server/qp_server.py` | Socket command handlers |
| `microscope_command_server/.../acquisition/workflow.py` | Acquisition-time calibration loading and application |
| `microscope_configurations/imageprocessing_PPM.yml` | Calibration results storage (per-angle exposures and gains) |

### Calibration Algorithm Summary

The `JAIWhiteBalanceCalibrator.calibrate_simple()` method:

1. Starts with a user-provided initial exposure (applied to all 3 channels)
2. Captures a frame and measures mean intensity per channel
3. Adjusts per-channel exposures proportionally to reach the target intensity
4. If any channel's exposure would exceed the soft cap, uses per-channel gain instead
5. Iterates until all channels are within tolerance of the target, or max iterations reached
6. Saves the final per-channel exposures and gains

For PPM (`calibrate_ppm()`), this process repeats independently for each of the 4 polarizer angles, with gains reset to 1.0 between angles to prevent carryover.

### PPM Multi-Angle Calibration

PPM captures at 4 polarizer angles with very different light levels:

| Angle | Name | Typical Intensity | Why |
|-------|------|-------------------|-----|
| 0 deg | Crossed | Very dim (~125) | Polarizers at 90 deg, blocks most light |
| 90 deg | Uncrossed | Very bright (~245) | Polarizers parallel, maximum transmission |
| +7 deg | Positive | Moderate (~160) | Slight rotation, partial transmission |
| -7 deg | Negative | Moderate (~160) | Slight rotation, partial transmission |

Each angle gets independently calibrated per-channel exposures and gains, stored under `imaging_profiles.ppm.{objective}.{detector}.exposures_ms.{angle}.{r,g,b}` in the imageprocessing YAML.

### YAML Configuration Structure

```yaml
# imageprocessing_{microscope}.yml
imaging_profiles:
  ppm:
    LOCI_OBJECTIVE_OLYMPUS_20X_POL_001:       # Objective ID
      LOCI_DETECTOR_JAI_001:                   # Detector ID
        white_balance:
          enabled: false                       # Disable software WB (using hardware)
        white_balance_calibration:
          defocus_offset_um: 75                # Defocus for blank-field calibration
          target_intensity: 180                # 8-bit target (0-255)
          tolerance: 5                         # Acceptable deviation
          max_iterations: 15
          max_exposure_ratio: 2.0
          calibrate_black_level: true
        exposures_ms:
          uncrossed:
            all: 15                            # Unified exposure (reference)
            r: 12                              # Calibrated per-channel
            g: 15
            b: 18
          crossed:
            all: 1200
            r: 1150
            g: 1200
            b: 1250
          # ... (positive, negative)
        gains:
          uncrossed: {r: 1.2, g: 1.0, b: 1.1}
          crossed: {r: 1.1, g: 1.0, b: 1.05}
          # ... (positive, negative)

calibration_targets:
  target_intensities:
    uncrossed: 245.0
    positive: 160.0
    negative: 160.0
    crossed: 125.0
    default: 180.0
```

### Socket Protocol

All commands use 8-byte padded identifiers. See `protocol.py` for the full list.

| Command | Payload | JAI-Specific? | Purpose |
|---------|---------|---------------|---------|
| `WBSIMPLE` | Text flags | Yes | Single-condition calibration |
| `WBPPM` | Text flags (per-angle) | Yes | 4-angle PPM calibration |
| `SETMODE` | 2 bytes | Yes | Switch unified/individual mode |
| `SETEXP` | 1 count + N floats | count=1: generic, count>=3: JAI | Set exposure values |
| `SETGAIN` | 1 count + N floats | Yes (both paths) | Set gain values |
| `SETWBMD` | 1 byte | Yes | Hardware auto-WB mode control |

## 4. Implementing WB for a New Camera

Follow these steps when adding white balance support for a camera other than JAI.

### Step 1: Assess Camera Capabilities

Determine what WB controls the camera exposes through Micro-Manager:

- **Per-channel exposure?** (rare -- mostly trilinear cameras like JAI)
- **Per-channel gain?** (more common in industrial cameras)
- **Hardware auto-WB?** (common but not reproducible)
- **Only unified exposure/gain?** (most consumer/scientific cameras)

If the camera only supports unified controls, hardware WB is not possible -- use software WB in `pipeline.py` instead.

### Step 2: Create Properties Module

Create a new package under `microscope_control/microscope_control/{camera}/`:

```
microscope_control/microscope_control/{camera}/
    __init__.py          # Export main classes
    properties.py        # Camera property access (like JAICameraProperties)
    calibration.py       # Calibration algorithm (if applicable)
```

The properties module should provide:
- Camera validation (confirm the expected camera is connected)
- Per-channel exposure getters/setters (if supported)
- Per-channel gain getters/setters (if supported)
- Mode switching (unified vs individual, if applicable)

### Step 3: Implement Calibration

If the camera supports per-channel controls, implement an iterative calibration similar to `JAIWhiteBalanceCalibrator`:

1. Set initial exposure (equal across channels)
2. Capture and measure per-channel intensity
3. Adjust channels proportionally toward target
4. Use gain as overflow when exposure limits are reached
5. Iterate until convergence

Key design decisions:
- **Target intensity**: What 8-bit-equivalent value to aim for (typically 160-200)
- **Convergence tolerance**: How close is "close enough" (typically 3-5 units)
- **Max iterations**: Upper bound to prevent infinite loops (typically 10-20)
- **Gain threshold**: When to switch from exposure adjustment to gain compensation

### Step 4: Add Socket Commands (if remote calibration needed)

If calibration must be triggered from QuPath over the socket:

1. Define new command constants in `protocol.py` (8-byte padded)
2. Add handler blocks in `qp_server.py` following the existing pattern
3. Document: protocol format, response codes, JAI-specificity, cleanup behavior

Follow the existing handler patterns:
- Send `STARTED:` acknowledgment immediately
- Send `SUCCESS:` or `FAILED:` on completion
- Reset camera state in a `finally` block

### Step 5: Update YAML Configuration

Add camera-specific sections to `imageprocessing_{microscope}.yml`:

```yaml
imaging_profiles:
  {modality}:
    {objective_id}:
      {detector_id}:
        white_balance:
          enabled: false         # false = using hardware WB
        white_balance_calibration:
          # Camera-specific calibration parameters
        exposures_ms:
          # Calibration results (populated by calibrator)
        gains:
          # Calibration results (populated by calibrator)
```

### Step 6: Update Acquisition-Time Application

Modify or extend `workflow.py` to:
- Load calibration for the new camera type
- Apply camera-specific settings before capture
- Reset camera state after capture

The existing functions check for `"JAI"` in the camera name. For a new camera, you can either:
- Add a new function pair (e.g., `load_{camera}_calibration` / `apply_{camera}_calibration`)
- Generalize the existing functions with a camera-type dispatch

## 5. Key Gotchas

### Frame Rate vs Exposure

On some cameras (including JAI), the maximum exposure time is limited by the current frame rate setting. If the frame rate is set to 30 fps, no channel can exceed ~33ms exposure. The calibrator must either lower the frame rate first or use gain compensation when exposure limits are reached.

### Mode Switching Side Effects

Switching between unified and individual exposure/gain modes on the JAI camera can reset channel values to defaults. Always set mode FIRST, then set values. The `auto_enable=True` parameter on `set_channel_exposures()` and `set_analog_gains()` handles this automatically.

### Live Mode Interaction

If Micro-Manager's live mode is active during calibration, it can interfere with frame capture. The calibrator should either stop live mode before calibration or use a separate capture path.

### Camera AWB (Auto White Balance)

The JAI camera's hardware AWB **cannot be reliably controlled through Pycromanager** due to issues on both the camera and MicroManager side. The programmatic `run_auto_white_balance()` and `set_white_balance_mode()` methods have been removed.

**To use Camera AWB:**

1. Open MicroManager's **Device Property Browser**
2. Find **JAICamera -> WhiteBalance**
3. Set to **"Continuous"** (ensure Live mode is active so the camera receives frames)
4. Wait for colors to converge (~3-5 seconds)
5. Set WhiteBalance back to **"Off"**

**To clear AWB:** Restart MicroManager and wait ~30 seconds. The `clear_awb_corrections()` method clears analog gains and sets WhiteBalance to Off, but AWB corrections set through MicroManager's GUI may persist.

**Temperature property (physical sensor temperature):**
- The `Temperature` property (JAICamera-Temperature) is the physical sensor temperature in degrees C -- a thermal reading
- It is NOT a color temperature or white balance parameter
- AWB corrections are stored in an opaque internal pipeline and cannot be read via any exposed property

### Gain State Persistence

JAI per-channel gains persist across captures even after switching back to unified mode. The post-calibration cleanup in the socket handlers explicitly resets gains to (1.0, 1.0, 1.0) and switches back to unified mode. Missing this reset causes gain settings to "leak" into subsequent acquisitions.

### Unified vs Per-Channel Interaction

On JAI cameras, enabling individual gain mode with unity values (1.0, 1.0, 1.0) behaves differently than having individual gain mode disabled entirely. Only enable individual gain mode when actual per-channel gain compensation is needed (any channel > 0.01 from 1.0).

### Post-Calibration Reset

After calibration completes (or fails), the `finally` block in the socket handlers must:
1. Disable individual exposure mode (back to unified)
2. Disable individual gain mode (back to unified)
3. Set all analog gains to (1.0, 1.0, 1.0)

This ensures the camera is returned to a clean state regardless of whether calibration succeeded.

### Saved Exposures as Starting Points

Background collection saves per-angle exposures to `calibration_targets.background_exposures` in `imageprocessing_PPM.yml` after each run. On the next background collection, these saved exposures are loaded as starting points for the adaptive exposure loop instead of the client's defaults (typically 10ms).

This is critical for angles like uncrossed/90-deg at 20x, where the correct exposure is ~0.5ms. Starting from 10ms and using proportional reduction of ~13.5% per step would need ~20 iterations to converge. Starting from the saved 0.5ms value converges immediately.

Saved exposures are loaded from `background_exposures.angles.{name}.exposure_ms` keyed by `angle_degrees`. If no saved data exists (first run), the client-provided defaults are used as fallback.

See `workflow.py:simple_background_collection()` (line ~3647) for the implementation.

### Adaptive Exposure Halving for Saturated Images

When all channels in a background image are fully clipped (median >= 254), the normal proportional reduction factor `(target * 0.90) / median = (245 * 0.90) / 255 = 0.865` only reduces exposure by ~13.5% per step. This is far too slow to reach a valid exposure range from a very high starting point.

When `max_ch_median >= 254`, the code halves the exposure instead (`reduction = 0.5`), reaching 0.6ms from 10ms in only 4 iterations. Partial saturation (median 245-253) still uses proportional reduction since there is some intensity information to guide the adjustment.

See `workflow.py:acquire_background_with_target_intensity()` (line ~2976) for the implementation.

### ASCII-Only Policy

Per project policy, all Python logging, error messages, and comments must use ASCII characters only. Do not use unicode arrows, degree symbols, or special characters. Use `->` not `-->`, `deg` not the degree symbol, `um` not the micro symbol. See CLAUDE.md for the full policy.
