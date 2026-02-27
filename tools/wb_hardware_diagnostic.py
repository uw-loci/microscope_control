"""
White Balance Hardware Diagnostic Script
=========================================

TEMPORARY TEST SCRIPT -- remove after hardware verification is complete.
See: claude-reports/white-balance/WB_HARDWARE_TEST_CLEANUP.md

Tests three specific JAI camera hardware behaviors:
  Issue 1: Do Camera AWB internal gains persist after reset?
  Issue 6: Do per-channel gains leak between PPM angle changes?
  Issue 7: Does frame rate auto-adjust for long exposures?

Requirements:
  - Micro-Manager running with JAI AP-3200T-USB connected
  - microscope_control package installed (pip install -e .)
  - Run from Windows machine with camera access

Usage:
    python wb_hardware_diagnostic.py [--test 1|6|7|all] [--verbose]

All values are hardcoded from imageprocessing_PPM.yml (20x objective).
Edit the HARDCODED VALUES section below to match your current setup.
"""

import argparse
import logging
import sys
import time

# ---------------------------------------------------------------------------
# HARDCODED VALUES -- edit these to match your setup
# Source: microscope_configurations/imageprocessing_PPM.yml (20x + JAI)
# ---------------------------------------------------------------------------

# PPM angles with calibrated per-channel exposures (ms) and gains
PPM_ANGLES = {
    "uncrossed": {
        "exposure_r": 0.66, "exposure_g": 0.97, "exposure_b": 4.76,
        "unified_gain": 1.0, "analog_red": 1.0, "analog_blue": 1.0,
    },
    "negative": {
        "exposure_r": 6.1, "exposure_g": 6.19, "exposure_b": 76.03,
        "unified_gain": 3.0, "analog_red": 1.0, "analog_blue": 1.0,
    },
    "crossed": {
        "exposure_r": 17.65, "exposure_g": 19.13, "exposure_b": 68.17,
        "unified_gain": 3.0, "analog_red": 1.0, "analog_blue": 1.0,
    },
    "positive": {
        "exposure_r": 12.06, "exposure_g": 29.74, "exposure_b": 40.39,
        "unified_gain": 2.0, "analog_red": 1.0, "analog_blue": 1.0,
    },
}

# Long exposure for frame rate coupling test (ms)
LONG_EXPOSURE_MS = 500.0
VERY_LONG_EXPOSURE_MS = 2000.0

# Settle time after property changes (seconds)
SETTLE_TIME = 0.3

# ---------------------------------------------------------------------------
# End of hardcoded values
# ---------------------------------------------------------------------------

logger = logging.getLogger("wb_diagnostic")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logging.basicConfig(level=level, handlers=[handler])


def connect_camera():
    """Initialize Pycromanager and return (JAICameraProperties, PycromanagerHardware)."""
    from microscope_control.hardware.pycromanager import init_pycromanager, PycromanagerHardware
    from microscope_control.jai.properties import JAICameraProperties

    logger.info("Connecting to Micro-Manager via Pycromanager...")
    core, studio = init_pycromanager(timeout_seconds=15)
    if core is None:
        logger.error("Failed to connect to Micro-Manager. Is it running?")
        sys.exit(1)

    props = JAICameraProperties(core)
    if not props.validate_camera():
        logger.error("JAI camera not detected or not active.")
        sys.exit(1)

    hardware = PycromanagerHardware(core, studio, settings={})
    logger.info("JAI camera connected and validated.")
    return props, hardware


def snap_rgb_means(hardware):
    """Snap an image and return per-channel mean values as (R, G, B) tuple."""
    import numpy as np
    img, _tags = hardware.snap_image()
    if img.ndim == 3 and img.shape[2] >= 3:
        r_mean = float(np.mean(img[:, :, 0]))
        g_mean = float(np.mean(img[:, :, 1]))
        b_mean = float(np.mean(img[:, :, 2]))
    else:
        val = float(np.mean(img))
        r_mean = g_mean = b_mean = val
    return r_mean, g_mean, b_mean


def log_rgb(label, rgb):
    """Log RGB means with a label."""
    r, g, b = rgb
    # Color balance ratio: how far from neutral (1.0 = perfectly balanced)
    avg = (r + g + b) / 3.0 if (r + g + b) > 0 else 1.0
    ratio_r = r / avg if avg > 0 else 0
    ratio_b = b / avg if avg > 0 else 0
    logger.info(
        f"[{label}] RGB means: R={r:.1f}  G={g:.1f}  B={b:.1f}  "
        f"(R/avg={ratio_r:.3f}, B/avg={ratio_b:.3f})"
    )
    return rgb


def dump_state(props, label=""):
    """Read and print all relevant camera state."""
    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}--- Camera State Snapshot ---")

    # Exposure mode
    exp_indiv = props.is_individual_exposure_enabled()
    logger.info(f"{prefix}ExposureIsIndividual: {exp_indiv}")

    # Per-channel exposures (always readable regardless of mode)
    try:
        ch_exp = props.get_channel_exposures()
        logger.info(
            f"{prefix}Channel exposures: "
            f"R={ch_exp['red']:.3f}ms  G={ch_exp['green']:.3f}ms  B={ch_exp['blue']:.3f}ms"
        )
    except Exception as e:
        logger.warning(f"{prefix}Could not read channel exposures: {e}")

    # Gain mode
    gain_indiv = props.is_individual_gain_enabled()
    logger.info(f"{prefix}GainIsIndividual: {gain_indiv}")

    # Unified gain
    try:
        ug = props.get_unified_gain()
        logger.info(f"{prefix}Unified gain: {ug:.3f}")
    except Exception as e:
        logger.warning(f"{prefix}Could not read unified gain: {e}")

    # R/B analog gains
    try:
        rb = props.get_rb_analog_gains()
        logger.info(f"{prefix}Analog gains: R={rb['red']:.4f}  B={rb['blue']:.4f}")
    except Exception as e:
        logger.warning(f"{prefix}Could not read R/B analog gains: {e}")

    # White balance mode
    try:
        wb = props.get_white_balance_mode()
        logger.info(f"{prefix}WhiteBalance mode: {wb}")
    except Exception as e:
        logger.warning(f"{prefix}Could not read WB mode: {e}")

    # Frame rate
    try:
        fr = float(props._get_property(props.FRAME_RATE))
        logger.info(f"{prefix}Frame rate: {fr:.3f} Hz")
    except Exception as e:
        logger.warning(f"{prefix}Could not read frame rate: {e}")

    logger.info(f"{prefix}--- End Snapshot ---")
    return


def read_gains_compact(props):
    """Return a compact dict of gain state for comparison."""
    result = {}
    try:
        result["unified_gain"] = props.get_unified_gain()
    except Exception:
        result["unified_gain"] = None
    try:
        rb = props.get_rb_analog_gains()
        result["analog_red"] = rb["red"]
        result["analog_blue"] = rb["blue"]
    except Exception:
        result["analog_red"] = None
        result["analog_blue"] = None
    try:
        result["gain_individual"] = props.is_individual_gain_enabled()
    except Exception:
        result["gain_individual"] = None
    return result


# ===========================================================================
# TEST 1: Camera AWB Gain Persistence
# ===========================================================================

def apply_awb(props, dwell=5.0):
    """Apply Continuous AWB for dwell seconds, then set to Off."""
    props._set_property(props.WHITE_BALANCE, "Continuous")
    time.sleep(dwell)
    props._set_property(props.WHITE_BALANCE, "Off")
    time.sleep(0.5)
    # Verify mode actually changed
    actual = props.get_white_balance_mode()
    if actual != "Off":
        logger.warning(f"WB mode after Off write: '{actual}' (expected 'Off')")
    return actual


def test_issue_1_awb_persistence(props, hardware):
    """
    Issue 1: Do camera AWB internal gains persist after explicit reset?

    Uses Continuous mode (the real Camera AWB workflow) with image-based
    verification. Snaps images to measure RGB channel means before/after
    each reset strategy to determine which call actually clears AWB.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Camera AWB Persistence & Reset Strategies (Issue #1)")
    print("=" * 70)
    print("Uses Continuous mode + image snaps to detect AWB state.")
    print()

    AWB_DWELL = 5.0  # seconds for Continuous AWB to converge

    # Step 1: Set frame rate high so AWB has frames to work with
    logger.info("Step 1: Setting frame rate to 38 Hz for AWB convergence...")
    props._set_property(props.FRAME_RATE, props.FRAME_RATE_MAX)
    time.sleep(SETTLE_TIME)

    # Step 2: Reset to known neutral state and snap baseline image
    logger.info("Step 2: Resetting to neutral state...")
    props.reset_to_defaults()
    props.set_unified_gain(1.0)
    props.set_rb_analog_gains(red=1.0, blue=1.0)
    props._set_property(props.WHITE_BALANCE, "Off")
    time.sleep(1.0)

    baseline_rgb = snap_rgb_means(hardware)
    log_rgb("BASELINE (no AWB)", baseline_rgb)
    dump_state(props, "BASELINE")

    # Step 3: Apply Continuous AWB and snap image to confirm it worked
    logger.info(f"\nStep 3: Applying Continuous AWB for {AWB_DWELL}s...")
    props._set_property(props.WHITE_BALANCE, "Continuous")
    time.sleep(AWB_DWELL)

    awb_active_rgb = snap_rgb_means(hardware)
    log_rgb("AWB ACTIVE (Continuous)", awb_active_rgb)

    wb_mode_during = props.get_white_balance_mode()
    logger.info(f"WB mode while active: {wb_mode_during}")

    # Step 4: Set WB to Off and check if correction persists
    logger.info("\nStep 4: Writing WB='Off'...")
    props._set_property(props.WHITE_BALANCE, "Off")
    time.sleep(1.0)

    wb_mode_after_off = props.get_white_balance_mode()
    logger.info(f"WB mode after Off write: {wb_mode_after_off}")

    awb_off_rgb = snap_rgb_means(hardware)
    log_rgb("AFTER WB->Off", awb_off_rgb)

    # Check if AWB actually changed anything
    baseline_r, baseline_g, baseline_b = baseline_rgb
    awb_r, awb_g, awb_b = awb_active_rgb
    channel_shift = abs(awb_r - baseline_r) + abs(awb_g - baseline_g) + abs(awb_b - baseline_b)

    if channel_shift < 5.0:
        print("\n  WARNING: AWB did not significantly change image appearance.")
        print(f"  Total channel shift: {channel_shift:.1f} (threshold: 5.0)")
        print("  The camera may need a more strongly colored scene.")
        print("  Continuing with reset tests anyway...\n")

    # ===================================================================
    # Step 5: Try each reset strategy with image verification
    # ===================================================================
    # For each strategy: re-apply AWB, then try the reset, then snap image.
    # Compare snapped image to baseline to determine if AWB was cleared.

    reset_strategies = [
        {
            "name": "A: set_white_balance_mode('Off') only",
            "desc": "Current approach -- write Off to WB property",
            "fn": lambda: props._set_property(props.WHITE_BALANCE, "Off"),
        },
        {
            "name": "B: Preset6500K then Off",
            "desc": "Transition through a preset to break out of Continuous",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Preset6500K"),
                time.sleep(0.5),
                props._set_property(props.WHITE_BALANCE, "Off"),
            ),
        },
        {
            "name": "C: Once then Off",
            "desc": "Transition through Once mode (auto-returns to Off)",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Once"),
                time.sleep(1.0),
                props._set_property(props.WHITE_BALANCE, "Off"),
            ),
        },
        {
            "name": "D: Off with 2s delay",
            "desc": "Write Off then wait longer for camera to process",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Off"),
                time.sleep(2.0),
            ),
        },
        {
            "name": "E: set_rb_analog_gains(1.0, 1.0) after Off",
            "desc": "Write Off then overwrite analog gains",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Off"),
                time.sleep(0.5),
                props.set_rb_analog_gains(red=1.0, blue=1.0),
            ),
        },
        {
            "name": "F: set_unified_gain(1.0) after Off",
            "desc": "Write Off then overwrite unified gain",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Off"),
                time.sleep(0.5),
                props.set_unified_gain(1.0),
            ),
        },
        {
            "name": "G: Toggle individual gain mode after Off",
            "desc": "Write Off, enable individual gain, disable it",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Off"),
                time.sleep(0.5),
                props.enable_individual_gain(),
                time.sleep(0.3),
                props.disable_individual_gain(),
            ),
        },
        {
            "name": "H: Full cleanup (Off + all resets)",
            "desc": "Write Off + analog gains + unified gain + mode toggles",
            "fn": lambda: (
                props._set_property(props.WHITE_BALANCE, "Off"),
                time.sleep(0.5),
                props.set_rb_analog_gains(red=1.0, blue=1.0),
                props.set_unified_gain(1.0),
                props.disable_individual_exposure(),
                props.disable_individual_gain(),
            ),
        },
    ]

    strategy_results = []
    for strategy in reset_strategies:
        name = strategy["name"]
        desc = strategy["desc"]
        fn = strategy["fn"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Strategy {name}")
        logger.info(f"  {desc}")

        # Re-apply AWB fresh
        logger.info(f"  Re-applying Continuous AWB for {AWB_DWELL}s...")
        props._set_property(props.WHITE_BALANCE, "Continuous")
        time.sleep(AWB_DWELL)

        # Snap to confirm AWB is active
        awb_rgb = snap_rgb_means(hardware)
        log_rgb(f"  {name} AWB-ACTIVE", awb_rgb)

        # Apply the reset strategy
        logger.info(f"  Applying reset strategy...")
        try:
            fn()
        except Exception as e:
            logger.warning(f"  Strategy raised: {e}")
        time.sleep(1.0)

        # Snap after reset
        post_rgb = snap_rgb_means(hardware)
        log_rgb(f"  {name} AFTER-RESET", post_rgb)

        # Check WB mode
        wb_after = props.get_white_balance_mode()
        logger.info(f"  WB mode after reset: {wb_after}")

        # Compare to baseline: how close is the post-reset image to baseline?
        post_r, post_g, post_b = post_rgb
        dist_to_baseline = (
            abs(post_r - baseline_r) + abs(post_g - baseline_g) + abs(post_b - baseline_b)
        )
        dist_to_awb = (
            abs(post_r - awb_r) + abs(post_g - awb_g) + abs(post_b - awb_b)
        )

        # AWB cleared if post-reset is closer to baseline than to AWB-active
        cleared = dist_to_baseline < dist_to_awb
        verdict = "CLEARED AWB" if cleared else "AWB still active"

        logger.info(
            f"  Distance to baseline: {dist_to_baseline:.1f}, "
            f"distance to AWB: {dist_to_awb:.1f} -> {verdict}"
        )

        strategy_results.append({
            "name": name,
            "awb_rgb": awb_rgb,
            "post_rgb": post_rgb,
            "wb_mode": wb_after,
            "dist_baseline": dist_to_baseline,
            "dist_awb": dist_to_awb,
            "cleared": cleared,
        })

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: AWB Reset Strategies")
    print("=" * 70)
    print(f"\n  Baseline RGB:  R={baseline_r:.1f}  G={baseline_g:.1f}  B={baseline_b:.1f}")
    print(f"  AWB active:    R={awb_r:.1f}  G={awb_g:.1f}  B={awb_b:.1f}")
    print()

    any_cleared = False
    print(f"  {'Strategy':<50s} {'WB Mode':<12s} {'Dist->Base':<12s} {'Dist->AWB':<12s} {'Result'}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
    for r in strategy_results:
        result_str = "CLEARED" if r["cleared"] else "still active"
        if r["cleared"]:
            any_cleared = True
        print(
            f"  {r['name']:<50s} {r['wb_mode']:<12s} "
            f"{r['dist_baseline']:<12.1f} {r['dist_awb']:<12.1f} {result_str}"
        )

    if any_cleared:
        cleared_names = [r["name"] for r in strategy_results if r["cleared"]]
        print(f"\n  FOUND WORKING RESET: {cleared_names[0]}")
        if len(cleared_names) > 1:
            print(f"  Also worked: {', '.join(cleared_names[1:])}")
    else:
        print("\n  NO STRATEGY CLEARED AWB.")
        print("  The camera may require MM restart to clear AWB corrections.")

    print()
    return


# ===========================================================================
# TEST 6: Gain State Persistence Across Angles
# ===========================================================================

def test_issue_6_gain_persistence_across_angles(props):
    """
    Issue 6: Do gains from one PPM angle leak into the next?

    Procedure:
      1. Reset to known state
      2. For each PPM angle:
         a. Set per-channel gains for this angle
         b. Read back immediately -- verify they match
         c. Sleep briefly (simulates capture time)
         d. Read back again -- verify still correct
      3. After all angles: read back again without setting -- are last angle's gains still there?
    """
    print("\n" + "=" * 70)
    print("TEST 6: Gain State Persistence Across PPM Angles (Issue #6)")
    print("=" * 70)
    print("Question: Are gains reliably set per-angle without cross-contamination?")
    print()

    # Step 1: Reset
    logger.info("Step 1: Resetting to neutral state...")
    props.reset_to_defaults()
    props.set_unified_gain(1.0)
    props.set_rb_analog_gains(red=1.0, blue=1.0)
    time.sleep(SETTLE_TIME)

    angle_names = list(PPM_ANGLES.keys())
    results = {}

    # Step 2: Cycle through angles
    for angle_name in angle_names:
        angle = PPM_ANGLES[angle_name]
        logger.info(f"\n--- Angle: {angle_name} ---")

        # Set gains
        logger.info(
            f"Setting: unified={angle['unified_gain']}, "
            f"analog_r={angle['analog_red']}, analog_b={angle['analog_blue']}"
        )
        props.set_unified_gain(angle["unified_gain"])
        props.set_rb_analog_gains(red=angle["analog_red"], blue=angle["analog_blue"])

        # Also set per-channel exposures to simulate real workflow
        props.enable_individual_exposure()
        props.set_channel_exposures(
            red=angle["exposure_r"],
            green=angle["exposure_g"],
            blue=angle["exposure_b"],
        )
        time.sleep(SETTLE_TIME)

        # Immediate read-back
        gains_immediate = read_gains_compact(props)
        logger.info(
            f"Read-back (immediate): unified={gains_immediate['unified_gain']}, "
            f"R={gains_immediate['analog_red']}, B={gains_immediate['analog_blue']}"
        )

        # Simulate capture delay
        time.sleep(0.5)

        # Delayed read-back
        gains_delayed = read_gains_compact(props)
        logger.info(
            f"Read-back (after 0.5s): unified={gains_delayed['unified_gain']}, "
            f"R={gains_delayed['analog_red']}, B={gains_delayed['analog_blue']}"
        )

        results[angle_name] = {
            "requested": {
                "unified_gain": angle["unified_gain"],
                "analog_red": angle["analog_red"],
                "analog_blue": angle["analog_blue"],
            },
            "immediate": gains_immediate,
            "delayed": gains_delayed,
        }

    # Step 3: Final read without setting (should still be last angle's values)
    time.sleep(0.5)
    final_gains = read_gains_compact(props)

    # Analysis
    print("\n--- ANALYSIS ---")
    all_match = True
    for angle_name in angle_names:
        r = results[angle_name]
        req = r["requested"]
        imm = r["immediate"]
        dly = r["delayed"]

        unified_ok = (
            imm["unified_gain"] == req["unified_gain"]
            and dly["unified_gain"] == req["unified_gain"]
        )
        red_ok = (
            imm["analog_red"] == req["analog_red"]
            and dly["analog_red"] == req["analog_red"]
        )
        blue_ok = (
            imm["analog_blue"] == req["analog_blue"]
            and dly["analog_blue"] == req["analog_blue"]
        )

        status = "OK" if (unified_ok and red_ok and blue_ok) else "MISMATCH"
        if status == "MISMATCH":
            all_match = False

        print(f"  {angle_name:12s}: [{status}]")
        if status == "MISMATCH":
            print(f"    Requested: unified={req['unified_gain']}, R={req['analog_red']}, B={req['analog_blue']}")
            print(f"    Immediate: unified={imm['unified_gain']}, R={imm['analog_red']}, B={imm['analog_blue']}")
            print(f"    Delayed:   unified={dly['unified_gain']}, R={dly['analog_red']}, B={dly['analog_blue']}")

    last_angle = angle_names[-1]
    last_req = results[last_angle]["requested"]
    persistence_ok = (
        final_gains["unified_gain"] == last_req["unified_gain"]
        and final_gains["analog_red"] == last_req["analog_red"]
        and final_gains["analog_blue"] == last_req["analog_blue"]
    )
    print(f"\n  Final read (no set): unified={final_gains['unified_gain']}, "
          f"R={final_gains['analog_red']}, B={final_gains['analog_blue']}")
    print(f"  Matches last angle ({last_angle}): {'Yes' if persistence_ok else 'No'}")

    if all_match:
        print("\n  PASS: All angles received correct gains. No cross-contamination detected.")
    else:
        print("\n  FAIL: Gain read-back did not match requested values for some angles.")
        print("  Check for timing issues or hardware register write delays.")

    # Cleanup
    props.reset_to_defaults()
    props.set_unified_gain(1.0)
    props.set_rb_analog_gains(red=1.0, blue=1.0)

    print()
    return


# ===========================================================================
# TEST 7: Frame Rate / Exposure Coupling
# ===========================================================================

def test_issue_7_framerate_exposure_coupling(props):
    """
    Issue 7: Does the camera correctly handle long exposures via frame rate?

    Procedure:
      1. Reset to default frame rate (max ~38 Hz)
      2. Read current frame rate
      3. Set a long per-channel exposure (500ms) via set_channel_exposures
         (which should auto-call _adjust_frame_rate_for_exposure)
      4. Read frame rate -- should be much lower
      5. Read back exposure -- verify it actually took effect
      6. Try an even longer exposure (2000ms) -- verify
      7. Test: what happens if we set frame rate high THEN set long exposure
         manually without going through set_channel_exposures?
    """
    print("\n" + "=" * 70)
    print("TEST 7: Frame Rate / Exposure Coupling (Issue #7)")
    print("=" * 70)
    print("Question: Does _adjust_frame_rate_for_exposure() work correctly?")
    print()

    # Step 1: Reset
    logger.info("Step 1: Resetting to defaults...")
    props.reset_to_defaults()
    props.set_unified_gain(1.0)
    time.sleep(SETTLE_TIME)

    # Step 2: Read default frame rate
    default_fr = float(props._get_property(props.FRAME_RATE))
    logger.info(f"Step 2: Default frame rate: {default_fr:.3f} Hz")

    # Step 3: Set long exposure via API (triggers frame rate adjustment)
    logger.info(f"Step 3: Setting per-channel exposure to {LONG_EXPOSURE_MS}ms via API...")
    props.enable_individual_exposure()
    props.set_channel_exposures(
        red=LONG_EXPOSURE_MS, green=LONG_EXPOSURE_MS, blue=LONG_EXPOSURE_MS
    )
    time.sleep(SETTLE_TIME)

    # Step 4: Read adjusted frame rate
    adjusted_fr = float(props._get_property(props.FRAME_RATE))
    ch_exp = props.get_channel_exposures()
    logger.info(f"Step 4: Adjusted frame rate: {adjusted_fr:.3f} Hz")
    logger.info(
        f"  Read-back exposures: R={ch_exp['red']:.3f}ms  "
        f"G={ch_exp['green']:.3f}ms  B={ch_exp['blue']:.3f}ms"
    )

    # Step 5: Even longer exposure
    logger.info(f"Step 5: Setting per-channel exposure to {VERY_LONG_EXPOSURE_MS}ms...")
    props.set_channel_exposures(
        red=VERY_LONG_EXPOSURE_MS, green=VERY_LONG_EXPOSURE_MS, blue=VERY_LONG_EXPOSURE_MS
    )
    time.sleep(SETTLE_TIME)
    very_long_fr = float(props._get_property(props.FRAME_RATE))
    ch_exp_long = props.get_channel_exposures()
    logger.info(f"  Frame rate after {VERY_LONG_EXPOSURE_MS}ms: {very_long_fr:.3f} Hz")
    logger.info(
        f"  Read-back exposures: R={ch_exp_long['red']:.3f}ms  "
        f"G={ch_exp_long['green']:.3f}ms  B={ch_exp_long['blue']:.3f}ms"
    )

    # Step 6: Test bypass -- set frame rate high, then set long exposure directly
    # This simulates what happens if someone forgets to call _adjust_frame_rate
    logger.info("Step 6: BYPASS TEST - Set high frame rate, then write long exposure directly...")
    props._set_property(props.FRAME_RATE, props.FRAME_RATE_MAX)
    time.sleep(SETTLE_TIME)
    bypass_fr_before = float(props._get_property(props.FRAME_RATE))
    logger.info(f"  Frame rate forced to: {bypass_fr_before:.3f} Hz")

    # Write exposure directly to the property (bypassing _adjust_frame_rate)
    try:
        props._set_property(props.EXPOSURE_RED, LONG_EXPOSURE_MS)
        props._set_property(props.EXPOSURE_GREEN, LONG_EXPOSURE_MS)
        props._set_property(props.EXPOSURE_BLUE, LONG_EXPOSURE_MS)
        time.sleep(SETTLE_TIME)
        ch_exp_bypass = props.get_channel_exposures()
        bypass_fr_after = float(props._get_property(props.FRAME_RATE))
        logger.info(
            f"  After direct property write: R={ch_exp_bypass['red']:.3f}ms  "
            f"G={ch_exp_bypass['green']:.3f}ms  B={ch_exp_bypass['blue']:.3f}ms"
        )
        logger.info(f"  Frame rate: {bypass_fr_after:.3f} Hz")
        bypass_clipped = (
            ch_exp_bypass["red"] < LONG_EXPOSURE_MS * 0.9
            or ch_exp_bypass["green"] < LONG_EXPOSURE_MS * 0.9
            or ch_exp_bypass["blue"] < LONG_EXPOSURE_MS * 0.9
        )
    except Exception as e:
        logger.info(f"  Direct property write raised error: {e}")
        bypass_clipped = True
        ch_exp_bypass = {"red": 0, "green": 0, "blue": 0}

    # Analysis
    print("\n--- ANALYSIS ---")

    # Check: did frame rate actually drop?
    fr_dropped = adjusted_fr < default_fr * 0.5
    print(f"  Default frame rate:   {default_fr:.3f} Hz")
    print(f"  After {LONG_EXPOSURE_MS}ms exp:    {adjusted_fr:.3f} Hz  "
          f"{'(dropped -- OK)' if fr_dropped else '(DID NOT DROP -- check logic)'}")

    # Check: did exposure take effect?
    exp_applied = all(
        abs(ch_exp[ch] - LONG_EXPOSURE_MS) < LONG_EXPOSURE_MS * 0.1
        for ch in ["red", "green", "blue"]
    )
    print(f"  Exposure {LONG_EXPOSURE_MS}ms applied: "
          f"{'Yes' if exp_applied else 'No (clipped or rejected)'}")

    # Check: very long exposure
    very_long_fr_ok = very_long_fr < 1.0
    very_long_exp_ok = all(
        abs(ch_exp_long[ch] - VERY_LONG_EXPOSURE_MS) < VERY_LONG_EXPOSURE_MS * 0.1
        for ch in ["red", "green", "blue"]
    )
    print(f"  After {VERY_LONG_EXPOSURE_MS}ms exp: FR={very_long_fr:.3f} Hz  "
          f"{'(OK)' if very_long_fr_ok else '(frame rate too high!)'}")
    print(f"  Exposure {VERY_LONG_EXPOSURE_MS}ms applied: "
          f"{'Yes' if very_long_exp_ok else 'No (clipped or rejected)'}")

    # Check: bypass behavior
    print(f"\n  BYPASS TEST (direct property write at {props.FRAME_RATE_MAX} Hz):")
    if bypass_clipped:
        print(f"  Camera SILENTLY CLIPPED exposure when frame rate was too high.")
        print(f"  Read-back: R={ch_exp_bypass['red']:.3f}ms  "
              f"G={ch_exp_bypass['green']:.3f}ms  B={ch_exp_bypass['blue']:.3f}ms")
        print("  This confirms _adjust_frame_rate_for_exposure() is ESSENTIAL.")
    else:
        print(f"  Camera accepted long exposure even at high frame rate.")
        print("  The camera may auto-adjust frame rate internally.")

    # Overall verdict
    if fr_dropped and exp_applied and very_long_exp_ok:
        print("\n  PASS: Frame rate auto-adjustment works correctly for long exposures.")
    else:
        print("\n  FAIL: Frame rate / exposure coupling has issues. See details above.")

    # Cleanup
    props.reset_to_defaults()
    props.set_unified_gain(1.0)
    props._set_property(props.FRAME_RATE, props.FRAME_RATE_MAX)

    print()
    return


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WB Hardware Diagnostic -- tests Issues 1, 6, 7"
    )
    parser.add_argument(
        "--test", choices=["1", "6", "7", "all"], default="all",
        help="Which test to run (default: all)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging"
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    print("=" * 70)
    print("  White Balance Hardware Diagnostic")
    print("  Tests Issues 1 (AWB persistence), 6 (gain leak), 7 (FR coupling)")
    print("  TEMPORARY SCRIPT -- see WB_HARDWARE_TEST_CLEANUP.md for removal")
    print("=" * 70)
    print()

    props, hardware = connect_camera()

    # Initial state dump
    dump_state(props, "INITIAL")

    tests_to_run = args.test
    try:
        if tests_to_run in ("1", "all"):
            test_issue_1_awb_persistence(props, hardware)

        if tests_to_run in ("6", "all"):
            test_issue_6_gain_persistence_across_angles(props)

        if tests_to_run in ("7", "all"):
            test_issue_7_framerate_exposure_coupling(props)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        # Always try to reset camera to safe state
        try:
            props.reset_to_defaults()
            props.set_unified_gain(1.0)
            props.set_rb_analog_gains(red=1.0, blue=1.0)
            props._set_property(props.FRAME_RATE, props.FRAME_RATE_MAX)
            logger.info("Camera reset to safe defaults.")
        except Exception:
            logger.warning("Could not reset camera to defaults.")

    print("\nDone. Copy the output above into a test session in WB_TEST_LOG.md.")


if __name__ == "__main__":
    main()
