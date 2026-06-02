"""Regression tests for the autofocus focus-runaway containment guards.

Background: on 2026-05-31 a PPM 40x acquisition lost focus mid-run and never
recovered. A contrast-inverted (saturated red channel) focus metric produced a
score curve that ramped toward one edge. The sweep drift-check's edge-retry
chased that ramp: the first window peaked at its high edge, the retry extended
the window upward, the inverted curve peaked in the *interior* of the extended
window, and the check committed a +26.64 um "correction". That bad Z propagated
forward as the hint for every subsequent tile, walking the stage from ~7 um to
104 um.

These tests lock in the containment guards added in
``claude-reports/2026-06-02_autofocus-focus-runaway.md``:

* ``autofocus_sweep_drift_check`` must not commit a correction larger than its
  drift cap (one search window by default) -- it holds at the starting Z and
  leaves real focus loss to the standard AF / manual path.
* ``AutofocusUtils.validate_focus_peak`` must flag a one-sided ramp via
  ``should_extend_direction`` -- the signal the standard-AF refusal path keys on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.hardware.pycromanager import PycromanagerHardware


class _FakeCore:
    """Minimal Micro-Manager core stand-in for the sweep drift check.

    Tracks the commanded Z, latches the Z at ``snap_image`` time (so the
    pipelined "snap at i, move to i+1, score frame i" loop scores the right
    position), and serves trivial geometry queries.
    """

    def __init__(self, start_z):
        self.z = float(start_z)
        self.snapped_z = float(start_z)

    # --- focus device / geometry ---
    def get_focus_device(self):
        return "Z"

    def set_focus_device(self, dev):
        pass

    def get_image_width(self):
        return 100

    def get_image_height(self):
        return 100

    def get_number_of_components(self):
        return 1

    def get_exposure(self):
        return 5.0

    # --- motion ---
    def set_position(self, z):
        self.z = float(z)

    def get_position(self):
        return self.z

    def wait_for_device(self, dev):
        pass

    # --- capture ---
    def snap_image(self):
        self.snapped_z = self.z

    def get_tagged_image(self):
        return SimpleNamespace(pix=None)


def _make_hardware(start_z, score_fn):
    """Build a PycromanagerHardware that scores frames via ``score_fn(z)``.

    ``score_fn`` maps the latched snap Z to a focus score, letting a test
    impose an arbitrary (e.g. contrast-inverted) focus curve.
    """
    hw = PycromanagerHardware.__new__(PycromanagerHardware)
    core = _FakeCore(start_z)
    hw.core = core
    hw.settings = {
        "stage": {
            "z_stage": "Z",
            "limits": {"z_um": {"low": -100.0, "high": 1000.0}},
        }
    }
    # _stage is only used for an optional lock; None disables locking.
    hw._stage = SimpleNamespace(lock=None, get_z=lambda: core.z)
    # camera.stop_if_streaming() is the only camera call in the sweep.
    hw._active_detector_id = "fake"
    hw._camera_registry = {"fake": SimpleNamespace(stop_if_streaming=lambda: None)}
    # Score the latched snap Z, bypassing real pixel scoring.
    hw._score_single_metric = lambda *a, **k: (float(score_fn(core.snapped_z)), 0.0)
    return hw, core


def test_drift_check_refuses_runaway_interior_peak():
    """The exact 2026-05-31 failure: a contrast-inverted curve peaking far
    from the start (Z=34) reached via edge-retry must NOT be committed.

    Starting near Z=7.5 with a 20 um window, the first window [-2.5, 17.5]
    ramps toward its high edge; the edge-retry extends to [17.5, 37.5] where
    the parabola peaks interior at Z=34. Without the drift cap this commits
    ~+26 um. With the cap it holds at the starting Z.
    """
    start_z = 7.5
    # Parabola peaking at Z=34 -- monotonic-rising across the first window,
    # interior peak in the first extended window (defeats the boundary-hold
    # and U-shape guards, exactly as the saturated metric did in the field).
    hw, core = _make_hardware(start_z, lambda z: -((z - 34.0) ** 2))

    result = hw.autofocus_sweep_drift_check(
        range_um=20.0, n_steps=5, score_metric="normalized_variance", max_retries=2
    )

    assert result == pytest.approx(start_z), (
        f"drift check committed a runaway correction to Z={result} "
        f"(expected hold at start Z={start_z})"
    )
    assert core.z == pytest.approx(start_z), "stage was left off the starting Z"


def test_drift_check_commits_small_real_drift():
    """A genuine small drift (peak 2 um from start, bracketed inside the first
    window) is still corrected -- the cap must not break normal operation."""
    start_z = 7.5
    peak_z = 9.5  # +2 um, sampled exactly by the [-2.5..17.5] step grid
    hw, core = _make_hardware(start_z, lambda z: -((z - peak_z) ** 2))

    result = hw.autofocus_sweep_drift_check(
        range_um=20.0, n_steps=5, score_metric="normalized_variance", max_retries=2
    )

    assert result == pytest.approx(peak_z, abs=0.5), (
        f"drift check failed to commit a legitimate small correction "
        f"(got Z={result}, expected ~{peak_z})"
    )


def test_drift_check_custom_cap_is_respected():
    """A tighter explicit cap rejects a correction the default would allow."""
    start_z = 7.5
    peak_z = 12.5  # +5 um: within the default (20) cap but outside a 3 um cap
    hw, core = _make_hardware(start_z, lambda z: -((z - peak_z) ** 2))

    result = hw.autofocus_sweep_drift_check(
        range_um=20.0,
        n_steps=5,
        score_metric="normalized_variance",
        max_retries=2,
        max_total_drift_um=3.0,
    )

    assert result == pytest.approx(
        start_z
    ), f"tight drift cap was not honored (committed Z={result})"


def test_validate_focus_peak_flags_one_sided_ramp():
    """A monotonic ramp toward high Z must set should_extend_direction='high'.

    This is the signal the standard-AF refusal path keys on: when edge retries
    are exhausted and this is still set, the metric never bracketed focus and
    the boundary peak must be refused rather than committed.
    """
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # rising to high edge

    result = AutofocusUtils.validate_focus_peak(z, scores)

    assert result["peak_at_edge"] is True
    assert result["should_extend_direction"] == "high"
    assert result["has_descending"] is False


def test_validate_focus_peak_brackets_clean_interior_peak():
    """A clean interior peak must NOT request extension (bracketed focus)."""
    # 7 points so both flanks have the >=3 samples the validator needs to
    # confirm a trend; peak centered at idx 3.
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    scores = np.array([10.0, 25.0, 45.0, 60.0, 44.0, 24.0, 9.0])  # peak at idx 3

    result = AutofocusUtils.validate_focus_peak(z, scores)

    assert result["is_valid"] is True
    assert result["should_extend_direction"] is None
