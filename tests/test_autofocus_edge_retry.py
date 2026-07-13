"""Tests for the AF edge-retry window placement and sample retention.

Background (claude-reports/TODO_LIST.md, added 2026-07-13): the peak-at-edge
retry used to CENTER the widened retry window on the previous window's edge,
so ~half of every retry re-imaged Z the previous sweep had already covered.
Two changes under test here:

1. ``_compute_edge_retry_window`` now places the covered band's edge ~10%
   into the new window (90% new ground) and takes the covered band's bounds
   (all attempts, not just the last window) for its extend-direction guard.
2. ``autofocus`` retains every (z, score) sample across edge retries and
   images ONLY targets outside the covered band, running peak detection on
   the merged set -- no re-imaging at all.
"""

from types import SimpleNamespace

import numpy as np

from microscope_control.hardware.base import Position
from microscope_control.hardware.pycromanager import PycromanagerHardware

WIDE_LIMITS = {"low": -1000.0, "high": 1000.0}


def _bare_hardware(z_limits=WIDE_LIMITS):
    hw = PycromanagerHardware.__new__(PycromanagerHardware)
    hw.settings = {"stage": {"limits": {"z_um": dict(z_limits)}}}
    return hw


def _window(validation, hw=None, **kwargs):
    hw = hw or _bare_hardware()
    defaults = {
        "cur_center": 0.0,
        "cur_range": 30.0,
        "widen_factor": 2.0,
        "z_min": WIDE_LIMITS["low"],
        "z_max": WIDE_LIMITS["high"],
        "covered_low": -15.0,
        "covered_high": 15.0,
    }
    defaults.update(kwargs)
    return hw._compute_edge_retry_window(validation=validation, **defaults)


ASC_ONLY = {"has_ascending": True, "has_descending": False}
DESC_ONLY = {"has_ascending": False, "has_descending": True}


def test_retry_window_puts_old_edge_ten_percent_in_extend_high():
    center, rng = _window(ASC_ONLY)
    assert rng == 60.0
    # Window [9, 69]: old high edge (15) sits 6 um = 10% of 60 into it.
    assert abs(center - 39.0) < 1e-6
    overlap = 15.0 - (center - rng / 2.0)
    assert abs(overlap - 0.1 * rng) < 1e-6


def test_retry_window_puts_old_edge_ten_percent_in_extend_low():
    center, rng = _window(DESC_ONLY)
    assert rng == 60.0
    assert abs(center - (-39.0)) < 1e-6
    overlap = (center + rng / 2.0) - (-15.0)
    assert abs(overlap - 0.1 * rng) < 1e-6


def test_retry_window_guard_uses_covered_band_not_last_window():
    # Covered band already reaches the usable Z ceiling (z_max - margin =
    # 15), so an extend-high retry cannot add new ground even though a
    # window computed from the last attempt alone would appear to.
    center, rng = _window(ASC_ONLY, z_max=20.0, covered_high=15.0)
    assert center is None and rng is None


def test_retry_window_refuses_non_directional_trend():
    both = {"has_ascending": True, "has_descending": True}
    assert _window(both) == (None, None)
    neither = {"has_ascending": False, "has_descending": False}
    assert _window(neither) == (None, None)


# ---------------------------------------------------------------------------
# Sample retention across edge retries in the dense autofocus() path.
# ---------------------------------------------------------------------------


def _make_dense_hardware(score_fn, start_z=0.0):
    """PycromanagerHardware whose stage/camera are faked so that the dense
    ``autofocus`` path scores frames via ``score_fn(z)``. Returns the
    hardware and the list of Z positions actually imaged, in order."""
    hw = _bare_hardware()
    z_holder = {"z": float(start_z)}
    snapped = []

    hw._active_detector_id = "fake"
    hw._camera_registry = {
        "fake": SimpleNamespace(
            stop_if_streaming=lambda: None,
            extract_green_channel=lambda img: img,
        )
    }
    hw.get_current_position = lambda: Position(0.0, 0.0, z_holder["z"])

    def _move(pos):
        if pos.z is not None:
            z_holder["z"] = float(pos.z)

    hw.move_to_position = _move

    def _snap():
        z = z_holder["z"]
        snapped.append(z)
        return np.full((8, 8), float(score_fn(z))), {}

    hw.snap_image = _snap
    return hw, snapped


def test_autofocus_edge_retry_images_only_new_ground():
    # True focus at Z=40, first window [-15, 15]: a one-sided ascending
    # ramp toward the high edge. The retry must reuse the 7 samples
    # already taken and image only targets above the covered band.
    def score_fn(z):
        return 1000.0 * np.exp(-0.5 * ((z - 40.0) / 8.0) ** 2) + 10.0

    hw, snapped = _make_dense_hardware(score_fn)
    result = hw.autofocus(
        n_steps=7,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
        edge_retries=2,
    )

    assert isinstance(result, float)
    # Attempt 1: 7 snaps over [-15, 15]. Retry window [9, 69] yields 7
    # targets of which [9, 19] fall inside covered-band tolerance -> only
    # 5 new snaps. (Pre-change behavior: 14 snaps, half re-imaged.)
    assert len(snapped) == 12
    first, second = snapped[:7], snapped[7:]
    assert max(first) <= 15.0 + 1e-6
    covered_high = max(first)
    assert all(z > covered_high for z in second), (
        "edge retry re-imaged Z inside the already-covered band: %s" % second
    )
    # Every attempt approaches its samples from below (ascending order),
    # keeping the backlash direction consistent.
    assert second == sorted(second)
    # Merged-set peak detection brackets the true focus at Z=40.
    assert 30.0 <= result <= 50.0


def test_autofocus_without_retry_is_unchanged():
    # Interior peak: no retry, one window, n_steps snaps -- the retention
    # plumbing must not alter the plain single-sweep behavior.
    def score_fn(z):
        return 1000.0 * np.exp(-0.5 * (z / 4.0) ** 2) + 10.0

    hw, snapped = _make_dense_hardware(score_fn)
    result = hw.autofocus(
        n_steps=9,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
        edge_retries=2,
    )
    assert len(snapped) == 9
    assert abs(result) <= 4.0
