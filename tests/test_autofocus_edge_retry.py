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


# ---------------------------------------------------------------------------
# The time ceiling is on the projected finish, not on time already spent.
# ---------------------------------------------------------------------------


def _slow_stage(monkeypatch, hw, snapped, seconds_per_snap):
    """Advance a fake clock by ``seconds_per_snap`` on every snap."""
    import microscope_control.hardware.pycromanager as pm

    clock = {"t": 0.0}
    monkeypatch.setattr(pm.time, "perf_counter", lambda: clock["t"])
    inner = hw.snap_image

    def _snap():
        clock["t"] += seconds_per_snap
        return inner()

    hw.snap_image = _snap


def test_slow_first_sweep_still_gets_its_edge_retry(monkeypatch):
    # OWS3 2026-10-27 in proportion: a 132 s first sweep (21 samples standing in for
    # 100 at 1.32 s). The old rule (no retry after 90 s elapsed) refused the retry,
    # so focus outside the first window was never searched for.
    def score_fn(z):
        return 1000.0 * np.exp(-0.5 * ((z - 40.0) / 8.0) ** 2) + 10.0

    hw, snapped = _make_dense_hardware(score_fn)
    _slow_stage(monkeypatch, hw, snapped, 132.0 / 21)
    result = hw.autofocus(
        n_steps=21,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
        edge_retries=2,
    )
    assert len(snapped) > 21, "the retry never ran"
    assert isinstance(result, float)
    assert 30.0 <= result <= 50.0


def test_edge_retry_that_would_finish_past_the_ceiling_is_skipped(monkeypatch):
    # 60 s per sample: the first sweep alone is 420 s, so any retry projects past
    # the 270 s ceiling and must not start.
    def score_fn(z):
        return 1000.0 * np.exp(-0.5 * ((z - 40.0) / 8.0) ** 2) + 10.0

    hw, snapped = _make_dense_hardware(score_fn)
    _slow_stage(monkeypatch, hw, snapped, 60.0)
    hw.autofocus(
        n_steps=7,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
        edge_retries=2,
    )
    assert len(snapped) == 7
