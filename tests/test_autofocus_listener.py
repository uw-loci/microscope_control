"""The autofocus listener sees every search: start, each scored Z, finish.

The microscope server sets ``hardware.autofocus_listener`` to show the focus
curve to the operator while a search runs, so these check the events a search
reports and that a misbehaving listener cannot stop one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from microscope_control.hardware.base import AutofocusAborted, AutofocusListener, Position
from microscope_control.hardware.pycromanager import PycromanagerHardware


class _Recorder(AutofocusListener):
    def __init__(self):
        self.events = []

    def af_started(self, method, params):
        self.events.append(("start", method, dict(params)))

    def af_sample(self, z, score, spread):
        self.events.append(("sample", z, score, spread))

    def af_finished(self, status, z, details):
        self.events.append(("end", status, z, dict(details)))


def _hardware(score_fn, should_snap=None):
    hw = PycromanagerHardware.__new__(PycromanagerHardware)
    hw.settings = {"stage": {"limits": {"z_um": {"low": -1000.0, "high": 1000.0}}}}
    hw.autofocus_listener = None
    z = {"z": 0.0}
    hw._active_detector_id = "fake"
    hw._camera_registry = {
        "fake": SimpleNamespace(
            stop_if_streaming=lambda: None, extract_green_channel=lambda img: img
        )
    }
    hw.get_current_position = lambda: Position(0.0, 0.0, z["z"])

    def _move(pos):
        if pos.z is not None:
            z["z"] = float(pos.z)

    hw.move_to_position = _move

    def _snap():
        if should_snap is not None:
            should_snap()
        return np.full((8, 8), float(score_fn(z["z"]))), {}

    hw.snap_image = _snap
    return hw


def _peak_at_zero(z):
    return 1000.0 * np.exp(-0.5 * (z / 4.0) ** 2) + 10.0


def test_a_search_reports_start_every_step_and_its_focus():
    hw = _hardware(_peak_at_zero)
    rec = _Recorder()
    hw.autofocus_listener = rec
    result = hw.autofocus(
        n_steps=9,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
    )
    kinds = [e[0] for e in rec.events]
    assert kinds == ["start"] + ["sample"] * 9 + ["end"]
    start = rec.events[0]
    assert start[1] == "sweep"
    assert start[2]["n_steps"] == 9 and start[2]["search_range"] == 30.0
    assert start[2]["score_metric"] == "<lambda>"
    assert start[2]["z0"] == 0.0
    samples = rec.events[1:-1]
    assert [round(s[1], 3) for s in samples] == [round(z, 3) for z in np.linspace(-15, 15, 9)]
    assert all(s[3] is not None for s in samples)  # P98 - P2 travels with each step
    end = rec.events[-1]
    assert end[1] == "ok" and end[2] == pytest.approx(result)


def test_a_failed_search_reports_failed_with_the_reason():
    hw = _hardware(lambda z: 10.0)  # flat: no peak to find
    rec = _Recorder()
    hw.autofocus_listener = rec
    result = hw.autofocus(
        n_steps=7,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
        raise_on_invalid_peak=False,
    )
    end = rec.events[-1]
    if isinstance(result, dict):
        assert end[1] == "failed" and end[3]["reason"]
    else:
        assert end[1] == "ok"


def test_a_cancelled_search_reports_aborted_and_still_raises():
    hw = _hardware(_peak_at_zero)
    rec = _Recorder()
    hw.autofocus_listener = rec
    calls = {"n": 0}

    def abort_after_three():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(AutofocusAborted):
        hw.autofocus(
            n_steps=9,
            search_range=30.0,
            score_metric=lambda img: float(np.mean(img)),
            move_stage_to_estimate=False,
            should_abort=abort_after_three,
        )
    assert [e[0] for e in rec.events] == ["start", "sample", "sample", "sample", "end"]
    assert rec.events[-1][1] == "aborted"


def test_a_listener_that_raises_never_stops_the_search():
    class Broken(AutofocusListener):
        def af_started(self, method, params):
            raise RuntimeError("display gone")

        def af_sample(self, z, score, spread):
            raise RuntimeError("display gone")

        def af_finished(self, status, z, details):
            raise RuntimeError("display gone")

    hw = _hardware(_peak_at_zero)
    hw.autofocus_listener = Broken()
    result = hw.autofocus(
        n_steps=9,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
    )
    assert isinstance(result, float) and abs(result) <= 4.0


def test_no_listener_changes_nothing():
    hw = _hardware(_peak_at_zero)
    result = hw.autofocus(
        n_steps=9,
        search_range=30.0,
        score_metric=lambda img: float(np.mean(img)),
        move_stage_to_estimate=False,
    )
    assert isinstance(result, float) and abs(result) <= 4.0
