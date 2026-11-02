"""The slide-change focus of 2026-11-01, on the LC-PolScope.

The sweep started at Z=53.4, focus was at Z=33.1, and the drift check's 12 um cap held it
at 53.4. That was reported as a success with a shift of 0.00, so the slide stayed out of
focus with Ready on screen. A drift check is right to refuse that move; what was wrong is
that nothing else searched, and that the refusal read as a measurement.
"""

from types import SimpleNamespace

import numpy as np

from microscope_control.autofocus import test as af_test
from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.hardware.base import Position
from tests.test_autofocus_drift_runaway import _make_hardware
from tests.test_autofocus_edge_retry import _make_dense_hardware

START_Z = 53.4
FOCUS_Z = 33.1


def focus_curve(z, focus=FOCUS_Z):
    """Sharp peak on a long tail: the far side still rises toward focus, as on the rig."""
    return 1000.0 / (1.0 + ((z - focus) / 4.0) ** 2) + 50.0


def _settings(**overrides):
    settings = {
        "n_steps": 30,
        "search_range": 30.0,
        "interp_strength": 100,
        "interp_kind": "quadratic",
        "score_metric": lambda img: float(np.mean(img)),
        "score_metric_name": "brenner_gradient",
        "edge_retries": 2,
    }
    settings.update(overrides)
    return settings


def _search(monkeypatch, tmp_path, score_fn, should_abort=None, **overrides):
    hw, snapped = _make_dense_hardware(score_fn)
    hw.get_current_position()  # the fake starts at Z=0
    hw.move_to_position(Position(0.0, 0.0, START_Z))
    monkeypatch.setattr(af_test, "_load_autofocus_settings", lambda *a: _settings(**overrides))
    result = af_test.focus_search_at_current_position(
        hw, "config_x.yml", str(tmp_path), "OBJ", should_abort=should_abort
    )
    return result, hw, snapped


def test_drift_check_says_when_it_held():
    hw, core = _make_hardware(START_Z, focus_curve)
    outcome = {}
    z = hw.autofocus_sweep_drift_check(
        range_um=12.0,
        n_steps=12,
        score_metric="brenner_gradient",
        outcome_out=outcome,
    )
    assert z == START_Z
    assert core.z == START_Z
    assert "still rising" in outcome["held"]


def test_drift_check_says_nothing_when_it_found_focus():
    hw, core = _make_hardware(START_Z, lambda z: focus_curve(z, focus=START_Z + 2.0))
    outcome = {}
    z = hw.autofocus_sweep_drift_check(
        range_um=12.0,
        n_steps=12,
        score_metric="brenner_gradient",
        outcome_out=outcome,
    )
    assert abs(z - (START_Z + 2.0)) < 0.5
    assert outcome == {}


def test_search_finds_focus_a_drift_check_refused(monkeypatch, tmp_path):
    result, hw, _ = _search(monkeypatch, tmp_path, focus_curve)
    assert result["success"]
    assert abs(result["final_z"] - FOCUS_Z) < 2.0
    assert abs(hw.get_current_position().z - result["final_z"]) < 1e-6


def test_search_reports_what_it_moved(monkeypatch, tmp_path):
    result, _, _ = _search(monkeypatch, tmp_path, focus_curve)
    assert result["initial_z"] == START_Z
    assert abs(result["z_shift"] - (result["final_z"] - START_Z)) < 1e-9
    assert "Focus search completed" in result["message"]


def test_search_without_a_peak_goes_back_to_where_it_started(monkeypatch, tmp_path):
    result, hw, _ = _search(monkeypatch, tmp_path, lambda z: 100.0 + 0.01 * np.sin(z * 37.0))
    assert not result["success"]
    assert result["final_z"] is None
    assert "No focus found" in result["message"]
    assert hw.get_current_position().z == START_Z


def test_blank_field_is_refused_rather_than_followed(monkeypatch, tmp_path):
    """Noise on the p98_p2 curve read as a trend: the retries walked 44 um and it committed."""
    rng = np.random.default_rng(3)
    metric = AutofocusUtils.autofocus_profile_brenner_gradient
    hw, _ = _make_dense_hardware(lambda z: 0.0, start_z=START_Z)
    hw.snap_image = lambda: (12000.0 + rng.normal(0, 40.0, (256, 256)), {})
    monkeypatch.setattr(
        af_test, "_load_autofocus_settings", lambda *a: _settings(score_metric=metric)
    )
    result = af_test.focus_search_at_current_position(hw, "config_x.yml", str(tmp_path), "OBJ")
    assert not result["success"]
    assert hw.get_current_position().z == START_Z


def test_cancelled_search_is_not_a_failure(monkeypatch, tmp_path):
    calls = []

    def abort_on_fifth_step():
        calls.append(1)
        return len(calls) >= 5

    result, hw, _ = _search(monkeypatch, tmp_path, focus_curve, should_abort=abort_on_fifth_step)
    assert result["cancelled"]
    assert not result["success"]
    assert hw.get_current_position().z == START_Z


def test_edge_retries_come_from_the_autofocus_config(tmp_path):
    (tmp_path / "config_x.yml").write_text("{}")
    (tmp_path / "autofocus_x.yml").write_text(
        "autofocus_settings:\n- objective: OBJ\n  edge_retries: 4\n"
    )
    log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    settings = af_test._load_autofocus_settings(str(tmp_path / "config_x.yml"), "OBJ", log)
    assert settings["edge_retries"] == 4
