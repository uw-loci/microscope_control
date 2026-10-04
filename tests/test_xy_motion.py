"""Tests for the XY motion profile and the origin-change detection on it.

The failure these guard against is specific: a Prior controller that restarts
re-zeros wherever it stands, then goes exactly where commanded and reports
exactly that. The test that matters most here is
``reported_matching_the_move_is_not_evidence`` -- it pins down the fact that the
obvious check (commanded vs reported after the move) CANNOT see this, so nobody
replaces the cross-time comparison with it later.
"""

import logging

import pytest

from microscope_control.hardware import xy_motion

# --------------------------------------------------------------------------
# resolve_profile
# --------------------------------------------------------------------------


def test_no_config_means_no_profile():
    # Without a declared cap there is no value to compare a readback against,
    # so the honest answer is to disable both checks rather than invent one.
    assert xy_motion.resolve_profile(None) is None
    assert xy_motion.resolve_profile({}) is None
    assert xy_motion.resolve_profile({"stage": {}}) is None
    assert xy_motion.resolve_profile({"stage": {"xy_motion": {}}}) is None


def test_resolve_profile_reads_properties_and_thresholds():
    profile = xy_motion.resolve_profile(
        {
            "stage": {
                "xy_motion": {
                    "properties": {"MaxSpeed": "50", "Acceleration": 40},
                    "verify_after_move_um": 1234,
                    "drift_tolerance_um": 77,
                }
            }
        }
    )
    assert profile is not None
    assert dict(profile.properties) == {"MaxSpeed": "50", "Acceleration": "40"}
    assert profile.verify_after_move_um == 1234.0
    assert profile.drift_tolerance_um == 77.0
    assert not profile.is_empty


def test_resolve_profile_defaults_the_thresholds():
    profile = xy_motion.resolve_profile(
        {"stage": {"xy_motion": {"properties": {"MaxSpeed": "50"}}}}
    )
    assert profile.verify_after_move_um == xy_motion.DEFAULT_VERIFY_AFTER_MOVE_UM
    assert profile.drift_tolerance_um == xy_motion.DEFAULT_DRIFT_TOLERANCE_UM


def test_malformed_properties_are_refused_not_guessed(caplog):
    with caplog.at_level(logging.WARNING):
        assert (
            xy_motion.resolve_profile({"stage": {"xy_motion": {"properties": "MaxSpeed=50"}}})
            is None
        )
    assert "expected a mapping" in caplog.text


def test_properties_with_no_value_are_dropped():
    assert (
        xy_motion.resolve_profile({"stage": {"xy_motion": {"properties": {"MaxSpeed": None}}}})
        is None
    )


# --------------------------------------------------------------------------
# values_agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reported,expected,agree",
    [
        ("50", "50", True),
        # An adapter is free to answer "50" as "50.0000"; comparing as text
        # would report a change that did not happen.
        ("50.0000", "50", True),
        (" 50 ", "50", True),
        (50.0, "50", True),
        ("100", "50", False),
        ("Fast", "Fast", True),
        ("Fast", "Slow", False),
        (None, "50", False),
    ],
)
def test_values_agree(reported, expected, agree):
    assert xy_motion.values_agree(reported, expected) is agree


# --------------------------------------------------------------------------
# apply_profile / describe_profile_mismatch
# --------------------------------------------------------------------------


class FakeCore:
    """Minimal MM Core stand-in that records property writes."""

    def __init__(self, initial=None, reject=(), clamp=None):
        self.props = dict(initial or {})
        self.reject = set(reject)
        self.clamp = clamp or {}
        self.writes = []

    def set_property(self, device, name, value):
        if name in self.reject:
            raise RuntimeError(f"{name} is not writable")
        self.writes.append((device, name, value))
        self.props[name] = self.clamp.get(name, value)

    def get_property(self, device, name):
        if name not in self.props:
            raise RuntimeError(f"no such property {name}")
        return self.props[name]


PROFILE = xy_motion.XyMotionProfile(
    properties=(("MaxSpeed", "50"), ("Acceleration", "50")),
    verify_after_move_um=5000.0,
    drift_tolerance_um=100.0,
)


def test_apply_profile_writes_every_property():
    core = FakeCore({"MaxSpeed": "100", "Acceleration": "100"})
    applied = xy_motion.apply_profile(core, "XYStage", PROFILE)
    assert [w[1:] for w in core.writes] == [("MaxSpeed", "50"), ("Acceleration", "50")]
    assert applied == {"MaxSpeed": "50", "Acceleration": "50"}


def test_apply_profile_survives_a_property_the_adapter_refuses(caplog):
    # A rig whose adapter rejects one of these should run at the adapter's own
    # speed with a warning, not fail to acquire.
    core = FakeCore({"MaxSpeed": "100"}, reject={"Acceleration"})
    with caplog.at_level(logging.WARNING):
        applied = xy_motion.apply_profile(core, "XYStage", PROFILE)
    assert applied == {"MaxSpeed": "50"}
    assert "Could not set XYStage.Acceleration" in caplog.text


def test_apply_profile_reports_a_clamping_adapter(caplog):
    core = FakeCore({"MaxSpeed": "100", "Acceleration": "100"}, clamp={"MaxSpeed": "75"})
    with caplog.at_level(logging.WARNING):
        xy_motion.apply_profile(core, "XYStage", PROFILE)
    assert "reads back 75" in caplog.text


def test_profile_that_still_holds_is_not_a_mismatch():
    core = FakeCore({"MaxSpeed": "50", "Acceleration": "50.0000"})
    assert xy_motion.describe_profile_mismatch(core, "XYStage", PROFILE) is None


def test_a_default_back_in_place_is_reported_as_a_restart():
    # This is the canary: we hold these below the controller's own defaults all
    # session, so finding a default back means the controller restarted.
    core = FakeCore({"MaxSpeed": "100", "Acceleration": "50"})
    message = xy_motion.describe_profile_mismatch(core, "XYStage", PROFILE)
    assert message is not None
    assert "MaxSpeed=100 (expected 50)" in message
    assert "Acceleration" not in message
    assert "0,0" in message


def test_unreadable_properties_are_not_evidence():
    # Reading nothing tells us nothing; it must not be reported as a restart.
    core = FakeCore({})
    assert xy_motion.describe_profile_mismatch(core, "XYStage", PROFILE) is None


# --------------------------------------------------------------------------
# describe_drift
# --------------------------------------------------------------------------


def test_drift_within_tolerance_is_silent():
    assert xy_motion.describe_drift((100.0, 200.0), (100.0, 200.0), 100.0) is None
    assert xy_motion.describe_drift((150.0, 200.0), (100.0, 200.0), 100.0) is None


def test_drift_needs_both_positions():
    # The first move of a session has nothing to compare against.
    assert xy_motion.describe_drift(None, (1.0, 2.0), 100.0) is None
    assert xy_motion.describe_drift((1.0, 2.0), None, 100.0) is None


def test_drift_beyond_tolerance_names_both_positions():
    # The 2026-10-02 shape: the controller re-zeroed while parked on slide 4,
    # so it reports the origin where we left it at (-41453, -7012).
    message = xy_motion.describe_drift((0.0, 0.0), (-41453.0, -7012.0), 100.0)
    assert message is not None
    assert "(0.0, 0.0)" in message
    assert "(-41453.0, -7012.0)" in message
    assert "nothing commanded in" in message


def test_drift_is_measured_as_distance_not_per_axis():
    # 80 um on each axis is 113 um of real disagreement; a per-axis test would
    # pass both and miss it.
    assert xy_motion.describe_drift((80.0, 80.0), (0.0, 0.0), 100.0) is not None
    assert xy_motion.describe_drift((80.0, 0.0), (0.0, 0.0), 100.0) is None
