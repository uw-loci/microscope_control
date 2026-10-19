"""Sequence tests for the stage's XY motion profile and frame guard.

Every assertion here is about behaviour ACROSS several moves, because that is
the only place this failure is visible. A single move always looks healthy: the
controller goes where it is told and reports where it was told to go, both
before and after its origin moves.
"""

import logging

import pytest

from microscope_control.hardware.stage import PycromanagerStage

CONTROLLER_DEFAULTS = {"MaxSpeed": "100", "Acceleration": "100"}

SETTINGS = {
    "stage": {
        "xy_motion": {
            "properties": {"MaxSpeed": "50", "Acceleration": "50"},
            "verify_after_move_um": 5000,
            "drift_tolerance_um": 100,
        }
    }
}


class FakeCore:
    """MM Core stand-in holding the controller's own idea of where it is.

    ``x``/``y`` are the CONTROLLER's coordinates, not a physical location --
    which is the whole point. :meth:`restart` models a ProScan III coming back
    up: its position becomes 0,0 wherever it is standing, and its own property
    defaults return.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.props = dict(CONTROLLER_DEFAULTS)
        self.busy = False
        self.property_writes = []
        self.property_reads = []

    # --- stage ---
    def get_xy_stage_device(self):
        return "XYStage"

    def set_xy_position(self, x, y):
        self.x = x
        self.y = y

    def get_x_position(self):
        return self.x

    def get_y_position(self):
        return self.y

    def wait_for_device(self, device):
        pass

    def device_busy(self, device):
        return self.busy

    # --- properties ---
    def get_property(self, device, name):
        self.property_reads.append(name)
        return self.props[name]

    def set_property(self, device, name, value):
        self.property_writes.append((name, value))
        self.props[name] = value

    # --- fault injection ---
    def restart(self):
        self.x = 0.0
        self.y = 0.0
        self.props = dict(CONTROLLER_DEFAULTS)

    def restart_properties_only(self):
        self.props = dict(CONTROLLER_DEFAULTS)

    def nudge(self, x, y):
        """Model the operator moving the stage from the joystick.

        The position changes; the controller keeps its reference and the
        properties we set, because nothing restarted.
        """
        self.x = x
        self.y = y


@pytest.fixture
def core():
    return FakeCore()


@pytest.fixture
def stage(core):
    return PycromanagerStage(core, SETTINGS)


def test_profile_is_applied_on_the_first_move_and_not_again(stage, core):
    stage.move_xy(1000.0, 1000.0)
    assert core.props["MaxSpeed"] == "50"
    assert core.props["Acceleration"] == "50"
    assert len(core.property_writes) == 2

    stage.move_xy(1200.0, 1000.0)
    assert len(core.property_writes) == 2


def test_no_profile_configured_leaves_the_stage_alone(core):
    stage = PycromanagerStage(core, {"stage": {}})
    stage.move_xy(1000.0, 1000.0)
    assert core.property_writes == []
    assert stage.xy_frame_suspect is None


def test_reported_matching_the_move_is_not_evidence(stage, core):
    """The check that CANNOT see this failure, pinned down so it stays out.

    Commanded and reported agree immediately after every move, healthy or not,
    so an arrival comparison is blind here. Nothing about a sequence of correct
    moves may raise suspicion.
    """
    stage.set_frame_check_armed(True)
    for x in (10000.0, 10300.0, 10600.0, -41453.0):
        stage.move_xy(x, -7012.0)
        assert (core.get_x_position(), core.get_y_position()) == (x, -7012.0)
        assert stage.xy_frame_suspect is None


def test_a_restart_between_moves_is_caught_on_the_next_move(stage, core, caplog):
    stage.set_frame_check_armed(True)
    stage.move_xy(-41453.0, -7012.0)
    assert stage.xy_frame_suspect is None

    # The controller restarts while parked on slide 4. Nothing is commanded, so
    # the next move's pre-check is comparing a stationary stage against the
    # position we last sent it to.
    core.restart()

    with caplog.at_level(logging.ERROR):
        stage.move_xy(-41153.0, -7012.0)

    suspect = stage.xy_frame_suspect
    assert suspect is not None
    assert "(-41453.0, -7012.0)" in suspect
    assert "STAGE FRAME SUSPECT" in caplog.text


def test_a_joystick_nudge_while_disarmed_is_not_reported(stage, core):
    # Outside an acquisition the operator may move the stage by hand, and that
    # disagreement with our last commanded position is not evidence of
    # anything. This is why the drift check is armed only for an acquisition.
    stage.move_xy(-41453.0, -7012.0)
    core.nudge(-30000.0, -7012.0)
    stage.move_xy(-41153.0, -7012.0)
    assert stage.xy_frame_suspect is None


def test_the_same_nudge_is_reported_once_armed(stage, core):
    stage.move_xy(-41453.0, -7012.0)
    stage.set_frame_check_armed(True)
    core.nudge(-30000.0, -7012.0)
    stage.move_xy(-41153.0, -7012.0)
    assert stage.xy_frame_suspect is not None


def test_a_restart_is_caught_even_while_disarmed(stage, core):
    # The readback does not depend on the drift check: a restart restores the
    # controller's own property defaults, and we never write those.
    stage.move_xy(-41453.0, -7012.0)
    core.restart()
    stage.move_xy(-41153.0, -7012.0)
    assert "MaxSpeed=100 (expected 50)" in (stage.xy_frame_suspect or "")


def test_motion_in_flight_is_not_mistaken_for_drift(stage, core):
    stage.set_frame_check_armed(True)
    stage.move_xy_no_wait(10000.0, 0.0)
    # A previous non-blocking move still running: a position read is a point on
    # the way, not an arrival, so there is nothing to compare.
    core.busy = True
    core.x, core.y = 4000.0, 0.0
    stage.move_xy_no_wait(20000.0, 0.0)
    assert stage.xy_frame_suspect is None


def test_the_profile_canary_fires_on_a_long_move(stage, core):
    # Disarmed, so only the readback can speak. The properties come back at the
    # controller's defaults; the position is untouched.
    stage.move_xy(1000.0, 0.0)
    core.restart_properties_only()

    stage.move_xy(20000.0, 0.0)

    suspect = stage.xy_frame_suspect
    assert suspect is not None
    assert "MaxSpeed=100 (expected 50)" in suspect


def test_a_short_move_does_not_read_the_profile_back(stage, core):
    stage.move_xy(1000.0, 0.0)
    core.property_reads.clear()
    # Under verify_after_move_um: a readback per tile step would buy a serial
    # round trip per tile and no new information.
    stage.move_xy(1300.0, 0.0)
    assert core.property_reads == []


def test_suspect_is_sticky(stage, core):
    stage.set_frame_check_armed(True)
    stage.move_xy(-41453.0, -7012.0)
    core.restart()
    stage.move_xy(-41153.0, -7012.0)
    first = stage.xy_frame_suspect
    assert first is not None

    # Healthy moves afterwards must not clear it. The frame does not heal; it
    # has to be re-established at the calibration fiducial.
    stage.move_xy(-41000.0, -7012.0)
    stage.move_xy(-40800.0, -7012.0)
    assert stage.xy_frame_suspect == first


def test_a_move_by_hand_before_arming_is_not_reported(stage, core):
    # 2026-10-19: the operator moved the stage by hand between our last move
    # and the start of the next acquisition. Arming must measure from where the
    # stage stands, not from where we last sent it.
    stage.move_xy(2318.5, -1507.9)
    core.nudge(3628.0, 1636.0)
    stage.set_frame_check_armed(True)
    stage.move_xy(3928.0, 1636.0)
    assert stage.xy_frame_suspect is None


def test_a_move_by_hand_during_a_pause_is_not_reported(stage, core):
    # Manual focus: the run disarms, the operator finds tissue with the
    # joystick, the run re-arms and moves the stage back.
    stage.set_frame_check_armed(True)
    stage.move_xy(2318.5, -1507.9)
    stage.set_frame_check_armed(False)
    core.nudge(3628.0, 1636.0)
    stage.set_frame_check_armed(True)
    stage.move_xy(2318.5, -1507.9)
    assert stage.xy_frame_suspect is None


def test_drift_after_re_arming_is_still_reported(stage, core):
    stage.move_xy(1000.0, 0.0)
    stage.set_frame_check_armed(True)
    core.nudge(3000.0, 0.0)
    stage.move_xy(1300.0, 0.0)
    assert stage.xy_frame_suspect is not None


def test_a_restart_during_a_pause_is_caught_on_re_arming(stage, core):
    # Re-baselining would hide the position jump, so arming reads the profile
    # back: a restarted controller has its own defaults in place.
    stage.set_frame_check_armed(True)
    stage.move_xy(-41453.0, -7012.0)
    stage.set_frame_check_armed(False)
    core.restart()
    stage.set_frame_check_armed(True)
    assert "MaxSpeed=100 (expected 50)" in (stage.xy_frame_suspect or "")


def test_arming_before_the_first_move_is_not_a_restart(stage, core):
    # The profile has not been written yet, so the defaults are expected.
    stage.set_frame_check_armed(True)
    assert stage.xy_frame_suspect is None
    assert stage.frame_check_armed is True
