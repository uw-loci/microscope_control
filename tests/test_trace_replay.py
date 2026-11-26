"""ReplayCore serves a trace in order and reports every departure as a grouped Difference."""

import json

import numpy as np
import pytest

from microscope_control.trace import (
    Difference,
    ReplayDivergence,
    ReplaySession,
    Trace,
    TraceWriter,
    group_by_fingerprint,
    read_differences,
    record_core,
    record_studio,
    write_differences,
)
from microscope_control.trace.diff import (
    KIND_ARGS,
    KIND_EXTRA,
    KIND_METHOD,
    KIND_MISSING,
    KIND_UNRECORDED_QUERY,
    format_groups,
    main as group_main,
)
from microscope_control.trace.replay import is_query
from microscope_control.trace.values import RecordedError


class FakeCore:
    """A stage that is busy for two polls after a move, and a camera with one frame."""

    def __init__(self):
        self.x = 0.0
        self.z = 5.0
        self.busy_polls = 0
        self.exposure = 10.0
        self.frame = np.full((4, 5), 7, dtype=np.uint8)

    def set_xy_position(self, x, y):
        self.x = x
        self.busy_polls = 2

    def device_busy(self, device):
        if self.busy_polls:
            self.busy_polls -= 1
            return True
        return False

    def wait_for_device(self, device):
        self.busy_polls = 0

    def get_x_position(self):
        return self.x

    def get_position(self):
        return self.z

    def set_position(self, z):
        self.z = z

    def set_exposure(self, ms):
        self.exposure = ms

    def get_exposure(self):
        return self.exposure

    def snap_image(self):
        pass

    def get_image(self):
        return self.frame

    def set_property(self, device, name, value):
        if name == "Nope":
            raise RuntimeError("No such property")


class FakeLive:
    def set_live_mode(self, on):
        pass

    def is_live_mode_on(self):
        return False


class FakeStudio:
    def live(self):
        return FakeLive()


def session_script(core, studio):
    """The calls the 'recorded session' makes; replayed verbatim by the unchanged code."""
    out = {}
    core.set_xy_position(100.0, 200.0)
    polls = []
    while core.device_busy("XYStage"):
        polls.append(True)
    out["polls"] = len(polls)
    out["x"] = core.get_x_position()
    core.set_exposure(12.5)
    out["exposure"] = core.get_exposure()
    core.snap_image()
    out["frame"] = core.get_image()
    core.set_position(7.5)
    out["z"] = core.get_position()
    try:
        core.set_property("DCam", "Nope", "1")
    except RuntimeError as exc:
        out["error"] = str(exc)
    studio.live().set_live_mode(True)
    out["live"] = studio.live().is_live_mode_on()
    return out


@pytest.fixture
def recorded(tmp_path):
    writer = TraceWriter(tmp_path / "trace")
    core = FakeCore()
    writer.set_stage(1, "MOVE")
    recorded_out = session_script(
        record_core(core, writer, "hash"), record_studio(FakeStudio(), writer)
    )
    writer.close()
    return Trace(writer.directory), recorded_out


def test_query_classification():
    assert (
        is_query("get_x_position") and is_query("device_busy") and is_query("is_sequence_running")
    )
    assert is_query("get_remaining_image_count") and is_query("has_property_limits")
    assert not is_query("get_image") and not is_query("get_tagged_image")
    assert not is_query("set_xy_position") and not is_query("snap_image") and not is_query("home")


def test_unchanged_code_replays_with_zero_differences(recorded):
    trace, recorded_out = recorded
    session = ReplaySession(trace, mode="strict")
    out = session_script(session.core(), session.studio())
    assert out["polls"] == recorded_out["polls"] == 2  # device_busy True, True, False as recorded
    assert out["x"] == 100.0 and out["exposure"] == 12.5 and out["z"] == 7.5
    assert out["frame"].shape == (4, 5) and out["frame"].dtype == np.uint8
    assert int(out["frame"][0, 0]) == 7  # hash mode: filled with the recorded mean
    assert out["error"] == recorded_out["error"] == "No such property"
    assert out["live"] is False
    assert session.finish() == []
    assert session.summary().startswith("7/7 recorded hardware commands consumed")


def test_queries_are_answered_from_the_state_after_the_last_command(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="strict")
    core = session.core()
    # Asked out of order and more often than recorded: still the recorded state.
    assert core.get_x_position() == 100.0  # first recorded read after the start
    core.set_xy_position(100.0, 200.0)
    for _ in range(10):
        core.device_busy("XYStage")
    assert core.device_busy("XYStage") is False
    assert core.get_exposure() == 12.5  # the nearest recorded read after the cursor


def test_a_different_argument_is_a_difference_with_the_field_named(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="strict")
    with pytest.raises(ReplayDivergence) as info:
        session.core().set_xy_position(-100.0, 200.0)
    diff = info.value.difference
    assert diff.kind == KIND_ARGS and diff.subject == "core.set_xy_position"
    assert (
        diff.field == "args[0]"
        and diff.expected == [100.0, 200.0]
        and diff.actual == [-100.0, 200.0]
    )
    assert (
        diff.stage == 1 and diff.fingerprint == "-|hardware_call_args|core.set_xy_position|args[0]"
    )


def test_arguments_within_the_tolerance_match(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="strict", tolerances={"set_xy_position": 0.5})
    session.core().set_xy_position(100.3, 199.8)
    assert session.differences == []


def test_a_different_method_and_an_extra_call_are_differences(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="strict")
    with pytest.raises(ReplayDivergence) as info:
        session.core().snap_image()
    assert info.value.difference.kind == KIND_METHOD
    assert info.value.difference.expected == "core.set_xy_position"
    assert "expected core.set_xy_position(100.0, 200.0) next" in info.value.difference.detail

    session = ReplaySession(trace, mode="record")
    session_script(session.core(), session.studio())
    assert session.core().set_position(1.0) is None  # record mode: noted, not raised
    assert session.differences[-1].kind == KIND_EXTRA
    assert session.differences[-1].subject == "core.set_position"


def test_missing_calls_are_reported_at_the_end(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="record")
    session.core().set_xy_position(100.0, 200.0)
    diffs = session.finish()
    assert [d.kind for d in diffs] == [KIND_MISSING] * 6
    assert diffs[0].subject == "core.set_exposure" and diffs[0].expected == [12.5]
    assert diffs[-1].subject == "studio.live.set_live_mode"


def test_an_unrecorded_query_cannot_be_answered(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="record")
    with pytest.raises(ReplayDivergence) as info:
        session.core().get_property("DCam", "Binning")
    assert info.value.difference.kind == KIND_UNRECORDED_QUERY
    assert info.value.difference.subject == "core.get_property"


def test_record_mode_notes_differences_and_keeps_serving(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="record")
    core = session.core()
    core.set_xy_position(-100.0, 200.0)  # noted, not raised
    core.set_exposure(12.5)
    core.snap_image()
    frame = core.get_image()
    assert frame.shape == (4, 5)
    diffs = session.finish()
    kinds = [d.kind for d in diffs]
    assert kinds[0] == KIND_ARGS and KIND_MISSING in kinds


def test_recorded_errors_are_raised_on_replay(recorded):
    trace, _ = recorded
    session = ReplaySession(trace, mode="record")
    core = session.core()
    for _ in range(5):  # consume up to the failing set_property
        session.cursor += 1
        session.cursor_seq = session.commands[session.cursor - 1]["seq"]
    with pytest.raises(RecordedError, match="No such property"):
        core.set_property("DCam", "Nope", "1")


def test_differences_write_read_and_group(tmp_path):
    diffs = [
        Difference(3, "MOVE", KIND_ARGS, "core.set_xy_position", "args[0]", 1.0, -1.0),
        Difference(8, "MOVE", KIND_ARGS, "core.set_xy_position", "args[0]", 2.0, -2.0),
        Difference(14, "ACQUIRE", KIND_MISSING, "core.snap_image", "args", [], None),
    ]
    path = tmp_path / "differences.jsonl"
    assert write_differences(path, diffs) == 3
    back = read_differences(path)
    assert [d.fingerprint for d in back] == [d.fingerprint for d in diffs]
    groups = group_by_fingerprint(back)
    assert list(groups) == [
        "MOVE|hardware_call_args|core.set_xy_position|args[0]",
        "ACQUIRE|hardware_call_missing|core.snap_image|args",
    ]
    assert len(groups["MOVE|hardware_call_args|core.set_xy_position|args[0]"]) == 2
    text = format_groups(groups)
    assert "3 difference(s) in 2 group(s)" in text and "stages: 3, 8" in text
    assert group_main([str(path)]) == 1
    assert group_main([str(tmp_path / "none.jsonl")]) == 0
    assert all(b < 128 for b in path.read_bytes())


def test_oracle_records_feed_the_skill_fingerprint_tool(tmp_path):
    from microscope_control.trace import write_oracle_records

    diff = Difference(3, "MOVE", KIND_ARGS, "core.set_xy_position", "args[0]", 1.0, -1.0, "tol")
    out = tmp_path / "records.json"
    assert write_oracle_records(out, [diff]) == 1
    rec = json.loads(out.read_text())[0]
    assert rec["stage"] == "MOVE" and rec["output"] == "core.set_xy_position"
    assert rec["kind"] == KIND_ARGS and rec["location"] == "args[0]"
    assert rec["value_a"] == 1.0 and rec["value_b"] == -1.0
    assert rec["location_class"] == "args" and rec["detail"]["replay_stage"] == 3


def test_fingerprint_leaves_out_stage_and_values():
    a = Difference(1, "SNAP", KIND_ARGS, "core.set_exposure", "args[0]", 1.0, 2.0)
    b = Difference(9, "SNAP", KIND_ARGS, "core.set_exposure", "args[0]", 3.0, 4.0)
    assert a.fingerprint == b.fingerprint == "SNAP|hardware_call_args|core.set_exposure|args[0]"
    assert "stage 1 SNAP" in a.one_line()
