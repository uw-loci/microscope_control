"""The trace recorder: every Core call lands in hardware.jsonl, ASCII only, frames as digests."""

import json
import threading
import time

import numpy as np
import pytest

from microscope_control.trace import (
    RecordingProxy,
    Trace,
    TraceWriter,
    frame_mode_from_env,
    record_core,
    record_studio,
)
from microscope_control.trace.values import (
    StrVector,
    TaggedImage,
    decode_value,
    encode_value,
    values_match,
)


class FakeCore:
    """Just enough of the MMCore surface for the recorder to have something to wrap."""

    def __init__(self):
        self.x = 0.0
        self.frame = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        self.props = {}

    def get_x_position(self):
        return self.x

    def set_xy_position(self, x, y):
        self.x = x

    def get_image(self):
        return self.frame

    def get_tagged_image(self):
        return TaggedImage(self.frame.reshape(-1), {"Width": 4, "Height": 3})

    def get_loaded_devices(self):
        return StrVector(["Core", "DCam"])

    def set_property(self, device, name, value):
        self.props[(device, name)] = value

    def boom(self):
        raise RuntimeError("Device not found: Nope")


class FakeLive:
    def __init__(self):
        self.on = False

    def set_live_mode(self, on):
        self.on = on

    def is_live_mode_on(self):
        return self.on


class FakeStudio:
    def __init__(self):
        self._live = FakeLive()

    def live(self):
        return self._live


def _ascii_only(path):
    data = path.read_bytes()
    return all(b < 128 for b in data)


@pytest.fixture
def writer(tmp_path):
    w = TraceWriter(tmp_path / "trace", {"server": "test"})
    yield w
    w.close()


def test_calls_are_recorded_with_args_returns_and_timing(writer):
    core = FakeCore()
    rc = record_core(core, writer, "hash")
    writer.set_stage(3, "MOVE")
    rc.set_xy_position(10.0, 20.0)
    assert rc.get_x_position() == 10.0  # passes straight through
    writer.close()

    trace = Trace(writer.directory)
    calls = trace.hardware_calls()
    assert [c["method"] for c in calls] == ["set_xy_position", "get_x_position"]
    move = calls[0]
    assert move["args"] == [10.0, 20.0] and move["ret"] is None
    assert move["stage"] == 3 and move["obj"] == "core"
    assert move["thread"] == threading.current_thread().name
    assert move["dt_ms"] >= 0 and move["t"] >= 0
    assert calls[1]["ret"] == 10.0
    assert calls[0]["seq"] < calls[1]["seq"]


def test_images_are_stored_as_digest_shape_and_dtype_not_pixels(writer):
    core = FakeCore()
    rc = record_core(core, writer, "hash")
    frame = rc.get_image()
    assert frame is core.frame
    tagged = rc.get_tagged_image()
    assert tagged.tags["Width"] == 4
    writer.close()

    text = (writer.directory / "hardware.jsonl").read_text()
    assert "23" not in json.dumps(
        json.loads(text.splitlines()[0])["ret"]["__ndarray__"].get("data", "")
    )
    img = json.loads(text.splitlines()[0])["ret"]["__ndarray__"]
    assert img["shape"] == [2, 3, 4] and img["dtype"] == "uint16"
    assert len(img["digest"]) == 32 and img["min"] == 0 and img["max"] == 23
    assert not (writer.directory / "frames").exists()
    pix = json.loads(text.splitlines()[1])["ret"]["__tagged_image__"]["pix"]["__ndarray__"]
    assert pix["shape"] == [24]


def test_errors_are_recorded_and_re_raised(writer):
    rc = record_core(FakeCore(), writer, "hash")
    with pytest.raises(RuntimeError, match="Nope"):
        rc.boom()
    writer.close()
    rec = Trace(writer.directory).hardware_calls()[0]
    assert rec["error"] == {"type": "RuntimeError", "message": "Device not found: Nope"}
    assert "ret" not in rec


def test_non_ascii_values_are_escaped_so_the_file_stays_cp1252_safe(writer):
    rc = record_core(FakeCore(), writer, "hash")
    rc.set_property("DCam", "Label", "5 µm → ok")
    writer.update_manifest(note="café")
    writer.close()
    assert _ascii_only(writer.directory / "hardware.jsonl")
    assert _ascii_only(writer.directory / "manifest.json")
    rec = Trace(writer.directory).hardware_calls()[0]
    assert rec["args"][2] == "5 \\xb5m \\u2192 ok"


def test_strvector_round_trips_as_a_list_that_answers_size_and_get(writer):
    rc = record_core(FakeCore(), writer, "hash")
    rc.get_loaded_devices()
    writer.close()
    rec = Trace(writer.directory).hardware_calls()[0]
    assert rec["ret"] == {"__strvector__": ["Core", "DCam"]}
    back = decode_value(rec["ret"])
    assert back.size() == 2 and back.get(1) == "DCam" and list(back) == ["Core", "DCam"]


def test_studio_live_calls_are_recorded_under_their_own_object_name(writer):
    studio = record_studio(FakeStudio(), writer, "hash")
    studio.live().set_live_mode(True)
    assert studio.live().is_live_mode_on() is True
    writer.close()
    calls = Trace(writer.directory).hardware_calls()
    assert [(c["obj"], c["method"]) for c in calls] == [
        ("studio.live", "set_live_mode"),
        ("studio.live", "is_live_mode_on"),
    ]
    assert record_studio(None, writer) is None


@pytest.mark.parametrize("mode", ["none", "hash", "preview", "full"])
def test_frame_modes_and_what_replay_gets_back(tmp_path, mode):
    writer = TraceWriter(tmp_path / mode)
    core = FakeCore()
    core.frame = np.random.default_rng(1).integers(0, 4000, (300, 400), dtype=np.uint16)
    rc = record_core(core, writer, mode)
    rc.get_image()
    writer.close()
    trace = Trace(writer.directory)
    info = trace.hardware_calls()[0]["ret"]["__ndarray__"]
    back = decode_value(trace.hardware_calls()[0]["ret"], trace.frames_dir)
    assert back.shape == (300, 400) and back.dtype == np.uint16
    if mode == "none":
        assert "digest" not in info and back.max() == 0
    elif mode == "hash":
        assert "preview_file" not in info and "file" not in info
        assert int(back[0, 0]) == int(round(info["mean"]))  # filled with the mean
    elif mode == "preview":
        assert (trace.frames_dir / info["preview_file"]).exists()
        assert info["preview_step"] == 4  # 400 px wide -> 128 px preview
        assert np.array_equal(back[::4, ::4], core.frame[::4, ::4])
    else:
        assert np.array_equal(back, core.frame)


def test_frame_mode_env(monkeypatch):
    monkeypatch.delenv("MICROSCOPE_TRACE_FRAMES", raising=False)
    assert frame_mode_from_env() == "hash"
    monkeypatch.setenv("MICROSCOPE_TRACE_FRAMES", "FULL")
    assert frame_mode_from_env() == "full"
    monkeypatch.setenv("MICROSCOPE_TRACE_FRAMES", "pixels")
    with pytest.raises(ValueError):
        frame_mode_from_env()


def test_values_match_uses_the_tolerance_and_names_the_first_differing_field():
    assert values_match([1.0, 2.0], [1.0, 2.0000001], abs_tol=1e-6) == (True, [])
    ok, path = values_match([1.0, {"z": 3.0}], [1.0, {"z": 3.5}], abs_tol=0.1)
    assert not ok and path == ["1", "z"]
    a = encode_value(np.zeros((2, 2), np.uint8))
    b = encode_value(np.ones((2, 2), np.uint8))
    ok, path = values_match([a], [b])
    assert not ok and path == ["0", "digest"]
    assert values_match([a], [encode_value(np.zeros((2, 2), np.uint8))])[0]


def test_manifest_and_summary(writer):
    rc = record_core(FakeCore(), writer, "hash")
    rc.set_xy_position(1.0, 1.0)
    writer.manifest_append("config_files", {"path": "a.yml"})
    writer.manifest_append("config_files", {"path": "a.yml"})  # no duplicate
    writer.close()
    trace = Trace(writer.directory)
    assert trace.manifest["server"] == "test"
    assert trace.manifest["config_files"] == [{"path": "a.yml"}]
    assert trace.manifest["format_version"] == 1
    assert "end_time" in trace.manifest and trace.manifest["duration_s"] >= 0
    summary = trace.summary()
    assert summary["hardware_calls"] == 1 and summary["methods"] == {"core.set_xy_position": 1}


def test_recording_proxy_rejects_an_unknown_frame_mode(writer):
    with pytest.raises(ValueError):
        RecordingProxy(FakeCore(), writer, frame_mode="pixels")


def test_recording_adds_microseconds_per_call_not_milliseconds(writer):
    core = FakeCore()
    rc = record_core(core, writer, "hash")
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        rc.get_x_position()
    per_call = (time.perf_counter() - t0) / n
    # Measured 10 us on the dev box; 500 us leaves room for a slow CI disk.
    assert per_call < 500e-6, f"{per_call * 1e6:.0f} us per recorded call"
