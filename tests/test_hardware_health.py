"""The per-part hardware status the server reports (HWSTAT)."""

from microscope_control.hardware.health import (
    CAMERA,
    MICROMANAGER,
    STAGE,
    HardwareHealth,
    is_bridge_timeout,
)


class _FakeCache:
    def __init__(self, status):
        self._status = status

    def down_status(self):
        return dict(self._status)


def test_a_part_fails_from_its_first_failure_until_it_answers():
    h = HardwareHealth()
    assert h.snapshot()[CAMERA]["ok"] is True
    h.failed(CAMERA, RuntimeError("Camera snap failed\nmore"))
    h.failed(CAMERA, RuntimeError("again"))
    cam = h.snapshot()[CAMERA]
    assert cam["ok"] is False
    assert cam["failures"] == 2
    assert cam["message"] == "again"
    first_since = cam["since"]
    assert first_since is not None
    h.failed(CAMERA, "third")
    assert h.snapshot()[CAMERA]["since"] == first_since, "since is the first failure"
    h.succeeded(CAMERA)
    assert h.snapshot()[CAMERA] == {"ok": True, "since": None, "message": "", "failures": 0}


def test_the_stage_comes_from_the_position_cache():
    h = HardwareHealth()
    assert h.snapshot()[STAGE]["known"] is False
    down = {"ok": False, "since": 100.0, "message": "Stage not responding", "failures": 5}
    s = h.snapshot(_FakeCache(down))[STAGE]
    assert s["ok"] is False and s["known"] is True and s["message"] == "Stage not responding"
    assert h.snapshot()[MICROMANAGER]["ok"] is True


def test_bridge_timeouts_are_recognized():
    assert is_bridge_timeout(TimeoutError("timed out"))
    assert is_bridge_timeout(RuntimeError("zmq.error.Again: Resource temporarily unavailable"))
    assert not is_bridge_timeout(RuntimeError("Error in device Camera: Exposure out of range"))
