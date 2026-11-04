"""StagePositionCache with a stage that stops answering (controller off or unplugged)."""

import logging

import pytest

from microscope_control.hardware import stage_cache as sc
from microscope_control.hardware.stage_cache import StagePositionCache, StageUnavailableError

SERIAL_ERROR = (
    'java.lang.Exception: Error in device "XYStage": Serial command failed.  '
    "Is the device connected to the serial port? (14)\n"
    "mmcorej.MMCoreJJNI.CMMCore_getXPosition__SWIG_1(Native Method)\n"
    "mmcorej.CMMCore.getXPosition(CMMCore.java:1315)"
)


class FakeClock:
    """Stands in for the time module inside stage_cache."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class FakeStage:
    """get_xyz() raises while `off` is True, else returns a fixed position."""

    def __init__(self):
        self.off = False
        self.calls = 0

    def get_xyz(self):
        self.calls += 1
        if self.off:
            raise Exception(SERIAL_ERROR)
        return (100.0, 200.0, -30.0)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(sc, "time", fake)
    return fake


@pytest.fixture
def stage():
    return FakeStage()


@pytest.fixture
def cache(stage, clock):
    # start() is not called: the tests drive reads themselves, no thread runs.
    return StagePositionCache(stage)


def _fail(cache, n, clock, step_s=1.0):
    for _ in range(n):
        clock.now += step_s
        with pytest.raises(Exception):
            cache.force_refresh()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_short_blip_logs_no_warning_and_blocks_nothing(cache, stage, clock, caplog):
    caplog.set_level(logging.DEBUG, logger=sc.__name__)
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD - 1, clock)
    assert not cache.is_stage_down
    assert _warnings(caplog) == []

    stage.off = False
    calls = stage.calls
    assert cache.get_cached_position(max_age_ms=250.0).x == 100.0
    assert stage.calls == calls + 1


def test_a_stage_that_stays_off_gets_one_warning_with_the_first_line_only(
    cache, stage, clock, caplog
):
    caplog.set_level(logging.DEBUG, logger=sc.__name__)
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD + 20, clock)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "Serial command failed" in message
    assert "powered on" in message
    assert "mmcorej" not in message
    assert cache.is_stage_down


def test_while_down_position_requests_fail_fast_without_a_serial_read(cache, stage, clock):
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD, clock)
    calls = stage.calls

    clock.now += sc._DOWN_RETRY_S - 0.1
    with pytest.raises(StageUnavailableError) as err:
        cache.get_cached_position(max_age_ms=250.0)
    assert stage.calls == calls
    assert "Serial command failed" in str(err.value)
    assert "\n" not in str(err.value)


def test_while_down_a_request_after_the_retry_interval_reads_the_stage(cache, stage, clock):
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD, clock)
    calls = stage.calls

    clock.now += sc._DOWN_RETRY_S + 0.1
    with pytest.raises(Exception) as err:
        cache.get_cached_position(max_age_ms=250.0)
    assert not isinstance(err.value, StageUnavailableError)
    assert stage.calls == calls + 1


def test_force_refresh_reads_the_stage_even_while_down(cache, stage, clock):
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD, clock)
    calls = stage.calls

    with pytest.raises(Exception) as err:
        cache.force_refresh()
    assert not isinstance(err.value, StageUnavailableError)
    assert stage.calls == calls + 1


def test_the_warning_repeats_after_the_reminder_interval(cache, stage, clock, caplog):
    caplog.set_level(logging.DEBUG, logger=sc.__name__)
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD, clock)
    _fail(cache, 10, clock, step_s=sc._DOWN_REMINDER_S / 20.0)
    assert len(_warnings(caplog)) == 1

    _fail(cache, 11, clock, step_s=sc._DOWN_REMINDER_S / 20.0)
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    assert "still not responding" in warnings[1].getMessage()


def test_recovery_is_logged_and_clears_the_state(cache, stage, clock, caplog):
    caplog.set_level(logging.DEBUG, logger=sc.__name__)
    stage.off = True
    _fail(cache, sc._SUSTAINED_ERROR_THRESHOLD + 2, clock)

    stage.off = False
    clock.now += sc._DOWN_RETRY_S + 0.1
    assert cache.get_cached_position(max_age_ms=250.0).y == 200.0
    assert not cache.is_stage_down
    assert any(
        r.levelno == logging.INFO and "responding again" in r.getMessage() for r in caplog.records
    )

    # Back to normal: a fresh request is served without raising.
    clock.now += 0.1
    assert cache.get_cached_position(max_age_ms=250.0).z == -30.0


class _CountingStop:
    """Replaces the cache's stop event: records each wait, stops after `limit`."""

    def __init__(self, limit):
        self.limit = limit
        self.waits = []

    def is_set(self):
        return len(self.waits) >= self.limit

    def wait(self, timeout):
        self.waits.append(timeout)
        return len(self.waits) >= self.limit

    def clear(self):
        pass


def test_the_poll_loop_slows_down_once_the_stage_counts_as_down(cache, stage, clock):
    stage.off = True
    stop = _CountingStop(limit=6)
    cache._stop_event = stop
    cache._poll_loop()

    threshold = sc._SUSTAINED_ERROR_THRESHOLD
    assert stop.waits[: threshold - 1] == [cache._retry_backoff_s] * (threshold - 1)
    assert stop.waits[threshold - 1 :] == [sc._DOWN_RETRY_S] * (6 - threshold + 1)


def test_the_poll_loop_uses_the_normal_interval_when_the_stage_answers(cache, stage, clock):
    stop = _CountingStop(limit=3)
    cache._stop_event = stop
    cache._poll_loop()
    assert stop.waits == [cache._poll_interval_s] * 3
    assert cache.poll_count == 3
