"""Background-polled stage position cache.

A single shared cache of the (x, y, z) stage position so that many
independent callers (live position display, frame overlays, progress
loggers, acquisition tile metadata) can read a recent stage position
without each issuing its own serial query through the Pycromanager
ZMQ bridge.

Why this exists
---------------
On microscopes whose XY stage talks over a slow serial link
(Prior ProScan via Micro-Manager, ~30-100 ms per get_position call),
multiple independent readers competing for the bus serialize at the
ZMQ bridge and occasionally time out with "Serial command failed".
Observed scenario: the live viewer's position panel polls every
500 ms while a Smooth Focus scan is also reading Z and an
acquisition workflow is reading XY for tile metadata. Three readers,
one bus, transient failures.

The fix is the classic shared-cache pattern: one background thread
polls the stage at a fixed cadence, and every non-critical reader
hits the in-memory cache instead of the wire. Critical readers
(starting/ending positions for a focus scan, per-tile acquisition
metadata) bypass the cache via force_refresh().

Lifecycle
---------
The cache is owned by PycromanagerHardware and (re)created whenever
the underlying stage is (re)created. The polling thread is a daemon,
so it dies with the process; explicit stop() is only required on
config reload when the stage is rebuilt against a different
microscope.

Threading model
---------------
- One daemon polling thread, started by start().
- self._lock guards the (x, y, z, timestamp) snapshot only.
- Stage hardware calls (get_xyz, force_refresh) are not held under
  the lock -- the bridge already serializes them, and holding a
  Python lock across a 100 ms serial round-trip would defeat the
  purpose of the cache.
- pause()/resume() let callers temporarily quiesce the polling
  thread for tight loops where they need the bus to themselves
  (e.g. a Z scan that is sensitive to bus contention).

Stage not responding
--------------------
After _SUSTAINED_ERROR_THRESHOLD reads in a row fail (stage controller
off or unplugged), the cache counts the stage as down: it logs one
WARNING naming the error, retries every _DOWN_RETRY_S instead of at the
poll rate, repeats the WARNING every _DOWN_REMINDER_S, and logs INFO
when a read succeeds again. While it is down, get_cached_position()
raises StageUnavailableError at once instead of making its own serial
read, so position requests from the client neither wait for a serial
timeout nor log a stack trace each. force_refresh() always reads the
stage, so callers that need a live position still get the real error.
Failures from both paths count toward the same state.
"""

import logging
import threading
import time
from typing import Optional

from microscope_control.hardware.base import Position
from microscope_control.hardware.stage import Stage

logger = logging.getLogger(__name__)

# Number of consecutive failed reads before the stage counts as down.
# Below this, transient serial-bus blips are logged at DEBUG so they
# don't dominate long-running session logs.
_SUSTAINED_ERROR_THRESHOLD = 3

# While the stage is down: seconds between read attempts, and seconds
# between repeats of the "not responding" WARNING.
_DOWN_RETRY_S = 5.0
_DOWN_REMINDER_S = 600.0


def _first_line(exc: BaseException) -> str:
    """The first line of an exception's text.

    Java exceptions from mmcorej (through the Pycromanager ZMQ bridge)
    carry their whole stack trace in str(e).
    """
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


class StageUnavailableError(RuntimeError):
    """The stage has not answered its last few reads; no live read was made.

    Raised by StagePositionCache.get_cached_position() while the stage
    counts as down. The message names the last error and how long the
    stage has been failing.
    """


class StagePositionCache:
    """Shared cache of the latest stage XYZ position.

    Construct with a Stage, call start() once, then read via
    get_cached_position() (cheap, in-memory) or force_refresh()
    (live, hits hardware).
    """

    def __init__(
        self,
        stage: Stage,
        poll_interval_s: float = 0.5,
        retry_backoff_s: float = 0.5,
    ):
        """
        Args:
            stage: The Stage instance to poll.
            poll_interval_s: Time between background polls. 0.5 s
                is fast enough for the 500 ms live position display
                refresh and keeps contention with stage operations
                low. Earlier 0.1 s (10 Hz) was too aggressive -- it
                added constant serial traffic that competed with
                Z-scroll move_z busy-polls for the Prior stage's
                single serial bus and made the 2026-04-15 connection
                storm much worse. With the stage-level RLock now
                serializing all stage access, high-frequency cache
                polling just queues behind moves and adds no value.
            retry_backoff_s: How long to sleep after a polling
                error before trying again, until the stage counts as
                down (then _DOWN_RETRY_S). Kept short so transient
                serial blips don't visibly stale the cache.
        """
        self._stage = stage
        self._poll_interval_s = float(poll_interval_s)
        self._retry_backoff_s = float(retry_backoff_s)

        self._lock = threading.Lock()
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._z: Optional[float] = None
        self._timestamp: float = 0.0  # monotonic seconds; 0 means "never"

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set => paused
        self._thread: Optional[threading.Thread] = None

        # Counters for diagnostics
        self._poll_count = 0
        self._error_count = 0

        # Failure streak, shared by the poll thread and force_refresh().
        # Guarded by self._lock. Times are time.monotonic() seconds.
        self._consecutive_errors = 0
        self._first_error_at = 0.0
        self._last_attempt_at = 0.0
        self._last_warning_at = 0.0
        self._last_error = ""

    # --- Lifecycle ---

    def start(self) -> None:
        """Start the background polling thread.

        Performs one synchronous read first so that the cache is
        populated by the time start() returns. If that read fails,
        the cache is left empty and the polling thread will retry.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.debug("StagePositionCache.start(): already running")
            return

        # Best-effort initial read so the first cached lookup
        # doesn't have to wait for the poll thread.
        try:
            self.force_refresh()
        except Exception as e:
            logger.warning(
                "StagePositionCache: initial read failed (%s); polling thread will retry",
                _first_line(e),
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="StagePositionCache",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "StagePositionCache: started polling at %.0f ms interval",
            self._poll_interval_s * 1000.0,
        )

    def stop(self, join_timeout_s: float = 1.0) -> None:
        """Stop the polling thread.

        Called when the stage is being replaced (e.g. config reload)
        so the old cache doesn't keep hitting hardware that has been
        rebuilt under it. Daemon thread will also die with the
        process if stop() is never called.
        """
        self._stop_event.set()
        # Make sure a paused thread can exit promptly.
        self._pause_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None
        logger.info(
            "StagePositionCache: stopped (polls=%d, errors=%d)",
            self._poll_count,
            self._error_count,
        )

    # --- Pause / resume ---

    def pause(self) -> None:
        """Temporarily suspend background polling.

        Use when a caller needs exclusive access to the serial bus
        for a short, latency-sensitive operation (e.g. a focus
        scan). force_refresh() still works while paused -- pause
        only stops the *background* thread.
        """
        self._pause_event.set()
        logger.debug("StagePositionCache: paused")

    def resume(self) -> None:
        """Resume background polling after pause()."""
        self._pause_event.clear()
        logger.debug("StagePositionCache: resumed")

    # --- Reads ---

    def get_cached_position(self, max_age_ms: Optional[float] = None) -> Position:
        """Return the most recent cached position.

        Args:
            max_age_ms: If set, the cache must be no older than this.
                If the cached snapshot is older (or missing entirely),
                a synchronous force_refresh() is issued and its result
                is returned. If None (default), whatever is in the
                cache is returned -- and only if the cache has never
                been populated does force_refresh() run.

        Returns:
            Position with x, y, z all set. Never returns None coords
            -- if the cache is empty and a refresh fails, the
            underlying exception propagates.

        Raises:
            StageUnavailableError: the stage counts as down and was
                tried less than _DOWN_RETRY_S ago; no read was made.
        """
        with self._lock:
            x, y, z, ts = self._x, self._y, self._z, self._timestamp
            down = self._consecutive_errors >= _SUSTAINED_ERROR_THRESHOLD
            since_attempt_s = time.monotonic() - self._last_attempt_at
            if down and since_attempt_s < _DOWN_RETRY_S:
                raise StageUnavailableError(self._down_message_locked())

        if ts == 0.0:
            # Cache empty -- force a read so callers never see None.
            return self.force_refresh()

        if max_age_ms is not None:
            age_ms = (time.monotonic() - ts) * 1000.0
            if age_ms > max_age_ms:
                # Cache is stale. Normally force a live read -- but that
                # acquires the stage lock, and a long stage-locked operation
                # (sweep autofocus holds it for ~13s while it steps through Z)
                # would block this read for the WHOLE operation. That stall
                # propagates to the client: the aux-socket position poll
                # blocks while holding the client's aux-socket lock, which in
                # turn blocks the ABORTAF cancel command -- making sweep
                # autofocus effectively uncancellable (the cancel only takes
                # effect once AF finishes on its own). Probe the stage lock
                # without blocking; if the stage is busy, return the stale
                # snapshot (harmless for a position display) so the aux socket
                # stays responsive and cancel reaches the server promptly.
                stage_lock = getattr(self._stage, "lock", None)
                if stage_lock is None or stage_lock.acquire(blocking=False):
                    try:
                        return self.force_refresh()
                    finally:
                        if stage_lock is not None:
                            stage_lock.release()
                logger.debug(
                    "StagePositionCache: stage busy (locked); returning stale "
                    "position (age=%.0f ms) instead of blocking on force_refresh",
                    age_ms,
                )
                return Position(x, y, z)

        return Position(x, y, z)

    def force_refresh(self) -> Position:
        """Live-query the stage, update the cache, and return.

        Use for callers that need the exact current position
        (focus-scan endpoints, per-tile acquisition metadata) and
        cannot tolerate up-to-100 ms staleness from the cache.
        Always reads the stage, even while it counts as down; the
        outcome updates the failure streak.
        """
        try:
            x, y, z = self._stage.get_xyz()
        except Exception as e:
            self._record_failure(e)
            raise
        with self._lock:
            self._x = x
            self._y = y
            self._z = z
            self._timestamp = time.monotonic()
        self._record_success()
        return Position(x, y, z)

    # --- Failure streak ---

    def _down_message_locked(self) -> str:
        """Why the stage counts as down. Caller holds self._lock."""
        failing_s = time.monotonic() - self._first_error_at
        return "Stage not responding (%d failed reads over %.0f s): %s" % (
            self._consecutive_errors,
            failing_s,
            self._last_error,
        )

    def _record_failure(self, exc: BaseException) -> None:
        """Count a failed read and log it: DEBUG, one WARNING, then reminders."""
        summary = _first_line(exc)
        now = time.monotonic()
        with self._lock:
            self._error_count += 1
            self._consecutive_errors += 1
            if self._consecutive_errors == 1:
                self._first_error_at = now
            self._last_attempt_at = now
            self._last_error = summary
            count = self._consecutive_errors
            failing_s = now - self._first_error_at
            if count < _SUSTAINED_ERROR_THRESHOLD:
                level = "debug"
            elif count == _SUSTAINED_ERROR_THRESHOLD:
                level = "first"
                self._last_warning_at = now
            elif now - self._last_warning_at >= _DOWN_REMINDER_S:
                level = "reminder"
                self._last_warning_at = now
            else:
                level = "quiet"

        if level == "first":
            logger.warning(
                "Stage not responding (%d failed reads): %s. Is the stage "
                "controller powered on and connected? Retrying every %.0f s; "
                "position requests get a hardware error until it answers.",
                count,
                summary,
                _DOWN_RETRY_S,
            )
        elif level == "reminder":
            logger.warning(
                "Stage still not responding (%d failed reads over %.0f min): %s",
                count,
                failing_s / 60.0,
                summary,
            )
        else:
            logger.debug("StagePositionCache: read failed (%d in a row): %s", count, summary)

    def _record_success(self) -> None:
        """End a failure streak; log the recovery if the stage counted as down."""
        now = time.monotonic()
        with self._lock:
            count = self._consecutive_errors
            failing_s = now - self._first_error_at
            self._consecutive_errors = 0
            self._last_attempt_at = now
            self._last_error = ""
        if count >= _SUSTAINED_ERROR_THRESHOLD:
            logger.info(
                "Stage responding again after %d failed reads over %.0f s",
                count,
                failing_s,
            )

    @property
    def is_stage_down(self) -> bool:
        """True while the stage has failed its last few reads in a row."""
        with self._lock:
            return self._consecutive_errors >= _SUSTAINED_ERROR_THRESHOLD

    # --- Background polling ---

    def _poll_loop(self) -> None:
        """Daemon thread body: poll get_xyz() in a loop.

        Exceptions are caught so a failing serial bus doesn't kill
        the polling thread; force_refresh() counts and logs them (see
        "Stage not responding" in the module docstring). The cache
        goes stale until reads succeed again.
        """
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Paused: sleep in short chunks so resume()/stop()
                # take effect promptly without burning CPU.
                if self._stop_event.wait(timeout=0.05):
                    return
                continue

            try:
                self.force_refresh()
                self._poll_count += 1
            except Exception:
                # Back off on errors: briefly for a transient blip, then
                # every _DOWN_RETRY_S once the stage counts as down, so
                # a switched-off controller is not asked twice a second.
                backoff_s = _DOWN_RETRY_S if self.is_stage_down else self._retry_backoff_s
                if self._stop_event.wait(timeout=backoff_s):
                    return
                continue

            if self._stop_event.wait(timeout=self._poll_interval_s):
                return

    # --- Diagnostics ---

    @property
    def poll_count(self) -> int:
        return self._poll_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
