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
"""

import logging
import threading
import time
from typing import Optional

from microscope_control.hardware.base import Position
from microscope_control.hardware.stage import Stage

logger = logging.getLogger(__name__)

# Number of consecutive poll failures before we treat it as a sustained
# problem worth a WARNING. Below this, transient serial-bus blips are
# logged at DEBUG so they don't dominate long-running session logs.
_SUSTAINED_ERROR_THRESHOLD = 3


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
                error before trying again. Kept short so transient
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
                "StagePositionCache: initial read failed (%s); " "polling thread will retry",
                e,
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
        """
        with self._lock:
            x, y, z, ts = self._x, self._y, self._z, self._timestamp

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
        """
        x, y, z = self._stage.get_xyz()
        with self._lock:
            self._x = x
            self._y = y
            self._z = z
            self._timestamp = time.monotonic()
        return Position(x, y, z)

    # --- Background polling ---

    def _poll_loop(self) -> None:
        """Daemon thread body: poll get_xyz() in a loop.

        Exceptions are caught and logged at warning (first few) /
        debug (steady state) so a failing serial bus doesn't kill
        the polling thread. The cache simply goes stale until
        reads start succeeding again.
        """
        consecutive_errors = 0
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Paused: sleep in short chunks so resume()/stop()
                # take effect promptly without burning CPU.
                if self._stop_event.wait(timeout=0.05):
                    return
                continue

            try:
                x, y, z = self._stage.get_xyz()
                with self._lock:
                    self._x = x
                    self._y = y
                    self._z = z
                    self._timestamp = time.monotonic()
                self._poll_count += 1
                # Only announce recovery if we previously logged a WARNING.
                # Singleton blips (1-2 errors) don't get a WARN, so they
                # don't need a paired recovery INFO either -- that pair
                # was the bulk of the weekend log spam.
                if consecutive_errors >= _SUSTAINED_ERROR_THRESHOLD:
                    logger.info(
                        "StagePositionCache: polling recovered after %d consecutive errors",
                        consecutive_errors,
                    )
                consecutive_errors = 0
            except Exception as e:
                self._error_count += 1
                consecutive_errors += 1
                # Java exceptions from mmcorej (Pycromanager ZMQ bridge)
                # render their full stack trace via __str__; strip to the
                # first line so the log isn't dominated by 13-line stacks
                # for transient serial-bus blips.
                summary = str(e).splitlines()[0] if str(e) else type(e).__name__
                # Quiet for short transients (~500 ms cache staleness is
                # harmless), loud once errors persist enough to suggest a
                # real wire / device problem worth investigating.
                if consecutive_errors < _SUSTAINED_ERROR_THRESHOLD:
                    logger.debug(
                        "StagePositionCache: poll failed (%d in a row): %s",
                        consecutive_errors,
                        summary,
                    )
                else:
                    logger.warning(
                        "StagePositionCache: poll failed (%d in a row): %s",
                        consecutive_errors,
                        summary,
                    )
                # Back off slightly on errors so we don't hammer a
                # broken bus at full poll rate.
                if self._stop_event.wait(timeout=self._retry_backoff_s):
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
