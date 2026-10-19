"""Stage abstraction for microscope XY and Z positioning.

Provides a Stage ABC and a Pycromanager-based concrete implementation.
Any microscope control backend that can move XY and Z axes can implement
the Stage interface to work with the QPSC acquisition system.
"""

import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from microscope_control.hardware import xy_motion

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract base class for microscope XYZ stages.

    Provides a unified interface for positioning hardware. Implementations
    handle the specifics of communicating with the physical stage controller
    (e.g. via Micro-Manager Core, direct serial, etc.).

    Coordinate conventions:
    - X, Y: lateral stage axes in micrometers
    - Z: focus axis in micrometers
    - Positive directions are hardware-dependent (configured per microscope)
    """

    # --- XY operations ---

    @abstractmethod
    def move_xy(self, x: float, y: float) -> None:
        """Move XY stage to position and wait for arrival.

        Args:
            x: Target X position in micrometers
            y: Target Y position in micrometers
        """
        ...

    @abstractmethod
    def move_xy_no_wait(self, x: float, y: float) -> None:
        """Issue XY move without waiting for completion.

        Call wait_xy() before any operation that depends on the
        stage being at the target position.
        """
        ...

    @abstractmethod
    def get_xy(self) -> Tuple[float, float]:
        """Get current XY position.

        Returns:
            Tuple of (x, y) in micrometers
        """
        ...

    @abstractmethod
    def wait_xy(self) -> None:
        """Block until XY stage reaches its target position."""
        ...

    # --- Z operations ---

    @abstractmethod
    def move_z(self, z: float) -> None:
        """Move Z (focus) stage to position and wait for arrival.

        Args:
            z: Target Z position in micrometers
        """
        ...

    @abstractmethod
    def move_z_no_wait(self, z: float) -> None:
        """Issue Z move without waiting for completion.

        Call wait_z() before any operation that depends on focus.
        """
        ...

    @abstractmethod
    def get_z(self) -> float:
        """Get current Z position in micrometers."""
        ...

    @abstractmethod
    def wait_z(self) -> None:
        """Block until Z stage reaches its target position."""
        ...

    # --- Combined operations ---

    def get_xyz(self) -> Tuple[float, float, float]:
        """Get current XYZ position.

        Default: calls get_xy() and get_z() separately.

        Returns:
            Tuple of (x, y, z) in micrometers
        """
        x, y = self.get_xy()
        z = self.get_z()
        return x, y, z

    def wait_all(self) -> None:
        """Block until both XY and Z stages have arrived.

        Default: calls wait_xy() then wait_z().
        """
        self.wait_xy()
        self.wait_z()

    # --- Optional secondary Z axis (condenser / F-stage) ---

    def has_secondary_z(self) -> bool:
        """Whether this stage has a secondary Z axis (e.g. condenser)."""
        return False

    def move_secondary_z(self, z: float) -> None:
        """Move the secondary Z axis (condenser / F-stage).

        Args:
            z: Target position in micrometers

        Raises:
            RuntimeError: If no secondary Z axis is configured
        """
        raise RuntimeError("No secondary Z axis configured")

    def get_secondary_z(self) -> float:
        """Get current secondary Z position."""
        raise RuntimeError("No secondary Z axis configured")

    # --- Optional objective turret ---

    def has_turret(self) -> bool:
        """Whether this stage has an objective turret."""
        return False

    def set_turret_position(self, label: str) -> None:
        """Switch objective turret to the named position.

        Args:
            label: Position label (e.g. 'Position-1' for 20x)
        """
        raise RuntimeError("No objective turret configured")

    def get_turret_position(self) -> str:
        """Get current turret position label."""
        raise RuntimeError("No objective turret configured")


class PycromanagerStage(Stage):
    """Stage implementation using Pycromanager/Micro-Manager Core.

    Wraps MM Core's XY stage and focus device APIs. Works with any
    stage hardware that Micro-Manager supports (Prior, ASI, Sutter,
    Thorlabs, etc.) -- the MM device adapter handles the specifics.
    """

    def __init__(self, core, settings: Dict[str, Any]):
        """
        Args:
            core: Pycromanager Core object
            settings: Microscope configuration dict (contains stage.z_stage,
                stage.limits, etc.)
        """
        self._core = core
        self._settings = settings
        # Reentrant lock guarding every stage operation. Serializes concurrent
        # access from multiple client threads (position pollers, Z scroll,
        # acquisition workflow, Smooth Focus, background cache) so that:
        #   - no two move_z() calls can retarget the Prior stage mid-wait,
        #     which was causing wait_z busy-poll to hang for the full 10 s
        #     timeout during rapid Z scroll (2026-04-15 incident),
        #   - get_xyz() reads don't interleave mid-move and return mixed
        #     pre/post-move coordinates,
        #   - secondary Z (F-stage) operations that temporarily rebind the
        #     MM focus device can't race with a primary Z move.
        # Reentrant so that composed operations (move_to_position, which
        # calls move_z_no_wait + wait_z) can hold the lock at the top level
        # and have the inner calls re-enter it cheaply.
        self._lock = threading.RLock()

        # XY motion profile (speed/acceleration cap) and the origin-change
        # detection that rides on it. See hardware/xy_motion.py for why a
        # cross-time comparison is the only check that can see this failure.
        self._xy_profile = xy_motion.resolve_profile(settings)
        self._xy_profile_applied = False
        self._last_commanded_xy: Optional[Tuple[float, float]] = None
        self._xy_frame_suspect: Optional[str] = None
        self._xy_frame_suspect_since: Optional[float] = None
        # The drift check is disarmed by default because a physical joystick
        # nudge between runs legitimately moves the stage away from the last
        # commanded position, and flagging that would be a false alarm. The
        # acquisition workflow arms it for the duration of an acquisition --
        # exactly the window where no one should be touching the stage and
        # where an undetected origin change costs the whole run.
        self._xy_frame_check_armed = False

        logger.info("Initialized PycromanagerStage")

    @property
    def core(self):
        """Access to the underlying MM Core (for autofocus and other
        low-level operations that need direct Z control)."""
        return self._core

    @property
    def lock(self) -> threading.RLock:
        """The stage's reentrant serialization lock.

        Composed operations like ``PycromanagerHardware.move_to_position``
        should acquire this lock at the top level so the non-blocking set
        and the subsequent wait run as an atomic unit. Individual stage
        method calls also acquire this lock internally; the RLock allows
        those inner acquires to succeed without deadlock.
        """
        return self._lock

    def _ensure_z_device(self) -> None:
        """Ensure the correct Z focus device is active.

        Some microscopes (e.g. CAMM) have multiple Z axes (Z stage + F
        piezo). This checks the config and switches if needed.
        """
        stage_config = self._settings.get("stage", {})
        z_stage_device = stage_config.get("z_stage", None)
        if z_stage_device and self._core.get_focus_device() != z_stage_device:
            self._core.set_focus_device(z_stage_device)

    def _wait_z_via_busy(
        self,
        timeout_ms: float = 10000.0,
        poll_interval_ms: float = 3.0,
        target_z: Optional[float] = None,
        tolerance_um: float = 0.5,
    ) -> None:
        """Wait for Z stage to finish moving via tight device_busy polling.

        Much faster than core.wait_for_device() on hardware where the MM
        core polls at 50-100 ms granularity internally (e.g. Prior
        ProScan via serial). device_busy hits the same serial line but
        *we* control the poll rate.

        Correctness safeguards:
        - Requires 5 consecutive not-busy reads to confirm arrival
          (was 2 -- bumped 2026-05-05 after the user observed AF feeling
          "incredibly fast" through the eyepiece, suggesting the
          firmware was reporting not-busy before the stage settled).
        - When ``target_z`` is provided, after the busy clears we
          query the actual position and confirm it's within
          ``tolerance_um`` of the target. If not, fall back to
          ``core.wait_for_device`` and log a WARNING -- this catches
          the case where ``device_busy`` reports clear before the
          stage has actually arrived at the commanded Z. The warning
          also gives operators a visible signal that something is
          wrong with stage control rather than silent bad data.
        - Falls back to core.wait_for_device() on any exception.
        - Optional ``stage.z_settle_ms`` config knob adds a fixed
          post-arrival sleep for piezos that need mechanical settling
          beyond the electrical "not busy" report.

        Args:
            timeout_ms: Hard cap on busy-poll duration before fallback.
            poll_interval_ms: Time between device_busy calls.
            target_z: Optional commanded Z target. When provided,
                arrival is verified by reading actual position.
            tolerance_um: Max allowed distance from ``target_z`` to
                consider arrival successful (default 0.5 um -- tight
                enough to catch real misses, loose enough to absorb
                encoder noise on stages without sub-100-nm precision).
        """
        focus_dev = self._core.get_focus_device()
        clear_required = 5
        try:
            deadline = time.perf_counter() + (timeout_ms / 1000.0)
            consecutive_clear = 0
            sleep_s = poll_interval_ms / 1000.0
            while time.perf_counter() < deadline:
                try:
                    busy = self._core.device_busy(focus_dev)
                except Exception as e:
                    logger.debug("device_busy(%s) failed, falling back: %s", focus_dev, e)
                    break
                if not busy:
                    consecutive_clear += 1
                    if consecutive_clear >= clear_required:
                        break
                else:
                    consecutive_clear = 0
                time.sleep(sleep_s)
            else:
                logger.warning(
                    "wait_z busy-poll timed out after %.0f ms on '%s'; "
                    "falling back to wait_for_device",
                    timeout_ms,
                    focus_dev,
                )
                try:
                    self._core.wait_for_device(focus_dev)
                except Exception as e:
                    logger.warning("wait_for_device fallback failed: %s", e)
        except Exception as e:
            logger.debug("wait_z busy-poll errored, falling back: %s", e)
            try:
                self._core.wait_for_device(focus_dev)
            except Exception as e2:
                logger.warning("wait_for_device fallback failed: %s", e2)

        # Optional mechanical settle delay -- piezos in particular can
        # report electrical "not busy" before the trajectory has
        # mechanically settled. Configurable per-microscope so this is
        # zero-overhead on stages that don't need it.
        stage_cfg = self._settings.get("stage", {}) if self._settings else {}
        settle_ms = float(stage_cfg.get("z_settle_ms", 0.0))
        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)

        # Arrival verification when a target was supplied. The user's
        # 2026-05-05 hypothesis: "this really feels like a break in the
        # stage control that we aren't catching." If the busy-poll
        # cleared but actual Z is far from target, that's exactly the
        # break the warning should surface.
        if target_z is not None:
            try:
                actual_z = self._core.get_position()
                err = abs(actual_z - target_z)
                if err > tolerance_um:
                    logger.warning(
                        "wait_z arrival check FAILED: target=%.3f um, "
                        "actual=%.3f um, error=%.3f um (tolerance=%.3f um) "
                        "on '%s'. Stage reported not-busy but did not "
                        "arrive. Falling back to wait_for_device.",
                        target_z,
                        actual_z,
                        err,
                        tolerance_um,
                        focus_dev,
                    )
                    try:
                        self._core.wait_for_device(focus_dev)
                    except Exception as e:
                        logger.warning("wait_for_device fallback failed: %s", e)
                    # Re-check after the safer wait
                    try:
                        actual_z2 = self._core.get_position()
                        err2 = abs(actual_z2 - target_z)
                        if err2 > tolerance_um:
                            logger.error(
                                "wait_z STILL off-target after fallback: "
                                "target=%.3f, actual=%.3f, err=%.3f um. "
                                "Stage controller may be in a bad state.",
                                target_z,
                                actual_z2,
                                err2,
                            )
                    except Exception as e:
                        logger.debug("post-fallback position read failed: %s", e)
                else:
                    logger.debug(
                        "wait_z arrival OK: target=%.3f, actual=%.3f, " "err=%.3f um",
                        target_z,
                        actual_z,
                        err,
                    )
            except Exception as e:
                logger.debug("wait_z arrival check skipped (read failed): %s", e)

    # --- XY motion profile and origin-change detection ---

    @property
    def xy_frame_suspect(self) -> Optional[str]:
        """Why the XY coordinate frame stopped being trustworthy, or None.

        Sticky. An origin change is not something the controller recovers from
        on its own, so this stays set until the frame is re-established at the
        calibration fiducial and the server restarted. Callers treat a non-None
        value as "stop", not "retry": every coordinate they hold describes a
        place the sample no longer is.
        """
        return self._xy_frame_suspect

    @property
    def xy_frame_suspect_since(self) -> Optional[float]:
        """When the frame was first called suspect (``time.time()``), or None.

        The flag is sticky, so a run started an hour later fails with the same numbers; this
        is what lets the message say the fault is old rather than read as a new one.
        """
        return self._xy_frame_suspect_since

    @property
    def frame_check_armed(self) -> bool:
        """Whether the commanded-vs-reported drift check is currently armed."""
        return self._xy_frame_check_armed

    def set_frame_check_armed(self, armed: bool) -> None:
        """Arm or disarm the commanded-vs-reported drift check.

        Armed for the length of an acquisition, when nothing should be moving
        the stage except us. Disarmed otherwise, so a joystick nudge during
        setup is not reported as an origin change.

        Arming takes the stage's current position as the one we last commanded.
        Moves made by hand while disarmed (the joystick, Micro-Manager's own
        controls) never pass through here, so the position we sent last may be
        far from where the stage now stands, and comparing against it would
        report the operator's move as a frame change. 2026-10-19: a liver run
        stopped 3.4 mm "off" this way. A controller restart while disarmed is
        still caught: arming reads the motion profile back, and a restarted
        controller has its own defaults in place.
        """
        with self._lock:
            if armed:
                self._rebaseline_frame_check()
            self._xy_frame_check_armed = bool(armed)

    def _rebaseline_frame_check(self) -> None:
        if self._xy_profile is None:
            return
        if self._xy_busy():
            # A move of ours is still running; its target is already the
            # commanded position, and the next move skips the comparison anyway.
            return
        try:
            self._last_commanded_xy = (self._core.get_x_position(), self._core.get_y_position())
        except Exception as e:
            logger.debug("Could not read the XY position when arming the frame check: %s", e)
            self._last_commanded_xy = None
        if not self._xy_profile_applied:
            # Nothing written yet, so the controller's defaults are expected.
            return
        device = self._xy_device()
        if device is None:
            return
        mismatch = xy_motion.describe_profile_mismatch(self._core, device, self._xy_profile)
        if mismatch:
            self._note_frame_suspect(mismatch)

    def ensure_xy_motion_profile(self, force: bool = False) -> None:
        """Apply the configured speed/acceleration cap, once per session."""
        if self._xy_profile is None or (self._xy_profile_applied and not force):
            return
        device = self._xy_device()
        if device is None:
            return
        with self._lock:
            xy_motion.apply_profile(self._core, device, self._xy_profile)
        self._xy_profile_applied = True

    def _xy_device(self) -> Optional[str]:
        try:
            return str(self._core.get_xy_stage_device())
        except Exception as e:
            logger.debug("Could not resolve the XY stage device: %s", e)
            return None

    def _xy_busy(self) -> bool:
        device = self._xy_device()
        if device is None:
            return False
        try:
            return bool(self._core.device_busy(device))
        except Exception:
            return False

    def _note_frame_suspect(self, message: str) -> None:
        if self._xy_frame_suspect:
            return
        self._xy_frame_suspect = message
        self._xy_frame_suspect_since = time.time()
        logger.error("STAGE FRAME SUSPECT: %s", message)

    def _before_xy_move(self, x: float, y: float) -> float:
        """Verify the frame still holds; return the distance about to be moved.

        Returns 0.0 whenever the distance cannot be established honestly, which
        costs only a skipped profile readback on that move.
        """
        self.ensure_xy_motion_profile()
        profile = self._xy_profile
        if profile is None:
            return 0.0
        if self._xy_busy():
            # A previous non-blocking move is still running, so a position read
            # is a point along the way rather than an arrival. Nothing to
            # compare, and comparing anyway would report motion as drift.
            return 0.0
        try:
            reported = (self._core.get_x_position(), self._core.get_y_position())
        except Exception as e:
            logger.debug("Could not read the XY position before a move: %s", e)
            return 0.0
        if self._xy_frame_check_armed:
            drift = xy_motion.describe_drift(
                reported, self._last_commanded_xy, profile.drift_tolerance_um
            )
            if drift:
                self._note_frame_suspect(drift)
        dx = x - reported[0]
        dy = y - reported[1]
        return (dx * dx + dy * dy) ** 0.5

    def _after_xy_move(self, x: float, y: float, distance_um: float) -> None:
        self._last_commanded_xy = (x, y)
        profile = self._xy_profile
        if profile is None or distance_um < profile.verify_after_move_um:
            return
        device = self._xy_device()
        if device is None:
            return
        mismatch = xy_motion.describe_profile_mismatch(self._core, device, profile)
        if mismatch:
            self._note_frame_suspect(mismatch)

    # --- XY operations ---

    def move_xy(self, x: float, y: float) -> None:
        with self._lock:
            distance = self._before_xy_move(x, y)
            self._core.set_xy_position(x, y)
            self._core.wait_for_device(self._core.get_xy_stage_device())
            self._after_xy_move(x, y, distance)
            logger.debug("move_xy: X=%.2f, Y=%.2f", x, y)

    def move_xy_no_wait(self, x: float, y: float) -> None:
        with self._lock:
            distance = self._before_xy_move(x, y)
            self._core.set_xy_position(x, y)
            self._after_xy_move(x, y, distance)
            logger.debug("move_xy_no_wait: target X=%.2f, Y=%.2f", x, y)

    def get_xy(self) -> Tuple[float, float]:
        with self._lock:
            return self._core.get_x_position(), self._core.get_y_position()

    def wait_xy(self) -> None:
        with self._lock:
            self._core.wait_for_device(self._core.get_xy_stage_device())

    # --- Z operations ---

    def move_z(self, z: float) -> None:
        with self._lock:
            self._ensure_z_device()
            self._core.set_position(z)
            self._wait_z_via_busy(target_z=z)
            logger.debug("move_z: Z=%.2f", z)

    def move_z_no_wait(self, z: float) -> None:
        with self._lock:
            self._ensure_z_device()
            self._core.set_position(z)
            # Cache the most recent target so wait_z() can verify
            # arrival even when the caller used the no-wait + wait split
            # rather than the all-in-one move_z. The cache is best-effort
            # -- if multiple targets are queued, only the last one is
            # checked, which matches MM's own behavior of overriding
            # in-flight commands.
            self._last_z_target = float(z)
            logger.debug("move_z_no_wait: target Z=%.2f", z)

    def get_z(self) -> float:
        with self._lock:
            return self._core.get_position()

    def wait_z(self) -> None:
        with self._lock:
            target = getattr(self, "_last_z_target", None)
            self._wait_z_via_busy(target_z=target)

    # --- Combined (optimized) ---

    def get_xyz(self) -> Tuple[float, float, float]:
        """Get XYZ in a single call (no redundant device queries)."""
        with self._lock:
            return (
                self._core.get_x_position(),
                self._core.get_y_position(),
                self._core.get_position(),
            )

    # --- Secondary Z (condenser / F-stage) ---

    def has_secondary_z(self) -> bool:
        """True if an F-stage / condenser device is configured."""
        stage_config = self._settings.get("stage", {})
        return bool(stage_config.get("f_stage"))

    def _ensure_secondary_z_device(self) -> None:
        """Temporarily switch focus device to the F-stage."""
        stage_config = self._settings.get("stage", {})
        f_device = stage_config.get("f_stage")
        if not f_device:
            raise RuntimeError("No secondary Z (f_stage) configured")
        if self._core.get_focus_device() != f_device:
            self._core.set_focus_device(f_device)

    def move_secondary_z(self, z: float) -> None:
        """Move condenser / F-stage to position.

        Temporarily switches the MM focus device to the F-stage,
        issues the move, waits, then switches back to the primary Z.
        """
        with self._lock:
            self._ensure_secondary_z_device()
            self._core.set_position(z)
            self._core.wait_for_device(self._core.get_focus_device())
            logger.debug("move_secondary_z (F-stage): %.2f", z)
            # Switch back to primary Z
            self._ensure_z_device()

    def get_secondary_z(self) -> float:
        """Get current condenser / F-stage position."""
        with self._lock:
            self._ensure_secondary_z_device()
            pos = self._core.get_position()
            # Switch back to primary Z
            self._ensure_z_device()
            return pos

    # --- Objective turret ---

    def has_turret(self) -> bool:
        """True if an objective turret is configured."""
        turret_config = self._settings.get("obj_slider")
        return bool(turret_config)

    def set_turret_position(self, label: str) -> None:
        """Switch objective turret to the named position.

        Args:
            label: Position label (e.g. 'Position-1', 'Position-2')
        """
        turret = self._settings.get("obj_slider")
        if not turret:
            raise RuntimeError("No objective turret (obj_slider) configured")
        with self._lock:
            # obj_slider is [device_name, property_name] in config
            self._core.set_property(turret[0], turret[1], label)
            self._core.wait_for_device(turret[0])
            logger.info("Turret set to %s", label)

    def get_turret_position(self) -> str:
        """Get current turret position label."""
        turret = self._settings.get("obj_slider")
        if not turret:
            raise RuntimeError("No objective turret configured")
        with self._lock:
            return self._core.get_property(turret[0], turret[1])
