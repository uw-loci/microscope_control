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

    def _wait_z_via_busy(self, timeout_ms: float = 10000.0,
                         poll_interval_ms: float = 3.0) -> None:
        """Wait for Z stage to finish moving via tight device_busy polling.

        Much faster than core.wait_for_device() on hardware where the MM
        core polls at 50-100 ms granularity internally (e.g. Prior
        ProScan via serial). device_busy hits the same serial line but
        *we* control the poll rate, dropping the 'stage is done but MM
        hasn't noticed yet' latency from ~150 ms to ~6 ms per move.

        Measured on PPM with PROBEZ: 20 um blocking move went from
        ~240 ms (wait_for_device) to ~80 ms (busy polling) -- ~3-4x
        speedup with no behavioral change.

        Correctness safeguards:
        - Requires 2 consecutive not-busy reads to confirm arrival
          (cheap defense against a race where MM reports not-busy for
          one cycle while the serial adapter is mid-command).
        - Falls back to core.wait_for_device() on any exception (keeps
          legacy behavior for hardware where device_busy is flaky).
        - Hard timeout of 10 s; also falls back to wait_for_device on
          timeout so we never return with the stage still moving.

        Args:
            timeout_ms: Hard cap on busy-poll duration before fallback.
                Should be larger than any reasonable Z move time.
            poll_interval_ms: Time between device_busy calls.
        """
        focus_dev = self._core.get_focus_device()
        try:
            deadline = time.perf_counter() + (timeout_ms / 1000.0)
            consecutive_clear = 0
            sleep_s = poll_interval_ms / 1000.0
            while time.perf_counter() < deadline:
                try:
                    busy = self._core.device_busy(focus_dev)
                except Exception as e:
                    # device_busy not supported or errored -- fall
                    # through to the safe path below.
                    logger.debug("device_busy(%s) failed, falling back: %s",
                                 focus_dev, e)
                    break
                if not busy:
                    consecutive_clear += 1
                    if consecutive_clear >= 2:
                        return
                else:
                    consecutive_clear = 0
                time.sleep(sleep_s)
            else:
                logger.warning(
                    "wait_z busy-poll timed out after %.0f ms on '%s'; "
                    "falling back to wait_for_device",
                    timeout_ms, focus_dev,
                )
        except Exception as e:
            logger.debug("wait_z busy-poll errored, falling back: %s", e)

        # Safe fallback: whatever MM core's native blocking wait does.
        try:
            self._core.wait_for_device(focus_dev)
        except Exception as e:
            logger.warning("wait_for_device fallback failed: %s", e)

    # --- XY operations ---

    def move_xy(self, x: float, y: float) -> None:
        with self._lock:
            self._core.set_xy_position(x, y)
            self._core.wait_for_device(self._core.get_xy_stage_device())
            logger.debug("move_xy: X=%.2f, Y=%.2f", x, y)

    def move_xy_no_wait(self, x: float, y: float) -> None:
        with self._lock:
            self._core.set_xy_position(x, y)
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
            self._wait_z_via_busy()
            logger.debug("move_z: Z=%.2f", z)

    def move_z_no_wait(self, z: float) -> None:
        with self._lock:
            self._ensure_z_device()
            self._core.set_position(z)
            logger.debug("move_z_no_wait: target Z=%.2f", z)

    def get_z(self) -> float:
        with self._lock:
            return self._core.get_position()

    def wait_z(self) -> None:
        with self._lock:
            self._wait_z_via_busy()

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
