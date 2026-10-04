"""XY motion profile, and detection of the stage origin moving underneath us.

Two jobs in one module, because they are two halves of one guard.

The PROFILE caps how hard the XY stage is driven. The Prior XYStage adapter on
PPM comes up at ``MaxSpeed=100`` and ``Acceleration=100`` -- every value its own
maximum -- and the one thing a multi-slide run does that a single-slide run
never does is a 28-85 mm slot-to-slot traverse at that full rate. Capping speed
and acceleration is the mitigation.

The CANARY falls out of the cap. Because the profile holds the stage BELOW the
controller's own defaults for the whole session, a readback that comes back at
the default is evidence that the controller restarted. That matters far more
than the speed does: a ProScan III has no absolute reference and nothing to home
to, so a controller that restarts resumes with position 0,0 wherever it happens
to be standing. Every coordinate afterwards is silently offset -- the holder
calibration, every per-slide alignment, and the stage ``TileConfiguration`` of
data already written to disk.

**Why a readback and a cross-time comparison, and not an arrival check.** After
the origin moves, the stage still goes exactly where it is told and still
reports the position it was told to go to. So comparing a commanded position
against the position reported immediately after that move ALWAYS agrees, and can
never see this failure; it is the wrong instrument, however reasonable it looks.
What does see it is a comparison across time: the position the controller
reports BEFORE the next move, against the position we last commanded. Nothing
moved in between, so those must agree. When they do not, either the stage moved
on its own or 0,0 moved under it, and :func:`describe_drift` reports the
disagreement rather than guessing which.

Neither check can correct anything. Both exist so that a run stops within
seconds instead of spending the next thirty hours imaging a slide label.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# How far the reported position may sit from the last commanded position before
# it counts as evidence rather than noise. A Prior step is 1 um and the
# controller reports whole steps, so single microns are measurement granularity;
# 100 um is over a third of a 20x field and cannot be anything but a real
# disagreement.
DEFAULT_DRIFT_TOLERANCE_UM = 100.0

# Only moves at least this long are worth a profile readback. Short moves are
# the per-tile steps -- tens of thousands of them in a run -- and a readback on
# each would add a serial round trip per tile for no new information, since a
# controller restart shows up on the first long move after it.
DEFAULT_VERIFY_AFTER_MOVE_UM = 5000.0


@dataclass(frozen=True)
class XyMotionProfile:
    """Device properties to hold the XY stage at, plus the two thresholds.

    ``properties`` maps an MM device property name to the value it must keep.
    It is deliberately an open mapping rather than named speed/acceleration
    fields: property names differ between adapters, and which properties are
    safe to touch is a per-rig question. On the Prior adapter ``MaxSpeed`` and
    ``Acceleration`` are both 1-100 percent scales where lower is gentler, so
    their direction is unambiguous. ``SCurve`` is NOT, which is exactly why it
    is left to the config rather than assumed here -- establish a property's
    direction by measurement before capping it.
    """

    properties: Tuple[Tuple[str, str], ...]
    verify_after_move_um: float = DEFAULT_VERIFY_AFTER_MOVE_UM
    drift_tolerance_um: float = DEFAULT_DRIFT_TOLERANCE_UM

    @property
    def is_empty(self) -> bool:
        return not self.properties


def resolve_profile(settings: Optional[Dict[str, Any]]) -> Optional[XyMotionProfile]:
    """Read ``stage.xy_motion`` from the microscope config.

    Returns None when the rig declares no profile, which leaves the stage at
    whatever the adapter came up with and disables both checks. That is the
    honest default: without a declared cap there is no value to compare a
    readback against, so the canary would have nothing to say.
    """
    stage_cfg = (settings or {}).get("stage") or {}
    cfg = stage_cfg.get("xy_motion") or {}
    if not cfg:
        return None

    raw = cfg.get("properties") or {}
    if not isinstance(raw, dict):
        logger.warning(
            "stage.xy_motion.properties is %s, expected a mapping of property name to value. "
            "Ignoring the XY motion profile.",
            type(raw).__name__,
        )
        return None

    properties = tuple((str(name), str(value)) for name, value in raw.items() if value is not None)
    if not properties:
        logger.warning("stage.xy_motion declares no properties; XY motion profile ignored.")
        return None

    return XyMotionProfile(
        properties=properties,
        verify_after_move_um=float(cfg.get("verify_after_move_um", DEFAULT_VERIFY_AFTER_MOVE_UM)),
        drift_tolerance_um=float(cfg.get("drift_tolerance_um", DEFAULT_DRIFT_TOLERANCE_UM)),
    )


def values_agree(reported: Any, expected: Any) -> bool:
    """Compare two MM property values the way the device means them.

    MM hands properties back as strings, and an adapter is free to answer "50"
    as "50.0000". Comparing those as text reports a change that did not happen,
    so numbers are compared numerically and anything else as text.
    """
    if reported is None:
        return False
    text = str(reported).strip()
    want = str(expected).strip()
    try:
        return float(text) == float(want)
    except (TypeError, ValueError):
        return text == want


def apply_profile(core: Any, device: str, profile: XyMotionProfile) -> Dict[str, str]:
    """Write every property in the profile and report what the device then held.

    Never raises: a rig whose adapter rejects one of these properties should run
    at the adapter's own speed with a warning in the log, not fail to acquire.
    The returned mapping is what a readback saw, so a caller can see which
    properties actually took.
    """
    applied: Dict[str, str] = {}
    for name, value in profile.properties:
        try:
            core.set_property(device, name, value)
        except Exception as e:
            logger.warning("Could not set %s.%s=%s: %s", device, name, value, e)
            continue
        read = _read_property(core, device, name)
        if read is None:
            continue
        applied[name] = read
        if not values_agree(read, value):
            logger.warning(
                "%s.%s was set to %s but reads back %s; the adapter may clamp this property.",
                device,
                name,
                value,
                read,
            )
    if applied:
        logger.info(
            "XY motion profile applied to %s: %s",
            device,
            ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
        )
    return applied


def describe_profile_mismatch(core: Any, device: str, profile: XyMotionProfile) -> Optional[str]:
    """Return a description of any property that no longer holds its value.

    A mismatch is the canary described in the module docstring: we hold these
    below the controller's defaults all session, so finding a default back in
    place says the controller restarted -- and a restarted controller has
    re-zeroed. Returns None when every property still agrees, or when none could
    be read (an unreadable property is not evidence of anything).
    """
    drifted = []
    for name, value in profile.properties:
        read = _read_property(core, device, name)
        if read is None:
            continue
        if not values_agree(read, value):
            drifted.append(f"{name}={read} (expected {value})")
    if not drifted:
        return None
    return (
        f"{device} motion profile no longer holds the values we set: {', '.join(drifted)}. "
        "The controller has most likely restarted, and a Prior controller that restarts "
        "resumes with position 0,0 wherever it is standing -- so the stage origin, the "
        "holder calibration and every slide alignment are no longer trustworthy."
    )


def describe_drift(
    reported: Optional[Tuple[float, float]],
    commanded: Optional[Tuple[float, float]],
    tolerance_um: float,
) -> Optional[str]:
    """Compare where the controller says it is against where we last sent it.

    Called before the next move, with the stage stationary, so the two must
    agree: nothing was commanded in between. Returns None when they agree within
    ``tolerance_um``, or when either is unknown (the first move of a session has
    nothing to compare against).
    """
    if reported is None or commanded is None:
        return None
    dx = reported[0] - commanded[0]
    dy = reported[1] - commanded[1]
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= tolerance_um:
        return None
    return (
        f"Stage reports ({reported[0]:.1f}, {reported[1]:.1f}) um but the last commanded "
        f"position was ({commanded[0]:.1f}, {commanded[1]:.1f}) um -- off by "
        f"({dx:.1f}, {dy:.1f}) um, {distance:.1f} um total, with nothing commanded in "
        "between. Either the stage moved on its own or its 0,0 moved underneath it; "
        "either way the coordinates in use no longer describe the sample."
    )


def _read_property(core: Any, device: str, name: str) -> Optional[str]:
    try:
        return str(core.get_property(device, name))
    except Exception as e:
        logger.debug("Could not read %s.%s: %s", device, name, e)
        return None
