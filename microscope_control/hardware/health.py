"""Which part of the microscope is not answering, from failures that already happen.

The server reports a part as failing only from calls it makes anyway: camera snaps, and
Micro-Manager bridge timeouts seen on those snaps or on stage reads. It never probes a device
to find out: probing costs serial traffic and multi-second timeouts, and most Micro-Manager
property reads return cached values without touching the device. The stage's own state comes
from the stage position cache, which already counts failed reads (see stage_cache.py).

A part is "failing" from its first failure after a success until its next success.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

CAMERA = "camera"
MICROMANAGER = "micromanager"
STAGE = "stage"


@dataclass
class PartHealth:
    """One part's state. ``since`` is wall-clock seconds (time.time()) of the first failure."""

    ok: bool = True
    since: Optional[float] = None
    message: str = ""
    failures: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "since": self.since,
            "message": self.message,
            "failures": self.failures,
        }


def is_bridge_timeout(exc: BaseException) -> bool:
    """True when ``exc`` looks like the Micro-Manager (pycromanager / ZMQ) bridge timing out."""
    text = "%s %s" % (type(exc).__name__, exc)
    lowered = text.lower()
    return (
        "timed out" in lowered
        or "timeout" in lowered
        or "zmq" in lowered
        or type(exc).__name__ == "Again"
    )


@dataclass
class HardwareHealth:
    """Thread-safe per-part failure tracker."""

    _parts: Dict[str, PartHealth] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def failed(self, part: str, error: object) -> None:
        """Record a failure of ``part``; ``error`` is an exception or a message."""
        message = str(error).splitlines()[0] if str(error) else type(error).__name__
        with self._lock:
            h = self._parts.setdefault(part, PartHealth())
            if h.ok:
                h.ok = False
                h.since = time.time()
                h.failures = 0
            h.failures += 1
            h.message = message

    def succeeded(self, part: str) -> None:
        """Record that ``part`` answered."""
        with self._lock:
            self._parts[part] = PartHealth()

    def snapshot(self, stage_cache=None) -> dict:
        """Every part's state as plain dicts; the stage from ``stage_cache`` when given."""
        with self._lock:
            out = {name: h.as_dict() for name, h in self._parts.items()}
        for name in (CAMERA, MICROMANAGER):
            out.setdefault(name, PartHealth().as_dict())
        out[STAGE] = _stage_state(stage_cache)
        return out


def _stage_state(stage_cache) -> dict:
    if stage_cache is None:
        return {"ok": True, "since": None, "message": "", "failures": 0, "known": False}
    status = stage_cache.down_status()
    status["known"] = True
    return status


#: The one tracker the server and the hardware classes share.
health = HardwareHealth()
