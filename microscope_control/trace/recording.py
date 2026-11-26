"""Recording proxies for the Micro-Manager ``Core`` and ``Studio`` objects.

A :class:`RecordingProxy` stands in front of the live object. Every method
call goes straight through and one JSON line lands in ``hardware.jsonl``:

``{"kind":"call","obj":"core","method":"set_xy_position","args":[1000.0,2000.0],
"kwargs":{},"ret":null,"dt_ms":12.3,"thread":"MainThread","seq":17,"t":4.2,
"stage":3}``

An exception is recorded as ``"error":{"type":...,"message":...}`` and
re-raised unchanged. Images are stored as a digest plus shape, dtype and
statistics unless :data:`TRACE_FRAMES_ENV` asks for a preview or full copy.
The proxy adds one dictionary lookup, one time read and one JSON line per
call; wrapped methods are cached per name after the first call.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Iterable, Optional

from microscope_control.trace.values import FRAME_MODES, encode_value
from microscope_control.trace.writer import TRACE_FRAMES_ENV, HARDWARE_STREAM, TraceWriter

# Methods whose return value is a frame; they get a stable file name in
# frame_mode 'full' and are the only ones that can be large.
FRAME_METHODS = frozenset(
    {
        "get_image",
        "get_tagged_image",
        "get_last_image",
        "get_last_tagged_image",
        "pop_next_image",
        "pop_next_tagged_image",
        "get_n_before_last_tagged_image",
    }
)


def frame_mode_from_env(default: str = "hash") -> str:
    """The frame mode named by :data:`TRACE_FRAMES_ENV`, or ``default``."""
    mode = os.environ.get(TRACE_FRAMES_ENV, "").strip().lower() or default
    if mode not in FRAME_MODES:
        raise ValueError(f"{TRACE_FRAMES_ENV} must be one of {FRAME_MODES}, not {mode!r}")
    return mode


class RecordingProxy:
    """Records every method call on ``target`` to ``writer`` and passes it through.

    ``nested`` maps a method name to the ``obj`` label for a proxy wrapped
    around that method's result (``{"live": "studio.live"}`` records the calls
    made on ``studio.live()``); such accessor calls are not themselves recorded.
    """

    def __init__(
        self,
        target: Any,
        writer: TraceWriter,
        obj: str = "core",
        frame_mode: str = "hash",
        nested: Optional[Dict[str, str]] = None,
        skip: Iterable[str] = (),
    ):
        if frame_mode not in FRAME_MODES:
            raise ValueError(f"frame_mode must be one of {FRAME_MODES}, not {frame_mode!r}")
        self._target = target
        self._writer = writer
        self._obj = obj
        self._frame_mode = frame_mode
        self._nested = dict(nested or {})
        self._skip = frozenset(skip)
        self._wrapped: Dict[str, Any] = {}
        self._nested_proxies: Dict[str, "RecordingProxy"] = {}

    @property
    def recording_target(self) -> Any:
        """The live object behind the proxy."""
        return self._target

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not on the proxy itself.
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        wrapped = self._wrapped.get(name)
        if wrapped is None:
            wrapped = self._wrap(name, attr)
            self._wrapped[name] = wrapped
        return wrapped

    def _wrap(self, name: str, method: Any):
        if name in self._nested:
            label = self._nested[name]

            def accessor(*args: Any, **kwargs: Any) -> Any:
                result = method(*args, **kwargs)
                proxy = self._nested_proxies.get(label)
                if proxy is None or proxy.recording_target is not result:
                    proxy = RecordingProxy(result, self._writer, label, self._frame_mode)
                    self._nested_proxies[label] = proxy
                return proxy

            return accessor
        if name in self._skip:
            return method

        writer = self._writer
        obj = self._obj
        frame_mode = self._frame_mode
        frames_dir = writer.frames_dir

        def recorded(*args: Any, **kwargs: Any) -> Any:
            seq = writer.next_seq()
            t_start = writer.now()
            record: Dict[str, Any] = {
                "kind": "call",
                "obj": obj,
                "method": name,
                "args": [encode_value(a, "hash") for a in args],
                "kwargs": {k: encode_value(v, "hash") for k, v in kwargs.items()},
                "seq": seq,
                "t": round(t_start, 6),
                "thread": threading.current_thread().name,
            }
            try:
                result = method(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
                record["error"] = encode_value(exc)["__error__"]
                record["dt_ms"] = round((writer.now() - t_start) * 1000.0, 3)
                writer.append(HARDWARE_STREAM, record)
                raise
            record["dt_ms"] = round((writer.now() - t_start) * 1000.0, 3)
            frame_name = f"{seq:08d}_{name}" if name in FRAME_METHODS else None
            mode = frame_mode if name in FRAME_METHODS or frame_mode != "full" else "hash"
            record["ret"] = encode_value(result, mode, frames_dir, frame_name)
            writer.append(HARDWARE_STREAM, record)
            return result

        recorded.__name__ = name
        return recorded


def record_core(core: Any, writer: TraceWriter, frame_mode: Optional[str] = None) -> RecordingProxy:
    """A recording stand-in for a Pycro-Manager ``Core`` (or the demo core)."""
    return RecordingProxy(core, writer, "core", frame_mode or frame_mode_from_env())


def record_studio(studio: Any, writer: TraceWriter, frame_mode: Optional[str] = None) -> Any:
    """A recording stand-in for ``Studio``; the calls on ``studio.live()`` are recorded."""
    if studio is None:
        return None
    return RecordingProxy(
        studio,
        writer,
        "studio",
        frame_mode or frame_mode_from_env(),
        nested={"live": "studio.live"},
    )
