"""JSON-safe encoding of the values that cross the Micro-Manager Core boundary.

Everything written to a trace is plain JSON built from ASCII text. Images are
never written whole by default: an array becomes its hash, shape, dtype and a
few statistics, optionally with a small preview or, on request, a full copy
saved beside the stream as ``.npy``.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# How frames are stored. ``hash``: digest, shape, dtype, min/max/mean.
# ``preview``: hash plus a strided copy no larger than PREVIEW_PX on a side.
# ``full``: hash plus the whole array as a .npy file. ``none``: shape and
# dtype only (the cheapest; frames cannot be told apart).
FRAME_MODES = ("none", "hash", "preview", "full")
PREVIEW_PX_ENV = "MICROSCOPE_TRACE_PREVIEW_PX"
DEFAULT_PREVIEW_PX = 128

ARRAY_KEY = "__ndarray__"
TUPLE_KEY = "__tuple__"
STRVECTOR_KEY = "__strvector__"
TAGGED_KEY = "__tagged_image__"
ERROR_KEY = "__error__"
BYTES_KEY = "__bytes__"
REPR_KEY = "__repr__"


def _ascii(text: str) -> str:
    """Escape anything outside printable ASCII so the file stays cp1252-safe."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _finite(value: float) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # 'nan', 'inf', '-inf'
    return value


def array_digest(array: np.ndarray) -> str:
    """Short BLAKE2b digest of the array bytes (shape and dtype are stored beside it)."""
    buf = np.ascontiguousarray(array)
    return hashlib.blake2b(buf.view(np.uint8).reshape(-1), digest_size=16).hexdigest()


def preview_px() -> int:
    """Longest side of a stored preview, from :data:`PREVIEW_PX_ENV` (default 128)."""
    raw = os.environ.get(PREVIEW_PX_ENV, "").strip()
    return int(raw) if raw else DEFAULT_PREVIEW_PX


def preview_step(shape: Tuple[int, ...], max_px: int) -> int:
    return max(1, int(math.ceil(max(shape[:2] or (1,)) / float(max_px))))


def preview_of(array: np.ndarray, max_px: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """A strided copy no larger than ``max_px`` on either image axis, and the stride."""
    if array.ndim == 0:
        return array.reshape(1), 1
    step = preview_step(array.shape, max_px or preview_px())
    if array.ndim == 1:
        return np.ascontiguousarray(array[::step]), step
    return np.ascontiguousarray(array[::step, ::step]), step


def encode_array(
    array: np.ndarray,
    frame_mode: str = "hash",
    frames_dir: Optional[Path] = None,
    frame_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize an array for the trace; see :data:`FRAME_MODES`."""
    if frame_mode not in FRAME_MODES:
        raise ValueError(f"frame_mode must be one of {FRAME_MODES}, not {frame_mode!r}")
    info: Dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if frame_mode == "none":
        return {ARRAY_KEY: info}
    info["digest"] = array_digest(array)
    if array.size and array.dtype.kind in "biuf":
        info["min"] = _finite(float(array.min()))
        info["max"] = _finite(float(array.max()))
        info["mean"] = _finite(float(array.mean()))
    if frame_mode in ("preview", "full"):
        if frames_dir is None or frame_name is None:
            raise ValueError(f"frame_mode {frame_mode!r} needs frames_dir and frame_name")
        frames_dir.mkdir(parents=True, exist_ok=True)
        if frame_mode == "preview":
            small, step = preview_of(array)
            path = frames_dir / f"{frame_name}_preview.npy"
            np.save(path, small)
            info["preview_file"] = path.name
            info["preview_step"] = step
        else:
            path = frames_dir / f"{frame_name}.npy"
            np.save(path, array)
            info["file"] = path.name
    return {ARRAY_KEY: info}


def encode_value(
    value: Any,
    frame_mode: str = "hash",
    frames_dir: Optional[Path] = None,
    frame_name: Optional[str] = None,
) -> Any:
    """Turn a Core argument or return value into JSON-safe data.

    Arrays go through :func:`encode_array`. Objects shaped like MMCore's
    ``StrVector`` (``size()``/``get(i)``) become a list of strings and
    ``TaggedImage``-like objects (``.pix``/``.tags``) keep both parts. Anything
    else unknown is kept as its ``repr``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return _ascii(value) if isinstance(value, str) else value
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return encode_array(value, frame_mode, frames_dir, frame_name)
    if isinstance(value, (bytes, bytearray)):
        return {BYTES_KEY: bytes(value).hex()}
    if isinstance(value, tuple):
        return {TUPLE_KEY: [encode_value(v, frame_mode, frames_dir, frame_name) for v in value]}
    size = getattr(value, "size", None)
    get = getattr(value, "get", None)
    if callable(size) and callable(get) and not isinstance(value, dict):
        # MMCore's StrVector, or the demo core's list subclass that mimics it
        try:
            return {STRVECTOR_KEY: [_ascii(str(get(i))) for i in range(int(size()))]}
        except Exception:  # noqa: BLE001 - fall through to the plain encodings
            pass
    if isinstance(value, list):
        return [encode_value(v, frame_mode, frames_dir, frame_name) for v in value]
    if isinstance(value, dict):
        return {
            str(k): encode_value(v, frame_mode, frames_dir, frame_name) for k, v in value.items()
        }
    if isinstance(value, BaseException):
        return {ERROR_KEY: {"type": type(value).__name__, "message": _ascii(str(value))}}
    pix = getattr(value, "pix", None)
    tags = getattr(value, "tags", None)
    if pix is not None and tags is not None:
        return {
            TAGGED_KEY: {
                "pix": encode_value(np.asarray(pix), frame_mode, frames_dir, frame_name),
                "tags": encode_value(dict(tags), frame_mode),
            }
        }
    return {REPR_KEY: _ascii(repr(value))}


class StrVector(list):
    """A list that also answers ``size()`` / ``get(i)`` like MMCore's StrVector."""

    def size(self) -> int:
        return len(self)

    def get(self, i: int):
        return self[i]


class TaggedImage:
    """The ``.pix`` / ``.tags`` pair that ``core.get_tagged_image()`` returns."""

    def __init__(self, pix: np.ndarray, tags: Dict[str, Any]):
        self.pix = pix
        self.tags = tags


class RecordedError(RuntimeError):
    """Raised on replay in place of an exception the live Core raised.

    ``str(exc)`` is the recorded message unchanged, so code that reads the
    text behaves as it did; ``recorded_type`` names the original class.
    """

    def __init__(self, message: str, recorded_type: str = "Exception"):
        super().__init__(message)
        self.recorded_type = recorded_type


def _decode_float(value: Any) -> Any:
    if isinstance(value, str) and value in ("nan", "inf", "-inf"):
        return float(value)
    return value


def decode_array(info: Dict[str, Any], frames_dir: Optional[Path] = None) -> np.ndarray:
    """Rebuild an array from its trace summary.

    A full copy is loaded from disk when one was saved; a preview is
    nearest-neighbor upsampled to the recorded shape; otherwise the frame is
    filled with the recorded mean (or zeros), so the code under test gets an
    array of the right shape and dtype but not the real pixels.
    """
    shape = tuple(int(n) for n in info["shape"])
    dtype = np.dtype(info["dtype"])
    if frames_dir is not None and "file" in info and (frames_dir / info["file"]).exists():
        return np.load(frames_dir / info["file"])
    if (
        frames_dir is not None
        and "preview_file" in info
        and (frames_dir / info["preview_file"]).exists()
    ):
        small = np.load(frames_dir / info["preview_file"])
        step = int(info.get("preview_step", 1))
        out = small
        for axis in range(min(2, len(shape))):
            out = np.repeat(out, step, axis=axis)
        slices = tuple(slice(0, n) for n in shape[: min(2, len(shape))])
        out = out[slices]
        return np.ascontiguousarray(out).astype(dtype, copy=False)
    fill = info.get("mean")
    fill = _decode_float(fill) if fill is not None else 0
    if dtype.kind in "iu":
        fill = int(round(fill))
    return np.full(shape, fill, dtype=dtype)


def decode_value(value: Any, frames_dir: Optional[Path] = None) -> Any:
    """Inverse of :func:`encode_value` as far as the trace allows."""
    if isinstance(value, list):
        return [decode_value(v, frames_dir) for v in value]
    if not isinstance(value, dict):
        return _decode_float(value)
    if ARRAY_KEY in value:
        return decode_array(value[ARRAY_KEY], frames_dir)
    if TUPLE_KEY in value:
        return tuple(decode_value(v, frames_dir) for v in value[TUPLE_KEY])
    if STRVECTOR_KEY in value:
        return StrVector(value[STRVECTOR_KEY])
    if TAGGED_KEY in value:
        inner = value[TAGGED_KEY]
        return TaggedImage(decode_value(inner["pix"], frames_dir), decode_value(inner["tags"]))
    if BYTES_KEY in value:
        return bytes.fromhex(value[BYTES_KEY])
    if ERROR_KEY in value:
        err = value[ERROR_KEY]
        return RecordedError(err["message"], err["type"])
    if REPR_KEY in value:
        return value[REPR_KEY]
    return {k: decode_value(v, frames_dir) for k, v in value.items()}


def is_error(value: Any) -> bool:
    return isinstance(value, dict) and ERROR_KEY in value


def values_match(expected: Any, actual: Any, abs_tol: float = 0.0) -> Tuple[bool, List[str]]:
    """Compare two encoded values; numbers within ``abs_tol`` count as equal.

    Returns ``(matched, path)`` where ``path`` names the first differing field
    (``["1", "x"]`` style), empty when they match.
    """

    def walk(e: Any, a: Any, path: List[str]) -> Optional[List[str]]:
        if isinstance(e, bool) or isinstance(a, bool):
            return None if e == a else path
        if isinstance(e, (int, float)) and isinstance(a, (int, float)):
            return None if abs(float(e) - float(a)) <= abs_tol else path
        if isinstance(e, dict) and isinstance(a, dict):
            if ARRAY_KEY in e and ARRAY_KEY in a:
                ei, ai = e[ARRAY_KEY], a[ARRAY_KEY]
                for key in ("shape", "dtype", "digest"):
                    if key in ei and key in ai and ei[key] != ai[key]:
                        return path + [key]
                return None
            if set(e) != set(a):
                return path + [sorted(set(e) ^ set(a))[0]]
            for key in e:
                bad = walk(e[key], a[key], path + [str(key)])
                if bad is not None:
                    return bad
            return None
        if isinstance(e, list) and isinstance(a, list):
            if len(e) != len(a):
                return path + ["len"]
            for i, (x, y) in enumerate(zip(e, a)):
                bad = walk(x, y, path + [str(i)])
                if bad is not None:
                    return bad
            return None
        return None if e == a else path

    bad = walk(expected, actual, [])
    return (bad is None), (bad or [])
