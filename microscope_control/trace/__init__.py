"""Record-and-replay traces of the Micro-Manager Core boundary.

A trace is one directory holding ``manifest.json`` plus append-only JSONL
streams. ``hardware.jsonl`` carries every call that crossed into the
Micro-Manager ``Core`` (and ``Studio``) with its arguments, return value and
timing; the command server adds ``socket.jsonl`` with the client traffic.

Recording: :func:`record_core` / :func:`record_studio` wrap the live objects
in a proxy that appends one JSON line per call. Replay: :class:`ReplayCore`
serves the recorded return values in order and raises
:class:`ReplayDivergence` when the code under test asks for something else.
Differences are :class:`Difference` records a fingerprint tool can group.
"""

from microscope_control.trace.diff import (
    Difference,
    group_by_fingerprint,
    read_differences,
    to_oracle_record,
    write_differences,
    write_oracle_records,
)
from microscope_control.trace.reader import Trace
from microscope_control.trace.recording import (
    FRAME_MODES,
    RecordingProxy,
    frame_mode_from_env,
    record_core,
    record_studio,
)
from microscope_control.trace.replay import (
    ReplayCore,
    ReplayDivergence,
    ReplaySession,
    ReplayStudio,
)
from microscope_control.trace.writer import (
    TRACE_DIR_ENV,
    TRACE_FRAMES_ENV,
    TraceWriter,
    new_trace_dir,
)

__all__ = [
    "Difference",
    "FRAME_MODES",
    "RecordingProxy",
    "ReplayCore",
    "ReplayDivergence",
    "ReplaySession",
    "ReplayStudio",
    "TRACE_DIR_ENV",
    "TRACE_FRAMES_ENV",
    "Trace",
    "TraceWriter",
    "frame_mode_from_env",
    "group_by_fingerprint",
    "new_trace_dir",
    "read_differences",
    "record_core",
    "record_studio",
    "to_oracle_record",
    "write_differences",
    "write_oracle_records",
]
