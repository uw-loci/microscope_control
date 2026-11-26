"""A fake Micro-Manager Core that serves a recorded trace.

:class:`ReplayCore` stands where the live ``Core`` (or the demo core) stood.
The code under test runs unchanged against it; every call is checked against
the trace and answered with the recorded return value.

Two classes of call are told apart:

* **Commands** change state or consume a frame (``set_*``, ``snap_image``,
  ``get_image``, ``wait_for_device``, ...). They must come in the recorded
  order with the recorded arguments (numbers within a tolerance). Anything
  else is a :class:`Difference` and, in strict mode, a
  :class:`ReplayDivergence`.
* **Queries** read state (``get_*`` other than frames, ``is_*``, ``has_*``,
  ``device_busy``). Polling loops and the stage position cache's background
  thread ask these a run-dependent number of times, so they are not order
  checked. Each is answered from the recorded reads of the same method and
  arguments made after the last consumed command and before the next one
  (so a ``device_busy`` loop sees True, True, False exactly as recorded and
  then keeps reading False); with no such read, the nearest recorded one is
  used. A query the trace never saw is a difference.

``mode="strict"`` raises on the first difference of a call; ``mode="record"``
notes it, answers with the recorded value anyway and moves on, which lets a
whole session be compared in one run.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

from microscope_control.trace.diff import (
    KIND_ARGS,
    KIND_EXTRA,
    KIND_METHOD,
    KIND_MISSING,
    KIND_UNRECORDED_QUERY,
    Difference,
)
from microscope_control.trace.reader import Trace
from microscope_control.trace.recording import FRAME_METHODS
from microscope_control.trace.values import (
    decode_value,
    encode_value,
    values_match,
)

QUERY_PREFIXES = ("get_", "is_", "has_")
QUERY_METHODS = frozenset({"device_busy", "system_busy", "get_remaining_image_count"})
MODES = ("strict", "record")


def is_query(method: str) -> bool:
    """Reads that are answered from recorded state rather than order-checked."""
    if method in FRAME_METHODS:
        return False
    if method in QUERY_METHODS:
        return True
    return method.startswith(QUERY_PREFIXES)


class ReplayDivergence(AssertionError):
    """The code under test asked the hardware for something the trace does not have."""

    def __init__(self, difference: Difference):
        super().__init__(difference.one_line())
        self.difference = difference


class ReplaySession:
    """Shared cursor over one trace's hardware calls, for a ReplayCore and ReplayStudio."""

    def __init__(
        self,
        trace: Trace,
        abs_tol: float = 1e-6,
        tolerances: Optional[Dict[str, float]] = None,
        mode: str = "strict",
        max_differences: int = 1000,
        stage_of: Optional[Any] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.trace = trace
        self.abs_tol = float(abs_tol)
        self.tolerances = dict(tolerances or {})
        self.mode = mode
        self.max_differences = int(max_differences)
        self._stage_of = stage_of  # callable () -> (stage, command) of the replay, or None
        calls = [r for r in trace.hardware if r.get("kind") == "call"]
        self.commands: List[Dict[str, Any]] = [c for c in calls if not is_query(c["method"])]
        self._queries: Dict[str, List[Dict[str, Any]]] = {}
        for rec in calls:
            if is_query(rec["method"]):
                self._queries.setdefault(
                    self._key(rec["obj"], rec["method"], rec["args"]), []
                ).append(rec)
        self.cursor = 0
        self.cursor_seq = 0
        self.lookahead = 200
        self._query_counts: Dict[str, int] = {}
        self.differences: List[Difference] = []
        self._lock = threading.RLock()
        self._finished = False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _key(obj: str, method: str, enc_args: Any) -> str:
        return f"{obj}.{method}{json.dumps(enc_args, sort_keys=True)}"

    def _current_stage(self, recorded: Optional[Dict[str, Any]]) -> Tuple[int, str]:
        if self._stage_of is not None:
            try:
                stage, command = self._stage_of()
                return int(stage), str(command)
            except Exception:  # noqa: BLE001
                pass
        if recorded is not None:
            stage = int(recorded.get("stage", 0))
            return stage, self.trace.stage_name(stage)
        return 0, ""

    def _tolerance(self, method: str) -> float:
        return float(self.tolerances.get(method, self.abs_tol))

    def _note(self, diff: Difference) -> None:
        self.differences.append(diff)
        if len(self.differences) >= self.max_differences:
            raise ReplayDivergence(
                Difference(
                    diff.stage,
                    diff.command,
                    "replay_error",
                    "session",
                    "max_differences",
                    self.max_differences,
                    len(self.differences),
                    "too many differences; the replay stopped",
                )
            )
        if self.mode == "strict":
            raise ReplayDivergence(diff)

    @staticmethod
    def _result(recorded: Dict[str, Any], frames_dir: Any) -> Any:
        if "error" in recorded:
            raise decode_value({"__error__": recorded["error"]})
        return decode_value(recorded.get("ret"), frames_dir)

    def _lookahead(self, subject: str, enc_args: Any, method: str) -> Optional[int]:
        """Index of the next recorded ``subject`` call, matching arguments preferred."""
        end = min(len(self.commands), self.cursor + self.lookahead)
        first_same_method = None
        for j in range(self.cursor, end):
            rec = self.commands[j]
            if f"{rec['obj']}.{rec['method']}" != subject:
                continue
            if values_match(rec["args"], enc_args, self._tolerance(method))[0]:
                return j
            if first_same_method is None:
                first_same_method = j
        return first_same_method

    def _next_command_seq(self) -> float:
        if self.cursor < len(self.commands):
            return float(self.commands[self.cursor]["seq"])
        return math.inf

    # -- the call -----------------------------------------------------------

    def call(self, obj: str, method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        enc_args = [encode_value(a, "hash") for a in args]
        enc_kwargs = {k: encode_value(v, "hash") for k, v in kwargs.items()}
        with self._lock:
            if is_query(method):
                return self._query(obj, method, enc_args, enc_kwargs)
            return self._command(obj, method, enc_args, enc_kwargs)

    def _query(self, obj: str, method: str, enc_args: Any, enc_kwargs: Any) -> Any:
        key = self._key(obj, method, enc_args)
        occurrences = self._queries.get(key, [])
        if not occurrences:
            stage, command = self._current_stage(None)
            self._note(
                Difference(
                    stage,
                    command,
                    KIND_UNRECORDED_QUERY,
                    f"{obj}.{method}",
                    "args",
                    None,
                    enc_args,
                    "the trace never saw this read; nothing to answer with",
                    threading.current_thread().name,
                    self.cursor_seq,
                )
            )
            raise ReplayDivergence(self.differences[-1])
        k = self._query_counts.get(key, 0)
        self._query_counts[key] = k + 1
        upper = self._next_command_seq()
        segment = [r for r in occurrences if self.cursor_seq < r["seq"] < upper]
        if segment:
            chosen = segment[min(k, len(segment) - 1)]
        else:
            after = [r for r in occurrences if r["seq"] > self.cursor_seq]
            chosen = after[0] if after else occurrences[-1]
        return self._result(chosen, self.trace.frames_dir)

    def _command(self, obj: str, method: str, enc_args: Any, enc_kwargs: Any) -> Any:
        subject = f"{obj}.{method}"
        thread = threading.current_thread().name
        if self.cursor >= len(self.commands):
            stage, command = self._current_stage(None)
            self._note(
                Difference(
                    stage,
                    command,
                    KIND_EXTRA,
                    subject,
                    "args",
                    None,
                    enc_args,
                    "the trace has no more hardware commands",
                    thread,
                    self.cursor_seq,
                )
            )
            return None
        expected = self.commands[self.cursor]
        stage, command = self._current_stage(expected)
        exp_subject = f"{expected['obj']}.{expected['method']}"
        if exp_subject != subject:
            self._note(
                Difference(
                    stage,
                    command,
                    KIND_METHOD,
                    subject,
                    "method",
                    exp_subject,
                    subject,
                    f"expected {exp_subject}({_fmt(expected['args'])}) next, "
                    f"got {subject}({_fmt(enc_args)})",
                    thread,
                    expected.get("seq"),
                )
            )
            # Record mode: find where the trace next makes this call, report
            # what was skipped as missing, and answer from there; answering a
            # different method with this entry's value would only fail later.
            ahead = self._lookahead(subject, enc_args, method)
            if ahead is None:
                self.differences.append(
                    Difference(
                        stage,
                        command,
                        KIND_EXTRA,
                        subject,
                        "args",
                        None,
                        enc_args,
                        f"no later {subject} in the trace within {self.lookahead} commands",
                        thread,
                        self.cursor_seq,
                    )
                )
                return None
            for skipped in self.commands[self.cursor : ahead]:
                skipped_stage = int(skipped.get("stage", 0))
                self.differences.append(
                    Difference(
                        skipped_stage,
                        self.trace.stage_name(skipped_stage),
                        KIND_MISSING,
                        f"{skipped['obj']}.{skipped['method']}",
                        "args",
                        skipped["args"],
                        None,
                        "skipped: the replay went on to a later recorded call",
                        skipped.get("thread", ""),
                        skipped.get("seq"),
                    )
                )
            self.cursor = ahead
            expected = self.commands[self.cursor]
        else:
            ok, path = values_match(expected["args"], enc_args, self._tolerance(method))
            if ok:
                ok, path = values_match(
                    expected.get("kwargs", {}), enc_kwargs, self._tolerance(method)
                )
                path = ["kwargs"] + path if not ok else path
            else:
                path = ["args"] + path
            if not ok:
                field = "".join(f"[{p}]" if p.isdigit() else f".{p}" for p in path).lstrip(".")
                self._note(
                    Difference(
                        stage,
                        command,
                        KIND_ARGS,
                        subject,
                        field,
                        expected["args"] if path[0] == "args" else expected.get("kwargs", {}),
                        enc_args if path[0] == "args" else enc_kwargs,
                        f"tolerance {self._tolerance(method)}",
                        thread,
                        expected.get("seq"),
                    )
                )
        self.cursor += 1
        self.cursor_seq = int(expected["seq"])
        self._query_counts.clear()
        return self._result(expected, self.trace.frames_dir)

    # -- the end --------------------------------------------------------------

    def finish(self) -> List[Difference]:
        """Report the recorded commands that never came; returns every difference."""
        with self._lock:
            if not self._finished:
                self._finished = True
                for rec in self.commands[self.cursor :]:
                    stage = int(rec.get("stage", 0))
                    self.differences.append(
                        Difference(
                            stage,
                            self.trace.stage_name(stage),
                            KIND_MISSING,
                            f"{rec['obj']}.{rec['method']}",
                            "args",
                            rec["args"],
                            None,
                            "recorded but never called in the replay",
                            rec.get("thread", ""),
                            rec.get("seq"),
                        )
                    )
            return list(self.differences)

    def core(self) -> "ReplayCore":
        return ReplayCore(self)

    def studio(self) -> "ReplayStudio":
        return ReplayStudio(self)

    def summary(self) -> str:
        done = min(self.cursor, len(self.commands))
        return (
            f"{done}/{len(self.commands)} recorded hardware commands consumed, "
            f"{len(self.differences)} difference(s)"
        )


def _fmt(enc_args: Any) -> str:
    try:
        text = json.dumps(enc_args)
    except TypeError:
        text = repr(enc_args)
    return text[1:-1] if text.startswith("[") else text


class _ReplayProxy:
    def __init__(self, session: ReplaySession, obj: str):
        self._session = session
        self._obj = obj

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        session = self._session
        obj = self._obj

        def method(*args: Any, **kwargs: Any) -> Any:
            return session.call(obj, name, args, kwargs)

        method.__name__ = name
        return method


class ReplayCore(_ReplayProxy):
    """Answers the Micro-Manager Core API from a trace; see the module docstring."""

    def __init__(self, session: ReplaySession):
        super().__init__(session, "core")

    @property
    def session(self) -> ReplaySession:
        return self._session


class _ReplayLive(_ReplayProxy):
    def __init__(self, session: ReplaySession):
        super().__init__(session, "studio.live")


class ReplayStudio:
    """``studio.live()`` answered from the trace; nothing else of Studio is used."""

    def __init__(self, session: ReplaySession):
        self._session = session
        self._live = _ReplayLive(session)

    def live(self) -> _ReplayLive:
        return self._live
