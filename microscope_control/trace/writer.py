"""Append-only trace writer: one directory, a manifest and JSONL streams."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_DIR_ENV = "MICROSCOPE_TRACE_DIR"
TRACE_FRAMES_ENV = "MICROSCOPE_TRACE_FRAMES"

MANIFEST_NAME = "manifest.json"
HARDWARE_STREAM = "hardware.jsonl"
SOCKET_STREAM = "socket.jsonl"
FRAMES_DIR = "frames"
FORMAT_VERSION = 1


def _ascii_json(data: Any) -> str:
    # ensure_ascii escapes anything outside ASCII, so the file is cp1252-safe.
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=False)


def new_trace_dir(parent: Path, label: str = "session") -> Path:
    """A fresh ``<parent>/<label>_<YYYYmmdd_HHMMSS>`` directory (suffixed if taken)."""
    parent = Path(parent)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = parent / f"{label}_{stamp}"
    path = base
    n = 1
    while path.exists():
        n += 1
        path = parent / f"{base.name}_{n}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TraceWriter:
    """Writes the manifest and appends records to the trace streams.

    Each record gets ``seq`` (one counter across every stream, so the streams
    can be merged in order), ``t`` (seconds since the session started) and
    ``stage`` (the client command in progress, kept by the server recorder).
    Appends are serialized by one lock and flushed per line, so a crash loses
    at most the line being written.
    """

    def __init__(self, directory: Path, manifest: Optional[Dict[str, Any]] = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.directory / FRAMES_DIR
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.perf_counter()
        self._files: Dict[str, Any] = {}
        self.stage = 0
        self.stage_name = ""
        self._manifest: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "streams": [HARDWARE_STREAM, SOCKET_STREAM],
        }
        if manifest:
            self._manifest.update(manifest)
        self._write_manifest()

    # -- manifest -----------------------------------------------------------

    def update_manifest(self, **fields: Any) -> None:
        with self._lock:
            self._manifest.update(fields)
            self._write_manifest()

    def manifest_append(self, key: str, item: Any) -> None:
        """Append ``item`` to the manifest list under ``key`` (created if missing)."""
        with self._lock:
            items = self._manifest.setdefault(key, [])
            if item not in items:
                items.append(item)
            self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.directory / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(self._manifest, ensure_ascii=True, indent=2) + "\n")
        os.replace(tmp, self.directory / MANIFEST_NAME)

    @property
    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    # -- streams --------------------------------------------------------------

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def set_stage(self, stage: int, name: str) -> None:
        """Label later records with the client command now in progress."""
        with self._lock:
            self.stage = int(stage)
            self.stage_name = name

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def append(self, stream: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append one record; ``seq``, ``t`` and ``stage`` are filled in when absent."""
        with self._lock:
            if "seq" not in record:
                self._seq += 1
                record["seq"] = self._seq
            record.setdefault("t", round(self.now(), 6))
            record.setdefault("stage", self.stage)
            fh = self._files.get(stream)
            if fh is None:
                fh = open(  # noqa: SIM115 - kept open for the session, closed in close()
                    self.directory / stream, "a", encoding="ascii", newline="\n"
                )
                self._files[stream] = fh
            fh.write(_ascii_json(record))
            fh.write("\n")
            fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            for fh in self._files.values():
                with contextlib.suppress(Exception):
                    fh.close()
            self._files.clear()
            self._manifest["end_time"] = datetime.now().isoformat(timespec="seconds")
            self._manifest["duration_s"] = round(self.now(), 3)
            self._write_manifest()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
