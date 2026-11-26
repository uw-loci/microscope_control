"""Read a trace directory back: manifest and streams, in sequence order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from microscope_control.trace.writer import (
    FRAMES_DIR,
    HARDWARE_STREAM,
    MANIFEST_NAME,
    SOCKET_STREAM,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Every record of a JSONL file; a truncated last line (a crash) is dropped."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


class Trace:
    """A recorded session: ``manifest``, ``hardware`` and ``socket`` records."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        manifest_path = self.directory / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"not a trace directory (no {MANIFEST_NAME}): {self.directory}")
        self.manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="ascii"))
        self.hardware: List[Dict[str, Any]] = read_jsonl(self.directory / HARDWARE_STREAM)
        self.socket: List[Dict[str, Any]] = read_jsonl(self.directory / SOCKET_STREAM)
        self.frames_dir = self.directory / FRAMES_DIR

    def merged(self) -> Iterator[Dict[str, Any]]:
        """Hardware and socket records interleaved by ``seq``."""
        return iter(sorted(self.hardware + self.socket, key=lambda r: r.get("seq", 0)))

    def commands(self) -> List[Dict[str, Any]]:
        """The socket records that opened a client command (``kind == 'command'``)."""
        return [r for r in self.socket if r.get("kind") == "command"]

    def stage_name(self, stage: int) -> str:
        for rec in self.socket:
            if rec.get("kind") == "command" and rec.get("stage") == stage:
                return str(rec.get("command", ""))
        return ""

    def hardware_calls(self, obj: Optional[str] = None) -> List[Dict[str, Any]]:
        calls = [r for r in self.hardware if r.get("kind") == "call"]
        if obj is not None:
            calls = [r for r in calls if r.get("obj") == obj]
        return calls

    def summary(self) -> Dict[str, Any]:
        """Counts for a quick look at what a trace holds."""
        methods: Dict[str, int] = {}
        for rec in self.hardware_calls():
            key = f"{rec.get('obj', 'core')}.{rec.get('method', '?')}"
            methods[key] = methods.get(key, 0) + 1
        commands: Dict[str, int] = {}
        for rec in self.commands():
            name = str(rec.get("command", "?"))
            commands[name] = commands.get(name, 0) + 1
        return {
            "directory": str(self.directory),
            "start_time": self.manifest.get("start_time"),
            "duration_s": self.manifest.get("duration_s"),
            "hardware_calls": len(self.hardware_calls()),
            "client_commands": len(self.commands()),
            "methods": dict(sorted(methods.items())),
            "commands": dict(sorted(commands.items())),
        }
