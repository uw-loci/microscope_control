"""Difference records: what a replay found that the recorded session did not do.

Every record carries a ``fingerprint`` built from the stage's command, the
kind of difference, the subject (a Core method or a reply) and the first
differing field, but never the stage number or the values, so the same
defect repeating across tiles groups into one line.

Run ``python -m microscope_control.trace group differences.jsonl`` to group a
file of records.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Kinds a replay produces. Hardware-side kinds come from ReplayCore, reply-side
# kinds from the server replayer.
KIND_METHOD = "hardware_call_method"  # a different Core method was called
KIND_ARGS = "hardware_call_args"  # same method, arguments outside tolerance
KIND_EXTRA = "hardware_call_extra"  # a call the trace does not have at all
KIND_MISSING = "hardware_call_missing"  # a recorded call that never came
KIND_UNRECORDED_QUERY = "hardware_query_unrecorded"  # a read the trace never saw
KIND_REPLY_VALUE = "reply_value"  # a decoded reply field differs
KIND_REPLY_BYTES = "reply_bytes"  # raw reply bytes differ
KIND_REPLY_LENGTH = "reply_length"  # the server sent more or fewer bytes
KIND_REPLY_TIMEOUT = "reply_timeout"  # the recorded reply never arrived
KIND_REPLAY_ERROR = "replay_error"  # the replay itself could not go on


def _short(value: Any, limit: int = 200) -> Any:
    """Keep values readable in a record (long strings and lists are cut)."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(+%d)" % (len(value) - limit)
    if isinstance(value, list) and len(value) > 20:
        return value[:20] + ["...(+%d)" % (len(value) - 20)]
    return value


@dataclass
class Difference:
    stage: int
    command: str
    kind: str
    subject: str
    field: str
    expected: Any = None
    actual: Any = None
    detail: str = ""
    thread: str = ""
    seq: Optional[int] = None
    fingerprint: str = field(default="", init=True)

    def __post_init__(self) -> None:
        self.expected = _short(self.expected)
        self.actual = _short(self.actual)
        if not self.fingerprint:
            self.fingerprint = make_fingerprint(self.command, self.kind, self.subject, self.field)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def one_line(self) -> str:
        where = f"stage {self.stage} {self.command}".strip()
        text = f"[{where}] {self.kind}: {self.subject}"
        if self.field:
            text += f".{self.field}"
        text += f" expected={self.expected!r} actual={self.actual!r}"
        if self.detail:
            text += f" ({self.detail})"
        return text


def make_fingerprint(command: str, kind: str, subject: str, field: str) -> str:
    return "|".join([command or "-", kind, subject or "-", field or "-"])


def write_differences(path: Path, differences: Iterable[Difference]) -> int:
    """Append the records to a JSONL file; returns how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="ascii", newline="\n") as fh:
        for diff in differences:
            fh.write(json.dumps(diff.as_dict(), ensure_ascii=True, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def read_differences(path: Path) -> List[Difference]:
    out: List[Difference] = []
    path = Path(path)
    if not path.exists():
        return out
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data = json.loads(line)
                out.append(Difference(**data))
    return out


def to_oracle_record(diff: Difference) -> Dict[str, Any]:
    """The record shape ``verify-by-oracle``'s ``oracle.py fingerprint`` groups.

    Its fields: ``stage`` (here the client command), ``output`` (the Core
    method or ``reply``), ``kind``, ``location`` (the differing field),
    ``value_a`` (recorded) and ``value_b`` (replayed).
    """
    return {
        "stage": diff.command or "-",
        "output": diff.subject,
        "kind": diff.kind,
        "location": diff.field,
        "location_class": diff.field.split("[")[0] if diff.field else "",
        "shape_relation": "",
        "value_a": diff.expected,
        "value_b": diff.actual,
        "detail": {"text": diff.detail, "replay_stage": diff.stage, "thread": diff.thread},
    }


def write_oracle_records(path: Path, differences: Iterable[Difference]) -> int:
    """Write the differences as the JSON list ``oracle.py fingerprint`` reads."""
    records = [to_oracle_record(d) for d in differences]
    Path(path).write_text(json.dumps(records, ensure_ascii=True, indent=1) + "\n")
    return len(records)


def group_by_fingerprint(differences: Iterable[Difference]) -> Dict[str, List[Difference]]:
    """Records grouped by fingerprint, largest group first."""
    groups: Dict[str, List[Difference]] = {}
    for diff in differences:
        groups.setdefault(diff.fingerprint, []).append(diff)
    return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def format_groups(groups: Dict[str, List[Difference]]) -> str:
    lines: List[str] = []
    total = sum(len(v) for v in groups.values())
    lines.append(f"{total} difference(s) in {len(groups)} group(s)")
    for fp, items in groups.items():
        first = items[0]
        stages = sorted({d.stage for d in items})
        shown = ", ".join(str(s) for s in stages[:8]) + (" ..." if len(stages) > 8 else "")
        lines.append(f"  {len(items):5d}  {fp}")
        lines.append(f"         stages: {shown}")
        lines.append(f"         first: expected={first.expected!r} actual={first.actual!r}")
        if first.detail:
            lines.append(f"         {first.detail}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m microscope_control.trace group",
        description="Group a replay's difference records by fingerprint.",
    )
    parser.add_argument("path", help="differences.jsonl written by a replay")
    args = parser.parse_args(argv)
    diffs = read_differences(Path(args.path))
    print(format_groups(group_by_fingerprint(diffs)))
    return 1 if diffs else 0
