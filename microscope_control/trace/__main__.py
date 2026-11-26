"""``python -m microscope_control.trace summary TRACE_DIR`` / ``group DIFFERENCES.jsonl``."""

import json
import sys
from pathlib import Path

from microscope_control.trace.diff import main as group_main
from microscope_control.trace.diff import read_differences, write_oracle_records
from microscope_control.trace.reader import Trace

USAGE = (
    "usage: python -m microscope_control.trace summary TRACE_DIR\n"
    "       python -m microscope_control.trace group DIFFERENCES.jsonl\n"
    "       python -m microscope_control.trace oracle-records DIFFERENCES.jsonl OUT.json"
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "summary" and len(argv) == 2:
        print(json.dumps(Trace(Path(argv[1])).summary(), indent=2))
        return 0
    if argv[0] == "group":
        return group_main(argv[1:])
    if argv[0] == "oracle-records" and len(argv) == 3:
        n = write_oracle_records(Path(argv[2]), read_differences(Path(argv[1])))
        print(f"{n} record(s) written to {argv[2]}")
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
