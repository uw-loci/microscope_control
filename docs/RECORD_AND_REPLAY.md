# Record and replay the Micro-Manager boundary

`microscope_control.trace` records every call this library makes into the Micro-Manager
`Core` (and `Studio.live()`) and plays a recording back as a fake core, so a session captured
once at the microscope can be rerun against changed code on any machine. The command server
switches it on with `--trace-dir` and adds the client traffic; see
`microscope_command_server/README.md`, section *Record and replay a session*. This page
covers the library side: the trace format, what the fake core checks, and how to use it in
a test.

## What a trace holds

One session is one directory:

| File | Content |
|---|---|
| `manifest.json` | format version, start and end time, server version and the git ref of every lab package, the config files CONFIG named, the paths the client sent, connections |
| `hardware.jsonl` | one line per Core or Studio call: `obj`, `method`, `args`, `kwargs`, `ret` or `error`, `dt_ms`, `thread`, `seq`, `t` (seconds since start) and `stage` (the client command in progress) |
| `socket.jsonl` | the client commands and their payload and reply bytes (written by the command server) |
| `config/`, `inputs/` | copies of the YAML bundle and the `TileConfiguration.txt` files the session read |
| `frames/` | `.npy` frames, only when recorded with frame mode `preview` or `full` |

Everything is ASCII JSON; strings outside ASCII are escaped. A frame is never written as
pixels unless asked for: by default an image becomes its BLAKE2b digest, shape, dtype and
min, max and mean. `MICROSCOPE_TRACE_FRAMES` picks the mode:

| Mode | Stored per frame | Cost for a 2048 x 2448 x 3 uint8 frame | Replay gets |
|---|---|---|---|
| `none` | shape and dtype | none | an array of zeros |
| `hash` (default) | digest, shape, dtype, min, max, mean | 26 ms | an array filled with the recorded mean |
| `preview` | the above plus a strided copy, longest side `MICROSCOPE_TRACE_PREVIEW_PX` (128) | 31 ms | the preview upsampled to full size |
| `full` | the above plus the whole frame as `.npy` | 43 ms | the recorded pixels |

A plain call costs about 10 us through the recording proxy (measured on the development
machine: 0.04 us direct, 9.9 us recorded). Stage moves and exposures run for
milliseconds to seconds, so the recorder does not change their timing in any measurable way.

**Choose `full` when the session autofocuses.** Autofocus picks its Z from the pixels, and so
do the saturation and tissue checks. With digests alone, a replay feeds those decisions a
flat frame and they come out differently; that is a true report about the replay, not about
the code. The demo session with autofocus replays to zero differences with `full` frames and
diverges inside TESTAF with `hash` frames.

## The fake core

```python
from microscope_control.trace import Trace, ReplaySession

session = ReplaySession(Trace(trace_dir), mode="strict")
hardware = PycromanagerHardware(session.core(), session.studio(), settings)
...  # run the code under test
differences = session.finish()   # recorded commands that never came are added here
```

Calls are of two kinds:

- **Commands** change state or consume a frame: `set_*`, `snap_image`, `get_image`,
  `get_tagged_image`, `wait_for_device`, `home`, `set_config`, ... They must arrive in the
  recorded order with the recorded arguments; numbers are compared with `abs_tol` (default
  1e-6) or a per-method value (`tolerances={"set_xy_position": 0.5}`).
- **Queries** read state: `get_*` other than frames, `is_*`, `has_*`, `device_busy`. The stage
  position cache polls these from a background thread and `wait_z` polls `device_busy` in a
  loop, a run-dependent number of times, so they are not order checked. A query is answered
  from the recorded reads of the same method and arguments between the last consumed command
  and the next one: a `device_busy` loop sees True, True, False as recorded and then False.
  A query the trace never saw is a difference.

`mode="strict"` raises `ReplayDivergence` on the first departure, which is what a unit test
wants. `mode="record"` notes it and keeps answering from the trace (on a wrong method it
skips ahead to the next recorded call of that method and reports what it skipped as
missing), which is what a whole-session replay wants. A recorded exception is raised again
as `RecordedError` with the same message.

## Differences

Every departure is a `Difference` record: `stage`, `command`, `kind`, `subject`, `field`,
`expected`, `actual`, `detail`, `thread`, `seq` and a `fingerprint` of
`command|kind|subject|field` (never the stage number or the values, so the same defect across
40 tiles is one group). Kinds: `hardware_call_method`, `hardware_call_args`,
`hardware_call_extra`, `hardware_call_missing`, `hardware_query_unrecorded`, and the reply
kinds the command server adds (`reply_value`, `reply_bytes`, `reply_length`,
`reply_timeout`).

```
python -m microscope_control.trace summary TRACE_DIR          # what a trace holds
python -m microscope_control.trace group differences.jsonl    # grouped by fingerprint
python -m microscope_control.trace oracle-records differences.jsonl records.json
```

The last one writes the list the `verify-by-oracle` skill's `oracle.py fingerprint` reads.
After a difference is understood, the correction becomes a test: either a unit test on the
fake core (`tests/test_trace_replay.py` shows the pattern) or a replay of the trace in the
command server's suite.

## Limits

- Only what crosses the Core boundary is recorded. File inputs are copied for CONFIG and
  ACQUIRE (`TileConfiguration.txt`); background folders, white-balance folders and
  flat-field profiles are not, so a replay of a run that used them needs them on disk at
  the rewritten path (`--rewrite OLD=NEW` in the replayer).
- The position cache's timing is not reproduced. GETXY, GETZ and GETXYZ replies are waited
  for but not compared; a stage bug still shows in `set_xy_position` arguments.
- Recording is per process. A server restart starts a new session directory.
