#!/usr/bin/env python3
"""Reproduce a multi-slide stage traverse over the serial port, with nothing else attached.

WHY THIS EXISTS
---------------
On a four-slide run (2026-10-02, Prior ProScan III on a PPM scope) the first slide
acquired correctly and the three acquired 9, 12 and 30 hours later were progressively
offset in Y -- the last badly enough to image the slide label for 1,714 tiles. The
Micro-Manager CoreLog for that session rules out everything a log can rule out:

  * devices initialised once and were unloaded once, with no reset in between;
  * six serial timeouts in four days, none during any acquisition;
  * for every move over 10 mm -- 42 of them across two four-day sessions, including an
    85.9 mm traverse of the whole holder -- the position read back after the move equalled
    the commanded position EXACTLY, on both axes.

That last point matters. A controller that re-zeroed would report a position near zero
afterwards, not the commanded value, so a coordinate reset would have been visible and was
not. The register is faithful. Whatever is lost is lost between the register and the
stage, which an open-loop controller cannot report -- it counts what it was asked to do.

The remaining suspicion is therefore mechanical or electrical: a large, fast move skipping
steps. The intended XY motion between slides is large and fast, and the offset is always in
Y while the big component of the move is in X, which raises the possibility of a
cross-axis effect (for instance a fast X move drawing enough current that Y loses steps).

This script drives only the stage, so that possibility can be tested without a camera, a
sample, autofocus, tiling, Micro-Manager or any of our software in the way -- and so the
result can be handed to the manufacturer as something they can run themselves.

WHAT IT CAN AND CANNOT SEE
--------------------------
Over the serial port alone it can detect a CONTROLLER-side fault: a coordinate reset, a
refused command, a timeout, a move that reports an arrival disagreeing with its target.

It CANNOT detect lost steps, because nothing in the protocol reports them. For that, put
a fiducial under the objective, note where it sits, run this, and look again. The script
returns to the starting position at the end of every cycle precisely so that check is a
matter of looking rather than arithmetic.

USAGE
-----
    python3 prior_xy_slide_change_stress.py --port COM1 --speed 100 --accel 100 --cycles 20

The speed and acceleration are the point of the exercise: sweep them. The values that
produced the failure were ``--speed 100 --accel 100``, the adapter's maxima, which is what
Micro-Manager had been leaving them at.

    !!  A LARGE FAST MOVE IS NOT A FREE PARAMETER.  !!
    Driving an XY stage at its maximum speed and acceleration can skip steps, stall, or
    stress the mechanics, and the consequences do not announce themselves -- a stage can
    lose position while reporting that it has not. Start low and work up, keep the
    objective well clear of the sample, and ask the manufacturer what this stage is rated
    for before trusting a value.

Baud rate must match whatever the Micro-Manager configuration uses for this port; the
default below is the Prior factory setting, not a guess at your rig.
"""

import argparse
import contextlib
import statistics
import sys
import time
from typing import List, Optional, Tuple

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial is required: pip install pyserial")


# Slot positions from the 2026-10-02 run (um), in the order the acquire pass visited them.
# Each entry is the first autofocus position of that slide's first region, so the moves
# between them are the real slide-change traverses: 85.9 mm into slot 1, then 37.5, 28.8
# and 20.2 mm. Override with --positions to match a different holder.
DEFAULT_POSITIONS: List[Tuple[float, float]] = [
    (38111, -14028),  # slot 1
    (2858, -30919),  # slot 2
    (-23759, -18944),  # slot 3
    (-41453, -7012),  # slot 4
]

# The controller answers a status query with a bitmask; 0 means every axis has stopped.
STATUS_IDLE = 0


class Prior:
    """The handful of ProScan commands this test needs, and nothing else."""

    def __init__(self, port: str, baud: int, timeout: float, verbose: bool):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.verbose = verbose
        time.sleep(0.25)
        self.ser.reset_input_buffer()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.ser.close()

    def command(self, text: str) -> str:
        """Send one command and return its reply, stripped."""
        self.ser.reset_input_buffer()
        self.ser.write((text + "\r").encode("ascii"))
        reply = self.ser.readline().decode("ascii", errors="replace").strip()
        if self.verbose:
            print(f"    -> {text:<24} <- {reply!r}")
        if not reply:
            raise TimeoutError(f"no reply to {text!r} within the serial timeout")
        return reply

    # --- motion profile ---
    def set_profile(self, speed: int, accel: int) -> None:
        for cmd in (f"SMS,{speed}", f"SAS,{accel}"):
            reply = self.command(cmd)
            if reply.upper() not in ("R", "0"):
                print(f"  WARNING: {cmd} answered {reply!r} rather than an acknowledgement")

    def read_profile(self) -> Tuple[str, str]:
        return self.command("SMS"), self.command("SAS")

    # --- position ---
    def position(self) -> Tuple[float, float]:
        return float(self.command("PX")), float(self.command("PY"))

    def goto(self, x: float, y: float) -> None:
        reply = self.command(f"G,{round(x)},{round(y)}")
        if reply.upper() != "R":
            raise RuntimeError(f"move to ({x}, {y}) was not acknowledged: {reply!r}")

    def wait_until_stopped(self, timeout_s: float) -> float:
        """Poll status until every axis stops. Returns how long the move took."""
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if int(self.command("$")) == STATUS_IDLE:
                return time.monotonic() - start
            time.sleep(0.015)
        raise TimeoutError(f"stage still moving after {timeout_s:.1f} s")


def parse_positions(raw: Optional[str]) -> List[Tuple[float, float]]:
    if not raw:
        return DEFAULT_POSITIONS
    points = []
    for pair in raw.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    if len(points) < 2:
        sys.exit("--positions needs at least two points to move between")
    return points


def main() -> int:
    p = argparse.ArgumentParser(
        description="Drive a Prior XY stage through a multi-slide traverse sequence over serial.",
        epilog="A large fast move can skip steps or stall. Start low, keep the objective "
        "clear of the sample, and check the stage's rating with the manufacturer.",
    )
    p.add_argument("--port", required=True, help="serial port, e.g. COM1 or /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=9600, help="must match the Micro-Manager config")
    p.add_argument("--speed", type=int, required=True, help="SMS value (Prior: 1-100 percent)")
    p.add_argument("--accel", type=int, required=True, help="SAS value (Prior: 1-100 percent)")
    p.add_argument("--cycles", type=int, default=10, help="times to walk the whole sequence")
    p.add_argument(
        "--positions",
        help='override the slot positions: "x1,y1;x2,y2;..." in um. '
        "Defaults to the four slots of the 2026-10-02 run.",
    )
    p.add_argument("--settle", type=float, default=0.5, help="pause after each move (s)")
    p.add_argument("--move-timeout", type=float, default=30.0, help="per-move limit (s)")
    p.add_argument("--serial-timeout", type=float, default=2.0, help="per-reply limit (s)")
    p.add_argument("--verbose", action="store_true", help="log every command and reply")
    args = p.parse_args()

    positions = parse_positions(args.positions)

    print("Prior XY slide-change stress test")
    print(f"  port={args.port} baud={args.baud}  SMS={args.speed} SAS={args.accel}")
    print(f"  {len(positions)} positions, {args.cycles} cycles")
    print(
        "  NOTE: this detects CONTROLLER faults (resets, refusals, arrival disagreements).\n"
        "        Lost steps are invisible over serial -- watch a fiducial for those.\n"
    )

    stage = Prior(args.port, args.baud, args.serial_timeout, args.verbose)
    disagreements = 0
    durations: List[float] = []
    try:
        before_speed, before_accel = stage.read_profile()
        print(f"  profile before: SMS={before_speed} SAS={before_accel}")
        stage.set_profile(args.speed, args.accel)
        after_speed, after_accel = stage.read_profile()
        print(f"  profile now:    SMS={after_speed} SAS={after_accel}")

        origin = stage.position()
        print(f"  starting position: ({origin[0]:.0f}, {origin[1]:.0f}) um\n")

        for cycle in range(1, args.cycles + 1):
            # Back to the start each cycle, so a fiducial check is just looking.
            for target in list(positions) + [origin]:
                here = stage.position()
                distance = ((target[0] - here[0]) ** 2 + (target[1] - here[1]) ** 2) ** 0.5
                stage.goto(*target)
                elapsed = stage.wait_until_stopped(args.move_timeout)
                time.sleep(args.settle)
                got = stage.position()
                dx = got[0] - round(target[0])
                dy = got[1] - round(target[1])
                durations.append(elapsed)
                flag = ""
                if abs(dx) > 2 or abs(dy) > 2:
                    disagreements += 1
                    flag = "   <-- ARRIVAL DISAGREES WITH TARGET"
                print(
                    f"  cycle {cycle:3d}  {distance / 1000:6.1f} mm in {elapsed:5.2f} s  "
                    f"-> ({got[0]:8.0f}, {got[1]:8.0f})  dX={dx:+4.0f} dY={dy:+4.0f}{flag}"
                )

            # A profile that no longer holds is evidence the controller restarted: we set
            # these below its defaults and nothing else writes them during the test.
            now_speed, now_accel = stage.read_profile()
            if (now_speed, now_accel) != (after_speed, after_accel):
                print(
                    f"  !! cycle {cycle}: motion profile changed by itself "
                    f"(SMS {after_speed}->{now_speed}, SAS {after_accel}->{now_accel}). "
                    "That is what a controller restart looks like."
                )
                disagreements += 1

        final = stage.position()
        print(
            f"\n  returned to ({final[0]:.0f}, {final[1]:.0f}) um; started at "
            f"({origin[0]:.0f}, {origin[1]:.0f}) um"
        )
        if durations:
            print(
                f"  move times: median {statistics.median(durations):.2f} s, "
                f"max {max(durations):.2f} s over {len(durations)} moves"
            )
        print(f"  arrival disagreements: {disagreements}")
        if disagreements == 0:
            print(
                "\n  The controller reported no fault. This does NOT mean the stage kept\n"
                "  position: check the fiducial. A stage that reports the count it was given\n"
                "  looks identical whether or not it travelled that far."
            )
        return 1 if disagreements else 0
    finally:
        stage.close()


if __name__ == "__main__":
    sys.exit(main())
