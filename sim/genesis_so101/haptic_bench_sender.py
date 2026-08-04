#!/usr/bin/env python3
"""Send a bounded synthetic contact signal for the single-joint bench gate."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

from .haptic_hardware import ARM_MOTORS, BENCH_MOTORS
from .live_protocol import HapticFeedbackPacket, encode_haptic_packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58082)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--motor", choices=BENCH_MOTORS, required=True)
    parser.add_argument("--duration-s", type=float, default=10.5)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--contact-force-n", type=float, default=2.0)
    parser.add_argument("--reaction-effort", type=float, default=3.35)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if not 10.0 <= args.duration_s <= 12.0:
        raise ValueError("bench sender duration must be between 10 and 12 seconds")
    if not 20.0 <= args.hz <= 30.0:
        raise ValueError("bench sender rate must be between 20 and 30 Hz")
    if not 0.5 <= args.contact_force_n <= 5.0:
        raise ValueError("bench contact force must be between 0.5 and 5 N")
    if not 0.0 < abs(args.reaction_effort) <= 3.35:
        raise ValueError("bench reaction effort magnitude must be in (0, 3.35]")

    index = (0 if args.side == "left" else len(ARM_MOTORS)) + ARM_MOTORS.index(
        args.motor
    )
    efforts = [0.0] * 12
    efforts[index] = args.reaction_effort
    forces = (
        (args.contact_force_n, 0.0)
        if args.side == "left"
        else (0.0, args.contact_force_n)
    )
    destination = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period_s = 1.0 / args.hz
    started = time.monotonic()
    next_tick = started
    sent = 0
    try:
        while time.monotonic() - started < args.duration_s:
            packet = HapticFeedbackPacket(
                sequence_id=sent,
                captured_monotonic_ns=time.monotonic_ns(),
                captured_unix_ns=time.time_ns(),
                joint_reaction_effort=tuple(efforts),
                contact_force_n=forces,
            )
            sock.sendto(encode_haptic_packet(packet), destination)
            sent += 1
            next_tick += period_s
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        sock.close()
    elapsed_s = max(time.monotonic() - started, 1e-9)
    report = {
        "schema_version": "radeon_oneloop.haptic_bench_sender.v1",
        "side": args.side,
        "motor": args.motor,
        "duration_s": elapsed_s,
        "packets_sent": sent,
        "effective_hz": sent / elapsed_s,
        "contact_force_n": args.contact_force_n,
        "reaction_effort": args.reaction_effort,
        "physical_output_commands": False,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(payload, encoding="utf-8")
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
