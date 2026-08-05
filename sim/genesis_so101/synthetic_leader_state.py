#!/usr/bin/env python3
"""Publish hardware-free dual-leader packets to a loopback Genesis gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import socket
import time

from .live_protocol import LeaderActionPacket, encode_packet


HOME_ARM_ACTION = (0.0, -55.0, 70.0, 70.0, 0.0, 35.0)


def synthetic_action(phase: float) -> tuple[float, ...]:
    """Return a bounded 12-DoF LeRobot-unit action over a 0..1 cycle."""

    if not math.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("phase must be finite and inside [0, 1]")
    cycle = 2.0 * math.pi * phase
    left = list(HOME_ARM_ACTION)
    right = list(HOME_ARM_ACTION)
    left[0] += 4.0 * math.sin(cycle)
    right[0] -= 4.0 * math.sin(cycle)
    left[4] += 6.0 * math.sin(2.0 * cycle)
    right[4] -= 6.0 * math.sin(2.0 * cycle)
    left[5] += 8.0 * (0.5 + 0.5 * math.sin(cycle))
    right[5] += 8.0 * (0.5 - 0.5 * math.sin(cycle))
    return tuple(left + right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58281)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--cycle-s", type=float, default=6.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("synthetic leader state is restricted to loopback")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.duration_s <= 0.0 or args.cycle_s <= 0.0:
        raise ValueError("durations must be positive")
    if not 20.0 <= args.hz <= 120.0:
        raise ValueError("hz must be between 20 and 120")

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = (args.host, args.port)
    interval_s = 1.0 / args.hz
    started = time.monotonic()
    next_send = started
    sent = 0
    send_errors = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        while not stop_requested:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= args.duration_s:
                break
            if now < next_send:
                time.sleep(min(next_send - now, 0.005))
                continue
            phase = (elapsed % args.cycle_s) / args.cycle_s
            packet = LeaderActionPacket(
                sequence_id=sent,
                captured_monotonic_ns=time.monotonic_ns(),
                captured_unix_ns=time.time_ns(),
                action=synthetic_action(phase),
            )
            try:
                udp.sendto(encode_packet(packet), destination)
            except OSError:
                send_errors += 1
                raise
            sent += 1
            next_send += interval_s
            if next_send < time.monotonic() - interval_s:
                next_send = time.monotonic()

    elapsed_s = time.monotonic() - started
    report = {
        "schema_version": "radeon_oneloop.synthetic_leader_state.v1",
        "formal": False,
        "accepted": sent >= max(int(elapsed_s * args.hz * 0.8), 1) and send_errors == 0,
        "termination": "signal" if stop_requested else "duration",
        "destination": {"host": args.host, "port": args.port},
        "requested_hz": args.hz,
        "elapsed_s": elapsed_s,
        "packets_sent": sent,
        "send_errors": send_errors,
        "trajectory": "bounded_dual_arm_joint_and_gripper_cycle",
        "serial_or_usb_access": False,
        "physical_output": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise RuntimeError("synthetic leader-state publisher gate failed")


if __name__ == "__main__":
    main()
