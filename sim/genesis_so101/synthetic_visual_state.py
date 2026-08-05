#!/usr/bin/env python3
"""Publish a hardware-free visual-state sweep for the live Gaussian gate.

This process only writes versioned UDP snapshots to a loopback socket.  It
does not import LeRobot, enumerate serial devices, step Genesis physics, or
issue motor commands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import time

from radeon_oneloop.contracts import lerobot_arm_to_genesis

from .visual_state_protocol import VisualStatePacket, encode_visual_state


HOME_ARM_ACTION = (0.0, -55.0, 70.0, 70.0, 0.0, 35.0)
OBJECT_START_POS = (0.10, -0.26, 0.47)


def synthetic_packet(sequence_id: int, phase: float) -> VisualStatePacket:
    """Return one deterministic snapshot over a normalized 0..1 sweep."""

    if sequence_id < 0:
        raise ValueError("sequence_id must be non-negative")
    if not math.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("phase must be finite and inside [0, 1]")

    cycle = 2.0 * math.pi * phase
    left = list(lerobot_arm_to_genesis(HOME_ARM_ACTION))
    right = list(lerobot_arm_to_genesis(HOME_ARM_ACTION))
    # Small visual-only joint excursions make stale follower transforms easy
    # to spot without bringing any physical controller into the process.
    left[0] += math.radians(4.0) * math.sin(cycle)
    right[0] -= math.radians(4.0) * math.sin(cycle)
    left[4] += math.radians(6.0) * math.sin(2.0 * cycle)
    right[4] -= math.radians(6.0) * math.sin(2.0 * cycle)

    # One full yaw exposes every azimuth while a small translation checks that
    # the Gaussian and the invisible collision entity share the same root.
    yaw = cycle
    position = (
        OBJECT_START_POS[0] + 0.02 * math.sin(cycle),
        OBJECT_START_POS[1],
        OBJECT_START_POS[2] + 0.005 * math.sin(2.0 * cycle),
    )
    quaternion = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return VisualStatePacket(
        sequence_id=sequence_id,
        captured_monotonic_ns=time.monotonic_ns(),
        captured_unix_ns=time.time_ns(),
        joint_positions_rad=tuple(left + right),
        object_position_m=position,
        object_quaternion_wxyz=quaternion,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58183)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("synthetic visual state is restricted to loopback")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.duration_s <= 0.0:
        raise ValueError("duration-s must be positive")
    if not 20.0 <= args.hz <= 120.0:
        raise ValueError("hz must be between 20 and 120")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = (args.host, args.port)
    interval_s = 1.0 / args.hz
    started = time.monotonic()
    next_send = started
    sent = 0
    send_errors = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= args.duration_s:
                break
            if now < next_send:
                time.sleep(min(next_send - now, 0.005))
                continue
            phase = min(elapsed / args.duration_s, 1.0)
            packet = synthetic_packet(sent, phase)
            try:
                udp.sendto(encode_visual_state(packet), destination)
            except OSError:
                send_errors += 1
                raise
            sent += 1
            next_send += interval_s
            if next_send < time.monotonic() - interval_s:
                next_send = time.monotonic()

    elapsed_s = time.monotonic() - started
    report = {
        "schema_version": "radeon_oneloop.synthetic_visual_state.v1",
        "formal": False,
        "accepted": sent >= int(args.duration_s * args.hz * 0.9) and send_errors == 0,
        "destination": {"host": args.host, "port": args.port},
        "requested_hz": args.hz,
        "duration_s": args.duration_s,
        "elapsed_s": elapsed_s,
        "packets_sent": sent,
        "send_errors": send_errors,
        "trajectory": "full_yaw_plus_small_xyz_and_joint_sweep",
        "hardware_access": False,
        "physical_output": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise RuntimeError("synthetic visual-state publisher gate failed")


if __name__ == "__main__":
    main()
