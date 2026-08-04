#!/usr/bin/env python3
"""Read-only ten-joint preflight for a candidate dual-arm haptic stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .haptic_arm_readonly_preflight import (
    arm_command_envelope,
    evaluate_arm_hardware_snapshot,
)
from .haptic_hardware import BENCH_MOTORS
from .haptic_readonly_preflight import READ_ONLY_REGISTERS, read_register_snapshot
from .leader_publisher import _connect_read_only, _make_leader, _read_arm


def evaluate_dual_hardware_snapshot(
    *,
    action: tuple[float, ...],
    registers_by_side: dict[str, dict[str, dict[str, int]]],
    envelopes_by_side: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, dict[str, dict[str, Any]]]]:
    per_side_checks: dict[str, dict[str, bool]] = {}
    position_gates: dict[str, dict[str, dict[str, Any]]] = {}
    for side in ("left", "right"):
        checks, per_motor = evaluate_arm_hardware_snapshot(
            action=action,
            side=side,
            registers_by_motor=registers_by_side.get(side, {}),
            envelope=envelopes_by_side.get(side, {}),
        )
        per_side_checks[side] = checks
        position_gates[side] = per_motor
    checks = {
        "left_arm_readonly_preflight_accepted": all(
            per_side_checks["left"].values()
        ),
        "right_arm_readonly_preflight_accepted": all(
            per_side_checks["right"].values()
        ),
        "all_ten_non_gripper_motors_present": all(
            set(registers_by_side.get(side, {})) == set(BENCH_MOTORS)
            for side in ("left", "right")
        ),
        "both_command_envelopes_accepted": all(
            envelopes_by_side.get(side, {}).get("accepted") is True
            for side in ("left", "right")
        ),
    }
    checks.update(
        {
            f"{side}_{name}": value
            for side, side_checks in per_side_checks.items()
            for name, value in side_checks.items()
        }
    )
    return checks, position_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-id", required=True)
    parser.add_argument("--right-id", required=True)
    parser.add_argument("--simulated-effort-full-scale", type=float, required=True)
    parser.add_argument("--reaction-effort", type=float, required=True)
    parser.add_argument("--max-torque-limit-raw", type=int, default=15)
    parser.add_argument("--max-position-offset-deg", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.left_port == args.right_port:
        raise ValueError("left and right ports must differ")
    if not 0.0 < args.max_position_offset_deg <= 0.4:
        raise ValueError("dual preflight offset candidate must be in (0, 0.4]")
    if not 1 <= args.max_torque_limit_raw <= 15:
        raise ValueError("dual preflight torque candidate must be in [1, 15]")

    envelopes_by_side = {
        side: arm_command_envelope(
            side=side,
            simulated_effort_full_scale=args.simulated_effort_full_scale,
            reaction_effort=args.reaction_effort,
            max_torque_limit_raw=args.max_torque_limit_raw,
            max_position_offset_deg=args.max_position_offset_deg,
        )
        for side in ("left", "right")
    }
    leaders = [
        _make_leader(args.left_port, args.left_id),
        _make_leader(args.right_port, args.right_id),
    ]
    connected: list[Any] = []
    started = time.monotonic()
    try:
        for leader in leaders:
            _connect_read_only(leader)
            connected.append(leader)
        action = _read_arm(leaders[0]) + _read_arm(leaders[1])
        registers_by_side = {
            side: {
                motor: read_register_snapshot(leader.bus, motor)
                for motor in BENCH_MOTORS
            }
            for side, leader in zip(("left", "right"), leaders, strict=True)
        }
    finally:
        for leader in reversed(connected):
            if leader.bus.is_connected:
                leader.bus.disconnect(disable_torque=False)

    checks, position_gates = evaluate_dual_hardware_snapshot(
        action=action,
        registers_by_side=registers_by_side,
        envelopes_by_side=envelopes_by_side,
    )
    report = {
        "schema_version": "radeon_oneloop.haptic_dual_arm_readonly_preflight.v1",
        "formal": False,
        "stage": "dual_arm_readonly_preflight",
        "accepted": all(checks.values()),
        "checks": checks,
        "selection": {
            "sides": ["left", "right"],
            "motors_per_side": list(BENCH_MOTORS),
        },
        "registers_by_side": registers_by_side,
        "action": list(action),
        "position_gates_by_side": position_gates,
        "command_envelopes_by_side": envelopes_by_side,
        "elapsed_s": time.monotonic() - started,
        "bus_access": "read_only_monitor_connection",
        "selected_register_reads": (
            len(READ_ONLY_REGISTERS) * len(BENCH_MOTORS) * 2
        ),
        "leader_position_values_read": 12,
        "serial_register_writes": 0,
        "torque_enable_commands": 0,
        "physical_output_commands": False,
        "operator_estop_attestation": "not_requested_read_only_preflight",
        "not_authorized": ["physical_motor_output", "dual_arm_haptics"],
        "candidate_requires_single_arm_empirical_acceptance": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("dual-arm read-only haptic preflight failed")


if __name__ == "__main__":
    main()
