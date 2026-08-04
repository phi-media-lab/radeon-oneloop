#!/usr/bin/env python3
"""Read-only hardware and command-envelope preflight for one haptic joint.

This command opens the two calibrated leader buses exactly like the monitor
publisher, reads positions and health registers, and disconnects without
writing calibration, torque, limits, modes, or goals.  The candidate feedback
mapping is exercised only through the pure ``SafeHapticController`` kernel.
It cannot authorize or perform physical output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from .haptic_hardware import BENCH_MOTORS, HapticBenchConfig
from .haptic_safety import HapticSafetyConfig, SafeHapticController
from .leader_publisher import _connect_read_only, _make_leader, _read_arm
from .live_protocol import (
    HapticFeedbackPacket,
    SO101_MODEL_ACTION_MAX,
    SO101_MODEL_ACTION_MIN,
)


READ_ONLY_REGISTERS = (
    "Torque_Enable",
    "Operating_Mode",
    "Torque_Limit",
    "Present_Current",
    "Present_Temperature",
    "Present_Voltage",
    "Status",
)


def read_register_snapshot(bus: Any, motor: str) -> dict[str, int]:
    return {
        register: int(bus.read(register, motor, normalize=False, num_retry=2))
        for register in READ_ONLY_REGISTERS
    }


def command_envelope(
    *,
    side: str,
    motor: str,
    simulated_effort_full_scale: float,
    reaction_effort: float,
    max_torque_limit_raw: int = 30,
    max_position_offset_deg: float = 1.0,
) -> dict[str, Any]:
    bench = HapticBenchConfig(
        side=side,
        motor=motor,
        max_torque_limit_raw=max_torque_limit_raw,
    )
    controller = SafeHapticController(
        HapticSafetyConfig(
            simulated_effort_full_scale=simulated_effort_full_scale,
            max_torque_limit_raw=max_torque_limit_raw,
            max_position_offset_deg=max_position_offset_deg,
        )
    )
    # SafeHapticController is pure and opens no hardware.  This synthetic arm
    # call is only a way to evaluate its bounded post-arm command envelope; it
    # is not an operator attestation and is never forwarded to a renderer.
    controller.arm(physical_estop_confirmed=True)
    selected = bench.action_index
    no_contact = HapticFeedbackPacket(
        sequence_id=0,
        captured_monotonic_ns=1,
        captured_unix_ns=1,
        joint_reaction_effort=(0.0,) * 12,
        contact_force_n=(0.0, 0.0),
    )
    zero = controller.update(no_contact, arrival_monotonic_ns=1_000_000_000)
    efforts = [0.0] * 12
    efforts[selected] = float(reaction_effort)
    forces = (2.0, 0.0) if side == "left" else (0.0, 2.0)
    commands = []
    last_arrival_ns = 1_000_000_000
    for sequence_id in range(1, 13):
        last_arrival_ns += 20_000_000
        packet = HapticFeedbackPacket(
            sequence_id=sequence_id,
            captured_monotonic_ns=sequence_id + 1,
            captured_unix_ns=sequence_id + 1,
            joint_reaction_effort=tuple(efforts),
            contact_force_n=forces,
        )
        commands.append(
            controller.update(packet, arrival_monotonic_ns=last_arrival_ns)
        )
    fail_zero = controller.watchdog(now_monotonic_ns=last_arrival_ns + 101_000_000)
    assert fail_zero is not None
    max_offset = max(abs(command.position_offset_deg[selected]) for command in commands)
    max_torque = max(command.torque_limit_raw[selected] for command in commands)
    expected_normalized = min(
        abs(reaction_effort) / simulated_effort_full_scale,
        controller.config.max_normalized_effort,
    )
    expected_offset = expected_normalized * max_position_offset_deg
    checks = {
        "no_contact_is_zero": (
            zero.position_offset_deg == (0.0,) * 12
            and zero.torque_limit_raw == (0,) * 12
        ),
        "selected_channel_only": all(
            all(
                value == 0.0
                for index, value in enumerate(command.position_offset_deg)
                if index != selected
            )
            for command in commands
        ),
        "offset_within_limit": max_offset <= max_position_offset_deg,
        "torque_within_limit": max_torque <= max_torque_limit_raw,
        "steady_state_matches_calibration": math.isclose(
            max_offset, expected_offset, rel_tol=0.0, abs_tol=1e-12
        ),
        "watchdog_fails_zero": (
            fail_zero.enabled is False
            and fail_zero.position_offset_deg == (0.0,) * 12
            and fail_zero.torque_limit_raw == (0,) * 12
            and fail_zero.reason == "feedback_timeout"
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "selected_action_index": selected,
        "reaction_effort": reaction_effort,
        "simulated_effort_full_scale": simulated_effort_full_scale,
        "max_observed_position_offset_deg": max_offset,
        "max_position_offset_limit_deg": max_position_offset_deg,
        "max_observed_torque_limit_raw": max_torque,
        "watchdog_ms": controller.config.watchdog_ms,
    }


def evaluate_hardware_snapshot(
    *,
    action: Sequence[float],
    registers: dict[str, int],
    envelope: dict[str, Any],
    model_limit_margin_deg: float = 5.0,
) -> dict[str, bool]:
    selected = int(envelope.get("selected_action_index", -1))
    max_offset = float(envelope.get("max_position_offset_limit_deg", math.nan))
    position_has_margin = False
    if 0 <= selected < len(action) and math.isfinite(max_offset):
        position = float(action[selected])
        position_has_margin = (
            position - max_offset
            >= SO101_MODEL_ACTION_MIN[selected] + model_limit_margin_deg
            and position + max_offset
            <= SO101_MODEL_ACTION_MAX[selected] - model_limit_margin_deg
        )
    return {
        "dual_arm_action_finite": (
            len(action) == 12 and all(math.isfinite(float(value)) for value in action)
        ),
        "selected_motor_torque_disabled": registers.get("Torque_Enable") == 0,
        "selected_motor_position_mode": registers.get("Operating_Mode") == 0,
        "selected_position_has_bidirectional_model_margin": position_has_margin,
        "present_current_within_bound": abs(registers.get("Present_Current", 10_000))
        <= 150,
        "temperature_within_bound": registers.get("Present_Temperature", 10_000)
        <= 45,
        "voltage_within_bound": 60
        <= registers.get("Present_Voltage", -1)
        <= 84,
        "status_clear": registers.get("Status") == 0,
        "synthetic_command_envelope_accepted": envelope.get("accepted") is True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-id", required=True)
    parser.add_argument("--right-id", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--motor", choices=BENCH_MOTORS, required=True)
    parser.add_argument("--simulated-effort-full-scale", type=float, required=True)
    parser.add_argument("--reaction-effort", type=float, required=True)
    parser.add_argument("--max-torque-limit-raw", type=int, default=30)
    parser.add_argument("--max-position-offset-deg", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.left_port == args.right_port:
        raise ValueError("left and right ports must differ")
    if not 0.0 < args.max_position_offset_deg <= 1.0:
        raise ValueError("preflight offset limit must be in (0, 1] degree")

    envelope = command_envelope(
        side=args.side,
        motor=args.motor,
        simulated_effort_full_scale=args.simulated_effort_full_scale,
        reaction_effort=args.reaction_effort,
        max_torque_limit_raw=args.max_torque_limit_raw,
        max_position_offset_deg=args.max_position_offset_deg,
    )
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
        selected_leader = leaders[0] if args.side == "left" else leaders[1]
        registers = read_register_snapshot(selected_leader.bus, args.motor)
    finally:
        for leader in reversed(connected):
            if leader.bus.is_connected:
                leader.bus.disconnect(disable_torque=False)

    checks = evaluate_hardware_snapshot(
        action=action,
        registers=registers,
        envelope=envelope,
    )
    selected = int(envelope["selected_action_index"])
    model_limit_margin_deg = 5.0
    report = {
        "schema_version": "radeon_oneloop.haptic_readonly_preflight.v1",
        "formal": False,
        "accepted": all(checks.values()),
        "checks": checks,
        "selection": {"side": args.side, "motor": args.motor},
        "registers": registers,
        "action": list(action),
        "selected_position_gate": {
            "position_deg": float(action[selected]),
            "model_min_deg": SO101_MODEL_ACTION_MIN[selected],
            "model_max_deg": SO101_MODEL_ACTION_MAX[selected],
            "required_bidirectional_margin_deg": model_limit_margin_deg,
            "candidate_max_abs_offset_deg": args.max_position_offset_deg,
            "accepted_position_range_deg": [
                SO101_MODEL_ACTION_MIN[selected]
                + model_limit_margin_deg
                + args.max_position_offset_deg,
                SO101_MODEL_ACTION_MAX[selected]
                - model_limit_margin_deg
                - args.max_position_offset_deg,
            ],
        },
        "command_envelope": envelope,
        "elapsed_s": time.monotonic() - started,
        "bus_access": "read_only_monitor_connection",
        "selected_register_reads": len(READ_ONLY_REGISTERS),
        "leader_position_values_read": 12,
        "serial_register_writes": 0,
        "torque_enable_commands": 0,
        "physical_output_commands": False,
        "operator_estop_attestation": "not_requested_read_only_preflight",
        "not_authorized": [
            "physical_motor_output",
            "single_arm_haptics",
            "dual_arm_haptics",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("read-only haptic preflight failed")


if __name__ == "__main__":
    main()
