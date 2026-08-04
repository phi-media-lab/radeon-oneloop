#!/usr/bin/env python3
"""Read-only five-joint preflight for a candidate single-arm haptic stage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from .haptic_hardware import ARM_MOTORS, BENCH_MOTORS
from .haptic_readonly_preflight import READ_ONLY_REGISTERS, read_register_snapshot
from .haptic_safety import HapticSafetyConfig, SafeHapticController
from .leader_publisher import _connect_read_only, _make_leader, _read_arm
from .live_protocol import (
    HapticFeedbackPacket,
    SO101_MODEL_ACTION_MAX,
    SO101_MODEL_ACTION_MIN,
)


def arm_command_envelope(
    *,
    side: str,
    simulated_effort_full_scale: float,
    reaction_effort: float,
    max_torque_limit_raw: int = 20,
    max_position_offset_deg: float = 0.5,
) -> dict[str, Any]:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    controller = SafeHapticController(
        HapticSafetyConfig(
            simulated_effort_full_scale=simulated_effort_full_scale,
            max_torque_limit_raw=max_torque_limit_raw,
            max_position_offset_deg=max_position_offset_deg,
        )
    )
    controller.arm(physical_estop_confirmed=True)
    side_offset = 0 if side == "left" else len(ARM_MOTORS)
    selected_indices = tuple(side_offset + index for index in range(len(BENCH_MOTORS)))
    efforts = [0.0] * 12
    for index in selected_indices:
        efforts[index] = float(reaction_effort)
    force = (2.0, 0.0) if side == "left" else (0.0, 2.0)
    commands = []
    last_arrival_ns = 1_000_000_000
    for sequence_id in range(1, 13):
        last_arrival_ns += 20_000_000
        packet = HapticFeedbackPacket(
            sequence_id=sequence_id,
            captured_monotonic_ns=sequence_id,
            captured_unix_ns=sequence_id,
            joint_reaction_effort=tuple(efforts),
            contact_force_n=force,
        )
        commands.append(
            controller.update(packet, arrival_monotonic_ns=last_arrival_ns)
        )
    fail_zero = controller.watchdog(now_monotonic_ns=last_arrival_ns + 101_000_000)
    assert fail_zero is not None
    observed_offset_by_motor = {
        motor: max(
            abs(command.position_offset_deg[index]) for command in commands
        )
        for motor, index in zip(BENCH_MOTORS, selected_indices, strict=True)
    }
    observed_torque_by_motor = {
        motor: max(command.torque_limit_raw[index] for command in commands)
        for motor, index in zip(BENCH_MOTORS, selected_indices, strict=True)
    }
    unselected = set(range(12)) - set(selected_indices)
    expected_normalized = min(
        abs(reaction_effort) / simulated_effort_full_scale,
        controller.config.max_normalized_effort,
    )
    expected_offset = expected_normalized * max_position_offset_deg
    checks = {
        "five_non_gripper_channels_selected": len(selected_indices) == 5,
        "unselected_channels_zero": all(
            all(
                command.position_offset_deg[index] == 0.0
                and command.torque_limit_raw[index] == 0
                for index in unselected
            )
            for command in commands
        ),
        "offsets_within_limit": all(
            value <= max_position_offset_deg
            for value in observed_offset_by_motor.values()
        ),
        "torques_within_limit": all(
            value <= max_torque_limit_raw
            for value in observed_torque_by_motor.values()
        ),
        "steady_state_matches_calibration": all(
            math.isclose(value, expected_offset, rel_tol=0.0, abs_tol=1e-12)
            for value in observed_offset_by_motor.values()
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
        "selected_action_indices": list(selected_indices),
        "selected_motors": list(BENCH_MOTORS),
        "reaction_effort": reaction_effort,
        "simulated_effort_full_scale": simulated_effort_full_scale,
        "max_observed_position_offset_deg_by_motor": observed_offset_by_motor,
        "max_position_offset_limit_deg": max_position_offset_deg,
        "max_observed_torque_limit_raw_by_motor": observed_torque_by_motor,
        "max_torque_limit_raw": max_torque_limit_raw,
        "watchdog_ms": controller.config.watchdog_ms,
    }


def evaluate_arm_hardware_snapshot(
    *,
    action: Sequence[float],
    side: str,
    registers_by_motor: dict[str, dict[str, int]],
    envelope: dict[str, Any],
    model_limit_margin_deg: float = 5.0,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    side_offset = 0 if side == "left" else len(ARM_MOTORS)
    max_offset = float(envelope.get("max_position_offset_limit_deg", math.nan))
    per_motor: dict[str, dict[str, Any]] = {}
    for motor_index, motor in enumerate(BENCH_MOTORS):
        action_index = side_offset + motor_index
        registers = registers_by_motor.get(motor, {})
        position = float(action[action_index]) if len(action) == 12 else math.nan
        has_margin = (
            math.isfinite(position)
            and math.isfinite(max_offset)
            and position - max_offset
            >= SO101_MODEL_ACTION_MIN[action_index] + model_limit_margin_deg
            and position + max_offset
            <= SO101_MODEL_ACTION_MAX[action_index] - model_limit_margin_deg
        )
        per_motor[motor] = {
            "action_index": action_index,
            "position_deg": position,
            "accepted_position_range_deg": [
                SO101_MODEL_ACTION_MIN[action_index]
                + model_limit_margin_deg
                + max_offset,
                SO101_MODEL_ACTION_MAX[action_index]
                - model_limit_margin_deg
                - max_offset,
            ],
            "torque_disabled": registers.get("Torque_Enable") == 0,
            "position_mode": registers.get("Operating_Mode") == 0,
            "bidirectional_model_margin": has_margin,
            "current_within_bound": (
                abs(registers.get("Present_Current", 10_000)) <= 150
            ),
            "temperature_within_bound": (
                registers.get("Present_Temperature", 10_000) <= 45
            ),
            "voltage_within_bound": (
                60 <= registers.get("Present_Voltage", -1) <= 84
            ),
            "status_clear": registers.get("Status") == 0,
        }
    checks = {
        "dual_arm_action_finite": (
            len(action) == 12
            and all(math.isfinite(float(value)) for value in action)
        ),
        "all_selected_motors_present": set(registers_by_motor) == set(BENCH_MOTORS),
        "all_selected_motors_torque_disabled": all(
            value["torque_disabled"] for value in per_motor.values()
        ),
        "all_selected_motors_position_mode": all(
            value["position_mode"] for value in per_motor.values()
        ),
        "all_selected_positions_have_bidirectional_model_margin": all(
            value["bidirectional_model_margin"] for value in per_motor.values()
        ),
        "all_selected_currents_within_bound": all(
            value["current_within_bound"] for value in per_motor.values()
        ),
        "all_selected_temperatures_within_bound": all(
            value["temperature_within_bound"] for value in per_motor.values()
        ),
        "all_selected_voltages_within_bound": all(
            value["voltage_within_bound"] for value in per_motor.values()
        ),
        "all_selected_status_clear": all(
            value["status_clear"] for value in per_motor.values()
        ),
        "synthetic_arm_command_envelope_accepted": envelope.get("accepted") is True,
    }
    return checks, per_motor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-id", required=True)
    parser.add_argument("--right-id", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--simulated-effort-full-scale", type=float, required=True)
    parser.add_argument("--reaction-effort", type=float, required=True)
    parser.add_argument("--max-torque-limit-raw", type=int, default=20)
    parser.add_argument("--max-position-offset-deg", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.left_port == args.right_port:
        raise ValueError("left and right ports must differ")
    if not 0.0 < args.max_position_offset_deg <= 0.5:
        raise ValueError("arm preflight offset limit must be in (0, 0.5] degree")
    if not 1 <= args.max_torque_limit_raw <= 20:
        raise ValueError("arm preflight torque candidate must be in [1, 20]")

    envelope = arm_command_envelope(
        side=args.side,
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
        registers_by_motor = {
            motor: read_register_snapshot(selected_leader.bus, motor)
            for motor in BENCH_MOTORS
        }
    finally:
        for leader in reversed(connected):
            if leader.bus.is_connected:
                leader.bus.disconnect(disable_torque=False)

    checks, per_motor = evaluate_arm_hardware_snapshot(
        action=action,
        side=args.side,
        registers_by_motor=registers_by_motor,
        envelope=envelope,
    )
    report = {
        "schema_version": "radeon_oneloop.haptic_arm_readonly_preflight.v1",
        "formal": False,
        "stage": "single_arm_readonly_preflight",
        "accepted": all(checks.values()),
        "checks": checks,
        "selection": {"side": args.side, "motors": list(BENCH_MOTORS)},
        "registers_by_motor": registers_by_motor,
        "action": list(action),
        "position_gates_by_motor": per_motor,
        "command_envelope": envelope,
        "elapsed_s": time.monotonic() - started,
        "bus_access": "read_only_monitor_connection",
        "selected_register_reads": len(READ_ONLY_REGISTERS) * len(BENCH_MOTORS),
        "leader_position_values_read": 12,
        "serial_register_writes": 0,
        "torque_enable_commands": 0,
        "physical_output_commands": False,
        "operator_estop_attestation": "not_requested_read_only_preflight",
        "not_authorized": ["physical_motor_output", "dual_arm_haptics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("single-arm read-only haptic preflight failed")


if __name__ == "__main__":
    main()
