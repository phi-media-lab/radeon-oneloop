#!/usr/bin/env python3
"""Read two calibrated SO-101 leaders and publish the frozen 12-DoF action."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from radeon_oneloop.contracts import ACTION_NAMES

from .haptic_arm_hardware import FeetechHapticArmRenderer, HapticArmConfig
from .haptic_arm_readonly_preflight import (
    arm_command_envelope,
    evaluate_arm_hardware_snapshot,
)
from .haptic_hardware import (
    BENCH_MOTORS,
    FeetechHapticBenchRenderer,
    HapticBenchConfig,
    HapticHardwareError,
)
from .haptic_intervention import StableSafePoseConfig, StableSafePoseGate
from .haptic_readonly_preflight import (
    READ_ONLY_REGISTERS,
    command_envelope as single_joint_command_envelope,
    evaluate_hardware_snapshot as evaluate_single_joint_hardware_snapshot,
    read_register_snapshot,
)
from .haptic_safety import HapticSafetyConfig, SafeHapticController
from .leader_hardware import connect_read_only, make_leader, read_arm
from .live_protocol import (
    MAX_PACKET_BYTES,
    SO101_MODEL_ACTION_MAX,
    SO101_MODEL_ACTION_MIN,
    LeaderActionPacket,
    LiveProtocolError,
    decode_haptic_packet,
    encode_packet,
)


# Preserve the private names for compatibility with earlier local tooling.
_make_leader = make_leader
_connect_read_only = connect_read_only
_read_arm = read_arm


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class ActionRangeTracker:
    """Accumulate a compact, auditable range summary for the 12-DoF stream."""

    def __init__(self) -> None:
        self.minimum: list[float] | None = None
        self.maximum: list[float] | None = None

    def update(self, action: tuple[float, ...]) -> None:
        if len(action) != len(ACTION_NAMES):
            raise ValueError(f"action must contain {len(ACTION_NAMES)} values")
        if self.minimum is None:
            self.minimum = list(action)
            self.maximum = list(action)
            return
        assert self.maximum is not None
        for index, value in enumerate(action):
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)

    def as_dict(self) -> dict[str, object]:
        if self.minimum is None or self.maximum is None:
            return {
                "action_names": list(ACTION_NAMES),
                "minimum": None,
                "maximum": None,
                "span": None,
            }
        return {
            "action_names": list(ACTION_NAMES),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "span": [
                maximum - minimum
                for minimum, maximum in zip(
                    self.minimum, self.maximum, strict=True
                )
            ],
        }


def _build_same_process_preflight(
    *,
    output_mode: str,
    side: str,
    motor: str | None,
    selected_bus: Any,
    action: tuple[float, ...],
    intervention_gate: StableSafePoseGate,
    now_ns: int,
    elapsed_s: float,
    simulated_effort_full_scale: float,
    reaction_effort: float,
    max_torque_limit_raw: int,
    max_position_offset_deg: float,
) -> dict[str, object]:
    common: dict[str, object] = {
        "formal": False,
        "action": list(action),
        "elapsed_s": elapsed_s,
        "bus_access": "same_process_read_only_intervention_transition",
        "leader_position_values_read": 12,
        "serial_register_writes": 0,
        "torque_enable_commands": 0,
        "physical_output_commands": False,
        "operator_estop_attestation": "received_by_guarded_runner",
        "same_process_transition": True,
        "intervention": intervention_gate.as_dict(now_ns=now_ns),
        "not_authorized": ["dual_arm_haptics"],
    }
    if output_mode == "bench-single-joint":
        if motor is None:
            raise HapticHardwareError("single-joint intervention requires a motor")
        registers = read_register_snapshot(selected_bus, motor)
        envelope = single_joint_command_envelope(
            side=side,
            motor=motor,
            simulated_effort_full_scale=simulated_effort_full_scale,
            reaction_effort=reaction_effort,
            max_torque_limit_raw=max_torque_limit_raw,
            max_position_offset_deg=max_position_offset_deg,
        )
        checks = evaluate_single_joint_hardware_snapshot(
            action=action,
            registers=registers,
            envelope=envelope,
        )
        selected = int(envelope["selected_action_index"])
        model_limit_margin_deg = 5.0
        return {
            **common,
            "schema_version": "radeon_oneloop.haptic_readonly_preflight.v1",
            "accepted": all(checks.values()),
            "checks": checks,
            "selection": {"side": side, "motor": motor},
            "registers": registers,
            "selected_position_gate": {
                "position_deg": float(action[selected]),
                "model_min_deg": SO101_MODEL_ACTION_MIN[selected],
                "model_max_deg": SO101_MODEL_ACTION_MAX[selected],
                "required_bidirectional_margin_deg": model_limit_margin_deg,
                "candidate_max_abs_offset_deg": max_position_offset_deg,
                "accepted_position_range_deg": [
                    SO101_MODEL_ACTION_MIN[selected]
                    + model_limit_margin_deg
                    + max_position_offset_deg,
                    SO101_MODEL_ACTION_MAX[selected]
                    - model_limit_margin_deg
                    - max_position_offset_deg,
                ],
            },
            "command_envelope": envelope,
            "selected_register_reads": len(READ_ONLY_REGISTERS),
            "not_authorized": [
                "single_arm_haptics",
                "dual_arm_haptics",
            ],
        }
    if output_mode != "physical-single-arm":
        raise HapticHardwareError(
            f"unsupported intervention output mode: {output_mode}"
        )
    registers_by_motor = {
        selected_motor: read_register_snapshot(selected_bus, selected_motor)
        for selected_motor in BENCH_MOTORS
    }
    envelope = arm_command_envelope(
        side=side,
        simulated_effort_full_scale=simulated_effort_full_scale,
        reaction_effort=reaction_effort,
        max_torque_limit_raw=max_torque_limit_raw,
        max_position_offset_deg=max_position_offset_deg,
    )
    checks, position_gates = evaluate_arm_hardware_snapshot(
        action=action,
        side=side,
        registers_by_motor=registers_by_motor,
        envelope=envelope,
    )
    return {
        **common,
        "schema_version": "radeon_oneloop.haptic_arm_readonly_preflight.v1",
        "stage": "single_arm_readonly_preflight",
        "accepted": all(checks.values()),
        "checks": checks,
        "selection": {"side": side, "motors": list(BENCH_MOTORS)},
        "registers_by_motor": registers_by_motor,
        "position_gates_by_motor": position_gates,
        "command_envelope": envelope,
        "selected_register_reads": len(READ_ONLY_REGISTERS) * len(BENCH_MOTORS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-id", required=True)
    parser.add_argument("--right-id", required=True)
    parser.add_argument("--destination-host", default="127.0.0.1")
    parser.add_argument("--destination-port", type=int, default=58081)
    parser.add_argument("--feedback-bind-host")
    parser.add_argument("--feedback-port", type=int, default=58082)
    parser.add_argument("--feedback-source-host")
    parser.add_argument(
        "--haptic-output-mode",
        choices=("monitor", "bench-single-joint", "physical-single-arm"),
        default="monitor",
    )
    parser.add_argument("--haptic-bench-side", choices=("left", "right"))
    parser.add_argument("--haptic-bench-motor", choices=BENCH_MOTORS)
    parser.add_argument("--haptic-max-torque-limit-raw", type=int, default=30)
    parser.add_argument("--haptic-max-position-offset-deg", type=float, default=1.0)
    parser.add_argument(
        "--haptic-simulated-effort-full-scale", type=float, default=3.35
    )
    parser.add_argument("--haptic-max-output-duration-s", type=float, default=10.0)
    parser.add_argument("--haptic-health-hz", type=float, default=5.0)
    parser.add_argument("--physical-estop-confirmed", action="store_true")
    parser.add_argument("--intervention-assisted-arm", action="store_true")
    parser.add_argument("--intervention-stable-duration-s", type=float, default=0.4)
    parser.add_argument("--intervention-max-span-deg", type=float, default=2.0)
    parser.add_argument("--intervention-timeout-s", type=float, default=90.0)
    parser.add_argument("--intervention-ready-file", type=Path)
    parser.add_argument("--intervention-preflight-output", type=Path)
    parser.add_argument("--haptic-test-reaction-effort", type=float)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="Zero streams until interrupted."
    )
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--action-range-start-file",
        type=Path,
        help="If set, collect action ranges only after this file exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Read and validate without sending UDP."
    )
    args = parser.parse_args()
    if args.left_port == args.right_port:
        raise ValueError("left and right ports must differ")
    if not 1.0 <= args.hz <= 120.0:
        raise ValueError("hz must be between 1 and 120")
    if args.duration_s < 0:
        raise ValueError("duration-s must be non-negative")
    if args.print_every < 0:
        raise ValueError("print-every must be non-negative")
    if not 1 <= args.destination_port <= 65535:
        raise ValueError("destination-port must be between 1 and 65535")
    if not 1 <= args.feedback_port <= 65535:
        raise ValueError("feedback-port must be between 1 and 65535")
    if not 1.0 <= args.haptic_health_hz <= 10.0:
        raise ValueError("haptic-health-hz must be between 1 and 10")
    if not 0.01 <= args.haptic_simulated_effort_full_scale <= 10.0:
        raise ValueError("haptic simulated effort full scale must be in [0.01, 10]")
    if args.haptic_output_mode in ("bench-single-joint", "physical-single-arm"):
        if args.feedback_bind_host is None:
            raise ValueError("physical haptics require the feedback UDP listener")
        if args.haptic_bench_side is None:
            raise ValueError("physical haptics require one explicit side")
        if not args.physical_estop_confirmed:
            raise ValueError("physical haptics require --physical-estop-confirmed")
        if not 0.0 < args.haptic_max_output_duration_s <= 10.0:
            raise ValueError("physical haptic output must be time-bounded to 10 seconds")
    if args.haptic_output_mode == "bench-single-joint":
        if args.haptic_bench_motor is None:
            raise ValueError("bench haptics require one explicit motor")
        if not 0.0 < args.haptic_max_position_offset_deg <= 1.0:
            raise ValueError("first-bench position offset must be in (0, 1] degree")
    if args.haptic_output_mode == "physical-single-arm":
        if args.haptic_bench_motor is not None:
            raise ValueError("single-arm haptics do not accept one bench motor")
        if not 1 <= args.haptic_max_torque_limit_raw <= 20:
            raise ValueError("single-arm torque limit must be in [1, 20]")
        if not 0.0 < args.haptic_max_position_offset_deg <= 0.5:
            raise ValueError("single-arm position offset must be in (0, 0.5] degree")
        if not 0.0 < args.haptic_max_output_duration_s <= 5.0:
            raise ValueError("first single-arm output must be bounded to 5 seconds")
    intervention_paths = (
        args.intervention_ready_file,
        args.intervention_preflight_output,
    )
    if args.intervention_assisted_arm:
        if args.haptic_output_mode not in (
            "bench-single-joint",
            "physical-single-arm",
        ):
            raise ValueError(
                "intervention-assisted arming requires a physical haptic mode"
            )
        if any(path is None for path in intervention_paths):
            raise ValueError(
                "intervention-assisted arming requires ready and preflight outputs"
            )
        if args.haptic_test_reaction_effort is None or not (
            0.0 < abs(args.haptic_test_reaction_effort) <= 3.35
        ):
            raise ValueError(
                "intervention-assisted arming requires a bounded test reaction effort"
            )
        if not 5.0 <= args.intervention_timeout_s <= 120.0:
            raise ValueError("intervention timeout must be in [5, 120] seconds")
        if any(path.exists() for path in intervention_paths if path is not None):
            raise ValueError("intervention output paths must not already exist")
    elif any(path is not None for path in intervention_paths):
        raise ValueError(
            "intervention output paths require --intervention-assisted-arm"
        )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    leaders = [
        _make_leader(args.left_port, args.left_id),
        _make_leader(args.right_port, args.right_id),
    ]
    connected: list[Any] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.feedback_bind_host is not None:
        sock.bind((args.feedback_bind_host, args.feedback_port))
        sock.setblocking(False)
    destination = (args.destination_host, args.destination_port)
    samples = 0
    send_errors = 0
    read_times_ms: list[float] = []
    action_range = ActionRangeTracker()
    action_range_samples = 0
    action_range_started_monotonic_ns: int | None = None
    feedback_accepted = 0
    feedback_rejected = 0
    feedback_last_sequence_id: int | None = None
    feedback_last_arrival_ns: int | None = None
    feedback_max_contact_force_n = [0.0, 0.0]
    feedback_max_joint_reaction_effort = 0.0
    physical_output_armed_ever = False
    physical_output_armed_ns: int | None = None
    latest_health: dict[str, int] | None = None
    last_health_check_ns: int | None = None
    haptic_controller: SafeHapticController | None = None
    haptic_renderer: FeetechHapticBenchRenderer | FeetechHapticArmRenderer | None = None
    intervention_gate: StableSafePoseGate | None = None
    intervention_candidate_announced = False
    intervention_preflight: dict[str, object] | None = None
    shutdown_error: str | None = None
    if args.haptic_output_mode == "bench-single-joint":
        bench_config = HapticBenchConfig(
            side=args.haptic_bench_side,
            motor=args.haptic_bench_motor,
            max_torque_limit_raw=args.haptic_max_torque_limit_raw,
        )
        haptic_controller = SafeHapticController(
            HapticSafetyConfig(
                max_torque_limit_raw=bench_config.max_torque_limit_raw,
                max_position_offset_deg=args.haptic_max_position_offset_deg,
                simulated_effort_full_scale=(
                    args.haptic_simulated_effort_full_scale
                ),
            )
        )
        leader_index = 0 if bench_config.side == "left" else 1
        haptic_renderer = FeetechHapticBenchRenderer(
            leaders[leader_index].bus, bench_config
        )
        if args.intervention_assisted_arm:
            intervention_gate = StableSafePoseGate(
                StableSafePoseConfig(
                    side=bench_config.side,
                    motors=(bench_config.motor,),
                    hold_s=args.intervention_stable_duration_s,
                    max_span_deg=args.intervention_max_span_deg,
                    max_position_offset_deg=args.haptic_max_position_offset_deg,
                )
            )
    elif args.haptic_output_mode == "physical-single-arm":
        arm_config = HapticArmConfig(
            side=args.haptic_bench_side,
            max_torque_limit_raw=args.haptic_max_torque_limit_raw,
        )
        haptic_controller = SafeHapticController(
            HapticSafetyConfig(
                max_torque_limit_raw=arm_config.max_torque_limit_raw,
                max_position_offset_deg=args.haptic_max_position_offset_deg,
                simulated_effort_full_scale=(
                    args.haptic_simulated_effort_full_scale
                ),
            )
        )
        leader_index = 0 if arm_config.side == "left" else 1
        haptic_renderer = FeetechHapticArmRenderer(
            leaders[leader_index].bus, arm_config
        )
        if args.intervention_assisted_arm:
            intervention_gate = StableSafePoseGate(
                StableSafePoseConfig(
                    side=arm_config.side,
                    motors=BENCH_MOTORS,
                    hold_s=args.intervention_stable_duration_s,
                    max_span_deg=args.intervention_max_span_deg,
                    max_position_offset_deg=args.haptic_max_position_offset_deg,
                )
            )
    started = time.monotonic()
    period_s = 1.0 / args.hz
    next_tick = started
    try:
        for leader in leaders:
            _connect_read_only(leader)
            connected.append(leader)

        while not stop_requested:
            if args.duration_s and time.monotonic() - started >= args.duration_s:
                break
            if (
                physical_output_armed_ns is not None
                and (time.monotonic_ns() - physical_output_armed_ns) / 1_000_000_000
                >= args.haptic_max_output_duration_s
            ):
                break
            read_started = time.perf_counter()
            left = _read_arm(leaders[0])
            right = _read_arm(leaders[1])
            current_action = left + right
            now_ns = time.monotonic_ns()
            intervention_candidate_ready = True
            if (
                intervention_gate is not None
                and haptic_renderer is not None
                and not haptic_renderer.armed
            ):
                intervention_candidate_ready = intervention_gate.update(
                    current_action, now_ns=now_ns
                )
                if time.monotonic() - started >= args.intervention_timeout_s:
                    raise HapticHardwareError(
                        "timed out before same-process intervention arming"
                    )
                if (
                    intervention_candidate_ready
                    and not intervention_candidate_announced
                ):
                    assert args.intervention_ready_file is not None
                    _write_json_atomic(
                        args.intervention_ready_file,
                        {
                            "schema_version": (
                                "radeon_oneloop.haptic_intervention_ready.v1"
                            ),
                            "candidate_ready": True,
                            "physical_output_commands": False,
                            "intervention": intervention_gate.as_dict(now_ns=now_ns),
                        },
                    )
                    intervention_candidate_announced = True
            if (
                args.action_range_start_file is None
                or args.action_range_start_file.is_file()
            ):
                if action_range_started_monotonic_ns is None:
                    action_range_started_monotonic_ns = time.monotonic_ns()
                action_range.update(current_action)
                action_range_samples += 1
            packet = LeaderActionPacket(
                sequence_id=samples,
                captured_monotonic_ns=time.monotonic_ns(),
                captured_unix_ns=time.time_ns(),
                action=current_action,
            )
            read_times_ms.append((time.perf_counter() - read_started) * 1000.0)
            if not args.dry_run:
                try:
                    sock.sendto(encode_packet(packet), destination)
                except OSError:
                    send_errors += 1
                    raise
            if args.feedback_bind_host is not None:
                latest_feedback = None
                latest_feedback_arrival_ns = None
                while True:
                    try:
                        feedback_payload, feedback_source = sock.recvfrom(
                            MAX_PACKET_BYTES + 1
                        )
                    except BlockingIOError:
                        break
                    try:
                        if (
                            args.feedback_source_host is not None
                            and feedback_source[0] != args.feedback_source_host
                        ):
                            raise LiveProtocolError(
                                f"unexpected haptic source {feedback_source[0]!r}"
                            )
                        feedback = decode_haptic_packet(feedback_payload)
                        if (
                            feedback_last_sequence_id is not None
                            and feedback.sequence_id <= feedback_last_sequence_id
                        ):
                            raise LiveProtocolError(
                                "non-monotonic haptic sequence_id "
                                f"{feedback.sequence_id} after "
                                f"{feedback_last_sequence_id}"
                            )
                    except LiveProtocolError:
                        feedback_rejected += 1
                        continue
                    feedback_last_sequence_id = feedback.sequence_id
                    feedback_last_arrival_ns = time.monotonic_ns()
                    latest_feedback = feedback
                    latest_feedback_arrival_ns = feedback_last_arrival_ns
                    feedback_accepted += 1
                    feedback_max_joint_reaction_effort = max(
                        feedback_max_joint_reaction_effort,
                        max(abs(value) for value in feedback.joint_reaction_effort),
                    )
                    for arm_index, value in enumerate(feedback.contact_force_n):
                        feedback_max_contact_force_n[arm_index] = max(
                            feedback_max_contact_force_n[arm_index], value
                        )
                if haptic_renderer is not None and haptic_controller is not None:
                    if latest_feedback is not None:
                        assert latest_feedback_arrival_ns is not None
                        if not haptic_renderer.armed:
                            if not intervention_candidate_ready:
                                # Feedback cannot arm the bus until the operator's
                                # pose is currently safe and stable in this process.
                                latest_feedback = None
                                continue
                            if intervention_gate is not None:
                                assert args.haptic_test_reaction_effort is not None
                                assert args.intervention_preflight_output is not None
                                intervention_preflight = (
                                    _build_same_process_preflight(
                                        output_mode=args.haptic_output_mode,
                                        side=args.haptic_bench_side,
                                        motor=args.haptic_bench_motor,
                                        selected_bus=leaders[leader_index].bus,
                                        action=current_action,
                                        intervention_gate=intervention_gate,
                                        now_ns=now_ns,
                                        elapsed_s=time.monotonic() - started,
                                        simulated_effort_full_scale=(
                                            args.haptic_simulated_effort_full_scale
                                        ),
                                        reaction_effort=(
                                            args.haptic_test_reaction_effort
                                        ),
                                        max_torque_limit_raw=(
                                            args.haptic_max_torque_limit_raw
                                        ),
                                        max_position_offset_deg=(
                                            args.haptic_max_position_offset_deg
                                        ),
                                    )
                                )
                                _write_json_atomic(
                                    args.intervention_preflight_output,
                                    intervention_preflight,
                                )
                                if not intervention_preflight["accepted"]:
                                    raise HapticHardwareError(
                                        "same-process intervention preflight failed"
                                    )
                            haptic_controller.arm(
                                physical_estop_confirmed=args.physical_estop_confirmed
                            )
                            haptic_renderer.arm(
                                current_action,
                                physical_estop_confirmed=args.physical_estop_confirmed,
                            )
                            physical_output_armed_ever = True
                            physical_output_armed_ns = time.monotonic_ns()
                            # arm() has already sampled all health registers;
                            # avoid an immediate duplicate transaction burst.
                            last_health_check_ns = physical_output_armed_ns
                        haptic_command = haptic_controller.update(
                            latest_feedback,
                            arrival_monotonic_ns=latest_feedback_arrival_ns,
                        )
                        haptic_renderer.apply(haptic_command, current_action)
                    elif haptic_renderer.armed:
                        watchdog_command = haptic_controller.watchdog(
                            now_monotonic_ns=time.monotonic_ns()
                        )
                        if watchdog_command is not None:
                            haptic_renderer.apply(watchdog_command, current_action)
                            haptic_renderer.emergency_release()
                            raise HapticHardwareError("haptic feedback watchdog expired")

                    now_ns = time.monotonic_ns()
                    health_interval_ns = int(1_000_000_000 / args.haptic_health_hz)
                    if haptic_renderer.armed and (
                        last_health_check_ns is None
                        or now_ns - last_health_check_ns >= health_interval_ns
                    ):
                        try:
                            latest_health = haptic_renderer.check_health()
                        except Exception:
                            haptic_controller.latch_estop()
                            haptic_renderer.emergency_release()
                            raise
                        last_health_check_ns = now_ns
            samples += 1
            if args.print_every and samples % args.print_every == 0:
                print(json.dumps(packet.as_dict(), separators=(",", ":")), flush=True)

            next_tick += period_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period_s:
                next_tick = time.monotonic()
    finally:
        if haptic_renderer is not None:
            try:
                haptic_renderer.close()
            except Exception as exc:
                haptic_renderer.emergency_release()
                print(f"haptic renderer shutdown error: {exc}", file=sys.stderr)
                shutdown_error = str(exc)
        for leader in reversed(connected):
            if leader.bus.is_connected:
                leader.bus.disconnect(disable_torque=False)
        sock.close()

    elapsed_s = max(time.monotonic() - started, 1e-9)
    report = {
        "schema_version": "radeon_oneloop.leader_publisher.v1",
        "dry_run": args.dry_run,
        "destination": f"{args.destination_host}:{args.destination_port}",
        "requested_hz": args.hz,
        "samples": samples,
        "effective_hz": samples / elapsed_s,
        "send_errors": send_errors,
        "read_ms": {
            "mean": sum(read_times_ms) / len(read_times_ms) if read_times_ms else None,
            "max": max(read_times_ms) if read_times_ms else None,
        },
        "action_range": {
            **action_range.as_dict(),
            "samples": action_range_samples,
            "capture_start_gated": args.action_range_start_file is not None,
            "capture_started": action_range_started_monotonic_ns is not None,
        },
        "haptic_feedback": {
            "mode": (
                args.haptic_output_mode
                if args.feedback_bind_host is not None
                else "off"
            ),
            "accepted": feedback_accepted,
            "rejected": feedback_rejected,
            "last_sequence_id": feedback_last_sequence_id,
            "age_ms_at_shutdown": (
                (time.monotonic_ns() - feedback_last_arrival_ns) / 1_000_000.0
                if feedback_last_arrival_ns is not None
                else None
            ),
            "max_contact_force_n": feedback_max_contact_force_n,
            "max_abs_joint_reaction_effort": feedback_max_joint_reaction_effort,
            "bench_selection": (
                {
                    "side": args.haptic_bench_side,
                    "motor": args.haptic_bench_motor,
                    "max_torque_limit_raw": args.haptic_max_torque_limit_raw,
                    "max_position_offset_deg": args.haptic_max_position_offset_deg,
                    "simulated_effort_full_scale": (
                        args.haptic_simulated_effort_full_scale
                    ),
                    "max_output_duration_s": args.haptic_max_output_duration_s,
                }
                if args.haptic_output_mode == "bench-single-joint"
                else None
            ),
            "arm_selection": (
                {
                    "side": args.haptic_bench_side,
                    "motors": list(BENCH_MOTORS),
                    "max_torque_limit_raw": args.haptic_max_torque_limit_raw,
                    "max_position_offset_deg": args.haptic_max_position_offset_deg,
                    "simulated_effort_full_scale": (
                        args.haptic_simulated_effort_full_scale
                    ),
                    "max_output_duration_s": args.haptic_max_output_duration_s,
                }
                if args.haptic_output_mode == "physical-single-arm"
                else None
            ),
            "intervention": (
                intervention_preflight.get("intervention")
                if intervention_preflight is not None
                else None
            ),
            "output_armed_ever": physical_output_armed_ever,
            "output_commands": (
                haptic_renderer.output_commands if haptic_renderer is not None else 0
            ),
            "latest_health": latest_health,
            "peak_abs_current_raw": (
                haptic_renderer.peak_abs_current_raw
                if haptic_renderer is not None
                else 0
            ),
            "peak_temperature_c": (
                haptic_renderer.peak_temperature_c
                if haptic_renderer is not None
                else 0
            ),
            "physical_output_commands": physical_output_armed_ever,
            "shutdown_error": shutdown_error,
            "release_attempted": (
                haptic_renderer.release_attempted
                if haptic_renderer is not None
                else False
            ),
            "release_verified": (
                haptic_renderer.release_verified
                if haptic_renderer is not None
                else False
            ),
            "restore_verified": (
                haptic_renderer.restore_verified
                if haptic_renderer is not None
                else False
            ),
            "output_armed_at_shutdown": (
                haptic_renderer.armed if haptic_renderer is not None else False
            ),
        },
        "physical_output_commands": physical_output_armed_ever,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(payload, encoding="utf-8")
    print(payload, end="", flush=True)
    if shutdown_error is not None:
        raise HapticHardwareError(f"haptic shutdown verification failed: {shutdown_error}")


if __name__ == "__main__":
    main()
