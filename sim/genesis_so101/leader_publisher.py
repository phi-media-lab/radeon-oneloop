#!/usr/bin/env python3
"""Read two calibrated SO-101 leaders and publish the frozen 12-DoF action."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from radeon_oneloop.contracts import ARM_JOINTS

from .haptic_hardware import (
    BENCH_MOTORS,
    FeetechHapticBenchRenderer,
    HapticBenchConfig,
    HapticHardwareError,
)
from .haptic_safety import HapticSafetyConfig, SafeHapticController
from .live_protocol import (
    MAX_PACKET_BYTES,
    LeaderActionPacket,
    LiveProtocolError,
    decode_haptic_packet,
    encode_packet,
)


def _make_leader(port: str, arm_id: str) -> Any:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    return SO101Leader(
        SO101LeaderConfig(port=port, id=arm_id, use_degrees=True)
    )


def _connect_read_only(leader: Any) -> None:
    if not leader.calibration:
        raise RuntimeError(
            f"missing calibration for {leader.id}: {leader.calibration_fpath}"
        )
    # Deliberately bypass SO101Leader.connect(): that method configures motor
    # registers. A leader bridge only needs to open the bus and read positions.
    leader.bus.connect()
    if not leader.bus.is_calibrated:
        raise RuntimeError(
            f"motor calibration does not match {leader.calibration_fpath}; "
            "refusing to write calibration from the live reader"
        )


def _read_arm(leader: Any) -> tuple[float, ...]:
    values = leader.get_action()
    return tuple(float(values[name]) for name in ARM_JOINTS)


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
        choices=("monitor", "bench-single-joint"),
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
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="Zero streams until interrupted."
    )
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--metrics-output", type=Path)
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
    if args.haptic_output_mode == "bench-single-joint":
        if args.feedback_bind_host is None:
            raise ValueError("bench haptics require the feedback UDP listener")
        if args.haptic_bench_side is None or args.haptic_bench_motor is None:
            raise ValueError("bench haptics require one explicit side and motor")
        if not args.physical_estop_confirmed:
            raise ValueError("bench haptics require --physical-estop-confirmed")
        if not 0.0 < args.haptic_max_output_duration_s <= 10.0:
            raise ValueError("physical haptic output must be time-bounded to 10 seconds")
        if not 0.0 < args.haptic_max_position_offset_deg <= 1.0:
            raise ValueError("first-bench position offset must be in (0, 1] degree")

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
    haptic_renderer: FeetechHapticBenchRenderer | None = None
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
