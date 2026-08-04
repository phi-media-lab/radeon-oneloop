#!/usr/bin/env python3
"""Drive two Genesis SO-101 entities from a live UDP leader stream."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import statistics
import time
from pathlib import Path

import numpy as np

from radeon_oneloop.contracts import CAMERA_KEYS

from .gaussian_appearance import (
    SafeAppearanceBinding,
    VkSplatAppearanceRenderer,
    observed_core_asset,
)
from .live_protocol import (
    MAX_PACKET_BYTES,
    HapticFeedbackPacket,
    LeaderActionGate,
    LeaderActionPacket,
    LiveProtocolError,
    clamp_action_to_model,
    decode_packet,
    encode_haptic_packet,
)
from .scene import (
    ARM_BASE_SEPARATION_M,
    LEFT_BASE_POS,
    MODEL_FORWARD_UNIT,
    MODEL_LATERAL_UNIT,
    RIGHT_BASE_POS,
    SHARED_BASE_EULER_DEG,
    build,
)
from .visual_state_protocol import VisualStatePacket, encode_visual_state


class UdpLeaderReceiver:
    def __init__(self, host: str, port: int, *, source_host: str | None = None):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.source_host = source_host
        self.gate = LeaderActionGate()
        self.accepted = 0
        self.rejected = 0
        self.rebased = 0
        self.last_arrival_monotonic_ns: int | None = None
        self.last_source: tuple[str, int] | None = None

    def close(self) -> None:
        self.socket.close()

    def poll_latest(self) -> LeaderActionPacket | None:
        latest = None
        while True:
            try:
                payload, source = self.socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                if self.source_host is not None and source[0] != self.source_host:
                    raise LiveProtocolError(
                        f"unexpected source {source[0]!r}; expected {self.source_host!r}"
                    )
                packet = decode_packet(payload)
                sender_gap_s = self.gate.sender_gap_s(packet)
                if (
                    sender_gap_s is not None
                    and sender_gap_s > self.gate.maximum_sender_dt_s
                ):
                    # A long pause is a stream discontinuity, not a permanent
                    # fault. Re-establish the rate baseline only after the
                    # packet passes ordering and absolute range checks. The
                    # simulator then interpolates from its held pose.
                    latest = self.gate.rebase(packet)
                    self.rebased += 1
                else:
                    latest = self.gate.accept(packet)
            except LiveProtocolError:
                self.rejected += 1
                continue
            self.accepted += 1
            self.last_arrival_monotonic_ns = time.monotonic_ns()
            self.last_source = source
        return latest

    def age_ms(self) -> float | None:
        if self.last_arrival_monotonic_ns is None:
            return None
        return (time.monotonic_ns() - self.last_arrival_monotonic_ns) / 1_000_000.0


def _wait_for_first_packet(
    receiver: UdpLeaderReceiver, timeout_s: float
) -> LeaderActionPacket:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        packet = receiver.poll_latest()
        if packet is not None:
            return packet
        time.sleep(0.005)
    raise TimeoutError(f"no valid leader packet received within {timeout_s:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58081)
    parser.add_argument("--source-host")
    parser.add_argument("--feedback-host")
    parser.add_argument("--feedback-port", type=int, default=58082)
    parser.add_argument("--feedback-hz", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--first-packet-timeout-s", type=float, default=30.0)
    parser.add_argument("--watchdog-ms", type=float, default=250.0)
    parser.add_argument("--sim-hz", type=float, default=120.0)
    parser.add_argument("--input-hz", type=float, default=30.0)
    parser.add_argument("--render-hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--appearance-mode", choices=("debug-mesh", "vksplat"), default="debug-mesh"
    )
    parser.add_argument("--observed-core-root", type=Path)
    parser.add_argument("--vksplat-root", type=Path)
    parser.add_argument("--visual-state-host")
    parser.add_argument("--visual-state-port", type=int, default=58083)
    parser.add_argument("--visual-state-hz", type=float, default=30.0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-delay-s", type=float, default=0.0)
    args = parser.parse_args()
    import imageio.v3 as iio
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not 1 <= args.feedback_port <= 65535:
        raise ValueError("feedback-port must be between 1 and 65535")
    if args.duration_s <= 0:
        raise ValueError("duration-s must be positive")
    if args.first_packet_timeout_s <= 0:
        raise ValueError("first-packet-timeout-s must be positive")
    if args.watchdog_ms <= 0:
        raise ValueError("watchdog-ms must be positive")
    if args.sim_hz != 120.0:
        raise ValueError("the frozen Genesis scene currently requires --sim-hz=120")
    if not 1.0 <= args.input_hz <= args.sim_hz:
        raise ValueError("input-hz must be between 1 and sim-hz")
    if not 0.0 <= args.render_hz <= args.sim_hz:
        raise ValueError("render-hz must be between 0 and sim-hz")
    if not 1.0 <= args.feedback_hz <= args.sim_hz:
        raise ValueError("feedback-hz must be between 1 and sim-hz")
    if args.appearance_mode == "vksplat":
        if args.observed_core_root is None or args.vksplat_root is None:
            raise ValueError(
                "vksplat appearance requires --observed-core-root and --vksplat-root"
            )
        if args.render_hz <= 0:
            raise ValueError("vksplat appearance requires a positive render-hz")
    if not 1 <= args.visual_state_port <= 65535:
        raise ValueError("visual-state-port must be between 1 and 65535")
    if not 1.0 <= args.visual_state_hz <= args.sim_hz:
        raise ValueError("visual-state-hz must be between 1 and sim-hz")
    if not 0.0 <= args.start_delay_s <= 30.0:
        raise ValueError("start-delay-s must be between 0 and 30 seconds")
    if args.ready_file is not None and args.ready_file.exists():
        raise FileExistsError(f"ready-file already exists: {args.ready_file}")

    args.output.mkdir(parents=True, exist_ok=True)
    receiver = UdpLeaderReceiver(args.bind_host, args.port, source_host=args.source_host)
    feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    visual_state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback_destination = (
        (args.feedback_host, args.feedback_port) if args.feedback_host else None
    )
    visual_state_destination = (
        (args.visual_state_host, args.visual_state_port)
        if args.visual_state_host
        else None
    )
    task = None
    handles = None
    appearance_binding: SafeAppearanceBinding | None = None
    frames: list[np.ndarray] = []
    step_times_ms: list[float] = []
    watchdog_events = 0
    watchdog_active = False
    first_packet = None
    last_packet = None
    last_raw_action = None
    processed_packets_with_clamping = 0
    processed_values_clamped = 0
    max_clamping_delta = 0.0
    steps = 0
    feedback_sent = 0
    feedback_send_errors = 0
    visual_state_sent = 0
    visual_state_send_errors = 0
    max_contact_force_n = [0.0, 0.0]
    max_joint_reaction_effort = 0.0
    stop_requested = False
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    build_started = time.perf_counter()
    try:
        # Bind before the relatively expensive Genesis build so the kernel can
        # retain incoming leader packets in the UDP receive buffer.
        task, handles = build(
            args.asset_root.resolve(), seed=args.seed, show_viewer=args.show_viewer
        )
        if args.appearance_mode == "vksplat":
            asset = observed_core_asset(args.observed_core_root)
            appearance_binding = SafeAppearanceBinding.create(
                lambda: VkSplatAppearanceRenderer(asset, args.vksplat_root)
            )
            task.set_appearance_binding(appearance_binding)
        build_seconds = time.perf_counter() - build_started
        first_packet = _wait_for_first_packet(receiver, args.first_packet_timeout_s)
        last_packet = first_packet
        last_raw_action = np.asarray(first_packet.action, dtype=np.float64)
        current = np.asarray(clamp_action_to_model(first_packet.action), dtype=np.float64)
        initial_clamping = np.abs(last_raw_action - current)
        if np.any(initial_clamping > 0.0):
            processed_packets_with_clamping += 1
            processed_values_clamped += int(np.count_nonzero(initial_clamping))
            max_clamping_delta = float(np.max(initial_clamping))
        target = current.copy()
        interpolation_start = current.copy()
        interpolation_step = 0
        interpolation_steps = max(int(round(args.sim_hz / args.input_hz)), 1)
        task.reset(current.tolist())

        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_ready = args.ready_file.with_name(
                args.ready_file.name + ".tmp"
            )
            temporary_ready.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.live_teleop_ready.v1",
                        "start_delay_s": args.start_delay_s,
                        "physical_output_commands": False,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_ready.replace(args.ready_file)
        delay_deadline = time.monotonic() + args.start_delay_s
        while not stop_requested and time.monotonic() < delay_deadline:
            time.sleep(min(0.05, delay_deadline - time.monotonic()))

        render_interval = (
            max(int(round(args.sim_hz / args.render_hz)), 1)
            if args.render_hz > 0
            else None
        )
        feedback_interval = max(
            int(round(args.sim_hz / args.feedback_hz)), 1
        )
        visual_state_interval = max(
            int(round(args.sim_hz / args.visual_state_hz)), 1
        )
        run_started = time.monotonic()
        next_step = run_started
        while (
            not stop_requested
            and time.monotonic() - run_started < args.duration_s
        ):
            packet = receiver.poll_latest()
            if packet is not None:
                last_packet = packet
                last_raw_action = np.asarray(packet.action, dtype=np.float64)
                interpolation_start = current.copy()
                target = np.asarray(
                    clamp_action_to_model(packet.action), dtype=np.float64
                )
                clamping = np.abs(last_raw_action - target)
                if np.any(clamping > 0.0):
                    processed_packets_with_clamping += 1
                    processed_values_clamped += int(np.count_nonzero(clamping))
                    max_clamping_delta = max(
                        max_clamping_delta, float(np.max(clamping))
                    )
                interpolation_step = 0

            age_ms = receiver.age_ms()
            stale = age_ms is None or age_ms > args.watchdog_ms
            if stale:
                if not watchdog_active:
                    watchdog_events += 1
                    watchdog_active = True
                # Fail-hold the virtual followers and cancel any unfinished
                # interpolation. No command in this process reaches real motors.
                target = current.copy()
                interpolation_start = current.copy()
                interpolation_step = interpolation_steps
            else:
                watchdog_active = False

            if interpolation_step < interpolation_steps:
                interpolation_step += 1
                alpha = interpolation_step / interpolation_steps
                command = interpolation_start + (target - interpolation_start) * alpha
            else:
                command = target

            render = render_interval is not None and steps % render_interval == 0
            step_started = time.perf_counter()
            observation = task.step(command.tolist(), render=render)
            step_times_ms.append((time.perf_counter() - step_started) * 1000.0)
            current = np.asarray(command, dtype=np.float64)
            if (
                visual_state_destination is not None
                and steps % visual_state_interval == 0
            ):
                snapshot = task.visual_state()
                visual_packet = VisualStatePacket(
                    sequence_id=visual_state_sent,
                    captured_monotonic_ns=time.monotonic_ns(),
                    captured_unix_ns=time.time_ns(),
                    joint_positions_rad=snapshot["joint_positions_rad"],
                    object_position_m=snapshot["object_position_m"],
                    object_quaternion_wxyz=snapshot["object_quaternion_wxyz"],
                )
                try:
                    visual_state_socket.sendto(
                        encode_visual_state(visual_packet), visual_state_destination
                    )
                    visual_state_sent += 1
                except OSError:
                    # Demo rendering is deliberately non-authoritative. Losing
                    # its UDP sink must not stop physics or haptic monitoring.
                    visual_state_send_errors += 1
            if feedback_destination is not None and steps % feedback_interval == 0:
                efforts, contact_force_n = task.haptic_feedback()
                feedback_packet = HapticFeedbackPacket(
                    sequence_id=feedback_sent,
                    captured_monotonic_ns=time.monotonic_ns(),
                    captured_unix_ns=time.time_ns(),
                    joint_reaction_effort=efforts,
                    contact_force_n=contact_force_n,
                )
                try:
                    feedback_socket.sendto(
                        encode_haptic_packet(feedback_packet), feedback_destination
                    )
                except OSError:
                    feedback_send_errors += 1
                    raise
                feedback_sent += 1
                max_joint_reaction_effort = max(
                    max_joint_reaction_effort,
                    max(abs(value) for value in efforts),
                )
                for arm_index, value in enumerate(contact_force_n):
                    max_contact_force_n[arm_index] = max(
                        max_contact_force_n[arm_index], value
                    )
            if render:
                front = np.asarray(observation[CAMERA_KEYS[0]], dtype=np.uint8)
                hand = np.asarray(observation[CAMERA_KEYS[1]], dtype=np.uint8)
                frames.append(np.concatenate((front, hand), axis=1))

            steps += 1
            next_step += 1.0 / args.sim_hz
            delay = next_step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -(1.0 / args.sim_hz):
                next_step = time.monotonic()

        run_elapsed_s = time.monotonic() - run_started
        final_observation = task.observe(render=True)
        for key in CAMERA_KEYS:
            image = np.asarray(final_observation[key], dtype=np.uint8)
            iio.imwrite(args.output / ("live_" + key.rsplit(".", 1)[-1] + ".png"), image)
        final_state = np.asarray(
            final_observation["observation.state"], dtype=np.float64
        )
        tracking_error = np.abs(final_state - current)
        if args.record_video and frames:
            iio.imwrite(
                args.output / "live_dual_leader.mp4",
                np.stack(frames),
                fps=args.render_hz,
                codec="libx264",
                pixelformat="yuv420p",
            )

        report = {
            "schema_version": "radeon_oneloop.genesis_live_teleop.v1",
            "formal": False,
            "backend": str(handles.gs.backend),
            "device": str(handles.gs.device),
            "bind": f"{args.bind_host}:{args.port}",
            "source": list(receiver.last_source) if receiver.last_source else None,
            "duration_s": args.duration_s,
            "run_elapsed_s": run_elapsed_s,
            "termination": {
                "reason": "signal" if stop_signal is not None else "duration",
                "signal": stop_signal,
            },
            "build_seconds": build_seconds,
            "operator_start_delay_s": args.start_delay_s,
            "ready_file_emitted": args.ready_file is not None,
            "sim_hz_requested": args.sim_hz,
            "sim_hz_effective": steps / run_elapsed_s,
            "steps": steps,
            "scene_layout": {
                "left_base_pos_m": list(LEFT_BASE_POS),
                "right_base_pos_m": list(RIGHT_BASE_POS),
                "base_separation_m": ARM_BASE_SEPARATION_M,
                "model_forward_unit": list(MODEL_FORWARD_UNIT),
                "model_lateral_unit": list(MODEL_LATERAL_UNIT),
                "arrangement": "side_by_side_parallel",
                "shared_base_euler_deg": list(SHARED_BASE_EULER_DEG),
            },
            "packets": {
                "accepted": receiver.accepted,
                "rejected": receiver.rejected,
                "rebased": receiver.rebased,
                "first_sequence_id": first_packet.sequence_id,
                "last_sequence_id": last_packet.sequence_id if last_packet else None,
            },
            "watchdog": {
                "timeout_ms": args.watchdog_ms,
                "events": watchdog_events,
                "active_at_end": watchdog_active,
            },
            "haptic_feedback": {
                "enabled": feedback_destination is not None,
                "destination": (
                    f"{args.feedback_host}:{args.feedback_port}"
                    if feedback_destination is not None
                    else None
                ),
                "requested_hz": args.feedback_hz,
                "packets_sent": feedback_sent,
                "send_errors": feedback_send_errors,
                "max_contact_force_n": max_contact_force_n,
                "max_abs_joint_reaction_effort": max_joint_reaction_effort,
                "physical_output_commands": False,
            },
            "rendered_frames": len(frames),
            "step_ms": {
                "mean": statistics.fmean(step_times_ms),
                "p50": float(np.percentile(step_times_ms, 50)),
                "p95": float(np.percentile(step_times_ms, 95)),
                "p99": float(np.percentile(step_times_ms, 99)),
            },
            "input_clamping": {
                "processed_packets_with_clamping": processed_packets_with_clamping,
                "processed_values_clamped": processed_values_clamped,
                "max_abs_delta": max_clamping_delta,
            },
            "solver_limit_saturation": task.solver_limit_diagnostics(),
            "last_received_action": (
                last_raw_action.tolist() if last_raw_action is not None else None
            ),
            "final_applied_action": current.tolist(),
            "final_observation_state": final_state.tolist(),
            "tracking_error": {
                "mean_abs": float(np.mean(tracking_error)),
                "max_abs": float(np.max(tracking_error)),
            },
            "physical_output_commands": False,
            "appearance": {
                "mode": args.appearance_mode,
                "generated_fill_enabled": False,
                "diagnostics": task.appearance_diagnostics(),
            },
            "visual_state_stream": {
                "enabled": visual_state_destination is not None,
                "destination": (
                    f"{args.visual_state_host}:{args.visual_state_port}"
                    if visual_state_destination is not None
                    else None
                ),
                "requested_hz": args.visual_state_hz,
                "packets_sent": visual_state_sent,
                "send_errors": visual_state_send_errors,
                "authoritative_for_control": False,
            },
        }
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        receiver.close()
        feedback_socket.close()
        visual_state_socket.close()
        if appearance_binding is not None:
            appearance_binding.close()
        if handles is not None:
            try:
                handles.gs.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
