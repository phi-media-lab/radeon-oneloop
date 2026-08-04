#!/usr/bin/env python3
"""Render a non-authoritative Gaussian view from live Genesis state snapshots."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from radeon_oneloop.contracts import CAMERA_KEYS

from .gaussian_appearance import (
    SafeAppearanceBinding,
    VkSplatAppearanceRenderer,
    observed_core_asset,
)
from .scene import build
from .visual_state_protocol import (
    MAX_PACKET_BYTES,
    VisualStatePacket,
    VisualStateProtocolError,
    decode_visual_state,
)


class VisualStateReceiver:
    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.accepted = 0
        self.rejected = 0
        self.sequence_gaps = 0
        self.last_sequence_id: int | None = None

    def close(self) -> None:
        self.socket.close()

    def poll_latest(self) -> VisualStatePacket | None:
        latest = None
        while True:
            try:
                payload, _source = self.socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                packet = decode_visual_state(payload)
                if self.last_sequence_id is not None:
                    if packet.sequence_id <= self.last_sequence_id:
                        raise VisualStateProtocolError("non-monotonic visual sequence")
                    self.sequence_gaps += max(
                        packet.sequence_id - self.last_sequence_id - 1, 0
                    )
                self.last_sequence_id = packet.sequence_id
            except VisualStateProtocolError:
                self.rejected += 1
                continue
            self.accepted += 1
            latest = packet
        return latest


def _apply_snapshot(handles: Any, packet: VisualStatePacket) -> None:
    joints = np.asarray(packet.joint_positions_rad, dtype=np.float32)
    handles.left.set_dofs_position(joints[:6])
    handles.right.set_dofs_position(joints[6:])
    handles.object.set_pos(np.asarray(packet.object_position_m, dtype=np.float32))
    handles.object.set_quat(
        np.asarray(packet.object_quaternion_wxyz, dtype=np.float32)
    )
    handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so101-asset-root", type=Path, required=True)
    parser.add_argument("--observed-core-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58083)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--first-packet-timeout-s", type=float, default=180.0)
    parser.add_argument("--render-hz", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--fault-exit-after-frames",
        type=int,
        default=0,
        help="Test-only hard process exit after N rendered frames; zero disables.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.duration_s <= 0.0 or args.first_packet_timeout_s <= 0.0:
        raise ValueError("durations must be positive")
    if not 1.0 <= args.render_hz <= 30.0:
        raise ValueError("render-hz must be between 1 and 30")
    if args.fault_exit_after_frames < 0:
        raise ValueError("fault-exit-after-frames must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    receiver = VisualStateReceiver(args.bind_host, args.port)
    task = None
    handles = None
    binding: SafeAppearanceBinding | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    frames: list[np.ndarray] = []
    render_times_ms: list[float] = []
    snapshot_ages_ms: list[float] = []
    latest: VisualStatePacket | None = None
    try:
        task, handles = build(
            args.so101_asset_root.resolve(),
            seed=args.seed,
            show_viewer=False,
        )
        asset = observed_core_asset(args.observed_core_root)
        binding = SafeAppearanceBinding.create(
            lambda: VkSplatAppearanceRenderer(asset, args.vksplat_root)
        )
        task.set_appearance_binding(binding)
        (args.output / "READY").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "binding": binding.metrics(),
                    "physical_output": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        deadline = time.monotonic() + args.first_packet_timeout_s
        while not stop_requested and latest is None and time.monotonic() < deadline:
            latest = receiver.poll_latest()
            if latest is None:
                time.sleep(0.005)
        if latest is None:
            raise TimeoutError("no visual state received before the first-packet timeout")

        started = time.monotonic()
        next_render = started
        while not stop_requested and time.monotonic() - started < args.duration_s:
            packet = receiver.poll_latest()
            if packet is not None:
                latest = packet
            now = time.monotonic()
            if now < next_render:
                time.sleep(min(next_render - now, 0.005))
                continue
            _apply_snapshot(handles, latest)
            render_started = time.perf_counter()
            # This mirror does not step physics. State is imposed from the
            # authoritative process, so explicitly invalidate Genesis' render
            # cache after every external snapshot update.
            observation = task.observe(render=True, force_render=True)
            render_times_ms.append((time.perf_counter() - render_started) * 1000.0)
            snapshot_ages_ms.append(
                (time.monotonic_ns() - latest.captured_monotonic_ns) / 1_000_000.0
            )
            front = np.asarray(observation[CAMERA_KEYS[0]], dtype=np.uint8)
            hand = np.asarray(observation[CAMERA_KEYS[1]], dtype=np.uint8)
            frames.append(np.concatenate((front, hand), axis=1))
            if (
                args.fault_exit_after_frames
                and len(frames) == args.fault_exit_after_frames
            ):
                marker = {
                    "schema_version": "radeon_oneloop.renderer_fault_injection.v1",
                    "fault": "hard_process_exit",
                    "exit_code": 86,
                    "frames_before_exit": len(frames),
                    "last_snapshot_sequence_id": latest.sequence_id,
                    "binding": task.appearance_diagnostics()["binding"],
                    "physical_output": False,
                }
                (args.output / "FAULT_INJECTED.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(marker, sort_keys=True), flush=True)
                os._exit(86)
            next_render += 1.0 / args.render_hz
            if next_render < time.monotonic() - 1.0 / args.render_hz:
                next_render = time.monotonic()
        elapsed_s = time.monotonic() - started

        if not frames:
            raise RuntimeError("live Gaussian renderer produced no frames")
        iio.imwrite(args.output / "live_gaussian_first.png", frames[0])
        iio.imwrite(args.output / "live_gaussian_final.png", frames[-1])
        if args.record_video:
            iio.imwrite(
                args.output / "live_gaussian.mp4",
                np.stack(frames),
                fps=args.render_hz,
                codec="libx264",
                pixelformat="yuv420p",
            )
        diagnostics = task.appearance_diagnostics()
        minimum_frames = max(int(args.duration_s * args.render_hz * 0.8), 1)
        accepted = bool(
            len(frames) >= minimum_frames
            and receiver.accepted >= int(args.duration_s * 20.0)
            and receiver.rejected == 0
            and diagnostics["fallback_frames"] == 0
            and diagnostics["binding"]["latched_error"] is None
        )
        report = {
            "schema_version": "radeon_oneloop.gaussian_live_view.v1",
            "formal": False,
            "accepted": accepted,
            "duration_s": args.duration_s,
            "elapsed_s": elapsed_s,
            "render": {
                "requested_hz": args.render_hz,
                "frames": len(frames),
                "effective_hz": len(frames) / elapsed_s,
                "time_ms": {
                    "mean": statistics.fmean(render_times_ms),
                    "p95": float(np.percentile(render_times_ms, 95)),
                    "max": max(render_times_ms),
                },
            },
            "snapshots": {
                "accepted": receiver.accepted,
                "rejected": receiver.rejected,
                "sequence_gaps": receiver.sequence_gaps,
                "last_sequence_id": receiver.last_sequence_id,
                "age_ms": {
                    "mean": statistics.fmean(snapshot_ages_ms),
                    "p95": float(np.percentile(snapshot_ages_ms, 95)),
                    "max": max(snapshot_ages_ms),
                },
            },
            "appearance": diagnostics,
            "architecture": {
                "authoritative_control_process": False,
                "state_input": "read_only_udp_snapshot",
                "intermediate_snapshots_may_drop": True,
                "renderer_failure_can_stop_control": False,
            },
            "physical_output": False,
            "generated_fill_enabled": False,
        }
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not accepted:
            raise RuntimeError("decoupled live Gaussian view gate failed")
    finally:
        receiver.close()
        if binding is not None:
            binding.close()
        if handles is not None:
            try:
                handles.gs.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
