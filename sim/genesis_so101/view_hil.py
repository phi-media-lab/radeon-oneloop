#!/usr/bin/env python3
"""Interactively loop a historical HIL trajectory in the Genesis viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

from .live_protocol import clamp_action_to_model
from .replay_hil import load_trajectory
from .scene import build


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--workspace-texture", type=Path, required=True)
    parser.add_argument("--front-camera-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-key", choices=("observation_state", "action"), default="observation_state"
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--front-camera-hz",
        type=float,
        default=0.0,
        help=(
            "Optional standalone calibrated camera window refresh rate. "
            "Requires a GUI-enabled OpenCV build; zero keeps only the Genesis viewer."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.playback_speed <= 4.0:
        raise ValueError("playback-speed must be between 0.1 and 4.0")
    if not 0.0 <= args.front_camera_hz <= 30.0:
        raise ValueError("front-camera-hz must be between 0 and 30")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    trajectory_path = args.trajectory.resolve()
    texture_path = args.workspace_texture.resolve()
    calibration_path = args.front_camera_calibration.resolve()
    trajectory = load_trajectory(trajectory_path, args.state_key)
    timestamps = np.asarray(trajectory["timestamp"], dtype=np.float64)
    raw_states = np.asarray(trajectory[args.state_key], dtype=np.float64)
    applied = np.asarray(
        [clamp_action_to_model(tuple(row)) for row in raw_states], dtype=np.float64
    )
    manifest = {
        "schema_version": "radeon_oneloop.genesis_hil_viewer.v1",
        "formal": False,
        "physical_output_commands": False,
        "physical_devices_opened": False,
        "state_key": args.state_key,
        "source_frames": len(timestamps),
        "source_duration_s": float(timestamps[-1] - timestamps[0]),
        "playback_speed": args.playback_speed,
        "loop": args.loop,
        "trajectory_sha256": sha256_file(trajectory_path),
        "workspace_texture_sha256": sha256_file(texture_path),
        "front_camera_calibration_sha256": sha256_file(calibration_path),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stop_requested = False
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    handles = None
    steps = 0
    cycles = 0
    started = time.monotonic()
    try:
        task, handles = build(
            args.asset_root.resolve(),
            seed=args.seed,
            show_viewer=True,
            workspace_texture=texture_path,
            front_camera_calibration=calibration_path,
            front_camera_gui=args.front_camera_hz > 0.0,
        )
        task.reset(applied[0].tolist())
        if args.front_camera_hz > 0.0:
            handles.front_camera.render(rgb=True)
        (args.output / "READY").touch()
        print("HIL viewer READY", flush=True)
        render_interval = (
            max(1, int(round(120.0 / args.front_camera_hz)))
            if args.front_camera_hz > 0.0
            else None
        )
        while not stop_requested:
            cycle_started = time.monotonic()
            for row_index in range(1, len(timestamps)):
                if stop_requested:
                    break
                delta_s = float(timestamps[row_index] - timestamps[row_index - 1])
                substeps = max(1, int(round(delta_s * 120.0)))
                start_state = applied[row_index - 1]
                target_state = applied[row_index]
                elapsed_before_s = float(timestamps[row_index - 1] - timestamps[0])
                for substep in range(1, substeps + 1):
                    if stop_requested:
                        break
                    alpha = substep / substeps
                    command = start_state + alpha * (target_state - start_state)
                    task.step(command.tolist(), render=False)
                    steps += 1
                    if render_interval is not None and steps % render_interval == 0:
                        handles.front_camera.render(rgb=True)
                    source_elapsed_s = elapsed_before_s + alpha * delta_s
                    deadline = cycle_started + source_elapsed_s / args.playback_speed
                    delay = deadline - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
            cycles += 1
            if not args.loop:
                break
            task.reset(applied[0].tolist())
        report = {
            **manifest,
            "backend": str(handles.gs.backend),
            "device": str(handles.gs.device),
            "cycles_completed": cycles,
            "sim_steps": steps,
            "viewer_runtime_s": time.monotonic() - started,
            "stop_signal": stop_signal,
        }
        metrics_path = args.output / "metrics.json"
        metrics_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output / "hashes.sha256").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in (manifest_path, metrics_path)
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output / ("STOPPED" if stop_signal is not None else "DONE")).touch()
    except BaseException:
        (args.output / "FAILED").touch()
        raise
    finally:
        if handles is not None:
            try:
                handles.gs.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
