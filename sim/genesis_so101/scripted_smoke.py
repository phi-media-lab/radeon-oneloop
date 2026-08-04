#!/usr/bin/env python3
"""Run deterministic dual-arm sweeps and headless camera checks on AMD GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from radeon_oneloop.contracts import CAMERA_KEYS, IMAGE_SHAPE_HWC

from .scene import (
    ARM_BASE_SEPARATION_M,
    HOME_ACTION,
    LEFT_BASE_POS,
    MODEL_FORWARD_UNIT,
    MODEL_LATERAL_UNIT,
    RIGHT_BASE_POS,
    SHARED_BASE_EULER_DEG,
    build,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--video-frames",
        type=int,
        default=0,
        help="Capture this many evenly spaced side-by-side frames (zero disables video).",
    )
    parser.add_argument("--video-fps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    if args.video_frames < 0 or args.video_frames > args.steps:
        raise ValueError("video-frames must be between zero and steps")
    if args.video_fps < 1:
        raise ValueError("video-fps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    task, handles = build(args.asset_root.resolve(), seed=args.seed, show_viewer=False)
    build_seconds = time.perf_counter() - started
    initial_inter_arm_contacts = int(
        handles.left.get_contacts(with_entity=handles.right)["geom_a"].shape[0]
    )
    step_times = []
    rendered = None
    video_frames: list[np.ndarray] = []
    capture_steps = (
        {
            index * (args.steps - 1) // (args.video_frames - 1)
            for index in range(args.video_frames)
        }
        if args.video_frames > 1
        else ({args.steps // 2} if args.video_frames == 1 else set())
    )
    for step in range(args.steps):
        action = list(HOME_ACTION)
        joint = (step // 140) % 5
        offset = 8.0 * math.sin(2.0 * math.pi * (step % 140) / 140.0)
        action[joint] += offset
        action[6 + joint] += offset
        begin = time.perf_counter()
        observation = task.step(
            action,
            render=(step in (0, args.steps - 1) or step in capture_steps),
        )
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - begin)
        if CAMERA_KEYS[0] in observation:
            rendered = observation
            if step in capture_steps:
                front = np.asarray(observation[CAMERA_KEYS[0]], dtype=np.uint8)
                hand = np.asarray(observation[CAMERA_KEYS[1]], dtype=np.uint8)
                video_frames.append(np.concatenate((front, hand), axis=1))
    if rendered is None:
        rendered = task.observe(render=True)
    for key in CAMERA_KEYS:
        image = np.asarray(rendered[key])
        if image.shape != IMAGE_SHAPE_HWC:
            raise RuntimeError(f"{key} shape mismatch: {image.shape}")
        iio.imwrite(args.output / (key.rsplit(".", 1)[-1] + ".png"), image.astype(np.uint8))
    state = np.asarray(rendered["observation.state"])
    if state.shape != (12,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid state: shape={state.shape}")
    video_path = None
    if video_frames:
        video_path = args.output / "genesis_dual_camera.mp4"
        iio.imwrite(
            video_path,
            np.stack(video_frames),
            fps=args.video_fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "radeon_oneloop.genesis_smoke.v1",
        "backend": str(handles.gs.backend),
        "device": str(handles.gs.device),
        "torch_device": torch.cuda.get_device_name(0),
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "steps": args.steps,
        "scene_layout": {
            "left_base_pos_m": list(LEFT_BASE_POS),
            "right_base_pos_m": list(RIGHT_BASE_POS),
            "base_separation_m": ARM_BASE_SEPARATION_M,
            "model_forward_unit": list(MODEL_FORWARD_UNIT),
            "model_lateral_unit": list(MODEL_LATERAL_UNIT),
            "arrangement": "side_by_side_parallel",
            "shared_base_euler_deg": list(SHARED_BASE_EULER_DEG),
            "initial_inter_arm_contacts": initial_inter_arm_contacts,
            "final_inter_arm_contacts": int(
                handles.left.get_contacts(with_entity=handles.right)["geom_a"].shape[0]
            ),
        },
        "solver_limit_saturation": task.solver_limit_diagnostics(),
        "build_seconds": build_seconds,
        "step_ms": {
            "mean": 1000.0 * float(np.mean(step_times)),
            "p50": 1000.0 * float(np.percentile(step_times, 50)),
            "p95": 1000.0 * float(np.percentile(step_times, 95)),
            "p99": 1000.0 * float(np.percentile(step_times, 99)),
        },
        "observation": {
            "state_shape": list(state.shape),
            "camera_shapes": {
                key: list(np.asarray(rendered[key]).shape) for key in CAMERA_KEYS
            },
        },
        "task_success": task.success(),
        "task_success_note": (
            "Joint-sweep smoke validates the scene and contracts; it is not a handover evaluation."
        ),
        "video": {
            "path": str(video_path) if video_path else None,
            "frames": len(video_frames),
            "fps": args.video_fps if video_frames else None,
            "metric_eligible": False,
            "note": "Optional joint-sweep visualization; captured render steps are included in step timings.",
        },
    }
    payload = json.dumps(report, indent=2) + "\n"
    (args.output / "metrics.json").write_text(payload, encoding="utf-8")
    if os.environ.get("ONELOOP_RUN_DIR"):
        (Path(os.environ["ONELOOP_RUN_DIR"]) / "metrics.json").write_text(
            payload, encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
