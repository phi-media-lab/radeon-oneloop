#!/usr/bin/env python3
"""Replay a synchronized historical SO-101 trajectory in Genesis on AMD GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from radeon_oneloop.contracts import CAMERA_KEYS, IMAGE_SHAPE_HWC

from .handover_asset import DEFAULT_MESH, load_spec
from .live_protocol import clamp_action_to_model
from .scene import ARM_BASE_SEPARATION_M, LEFT_BASE_POS, RIGHT_BASE_POS, build


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory(path: Path, state_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"action", "observation_state", "timestamp", "frame_index"}
        if not required.issubset(archive.files):
            missing = sorted(required - set(archive.files))
            raise ValueError(f"trajectory is missing keys: {missing}")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    states = np.asarray(result[state_key], dtype=np.float64)
    timestamps = np.asarray(result["timestamp"], dtype=np.float64)
    frames = np.asarray(result["frame_index"], dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != 12 or states.shape[0] < 2:
        raise ValueError(f"{state_key} must have shape (frames>=2, 12)")
    if timestamps.shape != (states.shape[0],) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite and match trajectory length")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.array_equal(frames, np.arange(states.shape[0])):
        raise ValueError("frame_index must be contiguous from zero")
    return result


def _matrix(position: object, quaternion: object) -> np.ndarray:
    import genesis.utils.geom as gu

    value = gu.trans_quat_to_T(position, quaternion)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"pose matrix has shape {matrix.shape}")
    return matrix


def main() -> None:
    import imageio.v3 as iio
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-texture", type=Path)
    parser.add_argument("--front-camera-calibration", type=Path)
    parser.add_argument(
        "--state-key", choices=("observation_state", "action"), default="observation_state"
    )
    parser.add_argument("--sim-hz", type=float, default=120.0)
    parser.add_argument("--render-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if args.sim_hz != 120.0:
        raise ValueError("the frozen Genesis scene requires --sim-hz=120")
    if not 0.0 <= args.render_fps <= 30.0:
        raise ValueError("render-fps must be between zero and 30")
    if args.max_seconds < 0.0:
        raise ValueError("max-seconds must be non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    trajectory = load_trajectory(args.trajectory, args.state_key)
    timestamps = np.asarray(trajectory["timestamp"], dtype=np.float64)
    raw_states = np.asarray(trajectory[args.state_key], dtype=np.float64)
    if args.max_seconds > 0.0:
        keep = timestamps <= timestamps[0] + args.max_seconds
        timestamps = timestamps[keep]
        raw_states = raw_states[keep]
    applied = np.asarray(
        [clamp_action_to_model(tuple(row)) for row in raw_states], dtype=np.float64
    )
    clamping = np.abs(raw_states - applied)

    started = time.perf_counter()
    task, handles = build(
        args.asset_root.resolve(),
        seed=args.seed,
        show_viewer=False,
        workspace_texture=args.workspace_texture,
        front_camera_calibration=args.front_camera_calibration,
    )
    build_seconds = time.perf_counter() - started
    task.reset(applied[0].tolist())
    left_link = handles.left.get_link("gripper")
    right_link = handles.right.get_link("gripper")
    video_frames = []
    pose_frames = []
    left_poses = []
    right_poses = []
    hand_camera_poses = []
    tracking_errors = []
    step_times_ms = []
    next_render_s = 0.0
    sim_steps = 0
    rendered_observation = None

    for row_index in range(1, len(timestamps)):
        delta_s = float(timestamps[row_index] - timestamps[row_index - 1])
        substeps = max(1, int(round(delta_s * args.sim_hz)))
        start_state = applied[row_index - 1]
        target_state = applied[row_index]
        render_this_row = (
            args.render_fps > 0.0
            and timestamps[row_index] - timestamps[0] >= next_render_s
        )
        observation = None
        for substep in range(1, substeps + 1):
            alpha = substep / substeps
            command = start_state + alpha * (target_state - start_state)
            render = render_this_row and substep == substeps
            step_started = time.perf_counter()
            observation = task.step(command.tolist(), render=render)
            torch.cuda.synchronize()
            step_times_ms.append((time.perf_counter() - step_started) * 1000.0)
            sim_steps += 1
            if render:
                rendered_observation = observation
        assert observation is not None
        observed = np.asarray(observation["observation.state"], dtype=np.float64)
        tracking_errors.append(np.abs(observed - target_state))
        handles.hand_camera.move_to_attach()
        pose_frames.append(row_index)
        left_poses.append(_matrix(left_link.get_pos(), left_link.get_quat()))
        right_poses.append(_matrix(right_link.get_pos(), right_link.get_quat()))
        hand_transform = handles.hand_camera.get_transform()
        if hasattr(hand_transform, "detach"):
            hand_transform = hand_transform.detach().cpu().numpy()
        hand_camera_poses.append(np.asarray(hand_transform, dtype=np.float64))
        if render_this_row:
            front = np.asarray(observation[CAMERA_KEYS[0]], dtype=np.uint8)
            hand = np.asarray(observation[CAMERA_KEYS[1]], dtype=np.uint8)
            if front.shape != IMAGE_SHAPE_HWC or hand.shape != IMAGE_SHAPE_HWC:
                raise RuntimeError("Genesis camera shape does not match the dataset contract")
            video_frames.append(np.concatenate((front, hand), axis=1))
            next_render_s += 1.0 / args.render_fps

    if rendered_observation is None:
        rendered_observation = task.observe(render=True)
    for key in CAMERA_KEYS:
        image = np.asarray(rendered_observation[key], dtype=np.uint8)
        iio.imwrite(args.output / ("replay_" + key.rsplit(".", 1)[-1] + ".png"), image)
    if args.record_video and video_frames:
        iio.imwrite(
            args.output / "hil_replay_dual_camera.mp4",
            np.stack(video_frames),
            fps=args.render_fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    np.savez_compressed(
        args.output / "kinematic_poses.npz",
        frame_index=np.asarray(pose_frames, dtype=np.int64),
        timestamp_s=timestamps[np.asarray(pose_frames, dtype=np.int64)],
        left_gripper_to_world=np.stack(left_poses),
        right_gripper_to_world=np.stack(right_poses),
        hand_camera_to_world=np.stack(hand_camera_poses),
    )
    errors = np.stack(tracking_errors)
    object_position = task._array(handles.object.get_pos()).reshape(-1)
    object_spec = load_spec()
    report = {
        "schema_version": "radeon_oneloop.genesis_hil_replay.v1",
        "formal": False,
        "physical_output_commands": False,
        "backend": str(handles.gs.backend),
        "device": str(handles.gs.device),
        "trajectory_sha256": sha256_file(args.trajectory),
        "state_key": args.state_key,
        "source_frames": int(len(timestamps)),
        "source_duration_s": float(timestamps[-1] - timestamps[0]),
        "sim_steps": sim_steps,
        "rendered_frames": len(video_frames),
        "build_seconds": build_seconds,
        "scene_layout": {
            "left_base_pos_m": list(LEFT_BASE_POS),
            "right_base_pos_m": list(RIGHT_BASE_POS),
            "base_separation_m": ARM_BASE_SEPARATION_M,
        },
        "workspace_texture": {
            "enabled": args.workspace_texture is not None,
            "name": args.workspace_texture.name if args.workspace_texture else None,
            "sha256": sha256_file(args.workspace_texture)
            if args.workspace_texture
            else None,
        },
        "front_camera_calibration": {
            "enabled": args.front_camera_calibration is not None,
            "name": args.front_camera_calibration.name
            if args.front_camera_calibration
            else None,
            "sha256": sha256_file(args.front_camera_calibration)
            if args.front_camera_calibration
            else None,
        },
        "input_clamping": {
            "frames_with_clamping": int(np.any(clamping > 0.0, axis=1).sum()),
            "values_clamped": int(np.count_nonzero(clamping)),
            "max_abs_delta": float(clamping.max()),
        },
        "tracking_error": {
            "mean_abs": float(errors.mean()),
            "p95_abs": float(np.percentile(errors, 95)),
            "max_abs": float(errors.max()),
        },
        "step_ms": {
            "mean": statistics.fmean(step_times_ms),
            "p50": float(np.percentile(step_times_ms, 50)),
            "p95": float(np.percentile(step_times_ms, 95)),
            "p99": float(np.percentile(step_times_ms, 99)),
        },
        "object_proxy": {
            "kind": "hil_derived_rigid_convex_proxy",
            "asset_name": object_spec.asset_name,
            "exact_sku_status": object_spec.exact_sku_status,
            "mesh_sha256": sha256_file(DEFAULT_MESH),
            "reference_manifest_sha256": object_spec.reference_manifest_sha256,
            "physical_parameters_calibrated": False,
            "final_position_m": object_position.tolist(),
            "handover_metric_eligible": False,
        },
        "kinematic_pose_samples": len(pose_frames),
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hashed = [
        metrics_path,
        args.output / "kinematic_poses.npz",
        args.output / "replay_front_cam.png",
        args.output / "replay_hand_cam.png",
    ]
    video = args.output / "hil_replay_dual_camera.mp4"
    if video.is_file():
        hashed.append(video)
    (args.output / "hashes.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in sorted(hashed))
        + "\n",
        encoding="utf-8",
    )
    (args.output / "DONE").touch()
    print(json.dumps(report, indent=2))
    try:
        handles.gs.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
