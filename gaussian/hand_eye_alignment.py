#!/usr/bin/env python3
"""Align unit-square camera poses to metric SO-101 gripper kinematics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_rotation(value: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(value, dtype=np.float64))
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def solve_hand_eye_rotation(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve R_A R_X = R_X R_B and return X plus per-pair errors."""
    if len(pairs) < 3:
        raise ValueError("at least three relative motion pairs are required")
    equations = []
    identity = np.eye(3)
    for arm_motion, camera_motion in pairs:
        rotation_a = arm_motion[:3, :3]
        rotation_b = camera_motion[:3, :3]
        equations.append(
            np.kron(identity, rotation_a) - np.kron(rotation_b.T, identity)
        )
    _, _, right = np.linalg.svd(np.concatenate(equations, axis=0))
    raw = right[-1].reshape(3, 3, order="F")
    candidates = []
    for sign in (1.0, -1.0):
        rotation = project_rotation(sign * raw)
        errors = np.asarray(
            [
                rotation_angle_deg(
                    arm[:3, :3]
                    @ rotation
                    @ (rotation @ camera[:3, :3]).T
                )
                for arm, camera in pairs
            ],
            dtype=np.float64,
        )
        candidates.append((rotation, errors))
    return min(candidates, key=lambda value: float(np.median(value[1])))


def solve_translation_and_scale(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]], rotation: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    matrices = []
    targets = []
    for arm, camera in pairs:
        block = np.empty((3, 4), dtype=np.float64)
        block[:, :3] = arm[:3, :3] - np.eye(3)
        block[:, 3] = -(rotation @ camera[:3, 3])
        matrices.append(block)
        targets.append(-arm[:3, 3])
    matrix = np.concatenate(matrices, axis=0)
    target = np.concatenate(targets)
    keep = np.ones(len(pairs), dtype=bool)
    solution = None
    pair_errors = None
    for _ in range(4):
        row_keep = np.repeat(keep, 3)
        solution, _, _, _ = np.linalg.lstsq(
            matrix[row_keep], target[row_keep], rcond=None
        )
        residual = (matrix @ solution - target).reshape(-1, 3)
        pair_errors = np.linalg.norm(residual, axis=1)
        keep = pair_errors <= np.percentile(pair_errors, 85)
    assert solution is not None and pair_errors is not None
    return solution[:3], float(solution[3]), pair_errors


def motion_pairs(
    gripper_to_world: np.ndarray,
    camera_to_target: np.ndarray,
    segments: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for span in (1, 2, 3, 5, 8, 12):
        for first in range(0, len(segments) - span):
            second = first + span
            if segments[first] != segments[second]:
                continue
            arm = np.linalg.inv(gripper_to_world[first]) @ gripper_to_world[second]
            camera = np.linalg.inv(camera_to_target[first]) @ camera_to_target[second]
            arm_rotation = rotation_angle_deg(arm[:3, :3])
            camera_rotation = rotation_angle_deg(camera[:3, :3])
            if max(arm_rotation, camera_rotation) < 1.0:
                continue
            if max(arm_rotation, camera_rotation) > 70.0:
                continue
            pairs.append((arm, camera))
    return pairs


def _average_world_to_target(
    gripper_to_world: np.ndarray,
    camera_to_target: np.ndarray,
    gripper_from_camera: np.ndarray,
    scale_m_per_target_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transforms = []
    for gripper, camera in zip(gripper_to_world, camera_to_target, strict=True):
        metric_camera = camera.copy()
        metric_camera[:3, 3] *= scale_m_per_target_unit
        transforms.append(metric_camera @ np.linalg.inv(gripper @ gripper_from_camera))
    translations = np.stack([value[:3, 3] for value in transforms])
    rotation = project_rotation(np.mean([value[:3, :3] for value in transforms], axis=0))
    translation = np.median(translations, axis=0)
    average = np.eye(4, dtype=np.float64)
    average[:3, :3] = rotation
    average[:3, 3] = translation
    translation_errors = np.linalg.norm(translations - translation, axis=1)
    rotation_errors = np.asarray(
        [rotation_angle_deg(value[:3, :3] @ rotation.T) for value in transforms]
    )
    return average, translation_errors, rotation_errors


def align(args: argparse.Namespace) -> dict[str, object]:
    planar_path = args.planar.resolve()
    kinematics_path = args.kinematics.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(planar_path, allow_pickle=False) as archive:
        camera_to_target_all = np.asarray(archive["camera_to_target"], dtype=np.float64)
        camera_frames = np.asarray(archive["frame_index"], dtype=np.int64)
        segments_all = np.asarray(archive["track_segment"], dtype=np.int64)
    with np.load(kinematics_path, allow_pickle=False) as archive:
        gripper_to_world_all = np.asarray(
            archive["left_gripper_to_world"], dtype=np.float64
        )
        gripper_frames = np.asarray(archive["frame_index"], dtype=np.int64)
    gripper_lookup = {int(frame): index for index, frame in enumerate(gripper_frames)}
    matched = [index for index, frame in enumerate(camera_frames) if int(frame) in gripper_lookup]
    if len(matched) < args.min_poses:
        raise RuntimeError(f"only {len(matched)} synchronized poses; minimum is {args.min_poses}")
    camera_to_target = camera_to_target_all[matched]
    segments = segments_all[matched]
    gripper_to_world = np.stack(
        [gripper_to_world_all[gripper_lookup[int(camera_frames[index])]] for index in matched]
    )
    pairs = motion_pairs(gripper_to_world, camera_to_target, segments)
    if len(pairs) < args.min_pairs:
        raise RuntimeError(f"only {len(pairs)} motion pairs; minimum is {args.min_pairs}")
    rotation, rotation_errors = solve_hand_eye_rotation(pairs)
    translation, scale, translation_errors = solve_translation_and_scale(pairs, rotation)
    gripper_from_camera = np.eye(4, dtype=np.float64)
    gripper_from_camera[:3, :3] = rotation
    gripper_from_camera[:3, 3] = translation
    world_to_target, world_translation_errors, world_rotation_errors = _average_world_to_target(
        gripper_to_world, camera_to_target, gripper_from_camera, scale
    )
    accepted = bool(
        args.min_target_side_m <= scale <= args.max_target_side_m
        and np.median(rotation_errors) <= args.max_rotation_error_deg
        and np.median(translation_errors) <= args.max_translation_error_m
    )
    report = {
        "schema_version": "radeon_oneloop.hil_hand_eye_alignment.v1",
        "formal": False,
        "accepted": accepted,
        "synchronized_poses": len(matched),
        "track_segments": int(len(np.unique(segments))),
        "motion_pairs": len(pairs),
        "target_side_m": scale,
        "gripper_from_camera_opencv": gripper_from_camera.tolist(),
        "world_to_target": world_to_target.tolist(),
        "rotation_error_deg": {
            "median": float(np.median(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
        },
        "translation_equation_error_m": {
            "median": float(np.median(translation_errors)),
            "p95": float(np.percentile(translation_errors, 95)),
        },
        "world_alignment_consistency": {
            "translation_median_m": float(np.median(world_translation_errors)),
            "translation_p95_m": float(np.percentile(world_translation_errors, 95)),
            "rotation_median_deg": float(np.median(world_rotation_errors)),
            "rotation_p95_deg": float(np.percentile(world_rotation_errors, 95)),
        },
        "inputs": {
            "planar_sha256": sha256_file(planar_path),
            "kinematics_sha256": sha256_file(kinematics_path),
        },
        "quality_gate": {
            "target_side_range_m": [args.min_target_side_m, args.max_target_side_m],
            "max_median_rotation_error_deg": args.max_rotation_error_deg,
            "max_median_translation_error_m": args.max_translation_error_m,
        },
    }
    metrics = output / "alignment.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "alignment.npz",
        gripper_from_camera_opencv=gripper_from_camera,
        world_to_target=world_to_target,
        target_side_m=np.asarray(scale),
        rotation_error_deg=rotation_errors,
        translation_error_m=translation_errors,
    )
    outputs = [metrics, output / "alignment.npz"]
    (output / "hashes.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in outputs) + "\n",
        encoding="utf-8",
    )
    (output / "DONE").touch()
    if args.require_quality and not accepted:
        (output / "REJECTED").touch()
        raise RuntimeError("hand-eye alignment did not pass its quality gate")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planar", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-poses", type=int, default=30)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--min-target-side-m", type=float, default=0.04)
    parser.add_argument("--max-target-side-m", type=float, default=0.15)
    parser.add_argument("--max-rotation-error-deg", type=float, default=5.0)
    parser.add_argument("--max-translation-error-m", type=float, default=0.02)
    parser.add_argument("--require-quality", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = align(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
