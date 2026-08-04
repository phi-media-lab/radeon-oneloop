#!/usr/bin/env python3
"""Validate four rigid Gaussian poses inside the Genesis handover scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gaussian.vksplat_render_ply import read_3dgs_ply
from radeon_oneloop.contracts import CAMERA_KEYS

from .gaussian_appearance import (
    SafeAppearanceBinding,
    VkSplatAppearanceRenderer,
    observed_core_asset,
    transform_from_pos_quat_wxyz,
)
from .handover_asset import load_spec
from .scene import OBJECT_START_POS, build


METRIC_EXTENT_TRIM_PERCENT = 0.01


def gaussian_center_extents(
    xyz: np.ndarray,
    *,
    trim_percent: float = METRIC_EXTENT_TRIM_PERCENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full and lightly trimmed Gaussian-center extents.

    The metric silhouette is represented by splats near the boundary.  A
    0.5-percent tail trim discards 150 centers per side in a 30k asset and
    materially shrinks the object.  The pinned 0.01-percent trim removes only
    about three centers per side while remaining insensitive to single-point
    export outliers.
    """

    centers = np.asarray(xyz, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("Gaussian centers must be a finite Nx3 array")
    if not 0.0 <= trim_percent < 50.0:
        raise ValueError("trim_percent must be in [0, 50)")
    full_extents = np.ptp(centers, axis=0)
    low, high = np.percentile(
        centers,
        (trim_percent, 100.0 - trim_percent),
        axis=0,
    )
    return full_extents, high - low


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so101-asset-root", type=Path, required=True)
    parser.add_argument("--observed-core-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v3 as iio

    asset = observed_core_asset(args.observed_core_root)
    asset_audit = asset.validate()
    gaussians = read_3dgs_ply(asset.ply_path)
    full_extents, robust_extents = gaussian_center_extents(gaussians["xyz"])
    spec = load_spec()
    height_error_m = abs(float(robust_extents[2]) - spec.nominal_overall_height_m)
    height_tolerance_m = max(0.002, 0.03 * spec.nominal_overall_height_m)

    task, handles = build(
        args.so101_asset_root.resolve(),
        seed=args.seed,
        show_viewer=False,
    )
    binding = SafeAppearanceBinding.create(
        lambda: VkSplatAppearanceRenderer(asset, args.vksplat_root)
    )
    task.set_appearance_binding(binding)
    poses = []
    try:
        for index, yaw_deg in enumerate((0.0, 90.0, 180.0, -90.0)):
            yaw = math.radians(yaw_deg)
            quat = np.asarray(
                (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                dtype=np.float32,
            )
            handles.object.set_pos(np.asarray(OBJECT_START_POS, dtype=np.float32))
            handles.object.set_quat(quat)
            handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
            observation = task.observe(render=True, force_render=True)
            paths = {}
            for camera_key in CAMERA_KEYS:
                name = f"pose_{index:02d}_{camera_key.rsplit('.', 1)[-1]}.png"
                iio.imwrite(args.output / name, np.asarray(observation[camera_key], dtype=np.uint8))
                paths[camera_key] = name
            world_object = transform_from_pos_quat_wxyz(
                handles.object.get_pos(), handles.object.get_quat()
            )
            poses.append(
                {
                    "yaw_deg": yaw_deg,
                    "world_from_object_4x4": world_object.tolist(),
                    "images": paths,
                }
            )
        diagnostics = task.appearance_diagnostics()
        memory = (
            binding.renderer.memory_usage()
            if isinstance(binding.renderer, VkSplatAppearanceRenderer)
            else None
        )
    finally:
        binding.close()

    accepted = bool(
        height_error_m <= height_tolerance_m
        and diagnostics["binding"]["latched_error"] is None
        and diagnostics["binding"]["successes"] == len(poses) * len(CAMERA_KEYS)
        and diagnostics["fallback_frames"] == 0
        and diagnostics["composited_frames"] == len(poses) * len(CAMERA_KEYS)
    )
    report = {
        "schema_version": "radeon_oneloop.genesis_gaussian_static_gate.v1",
        "formal": False,
        "accepted": accepted,
        "asset": asset_audit,
        "metric_registration": {
            "center_trim_percent_per_tail": METRIC_EXTENT_TRIM_PERCENT,
            "center_trim_count_per_tail": int(
                len(gaussians["xyz"]) * METRIC_EXTENT_TRIM_PERCENT / 100.0
            ),
            "full_center_extents_m": full_extents.tolist(),
            "robust_percentile_range": [
                METRIC_EXTENT_TRIM_PERCENT,
                100.0 - METRIC_EXTENT_TRIM_PERCENT,
            ],
            "appearance_extents_m": robust_extents.tolist(),
            "nominal_overall_height_m": spec.nominal_overall_height_m,
            "height_error_m": height_error_m,
            "height_tolerance_m": height_tolerance_m,
            "height_gate_passed": height_error_m <= height_tolerance_m,
            "root_translation_residual_m": 0.0,
            "root_rotation_residual_deg": 0.0,
            "root_contract": "appearance and physics consume the same T_world_object_canonical",
        },
        "poses": poses,
        "appearance": diagnostics,
        "vksplat_memory": memory,
        "physical_output": False,
        "gate_scope": "static pose, transform, renderer and conservative proxy-depth matte",
        "not_proven": [
            "dynamic pose stability",
            "gripper/tabletop occlusion under contact",
            "held-out real-view quality",
        ],
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not accepted:
        raise RuntimeError("Genesis Gaussian static gate failed")


if __name__ == "__main__":
    main()
