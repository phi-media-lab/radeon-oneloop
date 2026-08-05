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
    full_geometry_candidate_asset,
    layered_preview_asset,
    observed_core_asset,
    transform_from_pos_quat_wxyz,
)
from .handover_asset import load_spec
from .scene import OBJECT_START_POS, build


METRIC_EXTENT_TRIM_PERCENT = 0.01
METRIC_SUPPORT_SIGMA = 2.0


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


def gaussian_support_extents(
    xyz: np.ndarray,
    scales: np.ndarray,
    rotations_wxyz: np.ndarray,
    *,
    sigma: float = METRIC_SUPPORT_SIGMA,
    trim_percent: float = METRIC_EXTENT_TRIM_PERCENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return axis-aligned Gaussian support extents, including splat size.

    A 3DGS asset is an ellipsoid field, not a point cloud.  Center-only
    bounds systematically understate its visible metric envelope.  We use a
    fixed two-sigma covariance envelope and apply the same tiny per-tail trim
    used by the center diagnostic to reject isolated export outliers.
    """

    centers = np.asarray(xyz, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    quaternions = np.asarray(rotations_wxyz, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Gaussian centers must have shape Nx3")
    if scale_values.shape != centers.shape or quaternions.shape != (len(centers), 4):
        raise ValueError("Gaussian scales and rotations must have shapes Nx3 and Nx4")
    if not all(np.isfinite(value).all() for value in (centers, scale_values, quaternions)):
        raise ValueError("Gaussian support inputs must be finite")
    if np.any(scale_values <= 0.0):
        raise ValueError("Gaussian scales must be positive")
    if sigma <= 0.0 or not 0.0 <= trim_percent < 50.0:
        raise ValueError("sigma must be positive and trim_percent must be in [0, 50)")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("Gaussian rotations contain a zero quaternion")
    w, x, y, z = (quaternions / norms).T
    rotations = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    axis_sigma = np.sqrt(
        np.einsum("nij,nj,nij->ni", rotations, scale_values * scale_values, rotations)
    )
    lower = centers - sigma * axis_sigma
    upper = centers + sigma * axis_sigma
    full_extents = np.max(upper, axis=0) - np.min(lower, axis=0)
    robust_lower = np.percentile(lower, trim_percent, axis=0)
    robust_upper = np.percentile(upper, 100.0 - trim_percent, axis=0)
    return full_extents, robust_upper - robust_lower


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so101-asset-root", type=Path, required=True)
    parser.add_argument("--observed-core-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--layered-preview", action="store_true")
    parser.add_argument("--full-geometry-candidate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v3 as iio

    if args.layered_preview and args.full_geometry_candidate:
        raise ValueError("appearance asset modes are mutually exclusive")
    if args.full_geometry_candidate:
        asset = full_geometry_candidate_asset(args.observed_core_root)
    elif args.layered_preview:
        asset = layered_preview_asset(args.observed_core_root)
    else:
        asset = observed_core_asset(args.observed_core_root)
    asset_audit = asset.validate()
    gaussians = read_3dgs_ply(asset.ply_path)
    center_full_extents, center_robust_extents = gaussian_center_extents(gaussians["xyz"])
    full_extents, robust_extents = gaussian_support_extents(
        gaussians["xyz"], gaussians["scales"], gaussians["rotations"]
    )
    metric_extents = robust_extents if args.layered_preview else center_robust_extents
    metric_extent_method = (
        "anisotropic_gaussian_two_sigma_support"
        if args.layered_preview
        else "lightly_trimmed_gaussian_centers"
    )
    spec = load_spec()
    height_error_m = abs(float(metric_extents[2]) - spec.nominal_overall_height_m)
    height_tolerance_m = max(0.002, 0.03 * spec.nominal_overall_height_m)

    task, handles = build(
        args.so101_asset_root.resolve(),
        seed=args.seed,
        show_viewer=False,
        object_visualization=False,
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
        and diagnostics["object_visualization"] is False
        and diagnostics["object_mesh_path"].endswith("_collision.obj")
        and diagnostics["compositor"] == "gaussian_self_depth"
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
            "full_center_extents_m": center_full_extents.tolist(),
            "robust_center_extents_m": center_robust_extents.tolist(),
            "support_sigma": METRIC_SUPPORT_SIGMA,
            "full_support_extents_m": full_extents.tolist(),
            "robust_percentile_range": [
                METRIC_EXTENT_TRIM_PERCENT,
                100.0 - METRIC_EXTENT_TRIM_PERCENT,
            ],
            "robust_support_extents_m": robust_extents.tolist(),
            "appearance_metric_extents_m": metric_extents.tolist(),
            "metric_extent_method": metric_extent_method,
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
        "layered_preview": args.layered_preview,
        "full_geometry_candidate": args.full_geometry_candidate,
        "gate_scope": (
            "static pose, transform, renderer and Gaussian self-depth compositing"
        ),
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
