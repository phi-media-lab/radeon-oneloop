#!/usr/bin/env python3
"""Run VGGT-Omega on reviewed object anchors and export a gated metric pose candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

from gaussian.object_pose_init import (
    PoseInitError,
    canonical_orbit_direction,
    deterministic_confident_sample,
    fit_proper_similarity,
    sha256_file,
    validate_labeled_camera_layout,
)


def _load_inputs(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.object_asset_manifest.v1":
        raise PoseInitError("expected an M1 object asset manifest")
    if manifest.get("formal") is not False:
        raise PoseInitError("VGGT-Omega development inference must be nonformal")
    if manifest.get("summary", {}).get("mask_review_status") != "reviewed_pass":
        raise PoseInitError("M1 masks must have reviewed_pass status")
    views = [view for view in manifest["views"] if "pose" in view["roles"]]
    if len(views) != 4:
        raise PoseInitError(f"VGGT-Omega pose inference requires exactly four anchors, got {len(views)}")
    for view in views:
        if view["tier"] != "A" or view["provenance"] != "observed" or not view["prepared"]:
            raise PoseInitError(f"{view['id']} is not a prepared observed tier-A view")
        if view.get("nominal_camera_orbit_deg") is None:
            raise PoseInitError(f"{view['id']} is missing nominal orbit metadata")
    return manifest, views


def _checked_path(root: Path, record: dict[str, str], view_id: str, kind: str) -> Path:
    path = (root / record["relpath"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PoseInitError(f"{view_id} {kind} escaped the M1 root") from exc
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise PoseInitError(f"{view_id} {kind} is absent or has a hash mismatch")
    return path


def _balanced_shape(height: int, width: int, resolution: int, patch: int) -> tuple[int, int]:
    aspect = height / max(width, 1)
    tokens = (resolution // patch) ** 2
    width_patches = max(1, int(np.round(np.sqrt(tokens / aspect))))
    height_patches = max(1, int(np.round(tokens / width_patches)))
    return height_patches * patch, width_patches * patch


def _preprocess(image: np.ndarray, mask: np.ndarray, resolution: int, patch: int) -> tuple[torch.Tensor, np.ndarray]:
    height, width = image.shape[:2]
    target_height, target_width = _balanced_shape(height, width, resolution, patch)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return tensor, mask


def _depth_to_world(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    view_count, height, width = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    result = np.empty((view_count, height, width, 3), dtype=np.float32)
    for index in range(view_count):
        z = depth[index]
        intrinsic = intrinsics[index]
        x = (grid_x - intrinsic[0, 2]) / intrinsic[0, 0] * z
        y = (grid_y - intrinsic[1, 2]) / intrinsic[1, 1] * z
        camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        rotation = extrinsics[index, :3, :3]
        translation = extrinsics[index, :3, 3]
        result[index] = ((camera - translation) @ rotation).reshape(height, width, 3)
    return result


def _camera_centers_and_up(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    up_vectors = []
    for extrinsic in extrinsics:
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        centers.append(-rotation.T @ translation)
        up_vectors.append(-(rotation.T @ np.asarray([0.0, 1.0, 0.0])))
    return np.asarray(centers), np.asarray(up_vectors)


def _reprojection_audit(
    points: np.ndarray,
    source_views: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside_count = np.zeros(len(points), dtype=np.uint8)
    mask_hit_count = np.zeros(len(points), dtype=np.uint8)
    own_hit = np.zeros(len(points), dtype=bool)
    for camera_index, (extrinsic, intrinsic, mask) in enumerate(zip(extrinsics, intrinsics, masks)):
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        camera_points = points @ rotation.T + translation
        z = camera_points[:, 2]
        positive = z > 1.0e-6
        u = np.zeros(len(points), dtype=np.float64)
        v = np.zeros(len(points), dtype=np.float64)
        u[positive] = intrinsic[0, 0] * camera_points[positive, 0] / z[positive] + intrinsic[0, 2]
        v[positive] = intrinsic[1, 1] * camera_points[positive, 1] / z[positive] + intrinsic[1, 2]
        x = np.rint(u).astype(np.int64)
        y = np.rint(v).astype(np.int64)
        inside = positive & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        hit = np.zeros(len(points), dtype=bool)
        hit[inside] = mask[y[inside], x[inside]] >= 128
        inside_count += inside.astype(np.uint8)
        mask_hit_count += hit.astype(np.uint8)
        own = source_views == camera_index
        own_hit[own] = hit[own]
    return inside_count, mask_hit_count, own_hit


def _matrix(value: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in value.tolist()]


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors_u8):
            stream.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("VGGT-Omega job requires exactly one visible ROCm device via the torch cuda API")
    if "MI300X" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected MI300X, got {torch.cuda.get_device_name(0)}")
    manifest_path = args.m1_manifest.resolve()
    root = manifest_path.parent
    manifest, views = _load_inputs(manifest_path)
    args.output.mkdir(parents=True, exist_ok=False)

    tensors = []
    masks = []
    images_np = []
    input_records = []
    for view in views:
        image_path = _checked_path(root, view["neutral_image"], view["id"], "neutral_image")
        mask_path = _checked_path(root, view["hard_mask"], view["id"], "hard_mask")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise PoseInitError(f"OpenCV could not decode {view['id']} inputs")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, mask_rs = _preprocess(rgb, mask, args.image_resolution, args.patch_size)
        tensors.append(tensor)
        masks.append(mask_rs)
        images_np.append(tensor.permute(1, 2, 0).numpy())
        input_records.append(
            {
                "view_id": view["id"],
                "neutral_image_sha256": view["neutral_image"]["sha256"],
                "hard_mask_sha256": view["hard_mask"]["sha256"],
            }
        )
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise PoseInitError(f"preprocessed anchor shapes differ: {sorted(shapes)}")
    image_tensor = torch.stack(tensors).to(args.device)
    mask_stack = np.stack(masks)

    sys.path.insert(0, str(args.vggt_omega_root.resolve()))
    from vggt_omega.models import VGGTOmega  # noqa: WPS433
    from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: WPS433

    model = VGGTOmega(enable_alignment=False).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(image_tensor)
    extrinsic_tensor, intrinsic_tensor = encoding_to_camera(
        prediction["pose_enc"], prediction["images"].shape[-2:]
    )
    extrinsics = extrinsic_tensor.detach().float().cpu().numpy().squeeze(0)
    intrinsics = intrinsic_tensor.detach().float().cpu().numpy().squeeze(0)
    depth = prediction["depth"].detach().float().cpu().numpy().squeeze(0)[..., 0]
    confidence = prediction["depth_conf"].detach().float().cpu().numpy().squeeze(0)
    world_points = _depth_to_world(depth, extrinsics, intrinsics)

    point_chunks = []
    color_chunks = []
    confidence_chunks = []
    source_chunks = []
    view_summaries = []
    for index, view in enumerate(views):
        valid = (mask_stack[index] >= 128) & np.isfinite(depth[index]) & (depth[index] > 1.0e-6)
        threshold = float(np.quantile(confidence[index][valid], args.confidence_quantile))
        valid &= confidence[index] >= threshold
        selected = deterministic_confident_sample(
            valid,
            confidence[index],
            limit=args.max_points_per_view,
            seed=args.sample_seed + index,
        )
        points = world_points[index].reshape(-1, 3)[selected]
        colors = images_np[index].reshape(-1, 3)[selected]
        scores = confidence[index].reshape(-1)[selected]
        finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
        points, colors, scores = points[finite], colors[finite], scores[finite]
        point_chunks.append(points)
        color_chunks.append(colors)
        confidence_chunks.append(scores)
        source_chunks.append(np.full(len(points), index, dtype=np.uint8))
        view_summaries.append(
            {
                "view_id": view["id"],
                "mask_pixels": int((mask_stack[index] >= 128).sum()),
                "confidence_threshold": threshold,
                "selected_points": int(len(points)),
                "depth_min": float(depth[index][valid].min()),
                "depth_max": float(depth[index][valid].max()),
            }
        )
    points_raw = np.concatenate(point_chunks).astype(np.float32)
    colors = np.concatenate(color_chunks).astype(np.float32)
    scores = np.concatenate(confidence_chunks).astype(np.float32)
    source_views = np.concatenate(source_chunks)
    inside_count, mask_hit_count, own_hit = _reprojection_audit(
        points_raw, source_views, extrinsics, intrinsics, mask_stack
    )

    centers_raw, up_raw = _camera_centers_and_up(extrinsics)
    nominal = np.asarray(
        [
            canonical_orbit_direction(
                view["nominal_camera_orbit_deg"]["azimuth"],
                view["nominal_camera_orbit_deg"]["elevation"],
            )
            for view in views
        ]
    )
    layout_gate = validate_labeled_camera_layout(
        centers_raw,
        nominal,
        camera_up_vectors=up_raw,
        max_angular_error_deg=args.max_camera_angular_error_deg,
        max_radius_cv=args.max_camera_radius_cv,
        min_mean_up_dot=args.min_mean_up_dot,
    )
    sim_to_ring = fit_proper_similarity(centers_raw, nominal)
    points_ring = (sim_to_ring["scale"] * (sim_to_ring["rotation"] @ points_raw.T)).T + sim_to_ring[
        "translation"
    ]
    q_low, q_high = np.quantile(points_ring[:, 2], [args.height_low_quantile, args.height_high_quantile])
    height_ring = float(q_high - q_low)
    if not np.isfinite(height_ring) or height_ring <= 1.0e-9:
        raise PoseInitError("VGGT-Omega point cloud has a degenerate robust height")
    metric_height = float(manifest["metric_anchor"]["value_m"])
    metric_scale = metric_height / height_ring
    origin_ring = np.median(points_ring, axis=0)
    points_metric = ((points_ring - origin_ring) * metric_scale).astype(np.float32)
    total_scale = float(sim_to_ring["scale"] * metric_scale)
    total_rotation = sim_to_ring["rotation"]
    total_translation = (sim_to_ring["translation"] - origin_ring) * metric_scale

    cameras_metric = []
    metric_extrinsics = []
    for index, view in enumerate(views):
        center_ring = sim_to_ring["aligned"][index]
        center_metric = (center_ring - origin_ring) * metric_scale
        raw_rotation_c2w = extrinsics[index, :3, :3].T
        metric_rotation_c2w = total_rotation @ raw_rotation_c2w
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = metric_rotation_c2w
        camera_to_world[:3, 3] = center_metric
        world_to_camera = np.linalg.inv(camera_to_world)
        metric_extrinsics.append(world_to_camera[:3])
        cameras_metric.append(
            {
                "view_id": view["id"],
                "view_label": view["view_label"],
                "image_size_wh": [int(image_tensor.shape[-1]), int(image_tensor.shape[-2])],
                "intrinsic_3x3": _matrix(intrinsics[index]),
                "camera_center_m": center_metric.tolist(),
                "camera_up_canonical": (total_rotation @ up_raw[index]).tolist(),
                "world_to_camera_opencv_4x4": _matrix(world_to_camera),
                "camera_to_world_opencv_4x4": _matrix(camera_to_world),
                "nominal_orbit_deg": view["nominal_camera_orbit_deg"],
            }
        )
    metric_extrinsics_np = np.asarray(metric_extrinsics, dtype=np.float32)

    cross_view_support_fraction = float(np.mean(mask_hit_count >= 2))
    own_reprojection_fraction = float(np.mean(own_hit))
    numeric_gate = bool(
        layout_gate["passed"]
        and own_reprojection_fraction >= args.min_own_reprojection_fraction
        and cross_view_support_fraction >= args.min_cross_view_support_fraction
    )
    quality = {
        "schema_version": "radeon_oneloop.object_pose_quality.v1",
        "formal": False,
        "method": "vggt_omega_1b_512",
        "layout_gate": layout_gate,
        "reprojection": {
            "own_mask_hit_fraction": own_reprojection_fraction,
            "cross_view_mask_support_fraction_ge_2": cross_view_support_fraction,
            "mask_hit_count_mean": float(np.mean(mask_hit_count)),
            "inside_camera_count_mean": float(np.mean(inside_count)),
            "thresholds": {
                "min_own_mask_hit_fraction": float(args.min_own_reprojection_fraction),
                "min_cross_view_mask_support_fraction_ge_2": float(args.min_cross_view_support_fraction),
            },
        },
        "metric_alignment": {
            "anchor_height_m": metric_height,
            "raw_robust_height_after_ring_alignment": height_ring,
            "height_quantiles": [args.height_low_quantile, args.height_high_quantile],
            "metric_scale_after_ring_alignment": metric_scale,
            "origin_estimator": "median_of_selected_object_points",
            "origin_contract_status": "provisional_pending_body_landmark_review",
        },
        "point_count": int(len(points_raw)),
        "confidence_mean": float(np.mean(scores)),
        "confidence_p50": float(np.median(scores)),
        "numeric_gate_passed": numeric_gate,
        "acceptance_status": "pending_visual_point_cloud_and_identity_review"
        if numeric_gate
        else "rejected_by_numeric_pose_gate",
        "limitations": [
            "product listing views may describe object rotation rather than a calibrated camera orbit",
            "metric scale is imposed from the 95 mm product specification",
            "the plush-body origin is provisional until landmark review",
            "learned depth and pose are candidate priors, not measured geometry",
        ],
    }
    cameras_document = {
        "schema_version": "radeon_oneloop.object_cameras.v1",
        "formal": False,
        "asset_name": manifest["asset_name"],
        "method": "vggt_omega_1b_512_metric_candidate",
        "camera_model": "PINHOLE_OPENCV",
        "coordinate_convention": manifest["coordinate_convention"],
        "metric_anchor": manifest["metric_anchor"],
        "cameras": cameras_metric,
    }
    similarity = {
        "schema_version": "radeon_oneloop.object_similarity_transform.v1",
        "formal": False,
        "source_frame": "vggt_omega_world_unaligned",
        "target_frame": "object_canonical_metric_provisional_origin",
        "scale": total_scale,
        "rotation_3x3": _matrix(total_rotation),
        "translation_m": total_translation.tolist(),
        "proper_rotation_determinant": float(np.linalg.det(total_rotation)),
        "metric_anchor": manifest["metric_anchor"],
        "status": "candidate_pending_p2_visual_and_identity_gate",
    }

    np.savez_compressed(
        args.output / "vggt_omega_pose_depth.npz",
        positions_raw=points_raw,
        positions_metric=points_metric,
        colors=colors,
        confidence=scores,
        source_view_index=source_views,
        inside_camera_count=inside_count,
        mask_hit_count=mask_hit_count,
        own_mask_hit=own_hit,
        depth=depth.astype(np.float32),
        depth_confidence=confidence.astype(np.float32),
        masks=mask_stack.astype(np.uint8),
        extrinsics_raw_world_to_camera=extrinsics.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        extrinsics_metric_world_to_camera=metric_extrinsics_np,
        view_ids=np.asarray([view["id"] for view in views]),
    )
    _write_ply(args.output / "sparse_points_metric.ply", points_metric, colors)
    for name, value in (
        ("cameras_observed.json", cameras_document),
        ("similarity_transform.json", similarity),
        ("quality.json", quality),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "radeon_oneloop.vggt_omega_object_inference.v1",
        "formal": False,
        "model": "VGGT-Omega-1B-512",
        "m1_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vggt_omega_commit": args.vggt_omega_commit,
        "inputs": input_records,
        "preprocess": {
            "image_resolution": args.image_resolution,
            "patch_size": args.patch_size,
            "tensor_shape": list(image_tensor.shape),
            "neutral_background": True,
            "confidence_quantile": args.confidence_quantile,
            "sample_method": "deterministic_uniform_foreground_after_confidence_gate",
            "sample_seed": args.sample_seed,
            "max_points_per_view": args.max_points_per_view,
        },
        "state_missing_keys": len(missing),
        "state_unexpected_keys": len(unexpected),
        "torch_peak_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "view_summaries": view_summaries,
        "quality": quality,
    }
    (args.output / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vggt-omega-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-omega-commit", required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--confidence-quantile", type=float, default=0.10)
    parser.add_argument("--max-points-per-view", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    parser.add_argument("--height-low-quantile", type=float, default=0.005)
    parser.add_argument("--height-high-quantile", type=float, default=0.995)
    parser.add_argument("--max-camera-angular-error-deg", type=float, default=25.0)
    parser.add_argument("--max-camera-radius-cv", type=float, default=0.35)
    parser.add_argument("--min-mean-up-dot", type=float, default=0.5)
    parser.add_argument("--min-own-reprojection-fraction", type=float, default=0.98)
    parser.add_argument("--min-cross-view-support-fraction", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
