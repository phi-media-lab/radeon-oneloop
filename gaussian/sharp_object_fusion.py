#!/usr/bin/env python3
"""Align SHARP-family geometry to the metric object frame and optionally donate appearance.

The result is generated fill evidence.  It never replaces the observed core and is
never eligible for held-out-real or formal metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


VIEW_ORDER = ("anchor_front", "anchor_right", "anchor_rear", "anchor_left")
SH0 = math.sqrt(1.0 / (4.0 * math.pi))


class SharpFusionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual_m: np.ndarray
    inliers: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash_manifest(root: Path) -> None:
    hash_path = root / "hashes.sha256"
    if not (root / "DONE").is_file() or not hash_path.is_file():
        raise SharpFusionError(f"input run is incomplete: {root}")
    for line in hash_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = (root / relative.lstrip("* ")).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise SharpFusionError(f"hash entry escapes run: {relative}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise SharpFusionError(f"input hash mismatch: {candidate}")


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the proper Umeyama similarity mapping source to target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape Nx3")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1.0e-15:
        raise ValueError("source correspondences are degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, values, right_t = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ sign @ right_t
    scale = float(np.trace(np.diag(values) @ sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def fit_similarity_trimmed(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_quantile: float = 0.8,
    iterations: int = 5,
    minimum_residual_m: float = 0.001,
) -> Similarity:
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    inliers = finite.copy()
    if int(inliers.sum()) < 100:
        raise SharpFusionError("too few finite SHARP/VGGT correspondences")
    for _ in range(iterations):
        scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        threshold = max(float(np.quantile(residual[inliers], trim_quantile)), minimum_residual_m)
        inliers = finite & (residual <= threshold)
    scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
    residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    return Similarity(scale, rotation, translation, residual, inliers)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for broadcastable wxyz quaternions."""
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        factor = 2.0 * math.sqrt(trace + 1.0)
        result = np.array(
            [
                0.25 * factor,
                (matrix[2, 1] - matrix[1, 2]) / factor,
                (matrix[0, 2] - matrix[2, 0]) / factor,
                (matrix[1, 0] - matrix[0, 1]) / factor,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            factor = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            result = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / factor,
                    0.25 * factor,
                    (matrix[0, 1] + matrix[1, 0]) / factor,
                    (matrix[0, 2] + matrix[2, 0]) / factor,
                ]
            )
        elif axis == 1:
            factor = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            result = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / factor,
                    (matrix[0, 1] + matrix[1, 0]) / factor,
                    0.25 * factor,
                    (matrix[1, 2] + matrix[2, 1]) / factor,
                ]
            )
        else:
            factor = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            result = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / factor,
                    (matrix[0, 2] + matrix[2, 0]) / factor,
                    (matrix[1, 2] + matrix[2, 1]) / factor,
                    0.25 * factor,
                ]
            )
    if result[0] < 0:
        result = -result
    return result / np.linalg.norm(result)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def project_mask_support(
    points: np.ndarray, cameras: list[dict[str, Any]], masks: np.ndarray
) -> np.ndarray:
    support = np.zeros(len(points), dtype=np.uint8)
    height, width = masks.shape[1:]
    for camera, mask in zip(cameras, masks):
        world_to_camera = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
        intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
        local = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        positive = local[:, 2] > 1.0e-6
        u = np.zeros(len(points), dtype=np.float64)
        v = np.zeros(len(points), dtype=np.float64)
        u[positive] = intrinsic[0, 0] * local[positive, 0] / local[positive, 2] + intrinsic[0, 2]
        v[positive] = intrinsic[1, 1] * local[positive, 1] / local[positive, 2] + intrinsic[1, 2]
        x = np.rint(u).astype(np.int64)
        y = np.rint(v).astype(np.int64)
        inside = positive & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        hit = np.zeros(len(points), dtype=bool)
        hit[inside] = mask[y[inside], x[inside]] >= 128
        support += hit.astype(np.uint8)
    return support


def voxel_cross_source_support(
    positions: np.ndarray,
    source_views: np.ndarray,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return neighboring-source support, voxel keys, origin, and grid shape."""
    from scipy.ndimage import maximum_filter  # imported only on the fusion host

    origin = np.floor((positions.min(axis=0) - voxel_size_m) / voxel_size_m) * voxel_size_m
    coordinates = np.floor((positions - origin) / voxel_size_m).astype(np.int64)
    shape = coordinates.max(axis=0) + 2
    if np.any(shape <= 0) or int(np.prod(shape, dtype=np.int64)) > 50_000_000:
        raise SharpFusionError(f"unsafe voxel grid shape: {shape.tolist()}")
    keys = np.ravel_multi_index(coordinates.T, tuple(int(value) for value in shape))
    support = np.zeros(len(positions), dtype=np.uint8)
    for view_index in range(len(VIEW_ORDER)):
        occupancy = np.zeros(tuple(int(value) for value in shape), dtype=bool)
        view_keys = np.unique(keys[source_views == view_index])
        occupancy.reshape(-1)[view_keys] = True
        nearby = maximum_filter(occupancy, size=3, mode="constant")
        support += nearby.reshape(-1)[keys].astype(np.uint8)
    return support, keys, origin, shape


def select_best_per_voxel(keys: np.ndarray, opacity: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    if len(candidates) == 0:
        return candidates
    order = np.lexsort((-opacity[candidates], keys[candidates]))
    ordered = candidates[order]
    ordered_keys = keys[ordered]
    first = np.concatenate(([True], ordered_keys[1:] != ordered_keys[:-1]))
    return ordered[first]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_views(m1: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {
        view["id"]: view
        for view in m1["views"]
        if view.get("prepared") and view["id"] in VIEW_ORDER
    }
    if set(result) != set(VIEW_ORDER):
        raise SharpFusionError(f"M1 canonical views do not match {VIEW_ORDER}")
    return result


def _generator_ply_paths(
    generator_root: Path, generator_manifest: dict[str, Any]
) -> dict[str, Path]:
    """Resolve the shared SHARP-family PLY layout without erasing provenance."""
    model = generator_manifest.get("model")
    if model == "apple_ml_sharp":
        result = {view_id: generator_root / "ply" / f"{view_id}.ply" for view_id in VIEW_ORDER}
    elif model == "UniSHARP":
        result = {
            item["view_id"]: generator_root / item["relpath"]
            for item in generator_manifest.get("outputs", [])
            if item.get("view_id") in VIEW_ORDER and str(item.get("relpath", "")).endswith(".ply")
        }
    else:
        raise SharpFusionError(f"unsupported SHARP-family generator model: {model!r}")
    if set(result) != set(VIEW_ORDER):
        raise SharpFusionError(f"generated PLY views do not match {VIEW_ORDER}: {sorted(result)}")
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise SharpFusionError(f"generated PLY files are missing: {missing}")
    return result


def _ply_metadata(ply: Any) -> tuple[np.ndarray, tuple[int, int]]:
    elements = {element.name: element for element in ply.elements}
    intrinsic = np.asarray(elements["intrinsic"].data["intrinsic"], dtype=np.float64).reshape(3, 3)
    size = np.asarray(elements["image_size"].data["image_size"], dtype=np.int64)
    return intrinsic, (int(size[0]), int(size[1]))


def _vertex_arrays(vertex: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(vertex.data[name]) for name in vertex.data.dtype.names}


def _surface_donor_fields(
    arrays: dict[str, np.ndarray], *, layers: int, expected_pixels: int
) -> dict[str, np.ndarray]:
    count = len(arrays["x"])
    if count != expected_pixels * layers:
        raise SharpFusionError(
            f"appearance donor has {count} Gaussians, expected {expected_pixels * layers}"
        )
    opacity = arrays["opacity"].reshape(expected_pixels, layers)
    selected_layers = np.argmax(opacity, axis=1)
    selected = np.arange(expected_pixels) * layers + selected_layers
    return {
        name: arrays[name][selected]
        for name in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity")
    }


def _fit_view_alignment(
    arrays: dict[str, np.ndarray],
    sharp_intrinsic: np.ndarray,
    sharp_size: tuple[int, int],
    pose: Any,
    view_index: int,
    metric_scale: float,
    *,
    layers: int,
    confidence_quantile: float,
) -> tuple[Similarity, dict[str, Any]]:
    count = len(arrays["x"])
    pixels = count // layers
    side = int(round(math.sqrt(pixels)))
    if side * side * layers != count:
        raise SharpFusionError(f"SHARP vertex layout is not square pixels x {layers} layers")
    xyz = np.stack([arrays[name] for name in ("x", "y", "z")], axis=1).reshape(side, side, layers, 3)
    opacity = arrays["opacity"].reshape(side, side, layers)
    layer = np.argmax(opacity, axis=2)
    source = np.take_along_axis(xyz, layer[..., None, None], axis=2)[..., 0, :]

    width, height = sharp_size
    u = sharp_intrinsic[0, 0] * source[..., 0] / source[..., 2] + sharp_intrinsic[0, 2]
    v = sharp_intrinsic[1, 1] * source[..., 1] / source[..., 2] + sharp_intrinsic[1, 2]
    target_height, target_width = pose["depth"].shape[1:]
    x = np.clip(np.rint(u * target_width / width).astype(np.int64), 0, target_width - 1)
    y = np.clip(np.rint(v * target_height / height).astype(np.int64), 0, target_height - 1)
    mask = pose["masks"][view_index] >= 128
    confidence = pose["depth_confidence"][view_index]
    threshold = float(np.quantile(confidence[mask], confidence_quantile))
    valid = (
        np.isfinite(source).all(axis=-1)
        & (source[..., 2] > 1.0e-6)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & mask[y, x]
        & (confidence[y, x] >= threshold)
    )
    source_fit = source[valid].astype(np.float64)
    raw_depth = pose["depth"][view_index][y[valid], x[valid]].astype(np.float64)
    target_z = raw_depth * metric_scale
    intrinsic = pose["intrinsics"][view_index].astype(np.float64)
    target = np.stack(
        (
            (x[valid] - intrinsic[0, 2]) / intrinsic[0, 0] * target_z,
            (y[valid] - intrinsic[1, 2]) / intrinsic[1, 1] * target_z,
            target_z,
        ),
        axis=1,
    )
    fit = fit_similarity_trimmed(source_fit, target)
    quantiles = np.quantile(fit.residual_m, [0.5, 0.8, 0.95, 0.99])
    summary = {
        "correspondence_count": int(len(source_fit)),
        "trimmed_inlier_count": int(fit.inliers.sum()),
        "scale": fit.scale,
        "rotation_correction_deg": rotation_angle_deg(fit.rotation),
        "translation_camera_m": fit.translation.tolist(),
        "residual_mm_p50_p80_p95_p99": (quantiles * 1000.0).tolist(),
        "confidence_threshold": threshold,
        "surface_selection": "maximum_opacity_of_eight_SHARP_layers_per_pixel",
    }
    return fit, summary


def _write_fused_ply(path: Path, arrays: dict[str, np.ndarray]) -> None:
    from plyfile import PlyData, PlyElement

    float_fields = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = [(name, "<f4") for name in float_fields] + [
        ("source_view", "u1"),
        ("cross_view_source_count", "u1"),
        ("silhouette_support_count", "u1"),
    ]
    output = np.empty(len(arrays["x"]), dtype=dtype)
    for name in float_fields:
        output[name] = arrays[name].astype(np.float32, copy=False)
    for name in ("source_view", "cross_view_source_count", "silhouette_support_count"):
        output[name] = arrays[name].astype(np.uint8, copy=False)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(path)


def _render_points(
    positions: np.ndarray,
    colors: np.ndarray,
    camera: dict[str, Any],
    *,
    size: int = 512,
    radius: int = 2,
) -> np.ndarray:
    world_to_camera = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
    intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
    local = positions @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    valid = local[:, 2] > 1.0e-6
    u = np.rint(intrinsic[0, 0] * local[:, 0] / np.maximum(local[:, 2], 1.0e-6) + intrinsic[0, 2]).astype(np.int64)
    v = np.rint(intrinsic[1, 1] * local[:, 1] / np.maximum(local[:, 2], 1.0e-6) + intrinsic[1, 2]).astype(np.int64)
    image = np.full((size, size, 3), 32, dtype=np.uint8)
    zbuffer = np.full(size * size, np.inf, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = u + dx
            y = v + dy
            inside = valid & (x >= 0) & (x < size) & (y >= 0) & (y < size)
            indices = np.flatnonzero(inside)
            flat = y[inside] * size + x[inside]
            order = np.lexsort((local[indices, 2], flat))
            ordered_flat = flat[order]
            ordered_indices = indices[order]
            first = np.concatenate(([True], ordered_flat[1:] != ordered_flat[:-1]))
            selected_flat = ordered_flat[first]
            selected_indices = ordered_indices[first]
            nearer = local[selected_indices, 2] < zbuffer[selected_flat]
            selected_flat = selected_flat[nearer]
            selected_indices = selected_indices[nearer]
            zbuffer[selected_flat] = local[selected_indices, 2]
            image.reshape(-1, 3)[selected_flat] = colors[selected_indices]
    return image


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image
    from plyfile import PlyData

    m1_path = args.m1_manifest.resolve()
    m1_root = m1_path.parent
    sharp_root = args.sharp_run.resolve()
    appearance_root = args.appearance_run.resolve() if args.appearance_run is not None else None
    pose_root = args.pose_run.resolve()
    pose_candidate = pose_root / "pose_candidate"
    verify_hash_manifest(m1_root)
    verify_hash_manifest(sharp_root)
    if appearance_root is not None:
        verify_hash_manifest(appearance_root)
    verify_hash_manifest(pose_root)
    verify_hash_manifest(args.pose_audit_manifest.resolve().parent)

    m1 = _load_json(m1_path)
    sharp_manifest = _load_json(sharp_root / "manifest.json")
    appearance_manifest = (
        _load_json(appearance_root / "manifest.json") if appearance_root is not None else None
    )
    pose_manifest = _load_json(pose_root / "manifest.json")
    pose_audit = _load_json(args.pose_audit_manifest.resolve())
    if pose_audit.get("review", {}).get("status") != "accepted_pose_and_coarse_geometry_initializer":
        raise SharpFusionError("pose visual audit has not accepted the metric initializer")
    if not pose_manifest.get("numeric_gate_passed"):
        raise SharpFusionError("pose run did not pass its numeric gate")
    generator_model = sharp_manifest.get("model")
    generator_plys = _generator_ply_paths(sharp_root, sharp_manifest)
    appearance_model = appearance_manifest.get("model") if appearance_manifest is not None else None
    appearance_plys = (
        _generator_ply_paths(appearance_root, appearance_manifest)
        if appearance_root is not None and appearance_manifest is not None
        else None
    )

    views = _canonical_views(m1)
    sharp_inputs = {item["basename"]: item["sha256"] for item in sharp_manifest["inputs"]}
    appearance_inputs = (
        {item["basename"]: item["sha256"] for item in appearance_manifest["inputs"]}
        if appearance_manifest is not None
        else None
    )
    for view_id, view in views.items():
        if sharp_inputs.get(f"{view_id}.png") != view["neutral_image"]["sha256"]:
            raise SharpFusionError(f"SHARP input does not match corrected M1 neutral image: {view_id}")
        if (
            appearance_inputs is not None
            and appearance_inputs.get(f"{view_id}.png") != view["neutral_image"]["sha256"]
        ):
            raise SharpFusionError(
                f"appearance donor input does not match corrected M1 neutral image: {view_id}"
            )

    pose = np.load(pose_candidate / "vggt_omega_pose_depth.npz")
    if tuple(pose["view_ids"].tolist()) != VIEW_ORDER:
        raise SharpFusionError(f"unexpected pose view order: {pose['view_ids'].tolist()}")
    cameras_document = _load_json(pose_candidate / "cameras_observed.json")
    cameras = cameras_document["cameras"]
    metric_similarity = _load_json(pose_candidate / "similarity_transform.json")
    metric_scale = float(metric_similarity["scale"])

    all_candidates: list[dict[str, np.ndarray]] = []
    alignments: list[dict[str, Any]] = []
    for view_index, view_id in enumerate(VIEW_ORDER):
        ply = PlyData.read(generator_plys[view_id])
        arrays = _vertex_arrays(ply["vertex"])
        intrinsic, image_size = _ply_metadata(ply)
        appearance_surface = None
        if appearance_plys is not None:
            appearance_ply = PlyData.read(appearance_plys[view_id])
            appearance_arrays = _vertex_arrays(appearance_ply["vertex"])
            appearance_intrinsic, appearance_size = _ply_metadata(appearance_ply)
            geometry_aspect = image_size[0] / image_size[1]
            appearance_aspect = appearance_size[0] / appearance_size[1]
            if abs(geometry_aspect - appearance_aspect) > 1.0e-6:
                raise SharpFusionError(f"appearance donor image aspect differs for {view_id}")
            appearance_surface = _surface_donor_fields(
                appearance_arrays,
                layers=args.sharp_layers,
                expected_pixels=len(arrays["x"]) // args.sharp_layers,
            )
        fit, alignment = _fit_view_alignment(
            arrays,
            intrinsic,
            image_size,
            pose,
            view_index,
            metric_scale,
            layers=args.sharp_layers,
            confidence_quantile=args.confidence_quantile,
        )
        alignment["view_id"] = view_id
        p50, _, p95, _ = alignment["residual_mm_p50_p80_p95_p99"]
        alignment["numeric_gate_passed"] = bool(
            alignment["correspondence_count"] >= args.min_correspondences
            and args.min_alignment_scale <= fit.scale <= args.max_alignment_scale
            and alignment["rotation_correction_deg"] <= args.max_rotation_correction_deg
            and p50 <= args.max_residual_p50_mm
            and p95 <= args.max_residual_p95_mm
            and np.linalg.det(fit.rotation) > 0.999
        )
        alignments.append(alignment)

        positions_camera = np.stack([arrays[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
        if len(positions_camera) % args.sharp_layers != 0:
            raise SharpFusionError("SHARP Gaussian count is not divisible by its layer count")
        if args.layer_policy == "max_opacity_surface":
            opacity_layers = arrays["opacity"].reshape(-1, args.sharp_layers)
            selected_layers = np.argmax(opacity_layers, axis=1)
            layer_keep = np.zeros_like(opacity_layers, dtype=bool)
            layer_keep[np.arange(len(layer_keep)), selected_layers] = True
            layer_keep = layer_keep.reshape(-1)
        else:
            layer_keep = np.ones(len(positions_camera), dtype=bool)
        width, height = image_size
        u = intrinsic[0, 0] * positions_camera[:, 0] / positions_camera[:, 2] + intrinsic[0, 2]
        v = intrinsic[1, 1] * positions_camera[:, 1] / positions_camera[:, 2] + intrinsic[1, 2]
        mask = pose["masks"][view_index]
        mask_height, mask_width = mask.shape
        x = np.clip(np.rint(u * mask_width / width).astype(np.int64), 0, mask_width - 1)
        y = np.clip(np.rint(v * mask_height / height).astype(np.int64), 0, mask_height - 1)
        own_mask = (
            np.isfinite(positions_camera).all(axis=1)
            & (positions_camera[:, 2] > 1.0e-6)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
            & (mask[y, x] >= 128)
            & (arrays["opacity"] >= args.min_opacity_logit)
            & layer_keep
        )

        camera_to_world = np.asarray(cameras[view_index]["camera_to_world_opencv_4x4"], dtype=np.float64)
        world_rotation = camera_to_world[:3, :3] @ fit.rotation
        world_translation = camera_to_world[:3, :3] @ fit.translation + camera_to_world[:3, 3]
        positions = fit.scale * (positions_camera @ world_rotation.T) + world_translation
        bounded = np.max(np.abs(positions), axis=1) <= args.max_canonical_abs_m
        silhouette_support = project_mask_support(positions, cameras, pose["masks"])
        keep = own_mask & bounded & (silhouette_support >= args.min_silhouette_views)
        pixel_index = np.arange(len(positions_camera), dtype=np.int64) // args.sharp_layers

        def appearance_values(name: str) -> np.ndarray:
            if appearance_surface is None:
                return arrays[name][keep].astype(np.float32)
            return appearance_surface[name][pixel_index[keep]].astype(np.float32)

        quaternions = np.stack([arrays[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float64)
        quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12)
        transformed_quaternions = quaternion_multiply(
            quaternion_from_rotation(world_rotation)[None, :], quaternions
        )
        transformed_quaternions /= np.maximum(
            np.linalg.norm(transformed_quaternions, axis=1, keepdims=True), 1.0e-12
        )
        candidate = {
            "x": positions[keep, 0].astype(np.float32),
            "y": positions[keep, 1].astype(np.float32),
            "z": positions[keep, 2].astype(np.float32),
            "f_dc_0": appearance_values("f_dc_0"),
            "f_dc_1": appearance_values("f_dc_1"),
            "f_dc_2": appearance_values("f_dc_2"),
            "opacity": appearance_values("opacity"),
            "scale_0": (arrays["scale_0"][keep] + math.log(fit.scale)).astype(np.float32),
            "scale_1": (arrays["scale_1"][keep] + math.log(fit.scale)).astype(np.float32),
            "scale_2": (arrays["scale_2"][keep] + math.log(fit.scale)).astype(np.float32),
            "rot_0": transformed_quaternions[keep, 0].astype(np.float32),
            "rot_1": transformed_quaternions[keep, 1].astype(np.float32),
            "rot_2": transformed_quaternions[keep, 2].astype(np.float32),
            "rot_3": transformed_quaternions[keep, 3].astype(np.float32),
            "source_view": np.full(int(keep.sum()), view_index, dtype=np.uint8),
            "silhouette_support_count": silhouette_support[keep].astype(np.uint8),
        }
        alignment["gaussian_counts"] = {
            "input": int(len(positions_camera)),
            "inside_own_observed_mask_and_opacity": int(own_mask.sum()),
            "inside_canonical_bound": int((own_mask & bounded).sum()),
            "silhouette_supported_candidate": int(keep.sum()),
        }
        all_candidates.append(candidate)

    if not all(item["numeric_gate_passed"] for item in alignments):
        diagnostic = [
            {
                "view_id": item["view_id"],
                "scale": item["scale"],
                "rotation_correction_deg": item["rotation_correction_deg"],
                "residual_mm_p50_p80_p95_p99": item["residual_mm_p50_p80_p95_p99"],
                "correspondence_count": item["correspondence_count"],
                "numeric_gate_passed": item["numeric_gate_passed"],
            }
            for item in alignments
        ]
        raise SharpFusionError(
            "one or more SHARP-family-to-VGGT alignment gates failed: "
            + json.dumps(diagnostic, sort_keys=True)
        )

    names = list(all_candidates[0])
    combined = {name: np.concatenate([item[name] for item in all_candidates]) for name in names}
    positions = np.stack([combined[name] for name in ("x", "y", "z")], axis=1)
    cross_source, voxel_keys, voxel_origin, voxel_shape = voxel_cross_source_support(
        positions,
        combined["source_view"],
        voxel_size_m=args.voxel_size_m,
    )
    eligible = cross_source >= args.min_source_views
    if args.reduction_policy == "keep_cross_source_supported":
        selected = np.flatnonzero(eligible)
        if len(selected) > args.max_fused_gaussians:
            ranking = np.argsort(combined["opacity"][selected], kind="stable")
            selected = np.sort(selected[ranking[-args.max_fused_gaussians :]])
    else:
        selected = select_best_per_voxel(voxel_keys, combined["opacity"], eligible)
    if len(selected) < args.min_fused_gaussians:
        raise SharpFusionError(f"only {len(selected)} fused generated Gaussians passed")

    fused = {name: value[selected] for name, value in combined.items()}
    fused["cross_view_source_count"] = cross_source[selected]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generator_slug = str(generator_model).lower().replace("-", "_")
    if appearance_model is not None:
        appearance_slug = str(appearance_model).lower().replace("-", "_")
        generator_slug = f"{generator_slug}_geometry_{appearance_slug}_appearance"
    ply_path = output / f"{generator_slug}_generated_fill_canonical.ply"
    _write_fused_ply(ply_path, fused)

    fused_positions = np.stack([fused[name] for name in ("x", "y", "z")], axis=1)
    fused_colors = np.clip(
        np.stack([fused[f"f_dc_{index}"] for index in range(3)], axis=1) * SH0 + 0.5,
        0.0,
        1.0,
    )
    fused_colors_u8 = np.round(fused_colors * 255.0).astype(np.uint8)
    target_images = []
    render_images = []
    for camera, view_id in zip(cameras, VIEW_ORDER):
        target = Image.open(m1_root / views[view_id]["neutral_image"]["relpath"]).convert("RGB")
        target_images.append(np.asarray(target.resize((512, 512), Image.Resampling.LANCZOS)))
        render_images.append(_render_points(fused_positions, fused_colors_u8, camera))
    montage = np.concatenate((np.concatenate(target_images, axis=1), np.concatenate(render_images, axis=1)), axis=0)
    Image.fromarray(montage).save(output / "projection_audit.jpg", quality=94)

    robust_extent = {
        axis: np.quantile(fused_positions[:, index], [0.005, 0.5, 0.995]).tolist()
        for index, axis in enumerate(("x_m", "y_m", "z_m"))
    }
    robust_height = robust_extent["z_m"][2] - robust_extent["z_m"][0]
    quality = {
        "schema_version": "radeon_oneloop.sharp_generated_fill_quality.v1",
        "formal": False,
        "acceptance_status": "pending_visual_generated_fill_review",
        "geometry_generator_model": generator_model,
        "appearance_generator_model": appearance_model or generator_model,
        "appearance_donor_policy": (
            "maximum_opacity_surface_color_and_opacity"
            if appearance_model is not None
            else "geometry_generator_native"
        ),
        "alignment": alignments,
        "fusion": {
            "layer_policy": args.layer_policy,
            "candidate_gaussians": int(len(positions)),
            "cross_source_supported_gaussians": int(eligible.sum()),
            "fused_voxel_gaussians": int(len(selected)),
            "output_gaussians": int(len(selected)),
            "reduction_policy": args.reduction_policy,
            "voxel_size_m": args.voxel_size_m,
            "voxel_origin_m": voxel_origin.tolist(),
            "voxel_grid_shape": voxel_shape.tolist(),
            "minimum_source_views_with_one_voxel_neighborhood": args.min_source_views,
            "minimum_observed_silhouette_views": args.min_silhouette_views,
            "source_view_counts": {
                VIEW_ORDER[index]: int(np.sum(fused["source_view"] == index))
                for index in range(len(VIEW_ORDER))
            },
            "cross_source_support_histogram": {
                str(value): int(np.sum(fused["cross_view_source_count"] == value))
                for value in range(1, len(VIEW_ORDER) + 1)
            },
            "robust_extent_p005_p50_p995": robust_extent,
            "robust_height_m": float(robust_height),
        },
        "numeric_gate_passed": bool(
            args.min_robust_height_m <= robust_height <= args.max_robust_height_m
            and len(selected) >= args.min_fused_gaussians
        ),
        "limitations": [
            f"{generator_model} is a generated geometry and appearance prior, not observed evidence",
            (
                f"{appearance_model} donates generated surface color and opacity only"
                if appearance_model is not None
                else "no separate appearance donor was used"
            ),
            "point-projection montage is a geometry audit, not a full Gaussian renderer",
            "four source views do not justify unobserved-view photometric claims",
        ],
    }
    (output / "quality.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "radeon_oneloop.generated_fill_provenance.v1",
        "formal": False,
        "asset_name": m1["asset_name"],
        "m1_manifest_sha256": sha256_file(m1_path),
        "pose_run_manifest_sha256": sha256_file(pose_root / "manifest.json"),
        "pose_visual_audit_manifest_sha256": sha256_file(args.pose_audit_manifest.resolve()),
        "generator_model": generator_model,
        "generator_run_manifest_sha256": sha256_file(sharp_root / "manifest.json"),
        "geometry_generator_model": generator_model,
        "geometry_generator_run_manifest_sha256": sha256_file(sharp_root / "manifest.json"),
        "appearance_generator_model": appearance_model,
        "appearance_generator_run_manifest_sha256": (
            sha256_file(appearance_root / "manifest.json") if appearance_root is not None else None
        ),
        "sharp_run_manifest_sha256": (
            sha256_file(sharp_root / "manifest.json") if generator_model == "apple_ml_sharp" else None
        ),
        "metric_anchor": m1["metric_anchor"],
        "coordinate_convention": m1["coordinate_convention"],
        "observed_core_mutated": False,
        "provenance_class": "generated_fill_candidate",
        "eligible_for_heldout_real_metrics": False,
        "eligible_for_formal_metrics": False,
        "output_ply_sha256": sha256_file(ply_path),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not quality["numeric_gate_passed"]:
        raise SharpFusionError("generated fill did not pass the metric extent gate")
    return {"quality": quality, "provenance": provenance}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--pose-audit-manifest", type=Path, required=True)
    parser.add_argument("--sharp-run", type=Path, required=True)
    parser.add_argument("--appearance-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sharp-layers", type=int, default=8)
    parser.add_argument(
        "--layer-policy",
        choices=("max_opacity_surface", "all_layers"),
        default="max_opacity_surface",
    )
    parser.add_argument("--confidence-quantile", type=float, default=0.1)
    parser.add_argument("--min-correspondences", type=int, default=20_000)
    parser.add_argument("--min-alignment-scale", type=float, default=0.15)
    parser.add_argument("--max-alignment-scale", type=float, default=0.45)
    parser.add_argument("--max-rotation-correction-deg", type=float, default=8.0)
    parser.add_argument("--max-residual-p50-mm", type=float, default=3.0)
    parser.add_argument("--max-residual-p95-mm", type=float, default=20.0)
    parser.add_argument("--min-opacity-logit", type=float, default=-2.944439)
    parser.add_argument("--min-silhouette-views", type=int, default=2)
    parser.add_argument("--max-canonical-abs-m", type=float, default=0.12)
    parser.add_argument("--voxel-size-m", type=float, default=0.0015)
    parser.add_argument("--min-source-views", type=int, default=2)
    parser.add_argument(
        "--reduction-policy",
        choices=("keep_cross_source_supported", "best_per_voxel"),
        default="keep_cross_source_supported",
    )
    parser.add_argument("--max-fused-gaussians", type=int, default=300_000)
    parser.add_argument("--min-fused-gaussians", type=int, default=10_000)
    parser.add_argument("--min-robust-height-m", type=float, default=0.07)
    parser.add_argument("--max-robust-height-m", type=float, default=0.12)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({
        "numeric_gate_passed": result["quality"]["numeric_gate_passed"],
        "fused_gaussians": result["quality"]["fusion"]["fused_voxel_gaussians"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
