#!/usr/bin/env python3
"""Fit and texture a complete metric surface carrier from reviewed real views.

The carrier is a deadline-safe, explicit mesh proposal.  Geometry begins from
the closed procedural Graffiti Mickey proxy, is locked to the 95 mm metric
anchor, and receives a bounded differentiable silhouette fit on an AMD device.
Real pixels are projected only onto observed, front-facing vertices; remaining
regions retain neutral material colors and explicit low confidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from gaussian.object_pose_init import _load_reviewed_manifest, build_manual_ring
from gaussian.prepare_vista4d_object_input import VISTA4D_FRAMES, vista4d_camera_track
from sim.genesis_so101.gaussian_orbit_audit import (
    canonical_orbit_extrinsic,
    scaled_intrinsic,
)
from sim.genesis_so101.handover_asset import (
    DEFAULT_CONFIG,
    MATERIALS,
    MeshPart,
    build_surface_carrier_parts,
    load_spec,
)


SCHEMA_VERSION = "radeon_oneloop.surface_carrier.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.surface_carrier_done.v1"


class SurfaceCarrierError(ValueError):
    """Raised when reviewed inputs cannot produce an auditable carrier."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine_parts(
    parts: Sequence[MeshPart],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Combine mesh parts while preserving one material and part label per vertex."""

    if not parts:
        raise SurfaceCarrierError("surface carrier requires at least one mesh part")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_materials: list[str] = []
    vertex_parts: list[str] = []
    offset = 0
    for part in parts:
        value = np.asarray(part.vertices, dtype=np.float64)
        triangles = np.asarray(part.faces, dtype=np.int64)
        if value.ndim != 2 or value.shape[1] != 3:
            raise SurfaceCarrierError(f"part {part.name} has invalid vertices")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise SurfaceCarrierError(f"part {part.name} is not triangular")
        vertices.append(value)
        faces.append(triangles + offset)
        vertex_materials.extend([part.material] * len(value))
        vertex_parts.extend([part.name] * len(value))
        offset += len(value)
    return (
        np.concatenate(vertices),
        np.concatenate(faces),
        vertex_materials,
        vertex_parts,
    )


def metric_base_vertices(
    vertices: np.ndarray, *, object_height_m: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Uniformly scale a proxy so its complete Z extent equals the metric anchor."""

    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise SurfaceCarrierError("vertices must be a finite N x 3 array")
    if not math.isfinite(object_height_m) or object_height_m <= 0.0:
        raise SurfaceCarrierError("object height must be finite and positive")
    extents = np.ptp(value, axis=0)
    if np.any(extents <= 0.0):
        raise SurfaceCarrierError("carrier extents must be positive")
    scale = object_height_m / float(extents[2])
    scaled = value * scale
    return scaled, {
        "proxy_extents_m": extents.tolist(),
        "uniform_metric_scale": scale,
        "metric_base_extents_m": np.ptp(scaled, axis=0).tolist(),
        "object_height_m": object_height_m,
    }


def analytical_lateral_initialization(
    cameras: Sequence[dict[str, Any]],
    base_vertices: np.ndarray,
    *,
    object_height_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Initialize X/Y scale from real silhouette aspect ratios."""

    by_label = {str(camera["view_label"]): camera for camera in cameras}
    if set(by_label) != {"front", "right", "rear", "left"}:
        raise SurfaceCarrierError("carrier fit requires front/right/rear/left cameras")

    def target_extent(labels: Sequence[str]) -> tuple[float, list[float]]:
        estimates = []
        for label in labels:
            x0, y0, x1, y1 = by_label[label]["foreground_bbox_xyxy"]
            height = float(y1 - y0)
            if height <= 0.0:
                raise SurfaceCarrierError(f"empty reviewed silhouette for {label}")
            estimates.append(object_height_m * float(x1 - x0) / height)
        return float(np.median(estimates)), estimates

    target_x, x_estimates = target_extent(("front", "rear"))
    target_y, y_estimates = target_extent(("right", "left"))
    extents = np.ptp(base_vertices, axis=0)
    scales = np.asarray((target_x / extents[0], target_y / extents[1]), dtype=np.float64)
    if not np.isfinite(scales).all() or np.any((scales < 0.5) | (scales > 1.5)):
        raise SurfaceCarrierError(f"implausible lateral initialization: {scales.tolist()}")
    return scales, {
        "method": "median_real_silhouette_aspect_ratio",
        "target_x_estimates_m": x_estimates,
        "target_y_estimates_m": y_estimates,
        "target_x_m": target_x,
        "target_y_m": target_y,
        "initial_xy_scale": scales.tolist(),
    }


def apply_xy_scale(vertices: np.ndarray, scale_xy: Sequence[float]) -> np.ndarray:
    scale = np.asarray((float(scale_xy[0]), float(scale_xy[1]), 1.0), dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise SurfaceCarrierError("carrier scale must be finite and positive")
    return np.asarray(vertices, dtype=np.float64) * scale


def _camera_projection(
    vertices: np.ndarray,
    camera: dict[str, Any],
    *,
    output_size_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
    intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
    source_size = tuple(int(value) for value in camera["image_size_wh"])
    if output_size_wh is not None:
        intrinsic = scaled_intrinsic(intrinsic, source_size, output_size_wh)
    local = vertices @ transform[:3, :3].T + transform[:3, 3]
    depth = local[:, 2]
    safe = np.maximum(depth, 1.0e-9)
    uv = np.stack(
        (
            intrinsic[0, 0] * local[:, 0] / safe + intrinsic[0, 2],
            intrinsic[1, 1] * local[:, 1] / safe + intrinsic[1, 2],
        ),
        axis=1,
    )
    return uv, depth


def silhouette_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=bool)
    b = np.asarray(rhs, dtype=bool)
    if a.shape != b.shape:
        raise SurfaceCarrierError("silhouette shapes do not match")
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def rasterize_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict[str, Any],
    *,
    size_wh: tuple[int, int],
) -> np.ndarray:
    import cv2

    width, height = size_wh
    uv, depth = _camera_projection(vertices, camera, output_size_wh=size_wh)
    mask = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        if np.any(depth[face] <= 1.0e-6):
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
    return mask


def _fit_xy_differentiable(
    base_vertices: np.ndarray,
    cameras: Sequence[dict[str, Any]],
    alpha_masks: Sequence[np.ndarray],
    initial_xy: np.ndarray,
    *,
    device: str,
    steps: int,
    resolution: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Bounded differentiable fit using soft point-splat silhouettes."""

    import cv2
    import torch
    import torch.nn.functional as functional

    if steps <= 0 or resolution < 32 or max_points < 512:
        raise SurfaceCarrierError("invalid differentiable-fit budget")
    torch.manual_seed(seed)
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise SurfaceCarrierError("requested AMD CUDA/HIP device is unavailable")
    if len(base_vertices) > max_points:
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(len(base_vertices), max_points, replace=False))
        fit_vertices = base_vertices[selected]
    else:
        fit_vertices = base_vertices

    points = torch.as_tensor(fit_vertices, dtype=torch.float32, device=requested)
    log_xy = torch.nn.Parameter(
        torch.log(torch.as_tensor(initial_xy, dtype=torch.float32, device=requested))
    )
    initial_log = log_xy.detach().clone()
    lower = torch.log(torch.as_tensor(initial_xy * 0.80, dtype=torch.float32, device=requested))
    upper = torch.log(torch.as_tensor(initial_xy * 1.20, dtype=torch.float32, device=requested))
    optimizer = torch.optim.Adam((log_xy,), lr=0.035)
    axis = torch.arange(resolution, dtype=torch.float32, device=requested) + 0.5
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)

    camera_tensors = []
    targets = []
    for camera, alpha in zip(cameras, alpha_masks, strict=True):
        transform = torch.as_tensor(
            np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float32),
            device=requested,
        )
        source_size = tuple(int(value) for value in camera["image_size_wh"])
        intrinsic = scaled_intrinsic(
            np.asarray(camera["intrinsic_3x3"], dtype=np.float64),
            source_size,
            (resolution, resolution),
        )
        camera_tensors.append(
            (
                transform,
                torch.as_tensor(intrinsic, dtype=torch.float32, device=requested),
            )
        )
        resized = cv2.resize(
            np.asarray(alpha, dtype=np.float32),
            (resolution, resolution),
            interpolation=cv2.INTER_AREA,
        )
        targets.append(torch.as_tensor(resized, dtype=torch.float32, device=requested))

    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.stack((torch.exp(log_xy[0]), torch.exp(log_xy[1]), log_xy.new_tensor(1.0)))
        transformed_points = points * scale
        view_losses = []
        view_dice = []
        for (transform, intrinsic), target in zip(camera_tensors, targets, strict=True):
            local = transformed_points @ transform[:3, :3].T + transform[:3, 3]
            depth = torch.clamp(local[:, 2], min=1.0e-6)
            uv = torch.stack(
                (
                    intrinsic[0, 0] * local[:, 0] / depth + intrinsic[0, 2],
                    intrinsic[1, 1] * local[:, 1] / depth + intrinsic[1, 2],
                ),
                dim=1,
            )
            min_distance = torch.cdist(grid, uv).amin(dim=1).reshape(resolution, resolution)
            prediction = torch.sigmoid((1.10 - min_distance) / 0.20)
            intersection = torch.sum(prediction * target)
            dice = (2.0 * intersection + 1.0) / (
                torch.sum(prediction) + torch.sum(target) + 1.0
            )
            bce = functional.binary_cross_entropy(
                torch.clamp(prediction, 1.0e-5, 1.0 - 1.0e-5), target
            )
            view_losses.append(1.0 - dice + 0.10 * bce)
            view_dice.append(dice)
        regularization = 0.01 * torch.mean((log_xy - initial_log) ** 2)
        loss = torch.stack(view_losses).mean() + regularization
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_xy.clamp_(lower, upper)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "view_soft_dice": [float(value.detach().cpu()) for value in view_dice],
                    "xy_scale": torch.exp(log_xy.detach()).cpu().numpy().tolist(),
                }
            )
    if requested.type == "cuda":
        torch.cuda.synchronize(requested)
    elapsed = time.perf_counter() - start
    final_scale = torch.exp(log_xy.detach()).cpu().numpy().astype(np.float64)
    hardware = {
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "requested_device": device,
        "resolved_device": str(requested),
        "accelerator": (
            torch.cuda.get_device_name(requested) if requested.type == "cuda" else "CPU"
        ),
    }
    report = {
        "method": "bounded_differentiable_soft_point_splat_silhouette_fit",
        "seed": seed,
        "steps": steps,
        "resolution": resolution,
        "fit_points": int(len(fit_vertices)),
        "initial_xy_scale": initial_xy.tolist(),
        "final_xy_scale": final_scale.tolist(),
        "bounds_relative_to_initial": [0.80, 1.20],
        "elapsed_s": elapsed,
        "history": history,
    }
    return final_scale, report, hardware


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1.0e-12)


def project_real_vertex_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_materials: Sequence[str],
    cameras: Sequence[dict[str, Any]],
    images: Sequence[np.ndarray],
    alpha_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Project real colors onto visible, front-facing vertices with provenance."""

    normals = vertex_normals(vertices, faces)
    fallback = np.asarray(
        [MATERIALS[material] for material in vertex_materials], dtype=np.float64
    )
    color_sum = np.zeros_like(fallback)
    weight_sum = np.zeros(len(vertices), dtype=np.float64)
    max_weight = np.zeros(len(vertices), dtype=np.float64)
    source_count = np.zeros(len(vertices), dtype=np.uint8)
    per_view = []
    for camera, image, alpha in zip(cameras, images, alpha_masks, strict=True):
        uv, depth = _camera_projection(vertices, camera)
        height, width = alpha.shape
        px = np.rint(uv[:, 0]).astype(np.int64)
        py = np.rint(uv[:, 1]).astype(np.int64)
        inside = (
            (depth > 1.0e-6)
            & (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        )
        safe_x = np.clip(px, 0, width - 1)
        safe_y = np.clip(py, 0, height - 1)
        sampled_alpha = np.asarray(alpha[safe_y, safe_x], dtype=np.float64)
        center = np.asarray(camera["camera_center_m"], dtype=np.float64)
        direction = center - vertices
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12)
        facing = np.clip(np.sum(normals * direction, axis=1), 0.0, 1.0)

        # A quantized vertex z-buffer rejects obvious rear-surface projections
        # without claiming full mesh-raster visibility.
        flat = safe_y * width + safe_x
        zbuffer = np.full(width * height, np.inf, dtype=np.float64)
        np.minimum.at(zbuffer, flat[inside], depth[inside])
        visible = depth <= zbuffer[flat] + 0.004
        weight = sampled_alpha * facing**2
        valid = inside & visible & (sampled_alpha >= 0.25) & (facing >= 0.08)
        weight = np.where(valid, weight, 0.0)
        sampled_rgb = np.asarray(image[safe_y, safe_x], dtype=np.float64)
        color_sum += sampled_rgb * weight[:, None]
        weight_sum += weight
        max_weight = np.maximum(max_weight, weight)
        source_count += (weight >= 0.05).astype(np.uint8)
        per_view.append(
            {
                "view_id": camera["view_id"],
                "contributing_vertices": int(np.count_nonzero(weight >= 0.05)),
                "mean_positive_weight": (
                    float(np.mean(weight[weight > 0.0])) if np.any(weight > 0.0) else 0.0
                ),
            }
        )
    observed = weight_sum > 1.0e-6
    colors = fallback.copy()
    colors[observed] = color_sum[observed] / weight_sum[observed, None]
    colors = np.round(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    confidence = np.clip(max_weight, 0.0, 1.0).astype(np.float32)
    return colors, confidence, source_count, {
        "method": "normal_weighted_real_view_projection_with_vertex_zbuffer",
        "observed_vertex_fraction": float(np.mean(observed)),
        "fallback_vertex_fraction": float(np.mean(~observed)),
        "source_count_histogram": {
            str(value): int(np.count_nonzero(source_count == value))
            for value in range(5)
        },
        "views": per_view,
        "fallback": "procedural_material_color",
    }


def write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    source_count: np.ndarray,
) -> None:
    """Write a portable ASCII PLY with per-vertex evidence fields."""

    if not (
        len(vertices) == len(colors) == len(confidence) == len(source_count)
        and faces.ndim == 2
        and faces.shape[1] == 3
    ):
        raise SurfaceCarrierError("PLY arrays have inconsistent shapes")
    lines = [
        "ply",
        "format ascii 1.0",
        "comment complete metric surface carrier; formal=false",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
        "property uchar source_count",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(
        f"{x:.9f} {y:.9f} {z:.9f} {int(r)} {int(g)} {int(b)} {float(c):.7f} {int(s)}"
        for (x, y, z), (r, g, b), c, s in zip(
            vertices, colors, confidence, source_count, strict=True
        )
    )
    lines.extend(f"3 {int(a)} {int(b)} {int(c)}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def render_colored_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    camera_from_object: np.ndarray,
    intrinsic: np.ndarray,
    size_wh: tuple[int, int],
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """Render a deterministic painter-sorted colored carrier proposal."""

    import cv2

    width, height = size_wh
    transform = np.asarray(camera_from_object, dtype=np.float64)
    local = vertices @ transform[:3, :3].T + transform[:3, 3]
    depth = local[:, 2]
    safe = np.maximum(depth, 1.0e-9)
    uv = np.stack(
        (
            intrinsic[0, 0] * local[:, 0] / safe + intrinsic[0, 2],
            intrinsic[1, 1] * local[:, 1] / safe + intrinsic[1, 2],
        ),
        axis=1,
    )
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background_rgb
    alpha = np.zeros((height, width), dtype=np.uint8)
    valid_faces = faces[np.all(depth[faces] > 1.0e-6, axis=1)]
    order = np.argsort(np.mean(depth[valid_faces], axis=1))[::-1]
    for face in valid_faces[order]:
        polygon = np.rint(uv[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].min() >= height
        ):
            continue
        color = tuple(int(value) for value in np.mean(colors[face], axis=0))
        cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(alpha, polygon, 255, lineType=cv2.LINE_AA)
    return image, alpha


def _load_reviewed_inputs(
    manifest_path: Path, *, radius_m: float
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    from PIL import Image

    root = manifest_path.parent
    manifest = _load_reviewed_manifest(manifest_path)
    done_path = root / "DONE"
    if not done_path.is_file():
        raise SurfaceCarrierError("reviewed M1 stage has no DONE marker")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise SurfaceCarrierError("reviewed M1 DONE marker does not bind the manifest")

    camera_manifest = copy.deepcopy(manifest)
    source_records = []
    for view in camera_manifest["views"]:
        if "pose" not in view.get("roles", []):
            continue
        view["image"] = copy.deepcopy(view["neutral_image"])
        view["hard_mask"] = copy.deepcopy(view["soft_alpha"])
        source_records.append(
            {
                "view_id": view["id"],
                "neutral_rgb_sha256": view["neutral_image"]["sha256"],
                "soft_alpha_sha256": view["soft_alpha"]["sha256"],
                "source_sha256": view["source_sha256"],
            }
        )
    cameras_document, _, _ = build_manual_ring(
        camera_manifest, root, radius_m=radius_m
    )
    views = {view["id"]: view for view in manifest["views"] if "pose" in view.get("roles", [])}
    images = []
    alphas = []
    for camera in cameras_document["cameras"]:
        view = views[camera["view_id"]]
        rgb_path = root / view["neutral_image"]["relpath"]
        alpha_path = root / view["soft_alpha"]["relpath"]
        if sha256_file(rgb_path) != view["neutral_image"]["sha256"]:
            raise SurfaceCarrierError(f"neutral RGB hash mismatch for {view['id']}")
        if sha256_file(alpha_path) != view["soft_alpha"]["sha256"]:
            raise SurfaceCarrierError(f"soft alpha hash mismatch for {view['id']}")
        images.append(np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64) / 255.0)
        alphas.append(np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float64) / 255.0)
    return cameras_document, images, alphas, source_records


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def _contact_sheet(frames: Sequence[np.ndarray], columns: int = 4) -> np.ndarray:
    if not frames:
        raise SurfaceCarrierError("contact sheet requires frames")
    height, width = frames[0].shape[:2]
    rows = math.ceil(len(frames) / columns)
    output = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        output[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import imageio.v3 as iio

    manifest_path = args.m1_manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite surface carrier: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        cameras_document, images, alphas, source_records = _load_reviewed_inputs(
            manifest_path, radius_m=args.radius_m
        )
        cameras = cameras_document["cameras"]
        object_height_m = float(cameras_document["metric_anchor"]["value_m"])
        spec = load_spec(args.config)
        if not math.isclose(spec.nominal_overall_height_m, object_height_m, abs_tol=1.0e-9):
            raise SurfaceCarrierError("proxy and reviewed M1 metric anchors disagree")
        parts = build_surface_carrier_parts(spec)
        proxy_vertices, faces, vertex_materials, vertex_parts = combine_parts(parts)
        base_vertices, metric = metric_base_vertices(
            proxy_vertices, object_height_m=object_height_m
        )
        initial_xy, analytical = analytical_lateral_initialization(
            cameras, base_vertices, object_height_m=object_height_m
        )
        fit_xy, fit_report, hardware = _fit_xy_differentiable(
            base_vertices,
            cameras,
            alphas,
            initial_xy,
            device=args.device,
            steps=args.fit_steps,
            resolution=args.fit_resolution,
            max_points=args.max_fit_points,
            seed=args.seed,
        )
        vertices = apply_xy_scale(base_vertices, fit_xy)
        colors, confidence, source_count, texture_report = project_real_vertex_colors(
            vertices,
            faces,
            vertex_materials,
            cameras,
            images,
            alphas,
        )

        carrier_dir = staging / "carrier"
        audit_dir = staging / "audit"
        orbit_dir = staging / "orbit"
        frame_dir = orbit_dir / "frames"
        alpha_dir = orbit_dir / "alpha"
        for directory in (carrier_dir, audit_dir, frame_dir, alpha_dir):
            directory.mkdir(parents=True)
        ply_path = carrier_dir / "complete_surface_carrier.ply"
        write_colored_ply(
            ply_path, vertices, faces, colors, confidence, source_count
        )
        np.save(carrier_dir / "confidence.npy", confidence, allow_pickle=False)
        np.save(carrier_dir / "source_count.npy", source_count, allow_pickle=False)

        audit_size = (args.audit_size, args.audit_size)
        comparisons = []
        view_metrics = []
        for camera, real_rgb, target_alpha in zip(cameras, images, alphas, strict=True):
            rendered_mask = rasterize_silhouette(
                vertices, faces, camera, size_wh=audit_size
            )
            target_mask = cv2.resize(
                target_alpha,
                audit_size,
                interpolation=cv2.INTER_AREA,
            )
            target_binary = target_mask >= 0.5
            iou = silhouette_iou(rendered_mask >= 128, target_binary)
            source_size = tuple(int(value) for value in camera["image_size_wh"])
            intrinsic = scaled_intrinsic(
                np.asarray(camera["intrinsic_3x3"], dtype=np.float64),
                source_size,
                audit_size,
            )
            rendered_rgb, _ = render_colored_mesh(
                vertices,
                faces,
                colors,
                camera_from_object=np.asarray(
                    camera["world_to_camera_opencv_4x4"], dtype=np.float64
                ),
                intrinsic=intrinsic,
                size_wh=audit_size,
            )
            real_u8 = np.round(
                cv2.resize(real_rgb, audit_size, interpolation=cv2.INTER_AREA) * 255.0
            ).astype(np.uint8)
            mask_panel = np.repeat(
                np.round(target_mask[..., None] * 255.0).astype(np.uint8), 3, axis=2
            )
            overlay = np.round(
                0.5 * real_u8.astype(np.float64) + 0.5 * rendered_rgb.astype(np.float64)
            ).astype(np.uint8)
            comparisons.append(np.concatenate((real_u8, mask_panel, rendered_rgb, overlay), axis=1))
            view_metrics.append(
                {
                    "view_id": camera["view_id"],
                    "silhouette_iou": iou,
                    "target_support_fraction": float(np.mean(target_binary)),
                    "carrier_support_fraction": float(np.mean(rendered_mask >= 128)),
                }
            )
        iio.imwrite(audit_dir / "real_mask_carrier_overlay.png", np.concatenate(comparisons, axis=0))

        front = cameras[0]
        source_size = tuple(int(value) for value in front["image_size_wh"])
        orbit_size = (args.width, args.height)
        orbit_intrinsic = scaled_intrinsic(
            np.asarray(front["intrinsic_3x3"], dtype=np.float64),
            source_size,
            orbit_size,
        )
        frames = []
        orbit_alpha = []
        render_ms = []
        for index in range(VISTA4D_FRAMES):
            start = time.perf_counter()
            frame, alpha = render_colored_mesh(
                vertices,
                faces,
                colors,
                camera_from_object=canonical_orbit_extrinsic(
                    360.0 * index / VISTA4D_FRAMES, distance_m=args.radius_m
                ),
                intrinsic=orbit_intrinsic,
                size_wh=orbit_size,
            )
            render_ms.append((time.perf_counter() - start) * 1000.0)
            iio.imwrite(frame_dir / f"{index:05d}.png", frame)
            iio.imwrite(alpha_dir / f"{index:05d}.png", alpha)
            frames.append(frame)
            orbit_alpha.append(alpha >= 128)
        closure_frame, _ = render_colored_mesh(
            vertices,
            faces,
            colors,
            camera_from_object=canonical_orbit_extrinsic(360.0, distance_m=args.radius_m),
            intrinsic=orbit_intrinsic,
            size_wh=orbit_size,
        )
        closure_mae = float(
            np.mean(np.abs(closure_frame.astype(np.float32) - frames[0].astype(np.float32)))
            / 255.0
        )
        iio.imwrite(
            orbit_dir / "source_carrier.mp4",
            np.stack(frames),
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        contact_indices = tuple(round(index * (VISTA4D_FRAMES - 1) / 11) for index in range(12))
        iio.imwrite(
            audit_dir / "carrier_orbit_contact_sheet.png",
            _contact_sheet([frames[index] for index in contact_indices]),
        )
        target_c2w, target_intrinsics = vista4d_camera_track(
            frames=VISTA4D_FRAMES,
            intrinsic_3x3=orbit_intrinsic,
            distance_m=args.radius_m,
        )
        np.savez(
            orbit_dir / "target_cameras.npz",
            cam_c2w=target_c2w,
            intrinsics=target_intrinsics,
        )

        ious = [value["silhouette_iou"] for value in view_metrics]
        final_extents = np.ptp(vertices, axis=0)
        accepted_numeric = bool(
            abs(float(final_extents[2]) - object_height_m) <= 1.0e-7
            and statistics.fmean(ious) >= args.min_mean_iou
            and min(ious) >= args.min_view_iou
            and texture_report["observed_vertex_fraction"] >= args.min_observed_vertex_fraction
            and closure_mae <= 1.0 / 255.0
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "formal": False,
            "host_role": args.host_role,
            "physical_output": False,
            "redistribution": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": cameras_document["asset_name"],
            "carrier_role": "complete_metric_surface_source_for_vista4d_proposal",
            "accepted_numeric": accepted_numeric,
            "visual_review_required": True,
            "inputs": {
                "m1_manifest_sha256": sha256_file(manifest_path),
                "proxy_config_sha256": spec.config_sha256,
                "reviewed_views": source_records,
            },
            "geometry": {
                **metric,
                "analytical_initialization": analytical,
                "differentiable_fit": fit_report,
                "final_extents_m": final_extents.tolist(),
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
                "parts": sorted(set(vertex_parts)),
                "coordinate_convention": cameras_document["coordinate_convention"],
                "ply_relpath": "carrier/complete_surface_carrier.ply",
                "ply_sha256": sha256_file(ply_path),
            },
            "appearance": texture_report,
            "real_view_audit": {
                "views": view_metrics,
                "silhouette_iou_mean": statistics.fmean(ious),
                "silhouette_iou_min": min(ious),
                "thresholds": {
                    "min_mean_iou": args.min_mean_iou,
                    "min_view_iou": args.min_view_iou,
                    "min_observed_vertex_fraction": args.min_observed_vertex_fraction,
                },
            },
            "orbit": {
                "frames": VISTA4D_FRAMES,
                "image_size_wh": list(orbit_size),
                "fps": args.fps,
                "distance_m": args.radius_m,
                "camera_track": "closed_level_canonical_orbit_without_duplicate_endpoint",
                "cycle_closure_rgb_mae": closure_mae,
                "alpha_support_fraction": {
                    "min": min(float(mask.mean()) for mask in orbit_alpha),
                    "mean": statistics.fmean(float(mask.mean()) for mask in orbit_alpha),
                    "max": max(float(mask.mean()) for mask in orbit_alpha),
                },
                "render_ms": {
                    "mean": statistics.fmean(render_ms),
                    "p95": float(np.percentile(render_ms, 95)),
                    "max": max(render_ms),
                },
                "frames_relpath": "orbit/frames",
                "alpha_relpath": "orbit/alpha",
                "video_relpath": "orbit/source_carrier.mp4",
                "target_cameras_relpath": "orbit/target_cameras.npz",
            },
            "hardware": hardware,
            "allowed_next_role": "vista4d_source_video_conditioning_proposal",
            "not_proven": [
                "generated geometric detail beyond the procedural carrier",
                "held-out real-view quality",
                "collision geometry replacement",
                "formal single-Radeon evidence",
            ],
        }
        manifest_path_out = staging / "manifest.json"
        manifest_path_out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA_VERSION,
                    "status": "complete_numeric_candidate_visual_review_required",
                    "manifest_sha256": sha256_file(manifest_path_out),
                    "hashes_sha256": hashes_sha,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except BaseException as error:
        try:
            (staging / "FAILED").write_text(
                json.dumps(
                    {
                        "schema_version": DONE_SCHEMA_VERSION,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            failed = output.parent / f"{output.name}.FAILED"
            if not failed.exists():
                os.replace(staging, failed)
            else:
                shutil.rmtree(staging)
        finally:
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host-role", default="amd_apu_nonformal_surface_carrier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radius-m", type=float, default=0.3)
    parser.add_argument("--fit-steps", type=int, default=60)
    parser.add_argument("--fit-resolution", type=int, default=64)
    parser.add_argument("--max-fit-points", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--audit-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--min-mean-iou", type=float, default=0.55)
    parser.add_argument("--min-view-iou", type=float, default=0.45)
    parser.add_argument("--min-observed-vertex-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if min(args.audit_size, args.width, args.height) <= 0:
        raise SurfaceCarrierError("render dimensions must be positive")
    if not 1.0 <= args.fps <= 60.0:
        raise SurfaceCarrierError("fps must be in [1, 60]")
    if not 0.0 < args.min_view_iou <= args.min_mean_iou <= 1.0:
        raise SurfaceCarrierError("invalid silhouette IoU thresholds")
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
