#!/usr/bin/env python3
"""Build a deterministic, accelerator-independent object COLMAP dataset.

The camera gauge comes from the reviewed 95 mm silhouette and nominal four-view
ring.  Initial geometry is a CPU visual hull: a fixed metric voxel grid is
projected into all four reviewed masks, its six-neighbour boundary is extracted,
and a fixed seed selects the final points.  No learned depth, generated view, or
secondary-accelerator artifact enters this dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from .object_colmap_export import rotation_matrix_to_colmap_qvec
from .object_pose_init import _load_reviewed_manifest, build_manual_ring


class ManualRingColmapError(ValueError):
    """Raised when the reviewed inputs cannot form a deterministic visual hull."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def six_neighbour_boundary(occupied: np.ndarray) -> np.ndarray:
    """Return occupied voxels that do not have all six occupied neighbours."""

    value = np.asarray(occupied, dtype=bool)
    if value.ndim != 3 or min(value.shape) < 3:
        raise ValueError("occupied grid must be 3-D with every dimension at least three")
    interior = np.zeros_like(value)
    interior[1:-1, 1:-1, 1:-1] = (
        value[1:-1, 1:-1, 1:-1]
        & value[:-2, 1:-1, 1:-1]
        & value[2:, 1:-1, 1:-1]
        & value[1:-1, :-2, 1:-1]
        & value[1:-1, 2:, 1:-1]
        & value[1:-1, 1:-1, :-2]
        & value[1:-1, 1:-1, 2:]
    )
    return value & ~interior


def _project(
    points: np.ndarray, camera: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
    intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    depth = camera_points[:, 2]
    safe_depth = np.where(depth > 1.0e-9, depth, 1.0)
    u = intrinsic[0, 0] * camera_points[:, 0] / safe_depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_points[:, 1] / safe_depth + intrinsic[1, 2]
    return u, v, depth


def visual_hull_surface(
    cameras: Sequence[dict[str, Any]],
    masks: Sequence[np.ndarray],
    *,
    half_extents_m: Sequence[float],
    resolution: int,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Carve and extract a metric visual-hull surface from four masks."""

    if len(cameras) != 4 or len(masks) != 4:
        raise ManualRingColmapError("visual hull requires exactly four cameras and masks")
    if resolution < 32:
        raise ManualRingColmapError("grid resolution must be at least 32")
    half = np.asarray(half_extents_m, dtype=np.float64)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise ManualRingColmapError("half extents must be three finite positive values")
    if chunk_size <= 0:
        raise ManualRingColmapError("chunk size must be positive")

    x = np.linspace(-half[0], half[0], resolution, dtype=np.float32)
    y = np.linspace(-half[1], half[1], resolution, dtype=np.float32)
    z = np.linspace(-half[2], half[2], resolution, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    occupied = np.ones(len(points), dtype=bool)
    for camera, raw_mask in zip(cameras, masks, strict=True):
        mask = np.asarray(raw_mask)
        if mask.ndim != 2:
            raise ManualRingColmapError("every mask must be a 2-D array")
        height, width = mask.shape
        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            u, v, depth = _project(points[start:end], camera)
            px = np.rint(u).astype(np.int64)
            py = np.rint(v).astype(np.int64)
            valid = (
                (depth > 1.0e-9)
                & (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
            )
            inside = np.zeros(end - start, dtype=bool)
            good = np.flatnonzero(valid)
            inside[good] = mask[py[good], px[good]] > 0
            occupied[start:end] &= inside

    volume = occupied.reshape(resolution, resolution, resolution)
    if not np.any(volume):
        raise ManualRingColmapError("reviewed masks have an empty visual-hull intersection")
    # The exact 95 mm vertical bounds are the metric gauge and may be touched.
    # Lateral bounds are deliberately padded and must not truncate the hull.
    lateral_boundary_hits = int(
        np.count_nonzero(volume[:, :, 0])
        + np.count_nonzero(volume[:, :, -1])
        + np.count_nonzero(volume[:, 0, :])
        + np.count_nonzero(volume[:, -1, :])
    )
    if lateral_boundary_hits:
        raise ManualRingColmapError(
            f"visual hull touches padded lateral bounds at {lateral_boundary_hits} voxels"
        )
    boundary = six_neighbour_boundary(volume)
    surface = points[np.flatnonzero(boundary.ravel())].astype(np.float64)
    spacing = (2.0 * half / (resolution - 1)).tolist()
    return surface, {
        "grid_resolution": resolution,
        "half_extents_m": half.tolist(),
        "voxel_spacing_m": spacing,
        "occupied_voxels": int(np.count_nonzero(volume)),
        "surface_voxels": int(len(surface)),
        "lateral_boundary_hits": lateral_boundary_hits,
    }


def _surface_colors(
    points: np.ndarray,
    cameras: Sequence[dict[str, Any]],
    images: Sequence[np.ndarray],
) -> np.ndarray:
    centers = np.asarray([camera["camera_center_m"] for camera in cameras])
    point_norm = np.linalg.norm(points, axis=1, keepdims=True)
    directions = points / np.maximum(point_norm, 1.0e-12)
    camera_directions = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    selected = np.argmax(directions @ camera_directions.T, axis=1)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for camera_index, (camera, image) in enumerate(
        zip(cameras, images, strict=True)
    ):
        indices = np.flatnonzero(selected == camera_index)
        if not len(indices):
            continue
        u, v, depth = _project(points[indices], camera)
        px = np.clip(np.rint(u).astype(np.int64), 0, image.shape[1] - 1)
        py = np.clip(np.rint(v).astype(np.int64), 0, image.shape[0] - 1)
        if np.any(depth <= 0.0):
            raise ManualRingColmapError("selected surface color projection is behind a camera")
        colors[indices] = image[py, px]
    return colors


def _write_hashes(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "hashes.sha256").write_text("".join(lines), encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    m1_path = args.m1_manifest.resolve()
    m1_root = m1_path.parent
    m1 = _load_reviewed_manifest(m1_path)
    cameras_document, similarity, pose_quality = build_manual_ring(
        m1, m1_root, radius_m=args.radius_m
    )
    cameras = cameras_document["cameras"]
    views = {view["id"]: view for view in m1["views"] if "photometric" in view["roles"]}
    if set(views) != {camera["view_id"] for camera in cameras}:
        raise ManualRingColmapError("manual cameras do not match the four photometric views")

    images = []
    masks = []
    for camera in cameras:
        view = views[camera["view_id"]]
        image_path = (m1_root / view["image"]["relpath"]).resolve()
        mask_path = (m1_root / view["hard_mask"]["relpath"]).resolve()
        if sha256_file(image_path) != view["image"]["sha256"]:
            raise ManualRingColmapError(f"image hash mismatch for {view['id']}")
        if sha256_file(mask_path) != view["hard_mask"]["sha256"]:
            raise ManualRingColmapError(f"mask hash mismatch for {view['id']}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ManualRingColmapError(f"failed to decode reviewed view {view['id']}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks.append(mask)

    height_m = float(m1["metric_anchor"]["value_m"])
    half_extents = (height_m * args.lateral_extent_ratio,) * 2 + (height_m / 2.0,)
    surface, hull = visual_hull_surface(
        cameras,
        masks,
        half_extents_m=half_extents,
        resolution=args.grid_resolution,
    )
    if len(surface) < args.max_points:
        raise ManualRingColmapError(
            f"visual-hull surface has only {len(surface)} points; need {args.max_points}"
        )
    generator = np.random.default_rng(args.sample_seed)
    selected = np.sort(
        generator.choice(len(surface), size=args.max_points, replace=False)
    )
    points = surface[selected]
    colors = _surface_colors(points, cameras, images)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        image_dir = staging / "images"
        mask_dir = staging / "masks"
        sparse_dir = staging / "sparse/0"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
        image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
        front = next(camera for camera in cameras if camera["view_id"] == "anchor_front")
        entries = [(front, True)] + [(camera, False) for camera in cameras]
        exported = []
        for image_id, (camera, duplicate) in enumerate(entries, start=1):
            view = views[camera["view_id"]]
            name = (
                "000_eval_probe_anchor_front.png"
                if duplicate
                else f"{camera['view_id']}.png"
            )
            shutil.copy2(m1_root / view["image"]["relpath"], image_dir / name)
            shutil.copy2(m1_root / view["hard_mask"]["relpath"], mask_dir / name)
            width, height = camera["image_size_wh"]
            intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
            transform = np.asarray(
                camera["world_to_camera_opencv_4x4"], dtype=np.float64
            )
            qvec = rotation_matrix_to_colmap_qvec(transform[:3, :3])
            values = [*qvec.tolist(), *transform[:3, 3].tolist()]
            camera_lines.append(
                f"{image_id} PINHOLE {width} {height} "
                f"{intrinsic[0, 0]:.12g} {intrinsic[1, 1]:.12g} "
                f"{intrinsic[0, 2]:.12g} {intrinsic[1, 2]:.12g}\n"
            )
            image_lines.append(
                f"{image_id} "
                + " ".join(f"{value:.17g}" for value in values)
                + f" {image_id} {name}\n\n"
            )
            exported.append(
                {
                    "view_id": camera["view_id"],
                    "image_id": image_id,
                    "evaluation_probe_duplicate": duplicate,
                    "included_in_training": not duplicate,
                    "image_sha256": view["image"]["sha256"],
                    "mask_sha256": view["hard_mask"]["sha256"],
                }
            )
        (sparse_dir / "cameras.txt").write_text("".join(camera_lines), encoding="utf-8")
        (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
        for point_id, (point, color) in enumerate(zip(points, colors, strict=True), start=1):
            point_lines.append(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")

        robust_extent = (
            np.quantile(points, 0.995, axis=0) - np.quantile(points, 0.005, axis=0)
        )
        manifest = {
            "schema_version": "radeon_oneloop.manual_ring_visual_hull_colmap.v1",
            "formal": False,
            "formal_input_eligible": True,
            "asset_name": m1["asset_name"],
            "m1_manifest_sha256": sha256_file(m1_path),
            "metric_anchor": m1["metric_anchor"],
            "camera_initialization": {
                "method": cameras_document["method"],
                "radius_m": args.radius_m,
                "similarity": similarity,
                "quality": pose_quality,
            },
            "visual_hull": {
                **hull,
                "sample_seed": args.sample_seed,
                "sampled_surface_points": int(len(points)),
                "robust_extent_m_p005_p995": robust_extent.tolist(),
            },
            "views": exported,
            "evaluation_split_rule": (
                "front duplicate is filename-sorted evaluation probe; all four unique "
                "reviewed views are training inputs; no held-out quality claim"
            ),
            "provenance": {
                "images": "reviewed_observed_tier_A_only",
                "masks": "reviewed_observed_masks_treated_as_dataset_annotations",
                "poses": "deterministic_manual_ring_CPU",
                "initial_points": "deterministic_reviewed_mask_visual_hull_CPU",
                "learned_depth": False,
                "generated_views": False,
                "generated_geometry": False,
                "secondary_accelerator_artifacts": False,
            },
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.object_asset_stage_done.v1",
                    "stage": "manual_ring_visual_hull_colmap",
                    "status": "done_formal_input_candidate",
                    "manifest_sha256": sha256_file(staging / "dataset_manifest.json"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "stage": "manual_ring_visual_hull_colmap",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED")
        if not failed.exists():
            os.replace(staging, failed)
        else:
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=0.30)
    parser.add_argument("--lateral-extent-ratio", type=float, default=0.65)
    parser.add_argument("--grid-resolution", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.radius_m <= 0.0 or args.lateral_extent_ratio <= 0.5:
        raise ValueError("radius must be positive and lateral extent ratio must exceed 0.5")
    if args.max_points < 10_000:
        raise ValueError("formal object initialization requires at least 10,000 points")
    return args


def main() -> None:
    manifest = build_dataset(parse_args())
    print(
        json.dumps(
            {
                "surface_points": manifest["visual_hull"]["surface_voxels"],
                "sampled_points": manifest["visual_hull"]["sampled_surface_points"],
                "secondary_accelerator_artifacts": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
