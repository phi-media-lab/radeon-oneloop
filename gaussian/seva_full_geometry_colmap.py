#!/usr/bin/env python3
"""Build an experimental full-orbit geometry dataset from a SEVA orbit.

This is intentionally distinct from :mod:`seva_pseudoview_colmap`.  The
reviewed SEVA orbit was accepted only as low-confidence appearance evidence.
Here its 49 silhouettes are used to create a *generated geometry hypothesis*
for visual real2sim, never metric truth, collision geometry, or formal
evidence.  The output remains pending a human geometry audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from gaussian.audit_seva_orbit import AUDIT_SCHEMA, ANCHORS
from gaussian.colmap_cardinal_camera_export import parse_colmap_text
from gaussian.export_observed_initialization import load_colmap_points
from gaussian.manual_ring_colmap import _project, six_neighbour_boundary
from gaussian.object_colmap_export import rotation_matrix_to_colmap_qvec
from gaussian.prepare_four_view_generation import sha256_file
from gaussian.provenance_quarantine import assert_not_quarantined
from gaussian.record_seva_four_view_run import SCHEMA_VERSION as SEVA_RUN_SCHEMA
from gaussian.record_seva_orbit_review import ACCEPTED, REVIEW_SCHEMA


SCHEMA = "radeon_oneloop.seva_full_geometry_colmap_dataset.v1"
DONE_SCHEMA = "radeon_oneloop.seva_full_geometry_colmap_dataset_done.v1"
BASE_DATASET_SCHEMA = "radeon_oneloop.seva_pseudoview_colmap_dataset.v1"
IMAGE_SIZE_WH = (576, 576)
ANCHOR_INDICES = frozenset(index for _, index in ANCHORS)
VIEW_ORDER = ("front", "right", "back", "left")


class SevaFullGeometryError(ValueError):
    """Raised when a generated geometry candidate violates its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SevaFullGeometryError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SevaFullGeometryError(f"{label} must be a JSON object")
    return value


def _verify_hash_index(root: Path) -> None:
    path = root / "hashes.sha256"
    if not path.is_file():
        raise SevaFullGeometryError(f"missing hash index: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as exc:
            raise SevaFullGeometryError(f"malformed hash line {line_number}: {path}") from exc
        relative = Path(relpath)
        candidate = root / relative
        if relative.is_absolute() or ".." in relative.parts or not candidate.is_file():
            raise SevaFullGeometryError(f"unsafe or missing hash target: {relpath}")
        if sha256_file(candidate) != digest:
            raise SevaFullGeometryError(f"hash mismatch: {root.name}/{relpath}")


def opengl_c2w_to_metric_opencv_w2c(c2w: np.ndarray, radius_m: float) -> np.ndarray:
    value = np.asarray(c2w, dtype=np.float64)
    if value.shape == (3, 4):
        value = np.vstack((value, np.asarray([0.0, 0.0, 0.0, 1.0])))
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise SevaFullGeometryError("SEVA camera must be one finite 3 x 4 or 4 x 4 matrix")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise SevaFullGeometryError("metric camera radius must be positive")
    center_norm = float(np.linalg.norm(value[:3, 3]))
    if center_norm <= 0:
        raise SevaFullGeometryError("SEVA camera center has zero radius")
    metric_gl = value.copy()
    metric_gl[:3, 3] *= radius_m / center_norm
    opencv_c2w = metric_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(opencv_c2w)


def _colmap_camera_line_square(camera_id: int, intrinsic: np.ndarray) -> str:
    width, height = IMAGE_SIZE_WH
    return (
        f"{camera_id} PINHOLE {width} {height} {intrinsic[0, 0]:.17g} "
        f"{intrinsic[1, 1]:.17g} {intrinsic[0, 2]:.17g} {intrinsic[1, 2]:.17g}\n"
    )


def _colmap_image_line(
    image_id: int, camera_id: int, name: str, world_to_camera: np.ndarray
) -> str:
    transform = np.asarray(world_to_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SevaFullGeometryError("COLMAP world-to-camera transform is invalid")
    qvec = rotation_matrix_to_colmap_qvec(transform[:3, :3])
    tx, ty, tz = transform[:3, 3]
    return (
        f"{image_id} {qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} "
        f"{qvec[3]:.17g} {tx:.17g} {ty:.17g} {tz:.17g} {camera_id} {name}\n\n"
    )


def required_support_count(view_count: int, minimum_fraction: float) -> int:
    """Return the inclusive silhouette-vote threshold for one voxel."""

    if view_count < 4:
        raise SevaFullGeometryError("at least four generated views are required")
    if not math.isfinite(minimum_fraction) or not 0.5 < minimum_fraction <= 1.0:
        raise SevaFullGeometryError("minimum support fraction must be in (0.5, 1]")
    return int(math.ceil(view_count * minimum_fraction))


def support_visual_hull_surface(
    cameras: Sequence[dict[str, Any]],
    masks: Sequence[np.ndarray],
    *,
    half_extents_m: Sequence[float],
    resolution: int,
    minimum_support_fraction: float,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Carve a support-voted visual hull from a noisy generated orbit."""

    if len(cameras) != len(masks):
        raise SevaFullGeometryError("camera and mask counts differ")
    support_required = required_support_count(len(cameras), minimum_support_fraction)
    if resolution < 32:
        raise SevaFullGeometryError("grid resolution must be at least 32")
    half = np.asarray(half_extents_m, dtype=np.float64)
    if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half <= 0.0):
        raise SevaFullGeometryError("half extents must be three finite positive values")
    if chunk_size <= 0:
        raise SevaFullGeometryError("chunk size must be positive")

    x = np.linspace(-half[0], half[0], resolution, dtype=np.float32)
    y = np.linspace(-half[1], half[1], resolution, dtype=np.float32)
    z = np.linspace(-half[2], half[2], resolution, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    support = np.zeros(len(points), dtype=np.uint8)
    for camera, raw_mask in zip(cameras, masks, strict=True):
        mask = np.asarray(raw_mask)
        if mask.ndim != 2:
            raise SevaFullGeometryError("every mask must be a 2-D array")
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
            good = np.flatnonzero(valid)
            votes = np.zeros(end - start, dtype=np.uint8)
            votes[good] = (mask[py[good], px[good]] > 0).astype(np.uint8)
            support[start:end] += votes

    occupied = support >= support_required
    volume = occupied.reshape(resolution, resolution, resolution)
    if not np.any(volume):
        raise SevaFullGeometryError("generated masks have an empty support visual hull")
    boundary = six_neighbour_boundary(volume)
    surface_indices = np.flatnonzero(boundary.ravel())
    surface = points[surface_indices].astype(np.float64)
    if not len(surface):
        raise SevaFullGeometryError("generated support hull has no surface voxels")
    boundary_hits = {
        "x_min": int(np.count_nonzero(volume[:, :, 0])),
        "x_max": int(np.count_nonzero(volume[:, :, -1])),
        "y_min": int(np.count_nonzero(volume[:, 0, :])),
        "y_max": int(np.count_nonzero(volume[:, -1, :])),
        "z_min": int(np.count_nonzero(volume[0, :, :])),
        "z_max": int(np.count_nonzero(volume[-1, :, :])),
    }
    surface_support = support[surface_indices]
    return surface, {
        "grid_resolution": resolution,
        "half_extents_m": half.tolist(),
        "voxel_spacing_m": (2.0 * half / (resolution - 1)).tolist(),
        "view_count": len(cameras),
        "minimum_support_fraction": minimum_support_fraction,
        "minimum_support_views": support_required,
        "occupied_voxels": int(np.count_nonzero(volume)),
        "surface_voxels": int(len(surface)),
        "surface_support_views_min": int(surface_support.min()),
        "surface_support_views_median": float(np.median(surface_support)),
        "boundary_hits": boundary_hits,
    }


def _load_base_dataset(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = root / "dataset_manifest.json"
    done_path = root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise SevaFullGeometryError("base SEVA pseudo-view dataset is incomplete")
    manifest = _load_json(manifest_path, "base SEVA dataset manifest")
    done = _load_json(done_path, "base SEVA dataset DONE")
    if manifest.get("schema_version") != BASE_DATASET_SCHEMA:
        raise SevaFullGeometryError("unexpected base SEVA dataset schema")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise SevaFullGeometryError("base dataset DONE does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(root / "hashes.sha256"):
        raise SevaFullGeometryError("base dataset DONE does not bind its hashes")
    _verify_hash_index(root)
    if manifest.get("initial_points", {}).get("source") != (
        "observed_real_mask_CPU_visual_hull"
    ):
        raise SevaFullGeometryError("base dataset lacks an observed metric-scale prior")
    points, _ = load_colmap_points(root / "sparse/0/points3D.txt")
    if len(points) < 1000 or not np.all(np.isfinite(points)):
        raise SevaFullGeometryError("base point cloud is invalid")
    assert_not_quarantined([("base_SEVA_dataset", manifest)])
    return manifest, points


def _load_sources(
    seva_root: Path, audit_root: Path, review_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    seva = _load_json(seva_root / "manifest.json", "SEVA manifest")
    seva_done = _load_json(seva_root / "DONE", "SEVA DONE")
    if seva.get("schema_version") != SEVA_RUN_SCHEMA or len(seva.get("frames", [])) != 49:
        raise SevaFullGeometryError("SEVA source must contain one 49-frame orbit")
    if seva_done.get("manifest_sha256") != sha256_file(seva_root / "manifest.json"):
        raise SevaFullGeometryError("SEVA DONE does not bind its manifest")
    if seva_done.get("hashes_sha256") != sha256_file(seva_root / "hashes.sha256"):
        raise SevaFullGeometryError("SEVA DONE does not bind its hashes")
    _verify_hash_index(seva_root)

    audit = _load_json(audit_root / "metrics.json", "SEVA audit")
    audit_done = _load_json(audit_root / "DONE", "SEVA audit DONE")
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise SevaFullGeometryError("unexpected SEVA audit schema")
    if audit_done.get("metrics_sha256") != sha256_file(audit_root / "metrics.json"):
        raise SevaFullGeometryError("SEVA audit DONE does not bind metrics")
    if audit_done.get("hashes_sha256") != sha256_file(audit_root / "hashes.sha256"):
        raise SevaFullGeometryError("SEVA audit DONE does not bind hashes")
    _verify_hash_index(audit_root)

    review = _load_json(review_path, "SEVA human review")
    if review.get("schema_version") != REVIEW_SCHEMA or review.get("decision") != ACCEPTED:
        raise SevaFullGeometryError("SEVA orbit lacks accepted appearance review")
    if review.get("evidence", {}).get("audit_metrics_sha256") != sha256_file(
        audit_root / "metrics.json"
    ):
        raise SevaFullGeometryError("SEVA review does not bind this audit")
    if review.get("accepted_role") != "generated_low_confidence_appearance_pseudoviews":
        raise SevaFullGeometryError("unexpected accepted SEVA role")

    transforms_path = seva_root / "inference/transforms.json"
    transforms = _load_json(transforms_path, "SEVA transforms")
    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) != 53:
        raise SevaFullGeometryError("SEVA transforms require 4 inputs plus 49 targets")
    assert_not_quarantined(
        [("SEVA_generation", seva), ("SEVA_audit", audit), ("SEVA_review", review)]
    )
    return seva, audit, review, frames


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None or value.shape != (576, 576, 3):
        raise SevaFullGeometryError(f"expected a 576p RGB frame: {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != (576, 576):
        raise SevaFullGeometryError(f"expected a 576p mask: {path}")
    return value


def _surface_colors(
    points: np.ndarray,
    cameras: Sequence[dict[str, Any]],
    images: Sequence[np.ndarray],
) -> np.ndarray:
    centers = np.asarray(
        [np.linalg.inv(camera["world_to_camera_opencv_4x4"])[:3, 3] for camera in cameras]
    )
    point_xy = points[:, :2]
    point_directions = point_xy / np.maximum(
        np.linalg.norm(point_xy, axis=1, keepdims=True), 1.0e-12
    )
    camera_xy = centers[:, :2]
    camera_directions = camera_xy / np.linalg.norm(camera_xy, axis=1, keepdims=True)
    selected = np.argmax(point_directions @ camera_directions.T, axis=1)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for camera_index, (camera, image) in enumerate(zip(cameras, images, strict=True)):
        indices = np.flatnonzero(selected == camera_index)
        if not len(indices):
            continue
        u, v, depth = _project(points[indices], camera)
        px = np.clip(np.rint(u).astype(np.int64), 0, image.shape[1] - 1)
        py = np.clip(np.rint(v).astype(np.int64), 0, image.shape[0] - 1)
        if np.any(depth <= 0.0):
            raise SevaFullGeometryError("surface-color projection fell behind a camera")
        colors[indices] = image[py, px]
    return colors


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    base_root = args.base_dataset.resolve()
    seva_root = args.seva_run.resolve()
    audit_root = args.audit.resolve()
    review_path = args.review.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.real_repeat < 1 or args.max_points < 1000 or args.mask_dilation_px < 0:
        raise SevaFullGeometryError("invalid repeat, point, or dilation argument")

    base, observed_points = _load_base_dataset(base_root)
    seva, audit, review, camera_frames = _load_sources(
        seva_root, audit_root, review_path
    )
    radius_m = float(base["camera_contract"]["metric_radius_m"])
    if not 0.05 <= radius_m <= 2.0:
        raise SevaFullGeometryError("base metric camera radius is implausible")

    generated_images: list[np.ndarray] = []
    generated_masks: list[np.ndarray] = []
    generated_cameras: list[dict[str, Any]] = []
    kernel = None
    if args.mask_dilation_px:
        size = 2 * args.mask_dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for index, source_record in enumerate(seva["frames"]):
        image_path = seva_root / source_record["relpath"]
        if sha256_file(image_path) != source_record["sha256"]:
            raise SevaFullGeometryError(f"SEVA frame hash mismatch: {index}")
        image = _read_rgb(image_path)
        mask_path = audit_root / "generated_masks" / f"{index:05d}.png"
        mask = _read_mask(mask_path)
        if kernel is not None:
            mask = cv2.dilate(mask, kernel, iterations=1)
        camera_frame = camera_frames[4 + index]
        intrinsic = np.asarray(
            [
                [camera_frame["fl_x"], 0.0, camera_frame["cx"]],
                [0.0, camera_frame["fl_y"], camera_frame["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w2c = opengl_c2w_to_metric_opencv_w2c(
            np.asarray(camera_frame["transform_matrix"], dtype=np.float64), radius_m
        )
        generated_images.append(image)
        generated_masks.append(mask)
        generated_cameras.append(
            {
                "image_size_wh": list(IMAGE_SIZE_WH),
                "intrinsic_3x3": intrinsic.tolist(),
                "world_to_camera_opencv_4x4": w2c.tolist(),
            }
        )

    old_half = np.max(np.abs(observed_points), axis=0)
    half_extents = old_half * args.bounds_padding
    surface, hull = support_visual_hull_surface(
        generated_cameras,
        generated_masks,
        half_extents_m=half_extents,
        resolution=args.grid_resolution,
        minimum_support_fraction=args.minimum_support_fraction,
        chunk_size=args.chunk_size,
    )
    if any(hull["boundary_hits"].values()):
        raise SevaFullGeometryError(
            f"generated visual hull touches padded bounds: {hull['boundary_hits']}"
        )
    if len(surface) < 1000:
        raise SevaFullGeometryError("generated visual hull surface is too sparse")
    selected_count = min(len(surface), args.max_points)
    generator = np.random.default_rng(args.sample_seed)
    selected = np.sort(generator.choice(len(surface), selected_count, replace=False))
    points = surface[selected]
    colors = _surface_colors(points, generated_cameras, generated_images)

    base_cameras, base_images = parse_colmap_text(base_root)
    real_sources = {}
    for label in VIEW_ORDER:
        name = f"real_{label}_w00.png"
        if name not in base_images:
            raise SevaFullGeometryError(f"base dataset lacks real anchor: {name}")
        image_record = base_images[name]
        real_sources[label] = {
            "image": _read_rgb(base_root / "images" / name),
            "mask": _read_mask(base_root / "masks" / name),
            "camera": {
                **base_cameras[image_record["camera_id"]],
                "world_to_camera_opencv_4x4": image_record[
                    "world_to_camera_opencv_4x4"
                ],
            },
            "source_name": name,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        image_dir = staging / "images"
        mask_dir = staging / "masks"
        sparse_dir = staging / "sparse/0"
        for directory in (image_dir, mask_dir, sparse_dir):
            directory.mkdir(parents=True)

        entries: list[dict[str, Any]] = []
        for index, (image, mask, camera) in enumerate(
            zip(generated_images, generated_masks, generated_cameras, strict=True)
        ):
            entries.append(
                {
                    "name": f"gen_{index:05d}.png",
                    "rgb": image,
                    "mask": mask,
                    "camera": camera,
                    "provenance": "generated_SEVA_full_orbit_geometry_hypothesis",
                    "source_index": index,
                    "sampling_duplicate": False,
                }
            )
        entries.append(
            {
                **entries[0],
                "name": "000_eval_probe_generated.png",
                "provenance": "generated_eval_probe_duplicate_not_training_evidence",
                "sampling_duplicate": True,
            }
        )
        for label in VIEW_ORDER:
            source = real_sources[label]
            for repeat in range(args.real_repeat):
                entries.append(
                    {
                        "name": f"real_{label}_w{repeat:02d}.png",
                        "rgb": source["image"],
                        "mask": source["mask"],
                        "camera": source["camera"],
                        "provenance": "observed_real_anchor_sampling_duplicate",
                        "source_index": None,
                        "sampling_duplicate": repeat > 0,
                    }
                )
        entries.sort(key=lambda item: item["name"])

        camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
        image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
        exported = []
        for image_id, entry in enumerate(entries, 1):
            name = entry["name"]
            if not cv2.imwrite(
                str(image_dir / name), cv2.cvtColor(entry["rgb"], cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write image: {name}")
            if not cv2.imwrite(str(mask_dir / name), entry["mask"]):
                raise RuntimeError(f"failed to write mask: {name}")
            camera = entry["camera"]
            intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
            w2c = np.asarray(
                camera["world_to_camera_opencv_4x4"], dtype=np.float64
            )
            camera_lines.append(_colmap_camera_line_square(image_id, intrinsic))
            image_lines.append(_colmap_image_line(image_id, image_id, name, w2c))
            exported.append(
                {
                    key: entry[key]
                    for key in (
                        "name",
                        "provenance",
                        "source_index",
                        "sampling_duplicate",
                    )
                }
            )
        (sparse_dir / "cameras.txt").write_text("".join(camera_lines), encoding="utf-8")
        (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
        for point_id, (point, color) in enumerate(zip(points, colors, strict=True), 1):
            point_lines.append(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")

        real_instances = 4 * args.real_repeat
        generated_instances = 49
        training_total = real_instances + generated_instances
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "status": "experimental_generated_geometry_candidate_pending_human_audit",
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "eligible_for_collision_geometry": False,
            "asset_name": base["asset_name"],
            "lineage": {
                "base_SEVA_dataset_manifest_sha256": sha256_file(
                    base_root / "dataset_manifest.json"
                ),
                "seva_manifest_sha256": sha256_file(seva_root / "manifest.json"),
                "seva_audit_metrics_sha256": sha256_file(audit_root / "metrics.json"),
                "seva_appearance_review_sha256": sha256_file(review_path),
                "inherited_mesh_or_procedural_surface": None,
                "metric_scale_source": "four_real_anchor_visual_hull_extent_and_camera_radius_only",
            },
            "source_review_boundary": {
                "accepted_role": review["accepted_role"],
                "geometry_role_was_not_accepted_by_source_review": True,
                "this_geometry_candidate_requires_new_human_visual_audit": True,
            },
            "sampling": {
                "generated_unique_views": generated_instances,
                "real_unique_views": 4,
                "real_repetitions_per_view": args.real_repeat,
                "real_training_instances": real_instances,
                "training_instances_total": training_total,
                "nominal_real_sampling_probability": real_instances / training_total,
                "generated_eval_probe_duplicates": 1,
            },
            "camera_contract": {
                "orientation_and_intrinsics": "exact_SEVA_target_camera_contract",
                "metric_radius_m": radius_m,
                "photogrammetrically_calibrated": False,
            },
            "initial_points": {
                "count": int(len(points)),
                "source": "49_view_generated_SEVA_support_visual_hull",
                "generated_geometry_prior": True,
                "observed_visual_hull_prior": False,
                "metric_truth": False,
                "collision_eligible": False,
                "bounds_gauge_source": "observed_four_view_visual_hull_AABB_with_padding",
                "old_surface_points_inherited": False,
                "visual_hull": hull,
            },
            "images": exported,
            "required_training_profile": {
                "freeze_geometry": False,
                "disable_refinement": False,
                "allow_means_scales_quaternions_optimization": True,
                "output_role": "nonformal_visual_asset_candidate_pending_human_audit",
            },
            "provenance_boundary": [
                "All 49 SEVA frames and masks are generated, not observed evidence.",
                "The generated visual hull is a visual geometry hypothesis, never metric truth.",
                "No rejected OBJ, mesh, or procedural surface initializes the point cloud.",
                "The old observed hull contributes only camera radius and padded search bounds.",
                "The result cannot be used as collision geometry or held-out evidence.",
                "A human full-orbit geometry audit is mandatory before live visual use.",
            ],
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA,
                    "status": "done_nonformal_generated_geometry_dataset_pending_audit",
                    "manifest_sha256": sha256_file(staging / "dataset_manifest.json"),
                    "hashes_sha256": hashes_sha,
                    "completed_utc": utc_now(),
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
                    "schema_version": "radeon_oneloop.seva_full_geometry_failure.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_utc": utc_now(),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--seva-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-repeat", type=int, default=12)
    parser.add_argument("--grid-resolution", type=int, default=128)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.90)
    parser.add_argument("--mask-dilation-px", type=int, default=2)
    parser.add_argument("--bounds-padding", type=float, default=1.20)
    parser.add_argument("--max-points", type=int, default=60_000)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser


def main() -> None:
    result = build_dataset(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "sampling": result["sampling"],
                "initial_points": result["initial_points"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
