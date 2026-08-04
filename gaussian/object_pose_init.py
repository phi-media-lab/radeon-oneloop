#!/usr/bin/env python3
"""Create an auditable metric camera baseline for a sparse object asset.

The manual-ring result is deliberately a deterministic initialization, not a
photogrammetric claim.  It fixes the distance/focal gauge, uses only reviewed
observed masks, and preserves the nominal view labels from the M1 manifest.
Learned pose candidates may replace it only after the P2 quality gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = "radeon_oneloop.object_pose_init.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.object_asset_stage_done.v1"


class PoseInitError(ValueError):
    """Raised when the M1 input cannot support a safe pose initialization."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise PoseInitError("cannot normalize a zero vector")
    return value / norm


def canonical_orbit_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return the camera-center direction for the documented +Y-front frame."""
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    cos_elevation = math.cos(elevation)
    return np.asarray(
        [
            math.sin(azimuth) * cos_elevation,
            math.cos(azimuth) * cos_elevation,
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def look_at_world_to_camera(
    camera_center: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Build a proper OpenCV world-to-camera matrix (+x right, +y down, +z forward)."""
    center = np.asarray(camera_center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0] if world_up is None else world_up, dtype=np.float64)
    forward = unit_vector(target - center)
    right = unit_vector(np.cross(forward, up))
    down = unit_vector(np.cross(forward, right))
    rotation = np.stack([right, down, forward], axis=0)
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8):
        raise PoseInitError("look-at rotation is not right-handed")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ center
    return transform


def camera_center_from_world_to_camera(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    return -rotation.T @ translation


def fit_proper_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Fit target ~= scale * source @ R.T + translation without reflection."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise PoseInitError("similarity fit requires matching N x 3 arrays with N >= 3")
    source_mean = src.mean(axis=0)
    target_mean = dst.mean(axis=0)
    source_centered = src - source_mean
    target_centered = dst - target_mean
    variance = float(np.sum(source_centered**2) / src.shape[0])
    if variance <= 1.0e-12:
        raise PoseInitError("source camera centers are degenerate")
    covariance = target_centered.T @ source_centered / src.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    if np.linalg.det(rotation) <= 0.0:
        raise PoseInitError("similarity fit produced a reflected rotation")
    scale = float(np.sum(singular_values * sign) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise PoseInitError("similarity scale is not positive and finite")
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ src.T)).T + translation
    residuals = np.linalg.norm(aligned - dst, axis=1)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "aligned": aligned,
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_residual": float(np.max(residuals)),
        "determinant": float(np.linalg.det(rotation)),
    }


def validate_labeled_camera_layout(
    camera_centers: np.ndarray,
    nominal_directions: np.ndarray,
    *,
    camera_up_vectors: np.ndarray | None = None,
    max_angular_error_deg: float,
    max_radius_cv: float,
    min_mean_up_dot: float = 0.5,
) -> dict[str, Any]:
    """Gate a learned layout after a proper-Sim(3) label alignment."""
    centers = np.asarray(camera_centers, dtype=np.float64)
    nominal = np.asarray(nominal_directions, dtype=np.float64)
    if centers.shape != nominal.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise PoseInitError("camera centers and nominal directions must be matching N x 3 arrays")
    centered = centers - centers.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    if float(np.min(radii)) <= 1.0e-9:
        raise PoseInitError("a learned camera center collapsed onto the orbit center")
    unit_centers = centered / radii[:, None]
    unit_nominal = nominal / np.linalg.norm(nominal, axis=1)[:, None]
    fit = fit_proper_similarity(unit_centers, unit_nominal)
    aligned = fit["aligned"] - fit["aligned"].mean(axis=0)
    aligned /= np.linalg.norm(aligned, axis=1)[:, None]
    cosines = np.clip(np.sum(aligned * unit_nominal, axis=1), -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(cosines))
    radius_cv = float(np.std(radii) / np.mean(radii))
    up_status: dict[str, Any]
    if camera_up_vectors is None:
        # Four horizontal orbit centers are rank-2, so center correspondences
        # alone cannot distinguish a reflection from a 180-degree 3D rotation.
        # Learned camera orientation is mandatory for the handedness gate.
        up_status = {
            "available": False,
            "mean_canonical_up_dot": None,
            "passed": False,
            "reason": "camera up vectors are required to resolve planar reflection ambiguity",
        }
    else:
        up = np.asarray(camera_up_vectors, dtype=np.float64)
        if up.shape != centers.shape:
            raise PoseInitError("camera up vectors must match the N x 3 camera-center array")
        norms = np.linalg.norm(up, axis=1)
        if float(np.min(norms)) <= 1.0e-9:
            raise PoseInitError("camera up vectors contain a zero vector")
        up = up / norms[:, None]
        aligned_up = (fit["rotation"] @ up.T).T
        mean_up_dot = float(np.mean(aligned_up @ np.asarray([0.0, 0.0, 1.0])))
        up_status = {
            "available": True,
            "mean_canonical_up_dot": mean_up_dot,
            "passed": bool(mean_up_dot >= min_mean_up_dot),
        }
    passed = bool(
        float(np.max(angular_errors)) <= max_angular_error_deg
        and radius_cv <= max_radius_cv
        and up_status["passed"]
    )
    return {
        "passed": passed,
        "proper_rotation_determinant": fit["determinant"],
        "angular_error_deg": angular_errors.tolist(),
        "angular_error_max_deg": float(np.max(angular_errors)),
        "radius_coefficient_of_variation": radius_cv,
        "up_direction_gate": up_status,
        "thresholds": {
            "max_angular_error_deg": float(max_angular_error_deg),
            "max_radius_coefficient_of_variation": float(max_radius_cv),
            "min_mean_up_dot": float(min_mean_up_dot),
        },
    }


def deterministic_confident_sample(
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample valid image pixels with a stable seed.

    Confidence is used to define ``valid_mask`` before this function.  Sampling
    only the global top scores creates severe spatial bias toward high-contrast
    ears and silhouettes, which is undesirable for a coarse object point cloud.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    scores = np.asarray(confidence)
    if valid.shape != scores.shape:
        raise PoseInitError("valid mask and confidence image must have the same shape")
    if limit <= 0:
        raise PoseInitError("sample limit must be positive")
    indices = np.flatnonzero(valid.reshape(-1))
    if len(indices) <= limit:
        return indices
    generator = np.random.default_rng(int(seed))
    selected = generator.choice(indices, size=limit, replace=False)
    return np.sort(selected)


def _load_reviewed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "radeon_oneloop.object_asset_manifest.v1":
        raise PoseInitError("input is not an M1 object-asset manifest")
    if value.get("formal") is not False:
        raise PoseInitError("object pose initialization must remain nonformal")
    if value.get("summary", {}).get("mask_review_status") != "reviewed_pass":
        raise PoseInitError("M1 masks must have reviewed_pass status")
    if value.get("coordinate_convention") != {
        "front_axis": "+Y",
        "up_axis": "+Z",
        "viewer_left_axis": "+X",
        "unit": "m",
        "origin": "plush_body_center",
    }:
        raise PoseInitError("unsupported canonical coordinate convention")
    return value


def _prepared_pose_views(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    selected = []
    for view in manifest["views"]:
        if "pose" not in view["roles"]:
            continue
        if view["provenance"] != "observed" or view["tier"] != "A" or not view["prepared"]:
            raise PoseInitError(f"pose view {view['id']} is not a prepared observed tier-A anchor")
        if view["mask_status"] != "reviewed_pass":
            raise PoseInitError(f"pose view {view['id']} has not passed mask review")
        orbit = view.get("nominal_camera_orbit_deg")
        if orbit is None:
            raise PoseInitError(f"pose view {view['id']} has no nominal orbit label")
        for key in ("image", "hard_mask"):
            file_record = view[key]
            file_path = (root / file_record["relpath"]).resolve()
            try:
                file_path.relative_to(root.resolve())
            except ValueError as exc:
                raise PoseInitError(f"{view['id']} {key} escaped the M1 root") from exc
            if not file_path.is_file() or sha256_file(file_path) != file_record["sha256"]:
                raise PoseInitError(f"{view['id']} {key} is missing or has a hash mismatch")
        selected.append(view)
    if len(selected) != 4:
        raise PoseInitError(f"manual ring requires exactly four pose anchors, got {len(selected)}")
    return selected


def _matrix_list(value: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in value.tolist()]


def build_manual_ring(
    manifest: dict[str, Any],
    m1_root: Path,
    *,
    radius_m: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise PoseInitError("ring radius must be positive and finite")
    height_m = float(manifest["metric_anchor"]["value_m"])
    views = _prepared_pose_views(manifest, m1_root)
    cameras = []
    for view in views:
        orbit = view["nominal_camera_orbit_deg"]
        direction = canonical_orbit_direction(orbit["azimuth"], orbit["elevation"])
        center = radius_m * direction
        world_to_camera = look_at_world_to_camera(center, np.zeros(3, dtype=np.float64))
        camera_to_world = np.linalg.inv(world_to_camera)
        width = int(view["normalization"]["output_width"])
        height = int(view["normalization"]["output_height"])
        x0, y0, x1, y1 = (int(item) for item in view["mask_qa"]["foreground_bbox_xyxy"])
        foreground_height_px = y1 - y0
        if foreground_height_px <= 0:
            raise PoseInitError(f"{view['id']} has an empty foreground bounding box")
        focal_px = float(foreground_height_px * radius_m / height_m)
        intrinsic = np.asarray(
            [
                [focal_px, 0.0, (width - 1.0) / 2.0],
                [0.0, focal_px, (height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cameras.append(
            {
                "view_id": view["id"],
                "view_label": view["view_label"],
                "source_sha256": view["source_sha256"],
                "image_sha256": view["image"]["sha256"],
                "hard_mask_sha256": view["hard_mask"]["sha256"],
                "image_size_wh": [width, height],
                "nominal_orbit_deg": {
                    "azimuth": float(orbit["azimuth"]),
                    "elevation": float(orbit["elevation"]),
                },
                "camera_center_m": center.tolist(),
                "world_to_camera_opencv_4x4": _matrix_list(world_to_camera),
                "camera_to_world_opencv_4x4": _matrix_list(camera_to_world),
                "intrinsic_3x3": _matrix_list(intrinsic),
                "focal_xy_px": [focal_px, focal_px],
                "principal_point_xy_px": [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
                "horizontal_fov_deg": math.degrees(2.0 * math.atan(width / (2.0 * focal_px))),
                "foreground_bbox_xyxy": [x0, y0, x1, y1],
                "foreground_height_px": foreground_height_px,
            }
        )

    camera_document = {
        "schema_version": "radeon_oneloop.object_cameras.v1",
        "formal": False,
        "asset_name": manifest["asset_name"],
        "method": "manual_ring_metric_silhouette_gauge",
        "camera_model": "PINHOLE_OPENCV",
        "coordinate_convention": manifest["coordinate_convention"],
        "metric_anchor": manifest["metric_anchor"],
        "cameras": cameras,
    }
    similarity = {
        "schema_version": "radeon_oneloop.object_similarity_transform.v1",
        "formal": False,
        "source_frame": "manual_ring_canonical_seed",
        "target_frame": "object_canonical_metric",
        "scale": 1.0,
        "rotation_3x3": _matrix_list(np.eye(3, dtype=np.float64)),
        "translation_m": [0.0, 0.0, 0.0],
        "metric_anchor": manifest["metric_anchor"],
        "status": "identity_by_construction",
    }
    quality = {
        "schema_version": "radeon_oneloop.object_pose_quality.v1",
        "formal": False,
        "method": "manual_ring_metric_silhouette_gauge",
        "view_count": len(cameras),
        "all_rotations_proper": all(
            np.isclose(
                np.linalg.det(np.asarray(camera["world_to_camera_opencv_4x4"])[:3, :3]),
                1.0,
                atol=1.0e-8,
            )
            for camera in cameras
        ),
        "metric_height_m": height_m,
        "ring_radius_m": float(radius_m),
        "gauge": "ring distance fixed by configuration; focal inferred independently from each reviewed silhouette height",
        "identity_orientation": {
            "front": "+Y",
            "viewer_left": "+X",
            "status": "labels fixed by reviewed source manifest",
            "automatic_color_landmark_check": "pending",
        },
        "gate_status": "baseline_only_pending_silhouette_optimization_and_learned_comparison",
        "limitations": [
            "nominal listing view labels are not measured camera angles",
            "distance and focal length are gauge-coupled without camera metadata",
            "95 mm scales the full reviewed silhouette and does not independently locate the plush-body origin",
            "this baseline has no learned or photogrammetric depth",
        ],
    }
    return camera_document, similarity, quality


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_hashes(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "hashes.sha256").write_text("".join(lines), encoding="utf-8")


def run_manual_ring(args: argparse.Namespace) -> Path:
    manifest_path = args.m1_manifest.resolve()
    m1_root = manifest_path.parent
    manifest = _load_reviewed_manifest(manifest_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        cameras, similarity, quality = build_manual_ring(manifest, m1_root, radius_m=args.radius_m)
        _write_json(staging / "cameras_observed.json", cameras)
        _write_json(staging / "similarity_transform.json", similarity)
        _write_json(staging / "quality.json", quality)
        stage_manifest = {
            "schema_version": SCHEMA_VERSION,
            "formal": False,
            "asset_name": manifest["asset_name"],
            "method": "manual_ring_metric_silhouette_gauge",
            "host_role": args.host_role,
            "m1_manifest_sha256": sha256_file(manifest_path),
            "parameters": {"radius_m": float(args.radius_m)},
            "outputs": {
                name: {"relpath": name, "sha256": sha256_file(staging / name)}
                for name in ("cameras_observed.json", "similarity_transform.json", "quality.json")
            },
            "acceptance_status": quality["gate_status"],
        }
        _write_json(staging / "manifest.json", stage_manifest)
        _write_hashes(staging)
        _write_json(
            staging / "DONE",
            {
                "schema_version": DONE_SCHEMA_VERSION,
                "stage": "M2_manual_ring_pose_initialization",
                "manifest_sha256": sha256_file(staging / "manifest.json"),
                "status": "done_candidate_pending_p2_gate",
            },
        )
        os.replace(staging, output)
    except BaseException as exc:
        try:
            _write_json(
                staging / "FAILED",
                {
                    "schema_version": DONE_SCHEMA_VERSION,
                    "stage": "M2_manual_ring_pose_initialization",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        finally:
            failed = output.parent / f"{output.name}.FAILED"
            if not failed.exists():
                os.replace(staging, failed)
            else:
                shutil.rmtree(staging)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=0.30)
    parser.add_argument("--host-role", default="amd_nonformal_pose_initialization")
    return parser.parse_args()


def main() -> None:
    output = run_manual_ring(parse_args())
    print(output)


if __name__ == "__main__":
    main()
