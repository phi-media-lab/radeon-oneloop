#!/usr/bin/env python3
"""Project four real views onto an aligned learned mesh and render an orbit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from gaussian.align_generated_mesh_four_views import (
    VIEW_ORDER,
    _fit_projection,
    _rasterize_silhouette,
    _verify_hunyuan_run,
    silhouette_iou,
)
from gaussian.prepare_four_view_generation import sha256_file, validate_generation_input


SCHEMA_VERSION = "radeon_oneloop.four_view_learned_mesh_texture_orbit.v2"
DONE_SCHEMA_VERSION = "radeon_oneloop.four_view_learned_mesh_texture_orbit_done.v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _verify_alignment(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("alignment root requires manifest and DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.four_view_learned_mesh_alignment.v1":
        raise ValueError("unexpected learned-mesh alignment schema")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("alignment DONE does not bind manifest")
    item = manifest["outputs"]["aligned_metric_ply"]
    path = root / item["relpath"]
    if sha256_file(path) != item["sha256"]:
        raise ValueError("aligned metric mesh hash mismatch")
    return manifest, path


def _view_basis_from_azimuth(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = math.radians(azimuth_deg)
    position = np.array([math.sin(azimuth), math.cos(azimuth), 0.0], dtype=np.float64)
    forward = -position
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def vista4d_unique_azimuths(frames: int = 49) -> np.ndarray:
    """Return the exact non-duplicated camera schedule consumed by Vista4D.

    Frame 49 does not exist, so a 49-frame sequence must sample ``[0, 360)``.
    Duplicating 0 degrees at frame 48 would bind that image to Vista4D's
    352.653-degree target camera and silently corrupt the camera contract.
    """

    if frames < 2:
        raise ValueError("a Vista4D orbit requires at least two frames")
    return np.arange(frames, dtype=np.float64) * (360.0 / frames)


def nearest_orbit_index(azimuths: np.ndarray, target_deg: float) -> int:
    values = np.asarray(azimuths, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("azimuths must be a non-empty finite vector")
    delta = np.abs((values - target_deg + 180.0) % 360.0 - 180.0)
    return int(np.argmin(delta))


def canonical_orbit_extrinsic(
    azimuth_deg: float, *, distance_m: float
) -> np.ndarray:
    """Return canonical-object to OpenCV-camera for a level orbit."""

    if not math.isfinite(azimuth_deg) or not math.isfinite(distance_m):
        raise ValueError("orbit parameters must be finite")
    if distance_m <= 0.0:
        raise ValueError("orbit distance must be positive")
    angle = math.radians(azimuth_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (
            (-cosine, -sine, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (sine, -cosine, 0.0, distance_m),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def vista4d_camera_track(
    *, frames: int, intrinsic_3x3: np.ndarray, distance_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build Vista4D external c2w matrices for the exact source orbit."""

    intrinsic = np.asarray(intrinsic_3x3, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
    matrices = []
    for azimuth in vista4d_unique_azimuths(frames):
        render_c2w = np.linalg.inv(
            canonical_orbit_extrinsic(float(azimuth), distance_m=distance_m)
        )
        matrices.append(conversion @ render_c2w)
    intrinsics = np.repeat(
        np.asarray(
            [[intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]],
            dtype=np.float64,
        ),
        frames,
        axis=0,
    )
    return np.stack(matrices), intrinsics


def project_real_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    images: dict[str, np.ndarray],
    alphas: dict[str, np.ndarray],
    *,
    visibility_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    color_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    weight_sum = np.zeros(len(vertices), dtype=np.float64)
    max_weight = np.zeros(len(vertices), dtype=np.float64)
    source_count = np.zeros(len(vertices), dtype=np.uint8)
    per_view = []
    for label in VIEW_ORDER:
        image = images[label]
        alpha = alphas[label]
        height, width = alpha.shape
        pixels, _, depth = _fit_projection(vertices, alpha >= 0.5, label)
        px = np.rint(pixels[:, 0]).astype(np.int64)
        py = np.rint(pixels[:, 1]).astype(np.int64)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        safe_x = np.clip(px, 0, width - 1)
        safe_y = np.clip(py, 0, height - 1)
        sampled_alpha = alpha[safe_y, safe_x]
        camera_direction = -_view_basis_from_azimuth(
            {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0}[label]
        )[2]
        facing = np.clip(normals @ camera_direction, 0.0, 1.0)
        flat = safe_y * width + safe_x
        closest = np.full(width * height, -np.inf, dtype=np.float64)
        np.maximum.at(closest, flat[inside], -depth[inside])
        visible = -depth >= closest[flat] - visibility_tolerance_m
        weight = sampled_alpha * facing**2
        valid = inside & visible & (sampled_alpha >= 0.25) & (facing >= 0.08)
        weight = np.where(valid, weight, 0.0)
        sampled_rgb = image[safe_y, safe_x]
        color_sum += sampled_rgb * weight[:, None]
        weight_sum += weight
        max_weight = np.maximum(max_weight, weight)
        source_count += (weight >= 0.05).astype(np.uint8)
        per_view.append(
            {
                "view": label,
                "contributing_vertices": int(np.count_nonzero(weight >= 0.05)),
                "mean_positive_weight": (
                    float(np.mean(weight[weight > 0])) if np.any(weight > 0) else 0.0
                ),
            }
        )
    observed = weight_sum > 1e-8
    colors = np.full((len(vertices), 3), 0.92, dtype=np.float64)
    colors[observed] = color_sum[observed] / weight_sum[observed, None]
    colors_u8 = np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    confidence = np.clip(max_weight, 0.0, 1.0).astype(np.float32)
    return colors_u8, confidence, source_count, {
        "method": "normal_weighted_four_real_view_projection_with_vertex_zbuffer",
        "visibility_tolerance_m": visibility_tolerance_m,
        "observed_vertex_fraction": float(np.mean(observed)),
        "neutral_unobserved_vertex_fraction": float(np.mean(~observed)),
        "neutral_unobserved_rgb": [0.92, 0.92, 0.92],
        "source_count_histogram": {
            str(value): int(np.count_nonzero(source_count == value)) for value in range(5)
        },
        "views": per_view,
    }


def _orbit_pixels(
    vertices: np.ndarray, azimuth_deg: float, size_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size_wh
    right, up, forward = _view_basis_from_azimuth(azimuth_deg)
    u = vertices @ right
    v = -(vertices @ up)
    span_u = max(float(np.ptp(u)), 1e-9)
    span_v = max(float(np.ptp(v)), 1e-9)
    scale = min(0.84 * width / span_u, 0.84 * height / span_v)
    pixels = np.stack(
        (
            (u - 0.5 * (u.min() + u.max())) * scale + width / 2.0,
            (v - 0.5 * (v.min() + v.max())) * scale + height / 2.0,
        ),
        axis=1,
    )
    return pixels, vertices @ forward


def _render(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    azimuth_deg: float,
    size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    width, height = size_wh
    pixels, depth = _orbit_pixels(vertices, azimuth_deg, size_wh)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    order = np.argsort(np.mean(depth[faces], axis=1))[::-1]
    face_colors = np.rint(np.mean(colors[faces], axis=1)).astype(np.uint8)
    for face_index in order:
        polygon = np.rint(pixels[faces[face_index]]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].min() >= height
        ):
            continue
        color = tuple(int(value) for value in face_colors[face_index])
        cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(alpha, polygon, 255, lineType=cv2.LINE_8)
    return image, alpha


def _render_perspective(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    camera_from_object: np.ndarray,
    intrinsic: np.ndarray,
    size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Render a fixed-intrinsic metric view for camera-bound generation."""

    import cv2

    width, height = size_wh
    transform = np.asarray(camera_from_object, dtype=np.float64)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if transform.shape != (4, 4) or matrix.shape != (3, 3):
        raise ValueError("perspective renderer requires 4x4 extrinsic and 3x3 intrinsic")
    local = vertices @ transform[:3, :3].T + transform[:3, 3]
    depth = local[:, 2]
    safe = np.maximum(depth, 1.0e-9)
    pixels = np.stack(
        (
            matrix[0, 0] * local[:, 0] / safe + matrix[0, 2],
            matrix[1, 1] * local[:, 1] / safe + matrix[1, 2],
        ),
        axis=1,
    )
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    valid_faces = faces[np.all(depth[faces] > 1.0e-6, axis=1)]
    order = np.argsort(np.mean(depth[valid_faces], axis=1))[::-1]
    face_colors = np.rint(np.mean(colors[valid_faces], axis=1)).astype(np.uint8)
    for face, face_color in zip(valid_faces[order], face_colors[order], strict=True):
        polygon = np.rint(pixels[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(
            image,
            polygon,
            tuple(int(value) for value in face_color),
            lineType=cv2.LINE_AA,
        )
        cv2.fillConvexPoly(alpha, polygon, 255, lineType=cv2.LINE_8)
    return image, alpha


def texture_and_render(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
        import imageio.v3 as iio
        import trimesh
    except ImportError as exc:  # pragma: no cover - remote real2sim environment
        raise RuntimeError("OpenCV, imageio, and trimesh are required") from exc
    input_root = args.input_root.resolve()
    alignment_root = args.alignment_root.resolve()
    output = args.output.resolve()
    if not math.isfinite(args.distance_m) or args.distance_m <= 0.0:
        raise ValueError("distance-m must be finite and positive")
    if not math.isfinite(args.horizontal_fov_deg) or not 20.0 <= args.horizontal_fov_deg <= 90.0:
        raise ValueError("horizontal-fov-deg must be finite and in [20, 90]")
    input_manifest = validate_generation_input(input_root)
    alignment_manifest, mesh_path = _verify_alignment(alignment_root)
    if alignment_manifest["input"]["four_view_manifest_sha256"] != sha256_file(
        input_root / "manifest.json"
    ):
        raise ValueError("alignment does not derive from the selected four-view contract")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite learned-mesh texture orbit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        mesh_dir = staging / "mesh"
        orbit_dir = staging / "orbit"
        frame_dir = orbit_dir / "frames"
        alpha_dir = orbit_dir / "alpha"
        audit_dir = staging / "audit"
        for directory in (mesh_dir, frame_dir, alpha_dir, audit_dir):
            directory.mkdir(parents=True)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        images: dict[str, np.ndarray] = {}
        alphas: dict[str, np.ndarray] = {}
        for label in VIEW_ORDER:
            rgb = cv2.imread(str(input_root / "anchors" / f"{label}.png"), cv2.IMREAD_COLOR)
            rgba = cv2.imread(
                str(input_root / "hunyuan3d_2mv" / f"{label}.png"), cv2.IMREAD_UNCHANGED
            )
            if rgb is None or rgba is None or rgba.shape[2] != 4:
                raise ValueError(f"cannot load real texture source: {label}")
            images[label] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            alphas[label] = rgba[..., 3].astype(np.float64) / 255.0
        colors, confidence, source_count, texture_report = project_real_colors(
            vertices,
            faces,
            normals,
            images,
            alphas,
            visibility_tolerance_m=args.visibility_tolerance_m,
        )
        textured_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        textured_mesh.visual.vertex_colors = np.concatenate(
            [colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1
        )
        ply_path = mesh_dir / "real_projected_learned_mesh.ply"
        glb_path = mesh_dir / "real_projected_learned_mesh.glb"
        textured_mesh.export(ply_path)
        textured_mesh.export(glb_path)
        np.save(mesh_dir / "confidence.npy", confidence, allow_pickle=False)
        np.save(mesh_dir / "source_count.npy", source_count, allow_pickle=False)

        focal_px = args.width / (
            2.0 * math.tan(math.radians(args.horizontal_fov_deg) / 2.0)
        )
        orbit_intrinsic = np.asarray(
            [
                [focal_px, 0.0, args.width / 2.0],
                [0.0, focal_px, args.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        frames = []
        masks = []
        render_ms = []
        azimuths = vista4d_unique_azimuths(49)
        for index, azimuth in enumerate(azimuths):
            started = time.perf_counter()
            frame, alpha = _render_perspective(
                vertices,
                faces,
                colors,
                camera_from_object=canonical_orbit_extrinsic(
                    float(azimuth), distance_m=args.distance_m
                ),
                intrinsic=orbit_intrinsic,
                size_wh=(args.width, args.height),
            )
            render_ms.append((time.perf_counter() - started) * 1000.0)
            iio.imwrite(frame_dir / f"{index:05d}.png", frame)
            iio.imwrite(alpha_dir / f"{index:05d}.png", alpha)
            frames.append(frame)
            masks.append(alpha >= 128)
        iio.imwrite(
            orbit_dir / "source.mp4",
            np.stack(frames),
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        target_c2w, target_intrinsics = vista4d_camera_track(
            frames=len(azimuths),
            intrinsic_3x3=orbit_intrinsic,
            distance_m=args.distance_m,
        )
        target_camera_path = orbit_dir / "target_cameras.npz"
        np.savez_compressed(
            target_camera_path,
            cam_c2w=target_c2w.astype(np.float64),
            intrinsics=target_intrinsics.astype(np.float64),
            azimuth_deg=azimuths.astype(np.float64),
        )
        contact_indices = np.linspace(0, 48, 12, dtype=int)
        contact_rows = [
            np.concatenate([frames[index] for index in contact_indices[start : start + 4]], axis=1)
            for start in range(0, 12, 4)
        ]
        contact_sheet = np.concatenate(contact_rows, axis=0)
        iio.imwrite(audit_dir / "orbit_contact_sheet.png", contact_sheet)

        anchor_metrics = []
        anchor_angles = {"front": 0.0, "right": 270.0, "back": 180.0, "left": 90.0}
        anchor_indices = {
            label: nearest_orbit_index(azimuths, angle)
            for label, angle in anchor_angles.items()
        }
        for label in VIEW_ORDER:
            index = anchor_indices[label]
            target = alphas[label] >= 0.5
            _, anchor_alpha = _render(
                vertices,
                faces,
                colors,
                float(azimuths[index]),
                (target.shape[1], target.shape[0]),
            )
            rendered = anchor_alpha >= 128
            anchor_metrics.append(
                {
                    "view": label,
                    "orbit_frame_index": index,
                    "silhouette_iou": silhouette_iou(rendered, target),
                }
            )
        last_to_first_mae = float(
            np.mean(np.abs(frames[0].astype(np.float32) - frames[-1].astype(np.float32)))
            / 255.0
        )
        last_to_first_mask_iou = silhouette_iou(masks[0], masks[-1])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": input_manifest["asset_name"],
            "input": {
                "four_view_manifest_sha256": sha256_file(input_root / "manifest.json"),
                "alignment_manifest_sha256": sha256_file(alignment_root / "manifest.json"),
                "aligned_learned_mesh_sha256": sha256_file(mesh_path),
                "inherited_procedural_geometry": None,
            },
            "texture_projection": texture_report,
            "mesh": {
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
                "ply_relpath": "mesh/real_projected_learned_mesh.ply",
                "ply_sha256": sha256_file(ply_path),
                "glb_relpath": "mesh/real_projected_learned_mesh.glb",
                "glb_sha256": sha256_file(glb_path),
                "confidence_sha256": sha256_file(mesh_dir / "confidence.npy"),
                "source_count_sha256": sha256_file(mesh_dir / "source_count.npy"),
            },
            "orbit": {
                "frames": 49,
                "image_size_wh": [args.width, args.height],
                "fps": args.fps,
                "azimuth_start_deg": 0.0,
                "azimuth_step_deg": float(360.0 / len(azimuths)),
                "azimuth_last_deg": float(azimuths[-1]),
                "azimuths_deg": azimuths.tolist(),
                "endpoint_duplicate": False,
                "camera_schedule": "vista4d_unique_49_frame_level_orbit",
                "render_camera_model": "PINHOLE_OPENCV_fixed_intrinsic",
                "distance_m": args.distance_m,
                "horizontal_fov_deg": args.horizontal_fov_deg,
                "intrinsic_3x3": orbit_intrinsic.tolist(),
                "target_cameras_relpath": "orbit/target_cameras.npz",
                "target_cameras_sha256": sha256_file(target_camera_path),
                "source_video_relpath": "orbit/source.mp4",
                "source_video_sha256": sha256_file(orbit_dir / "source.mp4"),
                "last_to_first_nominal_gap_deg": float(360.0 / len(azimuths)),
                "last_to_first_rgb_mae": last_to_first_mae,
                "last_to_first_mask_iou": last_to_first_mask_iou,
                "render_ms_mean": float(np.mean(render_ms)),
                "render_ms_p95": float(np.percentile(render_ms, 95)),
            },
            "four_real_view_silhouette_audit": {
                "comparison_model": "independent_orthographic_autofit_per_uncalibrated_product_photo",
                "camera_bound": False,
                "purpose": "alignment_consistency_only_not_source_camera_validation",
                "views": anchor_metrics,
            },
            "allowed_role": "continuous_real_projected_learned_mesh_source_pending_visual_review",
            "rejected_roles": [
                "observed_video",
                "final_texture",
                "heldout_real_evidence",
                "physics_collision_geometry",
            ],
            "review_status": "pending_visual_identity_review_before_vista4d",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
                lines.append(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}")
        hashes_path = staging / "hashes.sha256"
        hashes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        done = {
            "schema_version": DONE_SCHEMA_VERSION,
            "stage": "four_real_view_texture_projection_and_orbit_render",
            "status": "done_pending_visual_review_before_vista4d",
            "completed_utc": utc_now(),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "hashes_sha256": sha256_file(hashes_path),
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        failure_path = output.with_name(f"{output.name}.FAILED.json")
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.four_view_learned_mesh_texture_orbit_failure.v1",
                        "stage": "four_real_view_texture_projection_and_orbit_render",
                        "failed_utc": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility-tolerance-m", type=float, default=0.002)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--distance-m", type=float, default=0.24)
    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = texture_and_render(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
