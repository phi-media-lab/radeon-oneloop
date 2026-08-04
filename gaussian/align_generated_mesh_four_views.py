#!/usr/bin/env python3
"""Coarsely align a learned mesh to four reviewed real-image silhouettes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.prepare_four_view_generation import (
    sha256_file,
    validate_generation_input,
)


SCHEMA_VERSION = "radeon_oneloop.four_view_learned_mesh_alignment.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.four_view_learned_mesh_alignment_done.v1"
VIEW_ORDER = ("front", "right", "back", "left")
VIEW_CAMERA_POSITIONS = {
    "front": np.array([0.0, 1.0, 0.0]),
    "right": np.array([-1.0, 0.0, 0.0]),
    "back": np.array([0.0, -1.0, 0.0]),
    "left": np.array([1.0, 0.0, 0.0]),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def signed_permutation_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    unique = {tuple(matrix.reshape(-1).tolist()): matrix for matrix in rotations}
    if len(unique) != 24:
        raise RuntimeError("right-handed signed-permutation search must contain 24 rotations")
    return [unique[key] for key in sorted(unique)]


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("silhouette arrays must have the same shape")
    union = np.logical_or(a, b).sum()
    if union == 0:
        raise ValueError("silhouette union is empty")
    return float(np.logical_and(a, b).sum() / union)


def metric_transform(
    vertices: np.ndarray, raw_to_canonical_rotation: np.ndarray, height_m: float
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(raw_to_canonical_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("raw-to-canonical rotation must be orthonormal")
    if not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("metric height must be positive and finite")
    rotated = vertices @ rotation.T
    center = 0.5 * (rotated.min(axis=0) + rotated.max(axis=0))
    centered = rotated - center
    height_raw = float(np.ptp(centered[:, 2]))
    if height_raw <= 0:
        raise ValueError("aligned mesh has zero height")
    scale = height_m / height_raw
    transformed = centered * scale
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = -scale * center
    return transformed, matrix


def _view_basis(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = VIEW_CAMERA_POSITIONS[label]
    forward = -position / np.linalg.norm(position)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def _fit_projection(
    vertices: np.ndarray, target_mask: np.ndarray, label: str
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    right, up, forward = _view_basis(label)
    u = vertices @ right
    v = -(vertices @ up)
    ys, xs = np.nonzero(target_mask)
    if len(xs) == 0:
        raise ValueError(f"empty target mask for {label}")
    source_width = float(np.ptp(u))
    source_height = float(np.ptp(v))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("projected mesh has degenerate extent")
    target_width = float(xs.max() - xs.min() + 1)
    target_height = float(ys.max() - ys.min() + 1)
    scale = min(target_width / source_width, target_height / source_height)
    target_cx = 0.5 * (xs.min() + xs.max())
    target_cy = 0.5 * (ys.min() + ys.max())
    source_cx = 0.5 * (u.min() + u.max())
    source_cy = 0.5 * (v.min() + v.max())
    pixels = np.stack(
        (
            (u - source_cx) * scale + target_cx,
            (v - source_cy) * scale + target_cy,
        ),
        axis=1,
    )
    return (
        pixels,
        {
            "pixels_per_raw_unit": scale,
            "target_center_x_px": target_cx,
            "target_center_y_px": target_cy,
            "source_center_u_raw": source_cx,
            "source_center_v_raw": source_cy,
        },
        vertices @ forward,
    )


def _rasterize_silhouette(
    pixels: np.ndarray, faces: np.ndarray, size_wh: tuple[int, int]
) -> np.ndarray:
    import cv2

    width, height = size_wh
    polygons = np.rint(pixels[faces]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255, lineType=cv2.LINE_8)
    return mask


def _render_normal_mesh(
    pixels: np.ndarray,
    depths: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    size_wh: tuple[int, int],
) -> np.ndarray:
    import cv2

    width, height = size_wh
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    order = np.argsort(np.mean(depths[faces], axis=1))[::-1]
    colors = np.rint(np.clip((normals + 1.0) * 127.5, 0, 255)).astype(np.uint8)
    for face_index in order:
        face = faces[face_index]
        polygon = np.rint(pixels[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].min() >= height
        ):
            continue
        color = tuple(int(value) for value in np.mean(colors[face], axis=0))
        cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
    return image


def _verify_hunyuan_run(run_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = run_root / "manifest.json"
    done_path = run_root / "DONE"
    artifact = run_root / "artifact"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("Hunyuan run is missing manifest or DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.hunyuan3d_2mv_run.v1":
        raise ValueError("unexpected Hunyuan run schema")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Hunyuan DONE does not bind run manifest")
    artifact_manifest_path = artifact / "manifest.json"
    if manifest.get("artifact_manifest_sha256") != sha256_file(artifact_manifest_path):
        raise ValueError("Hunyuan run does not bind artifact manifest")
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("input", {}).get("inherited_geometry") is not None:
        raise ValueError("Hunyuan proposal illegally inherited geometry")
    mesh_path = artifact / artifact_manifest["mesh"]["ply_relpath"]
    if sha256_file(mesh_path) != artifact_manifest["mesh"]["ply_sha256"]:
        raise ValueError("Hunyuan mesh hash mismatch")
    return artifact_manifest, mesh_path


def align_mesh(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
        import imageio.v3 as iio
        import pymeshlab
        import trimesh
    except ImportError as exc:  # pragma: no cover - remote real2sim environment
        raise RuntimeError("OpenCV, imageio, pymeshlab, and trimesh are required") from exc

    input_root = args.input_root.resolve()
    hunyuan_run = args.hunyuan_run.resolve()
    output = args.output.resolve()
    input_manifest = validate_generation_input(input_root)
    parent_manifest, raw_mesh_path = _verify_hunyuan_run(hunyuan_run)
    if parent_manifest["input"]["generation_contract_manifest_sha256"] != sha256_file(
        input_root / "manifest.json"
    ):
        raise ValueError("Hunyuan proposal was not generated from the selected four-view contract")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mesh alignment: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        mesh_dir = staging / "mesh"
        audit_dir = staging / "audit"
        mesh_dir.mkdir(parents=True)
        audit_dir.mkdir(parents=True)
        decimated_path = mesh_dir / "decimated_raw.ply"
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(str(raw_mesh_path))
        loaded_faces = mesh_set.current_mesh().face_number()
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=args.target_faces,
            preservetopology=True,
            preservenormal=True,
            autoclean=True,
        )
        mesh_set.save_current_mesh(str(decimated_path))
        mesh = trimesh.load(decimated_path, force="mesh", process=False)
        vertices_raw = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(faces) < 1000:
            raise ValueError("decimated proposal is too small for alignment")

        targets: dict[str, np.ndarray] = {}
        observed_rgb: dict[str, np.ndarray] = {}
        for label in VIEW_ORDER:
            rgba = cv2.imread(
                str(input_root / "hunyuan3d_2mv" / f"{label}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            rgb = cv2.imread(str(input_root / "anchors" / f"{label}.png"), cv2.IMREAD_COLOR)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or rgb is None:
                raise ValueError(f"cannot load four-view target for {label}")
            targets[label] = rgba[..., 3] >= 128
            observed_rgb[label] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        candidates = []
        rotations = signed_permutation_rotations()
        for index, rotation in enumerate(rotations):
            vertices = vertices_raw @ rotation.T
            view_records = []
            for label in VIEW_ORDER:
                height, width = targets[label].shape
                pixels, fit, _ = _fit_projection(vertices, targets[label], label)
                rendered = _rasterize_silhouette(pixels, faces, (width, height)) >= 128
                view_records.append(
                    {"view": label, "silhouette_iou": silhouette_iou(rendered, targets[label]), **fit}
                )
            scores = [record["silhouette_iou"] for record in view_records]
            candidates.append(
                {
                    "rotation_index": index,
                    "raw_to_canonical_rotation_3x3": rotation.tolist(),
                    "mean_silhouette_iou": float(np.mean(scores)),
                    "min_silhouette_iou": float(np.min(scores)),
                    "views": view_records,
                }
            )
        candidates.sort(
            key=lambda item: (item["mean_silhouette_iou"], item["min_silhouette_iou"]),
            reverse=True,
        )
        selected = candidates[0]
        rotation = np.asarray(selected["raw_to_canonical_rotation_3x3"], dtype=np.float64)
        metric_vertices, transform = metric_transform(
            vertices_raw, rotation, input_manifest["metric_anchor"]["value_m"]
        )
        aligned_mesh = trimesh.Trimesh(
            vertices=metric_vertices,
            faces=faces,
            process=False,
            validate=False,
        )
        aligned_ply = mesh_dir / "aligned_metric_hunyuan3d_2mv.ply"
        aligned_glb = mesh_dir / "aligned_metric_hunyuan3d_2mv.glb"
        aligned_mesh.export(aligned_ply)
        aligned_mesh.export(aligned_glb)

        canonical_vertices = vertices_raw @ rotation.T
        canonical_mesh = trimesh.Trimesh(
            vertices=canonical_vertices, faces=faces, process=False, validate=False
        )
        normals = np.asarray(canonical_mesh.vertex_normals, dtype=np.float64)
        rows = []
        view_audit = []
        selected_by_view = {record["view"]: record for record in selected["views"]}
        for label in VIEW_ORDER:
            target = targets[label]
            height, width = target.shape
            pixels, _, depths = _fit_projection(canonical_vertices, target, label)
            silhouette = _rasterize_silhouette(pixels, faces, (width, height)) >= 128
            normal_render = _render_normal_mesh(
                pixels, depths, faces, normals, (width, height)
            )
            observed = observed_rgb[label]
            target_panel = np.repeat(target[..., None], 3, axis=2).astype(np.uint8) * 255
            overlay = observed.copy()
            contours, _ = cv2.findContours(
                silhouette.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 3)
            rows.append(np.concatenate((observed, target_panel, normal_render, overlay), axis=1))
            view_audit.append(
                {
                    **selected_by_view[label],
                    "rendered_support_fraction": float(silhouette.mean()),
                    "target_support_fraction": float(target.mean()),
                }
            )
        contact_sheet = np.concatenate(rows, axis=0)
        iio.imwrite(audit_dir / "four_real_view_alignment_contact_sheet.png", contact_sheet)
        (audit_dir / "orientation_candidates.json").write_text(
            json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        extents_m = np.ptp(metric_vertices, axis=0)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": input_manifest["asset_name"],
            "input": {
                "four_view_manifest_sha256": sha256_file(input_root / "manifest.json"),
                "hunyuan_run_manifest_sha256": sha256_file(hunyuan_run / "manifest.json"),
                "raw_mesh_sha256": sha256_file(raw_mesh_path),
                "inherited_procedural_geometry": None,
            },
            "decimation": {
                "loaded_raw_triangles": int(loaded_faces),
                "target_triangles": args.target_faces,
                "output_vertices": int(len(vertices_raw)),
                "output_triangles": int(len(faces)),
                "method": "pymeshlab_quadric_edge_collapse_preserve_topology_normals",
            },
            "orientation_search": {
                "candidate_count": len(rotations),
                "method": "right_handed_signed_permutations_orthographic_per_view_bbox_fit",
                "selected": selected,
                "top_five": candidates[:5],
            },
            "four_real_view_audit": view_audit,
            "metric_alignment": {
                "raw_to_canonical_metric_4x4": transform.tolist(),
                "height_anchor_m": input_manifest["metric_anchor"]["value_m"],
                "extents_m": extents_m.tolist(),
                "height_m": float(extents_m[2]),
                "origin_status": "bbox_center_approximation_pending_body_center_refinement",
                "camera_status": "per_view_orthographic_bbox_fit_not_photogrammetric_calibration",
            },
            "outputs": {
                "decimated_raw_ply": {
                    "relpath": "mesh/decimated_raw.ply",
                    "sha256": sha256_file(decimated_path),
                },
                "aligned_metric_ply": {
                    "relpath": "mesh/aligned_metric_hunyuan3d_2mv.ply",
                    "sha256": sha256_file(aligned_ply),
                },
                "aligned_metric_glb": {
                    "relpath": "mesh/aligned_metric_hunyuan3d_2mv.glb",
                    "sha256": sha256_file(aligned_glb),
                },
                "contact_sheet": {
                    "relpath": "audit/four_real_view_alignment_contact_sheet.png",
                    "sha256": sha256_file(
                        audit_dir / "four_real_view_alignment_contact_sheet.png"
                    ),
                },
            },
            "allowed_role": "coarsely_aligned_learned_mesh_prior_pending_visual_and_differentiable_refinement",
            "rejected_roles": [
                "observed_geometry",
                "final_metric_geometry",
                "physics_collision_geometry",
                "heldout_real_evidence",
            ],
            "review_status": "pending_visual_review_and_continuous_orbit_render",
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
            "stage": "four_real_view_coarse_learned_mesh_alignment",
            "status": "done_pending_visual_and_differentiable_refinement",
            "completed_utc": utc_now(),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "hashes_sha256": sha256_file(hashes_path),
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--hunyuan-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=60000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = align_mesh(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
