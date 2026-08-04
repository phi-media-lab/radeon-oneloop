#!/usr/bin/env python3
"""Export reviewed object views and a gated pose candidate as a COLMAP text dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


class ObjectColmapError(ValueError):
    """Raised when object evidence cannot be exported without breaking provenance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_matrix_to_colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP's Hamilton ``[qw, qx, qy, qz]`` quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ObjectColmapError("rotation must be 3 x 3")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-5):
        raise ObjectColmapError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-5):
        raise ObjectColmapError("rotation is not proper")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    value = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    value /= np.linalg.norm(value)
    if value[0] < 0:
        value *= -1.0
    return value


def scale_intrinsic(
    intrinsic: np.ndarray,
    source_size_wh: tuple[int, int],
    target_size_wh: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size_wh
    target_width, target_height = target_size_wh
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ObjectColmapError("image sizes must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    if result.shape != (3, 3):
        raise ObjectColmapError("intrinsic must be 3 x 3")
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result


def _safe_output_path(root: Path, relpath: str) -> Path:
    path = (root / relpath).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ObjectColmapError(f"pose output path escaped its run root: {relpath}") from exc
    return path


def _load_inputs(
    m1_manifest_path: Path, pose_run_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    m1 = json.loads(m1_manifest_path.read_text(encoding="utf-8"))
    pose_run = json.loads(pose_run_manifest_path.read_text(encoding="utf-8"))
    if m1.get("schema_version") != "radeon_oneloop.object_asset_manifest.v1":
        raise ObjectColmapError("input is not an M1 object manifest")
    if m1.get("formal") is not False or pose_run.get("formal") is not False:
        raise ObjectColmapError("object dataset export must remain nonformal")
    if m1.get("summary", {}).get("mask_review_status") != "reviewed_pass":
        raise ObjectColmapError("M1 masks have not passed review")
    m1_hash = sha256_file(m1_manifest_path)
    if pose_run.get("m1_manifest_sha256") != m1_hash:
        raise ObjectColmapError("pose candidate was not derived from this M1 manifest")
    by_name = {Path(item["relpath"]).name: item for item in pose_run.get("outputs", [])}
    required = {
        "cameras_observed.json": by_name.get("cameras_observed.json"),
        "vggt_omega_pose_depth.npz": by_name.get("vggt_omega_pose_depth.npz"),
    }
    missing = [name for name, item in required.items() if item is None]
    if missing:
        raise ObjectColmapError(f"pose run is missing required outputs: {missing}")
    root = pose_run_manifest_path.parent
    cameras_path = _safe_output_path(root, required["cameras_observed.json"]["relpath"])
    points_path = _safe_output_path(root, required["vggt_omega_pose_depth.npz"]["relpath"])
    for name, path in (("cameras", cameras_path), ("points", points_path)):
        expected = required[path.name]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ObjectColmapError(f"pose {name} output is missing or has a hash mismatch")
    return m1, pose_run, cameras_path, points_path


def validate_pose_audit(
    audit: dict[str, Any],
    *,
    expected_pose_run_sha256: str,
) -> None:
    if audit.get("schema_version") != "radeon_oneloop.object_pose_visual_audit.v1":
        raise ObjectColmapError("pose audit has an unsupported schema")
    if audit.get("formal") is not False:
        raise ObjectColmapError("pose audit must remain nonformal")
    if audit.get("source_run_manifest_sha256") != expected_pose_run_sha256:
        raise ObjectColmapError("pose audit does not refer to the selected pose run")
    if audit.get("review", {}).get("status") != "accepted_pose_and_coarse_geometry_initializer":
        raise ObjectColmapError("pose candidate has not passed visual identity review")


def _write_hashes(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "hashes.sha256").write_text("".join(lines), encoding="utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    m1_path = args.m1_manifest.resolve()
    pose_run_path = args.pose_run_manifest.resolve()
    pose_audit_path = args.pose_audit_manifest.resolve()
    m1_root = m1_path.parent
    m1, pose_run, cameras_path, points_path = _load_inputs(m1_path, pose_run_path)
    pose_audit = json.loads(pose_audit_path.read_text(encoding="utf-8"))
    validate_pose_audit(
        pose_audit,
        expected_pose_run_sha256=sha256_file(pose_run_path),
    )
    cameras_document = json.loads(cameras_path.read_text(encoding="utf-8"))
    if cameras_document.get("method") != "vggt_omega_1b_512_metric_candidate":
        raise ObjectColmapError("only the gated metric VGGT-Omega candidate is supported")
    camera_by_id = {camera["view_id"]: camera for camera in cameras_document["cameras"]}
    views = [view for view in m1["views"] if "photometric" in view["roles"]]
    if len(views) != 4 or {view["id"] for view in views} != set(camera_by_id):
        raise ObjectColmapError("four photometric M1 anchors must match the four learned cameras")
    heldout_view_id = args.heldout_view_id
    eval_probe_view_id = args.all_train_eval_probe_view_id
    selected_split_id = heldout_view_id or eval_probe_view_id
    if selected_split_id not in camera_by_id:
        raise ObjectColmapError(f"unknown split/probe view: {selected_split_id}")
    if heldout_view_id is not None:
        entries = [
            (view, False)
            for view in sorted(
                views,
                key=lambda view: (view["id"] != heldout_view_id, view["id"]),
            )
        ]
        split_rule = (
            "heldout filename is lexicographically first; pinned VkSplat sorts by filename before applying "
            "image_index modulo eval_interval"
        )
    else:
        probe_view = next(view for view in views if view["id"] == eval_probe_view_id)
        entries = [(probe_view, True)] + [
            (view, False) for view in sorted(views, key=lambda view: view["id"])
        ]
        split_rule = (
            "000_eval_probe filename is lexicographically first; all four unique observed views are training "
            "images; probe scores are not held-out metrics"
        )

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
        exported_views = []
        for index, (view, is_eval_probe_duplicate) in enumerate(entries, start=1):
            view_id = view["id"]
            image_source = _safe_output_path(m1_root, view["image"]["relpath"])
            mask_source = _safe_output_path(m1_root, view["hard_mask"]["relpath"])
            for kind, source, record in (
                ("image", image_source, view["image"]),
                ("mask", mask_source, view["hard_mask"]),
            ):
                if not source.is_file() or sha256_file(source) != record["sha256"]:
                    raise ObjectColmapError(f"{view_id} {kind} is missing or has a hash mismatch")
            is_heldout = heldout_view_id is not None and view_id == heldout_view_id
            if is_eval_probe_duplicate:
                image_name = f"000_eval_probe_{view_id}.png"
            elif is_heldout:
                image_name = f"000_heldout_{view_id}.png"
            else:
                image_name = f"{view_id}.png"
            shutil.copy2(image_source, image_dir / image_name)
            shutil.copy2(mask_source, mask_dir / image_name)
            camera = camera_by_id[view_id]
            target_size = (
                int(view["normalization"]["output_width"]),
                int(view["normalization"]["output_height"]),
            )
            source_size = tuple(int(item) for item in camera["image_size_wh"])
            intrinsic = scale_intrinsic(
                np.asarray(camera["intrinsic_3x3"], dtype=np.float64),
                source_size,
                target_size,
            )
            transform = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
            qvec = rotation_matrix_to_colmap_qvec(transform[:3, :3])
            translation = transform[:3, 3]
            camera_lines.append(
                f"{index} PINHOLE {target_size[0]} {target_size[1]} "
                f"{intrinsic[0, 0]:.12g} {intrinsic[1, 1]:.12g} "
                f"{intrinsic[0, 2]:.12g} {intrinsic[1, 2]:.12g}\n"
            )
            values = [*qvec.tolist(), *translation.tolist()]
            image_lines.append(
                f"{index} " + " ".join(f"{value:.17g}" for value in values) + f" {index} {image_name}\n\n"
            )
            exported_views.append(
                {
                    "view_id": view_id,
                    "image_id": index,
                    "camera_id": index,
                    "heldout": is_heldout,
                    "evaluation_probe_duplicate": is_eval_probe_duplicate,
                    "included_in_training": not (
                        is_eval_probe_duplicate
                        or (heldout_view_id is not None and view_id == heldout_view_id)
                    ),
                    "image_sha256": view["image"]["sha256"],
                    "mask_sha256": view["hard_mask"]["sha256"],
                    "target_size_wh": list(target_size),
                    "source_pose_size_wh": list(source_size),
                    "scaled_intrinsic_3x3": intrinsic.tolist(),
                }
            )
        (sparse_dir / "cameras.txt").write_text("".join(camera_lines), encoding="utf-8")
        (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")

        point_data = np.load(points_path)
        points = np.asarray(point_data["positions_metric"], dtype=np.float64)
        colors = np.clip(np.round(np.asarray(point_data["colors"]) * 255.0), 0, 255).astype(np.uint8)
        confidence = np.asarray(point_data["confidence"], dtype=np.float64)
        mask_support = np.asarray(point_data["mask_hit_count"], dtype=np.uint8)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (mask_support >= 2)
        candidates = np.flatnonzero(valid)
        if len(candidates) < args.min_points:
            raise ObjectColmapError(f"only {len(candidates)} valid initialization points")
        if len(candidates) > args.max_points:
            generator = np.random.default_rng(args.sample_seed)
            candidates = np.sort(generator.choice(candidates, size=args.max_points, replace=False))
        points = points[candidates]
        colors = colors[candidates]
        point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
        for point_id, (point, color) in enumerate(zip(points, colors), start=1):
            point_lines.append(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")

        robust_height = float(np.quantile(points[:, 2], 0.995) - np.quantile(points[:, 2], 0.005))
        manifest = {
            "schema_version": "radeon_oneloop.object_colmap_dataset.v1",
            "formal": False,
            "asset_name": m1["asset_name"],
            "m1_manifest_sha256": sha256_file(m1_path),
            "pose_run_manifest_sha256": sha256_file(pose_run_path),
            "pose_visual_audit_manifest_sha256": sha256_file(pose_audit_path),
            "pose_acceptance_status": pose_run["acceptance_status"],
            "pose_visual_review_status": pose_audit["review"]["status"],
            "heldout_view_id": heldout_view_id,
            "all_train_eval_probe_view_id": eval_probe_view_id,
            "evaluation_split_rule": split_rule,
            "views": exported_views,
            "initial_points": {
                "count": int(len(points)),
                "source_npz_sha256": sha256_file(points_path),
                "minimum_mask_support": 2,
                "sample_seed": int(args.sample_seed),
                "robust_height_m_p005_p995": robust_height,
                "metric_anchor_m": float(m1["metric_anchor"]["value_m"]),
            },
            "provenance": {
                "images": "observed_tier_A_only",
                "masks": "reviewed_observed_masks",
                "poses_and_initial_points": "nonformal_MI300X_VGGT_Omega_candidate",
                "generated_views": False,
                "generated_geometry": False,
            },
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "stage": "M2_object_COLMAP_export",
                    "status": "done_nonformal_dataset",
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
                {"stage": "M2_object_COLMAP_export", "status": "failed", "error": str(exc)},
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
    parser.add_argument("--pose-run-manifest", type=Path, required=True)
    parser.add_argument("--pose-audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--heldout-view-id")
    split.add_argument("--all-train-eval-probe-view-id")
    parser.add_argument("--min-points", type=int, default=10000)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    value = export(parse_args())
    print(
        json.dumps(
            {
                "heldout": value["heldout_view_id"],
                "all_train_eval_probe": value["all_train_eval_probe_view_id"],
                "points": value["initial_points"]["count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
