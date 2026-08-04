#!/usr/bin/env python3
"""Keep generated-fill Gaussians only where real anchor visibility is insufficient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.gaussian_appearance_delta import parse_vertex_layout, sha256_file
from gaussian.provenance_quarantine import assert_not_quarantined


SCHEMA = "radeon_oneloop.generated_fill_observed_visibility_prune.v1"
DONE_SCHEMA = "radeon_oneloop.generated_fill_observed_visibility_prune_done.v1"
DATASET_SCHEMAS = {
    "radeon_oneloop.hybrid_pseudoview_colmap_dataset.v1",
    "radeon_oneloop.seva_pseudoview_colmap_dataset.v1",
}


class VisibilityPruneError(ValueError):
    """Raised when generated fill cannot be separated from observed support."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def visibility_count(
    positions: np.ndarray,
    cameras: list[dict[str, Any]],
    masks: np.ndarray,
    *,
    surface_tolerance_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Count anchor cameras in which each center is a front visible surface sample."""

    xyz = np.asarray(positions, dtype=np.float64)
    alpha = np.asarray(masks, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.all(np.isfinite(xyz)):
        raise VisibilityPruneError("positions must be a finite N x 3 array")
    if alpha.ndim != 3 or len(alpha) != len(cameras):
        raise VisibilityPruneError("mask count must match camera count")
    if not np.isfinite(surface_tolerance_m) or surface_tolerance_m < 0:
        raise VisibilityPruneError("surface tolerance must be finite and nonnegative")
    counts = np.zeros(len(xyz), dtype=np.uint8)
    reports = []
    for camera_index, (camera, mask) in enumerate(zip(cameras, alpha, strict=True)):
        world_to_camera = np.asarray(
            camera["world_to_camera_opencv_4x4"], dtype=np.float64
        )
        intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
        if world_to_camera.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise VisibilityPruneError("camera matrices have invalid shapes")
        local = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = local[:, 2]
        safe_z = np.maximum(z, 1.0e-9)
        u = np.rint(intrinsic[0, 0] * local[:, 0] / safe_z + intrinsic[0, 2]).astype(
            np.int64
        )
        v = np.rint(intrinsic[1, 1] * local[:, 1] / safe_z + intrinsic[1, 2]).astype(
            np.int64
        )
        height, width = mask.shape
        inside = (z > 1.0e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        safe_u = np.clip(u, 0, width - 1)
        safe_v = np.clip(v, 0, height - 1)
        within_mask = mask[safe_v, safe_u] >= 128
        candidate = inside & within_mask
        flat = safe_v * width + safe_u
        depth_buffer = np.full(height * width, np.inf, dtype=np.float64)
        np.minimum.at(depth_buffer, flat[candidate], z[candidate])
        front_depth = depth_buffer[flat]
        visible = candidate & (z <= front_depth + surface_tolerance_m)
        counts += visible.astype(np.uint8)
        reports.append(
            {
                "camera_index": camera_index,
                "view_id": camera.get("view_id"),
                "projected_inside_mask": int(candidate.sum()),
                "front_visible_centers": int(visible.sum()),
            }
        )
    return counts, reports


def _read_ply(path: Path) -> tuple[int, int, np.dtype, np.memmap]:
    offset, count, dtype = parse_vertex_layout(path)
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    return offset, count, dtype, vertices


def _ply_header(path: Path) -> list[str]:
    lines = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise VisibilityPruneError("PLY header is incomplete")
            lines.append(raw.decode("ascii"))
            if raw.strip() == b"end_header":
                return lines


def _write_pruned_ply(
    source: Path,
    output: Path,
    vertices: np.memmap,
    keep: np.ndarray,
) -> None:
    header = _ply_header(source)
    source_count = len(vertices)
    replaced = False
    rewritten = []
    for line in header:
        if line.strip() == f"element vertex {source_count}":
            rewritten.append(f"element vertex {int(keep.sum())}\n")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        raise VisibilityPruneError("PLY vertex count line is missing")
    with output.open("xb") as handle:
        handle.write("".join(rewritten).encode("ascii"))
        handle.write(np.asarray(vertices[keep]).tobytes())


def _load_masks(dataset: Path, cameras: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    import cv2

    masks = []
    records = []
    for camera in cameras:
        name = str(camera["source_image_name"])
        path = dataset / "masks" / f"{Path(name).stem}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        expected_wh = camera["image_size_wh"]
        if mask is None or list(mask.shape[::-1]) != expected_wh:
            raise VisibilityPruneError(f"invalid real anchor mask: {path}")
        masks.append(mask)
        records.append(
            {
                "view_id": str(camera["view_id"]),
                "source_image_name": name,
                "mask_sha256": sha256_file(path),
            }
        )
    return np.stack(masks), records


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.ply.resolve()
    source_provenance_path = args.source_provenance.resolve()
    dataset = args.dataset.resolve()
    cameras_path = args.cameras.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.max_observed_views < 0 or args.max_observed_views >= 4:
        raise VisibilityPruneError("max-observed-views must be in [0, 3]")

    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    if source_provenance.get("provenance_class") != "generated_fill_candidate":
        raise VisibilityPruneError("source must be a generated-fill candidate")
    if source_provenance.get("output_ply_sha256") != sha256_file(source):
        raise VisibilityPruneError("source provenance does not bind the PLY")
    if source_provenance.get("formal") is not False:
        raise VisibilityPruneError("generated fill must remain nonformal")
    dataset_manifest_path = dataset / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("schema_version") not in DATASET_SCHEMAS:
        raise VisibilityPruneError("unsupported generated pseudo-view dataset")
    initial = dataset_manifest.get("initial_points", {})
    if initial.get("source") != "observed_real_mask_CPU_visual_hull" or initial.get(
        "generated_geometry_prior"
    ) is not False:
        raise VisibilityPruneError("generated fill does not originate from observed geometry")
    assert_not_quarantined(
        [
            ("generated_fill_source_provenance", source_provenance),
            ("generated_fill_dataset", dataset_manifest),
        ]
    )
    cameras_document = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras = cameras_document.get("cameras")
    if (
        cameras_document.get("camera_model") != "PINHOLE_OPENCV"
        or cameras_document.get("mode") != "cardinal_real"
        or not isinstance(cameras, list)
        or len(cameras) != 4
    ):
        raise VisibilityPruneError("visibility pruning requires four real cardinal cameras")
    if cameras_document.get("dataset_manifest_sha256") != sha256_file(dataset_manifest_path):
        raise VisibilityPruneError("real cameras do not bind the selected dataset")
    masks, mask_records = _load_masks(dataset, cameras)

    _, count, _, vertices = _read_ply(source)
    positions = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1).astype(
        np.float64
    )
    counts, camera_reports = visibility_count(
        positions,
        cameras,
        masks,
        surface_tolerance_m=args.surface_tolerance_m,
    )
    keep = counts <= args.max_observed_views
    if int(keep.sum()) < args.min_output_gaussians:
        raise VisibilityPruneError(
            f"visibility prune leaves too few Gaussians: {int(keep.sum())}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        output_ply = staging / "appearance_fill_unobserved.ply"
        _write_pruned_ply(source, output_ply, vertices, keep)
        mask_path = staging / "visibility.npz"
        np.savez_compressed(mask_path, observed_visibility_count=counts, keep=keep)
        histogram = {
            str(value): int(np.count_nonzero(counts == value)) for value in range(5)
        }
        metrics = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "source_ply_sha256": sha256_file(source),
            "source_provenance_sha256": sha256_file(source_provenance_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "real_cameras_sha256": sha256_file(cameras_path),
            "source_gaussians": count,
            "output_gaussians": int(keep.sum()),
            "removed_observed_support_gaussians": int((~keep).sum()),
            "observed_visibility_count_histogram": histogram,
            "max_observed_views_kept": args.max_observed_views,
            "surface_tolerance_m": args.surface_tolerance_m,
            "camera_reports": camera_reports,
            "real_masks": mask_records,
            "visibility_mask_sha256": sha256_file(mask_path),
            "output_ply_sha256": sha256_file(output_ply),
            "allowed_role": "low_confidence_low_visibility_appearance_fill_layer",
            "prohibited_roles": [
                "observed_core",
                "collision_geometry",
                "heldout_real_evidence",
                "formal_single_radeon_result",
            ],
        }
        metrics_path = staging / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = dict(source_provenance)
        provenance.update(
            {
                "parent_output_ply_sha256": sha256_file(source),
                "observed_visibility_prune_metrics_sha256": sha256_file(metrics_path),
                "observed_visibility_policy": (
                    f"keep_centers_visible_in_at_most_{args.max_observed_views}_real_anchors"
                ),
                "output_ply_sha256": sha256_file(output_ply),
                "gaussian_count": int(keep.sum()),
                "eligible_for_formal_metrics": False,
                "eligible_for_heldout_real_metrics": False,
            }
        )
        provenance_path = staging / "appearance_fill_unobserved.provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = []
        for path in sorted(staging.iterdir()):
            if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
                lines.append(f"{sha256_file(path)}  {path.name}")
        hashes_path = staging / "hashes.sha256"
        hashes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA,
                    "status": "done_unobserved_generated_fill_candidate",
                    "metrics_sha256": sha256_file(metrics_path),
                    "provenance_sha256": sha256_file(provenance_path),
                    "hashes_sha256": sha256_file(hashes_path),
                    "completed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return metrics
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.generated_fill_observed_visibility_prune_failure.v1",
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
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-observed-views", type=int, default=0)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.004)
    parser.add_argument("--min-output-gaussians", type=int, default=100)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
