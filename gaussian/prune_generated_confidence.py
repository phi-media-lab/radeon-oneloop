#!/usr/bin/env python3
"""Prune single-source generated Gaussians with accepted metric depth evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from gaussian.gaussian_appearance_delta import parse_vertex_layout, sha256_file


def compute_depth_prune_mask(
    positions: np.ndarray,
    source_view: np.ndarray,
    cross_source_count: np.ndarray,
    cameras: list[dict[str, Any]],
    masks: np.ndarray,
    depth_metric: np.ndarray,
    *,
    source_max_abs_depth_error_m: float,
    max_front_conflict_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = len(positions)
    source_valid = np.zeros(count, dtype=bool)
    source_error = np.full(count, np.inf, dtype=np.float64)
    front_conflict_count = np.zeros(count, dtype=np.uint8)
    for view_index, camera in enumerate(cameras):
        world_to_camera = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float64)
        intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float64)
        local = positions @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = local[:, 2]
        u = np.rint(intrinsic[0, 0] * local[:, 0] / np.maximum(z, 1.0e-9) + intrinsic[0, 2]).astype(np.int64)
        v = np.rint(intrinsic[1, 1] * local[:, 1] / np.maximum(z, 1.0e-9) + intrinsic[1, 2]).astype(np.int64)
        height, width = masks[view_index].shape
        inside = (z > 1.0e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        safe_u = np.clip(u, 0, width - 1)
        safe_v = np.clip(v, 0, height - 1)
        valid = inside & (masks[view_index, safe_v, safe_u] >= 128)
        expected = depth_metric[view_index, safe_v, safe_u]
        delta = z - expected
        is_source = source_view == view_index
        source_indices = valid & is_source
        source_valid[source_indices] = True
        source_error[source_indices] = np.abs(delta[source_indices])
        front_conflict_count += (
            valid & ~is_source & (delta < -max_front_conflict_m)
        ).astype(np.uint8)

    multi = cross_source_count >= 2
    single_keep = (
        source_valid
        & (source_error <= source_max_abs_depth_error_m)
        & (front_conflict_count == 0)
    )
    keep = multi | single_keep
    finite_source_error = source_error[np.isfinite(source_error)]
    report = {
        "input_gaussians": count,
        "cross_source_gaussians_kept": int(multi.sum()),
        "single_source_input": int((~multi).sum()),
        "single_source_depth_consistent_kept": int((~multi & single_keep).sum()),
        "single_source_missing_source_projection": int((~multi & ~source_valid).sum()),
        "single_source_front_conflict": int((~multi & (front_conflict_count > 0)).sum()),
        "output_gaussians": int(keep.sum()),
        "source_abs_depth_error_m_quantiles_p50_p80_p90_p95_p99": (
            np.quantile(finite_source_error, [0.5, 0.8, 0.9, 0.95, 0.99]).tolist()
            if len(finite_source_error)
            else None
        ),
        "source_max_abs_depth_error_m": source_max_abs_depth_error_m,
        "max_front_conflict_m": max_front_conflict_m,
    }
    return keep, report


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    ply = args.ply.resolve()
    offset, count, dtype = parse_vertex_layout(ply)
    required = {"x", "y", "z", "source_view", "cross_view_source_count"}
    if not required.issubset(dtype.names or ()):
        raise ValueError(f"PLY lacks confidence fields: {sorted(required - set(dtype.names or ()))}")
    vertices = np.memmap(ply, dtype=dtype, mode="r", offset=offset, shape=(count,))
    positions = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    pose = np.load(args.pose_npz.resolve())
    cameras_document = json.loads(args.cameras.resolve().read_text(encoding="utf-8"))
    cameras = cameras_document["cameras"]
    if len(cameras) != 4 or tuple(pose["view_ids"].tolist()) != (
        "anchor_front", "anchor_right", "anchor_rear", "anchor_left"
    ):
        raise ValueError("confidence pruning requires the accepted four-view camera order")
    similarity = json.loads(args.similarity.resolve().read_text(encoding="utf-8"))
    depth_metric = pose["depth"].astype(np.float64) * float(similarity["scale"])
    keep, report = compute_depth_prune_mask(
        positions,
        np.asarray(vertices["source_view"]),
        np.asarray(vertices["cross_view_source_count"]),
        cameras,
        pose["masks"],
        depth_metric,
        source_max_abs_depth_error_m=args.source_max_abs_depth_error_m,
        max_front_conflict_m=args.max_front_conflict_m,
    )
    metadata = {
        "schema_version": "radeon_oneloop.generated_fill_confidence_prune.v1",
        "formal": False,
        "source_ply_sha256": sha256_file(ply),
        "pose_npz_sha256": sha256_file(args.pose_npz.resolve()),
        "cameras_sha256": sha256_file(args.cameras.resolve()),
        "similarity_sha256": sha256_file(args.similarity.resolve()),
        "policy": "keep_all_cross_source_and_depth_consistent_non_front-conflicting_single_source",
        "eligible_for_formal_metrics": False,
        "report": report,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    np.savez_compressed(
        args.output,
        keep=keep,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    metadata["mask_sha256"] = sha256_file(args.output)
    metadata["mask_bytes"] = args.output.stat().st_size
    return metadata


def _ply_header(path: Path) -> tuple[list[str], int]:
    lines = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header is incomplete")
            lines.append(raw.decode("ascii"))
            if raw.strip() == b"end_header":
                return lines, handle.tell()


def apply(args: argparse.Namespace) -> dict[str, Any]:
    ply = args.ply.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.mask.resolve(), allow_pickle=False) as archive:
        keep = archive["keep"].astype(bool)
        metadata = json.loads(str(archive["metadata_json"]))
    if sha256_file(ply) != metadata["source_ply_sha256"]:
        raise ValueError("confidence mask source PLY hash mismatch")
    offset, count, dtype = parse_vertex_layout(ply)
    if keep.shape != (count,):
        raise ValueError("confidence mask length differs from PLY")
    vertices = np.memmap(ply, dtype=dtype, mode="r", offset=offset, shape=(count,))
    header, _ = _ply_header(ply)
    rewritten = []
    replaced = False
    for line in header:
        if line.strip() == f"element vertex {count}":
            rewritten.append(f"element vertex {int(keep.sum())}\n")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        raise ValueError("PLY vertex header was not found")
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write("".join(rewritten).encode("ascii"))
            handle.write(np.asarray(vertices[keep]).tobytes())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = {
        "schema_version": metadata["schema_version"],
        "formal": False,
        "parent_ply_sha256": metadata["source_ply_sha256"],
        "confidence_mask_sha256": sha256_file(args.mask.resolve()),
        "output_ply_sha256": sha256_file(output),
        "output_gaussians": int(keep.sum()),
        "pruning_report": metadata["report"],
        "eligible_for_formal_metrics": False,
    }
    if args.source_provenance is not None or args.output_provenance is not None:
        if args.source_provenance is None or args.output_provenance is None:
            raise ValueError("source and output provenance must be supplied together")
        provenance = json.loads(args.source_provenance.resolve().read_text(encoding="utf-8"))
        if provenance.get("output_ply_sha256") != result["parent_ply_sha256"]:
            raise ValueError("source provenance does not match parent PLY")
        provenance.update(
            {
                "parent_output_ply_sha256": result["parent_ply_sha256"],
                "confidence_pruning_mask_sha256": result["confidence_mask_sha256"],
                "confidence_pruning_report": result["pruning_report"],
                "output_ply_sha256": result["output_ply_sha256"],
                "eligible_for_heldout_real_metrics": False,
                "eligible_for_formal_metrics": False,
            }
        )
        args.output_provenance.resolve().write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["output_provenance_sha256"] = sha256_file(args.output_provenance.resolve())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--ply", type=Path, required=True)
    analyze_parser.add_argument("--pose-npz", type=Path, required=True)
    analyze_parser.add_argument("--cameras", type=Path, required=True)
    analyze_parser.add_argument("--similarity", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--source-max-abs-depth-error-m", type=float, default=0.008)
    analyze_parser.add_argument("--max-front-conflict-m", type=float, default=0.004)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--ply", type=Path, required=True)
    apply_parser.add_argument("--mask", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--source-provenance", type=Path)
    apply_parser.add_argument("--output-provenance", type=Path)
    args = parser.parse_args()
    result = analyze(args) if args.command == "analyze" else apply(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
