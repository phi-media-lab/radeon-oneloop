#!/usr/bin/env python3
"""Export the real-only CPU visual hull as a portable GS initializer."""

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

from gaussian.prepare_four_view_generation import sha256_file


SOURCE_SCHEMA = "radeon_oneloop.manual_ring_visual_hull_colmap.v1"
SCHEMA = "radeon_oneloop.observed_visual_hull_initialization.v1"
DONE_SCHEMA = "radeon_oneloop.observed_visual_hull_initialization_done.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_colmap_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    colors: list[list[int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"malformed COLMAP point on line {line_number}")
        vertices.append([float(value) for value in fields[1:4]])
        colors.append([int(value) for value in fields[4:7]])
    xyz = np.asarray(vertices, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 1000:
        raise ValueError("observed visual hull contains too few points")
    if not np.all(np.isfinite(xyz)) or np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError("observed visual hull contains invalid coordinates or colors")
    return xyz, rgb.astype(np.uint8)


def validate_source(root: Path) -> tuple[dict[str, Any], Path, np.ndarray, np.ndarray]:
    manifest_path = root / "dataset_manifest.json"
    done_path = root / "DONE"
    points_path = root / "sparse/0/points3D.txt"
    if not manifest_path.is_file() or not done_path.is_file() or not points_path.is_file():
        raise ValueError("source observed-only dataset is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SOURCE_SCHEMA or not manifest.get(
        "formal_input_eligible"
    ):
        raise ValueError("source is not the reviewed observed visual-hull dataset")
    provenance = manifest.get("provenance", {})
    required_false = (
        "generated_geometry",
        "generated_views",
        "learned_depth",
        "secondary_accelerator_artifacts",
    )
    if any(provenance.get(key) is not False for key in required_false):
        raise ValueError("observed initializer source contains generated or learned geometry")
    if provenance.get("initial_points") != "deterministic_reviewed_mask_visual_hull_CPU":
        raise ValueError("source initial points are not the deterministic CPU visual hull")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("source DONE does not bind its manifest")
    indexed = None
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        if line.endswith("  sparse/0/points3D.txt"):
            indexed = line.split("  ", 1)[0]
            break
    if indexed is None or indexed != sha256_file(points_path):
        raise ValueError("source hash index does not bind points3D.txt")
    vertices, colors = load_colmap_points(points_path)
    expected_count = int(manifest["visual_hull"]["sampled_surface_points"])
    if len(vertices) != expected_count:
        raise ValueError("observed visual-hull point count changed")
    return manifest, points_path, vertices, colors


def export_initialization(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source, points_path, vertices, _ = validate_source(source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copy2(points_path, staging / "points3D.txt")
        extents = np.ptp(vertices, axis=0)
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "host_role": args.host_role,
            "asset_name": source["asset_name"],
            "source": {
                "dataset_manifest_sha256": sha256_file(source_root / "dataset_manifest.json"),
                "dataset_hashes_sha256": sha256_file(source_root / "hashes.sha256"),
                "points3d_sha256": sha256_file(points_path),
                "m1_manifest_sha256": source["m1_manifest_sha256"],
                "formal_input_eligible": True,
            },
            "points": {
                "relpath": "points3D.txt",
                "sha256": sha256_file(staging / "points3D.txt"),
                "count": int(len(vertices)),
                "bounds_min_m": vertices.min(axis=0).tolist(),
                "bounds_max_m": vertices.max(axis=0).tolist(),
                "extents_m": extents.tolist(),
                "coordinate_frame": "object_canonical_metric",
            },
            "provenance": {
                "method": "deterministic_reviewed_mask_visual_hull_CPU",
                "observed_real_masks_only": True,
                "generated_geometry": False,
                "generated_views": False,
                "learned_depth": False,
                "secondary_accelerator_artifacts": False,
            },
            "allowed_role": "nonformal_generated_appearance_training_geometry_initializer",
            "required_downstream_constraints": [
                "freeze_geometry",
                "generated_views_low_confidence",
                "no_collision_geometry_claim",
                "no_heldout_quality_claim",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes = staging / "hashes.sha256"
        hashes.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in sorted(staging.iterdir())
                if path.is_file() and path.name not in {"hashes.sha256", "DONE"}
            )
            + "\n",
            encoding="utf-8",
        )
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "done_observed_visual_hull_initialization",
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "hashes_sha256": sha256_file(hashes),
            "completed_utc": utc_now(),
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
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-role", default="radeon_c_cpu_export_nonformal_derivative")
    return parser


def main() -> None:
    print(json.dumps(export_initialization(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
