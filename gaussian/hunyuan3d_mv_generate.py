#!/usr/bin/env python3
"""Generate an auditable Hunyuan3D-2mv mesh from the reviewed four views."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from gaussian.prepare_four_view_generation import (
    sha256_file,
    validate_generation_input,
)


SCHEMA_VERSION = "radeon_oneloop.hunyuan3d_2mv_mesh_proposal.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.hunyuan3d_2mv_mesh_done.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mesh_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("generated mesh must contain at least four 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("generated mesh must contain at least four triangular faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("generated mesh contains non-finite vertices")
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.equal(faces, np.floor(faces))):
            raise ValueError("generated face indices are not integers")
        faces = faces.astype(np.int64)
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("generated face index is out of bounds")
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extents = maximum - minimum
    if np.any(extents <= 0) or not np.all(np.isfinite(extents)):
        raise ValueError("generated mesh has degenerate bounds")
    unique_faces = np.unique(np.sort(faces.astype(np.int64), axis=1), axis=0)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "unique_triangles": int(len(unique_faces)),
        "duplicate_triangle_fraction": float(1.0 - len(unique_faces) / len(faces)),
        "bounds_min_raw": minimum.tolist(),
        "bounds_max_raw": maximum.tolist(),
        "extents_raw": extents.tolist(),
        "max_extent_raw": float(extents.max()),
        "finite": True,
    }


def _write_hash_index(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def _checkpoint_files(snapshot_root: Path, subfolder: str) -> list[dict[str, Any]]:
    folder = snapshot_root / subfolder
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relpath": path.relative_to(snapshot_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files or not any(item["relpath"].endswith(".safetensors") for item in files):
        raise RuntimeError("Hunyuan snapshot has no hashable safetensors checkpoint")
    return files


def validate_local_snapshot(path: Path, subfolder: str) -> tuple[Path, str]:
    snapshot = path.resolve()
    revision = snapshot.name
    if not snapshot.is_dir():
        raise FileNotFoundError(f"local Hunyuan snapshot does not exist: {snapshot}")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("local Hunyuan snapshot directory must be named by its 40-hex revision")
    _checkpoint_files(snapshot, subfolder)
    return snapshot, revision


def generate_mesh(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    output = args.output.resolve()
    input_manifest = validate_generation_input(input_root)
    contract = input_manifest["generator_contracts"]["hunyuan3d_2mv"]
    if contract["model"] != args.model or contract["subfolder"] != args.subfolder:
        raise ValueError("requested Hunyuan model differs from the bound generation contract")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Hunyuan proposal: {output}")
    if not 1 <= args.num_inference_steps <= 100:
        raise ValueError("num inference steps must be in [1, 100]")
    if not 128 <= args.octree_resolution <= 512:
        raise ValueError("octree resolution must be in [128, 512]")
    if not math.isfinite(args.guidance_scale) or not 0.0 <= args.guidance_scale <= 20.0:
        raise ValueError("guidance scale must be finite and in [0, 20]")

    try:
        import torch
        from huggingface_hub import model_info, snapshot_download
        from PIL import Image
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as exc:  # pragma: no cover - remote generator environment
        raise RuntimeError("Hunyuan3D-2mv runtime dependencies are missing") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Hunyuan generation requires exactly one visible ROCm device")
    hardware_name = torch.cuda.get_device_name(0)
    if args.require_mi300x and "MI300X" not in hardware_name:
        raise RuntimeError(f"expected MI300X, got {hardware_name}")

    if args.local_snapshot is not None:
        snapshot, resolved_revision = validate_local_snapshot(
            args.local_snapshot, args.subfolder
        )
        snapshot_resolution = "explicit_content_addressed_local_snapshot"
    else:
        info = model_info(args.model, revision=args.revision)
        resolved_revision = str(info.sha)
        snapshot = Path(
            snapshot_download(
                repo_id=args.model,
                revision=resolved_revision,
                allow_patterns=[f"{args.subfolder}/*"],
            )
        ).resolve()
        snapshot_resolution = "huggingface_api_resolved_snapshot"
    checkpoint_files = _checkpoint_files(snapshot, args.subfolder)

    input_dir = input_root / contract["input_relpath"]
    images = {}
    for key in contract["view_keys"]:
        path = input_dir / f"{key}.png"
        if not path.is_file():
            raise FileNotFoundError(f"missing Hunyuan input view: {key}")
        images[key] = Image.open(path).convert("RGBA")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        mesh_dir = staging / "mesh"
        mesh_dir.mkdir(parents=True)
        torch.cuda.reset_peak_memory_stats()
        started_utc = utc_now()
        started = time.perf_counter()
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(snapshot),
            subfolder=args.subfolder,
            variant="fp16",
            device="cuda",
            dtype=torch.float16,
        )
        load_runtime_s = time.perf_counter() - started
        generation_started = time.perf_counter()
        mesh = pipeline(
            image=images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            generator=torch.manual_seed(args.seed),
            output_type="trimesh",
        )[0]
        torch.cuda.synchronize()
        generation_runtime_s = time.perf_counter() - generation_started
        stats = mesh_statistics(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        stats.update(
            {
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "euler_number": int(mesh.euler_number),
                "body_count": int(mesh.body_count),
                "surface_area_raw": float(mesh.area),
                "volume_raw_if_watertight": float(mesh.volume) if mesh.is_watertight else None,
            }
        )
        glb_path = mesh_dir / "raw_hunyuan3d_2mv.glb"
        ply_path = mesh_dir / "raw_hunyuan3d_2mv.ply"
        mesh.export(glb_path)
        mesh.export(ply_path)
        if not glb_path.is_file() or not ply_path.is_file():
            raise RuntimeError("Hunyuan mesh export did not produce GLB and PLY")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "started_utc": started_utc,
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": input_manifest["asset_name"],
            "host_role": "phi_amd_work_mi300x_nonformal_generation_lab",
            "hardware": {
                "device": hardware_name,
                "device_count": torch.cuda.device_count(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "input": {
                "generation_contract_manifest_sha256": sha256_file(input_root / "manifest.json"),
                "generation_contract_hashes_sha256": sha256_file(input_root / "hashes.sha256"),
                "observed_views": 4,
                "view_keys": contract["view_keys"],
                "inherited_geometry": None,
                "provenance": "observed_identity_inputs_only",
            },
            "model": {
                "repo_id": args.model,
                "revision": resolved_revision,
                "snapshot_resolution": snapshot_resolution,
                "subfolder": args.subfolder,
                "variant": "fp16",
                "checkpoint_files": checkpoint_files,
                "license_boundary": "Tencent_Hunyuan_noncommercial_or_community_terms_apply",
            },
            "parameters": {
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "octree_resolution": args.octree_resolution,
                "num_chunks": args.num_chunks,
            },
            "runtime": {
                "model_load_s": load_runtime_s,
                "generation_and_mesh_export_s": generation_runtime_s,
            },
            "mesh": {
                **stats,
                "glb_relpath": "mesh/raw_hunyuan3d_2mv.glb",
                "glb_sha256": sha256_file(glb_path),
                "ply_relpath": "mesh/raw_hunyuan3d_2mv.ply",
                "ply_sha256": sha256_file(ply_path),
                "metric_scale_status": "unknown_until_four_real_view_alignment",
                "coordinate_alignment_status": "unknown_until_four_real_view_alignment",
                "appearance_status": "untextured_shape_proposal",
            },
            "allowed_role": "learned_complete_mesh_prior_pending_real_view_alignment",
            "rejected_roles": [
                "observed_geometry",
                "metric_geometry_before_alignment",
                "collision_geometry_before_physics_simplification",
                "heldout_real_evidence",
                "formal_single_radeon_result",
            ],
            "review_status": "pending_visual_and_four_real_view_alignment",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hash_index(staging)
        done = {
            "schema_version": DONE_SCHEMA_VERSION,
            "stage": "MI300X_Hunyuan3D_2mv_complete_mesh_proposal",
            "status": "done_candidate_pending_alignment",
            "completed_utc": utc_now(),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "hashes_sha256": hashes_sha,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv")
    parser.add_argument(
        "--local-snapshot",
        type=Path,
        help=(
            "Explicit content-addressed Hugging Face snapshot directory. Use this "
            "for an offline rerun when cached weights are already present."
        ),
    )
    parser.add_argument("--seed", type=int, default=10027)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=380)
    parser.add_argument("--num-chunks", type=int, default=20000)
    parser.add_argument("--require-mi300x", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_mesh(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
