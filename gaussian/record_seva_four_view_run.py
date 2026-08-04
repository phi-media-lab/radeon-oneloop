#!/usr/bin/env python3
"""Validate and bind a Stable Virtual Camera four-view orbit run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from gaussian.prepare_four_view_generation import (
    sha256_file,
    validate_generation_input,
)


SCHEMA_VERSION = "radeon_oneloop.seva_four_view_orbit.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.seva_four_view_orbit_done.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def camera_error(input_transforms: dict[str, Any], output_transforms: dict[str, Any]) -> float:
    source_frames = input_transforms.get("frames")
    generated_frames = output_transforms.get("frames")
    if not isinstance(source_frames, list) or len(source_frames) != 53:
        raise ValueError("SEVA input transforms must contain 4 inputs and 49 targets")
    if not isinstance(generated_frames, list) or len(generated_frames) != 53:
        raise ValueError("SEVA output transforms must contain 4 inputs and 49 targets")
    source = np.asarray(
        [frame["transform_matrix"] for frame in source_frames[4:]], dtype=np.float64
    )
    generated = np.asarray(
        [frame["transform_matrix"] for frame in generated_frames[4:]], dtype=np.float64
    )
    if source.shape != (49, 4, 4) or generated.shape != (49, 4, 4):
        raise ValueError("SEVA target camera matrix shape is invalid")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(generated)):
        raise ValueError("SEVA camera matrices contain non-finite values")
    return float(np.max(np.abs(source - generated)))


def _video_probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    value = json.loads(subprocess.check_output(command, text=True))
    return value["streams"][0]


def record_run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import imageio.v3 as iio
        import torch
    except ImportError as exc:  # pragma: no cover - remote SEVA environment
        raise RuntimeError("SEVA audit dependencies are missing") from exc
    run_dir = args.run_dir.resolve()
    input_root = args.input_root.resolve()
    seva_root = args.seva_root.resolve()
    local_model_root = args.local_model_root.resolve()
    input_manifest = validate_generation_input(input_root)
    if (run_dir / "DONE").exists() or (run_dir / "hashes.sha256").exists():
        raise FileExistsError("SEVA run is already finalized")
    inference = run_dir / "inference"
    frames_dir = inference / "samples-rgb"
    video = inference / "samples-rgb.mp4"
    output_transforms_path = inference / "transforms.json"
    frame_paths = sorted(frames_dir.glob("*.png"))
    if len(frame_paths) != 49:
        raise ValueError(f"SEVA output requires 49 generated frames, got {len(frame_paths)}")
    shapes = {tuple(iio.imread(path).shape) for path in frame_paths}
    if shapes != {(576, 576, 3)}:
        raise ValueError(f"SEVA frame shapes differ from 576p contract: {shapes}")
    probe = _video_probe(video)
    if int(probe["nb_read_frames"]) != 49:
        raise ValueError("SEVA video does not contain exactly 49 frames")
    if [int(probe["width"]), int(probe["height"])] != [576, 576]:
        raise ValueError("SEVA video resolution differs from the bound contract")

    input_scene = input_root / input_manifest["generator_contracts"]["seva"]["scene_relpath"]
    input_transforms = json.loads((input_scene / "transforms.json").read_text())
    output_transforms = json.loads(output_transforms_path.read_text())
    maximum_camera_error = camera_error(input_transforms, output_transforms)
    if maximum_camera_error > 1e-6:
        raise ValueError(f"SEVA output camera drift exceeds tolerance: {maximum_camera_error}")

    model = input_manifest["generator_contracts"]["seva"]["model"]
    version = input_manifest["generator_contracts"]["seva"]["version"]
    if not isinstance(args.revision, str) or len(args.revision) != 40 or any(
        value not in "0123456789abcdef" for value in args.revision
    ):
        raise ValueError("SEVA model revision must be an exact 40-character commit")
    resolved_revision = args.revision
    weight_path = local_model_root / f"modelv{version}.safetensors"
    config_path = local_model_root / "config.yaml"
    if not weight_path.is_file() or not config_path.is_file():
        raise ValueError("fixed-revision local SEVA model files are missing")
    commit = subprocess.check_output(
        ["git", "-C", str(seva_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(seva_root), "status", "--porcelain=v1"], text=True
    )

    generated_frames = [
        {
            "frame_index": index,
            "relpath": f"inference/samples-rgb/{path.name}",
            "sha256": sha256_file(path),
            "provenance": "generated",
            "eligible_for_observed_metrics": False,
        }
        for index, path in enumerate(frame_paths)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "formal": False,
        "eligible_for_formal_metrics": False,
        "eligible_for_heldout_real_metrics": False,
        "asset_name": input_manifest["asset_name"],
        "host_role": "phi_amd_work_mi300x_nonformal_generation_lab",
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "input": {
            "four_view_manifest_sha256": sha256_file(input_root / "manifest.json"),
            "four_view_hashes_sha256": sha256_file(input_root / "hashes.sha256"),
            "observed_views": 4,
            "inherited_geometry": None,
            "source_policy": "four_reviewed_same_instance_photos_only",
        },
        "model": {
            "repo_id": model,
            "version": version,
            "revision": resolved_revision,
            "weight_filename": weight_path.name,
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": sha256_file(weight_path),
            "config_sha256": sha256_file(config_path),
            "resolution": "fixed_revision_authorized_download_then_local_load",
            "seva_commit": commit,
            "seva_worktree_dirty": bool(git_status),
            "license_boundary": "Stable_Virtual_Camera_noncommercial_output_terms_apply",
        },
        "parameters": {
            "seed": args.seed,
            "task": "img2trajvid",
            "num_inputs": 4,
            "cfg": [3.0, 2.0],
            "use_traj_prior": True,
            "chunk_strategy": "interp-gt",
            "target_frames": 49,
        },
        "runtime_s": args.runtime_s,
        "camera_contract": {
            "target_matrices": 49,
            "maximum_absolute_matrix_error": maximum_camera_error,
            "tolerance": 1e-6,
            "status": "exact",
        },
        "video": {
            "relpath": "inference/samples-rgb.mp4",
            "sha256": sha256_file(video),
            "probe": probe,
        },
        "frames": generated_frames,
        "allowed_role": "camera_controlled_generated_orbit_video_prior_pending_visual_audit",
        "rejected_roles": [
            "observed_video",
            "metric_geometry",
            "heldout_real_evidence",
            "formal_single_radeon_result",
        ],
        "review_status": "pending_identity_temporal_and_loop_audit",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}")
    hashes_path = run_dir / "hashes.sha256"
    hashes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    done = {
        "schema_version": DONE_SCHEMA_VERSION,
        "stage": "MI300X_SEVA_four_view_orbit_generation",
        "status": "done_candidate_pending_visual_audit",
        "completed_utc": utc_now(),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "hashes_sha256": sha256_file(hashes_path),
    }
    (run_dir / "DONE").write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seva-root", type=Path, required=True)
    parser.add_argument("--local-model-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=10027)
    parser.add_argument("--runtime-s", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = record_run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
