#!/usr/bin/env python3
"""Run an external UniSHARP inference script while retaining lossless pseudo-views."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_w2c_from_eye(eye: list[float]) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, -eye[0]],
        [0.0, 1.0, 0.0, -eye[1]],
        [0.0, 0.0, 1.0, -eye[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_c2w_from_eye(eye: list[float]) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, eye[0]],
        [0.0, 1.0, 0.0, eye[1]],
        [0.0, 0.0, 1.0, eye[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_pseudoview_document(metadata: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
    if metadata.get("camera_kind") != "perspective":
        raise ValueError("lossless pseudo-view export currently supports perspective runs only")
    entry = metadata.get("camera_json_entry") or {}
    fx, fy, cx, cy = map(float, entry["intrinsics"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    crop_fraction = float(metadata.get("output_crop_border_fraction", 0.0))
    crop_x = int(round(width * crop_fraction))
    crop_y = int(round(height * crop_fraction))
    output_width = width - 2 * crop_x
    output_height = height - 2 * crop_y
    intrinsic = [
        [fx, 0.0, cx - crop_x],
        [0.0, fy, cy - crop_y],
        [0.0, 0.0, 1.0],
    ]

    views = []
    for kind in ("forward", "rotate"):
        frame_dir = sample_dir / f"{kind}_frames"
        paths = sorted(frame_dir.glob("frame_*.png"))
        if not paths:
            raise ValueError(f"lossless {kind} frames are missing: {frame_dir}")
        count = len(paths)
        for index, path in enumerate(paths):
            if kind == "forward":
                alpha = float(index + 1) / float(count)
                eye = [0.0, 0.0, float(metadata["forward_distance_m"]) * alpha]
            else:
                theta = -2.0 * math.pi * float(index) / float(count)
                radius = float(metadata["rotate_radius_m"])
                eye = [radius * math.sin(theta), radius * math.cos(theta), 0.0]
            views.append(
                {
                    "id": f"{kind}_{index:03d}",
                    "motion_kind": kind,
                    "relpath": str(path.relative_to(sample_dir)),
                    "sha256": sha256_file(path),
                    "image_size_wh": [output_width, output_height],
                    "intrinsic_3x3": intrinsic,
                    "camera_to_generator_world_4x4": _matrix_c2w_from_eye(eye),
                    "generator_world_to_camera_4x4": _matrix_w2c_from_eye(eye),
                }
            )
    return {
        "schema_version": "radeon_oneloop.unisharp_local_pseudoviews.v1",
        "formal": False,
        "coordinate_frame": "per_source_UniSHARP_generator_camera_frame",
        "source_camera_view_id": entry.get("source_camera_view_id"),
        "source_image_sha256": entry.get("source_image_sha256"),
        "crop_border_xy_px": [crop_x, crop_y],
        "eligible_for_observed_core": False,
        "eligible_for_heldout_real_metrics": False,
        "metric_geometry_status": "requires_independent_alignment_gate",
        "views": views,
    }


def _load_inference_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("radeon_oneloop_external_unisharp_infer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import UniSHARP inference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-script", type=Path, required=True)
    parser.add_argument("inference_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    inference_args = list(args.inference_args)
    if inference_args and inference_args[0] == "--":
        inference_args.pop(0)
    passthrough = argparse.ArgumentParser(add_help=False)
    passthrough.add_argument("--out-dir", type=Path, required=True)
    known, _ = passthrough.parse_known_args(inference_args)

    module = _load_inference_module(args.infer_script.resolve())
    original_save_gif = module._save_gif

    def save_gif_and_frames(frames: list[Any], out_file: Path, duration_ms: int) -> None:
        original_save_gif(frames, out_file, duration_ms)
        frame_dir = out_file.with_name(f"{out_file.stem}_frames")
        frame_dir.mkdir(parents=True, exist_ok=False)
        for index, frame in enumerate(frames):
            module.Image.fromarray(frame).save(frame_dir / f"frame_{index:03d}.png")

    module._save_gif = save_gif_and_frames
    previous_argv = sys.argv
    try:
        sys.argv = [str(args.infer_script), *inference_args]
        module.main()
    finally:
        sys.argv = previous_argv

    for metadata_path in sorted(known.out_dir.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        document = build_pseudoview_document(metadata, metadata_path.parent)
        (metadata_path.parent / "pseudo_view_cameras.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
