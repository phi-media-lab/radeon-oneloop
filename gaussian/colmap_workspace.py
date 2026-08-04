#!/usr/bin/env python3
"""Build and audit a COLMAP workspace from an exported HIL capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from typing import Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_text_model(model: Path) -> dict[str, int]:
    cameras = [
        line
        for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    image_lines = [
        line
        for line in (model / "images.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    registered = sum(
        Path(line.split()[-1]).suffix.lower() in IMAGE_SUFFIXES for line in image_lines
    )
    points = [
        line
        for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    return {"cameras": len(cameras), "registered_images": registered, "points3D": len(points)}


def _run(command: Sequence[str], log: Path) -> float:
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - started


def _write_hashes(workspace: Path, paths: Sequence[Path]) -> None:
    lines = []
    for path in sorted(set(item.resolve() for item in paths if item.is_file())):
        lines.append(f"{sha256_file(path)}  {path.relative_to(workspace).as_posix()}")
    (workspace / "colmap_hashes.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_colmap(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve()
    images = workspace / "images"
    capture_manifest = workspace / "manifest.json"
    if not images.is_dir() or not capture_manifest.is_file():
        raise FileNotFoundError("workspace must contain images/ and manifest.json")
    if not (workspace / "DONE").is_file():
        raise RuntimeError("capture workspace is missing its DONE marker")
    image_count = sum(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        for item in images.iterdir()
    )
    if image_count < 8:
        raise ValueError("COLMAP reconstruction requires at least eight images")

    database = workspace / "database.db"
    sparse = workspace / "sparse"
    logs = workspace / "colmap_logs"
    text_model = workspace / "sparse_txt"
    for path in (database, sparse, logs, text_model):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing COLMAP output: {path}")
    sparse.mkdir()
    logs.mkdir()
    text_model.mkdir()

    colmap = args.colmap
    timings = {}
    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        args.camera_model,
        "--SiftExtraction.use_gpu",
        "0",
        "--SiftExtraction.max_num_features",
        str(args.max_features),
        "--SiftExtraction.peak_threshold",
        str(args.peak_threshold),
    ]
    if args.camera_params:
        feature_command.extend(["--ImageReader.camera_params", args.camera_params])
    timings["feature_extractor_s"] = _run(
        feature_command, logs / "feature_extractor.log"
    )
    matcher_command = [
        colmap,
        f"{args.matcher}_matcher",
        "--database_path",
        str(database),
        "--SiftMatching.use_gpu",
        "0",
        "--SiftMatching.guided_matching",
        "1",
    ]
    if args.matcher == "sequential":
        matcher_command.extend(
            ["--SequentialMatching.overlap", str(args.sequential_overlap)]
        )
    timings["matcher_s"] = _run(matcher_command, logs / "matcher.log")
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.min_num_matches",
        str(args.min_num_matches),
        "--Mapper.init_min_num_inliers",
        str(args.init_min_num_inliers),
        "--Mapper.init_min_tri_angle",
        str(args.init_min_tri_angle),
        "--Mapper.abs_pose_min_num_inliers",
        str(args.abs_pose_min_num_inliers),
        "--Mapper.ba_refine_focal_length",
        "1" if args.refine_intrinsics else "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "1" if args.refine_intrinsics else "0",
    ]
    timings["mapper_s"] = _run(mapper_command, logs / "mapper.log")
    models = sorted(path for path in sparse.iterdir() if path.is_dir())
    if len(models) != 1:
        raise RuntimeError(f"expected one COLMAP model, found {len(models)}")
    model = models[0]
    convert_command = [
        colmap,
        "model_converter",
        "--input_path",
        str(model),
        "--output_path",
        str(text_model),
        "--output_type",
        "TXT",
    ]
    timings["model_converter_s"] = _run(
        convert_command, logs / "model_converter.log"
    )
    counts = parse_text_model(text_model)
    with sqlite3.connect(database) as connection:
        database_images = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        matched_pairs = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        )
    if counts["registered_images"] < args.min_registered_images:
        raise RuntimeError(
            f"only {counts['registered_images']} images registered; "
            f"minimum is {args.min_registered_images}"
        )
    report = {
        "schema_version": "radeon_oneloop.hil_colmap_workspace.v1",
        "formal": False,
        "camera_model": args.camera_model,
        "camera_params_initial": args.camera_params,
        "refine_intrinsics": args.refine_intrinsics,
        "matcher": args.matcher,
        "input_images": image_count,
        "database_images": database_images,
        "matched_pairs": matched_pairs,
        **counts,
        "registered_fraction": counts["registered_images"] / image_count,
        "timings_s": timings,
        "capture_manifest_sha256": sha256_file(capture_manifest),
        "model": model.relative_to(workspace).as_posix(),
    }
    metrics_path = workspace / "colmap_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hash_paths = [database, metrics_path]
    hash_paths.extend(path for path in logs.iterdir() if path.is_file())
    hash_paths.extend(path for path in model.iterdir() if path.is_file())
    hash_paths.extend(path for path in text_model.iterdir() if path.is_file())
    _write_hashes(workspace, hash_paths)
    (workspace / "COLMAP_DONE").touch()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--colmap", default=shutil.which("colmap") or "colmap")
    parser.add_argument(
        "--camera-model", choices=("PINHOLE", "SIMPLE_PINHOLE", "OPENCV"), default="OPENCV"
    )
    parser.add_argument(
        "--camera-params",
        help="COLMAP camera parameters, e.g. '500,320,240' for SIMPLE_PINHOLE.",
    )
    parser.add_argument("--refine-intrinsics", action="store_true")
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--sequential-overlap", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--peak-threshold", type=float, default=0.004)
    parser.add_argument("--min-num-matches", type=int, default=12)
    parser.add_argument("--init-min-num-inliers", type=int, default=30)
    parser.add_argument("--init-min-tri-angle", type=float, default=4.0)
    parser.add_argument("--abs-pose-min-num-inliers", type=int, default=15)
    parser.add_argument("--min-registered-images", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run_colmap(args)
    except Exception:
        args.workspace.mkdir(parents=True, exist_ok=True)
        (args.workspace / "COLMAP_FAILED").touch()
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
