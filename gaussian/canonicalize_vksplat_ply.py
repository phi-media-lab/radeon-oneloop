#!/usr/bin/env python3
"""Undo VkSplat's dataparser similarity and write a canonical observed-core PLY."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from gaussian.gaussian_appearance_delta import parse_vertex_layout, sha256_file
from gaussian.sharp_object_fusion import quaternion_from_rotation, quaternion_multiply


def inverse_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if transform.shape != (4, 4):
        raise ValueError("dataparser transform must be 4x4")
    inverse = np.linalg.inv(transform)
    linear = inverse[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("dataparser inverse must be a proper similarity")
    scale = determinant ** (1.0 / 3.0)
    rotation = linear / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-5):
        raise ValueError("dataparser inverse is not a uniform-scale rotation")
    return scale, rotation, inverse[:3, 3]


def canonicalize(args: argparse.Namespace) -> dict[str, object]:
    source = args.ply.resolve()
    output = args.output.resolve()
    provenance_path = args.output_provenance.resolve()
    if output.exists() or provenance_path.exists():
        raise FileExistsError(output if output.exists() else provenance_path)
    train_json_path = args.train_json.resolve()
    train = json.loads(train_json_path.read_text(encoding="utf-8"))
    transform = np.asarray(train["dataparser_transform"], dtype=np.float64)
    scale, rotation, translation = inverse_similarity(transform)

    training_lineage = None
    if args.training_run_manifest is not None or args.dataset_manifest is not None:
        if args.training_run_manifest is None or args.dataset_manifest is None:
            raise ValueError(
                "training-run-manifest and dataset-manifest must be supplied together"
            )
        training_path = args.training_run_manifest.resolve()
        dataset_path = args.dataset_manifest.resolve()
        training_manifest = json.loads(training_path.read_text(encoding="utf-8"))
        dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset_manifest_sha = sha256_file(dataset_path)
        if training_manifest.get("dataset_manifest_sha256") != dataset_manifest_sha:
            raise ValueError("training run does not bind the supplied dataset manifest")
        if dataset_manifest.get("provenance", {}).get("secondary_accelerator_artifacts") is not False:
            raise ValueError("dataset lineage does not exclude secondary-accelerator artifacts")
        training_lineage = {
            "training_run_manifest_sha256": sha256_file(training_path),
            "training_run_id": training_manifest.get("run_id"),
            "training_formal": training_manifest.get("formal"),
            "training_host_role": training_manifest.get("host_role"),
            "vksplat_commit": training_manifest.get("vksplat_commit"),
            "dataset_manifest_sha256": dataset_manifest_sha,
            "dataset_hash": training_manifest.get("dataset_hash"),
            "dataset_formal_input_eligible": dataset_manifest.get("formal_input_eligible"),
            "secondary_accelerator_artifacts": False,
        }

    shutil.copyfile(source, output)
    offset, count, dtype = parse_vertex_layout(output)
    vertices = np.memmap(output, dtype=dtype, mode="r+", offset=offset, shape=(count,))
    positions = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    transformed = scale * (positions @ rotation.T) + translation
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = transformed[:, index].astype(np.float32)
    for name in ("scale_0", "scale_1", "scale_2"):
        vertices[name] = (np.asarray(vertices[name], dtype=np.float64) + math.log(scale)).astype(np.float32)
    quaternions = np.stack([vertices[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float64)
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12)
    rotated = quaternion_multiply(quaternion_from_rotation(rotation)[None, :], quaternions)
    rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1.0e-12)
    for index in range(4):
        vertices[f"rot_{index}"] = rotated[:, index].astype(np.float32)
    vertices.flush()
    del vertices

    provenance = {
        "schema_version": "radeon_oneloop.observed_core_canonicalization.v1",
        "formal": False,
        "provenance_class": "observed_core_candidate",
        "observed_only_training": True,
        "input_ply_sha256": sha256_file(source),
        "train_json_sha256": sha256_file(train_json_path),
        "dataparser_transform_original_to_normalized": transform.tolist(),
        "inverse_similarity_normalized_to_canonical": {
            "scale": scale,
            "rotation_3x3": rotation.tolist(),
            "translation_m": translation.tolist(),
        },
        "output_ply_sha256": sha256_file(output),
        "gaussian_count": count,
        "training_lineage": training_lineage,
        "eligible_for_heldout_real_metrics": False,
        "eligible_for_formal_metrics": False,
    }
    temporary = provenance_path.with_name(f".{provenance_path.name}.tmp")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, provenance_path)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    result = canonicalize(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
