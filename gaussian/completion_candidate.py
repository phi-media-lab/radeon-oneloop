#!/usr/bin/env python3
"""Build and validate an immutable conditional point-completion candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np


SCHEMA_VERSION = "radeon_oneloop.completion_candidate.v1"
LABEL_OBSERVED = 0
LABEL_GENERATED = 1
LABEL_REJECTED = 2
PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    header_bytes: int
    vertex_count: int
    format: str
    properties: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ply_header(path: Path) -> PlyHeader:
    lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"incomplete PLY header: {path}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"non-ASCII PLY header: {path}") from error
            lines.append(line)
            if line == "end_header":
                header_bytes = handle.tell()
                break
    if not lines or lines[0] != "ply":
        raise ValueError(f"not a PLY file: {path}")
    format_name: str | None = None
    vertex_count: int | None = None
    in_vertex = False
    properties: list[tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if parts[:1] == ["format"] and len(parts) >= 2:
            format_name = parts[1]
        elif parts[:2] == ["element", "vertex"] and len(parts) == 3:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError(f"unsupported PLY vertex property: {line}")
            properties.append((parts[2], parts[1]))
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {format_name}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY must contain at least one vertex")
    names = {name for name, _ in properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex properties must include x, y, and z")
    return PlyHeader(header_bytes, vertex_count, format_name, tuple(properties))


def _binary_vertices(path: Path, header: PlyHeader) -> np.memmap:
    if header.format != "binary_little_endian":
        raise ValueError("automatic confidence derivation requires binary little-endian PLY")
    fields = []
    for name, ply_type in header.properties:
        if ply_type not in PLY_TYPES:
            raise ValueError(f"unsupported PLY property type: {ply_type}")
        fields.append((name, PLY_TYPES[ply_type]))
    dtype = np.dtype(fields, align=False)
    minimum_size = header.header_bytes + header.vertex_count * dtype.itemsize
    if path.stat().st_size < minimum_size:
        raise ValueError(f"truncated PLY vertex payload: {path}")
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header.header_bytes,
        shape=(header.vertex_count,),
    )


def derive_confidence(
    completed_ply: Path,
    header: PlyHeader,
    explicit: Path | None,
    default_generated_confidence: float | None,
) -> tuple[np.ndarray, str]:
    if explicit is not None:
        confidence = np.load(explicit, allow_pickle=False)
        derivation = f"explicit_npy_sha256:{sha256_file(explicit)}"
    else:
        names = {name for name, _ in header.properties}
        preferred = next(
            (name for name in ("generation_confidence", "confidence") if name in names),
            None,
        )
        if preferred is not None:
            vertices = _binary_vertices(completed_ply, header)
            confidence = np.asarray(vertices[preferred], dtype=np.float32)
            derivation = f"ply_property:{preferred}"
        elif {"cross_view_source_count", "silhouette_support_count"}.issubset(names):
            vertices = _binary_vertices(completed_ply, header)
            cross = np.asarray(vertices["cross_view_source_count"], dtype=np.float32)
            silhouette = np.asarray(vertices["silhouette_support_count"], dtype=np.float32)
            confidence = np.clip((cross + silhouette) / 8.0, 0.0, 1.0)
            derivation = "evidence_support:(cross_view_source_count+silhouette_support_count)/8"
        elif default_generated_confidence is not None:
            confidence = np.full(
                header.vertex_count, default_generated_confidence, dtype=np.float32
            )
            derivation = f"explicit_constant:{default_generated_confidence:.9g}"
        else:
            raise ValueError(
                "completion has no confidence fields; pass --confidence-npy or "
                "--default-generated-confidence"
            )
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.shape != (header.vertex_count,):
        raise ValueError(
            f"confidence shape {confidence.shape} differs from vertex count {header.vertex_count}"
        )
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence values must be finite and within [0, 1]")
    return confidence, derivation


def derive_source_labels(
    header: PlyHeader,
    point_set_role: str,
    explicit: Path | None,
) -> tuple[np.ndarray, str]:
    if explicit is not None:
        labels = np.load(explicit, allow_pickle=False)
        derivation = f"explicit_npy_sha256:{sha256_file(explicit)}"
    elif point_set_role == "generated_fill_only":
        labels = np.full(header.vertex_count, LABEL_GENERATED, dtype=np.uint8)
        derivation = "contract_semantics:all_points_are_generated_fill"
    else:
        raise ValueError("full_candidate requires --source-labels-npy")
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (header.vertex_count,):
        raise ValueError(
            f"source-label shape {labels.shape} differs from vertex count {header.vertex_count}"
        )
    if not set(np.unique(labels).tolist()).issubset(
        {LABEL_OBSERVED, LABEL_GENERATED, LABEL_REJECTED}
    ):
        raise ValueError("source labels must use only 0=observed, 1=generated, 2=rejected")
    if point_set_role == "full_candidate" and not np.any(labels == LABEL_OBSERVED):
        raise ValueError("full_candidate must contain at least one observed point label")
    return labels, derivation


def _artifact(path: Path, relpath: str, **extra: Any) -> dict[str, Any]:
    return {
        "relpath": relpath,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def _load_json_object(path: Path | None, default: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return default
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return value.startswith("/") or value.startswith("~")
    return False


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_transform(object_height_m: float) -> dict[str, Any]:
    return {
        "schema_version": "radeon_oneloop.canonical_transform.v1",
        "canonical_from_generator_4x4": np.eye(4).tolist(),
        "unit": "m",
        "object_height_m": object_height_m,
        "status": "identity_input_already_in_canonical_frame",
    }


def _validate_transform(document: dict[str, Any], object_height_m: float) -> None:
    matrix = np.asarray(document.get("canonical_from_generator_4x4"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("canonical transform must contain a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("canonical transform must be affine")
    if abs(np.linalg.det(matrix[:3, :3])) < 1.0e-12:
        raise ValueError("canonical transform is singular")
    declared_height = float(document.get("object_height_m", object_height_m))
    if not np.isclose(declared_height, object_height_m, rtol=0.0, atol=1.0e-9):
        raise ValueError("canonical transform object height differs from metric anchor")


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    observed = args.observed_ply.resolve()
    completed = args.completed_ply.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    observed_header = parse_ply_header(observed)
    completed_header = parse_ply_header(completed)
    confidence, confidence_derivation = derive_confidence(
        completed,
        completed_header,
        args.confidence_npy.resolve() if args.confidence_npy else None,
        args.default_generated_confidence,
    )
    labels, labels_derivation = derive_source_labels(
        completed_header,
        args.point_set_role,
        args.source_labels_npy.resolve() if args.source_labels_npy else None,
    )
    conditioning = _load_json_object(
        args.conditioning_json.resolve() if args.conditioning_json else None,
        {
            "schema_version": "radeon_oneloop.completion_conditioning.v1",
            "modalities": ["observed_point_cloud"],
            "observed_core_sha256": sha256_file(observed),
            "raw_private_paths_embedded": False,
        },
    )
    if _contains_absolute_path(conditioning):
        raise ValueError("conditioning JSON must not embed absolute or home-relative paths")
    transform = _load_json_object(
        args.canonical_transform_json.resolve() if args.canonical_transform_json else None,
        _default_transform(args.object_height_m),
    )
    _validate_transform(transform, args.object_height_m)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        observed_target = staging / "observed_points.ply"
        completed_target = staging / "completed_points.ply"
        shutil.copy2(observed, observed_target)
        shutil.copy2(completed, completed_target)
        confidence_target = staging / "confidence.npy"
        labels_target = staging / "source_labels.npy"
        np.save(confidence_target, confidence, allow_pickle=False)
        np.save(labels_target, labels, allow_pickle=False)
        conditioning_target = staging / "conditioning.json"
        transform_target = staging / "canonical_transform.json"
        _write_json(conditioning_target, conditioning)
        _write_json(transform_target, transform)

        checkpoint_sha256 = None
        if args.checkpoint is not None:
            checkpoint_sha256 = sha256_file(args.checkpoint.resolve())
        generator: dict[str, Any] = {
            "name": args.generator_name,
            "version": args.generator_version,
            "host_role": args.host_role,
            "accelerator": args.accelerator,
            "conditional": True,
            "seed": args.seed,
        }
        if checkpoint_sha256 is not None:
            generator["checkpoint_sha256"] = checkpoint_sha256
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "generator": generator,
            "coordinate_convention": {
                "front_axis": "+Y",
                "up_axis": "+Z",
                "viewer_left_axis": "+X",
                "unit": "m",
                "origin": "plush_body_center",
            },
            "metric_anchor": {
                "dimension": "overall_height",
                "value_m": args.object_height_m,
            },
            "observed_core": _artifact(
                observed_target,
                "observed_points.ply",
                vertex_count=observed_header.vertex_count,
                ply_format=observed_header.format,
            ),
            "completion": _artifact(
                completed_target,
                "completed_points.ply",
                vertex_count=completed_header.vertex_count,
                ply_format=completed_header.format,
                point_set_role=args.point_set_role,
                observed_core_frozen=True,
            ),
            "sidecars": {
                "confidence": _artifact(
                    confidence_target,
                    "confidence.npy",
                    dtype=str(confidence.dtype),
                    shape=list(confidence.shape),
                    derivation=confidence_derivation,
                ),
                "source_labels": _artifact(
                    labels_target,
                    "source_labels.npy",
                    dtype=str(labels.dtype),
                    shape=list(labels.shape),
                    derivation=labels_derivation,
                ),
            },
            "conditioning": _artifact(conditioning_target, "conditioning.json"),
            "canonical_transform": _artifact(
                transform_target, "canonical_transform.json"
            ),
            "source_label_values": {
                "observed": LABEL_OBSERVED,
                "generated": LABEL_GENERATED,
                "rejected": LABEL_REJECTED,
            },
        }
        manifest_target = staging / "manifest.json"
        _write_json(manifest_target, manifest)
        validate_candidate(staging, require_done=False)
        hash_lines = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
                hash_lines.append(f"{sha256_file(path)}  {path.name}")
        (staging / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "manifest_sha256": sha256_file(manifest_target),
            "hashes_sha256": sha256_file(staging / "hashes.sha256"),
        }
        _write_json(staging / "DONE", done)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_candidate(output, require_done=True)


def _schema_path() -> Path:
    return Path(__file__).with_name("completion_candidate.schema.json")


def _checked_artifact(root: Path, item: dict[str, Any]) -> Path:
    path = (root / item["relpath"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes candidate root: {item['relpath']}")
    if not path.is_file():
        raise ValueError(f"artifact is missing: {item['relpath']}")
    if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
        raise ValueError(f"artifact size or hash mismatch: {item['relpath']}")
    return path


def validate_candidate(root: Path, *, require_done: bool = True) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        manifest
    )
    observed_path = _checked_artifact(root, manifest["observed_core"])
    completed_path = _checked_artifact(root, manifest["completion"])
    confidence_path = _checked_artifact(root, manifest["sidecars"]["confidence"])
    labels_path = _checked_artifact(root, manifest["sidecars"]["source_labels"])
    conditioning_path = _checked_artifact(root, manifest["conditioning"])
    transform_path = _checked_artifact(root, manifest["canonical_transform"])
    observed_header = parse_ply_header(observed_path)
    completed_header = parse_ply_header(completed_path)
    if observed_header.vertex_count != manifest["observed_core"]["vertex_count"]:
        raise ValueError("observed-core vertex count differs from manifest")
    if completed_header.vertex_count != manifest["completion"]["vertex_count"]:
        raise ValueError("completion vertex count differs from manifest")
    confidence = np.load(confidence_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    expected_shape = (completed_header.vertex_count,)
    if confidence.shape != expected_shape or labels.shape != expected_shape:
        raise ValueError("candidate sidecar shape differs from completion vertex count")
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("candidate confidence is outside [0, 1]")
    if not set(np.unique(labels).tolist()).issubset(
        {LABEL_OBSERVED, LABEL_GENERATED, LABEL_REJECTED}
    ):
        raise ValueError("candidate contains an unknown source label")
    conditioning = json.loads(conditioning_path.read_text(encoding="utf-8"))
    if _contains_absolute_path(conditioning):
        raise ValueError("conditioning JSON embeds a non-portable path")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    _validate_transform(transform, manifest["metric_anchor"]["value_m"])
    if require_done:
        done_path = root / "DONE"
        if not done_path.is_file():
            raise ValueError("candidate has no atomic DONE marker")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "complete":
            raise ValueError("candidate DONE marker is not complete")
        if done.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("candidate DONE marker manifest hash mismatch")
        hashes_path = root / "hashes.sha256"
        if done.get("hashes_sha256") != sha256_file(hashes_path):
            raise ValueError("candidate DONE marker hash-index mismatch")
        for line in hashes_path.read_text(encoding="utf-8").splitlines():
            digest, relpath = line.split("  ", 1)
            indexed = (root / relpath).resolve()
            if not indexed.is_relative_to(root) or sha256_file(indexed) != digest:
                raise ValueError(f"candidate hash index mismatch: {relpath}")
    counts = {
        "observed": int(np.count_nonzero(labels == LABEL_OBSERVED)),
        "generated": int(np.count_nonzero(labels == LABEL_GENERATED)),
        "rejected": int(np.count_nonzero(labels == LABEL_REJECTED)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "observed_core_vertices": observed_header.vertex_count,
        "completion_vertices": completed_header.vertex_count,
        "source_label_counts": counts,
        "confidence_min": float(np.min(confidence)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_max": float(np.max(confidence)),
        "formal": False,
        "eligible_for_heldout_real_metrics": False,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--observed-ply", type=Path, required=True)
    build.add_argument("--completed-ply", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--point-set-role",
        choices=("generated_fill_only", "full_candidate"),
        default="generated_fill_only",
    )
    build.add_argument("--confidence-npy", type=Path)
    build.add_argument("--source-labels-npy", type=Path)
    build.add_argument("--default-generated-confidence", type=float)
    build.add_argument("--conditioning-json", type=Path)
    build.add_argument("--canonical-transform-json", type=Path)
    build.add_argument("--checkpoint", type=Path)
    build.add_argument("--generator-name", required=True)
    build.add_argument("--generator-version", required=True)
    build.add_argument("--host-role", default="phi-amd-work")
    build.add_argument("--accelerator", default="AMD Instinct MI300X")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--object-height-m", type=float, default=0.095)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_candidate(args)
    else:
        result = validate_candidate(args.candidate, require_done=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
