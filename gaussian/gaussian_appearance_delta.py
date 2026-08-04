#!/usr/bin/env python3
"""Create or apply a compact appearance-only delta between binary Gaussian PLYs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


TYPE_MAP = {
    "float": "<f4",
    "float32": "<f4",
    "uchar": "u1",
    "uint8": "u1",
}
APPEARANCE_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity")


class AppearanceDeltaError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vertex_layout(path: Path) -> tuple[int, int, np.dtype]:
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise AppearanceDeltaError(f"PLY header is incomplete: {path}")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise AppearanceDeltaError(f"PLY header is not ASCII: {path}") from error
            header.append(decoded)
            if decoded == "end_header":
                break
        header_bytes = handle.tell()
    if "format binary_little_endian 1.0" not in header:
        raise AppearanceDeltaError("appearance deltas require binary little-endian PLY")
    vertex_count = None
    in_vertex = False
    fields = []
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if len(parts) != 3 or parts[1] not in TYPE_MAP:
                raise AppearanceDeltaError(f"unsupported vertex property: {line}")
            fields.append((parts[2], TYPE_MAP[parts[1]]))
    if vertex_count is None or not fields:
        raise AppearanceDeltaError("PLY vertex layout is absent")
    dtype = np.dtype(fields, align=False)
    expected_size = header_bytes + vertex_count * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise AppearanceDeltaError(
            f"PLY has unsupported trailing elements or size mismatch: {path.stat().st_size} != {expected_size}"
        )
    missing = set(APPEARANCE_FIELDS) - set(dtype.names or ())
    if missing:
        raise AppearanceDeltaError(f"PLY lacks appearance fields: {sorted(missing)}")
    return header_bytes, vertex_count, dtype


def _vertices(path: Path, *, mode: str = "r") -> tuple[np.memmap, int, np.dtype]:
    header_bytes, count, dtype = parse_vertex_layout(path)
    return np.memmap(path, dtype=dtype, mode=mode, offset=header_bytes, shape=(count,)), header_bytes, dtype


def create_delta(base: Path, target: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    base_vertices, base_header, base_dtype = _vertices(base)
    target_vertices, target_header, target_dtype = _vertices(target)
    if base_header != target_header or base_dtype != target_dtype or len(base_vertices) != len(target_vertices):
        raise AppearanceDeltaError("base and target PLY layouts differ")
    immutable_fields = [name for name in base_dtype.names or () if name not in APPEARANCE_FIELDS]
    for name in immutable_fields:
        if not np.array_equal(base_vertices[name], target_vertices[name]):
            raise AppearanceDeltaError(f"geometry/provenance field changed: {name}")
    metadata = {
        "schema_version": "radeon_oneloop.gaussian_appearance_delta.v1",
        "base_ply_sha256": sha256_file(base),
        "target_ply_sha256": sha256_file(target),
        "vertex_count": len(base_vertices),
        "changed_fields": list(APPEARANCE_FIELDS),
    }
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{name: np.asarray(target_vertices[name]) for name in APPEARANCE_FIELDS},
    )
    metadata["delta_sha256"] = sha256_file(output)
    metadata["delta_bytes"] = output.stat().st_size
    return metadata


def apply_delta(base: Path, delta: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    with np.load(delta, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if sha256_file(base) != metadata["base_ply_sha256"]:
            raise AppearanceDeltaError("base PLY hash does not match appearance delta")
        shutil.copyfile(base, output)
        vertices, _, _ = _vertices(output, mode="r+")
        if len(vertices) != int(metadata["vertex_count"]):
            raise AppearanceDeltaError("vertex count does not match appearance delta")
        for name in APPEARANCE_FIELDS:
            values = archive[name]
            if values.shape != vertices[name].shape:
                raise AppearanceDeltaError(f"delta field shape differs: {name}")
            vertices[name] = values
        vertices.flush()
        del vertices
    actual = sha256_file(output)
    if actual != metadata["target_ply_sha256"]:
        raise AppearanceDeltaError(f"reconstructed target hash mismatch: {actual}")
    return {
        "schema_version": metadata["schema_version"],
        "base_ply_sha256": metadata["base_ply_sha256"],
        "delta_sha256": sha256_file(delta),
        "output_ply_sha256": actual,
        "vertex_count": metadata["vertex_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--base", type=Path, required=True)
    create.add_argument("--target", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--base", type=Path, required=True)
    apply.add_argument("--delta", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_delta(args.base.resolve(), args.target.resolve(), args.output.resolve())
    else:
        result = apply_delta(args.base.resolve(), args.delta.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
