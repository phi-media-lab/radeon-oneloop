#!/usr/bin/env python3
"""Export an audited surface carrier as a portable vertex-colored GLB.

Genesis 1.3.1 does not accept PLY files directly.  This exporter deliberately
implements only the small, explicit PLY subset emitted by ``surface_carrier``
so conversion does not introduce an unpinned geometry dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np


SCHEMA_VERSION = "radeon_oneloop.surface_carrier_glb.v2"
DONE_SCHEMA_VERSION = "radeon_oneloop.surface_carrier_glb_done.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_carrier_root(root: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    hashes_path = root / "hashes.sha256"
    for path in (manifest_path, done_path, hashes_path):
        if not path.is_file():
            raise FileNotFoundError(f"surface carrier is incomplete: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.surface_carrier.v1":
        raise ValueError("unsupported surface-carrier schema")
    if manifest.get("formal") is not False or manifest.get("accepted_numeric") is not True:
        raise ValueError("surface carrier must be an accepted formal=false candidate")
    if manifest.get("visual_review_required") is not True:
        raise ValueError("surface carrier must preserve the visual-review requirement")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("surface-carrier DONE marker does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise ValueError("surface-carrier DONE marker does not bind its hash index")

    source_hashes: dict[str, str] = {}
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        expected, relpath = line.split("  ", maxsplit=1)
        source = root / relpath
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"surface-carrier hash mismatch: {relpath}")
        source_hashes[relpath] = expected
    ply_relpath = str(manifest["geometry"]["ply_relpath"])
    ply_path = root / ply_relpath
    if source_hashes.get(ply_relpath) != manifest["geometry"]["ply_sha256"]:
        raise ValueError("surface-carrier PLY is not consistently hash-bound")
    return manifest, ply_path, source_hashes


def read_ascii_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact triangular, vertex-colored ASCII PLY subset we emit."""

    with path.open("r", encoding="ascii") as handle:
        if handle.readline().rstrip("\n") != "ply":
            raise ValueError("not a PLY file")
        if handle.readline().strip() != "format ascii 1.0":
            raise ValueError("only ASCII PLY 1.0 is supported")
        vertex_count = None
        face_count = None
        vertex_properties: list[str] = []
        current_element = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "end_header":
                break
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                current_element = "vertex"
            elif fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
                current_element = "face"
            elif fields[0] == "property" and current_element == "vertex":
                if len(fields) != 3:
                    raise ValueError("list-valued vertex properties are unsupported")
                vertex_properties.append(fields[2])
        if vertex_count is None or face_count is None:
            raise ValueError("PLY must declare vertex and face counts")
        required = ("x", "y", "z", "red", "green", "blue")
        missing = [name for name in required if name not in vertex_properties]
        if missing:
            raise ValueError(f"PLY is missing vertex properties: {missing}")
        indices = {name: vertex_properties.index(name) for name in required}
        vertices = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 4), dtype=np.uint8)
        colors[:, 3] = 255
        for row in range(vertex_count):
            fields = handle.readline().split()
            if len(fields) != len(vertex_properties):
                raise ValueError(f"invalid vertex row {row}")
            vertices[row] = [float(fields[indices[axis]]) for axis in ("x", "y", "z")]
            rgb = [int(fields[indices[channel]]) for channel in ("red", "green", "blue")]
            if any(value < 0 or value > 255 for value in rgb):
                raise ValueError(f"invalid vertex color at row {row}")
            colors[row, :3] = rgb
        faces = np.empty((face_count, 3), dtype=np.uint32)
        for row in range(face_count):
            fields = handle.readline().split()
            if len(fields) != 4 or fields[0] != "3":
                raise ValueError("only triangular PLY faces are supported")
            face = [int(value) for value in fields[1:]]
            if any(value < 0 or value >= vertex_count for value in face):
                raise ValueError(f"face index out of range at row {row}")
            faces[row] = face
    if not np.isfinite(vertices).all():
        raise ValueError("PLY contains non-finite positions")
    return vertices, colors, faces


def _pad4(payload: bytes, fill: bytes) -> bytes:
    return payload + fill * ((-len(payload)) % 4)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xFFFFFFFF
    )


def encode_rgb_png(image: np.ndarray) -> bytes:
    """Encode an RGB uint8 image without an optional imaging dependency."""

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("PNG input must be a non-empty HxWx3 array")
    height, width = image.shape[:2]
    scanlines = b"".join(b"\0" + row.tobytes() for row in image)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def build_triangle_texture_atlas(
    vertices: np.ndarray,
    colors_rgba: np.ndarray,
    faces: np.ndarray,
    *,
    cell_size: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate face vertices and bind each triangle to a padded atlas cell."""

    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors_rgba, dtype=np.uint8)
    faces = np.asarray(faces, dtype=np.uint32)
    if cell_size < 5:
        raise ValueError("atlas cells must be at least 5 pixels wide")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a non-empty Nx3 array")
    if colors.shape != (len(vertices), 4):
        raise ValueError("colors must be an Nx4 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty Mx3 array")
    if int(faces.max()) >= len(vertices):
        raise ValueError("face index is out of range")

    columns = int(np.ceil(np.sqrt(len(faces))))
    rows = int(np.ceil(len(faces) / columns))
    width = columns * cell_size
    height = rows * cell_size
    atlas = np.full((height, width, 3), 255, dtype=np.uint8)
    face_vertices = vertices[faces].reshape(-1, 3)
    face_colors = colors[faces, :3]
    vertex_normals = smooth_vertex_normals(vertices, faces)
    face_normals = vertex_normals[faces].reshape(-1, 3).astype(np.float32)
    uv = np.empty((len(faces), 3, 2), dtype=np.float32)

    low = 1.0
    high = float(cell_size - 2)
    local_uv_px = np.asarray(((low, low), (high, low), (low, high)), dtype=np.float64)
    grid_y, grid_x = np.mgrid[0:cell_size, 0:cell_size].astype(np.float64)
    weight_1 = (grid_x - low) / (high - low)
    weight_2 = (grid_y - low) / (high - low)
    weights = np.stack((1.0 - weight_1 - weight_2, weight_1, weight_2), axis=-1)
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1.0e-12)

    for face_index, triangle_colors in enumerate(face_colors):
        cell_x = (face_index % columns) * cell_size
        cell_y = (face_index // columns) * cell_size
        patch = np.rint(weights @ triangle_colors.astype(np.float64)).astype(np.uint8)
        atlas[cell_y : cell_y + cell_size, cell_x : cell_x + cell_size] = patch
        px = local_uv_px + np.asarray((cell_x + 0.5, cell_y + 0.5))
        uv[face_index, :, 0] = px[:, 0] / width
        # glTF UVs start at the bottom-left while PNG rows start at the top.
        uv[face_index, :, 1] = 1.0 - px[:, 1] / height

    indices = np.arange(len(face_vertices), dtype=np.uint32).reshape(-1, 3)
    return face_vertices, face_normals, uv.reshape(-1, 2), indices, atlas


def smooth_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    triangle_vertices = vertices[faces]
    triangle_normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], triangle_normals)
    normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    if np.any(normal_lengths <= 1.0e-12):
        raise ValueError("carrier contains a vertex without a stable surface normal")
    return (vertex_normals / normal_lengths).astype(np.float32)


def encode_quantized_material_glb(
    vertices: np.ndarray,
    colors_rgba: np.ndarray,
    faces: np.ndarray,
    *,
    levels: int = 6,
) -> tuple[bytes, dict[str, int]]:
    """Encode smooth indexed geometry using bounded, Genesis-safe materials."""

    vertices = np.asarray(vertices, dtype="<f4")
    colors = np.asarray(colors_rgba, dtype=np.uint8)
    faces = np.asarray(faces, dtype="<u4")
    if levels < 2 or levels > 8:
        raise ValueError("material quantization levels must be between 2 and 8")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must be a non-empty Nx3 array")
    if colors.shape != (len(vertices), 4):
        raise ValueError("colors must be an Nx4 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty Mx3 array")
    if int(faces.max()) >= len(vertices):
        raise ValueError("face index is out of range")

    normals = np.asarray(smooth_vertex_normals(vertices, faces), dtype="<f4")
    mean_rgb = colors[faces, :3].astype(np.float64).mean(axis=1)
    bins = np.rint(mean_rgb * (levels - 1) / 255.0).astype(np.uint8)
    codes = bins[:, 0].astype(np.int32) * levels * levels
    codes += bins[:, 1].astype(np.int32) * levels
    codes += bins[:, 2].astype(np.int32)
    occupied = np.unique(codes)

    position_bytes = vertices.tobytes(order="C")
    normal_bytes = normals.tobytes(order="C")
    binary_parts = [position_bytes, normal_bytes]
    buffer_views: list[dict[str, int]] = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
        {
            "buffer": 0,
            "byteOffset": len(position_bytes),
            "byteLength": len(normal_bytes),
            "target": 34962,
        },
    ]
    accessors: list[dict[str, Any]] = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(vertices),
            "type": "VEC3",
            "min": vertices.min(axis=0).astype(float).tolist(),
            "max": vertices.max(axis=0).astype(float).tolist(),
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3",
        },
    ]
    materials = []
    primitives = []
    offset = len(position_bytes) + len(normal_bytes)
    for material_index, code in enumerate(occupied.tolist()):
        face_indices = faces[codes == code].reshape(-1)
        index_bytes = face_indices.astype("<u4", copy=False).tobytes(order="C")
        view_index = len(buffer_views)
        accessor_index = len(accessors)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(index_bytes),
                "target": 34963,
            }
        )
        accessors.append(
            {
                "bufferView": view_index,
                "componentType": 5125,
                "count": int(face_indices.size),
                "type": "SCALAR",
                "min": [int(face_indices.min())],
                "max": [int(face_indices.max())],
            }
        )
        binary_parts.append(index_bytes)
        offset += len(index_bytes)
        r_bin = code // (levels * levels)
        g_bin = (code // levels) % levels
        b_bin = code % levels
        rgb = [channel / (levels - 1) for channel in (r_bin, g_bin, b_bin)]
        materials.append(
            {
                "name": f"carrier_rgb_{r_bin}_{g_bin}_{b_bin}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": rgb + [1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.82,
                },
            }
        )
        primitives.append(
            {
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": accessor_index,
                "material": material_index,
                "mode": 4,
            }
        )

    binary = _pad4(b"".join(binary_parts), b"\0")
    document = {
        "asset": {"version": "2.0", "generator": "radeon-oneloop surface carrier exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "complete_surface_carrier"}],
        "meshes": [{"name": "complete_surface_carrier", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN\0"),
            binary,
        )
    )
    return glb, {
        "render_vertices": int(len(vertices)),
        "material_buckets": int(len(occupied)),
        "quantization_levels_per_channel": levels,
    }


def encode_textured_glb(
    vertices: np.ndarray, colors_rgba: np.ndarray, faces: np.ndarray
) -> tuple[bytes, dict[str, int]]:
    """Encode a triangle-atlas GLB compatible with Genesis' texture path."""

    face_vertices, face_normals, uv, face_indices, atlas = build_triangle_texture_atlas(
        vertices, colors_rgba, faces
    )
    face_vertices = np.asarray(face_vertices, dtype="<f4")
    face_normals = np.asarray(face_normals, dtype="<f4")
    uv = np.asarray(uv, dtype="<f4")
    face_indices = np.asarray(face_indices, dtype="<u4")
    position_bytes = face_vertices.tobytes(order="C")
    uv_bytes = uv.tobytes(order="C")
    index_bytes = face_indices.reshape(-1).tobytes(order="C")
    png_bytes = encode_rgb_png(atlas)
    position_offset = 0
    normal_bytes = face_normals.tobytes(order="C")
    normal_offset = len(position_bytes)
    uv_offset = normal_offset + len(normal_bytes)
    index_offset = uv_offset + len(uv_bytes)
    image_offset = index_offset + len(index_bytes)
    binary = _pad4(
        position_bytes + normal_bytes + uv_bytes + index_bytes + png_bytes, b"\0"
    )
    document = {
        "asset": {"version": "2.0", "generator": "radeon-oneloop surface carrier exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "complete_surface_carrier"}],
        "meshes": [
            {
                "name": "complete_surface_carrier",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "observed_and_fallback_vertex_color",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.82,
                },
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [
            {"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}
        ],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(png_bytes)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(face_vertices),
                "type": "VEC3",
                "min": face_vertices.min(axis=0).astype(float).tolist(),
                "max": face_vertices.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(face_normals),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(uv),
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": int(face_indices.size),
                "type": "SCALAR",
                "min": [int(face_indices.min())],
                "max": [int(face_indices.max())],
            },
        ],
    }
    json_chunk = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN\0"),
            binary,
        )
    )
    return glb, {
        "render_vertices": int(len(face_vertices)),
        "atlas_width_px": int(atlas.shape[1]),
        "atlas_height_px": int(atlas.shape[0]),
    }


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def export_carrier(carrier_root: Path, output: Path) -> dict[str, Any]:
    carrier_root = carrier_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        source_manifest, ply_path, _ = _verify_carrier_root(carrier_root)
        vertices, colors, faces = read_ascii_colored_ply(ply_path)
        expected_extents = np.asarray(source_manifest["geometry"]["final_extents_m"])
        extents = np.ptp(vertices.astype(np.float64), axis=0)
        if not np.allclose(extents, expected_extents, atol=2.0e-8, rtol=0.0):
            raise ValueError("PLY extents do not match the surface-carrier manifest")
        glb_path = staging / "complete_surface_carrier.glb"
        glb, encoding_metrics = encode_quantized_material_glb(
            vertices, colors, faces
        )
        glb_path.write_bytes(glb)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "formal": False,
            "host_role": "portable_nonformal_visual_asset_conversion",
            "physical_output": False,
            "visual_review_required": True,
            "source": {
                "surface_carrier_manifest_sha256": sha256_file(carrier_root / "manifest.json"),
                "surface_carrier_hashes_sha256": sha256_file(carrier_root / "hashes.sha256"),
                "ply_sha256": sha256_file(ply_path),
            },
            "geometry": {
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
                "extents_m": extents.tolist(),
                "coordinate_convention": source_manifest["geometry"]["coordinate_convention"],
            },
            "appearance": {
                "encoding": "smooth_indexed_geometry_with_quantized_PBR_materials",
                **encoding_metrics,
                "observed_vertex_fraction": source_manifest["appearance"]["observed_vertex_fraction"],
                "fallback_vertex_fraction": source_manifest["appearance"]["fallback_vertex_fraction"],
            },
            "output": {
                "relpath": glb_path.name,
                "sha256": sha256_file(glb_path),
                "media_type": "model/gltf-binary",
                "intended_role": "collision_disabled_pose_following_Genesis_visual_fallback",
            },
            "not_proven": [
                "Genesis renderer compatibility before the downstream preview gate",
                "photorealistic completion",
                "collision suitability",
                "formal single-Radeon evidence",
            ],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        hashes_sha = _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA_VERSION,
                    "status": "complete_conversion_pending_genesis_visual_review",
                    "manifest_sha256": sha256_file(manifest_path),
                    "hashes_sha256": hashes_sha,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except BaseException as error:
        (staging / "FAILED").write_text(
            json.dumps({"error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED")
        if not failed.exists():
            os.replace(staging, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-carrier-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_carrier(args.surface_carrier_root, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
