import json
from pathlib import Path
import struct

import numpy as np
import pytest

from gaussian.export_surface_carrier_glb import (
    encode_rgb_png,
    encode_quantized_material_glb,
    encode_textured_glb,
    read_ascii_colored_ply,
)


def _write_triangle(path: Path) -> None:
    path.write_text(
        """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float confidence
property uchar source_count
element face 1
property list uchar int vertex_indices
end_header
0 0 0 255 0 0 1.0 2
1 0 0 0 255 0 0.5 1
0 1 0 0 0 255 0.0 0
3 0 1 2
""",
        encoding="ascii",
    )


def test_ascii_ply_to_vertex_colored_glb(tmp_path: Path) -> None:
    ply = tmp_path / "carrier.ply"
    _write_triangle(ply)
    vertices, colors, faces = read_ascii_colored_ply(ply)
    assert vertices.shape == (3, 3)
    assert colors.tolist() == [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ]
    assert faces.tolist() == [[0, 1, 2]]

    glb, metrics = encode_textured_glb(vertices, colors, faces)
    assert metrics == {
        "render_vertices": 3,
        "atlas_width_px": 6,
        "atlas_height_px": 6,
    }
    magic, version, total_length = struct.unpack_from("<4sII", glb, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(glb)
    json_length, json_kind = struct.unpack_from("<I4s", glb, 12)
    assert json_kind == b"JSON"
    document = json.loads(glb[20 : 20 + json_length].decode("utf-8"))
    assert document["meshes"][0]["primitives"][0]["attributes"] == {
        "POSITION": 0,
        "NORMAL": 1,
        "TEXCOORD_0": 2,
    }
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] == {
        "index": 0,
        "texCoord": 0,
    }
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["accessors"][1]["type"] == "VEC3"
    assert document["accessors"][2]["componentType"] == 5126
    assert document["accessors"][3]["count"] == 3
    assert encode_rgb_png(np.zeros((2, 3, 3), dtype=np.uint8)).startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    material_glb, material_metrics = encode_quantized_material_glb(
        vertices, colors, faces
    )
    assert material_glb.startswith(b"glTF")
    assert material_metrics["render_vertices"] == 3
    assert 1 <= material_metrics["material_buckets"] <= 6**3


def test_ascii_ply_rejects_non_triangular_faces(tmp_path: Path) -> None:
    ply = tmp_path / "carrier.ply"
    _write_triangle(ply)
    ply.write_text(
        ply.read_text(encoding="ascii").replace("3 0 1 2\n", "4 0 1 2 0\n"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="triangular"):
        read_ascii_colored_ply(ply)


def test_glb_rejects_out_of_range_face() -> None:
    with pytest.raises(ValueError, match="out of range"):
        encode_textured_glb(
            np.zeros((3, 3), dtype=np.float32),
            np.full((3, 4), 255, dtype=np.uint8),
            np.asarray([[0, 1, 3]], dtype=np.uint32),
        )
    with pytest.raises(ValueError, match="out of range"):
        encode_quantized_material_glb(
            np.zeros((3, 3), dtype=np.float32),
            np.full((3, 4), 255, dtype=np.uint8),
            np.asarray([[0, 1, 3]], dtype=np.uint32),
        )
