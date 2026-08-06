"""Pinned, dependency-light IO for the public MGPBD bunny reference meshes.

The upstream repository does not currently expose a license file, so reference
data are fetched into an untracked run directory and never vendored here.  This
module verifies exact source hashes before parsing the TetGen text files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .mgpbd_tet import boundary_faces, tetrahedral_rest_data


REFERENCE_REPOSITORY = "https://github.com/chunleili/mgpbd"
REFERENCE_COMMIT = "06761eb38dee8fb4165c6b9df8212c4f1744d131"
REFERENCE_SCENE_SHA256 = (
    "59caf188ea939cae78be1ffb885c2a31197c06267bdda95c977ff6910e5882d7"
)


@dataclass(frozen=True)
class ReferenceModelSpec:
    model: str
    nodes: int
    tetrahedra: int
    boundary_vertices: int
    boundary_faces: int
    boundary_edges: int
    node_sha256: str
    element_sha256: str
    face_sha256: str | None = None


REFERENCE_MODELS = {
    "bunny_small": ReferenceModelSpec(
        model="bunny_small",
        nodes=2_992,
        tetrahedra=12_298,
        boundary_vertices=2_000,
        boundary_faces=3_996,
        boundary_edges=5_994,
        node_sha256=(
            "d5c4f1d0af593074920180e7a6088fcf3e6edd083726aa8c644b3157f994a89d"
        ),
        element_sha256=(
            "cffdc6168764863049f1bf729e5f02ba1de7af348a2528ed16a90e9baed6ae7e"
        ),
        face_sha256=(
            "8a8e5b6529f3eee316a4efc8f3fe8dbbe2073aac1f5c54db4fadc77f2cf6ca73"
        ),
    ),
    "bunnyBig": ReferenceModelSpec(
        model="bunnyBig",
        nodes=60_678,
        tetrahedra=270_199,
        boundary_vertices=34_817,
        boundary_faces=69_630,
        boundary_edges=104_445,
        node_sha256=(
            "98298f7f1310d5bc75d01a0941cdeec0e760ca42b5841e8bac392ff2ad854543"
        ),
        element_sha256=(
            "8a6d6cd0f52b1e7f1c2db61c380f34b78fad4fbff6ac3d3b009e5e031b49bccf"
        ),
    ),
}


@dataclass(frozen=True)
class ReferenceTetMesh:
    positions: np.ndarray
    elements: np.ndarray
    boundary: np.ndarray
    diagnostics: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_lines(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        content = raw.split("#", 1)[0].strip()
        if content:
            rows.append(content.split())
    if not rows:
        raise ValueError(f"TetGen file is empty: {path}")
    return rows


def read_tetgen_nodes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _data_lines(path)
    count, dimension = int(rows[0][0]), int(rows[0][1])
    if dimension != 3 or count <= 0 or len(rows) - 1 != count:
        raise ValueError("TetGen node header/count is invalid")
    identifiers = np.asarray([int(row[0]) for row in rows[1:]], dtype=np.int64)
    positions = np.asarray(
        [[float(value) for value in row[1:4]] for row in rows[1:]],
        dtype=np.float64,
    )
    if len(np.unique(identifiers)) != count:
        raise ValueError("TetGen node identifiers are not unique")
    if not np.isfinite(positions).all():
        raise ValueError("TetGen node coordinates are not finite")
    return identifiers, positions


def read_tetgen_elements(
    path: Path, node_identifiers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = _data_lines(path)
    count, vertices_per_element = int(rows[0][0]), int(rows[0][1])
    if vertices_per_element != 4 or count <= 0 or len(rows) - 1 != count:
        raise ValueError("TetGen element header/count is invalid")
    identifiers = np.asarray([int(row[0]) for row in rows[1:]], dtype=np.int64)
    source_elements = np.asarray(
        [[int(value) for value in row[1:5]] for row in rows[1:]],
        dtype=np.int64,
    )
    if len(np.unique(identifiers)) != count:
        raise ValueError("TetGen element identifiers are not unique")
    node_lookup = {int(identifier): index for index, identifier in enumerate(node_identifiers)}
    try:
        elements = np.asarray(
            [[node_lookup[int(vertex)] for vertex in tet] for tet in source_elements],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError("TetGen element references an unknown node") from exc
    return identifiers, elements


def read_tetgen_faces(path: Path, node_identifiers: np.ndarray) -> np.ndarray:
    rows = _data_lines(path)
    count = int(rows[0][0])
    if count <= 0 or len(rows) - 1 != count:
        raise ValueError("TetGen face header/count is invalid")
    node_lookup = {int(identifier): index for index, identifier in enumerate(node_identifiers)}
    try:
        return np.asarray(
            [
                [node_lookup[int(value)] for value in row[1:4]]
                for row in rows[1:]
            ],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError("TetGen face references an unknown node") from exc


def signed_six_volumes(positions: np.ndarray, elements: np.ndarray) -> np.ndarray:
    tets = np.asarray(positions, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    return np.einsum(
        "ij,ij->i",
        np.cross(tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0]),
        tets[:, 3] - tets[:, 0],
    )


def normalize_consistent_orientation(
    positions: np.ndarray, elements: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    signed = signed_six_volumes(positions, elements)
    scale = max(float(np.max(np.ptp(positions, axis=0))) ** 3, 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 32.0
    positive = int(np.count_nonzero(signed > tolerance))
    negative = int(np.count_nonzero(signed < -tolerance))
    degenerate = int(len(signed) - positive - negative)
    if degenerate:
        raise ValueError("reference tet mesh contains a degenerate element")
    if positive and negative:
        raise ValueError("reference tet mesh has inconsistent element orientation")
    normalized = np.asarray(elements, dtype=np.int32).copy()
    method = "identity"
    if negative:
        normalized[:, [1, 2]] = normalized[:, [2, 1]]
        method = "swap_local_1_2"
    normalized_signed = signed_six_volumes(positions, normalized)
    if np.any(normalized_signed <= tolerance):
        raise ValueError("orientation normalization did not produce positive tets")
    return normalized, {
        "source_positive_tetrahedra": positive,
        "source_negative_tetrahedra": negative,
        "source_degenerate_tetrahedra": degenerate,
        "normalization": method,
        "normalized_nonpositive_tetrahedra": int(
            np.count_nonzero(normalized_signed <= tolerance)
        ),
    }


def boundary_topology(faces: np.ndarray) -> dict[str, object]:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edge_rows = np.sort(
        np.concatenate(
            (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
            axis=0,
        ),
        axis=1,
    )
    edges, incidence = np.unique(edge_rows, axis=0, return_counts=True)
    vertices = np.unique(triangles)
    parent = {int(vertex): int(vertex) for vertex in vertices}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges.tolist():
        union(int(left), int(right))
    components = len({find(int(vertex)) for vertex in vertices})
    euler = int(len(vertices) - len(edges) + len(triangles))
    genus = None
    if components == 1 and np.all(incidence == 2):
        genus = (2 - euler) / 2
    return {
        "boundary_vertices": int(len(vertices)),
        "boundary_faces": int(len(triangles)),
        "boundary_edges": int(len(edges)),
        "open_edges": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edges": int(np.count_nonzero(incidence > 2)),
        "edge_incidence_not_two": int(np.count_nonzero(incidence != 2)),
        "connected_components": components,
        "euler_characteristic": euler,
        "genus": genus,
    }


def load_reference_mesh(reference_root: Path, model: str) -> ReferenceTetMesh:
    if model not in REFERENCE_MODELS:
        raise ValueError(f"unsupported MGPBD reference model: {model}")
    spec = REFERENCE_MODELS[model]
    model_root = Path(reference_root).resolve() / "data" / "model" / model
    node_path = model_root / f"{model}.node"
    element_path = model_root / f"{model}.ele"
    face_path = model_root / f"{model}.face"
    node_hash, element_hash = sha256_file(node_path), sha256_file(element_path)
    if node_hash != spec.node_sha256 or element_hash != spec.element_sha256:
        raise ValueError("reference MGPBD node/element hash mismatch")
    node_ids, positions = read_tetgen_nodes(node_path)
    _element_ids, source_elements = read_tetgen_elements(element_path, node_ids)
    if len(positions) != spec.nodes or len(source_elements) != spec.tetrahedra:
        raise ValueError("reference MGPBD source count mismatch")
    elements, orientation = normalize_consistent_orientation(
        positions, source_elements
    )
    boundary = boundary_faces(elements)
    topology = boundary_topology(boundary)
    expected_topology = {
        "boundary_vertices": spec.boundary_vertices,
        "boundary_faces": spec.boundary_faces,
        "boundary_edges": spec.boundary_edges,
    }
    if any(topology[key] != value for key, value in expected_topology.items()):
        raise ValueError("derived reference boundary count mismatch")
    face_hash = None
    face_matches_derived = None
    if spec.face_sha256 is not None:
        face_hash = sha256_file(face_path)
        if face_hash != spec.face_sha256:
            raise ValueError("reference MGPBD face hash mismatch")
        stored_faces = read_tetgen_faces(face_path, node_ids)
        face_matches_derived = np.array_equal(
            np.unique(np.sort(stored_faces, axis=1), axis=0),
            np.unique(np.sort(boundary, axis=1), axis=0),
        )
        if not face_matches_derived:
            raise ValueError("reference face file does not match tet boundary")
    rest_inverse, rest_volumes = tetrahedral_rest_data(positions, elements)
    conditions = np.linalg.cond(np.linalg.inv(rest_inverse))
    diagnostics = {
        "repository": REFERENCE_REPOSITORY,
        "commit": REFERENCE_COMMIT,
        "model": model,
        "node_path": str(node_path),
        "element_path": str(element_path),
        "face_path": str(face_path) if spec.face_sha256 is not None else None,
        "node_sha256": node_hash,
        "element_sha256": element_hash,
        "face_sha256": face_hash,
        "nodes": int(len(positions)),
        "tetrahedra": int(len(elements)),
        "orientation": orientation,
        "topology": topology,
        "face_file_matches_derived_boundary": face_matches_derived,
        "rest_volume_sim_units": {
            "minimum": float(np.min(rest_volumes)),
            "maximum": float(np.max(rest_volumes)),
            "total": float(np.sum(rest_volumes)),
        },
        "rest_matrix_condition": {
            "maximum": float(np.max(conditions)),
            "p95": float(np.percentile(conditions, 95)),
        },
        "physical_unit_claim": False,
    }
    return ReferenceTetMesh(
        positions=positions.astype(np.float32),
        elements=elements,
        boundary=boundary,
        diagnostics=diagnostics,
    )


def verify_reference_scene(reference_root: Path) -> dict[str, str]:
    path = (
        Path(reference_root).resolve()
        / "data"
        / "scene"
        / "bunny_squash"
        / "bunny_squash.json"
    )
    digest = sha256_file(path)
    if digest != REFERENCE_SCENE_SHA256:
        raise ValueError("reference MGPBD bunny scene hash mismatch")
    return {"path": str(path), "sha256": digest}
