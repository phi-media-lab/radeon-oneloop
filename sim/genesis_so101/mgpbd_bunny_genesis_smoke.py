#!/usr/bin/env python3
"""Render an audited MGPBD bunny checkpoint through Genesis custom vertices.

This gate deliberately keeps ownership explicit: MGPBD produced the physical
tetrahedral state; Genesis owns only the boundary renderer.  It proves the
state/mesh bridge and surface continuity without claiming that Genesis' native
FEM, contact, gravity, or time integrator produced the deformation.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import numpy as np

from .handover_asset import sha256_file


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def read_triangle_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertex and triangular-face records used by conformance OBJ files."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields or fields[0] == "#":
            continue
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(f"non-3D OBJ vertex in {path}")
            vertices.append([float(value) for value in fields[1:]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"non-triangular OBJ face in {path}")
            face = []
            for value in fields[1:]:
                index = int(value.split("/", 1)[0])
                if index <= 0:
                    raise ValueError("negative/relative OBJ indices are unsupported")
                face.append(index - 1)
            faces.append(face)
    vertex_array = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    face_array = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if not len(vertex_array) or not len(face_array):
        raise ValueError(f"empty OBJ boundary: {path}")
    if int(face_array.max()) >= len(vertex_array):
        raise ValueError(f"OBJ face references a missing vertex: {path}")
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"OBJ contains non-finite vertices: {path}")
    return vertex_array, face_array


def compact_boundary(
    positions: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact boundary vertices, faces, and compact-to-volume indices."""

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    boundary_indices = np.unique(faces).astype(np.int32)
    volume_to_compact = np.full(len(positions), -1, dtype=np.int32)
    volume_to_compact[boundary_indices] = np.arange(
        len(boundary_indices), dtype=np.int32
    )
    compact_faces = volume_to_compact[faces]
    if np.any(compact_faces < 0):
        raise AssertionError("boundary compaction lost a referenced vertex")
    return positions[boundary_indices], compact_faces, boundary_indices


def boundary_topology(faces: np.ndarray) -> dict[str, int | bool]:
    """Audit edge incidence and edge-connected components of a triangle shell."""

    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for first, second in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(first), int(second))))
            edge_to_faces.setdefault(edge, []).append(face_index)
    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        for first in incident:
            adjacency[first].extend(other for other in incident if other != first)
    unseen = set(range(len(faces)))
    components = 0
    while unseen:
        components += 1
        queue = deque((unseen.pop(),))
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    incidence = Counter(len(value) for value in edge_to_faces.values())
    open_edges = int(incidence[1])
    nonmanifold_edges = int(
        sum(count for degree, count in incidence.items() if degree > 2)
    )
    return {
        "faces": int(len(faces)),
        "edges": int(len(edge_to_faces)),
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "edge_connected_components": int(components),
        "closed_two_manifold": bool(
            open_edges == 0 and nonmanifold_edges == 0 and components == 1
        ),
    }


def surface_metrics(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    rest_double_area: np.ndarray | None = None,
) -> dict[str, object]:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    triangles = positions[faces]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    result: dict[str, object] = {
        "finite": bool(np.isfinite(positions).all()),
        "vertices": int(len(positions)),
        "faces": int(len(faces)),
        "degenerate_faces": int(np.count_nonzero(double_area <= 1.0e-12)),
        "minimum_double_area_m2": float(np.min(double_area)),
        "extents_m": np.ptp(positions, axis=0).tolist(),
        "center_m": positions.mean(axis=0).tolist(),
    }
    if rest_double_area is not None:
        rest = np.asarray(rest_double_area, dtype=np.float64)
        ratio = double_area / np.maximum(rest, 1.0e-30)
        result["area_ratio_to_rest"] = {
            "minimum": float(np.min(ratio)),
            "p01": float(np.percentile(ratio, 1)),
            "median": float(np.median(ratio)),
            "maximum": float(np.max(ratio)),
        }
    return result


def bunny_world_transform(rest_positions: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return source center, metric scale, and translation for an upright bunny."""

    rest = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
    minimum = np.min(rest, axis=0)
    maximum = np.max(rest, axis=0)
    source_center = 0.5 * (minimum + maximum)
    scale = 0.18 / float(np.max(maximum - minimum))
    centered = rest - source_center
    # Upstream squash/recovery uses source Y as height.  Map it to Genesis Z;
    # source Z becomes depth while source X remains horizontal.
    rotated = np.column_stack(
        (centered[:, 0], -centered[:, 2], centered[:, 1])
    ) * scale
    translation = np.asarray((0.0, 0.0, 0.012 - np.min(rotated[:, 2])))
    return source_center, scale, translation


def transform_bunny(
    positions: np.ndarray,
    source_center: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(positions, dtype=np.float64) - source_center
    rotated = np.column_stack(
        (centered[:, 0], -centered[:, 2], centered[:, 1])
    )
    return (rotated * scale + translation).astype(np.float32)


def write_triangle_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        *(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices),
        *(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces),
    ]
    path.write_text("".join(lines), encoding="utf-8")


def nearest_rest_indices(
    visual_rest: np.ndarray, boundary_rest: np.ndarray
) -> tuple[np.ndarray, float]:
    """Map Genesis' possibly reordered/duplicated vverts to physical boundary nodes."""

    from scipy.spatial import cKDTree

    distances, indices = cKDTree(boundary_rest).query(visual_rest, k=1)
    return np.asarray(indices, dtype=np.int32), float(np.max(distances))


def main() -> None:
    import imageio.v3 as iio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    artifacts = source_run / "artifacts"
    rest_path = artifacts / "rest_boundary.obj"
    squashed_path = artifacts / "squashed_boundary.obj"
    checkpoint_path = artifacts / "last_safe_state.npz"
    for path in (rest_path, squashed_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rest, rest_faces = read_triangle_obj(rest_path)
    squashed, squashed_faces = read_triangle_obj(squashed_path)
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        accepted = np.asarray(checkpoint["positions"], dtype=np.float32)
        checkpoint_faces = np.asarray(checkpoint["boundary_faces"], dtype=np.int32)
        completed_outer = int(
            np.asarray(checkpoint["completed_outer_iterations"]).item()
        )
    if rest.shape != squashed.shape or rest.shape != accepted.shape:
        raise ValueError("MGPBD checkpoint stages have inconsistent vertex arrays")
    if not np.array_equal(rest_faces, squashed_faces) or not np.array_equal(
        rest_faces, checkpoint_faces
    ):
        raise ValueError("MGPBD checkpoint stages do not share one boundary topology")

    source_center, scale, translation = bunny_world_transform(rest)
    world_stages_full = {
        "rest": transform_bunny(rest, source_center, scale, translation),
        "squashed": transform_bunny(squashed, source_center, scale, translation),
        "accepted_outer": transform_bunny(
            accepted, source_center, scale, translation
        ),
    }
    compact_rest, compact_faces, boundary_indices = compact_boundary(
        world_stages_full["rest"], rest_faces
    )
    world_stages = {
        name: values[boundary_indices]
        for name, values in world_stages_full.items()
    }
    topology = boundary_topology(compact_faces)
    boundary_obj = output / "bunny_small_mgpbd_boundary.obj"
    write_triangle_obj(boundary_obj, compact_rest, compact_faces)

    import genesis as gs
    import torch

    gs.init(backend=gs.amdgpu, seed=args.seed)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis did not select AMD GPU: {gs.backend}")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.68, 0.68, 0.68),
            shadow=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    visual = scene.add_entity(
        gs.morphs.Mesh(
            file=str(boundary_obj),
            fixed=True,
            collision=False,
            convexify=False,
            decimate=False,
            watertighten=0,
            align=False,
            file_meshes_are_zup=True,
            enable_custom_vverts=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.72, 0.74, 0.78, 1.0),
            roughness=0.86,
        ),
    )
    for visual_geom in visual.vgeoms:
        visual_geom.surface.smooth = True
    camera = scene.add_camera(
        res=(720, 720),
        pos=(0.27, 0.42, 0.17),
        lookat=(0.0, 0.0, 0.105),
        up=(0.0, 0.0, 1.0),
        fov=31,
        GUI=False,
    )
    scene.build()

    visual_tensor = visual.get_vverts().detach().reshape(-1, 3)
    visual_rest = visual_tensor.cpu().numpy()
    visual_to_boundary, binding_error = nearest_rest_indices(
        visual_rest, compact_rest
    )
    captures: dict[str, str] = {}
    capture_images: dict[str, np.ndarray] = {}
    stage_metrics: dict[str, object] = {}
    rest_triangles = compact_rest[compact_faces]
    rest_double_area = np.linalg.norm(
        np.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0],
            rest_triangles[:, 2] - rest_triangles[:, 0],
        ),
        axis=1,
    )
    for stage_name in ("rest", "squashed", "accepted_outer"):
        compact_positions = world_stages[stage_name]
        rendered_positions = compact_positions[visual_to_boundary]
        visual.set_vverts(
            torch.as_tensor(
                rendered_positions,
                dtype=visual_tensor.dtype,
                device=visual_tensor.device,
            )
        )
        rgb, _, _, _ = camera.render(rgb=True, force_render=True)
        image_path = output / f"bunny_{stage_name}.png"
        image = as_numpy(rgb).astype(np.uint8)
        iio.imwrite(image_path, image)
        captures[stage_name] = image_path.name
        capture_images[stage_name] = image
        stage_metrics[stage_name] = surface_metrics(
            compact_positions,
            compact_faces,
            rest_double_area=rest_double_area,
        )

    source_markers = sorted(
        name
        for name in ("DONE", "FAILED", "GATE_PASSED", "GATE_FAILED")
        if (source_run / name).exists()
    )
    capture_hashes = {
        name: sha256_file(output / filename)
        for name, filename in captures.items()
    }
    pixel_differences: dict[str, dict[str, float]] = {}
    for first, second in (
        ("rest", "squashed"),
        ("squashed", "accepted_outer"),
        ("rest", "accepted_outer"),
    ):
        absolute = np.abs(
            capture_images[first].astype(np.int16)
            - capture_images[second].astype(np.int16)
        )
        pixel_differences[f"{first}_to_{second}"] = {
            "mean_absolute_channel_difference": float(np.mean(absolute)),
            "changed_pixel_fraction": float(
                np.mean(np.any(absolute > 0, axis=-1))
            ),
        }
    checks = {
        "source_run_completed": "DONE" in source_markers,
        "checkpoint_is_intermediate_not_full_pass": (
            "GATE_FAILED" in source_markers and "GATE_PASSED" not in source_markers
        ),
        "accepted_outer_checkpoint_present": completed_outer > 0,
        "boundary_closed_two_manifold": bool(topology["closed_two_manifold"]),
        "genesis_custom_vertex_binding_exact": binding_error <= 1.0e-6,
        "all_stage_vertices_finite": all(
            bool(record["finite"]) for record in stage_metrics.values()
        ),
        "rest_boundary_nondegenerate": (
            int(stage_metrics["rest"]["degenerate_faces"]) == 0
        ),
        "accepted_boundary_nondegenerate": (
            int(stage_metrics["accepted_outer"]["degenerate_faces"]) == 0
        ),
        "capture_hashes_are_stage_distinct": len(set(capture_hashes.values()))
        == len(capture_hashes),
        "squash_is_visibly_different_from_rest": (
            pixel_differences["rest_to_squashed"][
                "changed_pixel_fraction"
            ]
            >= 0.01
        ),
        "accepted_state_is_visibly_different_from_squash": (
            pixel_differences["squashed_to_accepted_outer"][
                "changed_pixel_fraction"
            ]
            >= 0.01
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "radeon_oneloop.genesis_mgpbd_bunny_bridge.v1",
        "formal": False,
        "passed": passed,
        "backend": str(gs.backend),
        "device": str(gs.device),
        "source_run": source_run.name,
        "source_markers": source_markers,
        "source_files": {
            path.name: sha256_file(path)
            for path in (rest_path, squashed_path, checkpoint_path)
        },
        "checkpoint": {
            "completed_outer_iterations": completed_outer,
            "scope": "last_atomically_accepted_outer_from_failed_full_P0a2",
            "full_P0a2_passed": False,
        },
        "ownership": {
            "physical_state": "custom_MGPBD_checkpoint",
            "Genesis": "custom_vverts_boundary_rendering_only",
            "Genesis_native_FEM_used": False,
            "contact_enabled": False,
            "gravity_enabled": False,
            "integration_enabled": False,
        },
        "transform": {
            "source_up": "+Y",
            "Genesis_up": "+Z",
            "metric_maximum_extent_m": 0.18,
            "scale_m_per_source_unit": scale,
            "source_center": source_center.tolist(),
            "world_translation_m": translation.tolist(),
        },
        "topology": {
            **topology,
            "volume_vertices": int(len(rest)),
            "boundary_vertices": int(len(boundary_indices)),
        },
        "binding": {
            "Genesis_visual_vertices": int(len(visual_rest)),
            "maximum_rest_mapping_error_m": binding_error,
            "method": "nearest_exact_boundary_node_after_Genesis_import",
        },
        "stages": stage_metrics,
        "captures": captures,
        "capture_sha256": capture_hashes,
        "capture_pixel_differences": pixel_differences,
        "checks": checks,
        "claim_scope": (
            "MGPBD checkpoint-to-Genesis renderer bridge and coherent boundary; "
            "not complete P0a2 convergence, dynamics, contact, or realtime"
        ),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashed = [boundary_obj, metrics_path]
    hashed.extend(output / name for name in captures.values())
    (output / "hashes.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}" for path in sorted(hashed)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Genesis MGPBD bunny bridge gate failed")


if __name__ == "__main__":
    main()
