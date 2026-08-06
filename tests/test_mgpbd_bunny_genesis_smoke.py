from __future__ import annotations

import numpy as np

from sim.genesis_so101.mgpbd_bunny_genesis_smoke import (
    boundary_topology,
    bunny_world_transform,
    compact_boundary,
    surface_metrics,
    transform_bunny,
)


def test_compact_closed_tet_boundary_is_one_manifold_component() -> None:
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (8.0, 8.0, 8.0),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int32,
    )
    compact, compact_faces, indices = compact_boundary(positions, faces)
    assert compact.shape == (4, 3)
    assert indices.tolist() == [0, 1, 2, 3]
    assert boundary_topology(compact_faces) == {
        "faces": 4,
        "edges": 6,
        "open_edges": 0,
        "nonmanifold_edges": 0,
        "edge_connected_components": 1,
        "closed_two_manifold": True,
    }


def test_bunny_transform_maps_source_y_to_genesis_z() -> None:
    rest = np.asarray(
        ((-1.0, 2.0, -0.5), (1.0, 4.0, 0.5)), dtype=np.float32
    )
    center, scale, translation = bunny_world_transform(rest)
    world = transform_bunny(rest, center, scale, translation)
    assert np.isclose(np.ptp(world[:, 2]), np.float32(2.0 * scale))
    assert np.isclose(float(np.min(world[:, 2])), 0.012)
    assert np.isclose(float(np.max(np.ptp(world, axis=0))), 0.18)


def test_surface_metrics_detects_degenerate_face() -> None:
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    metrics = surface_metrics(positions, np.asarray(((0, 1, 2),)))
    assert metrics["finite"]
    assert metrics["degenerate_faces"] == 1
