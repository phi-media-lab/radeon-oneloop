"""AMD/Vulkan XPBD shell-and-filler model for the team-owned plush toy.

The solver is intentionally independent of Genesis' rigid/MPM couplers.  A
closed triangle shell carries the visible surface.  Discrete interior grains
are sampled on a face-centred-cubic lattice, collide against every other grain,
and use dry tangential friction.  Unilateral grain-to-triangle contacts keep
the fill inside the moving fabric boundary without welding it to the shell;
far-side braces only prevent catastrophic shell self-collapse.  The global
shell-volume constraint is disabled for this granular mode: preserving one
incompressible volume makes the object behave like a water balloon.
Kinematic gripper boxes interact only through collision and Coulomb friction;
there are no grasp tethers or pose snaps.

Taichi is imported lazily so topology and gate-policy tests remain runnable on
machines that do not have the optional Vulkan runtime installed.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class XPBDPlushConfig:
    """Numerical and material parameters for the qualitative plush model."""

    dt_s: float = 1.0 / 120.0
    substeps: int = 2
    solver_iterations: int = 30
    gravity_m_s2: float = -9.81
    velocity_retention_per_substep: float = 0.993
    particle_radius_m: float = 0.0025
    filler_spacing_m: float = 0.00925
    shell_edge_compliance_m_n: float = 5.0e-9
    shell_bend_compliance_m_n: float = 5.0e-8
    shell_volume_compliance_m3_n: float = 2.0e-10
    shell_volume_constraint_enabled: bool = False
    shell_minimum_volume_ratio: float = 0.75
    filler_compliance_m_n: float = 5.0e-8
    shell_grain_compliance_m_n: float = 5.0e-8
    shell_grain_contact_relaxation: float = 1.50
    shell_grain_final_stabilization_passes: int = 6
    shell_grain_contact_patch_faces: int = 16
    shell_grain_friction: float = 0.90
    shell_grain_clearance_m: float = 0.00020
    support_base_flatten_height_m: float = 0.020
    shell_cross_compliance_m_n: float = 5.0e-9
    grain_contact_distance_m: float = 0.0092
    grain_friction: float = 0.90
    contact_friction: float = 1.40
    table_friction: float = 0.80
    contact_slop_m: float = 0.00035
    contact_release_m: float = 0.0015
    shell_inverse_mass: float = 1.0
    filler_inverse_mass: float = 0.30
    maximum_grain_speed_m_s: float = 1.0
    nominal_mass_kg: float = 0.040

    def validate(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("XPBD dt must be positive")
        if self.substeps < 1 or self.solver_iterations < 1:
            raise ValueError("XPBD substeps and iterations must be positive")
        if not 4 <= self.shell_grain_contact_patch_faces <= 32:
            raise ValueError("XPBD shell-grain contact patch must contain [4, 32] faces")
        if not 0.0 <= self.support_base_flatten_height_m <= 0.02:
            raise ValueError("XPBD support-base flatten height must be in [0, 0.02] m")
        if not 0.50 <= self.shell_minimum_volume_ratio <= 1.0:
            raise ValueError("XPBD minimum shell volume ratio must be in [0.50, 1.0]")
        if not 0.0 < self.velocity_retention_per_substep <= 1.0:
            raise ValueError("velocity retention must be in (0, 1]")
        if self.particle_radius_m <= 0.0 or self.filler_spacing_m <= 0.0:
            raise ValueError("particle radius and filler spacing must be positive")
        if not 2.0 * self.particle_radius_m <= self.grain_contact_distance_m <= self.filler_spacing_m:
            raise ValueError(
                "grain contact distance must be at least two particle radii and no larger than filler spacing"
            )
        if (
            self.contact_friction < 0.0
            or self.table_friction < 0.0
            or self.grain_friction < 0.0
            or self.shell_grain_friction < 0.0
        ):
            raise ValueError("contact friction cannot be negative")
        if self.shell_grain_clearance_m < 0.0:
            raise ValueError("shell-grain clearance cannot be negative")
        if not 0.0 < self.shell_grain_contact_relaxation <= 2.0:
            raise ValueError("shell-grain contact relaxation must be in (0, 2]")
        if not 1 <= self.shell_grain_final_stabilization_passes <= 8:
            raise ValueError("shell-grain final stabilization passes must be in [1, 8]")
        if self.contact_slop_m < 0.0:
            raise ValueError("contact slop cannot be negative")
        if self.contact_release_m <= self.contact_slop_m:
            raise ValueError("contact release distance must exceed contact slop")
        if self.shell_inverse_mass <= 0.0 or self.filler_inverse_mass <= 0.0:
            raise ValueError("inverse masses must be positive")
        if self.maximum_grain_speed_m_s <= 0.0:
            raise ValueError("maximum grain speed must be positive")

    def to_dict(self) -> dict[str, float | int | bool]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PlushTopology:
    shell_vertices: np.ndarray
    shell_faces: np.ndarray
    filler_points: np.ndarray
    constraint_pairs: np.ndarray
    rest_lengths_m: np.ndarray
    compliances_m_n: np.ndarray
    constraint_kinds: np.ndarray
    shell_face_contact_patches: np.ndarray

    @property
    def particle_count(self) -> int:
        return int(len(self.shell_vertices) + len(self.filler_points))

    @property
    def shell_count(self) -> int:
        return int(len(self.shell_vertices))

    def counts(self) -> dict[str, int]:
        names = (
            "shell_edge",
            "shell_bend",
            "filler",
            "shell_cross",
        )
        result = {
            name: int(np.count_nonzero(self.constraint_kinds == index))
            for index, name in enumerate(names)
        }
        result.update(
            shell_vertices=self.shell_count,
            shell_faces=int(len(self.shell_faces)),
            filler_particles=int(len(self.filler_points)),
            shell_grain_dynamic_contact_slots=int(
                len(self.filler_points) * self.shell_face_contact_patches.shape[1]
            ),
            shell_grain_contact_patch_faces=int(
                self.shell_face_contact_patches.shape[1]
            ),
            total_particles=self.particle_count,
            total_distance_constraints=int(len(self.constraint_pairs)),
            total_constraints=int(
                len(self.constraint_pairs)
                + len(self.filler_points) * self.shell_face_contact_patches.shape[1]
            ),
        )
        return result


def load_triangle_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only vertex positions and triangular faces from a simple OBJ."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("v "):
                values = line.split()
                vertices.append(tuple(float(value) for value in values[1:4]))
            elif line.startswith("f "):
                indices = [int(value.split("/", 1)[0]) for value in line.split()[1:]]
                if len(indices) != 3:
                    raise ValueError(f"XPBD shell must be triangular: {path}")
                if any(index <= 0 for index in indices):
                    raise ValueError("negative or relative OBJ indices are unsupported")
                faces.append(tuple(index - 1 for index in indices))
    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,) or not len(vertex_array):
        raise ValueError(f"OBJ contains no vertices: {path}")
    if face_array.ndim != 2 or face_array.shape[1:] != (3,) or not len(face_array):
        raise ValueError(f"OBJ contains no triangular faces: {path}")
    if int(face_array.max()) >= len(vertex_array):
        raise ValueError(f"OBJ face index is outside the vertex array: {path}")
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"OBJ contains non-finite vertices: {path}")
    return vertex_array, face_array


def _shell_edges_and_bends(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_opposites: dict[tuple[int, int], list[int]] = {}
    for a, b, c in np.asarray(faces, dtype=np.int32):
        for first, second, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edge = (int(min(first, second)), int(max(first, second)))
            edge_opposites.setdefault(edge, []).append(int(opposite))
    nonmanifold = [edge for edge, opposite in edge_opposites.items() if len(opposite) != 2]
    if nonmanifold:
        raise ValueError(
            f"XPBD shell must be a closed two-manifold; {len(nonmanifold)} edges fail"
        )
    edges = np.asarray(sorted(edge_opposites), dtype=np.int32)
    bends = np.asarray(
        sorted(
            {
                tuple(sorted(opposites))
                for opposites in edge_opposites.values()
                if opposites[0] != opposites[1]
            }
        ),
        dtype=np.int32,
    )
    return edges, bends


def _convex_face_planes(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("XPBD shell contains a degenerate face")
    normals /= lengths[:, None]
    center = vertices.mean(axis=0)
    inward = np.einsum("ij,ij->i", center - triangles[:, 0], normals) > 0.0
    normals[inward] *= -1.0
    offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
    return normals.astype(np.float32), offsets.astype(np.float32)


def sample_convex_filler(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    spacing_m: float,
    surface_clearance_m: float,
) -> np.ndarray:
    """Sample an FCC hard-grain lattice inside a convex watertight shell.

    ``spacing_m`` is the nearest-neighbour distance, not the cubic cell width.
    FCC provides a dense, isotropic starting packing instead of the 30--40%
    effective fill fraction produced by a boundary-clipped simple-cubic grid.
    The particles remain independent unilateral contacts after initialization.
    """

    if spacing_m <= 0.0 or surface_clearance_m < 0.0:
        raise ValueError("filler spacing must be positive and clearance nonnegative")
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    normals, offsets = _convex_face_planes(vertices, faces)
    cell_width = float(np.sqrt(2.0) * spacing_m)
    center = vertices.mean(axis=0)
    lower_index = np.floor((vertices.min(axis=0) - center) / cell_width).astype(int) - 1
    upper_index = np.ceil((vertices.max(axis=0) - center) / cell_width).astype(int) + 1
    axes = [
        np.arange(lower, upper + 1, dtype=np.int32)
        for lower, upper in zip(lower_index, upper_index, strict=True)
    ]
    cells = (
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        * cell_width
        + center
    )
    basis = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
        dtype=np.float32,
    ) * cell_width
    candidates = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    accepted: list[np.ndarray] = []
    # Batch the half-space test so the 4k-face source hull does not require a
    # large temporary allocation on laptops used only for scaffold tests.
    for start in range(0, len(candidates), 256):
        points = candidates[start : start + 256]
        inside = np.all(
            points @ normals.T <= offsets[None, :] - surface_clearance_m + 1e-6,
            axis=1,
        )
        accepted.append(points[inside])
    result = np.concatenate(accepted, axis=0) if accepted else np.empty((0, 3))
    if len(result) < 8:
        raise ValueError(
            f"filler sampling produced only {len(result)} particles; reduce spacing/clearance"
        )
    return np.asarray(result, dtype=np.float32)


def _pairs_within(points: np.ndarray, maximum_distance_m: float) -> np.ndarray:
    pairs: list[np.ndarray] = []
    for first in range(len(points) - 1):
        distances = np.linalg.norm(points[first + 1 :] - points[first], axis=1)
        seconds = np.flatnonzero(distances <= maximum_distance_m) + first + 1
        if len(seconds):
            pairs.append(
                np.column_stack(
                    (
                        np.full(len(seconds), first, dtype=np.int32),
                        seconds.astype(np.int32),
                    )
                )
            )
    if not pairs:
        raise ValueError("filler lattice has no neighbour constraints")
    return np.concatenate(pairs, axis=0)


def _all_pairs(point_count: int) -> np.ndarray:
    """Return every unordered pair for the small discrete grain population."""

    if point_count < 2:
        raise ValueError("granular fill needs at least two particles")
    first, second = np.triu_indices(point_count, k=1)
    return np.column_stack((first, second)).astype(np.int32)


def _face_contact_patches(faces: np.ndarray, *, patch_size: int) -> np.ndarray:
    """Build a deterministic local face patch around every shell triangle."""

    triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if patch_size < 1 or patch_size > len(triangles):
        raise ValueError("face contact patch size is invalid")
    vertex_faces: dict[int, set[int]] = {}
    for face_index, face in enumerate(triangles):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), set()).add(face_index)
    adjacency: list[set[int]] = []
    for face_index, face in enumerate(triangles):
        neighbours = {face_index}
        for vertex in face:
            neighbours.update(vertex_faces[int(vertex)])
        adjacency.append(neighbours)
    patches: list[list[int]] = []
    for root in range(len(triangles)):
        selected = {root}
        frontier = {root}
        while len(selected) < patch_size:
            expanded: set[int] = set()
            for face_index in frontier:
                expanded.update(adjacency[face_index])
            frontier = expanded - selected
            if not frontier:
                raise ValueError("shell face graph is disconnected")
            for face_index in sorted(frontier):
                selected.add(face_index)
                if len(selected) == patch_size:
                    break
        patches.append(sorted(selected))
    return np.asarray(patches, dtype=np.int32)


def _point_triangle_squared_distances(
    shell_vertices: np.ndarray,
    shell_faces: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Return exact squared point-to-finite-triangle distances (P by F)."""

    vertices = np.asarray(shell_vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shell_faces, dtype=np.int32).reshape(-1, 3)
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = vertices[faces]
    normals, offsets = _convex_face_planes(shell_vertices, shell_faces)
    signed_plane_distance = query @ normals.T - offsets[None, :]
    first = triangles[:, 0]
    edge_first = triangles[:, 1] - first
    edge_second = triangles[:, 2] - first
    dot_first_first = np.einsum("ij,ij->i", edge_first, edge_first)
    dot_first_second = np.einsum("ij,ij->i", edge_first, edge_second)
    dot_second_second = np.einsum("ij,ij->i", edge_second, edge_second)
    denominator = dot_first_first * dot_second_second - dot_first_second**2
    if np.any(denominator <= 1.0e-18):
        raise ValueError("shell-grain candidate search found a degenerate face")
    from_first_dot_first = (
        query @ edge_first.T
        - np.einsum("ij,ij->i", first, edge_first)[None, :]
    )
    from_first_dot_second = (
        query @ edge_second.T
        - np.einsum("ij,ij->i", first, edge_second)[None, :]
    )
    barycentric_second = (
        dot_second_second[None, :] * from_first_dot_first
        - dot_first_second[None, :] * from_first_dot_second
    ) / denominator[None, :]
    barycentric_third = (
        dot_first_first[None, :] * from_first_dot_second
        - dot_first_second[None, :] * from_first_dot_first
    ) / denominator[None, :]
    barycentric_first = 1.0 - barycentric_second - barycentric_third
    projection_inside = (
        (barycentric_first >= 0.0)
        & (barycentric_second >= 0.0)
        & (barycentric_third >= 0.0)
    )
    squared_distance = np.where(
        projection_inside,
        signed_plane_distance**2,
        np.inf,
    )
    # Outside the face interior, the closest point lies on one of its three
    # finite edges.  This prevents distant coplanar triangles from displacing
    # the genuinely local boundary facets in the fixed-size candidate list.
    for start, edge in (
        (triangles[:, 0], triangles[:, 1] - triangles[:, 0]),
        (triangles[:, 1], triangles[:, 2] - triangles[:, 1]),
        (triangles[:, 2], triangles[:, 0] - triangles[:, 2]),
    ):
        edge_squared = np.einsum("ij,ij->i", edge, edge)
        parameter = np.clip(
            (
                query @ edge.T
                - np.einsum("ij,ij->i", start, edge)[None, :]
            )
            / edge_squared[None, :],
            0.0,
            1.0,
        )
        closest = start[None, :, :] + parameter[:, :, None] * edge[None, :, :]
        edge_distance = np.sum(
            (query[:, None, :] - closest) ** 2, axis=2
        )
        squared_distance = np.minimum(squared_distance, edge_distance)
    return squared_distance


def _opposite_shell_pairs(vertices: np.ndarray) -> np.ndarray:
    """Pair shell points with their nearest reflected far-side counterpart."""

    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    center = points.mean(axis=0)
    targets = 2.0 * center - points
    pairs: set[tuple[int, int]] = set()
    minimum_chord = 0.45 * float(np.max(np.ptp(points, axis=0)))
    for start in range(0, len(points), 128):
        target = targets[start : start + 128]
        squared_distance = np.sum(
            (target[:, None, :] - points[None, :, :]) ** 2, axis=2
        )
        nearest = np.argmin(squared_distance, axis=1)
        for local_index, opposite in enumerate(nearest):
            index = start + local_index
            first, second = sorted((index, int(opposite)))
            if (
                first != second
                and np.linalg.norm(points[second] - points[first]) >= minimum_chord
            ):
                pairs.add((first, second))
    if len(pairs) < len(points) // 4:
        raise ValueError("too few opposite shell cross-braces were generated")
    return np.asarray(sorted(pairs), dtype=np.int32)


def build_plush_topology(
    shell_obj: Path, config: XPBDPlushConfig = XPBDPlushConfig()
) -> PlushTopology:
    """Build the closed shell, interior fill, and XPBD constraint families."""

    config.validate()
    vertices, faces = load_triangle_obj(shell_obj)
    edges, bends = _shell_edges_and_bends(faces)
    filler = sample_convex_filler(
        vertices,
        faces,
        spacing_m=config.filler_spacing_m,
        surface_clearance_m=(
            config.grain_contact_distance_m * 0.5
            + config.shell_grain_clearance_m
        ),
    )
    # The contact graph must change as grains rearrange.  At this scale (a few
    # hundred grains), an exhaustive broad phase is both deterministic and
    # cheap enough on Vulkan; a fixed initial-neighbour graph behaves like a
    # soft fluid/foam and misses newly formed contacts.
    filler_pairs = _all_pairs(len(filler))
    shell_face_contact_patches = _face_contact_patches(
        faces,
        patch_size=config.shell_grain_contact_patch_faces,
    )
    crosses = _opposite_shell_pairs(vertices)
    shell_count = len(vertices)
    filler_pairs = filler_pairs + shell_count
    families: tuple[tuple[np.ndarray, float, int], ...] = (
        (edges, config.shell_edge_compliance_m_n, 0),
        (bends, config.shell_bend_compliance_m_n, 1),
        (filler_pairs, config.filler_compliance_m_n, 2),
        (crosses, config.shell_cross_compliance_m_n, 3),
    )
    all_points = np.concatenate((vertices, filler), axis=0)
    pairs = np.concatenate([family[0] for family in families], axis=0).astype(np.int32)
    kinds = np.concatenate(
        [np.full(len(family[0]), family[2], dtype=np.uint8) for family in families]
    )
    rest = np.linalg.norm(all_points[pairs[:, 0]] - all_points[pairs[:, 1]], axis=1)
    if np.any(rest < 1e-7):
        raise ValueError("XPBD topology contains a zero-length constraint")
    # Filler pairs are a broad-phase neighbour list, not permanent springs.
    # All grains share one hard-core diameter and interact only on compression.
    rest[kinds == 2] = config.grain_contact_distance_m
    compliances = np.concatenate(
        [np.full(len(family[0]), family[1], dtype=np.float32) for family in families]
    )
    return PlushTopology(
        shell_vertices=vertices,
        shell_faces=faces,
        filler_points=filler,
        constraint_pairs=pairs,
        rest_lengths_m=rest.astype(np.float32),
        compliances_m_n=compliances,
        constraint_kinds=kinds,
        shell_face_contact_patches=shell_face_contact_patches,
    )


def granular_solid_fraction(
    topology: PlushTopology, config: XPBDPlushConfig
) -> float:
    """Return hard-sphere volume divided by the closed shell rest volume."""

    triangles = topology.shell_vertices[topology.shell_faces]
    shell_volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )
    if shell_volume <= 0.0:
        raise ValueError("XPBD shell must have positive outward rest volume")
    radius = config.grain_contact_distance_m * 0.5
    grain_volume = len(topology.filler_points) * 4.0 * np.pi * radius**3 / 3.0
    return float(grain_volume / shell_volume)


def grain_containment_diagnostics(
    shell_positions: np.ndarray,
    shell_faces: np.ndarray,
    grain_positions: np.ndarray,
    *,
    grain_radius_m: float,
) -> dict[str, object]:
    """Measure grain containment against the actual watertight triangle shell.

    A deformed fabric cage is not guaranteed to remain convex, so intersecting
    all current face half-spaces produces false escapes.  Generalized winding
    classifies each grain against the watertight surface; exact finite-triangle
    distance then measures either outside escape or inside shell penetration.
    """

    shell = np.asarray(shell_positions, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shell_faces, dtype=np.int32).reshape(-1, 3)
    grains = np.asarray(grain_positions, dtype=np.float64).reshape(-1, 3)
    if grain_radius_m <= 0.0:
        raise ValueError("grain radius must be positive")
    triangles = shell[faces]
    winding = np.empty(len(grains), dtype=np.float64)
    for start in range(0, len(grains), 64):
        points = grains[start : start + 64]
        first = triangles[None, :, 0, :] - points[:, None, :]
        second = triangles[None, :, 1, :] - points[:, None, :]
        third = triangles[None, :, 2, :] - points[:, None, :]
        first_length = np.linalg.norm(first, axis=2)
        second_length = np.linalg.norm(second, axis=2)
        third_length = np.linalg.norm(third, axis=2)
        numerator = np.einsum(
            "bfi,bfi->bf", first, np.cross(second, third)
        )
        denominator = (
            first_length * second_length * third_length
            + np.einsum("bfi,bfi->bf", first, second) * third_length
            + np.einsum("bfi,bfi->bf", second, third) * first_length
            + np.einsum("bfi,bfi->bf", third, first) * second_length
        )
        solid_angle = 2.0 * np.arctan2(numerator, denominator)
        winding[start : start + len(points)] = (
            np.sum(solid_angle, axis=1) / (4.0 * np.pi)
        )
    inside = np.abs(winding) >= 0.5
    squared_distance = _point_triangle_squared_distances(shell, faces, grains)
    surface_distance = np.sqrt(np.min(squared_distance, axis=1))
    violations = np.where(
        inside,
        np.maximum(grain_radius_m - surface_distance, 0.0),
        surface_distance + grain_radius_m,
    )
    return {
        "model": "watertight_triangle_winding_plus_closest_surface",
        "maximum_escape_m": float(np.max(violations, initial=0.0)),
        "escaped_grain_count_over_0_5mm": int(np.count_nonzero(violations > 5.0e-4)),
        "outside_grain_count": int(np.count_nonzero(~inside)),
        "minimum_center_to_surface_distance_m": float(
            np.min(surface_distance, initial=np.inf)
        ),
        "minimum_absolute_winding_number": float(
            np.min(np.abs(winding), initial=np.inf)
        ),
    }


def _make_taichi_runtime(
    topology: PlushTopology,
    config: XPBDPlushConfig,
    *,
    initial_center_m: np.ndarray,
    table_height_m: float,
    maximum_boxes: int,
):
    import taichi as ti

    @ti.data_oriented
    class _TaichiXPBDRuntime:
        MAX_COLLIDER_PLANES = 64

        def __init__(self) -> None:
            self.n_particles = topology.particle_count
            self.n_shell = topology.shell_count
            self.n_grains = len(topology.filler_points)
            self.n_constraints = len(topology.constraint_pairs)
            self.n_faces = len(topology.shell_faces)
            self.shell_grain_patch_faces = int(
                topology.shell_face_contact_patches.shape[1]
            )
            self.maximum_boxes = maximum_boxes
            self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
            self.x_previous = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
            self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
            self.inverse_mass = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.table_support_offset = ti.field(
                dtype=ti.f32, shape=self.n_particles
            )
            self.pairs = ti.Vector.field(2, dtype=ti.i32, shape=self.n_constraints)
            self.rest_length = ti.field(dtype=ti.f32, shape=self.n_constraints)
            self.compliance = ti.field(dtype=ti.f32, shape=self.n_constraints)
            self.constraint_kind = ti.field(dtype=ti.i32, shape=self.n_constraints)
            self.lagrange = ti.field(dtype=ti.f32, shape=self.n_constraints)
            self.faces = ti.Vector.field(3, dtype=ti.i32, shape=self.n_faces)
            self.face_rest_normal = ti.Vector.field(
                3, dtype=ti.f32, shape=self.n_faces
            )
            self.shell_face_contact_patch = ti.field(
                dtype=ti.i32,
                shape=(self.n_faces, self.shell_grain_patch_faces),
            )
            self.grain_closest_face = ti.field(dtype=ti.i32, shape=self.n_grains)
            self.grain_inside_shell = ti.field(dtype=ti.i32, shape=self.n_grains)
            self.grain_boundary_face = ti.field(dtype=ti.i32, shape=self.n_grains)
            self.grain_boundary_violation = ti.field(
                dtype=ti.f32, shape=self.n_grains
            )
            self.volume_gradient = ti.Vector.field(
                3, dtype=ti.f32, shape=self.n_shell
            )
            self.volume_value = ti.field(dtype=ti.f32, shape=())
            self.volume_weight = ti.field(dtype=ti.f32, shape=())
            self.volume_lagrange = ti.field(dtype=ti.f32, shape=())
            self.shell_center = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.delta = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
            self.delta_count = ti.field(dtype=ti.i32, shape=self.n_particles)
            self.box_center = ti.Vector.field(3, dtype=ti.f32, shape=maximum_boxes)
            self.box_previous_center = ti.Vector.field(
                3, dtype=ti.f32, shape=maximum_boxes
            )
            self.box_rotation = ti.Matrix.field(
                3, 3, dtype=ti.f32, shape=maximum_boxes
            )
            self.collider_plane_count = ti.field(
                dtype=ti.i32, shape=maximum_boxes
            )
            self.collider_plane_normal = ti.Vector.field(
                3,
                dtype=ti.f32,
                shape=(maximum_boxes, self.MAX_COLLIDER_PLANES),
            )
            self.collider_plane_offset = ti.field(
                dtype=ti.f32,
                shape=(maximum_boxes, self.MAX_COLLIDER_PLANES),
            )
            self.collider_friction_scale = ti.field(
                dtype=ti.f32, shape=maximum_boxes
            )
            self.contact_active = ti.field(
                dtype=ti.i32, shape=(self.n_particles, maximum_boxes)
            )
            self.contact_anchor_local = ti.Vector.field(
                3, dtype=ti.f32, shape=(self.n_particles, maximum_boxes)
            )
            self.contact_preload = ti.field(
                dtype=ti.f32, shape=(self.n_particles, maximum_boxes)
            )
            self.frame_max_post_projection_penetration = ti.field(
                dtype=ti.f32, shape=()
            )
            self.peak_max_post_projection_penetration = ti.field(
                dtype=ti.f32, shape=()
            )
            self.frame_contact_count = ti.field(
                dtype=ti.i32, shape=maximum_boxes
            )
            self.peak_contact_count = ti.field(
                dtype=ti.i32, shape=maximum_boxes
            )
            self.contact_projection_events = ti.field(
                dtype=ti.i32, shape=maximum_boxes
            )
            self.peak_pre_projection_penetration = ti.field(
                dtype=ti.f32, shape=maximum_boxes
            )
            self.shell_grain_projection_events = ti.field(dtype=ti.i32, shape=())
            self.peak_shell_grain_pre_projection_penetration = ti.field(
                dtype=ti.f32, shape=()
            )
            points = np.concatenate(
                (topology.shell_vertices, topology.filler_points), axis=0
            ).astype(np.float32)
            points += initial_center_m.astype(np.float32)
            self.x.from_numpy(points)
            self.x_previous.from_numpy(points)
            inverse_mass = np.concatenate(
                (
                    np.full(topology.shell_count, config.shell_inverse_mass),
                    np.full(len(topology.filler_points), config.filler_inverse_mass),
                )
            ).astype(np.float32)
            self.inverse_mass.from_numpy(inverse_mass)
            support_offsets = np.zeros(self.n_particles, dtype=np.float32)
            if config.support_base_flatten_height_m > 0.0:
                shell_height_from_bottom = (
                    topology.shell_vertices[:, 2]
                    - float(topology.shell_vertices[:, 2].min())
                )
                support_band = (
                    shell_height_from_bottom
                    < config.support_base_flatten_height_m
                )
                # Give the curved lower band an effective flat contact sole:
                # each selected shell particle reaches down to the same rest
                # plane without altering the visible or watertight shell.
                support_offsets[: topology.shell_count][support_band] = (
                    shell_height_from_bottom[support_band]
                )
            self.table_support_offset.from_numpy(support_offsets)
            self.pairs.from_numpy(topology.constraint_pairs)
            self.rest_length.from_numpy(topology.rest_lengths_m)
            self.compliance.from_numpy(topology.compliances_m_n)
            self.constraint_kind.from_numpy(
                topology.constraint_kinds.astype(np.int32)
            )
            self.faces.from_numpy(topology.shell_faces)
            rest_face_normals, _rest_face_offsets = _convex_face_planes(
                topology.shell_vertices, topology.shell_faces
            )
            self.face_rest_normal.from_numpy(rest_face_normals)
            self.shell_face_contact_patch.from_numpy(
                topology.shell_face_contact_patches
            )
            triangles = topology.shell_vertices[topology.shell_faces]
            self.rest_volume = float(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                ).sum()
                / 6.0
            )
            if self.rest_volume <= 0.0:
                raise ValueError("XPBD shell faces must have positive outward volume")
            self.active_boxes = 0
            self.collider_friction_scale.fill(1.0)
            self.table_height_m = float(table_height_m)
            self._host_centers = np.empty((0, 3), dtype=np.float32)
            self._host_rotations = np.empty((0, 3, 3), dtype=np.float32)
            self._host_plane_normals = np.empty((0, 0, 3), dtype=np.float32)
            self._host_plane_offsets = np.empty((0, 0), dtype=np.float32)
            self._host_plane_counts = np.empty(0, dtype=np.int32)

        @ti.kernel
        def _predict(self, dt: ti.f32, gravity: ti.f32):
            for index in range(self.n_particles):
                self.x_previous[index] = self.x[index]
                self.velocity[index][2] += gravity * dt
                self.x[index] += self.velocity[index] * dt

        @ti.kernel
        def _clear_constraint_state(self):
            for index in range(self.n_constraints):
                self.lagrange[index] = 0.0
            self.volume_lagrange[None] = 0.0

        @ti.kernel
        def _compute_shell_center(self):
            self.shell_center[None] = ti.Vector([0.0, 0.0, 0.0])
            for index in range(self.n_shell):
                for axis in ti.static(range(3)):
                    ti.atomic_add(
                        self.shell_center[None][axis],
                        self.x[index][axis] / ti.cast(self.n_shell, ti.f32),
                    )

        @ti.kernel
        def _clear_deltas(self):
            for index in range(self.n_particles):
                self.delta[index] = ti.Vector([0.0, 0.0, 0.0])
                self.delta_count[index] = 0

        @ti.kernel
        def _project_distances(self, dt: ti.f32, grain_friction: ti.f32):
            for constraint in range(self.n_constraints):
                pair = self.pairs[constraint]
                first = pair[0]
                second = pair[1]
                difference = self.x[second] - self.x[first]
                length = difference.norm()
                if length > 1e-8:
                    direction = difference / length
                    value = length - self.rest_length[constraint]
                    kind = self.constraint_kind[constraint]
                    # Shell edges and bends are elastic equality constraints.
                    # Grain contacts and far-side anti-collapse braces are
                    # unilateral: they resist overlap but never pull particles
                    # together like foam or preserve volume like liquid.
                    if kind < 2 or value < 0.0:
                        alpha = self.compliance[constraint] / (dt * dt)
                        weight = self.inverse_mass[first] + self.inverse_mass[second]
                        # Jacobi corrections are averaged by vertex degree below.
                        # Keeping a full accumulated lambda would falsely report a
                        # correction that was never fully applied and makes the
                        # compliant network collapse under gravity.
                        delta_lambda = -value / (weight + alpha)
                        first_delta = (
                            -self.inverse_mass[first] * delta_lambda * direction
                        )
                        second_delta = (
                            self.inverse_mass[second] * delta_lambda * direction
                        )
                        if kind == 2 and value < 0.0:
                            # Position-based dry friction.  Remove relative
                            # tangential motion up to Coulomb's mu*N bound.
                            # Referencing the substep-start separation makes
                            # the correction converge across solver iterations
                            # without permanently welding grains together.
                            previous_difference = (
                                self.x_previous[second]
                                - self.x_previous[first]
                            )
                            relative_motion = difference - previous_difference
                            tangent = (
                                relative_motion
                                - direction * relative_motion.dot(direction)
                            )
                            tangent_length = tangent.norm()
                            if tangent_length > 1e-8:
                                friction_limit = grain_friction * (-value)
                                friction_scale = ti.min(
                                    1.0, friction_limit / tangent_length
                                )
                                tangent_correction = tangent * friction_scale
                                first_delta += (
                                    self.inverse_mass[first]
                                    / weight
                                    * tangent_correction
                                )
                                second_delta -= (
                                    self.inverse_mass[second]
                                    / weight
                                    * tangent_correction
                                )
                        for axis in ti.static(range(3)):
                            ti.atomic_add(self.delta[first][axis], first_delta[axis])
                            ti.atomic_add(self.delta[second][axis], second_delta[axis])
                        ti.atomic_add(self.delta_count[first], 1)
                        ti.atomic_add(self.delta_count[second], 1)

        @ti.kernel
        def _find_closest_shell_faces(self):
            """Refresh every grain's boundary neighbourhood on the GPU."""

            for local_grain in range(self.n_grains):
                grain_position = self.x[self.n_shell + local_grain]
                closest_face = 0
                closest_squared_distance = 1.0e20
                for face_index in range(self.n_faces):
                    face = self.faces[face_index]
                    first = self.x[face[0]]
                    second = self.x[face[1]]
                    third = self.x[face[2]]
                    edge_first = second - first
                    edge_second = third - first
                    from_first = grain_position - first
                    dot_first_first = edge_first.dot(edge_first)
                    dot_first_second = edge_first.dot(edge_second)
                    dot_second_second = edge_second.dot(edge_second)
                    dot_point_first = from_first.dot(edge_first)
                    dot_point_second = from_first.dot(edge_second)
                    denominator = (
                        dot_first_first * dot_second_second
                        - dot_first_second * dot_first_second
                    )
                    squared_distance = 1.0e20
                    if denominator > 1e-14:
                        barycentric_second = (
                            dot_second_second * dot_point_first
                            - dot_first_second * dot_point_second
                        ) / denominator
                        barycentric_third = (
                            dot_first_first * dot_point_second
                            - dot_first_second * dot_point_first
                        ) / denominator
                        barycentric_first = (
                            1.0 - barycentric_second - barycentric_third
                        )
                        if (
                            barycentric_first >= 0.0
                            and barycentric_second >= 0.0
                            and barycentric_third >= 0.0
                        ):
                            cross = edge_first.cross(edge_second)
                            cross_squared = cross.dot(cross)
                            if cross_squared > 1e-20:
                                signed_numerator = cross.dot(from_first)
                                squared_distance = (
                                    signed_numerator
                                    * signed_numerator
                                    / cross_squared
                                )
                    first_edge_squared = ti.max(dot_first_first, 1e-20)
                    first_parameter = ti.min(
                        1.0,
                        ti.max(0.0, dot_point_first / first_edge_squared),
                    )
                    first_edge_difference = (
                        grain_position
                        - (first + first_parameter * edge_first)
                    )
                    squared_distance = ti.min(
                        squared_distance,
                        first_edge_difference.dot(first_edge_difference),
                    )
                    edge_third = third - second
                    from_second = grain_position - second
                    second_edge_squared = ti.max(edge_third.dot(edge_third), 1e-20)
                    second_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_second.dot(edge_third) / second_edge_squared,
                        ),
                    )
                    second_edge_difference = (
                        grain_position
                        - (second + second_parameter * edge_third)
                    )
                    squared_distance = ti.min(
                        squared_distance,
                        second_edge_difference.dot(second_edge_difference),
                    )
                    edge_fourth = first - third
                    from_third = grain_position - third
                    third_edge_squared = ti.max(edge_fourth.dot(edge_fourth), 1e-20)
                    third_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_third.dot(edge_fourth) / third_edge_squared,
                        ),
                    )
                    third_edge_difference = (
                        grain_position
                        - (third + third_parameter * edge_fourth)
                    )
                    squared_distance = ti.min(
                        squared_distance,
                        third_edge_difference.dot(third_edge_difference),
                    )
                    if squared_distance < closest_squared_distance:
                        closest_squared_distance = squared_distance
                        closest_face = face_index
                self.grain_closest_face[local_grain] = closest_face

        @ti.kernel
        def _classify_grains_inside_shell(self):
            """Classify grains by generalized winding of the current shell."""

            for local_grain in range(self.n_grains):
                point = self.x[self.n_shell + local_grain]
                solid_angle = 0.0
                for face_index in range(self.n_faces):
                    face = self.faces[face_index]
                    first = self.x[face[0]] - point
                    second = self.x[face[1]] - point
                    third = self.x[face[2]] - point
                    first_length = first.norm()
                    second_length = second.norm()
                    third_length = third.norm()
                    numerator = first.dot(second.cross(third))
                    denominator = (
                        first_length * second_length * third_length
                        + first.dot(second) * third_length
                        + second.dot(third) * first_length
                        + third.dot(first) * second_length
                    )
                    solid_angle += 2.0 * ti.atan2(numerator, denominator)
                self.grain_inside_shell[local_grain] = (
                    1 if ti.abs(solid_angle) > 6.283185307179586 else 0
                )

        @ti.kernel
        def _find_most_violated_shell_halfspace(self, grain_radius: ti.f32):
            """Find the current convex proxy face most violated by each grain."""

            for local_grain in range(self.n_grains):
                point = self.x[self.n_shell + local_grain]
                maximum_violation = -1.0e10
                boundary_face = 0
                for face_index in range(self.n_faces):
                    face = self.faces[face_index]
                    first = self.x[face[0]]
                    second = self.x[face[1]]
                    third = self.x[face[2]]
                    cross = (second - first).cross(third - first)
                    cross_length = cross.norm()
                    if cross_length > 1e-10:
                        normal = cross / cross_length
                        face_center = (first + second + third) / 3.0
                        if normal.dot(face_center - self.shell_center[None]) < 0.0:
                            normal = -normal
                        violation = normal.dot(point - first) + grain_radius
                        if violation > maximum_violation:
                            maximum_violation = violation
                            boundary_face = face_index
                self.grain_boundary_face[local_grain] = boundary_face
                self.grain_boundary_violation[local_grain] = maximum_violation

        @ti.kernel
        def _stabilize_shell_halfspace_barrier(self, grain_radius: ti.f32):
            """Project grains into the intersection of current shell halfspaces."""

            for local_grain in range(self.n_grains):
                violation = self.grain_boundary_violation[local_grain]
                if violation > 0.0:
                    grain = self.n_shell + local_grain
                    face = self.faces[self.grain_boundary_face[local_grain]]
                    first = self.x[face[0]]
                    second = self.x[face[1]]
                    third = self.x[face[2]]
                    cross = (second - first).cross(third - first)
                    cross_length = cross.norm()
                    if cross_length > 1e-10:
                        normal = cross / cross_length
                        face_center = (first + second + third) / 3.0
                        if normal.dot(face_center - self.shell_center[None]) < 0.0:
                            normal = -normal
                        correction_length = ti.min(violation, 2.0 * grain_radius)
                        correction = -correction_length * normal
                        self.x[grain] += correction
                        self.x_previous[grain] += correction

        @ti.kernel
        def _project_shell_grain_contacts(
            self,
            dt: ti.f32,
            grain_radius: ti.f32,
            compliance: ti.f32,
            friction: ti.f32,
            relaxation: ti.f32,
            shell_reaction_scale: ti.f32,
        ):
            """Keep independent hard grains inside the moving triangle shell.

            Each grain owns a conservative local set of rest-pose boundary
            triangles.  The constraint is unilateral and recomputes the face
            plane from current shell positions, so it transfers load to the
            fabric without a spring, tether, or fixed grain-to-shell offset.
            """

            for candidate in range(
                self.n_grains * self.shell_grain_patch_faces
            ):
                local_grain = candidate // self.shell_grain_patch_faces
                patch_slot = candidate % self.shell_grain_patch_faces
                closest_face = self.grain_closest_face[local_grain]
                face_index = self.shell_face_contact_patch[
                    closest_face, patch_slot
                ]
                grain = self.n_shell + local_grain
                face = self.faces[face_index]
                first = face[0]
                second = face[1]
                third = face[2]
                first_position = self.x[first]
                second_position = self.x[second]
                third_position = self.x[third]
                cross = (second_position - first_position).cross(
                    third_position - first_position
                )
                cross_length = cross.norm()
                if cross_length > 1e-10:
                    normal = cross / cross_length
                    face_center = (
                        first_position + second_position + third_position
                    ) / 3.0
                    if normal.dot(face_center - self.shell_center[None]) < 0.0:
                        normal = -normal
                    grain_position = self.x[grain]
                    signed_distance = normal.dot(grain_position - first_position)
                    edge_first = second_position - first_position
                    edge_second = third_position - first_position
                    from_first = grain_position - first_position
                    dot_first_first = edge_first.dot(edge_first)
                    dot_first_second = edge_first.dot(edge_second)
                    dot_second_second = edge_second.dot(edge_second)
                    dot_point_first = from_first.dot(edge_first)
                    dot_point_second = from_first.dot(edge_second)
                    barycentric_denominator = (
                        dot_first_first * dot_second_second
                        - dot_first_second * dot_first_second
                    )
                    barycentric_second = 0.0
                    barycentric_third = 0.0
                    barycentric_first = 0.0
                    closest_squared_distance = 1.0e20
                    if barycentric_denominator > 1e-14:
                        projected_second = (
                            dot_second_second * dot_point_first
                            - dot_first_second * dot_point_second
                        ) / barycentric_denominator
                        projected_third = (
                            dot_first_first * dot_point_second
                            - dot_first_second * dot_point_first
                        ) / barycentric_denominator
                        projected_first = (
                            1.0 - projected_second - projected_third
                        )
                        if (
                            projected_first >= 0.0
                            and projected_second >= 0.0
                            and projected_third >= 0.0
                        ):
                            barycentric_first = projected_first
                            barycentric_second = projected_second
                            barycentric_third = projected_third
                            closest_squared_distance = (
                                signed_distance * signed_distance
                            )

                    first_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            dot_point_first / ti.max(dot_first_first, 1e-20),
                        ),
                    )
                    first_closest = first_position + first_parameter * edge_first
                    first_difference = grain_position - first_closest
                    first_squared_distance = first_difference.dot(first_difference)
                    if first_squared_distance < closest_squared_distance:
                        closest_squared_distance = first_squared_distance
                        barycentric_first = 1.0 - first_parameter
                        barycentric_second = first_parameter
                        barycentric_third = 0.0

                    edge_third = third_position - second_position
                    from_second = grain_position - second_position
                    second_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_second.dot(edge_third)
                            / ti.max(edge_third.dot(edge_third), 1e-20),
                        ),
                    )
                    second_closest = second_position + second_parameter * edge_third
                    second_difference = grain_position - second_closest
                    second_squared_distance = second_difference.dot(second_difference)
                    if second_squared_distance < closest_squared_distance:
                        closest_squared_distance = second_squared_distance
                        barycentric_first = 0.0
                        barycentric_second = 1.0 - second_parameter
                        barycentric_third = second_parameter

                    edge_fourth = first_position - third_position
                    from_third = grain_position - third_position
                    third_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_third.dot(edge_fourth)
                            / ti.max(edge_fourth.dot(edge_fourth), 1e-20),
                        ),
                    )
                    third_closest = third_position + third_parameter * edge_fourth
                    third_difference = grain_position - third_closest
                    third_squared_distance = third_difference.dot(third_difference)
                    if third_squared_distance < closest_squared_distance:
                        closest_squared_distance = third_squared_distance
                        barycentric_first = third_parameter
                        barycentric_second = 0.0
                        barycentric_third = 1.0 - third_parameter

                    closest_position = (
                        first_position * barycentric_first
                        + second_position * barycentric_second
                        + third_position * barycentric_third
                    )
                    separation = grain_position - closest_position
                    distance = ti.sqrt(ti.max(closest_squared_distance, 0.0))
                    penetration = grain_radius - distance
                    if penetration > 0.0:
                        ti.atomic_add(self.shell_grain_projection_events[None], 1)
                        ti.atomic_max(
                            self.peak_shell_grain_pre_projection_penetration[None],
                            penetration,
                        )
                        face_weight = shell_reaction_scale * (
                            self.inverse_mass[first]
                            * barycentric_first
                            * barycentric_first
                            + self.inverse_mass[second]
                            * barycentric_second
                            * barycentric_second
                            + self.inverse_mass[third]
                            * barycentric_third
                            * barycentric_third
                        )
                        weight = self.inverse_mass[grain] + face_weight
                        alpha = compliance / (dt * dt)
                        delta_lambda = (
                            relaxation * penetration / (weight + alpha)
                        )
                        contact_direction = -normal
                        if signed_distance < 0.0 and distance > 1e-8:
                            # For an inside grain, the finite-triangle closest
                            # vector points away from the shell even at an edge
                            # or vertex.  This closes the gaps left by an
                            # infinite-plane-only contact model.
                            contact_direction = separation / distance
                        grain_delta = (
                            self.inverse_mass[grain]
                            * delta_lambda
                            * contact_direction
                        )
                        first_delta = (
                            -self.inverse_mass[first]
                            * delta_lambda
                            * contact_direction
                            * barycentric_first
                            * shell_reaction_scale
                        )
                        second_delta = (
                            -self.inverse_mass[second]
                            * delta_lambda
                            * contact_direction
                            * barycentric_second
                            * shell_reaction_scale
                        )
                        third_delta = (
                            -self.inverse_mass[third]
                            * delta_lambda
                            * contact_direction
                            * barycentric_third
                            * shell_reaction_scale
                        )

                        current_face_center = (
                            first_position * barycentric_first
                            + second_position * barycentric_second
                            + third_position * barycentric_third
                        )
                        previous_face_center = (
                            self.x_previous[first] * barycentric_first
                            + self.x_previous[second] * barycentric_second
                            + self.x_previous[third] * barycentric_third
                        )
                        relative_motion = (
                            grain_position
                            - current_face_center
                            - self.x_previous[grain]
                            + previous_face_center
                        )
                        tangent = (
                            relative_motion
                            - contact_direction
                            * relative_motion.dot(contact_direction)
                        )
                        tangent_length = tangent.norm()
                        if tangent_length > 1e-8:
                            friction_limit = friction * penetration
                            friction_scale = ti.min(
                                1.0, friction_limit / tangent_length
                            )
                            tangent_correction = tangent * friction_scale
                            grain_delta -= (
                                self.inverse_mass[grain]
                                / weight
                                * tangent_correction
                            )
                            first_delta += (
                                self.inverse_mass[first]
                                * barycentric_first
                                / weight
                                * tangent_correction
                                * shell_reaction_scale
                            )
                            second_delta += (
                                self.inverse_mass[second]
                                * barycentric_second
                                / weight
                                * tangent_correction
                                * shell_reaction_scale
                            )
                            third_delta += (
                                self.inverse_mass[third]
                                * barycentric_third
                                / weight
                                * tangent_correction
                                * shell_reaction_scale
                            )
                        for axis in ti.static(range(3)):
                            ti.atomic_add(
                                self.delta[grain][axis], grain_delta[axis]
                            )
                            ti.atomic_add(
                                self.delta[first][axis], first_delta[axis]
                            )
                            ti.atomic_add(
                                self.delta[second][axis], second_delta[axis]
                            )
                            ti.atomic_add(
                                self.delta[third][axis], third_delta[axis]
                            )
                        ti.atomic_add(self.delta_count[grain], 1)
                        ti.atomic_add(self.delta_count[first], 1)
                        ti.atomic_add(self.delta_count[second], 1)
                        ti.atomic_add(self.delta_count[third], 1)

        @ti.kernel
        def _stabilize_closest_shell_contact(self, grain_radius: ti.f32):
            """Apply one direct grain-side finite-triangle nonpenetration pass."""

            for local_grain in range(self.n_grains):
                grain = self.n_shell + local_grain
                face_index = self.grain_closest_face[local_grain]
                face = self.faces[face_index]
                first_position = self.x[face[0]]
                second_position = self.x[face[1]]
                third_position = self.x[face[2]]
                grain_position = self.x[grain]
                edge_first = second_position - first_position
                edge_second = third_position - first_position
                cross = edge_first.cross(edge_second)
                cross_length = cross.norm()
                if cross_length > 1e-10:
                    normal = cross / cross_length
                    face_center = (
                        first_position + second_position + third_position
                    ) / 3.0
                    if normal.dot(face_center - self.shell_center[None]) < 0.0:
                        normal = -normal
                    from_first = grain_position - first_position
                    signed_distance = normal.dot(from_first)
                    dot_first_first = edge_first.dot(edge_first)
                    dot_first_second = edge_first.dot(edge_second)
                    dot_second_second = edge_second.dot(edge_second)
                    dot_point_first = from_first.dot(edge_first)
                    dot_point_second = from_first.dot(edge_second)
                    denominator = (
                        dot_first_first * dot_second_second
                        - dot_first_second * dot_first_second
                    )
                    barycentric_first = 0.0
                    barycentric_second = 0.0
                    barycentric_third = 0.0
                    closest_squared_distance = 1.0e20
                    if denominator > 1e-14:
                        projected_second = (
                            dot_second_second * dot_point_first
                            - dot_first_second * dot_point_second
                        ) / denominator
                        projected_third = (
                            dot_first_first * dot_point_second
                            - dot_first_second * dot_point_first
                        ) / denominator
                        projected_first = 1.0 - projected_second - projected_third
                        if (
                            projected_first >= 0.0
                            and projected_second >= 0.0
                            and projected_third >= 0.0
                        ):
                            barycentric_first = projected_first
                            barycentric_second = projected_second
                            barycentric_third = projected_third
                            closest_squared_distance = (
                                signed_distance * signed_distance
                            )

                    first_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            dot_point_first / ti.max(dot_first_first, 1e-20),
                        ),
                    )
                    first_closest = first_position + first_parameter * edge_first
                    first_difference = grain_position - first_closest
                    first_squared_distance = first_difference.dot(first_difference)
                    if first_squared_distance < closest_squared_distance:
                        closest_squared_distance = first_squared_distance
                        barycentric_first = 1.0 - first_parameter
                        barycentric_second = first_parameter
                        barycentric_third = 0.0

                    edge_third = third_position - second_position
                    from_second = grain_position - second_position
                    second_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_second.dot(edge_third)
                            / ti.max(edge_third.dot(edge_third), 1e-20),
                        ),
                    )
                    second_closest = second_position + second_parameter * edge_third
                    second_difference = grain_position - second_closest
                    second_squared_distance = second_difference.dot(second_difference)
                    if second_squared_distance < closest_squared_distance:
                        closest_squared_distance = second_squared_distance
                        barycentric_first = 0.0
                        barycentric_second = 1.0 - second_parameter
                        barycentric_third = second_parameter

                    edge_fourth = first_position - third_position
                    from_third = grain_position - third_position
                    third_parameter = ti.min(
                        1.0,
                        ti.max(
                            0.0,
                            from_third.dot(edge_fourth)
                            / ti.max(edge_fourth.dot(edge_fourth), 1e-20),
                        ),
                    )
                    third_closest = third_position + third_parameter * edge_fourth
                    third_difference = grain_position - third_closest
                    third_squared_distance = third_difference.dot(third_difference)
                    if third_squared_distance < closest_squared_distance:
                        closest_squared_distance = third_squared_distance
                        barycentric_first = third_parameter
                        barycentric_second = 0.0
                        barycentric_third = 1.0 - third_parameter

                    closest_position = (
                        first_position * barycentric_first
                        + second_position * barycentric_second
                        + third_position * barycentric_third
                    )
                    correction = ti.Vector([0.0, 0.0, 0.0])
                    needs_correction = False
                    separation = grain_position - closest_position
                    distance = ti.sqrt(ti.max(closest_squared_distance, 0.0))
                    inward = self.shell_center[None] - closest_position
                    inward_length = inward.norm()
                    if inward_length > 1e-8:
                        inward_direction = inward / inward_length
                        if self.grain_inside_shell[local_grain] == 0:
                            # Ray parity supplies the global inside/outside
                            # decision.  The star-shaped proxy centre only
                            # supplies a safe inward correction direction.
                            correction = (
                                closest_position
                                + grain_radius * inward_direction
                                - grain_position
                            )
                            needs_correction = True
                            self.grain_inside_shell[local_grain] = 1
                        elif distance < grain_radius:
                            direction = inward_direction
                            if distance > 1e-8:
                                direction = separation / distance
                            correction = (grain_radius - distance) * direction
                            needs_correction = True
                        correction_length = correction.norm()
                        maximum_correction = 2.0 * grain_radius
                        if correction_length > maximum_correction:
                            correction *= maximum_correction / correction_length
                    if needs_correction:
                        self.x[grain] += correction
                        # This is a geometric post-stabilization after the
                        # symmetric contact impulse.  Shifting the substep
                        # reference by the same amount prevents the correction
                        # from becoming a non-physical launch velocity.
                        self.x_previous[grain] += correction

        @ti.kernel
        def _apply_deltas(self):
            for index in range(self.n_particles):
                count = self.delta_count[index]
                if count > 0:
                    self.x[index] += self.delta[index] / ti.cast(count, ti.f32)

        @ti.kernel
        def _clear_volume_projection(self):
            self.volume_value[None] = 0.0
            self.volume_weight[None] = 0.0
            for index in range(self.n_shell):
                self.volume_gradient[index] = ti.Vector([0.0, 0.0, 0.0])

        @ti.kernel
        def _accumulate_volume_and_gradient(self):
            for face_index in range(self.n_faces):
                face = self.faces[face_index]
                first = self.x[face[0]]
                second = self.x[face[1]]
                third = self.x[face[2]]
                ti.atomic_add(
                    self.volume_value[None], first.dot(second.cross(third)) / 6.0
                )
                gradients = ti.Matrix.rows(
                    [
                        second.cross(third) / 6.0,
                        third.cross(first) / 6.0,
                        first.cross(second) / 6.0,
                    ]
                )
                for corner in ti.static(range(3)):
                    vertex = face[corner]
                    for axis in ti.static(range(3)):
                        ti.atomic_add(
                            self.volume_gradient[vertex][axis],
                            gradients[corner, axis],
                        )

        @ti.kernel
        def _accumulate_volume_weight(self):
            for index in range(self.n_shell):
                gradient = self.volume_gradient[index]
                ti.atomic_add(
                    self.volume_weight[None],
                    self.inverse_mass[index] * gradient.dot(gradient),
                )

        @ti.kernel
        def _apply_volume_projection(
            self, dt: ti.f32, minimum_volume: ti.f32, compliance: ti.f32
        ):
            # A sealed, tightly stuffed shell can lose some volume by packing
            # its grains, but it cannot collapse through itself.  This is a
            # one-sided lower-volume barrier, not the equality constraint that
            # made earlier prototypes behave like incompressible water balls.
            value = self.volume_value[None] - minimum_volume
            if value < 0.0:
                alpha = compliance / (dt * dt)
                delta_lambda = (
                    -value - alpha * self.volume_lagrange[None]
                ) / (self.volume_weight[None] + alpha)
                self.volume_lagrange[None] += delta_lambda
                for index in range(self.n_shell):
                    self.x[index] += (
                        self.inverse_mass[index]
                        * self.volume_gradient[index]
                        * delta_lambda
                    )

        @ti.kernel
        def _project_contacts(
            self,
            active_boxes: ti.i32,
            particle_radius: ti.f32,
            friction: ti.f32,
            table_friction: ti.f32,
            contact_slop: ti.f32,
            contact_release: ti.f32,
            table_height: ti.f32,
        ):
            # External rigid bodies contact the fabric shell only.  Letting a
            # jaw or the table project hidden filler particles directly means
            # the rigid collider passes through the cloth boundary and drives
            # the whole fill like a fluid volume.
            for index in range(self.n_shell):
                point = self.x[index]
                previous = self.x_previous[index]
                floor = (
                    table_height
                    + particle_radius
                    + self.table_support_offset[index]
                )
                if point[2] < floor:
                    penetration = floor - point[2]
                    point[2] = floor
                    tangent = ti.Vector([point[0] - previous[0], point[1] - previous[1]])
                    tangent_norm = tangent.norm()
                    if tangent_norm > 1e-8:
                        scale = ti.min(
                            1.0, table_friction * penetration / tangent_norm
                        )
                        point[0] -= tangent[0] * scale
                        point[1] -= tangent[1] * scale
                for box in range(active_boxes):
                    rotation = self.box_rotation[box]
                    local = rotation.transpose() @ (point - self.box_center[box])
                    distance = -1.0e10
                    local_normal = ti.Vector([1.0, 0.0, 0.0])
                    plane_count = self.collider_plane_count[box]
                    for plane in range(plane_count):
                        candidate = (
                            self.collider_plane_normal[box, plane].dot(local)
                            - self.collider_plane_offset[box, plane]
                            - particle_radius
                        )
                        if candidate > distance:
                            distance = candidate
                            local_normal = self.collider_plane_normal[box, plane]
                    near_contact = distance < contact_release
                    if near_contact:
                        normal = rotation @ local_normal
                        touching = distance < contact_slop
                        if touching or self.contact_active[index, box] != 0:
                            if self.contact_active[index, box] == 0:
                                self.contact_active[index, box] = 1
                                self.contact_anchor_local[index, box] = local
                                self.contact_preload[index, box] = contact_slop
                            penetration = ti.max(0.0, -distance)
                            if penetration > 0.0:
                                ti.atomic_add(self.contact_projection_events[box], 1)
                                ti.atomic_max(
                                    self.peak_pre_projection_penetration[box],
                                    penetration,
                                )
                            point += normal * penetration
                            # The commanded jaw closure supplies the squeeze
                            # load represented by this persistent positional
                            # preload.  A fully open jaw keeps unilateral
                            # normal collision but has zero tangential load,
                            # preventing the solver state from acting as glue.
                            preload = ti.min(
                                contact_release,
                                ti.max(
                                    contact_slop,
                                    self.contact_preload[index, box] * 0.999
                                    + penetration,
                                ),
                            )
                            self.contact_preload[index, box] = preload
                            anchor_world = (
                                rotation @ self.contact_anchor_local[index, box]
                                + self.box_center[box]
                            )
                            anchor_error = anchor_world - point
                            tangent_error = (
                                anchor_error - normal * anchor_error.dot(normal)
                            )
                            tangent_norm = tangent_error.norm()
                            if tangent_norm > 1e-8:
                                scale = ti.min(
                                    1.0,
                                    friction
                                    * self.collider_friction_scale[box]
                                    * preload
                                    / tangent_norm,
                                )
                                point += tangent_error * scale
                                if scale < 0.999:
                                    # Dynamic slip consumes the static anchor;
                                    # the next iteration starts from the new
                                    # local contact point instead of tethering.
                                    self.contact_anchor_local[index, box] = (
                                        rotation.transpose()
                                        @ (point - self.box_center[box])
                                    )
                    else:
                        self.contact_active[index, box] = 0
                        self.contact_preload[index, box] = 0.0
                self.x[index] = point

        @ti.kernel
        def _update_velocity(
            self, dt: ti.f32, retention: ti.f32, maximum_grain_speed: ti.f32
        ):
            # Constraint projections alter both shell and fill positions.  Not
            # updating grain velocities here accumulates gravity forever and
            # eventually tunnels the fill through any unilateral boundary.
            for index in range(self.n_particles):
                self.velocity[index] = (
                    (self.x[index] - self.x_previous[index]) / dt * retention
                )
                if index >= self.n_shell:
                    speed = self.velocity[index].norm()
                    if speed > maximum_grain_speed:
                        self.velocity[index] *= maximum_grain_speed / speed

        @ti.kernel
        def _measure_post_projection_contacts(
            self,
            active_boxes: ti.i32,
            particle_radius: ti.f32,
            contact_release: ti.f32,
        ):
            self.frame_max_post_projection_penetration[None] = 0.0
            for box in range(self.maximum_boxes):
                self.frame_contact_count[box] = 0
            # Only the fabric shell is eligible for rigid-jaw contact.  Hidden
            # fill particles must not create false gripper-contact evidence.
            for index in range(self.n_shell):
                point = self.x[index]
                for box in range(active_boxes):
                    rotation = self.box_rotation[box]
                    local = rotation.transpose() @ (point - self.box_center[box])
                    signed_distance = -1.0e10
                    plane_count = self.collider_plane_count[box]
                    for plane in range(plane_count):
                        candidate = (
                            self.collider_plane_normal[box, plane].dot(local)
                            - self.collider_plane_offset[box, plane]
                            - particle_radius
                        )
                        signed_distance = ti.max(signed_distance, candidate)
                    penetration = ti.max(0.0, -signed_distance)
                    ti.atomic_max(
                        self.frame_max_post_projection_penetration[None], penetration
                    )
                    ti.atomic_max(
                        self.peak_max_post_projection_penetration[None], penetration
                    )
                    if signed_distance <= contact_release:
                        ti.atomic_add(self.frame_contact_count[box], 1)

        @ti.kernel
        def _accumulate_peak_contact_counts(self):
            for box in range(self.maximum_boxes):
                ti.atomic_max(
                    self.peak_contact_count[box], self.frame_contact_count[box]
                )

        @ti.kernel
        def _translate(self, offset: ti.types.vector(3, ti.f32)):
            for index in range(self.n_particles):
                self.x[index] += offset
                self.x_previous[index] += offset

        @ti.kernel
        def _zero_velocity(self):
            for index in range(self.n_particles):
                self.velocity[index] = ti.Vector([0.0, 0.0, 0.0])

        def set_boxes(
            self,
            centers: np.ndarray,
            rotations: np.ndarray,
            half_extents: np.ndarray,
        ) -> None:
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
            half_extents = np.asarray(half_extents, dtype=np.float32).reshape(-1, 3)
            if not (len(centers) == len(rotations) == len(half_extents)):
                raise ValueError("gripper box arrays must have equal lengths")
            if len(centers) > self.maximum_boxes:
                raise ValueError("too many gripper boxes for XPBD runtime")
            normals = np.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ),
                dtype=np.float32,
            )
            plane_normals = np.repeat(normals[None], len(centers), axis=0)
            plane_offsets = np.column_stack(
                (
                    half_extents[:, 0],
                    half_extents[:, 0],
                    half_extents[:, 1],
                    half_extents[:, 1],
                    half_extents[:, 2],
                    half_extents[:, 2],
                )
            ).astype(np.float32)
            self.set_convex_colliders(
                centers,
                rotations,
                plane_normals,
                plane_offsets,
                np.full(len(centers), 6, dtype=np.int32),
            )

        def set_convex_colliders(
            self,
            centers: np.ndarray,
            rotations: np.ndarray,
            plane_normals: np.ndarray,
            plane_offsets: np.ndarray,
            plane_counts: np.ndarray,
        ) -> None:
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
            plane_normals = np.asarray(plane_normals, dtype=np.float32)
            plane_offsets = np.asarray(plane_offsets, dtype=np.float32)
            plane_counts = np.asarray(plane_counts, dtype=np.int32).reshape(-1)
            if not (
                len(centers)
                == len(rotations)
                == len(plane_normals)
                == len(plane_offsets)
                == len(plane_counts)
            ):
                raise ValueError("convex collider arrays must have equal lengths")
            if len(centers) > self.maximum_boxes:
                raise ValueError("too many convex colliders for XPBD runtime")
            if plane_normals.ndim != 3 or plane_normals.shape[2] != 3:
                raise ValueError("collider plane normals must have shape BxPx3")
            if plane_offsets.shape != plane_normals.shape[:2]:
                raise ValueError("collider plane offsets must match plane normals")
            if np.any(plane_counts < 4) or np.any(
                plane_counts > self.MAX_COLLIDER_PLANES
            ):
                raise ValueError("convex collider plane count is out of range")
            normal_lengths = np.linalg.norm(plane_normals, axis=2)
            active_mask = (
                np.arange(plane_normals.shape[1])[None, :] < plane_counts[:, None]
            )
            if not np.allclose(normal_lengths[active_mask], 1.0, atol=2e-4):
                raise ValueError("convex collider plane normals must be unit length")
            old = self.box_center.to_numpy()
            padded_centers = np.zeros((self.maximum_boxes, 3), dtype=np.float32)
            padded_rotations = np.repeat(
                np.eye(3, dtype=np.float32)[None], self.maximum_boxes, axis=0
            )
            padded_normals = np.zeros(
                (self.maximum_boxes, self.MAX_COLLIDER_PLANES, 3), dtype=np.float32
            )
            padded_offsets = np.zeros(
                (self.maximum_boxes, self.MAX_COLLIDER_PLANES), dtype=np.float32
            )
            padded_counts = np.zeros(self.maximum_boxes, dtype=np.int32)
            padded_centers[: len(centers)] = centers
            padded_rotations[: len(rotations)] = rotations
            for index, count in enumerate(plane_counts):
                padded_normals[index, :count] = plane_normals[index, :count]
                padded_offsets[index, :count] = plane_offsets[index, :count]
            padded_counts[: len(plane_counts)] = plane_counts
            if self.active_boxes == 0:
                old = padded_centers.copy()
            self.box_previous_center.from_numpy(old)
            self.box_center.from_numpy(padded_centers)
            self.box_rotation.from_numpy(padded_rotations)
            self.collider_plane_normal.from_numpy(padded_normals)
            self.collider_plane_offset.from_numpy(padded_offsets)
            self.collider_plane_count.from_numpy(padded_counts)
            self.active_boxes = len(centers)
            self._host_centers = centers.copy()
            self._host_rotations = rotations.copy()
            self._host_plane_normals = plane_normals.copy()
            self._host_plane_offsets = plane_offsets.copy()
            self._host_plane_counts = plane_counts.copy()

        def set_collider_friction_scales(self, scales: np.ndarray) -> None:
            values = np.asarray(scales, dtype=np.float32).reshape(-1)
            if len(values) != self.active_boxes:
                raise ValueError(
                    "collider friction scale count must match active colliders"
                )
            if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(
                values > 1.0
            ):
                raise ValueError("collider friction scales must be finite in [0, 1]")
            padded = np.ones(self.maximum_boxes, dtype=np.float32)
            padded[: len(values)] = values
            self.collider_friction_scale.from_numpy(padded)

        def step(self) -> None:
            substep_dt = config.dt_s / config.substeps
            for _ in range(config.substeps):
                self._predict(substep_dt, config.gravity_m_s2)
                self._compute_shell_center()
                self._find_closest_shell_faces()
                self._clear_constraint_state()
                for _ in range(config.solver_iterations):
                    self._clear_deltas()
                    self._project_distances(substep_dt, config.grain_friction)
                    self._apply_deltas()
                    self._clear_deltas()
                    self._project_shell_grain_contacts(
                        substep_dt,
                        config.grain_contact_distance_m * 0.5,
                        config.shell_grain_compliance_m_n,
                        config.shell_grain_friction,
                        config.shell_grain_contact_relaxation,
                        1.0,
                    )
                    self._apply_deltas()
                    if config.shell_volume_constraint_enabled:
                        self._clear_volume_projection()
                        self._accumulate_volume_and_gradient()
                        self._accumulate_volume_weight()
                        self._apply_volume_projection(
                            substep_dt,
                            self.rest_volume * config.shell_minimum_volume_ratio,
                            config.shell_volume_compliance_m3_n,
                        )
                    self._project_contacts(
                        self.active_boxes,
                        config.particle_radius_m,
                        config.contact_friction,
                        config.table_friction,
                        config.contact_slop_m,
                        config.contact_release_m,
                        self.table_height_m,
                    )
                # Refresh the global closest surface after all shell and rigid
                # contact motion, then perform a direct grain-side collision
                # post-process.  The symmetric iterations above carry the
                # reaction force; this only removes residual interpenetration.
                for _ in range(config.shell_grain_final_stabilization_passes):
                    self._compute_shell_center()
                    self._find_most_violated_shell_halfspace(
                        config.grain_contact_distance_m * 0.5
                    )
                    self._stabilize_shell_halfspace_barrier(
                        config.grain_contact_distance_m * 0.5
                    )
                self._measure_post_projection_contacts(
                    self.active_boxes,
                    config.particle_radius_m,
                    config.contact_release_m,
                )
                self._accumulate_peak_contact_counts()
                self._update_velocity(
                    substep_dt,
                    config.velocity_retention_per_substep,
                    config.maximum_grain_speed_m_s,
                )

        def positions(self) -> np.ndarray:
            return self.x.to_numpy()

        def signed_distances(
            self, points: np.ndarray, *, radius_m: float
        ) -> np.ndarray:
            """Return point-to-expanded-convex signed distances, N by B."""

            values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            distances = np.empty((len(values), self.active_boxes), dtype=np.float32)
            for box in range(self.active_boxes):
                local = (
                    values - self._host_centers[box]
                ) @ self._host_rotations[box]
                count = int(self._host_plane_counts[box])
                scores = (
                    local @ self._host_plane_normals[box, :count].T
                    - self._host_plane_offsets[box, :count]
                    - float(radius_m)
                )
                distances[:, box] = np.max(scores, axis=1)
            return distances

        def contact_diagnostics(self) -> dict[str, object]:
            return {
                "active_colliders": int(self.active_boxes),
                "friction_scale_by_collider": self.collider_friction_scale.to_numpy()[
                    : self.active_boxes
                ].tolist(),
                "frame_max_post_projection_penetration_m": float(
                    self.frame_max_post_projection_penetration[None]
                ),
                "peak_max_post_projection_penetration_m": float(
                    self.peak_max_post_projection_penetration[None]
                ),
                "frame_particle_contact_count_by_collider": self.frame_contact_count.to_numpy()[
                    : self.active_boxes
                ].tolist(),
                "peak_particle_contact_count_by_collider": self.peak_contact_count.to_numpy()[
                    : self.active_boxes
                ].tolist(),
                "contact_projection_events_by_collider": self.contact_projection_events.to_numpy()[
                    : self.active_boxes
                ].tolist(),
                "peak_pre_projection_penetration_m_by_collider": self.peak_pre_projection_penetration.to_numpy()[
                    : self.active_boxes
                ].tolist(),
                "shell_grain_contact_model": "moving_triangle_unilateral_with_dry_friction",
                "shell_grain_neighbourhood": "dynamic_full_surface_nearest_face_plus_local_patch",
                "shell_grain_contact_patch_faces": int(
                    self.shell_grain_patch_faces
                ),
                "shell_grain_projection_events": int(
                    self.shell_grain_projection_events[None]
                ),
                "peak_shell_grain_pre_projection_penetration_m": float(
                    self.peak_shell_grain_pre_projection_penetration[None]
                ),
            }

        def frame_contact_counts(self) -> np.ndarray:
            """Return current shell-contact counts for active colliders only."""

            return self.frame_contact_count.to_numpy()[: self.active_boxes].copy()

        def measure_contacts(self) -> None:
            self._measure_post_projection_contacts(
                self.active_boxes,
                config.particle_radius_m,
                config.contact_release_m,
            )
            self._accumulate_peak_contact_counts()

        def translate(self, offset: np.ndarray) -> None:
            self._translate(np.asarray(offset, dtype=np.float32).reshape(3))

        def zero_velocity(self) -> None:
            self._zero_velocity()

        def reset(self, points: np.ndarray) -> None:
            points = np.asarray(points, dtype=np.float32).reshape(
                self.n_particles, 3
            )
            self.x.from_numpy(points)
            self.x_previous.from_numpy(points)
            self.velocity.fill(0.0)
            self.contact_active.fill(0)
            self.contact_preload.fill(0.0)
            self.frame_max_post_projection_penetration.fill(0.0)
            self.peak_max_post_projection_penetration.fill(0.0)
            self.frame_contact_count.fill(0)
            self.peak_contact_count.fill(0)
            self.contact_projection_events.fill(0)
            self.peak_pre_projection_penetration.fill(0.0)
            self.shell_grain_projection_events.fill(0)
            self.peak_shell_grain_pre_projection_penetration.fill(0.0)

    return _TaichiXPBDRuntime()


class XPBDPlushSolver:
    """Thin host wrapper around the lazily-created Taichi/Vulkan runtime."""

    def __init__(
        self,
        topology: PlushTopology,
        config: XPBDPlushConfig = XPBDPlushConfig(),
        *,
        initial_center_m: Iterable[float] = (0.0, 0.0, 0.0),
        table_height_m: float = 0.0,
        maximum_boxes: int = 4,
        initialize_vulkan: bool = True,
    ):
        config.validate()
        if maximum_boxes < 1:
            raise ValueError("XPBD runtime needs at least one collider slot")
        import taichi as ti

        if initialize_vulkan:
            runtime = ti.lang.impl.get_runtime()
            if runtime.prog is None:
                ti.init(arch=ti.vulkan, offline_cache=False)
        self.ti = ti
        self.topology = topology
        self.config = config
        self.table_height_m = float(table_height_m)
        self.runtime = _make_taichi_runtime(
            topology,
            config,
            initial_center_m=np.asarray(tuple(initial_center_m), dtype=np.float32),
            table_height_m=table_height_m,
            maximum_boxes=maximum_boxes,
        )

    def set_boxes(
        self,
        centers: np.ndarray,
        rotations: np.ndarray,
        half_extents: np.ndarray,
    ) -> None:
        self.runtime.set_boxes(centers, rotations, half_extents)

    def set_convex_colliders(
        self,
        centers: np.ndarray,
        rotations: np.ndarray,
        plane_normals: np.ndarray,
        plane_offsets: np.ndarray,
        plane_counts: np.ndarray,
    ) -> None:
        self.runtime.set_convex_colliders(
            centers, rotations, plane_normals, plane_offsets, plane_counts
        )

    def set_collider_friction_scales(self, scales: np.ndarray) -> None:
        self.runtime.set_collider_friction_scales(scales)

    def step(self, *, synchronize: bool = False) -> None:
        self.runtime.step()
        if synchronize:
            self.ti.sync()

    def positions(self) -> np.ndarray:
        return self.runtime.positions()

    def shell_positions(self) -> np.ndarray:
        return self.positions()[: self.topology.shell_count]

    def translate(self, offset: Iterable[float]) -> None:
        self.runtime.translate(np.asarray(tuple(offset), dtype=np.float32))

    def zero_velocity(self) -> None:
        self.runtime.zero_velocity()

    def reset(
        self,
        center_m: Iterable[float],
        row_rotation: np.ndarray | None = None,
    ) -> None:
        center = np.asarray(tuple(center_m), dtype=np.float32).reshape(3)
        if row_rotation is None:
            row_rotation = np.eye(3, dtype=np.float32)
        rotation = np.asarray(row_rotation, dtype=np.float32).reshape(3, 3)
        points = np.concatenate(
            (self.topology.shell_vertices, self.topology.filler_points), axis=0
        )
        self.runtime.reset(points @ rotation + center)

    def center(self) -> np.ndarray:
        return self.shell_positions().mean(axis=0)

    def signed_distances(
        self, points: np.ndarray, *, radius_m: float = 0.0
    ) -> np.ndarray:
        return self.runtime.signed_distances(points, radius_m=radius_m)

    def measure_contacts(self) -> None:
        self.runtime.measure_contacts()

    def frame_contact_counts(self) -> np.ndarray:
        return self.runtime.frame_contact_counts()

    def diagnostics(self) -> dict[str, object]:
        positions = self.positions()
        shell = positions[: self.topology.shell_count]
        grains = positions[self.topology.shell_count :]
        rest_extents = np.ptp(self.topology.shell_vertices, axis=0)
        current_extents = np.ptp(shell, axis=0)
        return {
            "kind": "custom_xpbd_closed_shell_plus_filler",
            "fill_model": "dynamic_all_pair_hard_grain_frictional_jamming",
            "rest_equivalent_hard_grain_solid_fraction": granular_solid_fraction(
                self.topology, self.config
            ),
            "backend": "Taichi_Vulkan",
            "configuration": self.config.to_dict(),
            "topology": self.topology.counts(),
            "center_m": shell.mean(axis=0).tolist(),
            "current_extents_m": current_extents.tolist(),
            "rest_shape": {
                "extents_m": rest_extents.tolist(),
                "current_extent_ratio": (current_extents / rest_extents).tolist(),
                "maximum_extent_error_ratio": float(
                    np.max(np.abs(current_extents / rest_extents - 1.0))
                ),
            },
            "table_contact": {
                "height_m": self.table_height_m,
                "minimum_shell_clearance_m": float(
                    np.min(shell[:, 2])
                    - self.table_height_m
                    - self.config.particle_radius_m
                ),
            },
            "grain_containment": grain_containment_diagnostics(
                shell,
                self.topology.shell_faces,
                grains,
                grain_radius_m=self.config.grain_contact_distance_m * 0.5,
            ),
            "authoritative_contact": "XPBD_external_shell_contact_plus_internal_grains",
            "external_contact_scope": "shell_particles_only",
            "synthetic_attachment": False,
            "nominal_mass_kg": self.config.nominal_mass_kg,
            "contacts": self.runtime.contact_diagnostics(),
        }


def evaluate_xpbd_gate(
    *,
    initial_extents_m: Iterable[float],
    squeezed_extents_m: Iterable[float],
    initial_center_m: Iterable[float],
    lifted_center_m: Iterable[float],
    recovered_extents_m: Iterable[float],
    finite: bool,
    maximum_gripper_penetration_m: float,
    p95_step_ms: float,
    minimum_lift_m: float = 0.035,
    maximum_p95_step_ms: float = 12.0,
    rest_extents_m: Iterable[float] | None = None,
    static_aligned_extents_m: Iterable[float] | None = None,
    recovered_aligned_extents_m: Iterable[float] | None = None,
    maximum_static_extent_error_ratio: float = 0.10,
    maximum_recovered_rest_extent_error_ratio: float = 0.12,
    maximum_orthogonal_squeeze_expansion_ratio: float = 1.10,
    maximum_grain_escape_m: float = 0.0,
    maximum_allowed_grain_escape_m: float = 5.0e-4,
) -> dict[str, object]:
    """Evaluate deformation, granular squeeze response, and performance."""

    initial_extents = np.asarray(tuple(initial_extents_m), dtype=np.float64)
    squeezed_extents = np.asarray(tuple(squeezed_extents_m), dtype=np.float64)
    recovered_extents = np.asarray(tuple(recovered_extents_m), dtype=np.float64)
    initial_center = np.asarray(tuple(initial_center_m), dtype=np.float64)
    lifted_center = np.asarray(tuple(lifted_center_m), dtype=np.float64)
    if any(values.shape != (3,) for values in (initial_extents, squeezed_extents, recovered_extents, initial_center, lifted_center)):
        raise ValueError("XPBD gate vectors must each contain three values")
    if np.any(initial_extents <= 0.0):
        raise ValueError("initial XPBD extents must be positive")
    squeeze_ratio = float(squeezed_extents[0] / initial_extents[0])
    orthogonal_squeeze_expansion_ratio = float(
        np.max(squeezed_extents[1:] / initial_extents[1:])
    )
    recovery_error = float(
        np.max(np.abs(recovered_extents / initial_extents - 1.0))
    )
    lift_m = float(lifted_center[2] - initial_center[2])
    static_extent_error = None
    recovered_rest_extent_error = None
    if rest_extents_m is not None:
        rest_extents = np.asarray(tuple(rest_extents_m), dtype=np.float64)
        if rest_extents.shape != (3,) or np.any(rest_extents <= 0.0):
            raise ValueError("rest XPBD extents must contain three positive values")
        static_extents = (
            np.asarray(tuple(static_aligned_extents_m), dtype=np.float64)
            if static_aligned_extents_m is not None
            else initial_extents
        )
        if static_extents.shape != (3,) or np.any(static_extents <= 0.0):
            raise ValueError("static aligned extents must contain three positive values")
        static_extent_error = float(np.max(np.abs(static_extents / rest_extents - 1.0)))
        if recovered_aligned_extents_m is not None:
            recovered_aligned_extents = np.asarray(
                tuple(recovered_aligned_extents_m), dtype=np.float64
            )
            if recovered_aligned_extents.shape != (3,) or np.any(
                recovered_aligned_extents <= 0.0
            ):
                raise ValueError(
                    "recovered aligned extents must contain three positive values"
                )
            recovered_rest_extent_error = float(
                np.max(np.abs(recovered_aligned_extents / rest_extents - 1.0))
            )
    checks = {
        "finite_state": bool(finite),
        "visible_compression": 0.60 <= squeeze_ratio <= 0.94,
        # A liquid-like, nearly incompressible fill turns jaw compression into
        # large expansion on the other axes.  The real object is packed with
        # hard granules: it should form a local dimple and jam instead.
        "no_water_balloon_bulging": (
            orthogonal_squeeze_expansion_ratio
            <= maximum_orthogonal_squeeze_expansion_ratio
        ),
        "lift_without_tether": lift_m >= minimum_lift_m,
        "release_shape_recovery": recovery_error <= 0.20,
        "gripper_nonpenetration": maximum_gripper_penetration_m <= 0.001,
        "hard_grains_remain_inside_shell": (
            maximum_grain_escape_m <= maximum_allowed_grain_escape_m
        ),
        "realtime_solver_step": p95_step_ms <= maximum_p95_step_ms,
    }
    if static_extent_error is not None:
        checks["static_shape_support"] = (
            static_extent_error <= maximum_static_extent_error_ratio
        )
    if recovered_rest_extent_error is not None:
        checks["recovered_rest_shape_support"] = (
            recovered_rest_extent_error
            <= maximum_recovered_rest_extent_error_ratio
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "squeeze_x_ratio": squeeze_ratio,
        "orthogonal_squeeze_expansion_ratio": (
            orthogonal_squeeze_expansion_ratio
        ),
        "lift_m": lift_m,
        "recovery_max_extent_error_ratio": recovery_error,
        "maximum_gripper_penetration_m": float(maximum_gripper_penetration_m),
        "maximum_grain_escape_m": float(maximum_grain_escape_m),
        "p95_step_ms": float(p95_step_ms),
        "static_max_extent_error_ratio": static_extent_error,
        "recovered_rest_max_extent_error_ratio": recovered_rest_extent_error,
        "thresholds": {
            "minimum_lift_m": minimum_lift_m,
            "maximum_p95_step_ms": maximum_p95_step_ms,
            "maximum_static_extent_error_ratio": maximum_static_extent_error_ratio,
            "maximum_recovered_rest_extent_error_ratio": (
                maximum_recovered_rest_extent_error_ratio
            ),
            "maximum_orthogonal_squeeze_expansion_ratio": (
                maximum_orthogonal_squeeze_expansion_ratio
            ),
            "maximum_allowed_grain_escape_m": maximum_allowed_grain_escape_m,
        },
    }
