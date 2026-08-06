"""Reusable particle-soft-body to textured-mesh binding for the plush asset."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quaternion_wxyz_to_matrix(quaternion: object) -> np.ndarray:
    """Return a right-handed row-vector rotation matrix."""

    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    w, x, y, z = values / norm
    # This is the transpose of the usual column-vector rotation matrix because
    # Genesis particle and visual vertex tensors are multiplied as row vectors.
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
            (2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
            (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def convex_support_planes(
    points: object,
    *,
    maximum_planes: int = 64,
    coplanar_tolerance: float = 2.0e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return a bounded half-space representation enclosing finite 3-D points.

    The returned local-space hull uses ``normal @ point <= offset``.  Meshes
    commonly triangulate one planar face into many identical ConvexHull facets,
    so coplanar facets are deduplicated first.  If a genuinely detailed hull
    still exceeds the Taichi field budget, normals are sampled over direction
    space and every retained offset is recomputed from all source vertices.
    That last step guarantees that simplification can enlarge, but never cut
    into, the authoritative Genesis link geometry.
    """

    from scipy.spatial import ConvexHull

    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("convex collider needs at least four finite vertices")
    if maximum_planes < 4:
        raise ValueError("maximum convex plane count must be at least four")
    if coplanar_tolerance <= 0.0:
        raise ValueError("coplanar tolerance must be positive")
    try:
        hull = ConvexHull(vertices)
    except Exception as error:
        raise ValueError("gripper vertices do not form a 3-D convex hull") from error

    equations = np.asarray(hull.equations, dtype=np.float64)
    normals = equations[:, :3]
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / lengths[:, None]
    offsets = -equations[:, 3] / lengths

    retained_normals: list[np.ndarray] = []
    retained_offsets: list[float] = []
    for normal, offset in zip(normals, offsets, strict=True):
        duplicate = any(
            float(np.dot(normal, candidate)) >= 1.0 - coplanar_tolerance
            and abs(float(offset - candidate_offset)) <= coplanar_tolerance
            for candidate, candidate_offset in zip(
                retained_normals, retained_offsets, strict=True
            )
        )
        if not duplicate:
            retained_normals.append(normal)
            retained_offsets.append(float(offset))

    unique_normals = np.asarray(retained_normals, dtype=np.float64)
    unique_plane_count = len(unique_normals)
    if unique_plane_count > maximum_planes:
        # Deterministic farthest-point sampling on the unit sphere keeps broad
        # directional coverage instead of merely taking the first mesh faces.
        first = int(np.lexsort(unique_normals.T[::-1])[0])
        selected = [first]
        minimum_angular_distance = 1.0 - unique_normals @ unique_normals[first]
        while len(selected) < maximum_planes:
            next_index = int(np.argmax(minimum_angular_distance))
            selected.append(next_index)
            angular_distance = 1.0 - unique_normals @ unique_normals[next_index]
            minimum_angular_distance = np.minimum(
                minimum_angular_distance, angular_distance
            )
            minimum_angular_distance[selected] = -1.0
        unique_normals = unique_normals[selected]

    # Recompute support offsets even for the unsimplified case so numerical
    # ConvexHull tolerances cannot leave a source vertex microscopically outside.
    support_offsets = np.max(vertices @ unique_normals.T, axis=0)
    signed_excess = vertices @ unique_normals.T - support_offsets[None, :]
    maximum_containment_error = float(max(0.0, np.max(signed_excess)))
    return (
        unique_normals.astype(np.float32),
        support_offsets.astype(np.float32),
        {
            "source_facet_count": int(len(equations)),
            "unique_plane_count": int(unique_plane_count),
            "retained_plane_count": int(len(unique_normals)),
            "maximum_containment_error_m": maximum_containment_error,
        },
    )


def matrix_to_quaternion_wxyz(row_rotation: object) -> np.ndarray:
    """Convert a right-handed row-vector rotation matrix to wxyz."""

    row_matrix = np.asarray(row_rotation, dtype=np.float64)
    if row_matrix.shape != (3, 3) or not np.isfinite(row_matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    matrix = row_matrix.T
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def nearest_particle_patch(
    particles: object,
    point: object,
    *,
    count: int,
    maximum_nearest_distance_m: float,
    excluded: object = (),
) -> np.ndarray:
    """Select a deterministic local patch, or no particles when out of reach."""

    positions = np.asarray(particles, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(point, dtype=np.float64).reshape(3)
    if count < 1:
        raise ValueError("grasp patch count must be positive")
    if maximum_nearest_distance_m <= 0.0:
        raise ValueError("maximum nearest distance must be positive")
    available = np.ones(len(positions), dtype=bool)
    excluded_indices = np.asarray(tuple(excluded), dtype=np.int64).reshape(-1)
    if len(excluded_indices):
        if np.any(excluded_indices < 0) or np.any(excluded_indices >= len(positions)):
            raise ValueError("excluded particle index is out of bounds")
        available[excluded_indices] = False
    indices = np.flatnonzero(available)
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    distances = np.linalg.norm(positions[indices] - target, axis=1)
    order = np.argsort(distances, kind="stable")
    if float(distances[order[0]]) > maximum_nearest_distance_m:
        return np.empty(0, dtype=np.int64)
    return indices[order[: min(count, len(order))]].astype(np.int64)


def separated_support_center(
    surface_point: object,
    outward: object,
    centered_point_sets: tuple[object, ...],
    *,
    clearance_m: float,
    floor_z_m: float | None = None,
) -> np.ndarray:
    """Place point sets outside a support plane and optional horizontal floor."""

    surface = np.asarray(surface_point, dtype=np.float64).reshape(3)
    direction = np.asarray(outward, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("support direction must be nonzero")
    if clearance_m < 0.0:
        raise ValueError("support clearance must be nonnegative")
    direction /= norm
    if not centered_point_sets:
        raise ValueError("at least one centered point set is required")
    back_extents = []
    minimum_z = np.inf
    for points in centered_point_sets:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("support point sets must be nonempty and finite")
        back_extents.append(-float(np.min(values @ direction)))
        minimum_z = min(minimum_z, float(np.min(values[:, 2])))
    outward_offset = max(back_extents) + clearance_m
    center = surface + direction * outward_offset
    if floor_z_m is None:
        return center

    required_center_z = float(floor_z_m) + clearance_m - minimum_z
    if center[2] >= required_center_z:
        return center
    # Find the intersection of the fingertip support half-space and the table
    # support half-space in the span of the gripper normal and world up.  A
    # naive vertical lift would violate the fingertip plane when the fingers
    # point down, which is exactly how the prior bad reset entered the table.
    outward_z = float(direction[2])
    denominator = 1.0 - outward_z * outward_z
    if denominator < 1e-8:
        raise ValueError("vertical support direction cannot also clear the floor")
    floor_offset = required_center_z - float(surface[2])
    outward_coefficient = (
        outward_offset - outward_z * floor_offset
    ) / denominator
    up_coefficient = (
        floor_offset - outward_z * outward_offset
    ) / denominator
    return (
        surface
        + direction * outward_coefficient
        + np.asarray((0.0, 0.0, up_coefficient), dtype=np.float64)
    )


class PlushVisualBinding:
    """Drive a textured visual with rigid motion plus local particle displacements."""

    def __init__(
        self,
        soft: object,
        visual: object,
        *,
        supports: int = 8,
        residual_gain: float = 1.0,
        maximum_residual_m: float | None = None,
        gaussian_kernel_sigma_m: float | None = None,
        contact_influence_provider: Callable[[], object] | None = None,
        contact_influence_gain: float = 1.0,
        contact_residual_gain: float = 1.0,
        contact_maximum_residual_m: float | None = None,
    ):
        import torch
        if supports < 1:
            raise ValueError("visual binding supports must be positive")
        if not 0.0 <= residual_gain <= 1.0:
            raise ValueError("visual residual gain must be in [0, 1]")
        if maximum_residual_m is not None and maximum_residual_m <= 0.0:
            raise ValueError("maximum visual residual must be positive")
        if gaussian_kernel_sigma_m is not None and gaussian_kernel_sigma_m <= 0.0:
            raise ValueError("Gaussian visual kernel sigma must be positive")
        if not residual_gain <= contact_residual_gain <= 1.0:
            raise ValueError(
                "contact visual residual gain must be in [base gain, 1]"
            )
        if contact_influence_gain <= 0.0:
            raise ValueError("contact visual influence gain must be positive")
        if (
            contact_maximum_residual_m is not None
            and contact_maximum_residual_m <= 0.0
        ):
            raise ValueError("maximum contact visual residual must be positive")
        self.soft = soft
        self.visual = visual
        self.residual_gain = float(residual_gain)
        self.maximum_residual_m = (
            None if maximum_residual_m is None else float(maximum_residual_m)
        )
        self.gaussian_kernel_sigma_m = (
            None
            if gaussian_kernel_sigma_m is None
            else float(gaussian_kernel_sigma_m)
        )
        self.contact_influence_provider = contact_influence_provider
        self.contact_influence_gain = float(contact_influence_gain)
        self.contact_residual_gain = float(contact_residual_gain)
        self.contact_maximum_residual_m = (
            None
            if contact_maximum_residual_m is None
            else float(contact_maximum_residual_m)
        )
        self._last_contact_influence = (0.0, 0.0, 0.0)
        self.rest_particles = soft.get_particles_pos().detach().clone()
        self.rest_visual = visual.get_vverts().detach().clone()
        if self.gaussian_kernel_sigma_m is None:
            from scipy.spatial import cKDTree

            rest_particles_np = as_numpy(self.rest_particles)
            rest_visual_np = as_numpy(self.rest_visual)
            distances, indices = cKDTree(rest_particles_np).query(
                rest_visual_np,
                k=min(supports, len(rest_particles_np)),
                workers=-1,
            )
            if indices.ndim == 1:
                indices = indices[:, None]
                distances = distances[:, None]
            inverse_distance = 1.0 / np.maximum(distances, 1e-5) ** 2
            weights = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)
            self.indices = torch.as_tensor(
                indices, dtype=torch.long, device=self.rest_particles.device
            )
            self.weights = torch.as_tensor(
                weights,
                dtype=self.rest_particles.dtype,
                device=self.rest_particles.device,
            )
            binding_kind = "rigid_kabsch_plus_knn_smooth_residual"
            support_count = int(self.indices.shape[1])
        else:
            # All-node Gaussian weights form a C-infinity displacement field.
            # A hard K-nearest cut changes support membership between adjacent
            # high-resolution visual vertices and produced visible triangle-
            # scale folds on the 60k-face TRELLIS surface.
            # Build the dense field on the selected Torch device. The full
            # TRELLIS asset has 221k vertices; doing this 36M-distance build
            # through a broadcasted NumPy temporary saturated the Ryzen CPU
            # for minutes before a live viewer could become READY. On the AMD
            # GPU this is a compact pairwise-distance kernel and the resulting
            # matrix remains resident for each deformation update.
            squared_distance = torch.cdist(
                self.rest_visual,
                self.rest_particles,
                p=2.0,
            ).square_()
            self.weights = torch.softmax(
                -0.5
                * squared_distance
                / (self.gaussian_kernel_sigma_m**2),
                dim=1,
            )
            self.indices = None
            binding_kind = "rigid_kabsch_plus_all_node_gaussian_residual"
            support_count = int(len(self.rest_particles))
        self.rest_center = self.rest_particles.mean(dim=0)
        self.rest_particles_centered = self.rest_particles - self.rest_center
        self.rest_visual_centered = self.rest_visual - self.rest_center
        faces: list[np.ndarray] = []
        vertex_offset = 0
        for vgeom in visual.vgeoms:
            local_faces = np.asarray(vgeom.vmesh.faces, dtype=np.int64).reshape(-1, 3)
            faces.append(local_faces + vertex_offset)
            vertex_offset += int(vgeom.n_vverts)
        if vertex_offset != len(self.rest_visual) or not faces:
            raise RuntimeError("visual face topology does not match custom vertices")
        self.faces = torch.as_tensor(
            np.concatenate(faces, axis=0),
            dtype=torch.long,
            device=self.rest_visual.device,
        )
        rest_triangles = self.rest_visual[self.faces]
        self.rest_face_cross = torch.linalg.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0],
            rest_triangles[:, 2] - rest_triangles[:, 0],
        )
        rest_face_double_area = torch.linalg.norm(self.rest_face_cross, dim=1)
        self.rest_face_valid = rest_face_double_area > 1.0e-12
        self.rest_face_double_area = rest_face_double_area.clamp_min(1.0e-12)
        self._last_visual_vertices = self.rest_visual.detach().clone()
        self.embedding_diagnostics = {
            "binding": binding_kind,
            "visual_vertices": int(len(self.rest_visual)),
            "physics_vertices": int(len(self.rest_particles)),
            "supports": support_count,
            "gaussian_kernel_sigma_m": self.gaussian_kernel_sigma_m,
            "residual_gain": self.residual_gain,
            "maximum_residual_m": self.maximum_residual_m,
            "contact_adaptive_residual": {
                "enabled": self.contact_influence_provider is not None,
                "contact_residual_gain": self.contact_residual_gain,
                "contact_influence_gain": self.contact_influence_gain,
                "contact_maximum_residual_m": (
                    self.contact_maximum_residual_m
                ),
                "influence_minimum": 0.0,
                "influence_mean": 0.0,
                "influence_maximum": 0.0,
            },
            "rest_reconstruction_error_m_max": 0.0,
            "render_surface_smooth": all(
                bool(vgeom.surface.smooth) for vgeom in visual.vgeoms
            ),
        }

    def rigid_transform(self) -> tuple[Any, Any, Any]:
        """Return current particles, centroid and best-fit row rotation."""

        import torch

        particles = self.soft.get_particles_pos()
        center = particles.mean(dim=0)
        centered = particles - center
        covariance = self.rest_particles_centered.T @ centered
        left, _singular, right_t = torch.linalg.svd(covariance)
        rotation = left @ right_t
        if torch.linalg.det(rotation) < 0:
            left = left.clone()
            left[:, -1] *= -1.0
            rotation = left @ right_t
        return particles, center, rotation

    def update(self) -> None:
        import torch

        with torch.no_grad():
            particles, center, rotation = self.rigid_transform()
            rigid_particles = self.rest_particles_centered @ rotation + center
            local_residual = particles - rigid_particles
            if self.indices is None:
                blended_residual = self.weights @ local_residual
            else:
                blended_residual = (
                    local_residual[self.indices] * self.weights.unsqueeze(-1)
                ).sum(dim=1)
            visual_contact_influence = None
            if self.contact_influence_provider is not None:
                node_influence = torch.as_tensor(
                    self.contact_influence_provider(),
                    dtype=particles.dtype,
                    device=particles.device,
                ).reshape(-1)
                if len(node_influence) != len(particles):
                    raise RuntimeError(
                        "contact influence must match physics particle count"
                    )
                node_influence = torch.clamp(node_influence, 0.0, 1.0)
                if self.indices is None:
                    visual_contact_influence = self.weights @ node_influence
                else:
                    visual_contact_influence = (
                        node_influence[self.indices] * self.weights
                    ).sum(dim=1)
                visual_contact_influence = torch.clamp(
                    visual_contact_influence * self.contact_influence_gain,
                    0.0,
                    1.0,
                )
                self._last_contact_influence = (
                    float(torch.min(visual_contact_influence)),
                    float(torch.mean(visual_contact_influence)),
                    float(torch.max(visual_contact_influence)),
                )
                adaptive_gain = self.residual_gain + visual_contact_influence * (
                    self.contact_residual_gain - self.residual_gain
                )
                blended_residual *= adaptive_gain[:, None]
            else:
                blended_residual *= self.residual_gain
            maximum_residual = self.maximum_residual_m
            if (
                visual_contact_influence is not None
                and self.contact_maximum_residual_m is not None
                and maximum_residual is not None
            ):
                maximum_residual = maximum_residual + visual_contact_influence * (
                    self.contact_maximum_residual_m - maximum_residual
                )
            if maximum_residual is not None:
                residual_norm = torch.linalg.norm(
                    blended_residual, dim=1, keepdim=True
                )
                residual_scale = torch.clamp(
                    (
                        maximum_residual
                        if torch.is_tensor(maximum_residual)
                        else torch.full_like(residual_norm, maximum_residual)
                    ).reshape(-1, 1)
                    / residual_norm.clamp_min(1.0e-12),
                    max=1.0,
                )
                blended_residual *= residual_scale
            visual_vertices = (
                self.rest_visual_centered @ rotation + center + blended_residual
            )
            self._last_visual_vertices = visual_vertices.detach().clone()
            self.visual.set_vverts(visual_vertices)
            contact_metrics = self.embedding_diagnostics[
                "contact_adaptive_residual"
            ]
            contact_metrics["influence_minimum"] = self._last_contact_influence[0]
            contact_metrics["influence_mean"] = self._last_contact_influence[1]
            contact_metrics["influence_maximum"] = self._last_contact_influence[2]

    def smooth_contact_correction(self, correction: object):
        """Low-pass a visual contact correction through physics supports."""

        import torch

        values = torch.as_tensor(
            correction,
            dtype=self.rest_visual.dtype,
            device=self.rest_visual.device,
        ).reshape(-1, 3)
        if len(values) != len(self.rest_visual):
            raise ValueError("visual correction must match visual vertex count")
        active = (torch.linalg.norm(values, dim=1) > 1.0e-9).to(
            values.dtype
        )[:, None]
        if not bool(torch.any(active)):
            return torch.zeros_like(values)
        if self.indices is not None:
            raise RuntimeError(
                "contact correction smoothing requires all-node Gaussian weights"
            )
        node_weight = self.weights.T @ active
        node_correction = (self.weights.T @ values) / node_weight.clamp_min(
            1.0e-8
        )
        node_correction = torch.where(
            node_weight > 1.0e-7,
            node_correction,
            torch.zeros_like(node_correction),
        )
        return self.weights @ node_correction

    def surface_quality_diagnostics(self) -> dict[str, object]:
        """Measure visible triangle folding independently of tet validity."""

        import torch

        with torch.no_grad():
            _particles, _center, rotation = self.rigid_transform()
            triangles = self._last_visual_vertices[self.faces]
            face_cross = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            double_area = torch.linalg.norm(face_cross, dim=1)
            rigid_rest_cross = self.rest_face_cross @ rotation
            orientation_cosine = torch.sum(
                face_cross * rigid_rest_cross, dim=1
            ) / (
                double_area.clamp_min(1.0e-12)
                * self.rest_face_double_area
            )
            area_ratio = double_area / self.rest_face_double_area
            finite_values = torch.isfinite(area_ratio) & torch.isfinite(
                orientation_cosine
            )
            evaluable = finite_values & self.rest_face_valid
            flipped = evaluable & (orientation_cosine <= 0.0)
            current_degenerate = evaluable & (area_ratio <= 1.0e-3)
            finite_ratio = area_ratio[evaluable]
            source_degenerate = int(torch.count_nonzero(~self.rest_face_valid))
            if not len(finite_ratio):
                return {
                    "finite": False,
                    "faces": int(len(self.faces)),
                    "flipped_faces": int(len(self.faces)),
                    "degenerate_faces": int(len(self.faces)),
                }
            return {
                "finite": bool(torch.all(finite_values)),
                "faces": int(len(self.faces)),
                "evaluated_faces": int(torch.count_nonzero(evaluable)),
                "source_degenerate_faces": source_degenerate,
                "flipped_faces": int(torch.count_nonzero(flipped)),
                "flipped_fraction": float(torch.mean(flipped.float())),
                "degenerate_faces": source_degenerate
                + int(torch.count_nonzero(current_degenerate)),
                "degenerate_fraction": float(
                    (
                        source_degenerate
                        + int(torch.count_nonzero(current_degenerate))
                    )
                    / len(self.faces)
                ),
                "area_ratio": {
                    "minimum": float(torch.min(finite_ratio)),
                    "p01": float(torch.quantile(finite_ratio, 0.01)),
                    "median": float(torch.median(finite_ratio)),
                    "p99": float(torch.quantile(finite_ratio, 0.99)),
                    "maximum": float(torch.max(finite_ratio)),
                },
            }


def position_weld_groups(
    positions: object, *, tolerance_m: float = 1.0e-7
) -> tuple[np.ndarray, int]:
    """Group UV-split vertices that occupy the same geometric position."""

    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("normal-weld positions must be finite and non-empty")
    if tolerance_m <= 0.0:
        raise ValueError("normal-weld tolerance must be positive")
    quantized = np.rint(points / tolerance_m).astype(np.int64)
    _unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.astype(np.int32), int(np.max(inverse)) + 1


def area_weighted_welded_normals(
    vertices: object,
    faces: object,
    weld_groups: object,
) -> np.ndarray:
    """Return smooth normals shared across duplicated UV seam vertices."""

    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    groups = np.asarray(weld_groups, dtype=np.int64).reshape(-1)
    if len(groups) != len(points) or not len(triangles):
        raise ValueError("normal-weld topology shape mismatch")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(points):
        raise ValueError("normal-weld face index out of range")
    if int(groups.min()) < 0:
        raise ValueError("normal-weld group index must be nonnegative")
    face_cross = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    group_normals = np.zeros((int(groups.max()) + 1, 3), dtype=np.float32)
    for local_index in range(3):
        np.add.at(group_normals, groups[triangles[:, local_index]], face_cross)
    lengths = np.linalg.norm(group_normals, axis=1, keepdims=True)
    group_normals /= np.maximum(lengths, 1.0e-12)
    return group_normals[groups]


def best_fit_row_rotation(rest_vertices: object, current_vertices: object) -> np.ndarray:
    """Return the proper row-vector rotation best aligning two point sets."""

    rest = np.asarray(rest_vertices, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(current_vertices, dtype=np.float64).reshape(-1, 3)
    if rest.shape != current.shape or len(rest) < 3:
        raise ValueError("rigid normal transport needs matching point sets")
    if not np.isfinite(rest).all() or not np.isfinite(current).all():
        raise ValueError("rigid normal transport points must be finite")
    covariance = (rest - rest.mean(axis=0)).T @ (
        current - current.mean(axis=0)
    )
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def rigidly_transport_rest_normals(
    rest_vertices: object,
    current_vertices: object,
    rest_normals: object,
) -> np.ndarray:
    """Transport authored rest normals by the visual's best-fit rotation.

    The TRELLIS surface contains texture seams and non-manifold decorative
    sheets. Recomputing normals from that topology after every custom-vvert
    update creates false triangular shards even when the surface displacement
    field is smooth. The tight plush renderer therefore keeps its authored
    high-quality rest normals and transports their global orientation while
    MGPBD still owns every physical/contact degree of freedom.
    """

    normals = np.asarray(rest_normals, dtype=np.float64).reshape(-1, 3)
    rest = np.asarray(rest_vertices, dtype=np.float64).reshape(-1, 3)
    if normals.shape != rest.shape or not np.isfinite(normals).all():
        raise ValueError("rest normals must match finite rest vertices")
    rotation = best_fit_row_rotation(rest, current_vertices)
    transported = normals @ rotation
    transported /= np.maximum(
        np.linalg.norm(transported, axis=1, keepdims=True), 1.0e-12
    )
    return transported.astype(np.float32)


def install_custom_vvert_rest_normal_transport(
    scene: object,
    visual: object,
) -> dict[str, object]:
    """Preserve the TRELLIS authored appearance under tight deformation."""

    rest_world = as_numpy(visual.get_vverts()).astype(np.float64, copy=True)
    targets_by_count: dict[
        int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    normal_cache_by_count: dict[int, dict[str, np.ndarray]] = {}
    vertex_offset = 0
    for vgeom in visual.vgeoms:
        count = int(vgeom.n_vverts)
        local_vertices = np.asarray(vgeom.vmesh.verts, dtype=np.float64).reshape(
            -1, 3
        )
        local_normals = np.asarray(
            vgeom.vmesh.normals, dtype=np.float64
        ).reshape(-1, 3)
        geom_rest_world = rest_world[vertex_offset : vertex_offset + count]
        if len(local_vertices) != count or len(geom_rest_world) != count:
            raise RuntimeError("custom-vvert normal transport topology mismatch")
        local_to_world = best_fit_row_rotation(local_vertices, geom_rest_world)
        rest_world_normals = local_normals @ local_to_world
        rest_world_normals /= np.maximum(
            np.linalg.norm(rest_world_normals, axis=1, keepdims=True), 1.0e-12
        )
        rotation_fit_indices = np.linspace(
            0,
            count - 1,
            num=min(count, 1024),
            dtype=np.int64,
        )
        targets_by_count.setdefault(count, []).append(
            (
                geom_rest_world.copy(),
                rest_world_normals.astype(np.float32),
                rotation_fit_indices,
            )
        )
        normal_cache_by_count[count] = {
            "sample": geom_rest_world[rotation_fit_indices].astype(
                np.float32, copy=True
            ),
            "normals": rest_world_normals.astype(np.float32, copy=True),
        }
        vertex_offset += count
    if vertex_offset != len(rest_world) or not vertex_offset:
        raise RuntimeError("custom-vvert visual has no transportable normals")

    jit = scene.visualizer.context.jit
    original_update_normal = jit.update_normal
    diagnostics: dict[str, object] = {
        "installed": True,
        "method": "rigidly_transported_authored_rest_normals",
        "visual_vertices": vertex_offset,
        "rotation_fit_vertices": min(vertex_offset, 1024),
        "normal_updates": 0,
        "transported_normal_updates": 0,
        "transport_recomputations": 0,
        "transport_cache_hits": 0,
    }

    def update_normal(node: object, vertices: object):
        diagnostics["normal_updates"] = int(diagnostics["normal_updates"]) + 1
        primitive = node.mesh.primitives[0]
        values = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        candidates = targets_by_count.get(len(values), ())
        if int(diagnostics["normal_updates"]) <= 3:
            print(
                "ONELOOP_NORMAL_UPDATE "
                f"call={diagnostics['normal_updates']} vertices={len(values)} "
                f"indexed={primitive.indices is not None} "
                f"candidate_count={len(candidates)}",
                flush=True,
            )
        if primitive.indices is None or len(candidates) != 1:
            return original_update_normal(node, vertices)
        diagnostics["transported_normal_updates"] = (
            int(diagnostics["transported_normal_updates"]) + 1
        )
        rest_vertices, rest_normals, fit_indices = candidates[0]
        cache = normal_cache_by_count[len(values)]
        current_sample = values[fit_indices]
        if np.max(np.abs(current_sample - cache["sample"]), initial=0.0) <= 1.0e-7:
            diagnostics["transport_cache_hits"] = (
                int(diagnostics["transport_cache_hits"]) + 1
            )
            if int(diagnostics["normal_updates"]) <= 3:
                print("ONELOOP_NORMAL_UPDATE result=cache_hit", flush=True)
            return cache["normals"]
        rotation = best_fit_row_rotation(
            rest_vertices[fit_indices],
            current_sample,
        )
        cache["sample"] = current_sample.copy()
        cache["normals"] = rest_normals @ rotation.astype(np.float32)
        diagnostics["transport_recomputations"] = (
            int(diagnostics["transport_recomputations"]) + 1
        )
        if int(diagnostics["normal_updates"]) <= 3:
            print("ONELOOP_NORMAL_UPDATE result=recomputed", flush=True)
        return cache["normals"]

    # Assigning an instance attribute is atomic under the Python GIL. Avoid
    # taking viewer_lock here: a full-resolution custom-vvert entity asks the
    # viewer to recompute 294k face normals at refresh rate, which can starve
    # the build thread before it gets a chance to install this replacement.
    jit.update_normal = update_normal
    return diagnostics


def install_custom_vvert_seam_normal_smoothing(
    scene: object,
    visual: object,
    *,
    tolerance_m: float = 1.0e-7,
) -> dict[str, object]:
    """Patch one Genesis custom-vvert renderer to smooth across UV seams.

    Genesis correctly recomputes indexed-mesh normals after custom vertex
    updates, but OBJ UV seams are represented by duplicated vertex indices.
    Its stock normal kernel therefore cannot average across those geometric
    duplicates.  This narrowly wraps the scene-local JIT normal callback and
    replaces normals only for the target visual's vertex counts.
    """

    targets_by_count: dict[int, list[np.ndarray]] = {}
    total_vertices = 0
    total_groups = 0
    for vgeom in visual.vgeoms:
        points = np.asarray(vgeom.vmesh.verts, dtype=np.float64).reshape(-1, 3)
        groups, group_count = position_weld_groups(
            points, tolerance_m=tolerance_m
        )
        targets_by_count.setdefault(len(points), []).append(groups)
        total_vertices += len(points)
        total_groups += group_count
    if not total_vertices:
        raise RuntimeError("custom-vvert visual has no vertices for normal smoothing")

    jit = scene.visualizer.context.jit
    original_update_normal = jit.update_normal
    diagnostics: dict[str, object] = {
        "installed": True,
        "method": "position_welded_area_weighted_dynamic_normals",
        "tolerance_m": tolerance_m,
        "visual_vertices": total_vertices,
        "welded_position_groups": total_groups,
        "duplicate_uv_vertices": total_vertices - total_groups,
        "normal_updates": 0,
        "welded_normal_updates": 0,
    }

    def update_normal(node: object, vertices: object):
        diagnostics["normal_updates"] = int(diagnostics["normal_updates"]) + 1
        normals = original_update_normal(node, vertices)
        if normals is None:
            return None
        primitive = node.mesh.primitives[0]
        values = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        candidates = targets_by_count.get(len(values), ())
        if primitive.indices is None or len(candidates) != 1:
            return normals
        diagnostics["welded_normal_updates"] = (
            int(diagnostics["welded_normal_updates"]) + 1
        )
        return area_weighted_welded_normals(
            values,
            primitive.indices,
            candidates[0],
        )

    with scene.visualizer.viewer_lock:
        jit.update_normal = update_normal
    return diagnostics


class XPBDShellProvider:
    """Expose a Taichi XPBD shell as the torch tensor expected by the binder."""

    def __init__(self, solver: object, visual: object):
        self.solver = solver
        prototype = visual.get_vverts()
        self.dtype = prototype.dtype
        self.device = prototype.device

    def get_particles_pos(self):
        import torch

        return torch.as_tensor(
            self.solver.shell_positions(), dtype=self.dtype, device=self.device
        )


class XPBDPlushObjectAdapter:
    """Genesis-facing pose, collider, and diagnostics adapter for custom XPBD."""

    def __init__(
        self,
        solver: object,
        provider: XPBDShellProvider,
        binding: object,
        *,
        table_height_m: float,
    ):
        self.solver = solver
        self.provider = provider
        self.binding = binding
        self.table_height_m = float(table_height_m)
        self._gripper_links: tuple[object, ...] = ()
        self._collider_plane_normals = np.empty((0, 64, 3), dtype=np.float32)
        self._collider_plane_offsets = np.empty((0, 64), dtype=np.float32)
        self._collider_plane_counts = np.empty(0, dtype=np.int32)
        self._collider_diagnostics: list[dict[str, object]] = []
        self._collider_local_source_vertices: list[np.ndarray] = []
        self._collider_source_vertex_indices: list[np.ndarray] = []
        self._collider_geoms: list[object] = []
        self._collider_roles: list[str] = []
        self._collider_arm_indices: list[int] = []
        self._contact_gate_injection: dict[str, object] | None = None

    @staticmethod
    def _world_link_pose(link: object) -> tuple[np.ndarray, np.ndarray]:
        """Return solver-world pose matching ``RigidLink.get_verts()``.

        Genesis link poses default to ``relative=True``, which strips the
        entity morph/base pose.  ``get_verts()`` is explicitly world-frame.
        Mixing those two contracts displaced each arm collider by its base
        transform while the rendered link stayed correct.
        """

        position = as_numpy(link.get_pos(relative=False)).reshape(3)
        row_rotation = quaternion_wxyz_to_matrix(
            as_numpy(link.get_quat(relative=False))
        )
        return position, row_rotation

    def _include_contact_geom(self, role: str, geom_index: int) -> bool:
        """Return whether one Genesis geometry owns custom plush contact."""

        del role, geom_index
        return True

    def _contact_proxy_vertex_indices(
        self, role: str, geom_index: int, local_vertices: np.ndarray
    ) -> np.ndarray:
        """Select source vertices used to build one convex contact proxy."""

        del role, geom_index
        return np.arange(len(local_vertices), dtype=np.int64)

    def configure_grippers(
        self,
        left_fixed: object,
        right_fixed: object,
        left_moving: object,
        right_moving: object,
        _left_wrist: object,
        _right_wrist: object,
    ) -> None:
        sources: list[tuple[object, object, str, int, int]] = []
        for arm_name, arm_index, fixed, moving in (
            ("left", 0, left_fixed, left_moving),
            ("right", 1, right_fixed, right_moving),
        ):
            for role, link in (("fixed", fixed), ("moving", moving)):
                geoms = tuple(link.geoms)
                if not geoms:
                    raise RuntimeError(f"SO-101 {arm_name} {role} link has no collision geoms")
                for geom_index, geom in enumerate(geoms):
                    role_name = f"{arm_name}_{role}_geom_{geom_index}"
                    if not self._include_contact_geom(role_name, geom_index):
                        continue
                    sources.append(
                        (
                            link,
                            geom,
                            role_name,
                            arm_index,
                            geom_index,
                        )
                    )
        plane_normals = np.zeros((len(sources), 64, 3), dtype=np.float32)
        plane_offsets = np.zeros((len(sources), 64), dtype=np.float32)
        plane_counts = np.zeros(len(sources), dtype=np.int32)
        collider_diagnostics: list[dict[str, object]] = []
        local_source_vertices: list[np.ndarray] = []
        source_vertex_indices: list[np.ndarray] = []
        links: list[object] = []
        roles: list[str] = []
        arm_indices: list[int] = []
        for link, geom, role, arm_index, geom_index in sources:
            position, row_rotation = self._world_link_pose(link)
            world_vertices = as_numpy(geom.get_verts()).reshape(-1, 3)
            if not len(world_vertices) or not np.isfinite(world_vertices).all():
                raise RuntimeError("SO-101 gripper geom has no finite vertices")
            all_local_vertices = (world_vertices - position) @ row_rotation.T
            source_indices = np.asarray(
                self._contact_proxy_vertex_indices(
                    role, geom_index, all_local_vertices
                ),
                dtype=np.int64,
            ).reshape(-1)
            if (
                len(source_indices) < 4
                or int(source_indices.min()) < 0
                or int(source_indices.max()) >= len(all_local_vertices)
            ):
                raise RuntimeError("custom gripper contact proxy selection is invalid")
            local_vertices = all_local_vertices[source_indices]
            normals, offsets, diagnostics = convex_support_planes(local_vertices)
            index = len(collider_diagnostics)
            count = len(normals)
            plane_normals[index, :count] = normals
            plane_offsets[index, :count] = offsets
            plane_counts[index] = count
            collider_diagnostics.append(
                {
                    **diagnostics,
                    "role": role,
                    "source_vertices": int(len(all_local_vertices)),
                    "selected_vertices": int(len(local_vertices)),
                }
            )
            local_source_vertices.append(local_vertices.astype(np.float32))
            source_vertex_indices.append(source_indices)
            links.append(link)
            roles.append(role)
            arm_indices.append(arm_index)
        self._gripper_links = tuple(links)
        self._collider_plane_normals = plane_normals
        self._collider_plane_offsets = plane_offsets
        self._collider_plane_counts = plane_counts
        self._collider_diagnostics = collider_diagnostics
        self._collider_local_source_vertices = local_source_vertices
        self._collider_source_vertex_indices = source_vertex_indices
        self._collider_geoms = [source[1] for source in sources]
        self._collider_roles = roles
        self._collider_arm_indices = arm_indices
        self.update_gripper_colliders()
        initial_alignment = self.collider_alignment_diagnostics()
        if initial_alignment["maximum_vertex_alignment_error_m"] > 2.0e-5:
            raise RuntimeError("XPBD gripper collider is not aligned to Genesis world vertices")

    def update_gripper_colliders(self) -> None:
        if not self._gripper_links:
            raise RuntimeError("XPBD grippers have not been configured")
        centers: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for link in self._gripper_links:
            position, row_rotation = self._world_link_pose(link)
            centers.append(position)
            # Taichi's box SDF uses column vectors; Genesis helpers here use
            # row vectors, so pass the transpose.
            rotations.append(row_rotation.T)
        self.solver.set_convex_colliders(
            np.asarray(centers, dtype=np.float32),
            np.asarray(rotations, dtype=np.float32),
            self._collider_plane_normals,
            self._collider_plane_offsets,
            self._collider_plane_counts,
        )
        set_reference_points = getattr(
            self.solver, "set_collider_reference_points", None
        )
        if callable(set_reference_points):
            reference_points = []
            for link, local_vertices in zip(
                self._gripper_links,
                self._collider_local_source_vertices,
                strict=True,
            ):
                position, row_rotation = self._world_link_pose(link)
                reference_points.append(
                    local_vertices.mean(axis=0) @ row_rotation + position
                )
            set_reference_points(
                np.asarray(reference_points, dtype=np.float32)
            )

    def set_gripper_closure(
        self, left_closed_fraction: float, right_closed_fraction: float
    ) -> None:
        """Scale dry-friction capacity by each gripper's closure command.

        The convex geometry always remains an authoritative unilateral
        collider.  Closure only controls tangential load capacity: fully open
        fingers cannot retain a stale static-friction anchor and tow the plush.
        """

        closures = np.asarray(
            (left_closed_fraction, right_closed_fraction), dtype=np.float32
        )
        if not np.isfinite(closures).all() or np.any(closures < 0.0) or np.any(
            closures > 1.0
        ):
            raise ValueError("gripper closure fractions must be finite in [0, 1]")
        if not self._collider_arm_indices:
            raise RuntimeError("XPBD grippers have not been configured")
        scales = np.asarray(
            [closures[index] for index in self._collider_arm_indices],
            dtype=np.float32,
        )
        self.solver.set_collider_friction_scales(scales)

    @staticmethod
    def _group_contact_counts(
        counts: object,
        roles: list[str],
        arm_indices: list[int],
    ) -> dict[str, object]:
        """Require simultaneous fixed- and moving-finger shell contact per arm."""

        values = np.asarray(counts, dtype=np.int64).reshape(-1)
        if len(values) != len(roles) or len(values) != len(arm_indices):
            raise ValueError("collider contact metadata length mismatch")
        if np.any(values < 0) or any(index not in (0, 1) for index in arm_indices):
            raise ValueError("collider contact counts/arm indices are invalid")
        role_counts = [
            {"fixed": 0, "moving": 0},
            {"fixed": 0, "moving": 0},
        ]
        for count, role, arm_index in zip(
            values.tolist(), roles, arm_indices, strict=True
        ):
            if "_fixed_" in role:
                role_name = "fixed"
            elif "_moving_" in role:
                role_name = "moving"
            else:
                raise ValueError(f"unrecognized gripper collider role: {role}")
            role_counts[arm_index][role_name] += int(count)
        active = [
            item["fixed"] > 0 and item["moving"] > 0 for item in role_counts
        ]
        return {
            "schema_version": "radeon_oneloop.xpbd_gripper_contact_evidence.v1",
            "mode": "identity_specific_shell_contact_both_finger_roles",
            "active_by_arm": active,
            "role_contact_counts": {
                "left": role_counts[0],
                "right": role_counts[1],
            },
            "collider_roles": list(roles),
            "frame_contact_count_by_collider": values.tolist(),
            "minimum_contact_count_per_finger_role": 1,
            "synthetic_attachment": False,
        }

    def gripper_contact_evidence(self) -> dict[str, object]:
        """Expose identity-specific current contact without inventing a force."""

        if not self._collider_roles:
            raise RuntimeError("XPBD grippers have not been configured")
        return self._group_contact_counts(
            self.solver.frame_contact_counts(),
            self._collider_roles,
            self._collider_arm_indices,
        )

    def collider_alignment_diagnostics(self) -> dict[str, object]:
        if not self._gripper_links:
            return {
                "maximum_vertex_alignment_error_m": None,
                "maximum_plane_containment_error_m": None,
                "links": [],
            }
        links: list[dict[str, object]] = []
        maximum_alignment_error = 0.0
        maximum_containment_error = 0.0
        for index, (link, geom, local_vertices, source_indices) in enumerate(
            zip(
                self._gripper_links,
                self._collider_geoms,
                self._collider_local_source_vertices,
                self._collider_source_vertex_indices,
                strict=True,
            )
        ):
            position, row_rotation = self._world_link_pose(link)
            all_actual_world = as_numpy(geom.get_verts()).reshape(-1, 3)
            if int(source_indices.max()) >= len(all_actual_world):
                raise RuntimeError("Genesis gripper link vertex count changed after build")
            actual_world = all_actual_world[source_indices]
            if actual_world.shape != local_vertices.shape:
                raise RuntimeError("Genesis gripper link vertex count changed after build")
            reconstructed_world = local_vertices @ row_rotation + position
            alignment_error = float(
                np.max(np.linalg.norm(reconstructed_world - actual_world, axis=1))
            )
            current_local = (actual_world - position) @ row_rotation.T
            count = int(self._collider_plane_counts[index])
            containment_error = float(
                max(
                    0.0,
                    np.max(
                        current_local @ self._collider_plane_normals[index, :count].T
                        - self._collider_plane_offsets[index, :count]
                    ),
                )
            )
            maximum_alignment_error = max(maximum_alignment_error, alignment_error)
            maximum_containment_error = max(
                maximum_containment_error, containment_error
            )
            links.append(
                {
                    "vertex_count": int(len(actual_world)),
                    "vertex_alignment_error_m": alignment_error,
                    "plane_containment_error_m": containment_error,
                    "world_bounds_m": [
                        actual_world.min(axis=0).tolist(),
                        actual_world.max(axis=0).tolist(),
                    ],
                }
            )
        return {
            "maximum_vertex_alignment_error_m": maximum_alignment_error,
            "maximum_plane_containment_error_m": maximum_containment_error,
            "links": links,
        }

    def contact_gate_center(
        self, collider_index: int, *, overlap_m: float
    ) -> np.ndarray:
        """Construct a deterministic shallow-overlap pose for integration tests."""

        if not 0 <= collider_index < len(self._gripper_links):
            raise ValueError("contact gate collider index is out of range")
        if not 0.0 < overlap_m <= 0.005:
            raise ValueError("contact gate overlap must be in (0, 0.005] m")
        self.update_gripper_colliders()
        link = self._gripper_links[collider_index]
        position, row_rotation = self._world_link_pose(link)
        count = int(self._collider_plane_counts[collider_index])
        local_normals = self._collider_plane_normals[collider_index, :count]
        world_normals = local_normals @ row_rotation
        # Semantic-left arm at +X points inward toward -X; right mirrors it.
        arm_index = self._collider_arm_indices[collider_index]
        inward = np.asarray(
            (-1.0, 0.0, 0.0) if arm_index == 0 else (1.0, 0.0, 0.0),
            dtype=np.float32,
        )
        horizontal = np.abs(world_normals[:, 2]) < 0.65
        scores = world_normals @ inward
        scores[~horizontal] = -np.inf
        plane_index = int(np.argmax(scores))
        local_normal = local_normals[plane_index]
        world_normal = world_normals[plane_index]
        local_vertices = self._collider_local_source_vertices[collider_index]
        support_scores = local_vertices @ local_normal
        support_maximum = float(np.max(support_scores))
        support_indices = np.flatnonzero(
            support_scores >= support_maximum - 1.0e-4
        )
        support_world = local_vertices[support_indices] @ row_rotation + position
        # The centroid of the coplanar support patch lies on the face interior.
        # A single extreme vertex is a convex corner: moving along only one
        # face normal can remain outside an adjacent face and create no contact.
        support_point = support_world.mean(axis=0)
        shell_vertices = self.solver.topology.shell_vertices
        shell_contact_index = int(np.argmin(shell_vertices @ world_normal))
        shell_contact_vertex = shell_vertices[shell_contact_index]
        # Align one concrete shell particle with the face interior.  Merely
        # matching the shell's scalar support distance leaves that particle's
        # tangential coordinates unconstrained and can miss a small jaw face.
        desired_particle = support_point + world_normal * (
            float(self.solver.config.particle_radius_m) - overlap_m
        )
        center = desired_particle - shell_contact_vertex
        injected_signed_distances = self.solver.signed_distances(
            desired_particle[None],
            radius_m=float(self.solver.config.particle_radius_m),
        )[0]
        self._contact_gate_injection = {
            "collider_index": int(collider_index),
            "plane_index": plane_index,
            "overlap_m": float(overlap_m),
            "support_point_world_m": support_point.tolist(),
            "support_vertex_count": int(len(support_indices)),
            "outward_normal_world": world_normal.tolist(),
            "object_center_world_m": center.tolist(),
            "shell_contact_vertex_index": shell_contact_index,
            "shell_contact_vertex_local_m": shell_contact_vertex.tolist(),
            "injected_particle_signed_distance_m_by_collider": (
                injected_signed_distances.tolist()
            ),
        }
        return center.astype(np.float32)

    def record_contact_gate_post_reset(self) -> None:
        if self._contact_gate_injection is None:
            raise RuntimeError("contact gate center was not constructed")
        shell_index = int(self._contact_gate_injection["shell_contact_vertex_index"])
        particle = self.solver.shell_positions()[shell_index]
        signed_distances = self.solver.signed_distances(
            particle[None],
            radius_m=float(self.solver.config.particle_radius_m),
        )[0]
        self.solver.measure_contacts()
        self._contact_gate_injection.update(
            post_reset_particle_world_m=particle.tolist(),
            post_reset_particle_signed_distance_m_by_collider=(
                signed_distances.tolist()
            ),
            post_reset_taichi_contact_measurement=(
                self.solver.runtime.contact_diagnostics()
            ),
        )

    def step_simulation(self) -> None:
        self.update_gripper_colliders()
        self.solver.step(synchronize=False)

    def get_pos(self) -> np.ndarray:
        _particles, center, rotation = self.binding.rigid_transform()
        # XPBD reset positions are ``raw_OBJ_vertex @ R + origin``.  The OBJ
        # origin is not its shell centroid (for this asset they differ by up
        # to 13 mm), so returning the centroid and feeding it back into
        # ``set_quat`` performs an unintended second translation.
        rest_shell_center = np.asarray(
            self.solver.topology.shell_vertices, dtype=np.float64
        ).mean(axis=0)
        origin = as_numpy(center) - rest_shell_center @ as_numpy(rotation)
        return np.asarray(origin, dtype=np.float32)

    def get_quat(self) -> np.ndarray:
        _particles, _center, rotation = self.binding.rigid_transform()
        return matrix_to_quaternion_wxyz(as_numpy(rotation)).astype(np.float32)

    def set_pos(self, position: object) -> None:
        self.solver.reset(np.asarray(position, dtype=np.float32).reshape(3))

    def set_quat(self, quaternion: object) -> None:
        center = self.get_pos()
        self.solver.reset(center, quaternion_wxyz_to_matrix(quaternion))

    def set_dofs_velocity(self, velocity: object) -> None:
        values = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("XPBD reset velocity must contain six finite values")
        if np.any(np.abs(values) > 1e-8):
            raise ValueError("XPBD adapter currently supports only zero reset velocity")
        self.solver.zero_velocity()

    def update_visual(self) -> None:
        self.binding.update()

    def diagnostics(self) -> dict[str, object]:
        diagnostics = self.solver.diagnostics()
        visual_vertices = as_numpy(self.binding.visual.get_vverts()).reshape(-1, 3)
        visual_signed_distances = self.solver.signed_distances(
            visual_vertices, radius_m=0.0
        )
        minimum_visual_distance = (
            float(np.min(visual_signed_distances))
            if visual_signed_distances.size
            else None
        )
        diagnostics.update(
            pose_origin_m=self.get_pos().tolist(),
            visual_vertices=int(self.binding.rest_visual.shape[0]),
            visual_contact={
                "current_minimum_signed_distance_m": minimum_visual_distance,
                "current_maximum_penetration_m": (
                    max(0.0, -minimum_visual_distance)
                    if minimum_visual_distance is not None
                    else None
                ),
                "note": "current terminal frame; XPBD contact metrics include run peaks",
            },
            gripper_colliders={
                "kind": "per_Genesis_collision_geom_convex_support_hulls",
                "count": len(self._gripper_links),
                "links": self._collider_diagnostics,
                "role_indices": {
                    role: [
                        index
                        for index, candidate in enumerate(self._collider_roles)
                        if candidate.startswith(role)
                    ]
                    for role in (
                        "left_fixed",
                        "left_moving",
                        "right_fixed",
                        "right_moving",
                    )
                },
                "world_alignment": self.collider_alignment_diagnostics(),
                "friction_model": (
                    "closure_scaled_persistent_static_then_Coulomb_limited_slip"
                ),
            },
            table_height_m=self.table_height_m,
            contact_gate_injection=self._contact_gate_injection,
        )
        embedding_diagnostics = getattr(
            self.binding, "embedding_diagnostics", None
        )
        if embedding_diagnostics is not None:
            diagnostics["visual_embedding"] = dict(embedding_diagnostics)
        surface_quality = getattr(
            self.binding, "surface_quality_diagnostics", None
        )
        if callable(surface_quality):
            diagnostics["visual_surface_quality"] = surface_quality()
        return diagnostics


class PlushObjectAdapter:
    """Expose the rigid-pose subset for an authoritative particle soft body."""

    TETHER_GAIN_PER_S = 25.0
    TETHER_MAX_SPEED_M_S = 0.35

    def __init__(
        self,
        soft: object,
        binding: PlushVisualBinding,
        *,
        particle_radius_m: float,
        support_floor_z_m: float,
        solver_kind: str = "PBD.Elastic",
    ):
        if particle_radius_m <= 0.0:
            raise ValueError("particle radius must be positive")
        self.soft = soft
        self.binding = binding
        self.particle_radius_m = float(particle_radius_m)
        self.support_floor_z_m = float(support_floor_z_m)
        self.solver_kind = str(solver_kind)
        self._gripper_links: tuple[Any, Any] | None = None
        self._gripper_point_links: tuple[tuple[Any, Any], tuple[Any, Any]] | None = None
        self._wrist_links: tuple[Any, Any] | None = None
        self._grasped_indices: list[np.ndarray] = [
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        ]
        self._grasp_local_offsets: list[np.ndarray] = [
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        ]
        self._attach_events = [0, 0]
        self._release_events = [0, 0]
        self._rejected_attach_events = [0, 0]
        self._nearest_distance_m: list[float | None] = [None, None]
        self._initial_min_tip_plane_clearance_m: float | None = None
        self._initial_min_visual_tip_plane_clearance_m: float | None = None
        self._initial_min_particle_floor_clearance_m: float | None = None
        self._initial_min_visual_floor_clearance_m: float | None = None
        self._maximum_tether_error_m = [0.0, 0.0]
        self._maximum_tether_speed_m_s = [0.0, 0.0]

    def configure_grippers(
        self,
        left_link: object,
        right_link: object,
        left_moving_jaw: object,
        right_moving_jaw: object,
        left_wrist: object,
        right_wrist: object,
    ) -> None:
        self._gripper_links = (left_link, right_link)
        self._gripper_point_links = (
            (left_link, left_moving_jaw),
            (right_link, right_moving_jaw),
        )
        self._wrist_links = (left_wrist, right_wrist)

    def grasp_geometry(self, arm_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the fingertip support point and outward unit direction."""

        if self._gripper_point_links is None or self._wrist_links is None:
            raise RuntimeError("PBD grippers have not been configured")
        fixed, moving = self._gripper_point_links[arm_index]
        jaw_center = 0.5 * (
            as_numpy(fixed.get_pos()).reshape(3)
            + as_numpy(moving.get_pos()).reshape(3)
        )
        wrist_center = as_numpy(self._wrist_links[arm_index].get_pos()).reshape(3)
        outward = jaw_center - wrist_center
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm < 1e-6:
            raise RuntimeError("gripper outward direction is degenerate")
        outward /= outward_norm
        jaw_vertices = np.concatenate(
            (
                as_numpy(fixed.get_verts()).reshape(-1, 3),
                as_numpy(moving.get_verts()).reshape(-1, 3),
            ),
            axis=0,
        )
        tip_projection = float(np.max(jaw_vertices @ outward))
        surface_point = jaw_center + outward * (
            tip_projection - float(jaw_center @ outward)
        )
        return surface_point, outward

    def grasp_point(self, arm_index: int) -> np.ndarray:
        return self.grasp_geometry(arm_index)[0]

    def collision_free_initial_center(self, arm_index: int) -> np.ndarray:
        """Place physical and visible geometry wholly outside the fingertip."""

        surface_point, outward = self.grasp_geometry(arm_index)
        return separated_support_center(
            surface_point,
            outward,
            (
                as_numpy(self.binding.rest_particles_centered),
                as_numpy(self.binding.rest_visual_centered),
            ),
            clearance_m=self.particle_radius_m,
            floor_z_m=self.support_floor_z_m,
        )

    def release_all_grasps(self) -> None:
        for arm_index in range(2):
            self._release_grasp(arm_index)

    def attach_initial_left_grasp(self) -> bool:
        """Encode the task initial condition: the left follower holds the toy."""

        surface_point, outward = self.grasp_geometry(0)
        particles = as_numpy(self.soft.get_particles_pos()).reshape(-1, 3)
        self._initial_min_tip_plane_clearance_m = float(
            np.min((particles - surface_point) @ outward)
        )
        visual_vertices = (
            as_numpy(self.binding.rest_visual_centered).reshape(-1, 3)
            + particles.mean(axis=0)
        )
        self._initial_min_visual_tip_plane_clearance_m = float(
            np.min((visual_vertices - surface_point) @ outward)
        )
        self._initial_min_particle_floor_clearance_m = float(
            np.min(particles[:, 2]) - self.support_floor_z_m
        )
        self._initial_min_visual_floor_clearance_m = float(
            np.min(visual_vertices[:, 2]) - self.support_floor_z_m
        )
        return self._attach_grasp(
            0,
            maximum_nearest_distance_m=0.12,
            patch_count=24,
        )

    def update_grasps(
        self,
        gripper_percent: tuple[float, float],
        *,
        allow_new_attachment: bool = True,
    ) -> None:
        """Apply a 45/65 percent close/open hysteresis grasp state machine."""

        for arm_index, percent in enumerate(gripper_percent):
            if percent >= 65.0:
                self._release_grasp(arm_index)
            elif (
                allow_new_attachment
                and percent <= 45.0
                and not len(self._grasped_indices[arm_index])
            ):
                self._attach_grasp(
                    arm_index,
                    maximum_nearest_distance_m=0.065,
                    patch_count=24,
                )
        for arm_index in range(2):
            self._apply_grasp_tether(arm_index)

    def _attach_grasp(
        self,
        arm_index: int,
        *,
        maximum_nearest_distance_m: float,
        patch_count: int,
    ) -> bool:
        if self._gripper_links is None:
            raise RuntimeError("PBD grippers have not been configured")
        link = self._gripper_links[arm_index]
        particles = as_numpy(self.soft.get_particles_pos()).reshape(-1, 3)
        point = self.grasp_point(arm_index)
        other = self._grasped_indices[1 - arm_index]
        distances = np.linalg.norm(particles - point, axis=1)
        self._nearest_distance_m[arm_index] = float(np.min(distances))
        selected = nearest_particle_patch(
            particles,
            point,
            count=patch_count,
            maximum_nearest_distance_m=maximum_nearest_distance_m,
            excluded=other,
        )
        if not len(selected):
            self._rejected_attach_events[arm_index] += 1
            return False
        link_position = as_numpy(link.get_pos()).reshape(3)
        link_rotation = quaternion_wxyz_to_matrix(as_numpy(link.get_quat()))
        self._grasp_local_offsets[arm_index] = (
            particles[selected] - link_position
        ) @ link_rotation.T
        self._grasped_indices[arm_index] = selected
        self._attach_events[arm_index] += 1
        self._apply_grasp_tether(arm_index)
        return True

    def _apply_grasp_tether(self, arm_index: int) -> None:
        """Softly follow a link while leaving particles free for collision."""

        selected = self._grasped_indices[arm_index]
        if not len(selected):
            return
        if self._gripper_links is None:
            raise RuntimeError("PBD grippers have not been configured")
        link = self._gripper_links[arm_index]
        link_position = as_numpy(link.get_pos()).reshape(3)
        link_rotation = quaternion_wxyz_to_matrix(as_numpy(link.get_quat()))
        targets = (
            self._grasp_local_offsets[arm_index] @ link_rotation + link_position
        )
        particles = as_numpy(self.soft.get_particles_pos()).reshape(-1, 3)[selected]
        correction = targets - particles
        error = np.linalg.norm(correction, axis=1)
        self._maximum_tether_error_m[arm_index] = max(
            self._maximum_tether_error_m[arm_index], float(np.max(error))
        )
        velocities = correction * self.TETHER_GAIN_PER_S
        speeds = np.linalg.norm(velocities, axis=1)
        too_fast = speeds > self.TETHER_MAX_SPEED_M_S
        if np.any(too_fast):
            velocities[too_fast] *= (
                self.TETHER_MAX_SPEED_M_S / speeds[too_fast]
            )[:, None]
        self._maximum_tether_speed_m_s[arm_index] = max(
            self._maximum_tether_speed_m_s[arm_index],
            float(np.max(np.linalg.norm(velocities, axis=1))),
        )
        self.soft.set_particles_vel(
            velocities.astype(np.float32), particles_idx_local=selected
        )

    def _release_grasp(self, arm_index: int) -> None:
        selected = self._grasped_indices[arm_index]
        if not len(selected):
            return
        self._grasped_indices[arm_index] = np.empty(0, dtype=np.int64)
        self._grasp_local_offsets[arm_index] = np.empty((0, 3), dtype=np.float64)
        self._release_events[arm_index] += 1

    def get_pos(self) -> np.ndarray:
        _particles, center, _rotation = self.binding.rigid_transform()
        return as_numpy(center).astype(np.float32)

    def get_quat(self) -> np.ndarray:
        _particles, _center, rotation = self.binding.rigid_transform()
        return matrix_to_quaternion_wxyz(as_numpy(rotation)).astype(np.float32)

    def set_pos(self, position: object) -> None:
        import torch

        target = torch.as_tensor(
            np.asarray(position, dtype=np.float32).reshape(3),
            dtype=self.binding.rest_particles.dtype,
            device=self.binding.rest_particles.device,
        )
        particles = self.soft.get_particles_pos()
        translated = particles + target - particles.mean(dim=0)
        self.soft.set_particles_pos(translated)

    def set_quat(self, quaternion: object) -> None:
        import torch

        center = self.soft.get_particles_pos().mean(dim=0)
        rotation = torch.as_tensor(
            quaternion_wxyz_to_matrix(quaternion),
            dtype=self.binding.rest_particles.dtype,
            device=self.binding.rest_particles.device,
        )
        self.soft.set_particles_pos(self.binding.rest_particles_centered @ rotation + center)

    def set_dofs_velocity(self, velocity: object) -> None:
        values = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("PBD reset velocity must contain six finite values")
        if np.any(np.abs(values) > 1e-8):
            raise ValueError("PBD adapter currently supports only zero reset velocity")
        self.soft.set_particles_vel(0.0)

    def update_visual(self) -> None:
        self.binding.update()

    def diagnostics(self) -> dict[str, object]:
        particles = as_numpy(self.soft.get_particles_pos()).reshape(-1, 3)
        rest = as_numpy(self.binding.rest_particles).reshape(-1, 3)
        current_extents = np.ptp(particles, axis=0)
        rest_extents = np.ptp(rest, axis=0)
        diagnostics = {
            "kind": self.solver_kind,
            "particles": int(len(particles)),
            "visual_vertices": int(self.binding.rest_visual.shape[0]),
            "current_extents_m": current_extents.tolist(),
            "extent_ratio_to_rest": (current_extents / rest_extents).tolist(),
            "rigid_contact_identity_available": False,
            "particle_radius_m": self.particle_radius_m,
        }
        if self.solver_kind == "MPM.Elastic":
            diagnostics["grasp_constraints"] = {
                "method": "native_rigid_MPM_Coulomb_friction_contact",
                "synthetic_attachment": False,
                "velocity_tether": False,
                "youngs_modulus_pa": 8.0e3,
                "poissons_ratio": 0.20,
                "gripper_force_limit_n": 0.8,
            }
        else:
            diagnostics["grasp_constraints"] = {
                "method": "collision_aware_velocity_tether_to_gripper_link",
                "particles_remain_free_for_rigid_pbd_collision": True,
                "patch_particles": [
                    int(len(indices)) for indices in self._grasped_indices
                ],
                "attach_events": list(self._attach_events),
                "release_events": list(self._release_events),
                "rejected_attach_events": list(self._rejected_attach_events),
                "last_nearest_distance_m": list(self._nearest_distance_m),
                "close_threshold_percent": 45.0,
                "open_threshold_percent": 65.0,
                "tether_gain_per_s": self.TETHER_GAIN_PER_S,
                "tether_max_speed_m_s": self.TETHER_MAX_SPEED_M_S,
                "maximum_tether_error_m": list(self._maximum_tether_error_m),
                "maximum_tether_speed_m_s": list(
                    self._maximum_tether_speed_m_s
                ),
                "initial_min_particle_center_tip_plane_clearance_m": (
                    self._initial_min_tip_plane_clearance_m
                ),
                "initial_min_visual_vertex_tip_plane_clearance_m": (
                    self._initial_min_visual_tip_plane_clearance_m
                ),
                "initial_min_particle_center_floor_clearance_m": (
                    self._initial_min_particle_floor_clearance_m
                ),
                "initial_min_visual_vertex_floor_clearance_m": (
                    self._initial_min_visual_floor_clearance_m
                ),
            }
        return diagnostics
